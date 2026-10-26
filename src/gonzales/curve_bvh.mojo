"""Curve leaves in clusters, each with a BVH built in a frame along its fibers (CurveBvh in curves.mojo), for the rays
next to the RT cores. Fur is thin, nearly straight pieces at every slant; their world-axis boxes are mostly empty. A
cluster rotates the ray once; distances are the same in both frames, so one tmax serves the box and the curve tests.
Blender lion: shadow pass 16.5 -> 8.5 s, closest curve hits 8.5 -> 8.1 s."""
from std.atomic import Atomic, Ordering
from std.bit import count_trailing_zeros, pop_count
from std.math import ceildiv, sqrt
from std.memory import bitcast, stack_allocation
from std.memory.alloc import unsafe_alloc
from std.sys.info import is_gpu
from max.gpu import block_idx, thread_idx, lane_id, WARP_SIZE
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from max.gpu.primitives import warp
from max.gpu.primitives.warp import match_any, shuffle_idx, vote
from max.gpu.sync import barrier, syncwarp
from .bvh import BVH2Node, _node_box_hit, _shadow_is_null_material
from .curves import Curve, curve_at, CurveBvh, intersect_curve, curve_piece_endpoints, _curve_perp_axis
from .geometry import Point3f, Vec3f, cross
from .gpu_bvh import TPB, MaskT, _buf, _p, _tid, _seg_of, _ord, _unord
from .materials import Material
from .primitives import PrimId, Ray

comptime CURVE_CLUSTER = 4096       # pieces a cluster covers at most


def _leaf_range(nodes: Pointer[BVH2Node, MutUntrackedOrigin], i: Int) -> Tuple[Int, Int]:
    """The PrimIds below node i: a depth-first BVH2's subtree covers one contiguous range."""
    var a = i
    while nodes[unsafe_offset=a].count == Int32(0):
        a += 1
    var b = i
    while nodes[unsafe_offset=b].count == Int32(0):
        b = Int(nodes[unsafe_offset=b].offset)
    return (Int(nodes[unsafe_offset=a].offset), Int(nodes[unsafe_offset=b].offset) + Int(nodes[unsafe_offset=b].count))


@always_inline
def _in_frame(f: Pointer[Float32, MutUntrackedOrigin], p: Vec3f) -> Vec3f:
    return Vec3f(f[unsafe_offset=0] * p.x + f[unsafe_offset=1] * p.y + f[unsafe_offset=2] * p.z,
                 f[unsafe_offset=3] * p.x + f[unsafe_offset=4] * p.y + f[unsafe_offset=5] * p.z,
                 f[unsafe_offset=6] * p.x + f[unsafe_offset=7] * p.y + f[unsafe_offset=8] * p.z)


def _fiber_frame(prim_ids: Pointer[PrimId, MutUntrackedOrigin], curves: Pointer[Curve, MutUntrackedOrigin], a: Int, n: Int,
                 f: Pointer[Float32, MutUntrackedOrigin]):
    """Rows x, y, z: z the main axis of sum(|d| u u^T) over the pieces (u = d / |d|), by power iteration."""
    var xx = Float64(0); var xy = Float64(0); var xz = Float64(0); var yy = Float64(0); var yz = Float64(0); var zz = Float64(0)
    var lx = Float64(0); var ly = Float64(0); var lz = Float64(0)
    for k in range(a, a + n):
        var p = prim_ids[unsafe_offset=k]
        var curve = curve_at(curves, Int(p.id1))
        for piece in range(Int(p.id2) // 8, Int(p.id2) // 8 + Int(p.id2) % 8):
            var e = curve_piece_endpoints(curve, piece)
            var dx = Float64(e[1].x - e[0].x); var dy = Float64(e[1].y - e[0].y); var dz = Float64(e[1].z - e[0].z)
            var l = sqrt(dx * dx + dy * dy + dz * dz)
            if l < 1e-20:
                continue
            xx += dx * dx / l; xy += dx * dy / l; xz += dx * dz / l; yy += dy * dy / l; yz += dy * dz / l; zz += dz * dz / l
            lx += abs(dx); ly += abs(dy); lz += abs(dz)
    var vx = lx + 1e-30; var vy = ly; var vz = lz
    for _ in range(24):
        var wx = xx * vx + xy * vy + xz * vz; var wy = xy * vx + yy * vy + yz * vz; var wz = xz * vx + yz * vy + zz * vz
        var wl = sqrt(wx * wx + wy * wy + wz * wz)
        if wl < 1e-30:
            break
        vx = wx / wl; vy = wy / wl; vz = wz / wl
    var vl = sqrt(vx * vx + vy * vy + vz * vz)
    var z = Vec3f(Float32(vx / vl), Float32(vy / vl), Float32(vz / vl))
    var x = _curve_perp_axis(z)
    var y = cross(z, x)
    f[unsafe_offset=0] = x.x; f[unsafe_offset=1] = x.y; f[unsafe_offset=2] = x.z
    f[unsafe_offset=3] = y.x; f[unsafe_offset=4] = y.y; f[unsafe_offset=5] = y.z
    f[unsafe_offset=6] = z.x; f[unsafe_offset=7] = z.y; f[unsafe_offset=8] = z.z


def curve_bvh_occluded(cb: CurveBvh, curves: Pointer[Curve, MutUntrackedOrigin],
                       materials: Pointer[Material, MutUntrackedOrigin], ray: Ray, tmax: Float32) -> Bool:
    return _curve_bvh_walk[True](cb, curves, materials, ray, tmax)[0] >= 0


def curve_bvh_closest(cb: CurveBvh, curves: Pointer[Curve, MutUntrackedOrigin],
                      ray: Ray, tmax: Float32) -> Tuple[Int, Float32, Float32, Float32]:
    """(PrimId index or -1, t, u, v) of the nearest curve hit before tmax."""
    return _curve_bvh_walk[False](cb, curves, Pointer[Material, MutUntrackedOrigin].unsafe_dangling(), ray, tmax)


@always_inline
def curve_bvh_prim(cb: CurveBvh, slot: Int) -> PrimId:
    """The type-5 PrimId of a leaf slot curve_bvh_closest returned."""
    var r = cb.refs.unsafe_offset(3 * slot)
    return PrimId(Int64(r[unsafe_offset=0]), Int64(r[unsafe_offset=1]), Int64(r[unsafe_offset=2]), Int32(-1), Int8(5), Int8(0), Int8(0), Int8(0))


comptime FS = 16                    # floats a cluster's frame takes
comptime Q_STEPS = Float32(65000)   # grid steps over a level's root box; the margin keeps rounded boxes in 16 bits
comptime Q_LEAF = UInt32(1) << 31
comptime Q_BIAS = Float32(8388608)  # 2^23: its bits or a 16-bit q give the float 2^23 + q, so decoding is one integer op
comptime Q_BIAS_BITS = UInt32(0x4B000000)


# A node of either level: its box in that level's 16-bit grid (lo x | lo y << 16, lo z | hi x << 16, hi y | hi z << 16) and a
# link: an interior node's right child (the left one follows it), or Q_LEAF | count << 16 | first slot, all
# local to its level. Half the size of a BVH2Node; the ray goes into grid units, plus Q_BIAS, once per level. Rounding
# that origin moves the ray by at most half a step, which the boxes' outward step covers.
@always_inline                      # invariant only on the GPU: the optimized CPU walk hoisted it above its index test
def _qnode(qn: Pointer[UInt32, MutUntrackedOrigin], i: Int) -> SIMD[DType.uint32, 4]:
    return qn.unsafe_load[width=4, alignment=16 if is_gpu() else 4, invariant=is_gpu()](4 * i)


@always_inline
def _q_hit(w: SIMD[DType.uint32, 4], rdir: Vec3f, o: Vec3f, nx: Bool, ny: Bool, nz: Bool, tmax: Float32) -> Tuple[Bool, Float32]:
    var lo = bitcast[DType.float32, 4](((SIMD[DType.uint32, 4](w[0], w[0], w[1], w[1]) >> SIMD[DType.uint32, 4](0, 16, 0, 16)) & UInt32(0xFFFF)) | Q_BIAS_BITS)
    var hi = bitcast[DType.float32, 4](((SIMD[DType.uint32, 4](w[2], w[2], 0, 0) >> SIMD[DType.uint32, 4](0, 16, 0, 0)) & UInt32(0xFFFF)) | Q_BIAS_BITS)
    return _node_box_hit(lo, hi, rdir, o, nx, ny, nz, tmax)


@always_inline
def _q_oc(w: SIMD[DType.uint32, 4]) -> SIMD[DType.int32, 2]:
    """(right child, 0) or (first slot, count), as _node_oc."""
    var leaf = UInt32(0) - (w[3] >> 31)                    # all ones for a leaf; masks, not a branch: lanes stay together
    return SIMD[DType.int32, 2](Int32(w[3] & (UInt32(0xFFFF) | ~leaf)), Int32((w[3] >> 16) & UInt32(0x7FFF) & leaf))


comptime WALK_STACK = 64


struct CurveWalk(Movable):
    """The two-level walk of one ray, a node or leaf a step: one loop and one stack over both levels; a top leaf names a cluster, whose BVH is walked with the ray in its frame;
    a -1 on the stack marks the way back to the top level. Both levels are _qnodes, so their steps are the same code.
    Pieces are tested with the world ray."""
    var tmax: Float32
    var best: Int
    var bu: Float32
    var bv: Float32
    var wo: Vec3f
    var wd: Vec3f
    var top_o: Vec3f
    var top_rdir: Vec3f
    var o: Vec3f
    var rdir: Vec3f
    var nx: Bool
    var ny: Bool
    var nz: Bool
    var qn: Pointer[UInt32, MutUntrackedOrigin]
    var slot0: Int
    var in_cluster: Bool
    var to_visit: Int
    var current: Int
    var cur_oc: SIMD[DType.int32, 2]

    @always_inline
    def __init__(out self, cb: CurveBvh, ray: Ray, tmax: Float32):
        self.tmax = tmax
        self.best = -1
        self.bu = Float32(0); self.bv = Float32(0)
        self.wo = Vec3f(ray.origin.x, ray.origin.y, ray.origin.z)
        self.wd = ray.direction
        var g = cb.grid
        self.top_o = Vec3f((self.wo.x - g[0]) / g[3] + Q_BIAS, (self.wo.y - g[1]) / g[4] + Q_BIAS, (self.wo.z - g[2]) / g[5] + Q_BIAS)
        self.top_rdir = Vec3f(g[3] / self.wd.x, g[4] / self.wd.y, g[5] / self.wd.z)
        self.o = self.top_o
        self.rdir = self.top_rdir
        self.nx = self.rdir.x >= Float32(0.0); self.ny = self.rdir.y >= Float32(0.0); self.nz = self.rdir.z >= Float32(0.0)
        self.qn = cb.top
        self.slot0 = 0
        self.in_cluster = False
        self.to_visit = 0
        self.current = 0
        self.cur_oc = _q_oc(_qnode(cb.top, 0))

    @always_inline
    def step[any_hit: Bool](mut self, cb: CurveBvh, curves: Pointer[Curve, MutUntrackedOrigin], materials: Pointer[Material, MutUntrackedOrigin],
                            stack: Pointer[Int32, MutUntrackedOrigin]) -> Bool:
        """One node or leaf; True when the walk is over (best, tmax, bu, bv hold its result)."""
        var pop = False
        if self.cur_oc[1] > 0:
            var off = Int(self.cur_oc[0])
            if not self.in_cluster:
                var c = Int(cb.top_cl[unsafe_offset=off])
                var f = cb.frames.unsafe_offset(FS * c)
                var qs = Vec3f(f[unsafe_offset=12], f[unsafe_offset=13], f[unsafe_offset=14])
                var co = _in_frame(f, self.wo) - Vec3f(f[unsafe_offset=9], f[unsafe_offset=10], f[unsafe_offset=11])
                co = Vec3f(co.x / qs.x + Q_BIAS, co.y / qs.y + Q_BIAS, co.z / qs.z + Q_BIAS)
                var cd = _in_frame(f, self.wd)
                var crdir = Vec3f(qs.x / cd.x, qs.y / cd.y, qs.z / cd.z)
                var cnx = crdir.x >= Float32(0.0); var cny = crdir.y >= Float32(0.0); var cnz = crdir.z >= Float32(0.0)
                var s0 = Int(cb.roots[unsafe_offset=c])
                var cq = cb.nodes.unsafe_offset(8 * s0)
                var w = _qnode(cq, 0)
                if _q_hit(w, crdir, co, cnx, cny, cnz, self.tmax)[0] and self.to_visit < WALK_STACK:
                    stack[unsafe_offset=self.to_visit] = Int32(-1)
                    self.to_visit += 1
                    self.o = co; self.rdir = crdir; self.nx = cnx; self.ny = cny; self.nz = cnz
                    self.qn = cq; self.slot0 = s0; self.in_cluster = True
                    self.current = 0; self.cur_oc = _q_oc(w)
                    return False
                pop = True
            else:
                off += self.slot0
                for j in range(Int(self.cur_oc[1])):
                    var r = cb.refs.unsafe_offset(3 * (off + j))
                    comptime if any_hit:
                        if _shadow_is_null_material(materials, Int64(r[unsafe_offset=2])):
                            continue
                    var id2 = Int(r[unsafe_offset=1])
                    var h = intersect_curve(self.wo, self.wd, curve_at(curves, Int(r[unsafe_offset=0])), id2 // 8, id2 % 8, self.tmax)
                    if h[0]:
                        self.best = off + j; self.tmax = h[1]; self.bu = h[2]; self.bv = h[3]
                        comptime if any_hit:
                            self.to_visit = 0                  # done: leave by the pop below
                            break
                pop = True
        else:
            var li = self.current + 1
            var ri = Int(self.cur_oc[0])
            var lw = _qnode(self.qn, li); var rw = _qnode(self.qn, ri)
            var lh = _q_hit(lw, self.rdir, self.o, self.nx, self.ny, self.nz, self.tmax)
            var rh = _q_hit(rw, self.rdir, self.o, self.nx, self.ny, self.nz, self.tmax)
            if lh[0] and rh[0]:
                var near_left = lh[1] <= rh[1]
                self.current = li if near_left else ri
                self.cur_oc = _q_oc(lw if near_left else rw)
                if self.to_visit < WALK_STACK:
                    stack[unsafe_offset=self.to_visit] = Int32(ri if near_left else li)
                    self.to_visit += 1
            elif lh[0]:
                self.current = li; self.cur_oc = _q_oc(lw)
            elif rh[0]:
                self.current = ri; self.cur_oc = _q_oc(rw)
            else:
                pop = True
        if pop:
            if self.to_visit == 0:
                return True
            self.to_visit -= 1
            self.current = Int(stack[unsafe_offset=self.to_visit])
            if self.current < 0:                               # back to the top level
                self.o = self.top_o; self.rdir = self.top_rdir
                self.nx = self.rdir.x >= Float32(0.0); self.ny = self.rdir.y >= Float32(0.0); self.nz = self.rdir.z >= Float32(0.0)
                self.qn = cb.top; self.in_cluster = False
                if self.to_visit == 0:
                    return True
                self.to_visit -= 1
                self.current = Int(stack[unsafe_offset=self.to_visit])
            self.cur_oc = _q_oc(_qnode(self.qn, self.current))
        return False


def _curve_bvh_walk[any_hit: Bool](cb: CurveBvh, curves: Pointer[Curve, MutUntrackedOrigin],
                                   materials: Pointer[Material, MutUntrackedOrigin], ray: Ray, tmax_in: Float32) -> Tuple[Int, Float32, Float32, Float32]:
    var stack = stack_allocation[WALK_STACK, Int32]()
    var w = CurveWalk(cb, ray, tmax_in)
    while not w.step[any_hit](cb, curves, materials, stack.unsafe_origin_cast[MutUntrackedOrigin]()):
        pass
    return (w.best, w.tmax, w.bu, w.bv)


# ---- the same clusters, built on the GPU (gpu_bvh) ----


struct CurveBvhDev(Movable):
    var bvh: CurveBvh                # device arrays, owned by bufs
    var bufs: List[DeviceBuffer[DType.uint8]]

    def __init__(out self, bvh: CurveBvh, var bufs: List[DeviceBuffer[DType.uint8]]):
        self.bvh = bvh; self.bufs = bufs^


@always_inline
def _prim_box[framed: Bool](p: PrimId, curves: Pointer[Curve, MutUntrackedOrigin], f: Pointer[Float32, MutUntrackedOrigin]) -> Tuple[SIMD[DType.float32, 4], SIMD[DType.float32, 4]]:
    var curve = curve_at(curves, Int(p.id1))
    var lo = SIMD[DType.float32, 4](Float32(1e30))
    var hi = SIMD[DType.float32, 4](Float32(-1e30))
    for piece in range(Int(p.id2) // 8, Int(p.id2) // 8 + Int(p.id2) % 8):
        var e = curve_piece_endpoints(curve, piece)
        var r = max(e[2], e[3])
        var a = e[0]; var b = e[1]
        comptime if framed:
            a = _in_frame(f, a); b = _in_frame(f, b)
        lo = min(lo, SIMD[DType.float32, 4](min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), Float32(0)) - r)
        hi = max(hi, SIMD[DType.float32, 4](max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), Float32(0)) + r)
    return (lo, hi)


@always_inline
def _put_box(out_b: Pointer[Float32, MutUntrackedOrigin], i: Int, lo: SIMD[DType.float32, 4], hi: SIMD[DType.float32, 4]):
    for q in range(3):
        out_b[unsafe_offset=6 * i + q] = lo[q]; out_b[unsafe_offset=6 * i + 3 + q] = hi[q]


def _world_boxes_kernel(prim_ids: Pointer[PrimId, MutUntrackedOrigin], curves: Pointer[Curve, MutUntrackedOrigin], a_dp: Int64, n_dp: Int64,
                        boxes: Pointer[Float32, MutUntrackedOrigin]):
    var k = _tid()
    if k < Int(n_dp):
        var b = _prim_box[False](prim_ids[unsafe_offset=Int(a_dp) + k], curves, boxes)
        _put_box(boxes, k, b[0], b[1])


comptime NB = 12                    # SAH buckets, as bvh._bvh_split


@always_inline
def _seg_at(seg: Pointer[Int32, MutUntrackedOrigin], i: Int, n: Int) -> Int32:
    return seg[unsafe_offset=i] if i < n else Int32(-1)


def _seg_kernel(starts: Pointer[Int32, MutUntrackedOrigin], ends: Pointer[Int32, MutUntrackedOrigin], k_dp: Int64, n_dp: Int64,
                seg: Pointer[Int32, MutUntrackedOrigin]):
    """seg[i]: the pending node whose range holds position i, or -1."""
    var i = _tid()
    if i >= Int(n_dp):
        return
    var s = _seg_of(starts, Int(k_dp), i)
    seg[unsafe_offset=i] = Int32(s) if i >= Int(starts[unsafe_offset=s]) and i < Int(ends[unsafe_offset=s]) else Int32(-1)


def _fill_kernel(p: Pointer[Int32, MutUntrackedOrigin], n_dp: Int64, stride_dp: Int64, hi_dp: Int64):
    """Records of `stride` ints: the first `hi` +inf (as _ord, for atomic minima), the rest 0."""
    var t = _tid()
    if t < Int(n_dp):
        p[unsafe_offset=t] = _ord(Float32(1e30)) if t % Int(stride_dp) < Int(hi_dp) else Int32(0)


@always_inline
def _centroid(wbox: Pointer[Float32, MutUntrackedOrigin], prim: Int) -> SIMD[DType.float32, 4]:
    var b = wbox.unsafe_offset(6 * prim)
    return SIMD[DType.float32, 4](b[unsafe_offset=0] + b[unsafe_offset=3], b[unsafe_offset=1] + b[unsafe_offset=4], b[unsafe_offset=2] + b[unsafe_offset=5],
                                  Float32(0)) * Float32(0.5)


def _node_bounds_kernel(wbox: Pointer[Float32, MutUntrackedOrigin], perm: Pointer[Int32, MutUntrackedOrigin], seg: Pointer[Int32, MutUntrackedOrigin],
                        n_dp: Int64, nb: Pointer[Int32, MutUntrackedOrigin]):
    """Per pending node: box min, box max, centroid min, centroid max (12 ordered ints); a block inside one node reduces first."""
    var n = Int(n_dp)
    var i = _tid()
    var s = Int(_seg_at(seg, i, n))
    var v = SIMD[DType.float32, 16](Float32(1e30))
    if s >= 0:
        var b = wbox.unsafe_offset(6 * Int(perm[unsafe_offset=i]))
        var c = _centroid(wbox, Int(perm[unsafe_offset=i]))
        v = SIMD[DType.float32, 16](b[unsafe_offset=0], b[unsafe_offset=1], b[unsafe_offset=2], -b[unsafe_offset=3], -b[unsafe_offset=4], -b[unsafe_offset=5],
                                    c[0], c[1], c[2], -c[0], -c[1], -c[2], 0, 0, 0, 0)
    var first = Int(block_idx.x) * TPB
    var s0 = Int(_seg_at(seg, first, n))
    if s0 >= 0 and Int(_seg_at(seg, min(first + TPB, n) - 1, n)) == s0:
        var sh = stack_allocation[TPB * 16, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
        var k = Int(thread_idx.x)
        sh.unsafe_store(16 * k, v)
        barrier()
        var w = TPB // 2
        while w > 0:
            if k < w:
                sh.unsafe_store(16 * k, min(sh.unsafe_load[width=16](16 * k), sh.unsafe_load[width=16](16 * (k + w))))
            barrier()
            w //= 2
        if k == 0:
            v = sh.unsafe_load[width=16](0)
            for q in range(12):
                _ = Atomic.min[ordering=Ordering.RELAXED](nb.unsafe_offset(12 * s0 + q), _ord(v[q]))
    elif s >= 0:
        for q in range(12):
            _ = Atomic.min[ordering=Ordering.RELAXED](nb.unsafe_offset(12 * s + q), _ord(v[q]))


@always_inline
def _bucket(c: SIMD[DType.float32, 4], axis: Int, cmin: Float32, inv: Float32) -> Int:
    return min(max(Int(Float32(NB) * ((c[axis] - cmin) * inv)), 0), NB - 1)


def _bin_kernel(wbox: Pointer[Float32, MutUntrackedOrigin], perm: Pointer[Int32, MutUntrackedOrigin], seg: Pointer[Int32, MutUntrackedOrigin], n_dp: Int64,
                axis: Pointer[Int32, MutUntrackedOrigin], cmin: Pointer[Float32, MutUntrackedOrigin], inv: Pointer[Float32, MutUntrackedOrigin],
                bins: Pointer[Int32, MutUntrackedOrigin]):
    """Per splitting node (axis >= 0) and bucket: box min, negated box max (ordered ints), count. A block inside one node
    bins in shared memory first."""
    var n = Int(n_dp)
    var i = _tid()
    var k = Int(thread_idx.x)
    var first = Int(block_idx.x) * TPB
    var s0 = Int(_seg_at(seg, first, n))
    var uniform = s0 >= 0 and Int(_seg_at(seg, min(first + TPB, n) - 1, n)) == s0 and axis[unsafe_offset=s0] >= Int32(0)
    var sh = stack_allocation[NB * 7, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    if uniform:
        if k < NB * 7:
            sh[unsafe_offset=k] = _ord(Float32(1e30)) if k % 7 < 6 else Int32(0)
        barrier()
    var s = Int(_seg_at(seg, i, n))
    if s >= 0 and axis[unsafe_offset=s] >= Int32(0):
        var p = Int(perm[unsafe_offset=i])
        var b = wbox.unsafe_offset(6 * p)
        var bk = _bucket(_centroid(wbox, p), Int(axis[unsafe_offset=s]), cmin[unsafe_offset=s], inv[unsafe_offset=s])
        if uniform:
            var dst = sh.unsafe_offset(7 * bk)
            for q in range(3):
                _ = Atomic.min[ordering=Ordering.RELAXED](dst.unsafe_offset(q), _ord(b[unsafe_offset=q]))
                _ = Atomic.min[ordering=Ordering.RELAXED](dst.unsafe_offset(3 + q), _ord(-b[unsafe_offset=3 + q]))
            _ = Atomic.fetch_add[ordering=Ordering.RELAXED](dst.unsafe_offset(6), Int32(1))
        else:
            var dst = bins.unsafe_offset((s * NB + bk) * 7)
            for q in range(3):
                _ = Atomic.min[ordering=Ordering.RELAXED](dst.unsafe_offset(q), _ord(b[unsafe_offset=q]))
                _ = Atomic.min[ordering=Ordering.RELAXED](dst.unsafe_offset(3 + q), _ord(-b[unsafe_offset=3 + q]))
            _ = Atomic.fetch_add[ordering=Ordering.RELAXED](dst.unsafe_offset(6), Int32(1))
    if uniform:
        barrier()
        if k < NB * 7:
            var g = bins.unsafe_offset(s0 * NB * 7 + k)
            if k % 7 == 6:
                if sh[unsafe_offset=k] != Int32(0):
                    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](g, sh[unsafe_offset=k])
            else:
                _ = Atomic.min[ordering=Ordering.RELAXED](g, sh[unsafe_offset=k])


def _partition_kernel(wbox: Pointer[Float32, MutUntrackedOrigin], perm: Pointer[Int32, MutUntrackedOrigin], perm2: Pointer[Int32, MutUntrackedOrigin],
                      seg: Pointer[Int32, MutUntrackedOrigin], n_dp: Int64, axis: Pointer[Int32, MutUntrackedOrigin], cmin: Pointer[Float32, MutUntrackedOrigin],
                      inv: Pointer[Float32, MutUntrackedOrigin], split: Pointer[Int32, MutUntrackedOrigin], starts: Pointer[Int32, MutUntrackedOrigin],
                      nleft: Pointer[Int32, MutUntrackedOrigin], cursor: Pointer[Int32, MutUntrackedOrigin]):
    """Splitting nodes: buckets <= split first (split -1: the first nleft positions, a median cut); the rest stay. Each warp
    takes its places on a side with one atomic."""
    var n = Int(n_dp)
    var i = _tid()
    var s = Int(_seg_at(seg, i, n))
    var moving = s >= 0 and axis[unsafe_offset=s] >= Int32(0)
    var p = Int32(0)
    var left = False
    if i < n:
        p = perm[unsafe_offset=i]
        if not moving:
            perm2[unsafe_offset=i] = p
    if moving:
        var st = Int(starts[unsafe_offset=s])
        if split[unsafe_offset=s] < Int32(0):
            left = i - st < Int(nleft[unsafe_offset=s])
        else:
            left = _bucket(_centroid(wbox, Int(p)), Int(axis[unsafe_offset=s]), cmin[unsafe_offset=s], inv[unsafe_offset=s]) <= Int(split[unsafe_offset=s])
    var lane = Int(lane_id())
    var below = (Scalar[MaskT](1) << Scalar[MaskT](lane)) - Scalar[MaskT](1)
    var peers = Scalar[MaskT](match_any(Int32(s) * 2 + Int32(1 if left else 0) if moving else Int32(-1 - lane)))
    var leader = Int(count_trailing_zeros(peers))
    var base = Int32(0)
    if moving and lane == leader:
        base = Atomic.fetch_add[ordering=Ordering.RELAXED](cursor.unsafe_offset(2 * s + (0 if left else 1)), Int32(pop_count(peers)))
    base = shuffle_idx(base, UInt32(leader))
    if moving:
        var st = Int(starts[unsafe_offset=s])
        var pos = st + (0 if left else Int(nleft[unsafe_offset=s])) + Int(base) + Int(pop_count(peers & below))
        perm2[unsafe_offset=pos] = p


def _frames_kernel(prim_ids: Pointer[PrimId, MutUntrackedOrigin], curves: Pointer[Curve, MutUntrackedOrigin], a_dp: Int64,
                   pslot: Pointer[Int32, MutUntrackedOrigin], cl_lo: Pointer[Int32, MutUntrackedOrigin], frames: Pointer[Float32, MutUntrackedOrigin]):
    """_fiber_frame, one block per cluster."""
    var c = Int(block_idx.x)
    var k = Int(thread_idx.x)
    var acc = SIMD[DType.float32, 16](0)
    var j = Int(cl_lo[unsafe_offset=c]) + k
    while j < Int(cl_lo[unsafe_offset=c + 1]):
        var p = prim_ids[unsafe_offset=Int(a_dp) + Int(pslot[unsafe_offset=j])]
        var curve = curve_at(curves, Int(p.id1))
        for piece in range(Int(p.id2) // 8, Int(p.id2) // 8 + Int(p.id2) % 8):
            var e = curve_piece_endpoints(curve, piece)
            var dx = e[1].x - e[0].x; var dy = e[1].y - e[0].y; var dz = e[1].z - e[0].z
            var l = sqrt(dx * dx + dy * dy + dz * dz)
            if l < Float32(1e-20):
                continue
            acc += SIMD[DType.float32, 16](dx * dx / l, dx * dy / l, dx * dz / l, dy * dy / l, dy * dz / l, dz * dz / l, abs(dx), abs(dy), abs(dz),
                                          0, 0, 0, 0, 0, 0, 0)
        j += TPB
    var sh = stack_allocation[TPB * 16, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    sh.unsafe_store(16 * k, acc)
    barrier()
    var w = TPB // 2
    while w > 0:
        if k < w:
            sh.unsafe_store(16 * k, sh.unsafe_load[width=16](16 * k) + sh.unsafe_load[width=16](16 * (k + w)))
        barrier()
        w //= 2
    if k != 0:
        return
    var m = sh.unsafe_load[width=16](0)
    var vx = m[6] + Float32(1e-30); var vy = m[7]; var vz = m[8]
    for _ in range(24):
        var wx = m[0] * vx + m[1] * vy + m[2] * vz; var wy = m[1] * vx + m[3] * vy + m[4] * vz; var wz = m[2] * vx + m[4] * vy + m[5] * vz
        var wl = sqrt(wx * wx + wy * wy + wz * wz)
        if wl < Float32(1e-30):
            break
        vx = wx / wl; vy = wy / wl; vz = wz / wl
    var vl = sqrt(vx * vx + vy * vy + vz * vz)
    var z = Vec3f(vx / vl, vy / vl, vz / vl)
    var x = _curve_perp_axis(z)
    var y = cross(z, x)
    var f = frames.unsafe_offset(FS * c)
    f[unsafe_offset=0] = x.x; f[unsafe_offset=1] = x.y; f[unsafe_offset=2] = x.z
    f[unsafe_offset=3] = y.x; f[unsafe_offset=4] = y.y; f[unsafe_offset=5] = y.z
    f[unsafe_offset=6] = z.x; f[unsafe_offset=7] = z.y; f[unsafe_offset=8] = z.z


def _frame_boxes_kernel(prim_ids: Pointer[PrimId, MutUntrackedOrigin], curves: Pointer[Curve, MutUntrackedOrigin], a_dp: Int64,
                        pslot: Pointer[Int32, MutUntrackedOrigin], cl_lo: Pointer[Int32, MutUntrackedOrigin], nc_dp: Int64,
                        frames: Pointer[Float32, MutUntrackedOrigin], n_dp: Int64, widx: Pointer[Int32, MutUntrackedOrigin],
                        wmin: Pointer[Float32, MutUntrackedOrigin], wmax: Pointer[Float32, MutUntrackedOrigin]):
    """Each piece's box in its cluster's frame, as build_bvh2's work arrays (indices local to the cluster)."""
    var j = _tid()
    if j >= Int(n_dp):
        return
    var c = _seg_of(cl_lo, Int(nc_dp), j)
    var b = _prim_box[True](prim_ids[unsafe_offset=Int(a_dp) + Int(pslot[unsafe_offset=j])], curves, frames.unsafe_offset(FS * c))
    widx[unsafe_offset=j] = Int32(j - Int(cl_lo[unsafe_offset=c]))
    for q in range(3):
        wmin[unsafe_offset=3 * j + q] = b[0][q]; wmax[unsafe_offset=3 * j + q] = b[1][q]


comptime CW = 4                     # clusters a block, a warp each


@always_inline
def _put_qnode(nodes: Pointer[UInt32, MutUntrackedOrigin], i: Int, bmin: Point3f, bmax: Point3f, qo: SIMD[DType.float32, 4],
               qs: SIMD[DType.float32, 4], link: UInt32):
    """A box rounded outward by a step more onto the cluster's grid."""
    var l = max((SIMD[DType.float32, 4](bmin.x, bmin.y, bmin.z, 0) - qo) / qs - Float32(1), SIMD[DType.float32, 4](0)).cast[DType.uint32]()
    var h = min((SIMD[DType.float32, 4](bmax.x, bmax.y, bmax.z, 0) - qo) / qs + Float32(2), SIMD[DType.float32, 4](65535)).cast[DType.uint32]()
    nodes.unsafe_store(4 * i, SIMD[DType.uint32, 4](l[0] | (l[1] << 16), l[2] | (h[0] << 16), h[1] | (h[2] << 16), link))


def _sah_clusters_kernel(cl_lo: Pointer[Int32, MutUntrackedOrigin], nc_dp: Int64, widx: Pointer[Int32, MutUntrackedOrigin], wtmp: Pointer[Int32, MutUntrackedOrigin],
                         wmin: Pointer[Float32, MutUntrackedOrigin], wmax: Pointer[Float32, MutUntrackedOrigin], nodes: Pointer[UInt32, MutUntrackedOrigin],
                         frames: Pointer[Float32, MutUntrackedOrigin], roots: Pointer[Int32, MutUntrackedOrigin], pslot: Pointer[Int32, MutUntrackedOrigin], a_dp: Int64,
                         prim_ids: Pointer[PrimId, MutUntrackedOrigin], refs: Pointer[Int32, MutUntrackedOrigin],
                         top_cl: Pointer[Int32, MutUntrackedOrigin]):
    """bvh._bvh_split's binned SAH, depth-first as build_bvh2_node, a warp per cluster; cluster c's nodes (_qnode) go to
    [2 lo, 2 hi), the grid of their boxes (from the root's) to its frame."""
    var lane = Int(lane_id())
    var w = Int(thread_idx.x) // WARP_SIZE
    var c = Int(block_idx.x) * CW + w
    if c >= Int(nc_dp):
        return
    var lo = Int(cl_lo[unsafe_offset=c]); var n = Int(cl_lo[unsafe_offset=c + 1]) - lo
    var idx = widx.unsafe_offset(lo); var tmp = wtmp.unsafe_offset(lo)
    var bmn = wmin.unsafe_offset(3 * lo); var bmx = wmax.unsafe_offset(3 * lo)
    var sh = stack_allocation[CW * NB * 7, Scalar[DType.int32], address_space=AddressSpace.SHARED]().unsafe_offset(w * NB * 7)
    var below = (Scalar[MaskT](1) << Scalar[MaskT](lane)) - Scalar[MaskT](1)
    var base = 2 * lo
    var st_s = stack_allocation[64, Int32](); var st_e = stack_allocation[64, Int32](); var st_p = stack_allocation[64, Int32]()
    var depth = 0
    var start = 0; var end = n; var parent = -1
    var count = 0
    var qo = SIMD[DType.float32, 4](0); var qs = SIMD[DType.float32, 4](1)
    while True:
        var my = count
        count += 1
        if parent >= 0 and lane == 0:
            nodes[unsafe_offset=4 * (base + parent) + 3] = UInt32(my)
        var v = SIMD[DType.float32, 16](Float32(1e30))
        for j in range(start + lane, end, WARP_SIZE):
            var e = 3 * Int(idx[unsafe_offset=j])
            var l3 = SIMD[DType.float32, 4](bmn[unsafe_offset=e], bmn[unsafe_offset=e + 1], bmn[unsafe_offset=e + 2], 0)
            var h3 = SIMD[DType.float32, 4](bmx[unsafe_offset=e], bmx[unsafe_offset=e + 1], bmx[unsafe_offset=e + 2], 0)
            var c3 = (l3 + h3) * Float32(0.5)
            v = min(v, SIMD[DType.float32, 16](l3[0], l3[1], l3[2], -h3[0], -h3[1], -h3[2], c3[0], c3[1], c3[2], -c3[0], -c3[1], -c3[2], 0, 0, 0, 0))
        comptime for q in range(12):
            v[q] = warp.min(v[q])
        var cnt = end - start
        var bmin = Point3f(v[0], v[1], v[2]); var bmax = Point3f(-v[3], -v[4], -v[5])
        if my == 0:
            qs = max((SIMD[DType.float32, 4](bmax.x, bmax.y, bmax.z, 0) - SIMD[DType.float32, 4](bmin.x, bmin.y, bmin.z, 0)) / Q_STEPS,
                     SIMD[DType.float32, 4](Float32(1e-30)))
            qo = SIMD[DType.float32, 4](bmin.x, bmin.y, bmin.z, 0) - Float32(200) * qs
            if lane == 0:
                var f = frames.unsafe_offset(FS * c)
                comptime for q in range(3):
                    f[unsafe_offset=9 + q] = qo[q]; f[unsafe_offset=12 + q] = qs[q]
        var cex = -v[9] - v[6]; var cey = -v[10] - v[7]; var cez = -v[11] - v[8]
        var dim = 0
        if cey > cex and cey >= cez:
            dim = 1
        elif cez > cex and cez > cey:
            dim = 2
        var cmin_d = v[6 + dim]; var cmax_d = -v[9 + dim]
        var dx = max(bmax.x - bmin.x, Float32(0)); var dy = max(bmax.y - bmin.y, Float32(0)); var dz = max(bmax.z - bmin.z, Float32(0))
        var sa = Float32(2.0) * (dx * dy + dy * dz + dz * dx)
        var mid = -1
        if not (sa == Float32(0.0) or cnt == 1 or cmax_d == cmin_d):
            if cnt == 2:
                if lane == 0:
                    var e0 = Int(idx[unsafe_offset=start]); var e1 = Int(idx[unsafe_offset=start + 1])
                    if bmn[unsafe_offset=3 * e1 + dim] + bmx[unsafe_offset=3 * e1 + dim] < bmn[unsafe_offset=3 * e0 + dim] + bmx[unsafe_offset=3 * e0 + dim]:
                        idx[unsafe_offset=start] = Int32(e1); idx[unsafe_offset=start + 1] = Int32(e0)
                syncwarp()
                mid = start + 1
            else:
                var inv_d = Float32(1.0) / (cmax_d - cmin_d)
                for k in range(lane, NB * 7, WARP_SIZE):
                    sh[unsafe_offset=k] = _ord(Float32(1e30)) if k % 7 < 6 else Int32(0)
                syncwarp()
                for j in range(start + lane, end, WARP_SIZE):
                    var e = 3 * Int(idx[unsafe_offset=j])
                    var ci = Float32(0.5) * (bmn[unsafe_offset=e + dim] + bmx[unsafe_offset=e + dim])
                    var o = 7 * min(max(Int(Float32(NB) * ((ci - cmin_d) * inv_d)), 0), NB - 1)
                    for q in range(3):
                        _ = Atomic.min[ordering=Ordering.RELAXED](sh.unsafe_offset(o + q), _ord(bmn[unsafe_offset=e + q]))
                        _ = Atomic.min[ordering=Ordering.RELAXED](sh.unsafe_offset(o + 3 + q), _ord(-bmx[unsafe_offset=e + q]))
                    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](sh.unsafe_offset(o + 6), Int32(1))
                syncwarp()
                var cost = SIMD[DType.float32, 16](0)          # cost[sp]: split after bucket sp; every lane the same sweep
                var pl = SIMD[DType.float32, 4](Float32(1e30)); var pu = SIMD[DType.float32, 4](Float32(1e30))
                var pc = 0
                comptime for sp in range(NB - 1):
                    var o = 7 * sp
                    pc += Int(sh[unsafe_offset=o + 6])
                    pl = min(pl, SIMD[DType.float32, 4](_unord(sh[unsafe_offset=o]), _unord(sh[unsafe_offset=o + 1]), _unord(sh[unsafe_offset=o + 2]), 0))
                    pu = min(pu, SIMD[DType.float32, 4](_unord(sh[unsafe_offset=o + 3]), _unord(sh[unsafe_offset=o + 4]), _unord(sh[unsafe_offset=o + 5]), 0))
                    var e = max(-pu - pl, SIMD[DType.float32, 4](0))
                    cost[sp] = Float32(pc) * Float32(2.0) * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0]) if pc > 0 else Float32(0)
                pl = SIMD[DType.float32, 4](Float32(1e30)); pu = SIMD[DType.float32, 4](Float32(1e30))
                pc = 0
                comptime for sp in range(NB - 1, 0, -1):
                    var o = 7 * sp
                    pc += Int(sh[unsafe_offset=o + 6])
                    pl = min(pl, SIMD[DType.float32, 4](_unord(sh[unsafe_offset=o]), _unord(sh[unsafe_offset=o + 1]), _unord(sh[unsafe_offset=o + 2]), 0))
                    pu = min(pu, SIMD[DType.float32, 4](_unord(sh[unsafe_offset=o + 3]), _unord(sh[unsafe_offset=o + 4]), _unord(sh[unsafe_offset=o + 5]), 0))
                    var e = max(-pu - pl, SIMD[DType.float32, 4](0))
                    if pc > 0:
                        cost[sp - 1] += Float32(pc) * Float32(2.0) * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0])
                var best = -1
                var best_cost = Float32(3.0e38)
                comptime for sp in range(NB - 1):
                    if cost[sp] < best_cost:
                        best_cost = cost[sp]; best = sp
                syncwarp()
                if cnt > 4 or (cnt > 1 and Float32(0.5) + best_cost / sa < Float32(cnt)):
                    var nl = 0
                    var nr = 0
                    for j0 in range(start, end, WARP_SIZE):
                        var j = j0 + lane
                        var e = Int(idx[unsafe_offset=j]) if j < end else 0
                        var left = False
                        if j < end:
                            var ci = Float32(0.5) * (bmn[unsafe_offset=3 * e + dim] + bmx[unsafe_offset=3 * e + dim])
                            left = min(max(Int(Float32(NB) * ((ci - cmin_d) * inv_d)), 0), NB - 1) <= best
                        var ml = vote[MaskT](left)
                        var mr = vote[MaskT](j < end and not left)
                        if left:
                            tmp[unsafe_offset=start + nl + Int(pop_count(ml & below))] = Int32(e)
                        elif j < end:
                            tmp[unsafe_offset=end - 1 - nr - Int(pop_count(mr & below))] = Int32(e)
                        nl += Int(pop_count(ml)); nr += Int(pop_count(mr))
                    syncwarp()
                    for j in range(start + lane, end, WARP_SIZE):
                        idx[unsafe_offset=j] = tmp[unsafe_offset=j]
                    syncwarp()
                    mid = start + nl
                    if mid == start or mid == end:
                        mid = -1
        if mid < 0 or depth == 64:
            if lane == 0:
                _put_qnode(nodes, base + my, bmin, bmax, qo, qs, Q_LEAF | (UInt32(cnt) << 16) | UInt32(start))
            if depth == 0:
                break
            depth -= 1
            start = Int(st_s[unsafe_offset=depth]); end = Int(st_e[unsafe_offset=depth]); parent = Int(st_p[unsafe_offset=depth])
        else:
            if lane == 0:
                _put_qnode(nodes, base + my, bmin, bmax, qo, qs, UInt32(0))
            st_s[unsafe_offset=depth] = Int32(mid); st_e[unsafe_offset=depth] = Int32(end); st_p[unsafe_offset=depth] = Int32(my)
            depth += 1
            end = mid; parent = -1
    syncwarp()
    if lane == 0:
        roots[unsafe_offset=c] = Int32(lo); top_cl[unsafe_offset=c] = Int32(c)
    for k in range(lane, n, WARP_SIZE):
        var p = prim_ids[unsafe_offset=Int(a_dp) + Int(pslot[unsafe_offset=lo + Int(idx[unsafe_offset=k])])]
        var r = refs.unsafe_offset(3 * (lo + k))
        r[unsafe_offset=0] = Int32(p.id1); r[unsafe_offset=1] = Int32(p.id2); r[unsafe_offset=2] = Int32(p.materialIndex)


def _dev_ints(ctx: DeviceContext, values: List[Int]) raises -> DeviceBuffer[DType.uint8]:
    var h = unsafe_alloc[Int32](max(len(values), 1))
    for i in range(len(values)):
        h[unsafe_offset=i] = Int32(values[i])
    var b = ctx.enqueue_create_buffer[DType.uint8](4 * max(len(values), 1))
    ctx.enqueue_copy(b, h.unsafe_bitcast[UInt8]())
    ctx.synchronize()
    h.unsafe_free()
    return b^


def _host_ints(ctx: DeviceContext, mut b: DeviceBuffer[DType.uint8], n: Int) raises -> List[Int]:
    var h = unsafe_alloc[UInt8](len(b))
    ctx.enqueue_copy(h, b)
    ctx.synchronize()
    var out_l = List[Int](capacity=n)
    for i in range(n):
        out_l.append(Int(h.unsafe_bitcast[Int32]()[unsafe_offset=i]))
    h.unsafe_free()
    return out_l^


struct _SahTop(Movable):
    var top: List[BVH2Node]                   # depth-first; leaves name a cluster
    var cl_lo: List[Int]                      # cluster c: positions [cl_lo[c], cl_lo[c + 1]) of perm
    var perm: DeviceBuffer[DType.uint8]       # position -> piece

    def __init__(out self, var top: List[BVH2Node], var cl_lo: List[Int], var perm: DeviceBuffer[DType.uint8]):
        self.top = top^; self.cl_lo = cl_lo^; self.perm = perm^


def _floats(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.uint8]:
    var h = unsafe_alloc[Float32](max(len(values), 1))
    for i in range(len(values)):
        h[unsafe_offset=i] = values[i]
    var b = ctx.enqueue_create_buffer[DType.uint8](4 * max(len(values), 1))
    ctx.enqueue_copy(b, h.unsafe_bitcast[UInt8]())
    ctx.synchronize()
    h.unsafe_free()
    return b^


def _emit_top(nodes_lo: List[SIMD[DType.float32, 4]], nodes_hi: List[SIMD[DType.float32, 4]], kids: List[Int], cluster: List[Int], i: Int,
              mut top: List[BVH2Node]):
    var my = len(top)
    var lo = nodes_lo[i]; var hi = nodes_hi[i]
    top.append(BVH2Node(Point3f(lo[0], lo[1], lo[2]), Point3f(hi[0], hi[1], hi[2]), Int32(cluster[i]), Int32(1)))
    if cluster[i] >= 0:
        return
    _emit_top(nodes_lo, nodes_hi, kids, cluster, kids[2 * i], top)
    var right = len(top)
    _emit_top(nodes_lo, nodes_hi, kids, cluster, kids[2 * i + 1], top)
    top[my] = BVH2Node(Point3f(lo[0], lo[1], lo[2]), Point3f(hi[0], hi[1], hi[2]), Int32(right), Int32(0))


def _sah_top(ctx: DeviceContext, mut wbox: DeviceBuffer[DType.uint8], n: Int) raises -> _SahTop:
    """bvh._bvh_split's binned SAH from the top, a level of nodes at a time on the device (the host picks each split from
    the bins), down to ranges of at most CURVE_CLUSTER pieces: the clusters."""
    var perm = _buf[Int32](ctx, n)
    var perm2 = _buf[Int32](ctx, n)
    var seg = _buf[Int32](ctx, n)
    var ident = List[Int](capacity=n)
    var ph = unsafe_alloc[Int32](n)
    for i in range(n):
        ph[unsafe_offset=i] = Int32(i)
    ctx.enqueue_copy(perm, ph.unsafe_bitcast[UInt8]())
    ctx.synchronize()
    ph.unsafe_free()
    _ = ident^
    var lo = List[SIMD[DType.float32, 4]](); var hi = List[SIMD[DType.float32, 4]]()
    var kids = List[Int](); var cluster = List[Int](); var nstart = List[Int](); var nend = List[Int]()
    var pending: List[Int] = [0]
    lo.append(SIMD[DType.float32, 4](0)); hi.append(SIMD[DType.float32, 4](0)); kids.append(-1); kids.append(-1); cluster.append(-1)
    nstart.append(0); nend.append(n)
    while len(pending) > 0:
        var k = len(pending)
        var st = List[Int](); var en = List[Int]()
        for q in range(k):
            st.append(nstart[pending[q]]); en.append(nend[pending[q]])
        var sb = _dev_ints(ctx, st); var eb = _dev_ints(ctx, en)
        ctx.enqueue_function[_seg_kernel](_p[Int32](sb), _p[Int32](eb), Int64(k), Int64(n), _p[Int32](seg), grid_dim=ceildiv(n, TPB), block_dim=TPB)
        var nb = _buf[Int32](ctx, 12 * k)
        ctx.enqueue_function[_fill_kernel](_p[Int32](nb), Int64(12 * k), Int64(12), Int64(12), grid_dim=ceildiv(12 * k, TPB), block_dim=TPB)
        ctx.enqueue_function[_node_bounds_kernel](_p[Float32](wbox), _p[Int32](perm), _p[Int32](seg), Int64(n), _p[Int32](nb), grid_dim=ceildiv(n, TPB), block_dim=TPB)
        var nbh = _host_ints(ctx, nb, 12 * k)
        var ax = List[Int](); var cm = List[Float32](); var iv = List[Float32]()
        var any_split = False
        for q in range(k):
            var v = pending[q]
            var f = SIMD[DType.float32, 16](0)
            for c in range(12):
                f[c] = _unord(Int32(nbh[12 * q + c]))
            lo[v] = SIMD[DType.float32, 4](f[0], f[1], f[2], 0); hi[v] = SIMD[DType.float32, 4](-f[3], -f[4], -f[5], 0)
            var cmn = SIMD[DType.float32, 4](f[6], f[7], f[8], 0); var cmx = SIMD[DType.float32, 4](-f[9], -f[10], -f[11], 0)
            var ext = cmx - cmn
            var dim = 0
            if ext[1] > ext[0] and ext[1] >= ext[2]:
                dim = 1
            elif ext[2] > ext[0] and ext[2] > ext[1]:
                dim = 2
            var big = en[q] - st[q] > CURVE_CLUSTER
            any_split = any_split or big
            ax.append(dim if big else -1); cm.append(cmn[dim]); iv.append(Float32(1) / ext[dim] if ext[dim] > Float32(0) else Float32(0))
            if not big:
                cluster[v] = 0                           # numbered below, in position order
        if not any_split:
            break
        var axb = _dev_ints(ctx, ax); var cmb = _floats(ctx, cm); var ivb = _floats(ctx, iv)
        var bins = _buf[Int32](ctx, 7 * NB * k)
        ctx.enqueue_function[_fill_kernel](_p[Int32](bins), Int64(7 * NB * k), Int64(7), Int64(6), grid_dim=ceildiv(7 * NB * k, TPB), block_dim=TPB)
        ctx.enqueue_function[_bin_kernel](_p[Float32](wbox), _p[Int32](perm), _p[Int32](seg), Int64(n), _p[Int32](axb), _p[Float32](cmb), _p[Float32](ivb),
                                          _p[Int32](bins), grid_dim=ceildiv(n, TPB), block_dim=TPB)
        var bh = _host_ints(ctx, bins, 7 * NB * k)
        var split = List[Int](); var nleft = List[Int]()
        var next = List[Int]()
        for q in range(k):
            var v = pending[q]
            if ax[q] < 0:
                split.append(-1); nleft.append(0)
                continue
            var cnt = en[q] - st[q]
            var best = -1
            var best_cost = Float32(3.0e38)
            var nb_left = 0
            if iv[q] > Float32(0):
                for sp in range(NB - 1):
                    var cost = Float32(0)
                    var c_below = 0
                    for side in range(2):
                        var bl = SIMD[DType.float32, 4](Float32(1e30)); var bu = SIMD[DType.float32, 4](Float32(-1e30))
                        var c = 0
                        for bkt in range(0 if side == 0 else sp + 1, sp + 1 if side == 0 else NB):
                            var o = (q * NB + bkt) * 7
                            if bh[o + 6] == 0:
                                continue
                            c += bh[o + 6]
                            bl = min(bl, SIMD[DType.float32, 4](_unord(Int32(bh[o])), _unord(Int32(bh[o + 1])), _unord(Int32(bh[o + 2])), 0))
                            bu = max(bu, SIMD[DType.float32, 4](-_unord(Int32(bh[o + 3])), -_unord(Int32(bh[o + 4])), -_unord(Int32(bh[o + 5])), 0))
                        if c > 0:
                            var e = max(bu - bl, SIMD[DType.float32, 4](0))
                            cost += Float32(c) * Float32(2.0) * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0])
                        if side == 0:
                            c_below = c
                    if cost < best_cost:
                        best_cost = cost; best = sp; nb_left = c_below
            if best < 0 or nb_left == 0 or nb_left == cnt:
                best = -1; nb_left = cnt // 2                 # degenerate: a median cut by position
            split.append(best); nleft.append(nb_left)
            var l = len(lo)
            for side in range(2):
                lo.append(SIMD[DType.float32, 4](0)); hi.append(SIMD[DType.float32, 4](0)); kids.append(-1); kids.append(-1); cluster.append(-1)
                nstart.append(st[q] if side == 0 else st[q] + nb_left); nend.append(st[q] + nb_left if side == 0 else en[q])
                next.append(l + side)
            kids[2 * v] = l; kids[2 * v + 1] = l + 1
        var spb = _dev_ints(ctx, split); var nlb = _dev_ints(ctx, nleft)
        var cur = _buf[Int32](ctx, 2 * k)
        ctx.enqueue_memset(cur, 0)
        ctx.enqueue_function[_partition_kernel](_p[Float32](wbox), _p[Int32](perm), _p[Int32](perm2), _p[Int32](seg), Int64(n), _p[Int32](axb), _p[Float32](cmb),
                                                _p[Float32](ivb), _p[Int32](spb), _p[Int32](sb), _p[Int32](nlb), _p[Int32](cur), grid_dim=ceildiv(n, TPB), block_dim=TPB)
        ctx.synchronize()
        swap(perm, perm2)
        _ = sb^; _ = eb^; _ = nb^; _ = axb^; _ = cmb^; _ = ivb^; _ = bins^; _ = spb^; _ = nlb^; _ = cur^
        pending = next^                               # children of a node follow each other: still in position order
    # clusters, numbered in position order
    var order = List[Int]()
    for v in range(len(cluster)):
        if cluster[v] >= 0:
            order.append(v)
    _sort_by(order, nstart)
    var cl_lo = List[Int]()
    for c in range(len(order)):
        cluster[order[c]] = c
        cl_lo.append(nstart[order[c]])
    cl_lo.append(n)
    var top = List[BVH2Node]()
    _emit_top(lo, hi, kids, cluster, 0, top)
    _ = perm2^; _ = seg^
    return _SahTop(top^, cl_lo^, perm^)


def _sort_by(mut idx: List[Int], key: List[Int]):
    """Insertion of a few thousand cluster indices by start (already nearly sorted)."""
    for i in range(1, len(idx)):
        var v = idx[i]
        var j = i - 1
        while j >= 0 and key[idx[j]] > key[v]:
            idx[j + 1] = idx[j]
            j -= 1
        idx[j + 1] = v


def gpu_build_curve_bvh(ctx: DeviceContext, prim_ids: Pointer[PrimId, MutUntrackedOrigin], curves: Pointer[Curve, MutUntrackedOrigin],
                        a: Int, n: Int) raises -> CurveBvhDev:
    """The curve PrimIds [a, a + n) cut by binned SAH into clusters, each with a binned-SAH BVH in its fiber frame."""
    var wbox = _buf[Float32](ctx, 6 * n)
    ctx.enqueue_function[_world_boxes_kernel](prim_ids, curves, Int64(a), Int64(n), _p[Float32](wbox), grid_dim=ceildiv(n, TPB), block_dim=TPB)
    var t = _sah_top(ctx, wbox, n)
    var nc = len(t.cl_lo) - 1
    if nc > 0xFFFF:
        raise Error("curve clusters: more than 65535")
    var n_top = len(t.top)
    var top = _buf[UInt32](ctx, 4 * n_top)
    var th = unsafe_alloc[UInt32](4 * n_top)
    var r0 = t.top[0]
    var qs = max((SIMD[DType.float32, 4](r0.max.x, r0.max.y, r0.max.z, 0) - SIMD[DType.float32, 4](r0.min.x, r0.min.y, r0.min.z, 0)) / Q_STEPS,
                 SIMD[DType.float32, 4](Float32(1e-30)))
    var qo = SIMD[DType.float32, 4](r0.min.x, r0.min.y, r0.min.z, 0) - Float32(200) * qs
    for k in range(n_top):
        var nd = t.top[k]
        _put_qnode(th, k, nd.min, nd.max, qo, qs, Q_LEAF | (UInt32(nd.count) << 16) | UInt32(nd.offset) if nd.count > Int32(0) else UInt32(nd.offset))
    ctx.enqueue_copy(top, th.unsafe_bitcast[UInt8]())
    var cl_lo = _dev_ints(ctx, t.cl_lo)
    th.unsafe_free()
    var pslot = _buf[Int32](ctx, 1)
    swap(pslot, t.perm)
    _ = wbox^
    var frames = _buf[Float32](ctx, FS * nc)
    ctx.enqueue_function[_frames_kernel](prim_ids, curves, Int64(a), _p[Int32](pslot), _p[Int32](cl_lo), _p[Float32](frames), grid_dim=nc, block_dim=TPB)
    var widx = _buf[Int32](ctx, n); var wmin = _buf[Float32](ctx, 3 * n); var wmax = _buf[Float32](ctx, 3 * n)
    ctx.enqueue_function[_frame_boxes_kernel](prim_ids, curves, Int64(a), _p[Int32](pslot), _p[Int32](cl_lo), Int64(nc), _p[Float32](frames), Int64(n),
                                              _p[Int32](widx), _p[Float32](wmin), _p[Float32](wmax), grid_dim=ceildiv(n, TPB), block_dim=TPB)
    var nodes = _buf[UInt32](ctx, 8 * n)
    var roots = _buf[Int32](ctx, nc)
    var refs = _buf[Int32](ctx, 3 * n)
    var top_cl = _buf[Int32](ctx, nc)
    var wtmp = _buf[Int32](ctx, n)
    ctx.enqueue_function[_sah_clusters_kernel](_p[Int32](cl_lo), Int64(nc), _p[Int32](widx), _p[Int32](wtmp), _p[Float32](wmin), _p[Float32](wmax),
                                               _p[UInt32](nodes), _p[Float32](frames), _p[Int32](roots), _p[Int32](pslot), Int64(a), prim_ids, _p[Int32](refs), _p[Int32](top_cl),
                                               grid_dim=ceildiv(nc, CW), block_dim=CW * WARP_SIZE)
    ctx.synchronize()
    _ = widx^; _ = wtmp^; _ = wmin^; _ = wmax^; _ = pslot^; _ = cl_lo^
    var bufs = List[DeviceBuffer[DType.uint8]]()
    bufs.append(top^); bufs.append(top_cl^); bufs.append(frames^); bufs.append(roots^); bufs.append(nodes^); bufs.append(refs^)
    var cb = CurveBvh(_p[UInt32](bufs[0]), _p[Int32](bufs[1]), _p[Float32](bufs[2]), _p[Int32](bufs[3]), _p[UInt32](bufs[4]), _p[Int32](bufs[5]), Int32(nc),
                      SIMD[DType.float32, 8](qo[0], qo[1], qo[2], qs[0], qs[1], qs[2], 0, 0))
    return CurveBvhDev(cb, bufs^)
