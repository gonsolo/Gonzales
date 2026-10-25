"""Curve leaves in clusters, each with a BVH built in a frame along its fibers (CurveBvh in curves.mojo), for shadow rays
next to the RT cores. Fur is thin, nearly straight pieces at every slant; their world-axis boxes are mostly empty. A
cluster rotates the ray once; distances are the same in both frames, so one tmax serves the box and the curve tests.
Blender lion: shadow pass 16.5 -> 8.5 s. Closest hits stay on the scene BVH's curve subtree, which measured faster."""
from max.algorithm import parallelize
from std.math import sqrt
from std.memory import stack_allocation
from std.sys.info import is_gpu
from std.memory.alloc import unsafe_alloc
from .bvh import BVH2Node, build_bvh2, _node_box_hit, _node_oc, _shadow_is_null_material
from .curves import Curve, CurveBvh, intersect_curve, curve_piece_endpoints, _curve_perp_axis
from .geometry import Vec3f, cross, dot
from .materials import Material
from .primitives import PrimId, Ray

comptime CURVE_CLUSTER = 4096       # pieces a cluster covers at most


struct CurveBvhBuild(Movable):
    var bvh: CurveBvh                # host arrays
    var n_top: Int
    var n_nodes: Int
    var n_prims: Int

    def __init__(out self, bvh: CurveBvh, n_top: Int, n_nodes: Int, n_prims: Int):
        self.bvh = bvh; self.n_top = n_top; self.n_nodes = n_nodes; self.n_prims = n_prims


def _leaf_range(nodes: Pointer[BVH2Node, MutUntrackedOrigin], i: Int) -> Tuple[Int, Int]:
    """The PrimIds below node i: a depth-first BVH2's subtree covers one contiguous range."""
    var a = i
    while nodes[unsafe_offset=a].count == Int32(0):
        a += 1
    var b = i
    while nodes[unsafe_offset=b].count == Int32(0):
        b = Int(nodes[unsafe_offset=b].offset)
    return (Int(nodes[unsafe_offset=a].offset), Int(nodes[unsafe_offset=b].offset) + Int(nodes[unsafe_offset=b].count))


def _cut(nodes: Pointer[BVH2Node, MutUntrackedOrigin], i: Int, cluster: Int, mut lo: List[Int32], mut hi: List[Int32], mut top: List[BVH2Node]):
    """Node i becomes a cluster when it covers at most `cluster` pieces; above the cut the scene BVH's own nodes are kept as
    the top level (depth-first, as there), its leaves naming one cluster each."""
    var r = _leaf_range(nodes, i)
    var nd = nodes[unsafe_offset=i]
    var my = len(top)
    if nd.count != Int32(0) or r[1] - r[0] <= cluster:
        top.append(BVH2Node(nd.min, nd.max, Int32(len(lo)), Int32(1)))
        lo.append(Int32(r[0])); hi.append(Int32(r[1]))
        return
    top.append(nd)
    _cut(nodes, i + 1, cluster, lo, hi, top)
    var right = len(top)
    _cut(nodes, Int(nd.offset), cluster, lo, hi, top)
    top[my] = BVH2Node(nd.min, nd.max, Int32(right), Int32(0))


# The scene BVH's node loads, but invariant only on the GPU: the optimized CPU build moved such loads ahead of the tests
# that make their index valid (a crash in test_curve_bvh, none unoptimized).
@always_inline
def _node_lo(nf: Pointer[Float32, MutUntrackedOrigin], idx: Int) -> SIMD[DType.float32, 4]:
    return nf.unsafe_load[width=4, alignment=16 if is_gpu() else 4, invariant=is_gpu()](idx * 8)

@always_inline
def _node_hi(nf: Pointer[Float32, MutUntrackedOrigin], idx: Int) -> SIMD[DType.float32, 4]:
    return nf.unsafe_load[width=4, alignment=16 if is_gpu() else 4, invariant=is_gpu()](idx * 8 + 4)


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
        var curve = curves[unsafe_offset=Int(p.id1)]
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


def build_curve_bvh(nodes: Pointer[BVH2Node, MutUntrackedOrigin], prim_ids: Pointer[PrimId, MutUntrackedOrigin],
                    curves: Pointer[Curve, MutUntrackedOrigin], root: Int) -> CurveBvhBuild:
    """Clusters = the subtrees of the scene BVH's curve subtree (at `root`) with at most CURVE_CLUSTER pieces."""
    var lo = List[Int32](); var hi = List[Int32]()
    var top_l = List[BVH2Node]()
    _cut(nodes, root, CURVE_CLUSTER, lo, hi, top_l)
    var nc = len(lo)
    var lo_p = unsafe_alloc[Int32](nc); var hi_p = unsafe_alloc[Int32](nc)
    for c in range(nc):
        lo_p[unsafe_offset=c] = lo[c]; hi_p[unsafe_offset=c] = hi[c]
    var frames = unsafe_alloc[Float32](9 * nc)
    var sub_nodes = unsafe_alloc[Pointer[BVH2Node, MutUntrackedOrigin]](nc)
    var sub_order = unsafe_alloc[Pointer[Int32, MutUntrackedOrigin]](nc)
    var sub_n = unsafe_alloc[Int](nc)

    def one(c: Int) {imm}:
        var a = Int(lo_p[unsafe_offset=c]); var n = Int(hi_p[unsafe_offset=c]) - a
        var f = frames.unsafe_offset(9 * c)
        _fiber_frame(prim_ids, curves, a, n, f)
        var b = unsafe_alloc[Float32](6 * n)
        for k in range(n):
            var p = prim_ids[unsafe_offset=a + k]
            var curve = curves[unsafe_offset=Int(p.id1)]
            for q in range(3):
                b[unsafe_offset=6 * k + q] = Float32(1e30); b[unsafe_offset=6 * k + 3 + q] = Float32(-1e30)
            for piece in range(Int(p.id2) // 8, Int(p.id2) // 8 + Int(p.id2) % 8):
                var e = curve_piece_endpoints(curve, piece)
                var r = max(e[2], e[3])
                var f0 = _in_frame(f, e[0]); var f1 = _in_frame(f, e[1])
                for q in range(3):
                    b[unsafe_offset=6 * k + q] = min(b[unsafe_offset=6 * k + q], min(f0[q], f1[q]) - r)
                    b[unsafe_offset=6 * k + 3 + q] = max(b[unsafe_offset=6 * k + 3 + q], max(f0[q], f1[q]) + r)
        var sn = unsafe_alloc[BVH2Node](2 * n + 2)
        var so = unsafe_alloc[Int32](max(n, 1))
        sub_n[unsafe_offset=c] = Int(build_bvh2(b, Int32(n), sn, so, parallel=False))
        sub_nodes[unsafe_offset=c] = sn.unsafe_origin_cast[MutUntrackedOrigin]()
        sub_order[unsafe_offset=c] = so.unsafe_origin_cast[MutUntrackedOrigin]()
        b.unsafe_free()

    parallelize(one, nc)
    var nbase = unsafe_alloc[Int](nc + 1); var pbase = unsafe_alloc[Int](nc + 1)
    nbase[unsafe_offset=0] = 0; pbase[unsafe_offset=0] = 0
    for c in range(nc):
        nbase[unsafe_offset=c + 1] = nbase[unsafe_offset=c] + sub_n[unsafe_offset=c]
        pbase[unsafe_offset=c + 1] = pbase[unsafe_offset=c] + Int(hi_p[unsafe_offset=c] - lo_p[unsafe_offset=c])
    var out_nodes = unsafe_alloc[BVH2Node](max(nbase[unsafe_offset=nc], 1))
    var prims = unsafe_alloc[Int32](max(pbase[unsafe_offset=nc], 1))
    var roots = unsafe_alloc[Int32](max(nc, 1))

    def copy(c: Int) {imm}:
        var nb = nbase[unsafe_offset=c]; var pb = pbase[unsafe_offset=c]
        for j in range(sub_n[unsafe_offset=c]):
            var nd = sub_nodes[unsafe_offset=c][unsafe_offset=j]
            nd.offset += Int32(nb) if nd.count == Int32(0) else Int32(pb)
            out_nodes[unsafe_offset=nb + j] = nd
        for k in range(Int(hi_p[unsafe_offset=c] - lo_p[unsafe_offset=c])):
            prims[unsafe_offset=pb + k] = lo_p[unsafe_offset=c] + sub_order[unsafe_offset=c][unsafe_offset=k]
        roots[unsafe_offset=c] = Int32(nb)
        sub_nodes[unsafe_offset=c].unsafe_free(); sub_order[unsafe_offset=c].unsafe_free()

    parallelize(copy, nc)
    var n_top = len(top_l)
    var top = unsafe_alloc[BVH2Node](max(n_top, 1))
    var top_cl = unsafe_alloc[Int32](max(nc, 1))
    for k in range(n_top):
        top[unsafe_offset=k] = top_l[k]
    for c in range(nc):
        top_cl[unsafe_offset=c] = Int32(c)
    var n_nodes = nbase[unsafe_offset=nc]; var n_prims = pbase[unsafe_offset=nc]
    lo_p.unsafe_free(); hi_p.unsafe_free(); sub_nodes.unsafe_free(); sub_order.unsafe_free(); sub_n.unsafe_free()
    nbase.unsafe_free(); pbase.unsafe_free()
    var cb = CurveBvh(top.unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin](), top_cl.unsafe_origin_cast[MutUntrackedOrigin](),
                      frames.unsafe_origin_cast[MutUntrackedOrigin](), roots.unsafe_origin_cast[MutUntrackedOrigin](),
                      out_nodes.unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin](), prims.unsafe_origin_cast[MutUntrackedOrigin](), Int32(nc))
    return CurveBvhBuild(cb, n_top, n_nodes, n_prims)


def free_curve_bvh(cb: CurveBvh):
    cb.top.unsafe_free(); cb.top_cl.unsafe_free(); cb.frames.unsafe_free(); cb.roots.unsafe_free(); cb.nodes.unsafe_free(); cb.prims.unsafe_free()


def curve_bvh_occluded(cb: CurveBvh, prim_ids: Pointer[PrimId, MutUntrackedOrigin], curves: Pointer[Curve, MutUntrackedOrigin],
                       materials: Pointer[Material, MutUntrackedOrigin], ray: Ray, tmax: Float32) -> Bool:
    """One loop and one stack over both levels: a top leaf names a cluster, whose BVH is walked with the ray in its frame;
    a -1 on the stack marks the way back to the top level. Pieces are tested with the world ray."""
    var wo = Vec3f(ray.origin.x, ray.origin.y, ray.origin.z)
    var wd = ray.direction
    var w_rdir = Vec3f(Float32(1.0) / wd.x, Float32(1.0) / wd.y, Float32(1.0) / wd.z)
    var o = wo
    var rdir = w_rdir
    var nx = rdir.x >= Float32(0.0); var ny = rdir.y >= Float32(0.0); var nz = rdir.z >= Float32(0.0)
    var nodes_f = cb.top
    var in_cluster = False
    var stack = stack_allocation[64, Int32]()
    var to_visit = 0
    var current = 0
    var cur_oc = _node_oc(_node_hi(nodes_f, 0))
    while True:
        var pop = False
        if cur_oc[1] > 0:
            var off = Int(cur_oc[0])
            if not in_cluster:
                var c = Int(cb.top_cl[unsafe_offset=off])
                var f = cb.frames.unsafe_offset(9 * c)
                var co = _in_frame(f, wo)
                var cd = _in_frame(f, wd)
                var crdir = Vec3f(Float32(1.0) / cd.x, Float32(1.0) / cd.y, Float32(1.0) / cd.z)
                var cnx = crdir.x >= Float32(0.0); var cny = crdir.y >= Float32(0.0); var cnz = crdir.z >= Float32(0.0)
                var r = Int(cb.roots[unsafe_offset=c])
                var rhi = _node_hi(cb.nodes, r)
                if _node_box_hit(_node_lo(cb.nodes, r), rhi, crdir, co, cnx, cny, cnz, tmax)[0] and to_visit < 64:
                    stack[unsafe_offset=to_visit] = Int32(-1)
                    to_visit += 1
                    o = co; rdir = crdir; nx = cnx; ny = cny; nz = cnz
                    nodes_f = cb.nodes; in_cluster = True
                    current = r; cur_oc = _node_oc(rhi)
                    continue
                pop = True
            else:
                for j in range(Int(cur_oc[1])):
                    var pi = Int(cb.prims[unsafe_offset=off + j])
                    var prim = prim_ids[unsafe_offset=pi]
                    if _shadow_is_null_material(materials, prim.materialIndex):
                        continue
                    var curve = curves[unsafe_offset=Int(prim.id1)]
                    if intersect_curve(wo, wd, curve, Int(prim.id2) // 8, Int(prim.id2) % 8, tmax)[0]:
                        return True
                pop = True
        else:
            var li = current + 1
            var ri = Int(cur_oc[0])
            var lhi = _node_hi(nodes_f, li); var rhi = _node_hi(nodes_f, ri)
            var lh = _node_box_hit(_node_lo(nodes_f, li), lhi, rdir, o, nx, ny, nz, tmax)
            var rh = _node_box_hit(_node_lo(nodes_f, ri), rhi, rdir, o, nx, ny, nz, tmax)
            if lh[0] and rh[0]:
                var near_left = lh[1] <= rh[1]
                current = li if near_left else ri
                cur_oc = _node_oc(lhi if near_left else rhi)
                if to_visit < 64:
                    stack[unsafe_offset=to_visit] = Int32(ri if near_left else li)
                    to_visit += 1
            elif lh[0]:
                current = li; cur_oc = _node_oc(lhi)
            elif rh[0]:
                current = ri; cur_oc = _node_oc(rhi)
            else:
                pop = True
        if pop:
            if to_visit == 0:
                break
            to_visit -= 1
            current = Int(stack[unsafe_offset=to_visit])
            if current < 0:                                    # back to the top level
                o = wo; rdir = w_rdir; nx = rdir.x >= Float32(0.0); ny = rdir.y >= Float32(0.0); nz = rdir.z >= Float32(0.0)
                nodes_f = cb.top; in_cluster = False
                if to_visit == 0:
                    break
                to_visit -= 1
                current = Int(stack[unsafe_offset=to_visit])
            cur_oc = _node_oc(_node_hi(nodes_f, current))
    return False
