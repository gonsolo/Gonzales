"""BVH2 builds on the GPU: H-PLOC (Benthin et al., HPG 2024) over sorted Morton codes, written out in bvh.build_bvh2's
depth-first layout. Segments (contiguous primitive ranges) get trees of their own: the segment id sits above the Morton
bits, so each segment is one node of the radix tree, and its clusters merge down to one root without leaving it."""
from std.math import ceildiv
from std.atomic import Atomic, Ordering
from std.bit import count_trailing_zeros, pop_count
from std.memory import bitcast, stack_allocation
from std.memory.alloc import unsafe_alloc
from std.sys.info import size_of
from max.gpu import block_idx, block_dim, thread_idx, lane_id, WARP_SIZE
from max.gpu.sync import barrier, syncwarp
from max.gpu.intrinsics import threadfence
from max.gpu.memory import AddressSpace
from max.gpu.primitives.warp import shuffle_idx, vote
from max.gpu.host import DeviceContext, DeviceBuffer
from .bvh import BVH2Node
from .geometry import Point3f

comptime TPB = 256
comptime THRESH = WARP_SIZE // 2        # clusters a range hands its parent
comptime RADIUS = 8                      # nearest-neighbour window
comptime ITEMS = 8                       # radix sort: keys per thread
comptime MaskT = DType.uint32 if WARP_SIZE == 32 else DType.uint64
comptime C_TRAV = Float32(0.5)           # bvh._bvh_split's costs: node 0.5, primitive 1
comptime U16x16 = SIMD[DType.uint16, 16]


struct GpuBvh(Movable):
    var nodes: DeviceBuffer[DType.uint8]     # BVH2Node, each segment's tree depth-first after the previous one
    var order: DeviceBuffer[DType.uint8]     # Int32: leaf slot -> input primitive
    var roots: DeviceBuffer[DType.uint8]     # Int32 per segment: its root in nodes
    var below: DeviceBuffer[DType.uint8]     # Int32 per node: primitives below (with `ranges`)
    var first: DeviceBuffer[DType.uint8]     # Int32 per node: its first leaf slot (with `ranges`)
    var n_nodes: Int

    def __init__(out self, var nodes: DeviceBuffer[DType.uint8], var order: DeviceBuffer[DType.uint8], var roots: DeviceBuffer[DType.uint8],
                 var below: DeviceBuffer[DType.uint8], var first: DeviceBuffer[DType.uint8], n_nodes: Int):
        self.nodes = nodes^; self.order = order^; self.roots = roots^; self.below = below^; self.first = first^; self.n_nodes = n_nodes


def _buf[T: AnyType](ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.uint8]:
    return ctx.enqueue_create_buffer[DType.uint8](max(n, 1) * size_of[T]())


@always_inline
def _p[T: AnyType](mut b: DeviceBuffer[DType.uint8]) -> Pointer[T, MutUntrackedOrigin]:
    return b.unsafe_ptr().unsafe_bitcast[T]().unsafe_origin_cast[MutUntrackedOrigin]()


@always_inline
def _tid() -> Int:
    return Int(block_idx.x * block_dim.x + thread_idx.x)


@always_inline
def _ord(f: Float32) -> Int32:
    """Float order as Int32 order, for atomic min / max."""
    var i = bitcast[DType.int32](f)
    return i if i >= Int32(0) else i ^ Int32(0x7FFFFFFF)


@always_inline
def _unord(i: Int32) -> Float32:
    return bitcast[DType.float32](i if i >= Int32(0) else i ^ Int32(0x7FFFFFFF))


@always_inline
def _half_area(lx: Float32, ly: Float32, lz: Float32, hx: Float32, hy: Float32, hz: Float32) -> Float32:
    var dx = max(hx - lx, Float32(0)); var dy = max(hy - ly, Float32(0)); var dz = max(hz - lz, Float32(0))
    return dx * dy + dy * dz + dz * dx


@always_inline
def _seg_of(seg_start: Pointer[Int32, MutUntrackedOrigin], nseg: Int, i: Int) -> Int:
    var a = 0
    var b = nseg - 1
    while a < b:
        var m = (a + b + 1) // 2
        if Int(seg_start[unsafe_offset=m]) <= i:
            a = m
        else:
            b = m - 1
    return a


# ---- exclusive scan of Int32 ----

def _scan_kernel(data: Pointer[Int32, MutUntrackedOrigin], n_dp: Int64, sums: Pointer[Int32, MutUntrackedOrigin]):
    var n = Int(n_dp)
    var sh = stack_allocation[TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var k = Int(thread_idx.x)
    var base = Int(block_idx.x) * TPB * 4 + k * 4
    var v = SIMD[DType.int32, 4](0)
    for j in range(4):
        if base + j < n:
            v[j] = data[unsafe_offset=base + j]
    var own = v[0] + v[1] + v[2] + v[3]
    var acc = own
    sh[unsafe_offset=k] = acc
    barrier()
    var off = 1
    while off < TPB:
        var add = sh[unsafe_offset=k - off] if k >= off else Int32(0)
        barrier()
        acc += add
        sh[unsafe_offset=k] = acc
        barrier()
        off *= 2
    var run = acc - own
    for j in range(4):
        if base + j < n:
            data[unsafe_offset=base + j] = run
        run += v[j]
    if k == TPB - 1:
        sums[unsafe_offset=Int(block_idx.x)] = acc


def _scan_add_kernel(data: Pointer[Int32, MutUntrackedOrigin], n_dp: Int64, sums: Pointer[Int32, MutUntrackedOrigin]):
    var i = _tid()
    if i < Int(n_dp):
        data[unsafe_offset=i] += sums[unsafe_offset=i // (TPB * 4)]


def _exclusive_scan(ctx: DeviceContext, data: Pointer[Int32, MutUntrackedOrigin], n: Int) raises:
    var nb = ceildiv(n, TPB * 4)
    var sums = _buf[Int32](ctx, nb)
    ctx.enqueue_function[_scan_kernel](data, Int64(n), _p[Int32](sums), grid_dim=nb, block_dim=TPB)
    if nb > 1:
        _exclusive_scan(ctx, _p[Int32](sums), nb)
        ctx.enqueue_function[_scan_add_kernel](data, Int64(n), _p[Int32](sums), grid_dim=ceildiv(n, TPB), block_dim=TPB)
    ctx.synchronize()
    _ = sums^                                            # a buffer goes back to the pool when dropped, run or not


# ---- keys: centroid bounds per segment, Morton codes ----

def _bounds_init_kernel(sb: Pointer[Int32, MutUntrackedOrigin], nseg_dp: Int64):
    var t = _tid()
    if t < Int(nseg_dp) * 6:
        sb[unsafe_offset=t] = _ord(Float32(1e30)) if t % 6 < 3 else _ord(Float32(-1e30))


def _bounds_kernel(boxes: Pointer[Float32, MutUntrackedOrigin], n_dp: Int64, seg_start: Pointer[Int32, MutUntrackedOrigin], nseg_dp: Int64,
                   sb: Pointer[Int32, MutUntrackedOrigin]):
    """Centroid bounds per segment; a block inside one segment reduces before its atomics."""
    var n = Int(n_dp); var nseg = Int(nseg_dp)
    var t = _tid()
    var lo = SIMD[DType.float32, 4](Float32(1e30))
    var hi = SIMD[DType.float32, 4](Float32(-1e30))
    if t < n:
        var b = boxes.unsafe_offset(6 * t)
        lo = SIMD[DType.float32, 4](b[unsafe_offset=0] + b[unsafe_offset=3], b[unsafe_offset=1] + b[unsafe_offset=4],
                                    b[unsafe_offset=2] + b[unsafe_offset=5], Float32(0)) * Float32(0.5)
        hi = lo
    var first = Int(block_idx.x) * TPB
    var s_first = _seg_of(seg_start, nseg, first)
    if s_first == _seg_of(seg_start, nseg, min(first + TPB, n) - 1):
        var sh = stack_allocation[TPB * 8, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
        var k = Int(thread_idx.x)
        sh.unsafe_store(8 * k, lo); sh.unsafe_store(8 * k + 4, hi)
        barrier()
        var w = TPB // 2
        while w > 0:
            if k < w:
                sh.unsafe_store(8 * k, min(sh.unsafe_load[width=4](8 * k), sh.unsafe_load[width=4](8 * (k + w))))
                sh.unsafe_store(8 * k + 4, max(sh.unsafe_load[width=4](8 * k + 4), sh.unsafe_load[width=4](8 * (k + w) + 4)))
            barrier()
            w //= 2
        if k == 0:
            lo = sh.unsafe_load[width=4](0); hi = sh.unsafe_load[width=4](4)
            for q in range(3):
                _ = Atomic.min[ordering=Ordering.RELAXED](sb.unsafe_offset(6 * s_first + q), _ord(lo[q]))
                _ = Atomic.max[ordering=Ordering.RELAXED](sb.unsafe_offset(6 * s_first + 3 + q), _ord(hi[q]))
    elif t < n:
        var s = _seg_of(seg_start, nseg, t)
        for q in range(3):
            _ = Atomic.min[ordering=Ordering.RELAXED](sb.unsafe_offset(6 * s + q), _ord(lo[q]))
            _ = Atomic.max[ordering=Ordering.RELAXED](sb.unsafe_offset(6 * s + 3 + q), _ord(hi[q]))


@always_inline
def _expand(v: UInt32) -> UInt32:
    var x = v & UInt32(0x3FF)
    x = (x | (x << 16)) & UInt32(0x030000FF)
    x = (x | (x << 8)) & UInt32(0x0300F00F)
    x = (x | (x << 4)) & UInt32(0x030C30C3)
    x = (x | (x << 2)) & UInt32(0x09249249)
    return x


def _morton_kernel(boxes: Pointer[Float32, MutUntrackedOrigin], n_dp: Int64, seg_start: Pointer[Int32, MutUntrackedOrigin], nseg_dp: Int64,
                   sb: Pointer[Int32, MutUntrackedOrigin], bits_dp: Int64, keys: Pointer[UInt32, MutUntrackedOrigin], vals: Pointer[Int32, MutUntrackedOrigin]):
    var t = _tid()
    if t >= Int(n_dp):
        return
    var bits = Int(bits_dp)
    var s = _seg_of(seg_start, Int(nseg_dp), t)
    var b = boxes.unsafe_offset(6 * t)
    var scale = Float32((1 << bits) - 1)
    var m = UInt32(0)
    for q in range(3):
        var lo = _unord(sb[unsafe_offset=6 * s + q]); var hi = _unord(sb[unsafe_offset=6 * s + 3 + q])
        var c = Float32(0.5) * (b[unsafe_offset=q] + b[unsafe_offset=q + 3])
        var x = (c - lo) / (hi - lo) * scale if hi > lo else Float32(0)
        m |= _expand(UInt32(Int(min(max(x, Float32(0)), scale)))) << UInt32(2 - q)
    keys[unsafe_offset=t] = UInt32(UInt64(s) << UInt64(3 * bits)) | m
    vals[unsafe_offset=t] = Int32(t)


# ---- stable LSD radix sort, four bits a pass ----

@always_inline
def _digits(keys: Pointer[UInt32, MutUntrackedOrigin], n: Int, base: Int, shift: Int) -> U16x16:
    var lanes = U16x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    var cnt = U16x16(0)
    for j in range(ITEMS):
        if base + j < n:
            var d = UInt16((keys[unsafe_offset=base + j] >> UInt32(shift)) & UInt32(15))
            cnt += lanes.eq(U16x16(d)).cast[DType.uint16]()
    return cnt


@always_inline
def _block_scan16(v: U16x16) -> Tuple[U16x16, U16x16]:
    """(exclusive prefix over the block's threads, block total), per digit."""
    var sh = stack_allocation[TPB * 16, Scalar[DType.uint16], address_space=AddressSpace.SHARED]()
    var k = Int(thread_idx.x)
    var acc = v
    sh.unsafe_store(16 * k, acc)
    barrier()
    var off = 1
    while off < TPB:
        var add = sh.unsafe_load[width=16](16 * (k - off)) if k >= off else U16x16(0)
        barrier()
        acc += add
        sh.unsafe_store(16 * k, acc)
        barrier()
        off *= 2
    return (acc - v, sh.unsafe_load[width=16](16 * (TPB - 1)))


def _radix_hist_kernel(keys: Pointer[UInt32, MutUntrackedOrigin], n_dp: Int64, shift_dp: Int64, hist: Pointer[Int32, MutUntrackedOrigin], nb_dp: Int64):
    var k = Int(thread_idx.x)
    var r = _block_scan16(_digits(keys, Int(n_dp), Int(block_idx.x) * TPB * ITEMS + k * ITEMS, Int(shift_dp)))
    if k < 16:
        hist[unsafe_offset=k * Int(nb_dp) + Int(block_idx.x)] = Int32(r[1][k])


def _radix_scatter_kernel(keys: Pointer[UInt32, MutUntrackedOrigin], vals: Pointer[Int32, MutUntrackedOrigin], keys_out: Pointer[UInt32, MutUntrackedOrigin],
                          vals_out: Pointer[Int32, MutUntrackedOrigin], n_dp: Int64, shift_dp: Int64, hist: Pointer[Int32, MutUntrackedOrigin], nb_dp: Int64):
    var n = Int(n_dp); var shift = Int(shift_dp); var nb = Int(nb_dp)
    var base = Int(block_idx.x) * TPB * ITEMS + Int(thread_idx.x) * ITEMS
    var run = _block_scan16(_digits(keys, n, base, shift))[0]
    var lanes = U16x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    for j in range(ITEMS):
        if base + j < n:
            var key = keys[unsafe_offset=base + j]
            var d = Int((key >> UInt32(shift)) & UInt32(15))
            var pos = Int(hist[unsafe_offset=d * nb + Int(block_idx.x)]) + Int(run[d])
            run += lanes.eq(U16x16(UInt16(d))).cast[DType.uint16]()
            keys_out[unsafe_offset=pos] = key
            vals_out[unsafe_offset=pos] = vals[unsafe_offset=base + j]


def _radix_sort(ctx: DeviceContext, mut keys: DeviceBuffer[DType.uint8], mut vals: DeviceBuffer[DType.uint8], n: Int, key_bits: Int) raises:
    var nb = ceildiv(n, TPB * ITEMS)
    var hist = _buf[Int32](ctx, 16 * nb)
    var keys2 = _buf[UInt32](ctx, n)
    var vals2 = _buf[Int32](ctx, n)
    var shift = 0
    while shift < key_bits:
        ctx.enqueue_function[_radix_hist_kernel](_p[UInt32](keys), Int64(n), Int64(shift), _p[Int32](hist), Int64(nb), grid_dim=nb, block_dim=TPB)
        _exclusive_scan(ctx, _p[Int32](hist), 16 * nb)
        ctx.enqueue_function[_radix_scatter_kernel](_p[UInt32](keys), _p[Int32](vals), _p[UInt32](keys2), _p[Int32](vals2), Int64(n), Int64(shift),
                                                    _p[Int32](hist), Int64(nb), grid_dim=nb, block_dim=TPB)
        swap(keys, keys2)
        swap(vals, vals2)
        shift += 4
    ctx.synchronize()
    _ = hist^; _ = keys2^; _ = vals2^


# ---- H-PLOC ----

@always_inline
def _delta(keys: Pointer[UInt32, MutUntrackedOrigin], i: Int) -> UInt64:
    """How far apart sorted keys i and i + 1 are; equal keys by index."""
    var x = keys[unsafe_offset=i] ^ keys[unsafe_offset=i + 1]
    return (UInt64(x) << 32) | UInt64(UInt32(i ^ (i + 1)))


@always_inline
def _exchange(p: Pointer[Int32, MutUntrackedOrigin], v: Int32) -> Int32:
    var e = Int32(-1)
    while not Atomic.compare_exchange[success_ordering=Ordering.RELAXED, failure_ordering=Ordering.RELAXED](p, e, v):
        pass
    return e


def _leaves_kernel(boxes: Pointer[Float32, MutUntrackedOrigin], vals: Pointer[Int32, MutUntrackedOrigin], n_dp: Int64,
                   nbox: Pointer[Float32, MutUntrackedOrigin], nl: Pointer[Int32, MutUntrackedOrigin], nr: Pointer[Int32, MutUntrackedOrigin],
                   ncnt: Pointer[Int32, MutUntrackedOrigin], nsize: Pointer[Int32, MutUntrackedOrigin], ncost: Pointer[Float32, MutUntrackedOrigin],
                   cl: Pointer[Int32, MutUntrackedOrigin], pidx: Pointer[Int32, MutUntrackedOrigin]):
    var i = _tid()
    if i >= Int(n_dp):
        return
    var b = boxes.unsafe_offset(6 * Int(vals[unsafe_offset=i]))
    nbox.unsafe_store(8 * i, SIMD[DType.float32, 4](b[unsafe_offset=0], b[unsafe_offset=1], b[unsafe_offset=2], Float32(0)))
    nbox.unsafe_store(8 * i + 4, SIMD[DType.float32, 4](b[unsafe_offset=3], b[unsafe_offset=4], b[unsafe_offset=5], Float32(0)))
    nl[unsafe_offset=i] = Int32(-1); nr[unsafe_offset=i] = Int32(i)
    ncnt[unsafe_offset=i] = Int32(1); nsize[unsafe_offset=i] = Int32(1)
    ncost[unsafe_offset=i] = _half_area(b[unsafe_offset=0], b[unsafe_offset=1], b[unsafe_offset=2], b[unsafe_offset=3], b[unsafe_offset=4], b[unsafe_offset=5])
    cl[unsafe_offset=i] = Int32(i); pidx[unsafe_offset=i] = Int32(-1)


def _hploc_kernel(keys: Pointer[UInt32, MutUntrackedOrigin], n_dp: Int64, seg_shift_dp: Int64, min_leaf_dp: Int64, max_leaf_dp: Int64,
                  cl: Pointer[Int32, MutUntrackedOrigin], pidx: Pointer[Int32, MutUntrackedOrigin], counter: Pointer[Int32, MutUntrackedOrigin],
                  nbox: Pointer[Float32, MutUntrackedOrigin], nl: Pointer[Int32, MutUntrackedOrigin], nr: Pointer[Int32, MutUntrackedOrigin],
                  ncnt: Pointer[Int32, MutUntrackedOrigin], nsize: Pointer[Int32, MutUntrackedOrigin], ncost: Pointer[Float32, MutUntrackedOrigin]):
    """Each thread climbs the radix tree from its leaf (Apetrei's bottom-up build); the second child to arrive goes on. A
    range larger than THRESH (or a whole segment) is reduced by its warp: PLOC merges of mutual nearest neighbours down to
    THRESH clusters (one for a whole segment), stored at the front of the range."""
    var n = Int(n_dp); var seg_shift = UInt32(seg_shift_dp)
    var min_leaf = Int32(min_leaf_dp); var max_leaf = Int32(max_leaf_dp)
    var lane = Int(lane_id())
    var wbase = (Int(thread_idx.x) // WARP_SIZE) * WARP_SIZE
    var s_id = stack_allocation[TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var below = (Scalar[MaskT](1) << Scalar[MaskT](lane)) - Scalar[MaskT](1)
    var i = _tid()
    var L = i
    var R = i
    var split = 0
    var active = i < n
    while vote[MaskT](active) != Scalar[MaskT](0):
        if active and L == 0 and R == n - 1:
            active = False
        if active:
            var prev: Int32
            if L == 0 or (R != n - 1 and _delta(keys, R) < _delta(keys, L - 1)):
                prev = _exchange(pidx.unsafe_offset(R), Int32(L))
                if prev != Int32(-1):
                    split = R + 1; R = Int(prev)
            else:
                prev = _exchange(pidx.unsafe_offset(L - 1), Int32(R))
                if prev != Int32(-1):
                    split = L; L = Int(prev)
            if prev == Int32(-1):
                active = False
        threadfence()                                    # acquire: the sibling's clusters and nodes, written before its exchange
        var merge = False
        var whole = False
        if active:
            var sl = keys[unsafe_offset=L] >> seg_shift
            var sr = keys[unsafe_offset=R] >> seg_shift
            whole = sl == sr and (L == 0 or keys[unsafe_offset=L - 1] >> seg_shift != sl) and (R == n - 1 or keys[unsafe_offset=R + 1] >> seg_shift != sr)
            merge = sl == sr and (R - L + 1 > THRESH or whole)
        var todo = vote[MaskT](merge)
        while todo != Scalar[MaskT](0):
            var src = UInt32(count_trailing_zeros(todo))
            todo &= todo - Scalar[MaskT](1)
            var bl = Int(shuffle_idx(Int32(L), src))
            var bs = Int(shuffle_idx(Int32(split), src))
            var br = Int(shuffle_idx(Int32(R), src))
            var target = 1 if shuffle_idx(Int32(1 if whole else 0), src) != Int32(0) else THRESH
            var n_left = min(bs - bl, THRESH)
            var n_load = n_left + min(br - bs + 1, THRESH)
            var id = Int32(-1)
            if lane < n_left:
                id = cl[unsafe_offset=bl + lane]
            elif lane < n_load:
                id = cl[unsafe_offset=bs + lane - n_left]
            var keep = vote[MaskT](id >= Int32(0))
            var num = Int(pop_count(keep))
            var lo = SIMD[DType.float32, 4](0)
            var hi = SIMD[DType.float32, 4](0)
            while True:
                syncwarp()
                if id >= Int32(0):
                    var pos = wbase + Int(pop_count(keep & below))
                    s_id[unsafe_offset=pos] = id
                syncwarp()
                id = s_id[unsafe_offset=wbase + lane] if lane < num else Int32(-1)
                if id >= Int32(0):
                    lo = nbox.unsafe_load[width=4](8 * Int(id)); hi = nbox.unsafe_load[width=4](8 * Int(id) + 4)
                if num <= target:
                    break
                var best = Float32(3.0e38)
                var bj = -1
                comptime for o in range(-RADIUS, RADIUS + 1):
                    comptime if o != 0:
                        var j = lane + o
                        var jj = UInt32(min(max(j, 0), WARP_SIZE - 1))
                        var ox = shuffle_idx(lo[0], jj); var oy = shuffle_idx(lo[1], jj); var oz = shuffle_idx(lo[2], jj)
                        var px = shuffle_idx(hi[0], jj); var py = shuffle_idx(hi[1], jj); var pz = shuffle_idx(hi[2], jj)
                        if j >= 0 and j < num and lane < num:
                            var d = _half_area(min(lo[0], ox), min(lo[1], oy), min(lo[2], oz), max(hi[0], px), max(hi[1], py), max(hi[2], pz))
                            if d < best:
                                best = d; bj = j
                var bjj = UInt32(max(bj, 0))
                var nn_nn = Int(shuffle_idx(Int32(bj), bjj))
                var pid = shuffle_idx(id, bjj)
                var mutual = lane < num and bj >= 0 and nn_nn == lane
                var makers = vote[MaskT](mutual and lane < bj)
                var base = Int32(0)
                if makers != Scalar[MaskT](0) and lane == Int(count_trailing_zeros(makers)):
                    base = Atomic.fetch_add[ordering=Ordering.RELAXED](counter, Int32(pop_count(makers)))
                base = shuffle_idx(base, UInt32(count_trailing_zeros(makers | (Scalar[MaskT](1) << Scalar[MaskT](WARP_SIZE - 1)))))
                if mutual and lane < bj:
                    var a = Int(id); var b = Int(pid)
                    var blo = nbox.unsafe_load[width=4](8 * b); var bhi = nbox.unsafe_load[width=4](8 * b + 4)
                    var ulo = min(lo, blo); var uhi = max(hi, bhi)
                    var nid = n + Int(base) + Int(pop_count(makers & below))
                    var c = ncnt[unsafe_offset=a] + ncnt[unsafe_offset=b]
                    var sa = _half_area(ulo[0], ulo[1], ulo[2], uhi[0], uhi[1], uhi[2])
                    var split_cost = C_TRAV * sa + ncost[unsafe_offset=a] + ncost[unsafe_offset=b]
                    var leaf_cost = sa * Float32(c)
                    if c <= max_leaf and (c <= min_leaf or leaf_cost <= split_cost):
                        nsize[unsafe_offset=nid] = Int32(1); ncost[unsafe_offset=nid] = leaf_cost
                    else:
                        nsize[unsafe_offset=nid] = Int32(1) + nsize[unsafe_offset=a] + nsize[unsafe_offset=b]; ncost[unsafe_offset=nid] = split_cost
                    nbox.unsafe_store(8 * nid, ulo); nbox.unsafe_store(8 * nid + 4, uhi)
                    nl[unsafe_offset=nid] = Int32(a); nr[unsafe_offset=nid] = Int32(b); ncnt[unsafe_offset=nid] = c
                    id = Int32(nid)
                elif mutual:
                    id = Int32(-1)
                threadfence()
                keep = vote[MaskT](id >= Int32(0))
                if Int(pop_count(keep)) == num:          # no mutual pair: only a bug gets here; don't hang the GPU
                    break
                num = Int(pop_count(keep))
            syncwarp()
            if lane < n_load:
                cl[unsafe_offset=bl + lane] = id
        threadfence()


# ---- depth-first layout ----

def _roots_kernel(cl: Pointer[Int32, MutUntrackedOrigin], seg_start: Pointer[Int32, MutUntrackedOrigin], nseg_dp: Int64, nsize: Pointer[Int32, MutUntrackedOrigin],
                  roots: Pointer[Int32, MutUntrackedOrigin]):
    var s = _tid()
    if s < Int(nseg_dp):
        roots[unsafe_offset=s] = nsize[unsafe_offset=Int(cl[unsafe_offset=Int(seg_start[unsafe_offset=s])])]


def _front_kernel(cl: Pointer[Int32, MutUntrackedOrigin], seg_start: Pointer[Int32, MutUntrackedOrigin], nseg_dp: Int64, nsize: Pointer[Int32, MutUntrackedOrigin],
                  roots: Pointer[Int32, MutUntrackedOrigin], dfs: Pointer[Int32, MutUntrackedOrigin], pst: Pointer[Int32, MutUntrackedOrigin],
                  front: Pointer[Int32, MutUntrackedOrigin], total: Pointer[Int32, MutUntrackedOrigin]):
    var s = _tid()
    var nseg = Int(nseg_dp)
    if s >= nseg:
        return
    var r = Int(cl[unsafe_offset=Int(seg_start[unsafe_offset=s])])
    dfs[unsafe_offset=r] = roots[unsafe_offset=s]
    pst[unsafe_offset=r] = seg_start[unsafe_offset=s]
    front[unsafe_offset=s] = Int32(r)
    if s == nseg - 1:
        total[unsafe_offset=0] = roots[unsafe_offset=s] + nsize[unsafe_offset=r]


def _layout_kernel(front: Pointer[Int32, MutUntrackedOrigin], nfront_dp: Int64, next: Pointer[Int32, MutUntrackedOrigin], nnext: Pointer[Int32, MutUntrackedOrigin],
                   dfs: Pointer[Int32, MutUntrackedOrigin], pst: Pointer[Int32, MutUntrackedOrigin], nbox: Pointer[Float32, MutUntrackedOrigin],
                   nl: Pointer[Int32, MutUntrackedOrigin], nr: Pointer[Int32, MutUntrackedOrigin], ncnt: Pointer[Int32, MutUntrackedOrigin],
                   nsize: Pointer[Int32, MutUntrackedOrigin], vals: Pointer[Int32, MutUntrackedOrigin], dst: Pointer[BVH2Node, MutUntrackedOrigin],
                   order: Pointer[Int32, MutUntrackedOrigin], below: Pointer[Int32, MutUntrackedOrigin], first: Pointer[Int32, MutUntrackedOrigin], ranges_dp: Int64):
    """One level: writes the front's nodes (dfs -1: inside a leaf) and their primitives' slots; the children go next.
    No early return: the warp appends the children together."""
    var t = _tid()
    var a = -1
    var b = -1
    if t < Int(nfront_dp):
        var v = Int(front[unsafe_offset=t])
        var d = Int(dfs[unsafe_offset=v]); var p = pst[unsafe_offset=v]
        var lo = nbox.unsafe_load[width=4](8 * v); var hi = nbox.unsafe_load[width=4](8 * v + 4)
        if d >= 0 and ranges_dp != Int64(0):
            below[unsafe_offset=d] = ncnt[unsafe_offset=v]; first[unsafe_offset=d] = p
        if nl[unsafe_offset=v] < Int32(0):
            order[unsafe_offset=Int(p)] = vals[unsafe_offset=Int(nr[unsafe_offset=v])]
            if d >= 0:
                dst[unsafe_offset=d] = BVH2Node(Point3f(lo[0], lo[1], lo[2]), Point3f(hi[0], hi[1], hi[2]), p, Int32(1))
        else:
            a = Int(nl[unsafe_offset=v]); b = Int(nr[unsafe_offset=v])
            pst[unsafe_offset=a] = p; pst[unsafe_offset=b] = p + ncnt[unsafe_offset=a]
            if d >= 0 and nsize[unsafe_offset=v] == Int32(1):
                dst[unsafe_offset=d] = BVH2Node(Point3f(lo[0], lo[1], lo[2]), Point3f(hi[0], hi[1], hi[2]), p, ncnt[unsafe_offset=v])
                d = -1
            elif d >= 0:
                var right = d + 1 + Int(nsize[unsafe_offset=a])
                dst[unsafe_offset=d] = BVH2Node(Point3f(lo[0], lo[1], lo[2]), Point3f(hi[0], hi[1], hi[2]), Int32(right), Int32(0))
                dfs[unsafe_offset=a] = Int32(d + 1); dfs[unsafe_offset=b] = Int32(right)
            if d < 0:
                dfs[unsafe_offset=a] = Int32(-1); dfs[unsafe_offset=b] = Int32(-1)
    var push = vote[MaskT](a >= 0)
    if push == Scalar[MaskT](0):
        return
    var lane = Int(lane_id())
    var leader = Int(count_trailing_zeros(push))
    var base = Int32(0)
    if lane == leader:
        base = Atomic.fetch_add[ordering=Ordering.RELAXED](nnext, Int32(2 * pop_count(push)))
    base = shuffle_idx(base, UInt32(leader))
    if a >= 0:
        var k = Int(base) + 2 * Int(pop_count(push & ((Scalar[MaskT](1) << Scalar[MaskT](lane)) - Scalar[MaskT](1))))
        next[unsafe_offset=k] = Int32(a); next[unsafe_offset=k + 1] = Int32(b)


def _read_int(ctx: DeviceContext, mut b: DeviceBuffer[DType.uint8]) raises -> Int:
    var h = unsafe_alloc[UInt8](len(b))
    ctx.enqueue_copy(h, b)
    ctx.synchronize()
    var v = Int(h.unsafe_bitcast[Int32]()[unsafe_offset=0])
    h.unsafe_free()
    return v


def gpu_morton_sort(ctx: DeviceContext, boxes: Pointer[Float32, MutUntrackedOrigin], n: Int, seg_start: Pointer[Int32, MutUntrackedOrigin],
                    nseg: Int) raises -> Tuple[DeviceBuffer[DType.uint8], DeviceBuffer[DType.uint8], Int]:
    """(keys, primitives in key order, Morton bits per axis); a segment's keys sort before the next one's."""
    var seg_bits = 0
    while (1 << seg_bits) < nseg:
        seg_bits += 1
    var bits = min(10, (32 - seg_bits) // 3)
    var sb = _buf[Int32](ctx, 6 * nseg)
    ctx.enqueue_function[_bounds_init_kernel](_p[Int32](sb), Int64(nseg), grid_dim=ceildiv(6 * nseg, TPB), block_dim=TPB)
    ctx.enqueue_function[_bounds_kernel](boxes, Int64(n), seg_start, Int64(nseg), _p[Int32](sb), grid_dim=ceildiv(n, TPB), block_dim=TPB)
    var keys = _buf[UInt32](ctx, n)
    var vals = _buf[Int32](ctx, n)
    ctx.enqueue_function[_morton_kernel](boxes, Int64(n), seg_start, Int64(nseg), _p[Int32](sb), Int64(bits), _p[UInt32](keys), _p[Int32](vals),
                                         grid_dim=ceildiv(n, TPB), block_dim=TPB)
    _radix_sort(ctx, keys, vals, n, 3 * bits + seg_bits)
    _ = sb^
    return (keys^, vals^, bits)


def gpu_build_bvh2(ctx: DeviceContext, boxes: Pointer[Float32, MutUntrackedOrigin], n: Int, seg_start: Pointer[Int32, MutUntrackedOrigin], nseg: Int,
                   min_leaf: Int = 1, max_leaf: Int = 4, ranges: Bool = False) raises -> GpuBvh:
    """`boxes`: six floats per primitive, on the device; `seg_start`: nseg + 1 offsets on the device (0, ..., n). Leaves
    index `order` as build_bvh2's do."""
    var (keys, vals, bits) = gpu_morton_sort(ctx, boxes, n, seg_start, nseg)

    var nn = 2 * n
    var nbox = _buf[Float32](ctx, 8 * nn)
    var nl = _buf[Int32](ctx, nn); var nr = _buf[Int32](ctx, nn)
    var ncnt = _buf[Int32](ctx, nn); var nsize = _buf[Int32](ctx, nn); var ncost = _buf[Float32](ctx, nn)
    var cl = _buf[Int32](ctx, n); var pidx = _buf[Int32](ctx, n)
    var counter = _buf[Int32](ctx, 1)
    ctx.enqueue_memset(counter, 0)
    ctx.enqueue_function[_leaves_kernel](boxes, _p[Int32](vals), Int64(n), _p[Float32](nbox), _p[Int32](nl), _p[Int32](nr), _p[Int32](ncnt),
                                         _p[Int32](nsize), _p[Float32](ncost), _p[Int32](cl), _p[Int32](pidx), grid_dim=ceildiv(n, TPB), block_dim=TPB)
    ctx.enqueue_function[_hploc_kernel](_p[UInt32](keys), Int64(n), Int64(3 * bits), Int64(min_leaf), Int64(max_leaf), _p[Int32](cl), _p[Int32](pidx),
                                        _p[Int32](counter), _p[Float32](nbox), _p[Int32](nl), _p[Int32](nr), _p[Int32](ncnt), _p[Int32](nsize),
                                        _p[Float32](ncost), grid_dim=ceildiv(n, TPB), block_dim=TPB)
    ctx.synchronize()
    _ = keys^; _ = pidx^; _ = ncost^

    var roots = _buf[Int32](ctx, nseg)
    ctx.enqueue_function[_roots_kernel](_p[Int32](cl), seg_start, Int64(nseg), _p[Int32](nsize), _p[Int32](roots), grid_dim=ceildiv(nseg, TPB), block_dim=TPB)
    _exclusive_scan(ctx, _p[Int32](roots), nseg)
    var dfs = _buf[Int32](ctx, nn); var pst = _buf[Int32](ctx, nn)
    var front = _buf[Int32](ctx, n + nseg); var next = _buf[Int32](ctx, n + nseg)
    var total = _buf[Int32](ctx, 1)
    ctx.enqueue_function[_front_kernel](_p[Int32](cl), seg_start, Int64(nseg), _p[Int32](nsize), _p[Int32](roots), _p[Int32](dfs), _p[Int32](pst),
                                        _p[Int32](front), _p[Int32](total), grid_dim=ceildiv(nseg, TPB), block_dim=TPB)
    var n_nodes = _read_int(ctx, total)
    var dst = _buf[BVH2Node](ctx, n_nodes)
    var order = _buf[Int32](ctx, n)
    var below = _buf[Int32](ctx, n_nodes if ranges else 1)
    var first = _buf[Int32](ctx, n_nodes if ranges else 1)
    var nfront = nseg
    while nfront > 0:
        ctx.enqueue_memset(counter, 0)
        ctx.enqueue_function[_layout_kernel](_p[Int32](front), Int64(nfront), _p[Int32](next), _p[Int32](counter), _p[Int32](dfs), _p[Int32](pst),
                                             _p[Float32](nbox), _p[Int32](nl), _p[Int32](nr), _p[Int32](ncnt), _p[Int32](nsize), _p[Int32](vals),
                                             _p[BVH2Node](dst), _p[Int32](order), _p[Int32](below), _p[Int32](first), Int64(1 if ranges else 0),
                                             grid_dim=ceildiv(nfront, TPB), block_dim=TPB)
        nfront = _read_int(ctx, counter)
        swap(front, next)
    _ = vals^; _ = nbox^; _ = nl^; _ = nr^; _ = ncnt^; _ = nsize^; _ = cl^; _ = counter^; _ = dfs^; _ = pst^; _ = front^; _ = next^; _ = total^
    return GpuBvh(dst^, order^, roots^, below^, first^, n_nodes)
