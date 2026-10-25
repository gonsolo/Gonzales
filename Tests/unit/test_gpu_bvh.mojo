from std.sys import has_accelerator
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_equal, TestSuite
from max.gpu.host import DeviceContext
from gonzales.bvh import BVH2Node, build_bvh2
from gonzales.gpu_bvh import gpu_build_bvh2

# The GPU build must give a valid depth-first BVH2 (boxes nest, every primitive once, each segment's tree over exactly its
# primitives) of about the CPU build's SAH cost.

struct _Lcg:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Float32:
        self.s = self.s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        return Float32(Int((self.s >> 40) & UInt64(0xFFFFFF))) / Float32(16777216.0)


def _boxes(n: Int, seed: UInt64) -> Pointer[Float32, MutUntrackedOrigin]:
    var r = _Lcg(seed)
    var b = unsafe_alloc[Float32](6 * max(n, 1))
    for i in range(n):
        var e = Float32(0.001) + r.next() * Float32(0.02)
        for q in range(3):
            var c = r.next() * Float32(10.0) * (Float32(0.2) if q == 1 else Float32(1.0))
            b[unsafe_offset=6 * i + q] = c - e
            b[unsafe_offset=6 * i + 3 + q] = c + e
    return b.unsafe_origin_cast[MutUntrackedOrigin]()


def _half_area(nd: BVH2Node) -> Float32:
    var dx = nd.max.x - nd.min.x; var dy = nd.max.y - nd.min.y; var dz = nd.max.z - nd.min.z
    return dx * dy + dy * dz + dz * dx


def _check(nodes: Pointer[BVH2Node, MutUntrackedOrigin], order: Pointer[Int32, MutUntrackedOrigin], boxes: Pointer[Float32, MutUntrackedOrigin],
           root: Int, p0: Int, p1: Int, n_nodes: Int) raises -> Float64:
    """Walks the tree at `root`; returns its SAH cost over its root's area."""
    var seen = List[Bool](length=p1 - p0, fill=False)
    var stack = List[Int]()
    stack.append(root)
    var next_slot = p0
    var visited = 0
    var cost = Float64(0)
    while len(stack) > 0:
        var i = stack.pop()
        assert_true(i >= 0 and i < n_nodes)
        visited += 1
        var nd = nodes[unsafe_offset=i]
        if nd.count > Int32(0):
            assert_equal(Int(nd.offset), next_slot)        # depth-first: leaves in slot order
            next_slot += Int(nd.count)
            cost += Float64(_half_area(nd)) * Float64(nd.count)
            for k in range(Int(nd.offset), Int(nd.offset) + Int(nd.count)):
                var p = Int(order[unsafe_offset=k])
                assert_true(p >= p0 and p < p1)
                assert_true(not seen[p - p0])
                seen[p - p0] = True
                for q in range(3):
                    assert_true(boxes[unsafe_offset=6 * p + q] >= nd.min[q] and boxes[unsafe_offset=6 * p + 3 + q] <= nd.max[q])
        else:
            cost += Float64(0.5) * Float64(_half_area(nd))
            for c in [i + 1, Int(nd.offset)]:
                var ch = nodes[unsafe_offset=c]
                for q in range(3):
                    assert_true(ch.min[q] >= nd.min[q] and ch.max[q] <= nd.max[q])
            stack.append(Int(nd.offset))
            stack.append(i + 1)
    assert_equal(next_slot, p1)
    return cost / Float64(_half_area(nodes[unsafe_offset=root]))


def _build(ctx: DeviceContext, boxes: Pointer[Float32, MutUntrackedOrigin], n: Int, starts: List[Int]) raises -> Tuple[Pointer[BVH2Node, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin], Int]:
    var nseg = len(starts) - 1
    var bb = ctx.enqueue_create_buffer[DType.uint8](24 * max(n, 1))
    ctx.enqueue_copy(bb, boxes.unsafe_bitcast[UInt8]())
    var ss = unsafe_alloc[Int32](nseg + 1)
    for s in range(nseg + 1):
        ss[unsafe_offset=s] = Int32(starts[s])
    var sb = ctx.enqueue_create_buffer[DType.uint8](4 * (nseg + 1))
    ctx.enqueue_copy(sb, ss.unsafe_bitcast[UInt8]())
    var g = gpu_build_bvh2(ctx, bb.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin](), n,
                           sb.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_origin_cast[MutUntrackedOrigin](), nseg)
    var nodes = unsafe_alloc[BVH2Node](g.n_nodes)
    var order = unsafe_alloc[Int32](n)
    var roots = unsafe_alloc[Int32](nseg)
    ctx.enqueue_copy(nodes.unsafe_bitcast[UInt8](), g.nodes)
    ctx.enqueue_copy(order.unsafe_bitcast[UInt8](), g.order)
    ctx.enqueue_copy(roots.unsafe_bitcast[UInt8](), g.roots)
    ctx.synchronize()
    _ = bb^; _ = sb^                                         # alive until the build is done
    ss.unsafe_free()
    return (nodes.unsafe_origin_cast[MutUntrackedOrigin](), order.unsafe_origin_cast[MutUntrackedOrigin](), roots.unsafe_origin_cast[MutUntrackedOrigin](), g.n_nodes)


def test_single_tree_matches_cpu_quality() raises:
    comptime if not has_accelerator():
        return
    var ctx = DeviceContext()
    for n in [1, 2, 33, 1000, 300000]:
        var boxes = _boxes(n, UInt64(n))
        var g = _build(ctx, boxes, n, [0, n])
        var gc = _check(g[0], g[1], boxes, Int(g[2][unsafe_offset=0]), 0, n, g[3])
        if n >= 1000:
            var cn = unsafe_alloc[BVH2Node](2 * n + 2)
            var co = unsafe_alloc[Int32](n)
            var m = Int(build_bvh2(boxes, Int32(n), cn, co))
            var cc = _check(cn, co, boxes, 0, 0, n, m)
            assert_true(gc < cc * 1.25)
            cn.unsafe_free(); co.unsafe_free()
        g[0].unsafe_free(); g[1].unsafe_free(); g[2].unsafe_free(); boxes.unsafe_free()


def test_segments_get_their_own_trees() raises:
    comptime if not has_accelerator():
        return
    var ctx = DeviceContext()
    var starts: List[Int] = [0, 7000, 7001, 7040, 50000, 50017, 120000]
    var n = starts[len(starts) - 1]
    var boxes = _boxes(n, UInt64(5))
    var g = _build(ctx, boxes, n, starts)
    var next_root = 0
    for s in range(len(starts) - 1):
        var r = Int(g[2][unsafe_offset=s])
        assert_equal(r, next_root)                          # the trees follow each other
        _ = _check(g[0], g[1], boxes, r, starts[s], starts[s + 1], g[3])
        var nd = g[0][unsafe_offset=r]
        next_root = r + 1
        if nd.count == Int32(0):
            var stack: List[Int] = [r]
            next_root = r
            while len(stack) > 0:
                var i = stack.pop()
                next_root = max(next_root, i + 1)
                var x = g[0][unsafe_offset=i]
                if x.count == Int32(0):
                    stack.append(i + 1); stack.append(Int(x.offset))
    assert_equal(next_root, g[3])
    g[0].unsafe_free(); g[1].unsafe_free(); g[2].unsafe_free(); boxes.unsafe_free()



def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
