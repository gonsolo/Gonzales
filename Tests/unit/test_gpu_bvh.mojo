from std.sys import has_accelerator
from std.sys.info import size_of
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_equal, TestSuite
from max.gpu.host import DeviceContext, DeviceBuffer
from gonzales.bvh import BVH2Node, build_bvh2
from gonzales.gpu_bvh import gpu_build_bvh2
from gonzales.geometry import Point3f, Vec3f
from gonzales.primitives import Ray, PrimId
from gonzales.curves import Curve, CurveBvh, intersect_curve
from gonzales.curve_bvh import gpu_build_curve_bvh, curve_bvh_occluded, curve_bvh_closest
from _scene_fixture import make_triangle_scene
from std.math import sqrt

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


def _unit(mut r: _Lcg) -> Vec3f:
    var v = Vec3f(r.next() - 0.5, r.next() - 0.5, r.next() - 0.5)
    return v * (Float32(1.0) / sqrt(max(Float32(1e-12), v.x * v.x + v.y * v.y + v.z * v.z)))


def _download[T: AnyType](ctx: DeviceContext, mut b: DeviceBuffer[DType.uint8]) raises -> Pointer[T, MutUntrackedOrigin]:
    var h = unsafe_alloc[UInt8](len(b))
    ctx.enqueue_copy(h, b)
    ctx.synchronize()
    return h.unsafe_bitcast[T]().unsafe_origin_cast[MutUntrackedOrigin]()


def test_curve_clusters_match_brute_force() raises:
    """Shadow and nearest hits through the GPU-built clusters equal a brute-force test; the PrimIds sit after 5 others."""
    comptime if not has_accelerator():
        return
    var ctx = DeviceContext()
    comptime N = 60000
    comptime A = 5
    var r = _Lcg(7)
    var curves = unsafe_alloc[Curve](N)
    var prims = unsafe_alloc[PrimId](N + A)
    for i in range(N):
        var c = Vec3f(r.next() * 10.0, r.next() * 10.0, r.next() * 10.0)
        var d = _unit(r) * (Float32(0.2) + r.next() * 0.4)
        var bend = _unit(r) * Float32(0.03)
        var p0 = c; var p1 = c + d * Float32(0.33) + bend; var p2 = c + d * Float32(0.66) + bend; var p3 = c + d
        var pieces = 1 if i % 3 != 0 else 7
        curves[unsafe_offset=i] = Curve(Point3f(p0.x, p0.y, p0.z), Point3f(p1.x, p1.y, p1.z), Point3f(p2.x, p2.y, p2.z),
                                        Point3f(p3.x, p3.y, p3.z), Float32(0.02), Float32(0.01), Int32(0), Int32(pieces), Int32(0))
        prims[unsafe_offset=A + i] = PrimId(Int64(i), Int64(pieces), Int64(0), Int32(-1), Int8(5), Int8(0), Int8(0), Int8(0))
    var cb = ctx.enqueue_create_buffer[DType.uint8](N * size_of[Curve]())
    ctx.enqueue_copy(cb, curves.unsafe_bitcast[UInt8]())
    var pb = ctx.enqueue_create_buffer[DType.uint8]((N + A) * size_of[PrimId]())
    ctx.enqueue_copy(pb, prims.unsafe_bitcast[UInt8]())
    var g = gpu_build_curve_bvh(ctx, pb.unsafe_ptr().unsafe_bitcast[PrimId]().unsafe_origin_cast[MutUntrackedOrigin](),
                                cb.unsafe_ptr().unsafe_bitcast[Curve]().unsafe_origin_cast[MutUntrackedOrigin](), A, N)
    _ = cb^; _ = pb^
    var h = CurveBvh(_download[UInt32](ctx, g.bufs[0]), _download[Int32](ctx, g.bufs[1]), _download[Float32](ctx, g.bufs[2]),
                     _download[Int32](ctx, g.bufs[3]), _download[UInt32](ctx, g.bufs[4]), _download[Int32](ctx, g.bufs[5]), g.bvh.n_clusters,
                     g.bvh.grid)
    assert_true(Int(h.n_clusters) > 10)
    var seen = List[Bool](length=N, fill=False)
    for k in range(N):
        var q = Int(h.refs[unsafe_offset=3 * k])
        assert_true(q >= 0 and q < N and not seen[q])
        seen[q] = True
    var fx = make_triangle_scene([Point3f(100.0, 100.0, 100.0), Point3f(101.0, 100.0, 100.0), Point3f(100.0, 101.0, 100.0)])
    var hits = 0
    for _ in range(600):
        var o = Vec3f(r.next() * 14.0 - 2.0, r.next() * 14.0 - 2.0, r.next() * 14.0 - 2.0)
        var d = _unit(r)
        var tmax = Float32(0.5) + r.next() * 15.0
        var near = tmax
        var want = False
        for i in range(N):
            var c = intersect_curve(o, d, curves[unsafe_offset=i], 0, Int(prims[unsafe_offset=A + i].id2), near)
            if c[0]:
                want = True; near = c[1]
        var ray = Ray(Point3f(o.x, o.y, o.z), d)
        assert_equal(curve_bvh_occluded(h, curves, fx.materials, ray, tmax), want)
        var got = curve_bvh_closest(h, curves, ray, tmax)
        assert_equal(got[0] >= 0, want)
        if want:
            assert_true(abs(got[1] - near) < Float32(1e-5))
        hits += 1 if want else 0
    assert_true(hits > 30 and hits < 570)
    curves.unsafe_free(); prims.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
