from std.math import sqrt
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_equal, TestSuite
from gonzales.geometry import Point3f, Vec3f
from gonzales.primitives import Ray, PrimId
from gonzales.bvh import BVH2Node, build_bvh2
from gonzales.curves import Curve, intersect_curve, curve_piece_bounds
from gonzales.curve_bvh import build_curve_bvh, curve_bvh_occluded, free_curve_bvh
from _scene_fixture import make_triangle_scene

# The fiber-aligned curve clusters must block exactly the shadow rays a brute-force test of every curve blocks.

struct _Lcg:
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Float32:
        self.s = self.s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        return Float32(Int((self.s >> 40) & UInt64(0xFFFFFF))) / Float32(16777216.0)


def _unit(mut r: _Lcg) -> Vec3f:
    var v = Vec3f(r.next() - 0.5, r.next() - 0.5, r.next() - 0.5)
    return v * (Float32(1.0) / sqrt(max(Float32(1e-12), v.x * v.x + v.y * v.y + v.z * v.z)))


def test_occlusion_matches_brute_force() raises:
    comptime N = 20000                   # several clusters
    var r = _Lcg(7)
    var curves = unsafe_alloc[Curve](N)
    var prims = unsafe_alloc[PrimId](N)
    var bounds = unsafe_alloc[Float32](6 * N)
    for i in range(N):
        var c = Vec3f(r.next() * 10.0, r.next() * 10.0, r.next() * 10.0)
        var d = _unit(r) * (Float32(0.2) + r.next() * 0.4)
        var bend = _unit(r) * Float32(0.03)
        var p0 = c; var p1 = c + d * Float32(0.33) + bend; var p2 = c + d * Float32(0.66) + bend; var p3 = c + d
        var pieces = 1 if i % 3 != 0 else 7
        curves[unsafe_offset=i] = Curve(Point3f(p0.x, p0.y, p0.z), Point3f(p1.x, p1.y, p1.z), Point3f(p2.x, p2.y, p2.z),
                                        Point3f(p3.x, p3.y, p3.z), Float32(0.02), Float32(0.01), Int32(0), Int32(pieces), Int32(0))
        prims[unsafe_offset=i] = PrimId(Int64(i), Int64(pieces), Int64(0), Int32(-1), Int8(5), Int8(0), Int8(0), Int8(0))
        var b = curve_piece_bounds(curves[unsafe_offset=i], 0)
        for p in range(1, pieces):
            var q = curve_piece_bounds(curves[unsafe_offset=i], p)
            b = (min(b[0], q[0]), min(b[1], q[1]), min(b[2], q[2]), max(b[3], q[3]), max(b[4], q[4]), max(b[5], q[5]))
        bounds[unsafe_offset=6 * i] = b[0]; bounds[unsafe_offset=6 * i + 1] = b[1]; bounds[unsafe_offset=6 * i + 2] = b[2]
        bounds[unsafe_offset=6 * i + 3] = b[3]; bounds[unsafe_offset=6 * i + 4] = b[4]; bounds[unsafe_offset=6 * i + 5] = b[5]
    var nodes = unsafe_alloc[BVH2Node](2 * N + 2)
    var order = unsafe_alloc[Int32](N)
    _ = build_bvh2(bounds, Int32(N), nodes, order)
    var sorted = unsafe_alloc[PrimId](N)
    for k in range(N):
        sorted[unsafe_offset=k] = prims[unsafe_offset=Int(order[unsafe_offset=k])]
    var b = build_curve_bvh(nodes, sorted, curves, 0)
    assert_true(Int(b.bvh.n_clusters) > 1)
    var fx = make_triangle_scene([Point3f(100.0, 100.0, 100.0), Point3f(101.0, 100.0, 100.0), Point3f(100.0, 101.0, 100.0)])
    var hits = 0
    for _ in range(600):
        var o = Vec3f(r.next() * 14.0 - 2.0, r.next() * 14.0 - 2.0, r.next() * 14.0 - 2.0)
        var d = _unit(r)
        var tmax = Float32(0.5) + r.next() * 15.0
        var want = False
        for i in range(N):
            var p = prims[unsafe_offset=i]
            if intersect_curve(o, d, curves[unsafe_offset=i], 0, Int(p.id2), tmax)[0]:
                want = True
                break
        var got = curve_bvh_occluded(b.bvh, sorted, curves, fx.materials, Ray(Point3f(o.x, o.y, o.z), d), tmax)
        assert_equal(got, want)
        hits += 1 if want else 0
    assert_true(hits > 30 and hits < 570)
    free_curve_bvh(b.bvh)
    curves.unsafe_free(); prims.unsafe_free(); bounds.unsafe_free(); nodes.unsafe_free(); order.unsafe_free(); sorted.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
