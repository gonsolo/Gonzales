"""RMIP displacement: the pyramid's range queries and the traversal against brute force over the same micro-triangles."""
from std.math import abs, sin, cos, min, max
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, TestSuite
from gonzales.core.geometry import Vec3f
from gonzales.core.rng import PCG32
from gonzales.scene.shapes.displacement import (Rmip, rmip_build, displaced_tri, intersect_displaced,
    intersect_displaced_brute)

comptime F2 = SIMD[DType.float32, 2]


def _map(w: Int, hh: Int, mut rng: PCG32) -> Pointer[Float32, MutUntrackedOrigin]:
    var h = unsafe_alloc[Float32](w * hh)
    for j in range(hh):
        for i in range(w):
            h[unsafe_offset=j * w + i] = Float32(0.3) * sin(Float32(i) * 0.4) * cos(Float32(j) * 0.3) + Float32(0.2) * rng.next_float()
    return h


def test_rmip_query_matches_brute_force() raises:
    var rng = PCG32(7, 1)
    var w = 37; var hh = 23
    var h = _map(w, hh, rng)
    var rm = rmip_build(h, w, hh)
    assert_true(rm.b > 1)   # compressed to blocks
    var bad = 0
    for _ in range(2000):
        var i0 = Int(rng.next_float() * 120) - 60; var j0 = Int(rng.next_float() * 80) - 40
        var i1 = i0 + Int(rng.next_float() * 45); var j1 = j0 + Int(rng.next_float() * 30)
        var lo = Float32(3e38); var hi = Float32(-3e38)
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                var v = rm.sample(i, j); lo = min(lo, v); hi = max(hi, v)
        var q = rm.query(i0, i1, j0, j1)
        if q[0] > lo or q[1] < hi: bad += 1   # blocks may loosen, never cut
    assert_true(bad == 0)
    h.unsafe_free(); rm.tab.unsafe_free()


def _tri(rm: Rmip, tilt: Float32, t0: F2, t1: F2, t2: F2) -> Tuple[Vec3f, Vec3f, Vec3f, Vec3f, Vec3f, Vec3f, F2, F2, F2]:
    var p0 = Vec3f(0, 0, 0); var p1 = Vec3f(4, 0.3, 0); var p2 = Vec3f(0.5, 3.5, 0.2)
    var n0 = Vec3f(-tilt, 0, 1).normalize(); var n1 = Vec3f(tilt, 0.1, 1).normalize(); var n2 = Vec3f(0, tilt, 1).normalize()
    return (p0, p1, p2, n0, n1, n2, t0, t1, t2)


def _compare(rm: Rmip, tilt: Float32, t0: F2, t1: F2, t2: F2, seed: UInt64) raises -> Tuple[Int, Int]:
    var g = _tri(rm, tilt, t0, t1, t2)
    var tri = displaced_tri(g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], rm)
    var rng = PCG32(seed, 3)
    var hits = 0; var bad = 0
    for _ in range(300):
        # aim at a point near the triangle from a random upper-hemisphere-ish direction
        var a = rng.next_float(); var b = rng.next_float() * (Float32(1) - a)
        var target = g[0] * (Float32(1) - a - b) + g[1] * a + g[2] * b + Vec3f(0, 0, (rng.next_float() - 0.5) * 0.6)
        var dir = Vec3f(rng.next_float() * 2 - 1, rng.next_float() * 2 - 1, rng.next_float() * 1.5 - 0.3).normalize()
        var o = target + dir * Float32(6)
        var d = -dir
        var r = intersect_displaced(tri, rm, o, d, Float32(1e30))
        var e = intersect_displaced_brute(tri, rm, o, d, Float32(1e30))
        if e.hit: hits += 1
        if r.hit != e.hit or (e.hit and abs(r.t - e.t) > Float32(1e-3) * e.t):
            bad += 1
    return (hits, bad)


def test_traversal_matches_brute_force() raises:
    var rng = PCG32(11, 1)
    var w = 64; var hh = 48
    var h = _map(w, hh, rng)
    var rm = rmip_build(h, w, hh)
    # 30 texels across, then the same tiled past the map's edge, then with tilted normals
    var r1 = _compare(rm, 0, F2(3, 4), F2(33, 6), F2(8, 30), 1)
    var r2 = _compare(rm, 0, F2(50, 40), F2(90, 44), F2(55, 85), 2)
    var r3 = _compare(rm, 0.4, F2(3, 4), F2(33, 6), F2(8, 30), 3)
    assert_true(r1[0] > 100 and r2[0] > 100 and r3[0] > 100)
    assert_true(r1[1] == 0 and r2[1] == 0 and r3[1] == 0)
    h.unsafe_free(); rm.tab.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
