from std.math import abs
from std.testing import assert_true, TestSuite
from gonzales.geometry import Point3f, Vec3f, Ray_C
from _scene_fixture import make_triangle_scene

comptime EPS: Float32 = 1e-3

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# ── traverse_bvh2_core, exercised through a REAL build_bvh2-built BVH ───────
# Previously only intersect_aabb (the box-vs-ray slab test) had direct
# coverage — the full node-stack traversal + triangle intersection + the
# real BVH-build pipeline (build_bvh2, the same function finalize_scene
# calls) had none. This is the fixture other test files (render_tile,
# sppm/bdpt render loops, gpu_sppm kernels) can reuse via _scene_fixture.

def test_single_triangle_hit_through_center() raises:
    var fx = make_triangle_scene([
        Point3f(0.0, 0.0, 0.0), Point3f(1.0, 0.0, 0.0), Point3f(0.0, 1.0, 0.0),
    ])
    var ray = Ray_C(Point3f(0.2, 0.2, -5.0), Vec3f(0.0, 0.0, 1.0))
    var hit = fx.intersect(ray, Float32(100.0))
    assert_true(Int(hit.hit) == 1)
    assert_true(_close(hit.tHit, Float32(5.0)))
    assert_true(Int(hit.primId.id1) == 0)  # mesh 0

def test_single_triangle_miss_outside_edge() raises:
    """Same mesh, ray offset past the hypotenuse (x+y>1 in the triangle's
    plane) — must report no hit."""
    var fx = make_triangle_scene([
        Point3f(0.0, 0.0, 0.0), Point3f(1.0, 0.0, 0.0), Point3f(0.0, 1.0, 0.0),
    ])
    var ray = Ray_C(Point3f(0.9, 0.9, -5.0), Vec3f(0.0, 0.0, 1.0))
    var hit = fx.intersect(ray, Float32(100.0))
    assert_true(Int(hit.hit) == 0)

def test_two_triangles_picks_the_nearer_one() raises:
    """Two coplanar-normal but depth-separated triangles directly in front
    of each other on the ray's path — traversal must report the nearer
    tHit, not just any hit or the first one built."""
    var fx = make_triangle_scene([
        Point3f(-1.0, -1.0, 5.0), Point3f(1.0, -1.0, 5.0), Point3f(0.0, 1.0, 5.0),  # far
        Point3f(-1.0, -1.0, 2.0), Point3f(1.0, -1.0, 2.0), Point3f(0.0, 1.0, 2.0),  # near
    ])
    var ray = Ray_C(Point3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 1.0))
    var hit = fx.intersect(ray, Float32(100.0))
    assert_true(Int(hit.hit) == 1)
    assert_true(_close(hit.tHit, Float32(2.0)))

def test_tmax_cutoff_hides_a_real_hit() raises:
    var fx = make_triangle_scene([
        Point3f(0.0, 0.0, 0.0), Point3f(1.0, 0.0, 0.0), Point3f(0.0, 1.0, 0.0),
    ])
    var ray = Ray_C(Point3f(0.2, 0.2, -5.0), Vec3f(0.0, 0.0, 1.0))
    var hit = fx.intersect(ray, Float32(2.0))  # true hit is at t=5
    assert_true(Int(hit.hit) == 0)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
