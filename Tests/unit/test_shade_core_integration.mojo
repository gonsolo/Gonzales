from std.math import abs
from std.memory import alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import Point3f, Vec3f, RGB, Ray_C, Intersection_C, PathState_C, Material_C, MatKind
from gonzales.spectrum import SampledWavelengths
from gonzales.shading import shade_core
from _scene_fixture import make_triangle_scene

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _dummy_path(ray: Ray_C, throughput: RGB) -> PathState_C:
    return PathState_C(
        ray, throughput, RGB(Float32(0.0)), RGB(Float32(0.0)),
        Int32(0), UInt64(1), UInt64(1), Int8(1), Int8(0), Int8(0), Int8(0),
        Float32(0.0), Int32(-1), Int32(0), UInt64(0),
        SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )

# ── shade_core, exercised against a REAL BVH hit (via _scene_fixture) ───────
# Previously shading.mojo's helpers were tested with hand-built
# GeomContext/Intersection_C values (test_shading_helpers.mojo). This closes
# the loop one level further: build a real triangle + BVH, traverse a real
# ray through traverse_bvh2_core, and feed the resulting genuine
# Intersection_C into shade_core — the same data flow shade_core sees in an
# actual render, not a synthetic approximation of it.

def test_shade_core_area_light_hit_adds_emission() raises:
    var fx = make_triangle_scene([
        Point3f(0.0, 0.0, 0.0), Point3f(1.0, 0.0, 0.0), Point3f(0.0, 1.0, 0.0),
    ])
    fx.materials[0] = Material_C(
        MatKind.area_light, Int8(0), Int8(0), Int8(0),
        RGB(Float32(0.0)),                                  # albedo (unused for area_light)
        RGB(Float32(2.0), Float32(3.0), Float32(4.0)),      # emission
        Int32(-1), Float32(0.0), Float32(0.0), Int32(-1), Int32(-1),
        RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(1.0), Float32(1.0),
    )

    var ray = Ray_C(Point3f(0.2, 0.2, -5.0), Vec3f(0.0, 0.0, 1.0))
    var inter = fx.intersect(ray, Float32(100.0))
    assert_true(Int(inter.hit) == 1)

    var paths = alloc[PathState_C](1)
    var intersections = alloc[Intersection_C](1)
    paths[0] = _dummy_path(ray, RGB(Float32(0.5), Float32(0.5), Float32(0.5)))
    intersections[0] = inter

    shade_core(paths, intersections, fx.meshes, fx.materials, 0)

    # estimate += throughput * emission = 0.5*(2,3,4) = (1,1.5,2); path retires.
    assert_true(_close(paths[0].estimate.r, Float32(1.0)))
    assert_true(_close(paths[0].estimate.g, Float32(1.5)))
    assert_true(_close(paths[0].estimate.b, Float32(2.0)))
    assert_true(Int(paths[0].active) == 0)
    paths.free(); intersections.free()

def test_shade_core_miss_deactivates_path() raises:
    """A ray that misses the fixture's triangle entirely — shade_core must
    see inter.hit==0 and retire the path without touching estimate."""
    var fx = make_triangle_scene([
        Point3f(0.0, 0.0, 0.0), Point3f(1.0, 0.0, 0.0), Point3f(0.0, 1.0, 0.0),
    ])
    var ray = Ray_C(Point3f(50.0, 50.0, -5.0), Vec3f(0.0, 0.0, 1.0))
    var inter = fx.intersect(ray, Float32(100.0))
    assert_true(Int(inter.hit) == 0)

    var paths = alloc[PathState_C](1)
    var intersections = alloc[Intersection_C](1)
    paths[0] = _dummy_path(ray, RGB(Float32(1.0)))
    intersections[0] = inter

    shade_core(paths, intersections, fx.meshes, fx.materials, 0)

    assert_true(Int(paths[0].active) == 0)
    assert_true(_close(paths[0].estimate.r, Float32(0.0)))
    paths.free(); intersections.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
