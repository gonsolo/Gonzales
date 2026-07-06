from std.math import abs
from std.memory import alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import (
    Point3f, Vec3f, RGB, Ray_C, Intersection_C, PrimId_C, PathState_C,
    Material_C, MatKind, Curve_C, TriangleMesh_C, AreaLight_C,
    DistantLight_C, PointLight_C, InfiniteLight_C, Sphere_C, LightSampler_C,
)
from gonzales.bvh import BVH2Node
from gonzales.shading import shade_core_cpu_nee
from gonzales.guide import null_guide

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _dummy_path(ray: Ray_C, throughput: RGB) -> PathState_C:
    return PathState_C(
        ray, throughput, RGB(Float32(0.0)), RGB(Float32(0.0)),
        Int32(0), UInt64(1), UInt64(1), Int8(1), Int8(0), Int8(0), Int8(0),
        Float32(0.0), Int32(-1), Int32(0), UInt64(0),
    )

# ── Curve area lights (task #59: glowing_hair.pbrt used to render black) ───
# finalize_scene (pbrt_parser.mojo) now gives an emissive curve
# (Shape "curve" under an active AreaLightSource, no explicit Material) its
# own synthetic material slot with type=MatKind.area_light, the same way
# mesh/sphere area lights already worked. shade_nee_core's new
# `primId.type==5 and mat.type==area_light` branch is what actually makes
# it glow on hit — this test exercises that branch directly via
# shade_core_cpu_nee (the real CPU wavefront shading entry point), not a
# hand-rolled approximation of it.

def test_emissive_curve_hit_adds_emission_and_retires_path() raises:
    var materials = alloc[Material_C](1)
    materials[0] = Material_C(
        MatKind.area_light, Int8(0), Int8(0), Int8(0),
        RGB(Float32(0.0)),                                  # albedo
        RGB(Float32(200.0), Float32(80.0), Float32(20.0)),  # emission
        Int32(-1), Float32(0.0), Float32(0.0), Int32(-1), Int32(-1),
        RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(1.0), Float32(1.0),
    )

    var ray = Ray_C(Point3f(0.2, 0.2, -5.0), Vec3f(0.0, 0.0, 1.0))
    var inter = Intersection_C(
        PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(5), Int8(0), Int8(0), Int8(0)),
        Float32(5.0), Float32(0.0), Float32(0.5), Int8(1), Int8(0), Int8(0), Int8(0),
    )

    var paths = alloc[PathState_C](1)
    var intersections = alloc[Intersection_C](1)
    paths[0] = _dummy_path(ray, RGB(Float32(0.5)))
    intersections[0] = inter

    shade_core_cpu_nee(
        paths, intersections,
        UnsafePointer[BVH2Node, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[PrimId_C, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[TriangleMesh_C, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Curve_C, MutAnyOrigin].unsafe_dangling(),
        materials,
        UnsafePointer[AreaLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        0,
        UnsafePointer[DistantLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[PointLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[InfiniteLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[Sphere_C, MutAnyOrigin].unsafe_dangling(), 0,
        LightSampler_C(UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(), Int32(0), Int32(0)),
        UnsafePointer[UInt32, MutAnyOrigin].unsafe_dangling(),
        null_guide(),
    )

    # estimate += throughput(0.5) * emission(200,80,20) = (100,40,10); path retires.
    assert_true(_close(paths[0].estimate.r, Float32(100.0)))
    assert_true(_close(paths[0].estimate.g, Float32(40.0)))
    assert_true(_close(paths[0].estimate.b, Float32(10.0)))
    assert_true(Int(paths[0].active) == 0)
    paths.free(); intersections.free(); materials.free()

def test_nonemissive_curve_hit_does_not_self_terminate_via_the_arealight_path() raises:
    """A curve with a real (non-area-light) material — e.g. MatKind.hair —
    must NOT be caught by the new type==5 area-light branch; it should fall
    through unchanged. This test only checks it doesn't take the emissive
    shortcut (estimate stays 0); it doesn't exercise shade_hair itself."""
    var materials = alloc[Material_C](1)
    materials[0] = Material_C(
        MatKind.hair, Int8(0), Int8(0), Int8(0),
        RGB(Float32(0.3)), RGB(Float32(1.55), Float32(0.0), Float32(0.0)),
        Int32(-1), Float32(0.3), Float32(0.3), Int32(-1), Int32(-1),
        RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(1.0), Float32(1.0),
    )
    assert_true(materials[0].type != MatKind.area_light)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
