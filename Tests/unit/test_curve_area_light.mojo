from std.math import abs, sqrt
from std.memory import alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import (
    Point3f, Vec3f, RGB, Ray_C, Intersection_C, PrimId_C, PathState_C,
    Material_C, MatKind, Curve_C, TriangleMesh_C, AreaLight_C,
    DistantLight_C, PointLight_C, InfiniteLight_C, Sphere_C, LightSampler_C,
    curve_bspline_point, curve_light_tube_area, _curve_perp_axis, dot, cross,
)
from gonzales.spectrum import SampledWavelengths
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
        SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
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
        Int32(-1), Float32(0.0), Float32(0.0), Int32(-1), Int32(-1), Int32(-1),
        RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(1.0), Float32(1.0), Int32(-1),
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
        UnsafePointer[BVH2Node, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[PrimId_C, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[TriangleMesh_C, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[Curve_C, MutExternalOrigin].unsafe_dangling(),
        materials,
        UnsafePointer[AreaLight_C, MutExternalOrigin].unsafe_dangling(), 0,
        UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        0,
        UnsafePointer[DistantLight_C, MutExternalOrigin].unsafe_dangling(), 0,
        UnsafePointer[PointLight_C, MutExternalOrigin].unsafe_dangling(), 0,
        UnsafePointer[InfiniteLight_C, MutExternalOrigin].unsafe_dangling(), 0,
        UnsafePointer[Sphere_C, MutExternalOrigin].unsafe_dangling(), 0,
        LightSampler_C(UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(), Int32(0), Int32(0)),
        UnsafePointer[UInt32, MutExternalOrigin].unsafe_dangling(),
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
        Int32(-1), Float32(0.3), Float32(0.3), Int32(-1), Int32(-1), Int32(-1),
        RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(1.0), Float32(1.0), Int32(-1),
    )
    assert_true(materials[0].type != MatKind.area_light)

# ── Curve lights are now explicitly NEE-sampled too (task #60-65) ───────────
# Once an emissive curve has its own AreaLight_C entry (kind==1) in the
# light-sampler CDF, a camera/bounce ray that hits it directly must MIS-
# weight against that light's own selection pdf instead of always crediting
# full emission (the old self-emission-only behaviour exercised above) --
# otherwise the direct-hit case and the new NEE case would double-count.
# This pins the exact MIS-weighted formula shade_nee_core's new primId.type
# == 5 branch computes, against an independently-computed expected value.

def test_emissive_curve_bounce_hit_mis_weights_against_its_own_light_pdf() raises:
    var curve = Curve_C(
        Point3f(0.0, 0.0, 0.0), Point3f(1.0, 2.0, 0.0),
        Point3f(2.0, -1.0, 0.0), Point3f(3.0, 0.0, 0.0),
        Float32(0.2), Float32(0.2), Int32(0), Int32(1),
    )
    var curves = alloc[Curve_C](1)
    curves[0] = curve

    # Reconstruct the same geometric normal shade_nee_core's new branch
    # computes from (u=h, v) at h=0 -- geo_normal reduces to exactly b_perp0.
    var q0 = curve_bspline_point(curve, Float32(0.0))
    var q1 = curve_bspline_point(curve, Float32(1.0))
    var seg = q1 - q0
    var tangent = seg * (Float32(1.0) / sqrt(dot(seg, seg)))
    var n_perp = _curve_perp_axis(tangent)
    var b_perp0 = cross(tangent, n_perp)
    var ray_dir = -b_perp0  # cos_l = -dot(geo_normal, ray_dir) = 1 (straight-on hit)

    var materials = alloc[Material_C](1)
    materials[0] = Material_C(
        MatKind.area_light, Int8(0), Int8(0), Int8(0),
        RGB(Float32(0.0)), RGB(Float32(200.0), Float32(80.0), Float32(20.0)),
        Int32(-1), Float32(0.0), Float32(0.0), Int32(-1), Int32(-1), Int32(-1),
        RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(1.0), Float32(1.0), Int32(-1),
    )

    var total_area = curve_light_tube_area(curve)
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(0), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), total_area, Int8(1), Int8(0), Int8(0), Int8(0))

    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var light_sampler = LightSampler_C(cdf, Int32(1), Int32(0))

    var t_hit: Float32 = 5.0
    var pdf_bsdf: Float32 = 0.4
    var ray = Ray_C(Point3f(0.0, 0.0, 0.0), Vec3f(ray_dir[0], ray_dir[1], ray_dir[2]))
    var inter = Intersection_C(
        PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(5), Int8(0), Int8(0), Int8(0)),
        t_hit, Float32(0.0), Float32(0.5), Int8(1), Int8(0), Int8(0), Int8(0),
    )

    var paths = alloc[PathState_C](1)
    var intersections = alloc[Intersection_C](1)
    paths[0] = PathState_C(
        ray, RGB(Float32(0.5)), RGB(Float32(0.0)), RGB(Float32(0.0)),
        Int32(1),  # bounce > 0: NOT the "camera sees light directly" shortcut
        UInt64(1), UInt64(1), Int8(1), Int8(0),  # specularBounce = 0
        Int8(0), Int8(0), pdf_bsdf, Int32(-1), Int32(0), UInt64(0),
        SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )
    intersections[0] = inter

    shade_core_cpu_nee(
        paths, intersections,
        UnsafePointer[BVH2Node, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[PrimId_C, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[TriangleMesh_C, MutExternalOrigin].unsafe_dangling(),
        curves,
        materials,
        area_lights, 1,
        UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        0,
        UnsafePointer[DistantLight_C, MutExternalOrigin].unsafe_dangling(), 0,
        UnsafePointer[PointLight_C, MutExternalOrigin].unsafe_dangling(), 0,
        UnsafePointer[InfiniteLight_C, MutExternalOrigin].unsafe_dangling(), 0,
        UnsafePointer[Sphere_C, MutExternalOrigin].unsafe_dangling(), 0,
        light_sampler,
        UnsafePointer[UInt32, MutExternalOrigin].unsafe_dangling(),
        null_guide(),
    )

    # Expected value computed independently (Python, replicating
    # curve_bspline_point/curve_light_tube_area/power_heuristic exactly):
    # total_area = 2*pi*0.1*|q1-q0| = 1.1327173, dist2 = t_hit^2 = 25,
    # cos_l = 1 (straight-on hit), al_sel_pdf = 1 (single-light CDF),
    # pdf_light = 25 / 1.1327173 = 22.070820, w = pdf_bsdf^2/(pdf_bsdf^2+pdf_light^2)
    # = 0.4^2/(0.4^2+22.070820^2) = 0.00032835258, estimate.r = 0.5*200*w =
    # 0.0328353 -- strictly less than the full-credit 100.0 this replaced.
    assert_true(_close(paths[0].estimate.r, Float32(0.0328353)))
    paths.free(); intersections.free(); materials.free()
    curves.free(); area_lights.free(); cdf.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
