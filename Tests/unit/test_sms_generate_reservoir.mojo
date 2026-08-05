# Unit tests for shading.mojo's sms_generate_reservoir (Phase 6's
# candidate-generation piece, docs/A2_restir_migration_plan.md). Mirrors
# test_restir_gi_generation.mojo's harness style (real BVH + real
# ShadeContext), extended with a REAL materials array since, unlike GI's
# candidate generation, this function's own _sms_probe_and_solve dereferences
# ctx.materials to detect dielectric glass.

from std.math import abs, sqrt
from std.memory import alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import (
    RGB, Point3f, Vec3f, Ray_C, PrimId_C, TriangleMesh_C, Material_C, MatKind,
    Curve_C, GpuTexture_C, ShadowTask_C, LightSampler_C, Instance_C,
    AreaLight_C, DistantLight_C, PointLight_C, InfiniteLight_C, Sphere_C,
    MeasuredBRDF_C,
)
from gonzales.spectrum import null_spectral_handle
from gonzales.bvh import BVH2Node
from gonzales.guide import null_guide
from gonzales.shading import ShadeContext, LightContext, GIPendingX1, sms_generate_reservoir
from gonzales.restir_gi import gi_reservoir_io_null
from gonzales.rng import PCG32

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _make_triangle_mesh(p0: SIMD[DType.float32, 3], p1: SIMD[DType.float32, 3], p2: SIMD[DType.float32, 3]) -> TriangleMesh_C:
    var points = alloc[Float32](4 * 3)
    points[0*4+0] = p0[0]; points[0*4+1] = p0[1]; points[0*4+2] = p0[2]; points[0*4+3] = Float32(0.0)
    points[1*4+0] = p1[0]; points[1*4+1] = p1[1]; points[1*4+2] = p1[2]; points[1*4+3] = Float32(0.0)
    points[2*4+0] = p2[0]; points[2*4+1] = p2[1]; points[2*4+2] = p2[2]; points[2*4+3] = Float32(0.0)
    var vidx = alloc[Int64](3)
    vidx[0] = 0; vidx[1] = 1; vidx[2] = 2
    return TriangleMesh_C(
        points, UnsafePointer[Int64, MutAnyOrigin].unsafe_dangling(), vidx,
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    )

def _make_one_leaf_bvh(tri_min: SIMD[DType.float32, 3], tri_max: SIMD[DType.float32, 3]) -> BVH2Node:
    return BVH2Node(Point3f(tri_min[0], tri_min[1], tri_min[2]),
        Point3f(tri_max[0], tri_max[1], tri_max[2]), Int32(0), Int32(1))

def _make_dielectric(ior: Float32) -> Material_C:
    return Material_C(
        MatKind.dielectric, Int8(0), Int8(0), Int8(0),
        RGB(ior), RGB(Float32(0.0)),
        Int32(-1), Float32(0.0), Float32(0.0),
        Int32(-1), Int32(-1), Int32(-1),
        RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(1.0), Float32(1.0), Int32(-1))

def _make_ctx(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    area_lights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    area_light_count: Int,
    light_sampler_cdf: UnsafePointer[Float32, MutAnyOrigin],
) -> ShadeContext:
    return ShadeContext(
        0, bvh2Nodes, primIds, meshes,
        UnsafePointer[Curve_C, MutAnyOrigin].unsafe_dangling(),
        materials,
        UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[GpuTexture_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        Float32(0.0),
        UnsafePointer[UInt32, MutAnyOrigin].unsafe_dangling(),
        null_guide(),
        False,
        LightContext(
            area_lights, area_light_count,
            UnsafePointer[DistantLight_C, MutAnyOrigin].unsafe_dangling(), 0,
            UnsafePointer[PointLight_C, MutAnyOrigin].unsafe_dangling(), 0,
            UnsafePointer[InfiniteLight_C, MutAnyOrigin].unsafe_dangling(), 0,
            UnsafePointer[Sphere_C, MutAnyOrigin].unsafe_dangling(), 0,
            LightSampler_C(light_sampler_cdf, Int32(area_light_count), Int32(0))),
        UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Instance_C, MutAnyOrigin].unsafe_dangling(),
        null_spectral_handle(),
        UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[GIPendingX1, MutAnyOrigin].unsafe_dangling(),
        gi_reservoir_io_null(),
    )

# ── Shared fixture: a flat glass plane at z=1 between hit_point (origin,
# normal +Z) and a tiny light triangle near (0.3,-0.2,4.0) -- same synthetic
# geometry test_sms.mojo's own n=1 regression test already validated the
# underlying Newton solve against, reused here to validate the NEW wiring
# (target-pdf evaluation + reservoir streaming + field population) on top
# of it. ──────────────────────────────────────────────────────────────────

def _make_light_mesh() -> TriangleMesh_C:
    return _make_triangle_mesh(
        SIMD[DType.float32, 3](Float32(0.299), Float32(-0.201), Float32(4.0)),
        SIMD[DType.float32, 3](Float32(0.301), Float32(-0.201), Float32(4.0)),
        SIMD[DType.float32, 3](Float32(0.299), Float32(-0.199), Float32(4.0)))

def _make_glass_mesh() -> TriangleMesh_C:
    # A right triangle with generous legs (20 units) so the ray's crossing
    # point at (~0.075,~-0.05,1) sits comfortably inside (u+v~0.50), not
    # near the hypotenuse -- a first version of this fixture used 10-unit
    # legs and the crossing point landed at u+v~1.0025, just OUTSIDE the
    # valid barycentric range, causing a silent miss.
    return _make_triangle_mesh(
        SIMD[DType.float32, 3](Float32(-5.0), Float32(-5.0), Float32(1.0)),
        SIMD[DType.float32, 3](Float32(15.0), Float32(-5.0), Float32(1.0)),
        SIMD[DType.float32, 3](Float32(-5.0), Float32(15.0), Float32(1.0)))

def test_sms_generate_curve_light_returns_empty() raises:
    """Al.kind != 0 (a curve light) must bail out immediately, matching
    _mnee_area_light_contribute's own scope restriction -- no BVH probing
    even attempted."""
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0)), Float32(0.000002), Int8(1), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx(
        UnsafePointer[BVH2Node, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[PrimId_C, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[TriangleMesh_C, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Material_C, MutAnyOrigin].unsafe_dangling(),
        area_lights, 1, cdf)

    var pcg = PCG32(UInt64(1), UInt64(1))
    var res = sms_generate_reservoir(
        ctx, SIMD[DType.float32, 3](0.0, 0.0, 0.0), SIMD[DType.float32, 3](0.0, 0.0, 1.0), RGB(Float32(0.8)),
        SIMD[DType.float32, 3](0.0, 0.0, 1.0), Float32(4.0), SIMD[DType.float32, 3](0.3, -0.2, 4.0),
        SIMD[DType.float32, 3](1.0, 0.0, 0.0), SIMD[DType.float32, 3](0.0, 1.0, 0.0),
        area_lights[0], Float32(1.0), pcg)
    assert_true(res.n_vertices == Int32(0))
    area_lights.free(); cdf.free()

def test_sms_generate_no_glass_in_the_way_returns_empty() raises:
    """No dielectric intervenes (empty scene beyond the light) -- must
    return an empty reservoir, matching _mnee_area_light_contribute's own
    `dielectric_found=False` early return."""
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(SIMD[DType.float32, 3](1000.0, 1000.0, 1000.0), SIMD[DType.float32, 3](1001.0, 1001.0, 1001.0))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_light_mesh()
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx(bvh, primIds, meshes,
        UnsafePointer[Material_C, MutAnyOrigin].unsafe_dangling(),
        area_lights, 1, cdf)

    var pcg = PCG32(UInt64(1), UInt64(1))
    var res = sms_generate_reservoir(
        ctx, SIMD[DType.float32, 3](0.0, 0.0, 0.0), SIMD[DType.float32, 3](0.0, 0.0, 1.0), RGB(Float32(0.8)),
        SIMD[DType.float32, 3](0.0, 0.0, 1.0), Float32(4.0), SIMD[DType.float32, 3](0.3, -0.2, 4.0),
        SIMD[DType.float32, 3](1.0, 0.0, 0.0), SIMD[DType.float32, 3](0.0, 1.0, 0.0),
        area_lights[0], Float32(1.0), pcg)
    assert_true(res.n_vertices == Int32(0))

    meshes[0].points.free(); meshes[0].vertexIndices.free(); meshes.free()
    area_lights.free(); primIds.free(); bvh.free(); cdf.free()

def test_sms_generate_real_glass_produces_a_streamed_candidate() raises:
    """A real flat glass plane sits between hit_point and the light -- the
    1-vertex MNEE fast path (inside _sms_probe_and_solve) must converge,
    sms_target_pdf must evaluate positive, and the resulting reservoir must
    be a real, streamed (m=1) candidate with n_vertices=1 and a populated
    chain/light payload."""
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(SIMD[DType.float32, 3](-5.0, -5.0, Float32(0.9)), SIMD[DType.float32, 3](15.0, 15.0, Float32(1.1)))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_glass_mesh()
    var materials = alloc[Material_C](1)
    materials[0] = _make_dielectric(Float32(1.5))
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx(bvh, primIds, meshes, materials, area_lights, 1, cdf)

    var hit_point = SIMD[DType.float32, 3](0.0, 0.0, 0.0)
    var normal = SIMD[DType.float32, 3](0.0, 0.0, 1.0)
    var light_point = SIMD[DType.float32, 3](0.3, -0.2, 4.0)
    var shadow_dir_v = light_point - hit_point
    var shadow_dist = sqrt(shadow_dir_v[0]*shadow_dir_v[0] + shadow_dir_v[1]*shadow_dir_v[1] + shadow_dir_v[2]*shadow_dir_v[2])
    var shadow_dir = shadow_dir_v * (Float32(1.0) / shadow_dist)

    var pcg = PCG32(UInt64(1), UInt64(1))
    var res = sms_generate_reservoir(
        ctx, hit_point, normal, RGB(Float32(0.8)),
        shadow_dir, shadow_dist, light_point,
        SIMD[DType.float32, 3](1.0, 0.0, 0.0), SIMD[DType.float32, 3](0.0, 1.0, 0.0),
        area_lights[0], Float32(1.0), pcg)

    assert_true(res.n_vertices == Int32(1))
    assert_true(res.state.m > Float32(0.0))
    assert_true(res.state.w_sum > Float32(0.0))
    assert_true(_close(res.le.r, Float32(200.0)))
    assert_true(_close(res.light_point[2], Float32(4.0)))

    meshes[0].points.free(); meshes[0].vertexIndices.free(); meshes.free()
    materials.free(); area_lights.free(); primIds.free(); bvh.free(); cdf.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
