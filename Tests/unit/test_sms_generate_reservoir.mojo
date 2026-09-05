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
    Curve_C, GpuTexture_C, NormalSlopeMap_C, ShadowTask_C, LightSampler_C, Instance_C,
    AreaLight_C, DistantLight_C, PointLight_C, InfiniteLight_C, Sphere_C,
    MeasuredBRDF_C, PathState_C,
)
from gonzales.spectrum import null_spectral_handle, SampledWavelengths
from gonzales.bvh import BVH2Node
from gonzales.guide import null_guide
from gonzales.shading import ShadeContext, LightContext, GIPendingX1, sms_generate_reservoir, sms_resolve, sms_temporal_step, _shade_diffuse_nee
from gonzales.restir_gi import gi_reservoir_io_null
from gonzales.restir_di import reservoir_io_null
from gonzales.restir_sms import SMSReservoir, SMSReservoirIO, sms_reservoir_io_null
from gonzales.rng import PCG32

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _make_triangle_mesh(p0: Vec3f, p1: Vec3f, p2: Vec3f) -> TriangleMesh_C:
    var points = alloc[Float32](4 * 3)
    points[0*4+0] = p0[0]; points[0*4+1] = p0[1]; points[0*4+2] = p0[2]; points[0*4+3] = Float32(0.0)
    points[1*4+0] = p1[0]; points[1*4+1] = p1[1]; points[1*4+2] = p1[2]; points[1*4+3] = Float32(0.0)
    points[2*4+0] = p2[0]; points[2*4+1] = p2[1]; points[2*4+2] = p2[2]; points[2*4+3] = Float32(0.0)
    var vidx = alloc[Int64](3)
    vidx[0] = 0; vidx[1] = 1; vidx[2] = 2
    return TriangleMesh_C(
        points, UnsafePointer[Int64, MutExternalOrigin].unsafe_dangling(), vidx,
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    )

def _make_one_leaf_bvh(tri_min: Vec3f, tri_max: Vec3f) -> BVH2Node:
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
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    area_lights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    area_light_count: Int,
    light_sampler_cdf: UnsafePointer[Float32, MutExternalOrigin],
) -> ShadeContext:
    return ShadeContext(
        0, bvh2Nodes, primIds, meshes,
        UnsafePointer[Curve_C, MutExternalOrigin].unsafe_dangling(),
        materials,
        UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[GpuTexture_C, MutExternalOrigin].unsafe_dangling(), 0,
        UnsafePointer[NormalSlopeMap_C, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[ShadowTask_C, MutExternalOrigin].unsafe_dangling(),
        Float32(0.0),
        UnsafePointer[UInt32, MutExternalOrigin].unsafe_dangling(),
        null_guide(),
        False,
        LightContext(
            area_lights, area_light_count,
            UnsafePointer[DistantLight_C, MutExternalOrigin].unsafe_dangling(), 0,
            UnsafePointer[PointLight_C, MutExternalOrigin].unsafe_dangling(), 0,
            UnsafePointer[InfiniteLight_C, MutExternalOrigin].unsafe_dangling(), 0,
            UnsafePointer[Sphere_C, MutExternalOrigin].unsafe_dangling(), 0,
            LightSampler_C(light_sampler_cdf, Int32(area_light_count), Int32(0))),
        UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[Instance_C, MutExternalOrigin].unsafe_dangling(),
        null_spectral_handle(),
        UnsafePointer[MeasuredBRDF_C, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling(),
        gi_reservoir_io_null(),
    )

# ── Shared fixture: a flat glass plane at z=1 between hit_point (origin,
# normal +Z) and a tiny light triangle near (0.3,-0.2,4.0) -- same synthetic
# geometry test_sms.mojo's own n=1 regression test already validated the
# underlying Newton solve against, reused here to validate the NEW wiring
# (target-pdf evaluation + reservoir streaming + field population) on top
# of it. ──────────────────────────────────────────────────────────────────

def _make_light_mesh() -> TriangleMesh_C:
    # Winding chosen so cross(p1-p0, p2-p0) points -Z (roughly toward the
    # origin, where every fixture's hit_point sits) -- _nee_area_lights'
    # own cos_l = -dot(light_normal, shadow_dir) gate needs the light's
    # normal facing back toward the shading point to accept the sample.
    # (Standalone sms_generate_reservoir/sms_resolve calls elsewhere in
    # this file bypass that gate entirely by passing light_point directly,
    # so they don't care about this winding -- only the _nee_area_lights-
    # routed wiring tests do.)
    return _make_triangle_mesh(
        Vec3f(Float32(0.299), Float32(-0.201), Float32(4.0)),
        Vec3f(Float32(0.299), Float32(-0.199), Float32(4.0)),
        Vec3f(Float32(0.301), Float32(-0.201), Float32(4.0)))

def _make_glass_mesh() -> TriangleMesh_C:
    # A right triangle with generous legs (20 units) so the ray's crossing
    # point at (~0.075,~-0.05,1) sits comfortably inside (u+v~0.50), not
    # near the hypotenuse -- a first version of this fixture used 10-unit
    # legs and the crossing point landed at u+v~1.0025, just OUTSIDE the
    # valid barycentric range, causing a silent miss.
    return _make_triangle_mesh(
        Vec3f(Float32(-5.0), Float32(-5.0), Float32(1.0)),
        Vec3f(Float32(15.0), Float32(-5.0), Float32(1.0)),
        Vec3f(Float32(-5.0), Float32(15.0), Float32(1.0)))

def test_sms_generate_curve_light_returns_empty() raises:
    """Al.kind != 0 (a curve light) must bail out immediately, matching
    _mnee_area_light_contribute's own scope restriction -- no BVH probing
    even attempted."""
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0)), Float32(0.000002), Int8(1), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx(
        UnsafePointer[BVH2Node, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[PrimId_C, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[TriangleMesh_C, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[Material_C, MutExternalOrigin].unsafe_dangling(),
        area_lights, 1, cdf)

    var pcg = PCG32(UInt64(1), UInt64(1))
    var gen_result = sms_generate_reservoir(
        ctx, Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 1.0), RGB(Float32(0.8)),
        Vec3f(0.0, 0.0, 1.0), Float32(4.0), Vec3f(0.3, -0.2, 4.0),
        Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0),
        area_lights[0], Float32(1.0), pcg)
    assert_true(gen_result[0] == False)
    assert_true(gen_result[1].n_vertices == Int32(0))
    area_lights.free(); cdf.free()

def test_sms_generate_no_glass_in_the_way_returns_empty() raises:
    """No dielectric intervenes (empty scene beyond the light) -- must
    return an empty reservoir, matching _mnee_area_light_contribute's own
    `dielectric_found=False` early return."""
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(Vec3f(1000.0, 1000.0, 1000.0), Vec3f(1001.0, 1001.0, 1001.0))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_light_mesh()
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx(bvh, primIds, meshes,
        UnsafePointer[Material_C, MutExternalOrigin].unsafe_dangling(),
        area_lights, 1, cdf)

    var pcg = PCG32(UInt64(1), UInt64(1))
    var gen_result = sms_generate_reservoir(
        ctx, Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 1.0), RGB(Float32(0.8)),
        Vec3f(0.0, 0.0, 1.0), Float32(4.0), Vec3f(0.3, -0.2, 4.0),
        Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0),
        area_lights[0], Float32(1.0), pcg)
    assert_true(gen_result[0] == False)
    assert_true(gen_result[1].n_vertices == Int32(0))

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
    bvh[0] = _make_one_leaf_bvh(Vec3f(-5.0, -5.0, Float32(0.9)), Vec3f(15.0, 15.0, Float32(1.1)))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_glass_mesh()
    var materials = alloc[Material_C](1)
    materials[0] = _make_dielectric(Float32(1.5))
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx(bvh, primIds, meshes, materials, area_lights, 1, cdf)

    var hit_point = Vec3f(0.0, 0.0, 0.0)
    var normal = Vec3f(0.0, 0.0, 1.0)
    var light_point = Vec3f(0.3, -0.2, 4.0)
    var shadow_dir_v = light_point - hit_point
    var shadow_dist = sqrt(shadow_dir_v[0]*shadow_dir_v[0] + shadow_dir_v[1]*shadow_dir_v[1] + shadow_dir_v[2]*shadow_dir_v[2])
    var shadow_dir = shadow_dir_v * (Float32(1.0) / shadow_dist)

    var pcg = PCG32(UInt64(1), UInt64(1))
    var gen_result = sms_generate_reservoir(
        ctx, hit_point, normal, RGB(Float32(0.8)),
        shadow_dir, shadow_dist, light_point,
        Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0),
        area_lights[0], Float32(1.0), pcg)
    var res = gen_result[1].copy()

    assert_true(gen_result[0] == True)
    assert_true(res.n_vertices == Int32(1))
    assert_true(res.state.m > Float32(0.0))
    assert_true(res.state.w_sum > Float32(0.0))
    assert_true(_close(res.le.r, Float32(200.0)))
    assert_true(_close(res.light_point[2], Float32(4.0)))

    meshes[0].points.free(); meshes[0].vertexIndices.free(); meshes.free()
    materials.free(); area_lights.free(); primIds.free(); bvh.free(); cdf.free()

# ── sms_resolve ──────────────────────────────────────────────────────────────

def _make_path() -> PathState_C:
    return PathState_C(
        Ray_C(Point3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 1.0)),
        RGB(Float32(1.0)), RGB(Float32(0.0)), RGB(Float32(0.0)),
        Int32(0), UInt64(1), UInt64(1), Int8(1), Int8(0), Int8(0), Int8(0), Int8(0), Vec3f(Float32(0.0)),
        Float32(0.0), Int32(-1), Int32(0), UInt64(0),
        SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )

def test_sms_resolve_on_empty_reservoir_is_a_noop() raises:
    """N_vertices=0 must return immediately without touching estimate or
    crashing on the dangling glass-chain data."""
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(Vec3f(1000.0, 1000.0, 1000.0), Vec3f(1001.0, 1001.0, 1001.0))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_light_mesh()
    var cdf = alloc[Float32](1)
    cdf[0] = Float32(0.0)
    var ctx = _make_ctx(bvh, primIds, meshes,
        UnsafePointer[Material_C, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[AreaLight_C, MutExternalOrigin].unsafe_dangling(), 0, cdf)

    var path_arr = alloc[PathState_C](1)
    path_arr[0] = _make_path()
    var pcg_gen0 = PCG32(UInt64(1), UInt64(1))
    var gen_result0 = sms_generate_reservoir(
        ctx, Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 1.0), RGB(Float32(0.8)),
        Vec3f(0.0, 0.0, 1.0), Float32(4.0), Vec3f(0.3, -0.2, 4.0),
        Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0),
        AreaLight_C(Int32(0), Int32(1), RGB(Float32(0.0)), Float32(0.0), Int8(0), Int8(0), Int8(0), Int8(0)),
        Float32(1.0), pcg_gen0)
    var res = gen_result0[1].copy()
    assert_true(res.n_vertices == Int32(0))

    sms_resolve(path_arr, ctx, Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 1.0), RGB(Float32(0.8)), res)
    assert_true(_close(path_arr[0].estimate.r, Float32(0.0)))

    meshes[0].points.free(); meshes[0].vertexIndices.free(); meshes.free()
    primIds.free(); bvh.free(); cdf.free(); path_arr.free()

def test_sms_resolve_on_real_glass_adds_positive_contribution() raises:
    """A real, unshadowed streamed candidate must finalize to state.w > 0
    and add a strictly positive contribution to path_ptr[].estimate."""
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(Vec3f(-5.0, -5.0, Float32(0.9)), Vec3f(15.0, 15.0, Float32(1.1)))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_glass_mesh()
    var materials = alloc[Material_C](1)
    materials[0] = _make_dielectric(Float32(1.5))
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx(bvh, primIds, meshes, materials, area_lights, 1, cdf)

    var hit_point = Vec3f(0.0, 0.0, 0.0)
    var normal = Vec3f(0.0, 0.0, 1.0)
    var light_point = Vec3f(0.3, -0.2, 4.0)
    var shadow_dir_v = light_point - hit_point
    var shadow_dist = sqrt(shadow_dir_v[0]*shadow_dir_v[0] + shadow_dir_v[1]*shadow_dir_v[1] + shadow_dir_v[2]*shadow_dir_v[2])
    var shadow_dir = shadow_dir_v * (Float32(1.0) / shadow_dist)
    var alb = RGB(Float32(0.8))

    var pcg_gen = PCG32(UInt64(1), UInt64(1))
    var gen_result = sms_generate_reservoir(
        ctx, hit_point, normal, alb, shadow_dir, shadow_dist, light_point,
        Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0),
        area_lights[0], Float32(1.0), pcg_gen)
    var res = gen_result[1].copy()
    assert_true(res.n_vertices == Int32(1))

    var path_arr = alloc[PathState_C](1)
    path_arr[0] = _make_path()
    sms_resolve(path_arr, ctx, hit_point, normal, alb, res)
    assert_true(res.state.w > Float32(0.0))
    assert_true(path_arr[0].estimate.r > Float32(0.0))
    assert_true(path_arr[0].estimate.g > Float32(0.0))
    assert_true(path_arr[0].estimate.b > Float32(0.0))

    meshes[0].points.free(); meshes[0].vertexIndices.free(); meshes.free()
    materials.free(); area_lights.free(); primIds.free(); bvh.free(); cdf.free(); path_arr.free()

# ── sms_temporal_step ────────────────────────────────────────────────────────

def test_sms_temporal_step_without_io_still_resolves_like_batch_mode() raises:
    """Pixel_idx=-1 / no real SMSReservoirIO -- must still generate+resolve
    a single-frame candidate (the batch/non-interactive fallback), matching
    di_temporal_step's own documented fallback behavior."""
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(Vec3f(-5.0, -5.0, Float32(0.9)), Vec3f(15.0, 15.0, Float32(1.1)))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_glass_mesh()
    var materials = alloc[Material_C](1)
    materials[0] = _make_dielectric(Float32(1.5))
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx(bvh, primIds, meshes, materials, area_lights, 1, cdf)

    var hit_point = Vec3f(0.0, 0.0, 0.0)
    var normal = Vec3f(0.0, 0.0, 1.0)
    var light_point = Vec3f(0.3, -0.2, 4.0)
    var shadow_dir_v = light_point - hit_point
    var shadow_dist = sqrt(shadow_dir_v[0]*shadow_dir_v[0] + shadow_dir_v[1]*shadow_dir_v[1] + shadow_dir_v[2]*shadow_dir_v[2])
    var shadow_dir = shadow_dir_v * (Float32(1.0) / shadow_dist)
    var alb = RGB(Float32(0.8))
    var pcg = PCG32(UInt64(1), UInt64(1))

    var path_arr = alloc[PathState_C](1)
    path_arr[0] = _make_path()
    var found = sms_temporal_step(
        path_arr, ctx, hit_point, normal, alb, shadow_dir, shadow_dist, light_point,
        Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0),
        area_lights[0], Float32(1.0), pcg)
    assert_true(found == True)
    assert_true(path_arr[0].estimate.r > Float32(0.0))

    meshes[0].points.free(); meshes[0].vertexIndices.free(); meshes.free()
    materials.free(); area_lights.free(); primIds.free(); bvh.free(); cdf.free(); path_arr.free()

def test_sms_temporal_step_second_frame_accumulates_confidence() raises:
    """Two consecutive frames at the SAME pixel with a real SMSReservoirIO
    buffer -- the second frame's stored reservoir must end up with HIGHER
    confidence (state.m) than a single frame alone, confirming temporal
    reservoir_combine actually ran (not just independent per-frame solves)."""
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(Vec3f(-5.0, -5.0, Float32(0.9)), Vec3f(15.0, 15.0, Float32(1.1)))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_glass_mesh()
    var materials = alloc[Material_C](1)
    materials[0] = _make_dielectric(Float32(1.5))
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx(bvh, primIds, meshes, materials, area_lights, 1, cdf)

    var hit_point = Vec3f(0.0, 0.0, 0.0)
    var normal = Vec3f(0.0, 0.0, 1.0)
    var light_point = Vec3f(0.3, -0.2, 4.0)
    var shadow_dir_v = light_point - hit_point
    var shadow_dist = sqrt(shadow_dir_v[0]*shadow_dir_v[0] + shadow_dir_v[1]*shadow_dir_v[1] + shadow_dir_v[2]*shadow_dir_v[2])
    var shadow_dir = shadow_dir_v * (Float32(1.0) / shadow_dist)
    var alb = RGB(Float32(0.8))

    var buf_a = alloc[SMSReservoir](1)
    var buf_b = alloc[SMSReservoir](1)
    var pcg_seed = PCG32(UInt64(99), UInt64(1))
    var res_empty = sms_generate_reservoir(
        ctx, hit_point, normal, alb, shadow_dir, shadow_dist, light_point,
        Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0),
        AreaLight_C(Int32(0), Int32(1), RGB(Float32(0.0)), Float32(0.0), Int8(0), Int8(0), Int8(0), Int8(0)),
        Float32(1.0), pcg_seed)
    buf_a[0] = res_empty[1].copy()
    buf_b[0] = res_empty[1].copy()

    var path_arr = alloc[PathState_C](1)

    # Frame 0: read=buf_a (empty), write=buf_b.
    var io0 = SMSReservoirIO(read=buf_a, write=buf_b,
        gbuf_normal=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_depth=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_material_id=UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
        gbuf_world_pos=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        frame_w=Int32(0), frame_h=Int32(0))
    path_arr[0] = _make_path()
    var pcg0 = PCG32(UInt64(1), UInt64(1))
    _ = sms_temporal_step(
        path_arr, ctx, hit_point, normal, alb, shadow_dir, shadow_dist, light_point,
        Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0),
        area_lights[0], Float32(1.0), pcg0, io0, 0)
    var m_after_frame0 = buf_b[0].state.m
    assert_true(buf_b[0].n_vertices == Int32(1))
    assert_true(m_after_frame0 > Float32(0.0))

    # Frame 1: read=buf_b (frame 0's result), write=buf_a -- mirrors
    # pipeline.mojo's own ping-pong swap between frames.
    var io1 = SMSReservoirIO(read=buf_b, write=buf_a,
        gbuf_normal=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_depth=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_material_id=UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
        gbuf_world_pos=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        frame_w=Int32(0), frame_h=Int32(0))
    path_arr[0] = _make_path()
    var pcg1 = PCG32(UInt64(2), UInt64(1))
    _ = sms_temporal_step(
        path_arr, ctx, hit_point, normal, alb, shadow_dir, shadow_dist, light_point,
        Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0),
        area_lights[0], Float32(1.0), pcg1, io1, 0)
    var m_after_frame1 = buf_a[0].state.m

    assert_true(buf_a[0].n_vertices == Int32(1))
    assert_true(m_after_frame1 > m_after_frame0)
    assert_true(path_arr[0].estimate.r > Float32(0.0))

    meshes[0].points.free(); meshes[0].vertexIndices.free(); meshes.free()
    materials.free(); area_lights.free(); primIds.free(); bvh.free(); cdf.free()
    path_arr.free(); buf_a.free(); buf_b.free()

# ── End-to-end wiring: _shade_diffuse_nee -> _nee_area_lights -> ───────────
# sms_temporal_step, mirroring test_restir_gi_generation.mojo's own
# end-to-end GI wiring tests. Confirms the ACTUAL integration point (not
# just the standalone functions already covered above): a real sms_io
# passed into _shade_diffuse_nee at bounce 0, with ctx.use_restir=False
# so _nee_area_lights (not di_temporal_step) handles this bounce.

def test_shade_diffuse_nee_sms_wiring_accumulates_confidence_across_frames() raises:
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(Vec3f(-5.0, -5.0, Float32(0.9)), Vec3f(15.0, 15.0, Float32(1.1)))
    # primId references mesh index 1 (the glass) -- mesh index 0 is the
    # light, sampled directly via ctx.meshes[al.meshIdx], never through
    # the BVH at all.
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(1), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](2)
    meshes[0] = _make_light_mesh()
    meshes[1] = _make_glass_mesh()
    var materials = alloc[Material_C](1)
    materials[0] = _make_dielectric(Float32(1.5))
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx(bvh, primIds, meshes, materials, area_lights, 1, cdf)

    var hit_point = Vec3f(0.0, 0.0, 0.0)
    var normal = Vec3f(0.0, 0.0, 1.0)
    var alb = RGB(Float32(0.8))

    var buf_a = alloc[SMSReservoir](1)
    var buf_b = alloc[SMSReservoir](1)
    var pcg_seed = PCG32(UInt64(99), UInt64(1))
    var shadow_dir_v = Vec3f(0.3, -0.2, 4.0) - hit_point
    var shadow_dist = sqrt(shadow_dir_v[0]*shadow_dir_v[0] + shadow_dir_v[1]*shadow_dir_v[1] + shadow_dir_v[2]*shadow_dir_v[2])
    var shadow_dir = shadow_dir_v * (Float32(1.0) / shadow_dist)
    var res_empty = sms_generate_reservoir(
        ctx, hit_point, normal, alb, shadow_dir, shadow_dist, Vec3f(0.3, -0.2, 4.0),
        Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0),
        AreaLight_C(Int32(0), Int32(1), RGB(Float32(0.0)), Float32(0.0), Int8(0), Int8(0), Int8(0), Int8(0)),
        Float32(1.0), pcg_seed)
    buf_a[0] = res_empty[1].copy()
    buf_b[0] = res_empty[1].copy()

    var path_arr = alloc[PathState_C](1)

    # Frame 0, through the real _shade_diffuse_nee entry point.
    var io0 = SMSReservoirIO(read=buf_a, write=buf_b,
        gbuf_normal=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_depth=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_material_id=UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
        gbuf_world_pos=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        frame_w=Int32(0), frame_h=Int32(0))
    path_arr[0] = _make_path()
    var pcg0 = PCG32(UInt64(1), UInt64(1))
    _shade_diffuse_nee[False, False](
        path_arr, ctx, normal, hit_point, alb, Vec3f(0.0, 0.0, -1.0),
        Float32(0.5), Float32(0.5), Float32(0.5), Float32(0.5), Float32(0.5),
        pcg0, null_guide(), reservoir_io_null(), 0, io0)
    var m_after_frame0 = buf_b[0].state.m
    assert_true(buf_b[0].n_vertices == Int32(1))
    assert_true(m_after_frame0 > Float32(0.0))
    assert_true(path_arr[0].estimate.r > Float32(0.0))

    # Frame 1: read=buf_b (frame 0's result), write=buf_a.
    var io1 = SMSReservoirIO(read=buf_b, write=buf_a,
        gbuf_normal=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_depth=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_material_id=UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
        gbuf_world_pos=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        frame_w=Int32(0), frame_h=Int32(0))
    path_arr[0] = _make_path()
    var pcg1 = PCG32(UInt64(2), UInt64(1))
    _shade_diffuse_nee[False, False](
        path_arr, ctx, normal, hit_point, alb, Vec3f(0.0, 0.0, -1.0),
        Float32(0.5), Float32(0.5), Float32(0.5), Float32(0.5), Float32(0.5),
        pcg1, null_guide(), reservoir_io_null(), 0, io1)
    var m_after_frame1 = buf_a[0].state.m

    assert_true(buf_a[0].n_vertices == Int32(1))
    assert_true(m_after_frame1 > m_after_frame0)
    assert_true(path_arr[0].estimate.r > Float32(0.0))

    meshes[0].points.free(); meshes[0].vertexIndices.free()
    meshes[1].points.free(); meshes[1].vertexIndices.free()
    meshes.free()
    materials.free(); area_lights.free(); primIds.free(); bvh.free(); cdf.free()
    path_arr.free(); buf_a.free(); buf_b.free()

def test_shade_diffuse_nee_sms_io_inactive_at_bounce_1_uses_plain_mnee() raises:
    """Sms_io is only threaded through at bounce 0 -- at bounce 1 it must
    fall back to plain per-frame MNEE (sms_io_this_bounce forced to the
    null sentinel), even though a real buffer was passed in. Confirms the
    bounce-0-only gate actually gates, not just that it compiles."""
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(Vec3f(-5.0, -5.0, Float32(0.9)), Vec3f(15.0, 15.0, Float32(1.1)))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(1), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](2)
    meshes[0] = _make_light_mesh()
    meshes[1] = _make_glass_mesh()
    var materials = alloc[Material_C](1)
    materials[0] = _make_dielectric(Float32(1.5))
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx(bvh, primIds, meshes, materials, area_lights, 1, cdf)

    var hit_point = Vec3f(0.0, 0.0, 0.0)
    var normal = Vec3f(0.0, 0.0, 1.0)
    var alb = RGB(Float32(0.8))

    var buf_a = alloc[SMSReservoir](1)
    var buf_b = alloc[SMSReservoir](1)
    var pcg_seed = PCG32(UInt64(99), UInt64(1))
    var shadow_dir_v = Vec3f(0.3, -0.2, 4.0) - hit_point
    var shadow_dist = sqrt(shadow_dir_v[0]*shadow_dir_v[0] + shadow_dir_v[1]*shadow_dir_v[1] + shadow_dir_v[2]*shadow_dir_v[2])
    var shadow_dir = shadow_dir_v * (Float32(1.0) / shadow_dist)
    var res_empty = sms_generate_reservoir(
        ctx, hit_point, normal, alb, shadow_dir, shadow_dist, Vec3f(0.3, -0.2, 4.0),
        Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0),
        AreaLight_C(Int32(0), Int32(1), RGB(Float32(0.0)), Float32(0.0), Int8(0), Int8(0), Int8(0), Int8(0)),
        Float32(1.0), pcg_seed)
    # reservoir_update always increments state.m by 1 whether the candidate
    # was accepted or not (Le=0 here forces rejection, but m still moves to
    # 1.0) -- the seed reservoir's m is 1.0, not 0.0.
    var seed_m = res_empty[1].state.m
    buf_a[0] = res_empty[1].copy()
    buf_b[0] = res_empty[1].copy()

    var path_arr = alloc[PathState_C](1)
    path_arr[0] = _make_path()
    path_arr[0].bounce = Int32(1)
    var io0 = SMSReservoirIO(read=buf_a, write=buf_b,
        gbuf_normal=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_depth=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_material_id=UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
        gbuf_world_pos=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        frame_w=Int32(0), frame_h=Int32(0))
    var pcg0 = PCG32(UInt64(1), UInt64(1))
    _shade_diffuse_nee[False, False](
        path_arr, ctx, normal, hit_point, alb, Vec3f(0.0, 0.0, -1.0),
        Float32(0.5), Float32(0.5), Float32(0.5), Float32(0.5), Float32(0.5),
        pcg0, null_guide(), reservoir_io_null(), 0, io0)
    # buf_b was never written to (sms_io_this_bounce forced null at bounce 1) --
    # still holds exactly its seeded value, unchanged.
    assert_true(_close(buf_b[0].state.m, seed_m))
    # The refracted contribution should still appear via plain MNEE.
    assert_true(path_arr[0].estimate.r > Float32(0.0))

    meshes[0].points.free(); meshes[0].vertexIndices.free()
    meshes[1].points.free(); meshes[1].vertexIndices.free()
    meshes.free()
    materials.free(); area_lights.free(); primIds.free(); bvh.free(); cdf.free()
    path_arr.free(); buf_a.free(); buf_b.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
