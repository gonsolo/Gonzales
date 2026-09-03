# Unit tests for shading.mojo's _gi_generate_recon_candidate (Phase 4.1's
# remaining generation half, docs/A2_restir_migration_plan.md) -- the
# x2-local half of ReSTIR GI candidate generation, scoped to x2 (the
# reconnection vertex) also being diffuse. Mirrors test_shading_helpers.mojo's
# harness style (_make_ctx/_make_triangle_mesh/_make_one_leaf_bvh) since this
# function needs a real BVH for its own shadow-ray visibility test, unlike
# restir_gi.mojo's ctx-free math (already covered in test_restir_gi.mojo).

from std.math import abs
from std.memory import alloc
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import (
    RGB, Point3f, Vec3f, Ray_C, PrimId_C, TriangleMesh_C, Material_C, Curve_C,
    GpuTexture_C, ShadowTask_C, LightSampler_C, Instance_C, AreaLight_C,
    DistantLight_C, PointLight_C, InfiniteLight_C, Sphere_C, MeasuredBRDF_C,
    PathState_C,
)
from gonzales.spectrum import null_spectral_handle, SampledWavelengths
from gonzales.bvh import BVH2Node
from gonzales.guide import null_guide
from gonzales.shading import ShadeContext, LightContext, GIPendingX1, gi_pending_x1_init, _gi_generate_recon_candidate, _shade_diffuse_nee
from gonzales.restir_gi import gi_reservoir_io_null
from gonzales.restir_di import reservoir_io_null
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

def _make_ctx_with_light(
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    area_lights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    area_light_count: Int,
    light_sampler_cdf: UnsafePointer[Float32, MutExternalOrigin],
    use_restir: Bool = False,
    gi_pending: UnsafePointer[GIPendingX1, MutExternalOrigin] = UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling(),
) -> ShadeContext:
    """A ShadeContext with a real BVH + area-light setup (everything
    _gi_generate_recon_candidate touches) and dangling sentinels for
    everything it doesn't (materials/textures/spectral/measured -- this
    function takes alb directly, never dereferences ctx.materials).
    use_restir/gi_pending default to off/dangling, matching every real
    non-restir call site; the end-to-end wiring tests below pass real
    values to exercise the full generate/combine/resolve chain."""
    return ShadeContext(
        0, bvh2Nodes, primIds, meshes,
        UnsafePointer[Curve_C, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[Material_C, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[GpuTexture_C, MutExternalOrigin].unsafe_dangling(), 0,
        UnsafePointer[ShadowTask_C, MutExternalOrigin].unsafe_dangling(),
        Float32(0.0),
        UnsafePointer[UInt32, MutExternalOrigin].unsafe_dangling(),
        null_guide(),
        use_restir,
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
        gi_pending,
        gi_reservoir_io_null(),
    )

# ── Shared fixture: a small light triangle near (0,10,0) facing straight
# down (winding chosen so cross(p1-p0,p2-p0) points -Y), an x2 shading point
# at the origin facing +Y, and a one-leaf BVH slot the tests point at either
# a real occluder or a far-away decoy. ──────────────────────────────────────

def _make_light_mesh() -> TriangleMesh_C:
    # Tiny triangle (~0.001 extent) centered near (0,10,0) -- small enough
    # that barycentric sampling always lands within ~0.001 of (0,10,0),
    # making the sampled shadow-ray direction effectively deterministic
    # (straight up) without needing to predict the PCG draw exactly.
    return _make_triangle_mesh(
        Vec3f(-0.001, 10.0, -0.001),
        Vec3f(0.001, 10.0, -0.001),
        Vec3f(-0.001, 10.0, 0.001))

def test_gi_generate_no_area_lights_returns_invalid() raises:
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(Vec3f(1000.0, 1000.0, 1000.0), Vec3f(1001.0, 1001.0, 1001.0))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_light_mesh()
    var ctx = _make_ctx_with_light(bvh, primIds, meshes,
        UnsafePointer[AreaLight_C, MutExternalOrigin].unsafe_dangling(), 0, cdf)

    var pcg = PCG32(UInt64(1), UInt64(1))
    var res = _gi_generate_recon_candidate(ctx, Vec3f(0.0, 0.0, 0.0),
        Vec3f(0.0, 1.0, 0.0), RGB(Float32(0.8)), pcg)
    assert_true(res.valid == Int8(0))

    meshes[0].points.free(); meshes[0].vertexIndices.free(); meshes.free()
    primIds.free(); bvh.free(); cdf.free()

def test_gi_generate_surface_facing_away_from_light_returns_invalid() raises:
    """X2's own normal faces -Y (away from the light at y=10) -- cos_s <= 0
    must be rejected before any shadow ray is even considered."""
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(Vec3f(1000.0, 1000.0, 1000.0), Vec3f(1001.0, 1001.0, 1001.0))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_light_mesh()
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx_with_light(bvh, primIds, meshes, area_lights, 1, cdf)

    var pcg = PCG32(UInt64(1), UInt64(1))
    var res = _gi_generate_recon_candidate(ctx, Vec3f(0.0, 0.0, 0.0),
        Vec3f(0.0, -1.0, 0.0), RGB(Float32(0.8)), pcg)
    assert_true(res.valid == Int8(0))

    meshes[0].points.free(); meshes[0].vertexIndices.free(); meshes.free()
    area_lights.free(); primIds.free(); bvh.free(); cdf.free()

def test_gi_generate_unoccluded_light_gives_valid_positive_lo() raises:
    """No occluder (the one BVH leaf is a decoy AABB far outside the shadow
    ray's ~10-unit length, so it can never be hit) -- a real, unshadowed
    light sample must produce a positive-luminance Lo at x2's own hit point/
    normal, with recon_point/recon_normal echoing x2's own (matching
    gi_target_pdf's expectation that x2's data, not x1's, lives here)."""
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(Vec3f(1000.0, 1000.0, 1000.0), Vec3f(1001.0, 1001.0, 1001.0))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_light_mesh()
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx_with_light(bvh, primIds, meshes, area_lights, 1, cdf)

    var hit_point = Vec3f(0.0, 0.0, 0.0)
    var normal = Vec3f(0.0, 1.0, 0.0)
    var pcg = PCG32(UInt64(1), UInt64(1))
    var res = _gi_generate_recon_candidate(ctx, hit_point, normal, RGB(Float32(0.8)), pcg)
    assert_true(res.valid == Int8(1))
    assert_true(res.recon_is_delta == Int8(0))
    assert_true(_close(res.recon_point[0], hit_point[0]))
    assert_true(_close(res.recon_point[1], hit_point[1]))
    assert_true(_close(res.recon_point[2], hit_point[2]))
    assert_true(_close(res.recon_normal[1], normal[1]))
    assert_true(res.lo.r > Float32(0.0) and res.lo.g > Float32(0.0) and res.lo.b > Float32(0.0))
    # state is left UNSTREAMED -- the caller (shading.mojo's own wiring)
    # is responsible for the reservoir_update step, not this function.
    assert_true(_close(res.state.m, Float32(0.0)))

    meshes[0].points.free(); meshes[0].vertexIndices.free(); meshes.free()
    area_lights.free(); primIds.free(); bvh.free(); cdf.free()

def test_gi_generate_occluded_light_gives_valid_zero_lo() raises:
    """A real occluder triangle crosses the straight-up shadow ray at
    y=5 -- must still report valid=1 (a real sample was drawn and resolved),
    but lo=0 exactly, matching ordinary NEE's own treatment of occlusion
    (a real zero contribution, not an invalid draw)."""
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(Vec3f(-1.0, 4.9, -1.0), Vec3f(2.0, 5.1, 2.0))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(1), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](2)
    meshes[0] = _make_light_mesh()
    # Occluder: right triangle with the right-angle corner at (-1,5,-1),
    # legs to (2,5,-1) and (-1,5,2) -- contains (0,5,0) (barycentric
    # s=t=1/3, s+t=2/3<=1), which is exactly where the ~straight-up shadow
    # ray crosses y=5.
    meshes[1] = _make_triangle_mesh(
        Vec3f(-1.0, 5.0, -1.0),
        Vec3f(2.0, 5.0, -1.0),
        Vec3f(-1.0, 5.0, 2.0))
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx_with_light(bvh, primIds, meshes, area_lights, 1, cdf)

    var pcg = PCG32(UInt64(1), UInt64(1))
    var res = _gi_generate_recon_candidate(ctx, Vec3f(0.0, 0.0, 0.0),
        Vec3f(0.0, 1.0, 0.0), RGB(Float32(0.8)), pcg)
    assert_true(res.valid == Int8(1))
    assert_true(_close(res.lo.r, Float32(0.0)))
    assert_true(_close(res.lo.g, Float32(0.0)))
    assert_true(_close(res.lo.b, Float32(0.0)))

    meshes[0].points.free(); meshes[0].vertexIndices.free()
    meshes[1].points.free(); meshes[1].vertexIndices.free()
    meshes.free()
    area_lights.free(); primIds.free(); bvh.free(); cdf.free()

# ── End-to-end wiring: _shade_diffuse_nee's bounce-0-mark / bounce-1-
# generate+combine+resolve chain (Phase 4.1, full path) ─────────────────────
# Drives _shade_diffuse_nee twice against one shared PathState_C (x1 at
# bounce 0, x2 at bounce 1), mirroring how the real per-bounce loop
# (rendering.mojo) reuses one path slot across bounces. x1's own context
# (ctx0) has ZERO area lights, so its bounce-0 di_temporal_step branch
# contributes nothing -- isolates path_ptr[].estimate to whatever bounce 1
# adds. Geometry is chosen so gi_target_pdf's own cos terms are exact,
# round numbers: x1=(0,3,-4) with normal=(0,-0.6,0.8) pointing exactly at
# x2=(0,0,0) (cos_x1=1, dist=5), x2's normal=(0,1,0) giving cos_x2=0.6.

def _make_path(org: Vec3f, dir: Vec3f) -> PathState_C:
    return PathState_C(
        Ray_C(Point3f(org[0], org[1], org[2]), Vec3f(dir[0], dir[1], dir[2])),
        RGB(Float32(1.0)), RGB(Float32(0.0)), RGB(Float32(0.0)),
        Int32(0), UInt64(1), UInt64(1), Int8(1), Int8(0), Int8(0), Int8(0),
        Float32(0.0), Int32(-1), Int32(0), UInt64(0),
        SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )

def _run_two_bounce(gi_active: Bool) -> RGB:
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(Vec3f(1000.0, 1000.0, 1000.0), Vec3f(1001.0, 1001.0, 1001.0))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_light_mesh()
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))

    var gi_pending_buf = alloc[GIPendingX1](1)
    gi_pending_buf[0] = gi_pending_x1_init()
    var real_gi_pending = gi_pending_buf if gi_active else UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling()

    # ctx0 (bounce 0, x1): zero area lights -- di_temporal_step's own RIS
    # loop draws nothing and di_resolve returns immediately (res.light_idx
    # stays < 0), so it never touches path_ptr[].estimate.
    var no_lights_cdf = alloc[Float32](1)
    no_lights_cdf[0] = Float32(0.0)
    var ctx0 = _make_ctx_with_light(
        UnsafePointer[BVH2Node, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[PrimId_C, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[TriangleMesh_C, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[AreaLight_C, MutExternalOrigin].unsafe_dangling(), 0, no_lights_cdf,
        use_restir=True, gi_pending=real_gi_pending)
    # ctx1 (bounce 1, x2): the real reconnection light, real BVH for both
    # GI's own shadow ray and _nee_area_lights' ordinary NEE (which also
    # fires at bounce 1 regardless of GI -- ctx.use_restir only replaces
    # bounce 0's area-light NEE with the DI reservoir).
    var ctx1 = _make_ctx_with_light(bvh, primIds, meshes, area_lights, 1, cdf,
        use_restir=True, gi_pending=real_gi_pending)

    var path_arr = alloc[PathState_C](1)
    path_arr[0] = _make_path(Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, -1.0))

    var x1_hit = Vec3f(0.0, 3.0, -4.0)
    var x1_normal = Vec3f(0.0, -0.6, 0.8)
    # Deliberately NON-uniform across channels: a uniform albedo here would
    # hide the real "used path_ptr[].throughput at bounce 1 instead of a
    # bounce-0 snapshot" bug this test now specifically guards against --
    # multiplying by a uniform scalar doesn't change the delta's channel
    # ratio, so the earlier version of this test (x1_alb=RGB(0.5)) could
    # not have caught it. See GIPendingX1's own docstring for the full story.
    var x1_alb = RGB(Float32(0.5), Float32(0.3), Float32(0.7))
    var pcg0 = PCG32(UInt64(1), UInt64(1))
    _shade_diffuse_nee[False, False](path_arr, ctx0, x1_normal, x1_hit, x1_alb,
        Vec3f(0.0, 0.0, 1.0),
        Float32(0.5), Float32(0.5), Float32(0.5), Float32(0.5), Float32(0.5),
        pcg0, null_guide(), reservoir_io_null(), -1)

    # Mimics shade_diffuse's own continuation-sampling epilogue
    # (path_ptr[].throughput *= alb * cos_theta/(pi*pdf_mix)), which for a
    # plain cosine-weighted sample with no guide reduces to *= alb exactly
    # (cos_theta/(pi*pdf_mix) == 1 when pdf_mix == cos_theta/pi). Real
    # renders ALWAYS run this between bounce 0 and bounce 1 -- omitting it
    # here (as the original version of this test did) leaves
    # path_ptr[].throughput == T_0 at bounce 1 by coincidence, which is
    # exactly the state gi_resolve's bug silently relied on.
    path_arr[0].throughput = path_arr[0].throughput * x1_alb

    path_arr[0].bounce = Int32(1)
    var x2_hit = Vec3f(0.0, 0.0, 0.0)
    var x2_normal = Vec3f(0.0, 1.0, 0.0)
    var x2_alb = RGB(Float32(0.8))
    var pcg1 = PCG32(UInt64(2), UInt64(1))
    _shade_diffuse_nee[False, False](path_arr, ctx1, x2_normal, x2_hit, x2_alb,
        Vec3f(0.0, 0.0, 1.0),
        Float32(0.5), Float32(0.5), Float32(0.5), Float32(0.5), Float32(0.5),
        pcg1, null_guide(), reservoir_io_null(), -1)

    var result = path_arr[0].estimate

    meshes[0].points.free(); meshes[0].vertexIndices.free(); meshes.free()
    area_lights.free(); primIds.free(); bvh.free(); cdf.free(); no_lights_cdf.free()
    gi_pending_buf.free(); path_arr.free()
    return result

def test_shade_diffuse_nee_gi_wiring_adds_positive_reconnection_contribution() raises:
    """A/B: identical two-bounce setup, differing only in whether
    ctx.gi_pending is a real (active) buffer or the production default
    (dangling). Bounce 0's own pcg draws and bounce 1's ordinary
    _nee_area_lights draws are IDENTICAL between both runs (nothing before
    the GI block depends on gi_pending), so the delta isolates exactly
    GI's own generate+combine+resolve contribution. Must be strictly
    positive in every channel -- the reconnection light is real, unoccluded,
    and both cos terms are positive by construction (see module docstring
    for the exact geometry)."""
    var enabled = _run_two_bounce(gi_active=True)
    var disabled = _run_two_bounce(gi_active=False)
    assert_true(enabled.r > disabled.r + Float32(1e-12))
    assert_true(enabled.g > disabled.g + Float32(1e-12))
    assert_true(enabled.b > disabled.b + Float32(1e-12))

def test_shade_diffuse_nee_gi_wiring_delta_channel_ratio_matches_expected_formula() raises:
    """The GI-attributable delta's chromatic variation comes from exactly
    two per-channel sources: the reconnection light's own emission
    (200,80,20) AND x1's own albedo (0.5,0.3,0.7 -- gi_resolve's
    bxdf_eval_diffuse(alb) term, x1's BSDF response toward the
    reconnection direction, deliberately non-uniform here so a bug using
    the WRONG throughput/albedo at the wrong bounce shows up as a ratio
    mismatch rather than being hidden by a uniform multiplier -- this is
    exactly how the real throughput bug (see GIPendingX1's docstring) was
    confirmed fixed: the naive "ratio == light emission ratio" expectation
    from an earlier, uniform-x1_alb version of this test could NOT have
    caught it. throughput/cosines/state.w are channel-independent scalars
    and drop out of the ratio; x2's own albedo (0.8, uniform) does too."""
    var enabled = _run_two_bounce(gi_active=True)
    var disabled = _run_two_bounce(gi_active=False)
    var delta = enabled - disabled
    assert_true(delta.r > Float32(0.0) and delta.g > Float32(0.0) and delta.b > Float32(0.0))
    var rg = delta.r / delta.g
    var rb = delta.r / delta.b
    var expected_rg = (Float32(0.5) * Float32(200.0)) / (Float32(0.3) * Float32(80.0))
    var expected_rb = (Float32(0.5) * Float32(200.0)) / (Float32(0.7) * Float32(20.0))
    assert_true(_close(rg, expected_rg))
    assert_true(_close(rb, expected_rb))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
