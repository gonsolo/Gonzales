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
    RGB, Point3f, Vec3f, PrimId_C, TriangleMesh_C, Material_C, Curve_C,
    GpuTexture_C, ShadowTask_C, LightSampler_C, Instance_C, AreaLight_C,
    DistantLight_C, PointLight_C, InfiniteLight_C, Sphere_C, MeasuredBRDF_C,
)
from gonzales.spectrum import null_spectral_handle
from gonzales.bvh import BVH2Node
from gonzales.guide import null_guide
from gonzales.shading import ShadeContext, LightContext, GIPendingX1, _gi_generate_recon_candidate
from gonzales.restir_gi import GIReservoir
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

def _make_ctx_with_light(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    area_lights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    area_light_count: Int,
    light_sampler_cdf: UnsafePointer[Float32, MutAnyOrigin],
) -> ShadeContext:
    """A ShadeContext with a real BVH + area-light setup (everything
    _gi_generate_recon_candidate touches) and dangling sentinels for
    everything it doesn't (materials/textures/spectral/measured/GI-tracking
    buffers -- this function takes alb directly, never dereferences
    ctx.materials)."""
    return ShadeContext(
        0, bvh2Nodes, primIds, meshes,
        UnsafePointer[Curve_C, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Material_C, MutAnyOrigin].unsafe_dangling(),
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
        UnsafePointer[GIReservoir, MutAnyOrigin].unsafe_dangling(),
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
        SIMD[DType.float32, 3](-0.001, 10.0, -0.001),
        SIMD[DType.float32, 3](0.001, 10.0, -0.001),
        SIMD[DType.float32, 3](-0.001, 10.0, 0.001))

def test_gi_generate_no_area_lights_returns_invalid() raises:
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(SIMD[DType.float32, 3](1000.0, 1000.0, 1000.0), SIMD[DType.float32, 3](1001.0, 1001.0, 1001.0))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_light_mesh()
    var ctx = _make_ctx_with_light(bvh, primIds, meshes,
        UnsafePointer[AreaLight_C, MutAnyOrigin].unsafe_dangling(), 0, cdf)

    var pcg = PCG32(UInt64(1), UInt64(1))
    var res = _gi_generate_recon_candidate(ctx, SIMD[DType.float32, 3](0.0, 0.0, 0.0),
        SIMD[DType.float32, 3](0.0, 1.0, 0.0), RGB(Float32(0.8)), pcg)
    assert_true(res.valid == Int8(0))

    meshes[0].points.free(); meshes[0].vertexIndices.free(); meshes.free()
    primIds.free(); bvh.free(); cdf.free()

def test_gi_generate_surface_facing_away_from_light_returns_invalid() raises:
    """X2's own normal faces -Y (away from the light at y=10) -- cos_s <= 0
    must be rejected before any shadow ray is even considered."""
    var cdf = alloc[Float32](2)
    cdf[0] = Float32(0.0); cdf[1] = Float32(1.0)
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(SIMD[DType.float32, 3](1000.0, 1000.0, 1000.0), SIMD[DType.float32, 3](1001.0, 1001.0, 1001.0))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_light_mesh()
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx_with_light(bvh, primIds, meshes, area_lights, 1, cdf)

    var pcg = PCG32(UInt64(1), UInt64(1))
    var res = _gi_generate_recon_candidate(ctx, SIMD[DType.float32, 3](0.0, 0.0, 0.0),
        SIMD[DType.float32, 3](0.0, -1.0, 0.0), RGB(Float32(0.8)), pcg)
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
    bvh[0] = _make_one_leaf_bvh(SIMD[DType.float32, 3](1000.0, 1000.0, 1000.0), SIMD[DType.float32, 3](1001.0, 1001.0, 1001.0))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = _make_light_mesh()
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx_with_light(bvh, primIds, meshes, area_lights, 1, cdf)

    var hit_point = SIMD[DType.float32, 3](0.0, 0.0, 0.0)
    var normal = SIMD[DType.float32, 3](0.0, 1.0, 0.0)
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
    bvh[0] = _make_one_leaf_bvh(SIMD[DType.float32, 3](-1.0, 4.9, -1.0), SIMD[DType.float32, 3](2.0, 5.1, 2.0))
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(1), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var meshes = alloc[TriangleMesh_C](2)
    meshes[0] = _make_light_mesh()
    # Occluder: right triangle with the right-angle corner at (-1,5,-1),
    # legs to (2,5,-1) and (-1,5,2) -- contains (0,5,0) (barycentric
    # s=t=1/3, s+t=2/3<=1), which is exactly where the ~straight-up shadow
    # ray crosses y=5.
    meshes[1] = _make_triangle_mesh(
        SIMD[DType.float32, 3](-1.0, 5.0, -1.0),
        SIMD[DType.float32, 3](2.0, 5.0, -1.0),
        SIMD[DType.float32, 3](-1.0, 5.0, 2.0))
    var area_lights = alloc[AreaLight_C](1)
    area_lights[0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(200.0), Float32(80.0), Float32(20.0)), Float32(0.000002), Int8(0), Int8(0), Int8(0), Int8(0))
    var ctx = _make_ctx_with_light(bvh, primIds, meshes, area_lights, 1, cdf)

    var pcg = PCG32(UInt64(1), UInt64(1))
    var res = _gi_generate_recon_candidate(ctx, SIMD[DType.float32, 3](0.0, 0.0, 0.0),
        SIMD[DType.float32, 3](0.0, 1.0, 0.0), RGB(Float32(0.8)), pcg)
    assert_true(res.valid == Int8(1))
    assert_true(_close(res.lo.r, Float32(0.0)))
    assert_true(_close(res.lo.g, Float32(0.0)))
    assert_true(_close(res.lo.b, Float32(0.0)))

    meshes[0].points.free(); meshes[0].vertexIndices.free()
    meshes[1].points.free(); meshes[1].vertexIndices.free()
    meshes.free()
    area_lights.free(); primIds.free(); bvh.free(); cdf.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
