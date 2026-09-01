# Unit tests for shared helper functions extracted from shading.mojo during
# this session's H1/H2/H3-style dedup pass:
#   - _get_tri_verts: decodes an Intersection_C's primId into (mesh,v0,v1,v2,ok)
#   - _apply_normal_map: tangent-space normal-map decode (no-normal-map fast path)
#   - _build_geom_context_full: shared NEE-material GeomContext builder
#   - _shadow_contribute: shared shadow-ray-or-direct-contribution helper
# (enqueue_shadow=False / GPU shadow-queue paths, and the texture-sampling
# branch of _apply_normal_map, need real GPU/file infrastructure and are not
# exercised here — see the module docstring above each test group.)

from std.math import abs, sqrt
from std.memory import alloc
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import (
    RGB, Point3f, Vec3f, Ray_C, PrimId_C, Intersection_C, TriangleMesh_C,
    Material_C, MatKind, Curve_C, GpuTexture_C, ShadowTask_C, LightSampler_C,
    Instance_C, PathState_C, AreaLight_C, DistantLight_C, PointLight_C,
    InfiniteLight_C, Sphere_C, MeasuredBRDF_C, dot, cross,
)
from gonzales.spectrum import SampledWavelengths, null_spectral_handle
from gonzales.bvh import BVH2Node
from gonzales.guide import GuideGrid, null_guide
from gonzales.shading import (
    ShadeContext, LightContext, _get_tri_verts, _apply_normal_map,
    _build_geom_context_full, _shadow_contribute, GIPendingX1,
)
from gonzales.restir_gi import gi_reservoir_io_null

comptime EPS: Float32 = 1e-3

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _simd_close(a: SIMD[DType.float32, 3], b: SIMD[DType.float32, 3]) -> Bool:
    return _close(a[0], b[0]) and _close(a[1], b[1]) and _close(a[2], b[2])

def _simd_len(v: SIMD[DType.float32, 3]) -> Float32:
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

# ── shared fixture builders ──────────────────────────────────────────────────

def _make_triangle_mesh(p0: SIMD[DType.float32, 3], p1: SIMD[DType.float32, 3], p2: SIMD[DType.float32, 3]) -> TriangleMesh_C:
    """Mirrors test_sppm_helpers.mojo's fixture: a single-triangle mesh with
    no UVs/normals (sentinel dangling pointers -> fallback paths)."""
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

def _make_material(albedo: RGB, normal_tex_idx: Int32) -> Material_C:
    return Material_C(Int8(1), Int8(0), Int8(0), Int8(0), albedo, RGB(Float32(0.0)),
        Int32(-1), Float32(0.0), Float32(0.0), normal_tex_idx, Int32(-1), Int32(-1),
        RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(1.0), Float32(1.0), Int32(-1))

def _null_light_context() -> LightContext:
    """No lights of any kind -- fine, since none of the functions under test
    touch ctx.lights at all (NEE light-sampling lives in the shade_* callers,
    not in these lower-level helpers)."""
    return LightContext(
        UnsafePointer[AreaLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[DistantLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[PointLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[InfiniteLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[Sphere_C, MutAnyOrigin].unsafe_dangling(), 0,
        LightSampler_C(UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(), Int32(0), Int32(0)),
    )

def _make_ctx(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    px_scale: Float32,
) -> ShadeContext:
    """A ShadeContext with every field the functions under test don't touch
    left as a dangling sentinel -- valid as long as the exercised code paths
    never dereference it (verified per-function below: e.g. tex_idx=-1 and
    normal_tex_idx=-1 mean the texture/normal-map branches never run)."""
    return ShadeContext(
        0, bvh2Nodes, primIds, meshes,
        UnsafePointer[Curve_C, MutAnyOrigin].unsafe_dangling(),
        materials,
        UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[GpuTexture_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        px_scale,
        UnsafePointer[UInt32, MutAnyOrigin].unsafe_dangling(),
        null_guide(),
        False,
        _null_light_context(),
        UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Instance_C, MutAnyOrigin].unsafe_dangling(),
        null_spectral_handle(),
        UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[GIPendingX1, MutAnyOrigin].unsafe_dangling(),
        gi_reservoir_io_null(),
    )

def _make_path(org: SIMD[DType.float32, 3], dir: SIMD[DType.float32, 3]) -> PathState_C:
    return PathState_C(
        Ray_C(Point3f(org[0], org[1], org[2]), Vec3f(dir[0], dir[1], dir[2])),
        RGB(Float32(1.0)), RGB(Float32(0.0)), RGB(Float32(0.0)),
        Int32(0), UInt64(1), UInt64(1), Int8(1), Int8(0), Int8(0), Int8(0),
        Float32(0.0), Int32(-1), Int32(0), UInt64(0),
        SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    )

# ── _get_tri_verts ────────────────────────────────────────────────────────────
# Decodes an Intersection_C's primId into (mesh, v0, v1, v2, ok). Only the
# vertex-index bookkeeping is under test -- point data is never touched by
# this function, so meshes[].points can be left dangling.

def _make_two_tri_mesh() -> TriangleMesh_C:
    var vidx = alloc[Int64](6)
    vidx[0] = 10; vidx[1] = 11; vidx[2] = 12
    vidx[3] = 20; vidx[4] = 21; vidx[5] = 22
    return TriangleMesh_C(
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Int64, MutAnyOrigin].unsafe_dangling(), vidx,
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    )

def test_get_tri_verts_type0_decodes_second_triangle_correctly() raises:
    """Type==0 encodes id1=mesh_idx, id2=base_vertex_idx directly (already
    tri_idx*3, per pbrt_parser.mojo) -- NOT tri_idx, so this must select the
    *second* triangle's vertex-index triple (20,21,22), not (0,1,2)."""
    var mesh = _make_two_tri_mesh()
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = mesh
    var pid = PrimId_C(Int64(0), Int64(3), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var inter = Intersection_C(pid, Float32(1.0), Float32(0.0), Float32(0.0), Int8(1), Int8(0), Int8(0), Int8(0))

    var (_, v0, v1, v2, ok) = _get_tri_verts(inter, meshes)
    assert_true(ok)
    assert_true(v0 == 20)
    assert_true(v1 == 21)
    assert_true(v2 == 22)

    meshes[0].vertexIndices.free()
    meshes.free()

def test_get_tri_verts_non_triangle_prim_returns_not_ok() raises:
    """PrimId.type==4 is a sphere hit (see shading.mojo's shade dispatch and
    bvh.mojo's leaf-type switch) -- _get_tri_verts must report ok=False so
    callers deactivate the path instead of misreading sphere data as a
    triangle leaf."""
    var mesh = _make_two_tri_mesh()
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = mesh
    var pid = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(4), Int8(0), Int8(0), Int8(0))
    var inter = Intersection_C(pid, Float32(1.0), Float32(0.0), Float32(0.0), Int8(1), Int8(0), Int8(0), Int8(0))

    var (_, _, _, _, ok) = _get_tri_verts(inter, meshes)
    assert_false(ok)

    meshes[0].vertexIndices.free()
    meshes.free()

# ── _apply_normal_map ─────────────────────────────────────────────────────────
# Only the documented "no normal map" fast path (mat.normal_tex_idx < 0) is
# testable without real texture-sampling infrastructure -- that path must
# return geom_normal completely unchanged.

def test_apply_normal_map_returns_geom_normal_unchanged_when_no_normal_map() raises:
    var mat = _make_material(RGB(Float32(0.5)), Int32(-1))
    var mesh = TriangleMesh_C(
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Int64, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Int64, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    )
    var pid = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var inter = Intersection_C(pid, Float32(1.0), Float32(0.25), Float32(0.25), Int8(1), Int8(0), Int8(0), Int8(0))
    var geom_normal = SIMD[DType.float32, 3](0.267261, 0.534522, 0.801784)  # normalize(1,2,3)
    var p0 = SIMD[DType.float32, 3](0.0, 0.0, 0.0)
    var p1 = SIMD[DType.float32, 3](1.0, 0.0, 0.0)
    var p2 = SIMD[DType.float32, 3](0.0, 1.0, 0.0)

    var result = _apply_normal_map[False](mat, 0, 1, 2, mesh, inter, geom_normal, p0, p1, p2,
        UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[GpuTexture_C, MutAnyOrigin].unsafe_dangling(), 0)
    assert_true(_simd_close(result, geom_normal))

# ── _build_geom_context_full ──────────────────────────────────────────────────
# Full NEE-material GeomContext builder. With tex_idx=-1, normal_tex_idx=-1,
# and px_scale=0 the texture/normal-map/pixel-footprint branches are all
# skipped, leaving a pure closed-form geometry computation to check against.

def test_build_geom_context_full_matches_closed_form_for_axis_aligned_hit() raises:
    """A unit right-triangle in the z=0 plane, hit by a ray travelling in -z
    from (0,0,5): geo_normal must be (0,0,1) (faceforward against the ray),
    hit_point = ray_org + ray_dir*tHit + normal*1e-4, wo = -ray_dir, and
    (since n=(0,0,1) is the fixed point of Duff's from_z construction) the
    tangent/bitangent come out to exactly (1,0,0)/(0,1,0)."""
    var p0 = SIMD[DType.float32, 3](0.0, 0.0, 0.0)
    var p1 = SIMD[DType.float32, 3](1.0, 0.0, 0.0)
    var p2 = SIMD[DType.float32, 3](0.0, 1.0, 0.0)
    var mesh = _make_triangle_mesh(p0, p1, p2)
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = mesh

    var albedo = RGB(Float32(0.2), Float32(0.4), Float32(0.6))
    var mat = _make_material(albedo, Int32(-1))
    var materials = alloc[Material_C](1)
    materials[0] = mat

    var ctx = _make_ctx(
        UnsafePointer[BVH2Node, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[PrimId_C, MutAnyOrigin].unsafe_dangling(),
        meshes, materials, Float32(0.0))

    var org = SIMD[DType.float32, 3](0.0, 0.0, 5.0)
    var dir = SIMD[DType.float32, 3](0.0, 0.0, -1.0)
    var path = _make_path(org, dir)
    var path_arr = alloc[PathState_C](1)
    path_arr[0] = path
    var t_hit = Float32(5.0)

    var pid = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var inter = Intersection_C(pid, t_hit, Float32(0.25), Float32(0.25), Int8(1), Int8(0), Int8(0), Int8(0))

    var (gc, ok) = _build_geom_context_full[False](path_arr, inter, mat, ctx)
    assert_true(ok)
    assert_true(_simd_close(gc.normal, SIMD[DType.float32, 3](0.0, 0.0, 1.0)))
    assert_true(_simd_close(gc.geo_normal, SIMD[DType.float32, 3](0.0, 0.0, 1.0)))
    assert_true(_simd_close(gc.wo, SIMD[DType.float32, 3](0.0, 0.0, 1.0)))
    var expected_hit = org + dir * t_hit + SIMD[DType.float32, 3](0.0, 0.0, 1.0) * Float32(0.0001)
    assert_true(_simd_close(gc.hit_point, expected_hit))
    assert_true(_close(gc.alb.r, albedo.r))
    assert_true(_close(gc.alb.g, albedo.g))
    assert_true(_close(gc.alb.b, albedo.b))
    # Duff et al.'s from_z(0,0,1) fixed point: tangent=(1,0,0), bitangent=(0,1,0).
    assert_true(_simd_close(gc.tangent, SIMD[DType.float32, 3](1.0, 0.0, 0.0)))
    assert_true(_simd_close(gc.bitangent, SIMD[DType.float32, 3](0.0, 1.0, 0.0)))

    meshes[0].points.free()
    meshes[0].vertexIndices.free()
    meshes.free()
    materials.free()
    path_arr.free()

def test_build_geom_context_full_sphere_prim_returns_exact_analytic_normal() raises:
    """Sphere hit (primId.type==4) must return ok=True with the exact
    analytic outward normal (hit - center)/radius, face-forwarded against
    the ray, and the material's flat albedo (no UV space exists for a
    sphere). This used to unconditionally return ok=False, causing every
    non-emissive material on an analytic sphere to render solid black --
    a real, pre-existing, non-Mitsuba-specific gonzales bug (see
    project_mitsuba_parser memory). Sphere at (0,0,-1) radius 1, ray from
    (0,0,5) along -z hits the front pole (0,0,0) with outward normal
    (0,0,1), pointed straight back at the camera."""
    var mesh = _make_triangle_mesh(
        SIMD[DType.float32, 3](0.0, 0.0, 0.0),
        SIMD[DType.float32, 3](1.0, 0.0, 0.0),
        SIMD[DType.float32, 3](0.0, 1.0, 0.0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = mesh
    var albedo = RGB(Float32(0.5), Float32(0.3), Float32(0.1))
    var mat = _make_material(albedo, Int32(-1))
    var materials = alloc[Material_C](1)
    materials[0] = mat

    var spheres = alloc[Sphere_C](1)
    spheres[0] = Sphere_C(Point3f(0.0, 0.0, -1.0), Float32(1.0), Int32(-1),
        Int8(0), Int8(0), Int8(0), Int8(0), RGB(Float32(0.0)))
    var lights = LightContext(
        UnsafePointer[AreaLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[DistantLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[PointLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[InfiniteLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        spheres, 1,
        LightSampler_C(UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(), Int32(0), Int32(0)),
    )
    var ctx = ShadeContext(
        0, UnsafePointer[BVH2Node, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[PrimId_C, MutAnyOrigin].unsafe_dangling(), meshes,
        UnsafePointer[Curve_C, MutAnyOrigin].unsafe_dangling(),
        materials,
        UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[GpuTexture_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        Float32(0.0),
        UnsafePointer[UInt32, MutAnyOrigin].unsafe_dangling(),
        null_guide(),
        False,
        lights^,
        UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Instance_C, MutAnyOrigin].unsafe_dangling(),
        null_spectral_handle(),
        UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[GIPendingX1, MutAnyOrigin].unsafe_dangling(),
        gi_reservoir_io_null(),
    )

    var org = SIMD[DType.float32, 3](0.0, 0.0, 5.0)
    var dir = SIMD[DType.float32, 3](0.0, 0.0, -1.0)
    var path = _make_path(org, dir)
    var path_arr = alloc[PathState_C](1)
    path_arr[0] = path

    var pid = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(4), Int8(0), Int8(0), Int8(0))
    var inter = Intersection_C(pid, Float32(5.0), Float32(0.0), Float32(0.0), Int8(1), Int8(0), Int8(0), Int8(0))
    var (gc, ok) = _build_geom_context_full[False](path_arr, inter, mat, ctx)
    assert_true(ok)
    var expected_n = SIMD[DType.float32, 3](0.0, 0.0, 1.0)
    assert_true(_simd_close(gc.normal, expected_n))
    assert_true(_simd_close(gc.geo_normal, expected_n))
    var expected_hit = org + dir * Float32(5.0) + expected_n * Float32(0.0001)
    assert_true(_simd_close(gc.hit_point, expected_hit))
    assert_true(_close(gc.alb.r, albedo.r))
    assert_true(_close(gc.alb.g, albedo.g))
    assert_true(_close(gc.alb.b, albedo.b))

    meshes[0].points.free()
    meshes[0].vertexIndices.free()
    meshes.free()
    materials.free()
    spheres.free()
    path_arr.free()

# ── _shadow_contribute[enqueue_shadow=False] ─────────────────────────────────
# The direct-evaluation branch: fires a real shadow ray through a one-leaf
# BVH2 and either adds `contrib` to path_ptr[].estimate (unoccluded) or
# leaves it untouched (occluded). The enqueue_shadow=True branch just writes
# a ShadowTask_C record and needs a real GPU queue, so it's not covered here.

def _make_one_leaf_bvh(
    tri_min: SIMD[DType.float32, 3], tri_max: SIMD[DType.float32, 3],
) -> BVH2Node:
    return BVH2Node(Point3f(tri_min[0], tri_min[1], tri_min[2]),
        Point3f(tri_max[0], tri_max[1], tri_max[2]), Int32(0), Int32(1))

def test_shadow_contribute_direct_adds_contribution_when_unoccluded() raises:
    """An occluder triangle sits at z=[4.9,5.1]; a shadow ray fired in -z
    (away from it) must find no hit, so the full `contrib` is added to
    path_ptr[].estimate exactly (estimate started at zero)."""
    var mesh = _make_triangle_mesh(
        SIMD[DType.float32, 3](-1.0, -1.0, 5.0),
        SIMD[DType.float32, 3](1.0, -1.0, 5.0),
        SIMD[DType.float32, 3](0.0, 1.0, 5.0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = mesh
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(SIMD[DType.float32, 3](-1.0, -1.0, 4.9), SIMD[DType.float32, 3](1.0, 1.0, 5.1))

    var materials = alloc[Material_C](1)
    materials[0] = _make_material(RGB(Float32(0.5)), Int32(-1))
    var ctx = _make_ctx(bvh, primIds, meshes, materials, Float32(0.0))
    var path = _make_path(SIMD[DType.float32, 3](0.0, 0.0, 0.0), SIMD[DType.float32, 3](0.0, 0.0, 1.0))
    var path_arr = alloc[PathState_C](1)
    path_arr[0] = path

    var contrib = RGB(Float32(1.0), Float32(2.0), Float32(3.0))
    _shadow_contribute[False](path_arr, ctx, SIMD[DType.float32, 3](0.0, 0.0, 0.0),
        SIMD[DType.float32, 3](0.0, 0.0, -1.0), Float32(10.0), contrib)
    assert_true(_close(path_arr[0].estimate.r, contrib.r))
    assert_true(_close(path_arr[0].estimate.g, contrib.g))
    assert_true(_close(path_arr[0].estimate.b, contrib.b))

    meshes[0].points.free()
    meshes[0].vertexIndices.free()
    meshes.free()
    primIds.free()
    bvh.free()
    materials.free()
    path_arr.free()

def test_shadow_contribute_direct_skips_when_occluded() raises:
    """Same scene, but the shadow ray is fired straight at the occluder: the
    any-hit test must find it, so `estimate` stays exactly zero."""
    var mesh = _make_triangle_mesh(
        SIMD[DType.float32, 3](-1.0, -1.0, 5.0),
        SIMD[DType.float32, 3](1.0, -1.0, 5.0),
        SIMD[DType.float32, 3](0.0, 1.0, 5.0))
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = mesh
    var primIds = alloc[PrimId_C](1)
    primIds[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var bvh = alloc[BVH2Node](1)
    bvh[0] = _make_one_leaf_bvh(SIMD[DType.float32, 3](-1.0, -1.0, 4.9), SIMD[DType.float32, 3](1.0, 1.0, 5.1))

    var materials = alloc[Material_C](1)
    materials[0] = _make_material(RGB(Float32(0.5)), Int32(-1))
    var ctx = _make_ctx(bvh, primIds, meshes, materials, Float32(0.0))
    var path = _make_path(SIMD[DType.float32, 3](0.0, 0.0, 0.0), SIMD[DType.float32, 3](0.0, 0.0, -1.0))
    var path_arr = alloc[PathState_C](1)
    path_arr[0] = path

    var contrib = RGB(Float32(1.0), Float32(2.0), Float32(3.0))
    _shadow_contribute[False](path_arr, ctx, SIMD[DType.float32, 3](0.0, 0.0, 0.0),
        SIMD[DType.float32, 3](0.0, 0.0, 1.0), Float32(10.0), contrib)
    assert_true(_close(path_arr[0].estimate.r, Float32(0.0)))
    assert_true(_close(path_arr[0].estimate.g, Float32(0.0)))
    assert_true(_close(path_arr[0].estimate.b, Float32(0.0)))

    meshes[0].points.free()
    meshes[0].vertexIndices.free()
    meshes.free()
    primIds.free()
    bvh.free()
    materials.free()
    path_arr.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
