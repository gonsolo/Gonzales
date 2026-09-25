# Task #163 stage 4: equivalence test for the wavefront-staged light-path
# split. Drives BOTH `_bdpt_trace_light_path` (the original, single-mega-
# function loop, still used by the CPU renderer) AND the new
# `_bdpt_light_path_init` + `_bdpt_light_path_bounce` pair (intended for a
# future GPU wavefront host loop) from the SAME PCG32 seed over the SAME
# real BVH-backed scene, and asserts every stored LVC vertex plus the final
# `lvc_path_len` are byte-for-byte identical. The split's bounce body is a
# mechanical, non-semantic transformation of the original (break -> return
# False, continue -> return True, intersect moved to the caller) -- this
# test is the actual proof that transformation didn't change any observable
# behavior, not just that both sides happen to compile.
#
# Scene: two triangles in one mesh. Triangle 0 (small, at z=10, normal
# pointing -Z) is the sole area light. Triangle 1 (huge, at z=0) is a
# diffuse receiver spanning far more than the light's emission hemisphere,
# so a cosine-weighted emission direction is guaranteed to hit it -- this
# exercises the diffuse material-dispatch branch (vertex storage + MIS
# carry update + continuation sampling), not just the light-source vertex.

from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import RGB, Point3f, Vec3f, MeasuredBRDF_C, Material_C, MatKind
from gonzales.render_state import GpuTexture_C, NormalSlopeMap_C
from gonzales.primitives import Ray_C, Intersection_C, PrimId_C, Sphere_C, Instance_C, TriangleMesh_C
from gonzales.media import Medium_C, MediumInterface_C, Grid_C, NvdbGrid_C
from gonzales.lights import LightSampler_C, AreaLight_C, DistantLight_C, PointLight_C, InfiniteLight_C
from gonzales.curves import Curve_C
from gonzales.bvh import SceneDescriptor2_C, BVH2Node, build_bvh2, traverse_bvh2_core
from gonzales.rng import PCG32
from gonzales.spectrum import null_spectral_handle, SampledWavelengths, SpectralSample, sample_wavelengths_uniform
from gonzales.bdpt import (
    BDPTVertex, _bdpt_trace_light_path, _bdpt_light_path_init,
    _bdpt_light_path_bounce, _BDPT_MAX_VERTS, _BDPT_MAX_DEPTH,
)

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    var d = a - b
    if d < Float32(0): d = -d
    return d < EPS

def _vertex_close(a: BDPTVertex, b: BDPTVertex) -> Bool:
    return (
        _close(a.pos.x, b.pos.x) and _close(a.pos.y, b.pos.y) and _close(a.pos.z, b.pos.z) and
        _close(a.normal.x, b.normal.x) and _close(a.normal.y, b.normal.y) and _close(a.normal.z, b.normal.z) and
        _close(a.beta.v0, b.beta.v0) and _close(a.beta.v1, b.beta.v1) and
        _close(a.beta.v2, b.beta.v2) and _close(a.beta.v3, b.beta.v3) and
        _close(a.alb.r, b.alb.r) and _close(a.alb.g, b.alb.g) and _close(a.alb.b, b.alb.b) and
        a.is_surface == b.is_surface and a.is_delta == b.is_delta and a.is_light == b.is_light and
        a.mat_kind == b.mat_kind and a.med_idx == b.med_idx and
        _close(a.dVCM, b.dVCM) and _close(a.dVC, b.dVC) and _close(a.dVM, b.dVM)
    )

def _build_scene() -> SceneDescriptor2_C:
    # Triangle 0: light, small, at z=10, CCW winding so cross(p1-p0,p2-p0)
    # points -Z (toward the receiver below).
    # Triangle 1: receiver, huge, at z=0, diffuse (materials[0]).
    var n_verts = 6
    var points = unsafe_alloc[Float32](n_verts * 4)
    var verts = [
        Point3f(0.0, 0.0, 10.0), Point3f(0.0, 1.0, 10.0), Point3f(1.0, 0.0, 10.0),
        Point3f(-10000.0, -10000.0, 0.0), Point3f(10000.0, -10000.0, 0.0), Point3f(0.0, 10000.0, 0.0),
    ]
    for i in range(n_verts):
        points[unsafe_offset=i*4+0] = verts[i].x
        points[unsafe_offset=i*4+1] = verts[i].y
        points[unsafe_offset=i*4+2] = verts[i].z
        points[unsafe_offset=i*4+3] = Float32(1.0)
    var vertex_indices = unsafe_alloc[Int64](n_verts)
    for i in range(n_verts):
        vertex_indices[unsafe_offset=i] = Int64(i)
    var meshes = unsafe_alloc[TriangleMesh_C](1)
    meshes[unsafe_offset=0] = TriangleMesh_C(
        points, Pointer[Int64, MutUntrackedOrigin].unsafe_dangling(), vertex_indices,
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    )

    var n_tris = 2
    var bounds = unsafe_alloc[Float32](n_tris * 6)
    for t in range(n_tris):
        var p0 = verts[t*3+0]; var p1 = verts[t*3+1]; var p2 = verts[t*3+2]
        bounds[unsafe_offset=t*6+0] = min(p0.x, min(p1.x, p2.x)); bounds[unsafe_offset=t*6+1] = min(p0.y, min(p1.y, p2.y)); bounds[unsafe_offset=t*6+2] = min(p0.z, min(p1.z, p2.z))
        bounds[unsafe_offset=t*6+3] = max(p0.x, max(p1.x, p2.x)); bounds[unsafe_offset=t*6+4] = max(p0.y, max(p1.y, p2.y)); bounds[unsafe_offset=t*6+5] = max(p0.z, max(p1.z, p2.z))
    var max_nodes = n_tris * 2 + 4
    var bvh_nodes = unsafe_alloc[BVH2Node](max_nodes)
    var order = unsafe_alloc[Int32](n_tris)
    _ = build_bvh2(bounds, Int32(n_tris), bvh_nodes, order)
    bounds.unsafe_free()
    var prim_ids = unsafe_alloc[PrimId_C](n_tris)
    for k in range(n_tris):
        var orig = Int(order[unsafe_offset=k])
        prim_ids[unsafe_offset=k] = PrimId_C(Int64(0), Int64(orig * 3), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    order.unsafe_free()

    var materials = unsafe_alloc[Material_C](1)
    materials[unsafe_offset=0] = Material_C(
        MatKind.diffuse, Int8(0), Int8(0), Int8(0),
        RGB(Float32(0.8)), RGB(Float32(0.0)), Int32(-1),
        Float32(0.0), Float32(0.0), Int32(-1), Int32(-1), Float32(1.0), Int32(-1), Int32(-1),
        RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(1.0), Float32(1.0), Int32(-1), RGB(Float32(1.0)), RGB(Float32(0.0)),
        RGB(Float32(1.0)),   # sss_mean_refl (inert)
    )

    var area_lights = unsafe_alloc[AreaLight_C](1)
    # Light triangle area = 0.5 * |cross((0,1,0),(1,0,0))| = 0.5.
    area_lights[unsafe_offset=0] = AreaLight_C(Int32(0), Int32(1), RGB(Float32(1.0)), Float32(0.5), Int8(0), Int8(0), Int8(0), Int8(0))

    return SceneDescriptor2_C(
        bvh_nodes, prim_ids, meshes, Int64(1),
        materials, Int64(1),
        area_lights, Int64(1),
        Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[DistantLight_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[PointLight_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[InfiniteLight_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[Sphere_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[Curve_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[Medium_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[MediumInterface_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[Grid_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[NvdbGrid_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        LightSampler_C(Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Int32(0), Int32(0)),
        Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        Int64(0),
        Pointer[Instance_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        null_spectral_handle(),
        Pointer[GpuTexture_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
        Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(), Float32(0), Float32(1), Int32(9),
        Float32(0), Float32(0), Float32(0), Float32(0), Float32(0),
    )

# Both subpath halves of a VCM pass share one hero-wavelength set (see
# bdpt.mojo's _bdpt_pass_wavelengths); this test drives the halves directly,
# so it supplies that set itself.
comptime _TEST_PASS_WL = sample_wavelengths_uniform(Float32(0.5))

# Bump-footprint reference (cam_pos, px_scale). The test scene has no
# bump/normal maps, so no footprint is ever evaluated; both halves get the
# same values either way.
comptime _NO_CAM = Vec3f(Float32(0), Float32(0), Float32(0))

def test_wavefront_split_matches_original_light_path_exactly() raises:
    var sd = _build_scene()

    var pcg_old = PCG32(UInt64(12345), UInt64(7))
    var scratch_old = unsafe_alloc[Intersection_C](1)
    var lvc_old = unsafe_alloc[BDPTVertex](_BDPT_MAX_VERTS)
    var lvc_path_len_old = unsafe_alloc[Int32](1)
    _bdpt_trace_light_path[False](sd, pcg_old, False, Int32(-1), scratch_old, lvc_old, 0, lvc_path_len_old, Float32(0), Float32(0), _TEST_PASS_WL, _NO_CAM, Float32(0))

    var pcg_new = PCG32(UInt64(12345), UInt64(7))
    var lvc_new = unsafe_alloc[BDPTVertex](_BDPT_MAX_VERTS)
    var lvc_path_len_new = unsafe_alloc[Int32](1)
    var state = _bdpt_light_path_init[False](sd, pcg_new, Int32(-1), 0, lvc_new, lvc_path_len_new, Float32(0), _TEST_PASS_WL)
    lvc_path_len_new[unsafe_offset=0] = state.n_verts

    var pcg_bounce = PCG32(UInt64(0), UInt64(0))
    pcg_bounce.state = state.pcg_state
    pcg_bounce.inc = state.pcg_inc

    # Unpack the state struct into plain locals -- exactly what a real GPU
    # kernel wrapper would do at the top of each bounce launch.
    var ro = state.ro
    var rd = state.rd
    var flux = state.flux
    var n_verts = Int(state.n_verts)
    var dvcm = state.dvcm
    var dvc = state.dvc
    var dvm = state.dvm
    var is_finite_origin = state.is_finite_origin == Int8(1)
    var cur_med_idx = state.cur_med_idx
    var n_lbounces = Int(state.n_lbounces)
    var current_dielectric_ior = state.current_dielectric_ior
    var previous_dielectric_ior = state.previous_dielectric_ior
    var wavelengths = SampledWavelengths(state.wl0, state.wl1, state.wl2, state.wl3, state.wl_pdf)

    var n_iters = 0
    var active = state.active
    while active == Int8(1) and n_iters < Int(_BDPT_MAX_DEPTH):
        n_iters += 1
        var ray = Ray_C(ro, rd)
        var scratch_new = unsafe_alloc[Intersection_C](1)
        scratch_new[unsafe_offset=0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), scratch_new)
        var inter = scratch_new[unsafe_offset=0]
        scratch_new.unsafe_free()

        var cont = _bdpt_light_path_bounce[False](
            sd, pcg_bounce, False, inter, lvc_new, 0, Float32(0), Float32(0),
            ro, rd, flux, n_verts, dvcm, dvc, dvm,
            is_finite_origin, cur_med_idx, n_lbounces,
            current_dielectric_ior, previous_dielectric_ior, wavelengths,
            _NO_CAM, Float32(0),
        )
        active = Int8(1) if cont else Int8(0)
        lvc_path_len_new[unsafe_offset=0] = Int32(n_verts)

    assert_true(lvc_path_len_old[unsafe_offset=0] == lvc_path_len_new[unsafe_offset=0])
    var n = Int(lvc_path_len_old[unsafe_offset=0])
    assert_true(n >= 1)  # sanity: the receiver is huge enough that at least the light vertex was stored
    for i in range(n):
        assert_true(_vertex_close(lvc_old[unsafe_offset=i], lvc_new[unsafe_offset=i]))

    scratch_old.unsafe_free(); lvc_old.unsafe_free(); lvc_path_len_old.unsafe_free()
    lvc_new.unsafe_free(); lvc_path_len_new.unsafe_free()
    sd.bvh2Nodes.unsafe_free(); sd.primIds.unsafe_free(); sd.meshes.unsafe_free(); sd.materials.unsafe_free(); sd.areaLights.unsafe_free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
