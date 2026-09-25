# Task #163 stage 4 part 2: equivalence test for the wavefront-staged
# camera-path split, the counterpart to test_vcm_light_path_bounce.mojo.
# Drives BOTH `_bdpt_trace_camera_and_connect` (the original, single-mega-
# function loop, still used by the CPU renderer) AND the new
# `_bdpt_camera_path_init` + `_bdpt_camera_path_bounce` pair (intended for a
# future GPU wavefront host loop) from the SAME PCG32 seed over the SAME
# real BVH-backed scene, and asserts the returned (total, first_alb) match
# closely. `path_len=0` is used throughout so the LVC connect/merge calls
# (gated by `if path_len > 0:` in the original) are skipped identically by
# both sides without needing a real Light Vertex Cache or merge grid --
# this test's scope is the bounce loop's own control-flow transformation
# (break/continue -> return False/True, intersect moved to the caller,
# accumulators threaded as mut state), not LVC connect/merge machinery,
# which is untouched by this split (`scratch`/software-BVH shadow rays are
# passed straight through unchanged, per the user-confirmed stage 4 scope).
#
# Scene: camera at the origin looking straight down +Z (crafted r2c/c2w so
# the primary ray is exactly (0,0,0)+t*(0,0,1), independent of realistic
# camera-matrix semantics -- both code paths consume the same raw floats
# identically, so "physically sensible camera" doesn't matter here, only
# "same inputs, same outputs"). A huge diffuse floor at z=10 is hit by the
# primary ray, storing a vertex and scattering; the next bounce escapes the
# scene (no infinite lights) and terminates via the miss-handling path --
# exercising both the "hit" and "escape" halves of the split. No lights are
# in this scene (NEE loops all zero-iteration) -- `sd.spectral` is the
# dangling `null_spectral_handle()` sentinel, which is only safe when
# nothing actually evaluates spectral NEE; the NEE/spectral call sites
# themselves are copied byte-for-byte unchanged by this split (see
# `_bdpt_camera_path_bounce`'s docstring), so their correctness is already
# covered elsewhere and out of scope for this test.

from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import RGB, Point3f, Vec3f, Ray_C, Intersection_C, PrimId_C, Sphere_C, Instance_C, MeasuredBRDF_C, GpuTexture_C, NormalSlopeMap_C, TriangleMesh_C, Material_C, MatKind
from gonzales.media import Medium_C, MediumInterface_C, Grid_C, NvdbGrid_C
from gonzales.lights import LightSampler_C, AreaLight_C, DistantLight_C, PointLight_C, InfiniteLight_C
from gonzales.curves import Curve_C
from gonzales.bvh import SceneDescriptor2_C, BVH2Node, build_bvh2, traverse_bvh2_core, test_spheres
from gonzales.rng import PCG32
from gonzales.sampling import film_filter_of
from gonzales.spectrum import sample_wavelengths_uniform, null_spectral_handle, SampledWavelengths
from gonzales.bdpt import (
    BDPTVertex, _bdpt_trace_camera_and_connect, _bdpt_camera_path_init,
    _bdpt_camera_path_bounce, _BDPT_MAX_VERTS, _BDPT_MAX_DEPTH,
)

comptime EPS: Float32 = 1e-3

def _close(a: Float32, b: Float32) -> Bool:
    var d = a - b
    if d < Float32(0): d = -d
    return d < EPS

def _build_scene() -> SceneDescriptor2_C:
    # One huge diffuse triangle at z=10.
    var n_verts = 3
    var points = unsafe_alloc[Float32](n_verts * 4)
    var verts = [
        Point3f(-10000.0, -10000.0, 10.0), Point3f(10000.0, -10000.0, 10.0), Point3f(0.0, 10000.0, 10.0),
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

    var n_tris = 1
    var bounds = unsafe_alloc[Float32](n_tris * 6)
    var p0 = verts[0]; var p1 = verts[1]; var p2 = verts[2]
    bounds[unsafe_offset=0] = min(p0.x, min(p1.x, p2.x)); bounds[unsafe_offset=1] = min(p0.y, min(p1.y, p2.y)); bounds[unsafe_offset=2] = min(p0.z, min(p1.z, p2.z))
    bounds[unsafe_offset=3] = max(p0.x, max(p1.x, p2.x)); bounds[unsafe_offset=4] = max(p0.y, max(p1.y, p2.y)); bounds[unsafe_offset=5] = max(p0.z, max(p1.z, p2.z))
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

    return SceneDescriptor2_C(
        bvh_nodes, prim_ids, meshes, Int64(1),
        materials, Int64(1),
        Pointer[AreaLight_C, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
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

def _identity_camera_matrices() -> Tuple[Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]]:
    # r2c crafted so that at px=py=0 (fX=fY=0.5): cx=cy=0, cz=1, cw=1 --
    # camera-space direction (0,0,1) regardless of fX/fY's exact value
    # (columns 0/1 are all zero), independent of realistic raster-to-camera
    # semantics -- see this file's module docstring.
    var r2c = unsafe_alloc[Float32](16)
    for i in range(16): r2c[unsafe_offset=i] = Float32(0.0)
    r2c[unsafe_offset=14] = Float32(1.0)
    r2c[unsafe_offset=15] = Float32(1.0)
    var c2w = unsafe_alloc[Float32](16)
    for i in range(16): c2w[unsafe_offset=i] = Float32(0.0)
    c2w[unsafe_offset=0] = Float32(1.0); c2w[unsafe_offset=5] = Float32(1.0); c2w[unsafe_offset=10] = Float32(1.0); c2w[unsafe_offset=15] = Float32(1.0)
    return (r2c, c2w)

# Both subpath halves of a VCM pass share one hero-wavelength set (see
# bdpt.mojo's _bdpt_pass_wavelengths); this test drives the two halves
# directly, so it supplies that set itself.
comptime _TEST_PASS_WL = sample_wavelengths_uniform(Float32(0.5))

# Box, pbrt's default radius 0.5: the test compares two implementations of
# the same camera path, so any filter works as long as both get it.
comptime _TEST_FILTER = film_filter_of(Int32(2), Float32(0.5), Float32(0.5), Float32(0.5))

def test_wavefront_split_matches_original_camera_path_closely() raises:
    var sd = _build_scene()
    var (r2c, c2w) = _identity_camera_matrices()
    comptime px_scale = Float32(0.01)
    comptime n_light_paths_f = Float32(1.0)

    var pcg_old = PCG32(UInt64(999), UInt64(3))
    var scratch_old = unsafe_alloc[Intersection_C](1)
    var (total_old, alb_old) = _bdpt_trace_camera_and_connect[False](
        r2c, c2w, 0, 0, sd, pcg_old, False, scratch_old,
        Pointer[BDPTVertex, MutUntrackedOrigin].unsafe_dangling(), 0, 0,
        Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
        Float32(0), Float32(0), Float32(0),
        px_scale, Float32(0), Float32(0), n_light_paths_f, _TEST_PASS_WL, _TEST_FILTER,
    )

    var pcg_new = PCG32(UInt64(999), UInt64(3))
    var scratch_new = unsafe_alloc[Intersection_C](1)
    var state = _bdpt_camera_path_init[False](r2c, c2w, 0, 0, pcg_new, px_scale, n_light_paths_f, _TEST_PASS_WL, _TEST_FILTER)

    var pcg_bounce = PCG32(UInt64(0), UInt64(0))
    pcg_bounce.state = state.pcg_state
    pcg_bounce.inc = state.pcg_inc

    var ro = state.ro
    var rd = state.rd
    var beta = state.beta
    var total = state.total
    var first_alb = state.first_alb
    var n_verts = Int(state.n_verts)
    var n_bounces = Int(state.n_bounces)
    var cur_med_idx = state.cur_med_idx
    var dvcm = state.dvcm
    var dvc = state.dvc
    var dvm = state.dvm
    var last_bsdf_pdf = state.last_bsdf_pdf
    var mis_null_dist = state.mis_null_dist
    var current_dielectric_ior = state.current_dielectric_ior
    var previous_dielectric_ior = state.previous_dielectric_ior
    var wavelengths = SampledWavelengths(state.wl0, state.wl1, state.wl2, state.wl3, state.wl_pdf)

    var n_iters = 0
    var active = state.active
    while active == Int8(1) and n_iters < Int(_BDPT_MAX_DEPTH):
        n_iters += 1
        var ray_o = ro; var ray_d = rd
        scratch_new[unsafe_offset=0].hit = Int8(0)
        var ray = Ray_C(ray_o, ray_d)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), scratch_new)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, scratch_new)
        var inter = scratch_new[unsafe_offset=0]

        var cont = _bdpt_camera_path_bounce[False](
            sd, pcg_bounce, False, inter, scratch_new,
            Pointer[BDPTVertex, MutUntrackedOrigin].unsafe_dangling(), 0, 0,
            Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
            Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
            Float32(0), Float32(0), Float32(0), Float32(0), Float32(0),
            ro, rd, beta, total, first_alb, n_verts, n_bounces, cur_med_idx,
            dvcm, dvc, dvm, last_bsdf_pdf, mis_null_dist,
            current_dielectric_ior, previous_dielectric_ior, wavelengths,
            # Bump footprint reference; the camera sits at the origin
            # (identity c2w), and the scene has no bump/normal maps anyway.
            Vec3f(Float32(0)), px_scale,
        )
        active = Int8(1) if cont else Int8(0)

    # `total` is spectral (BDPT/VCM transport carries hero wavelengths); the
    # two halves must agree lane for lane, which is the real invariant here.
    assert_true(_close(total.v0, total_old.v0))
    assert_true(_close(total.v1, total_old.v1))
    assert_true(_close(total.v2, total_old.v2))
    assert_true(_close(total.v3, total_old.v3))
    assert_true(_close(first_alb.r, alb_old.r))
    assert_true(_close(first_alb.g, alb_old.g))
    assert_true(_close(first_alb.b, alb_old.b))
    assert_true(n_verts >= 1)  # sanity: the floor is huge enough to guarantee a hit + a stored vertex
    assert_true(_close(first_alb.r, Float32(0.8)))  # sanity: the stored vertex's albedo is the real material, not a stale zero

    scratch_old.unsafe_free(); scratch_new.unsafe_free()
    r2c.unsafe_free(); c2w.unsafe_free()
    sd.bvh2Nodes.unsafe_free(); sd.primIds.unsafe_free(); sd.meshes.unsafe_free(); sd.materials.unsafe_free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
