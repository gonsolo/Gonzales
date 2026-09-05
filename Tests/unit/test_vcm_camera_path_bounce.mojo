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

from std.memory import alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import (
    RGB, Point3f, Vec3f, Ray_C, Intersection_C, LightSampler_C, PrimId_C,
    AreaLight_C, DistantLight_C, PointLight_C, InfiniteLight_C, Sphere_C,
    Medium_C, MediumInterface_C, Grid_C, Instance_C, MeasuredBRDF_C,
    GpuTexture_C, NormalSlopeMap_C, TriangleMesh_C, Material_C, MatKind, Curve_C,
)
from gonzales.bvh import SceneDescriptor2_C, BVH2Node, build_bvh2, traverse_bvh2_core, test_spheres
from gonzales.rng import PCG32
from gonzales.spectrum import null_spectral_handle, SampledWavelengths
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
    var points = alloc[Float32](n_verts * 4)
    var verts = [
        Point3f(-10000.0, -10000.0, 10.0), Point3f(10000.0, -10000.0, 10.0), Point3f(0.0, 10000.0, 10.0),
    ]
    for i in range(n_verts):
        points[i*4+0] = verts[i].x
        points[i*4+1] = verts[i].y
        points[i*4+2] = verts[i].z
        points[i*4+3] = Float32(1.0)
    var vertex_indices = alloc[Int64](n_verts)
    for i in range(n_verts):
        vertex_indices[i] = Int64(i)
    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = TriangleMesh_C(
        points, UnsafePointer[Int64, MutExternalOrigin].unsafe_dangling(), vertex_indices,
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    )

    var n_tris = 1
    var bounds = alloc[Float32](n_tris * 6)
    var p0 = verts[0]; var p1 = verts[1]; var p2 = verts[2]
    bounds[0] = min(p0.x, min(p1.x, p2.x)); bounds[1] = min(p0.y, min(p1.y, p2.y)); bounds[2] = min(p0.z, min(p1.z, p2.z))
    bounds[3] = max(p0.x, max(p1.x, p2.x)); bounds[4] = max(p0.y, max(p1.y, p2.y)); bounds[5] = max(p0.z, max(p1.z, p2.z))
    var max_nodes = n_tris * 2 + 4
    var bvh_nodes = alloc[BVH2Node](max_nodes)
    var order = alloc[Int32](n_tris)
    _ = build_bvh2(bounds, Int32(n_tris), bvh_nodes, order)
    bounds.free()
    var prim_ids = alloc[PrimId_C](n_tris)
    for k in range(n_tris):
        var orig = Int(order[k])
        prim_ids[k] = PrimId_C(Int64(0), Int64(orig * 3), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    order.free()

    var materials = alloc[Material_C](1)
    materials[0] = Material_C(
        MatKind.diffuse, Int8(0), Int8(0), Int8(0),
        RGB(Float32(0.8)), RGB(Float32(0.0)), Int32(-1),
        Float32(0.0), Float32(0.0), Int32(-1), Int32(-1), Int32(-1),
        RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(1.0), Float32(1.0), Int32(-1),
    )

    return SceneDescriptor2_C(
        bvh_nodes, prim_ids, meshes, Int64(1),
        materials, Int64(1),
        UnsafePointer[AreaLight_C, MutExternalOrigin].unsafe_dangling(), Int64(0),
        UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(), Int64(0),
        UnsafePointer[DistantLight_C, MutExternalOrigin].unsafe_dangling(), Int64(0),
        UnsafePointer[PointLight_C, MutExternalOrigin].unsafe_dangling(), Int64(0),
        UnsafePointer[InfiniteLight_C, MutExternalOrigin].unsafe_dangling(), Int64(0),
        UnsafePointer[Sphere_C, MutExternalOrigin].unsafe_dangling(), Int64(0),
        UnsafePointer[Curve_C, MutExternalOrigin].unsafe_dangling(), Int64(0),
        UnsafePointer[Medium_C, MutExternalOrigin].unsafe_dangling(), Int64(0),
        UnsafePointer[MediumInterface_C, MutExternalOrigin].unsafe_dangling(), Int64(0),
        UnsafePointer[Grid_C, MutExternalOrigin].unsafe_dangling(), Int64(0),
        LightSampler_C(UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(), Int32(0), Int32(0)),
        UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        Int64(0),
        UnsafePointer[Instance_C, MutExternalOrigin].unsafe_dangling(), Int64(0),
        UnsafePointer[MeasuredBRDF_C, MutExternalOrigin].unsafe_dangling(), Int64(0),
        null_spectral_handle(),
        UnsafePointer[GpuTexture_C, MutExternalOrigin].unsafe_dangling(), Int64(0),
        UnsafePointer[NormalSlopeMap_C, MutExternalOrigin].unsafe_dangling(),
    )

def _identity_camera_matrices() -> Tuple[UnsafePointer[Float32, MutExternalOrigin], UnsafePointer[Float32, MutExternalOrigin]]:
    # r2c crafted so that at px=py=0 (fX=fY=0.5): cx=cy=0, cz=1, cw=1 --
    # camera-space direction (0,0,1) regardless of fX/fY's exact value
    # (columns 0/1 are all zero), independent of realistic raster-to-camera
    # semantics -- see this file's module docstring.
    var r2c = alloc[Float32](16)
    for i in range(16): r2c[i] = Float32(0.0)
    r2c[14] = Float32(1.0)
    r2c[15] = Float32(1.0)
    var c2w = alloc[Float32](16)
    for i in range(16): c2w[i] = Float32(0.0)
    c2w[0] = Float32(1.0); c2w[5] = Float32(1.0); c2w[10] = Float32(1.0); c2w[15] = Float32(1.0)
    return (r2c, c2w)

def test_wavefront_split_matches_original_camera_path_closely() raises:
    var sd = _build_scene()
    var (r2c, c2w) = _identity_camera_matrices()
    comptime px_scale = Float32(0.01)
    comptime n_light_paths_f = Float32(1.0)

    var pcg_old = PCG32(UInt64(999), UInt64(3))
    var scratch_old = alloc[Intersection_C](1)
    var (total_old, alb_old) = _bdpt_trace_camera_and_connect[False](
        r2c, c2w, 0, 0, sd, pcg_old, False, scratch_old,
        UnsafePointer[BDPTVertex, MutExternalOrigin].unsafe_dangling(), 0, 0,
        UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
        Float32(0), Float32(0), Float32(0),
        px_scale, Float32(0), Float32(0), n_light_paths_f,
    )

    var pcg_new = PCG32(UInt64(999), UInt64(3))
    var scratch_new = alloc[Intersection_C](1)
    var state = _bdpt_camera_path_init[False](r2c, c2w, 0, 0, pcg_new, px_scale, n_light_paths_f)

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
    var wavelengths = SampledWavelengths(state.wl0, state.wl1, state.wl2, state.wl3, state.wl_pdf)

    var n_iters = 0
    var active = state.active
    while active == Int8(1) and n_iters < Int(_BDPT_MAX_DEPTH):
        n_iters += 1
        var ray_o = ro; var ray_d = rd
        scratch_new[0].hit = Int8(0)
        var ray = Ray_C(ray_o, ray_d)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), scratch_new)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, scratch_new)
        var inter = scratch_new[0]

        var cont = _bdpt_camera_path_bounce[False](
            sd, pcg_bounce, False, inter, scratch_new,
            UnsafePointer[BDPTVertex, MutExternalOrigin].unsafe_dangling(), 0, 0,
            UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
            UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
            Float32(0), Float32(0), Float32(0), Float32(0), Float32(0),
            ro, rd, beta, total, first_alb, n_verts, n_bounces, cur_med_idx,
            dvcm, dvc, dvm, last_bsdf_pdf, wavelengths,
        )
        active = Int8(1) if cont else Int8(0)

    assert_true(_close(total.r, total_old.r))
    assert_true(_close(total.g, total_old.g))
    assert_true(_close(total.b, total_old.b))
    assert_true(_close(first_alb.r, alb_old.r))
    assert_true(_close(first_alb.g, alb_old.g))
    assert_true(_close(first_alb.b, alb_old.b))
    assert_true(n_verts >= 1)  # sanity: the floor is huge enough to guarantee a hit + a stored vertex
    assert_true(_close(first_alb.r, Float32(0.8)))  # sanity: the stored vertex's albedo is the real material, not a stale zero

    scratch_old.free(); scratch_new.free()
    r2c.free(); c2w.free()
    sd.bvh2Nodes.free(); sd.primIds.free(); sd.meshes.free(); sd.materials.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
