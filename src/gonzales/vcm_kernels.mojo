# VCM GPU kernels: per-stage, merge grid and shadow resolve.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.collections import Array
from max.gpu import block_idx, thread_idx, block_dim, MAX_THREADS_PER_BLOCK_METADATA
from std.utils import StaticTuple
from .gpu_tuning import (
    MINCTA_VCM_EMIT, MINCTA_VCM_SPLAT, MINCTA_VCM_CONNECT, MINCTA_VCM_LBOUNCE, MINCTA_VCM_CBOUNCE,
    MINCTA_VCM_RESOLVE,
)
from std.math import sqrt, min
from std.atomic import Atomic
from .geometry import Point3f, Vec3f, RGB
from .materials import MatKind
from .primitives import Ray, Intersection
from .vcm_camis import CamisCamRecord, CamisCamCarry, CamisLightRecord, camis_light_carry_off
from .bvh import traverse_dispatch
from .bvh import SceneView, test_spheres, _is_real_ptr, ray_sphere_hit
from .sampling import FilmFilter, ThinLens
from .rng import PCG32
from .sppm import _HSIZE, _PHOTON_BUCKET_CAP
from .spectrum import SpectralSample, pass_wavelengths, spectral_sample_to_rgb
from .bdpt_vertex import _BDPT_MAX_VERTS, BDPTVertex
from .bdpt_nee import _visible_transmittance
from .vcm_grid import _CAMIS_CAM_RECS, _bdpt_count_merge_vertex, _bdpt_insert_merge_vertex, _vcm_reset_cell
from .bdpt_connect import _bdpt_splat_filtered, _bdpt_connect_to_camera
from .bdpt_camera import (
    _bdpt_trace_camera_and_connect, VCMCameraPathState, _bdpt_camera_path_init, _bdpt_camera_path_bounce,
)
from .bdpt_light import VCMLightPathState, _bdpt_light_path_init, _bdpt_light_path_bounce, _bdpt_trace_light_path

@always_inline
def _finite3(a: Float32, b: Float32, c: Float32) -> Bool:
    """False for NaN/inf: one such sample would otherwise zero its pixel's whole sum at normalisation."""
    return (a - a) == Float32(0) and (b - b) == Float32(0) and (c - c) == Float32(0)


@always_inline
def _accum_camera_sample(accum: Pointer[Float32, MutUntrackedOrigin], accum_merge: Pointer[Float32, MutUntrackedOrigin],
                         pix: Int, c: RGB, m: RGB, sample_clamp: Float32):
    """One camera sample onto the film, connections and merges clamped together as pbrt clamps a sample."""
    if not (_finite3(c.r, c.g, c.b) and _finite3(m.r, m.g, m.b)):
        return
    var k = (c + m).sensor_clamp_scale(sample_clamp)
    accum[unsafe_offset=pix*3]   += c.r * k
    accum[unsafe_offset=pix*3+1] += c.g * k
    accum[unsafe_offset=pix*3+2] += c.b * k
    accum_merge[unsafe_offset=pix*3]   += m.r * k
    accum_merge[unsafe_offset=pix*3+1] += m.g * k
    accum_merge[unsafe_offset=pix*3+2] += m.b * k


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_VCM_EMIT))
def _bdpt_emit_light_paths_gpu(
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin],   # _VCM_CAMIS records, parallel to lvc
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    inter_scratch: Pointer[Intersection, MutUntrackedOrigin],
    n_light_paths_dp: Int64,
    default_emit_med: Int32,
    seed: UInt64,
    pass_idx_dp: Int64,
    # camera_to_world + pixel angular size: the light subpath needs a
    # bump/normal-map footprint and has no differentials of its own, so it
    c2w: Pointer[Float32, MutUntrackedOrigin],
    px_scale: Float32,
    sd: SceneView,
):
    """One thread per light path, each writing only its own dedicated
    per-path slice of `lvc` (VCM Stage 2b, see _bdpt_store_lvc_vertex's"""
    var mediumCount = sd.mediumCount
    var n_light_paths = Int(n_light_paths_dp)
    var pass_idx = Int(pass_idx_dp)
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= n_light_paths:
        return
    var has_med = mediumCount > Int64(0)
    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))
    var scratch = inter_scratch.unsafe_offset(k)
    var pass_wl = pass_wavelengths(pass_idx)
    _bdpt_trace_light_path[True](sd, pcg, has_med, default_emit_med, scratch, lvc, k, lvc_path_len,
                                 mis_vc_weight_factor, mis_vm_weight_factor, pass_wl, lvc_camis)

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_VCM_SPLAT))
def _bdpt_splat_light_paths_gpu(
    accum: Pointer[Float32, MutUntrackedOrigin],
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    n_light_paths_dp: Int64,
    inter_scratch: Pointer[Intersection, MutUntrackedOrigin],
    w2c: Pointer[Float32, MutUntrackedOrigin],
    c2r: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    fw_dp: Int64, fh_dp: Int64,
    px_scale: Float32,
    mis_vm_weight_factor: Float32,
    # The scene's PixelFilter: each splat is spread over its footprint
    # (_bdpt_splat_filtered) so the light-traced half of the image is
    # reconstructed the same way as the camera half.
    film_filter: FilmFilter,
    sd: SceneView,
    splat_clamp: Float32,   # pbrt's per-splat clamp, scaled by n_pix / n_light_paths
    # _VCM_CAMIS: see _bdpt_merge_from_cache's matching parameter.
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin] = Pointer[CamisLightRecord, MutUntrackedOrigin].unsafe_dangling(),
):
    """GPU t=1 light tracing: one thread per light path, splatting each of
    its vertices onto the film through the same `_bdpt_connect_to_camera`"""
    var spectral_coeffs = sd.spectral.coeffs
    var spectral_res_dp = Int64(sd.spectral.res)
    var spectral_cie_x = sd.spectral.cie_x
    var spectral_cie_y = sd.spectral.cie_y
    var spectral_cie_z = sd.spectral.cie_z
    var spectral_d65 = sd.spectral.d65
    var n_light_paths = Int(n_light_paths_dp)
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= n_light_paths:
        return
    var scratch = inter_scratch.unsafe_offset(k)
    var base = k * _BDPT_MAX_VERTS
    var n_verts = Int(lvc_path_len[unsafe_offset=k])
    for local in range(min(n_verts, _BDPT_MAX_VERTS)):
        var r = _bdpt_connect_to_camera(
            lvc[unsafe_offset=base + local], sd, scratch, c2w,
            w2c, c2r, Int32(Int(fw_dp)), Int32(Int(fh_dp)), px_scale,
            Float32(n_light_paths), mis_vm_weight_factor,
            lvc, lvc_camis, base + local)
        if r[0]:
            var (cr, cg, cb) = spectral_sample_to_rgb(
                spectral_coeffs, Int(spectral_res_dp), spectral_cie_x, spectral_cie_y,
                spectral_cie_z, spectral_d65, r[3], lvc[unsafe_offset=base + local].wavelengths)
            if _finite3(cr, cg, cb):
                var c = RGB(cr, cg, cb).sensor_clamped(splat_clamp)
                _bdpt_splat_filtered[True](accum, r[1], r[2], c.r, c.g, c.b,
                                       Int(fw_dp), Int(fh_dp), film_filter)

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_VCM_CONNECT))
def _bdpt_camera_connect_gpu(
    accum: Pointer[Float32, MutUntrackedOrigin],
    accum_merge: Pointer[Float32, MutUntrackedOrigin],
    albedo_accum: Pointer[Float32, MutUntrackedOrigin],
    # Benchmark instrumentation only -- see
    # _bdpt_merge_from_cache's docstring paragraph. Per-pixel
    visit_accum: Pointer[Float32, MutUntrackedOrigin],
    n_pix_dp: Int64,
    fw_dp: Int64,
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    inter_scratch: Pointer[Intersection, MutUntrackedOrigin],
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin],   # _VCM_CAMIS records, parallel to lvc
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    merge_heads: Pointer[Int32, MutUntrackedOrigin],
    merge_inv_cell: Float32,
    merge_r2: Float32,
    merge_norm: Float32,
    px_scale: Float32,
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    n_light_paths_f: Float32,
    film_filter: FilmFilter,
    seed: UInt64,
    pass_idx_dp: Int64,
    sd: SceneView,
    sample_clamp: Float32,
):
    """One thread per pixel. Thin wrapper: build sd, seed this thread's own
    PCG32 (same seed formula vcm_render's CPU driver uses, keyed by pixel"""
    var mediumCount = sd.mediumCount
    var spectral_coeffs = sd.spectral.coeffs
    var spectral_cie_x = sd.spectral.cie_x
    var spectral_cie_y = sd.spectral.cie_y
    var spectral_cie_z = sd.spectral.cie_z
    var spectral_d65 = sd.spectral.d65
    var spectral_res_dp = Int64(sd.spectral.res)
    var spectral_res = Int(spectral_res_dp)
    var n_pix = Int(n_pix_dp)
    var fw = Int(fw_dp)
    var pass_idx = Int(pass_idx_dp)
    var pix = Int(block_idx.x * block_dim.x + thread_idx.x)
    if pix >= n_pix:
        return
    var has_med = mediumCount > Int64(0)
    var px = pix % fw
    var py = pix // fw
    var pcg = PCG32(seed ^ UInt64(pix * 6364136223846793005 + 1442695040888963407),
                     UInt64(pass_idx * 2654435761 + 1))
    var scratch = inter_scratch.unsafe_offset(pix)
    var pass_wl = pass_wavelengths(pass_idx)
    var (contrib, contrib_merge, alb, visit_naive, visit_footprint, visit_thin) = _bdpt_trace_camera_and_connect[True](
        r2c, c2w, px, py, sd, pcg, has_med, scratch, lvc, pix, Int(lvc_path_len[unsafe_offset=pix]),
        merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm,
        px_scale, mis_vc_weight_factor, mis_vm_weight_factor, n_light_paths_f, pass_wl, film_filter,
        lvc_camis=lvc_camis)
    # Benchmark instrumentation only -- see
    # _bdpt_merge_from_cache's docstring paragraph.
    visit_accum[unsafe_offset=pix*3]   += Float32(visit_naive)
    visit_accum[unsafe_offset=pix*3+1] += Float32(visit_footprint)
    visit_accum[unsafe_offset=pix*3+2] += Float32(visit_thin)
    # ── Output boundary: spectral transport -> RGB film ──────────────────
    var (cr, cg, cb) = spectral_sample_to_rgb(
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, contrib, pass_wl)
    var (mr, mg, mb) = spectral_sample_to_rgb(
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, contrib_merge, pass_wl)
    _accum_camera_sample(accum, accum_merge, pix, RGB(cr, cg, cb), RGB(mr, mg, mb), sample_clamp)
    albedo_accum[unsafe_offset=pix*3]   += alb.r
    albedo_accum[unsafe_offset=pix*3+1] += alb.g
    albedo_accum[unsafe_offset=pix*3+2] += alb.b


# ── Task #163 stage 4, part 3: wavefront-staged GPU kernels ─────────────────
# Thin per-bounce wrappers around _bdpt_light_path_init/_bounce and
# _bdpt_camera_path_init/_bounce (this file, above), the counterpart to

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def _bdpt_light_path_init_gpu(
    states: Pointer[VCMLightPathState, MutUntrackedOrigin],
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    mis_vc_weight_factor: Float32,
    n_light_paths_dp: Int64,
    default_emit_med: Int32,
    seed: UInt64,
    pass_idx_dp: Int64,
    sd: SceneView,
):
    """One thread per light path: seed this thread's own PCG32 (same seed
    formula _bdpt_emit_light_paths_gpu uses), call _bdpt_light_path_init,"""
    var n_light_paths = Int(n_light_paths_dp)
    var pass_idx = Int(pass_idx_dp)
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= n_light_paths:
        return
    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))
    var pass_wl = pass_wavelengths(pass_idx)
    states[unsafe_offset=k] = _bdpt_light_path_init[True](sd, pcg, default_emit_med, k, lvc, lvc_path_len, mis_vc_weight_factor, pass_wl)

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def _bdpt_light_path_intersect_gpu(
    sd: SceneView,
    states: Pointer[VCMLightPathState, MutUntrackedOrigin],
    results: Pointer[Intersection, MutUntrackedOrigin],
    count_dp: Int64,
):
    """Batched-per-thread primary/bounce-ray intersect for one light-path
    depth level -- separated from the material dispatch in"""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if states[unsafe_offset=tid].active == Int8(0):
        return
    var ray = Ray(states[unsafe_offset=tid].ro, states[unsafe_offset=tid].rd)
    results[unsafe_offset=tid].hit = Int8(0)
    traverse_dispatch(sd.bvh4, sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), results.unsafe_offset(tid),
                        sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
    test_spheres(sd.spheres, Int(sd.sphereCount), ray, results.unsafe_offset(tid))

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_VCM_LBOUNCE))
def _bdpt_light_path_bounce_gpu(
    states: Pointer[VCMLightPathState, MutUntrackedOrigin],
    results: Pointer[Intersection, MutUntrackedOrigin],
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    n_light_paths_dp: Int64,
    # Bump/normal-map footprint reference -- see _bdpt_emit_light_paths_gpu's
    # matching params (this is the wavefront-staged path to the same walk).
    c2w: Pointer[Float32, MutUntrackedOrigin],
    px_scale: Float32,
    sd: SceneView,
):
    """One bounce's material dispatch for one light path, reading the
    Intersection _bdpt_light_path_intersect_gpu already computed this
    depth level instead of tracing it inline -- see this section's opening
    comment."""
    var mediumCount = sd.mediumCount
    var n_light_paths = Int(n_light_paths_dp)
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= n_light_paths:
        return
    if states[unsafe_offset=k].active == Int8(0):
        return
    var has_med = mediumCount > Int64(0)
    var st = states[unsafe_offset=k]
    var pcg = PCG32(UInt64(0), UInt64(0))
    pcg.state = st.pcg_state
    pcg.inc = st.pcg_inc
    # No _VCM_CAMIS on the wavefront driver (see its camera twin).
    var camis_l = camis_light_carry_off()
    var cont = _bdpt_light_path_bounce[True](
        sd, pcg, has_med, results[unsafe_offset=k], lvc, k, mis_vc_weight_factor, mis_vm_weight_factor,
        st, camis_l, Pointer[CamisLightRecord, MutUntrackedOrigin].unsafe_dangling(),
    )
    lvc_path_len[unsafe_offset=k] = Int32(st.n_verts)
    st.active = Int8(1) if cont else Int8(0)
    st.pcg_state = pcg.state
    st.pcg_inc = pcg.inc
    states[unsafe_offset=k] = st

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def _bdpt_camera_path_init_gpu(
    states: Pointer[VCMCameraPathState, MutUntrackedOrigin],
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    n_pix_dp: Int64,
    fw_dp: Int64,
    px_scale: Float32,
    n_light_paths_f: Float32,
    film_filter: FilmFilter,
    seed: UInt64,
    pass_idx_dp: Int64,
    lens_radius: Float32, focal_distance: Float32,
):
    """One thread per pixel: seed this thread's own PCG32 (same seed formula
    _bdpt_camera_connect_gpu uses), call _bdpt_camera_path_init, store the
    resulting VCMCameraPathState. No scene params needed -- camera-ray
    generation doesn't touch the scene."""
    var n_pix = Int(n_pix_dp)
    var fw = Int(fw_dp)
    var pass_idx = Int(pass_idx_dp)
    var pix = Int(block_idx.x * block_dim.x + thread_idx.x)
    if pix >= n_pix:
        return
    var px = pix % fw
    var py = pix // fw
    var pcg = PCG32(seed ^ UInt64(pix * 6364136223846793005 + 1442695040888963407),
                     UInt64(pass_idx * 2654435761 + 1))
    var pass_wl = pass_wavelengths(pass_idx)
    states[unsafe_offset=pix] = _bdpt_camera_path_init[True](r2c, c2w, px, py, pcg, px_scale, n_light_paths_f, pass_wl, film_filter, lens=ThinLens(lens_radius, focal_distance))

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def _bdpt_camera_path_intersect_gpu(
    sd: SceneView,
    states: Pointer[VCMCameraPathState, MutUntrackedOrigin],
    results: Pointer[Intersection, MutUntrackedOrigin],
    count_dp: Int64,
):
    """Camera-path counterpart to _bdpt_light_path_intersect_gpu -- see its
    docstring."""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if states[unsafe_offset=tid].active == Int8(0):
        return
    var ray = Ray(states[unsafe_offset=tid].ro, states[unsafe_offset=tid].rd)
    results[unsafe_offset=tid].hit = Int8(0)
    traverse_dispatch(sd.bvh4, sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), results.unsafe_offset(tid),
                        sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
    test_spheres(sd.spheres, Int(sd.sphereCount), ray, results.unsafe_offset(tid))

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_VCM_CBOUNCE))
def _bdpt_camera_path_bounce_gpu(
    states: Pointer[VCMCameraPathState, MutUntrackedOrigin],
    results: Pointer[Intersection, MutUntrackedOrigin],
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    merge_heads: Pointer[Int32, MutUntrackedOrigin],
    merge_inv_cell: Float32,
    merge_r2: Float32,
    merge_norm: Float32,
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    n_pix_dp: Int64,
    # Bump/normal-map footprint reference -- see _bdpt_camera_path_bounce's
    # matching params.
    c2w: Pointer[Float32, MutUntrackedOrigin],
    px_scale: Float32,
    sd: SceneView,
    # Task #163 stage 5: see _bdpt_camera_path_bounce's own matching
    # params -- forwarded through unchanged, except Bool -> Int8 (raw kernel
    defer_shadow_rays: Int8 = Int8(0),
    shadow_rays: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    shadow_pending: Pointer[SpectralSample, MutUntrackedOrigin] = Pointer[SpectralSample, MutUntrackedOrigin].unsafe_dangling(),
    shadow_valid: Pointer[Int8, MutUntrackedOrigin] = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling(),
    shadow_seg_med: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
):
    """One bounce's material dispatch (incl. NEE/connect/merge/MNEE, all
    still on the existing software-BVH `results + pix` scratch slot -- see"""
    var mediumCount = sd.mediumCount
    var n_pix = Int(n_pix_dp)
    var pix = Int(block_idx.x * block_dim.x + thread_idx.x)
    if pix >= n_pix:
        return
    if states[unsafe_offset=pix].active == Int8(0):
        return
    var has_med = mediumCount > Int64(0)
    var st = states[unsafe_offset=pix]
    var pcg = PCG32(UInt64(0), UInt64(0))
    pcg.state = st.pcg_state
    pcg.inc = st.pcg_inc
    var camis = CamisCamCarry(False, Int32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))
    var camis_recs = Array[CamisCamRecord, _CAMIS_CAM_RECS](
        fill=CamisCamRecord(Float32(0), Float32(0), Float32(0), Float32(0), Float32(0)))
    var visit_naive = Int32(0)
    var visit_footprint = Int32(0)
    var visit_thin = Int32(0)

    var cont = _bdpt_camera_path_bounce[True](
        sd, pcg, has_med, results[unsafe_offset=pix], results.unsafe_offset(pix), lvc, pix, Int(lvc_path_len[unsafe_offset=pix]),
        merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm,
        mis_vc_weight_factor, mis_vm_weight_factor,
        st, visit_naive, visit_footprint, visit_thin, camis, camis_recs,
        defer_shadow_rays != Int8(0), shadow_rays, shadow_pending, shadow_valid, shadow_seg_med,
    )
    st.active = Int8(1) if cont else Int8(0)
    st.pcg_state = pcg.state
    st.pcg_inc = pcg.inc
    states[unsafe_offset=pix] = st

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def _bdpt_camera_path_accumulate_gpu(
    states: Pointer[VCMCameraPathState, MutUntrackedOrigin],
    accum: Pointer[Float32, MutUntrackedOrigin],
    accum_merge: Pointer[Float32, MutUntrackedOrigin],
    albedo_accum: Pointer[Float32, MutUntrackedOrigin],
    n_pix_dp: Int64,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin],
    spectral_res_dp: Int64,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    sample_clamp: Float32,
):
    """Runs once per `si` sample, after the camera-path bounce loop has
    fully terminated for every lane -- writes each pixel's now-complete"""
    var n_pix = Int(n_pix_dp)
    var pix = Int(block_idx.x * block_dim.x + thread_idx.x)
    if pix >= n_pix:
        return
    # ── Output boundary: spectral transport -> RGB film ──────────────────
    var wl_acc = states[unsafe_offset=pix].wl
    var (tr, tg, tb) = spectral_sample_to_rgb(
        spectral_coeffs, Int(spectral_res_dp), spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, states[unsafe_offset=pix].total, wl_acc)
    var (mr, mg, mb) = spectral_sample_to_rgb(
        spectral_coeffs, Int(spectral_res_dp), spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, states[unsafe_offset=pix].total_merge, wl_acc)
    _accum_camera_sample(accum, accum_merge, pix, RGB(tr, tg, tb), RGB(mr, mg, mb), sample_clamp)
    albedo_accum[unsafe_offset=pix*3]   += states[unsafe_offset=pix].first_alb.r
    albedo_accum[unsafe_offset=pix*3+1] += states[unsafe_offset=pix].first_alb.g
    albedo_accum[unsafe_offset=pix*3+2] += states[unsafe_offset=pix].first_alb.b

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def bdpt_merge_grid_reset_gpu(heads: Pointer[Int32, MutUntrackedOrigin], hsize_dp: Int64):
    var hsize = Int(hsize_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= hsize:
        return
    _vcm_reset_cell(heads, tid)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def bdpt_merge_grid_count_gpu(
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    lvc_cap_dp: Int64,
    heads: Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
):
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= Int(lvc_cap_dp):
        return
    _bdpt_count_merge_vertex(k, lvc, lvc_path_len, heads, inv_cell)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def bdpt_merge_grid_insert_gpu(
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    lvc_cap_dp: Int64,
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    heads: Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    sd: SceneView,
):
    var lvc_cap = Int(lvc_cap_dp)
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= lvc_cap:
        return
    _bdpt_insert_merge_vertex[True](k, lvc, lvc_path_len, merge_next, heads, inv_cell, sd)


comptime _VCM_BUDGET_REDUCE_THREADS = 4096


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def vcm_budget_reduce_gpu(
    stat: Pointer[Float32, MutUntrackedOrigin],
    heads: Pointer[Int32, MutUntrackedOrigin],
    red: Pointer[Float32, MutUntrackedOrigin],
):
    """One pass's per-cell budget sums, for the next pass's lambda:
    red[0] += sqrt(V Q n) and red[1] += Q min(n, cap) over every bucket --"""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var chunk = _HSIZE // _VCM_BUDGET_REDUCE_THREADS
    var s = Float32(0)
    var b = Float32(0)
    for h in range(tid * chunk, (tid + 1) * chunk):
        var q = stat[unsafe_offset=2 * h]
        var n = Float32(heads[unsafe_offset=_HSIZE + h])
        if q > Float32(0) and n > Float32(0):
            s += sqrt(stat[unsafe_offset=2 * h + 1] * q * n)
            b += q * min(n, Float32(_PHOTON_BUCKET_CAP))
    _ = Atomic[Float32].fetch_add(red.unsafe_offset(0), s)
    _ = Atomic[Float32].fetch_add(red.unsafe_offset(1), b)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def reset_shadow_valid_gpu(
    # Task #163 stage 5: zeroes EVERY pixel's shadow_valid slots, every
    # bounce, unconditionally -- including inactive-path and non-diffuse-
    shadow_valid: Pointer[Int8, MutUntrackedOrigin],
    n_pix_dp: Int64,
):
    var n_pix = Int(n_pix_dp)
    var pix = Int(block_idx.x * block_dim.x + thread_idx.x)
    if pix >= n_pix:
        return
    var base = pix * _BDPT_MAX_VERTS
    for local in range(_BDPT_MAX_VERTS):
        shadow_valid[unsafe_offset=base + local] = Int8(0)

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_VCM_RESOLVE))
def resolve_shadow_connect_gpu(
    # Perf (2026-07-13, task #163 stage 5 follow-up): ONE dispatch over
    # ALL n_pix*_BDPT_MAX_VERTS shadow-ray slots (replaces the earlier
    shadow_results: Pointer[Float32, MutUntrackedOrigin],
    mesh_material_idx: Pointer[Int64, MutUntrackedOrigin],
    n_meshes_vk_dp: Int64,
    cam_states: Pointer[VCMCameraPathState, MutUntrackedOrigin],
    shadow_pending: Pointer[SpectralSample, MutUntrackedOrigin],
    shadow_valid: Pointer[Int8, MutUntrackedOrigin],
    shadow_seg_med: Pointer[Int32, MutUntrackedOrigin],
    shadow_rays: Pointer[Float32, MutUntrackedOrigin],
    scratch: Pointer[Intersection, MutUntrackedOrigin],
    count_dp: Int64,
    sd: SceneView,
    instance_base_mesh: Pointer[Int32, MutUntrackedOrigin],
):
    var n_meshes_vk = Int(n_meshes_vk_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var idx = tid
    if shadow_valid[unsafe_offset=idx] == Int8(0):
        return

    var seg_med = shadow_seg_med[unsafe_offset=idx]
    var idx8 = idx * 8
    var org = Point3f(shadow_rays[unsafe_offset=idx8 + 0], shadow_rays[unsafe_offset=idx8 + 1], shadow_rays[unsafe_offset=idx8 + 2])
    var dir = Vec3f(shadow_rays[unsafe_offset=idx8 + 4], shadow_rays[unsafe_offset=idx8 + 5], shadow_rays[unsafe_offset=idx8 + 6])
    var dist = shadow_rays[unsafe_offset=idx8 + 7] / Float32(0.9995)

    var ridx = tid * 8
    var iresults = shadow_results.unsafe_bitcast[Int32]()
    var hitFlag = iresults[unsafe_offset=ridx + 6]

    var needs_fallback = False
    if hitFlag != Int32(1):
        if seg_med >= Int32(0):
            needs_fallback = True
        elif hitFlag == Int32(3):
            needs_fallback = True            # --rt-hardware: too many rejected alpha hits, the software trace decides
        elif sd.sphereCount > Int64(0):
            # Analytic spheres are not in the acceleration structure (--rt-hardware): one in the way needs the full
            # (dielectric / medium aware) visibility trace.
            var sray = Ray(org, dir)
            for si in range(Int(sd.sphereCount)):
                if ray_sphere_hit(sd.spheres[unsafe_offset=si].center, sd.spheres[unsafe_offset=si].radius, sray, Float32(1e-4), dist * Float32(0.9995)) > Float32(0.0):
                    needs_fallback = True
        # else: fully visible (Tr=1), pending already holds the correct
        # unweighted contribution -- nothing to multiply.
    else:
        var mi = Int(iresults[unsafe_offset=ridx + 4])
        if mi >= n_meshes_vk and _is_real_ptr(instance_base_mesh):
            mi = Int(instance_base_mesh[unsafe_offset=mi - n_meshes_vk]) + Int(iresults[unsafe_offset=ridx + 7])
        var mat_idx = Int64(0)
        if mi >= 0 and mi < n_meshes_vk:
            mat_idx = mesh_material_idx[unsafe_offset=mi]
        var mat = sd.materials[unsafe_offset=Int(mat_idx)]
        if seg_med >= Int32(0) or mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric or mat.type == MatKind.interface:
            needs_fallback = True
        else:
            # Opaque hit, no medium: fully occluded.
            shadow_pending[unsafe_offset=idx] = SpectralSample(Float32(0))

    if needs_fallback:
        # Per-thread scratch slot (offset by `tid`) -- matches the
        # established convention elsewhere in this file (e.g.
        var dst = org + dir * dist
        var cst = cam_states[unsafe_offset=idx // _BDPT_MAX_VERTS]
        var wl_sp = cst.wl
        var Tr = _visible_transmittance(org, dst, seg_med, sd, scratch.unsafe_offset(tid), wl_sp)
        var p = shadow_pending[unsafe_offset=idx]
        shadow_pending[unsafe_offset=idx] = p * Tr

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def sum_shadow_connect_gpu(
    states: Pointer[VCMCameraPathState, MutUntrackedOrigin],
    shadow_pending: Pointer[SpectralSample, MutUntrackedOrigin],
    shadow_valid: Pointer[Int8, MutUntrackedOrigin],
    n_pix_dp: Int64,
):
    var n_pix = Int(n_pix_dp)
    var pix = Int(block_idx.x * block_dim.x + thread_idx.x)
    if pix >= n_pix:
        return
    var base = pix * _BDPT_MAX_VERTS
    var sum = SpectralSample(Float32(0))
    for local in range(_BDPT_MAX_VERTS):
        if shadow_valid[unsafe_offset=base + local] != Int8(0):
            sum += shadow_pending[unsafe_offset=base + local]
    states[unsafe_offset=pix].total += sum
