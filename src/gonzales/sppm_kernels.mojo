# SPPM GPU kernels.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from max.gpu import block_idx, thread_idx, block_dim
from .primitives import Intersection
from .media import Medium, Grid, NvdbGrid
from .bvh import SceneView
from .sampling import FilmFilter
from .rng import PCG32
from .sppm import (
    grid_reset_cell, _sppm_count_photon, SPPMPixel, SPPMPhoton, _sppm_insert_photon, _sppm_gather_one,
    _sppm_nee_one, _sppm_finalize_albedo_one_pixel, _sppm_finalize_one_pixel, _sppm_trace_visible_point,
    _sppm_trace_photon,
)
from .spectrum import pass_wavelengths

def sppm_reset_i32_gpu(counter: Pointer[Int32, MutUntrackedOrigin]):
    if block_idx.x == 0 and thread_idx.x == 0:
        counter[unsafe_offset=0] = Int32(0)


def sppm_gen_vp_gpu(
    vps: Pointer[SPPMPixel, MutUntrackedOrigin],
    inter_scratch: Pointer[Intersection, MutUntrackedOrigin],
    n_pix_dp: Int64,
    vp_samples_dp: Int64,
    fw: Int32,
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    init_r2: Float32,
    seed: UInt64,
    max_depth_dp: Int64,
    film_filter: FilmFilter,
    sd: SceneView,
):
    """One thread per (pixel, vp_sample). Calls the SAME
    _sppm_trace_visible_point the CPU driver (_sppm_camera_pass) calls,
    with use_gpu=True (task #151: real image-texture reflectance)."""
    var n_pix = Int(n_pix_dp)
    var vp_samples = Int(vp_samples_dp)
    var combined = Int(block_idx.x * block_dim.x + thread_idx.x)
    if combined >= n_pix * vp_samples:
        return
    var pix = combined // vp_samples
    var px = pix % Int(fw)
    var py = pix // Int(fw)
    var pcg = PCG32(seed ^ UInt64(combined * 6364136223846793005 + 1), UInt64(1))
    vps[unsafe_offset=combined] = _sppm_trace_visible_point[True](sd, pcg, r2c, c2w, px, py, Int32(pix), init_r2, inter_scratch.unsafe_offset(combined), Int(max_depth_dp), film_filter, combined % vp_samples)


def sppm_emit_photons_gpu(
    photons: Pointer[SPPMPhoton, MutUntrackedOrigin],
    n_emit_dp: Int64,
    max_photons_dp: Int64,
    inter_scratch: Pointer[Intersection, MutUntrackedOrigin],
    stored_counter: Pointer[Int32, MutUntrackedOrigin],
    default_emit_med: Int32,
    seed: UInt64,
    pass_idx_dp: Int64,
    max_depth_dp: Int64,
    sd: SceneView,
):
    """One thread per emitted photon path. Calls the SAME _sppm_trace_photon
    the CPU driver (_sppm_photon_pass) calls, with use_gpu=True so"""
    var areaLightCount = sd.areaLightCount
    var sphereCount = sd.sphereCount
    var distantLightCount = sd.distantLightCount
    var infiniteLightCount = sd.infiniteLightCount
    var pointLightCount = sd.pointLightCount
    var spectral_coeffs = sd.spectral.coeffs
    var spectral_cie_x = sd.spectral.cie_x
    var spectral_cie_y = sd.spectral.cie_y
    var spectral_cie_z = sd.spectral.cie_z
    var spectral_d65 = sd.spectral.d65
    var spectral_res_dp = Int64(sd.spectral.res)
    var spectral_res = Int(spectral_res_dp)
    var n_emit = Int(n_emit_dp)
    var max_photons = Int(max_photons_dp)
    var pass_idx = Int(pass_idx_dp)
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    # sphereCount is part of the test because an analytic sphere can BE the
    # scene's only light (Sphere.isAreaLight); leaving it out made this
    if k >= n_emit or (areaLightCount == Int64(0) and distantLightCount == Int64(0)
                       and infiniteLightCount == Int64(0) and pointLightCount == Int64(0)
                       and sphereCount == Int64(0)):
        return
    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))
    _sppm_trace_photon[True, True](sd, pcg, inter_scratch.unsafe_offset(k), n_emit, photons, max_photons, stored_counter, default_emit_med, Int(max_depth_dp),
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, pass_wavelengths(pass_idx))


def sppm_grid_reset_gpu(heads: Pointer[Int32, MutUntrackedOrigin], hsize_dp: Int64):
    var hsize = Int(hsize_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= hsize:
        return
    grid_reset_cell(heads, tid)


def sppm_grid_count_gpu(
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    n_stored_dp: Int64,
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
):
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= Int(n_stored_dp):
        return
    _sppm_count_photon(k, photons, heads, inv_cell)


def sppm_grid_insert_gpu(
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    n_stored_dp: Int64,
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    pass_idx_dp: Int64,
):
    var n_stored = Int(n_stored_dp)
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= n_stored:
        return
    _sppm_insert_photon[True](k, photons, heads, inv_cell, Int(pass_idx_dp))


def sppm_gather_gpu(
    vps:      Pointer[SPPMPixel, MutUntrackedOrigin],
    n_pix_dp:    Int64,
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    sd: SceneView,
    pass_idx_dp: Int64 = Int64(0),
    med_arr_dp: Pointer[Medium, MutUntrackedOrigin] = Pointer[Medium, MutUntrackedOrigin].unsafe_dangling(),
    med_count_dp: Int64 = Int64(0),
    grids_dp: Pointer[Grid, MutUntrackedOrigin] = Pointer[Grid, MutUntrackedOrigin].unsafe_dangling(),
    nvdb_grids_dp: Pointer[NvdbGrid, MutUntrackedOrigin] = Pointer[NvdbGrid, MutUntrackedOrigin].unsafe_dangling(),
):
    """One thread per visible point."""
    var spectral_coeffs = sd.spectral.coeffs
    var spectral_cie_x = sd.spectral.cie_x
    var spectral_cie_y = sd.spectral.cie_y
    var spectral_cie_z = sd.spectral.cie_z
    var spectral_d65 = sd.spectral.d65
    var spectral_res_dp = Int64(sd.spectral.res)
    var spectral_res = Int(spectral_res_dp)
    var n_pix = Int(n_pix_dp)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n_pix:
        return
    _sppm_gather_one(vps, i, photons, heads, inv_cell, sd, med_arr_dp, Int(med_count_dp),
                     grids_dp, nvdb_grids_dp,
                     spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
                     spectral_cie_z, spectral_d65, pass_wavelengths(Int(pass_idx_dp)), Int(pass_idx_dp))


def sppm_nee_gpu(
    vps:    Pointer[SPPMPixel, MutUntrackedOrigin],
    n_vps_dp:  Int64,
    seed:   UInt64,
    pass_idx_dp: Int64,
    sd: SceneView,
):
    """One thread per visible point. Calls the SAME _sppm_nee_one the CPU
    driver (_sppm_nee_update) calls. The only one of SPPM's 4 GPU kernels"""
    var n_vps = Int(n_vps_dp)
    var pass_idx = Int(pass_idx_dp)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n_vps:
        return
    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + i), UInt64(11))
    _sppm_nee_one(vps, i, sd, pcg)


def sppm_finalize_gpu(
    vps:        Pointer[SPPMPixel, MutUntrackedOrigin],
    n_pix_dp:      Int64,
    vp_samples_dp: Int64,
    n_passes:   Int32,
    iso_scale:  Float32,
    out_global: Pointer[Float32, MutUntrackedOrigin],
    out_caustic: Pointer[Float32, MutUntrackedOrigin],
    albedo_out: Pointer[Float32, MutUntrackedOrigin],
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin],
    spectral_res_dp: Int64,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
):
    """One thread per pixel. Calls the SAME _sppm_finalize_one_pixel the CPU
    driver (sppm_render's tail loop) calls, plus the matching albedo AOV"""
    var n_pix = Int(n_pix_dp)
    var vp_samples = Int(vp_samples_dp)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n_pix:
        return
    var (acc_g, acc_c) = _sppm_finalize_one_pixel(vps, i, vp_samples, n_passes, iso_scale,
                                       spectral_coeffs, Int(spectral_res_dp), spectral_cie_x,
                                       spectral_cie_y, spectral_cie_z, spectral_d65)
    out_global[unsafe_offset=i * 3 + 0] = acc_g.r
    out_global[unsafe_offset=i * 3 + 1] = acc_g.g
    out_global[unsafe_offset=i * 3 + 2] = acc_g.b
    out_caustic[unsafe_offset=i * 3 + 0] = acc_c.r
    out_caustic[unsafe_offset=i * 3 + 1] = acc_c.g
    out_caustic[unsafe_offset=i * 3 + 2] = acc_c.b
    var alb = _sppm_finalize_albedo_one_pixel(vps, i, vp_samples)
    albedo_out[unsafe_offset=i * 3 + 0] = alb.r
    albedo_out[unsafe_offset=i * 3 + 1] = alb.g
    albedo_out[unsafe_offset=i * 3 + 2] = alb.b


# ── GPU host driver ───────────────────────────────────────────────────────────
