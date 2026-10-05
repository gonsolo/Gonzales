# SPPM GPU driver.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.sys import has_accelerator
from std.sys.info import size_of
from max.gpu import block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import max, min, ceildiv
from std.memory.alloc import unsafe_alloc
from .materials import LobeKind
from .primitives import Intersection
from .media import Medium, SSS_WALK_ROUNDS, Grid, NvdbGrid
from .bssrdf import dipole_max_radius
from .bvh import SceneView
from .sampling import film_filter_of
from .footprint import camera_footprint
from .pbrt_parser import ParsedScene_Mojo
from .sppm import _HSIZE, SPPMPixel, SPPMPhoton, _VP_SAMPLES, _MAX_B, _photon_depth_cap
from .gpu_scene import GpuSceneHandle
from max.gpu.host._nvidia_cuda import CUDA
from .progress import Progress
from .outputs import finish_render
from .sppm_kernels import (
    sppm_reset_i32_gpu, sppm_gen_vp_gpu, sppm_emit_photons_gpu, sppm_grid_reset_gpu, sppm_grid_count_gpu,
    sppm_grid_insert_gpu, sppm_gather_gpu, sppm_nee_gpu, sppm_finalize_gpu,
)

def sppm_render_gpu(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneView,
    n_passes: Int,
    n_photons_per_pass: Int,
    initial_radius: Float32,
    no_denoise: Bool,
    verbose:  Bool,
) -> Int32:
    """GPU-accelerated Stochastic Progressive Photon Mapping — same algorithm
    as sppm_render, parallelized: one thread per visible-point sample for the"""
    if Int(sd.areaLightCount) + Int(sd.distantLightCount) + Int(sd.infiniteLightCount) + Int(sd.pointLightCount) == 0 and not (sd.sphereLightCount > 0):
        print("SPPM: no lights in scene, cannot emit photons")
        return Int32(-1)

    var fw = Int(psc[unsafe_offset=0].film_w)
    var fh = Int(psc[unsafe_offset=0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[unsafe_offset=0].film_iso / Float32(100)

    print("SPPM (GPU): " + String(fw) + "x" + String(fh)
          + " " + String(n_passes) + " passes x "
          + String(n_photons_per_pass) + " photons  r=" + String(initial_radius))

    var default_emit_med = Int32(-1)
    if Int(sd.mediumCount) > 0 and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[unsafe_offset=mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    # A BSSRDF visible point gathers out to the material's DIFFUSION reach,
    # which for skin is several times the SPPM radius this scene would
    var eff_radius = initial_radius
    for _mi in range(Int(sd.mediumCount)):
        if sd.mediums[unsafe_offset=_mi].is_sss != Int32(0):
            var _rq = dipole_max_radius(sd.mediums[unsafe_offset=_mi].sigma_s, sd.mediums[unsafe_offset=_mi].sigma_a, sd.mediums[unsafe_offset=_mi].g)
            if _rq > eff_radius:
                eff_radius = _rq
    # init_r2 stays on the SCENE's radius -- widening it would blur every
    # ordinary surface visible point in the scene. Only the grid CELLS grow,
    var init_r2 = initial_radius * initial_radius
    var inv_cell = Float32(1.0) / eff_radius
    if verbose:
        print("SPPM: vp radius " + String(initial_radius) + ", grid cell " + String(eff_radius))

    var ret = Int32(0)
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            comptime block_size = 256

            var n_vps = n_pix * _VP_SAMPLES
            # Sized for the worst case, mirroring sppm.mojo's CPU driver
            # (_sppm_render_core) exactly -- see its comment for why
            var max_bounces_per_photon = min(Int(psc[unsafe_offset=0].max_depth), _photon_depth_cap(sd))
            # A subsurface interior blows this budget wide open: its random-walk
            # steps are deliberately NOT charged to maxdepth (see
            var has_sss_medium = False
            for mi in range(Int(sd.mediumCount)):
                if sd.mediums[unsafe_offset=mi].is_sss != Int32(0):
                    has_sss_medium = True
                    break
            # (The "--sppm + subsurface is unsupported" warning that stood here
            # was WRONG and has been removed. It rested on one experiment --
            if has_sss_medium:
                max_bounces_per_photon += SSS_WALK_ROUNDS
            var max_photons = n_photons_per_pass * max(max_bounces_per_photon, 1)
            var vps_buf     = handle[].ctx.enqueue_create_buffer[DType.uint8](n_vps * size_of[SPPMPixel]())
            var photons_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(max_photons, 1) * size_of[SPPMPhoton]())
            var heads_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](2 * _HSIZE * size_of[Int32]())   # heads | counts
            var inter_cam_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_vps * size_of[Intersection]())
            var inter_ph_buf  = handle[].ctx.enqueue_create_buffer[DType.uint8](max(n_photons_per_pass, 1) * size_of[Intersection]())
            var counter_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](size_of[Int32]())
            var out_buf     = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())
            var caustic_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())
            var albedo_out_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())

            var r2c_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with r2c_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[unsafe_offset=0].raster_to_camera.unsafe_bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]
            var c2w_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with c2w_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[unsafe_offset=0].camera_to_world.unsafe_bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]

            var vps_ptr    = vps_buf.unsafe_ptr().unsafe_bitcast[SPPMPixel]().unsafe_origin_cast[MutUntrackedOrigin]()
            var photons_ptr = photons_buf.unsafe_ptr().unsafe_bitcast[SPPMPhoton]().unsafe_origin_cast[MutUntrackedOrigin]()
            var heads_ptr  = heads_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_origin_cast[MutUntrackedOrigin]()
            var inter_cam_ptr = inter_cam_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_origin_cast[MutUntrackedOrigin]()
            var inter_ph_ptr  = inter_ph_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_origin_cast[MutUntrackedOrigin]()
            var counter_ptr = counter_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_origin_cast[MutUntrackedOrigin]()
            var out_ptr     = out_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin]()
            var caustic_ptr = caustic_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin]()
            var albedo_out_ptr = albedo_out_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin]()
            var r2c_ptr = r2c_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin]()
            var c2w_ptr = c2w_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin]()

            var mediums = handle[].media.mediums_buf.unsafe_ptr().unsafe_bitcast[Medium]()
            var grids_dev = handle[].media.grids_buf.unsafe_ptr().unsafe_bitcast[Grid]()
            var nvdb_grids_dev = handle[].media.nvdb_grids_buf.unsafe_ptr().unsafe_bitcast[NvdbGrid]()
            var n_mediums = Int64(handle[].media.n_mediums)
            var (spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65) = handle[].spectral.unsafe_ptrs()
            handle[].cam_fp = camera_footprint(psc[unsafe_offset=0].raster_to_camera,
                psc[unsafe_offset=0].camera_to_world,
                Int(psc[unsafe_offset=0].film_w), Int(psc[unsafe_offset=0].film_h), _VP_SAMPLES)
            var gsd = handle[].scene_descriptor()

            var grid_pix = ceildiv(n_pix, block_size)
            var grid_vps = ceildiv(n_vps, block_size)
            var grid_hsize = ceildiv(_HSIZE, block_size)

            # Camera/visible-point samples are traced ONCE for the whole
            # render, not per SPPM pass — see _sppm_trace_visible_point's
            var cam_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(0x9E3779B97F4A7C15 + 7)
            handle[].ctx.enqueue_function[sppm_gen_vp_gpu](
                vps_ptr,
                inter_cam_ptr,
                Int64(n_pix),
                Int64(_VP_SAMPLES),
                psc[unsafe_offset=0].film_w,
                r2c_ptr,
                c2w_ptr,
                init_r2,
                cam_seed,
                Int64(psc[unsafe_offset=0].max_depth),
                film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                               psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
                gsd,
                grid_dim=grid_vps,
                block_dim=block_size,
            )

            var prog = Progress(n_passes, "passes", quiet=verbose)
            for pass_idx in range(n_passes):
                handle[].ctx.enqueue_function[sppm_reset_i32_gpu](
                    counter_ptr, grid_dim=1, block_dim=1)

                var pass_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(pass_idx * 2654435761 + 1)
                var grid_emit = ceildiv(max(n_photons_per_pass, 1), block_size)
                handle[].ctx.enqueue_function[sppm_emit_photons_gpu](
                    photons_ptr,
                    Int64(n_photons_per_pass),
                    Int64(max_photons),
                    inter_ph_ptr,
                    counter_ptr,
                    default_emit_med,
                    pass_seed,
                    Int64(pass_idx),
                    Int64(psc[unsafe_offset=0].max_depth),
                    gsd,
                    grid_dim=grid_emit,
                    block_dim=block_size,
                )

                handle[].ctx.synchronize()
                var n_stored_raw: Int32
                with counter_buf.map_to_host() as host_buf:
                    var src = host_buf.unsafe_ptr().unsafe_bitcast[Int32]()
                    n_stored_raw = src[unsafe_offset=0]
                # A silent clamp is how dropped deposits stay invisible: the
                # estimator still divides by the FULL emitted count, so the
                if Int(n_stored_raw) > max_photons:
                    print("Warning: SPPM photon buffer saturated ("
                          + String(n_stored_raw) + " deposits into "
                          + String(max_photons) + " slots) — photons were dropped"
                          + " and this pass is biased dark. Raise --sppm-photons.")
                var n_stored = min(Int(n_stored_raw), max_photons)

                if n_stored > 0:
                    handle[].ctx.enqueue_function[sppm_grid_reset_gpu](
                        heads_ptr, Int64(_HSIZE), grid_dim=grid_hsize, block_dim=block_size)
                    var grid_ins = ceildiv(n_stored, block_size)
                    handle[].ctx.enqueue_function[sppm_grid_count_gpu](
                        photons_ptr, Int64(n_stored), heads_ptr, inv_cell,
                        grid_dim=grid_ins, block_dim=block_size)
                    handle[].ctx.enqueue_function[sppm_grid_insert_gpu](
                        photons_ptr, Int64(n_stored), heads_ptr, inv_cell, Int64(pass_idx),
                        grid_dim=grid_ins, block_dim=block_size)
                    handle[].ctx.enqueue_function[sppm_gather_gpu](
                        vps_ptr,
                        Int64(n_vps),
                        photons_ptr,
                        heads_ptr,
                        inv_cell,
                        gsd,
                        Int64(pass_idx),
                        mediums,
                        n_mediums,
                        grids_dev,
                        nvdb_grids_dev,
                        grid_dim=grid_vps,
                        block_dim=block_size,
                    )

                var nee_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(pass_idx * 0xBF58476D1CE4E5B9 + 3)
                handle[].ctx.enqueue_function[sppm_nee_gpu](
                    vps_ptr,
                    Int64(n_vps),
                    nee_seed,
                    Int64(pass_idx),
                    gsd,
                    grid_dim=grid_vps,
                    block_dim=block_size,
                )

                if verbose:
                    print("SPPM (GPU): pass " + String(pass_idx + 1) + "/" + String(n_passes)
                          + " stored=" + String(n_stored))
                prog.update(pass_idx + 1)

            _ = prog.finish()

            handle[].ctx.enqueue_function[sppm_finalize_gpu](
                vps_ptr, Int64(n_pix), Int64(_VP_SAMPLES), Int32(n_passes), iso_scale, out_ptr, caustic_ptr, albedo_out_ptr,
                spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                grid_dim=grid_pix, block_dim=block_size)
            handle[].ctx.synchronize()
            # --- why-is-this-pixel-black diagnostic -------------------------
            # Four hypotheses about SPPM's remaining black pixels were refuted
            if verbose:
                var n_novp = 0; var n_nophot = 0; var n_dark = 0; var n_tot = 0; var n_envonly = 0; var n_bssrdf = 0; var n_bssrdf_lit = 0; var n_bssrdf_nan = 0
                with vps_buf.map_to_host() as vh:
                    var vp_host = vh.unsafe_ptr().unsafe_bitcast[SPPMPixel]()
                    for pi in range(n_pix):
                        var any_valid = False
                        var any_phot = False
                        var any_light = False
                        for s_i in range(_VP_SAMPLES):
                            var v = vp_host[unsafe_offset=pi * _VP_SAMPLES + s_i]
                            if v.valid != Int32(0):
                                any_valid = True
                                if v.N_acc > Float32(0): any_phot = True
                            if (v.ld.v0 + v.ld.v1 + v.ld.v2 + v.ld.v3) > Float32(1e-12): any_light = True
                            if (v.env.r + v.env.g + v.env.b) > Float32(1e-12): any_light = True
                        n_tot += 1
                        for s_i in range(_VP_SAMPLES):
                            var v2 = vp_host[unsafe_offset=pi * _VP_SAMPLES + s_i]
                            if v2.mat_kind == LobeKind.bssrdf and v2.valid != Int32(0):
                                n_bssrdf += 1
                                var ts = v2.tau.r + v2.tau.g + v2.tau.b
                                if ts > Float32(0): n_bssrdf_lit += 1
                                elif ts != ts: n_bssrdf_nan += 1
                        if not any_valid:
                            # Split the no-VP case: a camera ray that MISSES all
                            # geometry legitimately has no visible point and
                            if not any_light: n_novp += 1
                            else: n_envonly += 1
                        elif not any_phot and not any_light: n_dark += 1
                        elif not any_phot: n_nophot += 1
                var n_ph_surf = 0; var n_ph_vol = 0; var n_ph_bssrdf = 0
                with photons_buf.map_to_host() as ph_h:
                    var ph_host = ph_h.unsafe_ptr().unsafe_bitcast[SPPMPhoton]()
                    var n_scan = min(max_photons, 200000)
                    for k in range(n_scan):
                        var kind = Int(ph_host[unsafe_offset=k].is_volume)
                        if kind == 0: n_ph_surf += 1
                        elif kind == 1: n_ph_vol += 1
                        elif kind == 2: n_ph_bssrdf += 1
                print("SPPM diag photons (last pass, first " + String(min(max_photons,200000))
                      + " slots): surface=" + String(n_ph_surf) + " volume=" + String(n_ph_vol)
                      + " bssrdf=" + String(n_ph_bssrdf))
                print("SPPM diag: " + String(n_tot) + " pixels | DEAD (no VP, no env): " + String(n_novp)
                      + " | no VP but env only: " + String(n_envonly)
                      + " | VP but zero photons AND no light: " + String(n_dark)
                      + " | VP with light but zero photons: " + String(n_nophot)
                      + " || BSSRDF VPs: " + String(n_bssrdf) + " of which tau>0: " + String(n_bssrdf_lit) + " NaN: " + String(n_bssrdf_nan))
            # Keep these device buffers alive (Mojo's ASAP destruction would
            # otherwise free them right after their own last syntactic
            _ = vps_buf^; _ = photons_buf^; _ = heads_buf^
            _ = inter_cam_buf^; _ = inter_ph_buf^; _ = r2c_buf^; _ = c2w_buf^

            var out_pixels = unsafe_alloc[Float32](n_pix * 3)
            with out_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = out_pixels.unsafe_bitcast[UInt8]()
                for i in range(n_pix * 3 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]

            # Global/caustic split -- see _sppm_finalize_one_pixel's
            # docstring and project_water_caustic_sppm_gap memory. Only
            var caustic_pixels = unsafe_alloc[Float32](n_pix * 3)
            with caustic_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = caustic_pixels.unsafe_bitcast[UInt8]()
                for i in range(n_pix * 3 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]

            # Denoise (never wired up before -- no_denoise was a dead
            # parameter): read back the albedo AOV finalized above, run a
            var albedo_pixels = unsafe_alloc[Float32](n_pix * 3)
            with albedo_out_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = albedo_pixels.unsafe_bitcast[UInt8]()
                for i in range(n_pix * 3 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]

            _ = finish_render(psc, sd, out_pixels, albedo_pixels, no_denoise, caustic_pixels)
            out_pixels.unsafe_free(); caustic_pixels.unsafe_free(); albedo_pixels.unsafe_free()
        except e:
            print("SPPM GPU render failed: " + String(e))
            ret = Int32(-1)
    else:
        print("SPPM GPU: no accelerator")
        ret = Int32(-1)
    return ret
