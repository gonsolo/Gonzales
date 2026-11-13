# SPPM GPU driver.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.sys import has_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer
from std.sys.info import size_of
from max.gpu import block_dim
from std.math import max, min, ceildiv, pow
from std.memory.alloc import unsafe_alloc
from gonzales.transport.medium_interface import default_emit_medium
from gonzales.scene.appearance.materials import LobeKind
from gonzales.scene.shapes.primitives import Intersection
from gonzales.scene.appearance.media import Medium, SSS_WALK_ROUNDS, Grid, NvdbGrid
from gonzales.scene.accel.bvh import SceneView
from gonzales.core.sampling import film_filter_of
from gonzales.scene.appearance.footprint import camera_footprint
from gonzales.scene.description.pbrt_parser import ParsedScene_Mojo
from gonzales.integrators.sppm.sppm import SPPMPixel, SPPMPhoton, _vp_samples_for, _VP_SAMPLES, _photon_depth_cap, _ALPHA
from gonzales.estimators.photon_grid import _HSIZE
from gonzales.gpu.scene import GpuSceneHandle
from gonzales.core.progress import Progress
from gonzales.integrators.outputs import finish_render, write_partial_image
from gonzales.gpu.checkpoint import Checkpoint, CheckpointConfig
from gonzales.estimators.irradiance_cache import IrradianceCache
from gonzales.core.geometry import Point3f, Vec3f
from gonzales.core.spectrum import SpectralSample
from gonzales.gpu.sppm.kernels import (
    sppm_jb_points_gpu, sppm_jb_aggregate_gpu, sppm_jb_vp_gpu,
    sppm_reset_i32_gpu, sppm_gen_vp_gpu, sppm_emit_photons_gpu, sppm_grid_reset_gpu, sppm_grid_count_gpu,
    sppm_grid_insert_gpu, sppm_gather_gpu, sppm_nee_gpu, sppm_finalize_gpu, sppm_flush_gpu, sppm_retrace_vp_gpu,
)


def _upload(ctx: DeviceContext, src: Pointer[UInt8, MutUntrackedOrigin], nbytes: Int) raises -> DeviceBuffer[DType.uint8]:
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    with buf.map_to_host() as h:
        var dst = h.unsafe_ptr()
        for i in range(nbytes):
            dst[unsafe_offset=i] = src[unsafe_offset=i]
    return buf^


def _dev[T: AnyType](buf: DeviceBuffer[DType.uint8]) -> Pointer[T, MutUntrackedOrigin]:
    return buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_bitcast[T]().unsafe_origin_cast[MutUntrackedOrigin]()


def _sppm_images(handle: Pointer[GpuSceneHandle, MutUntrackedOrigin], vps: Pointer[SPPMPixel, MutUntrackedOrigin], n_pix: Int,
                 vp_samples: Int, n_done: Int, iso_scale: Float32, out_buf: DeviceBuffer[DType.uint8],
                 caustic_buf: DeviceBuffer[DType.uint8], albedo_buf: DeviceBuffer[DType.uint8]) raises -> Tuple[
                 Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]]:
    """The image after n_done passes: global, caustic and albedo, on the host."""
    comptime block_size = 256
    var (coeffs, res, cie_x, cie_y, cie_z, d65) = handle[].spectral.unsafe_ptrs()
    handle[].ctx.enqueue_function[sppm_finalize_gpu](
        vps, Int64(n_pix), Int64(vp_samples), Int32(n_done), iso_scale,
        out_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin](),
        caustic_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin](),
        albedo_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin](),
        coeffs, Int64(res), cie_x, cie_y, cie_z, d65, grid_dim=ceildiv(n_pix, block_size), block_dim=block_size)
    var o = unsafe_alloc[Float32](n_pix * 3)
    var c = unsafe_alloc[Float32](n_pix * 3)
    var a = unsafe_alloc[Float32](n_pix * 3)
    handle[].ctx.enqueue_copy(o.unsafe_bitcast[UInt8](), out_buf)
    handle[].ctx.enqueue_copy(c.unsafe_bitcast[UInt8](), caustic_buf)
    handle[].ctx.enqueue_copy(a.unsafe_bitcast[UInt8](), albedo_buf)
    handle[].ctx.synchronize()
    return (o, c, a)


def sppm_render_gpu(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneView,
    n_passes: Int,
    n_photons_per_pass: Int,
    initial_radius: Float32,
    no_denoise: Bool,
    verbose:  Bool,
    jb: Bool = False,
    jb_spacing: Float32 = Float32(0),
    jb_near: Float32 = Float32(4),
    ckpt: CheckpointConfig = CheckpointConfig(),
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

    var default_emit_med = default_emit_medium(sd)

    var init_r2 = initial_radius * initial_radius
    var inv_cell = Float32(1.0) / initial_radius
    if verbose:
        print("SPPM: vp radius " + String(initial_radius) + ", grid cell " + String(initial_radius))

    var ret = Int32(0)
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            comptime block_size = 256

            # Subsurface random-walk steps are not charged to maxdepth, so they need their own budget.
            var has_sss_medium = False
            var has_real_medium = False
            for mi in range(Int(sd.mediumCount)):
                if sd.mediums[unsafe_offset=mi].is_sss != Int32(0):
                    has_sss_medium = True
                else:
                    has_real_medium = True
            # Scenes without participating media or curves retrace one visible point per pixel every pass (pbrt's
            # SPPM): camera samples = passes, and the gather touches 1 VP per pixel instead of 32. A skin's walk
            # medium ends its camera path at a surface VP, so it does not count.
            var retrace = not has_real_medium and Int(sd.curveCount) == 0
            var vp_samples = 1 if retrace else _vp_samples_for(n_pix, True)
            var n_vps = n_pix * vp_samples
            print("SPPM (GPU): " + ("one retraced visible point per pixel per pass" if retrace else String(vp_samples) + " visible points per pixel"))
            # Sized for the worst case, as the CPU driver (_sppm_render_core) does.
            var max_bounces_per_photon = min(Int(psc[unsafe_offset=0].max_depth), _photon_depth_cap(sd))
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
            var r2c_ptr = r2c_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin]()
            var c2w_ptr = c2w_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin]()

            var (spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65) = handle[].spectral.unsafe_ptrs()
            handle[].cam_fp = camera_footprint(psc[unsafe_offset=0].raster_to_camera,
                psc[unsafe_offset=0].camera_to_world,
                Int(psc[unsafe_offset=0].film_w), Int(psc[unsafe_offset=0].film_h), _VP_SAMPLES)
            var gsd = handle[].scene_descriptor()

            var grid_vps = ceildiv(n_vps, block_size)
            var grid_hsize = ceildiv(_HSIZE, block_size)

            # Jensen-Buhler: the points and their tree are built on the host, then live on the device.
            var cache = IrradianceCache.build(sd, psc[unsafe_offset=0].prim_ids_cpu, Int(psc[unsafe_offset=0].prim_count_cpu) if jb else 0,
                                              jb_spacing, jb_near, psc[unsafe_offset=0].rng_seed ^ UInt64(0x51ED))
            if jb:
                print("Jensen-Buhler (GPU): " + String(cache.n) + " irradiance points, spacing " + String(cache.spacing)
                      + ", " + String(cache.n_nodes) + " nodes")
            var n_pts = max(cache.n, 1)
            var n_nodes = max(cache.n_nodes, 1)
            var n_mat = max(Int(sd.materialCount), 1)
            var jb_pos = _upload(handle[].ctx, cache.pos.unsafe_bitcast[UInt8](), n_pts * size_of[Point3f]())
            var jb_nrm = _upload(handle[].ctx, cache.nrm.unsafe_bitcast[UInt8](), n_pts * size_of[Vec3f]())
            var jb_mat = _upload(handle[].ctx, cache.mat.unsafe_bitcast[UInt8](), n_pts * size_of[Int32]())
            var jb_area = _upload(handle[].ctx, cache.area.unsafe_bitcast[UInt8](), n_pts * size_of[Float32]())
            var jb_eta = _upload(handle[].ctx, cache.eta.unsafe_bitcast[UInt8](), n_pts * size_of[Float32]())
            var jb_lx = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pts * size_of[SpectralSample]())
            var jb_center = _upload(handle[].ctx, cache.center.unsafe_bitcast[UInt8](), n_nodes * size_of[Point3f]())
            var jb_far2 = _upload(handle[].ctx, cache.far2.unsafe_bitcast[UInt8](), n_nodes * size_of[Float32]())
            var jb_skip = _upload(handle[].ctx, cache.skip.unsafe_bitcast[UInt8](), n_nodes * size_of[Int32]())
            var jb_begin = _upload(handle[].ctx, cache.begin.unsafe_bitcast[UInt8](), n_nodes * size_of[Int32]())
            var jb_end = _upload(handle[].ctx, cache.end.unsafe_bitcast[UInt8](), n_nodes * size_of[Int32]())
            var jb_depth = _upload(handle[].ctx, cache.depth.unsafe_bitcast[UInt8](), n_nodes * size_of[Int32]())
            var jb_sum = handle[].ctx.enqueue_create_buffer[DType.uint8](n_nodes * size_of[SpectralSample]())
            var jb_first = _upload(handle[].ctx, cache.mat_first.unsafe_bitcast[UInt8](), n_mat * size_of[Int32]())
            var jb_last = _upload(handle[].ctx, cache.mat_end.unsafe_bitcast[UInt8](), n_mat * size_of[Int32]())
            var dcache = IrradianceCache(cache.n, _dev[Point3f](jb_pos), _dev[Vec3f](jb_nrm), _dev[Int32](jb_mat), _dev[Float32](jb_area), _dev[Float32](jb_eta),
                                         _dev[SpectralSample](jb_lx), cache.n_nodes, _dev[Point3f](jb_center), _dev[Float32](jb_far2),
                                         _dev[Int32](jb_skip), _dev[Int32](jb_begin), _dev[Int32](jb_end), _dev[Int32](jb_depth), _dev[SpectralSample](jb_sum),
                                         _dev[Int32](jb_first), _dev[Int32](jb_last), cache.spacing, cache.max_depth)
            cache.free()

            # Visible points are traced once per render, not per pass, unless retraced.
            var cam_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(0x9E3779B97F4A7C15 + 7)
            if not retrace:
                handle[].ctx.enqueue_function[sppm_gen_vp_gpu](
                    vps_ptr,
                    inter_cam_ptr,
                    Int64(n_pix),
                    Int64(vp_samples),
                    psc[unsafe_offset=0].film_w,
                    r2c_ptr,
                    c2w_ptr,
                    init_r2,
                    cam_seed,
                    Int64(psc[unsafe_offset=0].max_depth),
                    film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                                   psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
                    gsd,
                    Int32(1) if jb else Int32(0),
                    grid_dim=grid_vps,
                    block_dim=block_size,
                )

            # The visible points carry every pass's progressive state (r2, N_acc, tau, gsum, env, ...), and without
            # retracing their geometry too: they are the whole state.
            var ck = Checkpoint(ckpt, psc, "sppm")
            ck.add("passes", String(n_passes)); ck.add("photons", String(n_photons_per_pass))
            ck.add("radius", String(initial_radius)); ck.add("retrace", String(retrace)); ck.add("vp_samples", String(vp_samples))
            var start = ck.resume()
            if ckpt.resume.byte_length() > 0:
                ck.io("vps", handle[].ctx, vps_buf)
                ck.end_read()
            var prog = Progress(n_passes, "passes", quiet=verbose)
            prog.update(start)
            for pass_idx in range(start, n_passes):
                handle[].ctx.enqueue_function[sppm_reset_i32_gpu](
                    counter_ptr, grid_dim=1, block_dim=1)

                var pass_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(pass_idx * 2654435761 + 1)
                if retrace:
                    handle[].ctx.enqueue_function[sppm_retrace_vp_gpu](
                        vps_ptr, inter_cam_ptr, Int64(n_pix), psc[unsafe_offset=0].film_w,
                        r2c_ptr, c2w_ptr, init_r2, cam_seed ^ UInt64(pass_idx * 0x94D049BB133111EB + 5),
                        Int64(psc[unsafe_offset=0].max_depth),
                        film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                                       psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
                        Int64(pass_idx), gsd, Int32(1) if jb else Int32(0),
                        grid_dim=grid_vps, block_dim=block_size)
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
                    Int32(1) if jb else Int32(0),
                    grid_dim=grid_emit,
                    block_dim=block_size,
                )

                handle[].ctx.synchronize()
                var n_stored_raw: Int32
                with counter_buf.map_to_host() as host_buf:
                    var src = host_buf.unsafe_ptr().unsafe_bitcast[Int32]()
                    n_stored_raw = src[unsafe_offset=0]
                # Dropped deposits still count in the emitted total, so the image darkens: say so.
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
                        grid_dim=grid_vps,
                        block_dim=block_size,
                    )

                if jb and cache.n > 0:
                    if n_stored == 0:   # the points still take their NEE share
                        handle[].ctx.enqueue_function[sppm_grid_reset_gpu](
                            heads_ptr, Int64(_HSIZE), grid_dim=grid_hsize, block_dim=block_size)
                    var r2 = init_r2 * pow(Float32(pass_idx + 1), _ALPHA - Float32(1))
                    handle[].ctx.enqueue_function[sppm_jb_points_gpu](
                        dcache, photons_ptr, heads_ptr, inv_cell, r2, psc[unsafe_offset=0].rng_seed, Int64(pass_idx), gsd,
                        grid_dim=ceildiv(cache.n, block_size), block_dim=block_size)
                    for lv in range(dcache.max_depth, -1, -1):
                        handle[].ctx.enqueue_function[sppm_jb_aggregate_gpu](
                            dcache, Int64(lv), grid_dim=ceildiv(dcache.n_nodes, block_size), block_dim=block_size)
                    handle[].ctx.enqueue_function[sppm_jb_vp_gpu](
                        vps_ptr, Int64(n_vps), dcache, Int64(pass_idx), gsd, grid_dim=grid_vps, block_dim=block_size)

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

                handle[].ctx.enqueue_function[sppm_flush_gpu](
                    vps_ptr, Int64(n_vps), spectral_coeffs, Int64(spectral_res), spectral_cie_x, spectral_cie_y,
                    spectral_cie_z, spectral_d65, grid_dim=grid_vps, block_dim=block_size)

                if verbose:
                    print("SPPM (GPU): pass " + String(pass_idx + 1) + "/" + String(n_passes)
                          + " stored=" + String(n_stored))
                prog.update(pass_idx + 1)
                var done = pass_idx + 1
                var last = done == n_passes
                if ck.enabled() and (last or ck.due() or ck.stop_here(done)):
                    handle[].ctx.synchronize()
                    ck.begin_write(done)
                    ck.io("vps", handle[].ctx, vps_buf)
                    ck.end_write(done)
                    if not last:
                        var partial = _sppm_images(handle, vps_ptr, n_pix, vp_samples, done, iso_scale, out_buf, caustic_buf, albedo_out_buf)
                        _ = write_partial_image(psc, partial[0], partial[1])
                        partial[0].unsafe_free(); partial[1].unsafe_free(); partial[2].unsafe_free()
                        if ck.stop_here(done):
                            print("Stopped after " + String(done) + " of " + String(n_passes) + " passes")
                            return ret

            _ = prog.finish()
            var images = _sppm_images(handle, vps_ptr, n_pix, vp_samples, n_passes, iso_scale, out_buf, caustic_buf, albedo_out_buf)
            # Verbose: count pixels without visible points, photons or light.
            if verbose:
                var n_novp = 0; var n_nophot = 0; var n_dark = 0; var n_tot = 0; var n_envonly = 0
                with vps_buf.map_to_host() as vh:
                    var vp_host = vh.unsafe_ptr().unsafe_bitcast[SPPMPixel]()
                    for pi in range(n_pix):
                        var any_valid = False
                        var any_phot = False
                        var any_light = False
                        for s_i in range(vp_samples):
                            var v = vp_host[unsafe_offset=pi * vp_samples + s_i]
                            if v.valid != Int32(0):
                                any_valid = True
                                if v.N_acc > Float32(0): any_phot = True
                            if (v.ld.v0 + v.ld.v1 + v.ld.v2 + v.ld.v3) > Float32(1e-12): any_light = True
                            if (v.env.r + v.env.g + v.env.b) > Float32(1e-12): any_light = True
                        n_tot += 1
                        for s_i in range(vp_samples):
                            var v2 = vp_host[unsafe_offset=pi * vp_samples + s_i]
                        if not any_valid:
                            # A ray that misses everything has no visible point but may see the env.
                            if not any_light: n_novp += 1
                            else: n_envonly += 1
                        elif not any_phot and not any_light: n_dark += 1
                        elif not any_phot: n_nophot += 1
                var n_ph_surf = 0; var n_ph_vol = 0
                with photons_buf.map_to_host() as ph_h:
                    var ph_host = ph_h.unsafe_ptr().unsafe_bitcast[SPPMPhoton]()
                    var n_scan = min(max_photons, 200000)
                    for k in range(n_scan):
                        var kind = Int(ph_host[unsafe_offset=k].is_volume)
                        if kind == 0: n_ph_surf += 1
                        elif kind == 1: n_ph_vol += 1
                print("SPPM diag photons (last pass, first " + String(min(max_photons,200000))
                      + " slots): surface=" + String(n_ph_surf) + " volume=" + String(n_ph_vol))
                print("SPPM diag: " + String(n_tot) + " pixels | DEAD (no VP, no env): " + String(n_novp)
                      + " | no VP but env only: " + String(n_envonly)
                      + " | VP but zero photons AND no light: " + String(n_dark)
                      + " | VP with light but zero photons: " + String(n_nophot))
            # Keep the device buffers alive until the kernels that use them are done.
            _ = vps_buf^; _ = photons_buf^; _ = heads_buf^
            _ = inter_cam_buf^; _ = inter_ph_buf^; _ = r2c_buf^; _ = c2w_buf^
            _ = jb_pos^; _ = jb_nrm^; _ = jb_mat^; _ = jb_area^; _ = jb_eta^; _ = jb_lx^; _ = jb_center^; _ = jb_far2^
            _ = jb_skip^; _ = jb_begin^; _ = jb_end^; _ = jb_depth^; _ = jb_sum^; _ = jb_first^; _ = jb_last^

            var out_pixels = images[0]; var caustic_pixels = images[1]; var albedo_pixels = images[2]
            _ = finish_render(psc, sd, out_pixels, albedo_pixels, no_denoise, caustic_pixels)
            out_pixels.unsafe_free(); caustic_pixels.unsafe_free(); albedo_pixels.unsafe_free()
        except e:
            print("SPPM GPU render failed: " + String(e))
            ret = Int32(-1)
    else:
        print("SPPM GPU: no accelerator")
        ret = Int32(-1)
    return ret
