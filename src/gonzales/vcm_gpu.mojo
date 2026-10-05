# VCM GPU megakernel driver.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.sys import has_accelerator
from std.sys.info import size_of
from max.gpu import block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import tan, max, abs, ceildiv
from std.memory.alloc import unsafe_alloc
from .geometry import RGB, PI
from .primitives import Intersection
from .vcm_camis import CamisLightRecord
from .bvh import SceneView, _scene_bounding_sphere
from .sampling import film_filter_of
from .footprint import camera_footprint
from .transform import matrix_invert
from .pbrt_parser import ParsedScene_Mojo
from .sppm import _HSIZE
from .gpu_scene import GpuSceneHandle
from max.gpu.host._nvidia_cuda import CUDA
from .progress import Progress
from .outputs import finish_render, _sidecar_name, _write_channels
from .bdpt_vertex import _BDPT_MAX_VERTS, BDPTVertex
from .vcm_grid import (
    _VCM_MN_STRIDE, _VCM_HEADS_SIZE, _CAMIS_LVC_PER_SLOT, _VCM_FOOTPRINT_PIXELS, _vcm_grid_inv_cell,
    _camera_typical_distance, vcm_merge_radius,
)
from .bdpt_render import _vcm_finalize_one_pixel
from .vcm_kernels import (
    _bdpt_emit_light_paths_gpu, _bdpt_splat_light_paths_gpu, _bdpt_camera_connect_gpu,
    bdpt_merge_grid_reset_gpu, bdpt_merge_grid_count_gpu, bdpt_merge_grid_insert_gpu,
    _VCM_BUDGET_REDUCE_THREADS, vcm_budget_reduce_gpu,
)

def vcm_render_gpu(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneView,
    n_spp:    Int,
    n_photons_req: Int,
    no_denoise: Bool,
    verbose:  Bool,
    vcm_budget: Bool = False,
    vcm_cap: Int32 = Int32(0),
    vcm_no_keep_mis: Bool = False,
    vcm_radius_from_camera: Bool = False,
    # EXPERIMENTAL, only read when vcm_radius_from_camera: percentile of the
    # camera-distance sample (0.5 = median) and a multiplier on top of the
    vcm_radius_cam_percentile: Float32 = Float32(0.5),
    vcm_radius_cam_fraction_mult: Float32 = Float32(1.0),
    # Reproduces the "naive VCM" baseline (one fixed global radius, no
    # per-vertex footprint scaling) -- see the site in the sample loop below.
    vcm_no_footprint: Bool = False,
    # --vcm-radius-scale: multiplies the whole merge radius consistently
    # (ceiling, hash-grid cell, MIS density and footprint radius).
    vcm_radius_scale: Float32 = Float32(1.0),
) -> Int32:
    """GPU-accelerated Light Vertex Cache BDPT — same algorithm as
    vcm_render (CPU), same shared _bdpt_trace_light_path/"""
    var fw = Int(psc[unsafe_offset=0].film_w)
    var fh = Int(psc[unsafe_offset=0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[unsafe_offset=0].film_iso / Float32(100)
    var n_light_paths_merge = max(n_photons_req, n_pix)

    print("VCM (GPU): " + String(fw) + "x" + String(fh) + "  " + String(n_spp) + " spp  "
          + String(n_light_paths_merge) + " light paths/pass")

    var has_med = Int(sd.mediumCount) > 0
    var default_emit_med = Int32(-1)
    if has_med and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[unsafe_offset=mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    var lvc_cap = n_light_paths_merge * _BDPT_MAX_VERTS
    var base_seed = psc[unsafe_offset=0].rng_seed

    var ret = Int32(0)
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            comptime block_size = 256

            var lvc_buf     = handle[].ctx.enqueue_create_buffer[DType.uint8](max(lvc_cap, 1) * size_of[BDPTVertex]())
            var path_len_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(n_light_paths_merge, 1) * size_of[Int32]())
            # VCM vertex merging (Stage 1) grid buffers — see vcm_render's
            # matching CPU allocation for the merge_r2/merge_norm derivation.
            var merge_heads_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](_VCM_HEADS_SIZE * size_of[Int32]())   # heads | counts | fine levels
            # The previous pass's table, kept intact for _vcm_keep; the two swap each pass.
            var merge_heads_buf_prev = handle[].ctx.enqueue_create_buffer[DType.uint8](_VCM_HEADS_SIZE * size_of[Int32]())
            # --vcm-budget: per-bucket [Q, V] merge statistics, this pass's and
            # the previous one's (swapped like the heads), and the reduction
            # [sum sqrt(V Q n), sum Q min(n, cap)] that sets lambda.
            var stat_n = 2 * _HSIZE if vcm_budget else 2
            var stat_buf_a = handle[].ctx.enqueue_create_buffer[DType.float32](stat_n)
            var stat_buf_b = handle[].ctx.enqueue_create_buffer[DType.float32](stat_n)
            var budget_red_buf = handle[].ctx.enqueue_create_buffer[DType.float32](2)
            var stat_ptr_a = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(stat_buf_a.unsafe_ptr()))
            var stat_ptr_b = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(stat_buf_b.unsafe_ptr()))
            var budget_red_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(budget_red_buf.unsafe_ptr()))
            var vcm_lambda = Float32(0)
            var merge_next_buf  = handle[].ctx.enqueue_create_buffer[DType.uint8](max(lvc_cap, 1) * _VCM_MN_STRIDE * size_of[Int32]())
            # CAMIS light records, slot for slot with lvc_buf -- a single
            # dummy element when the hybrid is compiled out (see _VCM_CAMIS).
            var lvc_camis_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](
                max(lvc_cap * _CAMIS_LVC_PER_SLOT, 1) * size_of[CamisLightRecord]())
            var inter_light_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(n_light_paths_merge, 1) * size_of[Intersection]())
            var inter_cam_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[Intersection]())
            var accum_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())
            with accum_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                for i in range(n_pix * 3):
                    dst[unsafe_offset=i] = Float32(0)
            # Vertex-merging contribution, split from `accum` (connect + t=1
            # splats) -- see VCMCameraPathState.total_merge's docstring.
            var accum_merge_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())
            with accum_merge_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                for i in range(n_pix * 3):
                    dst[unsafe_offset=i] = Float32(0)
            var albedo_accum_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())
            with albedo_accum_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                for i in range(n_pix * 3):
                    dst[unsafe_offset=i] = Float32(0)
            # Benchmark instrumentation only -- see
            # _bdpt_merge_from_cache's docstring paragraph. Per-pixel
            var visit_accum_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())
            with visit_accum_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                for i in range(n_pix * 3):
                    dst[unsafe_offset=i] = Float32(0)

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

            # t=1 light tracing needs the same two camera-projection
            # matrices vcm_render builds on the CPU (see its matching
            var w2c_host = unsafe_alloc[Float32](16)
            _ = matrix_invert(psc[unsafe_offset=0].camera_to_world, w2c_host)
            var _r2c_h = psc[unsafe_offset=0].raster_to_camera
            var a0 = _r2c_h[unsafe_offset=0]; var a1 = _r2c_h[unsafe_offset=4]; var a2 = _r2c_h[unsafe_offset=12]
            var b0 = _r2c_h[unsafe_offset=1]; var b1 = _r2c_h[unsafe_offset=5]; var b2 = _r2c_h[unsafe_offset=13]
            var g0 = _r2c_h[unsafe_offset=2]; var g1 = _r2c_h[unsafe_offset=6]; var g2 = _r2c_h[unsafe_offset=14]
            var d0 = b1*g2 - b2*g1
            var d1 = b0*g2 - b2*g0
            var d2 = b0*g1 - b1*g0
            var det = a0*d0 - a1*d1 + a2*d2
            var idet = Float32(1) / det if abs(det) > Float32(1e-20) else Float32(0)
            var c2r_host = unsafe_alloc[Float32](9)
            c2r_host[unsafe_offset=0] =  d0*idet; c2r_host[unsafe_offset=1] = -(a1*g2 - a2*g1)*idet; c2r_host[unsafe_offset=2] =  (a1*b2 - a2*b1)*idet
            c2r_host[unsafe_offset=3] = -d1*idet; c2r_host[unsafe_offset=4] =  (a0*g2 - a2*g0)*idet; c2r_host[unsafe_offset=5] = -(a0*b2 - a2*b0)*idet
            c2r_host[unsafe_offset=6] =  d2*idet; c2r_host[unsafe_offset=7] = -(a0*g1 - a1*g0)*idet; c2r_host[unsafe_offset=8] =  (a0*b1 - a1*b0)*idet
            var w2c_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with w2c_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = w2c_host.unsafe_bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]
            var c2r_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](9 * size_of[Float32]())
            with c2r_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = c2r_host.unsafe_bitcast[UInt8]()
                for i in range(9 * size_of[Float32]()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]
            w2c_host.unsafe_free()
            c2r_host.unsafe_free()
            var w2c_ptr = w2c_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var c2r_ptr = c2r_buf.unsafe_ptr().unsafe_bitcast[Float32]()

            var lvc_ptr     = lvc_buf.unsafe_ptr().unsafe_bitcast[BDPTVertex]()
            var path_len_ptr = path_len_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var merge_heads_ptr_a = merge_heads_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var merge_heads_ptr_b = merge_heads_buf_prev.unsafe_ptr().unsafe_bitcast[Int32]()
            var merge_next_ptr  = merge_next_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var lvc_camis_ptr   = lvc_camis_buf.unsafe_ptr().unsafe_bitcast[CamisLightRecord]()
            var inter_light_ptr = inter_light_buf.unsafe_ptr().unsafe_bitcast[Intersection]()
            var inter_cam_ptr   = inter_cam_buf.unsafe_ptr().unsafe_bitcast[Intersection]()
            var accum_ptr   = accum_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var accum_merge_ptr = accum_merge_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var albedo_accum_ptr = albedo_accum_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var visit_accum_ptr = visit_accum_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var r2c_ptr = r2c_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var c2w_ptr = c2w_buf.unsafe_ptr().unsafe_bitcast[Float32]()

            handle[].cam_fp = camera_footprint(psc[unsafe_offset=0].raster_to_camera,
                psc[unsafe_offset=0].camera_to_world,
                Int(psc[unsafe_offset=0].film_w), Int(psc[unsafe_offset=0].film_h), n_spp)
            var gsd = handle[].scene_descriptor()

            var grid_light = ceildiv(max(n_light_paths_merge, 1), block_size)
            var grid_pix = ceildiv(n_pix, block_size)
            var grid_hsize = ceildiv(_HSIZE, block_size)

            var (_scene_center, scene_radius_bbox) = _scene_bounding_sphere(sd)
            var scene_radius = scene_radius_bbox
            if vcm_radius_from_camera:
                # EXPERIMENTAL: see _camera_typical_distance's docstring. Falls
                # back to the bounding-sphere radius when nothing is hit
                # (e.g. camera facing only background/infinite lights).
                var cam_dist = _camera_typical_distance(psc[unsafe_offset=0].raster_to_camera,
                    psc[unsafe_offset=0].camera_to_world, Int(psc[unsafe_offset=0].film_w), Int(psc[unsafe_offset=0].film_h), sd,
                    vcm_radius_cam_percentile)
                if cam_dist > Float32(0):
                    scene_radius = cam_dist * vcm_radius_cam_fraction_mult
                if verbose:
                    print("vcm-radius-from-camera: bbox scene_radius=" + String(scene_radius_bbox)
                        + " cam_dist(p=" + String(vcm_radius_cam_percentile) + ")=" + String(cam_dist)
                        + " fraction_mult=" + String(vcm_radius_cam_fraction_mult)
                        + " -> effective scene_radius=" + String(scene_radius))
            scene_radius *= vcm_radius_scale
            var px_scale = Float32(2.0) * tan(psc[unsafe_offset=0].camera_fov * Float32(3.14159265 / 360.0)) / Float32(fh)
            var vcm_max_depth = psc[unsafe_offset=0].max_depth   # clamped in _vcm_depth
            var c2w_h = psc[unsafe_offset=0].camera_to_world
            var vcm_cam = SIMD[DType.float32, 4](c2w_h[unsafe_offset=12], c2w_h[unsafe_offset=13], c2w_h[unsafe_offset=14], Float32(0))
            var vcm_radius_0 = vcm_merge_radius(scene_radius, 0)
            var n_light_paths_f = Float32(n_light_paths_merge)

            var grid_merge_ins = ceildiv(max(lvc_cap, 1), block_size)

            var prog = Progress(n_spp, "spp", quiet=verbose)
            for si in range(n_spp):
                # Stage 2c progressive radius -- see vcm_render (CPU)'s
                # matching per-sample loop for the full derivation comment.
                var radius_i = vcm_merge_radius(scene_radius, si)
                # Footprint radius shrinks on the same schedule as the global one.
                # --vcm-no-footprint (the "naive VCM" baseline): 0 makes
                # _vcm_merge_radius_at fall back to the old fixed global radius.
                var vcm_footprint = Float32(0) if vcm_no_footprint else _VCM_FOOTPRINT_PIXELS * vcm_radius_scale * px_scale * (radius_i / max(vcm_radius_0, Float32(1e-20)))
                var merge_heads_ptr = merge_heads_ptr_a if si % 2 == 0 else merge_heads_ptr_b
                # Variance-aware merge MIS reads the OTHER table: last pass's
                # counts, rescaled from its (larger) cells to this radius.
                var vcm_keep_ptr = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
                var vcm_keep_inv_cell = Float32(0)
                var vcm_keep_scale = Float32(1)
                # --vcm-no-keep-mis: the insert still thins and the gather still
                # weights 1/k, but merging's MIS is shown no keep table, so it
                if si > 0 and not vcm_no_keep_mis:
                    var radius_prev = vcm_merge_radius(scene_radius, si - 1)
                    var prev_tab = merge_heads_ptr_b if si % 2 == 0 else merge_heads_ptr_a
                    vcm_keep_ptr = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(prev_tab) + _HSIZE * size_of[Int32]())
                    vcm_keep_inv_cell = Float32(1.0) / max(radius_prev, Float32(1e-6))
                    vcm_keep_scale = (radius_i / radius_prev) * (radius_i / radius_prev)
                var merge_r2 = radius_i * radius_i
                var merge_inv_cell = _vcm_grid_inv_cell(sd, radius_i)
                var merge_norm = Float32(1.0) / (Float32(n_light_paths_merge) * PI * max(merge_r2, Float32(1e-12)))
                var eta_vcm = PI * max(merge_r2, Float32(1e-12)) * Float32(n_light_paths_merge)
                var mis_vm_weight_factor = eta_vcm
                var mis_vc_weight_factor = Float32(1.0) / eta_vcm

                var pass_seed = base_seed ^ UInt64(si * 2654435761 + 1)
                var vsd = gsd.with_vcm(vcm_keep_ptr, vcm_keep_inv_cell, vcm_keep_scale, vcm_max_depth, vcm_cam, vcm_footprint, radius_i).with_vcm_cap(vcm_cap)
                if vcm_budget:
                    # This pass accumulates into one table and reads the other,
                    # whose lambda the previous pass's reduction set (0 on the
                    # first pass: the fixed cap).
                    if si % 2 == 0:
                        stat_buf_a.enqueue_fill(Float32(0))
                    else:
                        stat_buf_b.enqueue_fill(Float32(0))
                    var stat_out = stat_ptr_a if si % 2 == 0 else stat_ptr_b
                    var stat_in = stat_ptr_b if si % 2 == 0 else stat_ptr_a
                    vsd = vsd.with_vcm_budget(
                        stat_in if si > 0 else Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
                        stat_out, vcm_lambda)
                handle[].ctx.enqueue_function[_bdpt_emit_light_paths_gpu](
                    lvc_ptr,
                    path_len_ptr,
                    lvc_camis_ptr,
                    mis_vc_weight_factor,
                    mis_vm_weight_factor,
                    inter_light_ptr,
                    Int64(n_light_paths_merge),
                    default_emit_med,
                    pass_seed,
                    Int64(si),
                    c2w_ptr,
                    px_scale,
                    vsd,
                    grid_dim=grid_light,
                    block_dim=block_size,
                )

                # VCM Stage 2b: light paths are deterministically paired with
                # pixels (the first n_pix of n_light_paths_merge total light
                handle[].ctx.enqueue_function[bdpt_merge_grid_reset_gpu](
                    merge_heads_ptr, Int64(_HSIZE), grid_dim=grid_hsize, block_dim=block_size)
                handle[].ctx.enqueue_function[bdpt_merge_grid_count_gpu](
                    lvc_ptr, path_len_ptr, Int64(lvc_cap), merge_heads_ptr, merge_inv_cell,
                    grid_dim=grid_merge_ins, block_dim=block_size)
                handle[].ctx.enqueue_function[bdpt_merge_grid_insert_gpu](
                    lvc_ptr, path_len_ptr, Int64(lvc_cap), merge_next_ptr, merge_heads_ptr, merge_inv_cell, vsd,
                    grid_dim=grid_merge_ins, block_dim=block_size)


                handle[].ctx.enqueue_function[_bdpt_camera_connect_gpu](
                    accum_ptr,
                    accum_merge_ptr,
                    albedo_accum_ptr,
                    visit_accum_ptr,
                    Int64(n_pix),
                    Int64(Int(psc[unsafe_offset=0].film_w)),
                    r2c_ptr,
                    c2w_ptr,
                    inter_cam_ptr,
                    lvc_ptr,
                    path_len_ptr,
                    lvc_camis_ptr,
                    merge_next_ptr,
                    merge_heads_ptr,
                    merge_inv_cell,
                    merge_r2,
                    merge_norm,
                    px_scale,
                    mis_vc_weight_factor,
                    mis_vm_weight_factor,
                    n_light_paths_f,
                    film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                                   psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
                    base_seed,
                    Int64(si),
                    vsd,
                    grid_dim=grid_pix,
                    block_dim=block_size,
                )

                # Phase 1.5: t=1 light tracing, the GPU counterpart of
                # vcm_render's splat pass. Same stream, launched AFTER the
                handle[].ctx.enqueue_function[_bdpt_splat_light_paths_gpu](
                    accum_ptr,
                    lvc_ptr,
                    path_len_ptr,
                    Int64(n_light_paths_merge),
                    inter_light_ptr,
                    w2c_ptr,
                    c2r_ptr,
                    c2w_ptr,
                    Int64(fw),
                    Int64(fh),
                    px_scale,
                    mis_vm_weight_factor,
                    film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                                   psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
                    vsd,
                    lvc_camis_ptr,
                    grid_dim=grid_light,
                    block_dim=block_size,
                )

                if vcm_budget:
                    # lambda for the next pass: the budget is the merge work
                    # the fixed cap would have spent, sum Q min(n, cap).
                    budget_red_buf.enqueue_fill(Float32(0))
                    handle[].ctx.enqueue_function[vcm_budget_reduce_gpu](
                        vsd.vcmStatOut, merge_heads_ptr, budget_red_ptr,
                        grid_dim=_VCM_BUDGET_REDUCE_THREADS // block_size, block_dim=block_size)
                    with budget_red_buf.map_to_host() as red:
                        var s_sum = red.unsafe_ptr()[unsafe_offset=0]
                        var b_sum = red.unsafe_ptr()[unsafe_offset=1]
                        vcm_lambda = b_sum / s_sum if s_sum > Float32(0) else Float32(0)
                    if verbose:
                        print("VCM (GPU): budget lambda " + String(vcm_lambda))

                if verbose:
                    print("VCM (GPU): sample " + String(si + 1) + "/" + String(n_spp))
                # Wait for the pass, so the line reports finished work.
                handle[].ctx.synchronize()
                prog.update(si + 1)

            handle[].ctx.synchronize()
            _ = prog.finish()


            # Benchmark instrumentation only -- see
            # _bdpt_merge_from_cache's docstring paragraph. Written
            var visit_sidecar = _sidecar_name(psc[unsafe_offset=0].film_filename, ".mergevisits.exr")
            with visit_accum_buf.map_to_host() as vhost:
                var vsrc = vhost.unsafe_ptr().unsafe_bitcast[Float32]()
                _ = _write_channels(visit_sidecar, vsrc, Int32(fw), Int32(fh), Int32(3),
                    "naive.Y,footprint.Y,thinning.Y", Int32(fw), Int32(fh), Int32(0), Int32(0))
            visit_sidecar.unsafe_free()
            # `pixels` (connect + t=1 splats) and `caustic_pixels` (vertex
            # merging) split, same reason/contract as _bdpt_render_core's
            var pixels = unsafe_alloc[Float32](n_pix * 3)
            var caustic_pixels = unsafe_alloc[Float32](n_pix * 3)
            with accum_buf.map_to_host() as host_buf:
                with accum_merge_buf.map_to_host() as host_buf_m:
                    var src = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                    var src_m = host_buf_m.unsafe_ptr().unsafe_bitcast[Float32]()
                    var inv_spp = iso_scale / Float32(n_spp)
                    for i in range(n_pix):
                        var conn = RGB(src[unsafe_offset=i*3], src[unsafe_offset=i*3+1], src[unsafe_offset=i*3+2])
                        var mrg = RGB(src_m[unsafe_offset=i*3], src_m[unsafe_offset=i*3+1], src_m[unsafe_offset=i*3+2])
                        var (c, m) = _vcm_finalize_one_pixel(conn, mrg, inv_spp)
                        pixels[unsafe_offset=i*3] = c.r; pixels[unsafe_offset=i*3+1] = c.g; pixels[unsafe_offset=i*3+2] = c.b
                        caustic_pixels[unsafe_offset=i*3] = m.r; caustic_pixels[unsafe_offset=i*3+1] = m.g; caustic_pixels[unsafe_offset=i*3+2] = m.b

            # Denoise (never wired up before -- no_denoise was a dead
            # parameter): read back the albedo AOV accumulated above, run
            var albedo_pixels = unsafe_alloc[Float32](n_pix * 3)
            with albedo_accum_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                var inv_spp_alb = Float32(1) / Float32(n_spp)
                for i in range(n_pix * 3):
                    albedo_pixels[unsafe_offset=i] = src[unsafe_offset=i] * inv_spp_alb

            _ = finish_render(psc, sd, pixels, albedo_pixels, no_denoise, caustic_pixels)
            pixels.unsafe_free(); caustic_pixels.unsafe_free(); albedo_pixels.unsafe_free()
        except e:
            print("VCM GPU render failed: " + String(e))
            ret = Int32(-1)
    else:
        print("VCM GPU: no accelerator")
        ret = Int32(-1)
    return ret

# ── Task #163 stage 5: Vulkan-RT-batched shadow ray resolution ──────────────
# Resolves the diffuse-branch connect shadow rays queued by
# _bdpt_connect_to_cache_deferred: after a batched Vulkan RT dispatch fills
