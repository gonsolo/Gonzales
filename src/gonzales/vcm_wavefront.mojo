# VCM GPU wavefront driver.
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
from .vulkaninterop import VulkanInteropRtSceneHandle
from max.gpu.host._nvidia_cuda import CUDA
from .progress import Progress
from .outputs import finish_render
from .spectrum import SpectralSample
from .bdpt_vertex import _BDPT_MAX_DEPTH, _BDPT_MAX_VERTS, BDPTVertex
from .vcm_grid import (
    _VCM_MN_STRIDE, _VCM_HEADS_SIZE, _VCM_CAMIS, _VCM_FOOTPRINT_PIXELS, _vcm_grid_inv_cell, vcm_merge_radius,
)
from .bdpt_camera import VCMCameraPathState
from .bdpt_light import VCMLightPathState
from .bdpt_render import _vcm_finalize_one_pixel
from .vcm_kernels import (
    _bdpt_splat_light_paths_gpu, _bdpt_light_path_init_gpu, _bdpt_light_path_intersect_gpu,
    _bdpt_light_path_bounce_gpu, _bdpt_camera_path_init_gpu, _bdpt_camera_path_intersect_gpu,
    _bdpt_camera_path_bounce_gpu, _bdpt_camera_path_accumulate_gpu, vulkaninterop_rt_traverse_light_paths_gpu,
    vulkaninterop_rt_traverse_camera_paths_gpu, vulkaninterop_rt_traverse_shadow_gpu,
    bdpt_merge_grid_reset_gpu, bdpt_merge_grid_count_gpu, bdpt_merge_grid_insert_gpu, reset_shadow_valid_gpu,
    resolve_shadow_connect_gpu, sum_shadow_connect_gpu,
)

def vcm_render_gpu_wavefront(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneView,
    n_spp:    Int,
    n_photons_req: Int,
    no_denoise: Bool,
    verbose:  Bool,
    # Task #163 stage 4 part 4: when set, both subpaths' per-bounce
    # intersect is routed through the CUDA/Vulkan interop mechanism
    use_vk: Bool = False,
    interop_scene: VulkanInteropRtSceneHandle = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
    interop_rays_buf: Optional[DeviceBuffer[DType.float32]] = None,
    interop_results_buf: Optional[DeviceBuffer[DType.float32]] = None,
    mesh_material_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    mesh_al_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    n_meshes_vk: Int = 0,
    # --rt-hardware (see rtcore_trace_unpack_gpu): alpha scratch sized for every ray the interop scene holds, and the
    # instance decode table for scenes with object instances.
    rt_scratch_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    instance_base_mesh_buf: Optional[DeviceBuffer[DType.uint8]] = None,
) -> Int32:
    """Task #163 stage 4 part 3: wavefront-staged variant of vcm_render_gpu,
    using _bdpt_light_path_init/_intersect/_bounce_gpu and"""
    comptime if _VCM_CAMIS:
        # Deferred by plan deep-hugging-locket: the camera's CAMIS records
        # would have to live in VCMCameraPathState across launches, and the
        print("VCM (GPU wavefront): not supported with _VCM_CAMIS; use --gpu --vcm")
        return Int32(-1)
    var fw = Int(psc[unsafe_offset=0].film_w)
    var fh = Int(psc[unsafe_offset=0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[unsafe_offset=0].film_iso / Float32(100)
    var n_light_paths_merge = max(n_photons_req, n_pix)

    print("VCM (GPU wavefront): " + String(fw) + "x" + String(fh) + "  " + String(n_spp) + " spp  "
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
            var merge_next_buf  = handle[].ctx.enqueue_create_buffer[DType.uint8](max(lvc_cap, 1) * _VCM_MN_STRIDE * size_of[Int32]())
            var inter_light_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(n_light_paths_merge, 1) * size_of[Intersection]())
            var inter_cam_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[Intersection]())
            var light_states_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(n_light_paths_merge, 1) * size_of[VCMLightPathState]())
            var cam_states_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[VCMCameraPathState]())
            # Task #163 stage 5: diffuse-branch connect shadow-ray queue,
            # strided _BDPT_MAX_VERTS slots per pixel -- see
            var shadow_cap = n_pix * _BDPT_MAX_VERTS
            var shadow_rays_buf    = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * 8 * size_of[Float32]())
            var shadow_pending_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * size_of[SpectralSample]())
            var shadow_valid_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * size_of[Int8]())
            var shadow_seg_med_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * size_of[Int32]())
            # One Intersection scratch slot PER THREAD (not per pixel) for
            # resolve_shadow_connect_gpu's _visible_transmittance fallback
            var shadow_scratch_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * size_of[Intersection]())
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

            var lvc_ptr     = lvc_buf.unsafe_ptr().unsafe_bitcast[BDPTVertex]()
            var path_len_ptr = path_len_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var merge_heads_ptr_a = merge_heads_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var merge_heads_ptr_b = merge_heads_buf_prev.unsafe_ptr().unsafe_bitcast[Int32]()
            var merge_next_ptr  = merge_next_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var inter_light_ptr = inter_light_buf.unsafe_ptr().unsafe_bitcast[Intersection]()
            var inter_cam_ptr   = inter_cam_buf.unsafe_ptr().unsafe_bitcast[Intersection]()
            var light_states_ptr = light_states_buf.unsafe_ptr().unsafe_bitcast[VCMLightPathState]()
            var cam_states_ptr   = cam_states_buf.unsafe_ptr().unsafe_bitcast[VCMCameraPathState]()
            var shadow_rays_ptr    = shadow_rays_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var shadow_pending_ptr = shadow_pending_buf.unsafe_ptr().unsafe_bitcast[SpectralSample]()
            var shadow_valid_ptr   = shadow_valid_buf.unsafe_ptr().unsafe_bitcast[Int8]()
            var shadow_seg_med_ptr = shadow_seg_med_buf.unsafe_ptr().unsafe_bitcast[Int32]()
            var shadow_scratch_ptr = shadow_scratch_buf.unsafe_ptr().unsafe_bitcast[Intersection]()
            # Task #163 stage 5 perf fix #3 (2026-07-13): scene-adaptive
            # shadow-ray batching. Investigation (dragon vs cornell-box)
            var shadow_batch_enabled = use_vk
            var accum_ptr   = accum_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var accum_merge_ptr = accum_merge_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var albedo_accum_ptr = albedo_accum_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var r2c_ptr = r2c_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var c2w_ptr = c2w_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            # t=1 light tracing needs the same two camera-projection
            # matrices vcm_render / vcm_render_gpu build (see vcm_render's
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

            handle[].cam_fp = camera_footprint(psc[unsafe_offset=0].raster_to_camera,
                psc[unsafe_offset=0].camera_to_world,
                Int(psc[unsafe_offset=0].film_w), Int(psc[unsafe_offset=0].film_h), n_spp)
            var gsd = handle[].scene_descriptor()

            var grid_light = ceildiv(max(n_light_paths_merge, 1), block_size)
            var grid_pix = ceildiv(n_pix, block_size)
            var grid_hsize = ceildiv(_HSIZE, block_size)

            var (_scene_center, scene_radius) = _scene_bounding_sphere(sd)
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
                var vcm_footprint = _VCM_FOOTPRINT_PIXELS * px_scale * (radius_i / max(vcm_radius_0, Float32(1e-20)))
                var merge_heads_ptr = merge_heads_ptr_a if si % 2 == 0 else merge_heads_ptr_b
                # Variance-aware merge MIS reads the OTHER table: last pass's
                # counts, rescaled from its (larger) cells to this radius.
                var vcm_keep_ptr = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
                var vcm_keep_inv_cell = Float32(0)
                var vcm_keep_scale = Float32(1)
                if si > 0:
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

                # Task #163 stage 4 part 3: wavefront-staged light pass --
                # init once, then intersect+bounce once per depth level,
                var vsd = gsd.with_vcm(vcm_keep_ptr, vcm_keep_inv_cell, vcm_keep_scale, vcm_max_depth, vcm_cam, vcm_footprint, radius_i)
                handle[].ctx.enqueue_function[_bdpt_light_path_init_gpu](
                    light_states_ptr,
                    lvc_ptr,
                    path_len_ptr,
                    mis_vc_weight_factor,
                    Int64(n_light_paths_merge),
                    default_emit_med,
                    pass_seed,
                    Int64(si),
                    gsd,
                    grid_dim=grid_light,
                    block_dim=block_size,
                )

                for _bounce_i in range(_BDPT_MAX_DEPTH):
                    if use_vk:
                        vulkaninterop_rt_traverse_light_paths_gpu(
                            handle[].ctx, light_states_buf, inter_light_buf, interop_scene,
                            interop_rays_buf.value(), interop_results_buf.value(),
                            mesh_material_idx_buf.value(), mesh_al_idx_buf.value(),
                            n_meshes_vk, n_light_paths_merge, sd, rt_scratch_buf, instance_base_mesh_buf)
                    else:
                        handle[].ctx.enqueue_function[_bdpt_light_path_intersect_gpu](
                            gsd,
                            light_states_ptr, inter_light_ptr, Int64(n_light_paths_merge),
                            grid_dim=grid_light, block_dim=block_size)
                    handle[].ctx.enqueue_function[_bdpt_light_path_bounce_gpu](
                        light_states_ptr,
                        inter_light_ptr,
                        lvc_ptr,
                        path_len_ptr,
                        mis_vc_weight_factor,
                        mis_vm_weight_factor,
                        Int64(n_light_paths_merge),
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

                if use_vk and si == 0:
                    var sum_pl = 0
                    with path_len_buf.map_to_host() as host_buf:
                        var pl = host_buf.unsafe_ptr().unsafe_bitcast[Int32]()
                        for pli in range(n_light_paths_merge):
                            sum_pl += Int(pl[unsafe_offset=pli])
                    var occupancy = (Float64(sum_pl) / Float64(n_light_paths_merge)) / Float64(_BDPT_MAX_VERTS)
                    if occupancy < 0.2:
                        shadow_batch_enabled = False
                    if verbose:
                        print("VCM (GPU wavefront): shadow-ray batching " + ("enabled" if shadow_batch_enabled else "disabled")
                              + " (light-path slot occupancy " + String(occupancy * 100) + "%)")

                # Task #163 stage 5 perf fix, ATTEMPTED AND REVERTED
                # (2026-07-13): tried bounding the per-bounce shadow-ray

                # Task #163 stage 4 part 3: wavefront-staged camera pass --
                # same shape as the light pass above. Connect/merge (still
                handle[].ctx.enqueue_function[_bdpt_camera_path_init_gpu](
                    cam_states_ptr, r2c_ptr, c2w_ptr, Int64(n_pix), Int64(Int(psc[unsafe_offset=0].film_w)),
                    px_scale, n_light_paths_f,
                    film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                                   psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
                    base_seed, Int64(si),
                    grid_dim=grid_pix, block_dim=block_size)

                for _bounce_i in range(_BDPT_MAX_DEPTH):
                    if use_vk:
                        vulkaninterop_rt_traverse_camera_paths_gpu(
                            handle[].ctx, cam_states_buf, inter_cam_buf, interop_scene,
                            interop_rays_buf.value(), interop_results_buf.value(),
                            mesh_material_idx_buf.value(), mesh_al_idx_buf.value(),
                            n_meshes_vk, n_pix, sd, rt_scratch_buf, instance_base_mesh_buf)
                    else:
                        handle[].ctx.enqueue_function[_bdpt_camera_path_intersect_gpu](
                            gsd,
                            cam_states_ptr, inter_cam_ptr, Int64(n_pix),
                            grid_dim=grid_pix, block_dim=block_size)
                    if shadow_batch_enabled:
                        handle[].ctx.enqueue_function[reset_shadow_valid_gpu](
                            shadow_valid_ptr, Int64(n_pix), grid_dim=grid_pix, block_dim=block_size)
                    handle[].ctx.enqueue_function[_bdpt_camera_path_bounce_gpu](
                        cam_states_ptr,
                        inter_cam_ptr,
                        lvc_ptr,
                        path_len_ptr,
                        merge_next_ptr,
                        merge_heads_ptr,
                        merge_inv_cell,
                        merge_r2,
                        merge_norm,
                        mis_vc_weight_factor,
                        mis_vm_weight_factor,
                        Int64(n_pix),
                        c2w_ptr,
                        px_scale,
                        vsd,
                        Int8(1) if shadow_batch_enabled else Int8(0),
                        shadow_rays_ptr,
                        shadow_pending_ptr,
                        shadow_valid_ptr,
                        shadow_seg_med_ptr,
                        grid_dim=grid_pix,
                        block_dim=block_size,
                    )

                    # Task #163 stage 5 perf follow-up (2026-07-13): resolve
                    # this bounce's diffuse-branch connect shadow rays in
                    if shadow_batch_enabled:
                        var shadow_grid = ceildiv(shadow_cap, block_size)
                        var instance_base_ptr_vcm = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
                        if instance_base_mesh_buf:
                            instance_base_ptr_vcm = instance_base_mesh_buf.value().unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
                        vulkaninterop_rt_traverse_shadow_gpu(
                            handle[].ctx, shadow_rays_buf, shadow_valid_buf,
                            interop_scene, interop_rays_buf.value(),
                            shadow_cap, interop_results_buf, rt_scratch_buf, sd, n_meshes_vk, instance_base_mesh_buf)
                        handle[].ctx.enqueue_function[resolve_shadow_connect_gpu](
                            interop_results_buf.value().unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                            mesh_material_idx_buf.value().unsafe_ptr().unsafe_bitcast[Int64]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                            Int64(n_meshes_vk),
                            cam_states_ptr,
                            shadow_pending_ptr,
                            shadow_valid_ptr,
                            shadow_seg_med_ptr,
                            shadow_rays_ptr,
                            shadow_scratch_ptr,
                            Int64(shadow_cap),
                            gsd,
                            instance_base_ptr_vcm,
                            grid_dim=shadow_grid,
                            block_dim=block_size,
                        )
                        handle[].ctx.enqueue_function[sum_shadow_connect_gpu](
                            cam_states_ptr, shadow_pending_ptr, shadow_valid_ptr, Int64(n_pix),
                            grid_dim=grid_pix, block_dim=block_size)


                handle[].ctx.enqueue_function[_bdpt_camera_path_accumulate_gpu](
                    cam_states_ptr, accum_ptr, accum_merge_ptr, albedo_accum_ptr, Int64(n_pix),
                    sd.spectral.coeffs, Int64(sd.spectral.res), sd.spectral.cie_x,
                    sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65,
                    grid_dim=grid_pix, block_dim=block_size)

                # Phase 1.5: t=1 light tracing -- the SAME kernel the
                # megakernel driver launches, reading the same fully-built
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
                    # _VCM_CAMIS: this driver refuses to run under it (see
                    # this function's own top-of-body guard) and allocates no
                    Pointer[CamisLightRecord, MutUntrackedOrigin].unsafe_dangling(),
                    grid_dim=grid_light,
                    block_dim=block_size,
                )

                if verbose:
                    print("VCM (GPU wavefront): sample " + String(si + 1) + "/" + String(n_spp))
                # Wait for the pass, so the line reports finished work.
                handle[].ctx.synchronize()
                prog.update(si + 1)

            handle[].ctx.synchronize()
            _ = prog.finish()

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
            print("VCM GPU wavefront render failed: " + String(e))
            ret = Int32(-1)
    else:
        print("VCM GPU wavefront: no accelerator")
        ret = Int32(-1)
    return ret




# ── SPPM GPU driver + kernels (relocated from sppm.mojo) ─────────────────────
# Moved here as a workaround for a reproducible Mojo 1.0.0 / Modular 26.5.0
# compiler bug: every ctx.enqueue_function[kernel] call defined/used from
# ── GPU kernels ───────────────────────────────────────────────────────────────
# Each kernel is a thin wrapper: compute this thread's index, then with the
# host-built SceneView `sd` call the EXACT SAME
