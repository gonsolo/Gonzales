# VCM GPU wavefront driver.
# Part of the BDPT/VCM machinery that used to be one file (bdpt.mojo).

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
    # (vulkaninterop_rt_traverse_light_paths_gpu/_camera_) instead of
    # traverse_bvh2_core/test_spheres -- see that section's own comment.
    # interop_scene must be sized for max(n_light_paths_merge, n_pix) rays
    # (n_light_paths_merge is always >= n_pix by construction, see
    # n_photons_req's docstring below) since it's reused sequentially for
    # both the light pass and the camera pass every sample.
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
    using _bdpt_light_path_init/_intersect/_bounce_gpu and
    _bdpt_camera_path_init/_intersect/_bounce_gpu instead of the single
    _bdpt_emit_light_paths_gpu/_bdpt_camera_connect_gpu mega-kernels --
    same algorithm, same LVC-then-connect ordering (light pass runs to
    full completion before the camera pass starts, since camera vertices
    connect/merge against the COMPLETE light-path cache, not a
    partially-built one), same output. The only behavioral difference is
    control-flow SHAPE: many small kernel launches (one intersect+bounce
    pair per _BDPT_MAX_DEPTH depth level, per subpath) instead of two
    single-kernel-traces-a-whole-subpath launches. Task #163 stage 4 part
    4: `use_vk=True` routes that per-bounce intersect through the CUDA/
    Vulkan interop mechanism (real hardware ray-query tracing, stage-1/2/3)
    instead of `traverse_bvh2_core`/`test_spheres` -- see this file's
    `vulkaninterop_rt_traverse_light_paths_gpu`/`_camera_` and their own
    section comment. `use_vk=False` (default) keeps the software-BVH path,
    still useful as a same-algorithm baseline for A/B comparison. Shadow
    rays (NEE/connect/merge/MNEE) stay on software BVH in BOTH modes, per
    the user-confirmed stage 4 scope -- only the primary/bounce ray is
    ever Vulkan-RT-traced. See vcm_render_gpu's own docstring for the
    shared algorithm-level documentation (n_photons_req/n_light_paths_merge
    derivation etc.), not repeated here."""
    comptime if _VCM_CAMIS:
        # Deferred by plan deep-hugging-locket: the camera's CAMIS records
        # would have to live in VCMCameraPathState across launches, and the
        # light side's pending scatter record in VCMLightPathState. Refused
        # at run time rather than by a `comptime assert`, which would fire
        # for every _VCM_CAMIS build -- pipeline.mojo reaches this function
        # behind a runtime flag, so it is always compiled.
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
            # _bdpt_connect_to_cache_deferred/resolve_shadow_connect_gpu.
            # Only allocated/used when use_vk (software-BVH _connect stays
            # the only path otherwise).
            var shadow_cap = n_pix * _BDPT_MAX_VERTS
            var shadow_rays_buf    = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * 8 * size_of[Float32]())
            var shadow_pending_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * size_of[SpectralSample]())
            var shadow_valid_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * size_of[Int8]())
            var shadow_seg_med_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(shadow_cap, 1) * size_of[Int32]())
            # One Intersection scratch slot PER THREAD (not per pixel) for
            # resolve_shadow_connect_gpu's _visible_transmittance fallback
            # call -- inter_light_buf (sized n_light_paths_merge) is too
            # small now that resolve dispatches n_pix*_BDPT_MAX_VERTS
            # threads in one go (2026-07-13 perf follow-up).
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
            # found the one-shot n_pix*_BDPT_MAX_VERTS dispatch (perf fix
            # #2, commit cb25f83) only pays off when light paths are long
            # enough to fill a real fraction of those _BDPT_MAX_VERTS
            # slots -- cornell-box's mean path_len is 5.18/10 (~52%
            # occupancy, a real win at every tested scale); dragon's is
            # 0.51/10 (~5% occupancy -- the dispatch is ~95% wasted
            # degenerate-ray padding, and got WORSE than software, even
            # worse than the earlier primary-ray-only swap). Decided ONCE
            # per render (not per sample -- a per-sample host sync was
            # already tried and found catastrophic, see perf fix #1's
            # revert note below) via a single path_len_buf readback after
            # si=0's light pass, before its camera pass starts. The 20%-
            # occupancy threshold is a judgment call from exactly 2 data
            # points (cornell-box/dragon), not a rigorously derived
            # constant -- revisit if a 3rd scene disagrees.
            var shadow_batch_enabled = use_vk
            var accum_ptr   = accum_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var accum_merge_ptr = accum_merge_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var albedo_accum_ptr = albedo_accum_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var r2c_ptr = r2c_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var c2w_ptr = c2w_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            # t=1 light tracing needs the same two camera-projection
            # matrices vcm_render / vcm_render_gpu build (see vcm_render's
            # comment): w2c = inverse(cameraToWorld), and c2r inverting the
            # 3x3 that turns (filmX, filmY, 1) into a camera-space direction.
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
                # instead of _bdpt_emit_light_paths_gpu's single mega-kernel.
                # Every kernel internally skips lanes whose state has already
                # gone inactive (mirrors gpu.mojo's traverse_paths_gpu/
                # shade_*_gpu `if paths[tid].active == 0: return` convention)
                # -- running the full _BDPT_MAX_DEPTH iterations regardless
                # of how many lanes are still active is the same fixed-
                # iteration-count wavefront shape the plain path tracer uses.
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
                # paths, task #152), so no host readback of a total vertex
                # count is needed anymore -- lvc_cap is already known at
                # compile/host time. Kernels stay ordered on one stream
                # without an explicit synchronize() here.
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
                # local-slot loop below by this sample's actual max light-
                # path length (one path_len_buf.map_to_host() readback per
                # SAMPLE). Correct in principle, but made things WORSE, not
                # better -- 256x256/16spp went from 6.4s to 34.9s. Root
                # cause: this whole per-sample loop is normally fully async/
                # pipelined across all `si` iterations with zero host syncs;
                # a single per-sample sync point breaks that overlap and
                # forces the CPU to idle-wait on GPU completion every
                # sample, which cost far more than the dispatches it saved.
                # Also, VCM's bounce loop doesn't respect the scene's own
                # `maxdepth` (grep confirms bdpt.mojo never reads
                # psc[0].max_depth) -- cornell-box's light paths routinely
                # reach the full _BDPT_MAX_VERTS(10) cap via RR-only
                # termination regardless of its maxdepth=4 setting, so even
                # a static maxdepth-based bound would be unsafe (could drop
                # real connect candidates) as well as ineffective (wouldn't
                # actually reduce shadow_locals below 10 for this scene).
                # See project_vulkan_rt_backend memory for the full story.

                # Task #163 stage 4 part 3: wavefront-staged camera pass --
                # same shape as the light pass above. Connect/merge (still
                # software-BVH shadow rays, per the user-confirmed stage 4
                # scope) run inline inside _bdpt_camera_path_bounce_gpu at
                # every non-delta vertex, exactly where
                # _bdpt_camera_connect_gpu's single mega-kernel already ran
                # them -- only the primary/bounce ray intersect moved out.
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
                    # ONE dispatch over all n_pix*_BDPT_MAX_VERTS slots
                    # (the interop scene's ray capacity was resized in
                    # pipeline.mojo specifically for this), replacing the
                    # earlier _BDPT_MAX_VERTS-separate-dispatches loop --
                    # cuts the per-bounce interop trace CALL count from 10
                    # to 1, the dominant cost identified by that follow-up's
                    # investigation. Then fold the resolved contributions
                    # into each pixel's running total.
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
                # LVC. Launched after the camera accumulate on the same
                # stream so that kernel's non-atomic per-pixel writes can
                # never overlap these atomic adds.
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
                    # lvc_camis buffer -- a dangling pointer is never
                    # dereferenced. GPU kernel launches need every argument
                    # passed explicitly; the callee's own default cannot be
                    # elided here the way an ordinary call can.
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
            # CPU tail -- see _vcm_finalize_one_pixel's docstring. No clamp
            # here; finish_render applies it once, to their sum, after
            # `pixels` is denoised.
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
            # a fresh normals/depth pass via the host-side sd (same
            # render_aux_buffers the CPU path/plain tracer use -- host-only,
            # so it runs on the CPU here too, not as a GPU kernel), then the
            # same CPU denoise() the CPU BDPT path uses.
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
# WITHIN sppm.mojo (regardless of kernel/host-function shape -- verified down
# to a 3-line minimal repro, and independently for a totally fresh,
# unrelated kernel added to that file) fails to typecheck ("no matching
# method in call to 'enqueue_function'", the kernel's own inferred type
# showing as plain "thin -> None" instead of the expected "capturing
# thin -> None"). The identical code compiles cleanly once moved to
# bdpt.mojo (confirmed empirically, twice: once for an existing sppm.mojo
# kernel called cross-file, once for a brand-new kernel+host-function pair
# defined directly in bdpt.mojo) -- this appears to be a real, per-FILE-
# specific compiler defect (a new file created for this code, e.g.
# `sppm_gpu.mojo`, was ALSO cursed). See reference_mojo_compiler_bug_6759
# memory / project_modular_26_5_0_migration memory for the full bisection.
#
# Re-checked 2026-09-17 on the current toolchain, and the move still fails --
# but the recorded "it is specifically the sppm.mojo file object" reading is
# WRONG: a minimal kernel + enqueue_function pair appended to sppm.mojo
# compiles fine, while THESE kernels fail both there and in a fresh
# sppm_gpu.mojo, with explicit imports as well as a wildcard one. So the
# trigger travels with this code, not with a file. Whatever it is has not
# been isolated; a smaller repro than "move all 735 lines" is the next step
# for anyone retrying.
# ── GPU kernels ───────────────────────────────────────────────────────────────
# Each kernel is a thin wrapper: compute this thread's index, then with the
# host-built SceneView `sd` call the EXACT SAME
# shared function the CPU driver above calls (comptime[use_gpu]-branching
# only at the two genuine concurrency-primitive divergence points: photon-
# slot reservation and hash-grid bucket insertion) — mirrors bdpt.mojo's
# GPU kernels, deliberately unlike the old gpu_sppm.mojo (a full duplicate
# reimplementation).
