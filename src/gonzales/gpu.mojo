from .geometry import TERMINAL_SEGMENT_GRACE_ROUNDS
from .materials import Material, MeasuredBRDF
from .media import Grid, MediumInterface, Medium, NvdbGrid
from .primitives import Instance, Intersection, Sphere
from .render_state import PathState, ShadowTask
from .restir_di import DIReservoir
from .restir_vol import VolReservoir
from .vulkaninterop import VulkanInteropRtSceneHandle
from max.gpu import block_dim
from max.gpu.host import DeviceBuffer
from std.math import ceildiv
from std.sys import has_accelerator
from .gpu_media import sample_medium_gpu, update_medium_gpu
from .gpu_scene import GpuSceneHandle
from .gpu_shade import shade_coated_conductor_gpu, shade_coated_diffuse_gpu, shade_conductor_gpu, shade_dielectric_gpu, shade_diffuse_gpu, shade_diffuse_transmit_gpu, shade_hair_gpu, shade_interface_gpu, shade_measured_gpu, shade_mix_gpu, shade_nee_preamble_gpu, shade_thin_dielectric_gpu
from .gpu_wavefront import accumulate_cone_gpu, accumulate_film_gpu, accumulate_film_wavefront_gpu, clear_film_gpu, compact_curve_paths_gpu, deactivate_paths_past_maxdepth_gpu, gen_primary_rays_gpu, gen_primary_rays_wavefront_gpu, reset_curve_counter_gpu, reset_restir_reservoirs_gpu, reset_restir_vol_reservoirs_gpu, reset_shadow_tasks_gpu, reset_vol_used_gpu, resolve_curve_candidates_gpu, traverse_paths_gpu, traverse_shadow_rays_gpu, vulkaninterop_rt_traverse_paths_gpu


def _gpu_bounce_kernels(
    handle: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    n: Int, grid_dim: Int, px_scale: Float32,
    max_depth: Int32,
    use_vulkan_rt: Bool = False,
    interop_scene: VulkanInteropRtSceneHandle = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
    interop_rays_buf: Optional[DeviceBuffer[DType.float32]] = None,
    interop_results_buf: Optional[DeviceBuffer[DType.float32]] = None,
    mesh_material_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    mesh_al_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    n_meshes_vk: Int = 0,
    # ReSTIR DI: only gpu_render_sample passes these (one path per pixel, so
    # tid is a valid pixel index). gpu_render_wavefront leaves them inert.
    use_restir: Bool = False,
    restir_read: Pointer[DIReservoir, MutUntrackedOrigin] = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling(),
    restir_write: Pointer[DIReservoir, MutUntrackedOrigin] = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling(),
    # Phase 7.3: same "only gpu_render_sample passes these" contract as
    # use_restir/restir_read/restir_write above, for volume-scatter vertices.
    use_vol_restir_reuse: Bool = False,
    restir_vol_read: Pointer[VolReservoir, MutUntrackedOrigin] = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
    restir_vol_write: Pointer[VolReservoir, MutUntrackedOrigin] = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
    # Per-pixel "already combined this frame" guard, reset by
    # gpu_render_sample before the bounce-round loop starts -- see
    # _sample_medium_core's own comment on vol_used for the bug this fixes.
    restir_vol_used: Pointer[Int8, MutUntrackedOrigin] = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling(),
    # Spatial reuse (2026-09-08): same G-buffers DI's own spatial reuse
    # already reads, harmless to pass unconditionally (see
    # _sample_medium_core's matching comment).
    restir_vol_gbuf_depth: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    restir_vol_gbuf_world_pos: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    restir_vol_frame_w: Int32 = Int32(0),
    restir_vol_frame_h: Int32 = Int32(0),
    # Object-instancing decode for Vulkan RT hits (see
    # vulkaninterop_unpack_results_kernel) -- None for scenes with no
    # instancing, matching every other Optional buffer above.
    instance_base_mesh_buf: Optional[DeviceBuffer[DType.uint8]] = None,
) raises:
    comptime block_size = 256
    var sd = handle[].scene_descriptor()
    handle[].ctx.enqueue_function[deactivate_paths_past_maxdepth_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n), max_depth,
        grid_dim=grid_dim, block_dim=block_size,
    )
    if use_vulkan_rt:
        vulkaninterop_rt_traverse_paths_gpu(
            handle[].ctx,
            handle[].path_buf,
            handle[].inter_buf,
            interop_scene,
            interop_rays_buf.value(),
            interop_results_buf.value(),
            mesh_material_idx_buf.value(),
            mesh_al_idx_buf.value(),
            n_meshes_vk,
            n,
            instance_base_mesh_buf,
            handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere](),
            handle[].n_spheres,
        )
    else:
        handle[].ctx.enqueue_function[traverse_paths_gpu](
            sd,
            handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            handle[].curves.cand_prim_ptr(),
            handle[].curves.cand_count_ptr(),
            Int64(n),
            grid_dim=grid_dim,
            block_dim=block_size,
        )
    # Both traversal branches converge here: grow the texture-footprint cone
    # by the segment just traced, before any material shading reads it.
    handle[].ctx.enqueue_function[accumulate_cone_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n),
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    # Deferred-candidate curve resolution only applies to the CUDA-native
    # intersection path (traverse_paths_gpu above, which defers candidates
    # into handle[]'s own curve_cand_* buffers). A Vulkan-RT render never
    # populates those buffers at all -- intersect_batch.comp resolves curve
    # hits itself, inline, as part of the SAME dispatch that traces meshes
    # (see vulkaninterop_rt_traverse_paths_gpu / vulkaninterop_unpack_
    # results_kernel's hitFlag==2 branch) -- so this whole compact+resolve
    # pass is CUDA-native-only.
    if handle[].curves.n_curves > 0 and not use_vulkan_rt:
        handle[].ctx.enqueue_function[reset_curve_counter_gpu](
            handle[].curves.compact_counter_ptr(),
            grid_dim=1, block_dim=1,
        )
        handle[].ctx.enqueue_function[compact_curve_paths_gpu](
            handle[].curves.cand_count_ptr(),
            Int64(n),
            handle[].curves.compact_path_ptr(),
            handle[].curves.compact_counter_ptr(),
            grid_dim=grid_dim, block_dim=block_size,
        )
        handle[].ctx.enqueue_function[resolve_curve_candidates_gpu](
            handle[].curves.compact_path_ptr(),
            handle[].curves.compact_counter_ptr(),
            handle[].curves.cand_prim_ptr(),
            handle[].curves.cand_count_ptr(),
            handle[].curves.cand_offset_ptr(),
            sd,
            handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            Int64(n),
            grid_dim=grid_dim,
            block_dim=block_size,
        )
    handle[].ctx.enqueue_function[sample_medium_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        Int64(n),
        Int32(1) if use_vol_restir_reuse else Int32(0),
        restir_vol_read,
        restir_vol_write,
        restir_vol_used,
        restir_vol_gbuf_depth,
        restir_vol_gbuf_world_pos,
        restir_vol_frame_w,
        restir_vol_frame_h,
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_nee_preamble_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n),
        px_scale,
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    # mix is a pure selector (see shade_mix_gpu's docstring) -- enqueued
    # FIRST among the per-material kernels so its pending_mat/materialIndex
    # redirect is visible to whichever real kernel the sub-material resolves
    # to, later in this SAME launch-ordered sequence.
    handle[].ctx.enqueue_function[shade_mix_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        Int64(n),
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    # Phase 0.4 (docs/A2_restir_migration_plan.md): reset_shadow_tasks_gpu
    # and the traverse_shadow_rays_gpu resolve call below are real, verified
    # machinery -- built, buffer-sized correctly (shadow_buf now matches
    # path_buf/inter_buf's n_pixels × WAVEFRONT_BATCH sizing), and confirmed
    # via render comparison against inline NEE for a single deferred shadow
    # ray per pixel per bounce. But every per-material kernel below is left
    # at enqueue_shadow=False (shadow_tasks is threaded through as a real
    # pointer and ready, not removed) rather than flipped on, because of a
    # real, structural mismatch discovered while verifying this: ShadowTask
    # holds ONE task per pixel, while _shade_diffuse_nee/_nee_area_lights/
    # _shade_conductor_nee/etc. (shading.mojo) each loop over MULTIPLE light
    # types per bounce (area, sphere, distant, point, infinite), calling
    # _shadow_contribute once per candidate. Under enqueue_shadow=True that
    # write is `ctx.shadow_tasks[ctx.path_idx] = ShadowTask(...)` --  an
    # OVERWRITE, not an accumulation -- so only the last light type resolved
    # this bounce survives; every earlier one silently vanishes. Measured on
    # real scenes: glass-of-water (mean radiance dropped 88%), curly-hair
    # (91%), material-testball (9%) -- cornell-box (single area light, one
    # NEE candidate per pixel) was the only scene unaffected, which is what
    # hid this until a multi-light comparison render caught it.
    # This single-slot design IS correct for its actual intended consumer:
    # Phase 2 (ReSTIR DI)'s reservoir winner is BY CONSTRUCTION exactly one
    # light candidate per pixel after resampling. Do not flip any of these 6
    # kernels' enqueue_shadow to True for today's multi-candidate NEE loops
    # without first either (a) giving ShadowTask N slots (N = max
    # simultaneous light types, currently 5) with accumulate semantics, or
    # (b) restricting deferral to a genuinely single-candidate call site.
    handle[].ctx.enqueue_function[reset_shadow_tasks_gpu](
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask](),
        Int64(n),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_diffuse_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n),
        px_scale,
        Int32(1) if use_restir else Int32(0),
        restir_read,
        restir_write,
        handle[].atrous_normals_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].atrous_depth_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].gbuf_material_id_buf.unsafe_ptr().unsafe_bitcast[Int32](),
        handle[].gbuf_worldpos_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].film.width,
        handle[].film.height,
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_coated_diffuse_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n),
        px_scale,
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask](),
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_diffuse_transmit_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n),
        px_scale,
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask](),
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_conductor_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n),
        px_scale,
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask](),
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_measured_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n),
        px_scale,
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask](),
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_dielectric_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        Int64(n),
        px_scale,
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_thin_dielectric_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        Int64(n),
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_coated_conductor_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n),
        px_scale,
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask](),
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_interface_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        Int64(n),
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    handle[].ctx.enqueue_function[update_medium_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        Int64(n),
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_hair_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        sd,
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n),
        px_scale,
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask](),
        grid_dim=grid_dim,
        block_dim=block_size,
    )
    # Phase 0.4: resolve whatever shadow rays this bounce's per-material
    # kernels deferred above. Currently a verified no-op in production --
    # every per-material kernel above still enqueues with enqueue_shadow=
    # False (see that block's own comment for why), so reset_shadow_tasks_gpu
    # zeroed every slot and nothing here ever finds task.active != 0. Kept
    # enqueued (not removed) because Phase 2 (ReSTIR DI) is expected to be
    # this resolve kernel's first real, single-candidate-per-pixel consumer.
    # Software BVH only (matches traverse_shadow_rays_gpu's own
    # implementation) even when use_vulkan_rt is set -- this machinery isn't
    # wired to Vulkan RT yet.
    handle[].ctx.enqueue_function[traverse_shadow_rays_gpu](
        sd,
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask](),
        Int64(n),
        grid_dim=grid_dim,
        block_dim=block_size,
    )


# Render one sample pass into the persistent film buffer.
# Ray generation runs on GPU — no CPU-side path buffer or PCIe upload needed.
def gpu_render_sample[Oc: Origin[mut=True]](
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    c2w: Pointer[Float32, Oc],
    si: Int32, log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    n: Int64,
    maxDepth: Int32,
    px_scale: Float32 = Float32(0.0),
    sample_clamp: Float32 = Float32(0.0),   # pbrt's per-sample maxcomponentvalue, iso-divided
    use_restir: Bool = False,
    frame_index: Int = 0,
    use_vol_restir_reuse: Bool = False,
):
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            # Update c2w for this frame
            handle[].ctx.enqueue_copy(handle[].c2w_buf, c2w.unsafe_bitcast[UInt8]())
            comptime block_size = 256
            var grid_dim = ceildiv(n_int, block_size)
            # Generate primary rays on GPU
            handle[].ctx.enqueue_function[gen_primary_rays_gpu](
                handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
                handle[].r2c_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                Int64(handle[].film.width), Int64(handle[].film.height),
                si, log2spp, n_base4,
                seed_dim0, seed_dim1,
                rng_seed_lo, rng_seed_hi,
                handle[].filter.sigma, handle[].filter.norm_x, handle[].filter.support_x,
                handle[].filter.norm_y, handle[].filter.support_y,
                handle[].filter.type,
                Int64(n_int),
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            # ReSTIR reservoir ping-pong. Spatial reuse reads neighbouring
            # pixels, which other threads are writing this frame, so the read
            # side must be the PREVIOUS frame's finished buffer. Alternating on
            # frame parity gives that without any copy.
            var restir_rd = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling()
            var restir_wr = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling()
            if use_restir:
                var buf_a = handle[].restir_a_buf.unsafe_ptr().unsafe_bitcast[DIReservoir]()
                var buf_b = handle[].restir_b_buf.unsafe_ptr().unsafe_bitcast[DIReservoir]()
                if frame_index % 2 == 0:
                    restir_rd = buf_a; restir_wr = buf_b
                else:
                    restir_rd = buf_b; restir_wr = buf_a
            # Phase 7.3: same ping-pong rule as DI's above, own buffer pair.
            # Temporal-only (no spatial neighbour reads), so the same-frame
            # in-flight-write race spatial reuse would need to worry about
            # doesn't apply here, but ping-ponging costs nothing and keeps
            # this consistent with every other reservoir buffer in the file.
            var vol_rd = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling()
            var vol_wr = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling()
            var vol_used_ptr = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling()
            var vol_gbuf_depth_ptr = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
            var vol_gbuf_world_pos_ptr = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
            var vol_fw = Int32(0)
            var vol_fh = Int32(0)
            if use_vol_restir_reuse:
                var vbuf_a = handle[].restir_vol_a_buf.unsafe_ptr().unsafe_bitcast[VolReservoir]()
                var vbuf_b = handle[].restir_vol_b_buf.unsafe_ptr().unsafe_bitcast[VolReservoir]()
                if frame_index % 2 == 0:
                    vol_rd = vbuf_a; vol_wr = vbuf_b
                else:
                    vol_rd = vbuf_b; vol_wr = vbuf_a
                vol_used_ptr = handle[].restir_vol_used_buf.unsafe_ptr().unsafe_bitcast[Int8]()
                handle[].ctx.enqueue_function[reset_vol_used_gpu](
                    vol_used_ptr, Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
                )
                # Spatial reuse: the SAME G-buffers DI's own spatial reuse
                # already reads (gen_aux_buffers_gpu populates them
                # unconditionally every frame, see gpu_gen_aux_buffers).
                vol_gbuf_depth_ptr = handle[].atrous_depth_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                vol_gbuf_world_pos_ptr = handle[].gbuf_worldpos_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                vol_fw = handle[].film.width
                vol_fh = handle[].film.height
            # See geometry.mojo's TERMINAL_SEGMENT_GRACE_ROUNDS: a maxdepth-
            # capped path still needs one more round to trace the ray from
            # its last real scatter and collect whatever it lands on or
            # escapes to. UNCONDITIONAL and universal, unlike the two
            # conditional margins below (a different reason: free null-
            # interface rounds / the SSS walk), which stay scene-gated.
            var gpu_max_rounds = Int(maxDepth) + TERMINAL_SEGMENT_GRACE_ROUNDS
            # Padding beyond the +1 above is CONDITIONAL on the scene
            # containing a medium or an `interface` material -- the only
            # things that make a null crossing -- because there is no
            # per-round host sync here (unlike the CPU loop's cheap
            # `anyActive` early-exit) to make extra rounds free: each one is a
            # real, unconditional dispatch of every kernel in
            # _gpu_bounce_kernels. Gating on media alone starved paths through
            # medium-free interface glass (pavilion interior 0.945x pbrt).
            comptime _MEDIUM_INTERFACE_MARGIN = 8
            # An SSS interior is walked one scattering event per round and
            # those steps are not charged to maxDepth (Medium.is_sss), so
            # the round count is what actually bounds the walk. Unlike the
            # margin above this is a large budget, and unlike the CPU loop
            # there is no `anyActive` early exit here -- every round is a real
            # dispatch. Gated on the scene actually containing an SSS medium
            # so no other scene pays for it.
            comptime _SSS_WALK_ROUNDS = 256
            if handle[].media.n_mediums > 0 or handle[].has_interface_material:
                gpu_max_rounds += _MEDIUM_INTERFACE_MARGIN
            if handle[].media.has_sss_medium:
                gpu_max_rounds += _SSS_WALK_ROUNDS
            for _ in range(gpu_max_rounds):
                _gpu_bounce_kernels(handle, n_int, grid_dim, px_scale, maxDepth,
                                    use_restir=use_restir,
                                    restir_read=restir_rd, restir_write=restir_wr,
                                    use_vol_restir_reuse=use_vol_restir_reuse,
                                    restir_vol_read=vol_rd, restir_vol_write=vol_wr,
                                    restir_vol_used=vol_used_ptr,
                                    restir_vol_gbuf_depth=vol_gbuf_depth_ptr,
                                    restir_vol_gbuf_world_pos=vol_gbuf_world_pos_ptr,
                                    restir_vol_frame_w=vol_fw, restir_vol_frame_h=vol_fh)
            handle[].ctx.enqueue_function[accumulate_film_gpu](
                handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                handle[].film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].albedo_film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(n_int), sample_clamp,
                        handle[].spectral.coeffs_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int64(handle[].spectral.res),
        handle[].spectral.cie_x_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_y_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_z_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.d65_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        grid_dim=grid_dim,
                block_dim=block_size,
            )
        except e:
            print("GPU render sample failed: " + String(e))


# Wavefront render: generates actual_batch samples worth of primary rays for all pixels,
# runs the full bounce loop over n_pixels × actual_batch paths together, then accumulates.
# Caller loops over spp in steps of WAVEFRONT_BATCH; progress reporting is up to the caller.
def gpu_render_wavefront(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    si_start: Int32, actual_batch: Int32,
    log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    n: Int64,
    maxDepth: Int32,
    px_scale: Float32 = Float32(0.0),
    sample_clamp: Float32 = Float32(0.0),   # pbrt's per-sample maxcomponentvalue, iso-divided
    # Task #163 stage 3: when use_vulkan_rt, every bounce's primary
    # intersection test is routed through the CUDA/Vulkan interop RT
    # backend (vulkaninterop_rt_traverse_paths_gpu, zero CPU sync) instead
    # of the traverse_paths_gpu CUDA kernel below -- caller must only set
    # this for scenes with no curves/spheres (object instancing IS
    # supported now -- see vulkaninterop_rt_create_scene/
    # vulkaninterop_rt_traverse_paths_gpu's docstrings). The interop_*/
    # mesh_*_buf/n_meshes_vk params are ignored when use_vulkan_rt is False.
    use_vulkan_rt: Bool = False,
    interop_scene: VulkanInteropRtSceneHandle = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
    interop_rays_buf: Optional[DeviceBuffer[DType.float32]] = None,
    interop_results_buf: Optional[DeviceBuffer[DType.float32]] = None,
    mesh_material_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    mesh_al_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    n_meshes_vk: Int = 0,
    instance_base_mesh_buf: Optional[DeviceBuffer[DType.uint8]] = None,
):
    # --restir batch rendering does not go through this function at all
    # (docs/A2_restir_migration_plan.md, pipeline.mojo's batch GPU branch):
    # WAVEFRONT_BATCH (8) concurrent samples per pixel per dispatch made
    # reservoir reuse either meaningless (plain RIS, no persistence) or
    # actively wrong (an attempted "N readers, 1 writer" reservoir-sharing
    # scheme measured a real, unexplained divergence and was reverted --
    # see project memory). Batch --restir instead loops gpu_render_sample
    # (1 sample/pixel/dispatch, same proven-correct path
    # --interactive-frames uses) so there is only ever one reader/writer
    # per pixel. This function is unconditionally RIS-and-reuse-free.
    var n_pix = Int(n)
    var batch  = Int(actual_batch)
    var n_total = n_pix * batch
    if n_total == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            handle[].ctx.enqueue_copy(handle[].c2w_buf, c2w.unsafe_bitcast[UInt8]())
            comptime block_size = 256
            var grid_total = ceildiv(n_total, block_size)
            var grid_pix   = ceildiv(n_pix, block_size)
            handle[].ctx.enqueue_function[gen_primary_rays_wavefront_gpu](
                handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
                handle[].r2c_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                Int64(handle[].film.width), Int64(handle[].film.height),
                si_start, log2spp, n_base4,
                seed_dim0, seed_dim1, rng_seed_lo, rng_seed_hi,
                handle[].filter.sigma, handle[].filter.norm_x, handle[].filter.support_x,
                handle[].filter.norm_y, handle[].filter.support_y,
                handle[].filter.type,
                Int64(n_total), Int64(n_pix),
                grid_dim=grid_total,
                block_dim=block_size,
            )
            # See geometry.mojo's TERMINAL_SEGMENT_GRACE_ROUNDS: a maxdepth-
            # capped path still needs one more round to trace the ray from
            # its last real scatter and collect whatever it lands on or
            # escapes to. UNCONDITIONAL and universal, unlike the two
            # conditional margins below (a different reason: free null-
            # interface rounds / the SSS walk), which stay scene-gated.
            var gpu_max_rounds = Int(maxDepth) + TERMINAL_SEGMENT_GRACE_ROUNDS
            # Padding beyond the +1 above is CONDITIONAL on the scene
            # containing a medium or an `interface` material -- the only
            # things that make a null crossing -- because there is no
            # per-round host sync here (unlike the CPU loop's cheap
            # `anyActive` early-exit) to make extra rounds free: each one is a
            # real, unconditional dispatch of every kernel in
            # _gpu_bounce_kernels. Gating on media alone starved paths through
            # medium-free interface glass (pavilion interior 0.945x pbrt).
            comptime _MEDIUM_INTERFACE_MARGIN = 8
            # An SSS interior is walked one scattering event per round and
            # those steps are not charged to maxDepth (Medium.is_sss), so
            # the round count is what actually bounds the walk. Unlike the
            # margin above this is a large budget, and unlike the CPU loop
            # there is no `anyActive` early exit here -- every round is a real
            # dispatch. Gated on the scene actually containing an SSS medium
            # so no other scene pays for it.
            comptime _SSS_WALK_ROUNDS = 256
            if handle[].media.n_mediums > 0 or handle[].has_interface_material:
                gpu_max_rounds += _MEDIUM_INTERFACE_MARGIN
            if handle[].media.has_sss_medium:
                gpu_max_rounds += _SSS_WALK_ROUNDS
            for _ in range(gpu_max_rounds):
                _gpu_bounce_kernels(
                    handle, n_total, grid_total, px_scale, maxDepth,
                    use_vulkan_rt, interop_scene, interop_rays_buf, interop_results_buf,
                    mesh_material_idx_buf, mesh_al_idx_buf, n_meshes_vk,
                    instance_base_mesh_buf=instance_base_mesh_buf,
                )
            handle[].ctx.enqueue_function[accumulate_film_wavefront_gpu](
                handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                handle[].film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].albedo_film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(n_pix), Int64(batch), sample_clamp,
                        handle[].spectral.coeffs_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int64(handle[].spectral.res),
        handle[].spectral.cie_x_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_y_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_z_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.d65_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        grid_dim=grid_pix,
                block_dim=block_size,
            )
        except e:
            print("GPU wavefront render failed: " + String(e))


def gpu_download_film(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    film: Pointer[Float32, MutUntrackedOrigin],
    n: Int64,
):
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            # Straight device-to-host copy: map_to_host would pin a ~1.3 GiB host
            # pool on first use (~1 s of page faults). film holds n_int*3 floats,
            # the size of film_buf.
            handle[].ctx.enqueue_copy(film.unsafe_bitcast[UInt8](), handle[].film_buf)
            handle[].ctx.synchronize()
        except e:
            print("GPU download film failed: " + String(e))


def gpu_download_albedo[Of: Origin[mut=True]](
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    film: Pointer[Float32, Of],
    n: Int64,
):
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            # See gpu_download_film.
            handle[].ctx.enqueue_copy(film.unsafe_bitcast[UInt8](), handle[].albedo_film_buf)
            handle[].ctx.synchronize()
        except e:
            print("GPU download albedo failed: " + String(e))


# ── À-trous wavelet denoiser (Dammertz et al. 2010) ─────────────────────────
# Three kernels: normalize, variance estimate, one à-trous pass (5× ping-pong).

def gpu_clear_film(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    n: Int64,
):
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            comptime block_size = 256
            var grid_dim = ceildiv(n_int, block_size)
            handle[].ctx.enqueue_function[clear_film_gpu](
                handle[].film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(n_int),
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            handle[].ctx.enqueue_function[clear_film_gpu](
                handle[].albedo_film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(n_int),
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            handle[].ctx.synchronize()
        except e:
            print("GPU clear film failed: " + String(e))


def gpu_clear_restir(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    n: Int64,
):
    """Reset both ReSTIR reservoir buffers. Call wherever gpu_clear_film is
    called: identity reprojection assumes the stored reservoir belongs to this
    pixel's shading point, which a camera move invalidates. Both buffers are
    cleared because which one is "read" this frame depends on frame parity."""
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            comptime block_size = 256
            var grid_dim = ceildiv(n_int, block_size)
            handle[].ctx.enqueue_function[reset_restir_reservoirs_gpu](
                handle[].restir_a_buf.unsafe_ptr().unsafe_bitcast[DIReservoir](),
                Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
            )
            handle[].ctx.enqueue_function[reset_restir_reservoirs_gpu](
                handle[].restir_b_buf.unsafe_ptr().unsafe_bitcast[DIReservoir](),
                Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
            )
            handle[].ctx.synchronize()
        except e:
            print("GPU clear restir failed: " + String(e))


def gpu_clear_restir_vol(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    n: Int64,
):
    """Reset both volume ReSTIR reservoir buffers (Phase 7.3). Call wherever
    gpu_clear_restir is called -- same identity-reprojection invalidation
    rule, same both-buffers-cleared reason (frame parity)."""
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            comptime block_size = 256
            var grid_dim = ceildiv(n_int, block_size)
            handle[].ctx.enqueue_function[reset_restir_vol_reservoirs_gpu](
                handle[].restir_vol_a_buf.unsafe_ptr().unsafe_bitcast[VolReservoir](),
                Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
            )
            handle[].ctx.enqueue_function[reset_restir_vol_reservoirs_gpu](
                handle[].restir_vol_b_buf.unsafe_ptr().unsafe_bitcast[VolReservoir](),
                Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
            )
            handle[].ctx.synchronize()
        except e:
            print("GPU clear restir vol failed: " + String(e))
