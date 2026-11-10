from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from std.utils import StaticTuple
from std.sys import get_defined_int
from gonzales.core.gpu_tuning import MINCTA_NEE, MINCTA_DIFFUSE, MINCTA_COATED, MINCTA_CONDUCTOR, MINCTA_MEASURED, MINCTA_DIELECTRIC
from gonzales.scene.accel.bvh import SceneView
from gonzales.core.geometry import _is_real_ptr, Vec3f, dot
from gonzales.scene.appearance.materials import MatKind, resolve_mix
from gonzales.scene.shapes.primitives import Intersection
from gonzales.transport.render_state import PathState, ShadowTask, SuppressTask
from gonzales.transport.restir_di import DIReservoir, ReservoirIO, reservoir_io_null
from gonzales.core.rng import PCG32
from gonzales.integrators.pt.materials import _shade_sss_exit, shade_coated_diffuse, shade_conductor, shade_dielectric, shade_diffuse, shade_diffuse_transmission, shade_hair, shade_interface, shade_measured, shade_thin_dielectric
from gonzales.integrators.pt.shading import shade_nee_core
from gonzales.integrators.pt.nee import _mnee_bsdf_weight, _glossy_sms_bsdf_weight
from gonzales.transport.glossy_sms import (GSMS_DEBUG, GLOSSY_SMS_MAX_TRIALS, SMSRoot, gsms_queue_count, gsms_queue_task, gsms_queue_long,
    gsms_queue_root, gsms_trial_rng, glossy_sms_refinds)
from gonzales.transport.bxdf import LobeTables
from gonzales.integrators.pt.common import _clamped
from std.memory import stack_allocation
from max.gpu.sync import barrier
from std.atomic import Atomic, Ordering
from max.gpu.memory import AddressSpace
from gonzales.transport.shade_context import ShadeContext, _shade_context
from max.gpu import block_dim, block_idx, thread_idx


def shared_shade_context(
    sd: SceneView,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    paths_base: Pointer[PathState, MutUntrackedOrigin],
    use_restir: Bool,
    shadow_tasks: Pointer[ShadowTask, MutUntrackedOrigin],
    mnee_stats: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
) -> ShadeContext:
    """The context of _shade_context with the per-thread path index replaced by the base of the path array."""
    var ctx = _shade_context(sd, sobol_matrices, use_restir=use_restir, shadow_tasks=shadow_tasks)
    ctx.paths_base = paths_base
    ctx.mnee_stats = mnee_stats
    return ctx^


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_NEE))
def shade_nee_preamble_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
    suppress_tasks: Pointer[SuppressTask, MutUntrackedOrigin] = Pointer[SuppressTask, MutUntrackedOrigin].unsafe_dangling(),
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].active == 0:
        return
    var inter = intersections[unsafe_offset=tid]
    # Do NOT early-exit on miss — shade_nee_core adds env-light contribution there.
    var ctx_no_shadow = _shade_context(sd, sobol_matrices, tid, suppress_tasks=suppress_tasks)
    shade_nee_core[True, False](path_ptr, inter, ctx_no_shadow)


# Decides, for the emitter hits shade_nee_core parked in `tasks`, whether MNEE owns the path (then the hit is dropped) or
# not (then its contribution is added). Kept out of shade_nee_preamble_gpu: the manifold solve is far too large to sit
# in the kernel every path runs through each bounce.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
def resolve_suppress_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    tasks: Pointer[SuppressTask, MutUntrackedOrigin],
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var kind = tasks[unsafe_offset=tid].active
    if kind == Int32(0):
        return
    tasks[unsafe_offset=tid].active = Int32(0)
    var path_ptr = paths.unsafe_offset(tid)
    var inter = intersections[unsafe_offset=tid]
    var ctx = _shade_context(sd, sobol_matrices, tid)
    var hp = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z) \
             + Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z) * inter.tHit
    var w = _mnee_bsdf_weight(ctx, path_ptr, inter, hp) if kind == Int32(1) else _glossy_sms_bsdf_weight(ctx, path_ptr, inter, hp)
    comptime if GSMS_DEBUG != 0:
        if GSMS_DEBUG == 1 or kind != Int32(2):
            return
        if GSMS_DEBUG == 3:
            w = Float32(1)
    if w > Float32(0.0):
        path_ptr[].estimate += tasks[unsafe_offset=tid].contrib * w


# ── Per-material GPU kernels (G1) ─────────────────────────────────────────────
# shade_nee_preamble_gpu handles miss + emission, then sets pending_mat.
# Each kernel below checks pending_mat, clears it, and calls the shade function.

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_DIFFUSE))
def shade_diffuse_gpu[mnee: Bool, defer_only: Bool = False](
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
    # ReSTIR DI (Phase 2, --restir). Only gpu_render_sample ever passes
    # use_restir=True here -- gpu_render_wavefront has no ReSTIR concept at
    # all (see its own docstring: batch --restir renders via
    # gpu_render_sample instead, precisely to avoid the
    # WAVEFRONT_BATCH-concurrent-samples-per-pixel problem). All defaulted-
    # inert so gpu_render_wavefront's dispatch is unaffected.
    # Int32 rather than Bool: GPU kernel arguments must be DevicePassable and
    # Bool is not, which the compiler only reports at the enqueue site.
    use_restir: Int32 = Int32(0),
    restir_read: Pointer[DIReservoir, MutUntrackedOrigin] = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling(),
    restir_write: Pointer[DIReservoir, MutUntrackedOrigin] = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling(),
    gbuf_normal: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    gbuf_depth: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    gbuf_material_id: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
    gbuf_world_pos: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    frame_w: Int32 = Int32(0),
    frame_h: Int32 = Int32(0),
    # Deferred shadow rays (--rt-hardware); a dangling pointer keeps every shadow ray inline.
    shadow_tasks: Pointer[ShadowTask, MutUntrackedOrigin] = Pointer[ShadowTask, MutUntrackedOrigin].unsafe_dangling(),
    # The one ShadeContext all threads share, built on the host (see shared_shade_context). Building it per thread put
    # its 480 bytes on every thread's stack (~1 GB of local-memory stores per launch).
    ctx_ptr: Pointer[ShadeContext, MutUntrackedOrigin] = Pointer[ShadeContext, MutUntrackedOrigin].unsafe_dangling(),
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.diffuse:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    if path_ptr[].sss.exiting != Int32(0):
        _shade_sss_exit[True](path_ptr, inter, ctx_ptr[])
        return
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var restir_on = use_restir != Int32(0)
    # tid IS the pixel index here: this kernel only ever sees use_restir=True
    # from gpu_render_sample, which runs exactly one path per pixel (see the
    # param block above -- gpu_render_wavefront never sets it). restir_on
    # without real buffers (non-restir renders) still needs pixel_idx=-1,
    # di_temporal_step's own "no reuse" sentinel.
    var restir_has_state = restir_on and _is_real_ptr(restir_read)
    var restir_io = reservoir_io_null()
    if restir_has_state:
        restir_io = ReservoirIO(
            read=restir_read, write=restir_write,
            gbuf_normal=gbuf_normal, gbuf_depth=gbuf_depth,
            gbuf_material_id=gbuf_material_id, gbuf_world_pos=gbuf_world_pos,
            frame_w=frame_w, frame_h=frame_h)
    shade_diffuse[True, True, mnee, defer_only](path_ptr, inter, ctx_ptr[], mat, ctx_ptr[].guide_write, restir_io, tid if restir_has_state else -1)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_COATED))
def shade_coated_diffuse_gpu[defer_only: Bool = False](
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
    shadow_tasks: Pointer[ShadowTask, MutUntrackedOrigin],
    ctx_ptr: Pointer[ShadeContext, MutUntrackedOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.coated_diffuse:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    mat = mat.glossy_filtered(path_ptr[].min_ray_pdf, sd.filterGlossy)
    shade_coated_diffuse[True, True, defer_only](path_ptr, inter, ctx_ptr[], mat)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def shade_diffuse_transmit_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
    shadow_tasks: Pointer[ShadowTask, MutUntrackedOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.diffuse_transmit:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var ctx = _shade_context(sd, sobol_matrices, path_idx=tid, shadow_tasks=shadow_tasks)
    shade_diffuse_transmission[True, True](path_ptr, inter, ctx)


# GPU-only: mix is a pure material SELECTOR, not a shader -- it has no BSDF,
# no NEE, no light/texture/spectral dependence of its own. The old
# implementation called shade_mix[True,False] with a full ShadeContext
# (33 parameters, mirroring every real per-material kernel's signature) which
# then called the @always_inline _shade_dispatch -- inlining the FULL shading
# code of every OTHER material (diffuse, conductor, dielectric,
# coated_diffuse, diffuse_transmission, coated_conductor, thin_dielectric,
# interface, measured) into this one function. That made shade_mix_gpu's
# compiled body easily the largest function in the codebase, and this
# machine's CUDA 13.3 driver / Modular 26.4.0 toolchain cannot produce valid
# PTX for it (confirmed via kernel-by-kernel bisection: shade_mix_gpu alone
# reproduces CUDA_ERROR_INVALID_PTX; every other kernel, including
# shade_measured_gpu, compiles and runs fine without it).
#
# Fix: don't shade anything here at all. Pick the sub-material (same RNG
# draw + mix-of-mix guard as shade_mix, shading.mojo) and redirect --
# overwrite this hit's materialIndex to the CHOSEN sub-material (still an
# index into the same `sd.materials` array) and re-tag pending_mat with the
# sub-material's real type, exactly mirroring what shade_nee_core's own
# GPU branch does for every material ("mark material for its dedicated
# per-material kernel"). CUDA kernels launched on one stream execute in
# launch order, so as long as this kernel is enqueued BEFORE every other
# per-material kernel in the same bounce's dispatch sequence (see both
# gpu_render_sample/gpu_render_wavefront call sites), the sub-material's
# real kernel picks up the redirected pending_mat/materialIndex later in
# this SAME pass and shades it with its own full NEE/BSDF logic --
# identical end result to the CPU path, just via a two-step handoff instead
# of one big inlined function. This can't be deferred to the NEXT bounce:
# the ray is never advanced here, so a fresh intersection at the start of
# the next bounce would just re-hit the same mix material and re-roll the
# choice forever.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def shade_mix_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.mix:
        return
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var mat_idx = Int(inter.primId.materialIndex)
    resolve_mix(sd.materials, pcg, mat, mat_idx)
    path_ptr[].pcgState = pcg.state
    intersections[unsafe_offset=tid].primId.materialIndex = Int64(mat_idx)
    path_ptr[].pending_mat = mat.type


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_CONDUCTOR))
def shade_conductor_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
    shadow_tasks: Pointer[ShadowTask, MutUntrackedOrigin],
    gsms_queue: Pointer[UInt8, MutUntrackedOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.conductor:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    mat = mat.glossy_filtered(path_ptr[].min_ray_pdf, sd.filterGlossy)
    var ctx = _shade_context(sd, sobol_matrices, path_idx=tid, shadow_tasks=shadow_tasks, gsms_queue=gsms_queue)
    shade_conductor[True, True](path_ptr, inter, ctx, mat)


def gsms_queue_reset_gpu(gsms_queue: Pointer[UInt8, MutUntrackedOrigin], cap: Int32):
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        gsms_queue.unsafe_bitcast[Int32]()[unsafe_offset=0] = Int32(0)
        gsms_queue.unsafe_bitcast[Int32]()[unsafe_offset=1] = cap
        gsms_queue.unsafe_bitcast[Int32]()[unsafe_offset=2] = Int32(0)


# The reciprocal estimates of the glossy SMS roots shade_conductor_gpu parked (transport/glossy_sms.mojo). Trial i of a
# root draws from its own sequence, so any lane may run it: a workgroup takes GSMS_GROUP trial indices of a root at a
# time and records the lowest that finds it; a root is done once its next index reaches that lowest one. The count is
# then exactly the sequential loop's. The first pass gives each root one group for GSMS_FIRST_ROUNDS takes; the second
# spreads all groups over the roots it left, which would otherwise hold the GPU for their thousands of trials.
comptime GSMS_GROUP = get_defined_int["GSMS_GROUP", 16]()
comptime GSMS_GROUPS = get_defined_int["GSMS_GROUPS", 2048]()
comptime GSMS_FIRST_ROUNDS = get_defined_int["GSMS_FIRST_ROUNDS", 4]()


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GSMS_GROUP)))
def resolve_gsms_gpu[long: Bool](sd: SceneView, gsms_queue: Pointer[UInt8, MutUntrackedOrigin]):
    var sh = stack_allocation[2, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var lane = Int(thread_idx.x)
    var g = Int(block_idx.x)
    var ctx = _shade_context(sd, Pointer[UInt32, MutUntrackedOrigin].unsafe_dangling())
    var tab = LobeTables(ctx.materials, ctx.curves, ctx.measured_brdfs, ctx.spectral)
    var head = gsms_queue.unsafe_bitcast[Int32]()
    var long_list = gsms_queue_long(gsms_queue)
    var count = Int(head[unsafe_offset=2]) if long else gsms_queue_count(gsms_queue)
    var j = 0
    var cur = -1
    var task = gsms_queue_task(gsms_queue, 0)[]
    var root = SMSRoot(fill=Vec3f(Float32(0)))
    while (j < count) if long else (g + j * GSMS_GROUPS < count):
        var t = Int(long_list[unsafe_offset=(g + j) % count]) if long else g + j * GSMS_GROUPS
        var tp = gsms_queue_task(gsms_queue, t)
        var nxt = Pointer(to=tp[].next).unsafe_origin_cast[MutUntrackedOrigin]()
        var fst = Pointer(to=tp[].first).unsafe_origin_cast[MutUntrackedOrigin]()
        if lane == 0:   # the group's next GSMS_GROUP trial indices, or none
            var lim = Atomic.load[ordering=Ordering.RELAXED](fst)
            var n_at = Atomic.load[ordering=Ordering.RELAXED](nxt)
            var stop = lim if long else min(lim, Int32(GSMS_GROUP * GSMS_FIRST_ROUNDS))
            var base = lim
            if n_at < stop:
                base = Atomic.fetch_add[ordering=Ordering.RELAXED](nxt, Int32(GSMS_GROUP))
            elif not long and n_at < lim:   # out of first-pass takes, unfinished
                long_list[unsafe_offset=Int(Atomic.fetch_add[ordering=Ordering.RELAXED](head.unsafe_offset(2), Int32(1)))] = Int32(t)
            sh[unsafe_offset=0] = lim
            sh[unsafe_offset=1] = base
        barrier()
        var lim = Int(sh[unsafe_offset=0])
        var base = Int(sh[unsafe_offset=1])
        barrier()
        if base >= lim:
            j += 1
            continue
        if cur != t:
            task = tp[]
            root = gsms_queue_root(gsms_queue, t, Int(task.n))
            cur = t
        var i = base + lane
        if i < lim:
            var pcg = gsms_trial_rng(task.seed, i)
            if glossy_sms_refinds(ctx, task.x0, task.bsdf, tab, task.wl, task.y, task.ldu, task.ldv, pcg, root, Int(task.n)):
                _ = Atomic.min[ordering=Ordering.RELAXED](fst, Int32(i))


# Each root's contribution times its trial count, the lowest index that found it plus one (the cap when none did).
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
def finish_gsms_gpu(paths: Pointer[PathState, MutUntrackedOrigin], sd: SceneView, gsms_queue: Pointer[UInt8, MutUntrackedOrigin]):
    var t = Int(block_idx.x * block_dim.x + thread_idx.x)
    if t >= gsms_queue_count(gsms_queue):
        return
    var task = gsms_queue_task(gsms_queue, t)[]
    var c = task.c * Float32(min(Int(task.first) + 1, GLOSSY_SMS_MAX_TRIALS))
    comptime if GSMS_DEBUG != 1:
        c = _clamped(_shade_context(sd, Pointer[UInt32, MutUntrackedOrigin].unsafe_dangling()), c, Int(task.bounce))
    # atomic: a path may have parked a root at several bounces
    var e = Pointer(to=paths.unsafe_offset(Int(task.path))[].estimate).unsafe_bitcast[Float32]().unsafe_origin_cast[MutUntrackedOrigin]()
    _ = Atomic[Float32].fetch_add(e, c.v0)
    _ = Atomic[Float32].fetch_add(e.unsafe_offset(1), c.v1)
    _ = Atomic[Float32].fetch_add(e.unsafe_offset(2), c.v2)
    _ = Atomic[Float32].fetch_add(e.unsafe_offset(3), c.v3)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_MEASURED))
def shade_measured_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
    shadow_tasks: Pointer[ShadowTask, MutUntrackedOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.measured:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var ctx = _shade_context(sd, sobol_matrices, path_idx=tid, shadow_tasks=shadow_tasks)
    shade_measured[True, True](path_ptr, inter, ctx, mat)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(MINCTA_DIELECTRIC))
def shade_dielectric_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
    shadow_tasks: Pointer[ShadowTask, MutUntrackedOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.dielectric:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    mat = mat.glossy_filtered(path_ptr[].min_ray_pdf, sd.filterGlossy)
    # sd.gpuTextures/camFp are here only so a dielectric carrying "texture
    # displacement"/"normalmap" gets it applied (barcelona-pavilion's water).
    # tex_filenames is CPU-only (GPU samples the uploaded texture table), so
    # the dangling default is correct on this path.
    var ctx = _shade_context(sd, sobol_matrices, path_idx=tid, shadow_tasks=shadow_tasks)
    shade_dielectric[True, True](path_ptr, inter, ctx, mat)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def shade_thin_dielectric_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.thin_dielectric:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    shade_thin_dielectric(path_ptr, inter, sd.meshes, mat, sd.spheres, sd.cycles)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def shade_coated_conductor_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
    shadow_tasks: Pointer[ShadowTask, MutUntrackedOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.coated_conductor:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    mat = mat.glossy_filtered(path_ptr[].min_ray_pdf, sd.filterGlossy)
    var ctx = _shade_context(sd, sobol_matrices, path_idx=tid, shadow_tasks=shadow_tasks)
    shade_conductor[True, True](path_ptr, inter, ctx, mat)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def shade_interface_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
):
    """Passthrough (interface) material: advance ray through the surface.
    Medium update is handled by update_medium_gpu which runs after all shaders."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.interface:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    shade_interface(path_ptr, inter)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def shade_hair_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
    shadow_tasks: Pointer[ShadowTask, MutUntrackedOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.hair:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    mat = mat.glossy_filtered(path_ptr[].min_ray_pdf, sd.filterGlossy)
    var ctx = _shade_context(sd, sobol_matrices, path_idx=tid, shadow_tasks=shadow_tasks)
    shade_hair[True, True](path_ptr, inter, ctx, mat)


# Phase 0.4 (docs/A2_restir_migration_plan.md): a task deferred by one
# material's per-pixel kernel this bounce (enqueue_shadow=True) must not be
# resolved twice, and a pixel shaded by a material that did NOT defer (still
# resolves its own shadow ray inline) must not have a stale prior-bounce
# task resolved in its place -- both need shadow_tasks[tid].active reset to
# 0 before any of this bounce's per-material kernels run, since only the one
# kernel matching pending_mat[tid] actually touches slot tid.
