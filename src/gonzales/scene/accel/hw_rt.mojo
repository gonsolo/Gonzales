from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from gonzales.core.geometry import Vec3f, _is_real_ptr
from gonzales.scene.accel.bvh import SceneView, trace_moving
from gonzales.scene.shapes.primitives import PrimKind
from max.gpu.host._nvidia_cuda import CUDA
from gonzales.scene.accel.rtcore import rtcore_active, rtcore_alpha_enabled, rtcore_shadow_enabled, rtcore_trace, rtcore_trace_interop
from gonzales.scene.accel.xert import xert_active, xert_alpha, xert_proc_table, xert_shadow_slots, xert_trace_interop

# Hardware ray tracing on either vendor: Intel Xe2 through xert, the NVIDIA RT cores through librtcore. Rays are 8
# floats each (ox oy oz t_min dx dy dz t_max) at device addresses; CUDA streams are only asked for on NVIDIA.

def hw_rt_active() -> Bool:
    return xert_active() or Int(rtcore_active()) != 0

# The hardware trace tests the curves itself (Intel: procedural primitives), so no software curve pass follows it.
def hw_curves() -> Bool:
    return xert_active() and xert_proc_table(1) != 0

def hw_shadow_slots() -> Int:
    return xert_shadow_slots() if xert_active() else Int(rtcore_shadow_enabled())

def hw_alpha() -> Bool:
    return xert_alpha() if xert_active() else Int(rtcore_alpha_enabled()) != 0

# Closest hits in the interop layout of vulkaninterop_unpack_results_kernel.
def hw_trace_interop(ctx: DeviceContext, rays: Int, results: Int, m: Int) raises:
    if xert_active():
        xert_trace_interop(ctx, rays, results, m, False)
    else:
        _ = rtcore_trace_interop(rtcore_active(), UInt64(rays), UInt64(results), Int32(m), CUDA(ctx.stream()))

# Shadow rays, any hit: True if the results are in the interop layout (word 6 = hit flag), False for librtcore's own
# (word 3 = 0xffffffff on a miss).
def hw_trace_shadow(ctx: DeviceContext, rays: Int, results: Int, m: Int) raises -> Bool:
    if xert_active():
        xert_trace_interop(ctx, rays, results, m, True)
        return True
    _ = rtcore_trace(rtcore_active(), UInt64(rays), UInt64(results), Int32(m), CUDA(ctx.stream()))
    return False

# Where a ray's shutter time is: the `time` field of the state (path, shadow task, VCM state) behind ray slot i,
# state (ids[i] or i) // div, `stride` floats apart.
@fieldwise_init
struct RayTimes(TrivialRegisterPassable):
    var time0: Pointer[Float32, MutUntrackedOrigin]
    var stride: Int64
    var ids: Pointer[Int32, MutUntrackedOrigin]
    var div: Int64
    var fixed_time: Float32   # the time of every ray when time0 dangles

    @staticmethod
    def fixed(time: Float32) -> RayTimes:
        return RayTimes(Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Int64(0), Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
                        Int64(1), time)

    @always_inline
    def of(self, i: Int) -> Float32:
        if not _is_real_ptr(self.time0):
            return self.fixed_time
        var s = Int(self.ids[unsafe_offset=i]) if _is_real_ptr(self.ids) else i
        return self.time0[unsafe_offset=(s // Int(self.div)) * Int(self.stride)]

# The moving instances the hardware structures leave out, traced in software at each ray's time; a nearer hit
# replaces the hardware's in the interop layout, as an instance hit or (flag 4) a curve of one.
def hw_motion_gpu(sd: SceneView, rays: Pointer[Float32, MutUntrackedOrigin], results: Pointer[Float32, MutUntrackedOrigin],
                  time0: Pointer[Float32, MutUntrackedOrigin], stride: Int64, ids: Pointer[Int32, MutUntrackedOrigin], div: Int64,
                  fixed_time: Float32, base_mesh: Pointer[Int32, MutUntrackedOrigin], n_meshes: Int64, any_hit: Int32, n: Int64):
    var times = RayTimes(time0, stride, ids, div, fixed_time)
    var i = global_idx.x
    if i >= Int(n):
        return
    var ir = results.unsafe_bitcast[Int32]()
    var flag = ir[unsafe_offset=8 * i + 6]
    if flag == Int32(3) or (any_hit != Int32(0) and flag != Int32(0)):
        return
    var r = rays.unsafe_load[width=8](8 * i)
    var t_min = r[3]
    var best = results[unsafe_offset=8 * i] if flag != Int32(0) else r[7]
    var d = Vec3f(r[4], r[5], r[6])
    var o = Vec3f(r[0], r[1], r[2]) + d * t_min
    var h = trace_moving(sd.meshes, sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances, sd.curves, o, d, best - t_min,
                         any_hit != Int32(0), motion=sd.motion.at(times.of(i)))
    if not h[0]:
        return
    var p = h[4]
    var k = Int(p.instanceIdx)
    results[unsafe_offset=8 * i] = h[1] + t_min
    results[unsafe_offset=8 * i + 1] = h[2]
    results[unsafe_offset=8 * i + 2] = h[3]
    if p.type == PrimKind.curve:
        ir[unsafe_offset=8 * i + 3] = Int32(k)
        ir[unsafe_offset=8 * i + 4] = Int32(p.id1)
        ir[unsafe_offset=8 * i + 5] = Int32(p.id2)
        ir[unsafe_offset=8 * i + 6] = Int32(4)
        ir[unsafe_offset=8 * i + 7] = Int32(p.materialIndex)
        return
    ir[unsafe_offset=8 * i + 3] = Int32(-1)
    ir[unsafe_offset=8 * i + 4] = Int32(Int(n_meshes) + k)
    ir[unsafe_offset=8 * i + 5] = Int32(p.id2 // 3)
    ir[unsafe_offset=8 * i + 6] = Int32(1)
    ir[unsafe_offset=8 * i + 7] = Int32(Int(p.id1) - Int(base_mesh[unsafe_offset=k])) if _is_real_ptr(base_mesh) else Int32(0)

# Enqueues hw_motion_gpu after a hardware trace of m rays when the scene has moving instances.
def hw_motion_pass(ctx: DeviceContext, sd: SceneView, rays: Int, results: Int, times: RayTimes,
                   base_mesh: Pointer[Int32, MutUntrackedOrigin], n_meshes: Int, any_hit: Bool, m: Int) raises:
    if sd.motion.n_moving == Int32(0) or m == 0:
        return
    ctx.enqueue_function[hw_motion_gpu](sd, Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=rays),
                                        Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=results),
                                        times.time0, times.stride, times.ids, times.div, times.fixed_time, base_mesh,
                                        Int64(n_meshes), Int32(1) if any_hit else Int32(0), Int64(m),
                                        grid_dim=ceildiv(m, 256), block_dim=256)
