from max.gpu.host import DeviceContext
from max.gpu.host._nvidia_cuda import CUDA
from .rtcore import rtcore_active, rtcore_alpha_enabled, rtcore_shadow_enabled, rtcore_trace, rtcore_trace_interop
from .xert import xert_active, xert_alpha, xert_shadow_slots, xert_trace_interop

# Hardware ray tracing on either vendor: Intel Xe2 through xert, the NVIDIA RT cores through librtcore. Rays are 8
# floats each (ox oy oz t_min dx dy dz t_max) at device addresses; CUDA streams are only asked for on NVIDIA.

def hw_rt_active() -> Bool:
    return xert_active() or Int(rtcore_active()) != 0

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
