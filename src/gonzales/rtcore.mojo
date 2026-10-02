from std.ffi import external_call
from max.gpu.host._nvidia_cuda import CUstream

# Hardware ray tracing from a plain CUDA kernel -- Mojo-side FFI wrapper for src/rtcore/rtcore.cpp. See rtcore.h and
# docs/rtcore/NOTES.md: the kernel is the NVIDIA driver's compiled ray-query code (built into build/rt_trace.cubin by
# docs/rtcore/exec/build_rt_cubin.py), traced against ONE bottom-level acceleration structure that Vulkan built
# (vulkanrt_build_scene + vulkanrt_debug_read_as). Rays and results are plain device buffers owned by the caller.

comptime RtCoreHandle = Pointer[UInt8, MutUntrackedOrigin]

# 32 bytes per ray: ox oy oz t_min dx dy dz t_max (floats). Results, 32 bytes per ray: t u v (floats), hit word
# (uint32: 0xffffffff = miss, else the triangle index is `word & 0x1fffffff`), 16 unused bytes.
comptime RTCORE_RAY_FLOATS = 8
comptime RTCORE_RESULT_FLOATS = 8
comptime RTCORE_MISS = UInt32(0xFFFFFFFF)
comptime RTCORE_PRIM_MASK = UInt32(0x1FFFFFFF)

def rtcore_create(
    cubin_path: Pointer[UInt8, MutUntrackedOrigin],
    as_bytes: Pointer[UInt8, MutUntrackedOrigin],
    as_size: Int64,
    as_vk_address: UInt64,
) -> RtCoreHandle:
    return external_call["rtcore_create", RtCoreHandle,
        Pointer[UInt8, MutUntrackedOrigin], Pointer[UInt8, MutUntrackedOrigin], Int64, UInt64](
        cubin_path, as_bytes, as_size, as_vk_address)

# Enqueues the trace on `stream`; does not synchronize. rays/results are device addresses.
def rtcore_trace(handle: RtCoreHandle, rays: UInt64, results: UInt64, ray_count: Int32, stream: CUstream) -> Int32:
    return external_call["rtcore_trace", Int32, RtCoreHandle, UInt64, UInt64, Int32, CUstream](
        handle, rays, results, ray_count, stream)

# Maps the merged geometry's global triangle index back to meshes: `tri_prefix` has n_meshes + 1 entries.
def rtcore_set_meshes(handle: RtCoreHandle, tri_prefix: Pointer[Int32, MutUntrackedOrigin], n_meshes: Int32) -> Int32:
    return external_call["rtcore_set_meshes", Int32, RtCoreHandle, Pointer[Int32, MutUntrackedOrigin], Int32](
        handle, tri_prefix, n_meshes)

# Like rtcore_trace, but `results` comes back in the Vulkan interop Result layout (32 bytes per ray: float t,u,v,pad;
# int mesh, triangle, hitFlag, geometryIndex), so vulkaninterop_unpack_results_kernel can consume it unchanged.
def rtcore_trace_interop(handle: RtCoreHandle, rays: UInt64, results: UInt64, ray_count: Int32, stream: CUstream) -> Int32:
    return external_call["rtcore_trace_interop", Int32, RtCoreHandle, UInt64, UInt64, Int32, CUstream](
        handle, rays, results, ray_count, stream)

# Process-wide active instance: the wavefront traversal uses the hardware trace when this is non-null.
def rtcore_set_active(handle: RtCoreHandle):
    external_call["rtcore_set_active", NoneType, RtCoreHandle](handle)

def rtcore_set_shadow(enabled: Int32):
    external_call["rtcore_set_shadow", NoneType, Int32](enabled)

def rtcore_shadow_enabled() -> Int32:
    return external_call["rtcore_shadow_enabled", Int32]()

def rtcore_active() -> RtCoreHandle:
    return external_call["rtcore_active", RtCoreHandle]()

def rtcore_destroy(handle: RtCoreHandle):
    external_call["rtcore_destroy", NoneType, RtCoreHandle](handle)
