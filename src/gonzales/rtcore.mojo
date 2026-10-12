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

# A structure from rt_build.mojo; gid maps the hit's primitive id to the merged triangle.
def rtcore_create_own(
    cubin_path: Pointer[UInt8, MutUntrackedOrigin],
    as_bytes: Pointer[UInt8, MutUntrackedOrigin],
    as_size: Int64,
    gid: Pointer[Int32, MutUntrackedOrigin],
    n_gid: Int64,
) -> RtCoreHandle:
    return external_call["rtcore_create_own", RtCoreHandle,
        Pointer[UInt8, MutUntrackedOrigin], Pointer[UInt8, MutUntrackedOrigin], Int64, Pointer[Int32, MutUntrackedOrigin], Int64](
        cubin_path, as_bytes, as_size, gid, n_gid)

# An instanced scene from rt_build.mojo: blas_bytes / blas_sizes are n_blas host pointers and sizes.
def rtcore_create_own_scene(
    cubin_path: Pointer[UInt8, MutUntrackedOrigin],
    n_blas: Int32,
    blas_bytes: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
    blas_sizes: Pointer[Int64, MutUntrackedOrigin],
    tlas_bytes: Pointer[UInt8, MutUntrackedOrigin],
    tlas_size: Int64,
    rec_off: Int64,
    n_rec: Int64,
    gid: Pointer[Int32, MutUntrackedOrigin],
    n_gid: Int64,
) -> RtCoreHandle:
    return external_call["rtcore_create_own_scene", RtCoreHandle,
        Pointer[UInt8, MutUntrackedOrigin], Int32, Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
        Pointer[Int64, MutUntrackedOrigin], Pointer[UInt8, MutUntrackedOrigin], Int64, Int64, Int64,
        Pointer[Int32, MutUntrackedOrigin], Int64](
        cubin_path, n_blas, blas_bytes, blas_sizes, tlas_bytes, tlas_size, rec_off, n_rec, gid, n_gid)

# Enqueues the trace on `stream`; does not synchronize. rays/results are device addresses.
def rtcore_trace(handle: RtCoreHandle, rays: UInt64, results: UInt64, ray_count: Int32, stream: CUstream) -> Int32:
    return external_call["rtcore_trace", Int32, RtCoreHandle, UInt64, UInt64, Int32, CUstream](
        handle, rays, results, ray_count, stream)

# Top-level structure: structure 0 is the TLAS, the rest the BLASes it references (see rtcore.h).
def rtcore_create_scene(
    cubin_path: Pointer[UInt8, MutUntrackedOrigin], n_as: Int32,
    as_bytes: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
    as_sizes: Pointer[Int64, MutUntrackedOrigin],
    as_addresses: Pointer[UInt64, MutUntrackedOrigin],
) -> RtCoreHandle:
    return external_call["rtcore_create_scene", RtCoreHandle, Pointer[UInt8, MutUntrackedOrigin], Int32,
        Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin], Pointer[Int64, MutUntrackedOrigin],
        Pointer[UInt64, MutUntrackedOrigin]](cubin_path, n_as, as_bytes, as_sizes, as_addresses)

# Decode tables for the hit instance / triangle (see rtcore.h): one domain per TLAS instance.
def rtcore_set_domains(
    handle: RtCoreHandle, n_domains: Int32,
    dom_base: Pointer[Int32, MutUntrackedOrigin], dom_n: Pointer[Int32, MutUntrackedOrigin],
    pre: Pointer[Int32, MutUntrackedOrigin], rawv: Pointer[Int32, MutUntrackedOrigin],
    geom: Pointer[Int32, MutUntrackedOrigin], n_entries: Int32,
) -> Int32:
    return external_call["rtcore_set_domains", Int32, RtCoreHandle, Int32,
        Pointer[Int32, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin],
        Pointer[Int32, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin], Int32](
        handle, n_domains, dom_base, dom_n, pre, rawv, geom, n_entries)

# Template instances decode from per-template tables (see rtcore.h); `inst_blas` null = read the TLAS records.
def rtcore_set_instance_decode(
    handle: RtCoreHandle, n_inst: Int32, inst_blas: UInt64, n_templates: Int32,
    tpre_off: Pointer[Int32, MutUntrackedOrigin], t_cnt: Pointer[Int32, MutUntrackedOrigin], raw_base: Int32,
) -> Int32:
    return external_call["rtcore_set_instance_decode", Int32, RtCoreHandle, Int32, UInt64, Int32,
        Pointer[Int32, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin], Int32](
        handle, n_inst, inst_blas, n_templates, tpre_off, t_cnt, raw_base)

# Swaps the flat id table for the compact one (see rt_convert.cu): head word, 8 byte deltas per group, wide groups of 8 ints.
def rtcore_set_gid_compact(
    handle: RtCoreHandle, n_groups: Int64, head: Pointer[UInt32, MutUntrackedOrigin], delta: Pointer[UInt8, MutUntrackedOrigin],
    n_wide: Int64, wide: Pointer[Int32, MutUntrackedOrigin],
) -> Int32:
    return external_call["rtcore_set_gid_compact", Int32, RtCoreHandle, Int64, Pointer[UInt32, MutUntrackedOrigin],
        Pointer[UInt8, MutUntrackedOrigin], Int64, Pointer[Int32, MutUntrackedOrigin]](
        handle, n_groups, head, delta, n_wide, wide)

# Physical device memory in bytes (0 if unknown).
def rtcore_device_total() -> UInt64:
    return external_call["rtcore_device_total", UInt64]()

# Page-locks host memory so kernels read it in place (the pointer is valid on the device).
def rtcore_host_register(p: UInt64, nbytes: UInt64) -> Int32:
    return external_call["rtcore_host_register", Int32, UInt64, UInt64](p, nbytes)

# Device address of the TLAS instance records (layout of primitives.Instance).
def rtcore_instance_records(handle: RtCoreHandle) -> UInt64:
    return external_call["rtcore_instance_records", UInt64, RtCoreHandle](handle)

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

def rtcore_set_alpha(enabled: Int32):
    external_call["rtcore_set_alpha", NoneType, Int32](enabled)

def rtcore_alpha_enabled() -> Int32:
    return external_call["rtcore_alpha_enabled", Int32]()

def rtcore_shadow_enabled() -> Int32:
    return external_call["rtcore_shadow_enabled", Int32]()

def rtcore_active() -> RtCoreHandle:
    return external_call["rtcore_active", RtCoreHandle]()

def rtcore_destroy(handle: RtCoreHandle):
    external_call["rtcore_destroy", NoneType, RtCoreHandle](handle)
