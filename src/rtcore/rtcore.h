#ifndef GONZALES_RTCORE_H
#define GONZALES_RTCORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Hardware ray tracing from a plain CUDA kernel (see docs/rtcore/NOTES.md). The kernel is the NVIDIA driver's own
// compiled Vulkan ray-query code with its post-processing replaced by raw result stores, wrapped in a cubin by
// docs/rtcore/exec/build_rt_cubin.py. It traces against ONE bottom-level acceleration structure built by Vulkan
// (vulkanrt_build_scene + vulkanrt_debug_read_as); top-level structures are not supported (the unit calls a
// driver-installed handler for instances that a CUDA context does not have -- see the update below: they do work once the TLAS's shifted BLAS references are relocated). Tied to sm_86 and the driver version
// that compiled the shader.

// Loads the cubin into the CUDA context current on the calling thread (the primary context of device 0 if none is
// current), copies the acceleration structure `as_bytes` to device memory and patches the absolute addresses the
// driver stored inside it (every 8-byte word equal to `as_vk_address` becomes the new device address).
// Returns NULL on failure (no CUDA, cubin missing, launch config rejected).
void* rtcore_create(const char* cubin_path, const uint8_t* as_bytes, int64_t as_size, uint64_t as_vk_address);

// Same for a top-level structure: structure 0 is the TLAS, the others the BLASes it references. Each is copied to a
// 64 KB aligned device address and every address stored inside (full, and the TLAS's `address >> 16` fields) relocated.
// The traced instance index is reported by the trace (record word 4 = instance + 1) and decoded by rtcore_set_domains.
void* rtcore_create_scene(const char* cubin_path, int32_t n_as, const uint8_t* const* as_bytes, const int64_t* as_sizes,
                          const uint64_t* as_vk_addresses);

// Enqueues a trace of `ray_count` rays on `cuda_stream` (a CUstream; 0 = default stream). `rays` is a device
// pointer to 32 bytes per ray: (ox, oy, oz, t_min, dx, dy, dz, t_max). `results` is a device pointer to 32 bytes per
// ray: float t (-NaN or garbage on a miss), float u, float v, uint32 hit word (0xffffffff = miss, otherwise
// bit 31..29 kind and the low 29 bits the triangle index: use `word & 0x1fffffff`); the last 16 bytes are left
// untouched. Does not synchronize. Returns 1 on success, 0 on failure.
int rtcore_trace(void* handle, uint64_t rays, uint64_t results, int32_t ray_count, void* cuda_stream);

// Maps the merged geometry's global triangle index back to meshes: `tri_prefix` has n_meshes + 1 entries (prefix sums of
// the per-mesh triangle counts). Needed by rtcore_trace_interop only.
int rtcore_set_domains(void* handle, int32_t nDomains, const int32_t* domBase, const int32_t* domN, const int32_t* pre,
                       const int32_t* rawv, const int32_t* geom, int32_t nEntries);
int rtcore_set_meshes(void* handle, const int32_t* tri_prefix, int32_t n_meshes);

// Like rtcore_trace, but writes `results` in the Vulkan interop Result layout (see rt_convert.cu): 32 bytes per ray,
// float t,u,v,pad; int mesh, triangle, hitFlag (1 hit / 0 miss), geometryIndex. A scratch buffer holds the raw records.
int rtcore_trace_interop(void* handle, uint64_t rays, uint64_t results, int32_t ray_count, void* cuda_stream);

// A process-wide "active" instance, so the wavefront traversal code can switch from Vulkan to the hardware trace without
// threading a handle through every call (NULL = not in use).
void rtcore_set_active(void* handle);
void* rtcore_active(void);
void rtcore_set_shadow(int slots);
int rtcore_shadow_enabled(void);
void rtcore_set_alpha(int enabled);
int rtcore_alpha_enabled(void);

void rtcore_destroy(void* handle);

#ifdef __cplusplus
}
#endif

#endif // GONZALES_RTCORE_H
