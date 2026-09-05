// CUDA-free stand-in for vulkaninterop.cpp.
//
// The real bridge shares a buffer between Mojo's CUDA kernels and a Vulkan
// compute dispatch, so it needs the CUDA toolkit's headers and libcudart. A
// machine without CUDA installed -- CI's container, or anyone building only
// the CPU renderer -- could not build gonzales AT ALL because of that: the
// Mojo link line always pulls in -lvulkaninterop, so a missing
// cuda_runtime.h failed the whole build rather than just this one optional
// backend. (That is what had CI red; see the Makefile's CUDA detection.)
//
// This provides the same exported symbols and reports "unavailable" from
// every one of them. Callers already handle that: vulkaninterop_create and
// vulkaninterop_rt_create_scene are documented to return NULL on failure
// (extension unsupported, import failed), and the trace/round-trip entry
// points return 0. So a CUDA-less build behaves exactly like a machine
// whose driver does not support the interop extensions -- the CUDA/Vulkan
// interop path stays off, and everything else builds and runs normally.
#include "vulkaninterop.h"

#include <stddef.h>

extern "C" {

void* vulkaninterop_create(int64_t) { return nullptr; }
void* vulkaninterop_get_cuda_ptr(void*) { return nullptr; }
int   vulkaninterop_round_trip(void*, void*) { return 0; }
void  vulkaninterop_destroy(void*) {}

void* vulkaninterop_rt_create_scene(
    const VulkanInteropMesh*, int64_t, const int64_t*, const int64_t*,
    int64_t, const int64_t*, const int64_t*,
    int64_t, const float*, const int32_t*,
    int64_t, const float*, const int32_t*, const int32_t*, const int32_t*,
    int64_t, const float*, const int32_t*,
    int64_t) { return nullptr; }

void* vulkaninterop_rt_get_rays_ptr(void*) { return nullptr; }
void* vulkaninterop_rt_get_results_ptr(void*) { return nullptr; }
int   vulkaninterop_rt_trace(void*, int32_t, void*) { return 0; }
void  vulkaninterop_rt_destroy_scene(void*) {}

}  // extern "C"
