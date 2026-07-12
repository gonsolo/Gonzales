#ifndef GONZALES_VULKANINTEROP_H
#define GONZALES_VULKANINTEROP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Task #163 stage 1: CUDA/Vulkan GPU-side interop foundation. Proves that a
// buffer can be shared between Mojo's CUDA-compiled kernels and a Vulkan
// compute dispatch WITHOUT any CPU synchronization round trip -- the
// prerequisite for ever making Vulkan RT faster than gonzales's existing
// software-BVH GPU path (see project_vulkan_rt_backend memory: the naive
// vulkanrt_trace_rays approach used by --vulkan-rt-shade syncs the whole
// GPU and copies through host memory on every call, ~94x slower than the
// CUDA baseline it replaces).
//
// Mechanism: the interop buffer's memory is allocated by Vulkan with
// VK_KHR_external_memory_fd (exportable as a POSIX fd) and imported into
// CUDA via cudaImportExternalMemory/cudaExternalMemoryGetMappedBuffer,
// giving a CUDA device pointer that ALIASES the same physical memory a
// Vulkan compute shader can also see. Two VkSemaphores (also exported via
// VK_KHR_external_semaphore_fd and imported into CUDA) hand off between the
// two APIs entirely on the GPU timeline: CUDA signals semaphore A after
// enqueuing work that writes the buffer, Vulkan's queue (already submitted,
// waiting on A) proceeds once the GPU sees the signal, runs its shader,
// signals semaphore B; CUDA waits on B (enqueued on the same CUDA stream)
// before any subsequently-enqueued Mojo kernel reads the buffer. No
// ctx.synchronize() / cudaStreamSynchronize / vkQueueWaitIdle anywhere in
// this handoff -- everything after buffer/semaphore setup is async.

// Builds an interop context: a headless Vulkan device (ray-query/AS
// extensions from vulkanrt.cpp's device requirements, PLUS
// VK_KHR_external_memory_fd/VK_KHR_external_semaphore_fd), one
// CUDA-importable float buffer of `elem_count` elements, two CUDA-imported
// semaphores, and a compute pipeline (interop_double.comp: multiplies every
// element by 2 -- a trivial stand-in for real ray tracing, isolating the
// interop mechanics from any actual RT-core work for this first stage).
// Returns NULL on failure (extension/interop unsupported, import failed).
void* vulkaninterop_create(int64_t elem_count);

// Returns the CUDA device pointer aliasing the interop buffer's memory --
// cast to whatever element type the caller wants (Mojo wraps it as a
// DeviceBuffer(ctx, ptr, count, owning=False) to read/write with ordinary
// enqueue_function kernels). NULL if `interop` is invalid.
void* vulkaninterop_get_cuda_ptr(void* interop);

// Enqueues one GPU-side round trip on `cuda_stream` (a real CUstream/
// cudaStream_t, e.g. from Mojo's CUDA(ctx.stream()) accessor): signal
// semaphore A on the stream, submit the pre-recorded Vulkan compute
// dispatch (waits on A, signals B), wait on semaphore B on the stream.
// Does NOT block the calling CPU thread on GPU completion -- vkQueueSubmit
// only blocks for CPU-side command submission, and the cuda*Async calls
// only enqueue onto the stream. Returns 1 on success, 0 on failure.
int vulkaninterop_round_trip(void* interop, void* cuda_stream);

// Destroys an interop context built by vulkaninterop_create.
void vulkaninterop_destroy(void* interop);

#ifdef __cplusplus
}
#endif

#endif // GONZALES_VULKANINTEROP_H
