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

// ---------------------------------------------------------------------------
// Stage 2: real VK_KHR_ray_query tracing through the same GPU-side interop
// mechanism proven above -- replaces interop_double.comp's trivial "double
// every element" shader with intersect_batch.comp (the same ray-query
// compute shader vulkanrt.cpp's vulkanrt_trace_rays uses), reusing the SAME
// external-memory/semaphore handoff so a full render's per-bounce
// intersection can run with zero CPU synchronization, unlike
// --vulkan-rt-shade's vulkanrt_trace_rays (which syncs the whole GPU and
// copies through host memory every call -- see project_vulkan_rt_backend
// memory for the ~94x-slower measurement that motivated this).
// ---------------------------------------------------------------------------

// Mirrors gonzales/geometry.mojo's TriangleMesh_C field-for-field (see
// vulkanrt.h's VulkanRtMesh for the full explanation -- redefined here
// rather than shared across bridge libraries, matching this codebase's
// existing per-bridge independence convention).
typedef struct {
    const float* points;
    const int64_t* faceIndices;
    const int64_t* vertexIndices;
    const float* uvs;
    const float* normals;
} VulkanInteropMesh;

// Builds an interop-AND-ray-query-capable Vulkan device, a real BLAS/TLAS
// scene, and two CUDA-importable buffers sized for up to `max_rays`
// rays/results at once: a rays buffer (max_rays * 8 floats, same
// (ox,oy,oz,tmin,dx,dy,dz,tmax) layout vulkanrt_trace_rays takes) and a
// results buffer (max_rays * 32 bytes, matching intersect_batch.comp's
// Result struct: float hitT,u,v,pad; int32 hitMesh,hitTriangle,geometryIndex;
// uint32 hitFlag). Returns NULL on failure.
//
// Object instancing (mirrors gonzales's own two-level BVH, see
// geometry.mojo's Instance_C / pbrt_parser.mojo's ObjectBegin/ObjectEnd/
// ObjectInstance handling): `meshes[i]` for i in any
// [template_mesh_start[t], template_mesh_end[t]) range is a TEMPLATE mesh
// -- excluded from the ordinary one-BLAS-per-mesh-with-identity-transform
// treatment every other mesh gets, and instead merged (as separate
// geometries, one per mesh, within ONE BLAS) into template t's own BLAS.
// Each of `instance_count` placements gets its own TLAS instance
// referencing template `instance_template_idx[k]`'s BLAS with the
// (column-major, 16-float, gonzales's own transform.mojo convention)
// transform `instance_obj_to_world[k*16 .. k*16+16)`. Pass template_count=0/
// instance_count=0 (with the corresponding arrays possibly NULL) for a
// scene with no instancing -- byte-identical to the pre-instancing
// behavior. A hit's instanceCustomIndex is `i` (< mesh_count) for an
// ordinary mesh, or `mesh_count + k` (instance index k) for an instance --
// see intersect_batch.comp / vulkaninterop_unpack_results_kernel (gpu.mojo)
// for the decode.
//
// Curves (gonzales's deferred-candidate scheme, see bvh.mojo's
// traverse_bvh2_core_defer_curves / gpu.mojo's resolve_curve_candidates_gpu):
// curves are NOT tessellated into triangle geometry. Instead
// `curve_leaf_aabbs` (n_curve_leaves * 6 floats: xmin,ymin,zmin,xmax,ymax,
// zmax, one box per curve BVH leaf -- a leaf may cover several
// locally-linear curve pieces, see pbrt_parser.mojo's curve-group bounds)
// becomes ONE procedural-AABB BLAS + TLAS instance, distinguishable from
// ordinary triangle geometry by intersection TYPE
// (gl_RayQueryCandidateIntersectionAABBEXT), not by instanceCustomIndex.
// `curve_leaf_prim_idx[i]` is leaf i's index into gonzales's own top-level
// `prim_ids` array (exactly the `primIdx` resolve_curve_candidates_gpu
// expects) -- uploaded as a plain read-only storage buffer the shader
// indexes by `rayQueryGetIntersectionPrimitiveIndexEXT`. Every ray's AABB
// candidates (up to CURVE_DEFER_K, matching geometry.mojo's constant) are
// recorded into the two curve-candidate buffers below by the shader itself,
// during the SAME dispatch that resolves the committed triangle/instance
// hit -- they never affect that committed hit (AABB candidates are never
// confirmed/generated), so the ordinary hit-decode path is unaffected.
// Pass n_curve_leaves=0 (arrays may then be NULL) for a scene with no
// curves -- byte-identical to before curve support existed.
void* vulkaninterop_rt_create_scene(
    const VulkanInteropMesh* meshes,
    int64_t mesh_count,
    const int64_t* point_counts,
    const int64_t* vertex_index_counts,
    int64_t template_count,
    const int64_t* template_mesh_start,
    const int64_t* template_mesh_end,
    int64_t instance_count,
    const float* instance_obj_to_world,
    const int32_t* instance_template_idx,
    int64_t n_curve_leaves,
    const float* curve_leaf_aabbs,
    const int64_t* curve_leaf_prim_idx,
    int64_t max_rays);

// CUDA device pointer for the shared curve-candidate buffer (max_rays *
// CURVE_DEFER_K int32s, row-major per ray -- same layout as gpu.mojo's
// curve_cand_prim_buf) -- a Mojo kernel copies this into
// curve_cand_prim_buf after the ray-query dispatch (see
// vulkaninterop_rt_traverse_paths_gpu). NULL if the scene has no curves.
void* vulkaninterop_rt_get_curve_cand_ptr(void* scene);

// CUDA device pointer for the shared curve-candidate COUNT buffer
// (max_rays int32s, one per ray -- same layout as gpu.mojo's
// curve_cand_count_buf). NULL if the scene has no curves. Curve candidates
// are no longer capped per-ray at CURVE_DEFER_K: the shader traces each ray
// twice (count, then place into a shared pool via a workgroup-local
// prefix sum + one atomicAdd per 64-ray workgroup) so a ray in a dense
// curl can use far more than the per-ray average by borrowing capacity
// left unused by sparse/missing rays. This count is now that ray's real,
// uncapped total (bounded only by the whole scene's shared pool capacity).
void* vulkaninterop_rt_get_curve_cand_count_ptr(void* scene);

// CUDA device pointer for the shared curve-candidate OFFSET buffer
// (max_rays int32s, one per ray): this ray's start offset into the flat
// curve-candidate pool (vulkaninterop_rt_get_curve_cand_ptr) -- read
// alongside curveCandCount to know exactly which slice of the pool belongs
// to this ray. NULL if the scene has no curves.
void* vulkaninterop_rt_get_curve_cand_offset_ptr(void* scene);

// CUDA device pointer for the shared rays buffer -- a Mojo kernel writes
// ray data directly into this (no host copy) before calling
// vulkaninterop_rt_trace.
void* vulkaninterop_rt_get_rays_ptr(void* scene);

// CUDA device pointer for the shared results buffer -- a Mojo kernel reads
// hit results directly from this (no host copy) after
// vulkaninterop_rt_trace's GPU-side handoff completes (ordering is
// guaranteed by enqueuing that read on the same cuda_stream, same as
// vulkaninterop_round_trip -- no ctx.synchronize() needed).
void* vulkaninterop_rt_get_results_ptr(void* scene);

// Enqueues one GPU-side ray-query dispatch on `cuda_stream`, tracing the
// first `ray_count` rays (<= max_rays) from the rays buffer into the
// results buffer -- same async CUDA-signal -> Vulkan-dispatch-wait ->
// CUDA-wait handoff as vulkaninterop_round_trip, just running
// intersect_batch.comp instead of interop_double.comp. Returns 1 on
// success, 0 on failure. Does not block on GPU completion.
int vulkaninterop_rt_trace(void* scene, int32_t ray_count, void* cuda_stream);

// Destroys a scene built by vulkaninterop_rt_create_scene.
void vulkaninterop_rt_destroy_scene(void* scene);

#ifdef __cplusplus
}
#endif

#endif // GONZALES_VULKANINTEROP_H
