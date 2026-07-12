#ifndef GONZALES_VULKANRT_H
#define GONZALES_VULKANRT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Mirrors gonzales/geometry.mojo's TriangleMesh_C field-for-field (5
// pointers, same order, no padding on either side) -- a
// UnsafePointer[TriangleMesh_C] from Mojo can be reinterpreted directly as
// a `const VulkanRtMesh*` with no conversion. faceIndices/uvs/normals are
// accepted for layout compatibility but unused by the scene builder below
// (it only needs points + vertexIndices for BLAS geometry).
typedef struct {
    const float* points;
    const int64_t* faceIndices;
    const int64_t* vertexIndices;
    const float* uvs;
    const float* normals;
} VulkanRtMesh;

// Task #162 step 1 smoke test: stand up a headless Vulkan instance/device
// requesting VK_KHR_ray_query + VK_KHR_acceleration_structure +
// VK_KHR_deferred_host_operations, build a single-triangle BLAS/TLAS, run
// one ray-query compute dispatch against it, and confirm the ray (hardcoded
// to pass through the triangle) actually reports a hit. No gonzales scene
// data involved yet -- this only proves the Mojo -> C++ bridge -> Vulkan ->
// RT-core round trip works end to end before building the real intersection
// backend on top of it.
//
// Returns 1 if the ray-query hit was reported correctly, 0 on any failure
// (extension unsupported, device creation failed, wrong/missing hit, etc).
// Prints diagnostics to stderr on failure.
int vulkanrt_smoke_test(void);

// Task #162 step 2: build a real Vulkan RT scene (one BLAS per input mesh,
// one TLAS instancing all of them with an identity transform -- gonzales
// already bakes each mesh's CTM into world-space point coordinates at parse
// time, see pbrt_parser.mojo's store_mesh, so no per-instance transform is
// needed here) from gonzales's own triangle mesh data, and trace real rays
// against it.
//
// meshes[i].points is an xyzw-interleaved float32 array (stride 16 bytes,
// w ignored); point_counts[i] is its vertex count. meshes[i].vertexIndices
// is a flat int64 array of triangle-vertex indices (a multiple of 3);
// vertex_index_counts[i] is its element count (== 3 * triangle count).
// Both arrays are read once during the build call and not touched again
// (no Mojo-side pointer needs to outlive the vulkanrt_build_scene call).
//
// Returns NULL on failure (extension unsupported, device creation failed,
// AS build failed). Diagnostics go to stderr.
void* vulkanrt_build_scene(
    const VulkanRtMesh* meshes,
    int64_t mesh_count,
    const int64_t* point_counts,
    const int64_t* vertex_index_counts);

// Traces one ray against a scene built by vulkanrt_build_scene. Returns 1 on
// a hit (out_t/out_mesh/out_triangle filled in: out_mesh is the 0-based
// input mesh index, out_triangle the 0-based triangle index within that
// mesh -- the same (mesh, triangle) numbering gonzales's own PrimId_C
// uses), 0 on a miss.
int vulkanrt_trace_ray(
    void* scene,
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float t_min, float t_max,
    float* out_t, int32_t* out_mesh, int32_t* out_triangle);

// Destroys a scene built by vulkanrt_build_scene.
void vulkanrt_destroy_scene(void* scene);

// Task #162 step 3: trace many rays in ONE dispatch, instead of
// vulkanrt_trace_ray's one queue-submit-and-wait per ray (far too slow to
// ever back a real renderer -- this is the batching step that makes Vulkan
// RT usable as an actual gonzales intersection backend).
//
// `rays` is a flat array of ray_count * 8 floats, 8 per ray in the order
// (ox, oy, oz, t_min, dx, dy, dz, t_max) -- the same layout
// vulkanrt_trace_ray takes as separate scalar args, just packed. Each
// out_* array must have ray_count elements; out_hit[i] is 1 on a hit, 0 on
// a miss (matching gonzales's own Intersection_C.hit convention), with
// out_t/out_u/out_v/out_mesh/out_triangle only meaningful when out_hit[i]
// is 1. out_u/out_v are the hardware's committed-intersection barycentric
// coordinates -- same (u, v) = (weight of vertex 1, weight of vertex 2)
// convention as gonzales's own Moller-Trumbore intersect_triangle, verified
// against it in Tests/unit/test_vulkanrt_batch.mojo.
//
// Returns 1 if the dispatch executed, 0 on failure (null scene, allocation
// failure) -- per-ray hit/miss is conveyed via out_hit, not this return
// value.
int vulkanrt_trace_rays(
    void* scene,
    int32_t ray_count,
    const float* rays,
    float* out_t, float* out_u, float* out_v,
    int32_t* out_mesh, int32_t* out_triangle, uint8_t* out_hit);

#ifdef __cplusplus
}
#endif

#endif // GONZALES_VULKANRT_H
