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

#ifdef __cplusplus
}
#endif

#endif // GONZALES_VULKANRT_H
