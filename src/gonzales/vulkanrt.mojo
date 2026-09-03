from std.ffi import external_call
from .geometry import TriangleMesh_C

# Task #162 step 1: Mojo-side FFI wrapper for the headless Vulkan
# ray-query smoke test (src/vulkanrt/vulkanrt.cpp). Completes the
# Mojo -> C++ bridge -> Vulkan -> RT-core round trip this step exists to
# prove -- see project_vulkan_rt_backend memory for the full architecture
# plan this is the first concrete step of.
#
# Builds a single hardcoded triangle + a ray known to hit it entirely on
# the C++ side (no gonzales scene data crosses the FFI boundary yet), runs
# one ray-query compute dispatch on the GPU, and reports whether the
# hardware-traced hit matches the expected result.
# Returns 1 on a correct hit, 0 on any failure (extension unsupported,
# device creation failed, wrong/missing hit). Diagnostics go to stderr.
def vulkanrt_smoke_test() -> Int32:
    return external_call["vulkanrt_smoke_test", Int32]()

# Opaque handle to a C++-owned VulkanRtScene (build_scene/trace_ray/
# destroy_scene below). Mirrors viewer.mojo's ViewerHandle pattern -- a
# UInt8* stays away from the !kgen.pointer<none> representation Mojo
# rejects.
comptime VulkanRtSceneHandle = UnsafePointer[UInt8, MutExternalOrigin]

# Task #162 step 2: build a real Vulkan RT scene (one BLAS per mesh, one
# TLAS instancing all of them with an identity transform) directly from
# gonzales's own TriangleMesh_C data -- no repacking on the Mojo side, the
# C++ bridge reads TriangleMesh_C's points/vertexIndices pointers as-is
# (see vulkanrt.h's VulkanRtMesh, a field-for-field mirror of
# TriangleMesh_C). point_counts[i]/vertex_index_counts[i] give the vertex
# count and vertexIndices element count (3 * triangle count) for mesh i,
# same convention gpu.mojo's gpu_upload_scene already uses.
#
# Returns a null handle on failure (extension unsupported, device creation
# failed, AS build failed) -- check before calling vulkanrt_trace_ray.
def vulkanrt_build_scene(
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    mesh_count: Int64,
    point_counts: UnsafePointer[Int64, MutExternalOrigin],
    vertex_index_counts: UnsafePointer[Int64, MutExternalOrigin],
) -> VulkanRtSceneHandle:
    return external_call["vulkanrt_build_scene", VulkanRtSceneHandle,
        UnsafePointer[TriangleMesh_C, MutExternalOrigin], Int64,
        UnsafePointer[Int64, MutExternalOrigin], UnsafePointer[Int64, MutExternalOrigin]](
        meshes, mesh_count, point_counts, vertex_index_counts)

# Traces one ray against a scene built by vulkanrt_build_scene. Returns 1 on
# a hit (out_t/out_mesh/out_triangle filled in: out_mesh is the 0-based
# input mesh index, out_triangle the 0-based triangle index within that
# mesh -- the same (mesh, triangle) numbering gonzales's own PrimId_C uses),
# 0 on a miss.
def vulkanrt_trace_ray(
    scene: VulkanRtSceneHandle,
    ox: Float32, oy: Float32, oz: Float32,
    dx: Float32, dy: Float32, dz: Float32,
    t_min: Float32, t_max: Float32,
    out_t: UnsafePointer[Float32, MutExternalOrigin],
    out_mesh: UnsafePointer[Int32, MutExternalOrigin],
    out_triangle: UnsafePointer[Int32, MutExternalOrigin],
) -> Int32:
    return external_call["vulkanrt_trace_ray", Int32,
        VulkanRtSceneHandle,
        Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32,
        UnsafePointer[Float32, MutExternalOrigin], UnsafePointer[Int32, MutExternalOrigin],
        UnsafePointer[Int32, MutExternalOrigin]](
        scene, ox, oy, oz, dx, dy, dz, t_min, t_max, out_t, out_mesh, out_triangle)

# Destroys a scene built by vulkanrt_build_scene.
def vulkanrt_destroy_scene(scene: VulkanRtSceneHandle):
    external_call["vulkanrt_destroy_scene", NoneType, VulkanRtSceneHandle](scene)

# Task #162 step 3: trace many rays in ONE dispatch, instead of
# vulkanrt_trace_ray's one queue-submit-and-wait per ray (far too slow to
# ever back a real renderer). `rays` is a flat array of ray_count * 8
# floats, 8 per ray in order (ox, oy, oz, t_min, dx, dy, dz, t_max). Each
# out_* array must have ray_count elements; out_hit[i] is 1 on a hit, 0 on
# a miss (matching Intersection_C.hit's convention), with
# out_t/out_u/out_v/out_mesh/out_triangle only meaningful when out_hit[i]
# is 1. out_u/out_v use the same barycentric convention as gonzales's own
# Moller-Trumbore intersect_triangle (verified in
# Tests/unit/test_vulkanrt_batch.mojo).
#
# Returns 1 if the dispatch executed, 0 on failure (null scene, allocation
# failure) -- per-ray hit/miss is conveyed via out_hit, not this return
# value.
def vulkanrt_trace_rays(
    scene: VulkanRtSceneHandle,
    ray_count: Int32,
    rays: UnsafePointer[Float32, MutExternalOrigin],
    out_t: UnsafePointer[Float32, MutExternalOrigin],
    out_u: UnsafePointer[Float32, MutExternalOrigin],
    out_v: UnsafePointer[Float32, MutExternalOrigin],
    out_mesh: UnsafePointer[Int32, MutExternalOrigin],
    out_triangle: UnsafePointer[Int32, MutExternalOrigin],
    out_hit: UnsafePointer[UInt8, MutExternalOrigin],
) -> Int32:
    return external_call["vulkanrt_trace_rays", Int32,
        VulkanRtSceneHandle, Int32, UnsafePointer[Float32, MutExternalOrigin],
        UnsafePointer[Float32, MutExternalOrigin], UnsafePointer[Float32, MutExternalOrigin],
        UnsafePointer[Float32, MutExternalOrigin], UnsafePointer[Int32, MutExternalOrigin],
        UnsafePointer[Int32, MutExternalOrigin], UnsafePointer[UInt8, MutExternalOrigin]](
        scene, ray_count, rays, out_t, out_u, out_v, out_mesh, out_triangle, out_hit)
