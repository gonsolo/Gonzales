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
comptime VulkanRtSceneHandle = UnsafePointer[UInt8, MutAnyOrigin]

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
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mesh_count: Int64,
    point_counts: UnsafePointer[Int64, MutAnyOrigin],
    vertex_index_counts: UnsafePointer[Int64, MutAnyOrigin],
) -> VulkanRtSceneHandle:
    return external_call["vulkanrt_build_scene", VulkanRtSceneHandle,
        UnsafePointer[TriangleMesh_C, MutAnyOrigin], Int64,
        UnsafePointer[Int64, MutAnyOrigin], UnsafePointer[Int64, MutAnyOrigin]](
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
    out_t: UnsafePointer[Float32, MutAnyOrigin],
    out_mesh: UnsafePointer[Int32, MutAnyOrigin],
    out_triangle: UnsafePointer[Int32, MutAnyOrigin],
) -> Int32:
    return external_call["vulkanrt_trace_ray", Int32,
        VulkanRtSceneHandle,
        Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32,
        UnsafePointer[Float32, MutAnyOrigin], UnsafePointer[Int32, MutAnyOrigin],
        UnsafePointer[Int32, MutAnyOrigin]](
        scene, ox, oy, oz, dx, dy, dz, t_min, t_max, out_t, out_mesh, out_triangle)

# Destroys a scene built by vulkanrt_build_scene.
def vulkanrt_destroy_scene(scene: VulkanRtSceneHandle):
    external_call["vulkanrt_destroy_scene", NoneType, VulkanRtSceneHandle](scene)
