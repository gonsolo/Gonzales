from std.ffi import external_call
from std.gpu import block_idx, thread_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host._nvidia_cuda import CUDA, CUstream
from .geometry import TriangleMesh_C

# Task #163 stage 1: Mojo-side FFI wrapper for the CUDA/Vulkan GPU-side
# interop bridge (src/vulkaninterop/vulkaninterop.cpp). Proves the whole
# chain works from gonzales's own Mojo GPU kernels, not just a standalone
# C/CUDA test: a Mojo enqueue_function kernel writes the shared buffer, the
# Vulkan compute dispatch transforms it, another Mojo kernel reads it back
# -- all on Mojo's own DeviceContext stream, with zero ctx.synchronize()
# calls in between. See vulkaninterop.h and project_vulkan_rt_backend
# memory for the full rationale (this is the prerequisite for a Vulkan RT
# backend that's actually faster than gonzales's existing software-BVH GPU
# path, not just correct).

comptime VulkanInteropHandle = UnsafePointer[UInt8, MutExternalOrigin]

def vulkaninterop_create(elem_count: Int64) -> VulkanInteropHandle:
    return external_call["vulkaninterop_create", VulkanInteropHandle, Int64](elem_count)

# Returns the CUDA device pointer aliasing the interop buffer's memory.
# Wrap it with `DeviceBuffer[DType.float32](ctx, ptr, elem_count,
# owning=False)` to read/write it with ordinary enqueue_function kernels --
# `owning=False` is essential, this memory is owned by the interop context
# (freed by vulkaninterop_destroy), not by the DeviceBuffer wrapper.
def vulkaninterop_get_cuda_ptr(interop: VulkanInteropHandle) -> UnsafePointer[Float32, MutExternalOrigin]:
    return external_call["vulkaninterop_get_cuda_ptr", UnsafePointer[Float32, MutExternalOrigin],
        VulkanInteropHandle](interop)

# Enqueues one GPU-side round trip on `cuda_stream` (get one via
# `CUDA(ctx.stream())`). Returns 1 on success, 0 on failure. Does not block
# the calling thread on GPU completion -- see vulkaninterop.h.
def vulkaninterop_round_trip(interop: VulkanInteropHandle, cuda_stream: CUstream) -> Int32:
    return external_call["vulkaninterop_round_trip", Int32,
        VulkanInteropHandle, CUstream](interop, cuda_stream)

def vulkaninterop_destroy(interop: VulkanInteropHandle):
    external_call["vulkaninterop_destroy", NoneType, VulkanInteropHandle](interop)

# Minimal GPU kernel used only by Tests/unit/test_vulkaninterop.mojo to
# exercise the interop buffer from a real Mojo enqueue_function kernel
# (proving the whole chain works from gonzales's own GPU code, not just a
# standalone C/CUDA test) -- lives in the package rather than the test file
# itself since fresh kernel definitions at `mojo run` top level for a
# standalone test script were not picked up for GPU codegen correctly.
def vulkaninterop_fill_kernel_test(data: UnsafePointer[Float32, MutExternalOrigin], n: Int32, base: Float32):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid < Int(n):
        data[tid] = base + Float32(tid)

# ---------------------------------------------------------------------------
# Stage 2: real VK_KHR_ray_query tracing through the same interop mechanism.
# ---------------------------------------------------------------------------

comptime VulkanInteropRtSceneHandle = UnsafePointer[UInt8, MutExternalOrigin]

# Builds an interop-AND-ray-query-capable scene from gonzales's own
# TriangleMesh_C data (same field-for-field-mirror trick as vulkanrt.mojo's
# vulkanrt_build_scene -- VulkanInteropMesh in vulkaninterop.h matches
# TriangleMesh_C's layout exactly, no repacking needed) -- one real BLAS
# per ordinary mesh (identity-transform TLAS instance) + one multi-geometry
# BLAS per object-instancing template (template_mesh_start/end mark which
# mesh-index ranges are template-only) + one TLAS instance per
# ObjectInstance placement (instance_obj_to_world, column-major 16 floats
# per instance, gonzales's own transform.mojo convention;
# instance_template_idx says which template's BLAS it references) + two
# CUDA-importable buffers sized for up to `max_rays` rays/results at once.
# Pass template_count=0/instance_count=0 (arrays may then be a
# `.unsafe_dangling()` placeholder) for a scene with no instancing --
# byte-identical to the pre-instancing behavior. Curves are NOT tessellated
# -- one procedural-AABB BLAS carrying each curve BVH leaf's bounding box.
# Unlike earlier designs (a deferred-candidate buffer of some kind, always
# finitely capped), intersect_batch.comp now does the real ray-vs-curve
# narrow-phase test itself and commits a valid hit inline (rayQueryGenerate
# IntersectionEXT), so no candidate buffer of any kind is needed -- just the
# curve geometry itself (curve_data/curve_n_pieces, one entry per curve) and
# 3 parallel per-leaf arrays (curve_leaf_curve_idx/piece_info/mat_idx) the
# shader uses to test a candidate and, on a hit, report enough for gpu.
# mojo's vulkaninterop_unpack_results_kernel to reconstruct a full PrimId_C
# with no further lookups. See vulkaninterop.h's own docstring for the
# exact per-leaf/per-curve encoding (matches pbrt_parser.mojo's finalize_
# scene exactly) and the hit-decode convention (hitFlag==2). Pass
# n_curve_leaves=0/n_curves=0 for a scene with no curves -- byte-identical
# to before curve support existed. Returns a null handle on failure.
def vulkaninterop_rt_create_scene(
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    mesh_count: Int64,
    point_counts: UnsafePointer[Int64, MutExternalOrigin],
    vertex_index_counts: UnsafePointer[Int64, MutExternalOrigin],
    template_count: Int64,
    template_mesh_start: UnsafePointer[Int64, MutExternalOrigin],
    template_mesh_end: UnsafePointer[Int64, MutExternalOrigin],
    instance_count: Int64,
    instance_obj_to_world: UnsafePointer[Float32, MutExternalOrigin],
    instance_template_idx: UnsafePointer[Int32, MutExternalOrigin],
    n_curve_leaves: Int64,
    curve_leaf_aabbs: UnsafePointer[Float32, MutExternalOrigin],
    curve_leaf_curve_idx: UnsafePointer[Int32, MutExternalOrigin],
    curve_leaf_piece_info: UnsafePointer[Int32, MutExternalOrigin],
    curve_leaf_mat_idx: UnsafePointer[Int32, MutExternalOrigin],
    n_curves: Int64,
    curve_data: UnsafePointer[Float32, MutExternalOrigin],
    curve_n_pieces: UnsafePointer[Int32, MutExternalOrigin],
    max_rays: Int64,
) -> VulkanInteropRtSceneHandle:
    return external_call["vulkaninterop_rt_create_scene", VulkanInteropRtSceneHandle,
        UnsafePointer[TriangleMesh_C, MutExternalOrigin], Int64,
        UnsafePointer[Int64, MutExternalOrigin], UnsafePointer[Int64, MutExternalOrigin],
        Int64, UnsafePointer[Int64, MutExternalOrigin], UnsafePointer[Int64, MutExternalOrigin],
        Int64, UnsafePointer[Float32, MutExternalOrigin], UnsafePointer[Int32, MutExternalOrigin],
        Int64, UnsafePointer[Float32, MutExternalOrigin],
        UnsafePointer[Int32, MutExternalOrigin], UnsafePointer[Int32, MutExternalOrigin], UnsafePointer[Int32, MutExternalOrigin],
        Int64, UnsafePointer[Float32, MutExternalOrigin], UnsafePointer[Int32, MutExternalOrigin],
        Int64](
        meshes, mesh_count, point_counts, vertex_index_counts,
        template_count, template_mesh_start, template_mesh_end,
        instance_count, instance_obj_to_world, instance_template_idx,
        n_curve_leaves, curve_leaf_aabbs,
        curve_leaf_curve_idx, curve_leaf_piece_info, curve_leaf_mat_idx,
        n_curves, curve_data, curve_n_pieces,
        max_rays)

# CUDA device pointer for the shared rays buffer (max_rays * 8 floats, same
# (ox,oy,oz,tmin,dx,dy,dz,tmax) layout vulkanrt_trace_rays takes). Wrap with
# DeviceBuffer[DType.float32](ctx, ptr, max_rays * 8, owning=False) to
# write ray data directly from a Mojo kernel -- no host round trip.
def vulkaninterop_rt_get_rays_ptr(scene: VulkanInteropRtSceneHandle) -> UnsafePointer[Float32, MutExternalOrigin]:
    return external_call["vulkaninterop_rt_get_rays_ptr", UnsafePointer[Float32, MutExternalOrigin],
        VulkanInteropRtSceneHandle](scene)

# CUDA device pointer for the shared results buffer (max_rays * 32 bytes,
# matching intersect_batch.comp's Result struct: float hitT,u,v,pad; int32
# hitMesh,hitTriangle,geometryIndex; uint32 hitFlag -- 8 x 4-byte fields).
# Wrap with DeviceBuffer[DType.float32](ctx, ptr, max_rays * 8,
# owning=False) and bitcast to read hitMesh/hitTriangle/hitFlag/
# geometryIndex as Int32/UInt32 (same raw-buffer-bitcast convention
# gpu.mojo already uses throughout) -- see
# vulkaninterop_unpack_results_kernel (gpu.mojo) for the instancing decode
# and the curve decode (hitFlag==2 -- fields mean something different there,
# see that kernel's own comment).
def vulkaninterop_rt_get_results_ptr(scene: VulkanInteropRtSceneHandle) -> UnsafePointer[Float32, MutExternalOrigin]:
    return external_call["vulkaninterop_rt_get_results_ptr", UnsafePointer[Float32, MutExternalOrigin],
        VulkanInteropRtSceneHandle](scene)

# Enqueues one GPU-side ray-query dispatch on `cuda_stream`, tracing the
# first `ray_count` rays (<= max_rays) from the rays buffer into the
# results buffer. Returns 1 on success, 0 on failure. Does not block on GPU
# completion -- see vulkaninterop.h.
def vulkaninterop_rt_trace(scene: VulkanInteropRtSceneHandle, ray_count: Int32, cuda_stream: CUstream) -> Int32:
    return external_call["vulkaninterop_rt_trace", Int32,
        VulkanInteropRtSceneHandle, Int32, CUstream](scene, ray_count, cuda_stream)

def vulkaninterop_rt_destroy_scene(scene: VulkanInteropRtSceneHandle):
    external_call["vulkaninterop_rt_destroy_scene", NoneType, VulkanInteropRtSceneHandle](scene)

# Minimal GPU kernel used only by Tests/unit/test_vulkaninterop_rt.mojo to
# write 3 known test rays directly into the interop rays buffer (same
# reasoning as vulkaninterop_fill_kernel_test for living in the package
# rather than the test file).
def vulkaninterop_rt_write_test_rays_kernel(rays: UnsafePointer[Float32, MutExternalOrigin]):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i == 0:
        # Hits mesh 0's triangle at (0.25, 0.25).
        rays[0] = Float32(0.25); rays[1] = Float32(0.25); rays[2] = Float32(-1.0); rays[3] = Float32(0.001)
        rays[4] = Float32(0.0);  rays[5] = Float32(0.0);  rays[6] = Float32(1.0);  rays[7] = Float32(10.0)
    elif i == 1:
        # Hits mesh 1's triangle at (5.25, 0.25).
        rays[8] = Float32(5.25); rays[9] = Float32(0.25); rays[10] = Float32(-1.0); rays[11] = Float32(0.001)
        rays[12] = Float32(0.0); rays[13] = Float32(0.0);  rays[14] = Float32(1.0); rays[15] = Float32(10.0)
    elif i == 2:
        # Misses both.
        rays[16] = Float32(100.0); rays[17] = Float32(100.0); rays[18] = Float32(-1.0); rays[19] = Float32(0.001)
        rays[20] = Float32(0.0);   rays[21] = Float32(0.0);   rays[22] = Float32(1.0);  rays[23] = Float32(10.0)
