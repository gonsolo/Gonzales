from std.ffi import external_call
from std.gpu import block_idx, thread_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.host._nvidia_cuda import CUDA, CUstream

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

comptime VulkanInteropHandle = UnsafePointer[UInt8, MutAnyOrigin]

def vulkaninterop_create(elem_count: Int64) -> VulkanInteropHandle:
    return external_call["vulkaninterop_create", VulkanInteropHandle, Int64](elem_count)

# Returns the CUDA device pointer aliasing the interop buffer's memory.
# Wrap it with `DeviceBuffer[DType.float32](ctx, ptr, elem_count,
# owning=False)` to read/write it with ordinary enqueue_function kernels --
# `owning=False` is essential, this memory is owned by the interop context
# (freed by vulkaninterop_destroy), not by the DeviceBuffer wrapper.
def vulkaninterop_get_cuda_ptr(interop: VulkanInteropHandle) -> UnsafePointer[Float32, MutAnyOrigin]:
    return external_call["vulkaninterop_get_cuda_ptr", UnsafePointer[Float32, MutAnyOrigin],
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
def vulkaninterop_fill_kernel_test(data: UnsafePointer[Float32, MutAnyOrigin], n: Int32, base: Float32):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid < Int(n):
        data[tid] = base + Float32(tid)
