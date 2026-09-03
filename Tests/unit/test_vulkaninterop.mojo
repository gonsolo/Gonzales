from std.memory import alloc
from std.sys import has_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host._nvidia_cuda import CUDA
from std.testing import assert_true, TestSuite
from gonzales.vulkaninterop import (
    vulkaninterop_create, vulkaninterop_get_cuda_ptr,
    vulkaninterop_round_trip, vulkaninterop_destroy,
    vulkaninterop_fill_kernel_test,
)

# Task #163 stage 1: proves the CUDA/Vulkan GPU-side interop bridge works
# from gonzales's own Mojo GPU kernels (not just the standalone C/CUDA test
# used during development) -- a real Mojo enqueue_function kernel writes
# the shared buffer, the Vulkan compute dispatch (see
# src/vulkaninterop/shaders/interop_double.comp) doubles every element
# through the SAME aliased memory, and the result is read back and
# checked. No ctx.synchronize() happens between the write kernel and the
# round trip -- only after, to inspect results for this test's assertions.

comptime N = 4096

def test_vulkaninterop_round_trip_doubles_buffer_via_mojo_kernel() raises:
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return

    var interop = vulkaninterop_create(Int64(N))
    assert_true(Int(interop) != 0)

    var cuda_ptr = vulkaninterop_get_cuda_ptr(interop)
    assert_true(Int(cuda_ptr) != 0)

    var ctx = DeviceContext()
    var buf = DeviceBuffer[DType.float32](ctx, cuda_ptr, N, owning=False)

    comptime block_size = 256
    comptime grid_size = (N + block_size - 1) // block_size
    ctx.enqueue_function[vulkaninterop_fill_kernel_test](buf.unsafe_ptr(), Int32(N), Float32(3.0),
                                                           grid_dim=grid_size, block_dim=block_size)

    var cuda_stream = CUDA(ctx.stream())
    var rc = vulkaninterop_round_trip(interop, cuda_stream)
    assert_true(Int(rc) == 1)

    ctx.synchronize()
    with buf.map_to_host() as host:
        for i in range(N):
            var expected = (Float32(3.0) + Float32(i)) * Float32(2.0)
            assert_true(host[i] == expected)

    vulkaninterop_destroy(interop)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
