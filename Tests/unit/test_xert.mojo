from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext
from gonzales.xert import xert_build_blas, xert_alignment, xert_free, xert_trace_rays_gpu, XERT_MISS

# Intel Xe2 hardware ray tracing: the driver builds a BLAS for the quad [-1, 1]^2 at z = 0 and a grid of rays straight
# down from z = 2 must hit exactly the grid points inside it, at t = 2, with barycentrics that give back the hit point.
# Skips without an Intel GPU (xert_alignment() == 0).

comptime N = 64

def test_xert_quad() raises:
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return
    if xert_alignment() == 0:
        print("SKIP: no Intel GPU with ray tracing")
        return
    var corners: List[Float32] = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0]
    var tv = unsafe_alloc[Float32](18)
    for k in range(18):
        tv[unsafe_offset=k] = corners[k]
    var size = Int64(0)
    var rtas = xert_build_blas(tv, 2, Pointer(to=size).unsafe_origin_cast[MutUntrackedOrigin]())
    assert_true(Int(rtas) != 0, "xert_build_blas failed")
    var ctx = DeviceContext()
    var accel = ctx.enqueue_create_buffer[DType.uint8](Int(size))
    with accel.map_to_host() as h:
        for k in range(Int(size)):
            h[k] = rtas[unsafe_offset=k]
    xert_free(rtas)
    var rays = ctx.enqueue_create_buffer[DType.float32](8 * N * N)
    with rays.map_to_host() as h:
        for i in range(N * N):
            var r: List[Float32] = [Float32(i % N) / Float32(N - 1) * 3.0 - 1.5, Float32(i // N) / Float32(N - 1) * 3.0 - 1.5,
                                    2.0, 0.0, 0.0, 0.0, -1.0, 1.0e30]
            for k in range(8):
                h[8 * i + k] = r[k]
    var results = ctx.enqueue_create_buffer[DType.uint32](8 * N * N)
    ctx.enqueue_function[xert_trace_rays_gpu](accel.unsafe_ptr(), rays.unsafe_ptr(), results.unsafe_ptr(), Int64(N * N), UInt32(0),
                                              grid_dim=(N * N + 63) // 64, block_dim=64)
    var hits = 0; var prims = List[Int](length=2, fill=0); var bad = 0
    with results.map_to_host() as h:
        with rays.map_to_host() as rh:
            for i in range(N * N):
                var word = h[8 * i + 3]
                if word == XERT_MISS:
                    continue
                hits += 1
                var t = bitcast[DType.float32](h[8 * i])
                var u = bitcast[DType.float32](h[8 * i + 1])
                var v = bitcast[DType.float32](h[8 * i + 2])
                var p = Int(word)
                if p > 1 or abs(t - 2.0) > 1.0e-5:
                    bad += 1
                    continue
                prims[p] += 1
                # Corners of triangle p: (v0, v1, v2) = corners[9p ...]; the hit point is (1 - u - v) v0 + u v1 + v v2.
                var x = (1 - u - v) * corners[9 * p] + u * corners[9 * p + 3] + v * corners[9 * p + 6]
                var y = (1 - u - v) * corners[9 * p + 1] + u * corners[9 * p + 4] + v * corners[9 * p + 7]
                if abs(x - rh[8 * i]) > 1.0e-4 or abs(y - rh[8 * i + 1]) > 1.0e-4:
                    bad += 1
    _ = accel^
    tv.unsafe_free()
    assert_equal(hits, 42 * 42)   # x = k / 63 * 3 - 1.5 lies in [-1, 1] for k = 11 ... 52
    assert_equal(bad, 0)
    assert_true(prims[0] > hits // 3 and prims[1] > hits // 3, "both triangles hit")

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
