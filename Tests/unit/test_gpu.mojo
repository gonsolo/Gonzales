from std.math import abs
from std.sys.info import size_of
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from gonzales.gpu import clear_film_gpu, accumulate_film_gpu
from gonzales.geometry import RGB, Point3f, Vec3f, Ray_C, PathState_C

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _dummy_path(estimate: RGB, albedo: RGB) -> PathState_C:
    return PathState_C(
        Ray_C(Point3f(0.0), Vec3f(0.0, 0.0, 1.0)),
        RGB(Float32(1.0)),  # throughput
        estimate,
        albedo,
        Int32(0),           # bounce
        UInt64(1),          # pcgState
        UInt64(1),          # pcgInc
        Int8(1),            # active
        Int8(0),            # specularBounce
        Int8(0),            # pending_mat
        Int8(0),            # volume_scattered
        Float32(0.0),       # lastBsdfPdf
        Int32(-1),          # current_medium_idx
        Int32(0),           # sampler_dim
        UInt64(0),          # sobol_idx
    )

def test_clear_film_gpu_zeroes_a_dirty_buffer() raises:
    """Clear_film_gpu is the simplest real kernel in gpu.mojo — one thread
    per pixel, writes 3 zeros. Fill the film buffer with garbage first so a
    no-op kernel (or one that only clears some pixels) would be caught."""
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return
    var ctx = DeviceContext()
    var n_pixels = 8
    var buf = ctx.enqueue_create_buffer[DType.float32](n_pixels * 3)
    buf.enqueue_fill(Float32(123.456))
    ctx.synchronize()

    ctx.enqueue_function[clear_film_gpu](
        buf.unsafe_ptr(), n_pixels,
        grid_dim=1, block_dim=n_pixels,
    )
    ctx.synchronize()

    with buf.map_to_host() as host:
        var p = host.unsafe_ptr()
        for i in range(n_pixels * 3):
            assert_true(p[i] == Float32(0.0))

def test_accumulate_film_gpu_adds_path_estimates_onto_existing_film() raises:
    """Accumulate_film_gpu does film[px] += path.estimate (and albedo_film
    += path.albedo), NOT an overwrite — pre-fill the film with a nonzero
    baseline so an accidental `=` instead of `+=` would be caught, and give
    each path a distinct estimate/albedo so a wrong-index bug would too."""
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return
    var ctx = DeviceContext()
    var n = 4

    var path_bytes = n * size_of[PathState_C]()
    var path_buf = ctx.enqueue_create_buffer[DType.uint8](path_bytes)
    with path_buf.map_to_host() as host:
        var paths = host.unsafe_ptr().bitcast[PathState_C]()
        for i in range(n):
            var f = Float32(i)
            paths[i] = _dummy_path(
                RGB(f * Float32(0.1), f * Float32(0.2), f * Float32(0.3)),
                RGB(f * Float32(0.01), f * Float32(0.02), f * Float32(0.03)),
            )

    var film_buf = ctx.enqueue_create_buffer[DType.float32](n * 3)
    var albedo_buf = ctx.enqueue_create_buffer[DType.float32](n * 3)
    film_buf.enqueue_fill(Float32(10.0))
    albedo_buf.enqueue_fill(Float32(1.0))
    ctx.synchronize()

    ctx.enqueue_function[accumulate_film_gpu](
        path_buf.unsafe_ptr().bitcast[PathState_C](),
        film_buf.unsafe_ptr(), albedo_buf.unsafe_ptr(), n,
        grid_dim=1, block_dim=n,
    )
    ctx.synchronize()

    with film_buf.map_to_host() as fh:
        var film = fh.unsafe_ptr()
        with albedo_buf.map_to_host() as ah:
            var alb = ah.unsafe_ptr()
            for i in range(n):
                var f = Float32(i)
                assert_true(_close(film[i*3+0], Float32(10.0) + f * Float32(0.1)))
                assert_true(_close(film[i*3+1], Float32(10.0) + f * Float32(0.2)))
                assert_true(_close(film[i*3+2], Float32(10.0) + f * Float32(0.3)))
                assert_true(_close(alb[i*3+0], Float32(1.0) + f * Float32(0.01)))
                assert_true(_close(alb[i*3+1], Float32(1.0) + f * Float32(0.02)))
                assert_true(_close(alb[i*3+2], Float32(1.0) + f * Float32(0.03)))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
