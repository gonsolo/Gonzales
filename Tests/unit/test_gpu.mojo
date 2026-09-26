from std.math import abs
from std.sys.info import size_of
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from gonzales.gpu_wavefront import clear_film_gpu, accumulate_film_gpu
from gonzales.geometry import RGB, Point3f, Vec3f
from gonzales.render_state import PathState
from gonzales.primitives import Ray
from gonzales.spectrum import SpectralSample, SampledWavelengths

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# Path transport is spectral (PathState.throughput/estimate are
# SpectralSample). These fixtures use a null spectral handle, under which
# spectrum.mojo's conversions carry plain R/G/B on lanes v0/v1/v2 (see
# rgb_to_spectral_sample's table-less fallback) -- so the assertions below
# read those lanes and mean exactly what the old RGB assertions meant.
def _dummy_path(estimate: SpectralSample, albedo: RGB) -> PathState:
    return PathState(
        Ray(Point3f(0.0), Vec3f(0.0, 0.0, 1.0)),
        SpectralSample(Float32(1.0)),  # throughput
        estimate,
        albedo,
        Int32(0),           # bounce
        UInt64(1),          # pcgState
        UInt64(1),          # pcgInc
        Int8(1),            # active
        Int8(0),            # specularBounce
        Int8(0),            # pending_mat
        Int8(0),            # volume_scattered
        Int8(0),            # at_cap
        Vec3f(Float32(0.0)),  # last_ns_n
        Vec3f(Float32(0.0)),  # last_ns_p
        Float32(0.0),       # lastBsdfPdf
        Int32(-1),          # current_medium_idx
        Float32(1.0),       # current_dielectric_ior
        Float32(1.0),       # previous_dielectric_ior
        Float32(1.0),       # eta_scale
        Int32(0),           # sampler_dim
        UInt64(0),          # sobol_idx
        SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),  # wavelengths
        Float32(0.0),   # mis_null_dist
        Float32(0.0),   # lastEnvNeePdf
        Float32(0.0),   # cone_len
    )

def _clear_film_gpu_body(ctx: DeviceContext) raises:
    """Clear_film_gpu is the simplest real kernel in gpu.mojo — one thread
    per pixel, writes 3 zeros. Fill the film buffer with garbage first so a
    no-op kernel (or one that only clears some pixels) would be caught."""
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return
    var n_pixels = 8
    var buf = ctx.enqueue_create_buffer[DType.float32](n_pixels * 3)
    buf.enqueue_fill(Float32(123.456))
    ctx.synchronize()

    ctx.enqueue_function[clear_film_gpu](
        buf.unsafe_ptr(), Int64(n_pixels),
        grid_dim=1, block_dim=n_pixels,
    )
    ctx.synchronize()

    with buf.map_to_host() as host:
        var p = host.unsafe_ptr()
        for i in range(n_pixels * 3):
            assert_true(p[unsafe_offset=i] == Float32(0.0))

def _accumulate_film_gpu_body(ctx: DeviceContext) raises:
    """Accumulate_film_gpu does film[px] += path.estimate (and albedo_film
    += path.albedo), NOT an overwrite — pre-fill the film with a nonzero
    baseline so an accidental `=` instead of `+=` would be caught, and give
    each path a distinct estimate/albedo so a wrong-index bug would too."""
    var n = 4

    var path_bytes = n * size_of[PathState]()
    var path_buf = ctx.enqueue_create_buffer[DType.uint8](path_bytes)
    with path_buf.map_to_host() as host:
        var paths = host.unsafe_ptr().unsafe_bitcast[PathState]()
        for i in range(n):
            var f = Float32(i)
            paths[unsafe_offset=i] = _dummy_path(
                SpectralSample(f * Float32(0.1), f * Float32(0.2), f * Float32(0.3), Float32(0.0)),
                RGB(f * Float32(0.01), f * Float32(0.02), f * Float32(0.03)),
            )

    var film_buf = ctx.enqueue_create_buffer[DType.float32](n * 3)
    var albedo_buf = ctx.enqueue_create_buffer[DType.float32](n * 3)
    film_buf.enqueue_fill(Float32(10.0))
    albedo_buf.enqueue_fill(Float32(1.0))
    ctx.synchronize()

    # A GPU kernel launch cannot use the signature's default arguments, so the
    # spectral tables are passed explicitly. The null handle (res 0 + dangling
    # pointers) selects spectral_sample_to_rgb's table-less path, which reads
    # R/G/B straight off lanes v0/v1/v2 -- so the film assertions below are
    # exactly the per-channel values this fixture wrote into `estimate`.
    var null_tbl = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
    ctx.enqueue_function[accumulate_film_gpu](
        path_buf.unsafe_ptr().unsafe_bitcast[PathState](),
        film_buf.unsafe_ptr(), albedo_buf.unsafe_ptr(), Int64(n),
        Float32(0.0),   # sample_clamp disabled: this test checks accumulation,
                        # not pbrt's per-sample maxcomponentvalue clamp
        null_tbl, Int64(0), null_tbl, null_tbl, null_tbl, null_tbl,
        grid_dim=1, block_dim=n,
    )
    ctx.synchronize()

    with film_buf.map_to_host() as fh:
        var film = fh.unsafe_ptr()
        with albedo_buf.map_to_host() as ah:
            var alb = ah.unsafe_ptr()
            for i in range(n):
                var f = Float32(i)
                assert_true(_close(film[unsafe_offset=i*3+0], Float32(10.0) + f * Float32(0.1)))
                assert_true(_close(film[unsafe_offset=i*3+1], Float32(10.0) + f * Float32(0.2)))
                assert_true(_close(film[unsafe_offset=i*3+2], Float32(10.0) + f * Float32(0.3)))
                assert_true(_close(alb[unsafe_offset=i*3+0], Float32(1.0) + f * Float32(0.01)))
                assert_true(_close(alb[unsafe_offset=i*3+1], Float32(1.0) + f * Float32(0.02)))
                assert_true(_close(alb[unsafe_offset=i*3+2], Float32(1.0) + f * Float32(0.03)))

def test_clear_film_gpu_and_accumulate_film_gpu() raises:
    """Both real kernels in this file, run against ONE shared DeviceContext
    -- a real, narrow, 100%-reproducible Modular/Mojo runtime bug (confirmed
    via a ~10-line minimal repro, isolated with gdb to a deadlock inside
    AsyncRT_DeviceContext_createBuffer_async) makes a SECOND independent
    DeviceContext's very first enqueue_create_buffer hang forever if an
    EARLIER DeviceContext in the same process ever called
    DeviceBuffer.map_to_host() on a buffer it OWNS (one it allocated itself
    via enqueue_create_buffer -- a non-owning DeviceBuffer wrapping
    externally-allocated memory, e.g. the Vulkan-interop tests' pattern,
    does NOT trigger it). Production code only ever creates one
    DeviceContext per process, so this has zero real-render impact -- it's
    purely a multi-test-in-one-file hazard, worked around here by giving
    both kernels one shared context instead of one each. See
    project_modular_26_5_0_migration memory for the full bisection."""
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return
    var ctx = DeviceContext()
    _clear_film_gpu_body(ctx)
    _accumulate_film_gpu_body(ctx)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
