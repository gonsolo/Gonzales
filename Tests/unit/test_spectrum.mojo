from std.math import abs
from std.testing import assert_true, TestSuite
from gonzales.spectrum import (
    SampledWavelengths, sample_wavelengths_uniform,
    rgb_to_spectral_sample, rgb_illuminant_to_spectral_sample, spectral_sample_to_rgb,
    SpectralContext, SpectralHandle, spectral_handle, LAMBDA_MIN, LAMBDA_MAX, N_SPECTRAL_SAMPLES,
)
from gonzales.rgb2spec import build_spectrum_table, build_cie_xyz_tables, SpectrumTable

comptime EPS: Float32 = 1e-3

def _close(a: Float32, b: Float32, tol: Float32 = EPS) -> Bool:
    return abs(a - b) < tol

comptime TEST_RES = 16

def _test_ctx() -> SpectralContext:
    var table = build_spectrum_table(TEST_RES)
    var cie = build_cie_xyz_tables()
    return SpectralContext(SpectrumTable(table^, TEST_RES), cie^)

# Stratified average over many hero-wavelength draws — removes per-sample MC
# noise so these tests check the underlying round-trip math, not variance.
comptime N_TRIALS = 2000

# NOTE: SpectralHandle's fields are passed DECOMPOSED (not `handle:
# SpectralHandle` as one by-value struct arg) into every function below --
# see spectrum.mojo's comment above rgb_to_spectral_sample: passing that
# 6-field struct by value across a real Mojo function-call boundary is a
# confirmed, reproducible miscompilation. `handle` itself stays a local
# SpectralHandle var in each test (that's fine, no boundary crossed) and its
# fields are unpacked at each call site: handle.coeffs, handle.res,
# handle.cie_x, handle.cie_y, handle.cie_z, handle.d65.
def _roundtrip(
    coeffs: UnsafePointer[Float32, MutAnyOrigin], res: Int,
    cie_x: UnsafePointer[Float32, MutAnyOrigin], cie_y: UnsafePointer[Float32, MutAnyOrigin],
    cie_z: UnsafePointer[Float32, MutAnyOrigin], d65: UnsafePointer[Float32, MutAnyOrigin],
    r: Float32, g: Float32, b: Float32,
) -> Tuple[Float32, Float32, Float32]:
    """A bare reflectance spectrum is fit assuming it's viewed under a D65
    illuminant (the sRGB standard's own convention — sRGB primaries are
    defined relative to the D65 white point), so it only round-trips back to
    its own RGB when multiplied by a reference D65 white light (RGB=1,1,1)
    before conversion — mirrors exactly how a real NEE term (albedo * light)
    behaves, and matches the diagnostic methodology already used to validate
    rgb2spec.mojo's product-of-spectra accuracy (see project_spectral_rendering
    memory). Converting a bare albedo directly (no light) measures its color
    under an idealized equal-energy illuminant instead — a different, and for
    saturated colors quite different, number — so that is NOT what's tested
    here."""
    var accR = Float32(0.0); var accG = Float32(0.0); var accB = Float32(0.0)
    for i in range(N_TRIALS):
        var u = (Float32(i) + Float32(0.5)) / Float32(N_TRIALS)
        var wl = sample_wavelengths_uniform(u)
        var alb = rgb_to_spectral_sample(coeffs, res, cie_x, cie_y, cie_z, d65, r, g, b, wl)
        var light = rgb_illuminant_to_spectral_sample(coeffs, res, cie_x, cie_y, cie_z, d65, Float32(1.0), Float32(1.0), Float32(1.0), wl)
        var product = alb * light
        var (rr, gg, bb) = spectral_sample_to_rgb(coeffs, res, cie_x, cie_y, cie_z, d65, product, wl)
        accR += rr; accG += gg; accB += bb
    return (accR / Float32(N_TRIALS), accG / Float32(N_TRIALS), accB / Float32(N_TRIALS))

# ── hero-wavelength sampling ─────────────────────────────────────────────────

def test_sample_wavelengths_uniform_stays_in_range() raises:
    for i in range(50):
        var u = (Float32(i) + Float32(0.5)) / Float32(50)
        var wl = sample_wavelengths_uniform(u)
        for k in range(N_SPECTRAL_SAMPLES):
            var lam = wl.get(k)
            assert_true(lam >= LAMBDA_MIN and lam <= LAMBDA_MAX)

def test_sample_wavelengths_uniform_pdf_is_positive_and_constant() raises:
    var wl_a = sample_wavelengths_uniform(Float32(0.1))
    var wl_b = sample_wavelengths_uniform(Float32(0.9))
    assert_true(wl_a.pdf > Float32(0.0))
    assert_true(_close(wl_a.pdf, wl_b.pdf))
    assert_true(_close(wl_a.pdf, Float32(1.0) / (LAMBDA_MAX - LAMBDA_MIN)))

def test_sample_wavelengths_uniform_strata_are_distinct() raises:
    """Hero sampling's whole point is decorrelated samples across the 4
    lanes — they should not all collapse to the same wavelength."""
    var wl = sample_wavelengths_uniform(Float32(0.37))
    assert_true(not _close(wl.lambda0, wl.lambda1, Float32(1.0)))
    assert_true(not _close(wl.lambda1, wl.lambda2, Float32(1.0)))
    assert_true(not _close(wl.lambda2, wl.lambda3, Float32(1.0)))

# ── RGB -> spectrum -> RGB round trip (real Jakob-Hanika table) ────────────
# Much tighter tolerances than the old basis-function approximation — this is
# the actual pbrt method, verified to sub-1% at the shipped res=64 (see
# rgb2spec.mojo/project_spectral_rendering memory); tests here use a smaller
# TEST_RES for speed, so tolerances are a bit looser than production.

def test_roundtrip_white_is_near_identity() raises:
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var (r, g, b) = _roundtrip(handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, Float32(1.0), Float32(1.0), Float32(1.0))
    assert_true(_close(r, Float32(1.0), Float32(0.02)))
    assert_true(_close(g, Float32(1.0), Float32(0.02)))
    assert_true(_close(b, Float32(1.0), Float32(0.02)))

def test_roundtrip_grey_scales_linearly() raises:
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var (r, g, b) = _roundtrip(handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, Float32(0.5), Float32(0.5), Float32(0.5))
    assert_true(_close(r, Float32(0.5), Float32(0.02)))
    assert_true(_close(g, Float32(0.5), Float32(0.02)))
    assert_true(_close(b, Float32(0.5), Float32(0.02)))

def test_roundtrip_black_is_black() raises:
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var (r, g, b) = _roundtrip(handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, Float32(0.0), Float32(0.0), Float32(0.0))
    assert_true(_close(r, Float32(0.0), Float32(0.02)))
    assert_true(_close(g, Float32(0.0), Float32(0.02)))
    assert_true(_close(b, Float32(0.0), Float32(0.02)))

def test_roundtrip_red_dominant_channel_preserved() raises:
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var (r, g, b) = _roundtrip(handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, Float32(0.8), Float32(0.05), Float32(0.05))
    assert_true(_close(r, Float32(0.8), Float32(0.03)))
    assert_true(_close(g, Float32(0.05), Float32(0.03)))
    assert_true(_close(b, Float32(0.05), Float32(0.03)))

def test_roundtrip_green_dominant_channel_preserved() raises:
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var (r, g, b) = _roundtrip(handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, Float32(0.05), Float32(0.8), Float32(0.05))
    assert_true(_close(r, Float32(0.05), Float32(0.03)))
    assert_true(_close(g, Float32(0.8), Float32(0.03)))
    assert_true(_close(b, Float32(0.05), Float32(0.03)))

def test_roundtrip_blue_dominant_channel_preserved() raises:
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var (r, g, b) = _roundtrip(handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, Float32(0.05), Float32(0.05), Float32(0.8))
    assert_true(_close(r, Float32(0.05), Float32(0.03)))
    assert_true(_close(g, Float32(0.05), Float32(0.03)))
    assert_true(_close(b, Float32(0.8), Float32(0.03)))

def test_spectral_sample_values_are_nonnegative() raises:
    """A reflectance/emission spectrum can never go negative at any
    wavelength (sigmoid is bounded in (0,1) by construction), regardless of
    input RGB."""
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var wl = sample_wavelengths_uniform(Float32(0.42))
    var spec = rgb_to_spectral_sample(handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, Float32(1.0), Float32(0.0), Float32(0.0), wl)
    assert_true(spec.v0 >= Float32(0.0))
    assert_true(spec.v1 >= Float32(0.0))
    assert_true(spec.v2 >= Float32(0.0))
    assert_true(spec.v3 >= Float32(0.0))

# ── illuminant (light-color) conversion ─────────────────────────────────────

def test_illuminant_roundtrip_matches_direct_rgb_for_neutral_light() raises:
    """A neutral (equal-RGB) light emission should round-trip to itself,
    same as a neutral albedo — the illuminant scale/D65-tint machinery
    shouldn't introduce color shift for the achromatic case."""
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var accR = Float32(0.0); var accG = Float32(0.0); var accB = Float32(0.0)
    for i in range(N_TRIALS):
        var u = (Float32(i) + Float32(0.5)) / Float32(N_TRIALS)
        var wl = sample_wavelengths_uniform(u)
        var spec = rgb_illuminant_to_spectral_sample(handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, Float32(10.0), Float32(10.0), Float32(10.0), wl)
        var (rr, gg, bb) = spectral_sample_to_rgb(handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, spec, wl)
        accR += rr; accG += gg; accB += bb
    accR /= Float32(N_TRIALS); accG /= Float32(N_TRIALS); accB /= Float32(N_TRIALS)
    assert_true(_close(accR, Float32(10.0), Float32(0.5)))
    assert_true(_close(accG, Float32(10.0), Float32(0.5)))
    assert_true(_close(accB, Float32(10.0), Float32(0.5)))

def test_illuminant_spectral_values_are_nonnegative() raises:
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var wl = sample_wavelengths_uniform(Float32(0.6))
    var spec = rgb_illuminant_to_spectral_sample(handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, Float32(17.0), Float32(12.0), Float32(4.0), wl)
    assert_true(spec.v0 >= Float32(0.0))
    assert_true(spec.v1 >= Float32(0.0))
    assert_true(spec.v2 >= Float32(0.0))
    assert_true(spec.v3 >= Float32(0.0))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
