from std.math import abs
from std.testing import assert_true, TestSuite
from gonzales.spectrum import (
    SampledWavelengths, sample_wavelengths_uniform,
    rgb_to_spectral_sample, spectral_sample_to_rgb,
    cie_x, cie_y, cie_z, LAMBDA_MIN, LAMBDA_MAX, N_SPECTRAL_SAMPLES,
)

comptime EPS: Float32 = 1e-3

def _close(a: Float32, b: Float32, tol: Float32 = EPS) -> Bool:
    return abs(a - b) < tol

# Stratified average over many hero-wavelength draws — removes per-sample MC
# noise so these tests check the underlying round-trip math, not variance.
comptime N_TRIALS = 2000

def _roundtrip(r: Float32, g: Float32, b: Float32) -> Tuple[Float32, Float32, Float32]:
    var accR = Float32(0.0); var accG = Float32(0.0); var accB = Float32(0.0)
    for i in range(N_TRIALS):
        var u = (Float32(i) + Float32(0.5)) / Float32(N_TRIALS)
        var wl = sample_wavelengths_uniform(u)
        var spec = rgb_to_spectral_sample(r, g, b, wl)
        var (rr, gg, bb) = spectral_sample_to_rgb(spec, wl)
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

# ── CIE color-matching-function sanity ──────────────────────────────────────

def test_cie_y_peaks_near_555nm() raises:
    var peak = Float32(-1.0); var peak_lam = Float32(0.0)
    var lam = LAMBDA_MIN
    while lam <= LAMBDA_MAX:
        var yv = cie_y(lam)
        if yv > peak:
            peak = yv; peak_lam = lam
        lam += Float32(2.0)
    assert_true(_close(peak_lam, Float32(555.0), Float32(6.0)))
    assert_true(_close(peak, Float32(1.0), Float32(0.05)))

def test_cie_curves_nonnegative_over_visible_range() raises:
    var lam = LAMBDA_MIN
    while lam <= LAMBDA_MAX:
        assert_true(cie_x(lam) >= Float32(-0.01))  # x-bar's small negative lobe near 500nm
        assert_true(cie_y(lam) >= Float32(-0.001))
        assert_true(cie_z(lam) >= Float32(-0.001))
        lam += Float32(5.0)

# ── RGB -> spectrum -> RGB round trip ───────────────────────────────────────
# The full pipeline is linear in the input RGB (basis evaluation -> CIE
# integral -> XYZ->RGB matrix), corrected by a precomputed M^-1 so achromatic
# colors and the primaries round-trip cleanly; see spectrum.mojo's comment on
# _M_INV_* for the derivation. These tests pin that guarantee down.

def test_roundtrip_white_is_near_identity() raises:
    var (r, g, b) = _roundtrip(Float32(1.0), Float32(1.0), Float32(1.0))
    assert_true(_close(r, Float32(1.0), Float32(0.01)))
    assert_true(_close(g, Float32(1.0), Float32(0.01)))
    assert_true(_close(b, Float32(1.0), Float32(0.01)))

def test_roundtrip_grey_scales_linearly() raises:
    var (r, g, b) = _roundtrip(Float32(0.5), Float32(0.5), Float32(0.5))
    assert_true(_close(r, Float32(0.5), Float32(0.01)))
    assert_true(_close(g, Float32(0.5), Float32(0.01)))
    assert_true(_close(b, Float32(0.5), Float32(0.01)))

def test_roundtrip_black_is_black() raises:
    var (r, g, b) = _roundtrip(Float32(0.0), Float32(0.0), Float32(0.0))
    assert_true(_close(r, Float32(0.0)))
    assert_true(_close(g, Float32(0.0)))
    assert_true(_close(b, Float32(0.0)))

def test_roundtrip_red_dominant_channel_preserved() raises:
    """Not an exact round trip (the spectrum's non-negativity clamp is a real,
    documented nonlinearity) — but the dominant channel must stay dominant
    and the off-channel bleed must stay small, or the approximation isn't
    good enough to be useful."""
    var (r, g, b) = _roundtrip(Float32(1.0), Float32(0.0), Float32(0.0))
    assert_true(r > Float32(0.9))
    assert_true(g < Float32(0.15))
    assert_true(b < Float32(0.05))

def test_roundtrip_green_dominant_channel_preserved() raises:
    var (r, g, b) = _roundtrip(Float32(0.0), Float32(1.0), Float32(0.0))
    assert_true(g > Float32(0.9))
    assert_true(r < Float32(0.15))
    assert_true(b < Float32(0.05))

def test_roundtrip_blue_dominant_channel_preserved() raises:
    var (r, g, b) = _roundtrip(Float32(0.0), Float32(0.0), Float32(1.0))
    assert_true(b > Float32(0.9))
    assert_true(r < Float32(0.1))
    assert_true(g < Float32(0.1))

def test_spectral_sample_values_are_nonnegative() raises:
    """A reflectance/emission spectrum can never go negative at any
    wavelength, regardless of input RGB (even out-of-gamut-leaning inputs
    that make the M^-1-corrected coefficients negative)."""
    var wl = sample_wavelengths_uniform(Float32(0.42))
    var spec = rgb_to_spectral_sample(Float32(1.0), Float32(0.0), Float32(0.0), wl)
    assert_true(spec.v0 >= Float32(0.0))
    assert_true(spec.v1 >= Float32(0.0))
    assert_true(spec.v2 >= Float32(0.0))
    assert_true(spec.v3 >= Float32(0.0))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
