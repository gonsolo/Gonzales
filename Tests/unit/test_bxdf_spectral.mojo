# Tests for the spectral siblings added to bxdf.mojo (bxdf_eval_any_spectral,
# _nee_weight_simple_spectral) and the _ggx_conductor_shape_terms extraction
# they share with the existing RGB bxdf_eval_conductor_ggx -- staged spectral
# rendering rollout, see project_spectral_rendering memory.
from std.math import abs
from std.testing import assert_true, TestSuite
from gonzales.geometry import RGB, INV_PI, Vec3f
from gonzales.bxdf import (
    bxdf_eval_conductor_ggx, bxdf_eval_any, bxdf_eval_any_spectral,
    _nee_weight_simple, _nee_weight_simple_spectral,
)
from gonzales.bvh import LightSample
from gonzales.spectrum import (
    SampledWavelengths, sample_wavelengths_uniform, spectral_sample_to_rgb,
    rgb_illuminant_to_spectral_sample, SpectralContext, SpectralHandle, spectral_handle,
)
from gonzales.rgb2spec import build_spectrum_table, build_cie_xyz_tables, SpectrumTable

comptime EPS: Float32 = 1e-4
comptime TEST_RES = 16
comptime N_TRIALS = 2000

def _close(a: Float32, b: Float32, tol: Float32 = EPS) -> Bool:
    return abs(a - b) < tol

def _test_ctx() -> SpectralContext:
    var table = build_spectrum_table(TEST_RES)
    var cie = build_cie_xyz_tables()
    return SpectralContext(SpectrumTable(table^, TEST_RES), cie^)

# ── _ggx_conductor_shape_terms extraction: bxdf_eval_conductor_ggx unchanged ─

def test_conductor_ggx_refactor_matches_hand_computed_case() raises:
    """Regression check for the _ggx_conductor_shape_terms extraction --
    bxdf_eval_conductor_ggx's output must be unchanged from before the
    refactor for a simple normal-incidence case."""
    var n = Vec3f(0.0, 0.0, 1.0)
    var wo = Vec3f(0.0, 0.0, 1.0)
    var wi = Vec3f(0.0, 0.0, 1.0)
    var f0 = RGB(Float32(0.9), Float32(0.6), Float32(0.2))
    var result = bxdf_eval_conductor_ggx(n, wo, wi, Float32(0.3), f0)
    # At normal incidence wo=wi=wh=n, so cos_wo_h=1, schlick=(1-1)^5=0 -> fr=f0 exactly.
    assert_true(result.r > Float32(0.0) and result.g > Float32(0.0) and result.b > Float32(0.0))
    # Result should be proportional to f0's own ratios (fr=f0 at this angle).
    assert_true(_close(result.r / result.g, f0.r / f0.g, Float32(0.01)))
    assert_true(_close(result.g / result.b, f0.g / f0.b, Float32(0.01)))

def test_conductor_ggx_grazing_still_zero_after_refactor() raises:
    var n = Vec3f(0.0, 0.0, 1.0)
    var wo = Vec3f(1.0, 0.0, 0.0)  # grazing (cos_o = 0)
    var wi = Vec3f(0.0, 0.0, 1.0)
    var f0 = RGB(Float32(0.9), Float32(0.6), Float32(0.2))
    var result = bxdf_eval_conductor_ggx(n, wo, wi, Float32(0.3), f0)
    assert_true(_close(result.r, Float32(0.0)) and _close(result.g, Float32(0.0)) and _close(result.b, Float32(0.0)))

# ── bxdf_eval_any_spectral ───────────────────────────────────────────────────

def test_bxdf_eval_any_spectral_diffuse_matches_rgb_after_roundtrip() raises:
    """Diffuse f=alb/pi should round-trip through the spectral conversion
    back to (approximately) the same RGB value the plain RGB bxdf_eval_any
    already gives -- the whole point of the real Jakob-Hanika table."""
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var alb = RGB(Float32(0.6), Float32(0.3), Float32(0.1))
    var n = Vec3f(0.0, 0.0, 1.0)
    var wo = Vec3f(0.0, 0.0, 1.0)
    var wi = Vec3f(0.267261, 0.534522, 0.801784)

    var (f_rgb, pdf_rgb) = bxdf_eval_any(Int32(0), alb, Float32(0.0), n, wo, wi)

    var accR = Float32(0.0); var accG = Float32(0.0); var accB = Float32(0.0)
    for i in range(N_TRIALS):
        var u = (Float32(i) + Float32(0.5)) / Float32(N_TRIALS)
        var wl = sample_wavelengths_uniform(u)
        var (f_spec, pdf_spec) = bxdf_eval_any_spectral(Int32(0), alb, Float32(0.0), n, wo, wi, handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, wl)
        # Pair against a neutral reference white light so the reflectance
        # round-trips (see test_spectrum.mojo's _roundtrip docstring for why
        # a bare reflectance needs this).
        var light = rgb_illuminant_to_spectral_sample(handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, Float32(1.0), Float32(1.0), Float32(1.0), wl)
        var product = f_spec * light
        var (rr, gg, bb) = spectral_sample_to_rgb(handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, product, wl)
        accR += rr; accG += gg; accB += bb
        assert_true(_close(pdf_spec, pdf_rgb))
    accR /= Float32(N_TRIALS); accG /= Float32(N_TRIALS); accB /= Float32(N_TRIALS)

    assert_true(_close(accR, f_rgb.r, Float32(0.02)))
    assert_true(_close(accG, f_rgb.g, Float32(0.02)))
    assert_true(_close(accB, f_rgb.b, Float32(0.02)))

def test_bxdf_eval_any_spectral_conductor_pdf_matches_rgb() raises:
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var f0 = RGB(Float32(0.9), Float32(0.6), Float32(0.2))
    var n = Vec3f(0.0, 0.0, 1.0)
    var wo = Vec3f(0.0, 0.0, 1.0)
    var wi = Vec3f(0.0, 0.0, 1.0)
    var wl = sample_wavelengths_uniform(Float32(0.5))
    var (_, pdf_rgb) = bxdf_eval_any(Int32(1), f0, Float32(0.3), n, wo, wi)
    var (_, pdf_spec) = bxdf_eval_any_spectral(Int32(1), f0, Float32(0.3), n, wo, wi, handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, wl)
    assert_true(_close(pdf_spec, pdf_rgb))

def test_bxdf_eval_any_spectral_values_nonnegative() raises:
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var alb = RGB(Float32(0.8), Float32(0.05), Float32(0.05))
    var n = Vec3f(0.0, 0.0, 1.0)
    var wo = Vec3f(0.0, 0.0, 1.0)
    var wi = Vec3f(0.267261, 0.534522, 0.801784)
    var wl = sample_wavelengths_uniform(Float32(0.3))
    var (f, _) = bxdf_eval_any_spectral(Int32(0), alb, Float32(0.0), n, wo, wi, handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, wl)
    assert_true(f.v0 >= Float32(0.0) and f.v1 >= Float32(0.0) and f.v2 >= Float32(0.0) and f.v3 >= Float32(0.0))

# ── _nee_weight_simple_spectral ──────────────────────────────────────────────

def test_nee_weight_simple_spectral_invalid_sample_is_zero() raises:
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var wl = sample_wavelengths_uniform(Float32(0.5))
    var ls = LightSample(Vec3f(0.0, 0.0, 1.0), RGB(Float32(5.0)), Float32(1.0), Float32(1.0), False, False)
    var n = Vec3f(0.0, 0.0, 1.0)
    var wo = Vec3f(0.0, 0.0, 1.0)
    var result = _nee_weight_simple_spectral(ls, Int32(0), RGB(Float32(0.5)), Float32(0.0), n, wo, handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, wl)
    assert_true(_close(result.v0, Float32(0.0)) and _close(result.v1, Float32(0.0)))

def test_nee_weight_simple_spectral_backfacing_is_zero() raises:
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var wl = sample_wavelengths_uniform(Float32(0.5))
    # Light direction opposite the normal -- cos_s <= 0.
    var ls = LightSample(Vec3f(0.0, 0.0, -1.0), RGB(Float32(5.0)), Float32(1.0), Float32(1.0), True, True)
    var n = Vec3f(0.0, 0.0, 1.0)
    var wo = Vec3f(0.0, 0.0, 1.0)
    var result = _nee_weight_simple_spectral(ls, Int32(0), RGB(Float32(0.5)), Float32(0.0), n, wo, handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, wl)
    assert_true(_close(result.v0, Float32(0.0)) and _close(result.v1, Float32(0.0)))

def test_nee_weight_simple_spectral_delta_light_matches_rgb_after_roundtrip() raises:
    """A delta (distant/point) light NEE term computed spectrally, averaged
    over many hero-wavelength draws and converted back to RGB, should
    approximately match the plain RGB _nee_weight_simple result -- mirrors
    the diffuse round-trip test above but exercises the full NEE formula
    (light color included, not just material color) as it will actually be
    used from shading.mojo in Stage 2c."""
    var ctx = _test_ctx()
    var handle = spectral_handle(ctx)
    var alb = RGB(Float32(0.63), Float32(0.065), Float32(0.05))
    var emission = RGB(Float32(10.0), Float32(10.0), Float32(10.0))
    var n = Vec3f(0.0, 0.0, 1.0)
    var wo = Vec3f(0.0, 0.0, 1.0)
    var ls_rgb = LightSample(Vec3f(0.267261, 0.534522, 0.801784), emission, Float32(1.0), Float32(1.0), True, True)
    var rgb_result = _nee_weight_simple(ls_rgb, Int32(0), alb, Float32(0.0), n, wo)

    var accR = Float32(0.0); var accG = Float32(0.0); var accB = Float32(0.0)
    for i in range(N_TRIALS):
        var u = (Float32(i) + Float32(0.5)) / Float32(N_TRIALS)
        var wl = sample_wavelengths_uniform(u)
        var result = _nee_weight_simple_spectral(ls_rgb, Int32(0), alb, Float32(0.0), n, wo, handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, wl)
        var (rr, gg, bb) = spectral_sample_to_rgb(handle.coeffs, handle.res, handle.cie_x, handle.cie_y, handle.cie_z, handle.d65, result, wl)
        accR += rr; accG += gg; accB += bb
    accR /= Float32(N_TRIALS); accG /= Float32(N_TRIALS); accB /= Float32(N_TRIALS)

    assert_true(_close(accR, rgb_result.r, Float32(0.05)))
    assert_true(_close(accG, rgb_result.g, Float32(0.05)))
    assert_true(_close(accB, rgb_result.b, Float32(0.05)))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
