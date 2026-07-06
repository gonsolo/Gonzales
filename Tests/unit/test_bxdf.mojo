from std.math import abs
from std.testing import assert_true, assert_false, TestSuite
from gonzales.bxdf import (
    ggx_D, ggx_lambda, ggx_G1, ggx_G2, ggx_vndf_pdf,
    bxdf_is_delta, bxdf_eval_diffuse, bxdf_pdf_diffuse, BxDFFlags,
)
from gonzales.geometry import RGB, PI, INV_PI

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# ── ggx_D (Trowbridge-Reitz microfacet distribution) ─────────────────────────

def test_ggx_d_normal_incidence() raises:
    """At cos_theta_h=1 the denominator collapses to alpha^2, giving the
    closed form D = 1/(pi*alpha^2)."""
    for alpha in [Float32(0.1), Float32(0.5), Float32(1.0)]:
        var d = ggx_D(Float32(1.0), alpha)
        var expected = Float32(1.0) / (PI * alpha * alpha)
        assert_true(_close(d, expected))

def test_ggx_d_never_negative() raises:
    var d1 = ggx_D(Float32(0.0), Float32(0.5))
    var d2 = ggx_D(Float32(-1.0), Float32(0.5))
    assert_true(d1 >= Float32(0.0))
    assert_true(d2 >= Float32(0.0))

# ── ggx_lambda / ggx_G1 / ggx_G2 (Smith masking-shadowing) ────────────────────

def test_ggx_lambda_zero_for_perfectly_smooth() raises:
    """Alpha=0 means no microfacet spread — Lambda (and hence masking) vanishes."""
    assert_true(_close(ggx_lambda(Float32(0.5), Float32(0.0)), Float32(0.0)))
    assert_true(_close(ggx_lambda(Float32(1.0), Float32(0.0)), Float32(0.0)))

def test_ggx_lambda_zero_at_or_below_horizon() raises:
    assert_true(_close(ggx_lambda(Float32(0.0), Float32(0.5)), Float32(0.0)))
    assert_true(_close(ggx_lambda(Float32(-0.3), Float32(0.5)), Float32(0.0)))

def test_ggx_g1_perfectly_smooth_is_one() raises:
    """No masking on a mirror surface — G1 == 1 regardless of view angle."""
    assert_true(_close(ggx_G1(Float32(0.3), Float32(0.0)), Float32(1.0)))
    assert_true(_close(ggx_G1(Float32(0.9), Float32(0.0)), Float32(1.0)))

def test_ggx_g1_in_unit_range() raises:
    var g = ggx_G1(Float32(0.5), Float32(0.5))
    assert_true(g > Float32(0.0) and g <= Float32(1.0))

def test_ggx_g2_perfectly_smooth_is_one() raises:
    assert_true(_close(ggx_G2(Float32(0.4), Float32(0.6), Float32(0.0)), Float32(1.0)))

def test_ggx_g2_symmetric_in_cos_o_cos_i() raises:
    """Height-correlated Smith G2 is symmetric under swapping wo/wi."""
    var a = ggx_G2(Float32(0.3), Float32(0.7), Float32(0.4))
    var b = ggx_G2(Float32(0.7), Float32(0.3), Float32(0.4))
    assert_true(_close(a, b))

# ── ggx_vndf_pdf ───────────────────────────────────────────────────────────

def test_ggx_vndf_pdf_zero_below_horizon() raises:
    assert_true(ggx_vndf_pdf(Float32(-0.1), Float32(0.5), Float32(1.0), Float32(0.5)) == Float32(0.0))
    assert_true(ggx_vndf_pdf(Float32(0.5), Float32(-0.1), Float32(1.0), Float32(0.5)) == Float32(0.0))

def test_ggx_vndf_pdf_independent_of_cos_wm() raises:
    """Pdf = G1(cos_o)*D*cos_wm/cos_o / (4*cos_wm) — cos_wm algebraically
    cancels, so the result must be identical for any two positive cos_wm."""
    var cos_o = Float32(0.6)
    var d = Float32(2.0)
    var alpha = Float32(0.3)
    var p1 = ggx_vndf_pdf(cos_o, Float32(0.2), d, alpha)
    var p2 = ggx_vndf_pdf(cos_o, Float32(0.9), d, alpha)
    assert_true(_close(p1, p2))

def test_ggx_vndf_pdf_matches_closed_form() raises:
    var cos_o = Float32(0.6)
    var alpha = Float32(0.3)
    var d = Float32(2.0)
    var p = ggx_vndf_pdf(cos_o, Float32(0.5), d, alpha)
    var expected = ggx_G1(cos_o, alpha) * d / (Float32(4.0) * cos_o)
    assert_true(_close(p, expected))

# ── BxDFFlags / bxdf_is_delta ──────────────────────────────────────────────

def test_bxdf_is_delta() raises:
    assert_true(bxdf_is_delta(BxDFFlags.delta))
    assert_true(bxdf_is_delta(BxDFFlags.delta | BxDFFlags.reflect))
    assert_false(bxdf_is_delta(BxDFFlags.glossy))
    assert_false(bxdf_is_delta(BxDFFlags.diffuse | BxDFFlags.reflect))

# ── Diffuse (Lambertian) helpers ───────────────────────────────────────────

def test_bxdf_eval_diffuse_scales_by_inv_pi() raises:
    var f = bxdf_eval_diffuse(RGB(Float32(1.0)))
    assert_true(_close(f.r, INV_PI) and _close(f.g, INV_PI) and _close(f.b, INV_PI))

    var f2 = bxdf_eval_diffuse(RGB(Float32(2.0), Float32(4.0), Float32(6.0)))
    assert_true(_close(f2.r, Float32(2.0) * INV_PI))
    assert_true(_close(f2.g, Float32(4.0) * INV_PI))
    assert_true(_close(f2.b, Float32(6.0) * INV_PI))

def test_bxdf_pdf_diffuse() raises:
    assert_true(bxdf_pdf_diffuse(Float32(-0.5)) == Float32(0.0))
    assert_true(_close(bxdf_pdf_diffuse(Float32(1.0)), INV_PI))
    assert_true(_close(bxdf_pdf_diffuse(Float32(0.5)), Float32(0.5) * INV_PI))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
