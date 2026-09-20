"""VCM's two camera-side environment-light MIS weights, pinned to the exact
values derived in Scenes/vcm_env_mis_derivation.py.

That harness enumerates every strategy's path density directly and forms the
balance-heuristic weight, reproducing it from the same dVCM/dVC recursion the
renderer carries -- agreement is exact to 1e-16. These tests pin the two
weight FUNCTIONS to those numbers, so a formula regression is caught here, in
milliseconds, instead of as a few percent of brightness in a render where it
competes with every other source of error.

The configuration is the harness's "1 surface vertex" case, its numbers copied
verbatim:

    lens   (0, 0, -4)
    x_1    (0.3, -0.2, 0), normal normalize(0.1, 0.15, -1)
    w_env  normalize(0.35, 0.8, -0.5)
    uniform environment, scene radius 4, merge radius 0.045, N = 12000

Tolerance is 2e-6 relative: these run in Float32 while the harness is double.
"""
from std.testing import assert_true, assert_equal, TestSuite
from gonzales.vcm_mis import vcm_env_nee_weight, vcm_env_escape_weight

# ── the harness's "1 surface vertex" configuration ─────────────────────────
comptime P_ENV = Float32(0.07957747154594767)
comptime EMISSION_PDF_W = Float32(0.0015831434944115277)
comptime PDF_BSDF_DIR = Float32(0.2039148626657762)
comptime PDF_BSDF_REV = Float32(0.3119951944408041)
comptime COS_OUT = Float32(0.640617434508574)
comptime MIS_VM = Float32(76.34070148223198)
comptime CAM_DVCM_ARRIVAL = Float32(16.456466495732197)
comptime CAM_DVC_ARRIVAL = Float32(0.0)
comptime POST_SCATTER_DVCM = Float32(4.904007422151646)
comptime POST_SCATTER_DVC = Float32(291.53090119351026)

comptime EXPECT_NEE = Float32(0.21074194480518355)
comptime EXPECT_ESCAPE = Float32(0.5400198560977768)


def _close(got: Float32, want: Float32, tol: Float32, label: String) raises:
    var err = abs(got - want) / max(abs(want), Float32(1e-30))
    if err > tol:
        raise Error(String("{}: got {}, want {}, rel err {}").format(
            label, got, want, err))


def test_env_nee_weight_matches_the_derivation() raises:
    """NEE's balance weight over ALL strategies, not just NEE vs BSDF."""
    var w = vcm_env_nee_weight(PDF_BSDF_DIR, PDF_BSDF_REV, P_ENV,
                               EMISSION_PDF_W, COS_OUT, MIS_VM,
                               CAM_DVCM_ARRIVAL, CAM_DVC_ARRIVAL)
    _close(w, EXPECT_NEE, Float32(2e-6), "env NEE weight")


def test_env_escape_weight_matches_the_derivation() raises:
    """The escape's balance weight, from the POST-SCATTER carries."""
    var w = vcm_env_escape_weight(P_ENV, EMISSION_PDF_W,
                                  POST_SCATTER_DVCM, POST_SCATTER_DVC)
    _close(w, EXPECT_ESCAPE, Float32(2e-6), "env escape weight")


def test_escape_with_arrival_carries_is_wrong() raises:
    """Guards the trap that cost the first implementation attempt.

    Feeding the ARRIVAL carries to the escape weight (instead of the
    post-scatter ones, one step later) is not a rounding difference: the
    arrival dVCM is smaller and dVC is still 0 at a first bounce, so the
    denominator collapses and the weight comes out ~20% LOW -- 0.4327 against
    the correct 0.5400, which is exactly what the derivation harness reported
    before the call site was corrected. If this ever matches the right answer,
    someone has changed what the carries mean and both call sites need
    re-checking."""
    var wrong = vcm_env_escape_weight(P_ENV, EMISSION_PDF_W,
                                      CAM_DVCM_ARRIVAL, CAM_DVC_ARRIVAL)
    assert_true(wrong < EXPECT_ESCAPE * Float32(0.85),
                "arrival carries must under-weight the escape by ~20%")
    assert_true(wrong > EXPECT_ESCAPE * Float32(0.70),
                "...by ~20%, not by an order of magnitude -- if it moved this "
                "far the carries themselves changed meaning")


def test_weights_shrink_as_merging_gets_more_share() raises:
    """Merging's share, `mis_vm_weight_factor`, IS part of the denominator, so raising
    it must lower NEE's weight monotonically. With it at zero the weight must
    collapse to the plain two-strategy balance weight, which is what the old
    power heuristic was approximating (badly)."""
    var w_no_merge = vcm_env_nee_weight(PDF_BSDF_DIR, PDF_BSDF_REV, P_ENV,
                                        EMISSION_PDF_W, COS_OUT, Float32(0.0),
                                        Float32(0.0), Float32(0.0))
    var balance = P_ENV / (P_ENV + PDF_BSDF_DIR)
    _close(w_no_merge, balance, Float32(2e-6), "NEE with no merging share")

    var prev = w_no_merge
    for i in range(1, 6):
        var w = vcm_env_nee_weight(PDF_BSDF_DIR, PDF_BSDF_REV, P_ENV,
                                   EMISSION_PDF_W, COS_OUT,
                                   MIS_VM * Float32(i),
                                   CAM_DVCM_ARRIVAL, CAM_DVC_ARRIVAL)
        assert_true(w < prev, "NEE weight must fall as merging's share grows")
        prev = w


def test_degenerate_light_pdf_is_zero_not_inf() raises:
    """A zero direct pdf must not divide."""
    assert_equal(vcm_env_nee_weight(PDF_BSDF_DIR, PDF_BSDF_REV, Float32(0.0),
                                    EMISSION_PDF_W, COS_OUT, MIS_VM,
                                    CAM_DVCM_ARRIVAL, CAM_DVC_ARRIVAL),
                 Float32(0.0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
