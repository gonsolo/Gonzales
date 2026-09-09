from std.math import abs
from std.os.path import exists
from std.testing import assert_true, assert_false, TestSuite
from gonzales.spd import (
    load_spd_rgb, load_spd_rgb_at, named_metal_rgb, named_glass_ior,
    SPD_LAMBDA_R, SPD_LAMBDA_G, SPD_LAMBDA_B,
)

comptime EPS: Float32 = 1e-4

# The gold .spd pair shipped with pbrt-v4's killeroos scene. Skipped (not
# failed) when the corpus isn't present, so this file stays runnable on a
# machine without the scene collection.
comptime AU_ETA = "/home/gonsolo/src/pbrt-v4-scenes/killeroos/spds/Au.eta.spd"
comptime AU_K = "/home/gonsolo/src/pbrt-v4-scenes/killeroos/spds/Au.k.spd"


def _close(a: Float32, b: Float32, tol: Float32 = EPS) -> Bool:
    return abs(a - b) < tol


def test_spd_reproduces_hardcoded_metal_au() raises:
    """THE anchor test for the .spd loader.

    material_builder.mojo's built-in `metal-Au` entry is eta
    (0.194, 0.608, 1.426) / k (3.060, 2.120, 1.846). Those are verbatim rows
    of Au.eta.spd / Au.k.spd at 619.920898 / 516.600769 / 459.200653 nm --
    that table was sampled straight out of these files. So sampling the
    loader at those same three wavelengths must return the table exactly.

    This pins the parser and the interpolator against an independent
    reference that already lived in the codebase: if the file scan, the
    column order, or the interpolation drifts, these equalities break."""
    if not exists(AU_ETA) or not exists(AU_K):
        return

    var (eta, eta_ok) = load_spd_rgb_at(
        AU_ETA, Float32(619.920898), Float32(516.600769), Float32(459.200653))
    assert_true(eta_ok, "Au.eta.spd should load")
    assert_true(_close(eta.r, Float32(0.194)), "Au eta R must match metal-Au table")
    assert_true(_close(eta.g, Float32(0.608)), "Au eta G must match metal-Au table")
    assert_true(_close(eta.b, Float32(1.426)), "Au eta B must match metal-Au table")

    var (k, k_ok) = load_spd_rgb_at(
        AU_K, Float32(619.920898), Float32(516.600769), Float32(459.200653))
    assert_true(k_ok, "Au.k.spd should load")
    assert_true(_close(k.r, Float32(3.060)), "Au k R must match metal-Au table")
    assert_true(_close(k.g, Float32(2.120)), "Au k G must match metal-Au table")
    assert_true(_close(k.b, Float32(1.846)), "Au k B must match metal-Au table")


def test_spd_gold_is_spectrally_gold() raises:
    """Gold's defining optical signature: k (the absorption term driving
    reflectance) rises steeply toward the red, so long wavelengths reflect far
    more than short ones. A loader that returned a plausible-looking but
    wavelength-scrambled triple would still pass a "is it yellow-ish" eyeball
    check; this asserts the actual monotone ordering."""
    if not exists(AU_K):
        return
    var (k, ok) = load_spd_rgb(AU_K)
    assert_true(ok, "Au.k.spd should load")
    assert_true(k.r > k.g, "gold k must increase toward red (R>G)")
    assert_true(k.g > k.b, "gold k must increase toward red (G>B)")

    var (eta, eta_ok) = load_spd_rgb(AU_ETA)
    assert_true(eta_ok, "Au.eta.spd should load")
    # eta runs the other way for gold -- lowest in the red.
    assert_true(eta.r < eta.b, "gold eta must fall toward red")


def test_spd_missing_file_reports_failure() raises:
    """A missing/unreadable .spd must return ok=False rather than a silent
    default -- the caller warns on it. Three silent asset-drop bugs were
    found in one session; see project_silent_asset_load_failures."""
    var (_, ok) = load_spd_rgb("/nonexistent/definitely/not/here.spd")
    assert_false(ok, "missing .spd must report failure, not a default")


def test_named_metal_cuzn_not_shadowed_by_cu() raises:
    """Regression: `startswith("metal-Cu")` also matches every metal-CuZn-*
    name, so brass silently rendered as copper. CuZn must be tested first."""
    var (cuzn, cuzn_ok) = named_metal_rgb("metal-CuZn-eta", False)
    var (cu, cu_ok) = named_metal_rgb("metal-Cu-eta", False)
    assert_true(cuzn_ok, "metal-CuZn-eta must resolve")
    assert_true(cu_ok, "metal-Cu-eta must resolve")
    assert_true(not _close(cuzn.g, cu.g, Float32(1e-2)),
                "CuZn must not resolve to Cu's values")


def test_named_spectrum_unknown_reports_failure() raises:
    var (_, ok) = named_metal_rgb("metal-Unobtainium-eta", False)
    assert_false(ok, "unknown metal must report failure so the caller warns")
    var (_, g_ok) = named_glass_ior("glass-NotReal")
    assert_false(g_ok, "unknown glass must report failure so the caller warns")


def test_named_glass_iors_are_sane() raises:
    """Every pbrt glass-* name the corpus uses must resolve to a physically
    sensible IOR (optical glasses run ~1.45-1.9)."""
    var names = ["glass-BK7", "glass-BAF10", "glass-FK51A",
                 "glass-LASF9", "glass-F5", "glass-F10", "glass-F11"]
    for n in names:
        var (ior, ok) = named_glass_ior(n)
        assert_true(ok, "glass name must resolve: " + n)
        assert_true(ior > Float32(1.4) and ior < Float32(2.0),
                    "IOR out of physical range for " + n)


def test_named_metal_tio2_is_nonabsorbing() raises:
    """TiO2 is a transparent high-index dielectric, not an absorbing metal:
    k is identically zero across the visible range, eta is ~2.9."""
    var (k, k_ok) = named_metal_rgb("metal-TiO2-k", True)
    assert_true(k_ok, "metal-TiO2-k must resolve")
    assert_true(_close(k.r, Float32(0.0)) and _close(k.g, Float32(0.0)),
                "TiO2 k must be zero")
    var (eta, eta_ok) = named_metal_rgb("metal-TiO2-eta", False)
    assert_true(eta_ok, "metal-TiO2-eta must resolve")
    assert_true(eta.r > Float32(2.5), "TiO2 eta must be high-index")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
