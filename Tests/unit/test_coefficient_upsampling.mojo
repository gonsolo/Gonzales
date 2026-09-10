from std.math import abs, min, max
from std.testing import assert_true, TestSuite
from gonzales.spectrum import (
    load_spectral_context, spectral_handle, sample_wavelengths_uniform,
    spec_refl, spec_refl_unbounded, rgb_bands_to_spectral_sample,
)

# Does a GREY RGB coefficient survive the smooth (Jakob-Hanika) upsampler
# unchanged, lane by lane?
#
# This matters because docs/02_spectra_and_color.md long asserted it does
# NOT -- "RGB(1,1,1) does not upsample to exactly 1 in every wavelength lane
# (the table has no reason to be 'flat' there)" -- and that claim is the
# entire stated justification for band-picking medium coefficients instead
# of upsampling them. Measured 2026-09-10, the claim is false: the table
# reproduces a grey coefficient exactly, in every lane, for both the
# clamped and unbounded entry points.
#
# The historical 3.5x (homogeneous) / 10.7x (NanoVDB) grey-medium bug that
# motivated band-picking is real, but its recorded signature -- "a spurious
# D65-shaped tint" -- is the fingerprint of the ILLUMINANT upsampler
# (spec_illum tints the D65 shape by construction), not of spec_refl. So
# the fix was very likely right for a different reason than recorded.
#
# Consequence, and why this test exists: matching pbrt's own convention for
# medium coefficients (RGBUnboundedSpectrum, a smooth sigmoid) is NOT
# blocked by a grey-invariant violation. If a future change swaps
# band-picking for spec_refl_unbounded and this test still passes, the grey
# invariant is intact and any remaining difference is a genuine spectral
# modelling choice, not a regression.

comptime EPS: Float32 = 1e-6
comptime DATA_DIR = "src/gonzales/data"


def _lanes_equal(v0: Float32, v1: Float32, v2: Float32, v3: Float32,
                 expect: Float32) -> Bool:
    return (abs(v0 - expect) < EPS and abs(v1 - expect) < EPS
            and abs(v2 - expect) < EPS and abs(v3 - expect) < EPS)


def test_grey_coefficient_upsamples_flat() raises:
    """Grey survives spec_refl / spec_refl_unbounded exactly."""
    var loaded = load_spectral_context(DATA_DIR)
    if not loaded[0]:
        print("SKIP test_grey_coefficient_upsamples_flat: no spectrum table")
        return
    var ctx = loaded[1].copy()
    var h = spectral_handle(ctx)

    # Several hero-wavelength sets, so this cannot pass by landing on one
    # lucky quadruple of lambdas.
    var us: List[Float32] = [0.0, 0.37, 0.61, 0.93]
    for i in range(len(us)):
        var wl = sample_wavelengths_uniform(us[i])

        var one = spec_refl(h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65,
                            Float32(1), Float32(1), Float32(1), wl)
        assert_true(_lanes_equal(one.v0, one.v1, one.v2, one.v3, Float32(1)),
                    "spec_refl(1,1,1) must be flat 1 in every lane")

        # The unbounded path is what pbrt's RGBUnboundedSpectrum corresponds
        # to, and what a medium coefficient (legitimately >1) needs.
        var two = spec_refl_unbounded(h.coeffs, h.res, h.cie_x, h.cie_y,
                                      h.cie_z, h.d65,
                                      Float32(2), Float32(2), Float32(2), wl)
        assert_true(_lanes_equal(two.v0, two.v1, two.v2, two.v3, Float32(2)),
                    "spec_refl_unbounded(2,2,2) must be flat 2 in every lane")

        # Band-picking trivially satisfies the same invariant; asserted here
        # so the two strategies are compared on one footing rather than the
        # incumbent being assumed correct.
        var bp = rgb_bands_to_spectral_sample(Float32(1), Float32(1),
                                              Float32(1), wl)
        assert_true(_lanes_equal(bp.v0, bp.v1, bp.v2, bp.v3, Float32(1)),
                    "band-pick(1,1,1) must be flat 1 in every lane")


def test_chromatic_coefficient_is_not_flat() raises:
    """Anti-vacuity: a NON-grey coefficient must NOT come back flat.

    Without this, the test above would still pass if the upsampler were
    stubbed to return its input's mean, or a constant."""
    var loaded = load_spectral_context(DATA_DIR)
    if not loaded[0]:
        print("SKIP test_chromatic_coefficient_is_not_flat: no spectrum table")
        return
    var ctx = loaded[1].copy()
    var h = spectral_handle(ctx)
    var wl = sample_wavelengths_uniform(Float32(0.37))

    var c = spec_refl(h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65,
                      Float32(0.25), Float32(0.5), Float32(1.0), wl)
    var lo = min(min(c.v0, c.v1), min(c.v2, c.v3))
    var hi = max(max(c.v0, c.v1), max(c.v2, c.v3))
    assert_true(hi - lo > Float32(0.01),
                "a chromatic coefficient must vary across lanes")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
