"""Does `bxdf_pdf_coated_exit` match the direction `coat_walk_scatter`
actually samples?

This is the check the volume-MIS attempt did not have. A MIS derivation that
sums to 1 in isolation proves nothing about whether the renderer's sampler
draws from the density the weights assume -- so this test drives the REAL
walk, histograms its exit directions, and compares against the analytic pdf.
If they ever disagree, putting coateddiffuse into _bdpt_vertex_mis_scoped
will silently lose energy (see project_vcm_volume_mis).
"""
from std.testing import assert_true, assert_equal, TestSuite
from std.math import sqrt, abs
from gonzales.bxdf import (
    coat_walk_begin, coat_walk_scatter, bxdf_pdf_coated_exit, coat_exit_norm,
    COAT_EXIT, COAT_WALKING,
)
from gonzales.geometry import Vec3f, RGB, dot, PI, fr_dielectric
from gonzales.rng import PCG32

comptime NBINS = 16


def _run_walks(ior: Float32, alpha: Float32, n: Int, mut hist: List[Float32]) -> Int:
    """Drive the real walk n times; bin each exit by cos_out. Returns exits."""
    var gn = Vec3f(Float32(0.0), Float32(0.0), Float32(1.0))
    var wo = Vec3f(Float32(0.0), Float32(0.6), Float32(0.8))
    var pcg = PCG32(UInt64(0x9E3779B97F4A7C15), UInt64(1))
    var exits = 0
    for _ in range(n):
        var w = coat_walk_begin(gn, wo, RGB(Float32(1.0)), ior, alpha, pcg)
        # Drive the walk directly: we want the exit-direction law of the base
        # walk, not the entry coin flip (which is a separate lobe).
        var guard = 0
        while w.event == COAT_WALKING and guard < 64:
            coat_walk_scatter(w, pcg)
            guard += 1
        if w.event == COAT_EXIT:
            var c = dot(w.wi, gn)
            if c > Float32(0.0):
                var b = Int(c * Float32(NBINS))
                if b >= NBINS: b = NBINS - 1
                hist[b] += Float32(1.0)
                exits += 1
    return exits


def test_coat_exit_norm_is_a_probability() raises:
    """Z is the escape fraction, so it must lie strictly in (0,1) and fall as
    the coat gets denser (more total internal reflection)."""
    var z15 = coat_exit_norm(Float32(1.5))
    assert_true(z15 > Float32(0.0) and z15 < Float32(1.0))
    assert_true(coat_exit_norm(Float32(2.5)) < z15)
    # eta -> 1 is an index-matched (invisible) interface: everything escapes.
    assert_true(coat_exit_norm(Float32(1.001)) > Float32(0.99))


def test_pdf_integrates_to_one() raises:
    """Numerically integrate the analytic pdf over the outside hemisphere.
    With azimuthal symmetry, int p dw = 2*pi*int_0^1 p(mu) dmu."""
    for ior_i in range(3):
        var ior = Float32(1.2) + Float32(ior_i) * Float32(0.5)   # 1.2, 1.7, 2.2
        comptime N = 2048
        var acc = Float32(0.0)
        for i in range(N):
            var mu = (Float32(i) + Float32(0.5)) / Float32(N)
            acc += bxdf_pdf_coated_exit(mu, ior)
        var integral = Float32(2.0) * PI * acc / Float32(N)
        assert_true(abs(integral - Float32(1.0)) < Float32(2e-3),
                    String("pdf must integrate to 1, got ") + String(integral)
                    + String(" at ior ") + String(ior))


def test_pdf_matches_the_real_sampler() raises:
    """THE test: histogram the real coat_walk_scatter's exit directions and
    compare bin-by-bin against the analytic pdf integrated over each bin.

    Bin [a,b] of cos_out has expected probability
        int_a^b p(mu) 2*pi dmu
    since p depends on direction only through cos_out."""
    var ior = Float32(1.5)
    var hist = List[Float32]()
    for _ in range(NBINS):
        hist.append(Float32(0.0))
    var n = 200000
    var exits = _run_walks(ior, Float32(0.0), n, hist)
    assert_true(exits > n // 2, String("expected most walks to exit, got ") + String(exits))

    var worst = Float32(0.0)
    for b in range(NBINS):
        var lo = Float32(b) / Float32(NBINS)
        var hi = Float32(b + 1) / Float32(NBINS)
        comptime S = 64
        var acc = Float32(0.0)
        for i in range(S):
            var mu = lo + (hi - lo) * (Float32(i) + Float32(0.5)) / Float32(S)
            acc += bxdf_pdf_coated_exit(mu, ior)
        var expect = Float32(2.0) * PI * acc * (hi - lo) / Float32(S)
        var got = hist[b] / Float32(exits)
        # Only judge bins with enough mass for the sample noise to be small.
        if expect > Float32(0.01):
            var rel = abs(got - expect) / expect
            if rel > worst: worst = rel
    assert_true(worst < Float32(0.03),
                String("sampled histogram must match the analytic pdf; worst relative error ")
                + String(worst))


def test_pdf_is_zero_below_the_horizon() raises:
    assert_equal(bxdf_pdf_coated_exit(Float32(0.0), Float32(1.5)), Float32(0.0))
    assert_equal(bxdf_pdf_coated_exit(Float32(-0.5), Float32(1.5)), Float32(0.0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
