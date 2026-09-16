"""Classical dipole BSSRDF: the kernel must integrate to the closed-form
total diffuse reflectance, and must conserve energy.

The value of this test is that the dipole has an ANALYTIC total: Jensen et al.
2001 eq. 15 gives

    Rd_total = (alpha'/2) (1 + exp(-4/3 A sqrt(3(1-alpha')))) exp(-sqrt(3(1-alpha')))

independently of the spatial profile. So numerically integrating our R_d(r)
over the plane, 2*pi*integral(r R_d(r) dr), has something exact to be checked
against -- not another implementation of the same thing.
"""
from std.math import exp, sqrt
from gonzales.rng import PCG32
from std.testing import assert_true, TestSuite
from gonzales.bssrdf import dipole_rd_channel, fdr_moment, dipole_max_radius, dipole_sample_radius, dipole_sample_pdf_mis, dipole_mis_sigma_tr
from gonzales.geometry import RGB


def _close(a: Float32, b: Float32, tol: Float32 = Float32(0.02)) -> Bool:
    var d = a - b
    if d < Float32(0): d = -d
    return d <= tol * (Float32(1.0) if b == Float32(0) else (b if b > Float32(0) else -b)) + Float32(1e-6)


def _analytic_total(sigma_s: Float32, sigma_a: Float32, g: Float32, eta: Float32) -> Float32:
    var ss_p = sigma_s * (Float32(1.0) - g)
    var st_p = ss_p + sigma_a
    var alpha_p = ss_p / st_p
    var fdr = fdr_moment(eta)
    var A = (Float32(1.0) + fdr) / (Float32(1.0) - fdr)
    var s = sqrt(Float32(3.0) * (Float32(1.0) - alpha_p))
    return (alpha_p / Float32(2.0)) * (Float32(1.0) + exp(-Float32(4.0)/Float32(3.0) * A * s)) * exp(-s)


def _numeric_total(sigma_s: Float32, sigma_a: Float32, g: Float32, eta: Float32) -> Float32:
    # 2*pi * integral_0^R r Rd(r) dr, midpoint rule over a generous range.
    var rmax = dipole_max_radius(RGB(sigma_s), RGB(sigma_a), g) * Float32(6.0)
    comptime N = 200000
    var dr = rmax / Float32(N)
    var acc = Float32(0.0)
    for i in range(N):
        var r = (Float32(i) + Float32(0.5)) * dr
        acc += r * dipole_rd_channel(sigma_s, sigma_a, g, eta, r) * dr
    return Float32(2.0) * Float32(3.14159265) * acc


def test_dipole_integrates_to_analytic_total() raises:
    """The spatial profile must carry exactly the energy the closed form says."""
    # A spread of albedos, including skin-like (very high) and absorbing.
    var cases_ss: List[Float32] = [100.0, 100.0, 10.0, 1000.0]
    var cases_sa: List[Float32] = [1.0,   10.0,  5.0,  1.0]
    for i in range(len(cases_ss)):
        var num = _numeric_total(cases_ss[i], cases_sa[i], Float32(0.0), Float32(1.33))
        var ana = _analytic_total(cases_ss[i], cases_sa[i], Float32(0.0), Float32(1.33))
        assert_true(_close(num, ana, Float32(0.05)),
                    "dipole profile must integrate to the analytic total")


def test_dipole_conserves_energy() raises:
    """No parameter choice may return more power than entered."""
    var cases_ss: List[Float32] = [1.0, 100.0, 1000.0, 10000.0]
    for i in range(len(cases_ss)):
        var t = _numeric_total(cases_ss[i], Float32(0.01), Float32(0.0), Float32(1.33))
        assert_true(t <= Float32(1.0) + Float32(1e-3), "dipole must not create energy")
        assert_true(t >= Float32(0.0), "dipole must not be negative")


def test_dipole_is_positive_and_decreasing() raises:
    var prev = dipole_rd_channel(Float32(100), Float32(1), Float32(0), Float32(1.33), Float32(0.001))
    for i in range(1, 60):
        var r = Float32(0.001) * Float32(i + 1)
        var cur = dipole_rd_channel(Float32(100), Float32(1), Float32(0), Float32(1.33), r)
        assert_true(cur >= Float32(0.0), "R_d must be non-negative")
        assert_true(cur <= prev + Float32(1e-9), "R_d must decrease with distance")
        prev = cur


def test_sampler_is_unbiased_against_the_analytic_total() raises:
    """Integrating R_d THROUGH the sampler must give the same analytic total
    the direct quadrature does. This is what a bidirectional integrator needs
    and a gather does not: an exit point with a known pdf, such that
    R_d/pdf is an unbiased estimator. If the pdf and the sampling routine ever
    drift apart, this is what catches it."""
    var rng = PCG32(UInt64(0x243F6A8885A308D3), UInt64(17))
    var cases_ss: List[Float32] = [100.0, 100.0, 10.0]
    var cases_sa: List[Float32] = [1.0,   10.0,  5.0]
    for i in range(len(cases_ss)):
        var ss = RGB(cases_ss[i]); var sa = RGB(cases_sa[i])
        var acc = Float32(0.0)
        comptime N = 400000
        for _ in range(N):
            # pick a channel uniformly, exactly as the renderer does
            var c = Int(rng.next_float() * Float32(3.0))
            if c > 2: c = 2
            var str_c = dipole_mis_sigma_tr(ss, sa, Float32(0.0), c)
            var r = dipole_sample_radius(str_c, rng.next_float())
            var pdf = dipole_sample_pdf_mis(ss, sa, Float32(0.0), r)
            if pdf > Float32(0.0):
                acc += dipole_rd_channel(cases_ss[i], cases_sa[i], Float32(0.0), Float32(1.33), r) / pdf
        var est = acc / Float32(N)
        var ana = _analytic_total(cases_ss[i], cases_sa[i], Float32(0.0), Float32(1.33))
        assert_true(_close(est, ana, Float32(0.06)),
                    "sampled estimate of the profile must match the analytic total")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
