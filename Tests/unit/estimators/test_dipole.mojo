from std.math import abs, exp, log, sqrt
from std.testing import assert_true, TestSuite
from gonzales.core.geometry import PI
from gonzales.estimators.dipole import Dipole, diffuse_fresnel_reflectance, F4

# R_d(r) integrated over the plane must equal Jensen et al. 2001's closed-form total diffuse reflectance,
# (alpha'/2) (1 + exp(-4/3 A sqrt(3(1 - alpha')))) exp(-sqrt(3(1 - alpha'))), at every lane.


def _total_closed_form(alpha_p: Float64, eta: Float32) -> Float64:
    var fdr = Float64(diffuse_fresnel_reflectance(eta))
    var a = (1.0 + fdr) / (1.0 - fdr)
    var s = sqrt(3.0 * (1.0 - alpha_p))
    return alpha_p / 2.0 * (1.0 + exp(-4.0 / 3.0 * a * s)) * exp(-s)


def test_dipole_integrates_to_total_diffuse_reflectance() raises:
    var sigma_a = F4(0.01, 0.1, 1.0, 0.5)
    var sigma_sp = F4(1.0, 1.0, 1.0, 2.0)
    var eta = Float32(1.33)
    var dip = Dipole.of(sigma_a, sigma_sp, eta)
    # r = e^u: the integral of R_d(r) 2 pi r dr is the integral of R_d(e^u) 2 pi e^(2u) du, trapezoid on a log grid.
    var u0 = log(1e-6)
    var u1 = log(200.0)
    comptime n = 40000
    var h = (u1 - u0) / Float64(n)
    var acc = SIMD[DType.float64, 4](0)
    for k in range(n + 1):
        var r = exp(u0 + Float64(k) * h)
        var v = dip.rd(Float32(r * r))
        var w = (0.5 if k == 0 or k == n else 1.0) * h * 2.0 * PI.cast[DType.float64]() * r * r
        acc += SIMD[DType.float64, 4](Float64(v.v0), Float64(v.v1), Float64(v.v2), Float64(v.v3)) * w
    for c in range(4):
        var ap = Float64(sigma_sp[c] / (sigma_a[c] + sigma_sp[c]))
        var want = _total_closed_form(ap, eta)
        assert_true(abs(acc[c] / want - 1.0) < 0.005,
                    "lane " + String(c) + ": integral " + String(acc[c]) + " vs closed form " + String(want))


def test_dipole_is_positive_and_falls_off() raises:
    var dip = Dipole.of(F4(0.05), F4(1.5), Float32(1.4))
    var near = dip.rd(Float32(0.01))
    var far = dip.rd(Float32(4.0))
    assert_true(near.v0 > Float32(0) and far.v0 > Float32(0), "R_d must be positive")
    assert_true(far.v0 < near.v0, "R_d must fall off with distance")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
