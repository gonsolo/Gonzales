from std.testing import assert_true, TestSuite
from std.math import abs
from gonzales.core.geometry import Vec3f, dot
from gonzales.core.rng import PCG32
from gonzales.transport.sss_walk import sss_leave


def test_lobe_weight_has_mean_one_and_stays_outward() raises:
    var n = Vec3f(Float32(0), Float32(0.6), Float32(0.8))
    for eta in [Float32(1.0001), Float32(1.33), Float32(1.6)]:
        var pcg = PCG32(UInt64(7), UInt64(3))
        var sum = Float64(0)
        var count = 200000
        for _ in range(count):
            var l = sss_leave(n, eta, pcg.next_float(), pcg.next_float())
            assert_true(dot(l.wi, n) >= Float32(0))
            sum += Float64(l.weight)
        assert_true(abs(sum / Float64(count) - Float64(1)) < Float64(0.01))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
