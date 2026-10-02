from std.collections import Array
from std.math import abs
from std.memory.alloc import unsafe_alloc
from std.testing import assert_equal, assert_true, TestSuite
from gonzales.sampling import (
    sobol_perm_lookup, fill_filter_lut, filter_lut_lookup, FILTER_LUT_N,
    gaussian_filter_sample_1d,
)

# The original one-byte-per-permutation table, verbatim from before sampling.mojo packed it into three 64-bit constants
# (each byte holds four 2-bit digits, first digit in the top bits). The packed words must decode to exactly these.
def _expected(p: Int, digit: Int) -> Int:
    var enc = [27, 30, 39, 45, 57, 54, 75, 78, 99, 108, 120, 114, 147, 156, 135, 141, 177, 180, 216, 210, 228, 225, 201, 198]
    return (enc[p] >> (2 * (3 - digit))) & 3

def test_sobol_perm_lookup_matches_original_byte_table() raises:
    for p in range(24):
        for digit in range(4):
            assert_equal(sobol_perm_lookup(p, digit), _expected(p, digit))

def test_filter_lut_matches_newton_inverse() raises:
    """The table is the exact inverse sampled at FILTER_LUT_N points; between
    points linear interpolation must stay within a small fraction of a pixel."""
    var lut = unsafe_alloc[Float32](2 * FILTER_LUT_N)
    fill_filter_lut(lut, Float32(2.5), Float32(5.0), Float32(3.0))
    var worst = Float32(0.0)
    for k in range(1, 400):
        var u = (Float32(k) + Float32(0.37)) / Float32(400.0)
        var dx = abs(filter_lut_lookup(lut, u) - gaussian_filter_sample_1d(u, Float32(2.5), Float32(5.0)))
        var dy = abs(filter_lut_lookup(lut.unsafe_offset(FILTER_LUT_N), u) - gaussian_filter_sample_1d(u, Float32(2.5), Float32(3.0)))
        worst = max(worst, max(dx, dy))
    lut.unsafe_free()
    assert_true(worst < Float32(2e-3))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
