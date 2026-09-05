# Tests for the Sobol direction-number table built by
# _generate_sobol_matrices from src/gonzales/data/new-joe-kuo-6.21201.
#
# Regression guard for a bug that was silent for a long time: the data file
# opens with a COLUMN-NAME HEADER line ("d s a m_i") and every data line has
# FOUR leading columns ("d s a m1 .. ms"). The parser skipped only '#'
# comments and read just three columns, so (a) the header fell through the
# number parser, wrote no direction numbers, and still advanced the
# dimension counter -- silently consuming dimension 1 and leaving its whole
# 52-entry matrix zero -- and (b) every data line landed one column to the
# left (d read as s, s as a, a as m1).
#
# The visible symptom of (a) was subtle enough to survive: sobol_sample(_, 1,
# _, _) returned exactly 0.0 for every sample, which pins the y pixel-filter
# offset at -yradius. That shifts every path-traced image DOWN by the
# filter's y radius with zero vertical antialiasing -- 5px on cornell-box,
# whose PixelFilter has yradius 5. Dimension 0 is hardcoded to the identity
# matrix above the parse loop, so x was unaffected and the image still
# looked plausible.
from std.testing import assert_true, assert_equal, TestSuite
from gonzales.pipeline import _generate_sobol_matrices
from gonzales.sampling import sobol_sample

comptime N_BITS = 52
comptime DATA = "src/gonzales/data/new-joe-kuo-6.21201"

def test_dimension_1_matrix_is_not_all_zero() raises:
    """The header line must not consume a dimension. Dimension 1's 52
    direction numbers come from the file's FIRST data row (d=2) and cannot
    be all zero -- if they are, sobol_sample returns a constant 0.0."""
    var opt = _generate_sobol_matrices(DATA)
    assert_true(Bool(opt), "Sobol data file failed to load")
    var m = opt.value()
    var nonzero = 0
    for i in range(N_BITS):
        if m[N_BITS + i] != UInt32(0):
            nonzero += 1
    assert_true(nonzero > 0, "dimension 1 matrix is entirely zero")
    m.free()

def test_dimension_1_columns_match_joe_kuo_first_row() raises:
    """The file's first data row is `2 1 0 1`: d=2 (=> 0-indexed dim 1),
    s=1, a=0, m=[1]. That fixes v[0] = 1<<31 and, since s=1 leaves the
    polynomial term empty, the recurrence v[i] = v[i-1] ^ (v[i-1] >> 1)
    for every later column. Reading the row one column to the left
    instead (s=2, a=1, m=[0,1]) gives v[0] = 0<<31 = 0 and
    v[1] = 1<<30 -- both caught here, the first by the v[0] check."""
    var opt = _generate_sobol_matrices(DATA)
    assert_true(Bool(opt), "Sobol data file failed to load")
    var m = opt.value()
    assert_equal(m[N_BITS + 0], UInt32(1) << UInt32(31))
    for i in range(1, N_BITS):
        var prev = m[N_BITS + i - 1]
        assert_equal(m[N_BITS + i], prev ^ (prev >> UInt32(1)))
    m.free()

def test_sobol_sample_dimension_1_varies_and_is_uniform() raises:
    """Dimension 1 must actually vary over [0,1) -- the property the pixel
    filter's y offset depends on. Checks spread, mean and both halves of
    the unit interval rather than just 'not always the same number'."""
    var opt = _generate_sobol_matrices(DATA)
    assert_true(Bool(opt), "Sobol data file failed to load")
    var m = opt.value()
    var n = 256
    var total = Float32(0)
    var lo = 0
    var hi = 0
    var vmin = Float32(2)
    var vmax = Float32(-1)
    for i in range(n):
        var u = sobol_sample(i, 1, UInt32(0x9e3779b9), m)
        assert_true(u >= Float32(0) and u < Float32(1), "sample out of [0,1)")
        total += u
        if u < Float32(0.5): lo += 1
        else: hi += 1
        if u < vmin: vmin = u
        if u > vmax: vmax = u
    m.free()
    assert_true(vmax - vmin > Float32(0.9), "dimension 1 barely varies")
    var mean = total / Float32(n)
    assert_true(mean > Float32(0.4) and mean < Float32(0.6), "mean far from 0.5")
    assert_true(lo > 0 and hi > 0, "all samples fell in one half")

def test_dimension_0_is_identity_and_differs_from_1() raises:
    """Dimension 0 is hardcoded (not parsed). It stayed correct throughout
    the bug, which is why only the y axis was affected -- assert it, and
    assert the two dimensions are genuinely decorrelated rather than
    accidentally identical sequences."""
    var opt = _generate_sobol_matrices(DATA)
    assert_true(Bool(opt), "Sobol data file failed to load")
    var m = opt.value()
    for j in range(N_BITS):
        assert_equal(m[j], UInt32(1) << UInt32(31 - j))
    var differing = 0
    for i in range(64):
        var u0 = sobol_sample(i, 0, UInt32(0x12345678), m)
        var u1 = sobol_sample(i, 1, UInt32(0x9e3779b9), m)
        if u0 != u1: differing += 1
    m.free()
    assert_true(differing > 32, "dimensions 0 and 1 are suspiciously alike")

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
