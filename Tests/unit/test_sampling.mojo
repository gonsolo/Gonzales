from std.math import abs, sqrt
from std.testing import assert_true, assert_false, TestSuite
from gonzales.sampling import (
    power_heuristic, sample_cosine_hemisphere,
    reverse_bits32, mix_bits_u64, encode_morton2, fast_owen_scramble,
)

comptime EPS: Float32 = 1e-5

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# ── power_heuristic ──────────────────────────────────────────────────────────
# Balance heuristic with β=2: w(f) = f²/(f²+g²).

def test_power_heuristic_equal_pdfs_split_evenly() raises:
    """Two strategies with identical pdfs must always get exactly half the
    weight each, regardless of the pdf's absolute magnitude."""
    for pdf in [Float32(0.5), Float32(2.0), Float32(100.0)]:
        assert_true(_close(power_heuristic(pdf, pdf), Float32(0.5)))

def test_power_heuristic_only_strategy_gets_full_weight() raises:
    """If the other strategy's pdf is 0 (it could not have produced this
    sample), the sampling strategy that did produce it gets weight 1."""
    assert_true(_close(power_heuristic(Float32(1.0), Float32(0.0)), Float32(1.0)))
    assert_true(_close(power_heuristic(Float32(3.7), Float32(0.0)), Float32(1.0)))

def test_power_heuristic_both_zero_hits_denom_guard() raises:
    assert_true(power_heuristic(Float32(0.0), Float32(0.0)) == Float32(0.0))

# ── sample_cosine_hemisphere ─────────────────────────────────────────────────
# Malley's method: z = cos(theta) = sqrt(1-u1) exactly by construction,
# independent of u2 (which only controls the azimuthal angle phi).

def test_sample_cosine_hemisphere_z_matches_sqrt_one_minus_u1() raises:
    for u1 in [Float32(0.0), Float32(0.5), Float32(0.99)]:
        var v = sample_cosine_hemisphere(u1, Float32(0.25))
        assert_true(_close(v.z, sqrt(Float32(1.0) - u1)))

def test_sample_cosine_hemisphere_is_unit_length() raises:
    """The construction guarantees x²+y² = u1 and z² = 1-u1, so the vector
    must land on the unit sphere for any (u1,u2) in range."""
    for u1 in [Float32(0.1), Float32(0.4), Float32(0.9)]:
        for u2 in [Float32(0.05), Float32(0.5), Float32(0.95)]:
            var v = sample_cosine_hemisphere(u1, u2)
            var len_sq = v.x * v.x + v.y * v.y + v.z * v.z
            assert_true(_close(len_sq, Float32(1.0)))

def test_sample_cosine_hemisphere_stays_in_upper_hemisphere() raises:
    var v = sample_cosine_hemisphere(Float32(0.3), Float32(0.7))
    assert_true(v.z >= Float32(0.0))

# ── reverse_bits32 ───────────────────────────────────────────────────────────

def test_reverse_bits32_zero_is_zero() raises:
    assert_true(reverse_bits32(UInt32(0)) == UInt32(0))

def test_reverse_bits32_single_low_bit_becomes_high_bit() raises:
    """The single set bit at position 0 must land at position 31."""
    assert_true(reverse_bits32(UInt32(1)) == UInt32(0x80000000))

def test_reverse_bits32_is_involution() raises:
    """Reversing twice must return the original value for arbitrary inputs."""
    for x in [UInt32(1), UInt32(0xdeadbeef), UInt32(0x12345678), UInt32(0xffffffff), UInt32(0x00ff00ff)]:
        assert_true(reverse_bits32(reverse_bits32(x)) == x)

# ── mix_bits_u64 ─────────────────────────────────────────────────────────────
# Weak avalanche sanity check only — not a claim of cryptographic strength,
# just that distinct inputs produce distinct outputs on the cases tried.

def test_mix_bits_u64_distinct_inputs_give_distinct_outputs() raises:
    var pairs = [
        (UInt64(0), UInt64(1)),
        (UInt64(1), UInt64(2)),
        (UInt64(12345), UInt64(12346)),
        (UInt64(0xdeadbeef), UInt64(0xfeedface)),
    ]
    for pair in pairs:
        assert_true(mix_bits_u64(pair[0]) != mix_bits_u64(pair[1]))

def test_mix_bits_u64_is_not_constant() raises:
    var a = mix_bits_u64(UInt64(0))
    var b = mix_bits_u64(UInt64(42))
    assert_true(a != b)

# ── encode_morton2 ───────────────────────────────────────────────────────────
# x-bits go to even output bit positions, y-bits (after <<1) to odd positions.

def test_encode_morton2_zero_is_zero() raises:
    assert_true(encode_morton2(UInt32(0), UInt32(0)) == UInt64(0))

def test_encode_morton2_x_bit_goes_to_even_position() raises:
    assert_true(encode_morton2(UInt32(1), UInt32(0)) == UInt64(1))
    assert_true(encode_morton2(UInt32(2), UInt32(0)) == UInt64(4))

def test_encode_morton2_y_bit_goes_to_odd_position() raises:
    assert_true(encode_morton2(UInt32(0), UInt32(1)) == UInt64(2))
    assert_true(encode_morton2(UInt32(0), UInt32(2)) == UInt64(8))

def test_encode_morton2_interleaves_both() raises:
    assert_true(encode_morton2(UInt32(1), UInt32(1)) == UInt64(3))
    assert_true(encode_morton2(UInt32(3), UInt32(3)) == UInt64(15))

# ── fast_owen_scramble ───────────────────────────────────────────────────────
# Only easily-testable invariant: changing the seed changes the output for a
# fixed input value (no claim about the quality of the scramble itself).

def test_fast_owen_scramble_seed_changes_output() raises:
    var v = UInt32(12345)
    var out_a = fast_owen_scramble(v, UInt32(1))
    var out_b = fast_owen_scramble(v, UInt32(2))
    assert_true(out_a != out_b)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
