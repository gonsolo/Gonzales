from std.testing import assert_true, assert_false, TestSuite
from gonzales.rng import PCG32

# ── PCG32 ────────────────────────────────────────────────────────────────────

def test_pcg32_same_seed_is_deterministic() raises:
    """Two generators constructed with the identical (state, inc) seed must
    produce the exact same sequence — this is the whole point of seeding a
    PRNG explicitly (reproducible renders / per-pixel streams)."""
    var a = PCG32(UInt64(12345), UInt64(67))
    var b = PCG32(UInt64(12345), UInt64(67))
    for _ in range(50):
        assert_true(a.next_float() == b.next_float())

def test_pcg32_next_float_stays_in_unit_range() raises:
    """Next_float must always land in [0,1) — callers (sampling code) divide
    by this and rely on it never returning 1.0 or a negative value."""
    var gen = PCG32(UInt64(999), UInt64(11))
    for _ in range(1000):
        var f = gen.next_float()
        assert_true(f >= Float32(0.0))
        assert_true(f < Float32(1.0))

def test_pcg32_different_seeds_diverge() raises:
    """A weak sanity check that the seed actually influences the output
    (not a statistical randomness claim): two different (state, inc) pairs
    must not produce an identical run of several draws."""
    var a = PCG32(UInt64(1), UInt64(1))
    var b = PCG32(UInt64(2), UInt64(1))
    var all_equal = True
    for _ in range(8):
        if a.next_float() != b.next_float():
            all_equal = False
    assert_false(all_equal)

def test_pcg32_different_seq_diverges() raises:
    """Same initstate but a different sequence-selection constant (inc) must
    also change the stream — inc is not a no-op parameter."""
    var a = PCG32(UInt64(42), UInt64(1))
    var b = PCG32(UInt64(42), UInt64(2))
    var all_equal = True
    for _ in range(8):
        if a.next_float() != b.next_float():
            all_equal = False
    assert_false(all_equal)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
