from std.math import abs
from std.testing import assert_true, assert_false, TestSuite
from gonzales.reservoir import (
    ReservoirState, reservoir_state_init, reservoir_update, reservoir_combine,
    reservoir_finalize, reservoir_cap_confidence,
)
from gonzales.rng import PCG32

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# ── reservoir_update ──────────────────────────────────────────────────────

def test_reservoir_update_first_positive_weight_candidate_always_accepted() raises:
    """An empty reservoir (w_sum=0) must accept the first candidate with any
    positive weight, for any u in [0,1) -- u*weight < weight reduces to
    u < 1, which is always true for a canonical [0,1) random number."""
    var s = reservoir_state_init()
    var accepted = reservoir_update(s, Float32(2.5), Float32(0.999999))
    assert_true(accepted)
    assert_true(_close(s.w_sum, Float32(2.5)))
    assert_true(_close(s.m, Float32(1.0)))

def test_reservoir_update_zero_weight_candidate_rejected_but_still_counts() raises:
    """A zero (or negative) weight candidate must never be accepted -- but
    per the docstring, m still increments, since confidence reflects how
    many candidates were considered, not which won."""
    var s = reservoir_state_init()
    var accepted = reservoir_update(s, Float32(0.0), Float32(0.0))
    assert_false(accepted)
    assert_true(_close(s.w_sum, Float32(0.0)))
    assert_true(_close(s.m, Float32(1.0)))

def test_reservoir_update_second_candidate_replaces_deterministically() raises:
    """With u chosen so u*w_sum >= weight, the second candidate must be
    rejected and the first remains the (implicit, caller-tracked) winner --
    verified via w_sum still reflecting both streamed weights."""
    var s = reservoir_state_init()
    _ = reservoir_update(s, Float32(1.0), Float32(0.0))   # always accepted (first)
    # w_sum after candidate 1 is 1.0; stream candidate 2 (weight 1.0) with
    # u=0.9999 -> u*w_sum(2.0) = 1.9998 >= weight(1.0) -> rejected.
    var accepted2 = reservoir_update(s, Float32(1.0), Float32(0.9999))
    assert_false(accepted2)
    assert_true(_close(s.w_sum, Float32(2.0)))
    assert_true(_close(s.m, Float32(2.0)))

def test_reservoir_update_selection_probability_matches_weight_ratio() raises:
    """Statistical correctness check: streaming two candidates (weights 1
    and 3) into a fresh reservoir each trial, the second candidate should
    win with probability weight2/(weight1+weight2) = 0.75 -- the defining
    property of weighted reservoir sampling. Fixed PCG32 seed keeps this
    deterministic; tolerance is well beyond binomial noise (std ~= 27 at
    N=4000, band is ~7 sigma)."""
    var pcg = PCG32(UInt64(12345), UInt64(1))
    comptime N = 4000
    var wins2 = 0
    for _ in range(N):
        var s = reservoir_state_init()
        _ = reservoir_update(s, Float32(1.0), pcg.next_float())
        var accepted2 = reservoir_update(s, Float32(3.0), pcg.next_float())
        if accepted2:
            wins2 += 1
    var frac = Float32(wins2) / Float32(N)
    assert_true(frac > Float32(0.70) and frac < Float32(0.80))

# ── reservoir_combine ─────────────────────────────────────────────────────

def test_reservoir_combine_accumulates_confidence_regardless_of_outcome() raises:
    """Confidence (m) must always accumulate (dst.m += src.m), whether or
    not src's sample is adopted -- it tracks candidates considered, not
    wins."""
    var dst = reservoir_state_init()
    _ = reservoir_update(dst, Float32(1.0), Float32(0.0))  # dst.m = 1
    var src = ReservoirState(w_sum=Float32(5.0), m=Float32(3.0), w=Float32(0.0))
    # target_pdf=0 -> weight=0 -> never accepted, but m must still accumulate.
    var accepted = reservoir_combine(dst, src, Float32(0.0), Float32(0.0))
    assert_false(accepted)
    assert_true(_close(dst.m, Float32(4.0)))

def test_reservoir_combine_high_value_source_can_win() raises:
    """A src reservoir with a large finalized weight and high target pdf in
    dst's domain must be able to outweigh dst's existing (low-weight)
    sample -- verifies the combine weight formula
    (target_pdf_in_dst_domain * src.w * src.m) is actually used, not
    ignored."""
    var dst = reservoir_state_init()
    _ = reservoir_update(dst, Float32(0.01), Float32(0.0))  # tiny existing weight
    var src = ReservoirState(w_sum=Float32(100.0), m=Float32(1.0), w=Float32(50.0))
    # combine weight = target_pdf(2.0) * src.w(50.0) * src.m(1.0) = 100.0,
    # vastly larger than dst's current w_sum(0.01). Acceptance threshold is
    # u < weight/new_w_sum = 100.0/100.01 ~= 0.9991 -- u=0.5 clears it with
    # wide margin without relying on floating-point precision near 1.0.
    var accepted = reservoir_combine(dst, src, Float32(2.0), Float32(0.5))
    assert_true(accepted)

# ── reservoir_finalize ─────────────────────────────────────────────────────

def test_reservoir_finalize_matches_hand_computed_ris_weight() raises:
    """W = w_sum / (m * p_hat(chosen)) -- verify against a hand-computed
    example rather than trusting the implementation's own arithmetic."""
    var s = ReservoirState(w_sum=Float32(12.0), m=Float32(4.0), w=Float32(0.0))
    reservoir_finalize(s, Float32(3.0))  # 12.0 / (4.0 * 3.0) = 1.0
    assert_true(_close(s.w, Float32(1.0)))

def test_reservoir_finalize_zero_target_pdf_gives_zero_weight_not_nan() raises:
    """A winner with zero target-function value (e.g. it turned out to be
    occluded) must finalize to W=0, not divide-by-zero garbage."""
    var s = ReservoirState(w_sum=Float32(5.0), m=Float32(2.0), w=Float32(0.0))
    reservoir_finalize(s, Float32(0.0))
    assert_true(_close(s.w, Float32(0.0)))

def test_reservoir_finalize_empty_reservoir_gives_zero_weight() raises:
    var s = reservoir_state_init()
    reservoir_finalize(s, Float32(1.0))
    assert_true(_close(s.w, Float32(0.0)))

# ── reservoir_cap_confidence ────────────────────────────────────────────────

def test_reservoir_cap_confidence_clamps_above_max() raises:
    var s = ReservoirState(w_sum=Float32(9.0), m=Float32(500.0), w=Float32(0.0))
    reservoir_cap_confidence(s, Float32(20.0))
    assert_true(_close(s.m, Float32(20.0)))

def test_reservoir_cap_confidence_leaves_below_max_untouched() raises:
    var s = ReservoirState(w_sum=Float32(9.0), m=Float32(5.0), w=Float32(0.0))
    reservoir_cap_confidence(s, Float32(20.0))
    assert_true(_close(s.m, Float32(5.0)))

def test_reservoir_cap_confidence_does_not_rescale_w_sum() raises:
    """Per the documented design (matches Bitterli et al. 2020's temporal
    clamp): capping m must NOT touch w_sum -- only future combines'
    confidence contribution shrinks, not this reservoir's own accumulated
    weight sum."""
    var s = ReservoirState(w_sum=Float32(9.0), m=Float32(500.0), w=Float32(0.0))
    reservoir_cap_confidence(s, Float32(20.0))
    assert_true(_close(s.w_sum, Float32(9.0)))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
