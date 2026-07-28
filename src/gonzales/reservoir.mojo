# Weighted reservoir sampling (RIS/GRIS core), Phase 0 of
# docs/A2_restir_migration_plan.md.
#
# Design note: this module holds no payload type. `ReservoirState` is just
# the bookkeeping triple (w_sum, m, w); the actual candidate (a light
# sample, a path suffix, a specular chain, ...) lives in the caller's own
# struct alongside a `ReservoirState` field, swapped in manually when
# `reservoir_update`/`reservoir_combine` return True. This sidesteps
# generics entirely (no parametric struct exists anywhere else in this
# codebase) and matches how real ReSTIR shader implementations (HLSL/Slang)
# do it: `if (Update(...)) { self.payloadField = candidate.field; ... }`.
# Payload-specific reservoirs (e.g. a future DIReservoir for Phase 2) should
# be thin wrappers composing a `ReservoirState`, not reimplementations of
# the math below.

@fieldwise_init
struct ReservoirState(TrivialRegisterPassable):
    var w_sum: Float32  # running sum of streamed resampling weights (RIS numerator)
    var m: Float32       # confidence weight -- number of candidates this reservoir represents
    var w: Float32        # finalized corrective weight; multiply the chosen sample's own
                           # contribution by this to get an unbiased estimate (see reservoir_finalize)

@always_inline
def reservoir_state_init() -> ReservoirState:
    return ReservoirState(Float32(0.0), Float32(0.0), Float32(0.0))

@always_inline
def _reservoir_stream(mut state: ReservoirState, weight: Float32, u: Float32) -> Bool:
    """Core weighted-reservoir-sampling step, shared by reservoir_update and
    reservoir_combine: stream one candidate of resampling weight `weight`,
    accept it (replacing the current sample) with probability
    weight / new_w_sum. Does NOT touch `m` -- callers increment confidence
    differently (update: +1 fresh candidate; combine: += the other
    reservoir's own m), so that bookkeeping is the caller's responsibility.
    `u` must be a fresh uniform random number in [0,1), independent of any
    number used to generate the candidate itself (reusing correlated random
    numbers here is a classic, hard-to-spot source of bias)."""
    if weight <= Float32(0.0):
        return False
    state.w_sum += weight
    return u * state.w_sum < weight

@always_inline
def reservoir_update(mut state: ReservoirState, weight: Float32, u: Float32) -> Bool:
    """Stream one freshly-generated candidate into the reservoir (e.g. one
    of the M initial RIS candidates for a pixel, Phase 2 step 2.2).
    `weight` is the candidate's own RIS weight, p_hat(candidate)/q(candidate)
    -- computed by the caller, since this module knows nothing about
    payloads or target functions. Returns True iff this candidate became the
    new winner: the caller must then copy the candidate's own fields into
    its sample slot. Always increments `m` by 1, whether accepted or not --
    confidence reflects how many candidates were considered, not which won."""
    var accept = _reservoir_stream(state, weight, u)
    state.m += Float32(1.0)
    return accept

@always_inline
def reservoir_combine(
    mut dst: ReservoirState, src: ReservoirState,
    target_pdf_in_dst_domain: Float32, u: Float32,
) -> Bool:
    """Merge `src`'s already-chosen sample into `dst` as one RIS candidate
    (GRIS-style spatial/temporal reservoir combination, Phase 2 steps
    2.3/2.5). `target_pdf_in_dst_domain` is p_hat(src's sample) as evaluated
    in DST's domain -- i.e. after applying whatever shift mapping and
    Jacobian correction make src's sample valid at dst (reconnection shift,
    manifold shift, ...). This module has no knowledge of shift mappings; the
    caller must compute that value before calling. Returns True iff src's
    sample becomes dst's new winner (caller must copy payload); `m` always
    accumulates (dst.m += src.m) regardless of outcome. `src.m` should
    already reflect the caller's own M-cap policy (reservoir_cap_confidence)
    before this call, so history doesn't grow unboundedly across repeated
    temporal reuse."""
    var weight = target_pdf_in_dst_domain * src.w * src.m
    var accept = _reservoir_stream(dst, weight, u)
    dst.m += src.m
    return accept

@always_inline
def reservoir_finalize(mut state: ReservoirState, target_pdf_of_chosen_sample: Float32):
    """Compute the final corrective weight W = w_sum / (m * p_hat(chosen)),
    the standard RIS estimator correction (Bitterli et al. 2020, Eq. 6).
    Call once after all streaming/combining for this pixel/frame is done,
    passing p_hat of whichever sample ended up as the winner. The caller
    then uses state.w to weight that sample's own (unshadowed, in the DI
    case) contribution -- contribution * state.w is the unbiased estimate.
    Degenerate cases (empty reservoir, zero-probability winner) set w=0
    rather than producing NaN/Inf, so a degenerate reservoir contributes
    nothing instead of corrupting the image."""
    if state.w_sum <= Float32(0.0) or target_pdf_of_chosen_sample <= Float32(0.0) or state.m <= Float32(0.0):
        state.w = Float32(0.0)
        return
    state.w = state.w_sum / (state.m * target_pdf_of_chosen_sample)

@always_inline
def reservoir_cap_confidence(mut state: ReservoirState, max_m: Float32):
    """Bound `m` to `max_m` (Phase 0.2; guideline from the migration plan:
    roughly 20x the per-frame candidate count). This is the standard ReSTIR
    temporal-history clamp (Bitterli et al. 2020): it does NOT rescale
    w_sum, only m -- so it must be called AFTER reservoir_finalize for the
    current frame's own shading (which should use the true, uncapped
    confidence), and BEFORE storing the reservoir for the next frame/pass to
    reuse (where capped confidence prevents old history from permanently
    dominating fresh candidates). Calling it before finalize would inflate
    the current frame's own W incorrectly."""
    if state.m > max_m:
        state.m = max_m
