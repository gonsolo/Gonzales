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
def reservoir_finalize(
    mut state: ReservoirState, target_pdf_of_chosen_sample: Float32,
    norm: Float32 = Float32(-1.0),
):
    """Compute the final corrective weight W = w_sum / (norm * p_hat(chosen)),
    the standard RIS estimator correction (Bitterli et al. 2020, Eq. 6).
    Call once after all streaming/combining for this pixel/frame is done,
    passing p_hat of whichever sample ended up as the winner. The caller
    then uses state.w to weight that sample's own (unshadowed, in the DI
    case) contribution -- contribution * state.w is the unbiased estimate.
    Degenerate cases (empty reservoir, zero-probability winner) set w=0
    rather than producing NaN/Inf, so a degenerate reservoir contributes
    nothing instead of corrupting the image.

    `norm` is the count the estimator divides by. Leave it negative (the
    default) to use `state.m`, which is correct whenever every combined
    reservoir shared ONE integration domain -- a single pixel's own
    candidates, or temporal reuse under identity reprojection, where the
    previous frame's domain IS this pixel's domain.

    Pass an explicit value for Bitterli et al. 2020 Algorithm 6's UNBIASED
    multi-domain combination, where it is `Z`: the sum of confidences of
    only those combined reservoirs whose OWN domain could actually have
    produced the chosen sample. Dividing by the full `m` there over-counts
    domains that could never have generated it, which is exactly the bias
    in that paper's Algorithm 4 -- measured here as a mean offset plus a
    convergence rate that stalls instead of falling (MSE floors at bias^2).
    Note `norm` deliberately does NOT touch state.m: the stored confidence
    must stay the true accumulated count for the next frame's reuse and
    M-cap, so only W is renormalized."""
    var denom = state.m if norm < Float32(0.0) else norm
    if state.w_sum <= Float32(0.0) or target_pdf_of_chosen_sample <= Float32(0.0) or denom <= Float32(0.0):
        state.w = Float32(0.0)
        return
    state.w = state.w_sum / (denom * target_pdf_of_chosen_sample)

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


# ── Reprojection strategy (Phase 0.2b) ──────────────────────────────────────
# Maps a pixel in the current frame to the reservoir it should pull temporal
# history from, if any. docs/A2_restir_migration_plan.md fact #3: gonzales
# has zero motion/animation support today, so only two modes are real yet --
# IDENTITY (interactive mode, static camera: pipeline.mojo's render_interactive
# already clears the film, and would need to clear stored reservoirs the same
# way, on any camera move -- so "previous frame, same pixel" is exactly
# correct whenever a reservoir survived) and NONE (no history available:
# final-frame mode's first pass, or any case a caller doesn't want reuse).
# A future host-supplied motion-vector mode (Phase 3.5) is a third value this
# struct exists to make cheap to add later -- don't collapse this back to a
# bare Bool once that lands.
struct ReprojectMode:
    comptime none     = Int32(0)
    comptime identity = Int32(1)

@always_inline
def reproject_pixel(mode: Int32, px: Int) -> Tuple[Bool, Int]:
    """Return (valid, src_pixel_index): where to read a prior reservoir from
    for pixel `px` of the current frame, under reprojection strategy `mode`.
    IDENTITY always reports the same pixel valid -- the caller is responsible
    for having already invalidated (cleared) stored reservoirs on any event
    that breaks the static-camera assumption, exactly as pipeline.mojo does
    for the film today. NONE always reports invalid, `px` unchanged only as
    a harmless placeholder -- callers must check `valid` before using it."""
    if mode == ReprojectMode.identity:
        return (True, px)
    return (False, px)
