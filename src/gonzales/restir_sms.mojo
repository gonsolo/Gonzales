# ReSTIR SMS (Hong et al. 2025, "Sample Space Partitioning and
# Spatiotemporal Resampling for Specular Manifold Sampling"), Phase 6 of
# docs/A2_restir_migration_plan.md. Mirrors restir_di.mojo/restir_gi.mojo's
# shape: payload type + pure target-function math, no ShadeContext
# dependency (candidate generation needs real ray tracing/BSDF evaluation
# via shading.mojo, kept separate to avoid a circular import).
#
# Scoped-down relative to the published method -- see
# project_sms_restir_phase6.md memory for the full research trail and
# rationale. Two deliberate simplifications, both documented there:
#   1. No tile-based sample-space partitioning / per-frame prior
#      distribution (the paper's Sections 3-4) -- candidates still come
#      from gonzales's existing straight-line-probe + Bernoulli-trial SMS
#      (sms.mojo's sms_solve_bernoulli), which is noisier/more expensive
#      per-candidate than the paper's own pipeline. ReSTIR still helps by
#      amortizing that cost across frames via temporal reuse.
#   2. Basic Bitterli-2020-style reservoir_combine + Z-normalization
#      (reservoir.mojo, already used by DI/GI) instead of the paper's
#      recommended "generalized pairwise heuristic" MIS (which needs
#      shifting the current sample to EVERY neighbor too -- 4 manifold
#      walks per neighbor reused from). Same simplification DI/GI already
#      made; pairwise MIS was separately investigated and ruled out this
#      session as not worth the rewrite risk for gonzales's reservoir
#      core (see project_restir_migration.md).
#
# The manifold SHIFT (this file's `sms_shift`) is a direct, mechanical
# reuse of sms.mojo's existing sms_walk -- no new Newton-solve code:
# forward-walk seeded from a neighbor's stored chain into the current
# pixel's domain, then backward-walk to verify invertibility. The
# bijectivity check formula and its 1e-4 threshold are taken verbatim from
# the reference implementation (github.com/Utah-Graphics-Lab/PSMS-ReSTIR,
# Shift.slang/SMS.slang), not guessed: reject the shift unless
# abs(dot(originalDir, baseDir) - 1.0) <= uniqueness_threshold, comparing
# DIRECTIONS from each domain's own shading point to its own chain's
# first vertex (not raw vertex positions -- direction-based comparison is
# scale-invariant between near/far solutions).

from std.math import sqrt, abs
from .geometry import RGB, dot, INV_PI
from .reservoir import ReservoirState, reservoir_state_init
from .sms import SMSVertex, MAX_SMS_VERTICES, sms_vertex_init, sms_walk

comptime SMS_UNIQUENESS_THRESHOLD: Float32 = Float32(1e-4)

@fieldwise_init
struct SMSReservoir(Copyable, Movable):
    """ReSTIR SMS's reservoir payload: an admissible specular chain plus
    the light it connects to. `n_vertices == 0` means no winner yet
    (mirrors DIReservoir's `light_idx < 0` / GIReservoir's `valid == 0`
    sentinel).

    Note: unlike DIReservoir/GIReservoir, this struct is NOT
    TrivialRegisterPassable -- Mojo's InlineArray never conforms to
    TrivialRegisterPassable regardless of element type (verified
    empirically: even InlineArray[SIMD[...], N] fails the same check),
    so a struct embedding one must fall back to plain Copyable/Movable.
    Copies need explicit `.copy()`.

    No `light_normal` field (unlike DIReservoir): sms_target_pdf has no
    cos_l term -- SMS's light-side geometry is already folded into
    `bsdf_product`/`dx1_dxlight` by the manifold walk itself, matching
    shading.mojo's pre-existing `mnee_wtN` formula, which also has no
    separate cos_l factor."""
    var n_vertices:   Int32
    var verts:        InlineArray[SMSVertex, MAX_SMS_VERTICES]
    var light_point:  SIMD[DType.float32, 3]
    var ldp_du:       SIMD[DType.float32, 3]
    var ldp_dv:       SIMD[DType.float32, 3]
    var le:           RGB
    var state:        ReservoirState

@always_inline
def sms_reservoir_init() -> SMSReservoir:
    var z3 = SIMD[DType.float32, 3](Float32(0.0))
    return SMSReservoir(
        n_vertices=Int32(0),
        verts=InlineArray[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init()),
        light_point=z3, ldp_du=z3, ldp_dv=z3,
        le=RGB(Float32(0.0)),
        state=reservoir_state_init(),
    )

@always_inline
def sms_target_pdf(
    hit_point: SIMD[DType.float32, 3], normal: SIMD[DType.float32, 3], alb: RGB,
    first_vertex: SIMD[DType.float32, 3], first_normal: SIMD[DType.float32, 3],
    le: RGB, bsdf_product: Float32, dx1_dxlight: Float32,
) -> Float32:
    """GRIS target function p̂(x̂) for an SMS candidate (Hong et al. 2025
    Eq. 12-13): ρ(x1,ωo,ω*)·Tr(x1↔x*↔x_n)·Le(x_n) -- the same quantity
    shading.mojo's `_mnee_area_light_contribute` already computes as its
    per-channel `mnee_wtN` contribution, returned here as luminance only
    (matching di_target_pdf/gi_target_pdf's scalar-target convention).

    `bsdf_product`/`dx1_dxlight` are sms_walk's own byproducts of solving
    (or shifting) the chain AT `hit_point` -- this function does no
    solving itself, it only evaluates the already-converged chain's
    contribution, exactly like di_target_pdf/gi_target_pdf are pure
    functions of already-known quantities. Excludes the sampling-side
    correction factors (inv_pdf_area, lobe_w, Bernoulli trial count) that
    belong to the RESAMPLING weight's OTHER factor (q, the generation
    density), not to p̂ itself -- same separation di_target_pdf makes by
    excluding 1/gen_pdf."""
    var wi_f = hit_point - first_vertex
    var wi_len = sqrt(dot(wi_f, wi_f))
    if wi_len < Float32(1e-8):
        return Float32(0.0)
    var wi_fn = wi_f * (Float32(1.0) / wi_len)
    var cos_s_x0 = dot(normal, -wi_fn)
    if cos_s_x0 <= Float32(0.0):
        return Float32(0.0)
    var dw0_dx1 = abs(dot(wi_fn, first_normal)) / (wi_len * wi_len)
    var g = min(dw0_dx1 * dx1_dxlight, Float32(2.0))
    var contrib = alb * le * (INV_PI * cos_s_x0 * g * bsdf_product)
    return contrib.r * Float32(0.2126) + contrib.g * Float32(0.7152) + contrib.b * Float32(0.0722)

@always_inline
def sms_shift(
    dst_x0: SIMD[DType.float32, 3], dst_light_point: SIMD[DType.float32, 3],
    dst_ldp_du: SIMD[DType.float32, 3], dst_ldp_dv: SIMD[DType.float32, 3],
    src_x0: SIMD[DType.float32, 3], src_light_point: SIMD[DType.float32, 3],
    src_ldp_du: SIMD[DType.float32, 3], src_ldp_dv: SIMD[DType.float32, 3],
    src_verts: InlineArray[SMSVertex, MAX_SMS_VERTICES], n: Int,
) -> Tuple[Bool, InlineArray[SIMD[DType.float32, 3], MAX_SMS_VERTICES], Float32, Float32]:
    """Manifold shift (Hong et al. 2025 Section 5.2): reuse a neighbor's
    (`src`) admissible specular chain at the current pixel (`dst`) by
    re-walking it seeded from the neighbor's own solution, then verifying
    the shift is invertible by walking back. Returns (accepted, shifted
    positions, bsdf_product, dx1_dxlight) -- the last two are sms_walk's
    own byproducts, ready to feed sms_target_pdf directly. `accepted =
    False` means either walk failed to converge OR the bijectivity check
    rejected it -- both must be treated as p̂=0 in dst's domain (GRIS
    shift-invalid convention), never as a silent fallback to the
    unshifted src chain.

    Bijectivity check ported verbatim from the reference implementation
    (Shift.slang/SMS.slang): compare the DIRECTION from src's own shading
    point to src's own first vertex (`original_dir`) against the
    direction obtained by walking the shifted (dst) solution back into
    src's domain (`base_dir`) -- reject unless
    abs(dot(original_dir, base_dir) - 1.0) <= SMS_UNIQUENESS_THRESHOLD.
    Direction-based (not raw position distance) so the check is scale-
    invariant between near and far solutions."""
    var zero_positions = InlineArray[SIMD[DType.float32, 3], MAX_SMS_VERTICES](fill=SIMD[DType.float32, 3](Float32(0.0)))
    var src_first = src_verts[0].pos
    var orig_dir_v = src_first - src_x0
    var orig_dir_len = sqrt(dot(orig_dir_v, orig_dir_v))
    if orig_dir_len < Float32(1e-8):
        return (False, zero_positions, Float32(0.0), Float32(0.0))
    var original_dir = orig_dir_v * (Float32(1.0) / orig_dir_len)

    # ── Forward shift: walk seeded from src's chain, into dst's domain ──
    var (fwd_ok, fwd_pos, fwd_bsdf, fwd_jac) = sms_walk(
        dst_x0, dst_light_point, src_verts, n, dst_ldp_du, dst_ldp_dv)
    if not fwd_ok:
        return (False, zero_positions, Float32(0.0), Float32(0.0))

    # ── Backward verification: walk the shifted chain back into src's domain ──
    var back_seed = src_verts
    for i in range(n):
        back_seed[i].pos = fwd_pos[i]
    var (back_ok, back_pos, _back_bsdf, _back_jac) = sms_walk(
        src_x0, src_light_point, back_seed, n, src_ldp_du, src_ldp_dv)
    if not back_ok:
        return (False, zero_positions, Float32(0.0), Float32(0.0))

    var base_dir_v = back_pos[0] - src_x0
    var base_dir_len = sqrt(dot(base_dir_v, base_dir_v))
    if base_dir_len < Float32(1e-8):
        return (False, zero_positions, Float32(0.0), Float32(0.0))
    var base_dir = base_dir_v * (Float32(1.0) / base_dir_len)

    if abs(dot(original_dir, base_dir) - Float32(1.0)) > SMS_UNIQUENESS_THRESHOLD:
        return (False, zero_positions, Float32(0.0), Float32(0.0))

    return (True, fwd_pos, fwd_bsdf, fwd_jac)
