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

from std.math import sqrt, abs, min, cos, sin
from .geometry import RGB, dot, INV_PI, Vec3f, _is_real_ptr
from .reservoir import ReservoirState, reservoir_state_init
from .sms import SMSVertex, MAX_SMS_VERTICES, sms_vertex_init, sms_walk, sms_refresh_solved_frames
from .rng import PCG32
from .reservoir import reservoir_combine, reservoir_finalize

comptime SMS_UNIQUENESS_THRESHOLD: Float32 = Float32(1e-4)

# ── Spatial reuse (Phase 6, second driver) ───────────────────────────────────
# Same tap count/radius and same G-buffer rejection thresholds restir_gi.mojo
# uses -- these are screen-space heuristics about "is my neighbour shading
# the same surface", which is a question about the G-buffer, not about which
# technique owns the reservoir, so there is no reason for them to differ.
comptime SMS_SPATIAL_NEIGHBORS: Int = 3
comptime SMS_SPATIAL_SLOTS: Int = 4
comptime SMS_SPATIAL_RADIUS_PX: Float32 = Float32(16.0)
comptime SMS_SPATIAL_NORMAL_DOT_MIN: Float32 = Float32(0.906)   # ~25 degrees
comptime SMS_SPATIAL_DEPTH_REL_MAX: Float32 = Float32(0.1)
# Same defensive clamp restir_gi.mojo carries, for the same reason: a
# finalized weight is fed back into every future combine that reuses it, so
# one anomalous value compounds instead of staying a one-frame outlier.
comptime SMS_MAX_FINALIZED_WEIGHT: Float32 = Float32(1.0e4)

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
    separate cos_l factor.

    `bsdf_product`/`dx1_dxlight` ARE stored (unlike DIReservoir, which
    recomputes everything di_target_pdf needs on demand): sms_target_pdf
    is a pure function of already-solved-chain quantities, not something
    that re-derives them from geometry alone, so a temporally-reused
    winner needs these carried forward to be re-evaluated without
    re-running the Newton solve. Under gonzales's identity-reprojection-
    only reuse (fact #2, docs/A2_restir_migration_plan.md -- no motion
    support, so x0/light_point are literally unchanged frame to frame at
    a given pixel), the stored values remain exactly valid: both
    quantities are properties of the chain's own vertices relative to
    ITS x0/light_point endpoints, which haven't moved."""
    var n_vertices:   Int32
    var verts:        InlineArray[SMSVertex, MAX_SMS_VERTICES]
    var light_point:  Vec3f
    var ldp_du:       Vec3f
    var ldp_dv:       Vec3f
    var le:           RGB
    var bsdf_product: Float32
    var dx1_dxlight:  Float32
    var state:        ReservoirState

@always_inline
def sms_reservoir_init() -> SMSReservoir:
    var z3 = Vec3f(Float32(0.0))
    return SMSReservoir(
        n_vertices=Int32(0),
        verts=InlineArray[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init()),
        light_point=z3, ldp_du=z3, ldp_dv=z3,
        le=RGB(Float32(0.0)),
        bsdf_product=Float32(0.0), dx1_dxlight=Float32(0.0),
        state=reservoir_state_init(),
    )

@always_inline
def sms_target_pdf(
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    first_vertex: Vec3f, first_normal: Vec3f,
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
    dst_x0: Vec3f, dst_light_point: Vec3f,
    dst_ldp_du: Vec3f, dst_ldp_dv: Vec3f,
    src_x0: Vec3f, src_light_point: Vec3f,
    src_ldp_du: Vec3f, src_ldp_dv: Vec3f,
    src_verts: InlineArray[SMSVertex, MAX_SMS_VERTICES], n: Int,
) -> Tuple[Bool, InlineArray[Vec3f, MAX_SMS_VERTICES], Float32, Float32]:
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
    var zero_positions = InlineArray[Vec3f, MAX_SMS_VERTICES](fill=Vec3f(Float32(0.0)))
    var src_first = src_verts[0].pos
    var orig_dir_v = src_first - src_x0
    var orig_dir_len = sqrt(dot(orig_dir_v, orig_dir_v))
    if orig_dir_len < Float32(1e-8):
        return (False, zero_positions.copy(), Float32(0.0), Float32(0.0))
    var original_dir = orig_dir_v * (Float32(1.0) / orig_dir_len)

    # ── Forward shift: walk seeded from src's chain, into dst's domain ──
    var _fwd = sms_walk(
        dst_x0, dst_light_point, src_verts, n, dst_ldp_du, dst_ldp_dv)
    var fwd_ok = _fwd[0]; var fwd_pos = _fwd[1].copy(); var fwd_bsdf = _fwd[2]; var fwd_jac = _fwd[3]
    if not fwd_ok:
        return (False, zero_positions.copy(), Float32(0.0), Float32(0.0))

    # ── Backward verification: walk the shifted chain back into src's domain ──
    var back_seed = src_verts.copy()
    for i in range(n):
        back_seed[i].pos = fwd_pos[i]
    var _back = sms_walk(
        src_x0, src_light_point, back_seed, n, src_ldp_du, src_ldp_dv)
    var back_ok = _back[0]; var back_pos = _back[1].copy()
    if not back_ok:
        return (False, zero_positions.copy(), Float32(0.0), Float32(0.0))

    var base_dir_v = back_pos[0] - src_x0
    var base_dir_len = sqrt(dot(base_dir_v, base_dir_v))
    if base_dir_len < Float32(1e-8):
        return (False, zero_positions.copy(), Float32(0.0), Float32(0.0))
    var base_dir = base_dir_v * (Float32(1.0) / base_dir_len)

    if abs(dot(original_dir, base_dir) - Float32(1.0)) > SMS_UNIQUENESS_THRESHOLD:
        return (False, zero_positions.copy(), Float32(0.0), Float32(0.0))

    return (True, fwd_pos^, fwd_bsdf, fwd_jac)

@fieldwise_init
struct SMSReservoirIO(TrivialRegisterPassable):
    """Per-pixel persistent reservoir buffers for temporal-only reuse
    (Phase 6's first driver -- no spatial reuse yet, so unlike
    restir_di.mojo's ReservoirIO/restir_gi.mojo's GIReservoirIO this
    carries no G-buffer pointers; add them alongside spatial reuse when
    that lands, not before). `read` is the previous frame's fully-
    resolved reservoirs; `write` is this frame's target -- pipeline.mojo's
    render_interactive swaps which physical buffer is which each frame
    rather than copying, same convention as the DI/GI buffers."""
    var read:  UnsafePointer[SMSReservoir, MutExternalOrigin]
    var write: UnsafePointer[SMSReservoir, MutExternalOrigin]
    # Phase 0.3's shared G-buffer, added with spatial reuse exactly as this
    # struct's docstring said it would be. `gbuf_world_pos` is not just for
    # rejection here the way it is for DI/GI: the manifold shift needs the
    # NEIGHBOUR'S OWN shading point as `src_x0`, because a specular chain is
    # only admissible relative to the point it was solved for.
    var gbuf_normal:      UnsafePointer[Float32, MutExternalOrigin]
    var gbuf_depth:       UnsafePointer[Float32, MutExternalOrigin]
    var gbuf_material_id: UnsafePointer[Int32, MutExternalOrigin]
    var gbuf_world_pos:   UnsafePointer[Float32, MutExternalOrigin]
    var frame_w: Int32
    var frame_h: Int32

@always_inline
def sms_reservoir_io_null() -> SMSReservoirIO:
    return SMSReservoirIO(
        read=UnsafePointer[SMSReservoir, MutExternalOrigin].unsafe_dangling(),
        write=UnsafePointer[SMSReservoir, MutExternalOrigin].unsafe_dangling(),
        gbuf_normal=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_depth=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_material_id=UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
        gbuf_world_pos=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        frame_w=Int32(0), frame_h=Int32(0),
    )

@always_inline
def sms_spatial_combine(
    mut res: SMSReservoir,
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    dst_light_point: Vec3f, dst_ldp_du: Vec3f, dst_ldp_dv: Vec3f,
    mut pcg: PCG32,
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
    pixel_idx: Int = -1,
):
    """Phase 6's spatial half: fold SMS_SPATIAL_NEIGHBORS neighbouring
    pixels' previous-frame reservoirs into `res`, each carried across by
    the manifold shift (`sms_shift`) rather than reused verbatim.

    That shift is the whole difference from restir_gi.mojo's spatial pass,
    and it is not optional. A GI reservoir stores a reconnection POINT,
    which means the same thing at any shading point that can see it; an SMS
    reservoir stores a specular CHAIN, which is a solution to a constraint
    system posed relative to one specific x0. Reusing a neighbour's chain
    unshifted would be reading a solution of a different equation. So each
    tap re-walks the neighbour's chain into this pixel's domain and is
    accepted only if the walk converges AND the backward walk lands where
    it started (bijectivity) -- both failures mean p̂ = 0 here, per GRIS's
    shift-invalid convention, never a silent fallback to the unshifted
    chain.

    The neighbour's own shading point comes from the G-buffer's world
    position, which is why this driver needs G-buffer pointers that the
    temporal-only one did not.

    Z normalization differs from GI's too, and more honestly. GI asks
    "is p̂ > 0 at the neighbour's point?" as a proxy for "could that
    neighbour have produced this sample?". For a shift-based reservoir the
    exact question is available: shift the WINNER back into the
    neighbour's domain and see whether it survives. That is one extra
    sms_shift per folded-in tap, and it is the same test the forward
    direction already performs, so it costs no new machinery.

    Does nothing without a real G-buffer and frame size -- callers with a
    null `sms_io` (every non-interactive path today) are unaffected."""
    if pixel_idx < 0 or not _is_real_ptr(sms_io.read):
        return
    if not _is_real_ptr(sms_io.gbuf_normal) or sms_io.frame_w <= Int32(0) or sms_io.frame_h <= Int32(0):
        return
    if not _is_real_ptr(sms_io.gbuf_world_pos):
        return

    var m_same_domain = res.state.m
    var nb_px_seen = InlineArray[Int32, SMS_SPATIAL_SLOTS](fill=Int32(-1))
    var nb_m_seen = InlineArray[Float32, SMS_SPATIAL_SLOTS](fill=Float32(0.0))
    var nb_seen = 0

    var self_px = Int32(pixel_idx) % sms_io.frame_w
    var self_py = Int32(pixel_idx) // sms_io.frame_w
    var self_depth = sms_io.gbuf_depth[pixel_idx]
    var self_mat = sms_io.gbuf_material_id[pixel_idx]

    for _ in range(SMS_SPATIAL_NEIGHBORS):
        var ang = pcg.next_float() * Float32(6.283185307)
        var rad = sqrt(pcg.next_float()) * SMS_SPATIAL_RADIUS_PX
        var nx = self_px + Int32(cos(ang) * rad)
        var ny = self_py + Int32(sin(ang) * rad)
        if nx < Int32(0) or nx >= sms_io.frame_w or ny < Int32(0) or ny >= sms_io.frame_h:
            continue
        var n_idx = Int(ny * sms_io.frame_w + nx)
        if n_idx == pixel_idx:
            continue
        var n_off = n_idx * 3
        var n_normal = Vec3f(
            sms_io.gbuf_normal[n_off], sms_io.gbuf_normal[n_off + 1], sms_io.gbuf_normal[n_off + 2])
        if dot(n_normal, normal) < SMS_SPATIAL_NORMAL_DOT_MIN:
            continue
        var n_depth = sms_io.gbuf_depth[n_idx]
        if self_depth <= Float32(0.0) or abs(n_depth - self_depth) > SMS_SPATIAL_DEPTH_REL_MAX * self_depth:
            continue
        if sms_io.gbuf_material_id[n_idx] != self_mat:
            continue

        var nb = sms_io.read[n_idx].copy()
        var n_v = Int(nb.n_vertices)
        if n_v <= 0:
            continue
        var nb_x0 = Vec3f(
            sms_io.gbuf_world_pos[n_off], sms_io.gbuf_world_pos[n_off + 1], sms_io.gbuf_world_pos[n_off + 2])

        var sh = sms_shift(
            hit_point, dst_light_point, dst_ldp_du, dst_ldp_dv,
            nb_x0, nb.light_point, nb.ldp_du, nb.ldp_dv,
            nb.verts, n_v)
        if not sh[0]:
            continue

        # The shift returns POSITIONS; the chain's frames still describe the
        # neighbour's solution until they are re-derived here (the same
        # staleness sms_refresh_solved_frames exists for).
        var shifted = nb.verts.copy()
        for i in range(n_v):
            shifted[i].pos = sh[1][i]
        sms_refresh_solved_frames(shifted, n_v)

        var p_hat_nb_here = sms_target_pdf(
            hit_point, normal, alb, shifted[0].pos, shifted[0].normal,
            nb.le, sh[2], sh[3])

        if nb_seen < SMS_SPATIAL_SLOTS:
            nb_px_seen[nb_seen] = Int32(n_idx)
            nb_m_seen[nb_seen] = nb.state.m
            nb_seen += 1

        if reservoir_combine(res.state, nb.state, p_hat_nb_here, pcg.next_float()):
            res.n_vertices = Int32(n_v)
            res.verts = shifted.copy()
            res.light_point = dst_light_point
            res.ldp_du = dst_ldp_du
            res.ldp_dv = dst_ldp_dv
            res.le = nb.le
            res.bsdf_product = sh[2]
            res.dx1_dxlight = sh[3]

    # ── Z normalization: which taps could have produced the winner? ──
    var z_norm = Float32(-1.0)
    if nb_seen > 0 and res.n_vertices > Int32(0):
        var z = m_same_domain
        var w_v = Int(res.n_vertices)
        for i in range(nb_seen):
            var np_off = Int(nb_px_seen[i]) * 3
            var n_hit = Vec3f(
                sms_io.gbuf_world_pos[np_off], sms_io.gbuf_world_pos[np_off + 1],
                sms_io.gbuf_world_pos[np_off + 2])
            var nb_i = sms_io.read[Int(nb_px_seen[i])].copy()
            # Shift the winner BACK into tap i's domain -- the exact
            # "could this domain have produced x?" test.
            var back = sms_shift(
                n_hit, nb_i.light_point, nb_i.ldp_du, nb_i.ldp_dv,
                hit_point, res.light_point, res.ldp_du, res.ldp_dv,
                res.verts, w_v)
            if back[0]:
                z += nb_m_seen[i]
        z_norm = z

    var p_hat_final = Float32(0.0)
    if res.n_vertices > Int32(0):
        p_hat_final = sms_target_pdf(
            hit_point, normal, alb, res.verts[0].pos, res.verts[0].normal,
            res.le, res.bsdf_product, res.dx1_dxlight)
    reservoir_finalize(res.state, p_hat_final, z_norm)
    if res.state.w > SMS_MAX_FINALIZED_WEIGHT:
        res.state.w = SMS_MAX_FINALIZED_WEIGHT
