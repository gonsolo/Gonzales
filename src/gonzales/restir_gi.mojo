# ReSTIR GI (Ouyang et al. 2021's path-reuse reservoir applied to indirect
# lighting), Phase 4 of docs/A2_restir_migration_plan.md. Mirrors
# restir_di.mojo's shape closely: payload type + pure target-function math +
# (unlike restir_di.mojo) the temporal/spatial COMBINE step too, since that
# step needs no ShadeContext -- see gi_temporal_spatial_combine's own
# docstring for why it can live here while di_temporal_step cannot.
#
# Still not wired into any render path. What's deliberately NOT here:
# candidate GENERATION (tracing a fresh continuation path from x1 to find a
# reconnection vertex and its Lo -- needs real ray tracing/BSDF sampling via
# PathState_C/ShadeContext, i.e. shading.mojo) and RESOLUTION (a shadow ray
# between x1 and the winning x2, MIS against BSDF sampling, injecting the
# result into the path's total -- also ctx-dependent, mirrors di_resolve).
# Both are separate, later, larger commits -- see project_restir_migration
# memory for why candidate generation specifically is the one genuinely
# hard remaining piece (it needs a second, path-local throughput tracked
# alongside the real one, reset at the reconnection vertex, to measure Lo
# uncontaminated by x1's own BSDF/cosine/pdf factors).
#
# Design note on why Lo is reused unmodified across the shift (no BSDF
# re-evaluation at the reconnection vertex): ReSTIR GI treats the
# reconnection vertex x2's own outgoing radiance Lo (leaving x2 back toward
# whichever x1 reconnects to it) as approximately view-independent -- valid
# exactly when x2's BSDF is non-delta (a delta BSDF's response changes
# completely with viewing direction, which is precisely why Phase 4.3's
# delta-BSDF rejection exists: never choose a delta vertex as a reconnection
# point). Only the FIRST segment (x1 -> x2) is genuinely re-evaluated under
# the shift -- x1's own BSDF, cosine, and the inverse-square/Jacobian term --
# exactly analogous to how Phase 2's light reconnection re-evaluates x1's
# BSDF toward a fixed light point while reusing the light's own Le verbatim.

from std.math import sqrt, cos, sin, abs
from .geometry import RGB, dot, INV_PI, _is_real_ptr, Vec3f
from .reservoir import ReservoirState, reservoir_state_init, reservoir_combine, reservoir_finalize, reservoir_cap_confidence
from .rng import PCG32

@fieldwise_init
struct GIReservoir(TrivialRegisterPassable):
    """ReSTIR GI's reservoir payload: a one-bounce-reconnection path suffix.

    `recon_point`/`recon_normal` are the reconnection vertex x2's world-space
    position and shading normal -- kept fixed in world space across temporal
    and spatial reuse, exactly like DIReservoir's `sample_point` is a light
    point fixed in world space. `lo` is the accumulated outgoing radiance
    leaving x2 back toward the path's ORIGINAL x1 (all further bounces'
    contribution already integrated) -- reused as-is under the shift (see
    module docstring). `recon_is_delta` records whether x2's own BSDF is a
    delta distribution; a reconnection vertex chosen there can never be
    validly reused from a different x1 (Phase 4.3), so any candidate/neighbor
    carrying this flag set must be rejected before ever computing a target
    pdf against it -- callers must check this themselves, since a pure
    geometry function can't know a material's BSDF kind.
    `valid` mirrors DIReservoir's `light_idx < 0` sentinel (0 = no winner
    yet; every other field is meaningless until a candidate is streamed)."""
    var recon_point:   Vec3f
    var recon_normal:  Vec3f
    var lo:            RGB
    var recon_is_delta: Int8
    var valid:          Int8
    var _pad0:          Int8
    var _pad1:          Int8
    var state:          ReservoirState

@always_inline
def gi_reservoir_init() -> GIReservoir:
    return GIReservoir(
        recon_point=Vec3f(Float32(0)),
        recon_normal=Vec3f(Float32(0)),
        lo=RGB(Float32(0)),
        recon_is_delta=Int8(0),
        valid=Int8(0),
        _pad0=Int8(0),
        _pad1=Int8(0),
        state=reservoir_state_init(),
    )

@always_inline
def gi_target_pdf(
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    recon_point: Vec3f, recon_normal: Vec3f, lo: RGB,
) -> Float32:
    """RIS target function p̂(candidate) for a one-bounce-reconnection GI
    sample: luminance of the UNSHADOWED Lambertian-at-x1 contribution
    f(x1) x cos(x1) x Lo(x2), where f(x1) = alb/pi (matching Phase 2's own
    diffuse-only-at-x1 scope). Deliberately NOT cos_x1*cos_x2/dist_sq --
    unlike di_target_pdf's `le` (a light's own EMITTED radiance, an
    area-measure quantity that genuinely needs the full geometric term to
    become incident radiance at x1), `lo` is already OUTGOING RADIANCE
    from x2 toward x1 under the reconnection-shift's view-independence
    assumption (see module header) -- radiance is invariant along a ray in
    vacuum, so transporting it to x1 needs only x1's own cos(x1)*BSDF, the
    same shape gi_resolve (shading.mojo) actually computes. Including
    cos_x2/dist_sq here was a real, shipped bug: for a SINGLE unstreamed
    candidate (state.m=1) the mismatch is invisible (state.w=1 regardless
    of p_hat's exact formula, by construction of the RIS finalize math),
    but once real reuse combines candidates with genuinely different
    cos_x2/dist_sq values, state.w = w_sum/(m*p_hat) over/under-compensates
    for a falloff gi_resolve's own contribution never had, and explodes
    whenever a reused candidate's cos_x2 happens to be small as seen from
    THIS pixel's x1 (even though gi_resolve's actual contribution doesn't
    depend on that cos_x2 at all) -- measured on cornell-box as a single-
    pixel resolve delta spiking past 16000 (vs. a normal ~0.001-0.6 range).
    cos_x2 is STILL checked for backface REJECTION (x1 can't see x2's back
    face) -- it's just not multiplied into the weight's magnitude.

    Same overall shape as di_target_pdf otherwise, with the reconnection
    vertex's position/normal/Lo standing in for a light sample's point/
    normal/Le -- visibility between x1 and x2 deliberately excluded here,
    to be resolved once for the reservoir's eventual winner, same as
    Phase 2.1."""
    var to_recon = recon_point - hit_point
    var dist_sq = dot(to_recon, to_recon)
    if dist_sq <= Float32(1e-8):
        return Float32(0.0)
    var dist = sqrt(dist_sq)
    var wi = to_recon * (Float32(1.0) / dist)
    var cos_x1 = dot(normal, wi)
    var cos_x2 = -dot(recon_normal, wi)
    if cos_x1 <= Float32(0.0) or cos_x2 <= Float32(0.0):
        return Float32(0.0)
    var contrib = alb * lo * (INV_PI * cos_x1)
    return contrib.r * Float32(0.2126) + contrib.g * Float32(0.7152) + contrib.b * Float32(0.0722)

# ── ReSTIR GI reuse (Phase 4.2, partial: combine only) ──────────────────────

@fieldwise_init
struct GIReservoirIO(TrivialRegisterPassable):
    """GI's counterpart to restir_di.mojo's ReservoirIO -- read/write
    persistent reservoir buffers plus the shared Phase 0.3 G-buffer pointers
    for spatial-neighbor rejection. Duplicated rather than shared with
    ReservoirIO (which is typed for DIReservoir specifically) -- this
    codebase avoids generics throughout (see reservoir.mojo's own docstring)
    in favor of small, independently-readable duplicated structs. All
    pointers default to `.unsafe_dangling()` and frame_w/frame_h to 0;
    gi_temporal_spatial_combine checks `_is_real_ptr`/`> 0` before ever
    touching them, matching ReservoirIO's own null-safety contract."""
    var read:  UnsafePointer[GIReservoir, MutExternalOrigin]
    var write: UnsafePointer[GIReservoir, MutExternalOrigin]
    var gbuf_normal:      UnsafePointer[Float32, MutExternalOrigin]
    var gbuf_depth:       UnsafePointer[Float32, MutExternalOrigin]
    var gbuf_material_id: UnsafePointer[Int32, MutExternalOrigin]
    var gbuf_world_pos:   UnsafePointer[Float32, MutExternalOrigin]
    var frame_w: Int32
    var frame_h: Int32

@always_inline
def gi_reservoir_io_null() -> GIReservoirIO:
    return GIReservoirIO(
        read=UnsafePointer[GIReservoir, MutExternalOrigin].unsafe_dangling(),
        write=UnsafePointer[GIReservoir, MutExternalOrigin].unsafe_dangling(),
        gbuf_normal=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_depth=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        gbuf_material_id=UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
        gbuf_world_pos=UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        frame_w=Int32(0), frame_h=Int32(0),
    )

# Placeholder tuning constants, copied from DI's MEASURED values
# (shading.mojo's DI_TEMPORAL_M_CAP/DI_SPATIAL_* -- see the tuning notes
# next to them) as a starting guess. UNVALIDATED for GI specifically: no
# candidate generation exists yet to render an A/B comparison against, and
# GI's reconnection vertices sit one bounce further from the camera than
# DI's light samples, so the right spatial radius/rejection thresholds may
# well differ. Re-measure empirically once Phase 4.1's generation half
# lands, the same way DI's own constants were tuned -- do not trust these
# as final.
comptime GI_TEMPORAL_M_CAP: Float32 = Float32(64.0)
comptime GI_SPATIAL_NEIGHBORS: Int = 4
comptime GI_SPATIAL_SLOTS: Int = GI_SPATIAL_NEIGHBORS + 1
comptime GI_SPATIAL_RADIUS_PX: Float32 = Float32(20.0)
comptime GI_SPATIAL_NORMAL_DOT_MIN: Float32 = Float32(0.9)
comptime GI_SPATIAL_DEPTH_REL_MAX: Float32 = Float32(0.1)
# Defensive cap on a finalized reservoir's own state.w -- see
# gi_temporal_spatial_combine's use of this constant for the full
# reasoning (reservoir_combine's own weight formula feeds a source
# reservoir's state.w back into future combines, so one anomalous value
# compounds into unbounded growth rather than staying a one-frame
# outlier; measured on cornell-box reaching >1,000,000 within ~20 frames
# without this clamp, versus ~1-10 for every healthy frame observed).
comptime GI_MAX_FINALIZED_WEIGHT: Float32 = Float32(10.0)

def gi_temporal_spatial_combine(
    mut res: GIReservoir, hit_point: Vec3f, normal: Vec3f, alb: RGB,
    mut pcg: PCG32,
    gi_io: GIReservoirIO = gi_reservoir_io_null(),
    pixel_idx: Int = -1,
):
    """Combine `res` (the caller's own freshly-generated, already-streamed
    M=1 candidate for this pixel/frame) with the previous frame's reservoir
    at the same pixel (temporal, identity reprojection -- fact #2, same as
    di_temporal_step) and GI_SPATIAL_NEIGHBORS random neighbors' previous-
    frame reservoirs (spatial), each gated by the same G-buffer rejection
    test shading.mojo's di_temporal_step uses (normal dot / depth delta /
    material match) PLUS a delta-BSDF check unique to GI (Phase 4.3): a
    reconnection vertex flagged `recon_is_delta` is skipped entirely, never
    folded into `res.state.m`, exactly like a G-buffer-incompatible neighbor
    is -- its Lo is not a valid quantity to reuse from a different x1 at
    all, not merely a low-quality one. Finalizes `res.state.w` via
    reservoir_finalize (Bitterli et al. 2020 Algorithm 6's Z-normalization
    when spatial neighbors were folded in, plain M-normalization otherwise
    -- same as di_temporal_step) and, when `gi_io` is real, caps confidence
    and stores the result for the next frame. Falls back to a plain
    single-frame finalize (no persistence, no spatial reuse) when `gi_io`
    isn't real or `pixel_idx < 0` -- same null-safety contract as
    di_temporal_step/ReservoirIO.

    Deliberately does NOT trace a shadow ray or touch any path's throughput/
    total -- that's resolution, a separate ctx-dependent step (see module
    header). Callers that only want to observe/test the combine math (as
    this file's own unit tests do) can call this directly on synthetic
    reservoirs with no rendering involved."""
    var has_temporal = pixel_idx >= 0 and _is_real_ptr(gi_io.read)
    var nb_px_seen = InlineArray[Int32, GI_SPATIAL_SLOTS](fill=Int32(-1))
    var nb_m_seen = InlineArray[Float32, GI_SPATIAL_SLOTS](fill=Float32(0))
    var nb_seen = 0
    var m_same_domain = Float32(0.0)
    if has_temporal:
        var prev = gi_io.read[pixel_idx]
        if prev.valid != Int8(0) and prev.recon_is_delta == Int8(0):
            var p_hat_prev_here = gi_target_pdf(hit_point, normal, alb, prev.recon_point, prev.recon_normal, prev.lo)
            var accept = reservoir_combine(res.state, prev.state, p_hat_prev_here, pcg.next_float())
            if accept:
                res.recon_point = prev.recon_point
                res.recon_normal = prev.recon_normal
                res.lo = prev.lo
                res.recon_is_delta = prev.recon_is_delta
                res.valid = Int8(1)

        m_same_domain = res.state.m

        if _is_real_ptr(gi_io.gbuf_normal) and gi_io.frame_w > Int32(0) and gi_io.frame_h > Int32(0):
            var self_px = Int32(pixel_idx) % gi_io.frame_w
            var self_py = Int32(pixel_idx) // gi_io.frame_w
            var self_depth = gi_io.gbuf_depth[pixel_idx]
            var self_mat = gi_io.gbuf_material_id[pixel_idx]
            for _ in range(GI_SPATIAL_NEIGHBORS):
                var ang = pcg.next_float() * Float32(6.283185307)
                var rad = sqrt(pcg.next_float()) * GI_SPATIAL_RADIUS_PX
                var nx = self_px + Int32(cos(ang) * rad)
                var ny = self_py + Int32(sin(ang) * rad)
                if nx < Int32(0) or nx >= gi_io.frame_w or ny < Int32(0) or ny >= gi_io.frame_h:
                    continue
                var n_idx = Int(ny * gi_io.frame_w + nx)
                if n_idx == pixel_idx:
                    continue
                var n_off = n_idx * 3
                var n_normal = Vec3f(
                    gi_io.gbuf_normal[n_off], gi_io.gbuf_normal[n_off + 1], gi_io.gbuf_normal[n_off + 2])
                if dot(n_normal, normal) < GI_SPATIAL_NORMAL_DOT_MIN:
                    continue
                var n_depth = gi_io.gbuf_depth[n_idx]
                if self_depth <= Float32(0.0) or abs(n_depth - self_depth) > GI_SPATIAL_DEPTH_REL_MAX * self_depth:
                    continue
                if gi_io.gbuf_material_id[n_idx] != self_mat:
                    continue
                var nb = gi_io.read[n_idx]
                if nb.valid == Int8(0) or nb.recon_is_delta != Int8(0):
                    continue
                var p_hat_nb_here = gi_target_pdf(hit_point, normal, alb, nb.recon_point, nb.recon_normal, nb.lo)
                if nb_seen < GI_SPATIAL_SLOTS:
                    nb_px_seen[nb_seen] = Int32(n_idx)
                    nb_m_seen[nb_seen] = nb.state.m
                    nb_seen += 1
                var accept_nb = reservoir_combine(res.state, nb.state, p_hat_nb_here, pcg.next_float())
                if accept_nb:
                    res.recon_point = nb.recon_point
                    res.recon_normal = nb.recon_normal
                    res.lo = nb.lo
                    res.recon_is_delta = nb.recon_is_delta
                    res.valid = Int8(1)

    # Z normalization (Bitterli et al. 2020, Algorithm 6) -- same rationale
    # as di_temporal_step's own Z pass: dividing by the full accumulated m
    # over-counts neighbor domains that could never have produced the
    # winning sample. Re-evaluates the target function at each folded-in
    # neighbor's OWN shading point (from the G-buffer), not this pixel's.
    var z_norm = Float32(-1.0)
    if nb_seen > 0 and _is_real_ptr(gi_io.gbuf_world_pos):
        var z = m_same_domain
        for i in range(nb_seen):
            var np_off = Int(nb_px_seen[i]) * 3
            var n_hit = Vec3f(
                gi_io.gbuf_world_pos[np_off], gi_io.gbuf_world_pos[np_off + 1], gi_io.gbuf_world_pos[np_off + 2])
            var n_nrm = Vec3f(
                gi_io.gbuf_normal[np_off], gi_io.gbuf_normal[np_off + 1], gi_io.gbuf_normal[np_off + 2])
            if gi_target_pdf(n_hit, n_nrm, alb, res.recon_point, res.recon_normal, res.lo) > Float32(0.0):
                z += nb_m_seen[i]
        z_norm = z

    var p_hat_final = Float32(0.0)
    if res.valid != Int8(0):
        p_hat_final = gi_target_pdf(hit_point, normal, alb, res.recon_point, res.recon_normal, res.lo)
    reservoir_finalize(res.state, p_hat_final, z_norm)

    # Defensive weight clamp -- NOT a complete fix, an honest safety valve.
    # reservoir_combine's own weight formula (reservoir.mojo) multiplies in
    # the SOURCE's own already-finalized state.w; without this clamp, one
    # anomalous value (measured cause: a winning candidate evaluated at
    # near-degenerate geometry relative to a DIFFERENT pixel's own x1,
    # e.g. near a scene corner/edge, not fully caught by this design's
    # G-buffer rejection heuristics) gets stored and FED BACK into every
    # future combine that reuses it, compounding into unbounded growth
    # instead of staying a one-frame outlier. DI does not need this same
    # clamp -- its target function (`le`, an exact scene property) doesn't
    # have GI's near-degenerate-geometry sensitivity in the first place.
    if res.state.w > GI_MAX_FINALIZED_WEIGHT:
        res.state.w = GI_MAX_FINALIZED_WEIGHT

    if has_temporal:
        # state.m still holds the TRUE accumulated confidence here (z_norm
        # renormalized only W, never m) -- cap AFTER finalize, same ordering
        # di_temporal_step uses and for the same reason (capping first would
        # shrink this frame's own W incorrectly).
        reservoir_cap_confidence(res.state, GI_TEMPORAL_M_CAP)
        gi_io.write[pixel_idx] = res
