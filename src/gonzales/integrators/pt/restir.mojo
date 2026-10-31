"""ReSTIR DI and GI on the path tracer."""
from std.collections import Array
from std.math import sqrt, abs
from gonzales.core.geometry import RGB, Point3f, Point2i, restir_jitter_pixel, Vec3f, dot, _is_real_ptr
from gonzales.transport.render_state import PathState
from gonzales.core.geometry import Ray
from gonzales.scene.appearance.lights import light_sampler_sample, light_sampler_pdf
from gonzales.transport.bxdf import bxdf_eval_diffuse, bxdf_pdf_diffuse
from gonzales.core.rng import PCG32
from gonzales.scene.accel.bvh import any_hit_dispatch
from gonzales.core.sampling import power_heuristic
from gonzales.core.spectrum import SpectralSample
from gonzales.core.reservoir import reservoir_update, reservoir_finalize, reservoir_combine, reservoir_cap_confidence
from gonzales.transport.restir_di import DIReservoir, di_reservoir_init, di_target_pdf, ReservoirIO, reservoir_io_null
from gonzales.transport.restir_gi import GIReservoir, gi_reservoir_init
from gonzales.transport.shade_context import ShadeContext
from gonzales.integrators.pt.common import _to_spec_diffuse_lit, _emitter_invisible, _shadow_contribute
from gonzales.integrators.pt.nee import _sample_light_point_and_normal, _mnee_area_light_contribute


# ── ReSTIR DI (Phase 2, restir_di.mojo's DIReservoir/di_target_pdf) ─────────
# Plain RIS: M candidates from the existing light sampler, weighted p̂/q,
# streamed into a reservoir (2.1+2.2); one shadow ray resolves the winner
# (2.6), MIS-weighted against BSDF sampling exactly like ordinary NEE (2.7).
# No temporal/spatial reuse yet (2.3/2.5 need persistent per-pixel GPU state
# across real interactive-viewer frames -- see restir_di.mojo's header and
# project_restir_migration.md memory for why that's scoped out for now).
# CPU-only (ctx.use_restir is only ever True from the CPU dispatch path,
# see ShadeContext's own docstring) -- traces the winner's shadow ray
# INLINE (_shadow_contribute[False]) rather than through Phase 0.4's
# deferred machinery the plan's 2.6 names: that machinery is GPU-only
# (shadow_buf lives in GpuSceneHandle), nothing CPU-side to defer into yet.
comptime DI_RIS_CANDIDATES: Int = 16


def _di_sample_candidate(
    ctx: ShadeContext, hit_point: Vec3f, mut pcg: PCG32,
) -> Tuple[Bool, Int32, Vec3f, Vec3f, RGB, Float32,
           Vec3f, Vec3f]:
    """One RIS candidate draw: (valid, light_idx, sample_point,
    light_normal, Le, generation_pdf, ldp_du, ldp_dv). Mirrors
    _sample_area_light_nee's math but keeps the sampled point/normal the
    reservoir payload needs -- that function only returns the derived
    wi/dist, not the point itself. The trailing tangent pair is the light
    surface's own parameterization at the sampled point, kept (rather than
    discarded as `_, _`) so a resampled winner can still run MNEE in
    di_resolve -- see DIReservoir's ldp_du/ldp_dv docs."""
    var zero3 = Vec3f(Float32(0))
    var invalid = (False, Int32(-1), zero3, zero3, RGB(Float32(0)), Float32(0), zero3, zero3)
    if ctx.lights.area_light_count == 0:
        return invalid
    var u_light = pcg.next_float()
    var ls_result = light_sampler_sample(ctx.lights.light_sampler, u_light)
    var light_idx = ls_result[0]
    var light_sel_pdf = ls_result[1]
    var al = ctx.lights.area_lights[unsafe_offset=light_idx]
    var r1 = pcg.next_float()
    var r2 = pcg.next_float()
    var (light_point, light_normal, ldp_du_v, ldp_dv_v) = _sample_light_point_and_normal(ctx, al, r1, r2, pcg, hit_point)
    var to_light = light_point - hit_point
    var dist_sq = dot(to_light, to_light)
    if dist_sq <= Float32(1e-8) or al.total_area <= Float32(0.0):
        return invalid
    var dist = sqrt(dist_sq)
    var wi = to_light * (Float32(1.0) / dist)
    var cos_l = -dot(light_normal, wi)
    if cos_l <= Float32(0.0):
        return invalid
    var pdf = dist_sq * light_sel_pdf / (cos_l * al.total_area)
    if pdf <= Float32(0.0):
        return invalid
    return (True, Int32(light_idx), light_point, light_normal, al.emission, pdf, ldp_du_v, ldp_dv_v)


def di_generate_reservoir(
    ctx: ShadeContext, hit_point: Vec3f, normal: Vec3f,
    alb: RGB, mut pcg: PCG32,
) -> DIReservoir:
    """Phase 2.1+2.2: stream DI_RIS_CANDIDATES area-light samples through
    weighted reservoir sampling, weight = p̂/q per candidate (reservoir.mojo).
    Visibility deliberately excluded (2.1) -- resolved once for the winner
    by di_resolve."""
    var res = di_reservoir_init()
    for _ in range(DI_RIS_CANDIDATES):
        var (valid, light_idx, sample_point, light_normal, le, gen_pdf, ldp_du_v, ldp_dv_v) = _di_sample_candidate(ctx, hit_point, pcg)
        if not valid:
            # A rejected candidate is still a REAL draw from q -- it just
            # landed where the target function is zero (light facing away,
            # cos_l <= 0, degenerate distance). RIS's W = w_sum / (M * p_hat)
            # needs M = the number of candidates DRAWN, not the number that
            # happened to be non-zero, so its zero weight must still be
            # streamed. Skipping this (an early `continue`) silently divides
            # by too small an M and inflates every W: on staircase2 roughly
            # 40% of candidates are backfacing, so m came out ~9 instead of
            # 16 and W was ~1.8x too large. reservoir_update with weight 0
            # returns before touching w_sum (and before reading `u`, hence
            # the dummy -- deliberately NOT a pcg draw, so the RNG stream is
            # identical to before this fix), but still does the m += 1 that
            # is the entire point here.
            _ = reservoir_update(res.state, Float32(0.0), Float32(0.0))
            continue
        var p_hat = di_target_pdf(hit_point, normal, alb, sample_point, light_normal, le)
        var weight = p_hat / gen_pdf
        var accept = reservoir_update(res.state, weight, pcg.next_float())
        if accept:
            res.light_idx = light_idx
            res.sample_point = sample_point
            res.light_normal = light_normal
            res.ldp_du = ldp_du_v
            res.ldp_dv = ldp_dv_v
            res.le = le
    return res


def di_resolve(
    path_ptr: Pointer[PathState, MutUntrackedOrigin], ctx: ShadeContext,
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    mut res: DIReservoir,
    z_norm: Float32 = Float32(-1.0),
):
    """Phase 2.6+2.7: finalize the reservoir's RIS weight, then trace ONE
    shadow ray for the winner and add its MIS-weighted contribution. The
    resampled winner is treated as a single light-technique sample for MIS
    purposes against BSDF sampling (same balance-heuristic pattern as
    ordinary NEE) -- standard RIS/ReSTIR practice; the RIS weight W already
    accounts for the M-candidate resampling itself.

    `z_norm` is forwarded verbatim to reservoir_finalize: negative (the
    default) means "divide by state.m", correct for single-domain
    combination; spatial reuse passes Bitterli 2020 Algorithm 6's Z
    instead. See di_temporal_step for how Z is accumulated."""
    if res.light_idx < Int32(0):
        return
    var p_hat = di_target_pdf(hit_point, normal, alb, res.sample_point, res.light_normal, res.le)
    reservoir_finalize(res.state, p_hat, z_norm)
    # Defensive weight clamp -- see DI_MAX_FINALIZED_WEIGHT's own comment
    # for the full mechanism (a real, live-traced bug: unbounded growth via
    # reservoir_combine's own feedback of a source's finalized state.w).
    # Must happen BEFORE the res.state.w <= 0 check below (clamping a
    # positive value can never make it <= 0) and BEFORE this frame's
    # result is stored for the next frame/neighbor to reuse.
    if res.state.w > DI_MAX_FINALIZED_WEIGHT:
        res.state.w = DI_MAX_FINALIZED_WEIGHT
    if res.state.w <= Float32(0.0):
        return
    var to_light = res.sample_point - hit_point
    var dist_sq = dot(to_light, to_light)
    if dist_sq <= Float32(1e-8):
        return
    var dist = sqrt(dist_sq)
    var wi = to_light * (Float32(1.0) / dist)
    var cos_s = dot(normal, wi)
    var cos_l = -dot(res.light_normal, wi)
    if cos_s <= Float32(0.0) or cos_l <= Float32(0.0):
        return
    var al = ctx.lights.area_lights[unsafe_offset=Int(res.light_idx)]
    if al.total_area <= Float32(0.0):
        return
    var light_sel_pdf = light_sampler_pdf(ctx.lights.light_sampler, res.light_idx)
    var pdf_light = dist_sq * light_sel_pdf / (cos_l * al.total_area)
    if pdf_light <= Float32(0.0):
        return
    var pdf_bsdf = bxdf_pdf_diffuse(cos_s)
    var mis_w = Float32(1.0) if _emitter_invisible(ctx, al) else power_heuristic(pdf_light, pdf_bsdf)
    # NOT cos_s*cos_l/dist_sq: pdf_light is a SOLID-ANGLE pdf (the
    # dist_sq/cos_l area-to-solid-angle Jacobian is already baked into its
    # own definition above). res.state.w (from reservoir_finalize) already
    # has a 1/p_hat(y) factor, and di_target_pdf's p_hat is built from the
    # FULL g=cos_s*cos_l/dist_sq -- multiplying by g again here double-counts
    # that Jacobian. Verified via the M=1 degenerate case (DI_RIS_CANDIDATES=1
    # forces exactly one candidate, always accepted): there, W collapses
    # algebraically to plain 1/pdf_light regardless of p_hat's shape (p_hat
    # cancels out of reservoir_finalize's w_sum/(M*p_hat) when w_sum IS
    # p_hat/gen_pdf), so contrib MUST reduce to _nee_area_lights's own
    # proven-correct cos_s/pdf_light formula -- which has no separate g
    # factor. Confirmed empirically too: adding g turned a ~3.5% mean
    # brightness bias into a ~28% one on staircase2 (400x400, vs a 128spp
    # plain-NEE reference).
    var contrib = path_ptr[].throughput * _to_spec_diffuse_lit(ctx, alb, res.le, path_ptr[].wavelengths) * (cos_s * res.state.w * mis_w)

    # MNEE, shared verbatim with _nee_area_lights. Without this the ReSTIR
    # path lost every bit of light reaching a surface THROUGH glass (the
    # straight shadow ray below is simply occluded by the dielectric), which
    # measured ~9% of staircase2's mean, spatially concentrated on the wall
    # and treads behind its glass railing.
    #
    # Measure conversion is the one thing that differs from the plain-NEE
    # caller: MNEE divides by an AREA-measure light pdf, but res.state.w
    # estimates 1/q in SOLID-ANGLE measure (_di_sample_candidate's gen_pdf
    # carries the dist^2/cos_l factor). Since q_sa = q_A * dist^2/cos_l,
    # 1/q_A = w * dist^2/cos_l, using the STRAIGHT-LINE dist/cos_l that
    # gen_pdf itself was built from (NOT the refracted path's geometry --
    # the manifold walk supplies its own G, and this factor is only undoing
    # the sampling measure, not describing the transport).
    var used_mnee = _mnee_area_light_contribute(
        path_ptr, ctx, normal, hit_point, alb, wi, dist,
        res.sample_point, res.ldp_du, res.ldp_dv, al,
        res.state.w * dist_sq / cos_l, Float32(1.0))
    if not used_mnee:
        _shadow_contribute[False](path_ptr, ctx, hit_point, wi, dist * Float32(0.9999), contrib)


# ── ReSTIR GI (Phase 4.1's remaining half: candidate generation) ────────────
# Scoped to x2 (the reconnection vertex) ALSO being diffuse -- mirrors how
# Phase 2 restricted x1 to diffuse first. This bounds the touch point to
# _shade_diffuse_nee alone: if bounce 1 lands on any other material, no GI
# candidate is generated for this path/sample at all (falls back to
# whatever that material's own shading does, completely unaffected). See
# project_restir_migration memory for why a fully general version (any
# material at x2) is real, separate, deferred work.
def _gi_generate_recon_candidate(
    ctx: ShadeContext, hit_point: Vec3f, normal: Vec3f, alb: RGB,
    mut pcg: PCG32,
) -> GIReservoir:
    """Generates x2's own contribution to Lo(x2): ONE area-light NEE sample
    (mirrors _di_sample_candidate's light pick, resolved as a plain single-
    sample estimate, NOT resampled via RIS -- ReSTIR GI's reconnection
    vertex gets ordinary NEE, not its own reservoir; the OUTER GI reservoir
    resamples over whole path suffixes, not over x2's own light picks).

    Explicit, documented scope limitation: Lo(x2) as computed here is
    area-light NEE ONLY -- no env/infinite/distant/point light contribution
    reaching x2 directly, and no further indirect bounces beyond x2. This
    is a real, honest simplification (matching Phase 2's own initial
    diffuse-and-area-lights-only scope), not a bug -- whoever generalizes
    this later should extend it the same incremental way Phase 2 itself
    was generalized, not treat this as a placeholder to silently rely on.

    No MIS weight against BSDF sampling is applied to the drawn light
    sample: since the complementary BSDF-sampling technique's own
    contribution is NOT separately included in this restricted definition
    of Lo, applying a MIS weight here would introduce a systematic
    negative bias rather than reduce variance -- correct only when treated
    as the sole technique for this quantity, which is exactly the case
    here.

    Returns an UNSTREAMED candidate (state left at reservoir_state_init())
    -- valid=0 when there are no area lights or the drawn sample is
    degenerate/backfacing at x2 (`normal`/`hit_point` here are x2's own).
    The caller must compute the real RIS weight using x1's OWN hit_point/
    normal/alb (not available here -- this runs at x2's shading call) via
    gi_target_pdf, then stream it with reservoir_update, before this
    becomes usable in gi_temporal_spatial_combine."""
    var (ok, _, sample_point, _, le, gen_pdf, _, _) = _di_sample_candidate(ctx, hit_point, pcg)
    if not ok:
        return gi_reservoir_init()
    var to_light = sample_point - hit_point
    var dist_sq = dot(to_light, to_light)
    var dist = sqrt(dist_sq)
    var wi = to_light * (Float32(1.0) / dist)
    var cos_s = dot(normal, wi)
    if cos_s <= Float32(0.0):
        return gi_reservoir_init()
    var lo = RGB(Float32(0.0))
    var shadow_ray = Ray(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(wi[0], wi[1], wi[2]))
    if not any_hit_dispatch(ctx.bvh4, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, shadow_ray, dist * Float32(0.9999),
                              ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                              ctx.lights.spheres, ctx.lights.sphere_count,
                              materials=ctx.materials):
        lo = bxdf_eval_diffuse(alb) * le * (cos_s / gen_pdf)
    var res = gi_reservoir_init()
    res.recon_point = hit_point
    res.recon_normal = normal
    res.lo = lo
    res.recon_is_delta = Int8(0)  # x2 is diffuse by construction (only called from _shade_diffuse_nee)
    res.valid = Int8(1)
    return res


def gi_resolve(
    path_ptr: Pointer[PathState, MutUntrackedOrigin], ctx: ShadeContext,
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    throughput: SpectralSample,
    res: GIReservoir,
):
    """Phase 4's resolution step, mirrors di_resolve: trace ONE shadow ray
    between x1 (hit_point/normal/alb -- x1's OWN data, passed in explicitly
    since this runs at bounce 1's shading call where x1 is long out of
    scope, see GIPendingX1) and the reservoir's finalized winning
    reconnection vertex, adding its RIS-weighted contribution to
    path_ptr[].estimate if visible. Assumes gi_temporal_spatial_combine has
    ALREADY been called on `res` -- state.w is a finalized RIS correction
    weight (reservoir_finalize's output), not a raw stream sum.

    `throughput` MUST be the path's throughput as it EXISTED ENTERING x1
    (before x1's own BSDF sample was applied) -- NOT path_ptr[].throughput
    read fresh here, which by bounce 1 already has x1's ORIGINAL sampled
    continuation direction's f/cos/pdf baked in. That original direction is
    irrelevant to the RESAMPLED reconnection direction this function
    connects along; using it caused a real, shipped ~10x energy inflation,
    caught only once interactive mode's temporal reuse was render-tested
    for the first time (see GIPendingX1's own docstring for the full story
    and why the earlier unit-level tests couldn't have caught it). Callers
    must pass the GIPendingX1 snapshot's own `throughput` field.

    No MIS against BSDF sampling, unlike di_resolve: the complementary
    "reach x2 anyway by continuing the path" contribution is exactly what
    THIS reservoir sample already represents (there is no separate,
    parallel BSDF-sampling estimate of the same x1->x2 connection to weigh
    against here) -- same reasoning as _gi_generate_recon_candidate's own
    no-MIS choice, applied one level up.

    Formula derivation is identical to di_resolve's own (verified there via
    the M=1 degenerate case): contrib = throughput * f(x1) * res.lo *
    (cos_s * state.w), NOT cos_s*cos_x2/dist_sq*state.w -- gi_target_pdf's
    p_hat already bakes in the full G=cos_x1*cos_x2/dist_sq term, and
    state.w carries a 1/p_hat(winner) factor that would double-count G if
    multiplied in again here."""
    if res.valid == Int8(0) or res.state.w <= Float32(0.0):
        return
    var to_recon = res.recon_point - hit_point
    var dist_sq = dot(to_recon, to_recon)
    if dist_sq <= Float32(1e-8):
        return
    var dist = sqrt(dist_sq)
    var wi = to_recon * (Float32(1.0) / dist)
    var cos_s = dot(normal, wi)
    var cos_x2 = -dot(res.recon_normal, wi)
    if cos_s <= Float32(0.0) or cos_x2 <= Float32(0.0):
        return
    var contrib = (throughput
        * _to_spec_diffuse_lit(ctx, alb, res.lo, path_ptr[].wavelengths)
        * (cos_s * res.state.w))
    _shadow_contribute[False](path_ptr, ctx, hit_point, wi, dist * Float32(0.9999), contrib)


# Temporal history clamp. The migration plan (and Bitterli et al. 2020)
# suggest ~20x the per-frame candidate count; measured here, 4x is better,
# so this deliberately departs from that guideline. staircase2 400x400,
# interactive 1spp/frame, MSE vs a 128spp plain-NEE reference:
#
#   temporal only, cap=20x (320):  0.001346 @128fr, converges 2.79x
#   temporal only, cap= 4x  (64):  0.001035 @128fr, converges 3.41x
#
# 1.30x better, with convergence rising to near plain NEE's own 3.75x.
# Mechanism: a reservoir pinned at a large cap gives each frame's fresh
# candidates only 16/320 ~ 5% of the total confidence, so it clings to
# recycled history and the effective independent-sample count grows far
# slower than m suggests. At 4x, fresh candidates carry 25%.
#
# Caveat worth respecting before trusting this number elsewhere: it is one
# scene, measured in the STATIC-camera accumulation regime, where trading
# sample independence for sample quality is a bad deal because the film is
# already averaging many frames. The cap's real job is bounding staleness
# across camera motion -- and render_interactive clears reservoirs outright
# on a camera move, so it rarely binds there at all. Re-measure before
# assuming 4x is right for a genuinely dynamic session.
comptime DI_TEMPORAL_M_CAP: Float32 = Float32(4 * DI_RIS_CANDIDATES)


# Phase 2.5 spatial reuse tuning, per the plan's own "k ~= 3-5 neighbors"
# and G-buffer rejection thresholds (docs/A2_restir_migration_plan.md).
#
# ENABLED. Getting here took three wrong verdicts, all caused by
# benchmarking in the wrong regime, so the reasoning is recorded in full.
#
# ReSTIR trades sample INDEPENDENCE for sample QUALITY. That pays when you
# display frame N; it costs when you AVERAGE N frames, because correlated
# frames do not average down. `--interactive-frames N` accumulates, so
# measuring at N=128 measures the regime ReSTIR is not for. Every earlier
# "ReSTIR loses" number here came from N=16/128.
#
# Measured on Scenes/restir-manylights.pbrt (256 equal-power lights, so
# the power-weighted CDF degenerates to uniform selection -- the case RIS
# exists to beat; ~500x spread in per-light contribution; all diffuse,
# maxdepth 1, so 100% of the image is inside ReSTIR's scope):
#
#   frames    plain    temporal   temporal+k=4    ReSTIR vs plain
#        1  0.07201     0.05772        0.05772    0.802x  WIN
#        2  0.02725     0.02152        0.02140    0.785x  WIN
#        4  0.01107     0.00934        0.00909    0.822x  WIN
#        8  0.00530     0.00513        0.00464    0.876x  WIN
#       16  0.00275     0.00316        0.00265    0.966x  WIN (spatial rescues it)
#
# Spatial contributes nothing at frame 1 (no neighbour has history yet)
# and grows to 0.838x of temporal-only by frame 16, turning what would be
# a 1.153x LOSS into a 0.966x win. It is off only in the sense that its
# benefit needs a frame or two to appear.
#
# Two earlier conclusions this overturned, both from bad benchmarks:
#   * "spatial is not worth enabling" -- measured on staircase2 (13
#     lights) and zero-day at 16/128 frames. Wrong on both axes.
#   * "ReSTIR never beats plain NEE" -- pure RIS with no reuse at all
#     (batch --restir) is 0.486x/0.499x on this scene, i.e. HALF the
#     error. The algorithm was always working; the benchmark was not.
#
# Also disproved here: the visibility theory (that RIS resamples toward
# unshadowed-looking-but-occluded lights). The generator emits an
# occluder-free variant, and ReSTIR behaves the same with and without
# occluders, so visibility is not what was limiting it.
comptime DI_SPATIAL_NEIGHBORS: Int = 4


# Fixed-size scratch for the Z pass, which must revisit each combined
# neighbour after the winner is known. +1 so the array is never zero-sized
# when someone sets DI_SPATIAL_NEIGHBORS = 0 to disable spatial reuse.
comptime DI_SPATIAL_SLOTS: Int = DI_SPATIAL_NEIGHBORS + 1


comptime DI_SPATIAL_RADIUS_PX: Float32 = Float32(20.0)


comptime DI_SPATIAL_NORMAL_DOT_MIN: Float32 = Float32(0.9)


comptime DI_SPATIAL_DEPTH_REL_MAX: Float32 = Float32(0.1)


# Defensive cap on a finalized reservoir's own state.w -- same fix, same
# root cause, as restir_gi.mojo's GI_MAX_FINALIZED_WEIGHT (see that
# constant's own comment for the full mechanism). reservoir_combine's
# weight formula (reservoir.mojo) multiplies in a SOURCE reservoir's own
# already-finalized state.w; without a bound, one anomalous value (root
# cause here: a stored light sample near-degenerate relative to a
# DIFFERENT pixel's own shading point -- e.g. a light sample point near
# the light mesh's own edge, viewed at grazing incidence from some pixel,
# not fully caught by the existing G-buffer rejection heuristics) gets
# fed back into every future combine that reuses it, compounding into
# unbounded growth. Confirmed via live tracing on cornell-box
# (--restir alone, no GI): w_sum grew ~1.3-1.7x per frame, reaching
# >600 billion within 30-40 frames while z_norm stayed roughly constant
# (~80) -- textbook exponential compounding, not a one-shot magnitude
# issue (this is why DI, unlike GI, WASN'T caught by this project's
# earlier extensive Phase 2 validation: that validation used interactive
# runs of a bounded frame count, per the plan's own noise-floor
# methodology, that happened to stay short enough to not yet manifest).
comptime DI_MAX_FINALIZED_WEIGHT: Float32 = Float32(10.0)


def di_temporal_step(
    path_ptr: Pointer[PathState, MutUntrackedOrigin], ctx: ShadeContext,
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    mut pcg: PCG32,
    restir_io: ReservoirIO = reservoir_io_null(),
    pixel_idx: Int = -1,
):
    """Phase 2.1/2.2/2.3/2.5/2.6/2.7: generate this frame's fresh RIS
    candidates (di_generate_reservoir), temporally combine with the previous
    frame's stored reservoir at the SAME pixel via IDENTITY reprojection
    (fact #2, docs/A2_restir_migration_plan.md -- static camera means no
    shift/Jacobian is needed, target_pdf is just di_target_pdf re-evaluated
    at this frame's hit_point using the previous winner's stored candidate),
    then spatially combine with DI_SPATIAL_NEIGHBORS random neighbor pixels'
    PREVIOUS-frame reservoirs (restir_io.read again, not this frame's --
    render_all_tiles parallelizes per-tile, so a same-frame neighbor read
    would race against whichever thread hasn't shaded it yet), each gated by
    a G-buffer rejection test. The "reconnection shift" the plan names is
    exactly di_target_pdf re-evaluated at THIS pixel's hit_point/normal/alb
    using the neighbor's stored sample_point/light_normal/le -- same
    mechanism already used for temporal reuse above, no separate Jacobian
    code needed (di_target_pdf's cos_l/dist_sq terms ARE that Jacobian).
    Resolves one shadow ray for the final combined winner, then M-caps and
    stores to restir_io.write for the next frame to read. Falls back to
    plain single-frame RIS (no persistence, no spatial reuse) when
    restir_io isn't real -- the batch (non-interactive) --restir path, which
    has no cross-frame concept."""
    var res = di_generate_reservoir(ctx, hit_point, normal, alb, pcg)
    var has_temporal = pixel_idx >= 0 and _is_real_ptr(restir_io.read)
    # Z-normalization bookkeeping, declared at function scope because the Z
    # pass below runs after the reuse block closes: which neighbour domains
    # took part, each one's confidence, and the confidence contributed by
    # domains identical to this pixel's own (see where it is assigned).
    var nb_px_seen = Array[Int32, DI_SPATIAL_SLOTS](fill=Int32(-1))
    var nb_m_seen = Array[Float32, DI_SPATIAL_SLOTS](fill=Float32(0))
    var nb_seen = 0
    var m_same_domain = Float32(0.0)
    if has_temporal:
        var prev = restir_io.read[unsafe_offset=pixel_idx]
        if prev.light_idx >= Int32(0):
            var p_hat_prev_here = di_target_pdf(hit_point, normal, alb, prev.sample_point, prev.light_normal, prev.le)
            var accept = reservoir_combine(res.state, prev.state, p_hat_prev_here, pcg.next_float())
            if accept:
                res.light_idx = prev.light_idx
                res.sample_point = prev.sample_point
                res.light_normal = prev.light_normal
                res.ldp_du = prev.ldp_du
                res.ldp_dv = prev.ldp_dv
                res.le = prev.le

        # Confidence accumulated so far comes from THIS pixel's own
        # candidates plus temporal reuse. Under identity reprojection the
        # previous frame's domain IS this pixel's domain (static camera), so
        # every bit of it can produce the winner and it all counts toward Z
        # unconditionally. Snapshot it before spatial combination starts,
        # since reservoir_combine grows state.m as it goes.
        m_same_domain = res.state.m

        if _is_real_ptr(restir_io.gbuf_normal) and restir_io.frame_w > Int32(0) and restir_io.frame_h > Int32(0):
            var self_px = Int32(pixel_idx) % restir_io.frame_w
            var self_py = Int32(pixel_idx) // restir_io.frame_w
            var self_depth = restir_io.gbuf_depth[unsafe_offset=pixel_idx]
            var self_mat = restir_io.gbuf_material_id[unsafe_offset=pixel_idx]
            for _ in range(DI_SPATIAL_NEIGHBORS):
                var ang = pcg.next_float() * Float32(6.283185307)
                var rad = sqrt(pcg.next_float()) * DI_SPATIAL_RADIUS_PX
                var nb_p = restir_jitter_pixel(Point2i(self_px, self_py), ang, rad)
                var nx = nb_p.x
                var ny = nb_p.y
                if nx < Int32(0) or nx >= restir_io.frame_w or ny < Int32(0) or ny >= restir_io.frame_h:
                    continue
                var n_idx = Int(ny * restir_io.frame_w + nx)
                if n_idx == pixel_idx:
                    continue
                var n_off = n_idx * 3
                var n_normal = Vec3f(
                    restir_io.gbuf_normal[unsafe_offset=n_off], restir_io.gbuf_normal[unsafe_offset=n_off + 1], restir_io.gbuf_normal[unsafe_offset=n_off + 2])
                if dot(n_normal, normal) < DI_SPATIAL_NORMAL_DOT_MIN:
                    continue
                var n_depth = restir_io.gbuf_depth[unsafe_offset=n_idx]
                if self_depth <= Float32(0.0) or abs(n_depth - self_depth) > DI_SPATIAL_DEPTH_REL_MAX * self_depth:
                    continue
                if restir_io.gbuf_material_id[unsafe_offset=n_idx] != self_mat:
                    continue
                var nb = restir_io.read[unsafe_offset=n_idx]
                if nb.light_idx < Int32(0):
                    continue
                var p_hat_nb_here = di_target_pdf(hit_point, normal, alb, nb.sample_point, nb.light_normal, nb.le)
                # Remember every neighbour actually folded in (winner or
                # not) -- Z is a property of which DOMAINS took part, not of
                # which one happened to win.
                if nb_seen < DI_SPATIAL_SLOTS:
                    nb_px_seen[nb_seen] = Int32(n_idx)
                    nb_m_seen[nb_seen] = nb.state.m
                    nb_seen += 1
                var accept_nb = reservoir_combine(res.state, nb.state, p_hat_nb_here, pcg.next_float())
                if accept_nb:
                    res.light_idx = nb.light_idx
                    res.sample_point = nb.sample_point
                    res.light_normal = nb.light_normal
                    res.ldp_du = nb.ldp_du
                    res.ldp_dv = nb.ldp_dv
                    res.le = nb.le

    # ── Z normalization (Bitterli et al. 2020, Algorithm 6) ───────────────
    # Dividing W by the full accumulated m assumes every combined reservoir
    # could have produced the chosen sample. Across DIFFERENT pixels that is
    # false: a neighbour whose surface faces away from the winning light
    # never could have generated it, yet its confidence still inflates the
    # denominator -- the systematic under-weighting that makes Algorithm 4
    # biased. Z instead counts only the domains that genuinely contain the
    # winner, which is the whole difference between the biased and unbiased
    # variants.
    #
    # "Domain i contains y" is tested by re-evaluating the target function
    # at neighbour i's OWN shading point (its world position + normal from
    # the Phase 0.3 G-buffer) rather than at this pixel's -- that is what
    # makes it a statement about neighbour i's integrand and not merely a
    # restatement of our own. Only the SIGN matters (>0 vs 0), so passing
    # this pixel's albedo is harmless: albedo scales p_hat, it cannot change
    # whether it is zero (and an all-black albedo contributes nothing under
    # either variant anyway). The test is therefore purely geometric --
    # exactly the cos_s>0 / cos_l>0 / dist>0 conditions inside di_target_pdf.
    var z_norm = Float32(-1.0)
    if nb_seen > 0 and _is_real_ptr(restir_io.gbuf_world_pos):
        var z = m_same_domain
        for i in range(nb_seen):
            var np_off = Int(nb_px_seen[i]) * 3
            var n_hit = Vec3f(
                restir_io.gbuf_world_pos[unsafe_offset=np_off],
                restir_io.gbuf_world_pos[unsafe_offset=np_off + 1],
                restir_io.gbuf_world_pos[unsafe_offset=np_off + 2])
            var n_nrm = Vec3f(
                restir_io.gbuf_normal[unsafe_offset=np_off],
                restir_io.gbuf_normal[unsafe_offset=np_off + 1],
                restir_io.gbuf_normal[unsafe_offset=np_off + 2])
            if di_target_pdf(n_hit, n_nrm, alb, res.sample_point, res.light_normal, res.le) > Float32(0.0):
                z += nb_m_seen[i]
        z_norm = z

    di_resolve(path_ptr, ctx, hit_point, normal, alb, res, z_norm)
    if has_temporal:
        # Note the ordering: state.m still holds the TRUE accumulated
        # confidence here (z_norm renormalized only W, never m), so the
        # M-cap and the stored history stay correct for the next frame.
        reservoir_cap_confidence(res.state, DI_TEMPORAL_M_CAP)
        restir_io.write[unsafe_offset=pixel_idx] = res
