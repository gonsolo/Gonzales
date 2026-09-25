# Volumetric ReSTIR (Lin, Kettunen, Bitterli, Pantaleoni, Yuksel, Wyman 2021 --
# "Fast Volume Rendering with Spatiotemporal Reservoir Resampling"), Phase 7 of
# docs/A2_restir_migration_plan.md -- read that section for the Ghost ReSTIR
# poster background and full implementation/wiring status. Payload type and
# pure target-function math only, no dependency on the medium sampler that
# calls it (gpu.mojo's _sample_medium_core) -- same one-directional layering
# restir_di.mojo and restir_gi.mojo use, for the same circular-import reason.
#
# WHAT IS RESAMPLED. A reservoir candidate is the PAIR (volume scattering
# vertex, light sample taken there), resampled jointly -- distance sampling
# alone is blind to where the light is, light sampling alone can't know which
# distances are unoccluded, so resampling the pair is what finds the few
# (t, light) combinations that carry the energy.
#
# WHY THE VERTEX IS STORED IN WORLD SPACE. Exactly as DIReservoir's
# `sample_point` and GIReservoir's `recon_point` are: reuse moves a sample to a
# different PIXEL, not to a different point in the scene, so a world-space
# vertex stays meaningful verbatim and the target function re-evaluated at the
# new pixel's own ray does all the reweighting. See vol_shift_scatter_vertex
# for the one seam where that assumption is stated explicitly.
#
# TWO DELIBERATE SEAMS (see A2's Phase 7 section for why -- the newer,
# ghost-vertex formulation exists only as a poster abstract, no derivation):
#   1. Transmittance. vol_target_pdf takes `tr` as an explicit argument rather
#      than computing it; VOL_TR_UNIT gives a target with no intermediate
#      transmittance, the property the newer formulation needs.
#   2. The domain mapping of the reconnection shift. vol_shift_scatter_vertex
#      is a mode dispatch (VolShiftMode), not an inlined assumption, so a
#      ghost-vertex bijection can be added as a second mode without touching
#      the reservoir plumbing.
#
# Scope actually implemented here: the payload, the target function, and the
# temporal+spatial combine (vol_temporal_spatial_combine) -- fully unit-tested
# (Tests/unit/test_restir_vol.mojo) but NOT YET WIRED into the render loop (no
# persistent per-pixel buffers, no G-buffer plumbing, no call site). Candidate
# generation and the shadow-ray resolve ARE wired, in gpu.mojo's
# _sample_medium_core, next to the medium sampler that owns the ray -- exactly
# as DI's generation half lives in shading.mojo.

from std.collections import Array
from std.math import sqrt, cos, sin, abs
from .geometry import RGB, dot, _is_real_ptr, Vec3f, Point2i, restir_jitter_pixel
from .media import hg_phase
from .reservoir import ReservoirState, reservoir_state_init, reservoir_combine, reservoir_finalize, reservoir_cap_confidence
from .rng import PCG32

# How many light candidates one volume scatter vertex resamples over. The
# asymmetry that makes this worth doing: a candidate costs a light pick plus an
# unshadowed target evaluation, while resolving one costs a visibility ray and,
# in a heterogeneous medium, a full ratio-tracking transmittance march -- so M
# candidates and ONE resolve is close to the price of the single sample it
# replaces.
#
# 1 reduces the estimator EXACTLY to that single sample (W = (p_hat/q)/p_hat =
# 1/q), which makes it the natural A/B baseline and a cheap correctness check.
# 8 is a starting value, not a measured optimum -- there is no tuning data for
# volumetric candidate counts in this renderer yet.
comptime VOL_RIS_CANDIDATES: Int = 8

# ── DISTANCE resampling (Phase 7.3's last piece) ────────────────────────────
# When true, each of the VOL_RIS_CANDIDATES candidates draws its OWN scatter
# distance along the segment as well as its own light sample, so the reservoir
# resamples the joint (distance, light) pair instead of the light alone at a
# fixed vertex.
#
# The derivation is short because the proposal is chosen to make it short.
# Draw candidate distances from the EXACT conditional collision density
#   q(t) = sigma_t e^{-sigma_t t} / P_c,  P_c = 1 - e^{-sigma_t t_surf}
# (analytic for a homogeneous medium), pair with the existing light sampler
# q_L, and use the target p_hat(t,y) = q(t) * c_hat(t,y) with c_hat the same
# unshadowed, transmittance-free `vol_target_pdf` 7.2 already uses. Then
#   w_i      = p_hat/Q = q(t_i)c_hat_i / (q(t_i) q_L)  = c_hat_i / q_L
#   F/p_hat  = q(t_Y)c_Y / (q(t_Y)c_hat_Y)             = c_Y / c_hat_Y
# -- q(t) cancels in BOTH places. So the weight formula and the resolve are
# bit-identical to 7.2's; the only difference is that c_hat is evaluated at
# each candidate's own point and the winner's point is what gets shadowed.
# No transmittance march, no 1/P_c divisor, no throughput correction.
#
# That last point is worth stating plainly because an earlier design pass
# (recorded in project_restir_migration) reached a DIFFERENT and more
# expensive answer -- a majorant-rate exponential proposal, corrected by
# h(t) = T(0,t)/P(collided) and costing two ratio-tracking marches per scatter
# event. That design is not merely costlier, it is BIASED for heterogeneous
# media: P(collided) = 1 - T(0,t_surf) has no closed form there, so T must be
# estimated by ratio tracking, and the estimator then divides by 1 - T_hat.
# Since x -> 1/(1-x) is strictly convex, E[1/(1-T_hat)] > 1/(1-E[T_hat]) by
# Jensen -- a systematic OVER-estimate, worst exactly where the medium is
# optically thin and T -> 1. Sampling the proposal from the exact conditional
# instead removes the offending factors rather than trying to estimate them.
#
# Scope, deliberately narrow: HOMOGENEOUS, achromatic media only (see the
# guard at the use site in gpu.mojo). Heterogeneous media would need candidate
# distances drawn from their own exact conditional too -- an independent
# delta-tracking walk per candidate, rejection-conditioned on collision, which
# is unbiased and needs no marches either but costs ~M walks per segment in
# expectation. Correct and known; simply not paid for until the cheap
# homogeneous case shows the technique earns its keep at all.
#
# MEASURED (2026-09-09, RE-measured after the sphere-boundary shadow-ray fix
# -- 64x64 GPU, --no-denoise, MSE at a matched 64spp budget over 5 seeds,
# against a 65536spp reference of the same estimator):
#
#   scene                                   MSE vs 7.2   bias @4096spp
#   thin fog + close light (favourable)       -19.5%        0.99939
#   dense fog + distant light                  -1.9%        0.99982
#
# ON by default as of that measurement. Every earlier figure for this feature
# was taken before the sphere-boundary fix, when a shadow ray leaving a
# sphere-bounded medium was Beer-Lambert'd across the vacuum all the way to
# the light -- both test scenes here are sphere-bounded, so those numbers were
# measured against a baseline 574x too dark and are void. The technique
# survived re-measurement; the conclusions drawn ALONGSIDE it did not (see
# VOL_TEMPORAL_M_CAP's note on temporal reuse).
#
# The unbiasedness check that carries the most weight is not the 4096spp
# ratios above but the two independent 65536spp references: the plain 7.2
# estimator and this one converge to 0.132659 vs 0.132683 on thin fog (0.018%
# apart) and 0.030218 vs 0.030220 on dense (0.006%). Two structurally
# different estimators agreeing to that tolerance is much harder to fake than
# either one matching itself.
#
# It is also FREE -- the added work per candidate is one RNG draw and one log,
# against a light pick, a target evaluation and a reservoir update.
#
# Compile-time constant rather than a CLI flag, matching VOL_SPATIAL_NEIGHBORS
# below: this code runs inside GPU kernels, where a runtime switch has to be
# threaded through every kernel signature.
comptime VOL_RIS_DISTANCE: Bool = True

# Transmittance seam sentinel -- see this file's header, seam 1. Passing this
# as vol_target_pdf's `tr` yields the transmittance-free target function.
comptime VOL_TR_UNIT: Float32 = Float32(1.0)

struct VolShiftMode:
    """How a stored scattering vertex is mapped onto a different pixel's camera
    ray. A dispatch struct rather than a bare Bool for the same reason
    reservoir.mojo's ReprojectMode is one: a second mode is expected (see this
    file's header, seam 2) and call sites should not have to change shape when
    it arrives."""

    # Reuse the world-space vertex verbatim and let the target function,
    # re-evaluated against the new ray, do the reweighting. This is the older
    # formulation's shift and it is exact for the vertex ITSELF; what it cannot
    # express is a change in the null-collision structure between the two rays,
    # which is precisely the gap ghost vertices close.
    comptime identity: Int32 = Int32(0)
    # Reserved: ghost-vertex bijection in null-scattering primary sample space.
    # NOT implemented -- no published derivation exists yet (poster abstract
    # only). vol_shift_scatter_vertex rejects it rather than silently falling
    # back to `identity`, so a caller cannot half-enable it by accident.
    comptime ghost: Int32 = Int32(1)
    # Keep the RECEIVING pixel's own scattering vertex and import only the
    # light sample. Correct -- and the ONLY correct choice -- when the vertex
    # was NOT drawn from a shared continuous proposal, i.e. when distance
    # resampling is off and the vertex is whatever delta-tracking's single
    # t_free happened to be for this path. Two frames' vertices are then point
    # masses with disjoint support: the receiver's proposal could never have
    # produced the donor's vertex, so importing it and dividing by the pooled
    # m over-counts. Re-targeting the light sample onto our own vertex is the
    # textbook ReSTIR DI reuse (the shading point is fixed by the pixel, only
    # the light sample travels) and sidesteps the question entirely.
    #
    # `identity` is right for the opposite case: with distance resampling on,
    # the vertex comes from q(t) = sigma_t e^{-sigma_t t}/(1 - e^{-sigma_t
    # t_surf}), which depends only on sigma_t and t_surf -- identical across
    # frames for a static camera at one pixel. The donor's vertex is then a
    # draw from exactly the receiver's own proposal, so the domains match and
    # the Jacobian is 1.
    comptime retarget: Int32 = Int32(2)

@fieldwise_init
struct VolReservoir(TrivialRegisterPassable):
    """One resampled (volume scattering vertex, light sample) pair.

    `scatter_point` is the vertex in world space. `sigma_s` and `phase_g` are
    the medium's scattering coefficient and Henyey-Greenstein asymmetry AT
    that vertex, carried on the payload rather than re-looked-up because a
    reusing pixel evaluates the target at a vertex that may sit in a
    different part of a heterogeneous medium than anything on its own ray --
    re-sampling the density there would answer the wrong question. `sigma_s`
    is the achromatic (hero/red-channel) coefficient, matching the convention
    the free-flight sampler itself uses.

    `light_*`/`le` are the light sample chosen at that vertex, kept so the
    eventual winner can be shadow-resolved once, mirroring DIReservoir.
    `medium_idx` gates reuse: a vertex from a different medium is not a
    lower-quality candidate for this pixel, it is a meaningless one.

    `valid` mirrors GIReservoir's own flag (0 = no winner yet; every other
    field is meaningless until a candidate has been streamed)."""
    var scatter_point: Vec3f
    var light_point:   Vec3f
    var light_normal:  Vec3f
    var le:            RGB
    var sigma_s:       Float32
    var phase_g:       Float32
    var light_idx:     Int32
    var medium_idx:    Int32
    var valid:         Int8
    var _pad0:         Int8
    var _pad1:         Int8
    var _pad2:         Int8
    var state:         ReservoirState

@always_inline
def vol_reservoir_init() -> VolReservoir:
    return VolReservoir(
        scatter_point=Vec3f(Float32(0)),
        light_point=Vec3f(Float32(0)),
        light_normal=Vec3f(Float32(0)),
        le=RGB(Float32(0)),
        sigma_s=Float32(0),
        phase_g=Float32(0),
        light_idx=Int32(-1),
        medium_idx=Int32(-1),
        valid=Int8(0),
        _pad0=Int8(0), _pad1=Int8(0), _pad2=Int8(0),
        state=reservoir_state_init(),
    )

@always_inline
def vol_shift_scatter_vertex(
    mode: Int32, scatter_point: Vec3f, ray_origin: Vec3f, ray_dir: Vec3f,
    receiver_vertex: Vec3f = Vec3f(Float32(0)),
) -> Tuple[Bool, Vec3f]:
    """Map a stored scattering vertex onto the camera ray (`ray_origin`,
    `ray_dir`) of the pixel now trying to reuse it. Returns (ok, vertex).

    Seam 2 (see this file's header). Under `identity` the vertex is reused
    verbatim -- reuse relocates a sample to another PIXEL, not to another
    point in the scene, and a world-space vertex is still a valid place for
    light to scatter no matter which ray asks about it. What the identity map
    does NOT model is the two rays having different null-collision structure
    through the medium; that is the gap ghost vertices are for, and it is why
    `ghost` is rejected here rather than aliased onto `identity`.

    Under `retarget` the donor's vertex is DISCARDED and `receiver_vertex` --
    the reusing pixel's own scattering vertex -- is returned, so only the
    light sample travels. See VolShiftMode.retarget for when each is correct;
    the short version is that `identity` needs the two vertices to be draws
    from a shared continuous proposal, which is true exactly when distance
    resampling is on.

    Whichever vertex comes back is the one the caller must BOTH evaluate the
    target at and, eventually, trace the shadow ray from: reservoir_finalize
    divides by p_hat of the chosen sample, so a resolve that shadows from a
    different point silently breaks the RIS identity.

    `ray_origin`/`ray_dir` are unused by both maps and are taken anyway so
    that adding a real (ray-dependent) mapping later does not change this
    function's signature, and therefore does not change any call site."""
    if mode == VolShiftMode.identity:
        return (True, scatter_point)
    if mode == VolShiftMode.retarget:
        return (True, receiver_vertex)
    # VolShiftMode.ghost, or anything unrecognised: refuse. Falling back to
    # identity would silently produce the older formulation's answer while the
    # caller believed it had the newer one.
    return (False, Vec3f(Float32(0)))

@always_inline
def vol_target_pdf(
    ray_dir: Vec3f,
    scatter_point: Vec3f,
    sigma_s: Float32, phase_g: Float32,
    light_point: Vec3f, light_normal: Vec3f, le: RGB,
    tr: Float32,
) -> Float32:
    """RIS target function p̂ for a (volume vertex, light sample) candidate:
    luminance of sigma_s x phase(wo, wi) x Tr x Le x G(x, y).

    `ray_dir` is the camera ray's direction of travel, so wo -- the direction
    back toward the viewer, which is what hg_phase's convention wants -- is
    its negation. G here is cos(y) / dist^2 and carries NO cosine at the
    scattering vertex: a phase function is not cosine-weighted, unlike
    di_target_pdf's surface BSDF. Getting that wrong is the volumetric
    analogue of the bug gi_target_pdf's docstring records at length.

    Visibility is deliberately excluded, resolved once for the eventual
    winner, exactly as in DI and GI. `tr` is the transmittance seam (see this
    file's header, seam 1): pass a real estimate for the published
    formulation, or VOL_TR_UNIT for a target function with no intermediate
    transmittance in it at all.

    Scalar (luminance) target rather than per-channel, matching DI/GI and
    standard ReSTIR practice -- keeps w_sum/w single floats."""
    if sigma_s <= Float32(0.0) or tr <= Float32(0.0):
        return Float32(0.0)
    var to_light = light_point - scatter_point
    var dist_sq = to_light.length_sq()
    if dist_sq < Float32(1e-12):
        return Float32(0.0)
    var dist = sqrt(dist_sq)
    var wi = to_light.to_simd() / dist
    # Emission leaves the front face only; a light seen edge-on or from
    # behind contributes nothing and must score 0, not a tiny positive.
    var cos_light = -dot(wi, light_normal.to_simd())
    if cos_light <= Float32(0.0):
        return Float32(0.0)
    var wo = -ray_dir.to_simd()
    var ph = hg_phase(dot(wo, wi), phase_g)
    var g_term = cos_light / dist_sq
    var scale = sigma_s * ph * tr * g_term
    var r = le.r * scale
    var g = le.g * scale
    var b = le.b * scale
    return r * Float32(0.2126) + g * Float32(0.7152) + b * Float32(0.0722)

@fieldwise_init
struct VolReservoirIO(TrivialRegisterPassable):
    """Persistent per-pixel reservoir buffers plus the Phase 0.3 G-buffer
    pointers used for spatial-neighbour rejection. Duplicated rather than
    shared with ReservoirIO/GIReservoirIO, which are typed for their own
    payloads -- this codebase avoids generics throughout (see reservoir.mojo's
    docstring) in favour of small independently-readable structs.

    All pointers default to `.unsafe_dangling()` and frame_w/frame_h to 0;
    vol_temporal_spatial_combine checks `_is_real_ptr`/`> 0` before touching
    them, the same null-safety contract the other two follow."""
    var read:  Pointer[VolReservoir, MutUntrackedOrigin]
    var write: Pointer[VolReservoir, MutUntrackedOrigin]
    var gbuf_depth:       Pointer[Float32, MutUntrackedOrigin]
    var gbuf_world_pos:   Pointer[Float32, MutUntrackedOrigin]
    var frame_w: Int32
    var frame_h: Int32

@always_inline
def vol_reservoir_io_null() -> VolReservoirIO:
    return VolReservoirIO(
        read=Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
        write=Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
        gbuf_depth=Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        gbuf_world_pos=Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        frame_w=Int32(0), frame_h=Int32(0),
    )

# Tuning constants. Started from GI's (restir_gi.mojo), themselves started from
# DI's MEASURED values. A volume vertex has neither a normal nor a material
# id, so two of DI/GI's three G-buffer rejection tests do not apply at all
# here -- what replaces them is the medium match plus the target function
# scoring an incompatible neighbour at 0 on its own.
#
# TEMPORAL REUSE (`--vol-restir-reuse`) NO LONGER SHOWS A MEASURABLE WIN, and
# the constants below are therefore untuned against anything real. It was
# recorded at 3-10x MSE reduction on GPU; re-measured 2026-09-09 after the
# sphere-boundary shadow-ray fix, on the same dense scene plus a thin-fog one,
# both against 65536spp references:
#
#   scene       temporal-only MSE vs off   bias @4096spp
#   thin fog             +2.9%                0.99998
#   dense fog            -0.7%                1.00004
#
# i.e. a wash on dense and mildly WORSE on thin. The original figure was taken
# when a shadow ray leaving a sphere-bounded medium was attenuated across the
# vacuum out to the light, so it compared two variants of a near-black image
# (that scene's mean was 5.7e-5; it is 0.0302 once correct) -- MSE ratios
# measured against a 574x-too-dark baseline do not survive the fix.
#
# It remains UNBIASED and correct, and it stays wired and opt-in rather than
# being removed: the machinery is shared with DI/GI, and a scene with genuine
# frame-to-frame coherence (interactive camera, many accumulated frames) is a
# different regime from these batch measurements. But nothing currently
# justifies defaulting it on, and its constants should be re-derived rather
# than trusted if it is ever revisited.
comptime VOL_TEMPORAL_M_CAP: Float32 = Float32(64.0)
# MEASURED 2026-09-08, now that temporal reuse + G-buffer wiring both exist
# (`Scenes/vol-restir-mesh-light.pbrt`, 5 seeds, matched cap, vs a 16384spp
# reference -- same methodology DI's own spatial-reuse verdict used):
# average MSE across seeds is a near-wash, mildly WORSE with spatial on
# (temporal-only 0.272 vs temporal+spatial 0.287, both ×1e-12 scale) and
# highly variable per seed (spatial ranged from 76% worse to 54% better on
# individual seeds) -- no consistent win. Both configurations are UNBIASED
# (converged mean within ~0.15% of the reference either way, Z-normalization
# already correct from Phase 7.1) -- this is DI's exact "correct but
# genuinely not worth enabling" verdict, now actually measured for the
# volumetric case instead of assumed by analogy. Machinery stays wired and
# correct; re-enabling is a one-constant change, same as DI's own history.
comptime VOL_SPATIAL_NEIGHBORS: Int = 0
comptime VOL_SPATIAL_SLOTS: Int = VOL_SPATIAL_NEIGHBORS + 1
comptime VOL_SPATIAL_RADIUS_PX: Float32 = Float32(20.0)
comptime VOL_SPATIAL_DEPTH_REL_MAX: Float32 = Float32(0.25)
# Defensive cap on a finalized reservoir's own state.w -- the same safety valve
# GI_MAX_FINALIZED_WEIGHT (restir_gi.mojo) provides, against the same real
# hazard: reservoir_combine feeds a source reservoir's already-finalized
# state.w back into future combines, so one anomalous value compounds into
# unbounded growth instead of staying a one-frame outlier (measured for GI at
# >1,000,000 within ~20 frames).
#
# DISABLED by default here, deliberately. GI's measured threshold of 10 does
# NOT transfer: for a single candidate W = w_sum / (m * p_hat) is exactly
# 1/q, the reciprocal of the source sampling pdf, which for volumetric
# distance sampling is legitimately small and therefore gives a legitimately
# large W. This file's own hand-computed unit test produces W ~= 168 from
# perfectly healthy geometry -- GI's constant would have silently cost that
# case a factor of 16. Phase 7.1 has no candidate generation yet, so there is
# nothing to measure a real threshold against; picking one by analogy would be
# inventing a number that quietly destroys energy. Set this once generation
# lands (Phase 7.2) the way DI's and GI's own constants were set: empirically,
# from observed healthy-vs-anomalous values on a real render.
comptime VOL_MAX_FINALIZED_WEIGHT: Float32 = Float32(3.4e38)

def vol_temporal_spatial_combine(
    mut res: VolReservoir,
    ray_origin: Vec3f, ray_dir: Vec3f,
    medium_idx: Int32,
    mut pcg: PCG32,
    vol_io: VolReservoirIO = vol_reservoir_io_null(),
    pixel_idx: Int = -1,
    shift_mode: Int32 = VolShiftMode.identity,
):
    """Combine `res` (this pixel's own freshly-generated, already-streamed M=1
    candidate) with the previous frame's reservoir at the same pixel
    (temporal, identity reprojection) and VOL_SPATIAL_NEIGHBORS random
    neighbours' previous-frame reservoirs (spatial).

    Rejection differs from DI/GI by necessity: a volume vertex has no normal
    and no material id, so the surviving gates are (a) the neighbour must be
    in the SAME medium -- a vertex from another medium is meaningless here,
    not merely poor -- (b) a coarse depth-similarity test on the primary
    surface behind each pixel, as a locality proxy, and (c) the shift
    returning ok. Beyond those, an incompatible neighbour scores p̂ = 0 and
    is harmless.

    Finalizes via reservoir_finalize with Bitterli et al. 2020 Algorithm 6's
    Z-normalization when spatial neighbours were folded in, and falls back to
    a plain single-frame finalize (no persistence, no spatial reuse) when
    `vol_io` isn't real or `pixel_idx < 0` -- the same null-safety contract
    di_temporal_step and gi_temporal_spatial_combine follow.

    Transmittance is passed as VOL_TR_UNIT at every target-function
    evaluation in here ON PURPOSE. This is the resampling stage, and it is
    exactly the stage the newer formulation identifies as needing a target
    with no intermediate transmittance so that it evaluates consistently
    wherever it is applied; the real transmittance belongs in the resolve,
    where it is computed once for the winner along a ray that is actually
    traced. See this file's header, seam 1.

    Deliberately does NOT trace anything or touch any path's throughput --
    that's resolution, a separate step owned by the medium sampler. Callers
    that only want the combine math (as this file's unit tests do) can call
    it directly on synthetic reservoirs with no rendering involved."""
    var has_temporal = pixel_idx >= 0 and _is_real_ptr(vol_io.read)
    var nb_px_seen = Array[Int32, VOL_SPATIAL_SLOTS](fill=Int32(-1))
    var nb_m_seen = Array[Float32, VOL_SPATIAL_SLOTS](fill=Float32(0))
    var nb_seen = 0
    var m_same_domain = Float32(0.0)

    # The receiving pixel's OWN vertex and the medium properties there, taken
    # before any donor can overwrite them. `retarget` maps every donor onto
    # this vertex, and the medium terms describe the VERTEX rather than the
    # light sample, so they have to travel with it -- in a heterogeneous
    # medium sigma_s at the donor's position is simply not sigma_s at ours.
    # (The medium_idx gate below only guarantees the same medium, not the same
    # density within it.)
    var recv_vertex = res.scatter_point
    var recv_sigma_s = res.sigma_s
    var recv_phase_g = res.phase_g
    var keep_recv_vertex = shift_mode == VolShiftMode.retarget

    if has_temporal:
        var prev = vol_io.read[unsafe_offset=pixel_idx]
        if prev.valid != Int8(0) and prev.medium_idx == medium_idx:
            var (ok_prev, pt_prev) = vol_shift_scatter_vertex(
                shift_mode, prev.scatter_point, ray_origin, ray_dir, recv_vertex)
            if ok_prev:
                var sig_prev = recv_sigma_s if keep_recv_vertex else prev.sigma_s
                var g_prev = recv_phase_g if keep_recv_vertex else prev.phase_g
                var p_hat_prev = vol_target_pdf(
                    ray_dir, pt_prev, sig_prev, g_prev,
                    prev.light_point, prev.light_normal, prev.le, VOL_TR_UNIT)
                if reservoir_combine(res.state, prev.state, p_hat_prev, pcg.next_float()):
                    res.scatter_point = pt_prev
                    res.light_point = prev.light_point
                    res.light_normal = prev.light_normal
                    res.le = prev.le
                    res.sigma_s = sig_prev
                    res.phase_g = g_prev
                    res.light_idx = prev.light_idx
                    res.medium_idx = prev.medium_idx
                    res.valid = Int8(1)

        m_same_domain = res.state.m

        if _is_real_ptr(vol_io.gbuf_depth) and vol_io.frame_w > Int32(0) and vol_io.frame_h > Int32(0):
            var self_px = Int32(pixel_idx) % vol_io.frame_w
            var self_py = Int32(pixel_idx) // vol_io.frame_w
            var self_depth = vol_io.gbuf_depth[unsafe_offset=pixel_idx]
            for _ in range(VOL_SPATIAL_NEIGHBORS):
                var ang = pcg.next_float() * Float32(6.283185307)
                var rad = sqrt(pcg.next_float()) * VOL_SPATIAL_RADIUS_PX
                var nb_p = restir_jitter_pixel(Point2i(self_px, self_py), ang, rad)
                var nx = nb_p.x
                var ny = nb_p.y
                if nx < Int32(0) or nx >= vol_io.frame_w or ny < Int32(0) or ny >= vol_io.frame_h:
                    continue
                var n_idx = Int(ny * vol_io.frame_w + nx)
                if n_idx == pixel_idx:
                    continue
                var n_depth = vol_io.gbuf_depth[unsafe_offset=n_idx]
                if self_depth <= Float32(0.0) or abs(n_depth - self_depth) > VOL_SPATIAL_DEPTH_REL_MAX * self_depth:
                    continue
                var nb = vol_io.read[unsafe_offset=n_idx]
                if nb.valid == Int8(0) or nb.medium_idx != medium_idx:
                    continue
                var (ok_nb, pt_nb) = vol_shift_scatter_vertex(
                    shift_mode, nb.scatter_point, ray_origin, ray_dir, recv_vertex)
                if not ok_nb:
                    continue
                var sig_nb = recv_sigma_s if keep_recv_vertex else nb.sigma_s
                var g_nb = recv_phase_g if keep_recv_vertex else nb.phase_g
                var p_hat_nb = vol_target_pdf(
                    ray_dir, pt_nb, sig_nb, g_nb,
                    nb.light_point, nb.light_normal, nb.le, VOL_TR_UNIT)
                if nb_seen < VOL_SPATIAL_SLOTS:
                    nb_px_seen[nb_seen] = Int32(n_idx)
                    nb_m_seen[nb_seen] = nb.state.m
                    nb_seen += 1
                if reservoir_combine(res.state, nb.state, p_hat_nb, pcg.next_float()):
                    res.scatter_point = pt_nb
                    res.light_point = nb.light_point
                    res.light_normal = nb.light_normal
                    res.le = nb.le
                    res.sigma_s = sig_nb
                    res.phase_g = g_nb
                    res.light_idx = nb.light_idx
                    res.medium_idx = nb.medium_idx
                    res.valid = Int8(1)

    # Z normalization (Bitterli et al. 2020, Algorithm 6): dividing by the
    # full accumulated m over-counts neighbour domains that could never have
    # produced the winning sample. Re-evaluates the target at each folded-in
    # neighbour's OWN camera ray, reconstructed from the G-buffer world
    # position (the ray through that pixel's primary hit), not this pixel's.
    var z_norm = Float32(-1.0)
    if nb_seen > 0 and _is_real_ptr(vol_io.gbuf_world_pos) and res.valid != Int8(0):
        var z = m_same_domain
        for i in range(nb_seen):
            var np_off = Int(nb_px_seen[i]) * 3
            var n_hit = Vec3f(
                vol_io.gbuf_world_pos[unsafe_offset=np_off],
                vol_io.gbuf_world_pos[unsafe_offset=np_off + 1],
                vol_io.gbuf_world_pos[unsafe_offset=np_off + 2])
            var n_seg = n_hit - ray_origin
            var n_len_sq = n_seg.length_sq()
            if n_len_sq < Float32(1e-12):
                continue
            var n_dir_s = n_seg.to_simd() / sqrt(n_len_sq)
            var n_dir = Vec3f(n_dir_s[0], n_dir_s[1], n_dir_s[2])
            if vol_target_pdf(
                n_dir, res.scatter_point, res.sigma_s, res.phase_g,
                res.light_point, res.light_normal, res.le, VOL_TR_UNIT) > Float32(0.0):
                z += nb_m_seen[i]
        z_norm = z

    var p_hat_final = Float32(0.0)
    if res.valid != Int8(0):
        p_hat_final = vol_target_pdf(
            ray_dir, res.scatter_point, res.sigma_s, res.phase_g,
            res.light_point, res.light_normal, res.le, VOL_TR_UNIT)
    reservoir_finalize(res.state, p_hat_final, z_norm)

    if res.state.w > VOL_MAX_FINALIZED_WEIGHT:
        res.state.w = VOL_MAX_FINALIZED_WEIGHT

    if has_temporal:
        # state.m still holds the TRUE accumulated confidence here (z_norm
        # renormalized only W, never m) -- cap AFTER finalize, the same
        # ordering and reason as DI's and GI's own combines.
        reservoir_cap_confidence(res.state, VOL_TEMPORAL_M_CAP)
        vol_io.write[unsafe_offset=pixel_idx] = res
