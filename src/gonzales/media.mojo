"""Participating media, split out of geometry.mojo (the per-cluster module
split; see project_geometry_module_split memory). Medium_C through
MediumInterface_C, and the later "one free-flight sampler for every
integrator" section (which had drifted far from the rest of the medium code
in geometry.mojo, separated by Frame/the geometry helpers/the SIMD math
helpers), were the two ranges that made up this cluster -- both depend only
on the core (Point3f/Vec3f/RGB/Frame) plus the same external modules
geometry.mojo already imported for them (nanovdb.mojo, spectrum.mojo,
rng.mojo), confirmed by a symbol-reference scan of each block with comments
and docstrings stripped before moving it."""
from std.math import sqrt, cos, sin, min, max, abs, floor, log, exp
from gonzales.spectrum import SampledWavelengths, SpectralSample, spec_refl, spec_refl_unbounded, rgb_illuminant_to_spectral_sample
from gonzales.nanovdb import nvdb_sample_index, nvdb_majorant_at, nvdb_leaf_base, nvdb_leaf_value
from gonzales.rng import PCG32
from .geometry import Point3f, Vec3f, RGB, Frame, INV_FOUR_PI, TWO_PI

@fieldwise_init
struct Medium_C(TrivialRegisterPassable):
    """Participating medium (PBRT-v4 HomogeneousMedium / GridMedium "uniformgrid").
    sigma_a + sigma_s are pre-scaled by 'scale'. For a heterogeneous medium
    (grid_idx >= 0), sigma_a/sigma_s are the PER-UNIT-DENSITY coefficients —
    actual extinction at a point is density(point) * (sigma_a + sigma_s), see
    Grid_C/grid_sample_density below.
    g = Henyey-Greenstein anisotropy in [-1, 1]; 0 = isotropic.
    """
    var sigma_a: RGB   # absorption coefficient (1/m), per unit density if grid_idx >= 0
    var sigma_s: RGB   # scattering coefficient (1/m), per unit density if grid_idx >= 0
    var g:       Float32           # HG anisotropy
    var grid_idx: Int32            # -1 = homogeneous; >=0 = index into scene.grids (dense "uniformgrid")
    var nvdb_idx: Int32            # -1 = none; >=0 = index into scene.nvdb_grids (sparse "nanovdb")
    # Emissive volumes (pbrt NanoVDBMedium): a SECOND nanovdb grid, named
    # "temperature", read from the same file and stored as an ordinary entry in
    # the same scene.nvdb_grids array -- so it reuses NvdbGrid_C, its upload,
    # and nvdb_sample_density unchanged. -1 = not emissive. Emitted radiance at
    # a point follows pbrt exactly: temp = (grid(p) - temp_offset) *
    # temp_scale, no emission at or below 100 K, then
    # Le = le_scale * blackbody(temp).
    var nvdb_temp_idx: Int32
    var le_scale:    Float32
    var temp_offset: Float32
    var temp_scale:  Float32
    # 1 if this medium is the INTERIOR of a `Material "subsurface"` object
    # (see material_builder.mojo). Subsurface scattering is rendered as a
    # plain random walk through this medium behind a dielectric boundary, so
    # the only thing that distinguishes it from an ordinary participating
    # medium is bookkeeping: its scattering events are interior random-walk
    # steps, not path bounces, and must NOT be charged to the path's
    # maxdepth budget. Skin1 at the scale sssdragon uses is ~37-50 extinction
    # events per scene unit with a red-channel albedo of 0.996, so a walk
    # routinely takes tens to hundreds of steps -- against pbrt's default
    # maxdepth of 5 the object would render nearly black. pbrt has the same
    # property for a different reason: its BSSRDF resolves the whole interior
    # analytically and never spends path depth on it either.
    var is_sss:      Int32

# Extra loop rounds a renderer must allow when the scene contains a subsurface
# interior (`Medium_C.is_sss` above). Those interior random-walk steps and the
# boundary crossings bracketing them are ONE BSSRDF event and are deliberately
# NOT charged to the path's maxdepth, so the depth budget alone would never
# end the walk -- a dense preset like Skin1 needs tens to hundreds of steps
# before a path escapes or is absorbed. This is the safety bound that does.
# Lives here, beside the flag it exists for, so the path tracer
# (rendering.mojo) and SPPM (sppm.mojo) share ONE number.
comptime SSS_WALK_ROUNDS: Int = 256


# ── Homogeneous-medium free-flight sampling ───────────────────────────────────
# Shared by BDPT, SPPM and the plain path tracer (CPU + GPU, gpu.mojo's
# _sample_medium_core) for the achromatic-decision, homogeneous case (glass-
# of-water / volumetric-caustic scenes) — no delta-tracking needed. Lives here
# (not in a higher-level integrator file) precisely so gpu.mojo can reach it
# too: geometry.mojo already sits below every integrator module and already
# imports spectrum.mojo (for SampledWavelengths/SpectralSample) and defines
# Medium_C/RGB, so it is the one place all three consumers can import from
# without a circular dependency (sppm.mojo imports gpu.mojo, so gpu.mojo can
# never import sppm.mojo's functions directly).
#
# Moved here 2026-09-10 from sppm.mojo (project_elegance_backlog_2026_09_10
# item 1) as part of collapsing three historically independent
# implementations of this same physical operation down to a smaller shared
# core. Full unification turned out to be a partial win, not a total one:
# BDPT and SPPM were ALREADY calling this exact function (bdpt.mojo imports
# it from sppm.mojo) by the time this pass started, so only the free-flight-
# DISTANCE sampling and the chromatic transmittance-RATIO math needed
# extracting for gpu.mojo to share too (see medium_transmittance_ratio_spectral
# below, and _sample_medium_core's own call site). gpu.mojo's homogeneous
# branch keeps its OWN scatter/absorb decision: it plays a physical
# Russian-roulette coin against sigma_s.r/sigma_t.r and terminates the path
# outright on absorption, whereas this struct's `weight`/`albedo` fields
# instead let BDPT/SPPM continue deterministically with throughput scaled by
# the exact albedo — two different, individually unbiased estimators of the
# same integral, not two buggy copies of one, so they were deliberately left
# separate rather than forced through one return shape. See
# docs/09_volumetric_media.md, "One shared free-flight core".
@fieldwise_init
struct FreeFlight(TrivialRegisterPassable):
    """Result of sampling a free-flight distance through a medium by its
    red/hero-wavelength extinction coefficient (the same "sample by one
    channel, let the rest cancel analytically" convention used throughout
    gonzales's spectral MIS). Produced by `sample_free_flight`, which picks
    the homogeneous closed form or heterogeneous delta tracking; the fields
    below mean the same thing either way, so a caller needs no branch."""
    var collided: Bool
    var t_free: Float32     # sampled distance (meaningful either way)
    var sig_t:   Float32    # red-channel extinction ACTUALLY IN FORCE at the sampled
                            # point: sigma_a.r + sigma_s.r for a homogeneous medium,
                            # density(x) * that for a heterogeneous one -- so a caller's
                            # `sig_t * exp(-sig_t * t_free)` pdf stays meaningful in both
                            # cases and reduces exactly to the old expression at density 1.
    var pdf:     Float32    # the density the outcome was ACTUALLY drawn from, and the
                            # single source of truth for every weight (all of which are
                            # f/pdf). Collided: a density, 1/length. Pass-through: the
                            # survival PROBABILITY, dimensionless. Under hero-wavelength
                            # MIS this is the uniform MIXTURE over the sampleable lanes
                            # (free_flight_mixture_pdf), NOT sig_t's lone exponential --
                            # which is exactly why it has to be carried rather than
                            # reconstructed by a caller from sig_t. A caller that
                            # reconstructs is silently wrong the moment the medium is
                            # chromatic; use this field.
    var albedo:  RGB        # single-scattering albedo at the collision point (only if collided)
    var weight:  RGB        # per-channel chromatic WEIGHT, meaningful in BOTH branches.
                            # A pure RATIO to the sampled (red) channel, never a raw
                            # Beer-Lambert factor -- exactly 1 on red, and exactly 1 on
                            # every channel for a grey medium. Callers must multiply it
                            # into beta/flux in BOTH branches. Heterogeneous media leave
                            # it at 1: their majorant/accept-reject decisions are
                            # red-channel-only, so there is no per-channel ratio to carry
                            # (real chromatic extinction in a density field is the same
                            # separate, unimplemented piece of work the path tracer's own
                            # heterogeneous branch documents).
    var emission: SpectralSample  # in-scattered volumetric EMISSION accumulated along the
                            # tracked segment (pbrt NanoVDBMedium's temperature grid),
                            # already spectral and already weighted by each candidate's
                            # absorption fraction -- the caller multiplies by its own
                            # throughput and adds. Zero for every non-emissive medium and
                            # for the whole homogeneous branch.

@always_inline
@always_inline
def _ff_lane(sig: SpectralSample, i: Int) -> Float32:
    """Lane `i` of a 4-lane extinction sample, by index (SpectralSample has no
    subscript)."""
    if i == 0: return sig.v0
    if i == 1: return sig.v1
    if i == 2: return sig.v2
    return sig.v3


@always_inline
def free_flight_mixture_pdf(sig: SpectralSample, n: Int, t: Float32, collided: Bool) -> Float32:
    """The density a free-flight distance drawn under hero-wavelength MIS was
    actually sampled from: the UNIFORM MIXTURE p_bar = (1/|A|) * sum_{j in A} p_j
    over the sampleable lanes A = {j < n : sigma_j > 0}.

    Choosing a lane uniformly and then sampling that lane's exponential IS
    sampling from this mixture, so `f_i / p_bar` is the single-sample
    balance-heuristic MIS estimator for lane i -- and it is bounded by |A|,
    where the old ratio-to-red `f_i / p_red` was unbounded. That bound is the
    entire point: a subsurface walk multiplies hundreds of these together, and
    a product of hundreds of mean-1 UNBOUNDED factors is log-normal, which is
    the firefly distribution that made `head.pbrt` diverge
    (project_pt_sss_energy_amplification).

    Lanes with sigma_j == 0 are excluded from A -- they cannot be sampled from
    (infinite mean free path) -- but they are still WEIGHTED by the caller,
    which stays unbiased because p_bar > 0 wherever any f_i > 0.

    n == 1 recovers the old behaviour EXACTLY: A = {red}, p_bar = p_red, and
    every weight collapses to the ratio-to-red this replaced. So the previous
    estimator is literally the one-lane case of this one, not a separate path."""
    var acc = Float32(0.0)
    var cnt = 0
    for i in range(n):
        var sj = _ff_lane(sig, i)
        if sj <= Float32(0.0):
            continue
        cnt += 1
        if collided:
            acc += sj * exp(-sj * t)
        else:
            acc += exp(-sj * t)
    if cnt == 0:
        # No lane can extinguish: nothing collides, everything survives intact.
        return Float32(0.0) if collided else Float32(1.0)
    return acc / Float32(cnt)


@always_inline
def _ff_pick_lane(sig: SpectralSample, n: Int, mut pcg: PCG32) -> Float32:
    """Uniformly choose one sampleable lane's extinction, or 0 if none is."""
    var cnt = 0
    for i in range(n):
        if _ff_lane(sig, i) > Float32(0.0):
            cnt += 1
    if cnt == 0:
        return Float32(0.0)
    if cnt == 1:
        # Draw NOTHING when there is no choice to make. Consuming a variate
        # here would shift the PCG stream for every existing caller, so the
        # n == 1 path would stop being bit-identical to the pre-MIS sampler
        # it is supposed to reduce to -- which is exactly what
        # test_free_flight_t_free_matches_closed_form_inversion_formula
        # caught.
        for i in range(n):
            var sj = _ff_lane(sig, i)
            if sj > Float32(0.0):
                return sj
        return Float32(0.0)
    var k = Int(pcg.next_float() * Float32(cnt))
    if k >= cnt: k = cnt - 1        # guard the u == 1.0 endpoint
    var seen = 0
    for i in range(n):
        var sj = _ff_lane(sig, i)
        if sj <= Float32(0.0):
            continue
        if seen == k:
            return sj
        seen += 1
    return Float32(0.0)


def sample_homogeneous_free_flight(
    med: Medium_C, t_surf: Float32, mut pcg: PCG32,
    # Hero-wavelength MIS. Supply these and the free flight is drawn from the
    # uniform MIXTURE over the 4 hero lanes' exponentials instead of from red
    # alone, which bounds every resulting weight by the lane count. Omit them
    # and `lane_sig` stays a single red lane, i.e. EXACTLY the old estimator --
    # correctness never depends on the plumbing, only variance does. Whatever
    # was used is recorded in FreeFlight.pdf, so a weight computed later as
    # f/ff.pdf can never disagree with what was actually sampled.
    lane_sig: SpectralSample = SpectralSample(Float32(0.0)),
    lane_n: Int = 0,
) -> FreeFlight:
    var sigma_t = med.sigma_a + med.sigma_s
    var sig_t = sigma_t.r
    if sig_t <= Float32(0.0):
        return FreeFlight(False, t_surf, sig_t, Float32(1.0), RGB(Float32(0)), RGB(Float32(1)), SpectralSample(Float32(0)))
    # n == 1 with lane 0 = red reproduces the pre-MIS sampler bit-for-bit.
    var sig = lane_sig if lane_n > 0 else SpectralSample(sig_t, Float32(0), Float32(0), Float32(0))
    var n = lane_n if lane_n > 0 else 1
    var sig_k = _ff_pick_lane(sig, n, pcg)
    if sig_k <= Float32(0.0):
        sig_k = sig_t
    var t_free = -log(max(pcg.next_float(), Float32(1e-7))) / sig_k
    if t_free < t_surf:
        var alb_s = med.sigma_s.r / sig_t
        var alb_g_s = med.sigma_s.g / sigma_t.g if sigma_t.g > Float32(0.0) else alb_s
        var alb_b_s = med.sigma_s.b / sigma_t.b if sigma_t.b > Float32(0.0) else alb_s
        # COLLISION WEIGHT. The distance was sampled from red's exponential,
        # pdf(t) = sigma_t.r * exp(-sigma_t.r * t), so lane c's contribution
        # sigma_s_c * exp(-sigma_t_c * t) needs
        #     sigma_s_c*exp(-sigma_t_c*t) / (sigma_t.r*exp(-sigma_t.r*t))
        # and `albedo` above already carries sigma_s_c/sigma_t_c, leaving
        #     exp(-(sigma_t_c - sigma_t.r)*t) * sigma_t_c/sigma_t.r.
        # (Algebraically identical to gpu.mojo's albedo_r * sigma_s_c/sigma_s.r
        # form, just factored to keep `albedo` per-channel here.)
        #
        # This weight was MISSING entirely: the collision branch returned a
        # flat RGB(1), so VCM/SPPM were chromatically BIASED in any medium
        # whose channels differ -- a red-heavy medium lost exactly the green
        # and blue extinction ratio at every scattering event, compounding
        # per bounce. gpu.mojo's _sample_medium_core has always applied it
        # (its comment spells out both factors); it simply never propagated
        # to this shared BDPT/SPPM sampler -- the same fixed-in-one-consumer
        # split that accounts for most defects in this codebase. Exactly 1 on
        # every channel for a grey medium, so grey renders are unaffected.
        # Both the RGB weight here and the spectral one in
        # spectral_free_flight_weight are now the SAME estimator, f/p_bar,
        # differing only in which basis f is evaluated on. p_bar reduces to
        # sigma_t.r*exp(-sigma_t.r*t) when no lanes were supplied, which makes
        # each of these exactly the ratio-to-red expression it replaced.
        var p_bar = free_flight_mixture_pdf(sig, n, t_free, True)
        if p_bar < Float32(1e-30): p_bar = Float32(1e-30)
        var wr = sigma_t.r * exp(-sigma_t.r * t_free) / p_bar
        var wg = (sigma_t.g * exp(-sigma_t.g * t_free) / p_bar
                  if sigma_t.g > Float32(0.0) else Float32(1.0))
        var wb = (sigma_t.b * exp(-sigma_t.b * t_free) / p_bar
                  if sigma_t.b > Float32(0.0) else Float32(1.0))
        return FreeFlight(True, t_free, sig_t, p_bar,
                                     RGB(alb_s, alb_g_s, alb_b_s),
                                     RGB(wr, wg, wb),
                                     SpectralSample(Float32(0)))
    # Pass-through WEIGHT, not the raw Beer-Lambert factor. The distance was
    # sampled from the red channel, so P(reach the surface) is ALREADY
    # exp(-sigma_t.r * t_surf) -- that channel's transmittance is carried by
    # the sampling probability itself. Only the RATIO of each channel's
    # transmittance to the sampled one survives as a weight:
    #     exp(-sigma_t_c * t) / exp(-sigma_t.r * t) = exp(-(sigma_t_c - sigma_t.r) * t)
    # which is exactly 1 on red, and exactly 1 on every channel for a grey
    # medium.
    #
    # This used to return the FULL exp(-sigma_t_c * t) on all three channels,
    # which callers multiply into beta/flux -- double-counting the sampled
    # channel's transmittance, so the expected contribution of a surface seen
    # through a medium was exp(-2*sigma_r*t) where it should be
    # exp(-sigma_r*t). Surfaces behind a medium came out too dark by exactly
    # the transmittance, and unlike the chromatic terms this bit GREY media
    # too. gpu.mojo's _sample_medium_core has always used the ratio (its own
    # comment records the same fix); it simply never propagated to this
    # BDPT/SPPM sampler.
    var p_bar = free_flight_mixture_pdf(sig, n, t_surf, False)
    if p_bar < Float32(1e-30): p_bar = Float32(1e-30)
    var Tr = RGB(exp(-sigma_t.r * t_surf) / p_bar,
                 exp(-sigma_t.g * t_surf) / p_bar,
                 exp(-sigma_t.b * t_surf) / p_bar)
    return FreeFlight(False, t_free, sig_t, p_bar, RGB(Float32(0)), Tr, SpectralSample(Float32(0)))


# ── Genuinely spectral free-flight weight (chromatic media) ─────────────────
# `FreeFlight.weight` above is a RATIO computed entirely in RGB
# (exp(-sigma_t.g*t), exp(-sigma_t.b*t), ...) and every caller used to lift it
# into the 4 hero lanes by BAND-PICKING that already-exponentiated triple --
# i.e. selecting, per lane, which of the 3 RGB ratios to read. Band-picking a
# genuine coefficient (see rgb_bands_to_spectral_sample's own docstring, "a
# coefficient is not a colour") is fine on its own, but here it throws away
# information for free: an extinction coefficient IS spectral data, and
# gonzales already has a real RGB->spectrum upsampler for exactly this shape
# of quantity (spec_refl_unbounded, used elsewhere for reflectance-shaped
# weights that may exceed 1 -- an extinction coefficient is the same kind of
# thing). The two orders are NOT interchangeable for a smooth upsampler:
# upsample(exp(-sigma*t)) != exp(-upsample(sigma)*t) in general (they
# coincide only for band-picking, a pure per-lane selection, since selection
# commutes with exp). Verified 2026-09-10 that the smooth upsampler's grey
# invariant survives exactly (spec_refl_unbounded(c,c,c) -> c in every lane,
# to machine precision) — see project_spectral_media_state.md,
# "Refutation 2" — so nothing is lost by switching sigma_t's upsampling from
# band-picking to the real curve; a grey medium is bit-for-bit unaffected.
#
# This computes the weight the CORRECT way: upsample sigma_t to per-lane
# values FIRST, then exponentiate PER LANE, using the same red-channel
# proposal density `sample_homogeneous_free_flight` already sampled `t_free`
# from (so this is purely a change to the WEIGHT, never to what distance was
# sampled or its acceptance probability -- no change to path continuation).
@always_inline
def medium_sigma_t_spectral(
    med: Medium_C, wavelengths: SampledWavelengths,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
) -> SpectralSample:
    """Upsample a medium's total extinction sigma_t = sigma_a + sigma_s from
    its 3 authored RGB channels to the 4 hero wavelengths via
    spec_refl_unbounded -- the same unbounded-coefficient upsampler used for
    reflectance-shaped weights that may exceed 1 (an extinction coefficient
    is the same kind of quantity). Shared by spectral_free_flight_weight,
    bdpt.mojo's _visible_transmittance and gpu.mojo's _sample_medium_core
    (via medium_transmittance_ratio_spectral below) so all consumers agree on
    what "the medium's colour" means."""
    var sig_t = med.sigma_a + med.sigma_s
    return spec_refl_unbounded(
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
        sig_t.r, sig_t.g, sig_t.b, wavelengths)

def medium_sigma_s_spectral(
    med: Medium_C, wavelengths: SampledWavelengths,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
) -> SpectralSample:
    """The SCATTERING coefficient sigma_s on the 4 hero lanes, upsampled with
    the SAME smooth curve medium_sigma_t_spectral uses for sigma_t.

    Using one conversion for sigma_t and a different one for sigma_s (this
    used to be band-picked from RGB in gpu.mojo) makes the per-lane single-
    scattering albedo sigma_s(lambda)/sigma_t(lambda) a ratio of two
    INCONSISTENT quantities. Harmless at two or three scatters, catastrophic
    in a subsurface walk: skin's albedo is ~0.99 and the walk runs hundreds
    of scatters, so a 1% per-lane albedo error compounds as 0.99^250 ~ 12x
    and inverts the hue. head.pbrt rendered BLUE-dominant (chromaticity
    b .406 against pbrt's .270, red 0.49x) until sigma_s came through here.

    Rule: whenever a per-lane RATIO of two medium coefficients is formed,
    both sides must come from the same upsampler."""
    # NOT an independent unbounded fit of sigma_s. sigma_s and sigma_t fitted
    # SEPARATELY are two different curves, and nothing makes the second stay
    # above the first, so their ratio -- the single-scattering albedo -- can
    # come out GREATER THAN 1 on some lanes. That is an energy-creating
    # medium, and a subsurface walk compounds it over hundreds of scatters:
    # measured on head.pbrt, 35% of pixels blown out and a max of 1.7e31.
    #
    # Upsample the ALBEDO instead, with the BOUNDED upsampler that clamps to
    # [0,1] (an albedo is exactly the reflectance-shaped quantity spec_refl
    # exists for), and rebuild sigma_s from it. sigma_s(lambda) <= sigma_t(lambda)
    # then holds on every lane by construction.
    var sig_t_rgb = med.sigma_a + med.sigma_s
    var alb_r = med.sigma_s.r / sig_t_rgb.r if sig_t_rgb.r > Float32(0.0) else Float32(0.0)
    var alb_g = med.sigma_s.g / sig_t_rgb.g if sig_t_rgb.g > Float32(0.0) else Float32(0.0)
    var alb_b = med.sigma_s.b / sig_t_rgb.b if sig_t_rgb.b > Float32(0.0) else Float32(0.0)
    var alb = spec_refl(
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
        alb_r, alb_g, alb_b, wavelengths)
    var sig_t = medium_sigma_t_spectral(
        med, wavelengths, spectral_coeffs, spectral_res,
        spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
    return SpectralSample(alb.v0 * sig_t.v0, alb.v1 * sig_t.v1,
                          alb.v2 * sig_t.v2, alb.v3 * sig_t.v3)

@always_inline
def medium_transmittance_ratio_spectral(
    med: Medium_C, t: Float32, pdf: Float32, wavelengths: SampledWavelengths,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
) -> SpectralSample:
    """The chromatic RATIO of each hero wavelength's transmittance over a
    homogeneous segment of length `t` to the density it was actually SAMPLED
    from, `pdf` (pass FreeFlight.pdf): exp(-sigma_t(lambda_i) * t) / pdf, with
    sigma_t upsampled to the 4 hero lanes FIRST (medium_sigma_t_spectral) then
    exponentiated PER LANE --
    exactly the `d0..d3` half of spectral_free_flight_weight below, pulled
    out because gpu.mojo's _sample_medium_core needs precisely this ratio
    (and nothing else -- it applies its own sigma_s/albedo factor separately,
    via a real scatter/absorb coin flip rather than a deterministic
    multiply) and can share this ordering-sensitive arithmetic without also
    taking on BDPT/SPPM's deterministic-continuation collision weight. Never
    fold a sigma_s/albedo factor into THIS function -- see the two callers'
    own docstrings for why they apply it differently."""
    var sig_t_r = med.sigma_a.r + med.sigma_s.r
    if sig_t_r <= Float32(0.0):
        return SpectralSample(Float32(1.0))
    var sig_t_spec = medium_sigma_t_spectral(
        med, wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
    var p = pdf
    if p < Float32(1e-30): p = Float32(1e-30)
    return SpectralSample(
        exp(-sig_t_spec.v0 * t) / p,
        exp(-sig_t_spec.v1 * t) / p,
        exp(-sig_t_spec.v2 * t) / p,
        exp(-sig_t_spec.v3 * t) / p)

@always_inline
def spectral_free_flight_weight(
    med: Medium_C, ff: FreeFlight, t_surf: Float32, wavelengths: SampledWavelengths,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
) -> SpectralSample:
    """Replaces `rgb_bands_to_spectral_sample(ff.weight.r, .g, .b, wl)` at
    every chromatic-media consumer. `ff` must come from
    `sample_homogeneous_free_flight(med, t_surf, ...)` -- SAME `med` and
    `t_surf` -- (this does not re-sample anything, it only re-derives the
    weight spectrally). `t_surf` is required explicitly, not read from
    `ff.t_free`: in the pass-through branch `ff.t_free` is the raw sampled
    distance (which exceeded `t_surf`, that's WHY it's pass-through), while
    the weight must use the actual segment length `t_surf` -- the same
    distinction `sample_homogeneous_free_flight`'s own RGB `Tr` makes
    (built from `t_surf`, not `t_free`, in that branch).
    Grey media pass through exactly, at every tau, since spec_refl_unbounded
    reproduces a grey coefficient exactly and the sig_t_r reference cancels
    to 1 on every lane. See docs/02_spectra_and_color.md, "Chromatic
    extinction" for the derivation and the measured before/after."""
    var sig_t_r = med.sigma_a.r + med.sigma_s.r
    if sig_t_r <= Float32(0.0):
        return SpectralSample(Float32(1.0))
    # A density-modulated medium has NO chromatic ratio to carry: its free
    # flight accepts/rejects on the red channel alone, so `ff.weight` comes
    # back at 1 and the correct spectral weight is 1 too. Guarding HERE rather
    # than at all five consumers (bdpt.mojo x4, sppm.mojo x1) keeps the rule in
    # the one place that owns it -- and it matters: sigma_a/sigma_s are
    # PER-UNIT-DENSITY coefficients for such a medium (see Medium_C), so
    # feeding them to the Beer-Lambert exponential below would weight by an
    # extinction the sampler never used. Real chromatic extinction in a density
    # field is the same separate, unimplemented piece of work the path tracer's
    # own heterogeneous branch documents.
    if medium_is_heterogeneous(med):
        return SpectralSample(Float32(1.0))
    # sig_t_spec computed ONCE and reused for both the d0..d3 ratio and (on
    # collision) the extra r0..r3 factor -- deliberately NOT routed through
    # medium_transmittance_ratio_spectral, which would upsample sigma_t a
    # second time on the collision branch (that helper exists for gpu.mojo's
    # simpler case, which never needs r0..r3).
    var sig_t_spec = medium_sigma_t_spectral(
        med, wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
    var t = ff.t_free if ff.collided else t_surf
    # Single-sample MIS estimator f_i / p_bar, where p_bar is the density the
    # distance was ACTUALLY drawn from. Read it off `ff` rather than rederiving
    # it: BDPT and SPPM call sample_free_flight without the lane arguments, so
    # a locally recomputed 4-lane mixture would silently disagree with the
    # 1-lane one the sampler used and bias every chromatic medium. Taking it
    # from `ff` makes the two impossible to desynchronise -- richer plumbing
    # then buys variance, never correctness.
    var p_bar = ff.pdf
    if p_bar < Float32(1e-30): p_bar = Float32(1e-30)
    if not ff.collided:
        return SpectralSample(exp(-sig_t_spec.v0 * t) / p_bar,
                              exp(-sig_t_spec.v1 * t) / p_bar,
                              exp(-sig_t_spec.v2 * t) / p_bar,
                              exp(-sig_t_spec.v3 * t) / p_bar)
    # Collision: f_i = sigma_t(lambda_i) * exp(-sigma_t(lambda_i) * t). The
    # caller applies `ff.albedo` (sigma_s/sigma_t) separately, so the product
    # is the physical sigma_s(lambda_i)*exp(-sigma_t(lambda_i)*t) / p_bar.
    return SpectralSample(sig_t_spec.v0 * exp(-sig_t_spec.v0 * t) / p_bar,
                          sig_t_spec.v1 * exp(-sig_t_spec.v1 * t) / p_bar,
                          sig_t_spec.v2 * exp(-sig_t_spec.v2 * t) / p_bar,
                          sig_t_spec.v3 * exp(-sig_t_spec.v3 * t) / p_bar)


@fieldwise_init
struct Grid_C(TrivialRegisterPassable):
    """Dense heterogeneous density grid backing a Medium_C (PBRT-v4
    "uniformgrid" — a flat array of nx*ny*nz density samples spanning the
    axis-aligned box [p0,p1] in the medium's own local space). density(x) is
    a unitless multiplier: real sigma_t at a world point = grid_sample_density(x)
    * (medium.sigma_a + medium.sigma_s). world_to_medium maps a world-space
    point into that local [p0,p1]-space box (same 4x4 column-major convention
    as transform.mojo's transform_points: m[0],m[4],m[8],m[12] combine for x).
    max_density is the majorant used for delta-tracking free-flight sampling.
    """
    var density: Pointer[Float32, MutUntrackedOrigin]  # flat, nz-major: idx = (z*ny + y)*nx + x
    var nx: Int32
    var ny: Int32
    var nz: Int32
    var p0: Point3f
    var p1: Point3f
    var world_to_medium: SIMD[DType.float32, 16]
    var max_density: Float32

@always_inline
def _grid_density_at(grid: Grid_C, xi: Int, yi: Int, zi: Int) -> Float32:
    var cx = max(0, min(Int(grid.nx) - 1, xi))
    var cy = max(0, min(Int(grid.ny) - 1, yi))
    var cz = max(0, min(Int(grid.nz) - 1, zi))
    return grid.density[unsafe_offset=(cz * Int(grid.ny) + cy) * Int(grid.nx) + cx]

@always_inline
def grid_sample_density(grid: Grid_C, p_world: Vec3f) -> Float32:
    """Trilinearly-interpolated density at a world-space point; 0 outside
    the grid's local bounds [p0,p1]."""
    var m = grid.world_to_medium
    var px = m[0]*p_world[0] + m[4]*p_world[1] + m[8]*p_world[2] + m[12]
    var py = m[1]*p_world[0] + m[5]*p_world[1] + m[9]*p_world[2] + m[13]
    var pz = m[2]*p_world[0] + m[6]*p_world[1] + m[10]*p_world[2] + m[14]

    var ext_x = grid.p1.x - grid.p0.x
    var ext_y = grid.p1.y - grid.p0.y
    var ext_z = grid.p1.z - grid.p0.z
    if ext_x <= Float32(0.0) or ext_y <= Float32(0.0) or ext_z <= Float32(0.0):
        return Float32(0.0)
    var u = (px - grid.p0.x) / ext_x
    var v = (py - grid.p0.y) / ext_y
    var w = (pz - grid.p0.z) / ext_z
    if u < Float32(0.0) or u > Float32(1.0) or v < Float32(0.0) or v > Float32(1.0) or w < Float32(0.0) or w > Float32(1.0):
        return Float32(0.0)

    # Continuous voxel coords (PBRT convention: sample centers at half-integer
    # offsets, so u=0..1 maps to [-0.5, nx-0.5]).
    var gx = u * Float32(grid.nx) - Float32(0.5)
    var gy = v * Float32(grid.ny) - Float32(0.5)
    var gz = w * Float32(grid.nz) - Float32(0.5)
    var x0 = Int(gx); var y0 = Int(gy); var z0 = Int(gz)
    var fx = gx - Float32(x0); var fy = gy - Float32(y0); var fz = gz - Float32(z0)
    if gx < Float32(0.0): x0 -= 1; fx = gx - Float32(x0)
    if gy < Float32(0.0): y0 -= 1; fy = gy - Float32(y0)
    if gz < Float32(0.0): z0 -= 1; fz = gz - Float32(z0)

    var d00 = _grid_density_at(grid, x0, y0, z0)     * (Float32(1.0) - fx) + _grid_density_at(grid, x0+1, y0, z0)     * fx
    var d10 = _grid_density_at(grid, x0, y0+1, z0)   * (Float32(1.0) - fx) + _grid_density_at(grid, x0+1, y0+1, z0)   * fx
    var d01 = _grid_density_at(grid, x0, y0, z0+1)   * (Float32(1.0) - fx) + _grid_density_at(grid, x0+1, y0, z0+1)   * fx
    var d11 = _grid_density_at(grid, x0, y0+1, z0+1) * (Float32(1.0) - fx) + _grid_density_at(grid, x0+1, y0+1, z0+1) * fx
    var d0 = d00 * (Float32(1.0) - fy) + d10 * fy
    var d1 = d01 * (Float32(1.0) - fy) + d11 * fy
    return d0 * (Float32(1.0) - fz) + d1 * fz

@fieldwise_init
struct NvdbGrid_C(TrivialRegisterPassable):
    """Sparse heterogeneous density grid backing a Medium_C (PBRT-v4
    "nanovdb" -- a decompressed .nvdb blob, sampled via nvdb_sample_index).
    Sibling to Grid_C, not a variant of it: NanoVDB's own index space has a
    DIFFERENT coordinate convention (integer voxel index, own affine map)
    from Grid_C's dense-array [p0,p1]-box convention, so this is its own
    struct rather than a `kind` flag bolted onto Grid_C.

    Two transforms compose to go from pbrt world space to an nvdb voxel
    index, mirroring the two-stage pipeline pbrt-v4 itself uses for nanovdb
    media: world_to_medium (this medium's own pbrt CTM inverse, same 4x4
    column-major convention as Grid_C's field of the same name) maps pbrt
    world space into the .nvdb file's OWN embedded world space, and
    inv_map/map_vec (that file's PNanoVDB "Map", read once from the blob
    at parse time) maps that into fractional index space:
        world_to_index(x) = inv_map * (x - map_vec)
    (matches pnanovdb_map_apply_inverse exactly; inv_map is row-major 3x3,
    packed into a SIMD16 with 7 unused padding lanes to reuse Grid_C's own
    storage convention rather than invent a 9-wide one).

    index_min/index_max are the blob's indexBBox (nvdb_index_bbox), for a
    cheap reject before touching the blob at all. max_density is the
    majorant (root-node max, nvdb_value_range) used for delta-tracking free
    -flight sampling, same role as Grid_C.max_density -- coarser than a
    per-leaf majorant would be, a documented, deliberate v1 scope choice.
    """
    var blob: Pointer[UInt8, MutUntrackedOrigin]
    var blob_size: Int64  # bytes -- CPU sampling never needs this (pure offset
                           # arithmetic, no bounds check), only the GPU upload's
                           # memcpy does; kept here rather than threaded as a
                           # separate parallel array alongside every other field.
    var world_to_medium: SIMD[DType.float32, 16]
    var inv_map: SIMD[DType.float32, 16]  # row-major 3x3 in lanes 0..8, rest unused
    var map_vec: Vec3f
    var index_min: Point3f
    var index_max: Point3f
    var max_density: Float32

@always_inline
def nvdb_sample_density(grid: NvdbGrid_C, p_world: Vec3f) -> Float32:
    """Point-sampled (NOT trilinear -- v1 scope, see project_nanovdb_media
    memory) density at a world-space point; 0 outside the grid's index
    bounds. Same "0 outside bounds" contract as grid_sample_density, so a
    shadow ray's ratio-tracking transmittance naturally stops attenuating
    once it exits the medium with no extra bookkeeping, exactly as that
    function's own docstring notes."""
    var m = grid.world_to_medium
    var mx = m[0]*p_world[0] + m[4]*p_world[1] + m[8]*p_world[2] + m[12]
    var my = m[1]*p_world[0] + m[5]*p_world[1] + m[9]*p_world[2] + m[13]
    var mz = m[2]*p_world[0] + m[6]*p_world[1] + m[10]*p_world[2] + m[14]

    var im = grid.inv_map
    var sx = mx - grid.map_vec.x
    var sy = my - grid.map_vec.y
    var sz = mz - grid.map_vec.z
    var ix = sx*im[0] + sy*im[1] + sz*im[2]
    var iy = sx*im[3] + sy*im[4] + sz*im[5]
    var iz = sx*im[6] + sy*im[7] + sz*im[8]

    # Trilinear, matching pbrt's NanoVDBMedium, which samples density with
    # nanovdb's SampleFromVoxels<..., 1, false> (order 1 = trilinear). NanoVDB's
    # convention places voxel VALUES at integer index coordinates, so the base
    # cell is floor(p) and the weights are the fractional part -- this is NOT
    # the half-integer cell-centre convention Grid_C's dense sampler uses, and
    # getting that wrong shifts the field by half a voxel.
    #
    # Point sampling (the original v1 scope) is not merely noisier: on a sparse
    # wispy cloud it keeps hard voxel edges where pbrt's trilinear bleeds each
    # occupied voxel into its neighbours, which changes the effective optical
    # thickness and rendered a visibly dimmer cloud than the reference.
    var fi = floor(ix); var fj = floor(iy); var fk = floor(iz)
    var i = Int32(fi); var j = Int32(fj); var k = Int32(fk)
    # Reject only when the whole 8-tap neighbourhood lies outside the grid;
    # taps that fall outside individually just read the background (0).
    if i < Int32(grid.index_min.x) - Int32(1) or i > Int32(grid.index_max.x): return Float32(0.0)
    if j < Int32(grid.index_min.y) - Int32(1) or j > Int32(grid.index_max.y): return Float32(0.0)
    if k < Int32(grid.index_min.z) - Int32(1) or k > Int32(grid.index_max.z): return Float32(0.0)
    var tx = ix - fi; var ty = iy - fj; var tz = iz - fk
    var v000: Float32; var v100: Float32; var v010: Float32; var v110: Float32
    var v001: Float32; var v101: Float32; var v011: Float32; var v111: Float32
    # Fast path: when the 2x2x2 stencil does not cross a leaf boundary (the
    # low 3 bits of every axis are < 7), one root->leaf descent locates the
    # leaf and the other seven values are pure offset arithmetic inside it --
    # 1 tree walk instead of 8. A leaf is 8^3, so this covers (7/8)^3 ~ 67%
    # of lookups; the rest fall back to eight independent descents, which is
    # exactly what this function did before. Both paths return identical
    # values by construction (the slow path would descend to this same leaf).
    var leaf_base = -1
    if (i & Int32(7)) < Int32(7) and (j & Int32(7)) < Int32(7) and (k & Int32(7)) < Int32(7):
        leaf_base = nvdb_leaf_base(grid.blob, i, j, k)
    if leaf_base >= 0:
        v000 = nvdb_leaf_value(grid.blob, leaf_base, i,           j,           k)
        v100 = nvdb_leaf_value(grid.blob, leaf_base, i + Int32(1), j,           k)
        v010 = nvdb_leaf_value(grid.blob, leaf_base, i,           j + Int32(1), k)
        v110 = nvdb_leaf_value(grid.blob, leaf_base, i + Int32(1), j + Int32(1), k)
        v001 = nvdb_leaf_value(grid.blob, leaf_base, i,           j,           k + Int32(1))
        v101 = nvdb_leaf_value(grid.blob, leaf_base, i + Int32(1), j,           k + Int32(1))
        v011 = nvdb_leaf_value(grid.blob, leaf_base, i,           j + Int32(1), k + Int32(1))
        v111 = nvdb_leaf_value(grid.blob, leaf_base, i + Int32(1), j + Int32(1), k + Int32(1))
    else:
        v000 = nvdb_sample_index(grid.blob, i,           j,           k)
        v100 = nvdb_sample_index(grid.blob, i + Int32(1), j,           k)
        v010 = nvdb_sample_index(grid.blob, i,           j + Int32(1), k)
        v110 = nvdb_sample_index(grid.blob, i + Int32(1), j + Int32(1), k)
        v001 = nvdb_sample_index(grid.blob, i,           j,           k + Int32(1))
        v101 = nvdb_sample_index(grid.blob, i + Int32(1), j,           k + Int32(1))
        v011 = nvdb_sample_index(grid.blob, i,           j + Int32(1), k + Int32(1))
        v111 = nvdb_sample_index(grid.blob, i + Int32(1), j + Int32(1), k + Int32(1))
    var v00 = v000 + (v100 - v000) * tx
    var v10 = v010 + (v110 - v010) * tx
    var v01 = v001 + (v101 - v001) * tx
    var v11 = v011 + (v111 - v011) * tx
    var v0 = v00 + (v10 - v00) * ty
    var v1 = v01 + (v11 - v01) * ty
    return v0 + (v1 - v0) * tz

@always_inline
def _slab_range(o: Vec3f, d: Vec3f, bmin: Vec3f, bmax: Vec3f) -> SIMD[DType.float32, 2]:
    """Ray/AABB slab test returning (t_enter, t_exit); t_exit < t_enter means
    "misses the box". The ray is NOT normalized here on purpose: `d` is passed
    in the SAME space as the box, transformed by the same affine map as `o`,
    so the returned t values stay in the caller's original (world) parameter
    units and can be compared directly against a world-space distance."""
    var t0 = Float32(-1.0e30)
    var t1 = Float32(1.0e30)
    for a in range(3):
        var di = d[a]
        var oi = o[a]
        var lo = bmin[a]
        var hi = bmax[a]
        if abs(di) < Float32(1e-12):
            if oi < lo or oi > hi:
                return SIMD[DType.float32, 2](Float32(1.0), Float32(-1.0))  # miss
        else:
            var inv = Float32(1.0) / di
            var ta = (lo - oi) * inv
            var tb = (hi - oi) * inv
            var tmin_a = ta if ta < tb else tb
            var tmax_a = tb if ta < tb else ta
            if tmin_a > t0: t0 = tmin_a
            if tmax_a < t1: t1 = tmax_a
    return SIMD[DType.float32, 2](t0, t1)

@always_inline
def nvdb_ray_range(grid: NvdbGrid_C, org: Vec3f, dir: Vec3f) -> SIMD[DType.float32, 2]:
    """(t_enter, t_exit) of the ray against this grid's index bbox, in WORLD t
    units. Needed because an infinite (environment) light's shadow ray has no
    finite distance to march: ratio-tracking it at the majorant step rate to a
    nominal "infinity" would burn millions of iterations through empty space
    for a transmittance that stops changing the moment the ray leaves the grid.
    Both transforms are affine, so t is preserved and the range can be
    intersected directly with the world-space shadow-ray length."""
    var m = grid.world_to_medium
    var ox = m[0]*org[0] + m[4]*org[1] + m[8]*org[2] + m[12]
    var oy = m[1]*org[0] + m[5]*org[1] + m[9]*org[2] + m[13]
    var oz = m[2]*org[0] + m[6]*org[1] + m[10]*org[2] + m[14]
    var dx = m[0]*dir[0] + m[4]*dir[1] + m[8]*dir[2]
    var dy = m[1]*dir[0] + m[5]*dir[1] + m[9]*dir[2]
    var dz = m[2]*dir[0] + m[6]*dir[1] + m[10]*dir[2]
    var im = grid.inv_map
    var sx = ox - grid.map_vec.x
    var sy = oy - grid.map_vec.y
    var sz = oz - grid.map_vec.z
    var oi = Vec3f(sx*im[0] + sy*im[1] + sz*im[2],
                   sx*im[3] + sy*im[4] + sz*im[5],
                   sx*im[6] + sy*im[7] + sz*im[8])
    var di = Vec3f(dx*im[0] + dy*im[1] + dz*im[2],
                   dx*im[3] + dy*im[4] + dz*im[5],
                   dx*im[6] + dy*im[7] + dz*im[8])
    return _slab_range(oi, di,
        Vec3f(grid.index_min.x, grid.index_min.y, grid.index_min.z),
        Vec3f(grid.index_max.x + Float32(1), grid.index_max.y + Float32(1), grid.index_max.z + Float32(1)))

@always_inline
def nvdb_index_ray(grid: NvdbGrid_C, org: Vec3f, dir: Vec3f) -> SIMD[DType.float32, 8]:
    """The world ray expressed in the grid's INDEX space, as
    (ox,oy,oz,|d|, dx,dy,dz,unused). Both transforms are affine and the
    direction is carried as a vector, so the ray parameter t is IDENTICAL in
    both spaces -- index-space box tests therefore return world-space t
    directly. Computed once per tracked ray so the per-segment majorant
    walk costs only arithmetic."""
    var m = grid.world_to_medium
    var ox = m[0]*org[0] + m[4]*org[1] + m[8]*org[2] + m[12]
    var oy = m[1]*org[0] + m[5]*org[1] + m[9]*org[2] + m[13]
    var oz = m[2]*org[0] + m[6]*org[1] + m[10]*org[2] + m[14]
    var dx = m[0]*dir[0] + m[4]*dir[1] + m[8]*dir[2]
    var dy = m[1]*dir[0] + m[5]*dir[1] + m[9]*dir[2]
    var dz = m[2]*dir[0] + m[6]*dir[1] + m[10]*dir[2]
    var im = grid.inv_map
    var sx = ox - grid.map_vec.x
    var sy = oy - grid.map_vec.y
    var sz = oz - grid.map_vec.z
    var iox = sx*im[0] + sy*im[1] + sz*im[2]
    var ioy = sx*im[3] + sy*im[4] + sz*im[5]
    var ioz = sx*im[6] + sy*im[7] + sz*im[8]
    var idx = dx*im[0] + dy*im[1] + dz*im[2]
    var idy = dx*im[3] + dy*im[4] + dz*im[5]
    var idz = dx*im[6] + dy*im[7] + dz*im[8]
    var dlen = sqrt(max(idx*idx + idy*idy + idz*idz, Float32(1e-20)))
    return SIMD[DType.float32, 8](iox, ioy, ioz, dlen, idx, idy, idz, Float32(0))

@always_inline
def nvdb_node_exit_t(iray: SIMD[DType.float32, 8], t_now: Float32, dim: Float32) -> Float32:
    """t at which the ray leaves the `dim`-aligned node box currently
    containing it. `dim` is a power of two (8/128/4096, the leaf/lower/upper
    extents), so the box base is the index coordinate with its low bits
    masked off -- an arithmetic shift, which is correct for negative
    coordinates too. A small nudge past the face is added so the next
    majorant query lands in the NEXT node and the walk always makes
    progress; without it a ray exactly on a node face would re-query the
    same node forever."""
    var px = iray[0] + t_now * iray[4]
    var py = iray[1] + t_now * iray[5]
    var pz = iray[2] + t_now * iray[6]
    var shift = Int32(3)
    if dim > Float32(2048): shift = Int32(12)
    elif dim > Float32(64): shift = Int32(7)
    var b0 = Float32((Int32(floor(px)) >> shift) << shift)
    var b1 = Float32((Int32(floor(py)) >> shift) << shift)
    var b2 = Float32((Int32(floor(pz)) >> shift) << shift)
    var t_exit = Float32(1.0e30)
    if abs(iray[4]) > Float32(1e-12):
        var bx = (b0 + dim) if iray[4] > Float32(0) else b0
        var tx = (bx - iray[0]) / iray[4]
        if tx < t_exit: t_exit = tx
    if abs(iray[5]) > Float32(1e-12):
        var by = (b1 + dim) if iray[5] > Float32(0) else b1
        var ty = (by - iray[1]) / iray[5]
        if ty < t_exit: t_exit = ty
    if abs(iray[6]) > Float32(1e-12):
        var bz = (b2 + dim) if iray[6] > Float32(0) else b2
        var tz = (bz - iray[2]) / iray[6]
        if tz < t_exit: t_exit = tz
    # nudge ~1e-3 voxel past the face
    var eps = Float32(1.0e-3) / iray[3]
    if t_exit <= t_now: t_exit = t_now
    return t_exit + eps

@always_inline
def nvdb_majorant_at_world(grid: NvdbGrid_C, p_world: Vec3f) -> SIMD[DType.float32, 2]:
    """LOCAL majorant (max, extent) at a WORLD-space point -- thin wrapper
    over nanovdb.mojo's nvdb_majorant_at that applies the same two-stage
    world->medium->index transform nvdb_sample_density uses."""
    var m = grid.world_to_medium
    var mx = m[0]*p_world[0] + m[4]*p_world[1] + m[8]*p_world[2] + m[12]
    var my = m[1]*p_world[0] + m[5]*p_world[1] + m[9]*p_world[2] + m[13]
    var mz = m[2]*p_world[0] + m[6]*p_world[1] + m[10]*p_world[2] + m[14]
    var im = grid.inv_map
    var sx = mx - grid.map_vec.x
    var sy = my - grid.map_vec.y
    var sz = mz - grid.map_vec.z
    var ix = sx*im[0] + sy*im[1] + sz*im[2]
    var iy = sx*im[3] + sy*im[4] + sz*im[5]
    var iz = sx*im[6] + sy*im[7] + sz*im[8]
    return nvdb_majorant_at(grid.blob, Int32(floor(ix)), Int32(floor(iy)), Int32(floor(iz)))

@always_inline
def grid_ray_range(grid: Grid_C, org: Vec3f, dir: Vec3f) -> SIMD[DType.float32, 2]:
    """(t_enter, t_exit) of the ray against a dense grid's [p0,p1] box, in
    WORLD t units. Same purpose as nvdb_ray_range -- see that docstring."""
    var m = grid.world_to_medium
    var o = Vec3f(m[0]*org[0] + m[4]*org[1] + m[8]*org[2] + m[12],
                  m[1]*org[0] + m[5]*org[1] + m[9]*org[2] + m[13],
                  m[2]*org[0] + m[6]*org[1] + m[10]*org[2] + m[14])
    var d = Vec3f(m[0]*dir[0] + m[4]*dir[1] + m[8]*dir[2],
                  m[1]*dir[0] + m[5]*dir[1] + m[9]*dir[2],
                  m[2]*dir[0] + m[6]*dir[1] + m[10]*dir[2])
    return _slab_range(o, d,
        Vec3f(grid.p0.x, grid.p0.y, grid.p0.z), Vec3f(grid.p1.x, grid.p1.y, grid.p1.z))

@always_inline
@always_inline
def _bb_lobe(x: Float32, m: Float32, s1: Float32, s2: Float32) -> Float32:
    """One asymmetric-Gaussian lobe of Wyman et al. 2013's analytic fit to the
    CIE 1931 colour-matching functions."""
    var sd = s1 if x < m else s2
    var t = (x - m) / sd
    return exp(Float32(-0.5) * t * t)


def blackbody_rgb(temp: Float32) -> RGB:
    """Linear-sRGB colour of a `temp` Kelvin blackbody, normalized to
    LUMINANCE 1 -- which is what pbrt's blackbody lights actually deliver, so
    a `"float scale"` beside a `"blackbody L"` means the same thing in both
    renderers.

    This used to be the Mitchell-Charity RGB approximation normalized so the
    max CHANNEL was 1, on the stated reasoning that pbrt's BlackbodySpectrum
    "likewise normalizes to a peak of 1 (via Wien's law)". That conflates two
    different things: pbrt normalizes the SPECTRUM's peak, and the RGB that
    spectrum then integrates to has neither unit max-channel nor unit
    luminance a priori. Measured against pbrt on a quad light (per unit L,
    with an `"rgb L" [1 1 1]` control confirming the harness at 1.0006):

        T=3500  pbrt [1.5642 0.8913 0.4063]   old code [1.0 0.756 0.555]
        T=6500  pbrt [1.0437 0.9827 1.0335]   old code [1.0 1.0   0.985]

    -- wrong in magnitude AND chromaticity. pbrt's own values have luminance
    0.9994 and 0.9994 respectively, i.e. exactly 1, which is what this now
    reproduces (to 0.7% at 3500 K and 0.2% at 6500 K, the residual being the
    analytic CIE fit rather than the quadrature).

    Planck's law integrated against Wyman et al. 2013's analytic CIE fits at
    10 nm over 360-830 nm, then XYZ->linear sRGB and divided by Y. 10 nm is
    indistinguishable from 1 nm here (<0.01%) because Planck is smooth and the
    fit's narrowest lobe is still ~12 nm wide -- worth keeping cheap, since
    this also runs per emission sample inside the GPU medium kernel for nanovdb
    temperature grids. Wavelengths are carried in MICROMETRES so lambda^-5 stays
    in a comfortable Float32 range.

    A future improvement worth noting: for the SPECTRAL medium path this is
    strictly worse than evaluating Planck at the four hero wavelengths
    directly -- that would be both cheaper (4 exps, not 48) and exact, with no
    RGB round trip at all."""
    if temp <= Float32(0.0):
        return RGB(Float32(0.0))
    var X = Float32(0.0)
    var Y = Float32(0.0)
    var Z = Float32(0.0)
    var lam = Float32(360.0)
    while lam <= Float32(830.0):
        var lum = lam * Float32(0.001)                  # micrometres
        var xarg = Float32(14388.0) / (lum * temp)      # c2 in um*K
        var inv: Float32
        if xarg > Float32(80.0):
            # exp(xarg) would overflow Float32; 1/(e^x - 1) -> e^-x there.
            inv = exp(-xarg)
        else:
            inv = Float32(1.0) / (exp(xarg) - Float32(1.0))
        var l2 = lum * lum
        var spec = inv / (l2 * l2 * lum)
        X += spec * (Float32(1.056) * _bb_lobe(lam, Float32(599.8), Float32(37.9), Float32(31.0))
                   + Float32(0.362) * _bb_lobe(lam, Float32(442.0), Float32(16.0), Float32(26.7))
                   - Float32(0.065) * _bb_lobe(lam, Float32(501.1), Float32(20.4), Float32(26.2)))
        Y += spec * (Float32(0.821) * _bb_lobe(lam, Float32(568.8), Float32(46.9), Float32(40.5))
                   + Float32(0.286) * _bb_lobe(lam, Float32(530.9), Float32(16.3), Float32(31.1)))
        Z += spec * (Float32(1.217) * _bb_lobe(lam, Float32(437.0), Float32(11.8), Float32(36.0))
                   + Float32(0.681) * _bb_lobe(lam, Float32(459.0), Float32(26.0), Float32(13.8)))
        lam += Float32(10.0)
    if Y <= Float32(0.0):
        return RGB(Float32(0.0))
    var iy = Float32(1.0) / Y
    var xn = X * iy
    var zn = Z * iy
    var r = Float32(3.2406) * xn - Float32(1.5372) - Float32(0.4986) * zn
    var g = Float32(-0.9689) * xn + Float32(1.8758) + Float32(0.0415) * zn
    var b = Float32(0.0557) * xn - Float32(0.2040) + Float32(1.0570) * zn
    return RGB(max(r, Float32(0.0)), max(g, Float32(0.0)), max(b, Float32(0.0)))


@always_inline
def hg_phase(cos_theta: Float32, g: Float32) -> Float32:
    """Henyey-Greenstein phase function, pbrt's exact form and convention:
    with `cos_theta = dot(wo, wi)` and wo pointing BACK along the incoming ray,
    g > 0 peaks at cos_theta = -1, i.e. wi continuing forward. Integrates to 1
    over the sphere, so it doubles as its own sampling pdf."""
    var gg = g * g
    var denom = Float32(1.0) + gg + Float32(2.0) * g * cos_theta
    if denom < Float32(1e-7):
        denom = Float32(1e-7)
    return INV_FOUR_PI * (Float32(1.0) - gg) / (denom * sqrt(denom))

@always_inline
def hg_sample(wo: Vec3f, g: Float32, u1: Float32, u2: Float32) -> SIMD[DType.float32, 4]:
    """Sample the HG phase function about `wo`; returns (wi.x, wi.y, wi.z, pdf).
    Falls back to the uniform-sphere sampler for |g| < 1e-3, both because the
    closed form is numerically unstable there and because that IS the isotropic
    case. Same inversion pbrt uses."""
    var cos_theta: Float32
    if abs(g) < Float32(1e-3):
        cos_theta = Float32(1.0) - Float32(2.0) * u1
    else:
        var gg = g * g
        var sq = (Float32(1.0) - gg) / (Float32(1.0) + g - Float32(2.0) * g * u1)
        cos_theta = -(Float32(1.0) + gg - sq * sq) / (Float32(2.0) * g)
    if cos_theta < Float32(-1.0): cos_theta = Float32(-1.0)
    if cos_theta > Float32(1.0): cos_theta = Float32(1.0)
    var sin_theta = sqrt(max(Float32(0.0), Float32(1.0) - cos_theta * cos_theta))
    var phi = TWO_PI * u2
    var f = Frame.from_z(wo)
    var lx = sin_theta * cos(phi)
    var ly = sin_theta * sin(phi)
    var wi = f.x * lx + f.y * ly + f.z * cos_theta
    return SIMD[DType.float32, 4](wi[0], wi[1], wi[2], hg_phase(cos_theta, g))

@fieldwise_init
struct MediumInterface_C(TrivialRegisterPassable):
    """Binds inside/outside media to a surface. -1 = vacuum."""
    var inside_medium_idx:  Int32
    var outside_medium_idx: Int32


# ── One free-flight sampler for every integrator ───────────────────────────────
# `sample_homogeneous_free_flight` above handles the closed-form case. Everything
# from here down is the HETEROGENEOUS half plus the dispatcher that picks between
# them, so that a caller writes one call and gets the right estimator.
#
# This used to live inline in gpu.mojo's `_sample_medium_core`, which meant the
# PLAIN PATH TRACER was the only integrator that ever sampled a density field:
# SPPM (sppm.mojo) and BDPT/VCM (bdpt.mojo) called
# `sample_homogeneous_free_flight` UNCONDITIONALLY, so for any "uniformgrid",
# "nanovdb" or procedural-"cloud" medium they used sigma_a/sigma_s -- which are
# PER-UNIT-DENSITY coefficients for such a medium, see Medium_C -- as if they
# were the absolute extinction, i.e. density identically 1 everywhere inside the
# bounding shape. bunny-cloud rendered as a featureless fog-filled sphere with no
# bunny in it, and clouds as flat noise, under both integrators (2026-09-16, found
# by the 4-way integrator-switcher gallery putting them beside the path tracer).
# Moving the loop here rather than copying it into two more files is the whole
# point: there is exactly one delta tracker, and it is this one.
comptime MEDIUM_TRACK_MAX_ITERS: Int = 10000  # delta/ratio-tracking loop safety bound

@always_inline
def medium_is_heterogeneous(med: Medium_C) -> Bool:
    """True if this medium's extinction is modulated by a density field, from
    EITHER source -- dense "uniformgrid" (grid_idx) or sparse "nanovdb"
    (nvdb_idx); the parser never sets both. The one place that question is
    asked, so a consumer never open-codes `grid_idx >= 0 or nvdb_idx >= 0` and
    silently forgets one of the two sources."""
    return med.grid_idx >= Int32(0) or med.nvdb_idx >= Int32(0)

@always_inline
def medium_grid_for(
    med: Medium_C, grids: Pointer[Grid_C, MutUntrackedOrigin]
) -> Grid_C:
    """`grids[med.grid_idx]`, or an inert zero-extent placeholder when this
    medium has no dense grid -- a homogeneous or nanovdb medium has
    grid_idx == -1 and must never index that array. Exists so the placeholder
    literal is written ONCE instead of at every site that needs a Grid_C in
    scope (free-flight sampling, NEE ratio tracking, shadow rays)."""
    if med.grid_idx < Int32(0):
        return Grid_C(
            Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
            Int32(0), Int32(0), Int32(0),
            Point3f(Float32(0), Float32(0), Float32(0)),
            Point3f(Float32(0), Float32(0), Float32(0)),
            SIMD[DType.float32, 16](0), Float32(0))
    return grids[unsafe_offset=Int(med.grid_idx)]

@always_inline
def medium_nvdb_for(
    med: Medium_C, nvdb_grids: Pointer[NvdbGrid_C, MutUntrackedOrigin]
) -> NvdbGrid_C:
    """`nvdb_grids[med.nvdb_idx]`, or an inert placeholder. See medium_grid_for."""
    if med.nvdb_idx < Int32(0):
        return NvdbGrid_C(
            Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(), Int64(0),
            SIMD[DType.float32, 16](0), SIMD[DType.float32, 16](0),
            Vec3f(Float32(0), Float32(0), Float32(0)),
            Point3f(Float32(0), Float32(0), Float32(0)),
            Point3f(Float32(0), Float32(0), Float32(0)), Float32(0))
    return nvdb_grids[unsafe_offset=Int(med.nvdb_idx)]

@always_inline
def medium_emission_spectral(
    c: RGB, wl: SampledWavelengths,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
) -> SpectralSample:
    """RGB emission/radiance -> spectral, at the light boundary inside a medium.
    Falls back to a flat spectrum when no spectral table is loaded, so a
    table-less build still transports the RGB magnitude. (Was gpu.mojo's
    `_med_spec_illum`; moved here so the shared tracker below can apply it to
    each emission candidate exactly where the path tracer used to.)"""
    if spectral_res <= 0:
        return SpectralSample(c.r, c.g, c.b, (c.r + c.g + c.b) * Float32(0.3333333))
    return rgb_illuminant_to_spectral_sample(spectral_coeffs, spectral_res,
        spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, c.r, c.g, c.b, wl)

def sample_free_flight(
    med: Medium_C,
    grids: Pointer[Grid_C, MutUntrackedOrigin],
    nvdb_grids: Pointer[NvdbGrid_C, MutUntrackedOrigin],
    ray_org: Vec3f,
    ray_dir: Vec3f,
    t_surf: Float32,
    mut pcg: PCG32,
    # Inert placeholder: only ever reached when a caller omits the emission
    # arguments, and then spectral_res is 0 too, so medium_emission_spectral
    # takes its flat-spectrum fallback and never reads these.
    wavelengths: SampledWavelengths = SampledWavelengths(
        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)),
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
) -> FreeFlight:
    """Sample a free-flight distance through `med` along `ray_org + t*ray_dir`,
    up to the surface at `t_surf`. THE entry point every integrator should use:
    homogeneous media take the analytic closed form, heterogeneous ones take
    delta tracking against a local majorant, and both come back in the same
    `FreeFlight` shape.

    The emission arguments are optional and only do anything for an emissive
    nanovdb medium (pbrt NanoVDBMedium's temperature grid); a caller that does
    not model volumetric emission can omit them and ignore `.emission`.

    Both heterogeneous sources use the RED channel exclusively for
    majorant/accept-reject decisions: exact for the achromatic density fields
    supported today, would need per-wavelength free-flight sampling with
    spectral MIS to extend to a colored density field. See
    docs/09_volumetric_media.md for the delta/ratio-tracking theory and the
    local-majorant optimization."""
    if not medium_is_heterogeneous(med):
        # Hand the homogeneous sampler its 4 hero-lane extinctions when a real
        # spectral table is available, so the free flight is drawn from the
        # lane mixture (bounded weights) rather than from red alone. Without a
        # table medium_sigma_t_spectral has no curve to evaluate, so stay on
        # the single-lane estimator, which is the same estimator with n = 1.
        if spectral_res > 0:
            var lane_sig = medium_sigma_t_spectral(
                med, wavelengths, spectral_coeffs, spectral_res,
                spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
            return sample_homogeneous_free_flight(med, t_surf, pcg, lane_sig, 4)
        return sample_homogeneous_free_flight(med, t_surf, pcg)

    var sigma_t = med.sigma_a + med.sigma_s
    # Resolve which density source this medium has ONCE, not per iteration:
    # they differ only in the per-candidate lookup and the majorant.
    var use_nvdb = med.nvdb_idx >= Int32(0)
    var grid = medium_grid_for(med, grids)
    var nvdb_grid = medium_nvdb_for(med, nvdb_grids)
    var majorant_density = nvdb_grid.max_density if use_nvdb else grid.max_density
    var sigma_maj = majorant_density * sigma_t.r
    var emission = SpectralSample(Float32(0))
    if sigma_maj <= Float32(0.0):
        return FreeFlight(False, t_surf, Float32(0), Float32(1.0), RGB(Float32(0)), RGB(Float32(1)), emission)

    # ── Segment-wise tracking with LOCAL majorants ─────────────────────────
    # Walk the ray one NanoVDB tree node at a time, using that node's own max
    # density as the local majorant (6.7x on disney-cloud vs one global
    # majorant -- see docs/09_volumetric_media.md, "Local majorants").
    # Unbiased by the memorylessness of the exponential: an overshoot just
    # resumes sampling from the segment boundary under the next node's
    # majorant. uniformgrid keeps one segment spanning the whole ray under the
    # global majorant (no per-node structure to walk).
    var t = Float32(0.0)
    var collided = False
    var iters = 0
    var seg_end = Float32(-1.0)       # < t forces a majorant query on entry
    var sigma_maj_seg = Float32(0.0)
    var density = Float32(0.0)
    var iray = nvdb_index_ray(nvdb_grid, ray_org, ray_dir) if use_nvdb else SIMD[DType.float32, 8](0)
    while iters < MEDIUM_TRACK_MAX_ITERS:
        iters += 1
        if t >= seg_end:
            if t >= t_surf:
                break
            if use_nvdb:
                var mr = nvdb_majorant_at_world(nvdb_grid, ray_org + t * ray_dir)
                sigma_maj_seg = mr[0] * sigma_t.r
                seg_end = min(nvdb_node_exit_t(iray, t, mr[1]), t_surf)
            else:
                sigma_maj_seg = sigma_maj
                seg_end = t_surf
            if sigma_maj_seg <= Float32(0.0):
                t = seg_end
                continue
        var u = pcg.next_float()
        var t_next = t + (-log(max(u, Float32(1e-7))) / sigma_maj_seg)
        if t_next >= seg_end:
            t = seg_end
            continue
        t = t_next
        var p_world = ray_org + t * ray_dir
        density = nvdb_sample_density(nvdb_grid, p_world) if use_nvdb else grid_sample_density(grid, p_world)
        # ── Volumetric emission (pbrt NanoVDBMedium) ───────────────────────
        # Accumulated at EVERY majorant candidate, weighted by the local
        # absorption fraction sigma_a/sigma_maj: that is the standard unbiased
        # estimator of the emitted-radiance integral along the segment when
        # distances are drawn against the majorant, and it must happen before
        # the collision test below (which breaks out of the loop) so emission
        # from the pass-through candidates is not silently dropped.
        # Temperature comes from a SECOND nanovdb grid stored as an ordinary
        # entry in the same array, so this reuses nvdb_sample_density
        # unchanged. Non-emissive media take the nvdb_temp_idx < 0 branch and
        # pay nothing.
        if med.nvdb_temp_idx >= Int32(0) and med.le_scale > Float32(0.0):
            var tgrid = nvdb_grids[unsafe_offset=Int(med.nvdb_temp_idx)]
            var tk = (nvdb_sample_density(tgrid, p_world) - med.temp_offset) * med.temp_scale
            if tk > Float32(100.0):
                var sigma_a_real = density * med.sigma_a.r
                emission += medium_emission_spectral(
                    blackbody_rgb(tk), wavelengths, spectral_coeffs, spectral_res,
                    spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65
                ) * (med.le_scale * sigma_a_real / sigma_maj_seg)
        var sigma_t_real = density * sigma_t.r
        var u2 = pcg.next_float()
        if u2 < sigma_t_real / sigma_maj_seg:
            collided = True
            break
    if not collided:
        return FreeFlight(False, t, Float32(0), Float32(1.0), RGB(Float32(0)), RGB(Float32(1)), emission)

    # `sig_t` is the extinction ACTUALLY in force at the collision point
    # (density-scaled), so a caller's analytic `sig_t * exp(-sig_t * t_free)`
    # pdf stays meaningful and reduces exactly to the homogeneous expression at
    # density 1. The single-scattering albedo needs no density factor at all --
    # it cancels between sigma_s and sigma_t.
    var inv_sig_t = Float32(1.0) / max(sigma_t.r, Float32(1e-7))
    var alb = RGB(med.sigma_s.r * inv_sig_t,
                  med.sigma_s.g / max(sigma_t.g, Float32(1e-7)) if sigma_t.g > Float32(0.0) else med.sigma_s.r * inv_sig_t,
                  med.sigma_s.b / max(sigma_t.b, Float32(1e-7)) if sigma_t.b > Float32(0.0) else med.sigma_s.r * inv_sig_t)
    # Delta tracking's collision density is not analytic; it accepts on the
    # red channel alone and carries no chromatic ratio (weight 1), so the
    # honest pdf to record is that lane's own exponential -- which is what
    # bdpt.mojo reconstructed here before, so heterogeneous media are
    # bit-for-bit unchanged by the MIS work.
    var het_sig = density * sigma_t.r
    return FreeFlight(True, t, het_sig, het_sig * exp(-het_sig * t), alb, RGB(Float32(1)), emission)
