# Stochastic Progressive Photon Mapping — CPU driver + GPU kernels, sharing
# one comptime[use_gpu]-parameterized core (same pattern as bdpt_*.mojo's
# LVC-BPT port; see project_unified_renderer_roadmap in memory).
# Reference: Hachisuka et al. 2008 "Progressive Photon Mapping"

from std.collections import Array
from std.sys.info import size_of
from max.algorithm import parallelize
from std.math import sqrt, cos, sin, floor, max, min, abs
from std.memory.alloc import unsafe_alloc
from std.atomic import Atomic
from std.memory import bitcast
from gonzales.transport.medium import MediumView
from gonzales.core.geometry import face_toward
from gonzales.transport.sss_walk import SssExit, sss_entered, sss_hop, sss_leave, sss_walks_below
from gonzales.transport.medium_interface import default_emit_medium, medium_after
from gonzales.core.geometry import offset_eps, TERMINAL_SEGMENT_GRACE_ROUNDS, RGB, Point3f, Point2f, Vec3f, vec3f, point3f, dot, cross, PI, INV_FOUR_PI, Frame, _is_real_ptr
from gonzales.scene.appearance.materials import Material, MatKind, LobeKind, PhotonKind, fr_dielectric, resolve_mix
from gonzales.scene.shapes.primitives import (
    Intersection, PrimId, TriangleMesh, Sphere, Instance, sphere_outward_normal, mesh_normal, mesh_vi,
)
from gonzales.core.geometry import Ray
from gonzales.scene.shapes.primitives import mesh_pos
from gonzales.scene.appearance.media import medium_inv_sigma_t_spectral
from gonzales.scene.appearance.media import Medium, Grid, NvdbGrid, SSS_WALK_ROUNDS
from gonzales.scene.appearance.lights import area_light_pick_triangle, area_light_side_pdf, AreaLight
from gonzales.scene.shapes.curves import Curve, curve_at, curve_piece_endpoints, _curve_perp_axis
from gonzales.scene.accel.bvh import traverse_dispatch, any_hit_dispatch
from gonzales.scene.accel.bvh import SceneView, curve_radius, _hair_sample_dir_u, _scene_bounding_sphere, _sample_disk_perpendicular, _sample_infinite_light_dir, _eval_infinite_light_and_pdf, _hair_precompute, LightSample, _sample_distant_light_nee, _sample_point_light_nee, _sample_sphere_light_nee, _sample_infinite_light_nee, light_path_pick, light_path_pick_pdf
from gonzales.transport.glossy_sms import (SMSChain, GLOSSY_SMS_MIN_PDF, GSMS_DEBUG, glossy_sms_lobe, glossy_sms_solve, glossy_sms_mis,
    glossy_sms_reproduce, glossy_sms_t_chain)
from gonzales.transport.sms import mnee_orthonormal_basis, sms_vertex_init
from gonzales.transport.manifold import _mnee_chain_transmittance, _mnee_pass
from gonzales.transport.shade_context import _shade_context
from gonzales.paths.vcm.eval import _vcm_mnee_light_point
from gonzales.transport.mis_policy import mis_policy_power, mis_policy_sole
from gonzales.core.sampling import power_heuristic
from gonzales.transport.bxdf import ggx_lambda, dielectric_interface, bxdf_sample_dielectric, bxdf_sample_thin_dielectric, BxDFFlags, GeomContext, bxdf_sample_conductor, BSDF, LobeTables
from gonzales.transport.surface import surface_hit
from gonzales.integrators.pt.common import area_light_hit_cos
from gonzales.transport.texturing import apply_surface_maps_at_hit
from gonzales.scene.shapes.geom_normal import _geom_normal, _shading_normal_at
from gonzales.estimators.photon_grid import (
    _HSIZE, _hash_cell, NEIGHBOUR_CELLS, neighbour_buckets, _PHOTON_BUCKET_CAP, grid_reset_cell, grid_count, grid_keep, grid_coin_bits, grid_push, grid_weight,
    gather_disk_contains, gather_disk_coverage, gather_ball_reach, disk_measure, ball_measure, ribbon_measure,
)
from gonzales.transport.shadow_media import shadow_transmittance
from gonzales.core.sampling import sample_ggx_vndf, camera_ray_from_film_xy, FilmFilter, film_filter_of, film_filter_offset
from gonzales.core.transform import transform_normal, Mat4
from gonzales.core.rng import PCG32
from gonzales.scene.description.pbrt_parser import ParsedScene_Mojo
from gonzales.core.progress import Progress
from gonzales.integrators.outputs import finish_render
from gonzales.core.spectrum import (
    SampledWavelengths, SpectralSample, sample_wavelengths,
    rgb_illuminant_to_spectral_sample, spectral_sample_to_rgb,
    spec_refl, spec_refl_unbounded, spec_illum, pass_wavelengths,
)


comptime _ALPHA  = Float32(2.0) / Float32(3.0)  # pbrt-v4's own value (integrators.cpp's SPPM radius-update `gamma`, Hachisuka/Jensen progressive photon mapping); was 0.7, a real but small (~5% asymptotic shrinkage-rate) mismatch


comptime _MAX_B  = 10  # hard safety ceiling, matching bdpt_*.mojo's own
                       # _BDPT_MAX_VERTS -- SPPM's bounce loops are additionally
                       # bounded by min(scene maxdepth, _MAX_B), see
                       # _sppm_trace_visible_point/_sppm_trace_photon's own
                       # `maxdepth` parameter. Before that bound was added, a
                       # scene with a small maxdepth (a common way to
                       # deliberately limit indirect bounces) was silently
                       # ignored and rendered up to 10 anyway -- measured on a
                       # cavity scene at maxdepth=1: SPPM read 2.05x pbrt where
                       # the (maxdepth-respecting) path tracer read 0.93x.


comptime _MAX_B_MEDIA = 50  # photon depth cap with media: a thick high-albedo cloud needs ~maxdepth scatters


@always_inline
def _photon_depth_cap(ref sd: SceneView) -> Int:
    return _MAX_B_MEDIA if Int(sd.mediumCount) > 0 else _MAX_B


# Independent visible-point samples per pixel, traced ONCE for the whole
# render (not re-traced every SPPM pass — see _sppm_camera_pass's docstring).
# Needed because the camera ray can cross a dielectric surface (stochastic
# reflect-vs-refract choice): with only 1 sample, a pixel unlucky enough to
# draw "reflect" would never see the diffuse surface behind the glass/water
# at all (the original black-speckle bug). With _VP_SAMPLES independent
# draws, P(all reflect) is (fresnel)^_VP_SAMPLES — negligible in practice —
# and each sample gets its own persistent (r2, tau, N_acc) accumulator that
# converges correctly since its surface/position never changes pass to pass.
comptime _VP_SAMPLES = 16


# GPU renders trace up to 32 visible points per pixel while they fit a VRAM budget: sparse
# bright features seen through glass stipple with 16 (about 1.7x the time). The CPU keeps 16.
comptime _VP_BUDGET_BYTES = 5_000_000_000


def _vp_samples_for(n_pix: Int, use_gpu: Bool) -> Int:
    if not use_gpu:
        return _VP_SAMPLES
    var n = 32
    while n > _VP_SAMPLES and n_pix * n * (size_of[SPPMPixel]() + size_of[Intersection]()) > _VP_BUDGET_BYTES:
        n //= 2
    return n


# ── Data structures ───────────────────────────────────────────────────────────


@fieldwise_init
struct SPPMPixel(TrivialRegisterPassable):
    """Visible point from one camera ray + SPPM accumulators."""
    var pos:    Point3f
    # The SHADING normal -- after bump/normal maps. What the BRDF is evaluated
    # against.
    var normal: Vec3f
    # The GEOMETRIC normal, for the gather's density tests (tangent disk and
    # which side a photon arrived from). Kept separate for the same reason as
    # BDPTVertex.normal/shading_normal: a density test against a PERTURBED
    # normal is a bias. With a normal tilted by theta, a photon genuinely on
    # the same flat surface sits |e| sin(theta) off the tangent plane, so the
    # disk test (one tenth of the radius) rejected every valid photon on any
    # bump steeper than ~5.7 degrees -- cornell-box-normalmap's SPPM cell
    # dropped 0.8% the moment the disk test went in, which is how it showed.
    var geo_normal: Vec3f
    # Camera-subpath throughput, RGB. It is the one transport quantity here
    # that deliberately stays RGB, because it has nowhere spectral to go: it
    # multiplies `tau`, which sums gathers from many passes at many
    # wavelengths and is RGB by construction (see below). Carrying it as a
    # SpectralSample and converting at finalize is WRONG and was measured to
    # be: spectral_sample_to_rgb is a radiance estimator, so a bare
    # throughput -- a flat unit spectrum, not an upsampled colour -- does not
    # come back as (1,1,1). A lossless white cavity, which must be exactly
    # neutral, rendered (0.446, 0.377, 0.375). The camera subpath here is
    # also short by construction (eye to the first diffuse/volume vertex,
    # i.e. a specular chain), so this is where RGB costs least.
    var beta: RGB  # camera throughput
    var alb:  RGB  # surface albedo
    # Gathered flux, accumulated ACROSS passes -- and each pass carries its
    # own hero wavelengths, so this one cannot be spectral: lane i would mean
    # a different wavelength in every term of the sum. Each pass's gather is
    # spectral internally and converts to RGB exactly once, here. That leaves
    # a single RGB product where the two subpaths meet (beta x tau); the
    # transport along each of them is spectral end to end, which is where the
    # per-bounce compounding error lives.
    var tau:  RGB  # accumulated flux (RGB: sums across passes/wavelengths)
    var N_acc:  Float32   # photon count (alpha-weighted sum)
    var r2:     Float32   # current search radius²
    var valid:  Int32     # 1 = has VP
    var pidx:   Int32     # flat pixel index
    var is_volume: Int32  # 1 = volume scatter VP (isotropic phase fn); 0 = surface
    # Direct (NEE) lighting accumulator — resampled fresh each SPPM pass (one
    # shadow ray per pass, same cadence as the photon pass), summed here and
    # divided by n_passes at finalize time. This is what pbrt's own SPPM
    # calls "pixel.Ld": a completely separate term from tau/photon-density,
    # capturing direct illumination at the visible point (which the
    # photon-density term alone can't reconstruct without heavy noise, since
    # it's estimating both direct AND indirect/caustic lighting through a
    # single, indirect-only channel otherwise). Applies to BOTH surface and
    # volume VPs: a volume VP uses the isotropic phase function (albedo/4pi,
    # no cosine) in place of a BRDF. Volume VPs were excluded from this
    # entirely until 2026-09-09, on the recorded assumption that "this scene
    # has no participating media" -- true of water-caustic, which is what
    # SPPM's NEE was validated against, and false of volumetric-caustic,
    # where ~86% of camera rays scatter in fog and thus received NO direct
    # lighting at all (the scene rendered as an unlit box, 8x too dark).
    var ld: SpectralSample
    # Index into sd.mediums of the medium this VP sits in, or -1 for vacuum.
    # Set for surface AND volume VPs alike: NEE from any point inside a
    # medium must attenuate the shadow ray by that medium's transmittance,
    # not just from a volume scatter vertex.
    var med_idx: Int32
    # Infinite-light radiance for a VP sample whose traced ray escaped the
    # scene entirely (valid stays 0 — there's no surface to gather photons
    # at or run NEE from) instead of hitting a diffuse/volume scatterer.
    # Already beta-weighted at trace time (see _sppm_trace_visible_point's
    # miss branch), so _sppm_finalize_one_pixel adds it directly rather than
    # multiplying by vp.beta again like the tau/ld terms.
    var env: RGB
    # Sum over passes of the direct-light and escape terms, already beta-weighted RGB (see
    # _sppm_flush_one); divided by n_passes at finalize. tau has beta folded in too.
    var gsum: RGB
    # 1 - F of the coat microfacet a split-coat VP sampled (alpha < 0 marks one); undoes the
    # 1/(1-F) in beta for the delta-light highlight, which NEE estimates whole.
    var coat_keep: Float32
    # Material dispatch for gather/NEE BRDF evaluation: 0 = Lambertian
    # (diffuse/coated_diffuse/diffuse_transmit — f_r = alb/π, angle-
    # independent), 1 = rough conductor/coated_conductor (GGX — f_r depends
    # on both wo and wi, evaluated via bxdf_eval_conductor_ggx), 2 = hair
    # (Marschner 3-lobe, evaluated via bvh.mojo's _hair_precompute/
    # _hair_eval_lobes). Volume VPs (is_volume=1) ignore this and always use
    # the isotropic phase function.
    var mat_kind: Int32
    # Outgoing direction (toward the camera) at this VP — only populated/used
    # when mat_kind=1 or 2, since both GGX and hair evaluation need both
    # directions unlike Lambertian's angle-independent f_r.
    var wo: Vec3f
    # GGX roughness (max(roughU, roughV), matching bdpt_vertex.mojo's BDPTVertex
    # convention) — only meaningful when mat_kind=1. alb doubles as F0 for
    # conductor VPs, same repurposing bdpt_vertex.mojo's BDPTVertex.alb already uses.
    var alpha: Float32
    # mat_kind=2 (hair) only: material index (to re-fetch eta/sigma_a/betaM/
    # betaN from sd.materials) + curve hit info (to re-derive the fiber frame
    # via _hair_precompute) — NOT the full ~30-field HairLobeConstants, to
    # keep this struct small for every other VP kind; mirrors bdpt_*.mojo's
    # BDPTVertex's own mat_idx/hair_curve_idx/hair_h/hair_v fields exactly.
    var mat_idx: Int32
    var hair_curve_idx: Int32
    var hair_h: Float32
    var hair_v: Float32
    # Hero-wavelength sample this VP's camera subpath was traced at (staged
    # spectral rollout, see project_spectral_rendering memory /
    # lovely-dazzling-meteor plan). Unused until Stage 4.
    var wavelengths: SampledWavelengths


@fieldwise_init
struct SPPMPhoton(TrivialRegisterPassable):
    """Photon stored at a scatter event (surface diffuse, conductor, or
    volume)."""
    var pos: Point3f
    var flux: SpectralSample  # flux at stored position, at the PASS's wavelengths
    var nxt:       Int32  # chained-list link in hash grid (-1 = end)
    var is_volume: Int32  # 1 = volume scatter photon; 0 = surface
    # Direction the photon was travelling when it was stored (i.e. the ray
    # direction, NOT negated) — needed at gather time to reconstruct wi
    # (= -dir_in) for a conductor VP's GGX evaluation. Lambertian/volume
    # gather ignores this (f_r is angle-independent), so it's set but unused
    # for those photon kinds.
    var dir_in: Vec3f
    # Hero-wavelength sample this photon was emitted at — independent of the
    # gathering VP's own wavelengths (real photons are independently
    # colored); see Stage 4 in project_spectral_rendering memory for how the
    # cross-wavelength gather is handled (converted to RGB at deposit time,
    # not kept spectral through tau's progressive accumulator). Unused until
    # Stage 4.
    var wavelengths: SampledWavelengths
    var curve: Int32   # the strand a photon on hair landed on, else -1: hair gathers per strand (the fibre kernel)


# ── Geometry helpers ──────────────────────────────────────────────────────────


@always_inline
def _cosine_hemisphere_sample(n: Vec3f, u1: Float32, u2: Float32) -> Vec3f:
    """Cosine-weighted random direction in the hemisphere around normal n
    (Frisvad tangent frame, same construction used for area-light emission
    sampling in _sppm_photon_pass)."""
    var r_samp = sqrt(u1)
    var theta = Float32(2.0) * PI * u2
    var lx = r_samp * cos(theta)
    var lz_loc = r_samp * sin(theta)
    var ly = sqrt(max(Float32(0.0), Float32(1.0) - u1))
    var sgn = Float32(1.0) if n[2] >= Float32(0.0) else Float32(-1.0)
    var a_tf = Float32(-1.0) / (sgn + n[2])
    var b_tf = n[0] * n[1] * a_tf
    var tangent   = Vec3f(Float32(1.0) + sgn*n[0]*n[0]*a_tf, sgn*b_tf, -sgn*n[0])
    var bitangent = Vec3f(b_tf, sgn + n[1]*n[1]*a_tf, -n[1])
    var d = tangent * lx + bitangent * lz_loc + n * ly
    var dl = dot(d, d)
    if dl > Float32(0.0): d = d * (Float32(1.0) / sqrt(dl))
    return d


# ── Dielectric bounce helper ──────────────────────────────────────────────────
# Returns (new_dir, new_org, radiance_scale, new_current_ior, new_previous_ior)
# after reflection or refraction. Mutates pcg. `radiance_scale` is the
# PBRT-style non-symmetric-scattering correction (1/eta² on transmission, 1
# on reflection) for transporting RADIANCE (camera/VP paths) across a change
# of IOR — solid angle compresses/expands across the interface, so radiance
# isn't conserved the way importance/flux is. Callers tracing a camera-origin
# subpath (SPPM's visible-point pass, BDPT's camera path) must multiply their
# beta by this; callers tracing a light-origin subpath (SPPM's
# photon-emission pass, BDPT's light path — TransportMode::Importance) must
# NOT apply it, or every transmissive light/photon path gets silently biased.
# Shared by both since the geometry math (entering/exiting, eta, Fresnel,
# TIR) is identical either way — only which mode multiplies the returned
# scale into its throughput differs, entirely at the call site.
#
# `current_ior`/`previous_ior` are the same depth-2 touching-dielectric-IOR
# stack bxdf_sample_dielectric (bxdf.mojo, the plain path tracer's dielectric
# core) carries on PathState — see PathState.current_dielectric_ior/
# previous_dielectric_ior's docstrings (geometry.mojo) for the full
# derivation and the transparent-machines repro that exposed both bugs this
# mirrors: entering used to assume vacuum unconditionally (eta = 1/ior),
# breaking touching same-material surfaces (an optically invisible seam
# reading as a lossy one); exiting used to assume vacuum unconditionally too
# (eta = ior), spuriously triggering total internal reflection at a boundary
# that should have been transparent. BDPT and SPPM call this same function
# (unlike the plain path tracer, which has its own independent
# bxdf_sample_dielectric) but never threaded current_ior/previous_ior through
# — so they still carried both original bugs after bxdf_sample_dielectric was
# fixed. `current_ior`/`previous_ior` default to vacuum so this is a
# behavior-preserving signature change for any caller that doesn't thread
# real values through.
@always_inline
def _vp_strat_u(vdc: Float32, pidx: Int32, event: Int) -> Float32:
    """vdc shifted by a per-(pixel, event) random offset (Cranley-Patterson), wrapped to [0,1). The event index must
    differ per crossing: depth stays put across a subsurface boundary, and a shared offset ties two choices together."""
    var shift_rng = PCG32(UInt64(pidx) * UInt64(0x9E3779B97F4A7C15) + UInt64(event + 1), UInt64(3))
    var sh = shift_rng.next_float()
    var v = vdc + sh
    return v - floor(v) if v >= Float32(1.0) else v


@always_inline
def _dielectric_bounce(
    ray_dir: Vec3f,
    hit_point: Vec3f,
    geom_normal: Vec3f,
    ior: Float32,
    # True only for a ray that cannot possibly be inside this dielectric yet,
    # i.e. the primary/first segment in vacuum. Used to fix inward-normal
    # meshes. Was `bounce: Int` with an internal `bounce == 0` test, which
    # breaks the moment a caller stops charging bounces for subsurface
    # boundaries (see Material.sss_boundary): `bounce` then stays 0 for the
    # whole interior walk and every boundary hit from INSIDE would be forced
    # to "entering", refracting inward again so light could never leave. The
    # caller knows whether it is in a medium; it passes the real question.
    force_entering: Bool,
    mut pcg: PCG32,
    current_ior: Float32 = Float32(1.0),    # IOR of the medium the ray is ALREADY in; 1.0 = vacuum
    previous_ior: Float32 = Float32(1.0),   # IOR one level below current_ior (what exiting restores)
    is_thin: Bool = False,                  # `thindielectric`: both interfaces at once
    radiance_mode: Bool = True,             # camera path; False for a photon/light path
    balance_branches: Bool = False,         # reflect with p=clamp(F,.25,.75), reweighted: a 5% reflection of a bright sky is not a coin flip
) -> Tuple[Vec3f, Vec3f, Float32, Float32, Float32]:
    """SPPM/VCM's dielectric bounce: a RAY-LEVEL adapter over the path
    tracer's BxDFs, holding no scattering physics of its own.

    It used to reimplement the reflect/refract/TIR split, the thin-slab
    Fresnel and the radiance correction. Every one of those drifted from the
    path tracer's copy and had to be fixed twice:

      * the touching-dielectric IOR stack (fixed in bxdf_sample_dielectric,
        then again here),
      * the thin-slab 2R/(1+R) branch (bxdf_sample_thin_dielectric had it;
        SPPM and VCM rendered a thin slab as thick glass, furnace 2.20),
      * the eta^2 radiance compression, which this file had INVERTED long
        after bxdf_sample_dielectric was corrected -- barcelona's pool, whose
        water is a single OPEN plane so the entering and exiting factors
        never cancel, read 7.5x the reference.

    The comment that used to sit here even predicted the failure: "BDPT and
    SPPM call this same function (unlike the plain path tracer, which has its
    own independent bxdf_sample_dielectric) -- so they still carried both
    original bugs after bxdf_sample_dielectric was fixed." Two copies of one
    physics is the bug; there is now one.

    What legitimately lives here is the part that is NOT physics: drawing the
    lobe-selection sample from `pcg`, and offsetting the continuation ray's
    origin off the surface it just left (+n on reflect, -n on transmit).
    `radiance_mode` is pbrt's TransportMode, passed through."""
    var u = pcg.next_float()
    if is_thin:
        # A thin slab's two interfaces coincide: no bend, no medium entry and
        # no radiance compression, so the ior stack is untouched and the mode
        # does not matter.
        var (bs_t, n_t) = bxdf_sample_thin_dielectric(geom_normal, ray_dir, ior, u)
        var off_t = n_t if (Int(bs_t.flags) & Int(BxDFFlags.reflect)) != 0 else -n_t
        return (bs_t.wi, hit_point + off_t * offset_eps(hit_point.x, hit_point.y, hit_point.z), bs_t.f.r,
                current_ior, previous_ior)
    var branch_w = Float32(1.0)
    if balance_branches:
        var di = dielectric_interface(geom_normal, ray_dir, ior, force_entering, current_ior, previous_ior)
        if not di.tir:
            var p = min(max(di.fresnel, Float32(0.25)), Float32(0.75))
            if u < p:
                u = di.fresnel * (u / p)
                branch_w = di.fresnel / p
            else:
                u = di.fresnel + (Float32(1.0) - di.fresnel) * ((u - p) / (Float32(1.0) - p))
                branch_w = (Float32(1.0) - di.fresnel) / (Float32(1.0) - p)
    var (bs, normal, new_current_ior, new_previous_ior) = bxdf_sample_dielectric(
        geom_normal, ray_dir, ior, force_entering, u, current_ior, previous_ior,
        radiance_mode)
    var off = normal if (Int(bs.flags) & Int(BxDFFlags.reflect)) != 0 else -normal
    return (bs.wi, hit_point + off * offset_eps(hit_point.x, hit_point.y, hit_point.z), bs.f.r * branch_w,
            new_current_ior, new_previous_ior)


# ── Uniform area-light sampling ───────────────────────────────────────────────
# Shared by BDPT's light-subpath emission and SPPM's photon emission + NEE
# (CPU + GPU) — all three use the same "uniform over all lights, uniform over
# triangles, uniform barycentric point" scheme (as opposed to shading.mojo's
# power-weighted light_sampler_sample used by the main path tracer's NEE).


@fieldwise_init
struct AreaLightSample(TrivialRegisterPassable):
    var light:  AreaLight
    var point:  Vec3f
    var normal: Vec3f


@always_inline
def area_light_emit_normal(s: AreaLightSample, mut pcg: PCG32) -> Vec3f:
    """The normal of the face a photon / light path leaves from: the sampled
    one, or for a two-sided light a random face (pdf area_light_side_pdf)."""
    if area_light_side_pdf(s.light) < Float32(1.0) and pcg.next_float() < Float32(0.5):
        return -s.normal
    return s.normal


@always_inline
def area_light_nee_normal(s: AreaLightSample, toward: Vec3f) -> Vec3f:
    """The normal for a shadow-ray connection: a two-sided light emits toward
    the shading point from whichever face looks at it."""
    if area_light_side_pdf(s.light) < Float32(1.0) and dot(s.normal, s.point - toward) > Float32(0.0):
        return -s.normal
    return s.normal


def sample_sphere_light_emission(
    ref sd: SceneView, k: Int, mut pcg: PCG32,
) -> Tuple[Int, Vec3f, Vec3f, Vec3f]:
    """Start of a light subpath on the k-th EMITTING analytic sphere:
    (its index in sd.spheres, a uniform point on its surface -- area density
    1/(4 pi r^2) -- the outward normal there, a cosine-weighted direction
    about that normal). The same emission model mesh lights use, so the
    flux is Le * pi * area / pick. sd.spheres holds every analytic sphere,
    emitters among them, hence the scan for the k-th emitter.

    Shared by SPPM's photons and VCM's light paths. Its NEE counterpart,
    _sample_sphere_light_nee, samples the visible cone instead -- a
    different strategy for the same light, which VCM's MIS weighs against
    this one (see _bdpt_light_path_init's sphere branch)."""
    var si = 0
    var seen = 0
    for i in range(Int(sd.sphereCount)):
        if sd.spheres[unsafe_offset=i].isAreaLight != Int8(0):
            if seen == k:
                si = i
                break
            seen += 1
    var sph = sd.spheres[unsafe_offset=si]
    var us1 = pcg.next_float(); var us2 = pcg.next_float()
    var cz = Float32(1) - Float32(2) * us1
    var sz = sqrt(max(Float32(0), Float32(1) - cz * cz))
    var sphi = Float32(2) * PI * us2
    var sn = Vec3f(sz * cos(sphi), sz * sin(sphi), cz)
    var du1 = pcg.next_float(); var du2 = pcg.next_float()
    var dir = _cosine_hemisphere_sample(sn, du1, du2)
    return (si, sph.center.to_simd() + sn * sph.radius, sn, dir)


@always_inline
def sample_area_light_uniform(
    areaLights: Pointer[AreaLight, MutUntrackedOrigin],
    meshes:     Pointer[TriangleMesh, MutUntrackedOrigin],
    n_lights:   Int,
    mut pcg:    PCG32,
    curves:     Pointer[Curve, MutUntrackedOrigin] = Pointer[Curve, MutUntrackedOrigin].unsafe_dangling(),
) -> AreaLightSample:
    """Uniformly picks one area light, then a point + geometric normal on
    it: a random triangle + barycentric point on a mesh light (kind==0,
    using the mesh's per-vertex shading normals when present — needed for
    e.g. ceiling lights whose winding gives an upward geometric normal but
    whose scene-specified normals point down into the room), or a random
    piece + point on a curve's swept tube (kind==1, `curves` must be a real
    pointer whenever any curve lights exist)."""
    var li = Int(pcg.next_uint() % UInt32(n_lights))
    return sample_area_light_point(areaLights[unsafe_offset=li], meshes, pcg, curves)


def sample_area_light_point(
    al:         AreaLight,
    meshes:     Pointer[TriangleMesh, MutUntrackedOrigin],
    mut pcg:    PCG32,
    curves:     Pointer[Curve, MutUntrackedOrigin] = Pointer[Curve, MutUntrackedOrigin].unsafe_dangling(),
) -> AreaLightSample:
    """A point + geometric normal on the GIVEN area light, uniform in area
    (sample_area_light_uniform without its pick). For a caller that chose
    the light itself -- VCM's power-proportional pick, whose densities
    assume the light it picked is the light that emits."""
    if al.kind == Int8(1):
        var curve = curve_at(curves, Int(al.meshIdx))
        var piece = Int(pcg.next_uint() % UInt32(max(Int(curve.n_pieces), 1)))
        var (q0, q1, r0, r1) = curve_piece_endpoints(curve, piece)
        var axis = q1 - q0
        var axis_len = sqrt(dot(axis, axis))
        var axis_dir = Vec3f(Float32(0.0), Float32(0.0), Float32(1.0))
        if axis_len > Float32(1e-8):
            axis_dir = axis * (Float32(1.0) / axis_len)
        var ru1 = pcg.next_float(); var ru2 = pcg.next_float()
        var r = r0 + (r1 - r0) * ru1
        var u_perp = _curve_perp_axis(axis_dir)
        var v_perp = cross(axis_dir, u_perp)
        var theta = ru2 * (Float32(2.0) * PI)
        var radial = u_perp * cos(theta) + v_perp * sin(theta)
        var point = q0 + axis_dir * (axis_len * ru1) + radial * r
        return AreaLightSample(al, point, radial)
    var lmesh = meshes[unsafe_offset=Int(al.meshIdx)]
    var ti = area_light_pick_triangle(al, pcg.next_float())
    var lb = ti * 3
    var lv0 = Int(mesh_vi(lmesh, lb)); var lv1 = Int(mesh_vi(lmesh, lb+1)); var lv2 = Int(mesh_vi(lmesh, lb+2))
    var lp0 = mesh_pos(lmesh, lv0)
    var lp1 = mesh_pos(lmesh, lv1)
    var lp2 = mesh_pos(lmesh, lv2)
    var ru1 = pcg.next_float(); var ru2 = pcg.next_float(); var sr1 = sqrt(ru1)
    var lp  = lp0*(Float32(1)-sr1) + lp1*(sr1*(Float32(1)-ru2)) + lp2*(sr1*ru2)
    var ln  = cross(lp1-lp0, lp2-lp0)
    var lnl = dot(ln, ln)
    if lnl > Float32(0): ln = ln*(Float32(1)/sqrt(lnl))
    # Use shading normals when provided — they give the correct emission hemisphere.
    if _is_real_ptr(lmesh.normals):
        var sn0 = mesh_normal(lmesh, lv0)
        var sn1 = mesh_normal(lmesh, lv1)
        var sn2 = mesh_normal(lmesh, lv2)
        var sn_avg = (sn0 + sn1 + sn2) / Float32(3)
        var snl = sn_avg.length()
        if snl > Float32(0): ln = (sn_avg / snl).to_simd()
    return AreaLightSample(al, lp, ln)


# ── Camera pass ───────────────────────────────────────────────────────────────


def _spectral_gain_rgb(ref sd: SceneView, w: SpectralSample, wl: SampledWavelengths) -> RGB:
    """A spectral throughput weight as an RGB gain: its film response over that of a flat 1."""
    ref s = sd.spectral
    var (r, g, b) = spectral_sample_to_rgb(s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65, w, wl)
    var (r1, g1, b1) = spectral_sample_to_rgb(s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65, SpectralSample(Float32(1)), wl)
    return RGB(r / max(r1, Float32(1e-6)), g / max(g1, Float32(1e-6)), b / max(b1, Float32(1e-6)))


@always_inline
def _after_glossy_weight(glossy_pdf: Float32, light_pdf: Float32) -> Float32:
    """The share of emission a camera path found right after a glossy vertex, whose NEE sampled it at light_pdf."""
    if glossy_pdf < Float32(0):
        return Float32(1)
    return power_heuristic(glossy_pdf, light_pdf) if glossy_pdf > Float32(0) else Float32(0)


@always_inline
def _sppm_gsms_on(ref sd: SceneView, bsdf: BSDF) -> Bool:
    return (sd.hasGlass & Int32(2)) != Int32(0) and Int(sd.areaLightCount) > 0 and glossy_sms_lobe(bsdf)


@no_inline   # one copy of the solver in the visible-point trace
def _sppm_gsms(
    ref sd: SceneView, bsdf: BSDF, x0: Vec3f, med: Int32, wl: SampledWavelengths, mut pcg: PCG32,
) -> SpectralSample:
    """Glossy SMS from a glossy camera vertex, power-weighted against its lobe sampling the same chain (as the path
    tracer's _glossy_sms_nee). Photons never reach these paths: no visible point sits on the glossy vertex."""
    var z = SpectralSample(Float32(0))
    var (slot, p_pick) = light_path_pick(sd, pcg.next_float())
    if p_pick <= Float32(0) or slot < 0 or slot >= Int(sd.areaLightCount):   # mesh lights only
        return z
    var (ok, y, l_du, l_dv, ln, area, side, emission) = _vcm_mnee_light_point(sd, slot, pcg)
    if not ok:
        return z
    var p_a = p_pick / area
    var (ldu, ldv) = mnee_orthonormal_basis(l_du, l_dv)
    var ctx = _shade_context(sd, Pointer[UInt32, MutUntrackedOrigin].unsafe_dangling())
    var tab = LobeTables(sd.materials, sd.curves, sd.measuredBrdfs, sd.spectral, sd.mediums)
    var verts = SMSChain(fill=sms_vertex_init())
    var r = glossy_sms_solve(ctx, x0, bsdf, tab, wl, y, ldu, ldv, pcg, verts)
    if not r[0]:
        return z
    var n = r[1]
    var cos_y = dot(ln, verts[n-1].pos - y)
    if side < Float32(1):
        cos_y = abs(cos_y)
    if cos_y <= Float32(0):
        return z
    var d0 = verts[0].pos - x0
    var ev = bsdf.f(d0 * (Float32(1) / sqrt(dot(d0, d0))), tab, wl)
    if ev.f_cos.is_black():
        return z
    var t_med = _mnee_chain_transmittance(ctx, x0, verts, n, y, med, wl)
    var w = glossy_sms_mis(ev.pdf_fwd, glossy_sms_t_chain(x0, verts, n) * _mnee_pass(t_med), r[3], p_a)
    return ev.f_cos * t_med * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y,
        sd.spectral.cie_z, sd.spectral.d65, emission.r, emission.g, emission.b, wl) * (r[2] * r[3] * r[4] * w / p_a)


@no_inline
def _sppm_gsms_bsdf_weight(
    ref sd: SceneView, x0: Vec3f, pdf0: Float32, dir0: Vec3f, n: Int, light_hit: Vec3f, light: Intersection, med: Int32,
    wl: SampledWavelengths,
) -> Float32:
    """The share of a camera path that left glossy vertex x0 along dir0 and reached light_hit through n chain vertices:
    the rest of _sppm_gsms' weight when re-tracing it by the chain rule lands there, everything otherwise."""
    var ctx = _shade_context(sd, Pointer[UInt32, MutUntrackedOrigin].unsafe_dangling())
    var verts = SMSChain(fill=sms_vertex_init())
    var rp = glossy_sms_reproduce(ctx, x0, dir0, n, light_hit, light, verts)
    if not rp[0]:
        return Float32(1)
    var al_idx = Int(light.primId.id1)
    var p_a = light_path_pick_pdf(sd, al_idx) / max(sd.areaLights[unsafe_offset=al_idx].total_area, Float32(1e-30))
    var t_med = _mnee_chain_transmittance(ctx, x0, verts, n, light_hit, med, wl)
    return Float32(1) - glossy_sms_mis(pdf0, glossy_sms_t_chain(x0, verts, n) * _mnee_pass(t_med), rp[1], p_a)


@always_inline
def _photon_walk[tex_gpu: Bool](ref sd: SceneView, mat: Material, inter: Intersection, hit: Point3f, refracted: Vec3f,
                                wl: SampledWavelengths, mut pcg: PCG32, mut flux: SpectralSample,
                                outside_ior: Float32) -> Tuple[Vec3f, Point3f, Bool]:
    """A photon through the walk and out of its exit: the next ray, and whether it lives."""
    var g_w = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
    var ex = sss_hop[tex_gpu, adjoint=True](sd, mat, inter, hit.to_simd(), g_w, refracted, wl, pcg, outside_ior)
    if not ex.ok:
        return (vec3f(refracted), hit, False)
    var out = sss_leave(ex.n_o, ex.eta, pcg.next_float(), pcg.next_float())
    flux *= ex.weight * SpectralSample(out.weight)
    var wi = vec3f(out.wi)
    return (wi, ex.x_o + wi * Float32(0.0002), True)


@always_inline
def _vp_at_walk_exit(mut vp: SPPMPixel, ex: SssExit, mat: Material, mat_idx: Int, ref sd: SceneView):
    """The path tracer's walk ended: a visible point at its exit, with the Fresnel exit lobe."""
    vp.beta = vp.beta * _spectral_gain_rgb(sd, ex.weight, vp.wavelengths)
    vp.pos = ex.x_o
    vp.normal = vec3f(ex.n_o)
    vp.geo_normal = vec3f(ex.n_o)
    vp.wo = vec3f(ex.n_o)
    vp.alb = RGB(Float32(1))
    vp.mat_kind = LobeKind.sss_exit
    vp.alpha = ex.eta
    vp.mat_idx = Int32(mat_idx)
    vp.is_volume = PhotonKind.surface
    vp.valid = Int32(1)


@always_inline
def _film_rgb(c: RGB, wl: SampledWavelengths, ref sd: SceneView) -> RGB:
    """Seen radiance through the film's spectral response (a named sensor's curves), as the path tracer sees it."""
    var sp = rgb_illuminant_to_spectral_sample(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, c.r, c.g, c.b, wl)
    var (r, g, b) = spectral_sample_to_rgb(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, sp, wl)
    return RGB(r, g, b)


def _sppm_trace_visible_point[use_gpu: Bool](
    ref sd:       SceneView,
    mut pcg:  PCG32,
    r2c:      Pointer[Float32, MutUntrackedOrigin],
    c2w:      Pointer[Float32, MutUntrackedOrigin],
    px: Int, py: Int,
    pidx:     Int32,
    init_r2:  Float32,
    scratch:  Pointer[Intersection, MutUntrackedOrigin],
    maxdepth: Int,
    film_filter: FilmFilter,
    vp_sample: Int,
    wl_pass: Int = -1,   # >=0: this pass's shared hero wavelengths (per-pass retrace)
) -> SPPMPixel:
    """Trace one primary ray for pixel (px,py), returning its visible point.
    Shared verbatim between the CPU driver (_sppm_camera_pass, [False]) and
    the GPU kernel (sppm_gen_vp_gpu, one slot per thread, [True]) — the
    bounce loop itself has zero CPU/GPU divergence EXCEPT `_tex_lookup`'s
    own CPU-filename vs GPU-texture-array dispatch for diffuse/coateddiffuse
    albedo (task #150/#151: bdpt_*.mojo/sppm.mojo previously never evaluated
    image textures at all, always falling back to a flat grey default).
    `scratch` is caller-owned (no internal alloc/free) so this is safe to
    call from a GPU kernel thread, same convention as bdpt_*.mojo's shared
    subpath tracers."""
    var has_media = Int(sd.mediumCount) > 0
    # Van der Corput value of this VP sample: stratifies the glass reflect/refract
    # choice across the pixel's samples (shifted per pixel and bounce below).
    var vdc = Float32(0)
    var vdc_f = Float32(0.5)
    var vdc_i = vp_sample
    while vdc_i > 0:
        if (vdc_i & 1) != 0:
            vdc += vdc_f
        vdc_i >>= 1
        vdc_f *= Float32(0.5)

    # One hero-wavelength sample for this VP's own camera subpath, carrying
    # its throughput (`beta`), its direct-lighting term (`ld`) and its
    # miss-escape emission (`env`). It is deliberately NOT a per-pass value:
    # SPPM traces visible points ONCE, before any photon pass exists, so a
    # VP's wavelengths cannot agree with the photons'. That is why `tau` is
    # RGB -- see SPPMPixel.tau.
    # Stratified over the pixel's VP samples: a VP keeps these wavelengths for the whole
    # render, so random ones leave a fixed colour error per visible point.
    var vp_wavelengths = sample_wavelengths(_vp_strat_u(vdc, pidx, 1000)) if wl_pass < 0 else pass_wavelengths(wl_pass)
    _ = pcg.next_float()

    var vp = SPPMPixel(
        pos=Point3f(Float32(0)),
        normal=Vec3f(Float32(0), Float32(1), Float32(0)),
        geo_normal=Vec3f(Float32(0), Float32(1), Float32(0)),
        beta=RGB(Float32(1)),
        alb=RGB(Float32(0)),
        tau=RGB(Float32(0)),
        N_acc=Float32(0), r2=init_r2, valid=Int32(0), pidx=pidx,
        is_volume=PhotonKind.surface,
        ld=SpectralSample(Float32(0)),
        env=RGB(Float32(0)),
        gsum=RGB(Float32(0)),
        coat_keep=Float32(1),
        mat_kind=LobeKind.lambertian,
        wo=Vec3f(Float32(0)),
        alpha=Float32(0),
        mat_idx=Int32(-1), hair_curve_idx=Int32(-1), hair_h=Float32(0), hair_v=Float32(0),
        wavelengths=vp_wavelengths,
        med_idx=Int32(-1),
    )

    # Sub-pixel position, drawn through the scene's PixelFilter -- the SAME
    # sampler the path tracer's primary rays use (sampling.mojo). This was a
    # uniform `px + rand` box, so SPPM ignored the filter entirely and rendered
    # visibly sharper than both the path tracer and pbrt. Still diversifies
    # which point on a dielectric-obscured surface (the caustic floor seen
    # through water) each of the vp_samples independent samples lands on.
    var u_fx = pcg.next_float()
    var u_fy = pcg.next_float()
    var (dfx, dfy) = film_filter_offset(u_fx, u_fy, film_filter)
    var fX = Float32(px) + Float32(0.5) + dfx
    var fY = Float32(py) + Float32(0.5) + dfy
    var ul0 = Float32(0.5); var ul1 = Float32(0.5)
    if sd.lens.radius > Float32(0):
        ul0 = pcg.next_float(); ul1 = pcg.next_float()
    var (rd, ro, _) = camera_ray_from_film_xy(fX, fY, r2c, c2w, sd.lens, ul0, ul1)

    # Total path length along the SPECULAR chain only -- a cone tracks the
    # camera's footprint, which survives refraction (the pool floor seen
    # through water) but means nothing after a diffuse scatter. Mirrors
    # accumulate_cone_gpu.
    # No specular gate is needed here, unlike the path tracer's: every
    # non-specular branch of this walk STORES a visible point and breaks, so
    # any segment that continues is specular by construction.
    var cone_len = Float32(0.0)

    var cur_med_idx = Int32(-1)  # camera starts in vacuum
    # Touching-dielectric IOR stack for _dielectric_bounce -- see that
    # function's docstring. Both start at vacuum (1.0).
    var current_dielectric_ior = Float32(1.0)
    var previous_dielectric_ior = Float32(1.0)

    # Depth is counted EXPLICITLY rather than by the loop variable, because a
    # subsurface interior must not be charged for it: the boundary crossings
    # into and out of a `Material "subsurface"` object, and the random-walk
    # scattering events between them, are all ONE subsurface event. Charging them
    # would kill the walk immediately -- head.pbrt declares maxdepth 2, so a
    # path entering skin died after ~2 scatters and no subsurface transport
    # happened at all (SPPM rendered the head flat white; the path tracer,
    # which has had this exemption, renders it correctly). Same rule and same
    # reasoning as shading.mojo's `charge_depth` / Medium.is_sss.
    #
    # `rounds` is only a safety bound so an exempt event cannot loop forever;
    # for a scene with no subsurface medium every event charges, so `bounce`
    # reaches the cap in exactly that many rounds and this is a no-op.
    var max_charged = min(maxdepth, _MAX_B)
    var bounce = 0
    var n_events = 0   # interactions so far, charged or not
    # After a glossy vertex that lit itself: > 0 its density, emission found next shares with its NEE; 0 emission is NEE's
    # alone (pbrt's SPPM); < 0 no such vertex, or a delta vertex since, which NEE could not see through.
    var glossy_pdf = Float32(-1)
    # The last glossy SMS vertex: its lobe density (0: none), position, sampled direction, medium; chain vertices since.
    var gs_pdf = Float32(0)
    var gs_x0 = Vec3f(Float32(0))
    var gs_dir = Vec3f(Float32(0))
    var gs_med = Int32(-1)
    var gs_n = 0
    var dbg = RGB(Float32(0))   # GSMS_DEBUG's only radiance
    while bounce < max_charged + TERMINAL_SEGMENT_GRACE_ROUNDS and n_events < max_charged + SSS_WALK_ROUNDS + TERMINAL_SEGMENT_GRACE_ROUNDS:
        n_events += 1
        # Charged up front, and the two subsurface-exempt branches below undo
        # it locally. Deliberately NOT a `charge_depth` flag applied at the
        # bottom of the body: several branches `continue` from the middle, so
        # a bottom-of-loop increment is silently skipped for exactly the
        # volume-scatter path that matters most.
        bounce += 1
        var ray = Ray(ro, rd)
        scratch[unsafe_offset=0].hit = Int8(0)
        # sd.spheres/sphereCount are REQUIRED: analytic spheres live in their
        # own flat array, not the mesh/curve BVH this walks, so omitting them
        # makes every `Shape "sphere"` invisible to SPPM -- which is exactly
        # why volumetric-caustic's glass sphere, and therefore its entire
        # caustic, was missing from the render until 2026-09-09.
        traverse_dispatch(sd.bvh4, sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1.0e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                           sd.spheres, Int(sd.sphereCount))
        if scratch[unsafe_offset=0].hit == Int8(0):
            for inf_i in range(Int(sd.infiniteLightCount)):
                var ilight = sd.infiniteLights[unsafe_offset=inf_i]
                var (Le, le_pdf) = _eval_infinite_light_and_pdf(ilight, rd, ro.to_simd())
                vp.env += vp.beta * _film_rgb(Le, vp.wavelengths, sd) * _after_glossy_weight(glossy_pdf, le_pdf)
            break

        var inter = scratch[unsafe_offset=0]
        var ray_dir = rd.to_simd()
        var t_hit = inter.tHit
        cone_len += t_hit

        # ── Volume free-flight ────────────────────────────────────────────
        if has_media and Int(cur_med_idx) >= 0:
            # ONE shared sampler for both medium kinds (geometry.mojo):
            # homogeneous closed form, or delta tracking against the real
            # density field. This call site used to be the homogeneous one
            # unconditionally, which rendered every "uniformgrid"/"nanovdb"/
            # "cloud" medium as uniform density-1 fog -- bunny-cloud came out a
            # featureless sphere with no bunny in it. See sample_free_flight.
            var ff = MediumView.at(sd, cur_med_idx).free_flight_emitting(Vec3f(ro.x, ro.y, ro.z), rd, t_hit, pcg, vp.wavelengths)
            var (em_r, em_g, em_b) = spectral_sample_to_rgb(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y,
                                                            sd.spectral.cie_z, sd.spectral.d65, ff.emission, vp.wavelengths)
            vp.env += vp.beta * RGB(em_r, em_g, em_b)
            if ff.collided:
                # Volume scatter — store VP here
                # Chromatic collision weight, as on pass-through and on the photon side.
                vp.beta *= ff.weight
                vp.pos = ro + rd * ff.t_free
                vp.normal = Vec3f(Float32(0), Float32(1), Float32(0))
                vp.geo_normal = Vec3f(Float32(0), Float32(1), Float32(0))
                vp.alb = ff.albedo
                vp.is_volume = PhotonKind.volume
                # wo is what an anisotropic (HG) phase function would need;
                # the gather and NEE both assume isotropic today, but store
                # it so a g != 0 phase is a local change here, not a
                # re-plumbing of the VP struct.
                vp.wo = vec3f((-rd).to_simd())
                vp.med_idx = cur_med_idx
                vp.valid = Int32(1)
                break
            else:
                # Transmittance through full segment to surface
                vp.beta *= ff.weight

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[unsafe_offset=mat_idx]
        var hit = ro + rd * t_hit

        # Direct hit on an emissive analytic sphere — checked BEFORE material
        # dispatch since the sphere's own material is often an inert
        # placeholder (e.g. pbrt's "Null" material on AreaLightSource
        # spheres), so mat.type never reflects that this primitive emits;
        # Sphere.isAreaLight is the only way to know. No MIS needed (unlike
        # bdpt_*.mojo's equivalent fix): this VP sample is either a direct
        # light hit (valid stays 0, credited here) XOR a real gatherable
        # surface that separately does its own NEE — mutually exclusive per
        # sample, same reasoning as the infinite-light miss-escape case
        # below, so there's no competing strategy to double-count against.
        if inter.primId.type == Int8(4):
            var sph_hit = sd.spheres[unsafe_offset=Int(inter.primId.id1)]
            if sph_hit.isAreaLight != Int8(0):
                if glossy_pdf < Float32(0):
                    vp.env += vp.beta * _film_rgb(sph_hit.emission, vp.wavelengths, sd)
                break

        resolve_mix(sd.materials, pcg, mat, mat_idx)

        if mat.type == MatKind.area_light:
            # Direct hit on a triangle/curve area light — same treatment as
            # the sphere case above (no MIS needed, same mutual-exclusivity
            # reasoning). id1 is the AreaLight index directly for a
            # type==3 hit, per pbrt_parser.mojo's own PrimId encoding.
            # Facing check: a one-sided area light emits nothing from its
            # back face (spheres above need no such check — always hit from
            # outside, always the front/emitting face).
            #
            # Emission and facing come from the SHARED resolvers the path
            # tracer and VCM use (shading.mojo). Testing the raw winding normal
            # here disagreed with the side sample_area_light_uniform emits from
            # whenever a mesh's declared normals oppose its winding, so SPPM saw
            # such an emitter as a back face and rendered it black --
            # sss-backlit-slab's emitter fills most of the frame and SPPM's
            # median there was 0. For a curve, id1 is the curve index, not an
            # AreaLight index, and the curve's emission lives in its own
            # material slot; a closed tube is always hit on its outside.
            if inter.primId.type == Int8(5):
                if glossy_pdf < Float32(0):
                    vp.env += vp.beta * _film_rgb(mat.emission, vp.wavelengths, sd)
            else:
                var cos_l = area_light_hit_cos(inter, sd.meshes, sd.instances, ray_dir)
                if cos_l > Float32(0):
                    var al_h = sd.areaLights[unsafe_offset=Int(inter.primId.id1)]
                    # as _sppm_nee_at's: uniform over the lights, then over the light's area
                    var pdf_l = t_hit * t_hit / (cos_l * Float32(sd.areaLightCount) * al_h.total_area)
                    var w_gs = Float32(1)   # share against glossy SMS from the last glossy vertex
                    var gs_shared = gs_pdf >= GLOSSY_SMS_MIN_PDF and gs_n >= 1 and al_h.kind == Int8(0)
                    if gs_shared:
                        w_gs = _sppm_gsms_bsdf_weight(sd, gs_x0, gs_pdf, gs_dir, gs_n, hit.to_simd(), inter, gs_med, vp.wavelengths)
                    var le_h = vp.beta * _film_rgb(al_h.emission, vp.wavelengths, sd)
                    comptime if GSMS_DEBUG >= 2:
                        if gs_shared:
                            dbg += le_h * (w_gs if GSMS_DEBUG == 2 else Float32(1))
                    vp.env += le_h * (_after_glossy_weight(glossy_pdf, pdf_l) * w_gs)
            break

        # See geometry.mojo's TERMINAL_SEGMENT_GRACE_ROUNDS: the loop bound
        # above was widened so the ray fired from the LAST charged bounce --
        # which may simply escape to an infinite light, already handled
        # above -- still gets traced and its emission (if any) collected.
        # That grace round must not become a real bounce of its own: once
        # `bounce` exceeds `max_charged`, a non-emissive hit here refuses to
        # scatter further (no VP store, no continuing dielectric/conductor
        # bounce), exactly as pbrt's own capped path does.
        if bounce > max_charged:
            break

        if mat.uses_lobe():
            if not mat.has_lobe():
                break   # e.g. a measured table that failed to load
            var sh = surface_hit[use_gpu](sd, inter, mat, hit, ray_dir, sd.camFp.cone_spread * cone_len)
            var gn = sh.gn
            var gn_geo = sh.gn_geo   # before any bump/normal map -- see SPPMPixel.geo_normal
            var eff_alb = sh.alb
            var rough_skin = mat.is_rough_dielectric() and sss_walks_below(mat, inter, cur_med_idx)
            var glossy_metal = mat.type == MatKind.conductor and max(mat.roughU, mat.roughV) > Float32(0.02)
            var sampled_glossy = rough_skin or glossy_metal
            if (mat.type == MatKind.hair or sampled_glossy) and bounce < max_charged:
                # pbrt's SPPM: hair, rough skin and glossy metal are glossy, so no visible point on them before the last
                # bounce. Light it here, radiance into env, and scatter on with the sampler's own f/pdf.
                var hv = vp
                hv.pos = hit
                hv.normal = vec3f(gn)
                hv.geo_normal = vec3f(gn_geo)
                hv.alb = eff_alb
                hv.mat_kind = mat.lobe()
                hv.mat_idx = Int32(mat_idx)
                hv.wo = vec3f((-rd).to_simd())
                hv.alpha = mat.lobe_param() if sampled_glossy else Float32(0)
                hv.hair_curve_idx = sh.curve_idx
                hv.hair_h = sh.curve_h
                hv.hair_v = sh.curve_v
                hv.is_volume = PhotonKind.surface
                hv.med_idx = cur_med_idx
                hv.valid = Int32(1)
                var (lr, lg, lb) = spectral_sample_to_rgb(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x,
                    sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, _sppm_nee_at(hv, sd, pcg, sampled_glossy), vp.wavelengths)
                vp.env += vp.beta * RGB(lr, lg, lb)
                if sampled_glossy:
                    var bsdf_g = sh.bsdf(mat, mat_idx)
                    var gsms = glossy_metal and _sppm_gsms_on(sd, bsdf_g)
                    if gsms:
                        var (gr, gg, gb) = spectral_sample_to_rgb(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x,
                            sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65,
                            _sppm_gsms(sd, bsdf_g, hit.to_simd(), cur_med_idx, vp.wavelengths, pcg), vp.wavelengths)
                        vp.env += vp.beta * RGB(gr, gg, gb)
                        comptime if GSMS_DEBUG == 1:
                            dbg += vp.beta * RGB(gr, gg, gb)
                    var s_v = bsdf_g.sample(pcg.next_float(), pcg.next_float(), pcg.next_float(), pcg.next_float(),
                        LobeTables(sd.materials, sd.curves, sd.measuredBrdfs, sd.spectral, sd.mediums), vp.wavelengths)
                    gs_pdf = s_v.pdf_fwd if gsms and s_v.valid and not s_v.is_delta else Float32(0)
                    gs_x0 = hit.to_simd()
                    gs_dir = s_v.wi
                    gs_med = cur_med_idx
                    gs_n = 0
                    if not s_v.valid:
                        break
                    if rough_skin and s_v.transmit:   # a walk leaves into the medium it came from, so no compression
                        var ex_r = sss_hop[use_gpu](sd, mat, inter, hit.to_simd(), _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd()),
                                                    s_v.wi, vp.wavelengths, pcg, current_dielectric_ior)
                        if ex_r.ok:
                            vp.beta = vp.beta * _spectral_gain_rgb(sd, s_v.weight * SpectralSample(mat.albedo.r * mat.albedo.r), vp.wavelengths)
                            _vp_at_walk_exit(vp, ex_r, mat, mat_idx, sd)
                        break
                    vp.beta = vp.beta * _spectral_gain_rgb(sd, s_v.weight, vp.wavelengths)
                    rd = vec3f(s_v.wi)
                    ro = sh.spawn(s_v.wi)
                    glossy_pdf = Float32(-1) if s_v.is_delta else s_v.pdf_fwd
                    continue
                var hc = _hair_precompute(mat, sd.curves, Int(sh.curve_idx), sh.curve_v, sh.curve_h, (-rd).to_simd())
                var hs = _hair_sample_dir_u(hc, pcg.next_float(), pcg.next_float(), pcg.next_float(), pcg.next_float())
                if hs[2] <= Float32(0):
                    break
                vp.beta *= hs[1] * (Float32(1) / hs[2])
                rd = vec3f(hs[0])
                ro = sh.spawn(hs[0])
                glossy_pdf = Float32(0)
                gs_pdf = Float32(0)
                continue
            # A smooth conductor is the one kind here that is genuinely
            # delta: it scatters but stores no visible point, exactly like
            # VCM's Material.is_mirror gate.
            var near_mirror = mat.type == MatKind.conductor and max(mat.roughU, mat.roughV) <= Float32(0.02)
            var coat_as_mirror = False
            if not mat.is_mirror() and not near_mirror:   # near-mirror chrome: the photon map can't resolve its lobe
                var coat_alpha = mat.coat_alpha()
                if mat.lobe() == LobeKind.layered and coat_alpha <= Float32(0.3):
                    # The coat's glossy lobe is too narrow for the gather: sample a microfacet,
                    # reflect off it with probability F and keep tracing, else the VP carries 1/(1-F).
                    var wo_s = (-rd).to_simd()
                    var m_s = gn
                    var g_refl = Float32(1)
                    if coat_alpha > Float32(0.05):
                        var frm_s = Frame.from_z(Vec3f(gn[0], gn[1], gn[2]))
                        var tg_s = Vec3f(frm_s.x.x, frm_s.x.y, frm_s.x.z)
                        var bt_s = Vec3f(frm_s.y.x, frm_s.y.y, frm_s.y.z)
                        var wo_l = Vec3f(dot(wo_s, tg_s), dot(wo_s, bt_s), dot(wo_s, gn))
                        var u_m = pcg.next_float()
                        var wh_l = sample_ggx_vndf(wo_l, coat_alpha, coat_alpha, min(u_m, Float32(0.9999)), pcg.next_float())
                        m_s = tg_s * wh_l.x + bt_s * wh_l.y + gn * wh_l.z
                        var wi_s = m_s * (Float32(2.0) * dot(m_s, wo_s)) - wo_s
                        var cos_i = dot(wi_s, gn)
                        if cos_i <= Float32(0) or dot(wi_s, gn_geo) <= Float32(0):
                            m_s = gn   # sampled reflection dips below the surface: mirror instead
                        else:
                            var lam_o = ggx_lambda(wo_l.z, coat_alpha)
                            g_refl = (Float32(1) + lam_o) / (Float32(1) + lam_o + ggx_lambda(cos_i, coat_alpha))
                    var f_coat = fr_dielectric(dot(m_s, wo_s), mat.emission.r)
                    if pcg.next_float() < f_coat:
                        rd = vec3f(m_s * (Float32(2.0) * dot(m_s, wo_s)) - wo_s)
                        ro = hit + rd * Float32(0.0002)
                        vp.beta *= g_refl
                        glossy_pdf = Float32(-1)
                        gs_pdf = Float32(0)
                        continue
                    vp.beta *= Float32(1.0) / (Float32(1.0) - f_coat)
                    vp.coat_keep = Float32(1.0) - f_coat
                    coat_as_mirror = True
                vp.pos = hit
                vp.normal = vec3f(gn)
                vp.geo_normal = vec3f(gn_geo)
                vp.alb = eff_alb
                # One lobe per material, from Material.lobe --
                # gather and NEE evaluate this VP through lobe_eval, so this
                # branch never asks which material it is again.
                vp.mat_kind = mat.lobe()
                vp.mat_idx = Int32(mat_idx)
                vp.wo = vec3f((-rd).to_simd())
                # smooth top: the gather skips the glossy coat lobe; a negative alpha keeps the real one for delta lights
                vp.alpha = (Float32(1e-4) if coat_alpha <= Float32(0.05) else -coat_alpha) if coat_as_mirror else mat.lobe_param()
                vp.hair_curve_idx = sh.curve_idx
                vp.hair_h = sh.curve_h
                vp.hair_v = sh.curve_v
                vp.is_volume = PhotonKind.surface
                vp.med_idx = cur_med_idx
                vp.valid = Int32(1)
                break
            # Smooth conductor: sample the mirror and keep tracing (RGB --
            # the VP-trace phase has no wavelengths yet, see SPPMPixel.beta).
            var wo_c = (-rd).to_simd()
            var frm_c = Frame.from_z(Vec3f(gn[0], gn[1], gn[2]))
            var gc_c = GeomContext(
                normal=gn, geo_normal=gn, hit_point=hit.to_simd(), wo=wo_c,
                tangent=Vec3f(frm_c.x.x, frm_c.x.y, frm_c.x.z),
                bitangent=Vec3f(frm_c.y.x, frm_c.y.y, frm_c.y.z),
                alb=eff_alb, pixel_uv=Float32(0),
            )
            var bs_c = bxdf_sample_conductor(gc_c, mat, pcg.next_float(), pcg.next_float())
            if bs_c.is_valid == Int8(0):
                break
            vp.beta *= bs_c.f
            glossy_pdf = Float32(-1)
            gs_pdf = Float32(0)
            rd = vec3f(bs_c.wi)
            ro = hit + rd * Float32(0.0002)

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            # Entering, leaving or total-internal-reflecting at the boundary of
            # a subsurface interior is part of the ONE subsurface event the walk
            # inside it belongs to, so it is not charged to maxdepth. Same rule
            # as the path tracer (Material.sss_boundary).
            if mat.sss_boundary != Int8(0):
                bounce -= 1
            var ior = mat.albedo.r
            var gn = _shading_normal_at(inter, sd.meshes, sd.instances, sd.spheres, hit)
            # Bump/normal maps. barcelona's water is a `dielectric` with a
            # "texture displacement", so this is the branch the corpus cares
            # about most. `orient_to` is the RAW interpolated normal, NOT a
            # face-forwarded one: _dielectric_bounce decides entering vs
            # exiting from dot(ray_dir, n) < 0, and flipping the perturbed
            # normal toward the ray would make that test tautological and
            # resurrect the 1/eta^4 transmission loss (same rule as
            # shading.mojo's dielectric site).
            gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes, sd.instances,
                gn, gn, hit.to_simd(), ray_dir, sd.camFp.cone_spread * cone_len, sd.camFp,
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            var (new_dir, new_org, radiance_scale, new_cur_ior, new_prev_ior) = _dielectric_bounce(
                ray_dir, hit.to_simd(), gn, ior, False, pcg, current_dielectric_ior, previous_dielectric_ior, mat.type == MatKind.thin_dielectric,
                not sss_walks_below(mat, inter, cur_med_idx), True)
            if sss_entered(mat, inter, ray_dir, new_dir, cur_med_idx, sd, hit.to_simd()):
                var g_w = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
                var ex = sss_hop[use_gpu](sd, mat, inter, hit.to_simd(), g_w, new_dir, vp.wavelengths, pcg, current_dielectric_ior)
                if ex.ok:
                    vp.beta = vp.beta * radiance_scale
                    _vp_at_walk_exit(vp, ex, mat, mat_idx, sd)
                break
            current_dielectric_ior = new_cur_ior
            previous_dielectric_ior = new_prev_ior
            glossy_pdf = Float32(-1)
            gs_n += 1
            vp.beta *= radiance_scale  # camera-path (Radiance mode): apply non-symmetric-scattering correction
            rd = vec3f(new_dir)
            ro = point3f(new_org)
            if has_media:
                cur_med_idx = medium_after(new_dir, cur_med_idx, inter, mat, sd, hit.to_simd())

        elif mat.type == MatKind.interface:
            # Transparent boundary — update medium, continue ray
            if has_media:
                cur_med_idx = medium_after(ray_dir, cur_med_idx, inter, mat, sd, hit.to_simd())
            ro = hit + rd * Float32(0.0002)

        else:
            break  # area_light, etc.

    comptime if GSMS_DEBUG != 0:
        vp.env = dbg
        vp.valid = Int32(0)
    return vp


def _sppm_store_photon[use_gpu: Bool](
    ph:          SPPMPhoton,
    photons:     Pointer[SPPMPhoton, MutUntrackedOrigin],
    max_photons: Int,
    counter:     Pointer[Int32, MutUntrackedOrigin],
):
    """Reserve the next photon slot and store `ph` there, dropping it if the
    buffer is already full. Comptime-branches only on the slot-reservation
    primitive (atomic fetch-add for racing GPU threads vs. a plain
    increment for the serial CPU loop) — mirrors bdpt_*.mojo's
    _bdpt_store_lvc_vertex[use_gpu] exactly."""
    comptime if use_gpu:
        var slot = Int(Atomic.fetch_add(counter, Int32(1)))
        if slot < max_photons:
            photons[unsafe_offset=slot] = ph
    else:
        var slot = Int(counter[unsafe_offset=0])
        counter[unsafe_offset=0] = Int32(slot + 1)
        if slot < max_photons:
            photons[unsafe_offset=slot] = ph


# The photon pass's bump/normal-map footprint reference, derived from the same
# camera the visible-point pass derives ITS footprint from -- so a photon and
# the visible point it will be gathered by filter the surface the same way.
#
# _sppm_trace_visible_point computes the identical quantity per pixel, where
def _sppm_trace_photon[use_gpu: Bool, tex_gpu: Bool](
    ref sd:               SceneView,
    mut pcg:          PCG32,
    scratch:          Pointer[Intersection, MutUntrackedOrigin],
    n_emit:           Int,
    photons:          Pointer[SPPMPhoton, MutUntrackedOrigin],
    max_photons:      Int,
    counter:          Pointer[Int32, MutUntrackedOrigin],
    default_emit_med: Int32,
    maxdepth: Int,
    # Decomposed spectral tables rather than reading sd.spectral. `sd` is a
    # SceneView passed BY VALUE, and it contains a SpectralHandle --
    # the 6-field TrivialRegisterPassable struct suspected (modular/modular#6759,
    # later retracted by its own author as unreproducible) of corrupting
    # under by-value passing across a real call boundary. Whatever the cause,
    # decomposing fixed a real observed symptom here: measured here, the GPU
    # kernel held sd.spectral.res == 64 immediately before the call and this
    # function read 0 from the same field, silently taking every conversion's
    # table-less path while the photons it was tracing had been built WITH
    # the table. CPU and GPU SPPM then disagreed by 10%, chromatically. This
    # only surfaced when the gather started reading sd.spectral on its common
    # path; before that only the measured-BxDF branch did, which no test
    # scene hit.
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin],
    spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    pass_wl:          SampledWavelengths,
):
    """Emit one photon path from a random light (area, distant, or
    infinite), trace through glass/media, storing at diffuse/volume hits via
    _sppm_store_photon. n_emit is the number of emitted photon PATHS (drives
    the flux-per-photon scale factor); max_photons is the storage buffer's
    capacity, which can be smaller than the number of storage attempts
    since the Russian-roulette diffuse-diffuse continuation (see the
    diffuse-hit branch below) means one emitted path can generate several
    storage attempts, not just one. Shared verbatim between the CPU driver
    (_sppm_photon_pass, one reused `scratch`/plain counter) and the GPU
    kernel (sppm_emit_photons_gpu, one scratch slot + atomic counter per
    thread) — only _sppm_store_photon's comptime branch differs.

    `use_gpu` and `tex_gpu` are DELIBERATELY separate comptime params, not
    one reused symbol: `use_gpu` means "use the atomic-safe photon-slot
    reservation" (both CPU and GPU pass True — parallel CPU workers race on
    the shared `photons` buffer/`counter` too, see _sppm_photon_pass's own
    comment), while `tex_gpu` means "actually running on a GPU device"
    (real CPU/GPU divergence for _tex_lookup's texture-format dispatch,
    task #151). Conflating them would make the CPU driver wrongly try to
    read a GPU-only GpuTexture array.

    Lights are chosen uniformly across ALL light types (area + distant +
    infinite) — `n_lights` below is this combined total. Distant/infinite
    lights have no finite position, so unlike an area light's surface point
    there's nothing meaningful to store as a "direct" photon at the light
    itself (SPPM never did that anyway — only diffuse/volume hits ever get
    stored); a disk-sampled point on the scene's bounding sphere (see
    bvh.mojo's _scene_bounding_sphere/_sample_disk_perpendicular) just seeds
    where this photon's path starts. Direct illumination from these lights
    is provided separately by _sppm_nee_one's own distant/infinite sampling."""
    var has_media = Int(sd.mediumCount) > 0
    var n_area = Int(sd.areaLightCount)
    # Analytic sphere lights are NOT in a pre-filtered lights array (sd.spheres
    # holds every sphere), so they need their own scan and their own slice of
    # the selection pool. They used to be left out of photon emission
    # entirely: a scene lit only by spheres emitted ZERO photons, so SPPM fell
    # back to visible-point NEE alone and lost every multiply-scattered path.
    # Scenes/vcm-media-sphere-light.pbrt read 0.0073 against pbrt's 0.0244
    # that way -- the fog ball kept its directly-lit rim and lost its glow.
    var n_sphere = Int(sd.sphereLightCount)
    var n_distant = Int(sd.distantLightCount)
    var n_infinite = Int(sd.infiniteLightCount)
    var n_point = Int(sd.pointLightCount)
    var n_lights = n_area + n_sphere + n_distant + n_infinite + n_point
    if n_lights == 0:
        return

    var ro: Point3f
    var rd: Vec3f
    # A photon's flux is EMISSION, so it enters the spectral domain through
    # the illuminant curve.
    var flux: SpectralSample
    # This PASS's hero wavelengths, shared by every photon in the pass. It has
    # to be shared: a visible point gathers photons from many different light
    # subpaths and sums them into one running total, so lane i only means the
    # same wavelength in every term if every photon in the pass agrees. Per-
    # photon draws (what this replaced) made that sum meaningless the moment
    # flux stopped being RGB.
    var ph_wavelengths = pass_wl

    # The light, picked by power (light_path_pick, the same table VCM uses):
    # a dim light no longer gets as many photons as one 100x brighter, and
    # every photon carries about the same flux. Each factor below that read
    # n_lights is 1 / this light's pick probability.
    var (light_pick, p_pick) = light_path_pick(sd, pcg.next_float())
    if p_pick <= Float32(0):
        return
    var inv_pick = Float32(1) / p_pick
    if light_pick < n_area:
        # A random triangle + barycentric point on the chosen area light.
        var light_sample = sample_area_light_point(sd.areaLights[unsafe_offset=light_pick], sd.meshes, pcg, sd.curves)
        var al = light_sample.light
        var lp = light_sample.point
        var ln = area_light_emit_normal(light_sample, pcg)

        # Sample cosine-weighted emission direction from the light hemisphere
        # (same Frisvad-frame construction as _cosine_hemisphere_sample).
        var du1 = pcg.next_float()
        var du2 = pcg.next_float()
        var pdir = _cosine_hemisphere_sample(ln, du1, du2)

        # Photon flux: total_light_power / n_emit
        # total_power = emission * pi * total_area / p_pick (the light's pick probability)
        var scale = PI * al.total_area * inv_pick / (area_light_side_pdf(al) * Float32(n_emit))
        flux = spec_illum(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, al.emission.r, al.emission.g, al.emission.b, ph_wavelengths) * scale
        ro = point3f(lp) + vec3f(ln) * Float32(0.0001)
        rd = vec3f(pdir)
    elif light_pick < n_area + n_sphere:
        # Sphere area light: pick the k-th emitting sphere, a uniform point on
        # its surface, and a cosine-weighted outward direction -- the same
        # emission model the mesh branch above uses, with 4*pi*r^2 for the
        # area. Its NEE counterpart is _sample_sphere_light_nee (cone
        # sampling), a different strategy for the same light; SPPM does not
        # MIS the two (photons and NEE cover disjoint path lengths here).
        var (si_e, sp_e, sn_e, sd_e) = sample_sphere_light_emission(sd, light_pick - n_area, pcg)
        var sph_e = sd.spheres[unsafe_offset=si_e]
        var area_e = Float32(4) * PI * sph_e.radius * sph_e.radius
        var scale_e = PI * area_e * inv_pick / Float32(n_emit)
        flux = spec_illum(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                          sph_e.emission.r * scale_e, sph_e.emission.g * scale_e, sph_e.emission.b * scale_e, ph_wavelengths)
        ro = point3f(sp_e + sn_e * (sph_e.radius * Float32(0.0001)))
        rd = vec3f(sd_e)
    elif light_pick < n_area + n_sphere + n_distant:
        var dl = sd.distantLights[unsafe_offset=light_pick - n_area - n_sphere]
        var (center, radius) = _scene_bounding_sphere(sd)
        var dir = Vec3f(dl.direction.x, dl.direction.y, dl.direction.z)
        var disk_pt = _sample_disk_perpendicular(dir, center, radius, Point2f(pcg.next_float(), pcg.next_float()))
        flux = spec_illum(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, dl.emission.r, dl.emission.g, dl.emission.b, ph_wavelengths) * (inv_pick * PI * radius * radius) / Float32(n_emit)
        ro = disk_pt
        rd = dir
    elif light_pick < n_area + n_sphere + n_distant + n_infinite:
        var il = sd.infiniteLights[unsafe_offset=light_pick - n_area - n_sphere - n_distant]
        var (center, radius) = _scene_bounding_sphere(sd)
        # _sample_infinite_light_dir returns env_dir in the NEE convention
        # ("direction FROM a shading point TOWARD the light"). A photon
        # leaving the light travels the opposite way — negate for emission.
        var (env_dir, env_rgb, pdf_dir) = _sample_infinite_light_dir(il, Point2f(pcg.next_float(), pcg.next_float()))
        if pdf_dir <= Float32(0):
            return
        var emit_dir = -env_dir
        var disk_pt = _sample_disk_perpendicular(emit_dir, center, radius, Point2f(pcg.next_float(), pcg.next_float()))
        if not il.portal_admits(disk_pt, emit_dir):
            return
        flux = spec_illum(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, env_rgb.r, env_rgb.g, env_rgb.b, ph_wavelengths) * (inv_pick * PI * radius * radius / pdf_dir) / Float32(n_emit)
        ro = disk_pt
        rd = emit_dir
    else:
        # Point light: has a real position (unlike distant/infinite), so no
        # bounding-sphere disk trick needed — emit directly from it, uniform
        # over the sphere (isotropic point light). pdf_dir = 1/(4π), so
        # flux = intensity × 4π × n_lights / n_emit (pdf_dir cancels).
        var pll = sd.pointLights[unsafe_offset=light_pick - n_area - n_sphere - n_distant - n_infinite]
        var u1p = pcg.next_float(); var u2p = pcg.next_float()
        var cos_p = Float32(1) - Float32(2) * u1p
        var sin_p = sqrt(max(Float32(0), Float32(1) - cos_p * cos_p))
        var phi_p = Float32(2) * PI * u2p
        var pdir_p = Vec3f(sin_p * cos(phi_p), sin_p * sin(phi_p), cos_p)
        flux = spec_illum(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, pll.intensity.r, pll.intensity.g, pll.intensity.b, ph_wavelengths) * (Float32(4) * PI * inv_pick / Float32(n_emit))
        ro = pll.position
        rd = pdir_p
    var cur_med_idx = default_emit_med  # start in medium if light is above one
    # Touching-dielectric IOR stack for _dielectric_bounce -- see that
    # function's docstring. Both start at vacuum (1.0).
    var current_dielectric_ior = Float32(1.0)
    var previous_dielectric_ior = Float32(1.0)

    # Depth is counted EXPLICITLY rather than by the loop variable, because a
    # subsurface interior must not be charged for it: the boundary crossings
    # into and out of a `Material "subsurface"` object, and the random-walk
    # scattering events between them, are all ONE subsurface event. Charging them
    # would kill the walk immediately -- head.pbrt declares maxdepth 2, so a
    # path entering skin died after ~2 scatters and no subsurface transport
    # happened at all (SPPM rendered the head flat white; the path tracer,
    # which has had this exemption, renders it correctly). Same rule and same
    # reasoning as shading.mojo's `charge_depth` / Medium.is_sss.
    #
    # `rounds` is only a safety bound so an exempt event cannot loop forever;
    # for a scene with no subsurface medium every event charges, so `bounce`
    # reaches the cap in exactly that many rounds and this is a no-op.
    var max_charged = min(maxdepth, _photon_depth_cap(sd))
    var bounce = 0
    var n_events = 0   # interactions so far, charged or not
    var n_null = 0     # of those, null-boundary crossings
    while bounce < max_charged + TERMINAL_SEGMENT_GRACE_ROUNDS and n_events < max_charged + SSS_WALK_ROUNDS + TERMINAL_SEGMENT_GRACE_ROUNDS:
        n_events += 1
        # Charged up front, and the two subsurface-exempt branches below undo
        # it locally. Deliberately NOT a `charge_depth` flag applied at the
        # bottom of the body: several branches `continue` from the middle, so
        # a bottom-of-loop increment is silently skipped for exactly the
        # volume-scatter path that matters most.
        bounce += 1
        var ray = Ray(ro, rd)
        scratch[unsafe_offset=0].hit = Int8(0)
        # sd.spheres/sphereCount are REQUIRED: analytic spheres live in their
        # own flat array, not the mesh/curve BVH this walks, so omitting them
        # makes every `Shape "sphere"` invisible to SPPM -- which is exactly
        # why volumetric-caustic's glass sphere, and therefore its entire
        # caustic, was missing from the render until 2026-09-09.
        traverse_dispatch(sd.bvh4, sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1.0e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                           sd.spheres, Int(sd.sphereCount))
        if scratch[unsafe_offset=0].hit == Int8(0):
            break  # miss

        var inter = scratch[unsafe_offset=0]
        var ray_dir = rd.to_simd()
        var t_hit = inter.tHit

        # ── Volume free-flight ────────────────────────────────────────────
        if has_media and Int(cur_med_idx) >= 0:
            var medv = MediumView.at(sd, cur_med_idx)
            # ONE shared sampler for both medium kinds (geometry.mojo):
            # homogeneous closed form, or delta tracking against the real
            # density field. This call site used to be the homogeneous one
            # unconditionally, which rendered every "uniformgrid"/"nanovdb"/
            # "cloud" medium as uniform density-1 fog -- bunny-cloud came out a
            # featureless sphere with no bunny in it. See sample_free_flight.
            var ff = medv.free_flight(Vec3f(ro.x, ro.y, ro.z), rd, t_hit, pcg, ph_wavelengths)
            if ff.collided:
                flux *= medv.weight(ff, t_hit, ph_wavelengths)   # chromatic collision weight, as VCM's light path
                # A scattering event INSIDE a subsurface interior is a step of
                # the subsurface random walk, not a path bounce, so it is not
                # charged to maxdepth -- the same exemption Medium.is_sss
                # buys the path tracer. Skin1 at the scale sssdragon/head use
                # needs tens to hundreds of these before a photon escapes or is
                # absorbed; charging them ends the walk almost immediately.
                if medv.med.is_sss != Int32(0):
                    bounce -= 1
                # Volume scatter — store photon and sample new direction
                var sp = ro + rd * ff.t_free
                # n_events > 1: skip storing at the light's own first
                # segment — that direct contribution is now covered
                # by _sppm_nee_update instead (matches pbrt's own
                # SPPM, which skips photon-grid gathering at depth 0
                # specifically to avoid double-counting with its NEE
                # term).
                if n_events - n_null > 1:
                    _sppm_store_photon[use_gpu](
                        SPPMPhoton(pos=sp, flux=flux, nxt=Int32(-1), is_volume=PhotonKind.volume, dir_in=rd, wavelengths=ph_wavelengths,
                                   curve=Int32(-1)),
                        photons, max_photons, counter)
                # Scatter: the albedo the gather and the path tracer use, sigma_s/sigma_t per wavelength.
                flux *= medv.albedo(ph_wavelengths)
                # Sample new isotropic direction (uniform sphere)
                var hs = medv.sample_phase((-rd).to_simd(), pcg.next_float(), pcg.next_float())
                rd = vec3f(Vec3f(hs[0], hs[1], hs[2]))
                ro = sp + rd * Float32(0.0001)
                continue
            else:
                # Apply Beer-Lambert transmittance through segment. Spectral:
                # upsample sigma_t to the 4 hero lanes FIRST, exponentiate
                # PER LANE -- see spectral_free_flight_weight's docstring for
                # why this differs from (and replaces) band-picking the
                # already-exponentiated RGB ratio.
                flux *= medv.weight(ff, t_hit, ph_wavelengths)

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[unsafe_offset=mat_idx]
        var hit = ro + rd * t_hit

        resolve_mix(sd.materials, pcg, mat, mat_idx)

        # See geometry.mojo's TERMINAL_SEGMENT_GRACE_ROUNDS and the matching
        # guard in _sppm_trace_visible_point above. A photon that simply
        # escapes deposits nothing either way, but without this guard a
        # photon could still scatter off a real material one bounce past
        # maxdepth in the grace round -- kept for consistency with the
        # camera side, not from an isolated repro.
        if bounce > max_charged:
            break

        if mat.uses_lobe():
            if not mat.has_lobe():
                break   # a measured table that did not load
            # The photon has arrived, so it is deposited whatever its continuation; a mirror stores none. Not at the
            # light's first segment: the visible point's NEE covers that (pbrt's SPPM skips depth 0 too).
            if not mat.is_mirror() and n_events - n_null > 1:
                _sppm_store_photon[use_gpu](
                    SPPMPhoton(pos=hit, flux=flux, nxt=Int32(-1), is_volume=PhotonKind.surface, dir_in=rd, wavelengths=ph_wavelengths,
                               curve=Int32(inter.primId.id1) if inter.primId.type == Int8(5) else Int32(-1)),
                    photons, max_photons, counter)
            var sh = surface_hit[tex_gpu](sd, inter, mat, hit, ray_dir, Float32(-1.0))
            var s_l = sh.bsdf(mat, mat_idx, adjoint=True).sample(pcg.next_float(), pcg.next_float(), pcg.next_float(),
                pcg.next_float(), LobeTables(sd.materials, sd.curves, sd.measuredBrdfs, sd.spectral, sd.mediums), ph_wavelengths)
            if not s_l.valid:
                break
            var rr_l = min(Float32(1.0), s_l.weight.max_component())
            if rr_l <= Float32(0.0) or pcg.next_float() >= rr_l:
                break
            flux *= s_l.weight * (Float32(1.0) / rr_l)
            if s_l.transmit and sss_walks_below(mat, inter, cur_med_idx):
                bounce -= 1
                var (wr, or_w, ok_w) = _photon_walk[tex_gpu](sd, mat, inter, hit, s_l.wi, ph_wavelengths, pcg, flux,
                                                     current_dielectric_ior)
                if not ok_w:
                    break
                rd = wr; ro = or_w
                continue
            rd = vec3f(s_l.wi)
            ro = sh.spawn(s_l.wi)
            continue

        if mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            # Entering, leaving or total-internal-reflecting at the boundary of
            # a subsurface interior is part of the ONE subsurface event the walk
            # inside it belongs to, so it is not charged to maxdepth. Same rule
            # as the path tracer (Material.sss_boundary).
            if mat.sss_boundary != Int8(0):
                bounce -= 1
            var ior = mat.albedo.r
            var gn = _shading_normal_at(inter, sd.meshes, sd.instances, sd.spheres, hit)
            # Bump/normal maps. barcelona's water is a `dielectric` with a
            # "texture displacement", so this is the branch the corpus cares
            # about most. `orient_to` is the RAW interpolated normal, NOT a
            # face-forwarded one: _dielectric_bounce decides entering vs
            # exiting from dot(ray_dir, n) < 0, and flipping the perturbed
            # normal toward the ray would make that test tautological and
            # resurrect the 1/eta^4 transmission loss (same rule as
            # shading.mojo's dielectric site).
            gn = apply_surface_maps_at_hit[tex_gpu](mat, inter, sd.meshes, sd.instances,
                gn, gn, hit.to_simd(), ray_dir, Float32(-1.0), sd.camFp,
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            # A visible point at a camera walk's exit gathers what lands here. Its lobe holds the Fresnel, so store before the choice.
            if sss_walks_below(mat, inter, cur_med_idx) and n_events - n_null > 1:
                _sppm_store_photon[use_gpu](
                    SPPMPhoton(pos=hit, flux=flux, nxt=Int32(-1), is_volume=PhotonKind.surface, dir_in=rd,
                               wavelengths=ph_wavelengths, curve=Int32(-1)),
                    photons, max_photons, counter)
            var (new_dir, new_org, _, new_cur_ior, new_prev_ior) = _dielectric_bounce(
                ray_dir, hit.to_simd(), gn, ior, False, pcg, current_dielectric_ior, previous_dielectric_ior, mat.type == MatKind.thin_dielectric, radiance_mode=False)
            if sss_entered(mat, inter, ray_dir, new_dir, cur_med_idx, sd, hit.to_simd()):
                var (wr, or_w, ok_w) = _photon_walk[tex_gpu](sd, mat, inter, hit, new_dir, ph_wavelengths, pcg, flux,
                                                     current_dielectric_ior)
                if not ok_w:
                    break
                rd = wr; ro = or_w
                continue
            current_dielectric_ior = new_cur_ior
            previous_dielectric_ior = new_prev_ior
            # Light path (TransportMode::Importance): do NOT apply the
            # radiance_scale non-symmetric-scattering correction to flux —
            # it's only for camera/Radiance-mode paths, see
            # _dielectric_bounce's docstring.
            rd = vec3f(new_dir)
            ro = point3f(new_org)
            if has_media:
                cur_med_idx = medium_after(new_dir, cur_med_idx, inter, mat, sd, hit.to_simd())

        elif mat.type == MatKind.interface:
            n_null += 1   # a null boundary is not an interaction for the surface "first segment" store gates
            if has_media:
                cur_med_idx = medium_after(ray_dir, cur_med_idx, inter, mat, sd, hit.to_simd())
            ro = hit + rd * Float32(0.0002)

        else:
            # area_light self-hit, etc.: absorb
            break


@always_inline
def _sppm_photon_bucket(k: Int, photons: Pointer[SPPMPhoton, MutUntrackedOrigin], inv_cell: Float32) -> Int:
    var p = photons[unsafe_offset=k].pos
    return _hash_cell(Int(floor(p.x * inv_cell)), Int(floor(p.y * inv_cell)), Int(floor(p.z * inv_cell)))


def _sppm_count_photon(
    k:        Int,
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
):
    grid_count(heads, _sppm_photon_bucket(k, photons, inv_cell))


def _sppm_insert_photon[use_gpu: Bool](
    k:        Int,
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    pass_idx: Int,
):
    """Insert stored photon `k` into the thinned hash grid (see grid_keep),
    after _sppm_count_photon has tallied every photon of the pass."""
    var h = _sppm_photon_bucket(k, photons, inv_cell)
    if not grid_keep(heads, h, pass_idx, photons[unsafe_offset=k].pos):
        return
    photons[unsafe_offset=k].nxt = grid_push[use_gpu](heads, h, k)


@always_inline
def _sppm_photon_on_vp_surface(vp_is_volume: Int32, vp_mat_kind: Int32, vp_normal: Vec3f,
                               e: Vec3f, dist2: Float32, r2: Float32, dir_in: Vec3f) -> Bool:
    """Whether a photon inside the gather ball may count toward a SURFACE
    visible point at all. Kinds whose estimate is not a flat-disk one are
    exempt: hair (a fibre has
    no tangent plane), and volume points (a 3-D estimate by construction).

    An opaque Lambertian point also requires the photon to have ARRIVED from
    the side the point faces -- its f = alb/pi is pulled out of the sum as
    angle-independent, so without this a photon on the back of a thin panel
    lit the front. A transmission lobe legitimately takes both sides, and the
    GGX/measured branches evaluate their BRDF per photon, which is already
    ~0 for a direction below the surface."""
    if vp_is_volume != PhotonKind.surface:
        return True
    if vp_mat_kind == LobeKind.hair:
        return True
    if not gather_disk_contains(e, dist2, r2, vp_normal):
        return False
    if vp_mat_kind == LobeKind.lambertian or vp_mat_kind == LobeKind.sss_exit:
        return dot(dir_in, vp_normal) < Float32(0.0)
    return True


@always_inline
def _vp_phase(ref sd: SceneView, vp: SPPMPixel, wi: Vec3f) -> Float32:
    """A volume visible point's phase function toward wi."""
    return MediumView.at(sd, vp.med_idx).phase(vp.wo.to_simd(), wi)


def _sppm_gather_one(
    vps:      Pointer[SPPMPixel, MutUntrackedOrigin],
    i:        Int,
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    ref sd:       SceneView,
    # Decomposed for exactly the reason the spectral tables below are: `sd` is
    # passed BY VALUE and its struct members do not read back reliably inside
    # the GPU kernel. Reading sd.mediumCount here returned 0 while the driver
    # held the real count.
    pass_wl:  SampledWavelengths,
    pass_idx: Int = 0,
):
    """Gather nearby photons into visible point `i` and apply the SPPM
    radius/flux update. Shared verbatim between the CPU driver
    (_gather_update, one call per pixel in a loop) and the GPU kernel
    (sppm_gather_gpu, one call per thread) — no divergence, each call only
    ever touches its own vps[i]. `sd` is only dereferenced for a mat_kind=2
    (hair) VP, to re-fetch the material + curve data _hair_precompute needs
    (see SPPMPixel's docstring for why that's not stored inline)."""
    if vps[unsafe_offset=i].valid == Int32(0):
        return
    var vp = vps[unsafe_offset=i]
    var spectral_coeffs = sd.spectral.coeffs
    var spectral_res = sd.spectral.res
    var spectral_cie_x = sd.spectral.cie_x
    var spectral_cie_y = sd.spectral.cie_y
    var spectral_cie_z = sd.spectral.cie_z
    var spectral_d65 = sd.spectral.d65
    var r2 = vp.r2
    if vp.is_volume == PhotonKind.volume and Int(vp.med_idx) >= 0:
        var cap = MediumView.at(sd, vp.med_idx).kernel_radius(vp.pos.to_simd())
        cap = max(gather_ball_reach(sd, vp.pos, cap), Float32(0.05) * cap)   # the ball stays inside the medium
        if r2 > cap * cap:   # same history rescale as the update below
            var q = cap * cap / r2
            vp.tau = vp.tau * (q * sqrt(q))
            r2 = cap * cap
            vps[unsafe_offset=i].tau = vp.tau
            vps[unsafe_offset=i].r2 = r2

    # Accumulate contributions from photons in 3x3x3 neighborhood
    # Gathered spectrally at THIS pass's wavelengths, then converted to RGB
    # exactly once on the way into `tau` (see SPPMPixel.tau).
    var phi_rgb = RGB(Float32(0))   # each photon converted at its own wavelengths
    var M = Float32(0)
    var cix = Int(floor(vp.pos.x * inv_cell))
    var ciy = Int(floor(vp.pos.y * inv_cell))
    var ciz = Int(floor(vp.pos.z * inv_cell))
    var buckets = neighbour_buckets(cix, ciy, ciz)
    for b in range(NEIGHBOUR_CELLS):
        var h = buckets[b]
        if h < 0:
            break
        # Survivors of a thinned bucket stand for n/cap photons each
        # (grid_keep): their flux AND their count, so the radius
        # update sees the same photon count in expectation.
        var bucket_w = grid_weight(heads, h)
        var k = Int(heads[unsafe_offset=h])
        while k != -1:
            var ph = photons[unsafe_offset=k]
            ph.flux *= bucket_w
            var e = ph.pos - vp.pos
            var dist2 = e.length_sq()
            # A hair visible point gathers its own strand's photons, every other one no hair photon.
            var same_strand = ph.curve == vp.hair_curve_idx if vp.mat_kind == LobeKind.hair else ph.curve < Int32(0)
            if dist2 <= r2 and ph.is_volume == vp.is_volume and same_strand and _sppm_photon_on_vp_surface(
                    Int32(vp.is_volume), Int32(vp.mat_kind), vp.geo_normal.to_simd(), e.to_simd(), dist2, r2,
                    ph.dir_in.to_simd()):
                # Volume VP: isotropic phase f=alb/(4π). Surface VP:
                # Lambertian f=alb/π (angle-independent, pulled out
                # of the sum) or, for a conductor VP, the raw GGX
                # BRDF evaluated per-photon (wi reconstructed from
                # the photon's stored incoming travel direction) —
                # NOT multiplied by any extra cosine, since the
                # photon's flux already encodes the appropriate
                # cosine-weighted density (same convention the
                # Lambertian/phase branches already rely on).
                if vp.is_volume == PhotonKind.volume:
                    if Int(vp.med_idx) >= 0:   # no `continue`: k advances at the bottom of this loop
                        var term_2 = MediumView.at(sd, vp.med_idx).photon_scale(vp.pos.to_simd(), ph.wavelengths) \
                               * _vp_phase(sd, vp, -ph.dir_in.to_simd()) * ph.flux
                        var (tr_2, tg_2, tb_2) = spectral_sample_to_rgb(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, term_2, ph.wavelengths)
                        phi_rgb += RGB(tr_2, tg_2, tb_2)
                else:
                    # The VP's BSDF, bare f: the density estimate carries the cosine.
                    var le = (BSDF(vp.mat_kind, True, False, vp.normal.to_simd(), vp.wo.to_simd(), vp.alb,
                                vp.mat_idx, _vp_base_alpha(vp.alpha), Float32(0), vp.hair_curve_idx,
                                vp.hair_h, vp.hair_v, True, False)).f[want_pdfs=False]((-ph.dir_in).to_simd(), LobeTables(sd.materials, sd.curves, sd.measuredBrdfs, sd.spectral, sd.mediums), ph.wavelengths)
                    if le.cos_used > Float32(1e-6):
                        var term_3 = le.f_cos * (Float32(1.0) / le.cos_used) * ph.flux
                        var (tr_3, tg_3, tb_3) = spectral_sample_to_rgb(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, term_3, ph.wavelengths)
                        phi_rgb += RGB(tr_3, tg_3, tb_3)
                M += bucket_w
            k = Int(ph.nxt)

    # SPPM update (only if new photons found)
    if M > Float32(0.0):
        # Truncation: this pass's photons came only from the part of the
        # gather disk that is surface (gather_disk_coverage, which VCM's merge
        # shares). The flux is per pass, at this pass's radius, so each pass's
        # contribution to tau is divided by its own coverage. Volume points
        # have no disk, and a hair fibre no tangent plane.
        if vp.is_volume != PhotonKind.volume and vp.mat_kind != LobeKind.hair:
            # 16 probes measured no better than 8 here (barcelona-pavilion
            # night: 99-99.9% bin WORSE, 0.998->1.012 ->1.096; top 0.1%
            # unchanged) -- the residual vs. an unbiased BDPT referee (whole
            # 0.998, mid-bin +5%, brightest -8%) is not probe-count noise.
            # Left at 8, matching VCM's merge (the open SPPM-specific residual is
            # described in gather_disk_coverage's docstring).
            var cov = gather_disk_coverage(sd, vp.pos, vp.geo_normal, sqrt(r2), salt=7 + pass_idx * 1000003)
            if cov > Float32(0):
                phi_rgb = phi_rgb * (Float32(1) / cov)
        if vp.mat_kind == LobeKind.hair:
            # The fibre kernel measures 2 r w, not pi r^2: rescale this pass into the disk convention tau carries.
            var w = Float32(2) * curve_radius(sd.curves, Int(vp.hair_curve_idx), vp.hair_v)
            phi_rgb = phi_rgb * (disk_measure(r2) / max(ribbon_measure(r2, w), Float32(1e-20)))
        var N = vp.N_acc
        var ratio = (N + _ALPHA * M) / (N + M)
        vps[unsafe_offset=i].r2  = r2 * ratio
        var (phi_r, phi_g, phi_b) = (phi_rgb.r, phi_rgb.g, phi_rgb.b)
        # tau must be rescaled by the KERNEL's own dimension, because tau is
        # later divided by that kernel. pbrt scales by Sqr(rNew)/Sqr(radius)
        # because its estimator divides by pi*r^2 -- it has only surface
        # visible points. Ours divides a VOLUME visible point by (4/3)pi*r^3,
        # so its history has to be rescaled by (rNew/r)^3, not (rNew/r)^2.
        # With ratio = (rNew/r)^2, that is ratio^1.5 for a volume VP and
        # ratio for a surface one. Using the surface rescale on a volume
        # kernel leaves a per-pass factor of ratio^-0.5 that COMPOUNDS over
        # every pass -- a systematic bias, which is why more photons never
        # moved the result.
        var tau_scale = ratio * sqrt(ratio) if vp.is_volume == PhotonKind.volume else ratio
        vps[unsafe_offset=i].tau = (vp.tau + vp.beta * RGB(phi_r, phi_g, phi_b)) * tau_scale
        vps[unsafe_offset=i].N_acc = N + _ALPHA * M


@always_inline
def _vp_base_alpha(alpha: Float32) -> Float32:
    """The lobe parameter for the gather and area/infinite NEE: a split-coat VP (alpha < 0, see
    _sppm_trace_visible_point) leaves the coat's glossy lobe to the continued camera path."""
    return Float32(1e-4) if alpha < Float32(0) else alpha


@always_inline
def _sppm_nee_weight(
    vp: SPPMPixel,
    ref sd: SceneView,
    vn: Vec3f,
    wo: Vec3f,
    ls: LightSample,
    alpha: Float32,
    shares: Bool = False,
) -> SpectralSample:
    """One distant, point, sphere or infinite light sample's NEE weight at a visible point, through its BSDF."""
    if vp.is_volume == PhotonKind.volume:
        # The phase function, and NO cosine factor -- a
        # volume vertex has no normal. Same convention the photon gather
        # already uses for volume VPs (see _sppm_gather_one), so the two
        # estimators agree on what a volume VP scatters.
        #
        # Deliberately NO MIS weight, unlike _nee_weight_simple_spectral's
        # sphere/infinite branch: MIS discounts a light sample by the chance
        # a competing BSDF/phase-sampling strategy would have found the same
        # path, and an SPPM visible point has no such strategy -- the VP
        # TERMINATES at this scatter (see _sppm_trace_visible_point's volume
        # branch, which stores and breaks). Weighting here would discard that
        # share to nobody. This matches the area-light branch of
        # _sppm_nee_one, which likewise applies none.
        if not ls.valid or ls.pdf <= Float32(0.0):
            return SpectralSample(Float32(0))
        var ph_v = MediumView.at(sd, vp.med_idx).albedo(vp.wavelengths) * _vp_phase(sd, vp, ls.wi)
        var li_v = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65,
                              ls.Li.r, ls.Li.g, ls.Li.b, vp.wavelengths)
        return ph_v * li_v * (Float32(1.0) / ls.pdf)
    # THE NEE weight (bxdf.mojo), from the SAME BSDF the gather and every
    # other consumer of a stored vertex uses -- one call in place of the
    # separate hair/measured branches this used to hand-dispatch to
    # (_nee_weight_hair, _nee_weight_measured) plus a third call for
    # everything else (_nee_weight_simple_spectral). mis_policy_sole(), NOT
    # the power-heuristic default: an SPPM visible point terminates the
    # camera path, and direct photons are gated out of the map, so NEE is
    # the only estimator of direct light here -- every kind gets this now;
    # measured never did before (barcelona-pavilion-day's shadowed-chair
    # deficit), and this is the same fix applied uniformly rather than at
    # one material's call site.
    return (BSDF(vp.mat_kind, True, False, vn, wo, vp.alb, vp.mat_idx, alpha,
                Float32(0), vp.hair_curve_idx, vp.hair_h, vp.hair_v, True, False)).nee_weight_mis(ls, LobeTables(sd.materials, sd.curves, sd.measuredBrdfs, sd.spectral, sd.mediums), vp.wavelengths, mis_policy_power() if shares else mis_policy_sole())


@always_inline
def _sppm_vp_shadow_eps(vp: SPPMPixel, ref sd: SceneView, wo: Vec3f) -> Float32:
    """Shadow-ray self-intersection offset for a stored SPPM visible point: none on hair, as in shade_hair."""
    if vp.mat_kind == LobeKind.hair:
        return Float32(0)   # from the hit itself, as shade_hair: an offset origin let the strand shadow its own light
    return offset_eps(vp.pos.x, vp.pos.y, vp.pos.z)


@always_inline
def _sppm_simple_light_count(ref sd: SceneView) -> Int:
    """Sibling of shading.mojo's _nee_simple_light_count / bdpt_*.mojo's
    _bdpt_simple_light_count -- a third, independent definition rather than
    a shared one, because sppm.mojo is typed against SceneView (like
    bdpt_*.mojo) but needs the (LightSample, tmax) PAIRING (like shading.mojo's
    ShadeContext version), and bdpt_*.mojo already imports from sppm.mojo (for
    _sppm_trace_visible_point etc.), so importing back the other way would be
    circular. distant + point + sphere; area and infinite stay at their own
    call sites, same rationale as the other two files."""
    return Int(sd.distantLightCount) + Int(sd.pointLightCount) + Int(sd.sphereCount)


@always_inline
def _sppm_sample_simple_light(
    ref sd: SceneView, i: Int, hit_point: Vec3f, mut pcg: PCG32,
) -> Tuple[LightSample, Float32]:
    """The i-th distant/point/sphere light, plus the any_hit_bvh2_core tmax
    to use for it -- same pairing shading.mojo's _nee_sample_simple_light
    centralizes, getting it wrong here is exactly the 08246179 bug (tmax
    computed from the wrong reference point overshot into the light itself).
    ORDER IS LOAD-BEARING: distant, then point, then sphere -- matching
    _sppm_nee_one's own existing order exactly, so converting its loop to
    this iterator is a pure collapse, not a reordering."""
    var nd = Int(sd.distantLightCount)
    var np_ = Int(sd.pointLightCount)
    if i < nd:
        var ls_d = _sample_distant_light_nee(sd.distantLights[unsafe_offset=i])
        var d = ls_d.dist
        return (ls_d^, d)
    if i < nd + np_:
        var ls_p = _sample_point_light_nee(sd.pointLights[unsafe_offset=i - nd], hit_point)
        var d = ls_p.dist * Float32(0.9999)
        return (ls_p^, d)
    var si = i - nd - np_
    var ls_s = _sample_sphere_light_nee(sd.spheres[unsafe_offset=si], Int(sd.sphereCount), hit_point, pcg)
    var d = ls_s.dist * Float32(0.9999)
    return (ls_s^, d)


@always_inline
def _sppm_shadow(
    vp: SPPMPixel, ref sd: SceneView, org: Point3f, wi: Vec3f, tmax: Float32, mut pcg: PCG32,
) -> SpectralSample:
    """Visibility times transmittance of a VP shadow ray: black if blocked, 1 in a medium-free scene."""
    if Int(sd.mediumCount) > 0:
        return shadow_transmittance(org.to_simd(), wi, tmax, vp.med_idx, sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves,
                                    sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances, sd.spheres, Int(sd.sphereCount),
                                    sd.materials, sd.mediums, sd.mediumInterfaces, sd.grids, sd.nvdbGrids,
                                    sd.spectral, vp.wavelengths, pcg)
    if any_hit_dispatch(sd.bvh4, sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, Ray(org, vec3f(wi)), tmax,
                         sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances, sd.spheres, Int(sd.sphereCount),
                         materials=sd.materials):
        return SpectralSample(Float32(0.0))
    return SpectralSample(Float32(1.0))


def _sppm_nee_at(
    vp:      SPPMPixel,
    ref sd:      SceneView,
    mut pcg: PCG32,
    shares: Bool = False,   # a glossy vertex: BSDF hits on area and infinite lights share them by MIS (_after_glossy_weight)
) -> SpectralSample:
    """Direct (NEE) lighting update for visible point `i`. Shared verbatim
    between the CPU driver (_sppm_nee_update, one call per VP in a loop) and
    the GPU kernel (sppm_nee_gpu, one call per thread) — no divergence, each
    call only ever touches its own visible point. Samples area lights (one random
    CDF-uniform pick), distant lights, and infinite lights (all of the
    latter two — few and typically dominant, matching shading.mojo's own
    NEE asymmetry rationale) — no MIS weighting needed for infinite lights
    here, unlike bdpt_*.mojo's camera-path NEE: a VP is only ever `valid` when
    its ONE traced ray hit a real surface, and _sppm_trace_visible_point's
    own miss-escape env contribution (vp.env) only fires when it didn't —
    mutually exclusive per sample, so there's no competing strategy to
    double-count against."""
    var ld = SpectralSample(Float32(0))
    if vp.valid == Int32(0):
        return ld
    var is_vol = vp.is_volume == PhotonKind.volume
    var vn   = vp.normal.to_simd()
    var wo   = vp.wo.to_simd()
    # A volume scatter point has no surface to self-intersect against, so it
    # needs no normal-offset epsilon -- and applying one would displace the
    # shadow origin along an arbitrary placeholder normal (volume VPs store
    # +Y), biasing every volume NEE sample by that offset.
    var shadow_eps = Float32(0) if is_vol else _sppm_vp_shadow_eps(vp, sd, wo)
    # EVERY shadow ray below starts at this offset point, so every light
    # sample must be measured FROM it too. Sampling a light from vp.pos while
    # firing the ray from vp.pos + n*eps makes the ray overshoot the light by
    # the offset's along-ray component (eps*cos_surface), so a tmax of
    # dist*0.999 runs INTO the light and reports a false occlusion whenever
    #     eps*cos_surface >= 0.001*dist   <=>   dist <= 1000*eps*cos_surface.
    # For a sphere light (real geometry, unlike a point/distant/infinite one)
    # with the 0.999 factor that is dist <= 0.1, and with the 0.9999 factor
    # used by the per-light-type blocks it is dist <= cos_surface, i.e. about
    # a full unit -- those lights were simply black. Same defect, same cause,
    # as the volume NEE one fixed in f79999f4; shading.mojo's surface NEE
    # avoids it by passing its own offset `hit_point` to the samplers, which
    # is what this mirrors.
    # Offset along the GEOMETRIC normal, like pbrt's OffsetRayOrigin: the
    # shading normal is turned toward wo (face_toward) and can point into the
    # surface, which would start every shadow ray underneath it.
    var shadow_org = vp.pos + vp.geo_normal * shadow_eps
    var spos = shadow_org.to_simd()
    # A two-sided lobe transports to BOTH sides, so the shadow-ray offset has
    # to follow the LIGHT DIRECTION rather than the normal. Offsetting along
    # +n for a direction on the -n side starts the ray on the wrong side and
    # it self-occludes on the surface it just left -- which is why adding the
    # diffuse_transmit lobe changed the furnace by EXACTLY nothing at first:
    # the evaluator returned the right value and visibility threw it away.
    # One-sided lobes only ever see cos > 0, so this is a no-op for them.
    # Lobes that transmit see lights on both sides; their shadow rays leave from the light's side.
    var two_sided_vp = vp.mat_kind == LobeKind.diffuse_transmit or vp.mat_kind == LobeKind.rough_dielectric \
        or vp.mat_kind == LobeKind.hair
    var shadow_org_back = vp.pos + vp.geo_normal * (-shadow_eps)

    var n_area = Int(sd.areaLightCount)
    if n_area > 0:
        # Pick a random area light + triangle + point on it (same scheme as
        # _sppm_trace_photon's emission sampling).
        var light_sample = sample_area_light_uniform(sd.areaLights, sd.meshes, n_area, pcg, sd.curves)
        var al = light_sample.light
        var lp = light_sample.point
        var ln = area_light_nee_normal(light_sample, spos)

        var to_light = lp - spos
        var dist2 = dot(to_light, to_light)
        var dist = sqrt(dist2)
        if dist > Float32(0.0):
            var wi = to_light * (Float32(1.0) / dist)
            # No cosine at a volume vertex (it has no normal); the isotropic
            # phase function replaces the BRDF below.
            var cos_surface = Float32(1.0) if is_vol else dot(vn, wi)
            var cos_light = -dot(ln, wi)
            if (cos_surface > Float32(0.0) or two_sided_vp) and cos_light > Float32(0.0):
                # Shadow ray, offset from both ends to avoid self-intersection.
                var org_a = shadow_org_back if (two_sided_vp and cos_surface < Float32(0)) else shadow_org
                var tr_a = _sppm_shadow(vp, sd, org_a, wi, dist * Float32(0.999), pcg)
                if not tr_a.is_black():
                    # pdf_area = 1/(n_area * total_area) — uniform-over-all-lights
                    # assumption, same as the emission-sampling flux scale factor.
                    var inv_pdf_area = Float32(n_area) * al.total_area
                    var geom = cos_light / dist2 * inv_pdf_area
                    if is_vol:
                        ld += (MediumView.at(sd, vp.med_idx).albedo(vp.wavelengths) * _vp_phase(sd, vp, wi)
                                      * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al.emission.r, al.emission.g, al.emission.b, vp.wavelengths)
                                      * geom * tr_a)
                    else:
                        # The lobe's own f*cos: its cosine is hair's fibre cosine, not the curve normal's.
                        var bsdf_vp = BSDF(vp.mat_kind, True, False, vn, wo, vp.alb,
                                    vp.mat_idx, _vp_base_alpha(vp.alpha), Float32(0), vp.hair_curve_idx,
                                    vp.hair_h, vp.hair_v, True, False)
                        var tab_vp = LobeTables(sd.materials, sd.curves, sd.measuredBrdfs, sd.spectral, sd.mediums)
                        var le_vp = bsdf_vp.f[want_pdfs=False](wi, tab_vp, vp.wavelengths)
                        if shares and al.kind == Int8(0):   # a curve light's hit takes no share: NEE keeps it whole
                            geom *= power_heuristic(dist2 / (cos_light * inv_pdf_area), bsdf_vp.f[want_pdfs=True](wi, tab_vp, vp.wavelengths).pdf_fwd)
                        var light_spec = rgb_illuminant_to_spectral_sample(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al.emission.r, al.emission.g, al.emission.b, vp.wavelengths)
                        ld += le_vp.f_cos * light_spec * geom * tr_a

    # Distant/point/sphere/infinite NEE through the Light interface and the BSDF (_sppm_nee_weight).
    # distant/point/sphere via the shared sampler -- pure loop collapse,
    # order was already distant,point,sphere. Sphere-light NEE loops ALL
    # analytic spheres (the sampler itself skips non-emissive ones), since
    # sd.spheres/sphereCount is the raw geometric array, not a pre-filtered
    # lights-only one like every other light type.
    var n_delta = Int(sd.distantLightCount) + Int(sd.pointLightCount)
    for li in range(_sppm_simple_light_count(sd)):
        var res = _sppm_sample_simple_light(sd, li, spos, pcg)
        var ls = res[0].copy()
        var tmax = res[1]
        var w = _sppm_nee_weight(vp, sd, vn, wo, ls, _vp_base_alpha(vp.alpha))
        if vp.alpha < Float32(0) and li < n_delta:
            # split coat: the camera path cannot hit a delta light, so NEE takes its glossy highlight
            w = w + (_sppm_nee_weight(vp, sd, vn, wo, ls, -vp.alpha) - w) * vp.coat_keep
        if not w.is_black():
            var s_org = shadow_org_back if (two_sided_vp and dot(vp.geo_normal.to_simd(), ls.wi) < Float32(0)) else shadow_org
            var tr_s = _sppm_shadow(vp, sd, s_org, ls.wi, tmax, pcg)
            if not tr_s.is_black():
                ld += w * tr_s

    for inf_i in range(Int(sd.infiniteLightCount)):
        var ls_e = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_i], Point2f(pcg.next_float(), pcg.next_float()), vp.pos)
        var w_e = _sppm_nee_weight(vp, sd, vn, wo, ls_e, _vp_base_alpha(vp.alpha), shares)
        if not w_e.is_black():
            var s_org_e = shadow_org_back if (two_sided_vp and dot(vp.geo_normal.to_simd(), ls_e.wi) < Float32(0)) else shadow_org
            var tr_e = _sppm_shadow(vp, sd, s_org_e, ls_e.wi, ls_e.dist, pcg)
            if not tr_e.is_black():
                ld += w_e * tr_e
    return ld


def _sppm_nee_one(
    vps:     Pointer[SPPMPixel, MutUntrackedOrigin],
    i:       Int,
    ref sd:      SceneView,
    mut pcg: PCG32,
):
    """_sppm_nee_at for visible point `i`."""
    vps[unsafe_offset=i].ld += _sppm_nee_at(vps[unsafe_offset=i], sd, pcg)


@always_inline
def _sppm_flush_one(
    vps: Pointer[SPPMPixel, MutUntrackedOrigin], i: Int,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin], spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin], spectral_d65: Pointer[Float32, MutUntrackedOrigin],
):
    """Moves this pass's direct-light term (ld, spectral at the VP's wavelengths) or, for a ray
    that escaped, its env radiance into the beta-weighted RGB sum gsum. Linear, so equal to
    converting the accumulated ld once at the end, and it lets the VP change every pass."""
    var vp = vps[unsafe_offset=i]
    if vp.valid == Int32(0):
        vps[unsafe_offset=i].gsum = vp.gsum + vp.env
        return
    var (dr, dg, db) = spectral_sample_to_rgb(
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, vp.ld, vp.wavelengths)
    # env is radiance the camera path already weighted: escapes, and the light at glossy vertices it passed.
    vps[unsafe_offset=i].gsum = vp.gsum + vp.env + vp.beta * RGB(dr, dg, db)
    vps[unsafe_offset=i].ld = SpectralSample(Float32(0))


# ── Finalize ──────────────────────────────────────────────────────────────────


@always_inline
def _sppm_finalize_albedo_one_pixel(
    vps:        Pointer[SPPMPixel, MutUntrackedOrigin],
    i:          Int,
    vp_samples: Int,
) -> RGB:
    """Averages the first-hit material albedo AOV across vp_samples VP
    samples for pixel i — same shape as _sppm_finalize_one_pixel, but for the
    denoiser's albedo guide buffer. SPPMPixel.alb is set once per VP sample
    at its first non-specular hit, in _sppm_trace_visible_point — already
    exactly the quantity the denoiser wants, no separate tracking needed
    (unlike bdpt_*.mojo, which had to add a new return value for this).
    Shared between the CPU driver (sppm_render) and the GPU finalize kernel
    (sppm_finalize_gpu)."""
    var acc = RGB(Float32(0))
    for vs in range(vp_samples):
        var vp = vps[unsafe_offset=i * vp_samples + vs]
        if vp.valid != Int32(0):
            acc += vp.alb
    return acc / Float32(vp_samples)


def _sppm_finalize_one_pixel(
    vps:        Pointer[SPPMPixel, MutUntrackedOrigin],
    i:          Int,
    vp_samples: Int,
    n_passes:   Int32,
    iso_scale:  Float32,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin],
    spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
) -> Tuple[RGB, RGB]:
    """Averages the vp_samples independently-converged samples for pixel i,
    returning (global, caustic) SEPARATELY instead of one summed RGB --
    see project_water_caustic_sppm_gap memory. The two have very different
    noise character: `global` (NEE direct lighting + escaped/env rays)
    behaves like ordinary path-traced noise and denoises fine; `caustic`
    (the photon-gather term, vp.tau) is a real, legitimately high-frequency
    spatial signal (a caustic thread's whole point is 1-2px bright lines on
    a dim background) that a generic spatial-variance edge-stopping filter
    cannot tell apart from noise -- it measured an ~85x energy loss on
    water-caustic's brightest pixels. Keeping the two separate lets the
    caller denoise ONLY `global` and add `caustic` back afterward,
    unsmoothed, instead of disabling denoising for the whole image (which
    is what an earlier version of this fix did, and loses real GI
    denoising on ordinary non-caustic SPPM scenes for no reason). This is
    the same caustic/global decomposition "Denoising Stochastic Progressive
    Photon Mapping Renderings Using a Multi-Residual Network" (Zheng et
    al., JCST 2020) uses for exactly this reason.

    Neither RGB carries the max-component sensor clamp -- that must be
    applied ONCE, by the caller, to (global + caustic) after global is
    denoised, not to each half separately (clamping each half against the
    same absolute limit independently would under-clamp their sum). Each
    IS NaN/negative-guarded here, since that guard is per-component and
    order-independent.

    Shared verbatim between the CPU driver (sppm_render's tail loop) and
    the GPU kernel (sppm_finalize_gpu)."""
    var acc_g = RGB(Float32(0))
    var acc_c = RGB(Float32(0))
    for vs in range(vp_samples):
        var vp = vps[unsafe_offset=i * vp_samples + vs]
        # Direct light and escape radiance, summed per pass by _sppm_flush_one.
        acc_g += vp.gsum / Float32(n_passes)
        # tau is kept even when this pass's VP is invalid: with per-pass retrace the last
        # visible point may have escaped while earlier passes gathered photons.
        if vp.N_acc > Float32(0.0) and vp.r2 > Float32(0.0):
            # Surface: L = tau / (pi * r^2 * n_passes) -- a photon lands ON a
            # surface, so the estimator normalises by the DISK AREA the search
            # radius sweeps out. tau already has albedo/pi folded in at gather
            # time.
            #
            # Volume: photons land THROUGHOUT a medium, so the same estimator
            # must normalise by the SPHERE VOLUME (4/3)pi r^3 instead -- the
            # standard volumetric photon-map density. Using the surface kernel
            # here understates a volume gather by (4/3)r, which at this
            # scene's r=0.05 is ~15x too dark. That went unnoticed because
            # volume VPs had no working direct-lighting term to compare
            # against until the same session fixed that.
            var denom = disk_measure(vp.r2) * Float32(n_passes)
            if vp.is_volume == PhotonKind.volume:
                denom = ball_measure(vp.r2) * Float32(n_passes)
            acc_c += vp.tau / denom
    acc_g = acc_g / Float32(vp_samples)
    acc_c = acc_c / Float32(vp_samples)

    # ISO exposure compensation (matches normalize_film) -- linear, so
    # applying it to each half separately and later summing them is exact.
    acc_g *= iso_scale
    acc_c *= iso_scale

    # NaN guard only (a negative is out-of-gamut colour the sensor matrix needs; apply_film_sensor clamps) -- safe to do
    # before the caller sums the two halves.
    if acc_g.r != acc_g.r: acc_g.r = Float32(0)
    if acc_g.g != acc_g.g: acc_g.g = Float32(0)
    if acc_g.b != acc_g.b: acc_g.b = Float32(0)
    if acc_c.r != acc_c.r: acc_c.r = Float32(0)
    if acc_c.g != acc_c.g: acc_c.g = Float32(0)
    if acc_c.b != acc_c.b: acc_c.b = Float32(0)
    return (acc_g, acc_c)


# ── Public entry point ────────────────────────────────────────────────────────
