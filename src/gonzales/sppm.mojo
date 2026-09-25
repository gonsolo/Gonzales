# Stochastic Progressive Photon Mapping — CPU driver + GPU kernels, sharing
# one comptime[use_gpu]-parameterized core (same pattern as bdpt.mojo's
# LVC-BPT port; see project_unified_renderer_roadmap in memory).
# Reference: Hachisuka et al. 2008 "Progressive Photon Mapping"

from std.sys import has_accelerator
from std.sys.info import size_of
from max.gpu import block_idx, thread_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from max.algorithm import parallelize
from std.math import sqrt, cos, sin, floor, log, exp, max, min, ceildiv
from std.memory.alloc import unsafe_alloc
from std.atomic import Atomic
from .geometry import face_toward, TERMINAL_SEGMENT_GRACE_ROUNDS, RGB, Point3f, Point2f, Vec3f, vec3f, point3f, dot, cross, PI, INV_FOUR_PI, Frame, _is_real_ptr
from .materials import Material_C, MatKind, LobeKind, PhotonKind, fr_dielectric, MeasuredBRDF_C
from .render_state import GpuTexture
from .primitives import Ray, Intersection, PrimId, TriangleMesh, Sphere, Instance, sphere_outward_normal
from .media import Medium, MediumInterface, Grid, NvdbGrid, FreeFlight, sample_homogeneous_free_flight, sample_free_flight, medium_is_heterogeneous, medium_sigma_t_spectral, medium_grid_for, medium_nvdb_for, grid_sample_density, nvdb_sample_density, SSS_WALK_ROUNDS, medium_transmittance_ratio_spectral, spectral_free_flight_weight
from .lights import area_light_pick_triangle, AreaLight, DistantLight, InfiniteLight, PointLight
from .curves import Curve_C, curve_piece_endpoints, _curve_perp_axis
from .bssrdf import dipole_rd, dipole_max_radius
from .bvh import (
    BVH2Node, SceneDescriptor2_C, traverse_bvh2_core, any_hit_bvh2_core,
    _scene_bounding_sphere, _sample_disk_perpendicular, _sample_infinite_light_dir, _eval_infinite_light_and_pdf,
    HairLobeConstants, _hair_precompute, _hair_eval_lobes, _hair_sample_dir, curve_offset_eps,
    LightSample, _sample_distant_light_nee, _sample_point_light_nee, _sample_sphere_light_nee, _sample_infinite_light_nee,
    render_aux_buffers,
)
from .vcm_mis import mis_policy_sole
from .layered import layered_sample
from .bxdf import dielectric_interface, bxdf_sample_dielectric, bxdf_sample_thin_dielectric, BxDFFlags, CoatWalk, coat_walk_begin, coat_walk_enter, coat_walk_at_base, coat_walk_scatter, COAT_WALKING, COAT_REFLECT, COAT_EXIT, COAT_ABSORB, GeomContext, BxDFSample, bxdf_sample_conductor, bxdf_sample_coated_conductor, bxdf_is_delta, bxdf_eval_conductor_ggx, _nee_weight_simple, _nee_weight_hair, _nee_weight_simple_spectral, LobeCtx, lobe_eval, lobe_sample, lobe_scoped, LobeTables, lobe_kind_of, lobe_param_of, lobe_is_delta_of, lobe_is_available_of, nee_weight_lobe
from .measured_bxdf_eval import bxdf_eval_measured, bxdf_sample_measured, _nee_weight_measured
from .shading import _tex_lookup, _get_tri_verts, _apply_surface_maps, \
    apply_surface_maps_at_hit, _camera_approx_footprint, area_light_hit_cos
from .sampling import power_heuristic, camera_ray_from_film_xy, FilmFilter, film_filter_of, film_filter_offset
from .transform import transform_normal_by_instance
from .rng import PCG32
from .pbrt_parser import ParsedScene_Mojo
from .postprocess import write_image, write_image_cropwindow, denoise
from .gpu_scene import GpuSceneHandle
from .spectrum import (
    SampledWavelengths, SpectralSample, sample_wavelengths_uniform,
    rgb_illuminant_to_spectral_sample, spectral_sample_to_rgb,
    rgb_bands_to_spectral_sample, spec_refl, spec_refl_unbounded, spec_illum, pass_wavelengths,
)


comptime _ALPHA  = Float32(0.7)
comptime _MAX_B  = 10  # hard safety ceiling, matching bdpt.mojo's own
                       # _BDPT_MAX_VERTS -- SPPM's bounce loops are additionally
                       # bounded by min(scene maxdepth, _MAX_B), see
                       # _sppm_trace_visible_point/_sppm_trace_photon's own
                       # `maxdepth` parameter. Before that bound was added, a
                       # scene with a small maxdepth (a common way to
                       # deliberately limit indirect bounces) was silently
                       # ignored and rendered up to 10 anyway -- measured on a
                       # cavity scene at maxdepth=1: SPPM read 2.05x pbrt where
                       # the (maxdepth-respecting) path tracer read 0.93x.
comptime _HSIZE  = 1048576   # 2^20 hash buckets
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
    # GGX roughness (max(roughU, roughV), matching bdpt.mojo's BDPTVertex
    # convention) — only meaningful when mat_kind=1. alb doubles as F0 for
    # conductor VPs, same repurposing bdpt.mojo's BDPTVertex.alb already uses.
    var alpha: Float32
    # mat_kind=2 (hair) only: material index (to re-fetch eta/sigma_a/betaM/
    # betaN from sd.materials) + curve hit info (to re-derive the fiber frame
    # via _hair_precompute) — NOT the full ~30-field HairLobeConstants, to
    # keep this struct small for every other VP kind; mirrors bdpt.mojo's
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


# ── Geometry helpers ──────────────────────────────────────────────────────────

@always_inline
def _geom_normal(
    inter: Intersection,
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    instances: Pointer[Instance, MutUntrackedOrigin] = Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(),
    spheres: Pointer[Sphere, MutUntrackedOrigin] = Pointer[Sphere, MutUntrackedOrigin].unsafe_dangling(),
    hit: Vec3f = Vec3f(Float32(0), Float32(0), Float32(0)),
) -> Vec3f:
    """Normalized geometric normal from triangle cross product. If this hit
    came from inside an instanced BLAS (primId.instanceIdx >= 0 — see
    bvh.mojo's traverse_bvh2_core type==6 branch), the mesh data is in that
    instance's object space, so the normal is transformed to world space
    before returning (the hit *point*, elsewhere computed as
    ray_org + ray_dir*tHit, needs no such fixup — see transform.mojo's
    transform_normal_by_instance for why).

    `spheres`/`hit` give analytic spheres (primId.type == 4) their exact
    outward normal, mirroring `_shading_normal_at` right below (which this
    function predates and was never given the same treatment). Without
    them, every caller either had to special-case type==4 itself before
    calling this -- duplicating the same sphere_outward_normal(hit, center)
    one-liner at each of the (as of 2026-09-15) 21 call sites across
    bdpt.mojo/sppm.mojo -- or, at 2 call sites that omitted the guard
    entirely, silently got the +Y placeholder below: a live bug (a diffuse
    analytic sphere corrupted both SPPM's visible-point normal and its
    photon-bounce normal identically to how a dielectric sphere corrupted
    refraction before _shading_normal_at's own fix). See
    project_mesh_only_geometry_assumption memory."""
    var mi: Int; var bv: Int
    if inter.primId.type == 0:
        mi = Int(inter.primId.id1); bv = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mi = Int(inter.primId.id2 >> 32); bv = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    elif inter.primId.type == Int8(4) and _is_real_ptr[Sphere](spheres):
        return sphere_outward_normal(Point3f(hit[0], hit[1], hit[2]), spheres[unsafe_offset=Int(inter.primId.id1)].center)
    else:
        return Vec3f(Float32(0), Float32(1), Float32(0))
    var m = meshes[unsafe_offset=mi]
    var v0 = Int(m.vertexIndices[unsafe_offset=bv])
    var v1 = Int(m.vertexIndices[unsafe_offset=bv + 1])
    var v2 = Int(m.vertexIndices[unsafe_offset=bv + 2])
    var p0 = Vec3f(m.points[unsafe_offset=v0*4], m.points[unsafe_offset=v0*4+1], m.points[unsafe_offset=v0*4+2])
    var p1 = Vec3f(m.points[unsafe_offset=v1*4], m.points[unsafe_offset=v1*4+1], m.points[unsafe_offset=v1*4+2])
    var p2 = Vec3f(m.points[unsafe_offset=v2*4], m.points[unsafe_offset=v2*4+1], m.points[unsafe_offset=v2*4+2])
    var n = cross(p1 - p0, p2 - p0)
    if inter.primId.instanceIdx >= Int32(0):
        n = transform_normal_by_instance(instances[unsafe_offset=Int(inter.primId.instanceIdx)].worldToObj, n)
    var l = dot(n, n)
    if l > Float32(0.0):
        n = n * (Float32(1.0) / sqrt(l))
    return n

@always_inline
def _shading_normal_at(
    inter: Intersection,
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    instances: Pointer[Instance, MutUntrackedOrigin] = Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(),
    spheres: Pointer[Sphere, MutUntrackedOrigin] = Pointer[Sphere, MutUntrackedOrigin].unsafe_dangling(),
    hit: Point3f = Point3f(Float32(0)),
) -> Vec3f:
    """Barycentrically-interpolated SMOOTH shading normal at a triangle hit,
    falling back to the flat geometric normal when the mesh has no per-vertex
    normals. Refraction through a finely-tessellated curved surface (e.g. the
    wavy water sheet in water-caustic, whose PLY carries per-vertex normals)
    MUST use the smooth normal — the flat per-triangle normal refracts each
    triangle's whole patch of light in one direction, producing a blocky/
    blotchy caustic and the wrong energy distribution. This is what pbrt (and
    gonzales's own main path tracer via shading.mojo::_shading_normal) does;
    SPPM previously used only _geom_normal here, which was the discrepancy.

    `spheres`/`hit` give analytic spheres (primId.type == 4) their exact
    outward normal. Without them this returned the +Y placeholder below for
    every sphere hit, so a dielectric sphere refracted every ray identically
    regardless of where it was struck -- silently destroying any caustic it
    should cast."""
    var mi: Int; var bv: Int
    if inter.primId.type == 0:
        mi = Int(inter.primId.id1); bv = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mi = Int(inter.primId.id2 >> 32); bv = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    elif inter.primId.type == Int8(4) and _is_real_ptr[Sphere](spheres):
        return sphere_outward_normal(hit, spheres[unsafe_offset=Int(inter.primId.id1)].center)
    else:
        return Vec3f(Float32(0), Float32(1), Float32(0))
    var m = meshes[unsafe_offset=mi]
    var v0 = Int(m.vertexIndices[unsafe_offset=bv])
    var v1 = Int(m.vertexIndices[unsafe_offset=bv + 1])
    var v2 = Int(m.vertexIndices[unsafe_offset=bv + 2])
    var p0 = Vec3f(m.points[unsafe_offset=v0*4], m.points[unsafe_offset=v0*4+1], m.points[unsafe_offset=v0*4+2])
    var p1 = Vec3f(m.points[unsafe_offset=v1*4], m.points[unsafe_offset=v1*4+1], m.points[unsafe_offset=v1*4+2])
    var p2 = Vec3f(m.points[unsafe_offset=v2*4], m.points[unsafe_offset=v2*4+1], m.points[unsafe_offset=v2*4+2])
    var gn = cross(p1 - p0, p2 - p0)
    if inter.primId.instanceIdx >= Int32(0):
        gn = transform_normal_by_instance(instances[unsafe_offset=Int(inter.primId.instanceIdx)].worldToObj, gn)
    var gl = dot(gn, gn)
    if gl > Float32(0.0): gn = gn * (Float32(1.0) / sqrt(gl))
    # No per-vertex normals → flat normal (sentinel addr <= 4, see GPU-nullable convention).
    if Int(m.normals) <= 4:
        return gn
    var w0 = Float32(1.0) - inter.u - inter.v
    var n0 = Vec3f(m.normals[unsafe_offset=v0*3], m.normals[unsafe_offset=v0*3+1], m.normals[unsafe_offset=v0*3+2])
    var n1 = Vec3f(m.normals[unsafe_offset=v1*3], m.normals[unsafe_offset=v1*3+1], m.normals[unsafe_offset=v1*3+2])
    var n2 = Vec3f(m.normals[unsafe_offset=v2*3], m.normals[unsafe_offset=v2*3+1], m.normals[unsafe_offset=v2*3+2])
    var sn = n0 * w0 + n1 * inter.u + n2 * inter.v
    if inter.primId.instanceIdx >= Int32(0):
        sn = transform_normal_by_instance(instances[unsafe_offset=Int(inter.primId.instanceIdx)].worldToObj, sn)
    var sl = dot(sn, sn)
    if sl <= Float32(1e-12):
        return gn
    sn = sn * (Float32(1.0) / sqrt(sl))
    if dot(sn, gn) < Float32(0.0):
        sn = -sn
    return sn

@always_inline
def _hash_cell(ix: Int, iy: Int, iz: Int) -> Int:
    var h = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791
    return (h % _HSIZE + _HSIZE) % _HSIZE


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
def _dielectric_bounce(
    ray_dir: Vec3f,
    hit_point: Vec3f,
    geom_normal: Vec3f,
    ior: Float32,
    # True only for a ray that cannot possibly be inside this dielectric yet,
    # i.e. the primary/first segment in vacuum. Used to fix inward-normal
    # meshes. Was `bounce: Int` with an internal `bounce == 0` test, which
    # breaks the moment a caller stops charging bounces for subsurface
    # boundaries (see Material_C.sss_boundary): `bounce` then stays 0 for the
    # whole interior walk and every boundary hit from INSIDE would be forced
    # to "entering", refracting inward again so light could never leave. The
    # caller knows whether it is in a medium; it passes the real question.
    force_entering: Bool,
    mut pcg: PCG32,
    current_ior: Float32 = Float32(1.0),    # IOR of the medium the ray is ALREADY in; 1.0 = vacuum
    previous_ior: Float32 = Float32(1.0),   # IOR one level below current_ior (what exiting restores)
    is_thin: Bool = False,                  # `thindielectric`: both interfaces at once
    radiance_mode: Bool = True,             # camera path; False for a photon/light path
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
        return (bs_t.wi, hit_point + off_t * Float32(0.0001), bs_t.f.r,
                current_ior, previous_ior)
    var (bs, normal, new_current_ior, new_previous_ior) = bxdf_sample_dielectric(
        geom_normal, ray_dir, ior, force_entering, u, current_ior, previous_ior,
        radiance_mode)
    var off = normal if (Int(bs.flags) & Int(BxDFFlags.reflect)) != 0 else -normal
    return (bs.wi, hit_point + off * Float32(0.0001), bs.f.r,
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
def sample_area_light_uniform(
    areaLights: Pointer[AreaLight, MutUntrackedOrigin],
    meshes:     Pointer[TriangleMesh, MutUntrackedOrigin],
    n_lights:   Int,
    mut pcg:    PCG32,
    curves:     Pointer[Curve_C, MutUntrackedOrigin] = Pointer[Curve_C, MutUntrackedOrigin].unsafe_dangling(),
) -> AreaLightSample:
    """Uniformly picks one area light, then a point + geometric normal on
    it: a random triangle + barycentric point on a mesh light (kind==0,
    using the mesh's per-vertex shading normals when present — needed for
    e.g. ceiling lights whose winding gives an upward geometric normal but
    whose scene-specified normals point down into the room), or a random
    piece + point on a curve's swept tube (kind==1, `curves` must be a real
    pointer whenever any curve lights exist)."""
    var li = Int(pcg.next_uint() % UInt32(n_lights))
    var al = areaLights[unsafe_offset=li]
    if al.kind == Int8(1):
        var curve = curves[unsafe_offset=Int(al.meshIdx)]
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
    var n_tris = Int(max(Int(al.n_tris), 1))
    var ti = area_light_pick_triangle(al, pcg.next_float())
    var lb = ti * 3
    var lv0 = Int(lmesh.vertexIndices[unsafe_offset=lb]); var lv1 = Int(lmesh.vertexIndices[unsafe_offset=lb+1]); var lv2 = Int(lmesh.vertexIndices[unsafe_offset=lb+2])
    var lp0 = Vec3f(lmesh.points[unsafe_offset=lv0*4], lmesh.points[unsafe_offset=lv0*4+1], lmesh.points[unsafe_offset=lv0*4+2])
    var lp1 = Vec3f(lmesh.points[unsafe_offset=lv1*4], lmesh.points[unsafe_offset=lv1*4+1], lmesh.points[unsafe_offset=lv1*4+2])
    var lp2 = Vec3f(lmesh.points[unsafe_offset=lv2*4], lmesh.points[unsafe_offset=lv2*4+1], lmesh.points[unsafe_offset=lv2*4+2])
    var ru1 = pcg.next_float(); var ru2 = pcg.next_float(); var sr1 = sqrt(ru1)
    var lp  = lp0*(Float32(1)-sr1) + lp1*(sr1*(Float32(1)-ru2)) + lp2*(sr1*ru2)
    var ln  = cross(lp1-lp0, lp2-lp0)
    var lnl = dot(ln, ln)
    if lnl > Float32(0): ln = ln*(Float32(1)/sqrt(lnl))
    # Use shading normals when provided — they give the correct emission hemisphere.
    if Int(lmesh.normals) > 4:
        var sn0 = Vec3f(lmesh.normals[unsafe_offset=lv0*3], lmesh.normals[unsafe_offset=lv0*3+1], lmesh.normals[unsafe_offset=lv0*3+2])
        var sn1 = Vec3f(lmesh.normals[unsafe_offset=lv1*3], lmesh.normals[unsafe_offset=lv1*3+1], lmesh.normals[unsafe_offset=lv1*3+2])
        var sn2 = Vec3f(lmesh.normals[unsafe_offset=lv2*3], lmesh.normals[unsafe_offset=lv2*3+1], lmesh.normals[unsafe_offset=lv2*3+2])
        var sn_avg = (sn0 + sn1 + sn2) / Float32(3)
        var snl = sn_avg.length()
        if snl > Float32(0): ln = (sn_avg / snl).to_simd()
    return AreaLightSample(al, lp, ln)


# ── Camera pass ───────────────────────────────────────────────────────────────

@always_inline
def medium_after_crossing(
    ray_dir: Vec3f,
    inter: Intersection,
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    mat: Material_C,
    ref sd: SceneDescriptor2_C,
    hit: Point3f = Point3f(Float32(0)),
) -> Int32:
    """Return new current_medium_idx after crossing a surface with MediumInterface.

    Shared by SPPM and VCM (bdpt.mojo had a byte-for-byte copy of this until
    2026-09-17, whose mesh branch also ignored instance transforms).

    `hit` is REQUIRED for analytic spheres: their outward normal is
    normalize(hit - center), so leaving it at the default origin
    makes the inside/outside test read one FIXED direction for every ray that
    crosses the sphere, regardless of where it actually struck. All four call
    sites omitted it until 2026-09-09, which is why volumetric-caustic's glass
    sphere never switched a ray between `gas` and its own vacuum interior --
    and so never produced the focused beam that is the whole point of the
    scene. Same mesh-only-assumption class as the three sphere bugs found
    earlier that day (see project_volume_area_light_nee_bug defect 3)."""
    if mat.medium_interface_idx < Int32(0) or sd.mediumIfaceCount == Int64(0):
        return Int32(-1)  # stays vacuum; caller keeps existing idx if needed
    var iface = sd.mediumInterfaces[unsafe_offset=Int(mat.medium_interface_idx)]
    var n = _geom_normal(inter, meshes, sd.instances, sd.spheres, hit.to_simd())
    var md = ray_dir[0]*n.x + ray_dir[1]*n.y + ray_dir[2]*n.z
    return iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx

def _sppm_trace_visible_point[use_gpu: Bool](
    ref sd:       SceneDescriptor2_C,
    mut pcg:  PCG32,
    r2c:      Pointer[Float32, MutUntrackedOrigin],
    c2w:      Pointer[Float32, MutUntrackedOrigin],
    px: Int, py: Int,
    pidx:     Int32,
    init_r2:  Float32,
    scratch:  Pointer[Intersection, MutUntrackedOrigin],
    maxdepth: Int,
    film_filter: FilmFilter,
) -> SPPMPixel:
    """Trace one primary ray for pixel (px,py), returning its visible point.
    Shared verbatim between the CPU driver (_sppm_camera_pass, [False]) and
    the GPU kernel (sppm_gen_vp_gpu, one slot per thread, [True]) — the
    bounce loop itself has zero CPU/GPU divergence EXCEPT `_tex_lookup`'s
    own CPU-filename vs GPU-texture-array dispatch for diffuse/coateddiffuse
    albedo (task #150/#151: bdpt.mojo/sppm.mojo previously never evaluated
    image textures at all, always falling back to a flat grey default).
    `scratch` is caller-owned (no internal alloc/free) so this is safe to
    call from a GPU kernel thread, same convention as bdpt.mojo's shared
    subpath tracers."""
    var org = Point3f(c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14])
    var has_media = Int(sd.mediumCount) > 0

    # One hero-wavelength sample for this VP's own camera subpath, carrying
    # its throughput (`beta`), its direct-lighting term (`ld`) and its
    # miss-escape emission (`env`). It is deliberately NOT a per-pass value:
    # SPPM traces visible points ONCE, before any photon pass exists, so a
    # VP's wavelengths cannot agree with the photons'. That is why `tau` is
    # RGB -- see SPPMPixel.tau.
    var vp_wavelengths = sample_wavelengths_uniform(pcg.next_float())

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
    var (rd, ro, cl) = camera_ray_from_film_xy(fX, fY, r2c, c2w)

    # Texture/bump footprint for this camera path, derived here rather than
    # plumbed: r2c's first column IS the per-pixel derivative of the
    # camera-space direction, so the pixel's angular size is |r2c[0..2]| / cl
    # (cl = that direction's length before normalising). No new parameter on
    # four call levels for a quantity the matrix already carries.
    #
    # Scaled by pbrt's max(0.125, 1/sqrt(n)) with n = _VP_SAMPLES, the
    # independent camera samples per pixel here -- same reasoning as the path
    # tracer's (cpu/integrators.cpp:251): the right filter width is the
    # SAMPLE SPACING, not the whole pixel.
    var _pxs = sqrt(r2c[unsafe_offset=0]*r2c[unsafe_offset=0]
                  + r2c[unsafe_offset=1]*r2c[unsafe_offset=1]
                  + r2c[unsafe_offset=2]*r2c[unsafe_offset=2])
    var px_scale = (_pxs / cl if cl > Float32(0.0) else Float32(0.0)) * max(
        Float32(0.125), Float32(1.0) / sqrt(Float32(_VP_SAMPLES)))
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
    # scattering events between them, are all ONE BSSRDF event. Charging them
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
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1.0e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                           sd.spheres, Int(sd.sphereCount))
        if scratch[unsafe_offset=0].hit == Int8(0):
            for inf_i in range(Int(sd.infiniteLightCount)):
                var ilight = sd.infiniteLights[unsafe_offset=inf_i]
                var (Le, _pdf_unused) = _eval_infinite_light_and_pdf(ilight, rd)
                vp.env += vp.beta * Le
            break

        var inter = scratch[unsafe_offset=0]
        var ray_dir = rd.to_simd()
        var t_hit = inter.tHit
        cone_len += t_hit

        # ── Volume free-flight ────────────────────────────────────────────
        if has_media and Int(cur_med_idx) >= 0:
            var med = sd.mediums[unsafe_offset=Int(cur_med_idx)]
            # ONE shared sampler for both medium kinds (geometry.mojo):
            # homogeneous closed form, or delta tracking against the real
            # density field. This call site used to be the homogeneous one
            # unconditionally, which rendered every "uniformgrid"/"nanovdb"/
            # "cloud" medium as uniform density-1 fog -- bunny-cloud came out a
            # featureless sphere with no bunny in it. See sample_free_flight.
            var ff = sample_free_flight(
                med, sd.grids, sd.nvdbGrids, Vec3f(ro.x, ro.y, ro.z), rd, t_hit, pcg)
            if ff.collided:
                # Volume scatter — store VP here
                # NOTE: ff.weight (the chromatic collision ratio) is deliberately
                # NOT applied here, unlike bdpt.mojo. Applying it to both the VP
                # and photon subpaths measurably WORSENED SPPM's agreement with
                # the path tracer on a chromatic scatterer (blue 1.22x -> 2.03x
                # of PT; chroma spread 1.38 -> 2.14), while VCM improved sharply
                # under the same change. SPPM's volume path has a separate,
                # unresolved discrepancy -- it already sits ~5% off PT on a GREY
                # scatterer -- so the weight is withheld here until that is
                # root-caused rather than shipping a measured regression.
                # See Scenes/media-chromatic-scatter.pbrt.
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
        # bdpt.mojo's equivalent fix): this VP sample is either a direct
        # light hit (valid stays 0, credited here) XOR a real gatherable
        # surface that separately does its own NEE — mutually exclusive per
        # sample, same reasoning as the infinite-light miss-escape case
        # below, so there's no competing strategy to double-count against.
        if inter.primId.type == Int8(4):
            var sph_hit = sd.spheres[unsafe_offset=Int(inter.primId.id1)]
            if sph_hit.isAreaLight != Int8(0):
                vp.env += vp.beta * sph_hit.emission
                break

        # Mix material: stochastically resolve to one of two sub-materials
        # (mirrors shading.mojo's shade_mix / bdpt.mojo's own resolution).
        if mat.type == MatKind.mix:
            var mix_idx1 = Int(mat.tex_idx & Int32(0xFFFF))
            var mix_idx2 = Int((mat.tex_idx >> 16) & Int32(0xFFFF))
            var mix_amount = mat.roughU
            var mix_chosen = mix_idx2 if pcg.next_float() < mix_amount else mix_idx1
            mat = sd.materials[unsafe_offset=mix_chosen]
            mat_idx = mix_chosen  # keep in sync with the resolved sub-material (hair needs the real index to re-fetch at gather/NEE time)
            if mat.type == MatKind.mix:
                mat.type = MatKind.diffuse

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
                vp.env += vp.beta * mat.emission
            elif area_light_hit_cos(inter, sd.meshes, sd.instances, ray_dir) > Float32(0):
                vp.env += vp.beta * sd.areaLights[unsafe_offset=Int(inter.primId.id1)].emission
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

        if mat.type == MatKind.diffuse or mat.type == MatKind.diffuse_transmit or mat.type == MatKind.coated_diffuse or mat.type == MatKind.conductor or mat.type == MatKind.measured or mat.type == MatKind.hair:
            if not lobe_is_available_of(mat):
                break   # e.g. a measured table that failed to load
            # GEOMETRY, not material: a curve hit has its own normal, and no
            # texture or bump map -- matches bdpt.mojo's own on_curve gate.
            var on_curve = inter.primId.type == Int8(5)
            var gn: Vec3f
            if on_curve:
                var hc_g = _hair_precompute(mat, sd.curves, Int(inter.primId.id1), inter.v, inter.u, (-rd).to_simd())
                gn = hc_g.geo_normal
            else:
                gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            if dot(gn, ray_dir) > Float32(0.0):
                gn = gn * Float32(-1.0)
            var gn_geo = gn   # before any bump/normal map -- see SPPMPixel.geo_normal
            var eff_alb = mat.albedo
            var (tex_mesh, tv0, tv1, tv2, tex_ok) = _get_tri_verts(inter, sd.meshes)
            if tex_ok and not on_curve:
                eff_alb = _tex_lookup[use_gpu](mat, inter, tv0, tv1, tv2, tex_mesh, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            if not on_curve:
                gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes,
                    gn, gn, ray_dir, px_scale * cone_len,
                    sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
                gn = face_toward(gn, -ray_dir)   # pbrt two-sided reflection, see face_toward
            # A smooth conductor is the one kind here that is genuinely
            # delta: it scatters but stores no visible point, exactly like
            # bdpt.mojo's own lobe_is_delta_of gate.
            if not lobe_is_delta_of(mat):
                vp.pos = hit
                vp.normal = vec3f(gn)
                vp.geo_normal = vec3f(gn_geo)
                vp.alb = eff_alb
                # One lobe per material, from bxdf.mojo's lobe_kind_of --
                # gather and NEE evaluate this VP through lobe_eval, so this
                # branch never asks which material it is again.
                vp.mat_kind = lobe_kind_of(mat.type)
                vp.mat_idx = Int32(mat_idx)
                vp.wo = vec3f((-rd).to_simd())
                vp.alpha = lobe_param_of(mat)
                if on_curve:
                    vp.hair_curve_idx = Int32(inter.primId.id1)
                    vp.hair_h = inter.u
                    vp.hair_v = inter.v
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
            rd = vec3f(bs_c.wi)
            ro = hit + rd * Float32(0.0002)

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            # ── Subsurface boundary: stop here, as a BSSRDF visible point ──
            # Instead of refracting INTO the skin and placing a volume visible
            # point a mean free path down (a 3D density estimate in a medium
            # whose mfp is ~0.001 scene units -- provably photon-count
            # insensitive, see bssrdf.mojo), keep the visible point ON the
            # surface and let a diffusion kernel carry the interior transport
            # at gather time. Photons stop on the surface too, so the photon
            # map stays two-dimensional -- which is where density estimation
            # actually works.
            #
            # Needs no new SPPMPixel fields: `med_idx` already reaches the
            # interior Medium for sigma_s/sigma_a/g, and `alpha` is unused
            # for this visible-point kind so it carries the boundary IOR.
            if mat.sss_boundary != Int8(0) and Int(cur_med_idx) < 0:
                var gn_s = _shading_normal_at(inter, sd.meshes, sd.instances, sd.spheres, hit)
                if dot(gn_s, rd) > Float32(0.0):
                    gn_s = gn_s * Float32(-1.0)
                var inside_idx = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                if inside_idx >= Int32(0):
                    vp.pos = hit
                    vp.normal = vec3f(gn_s)
                    vp.geo_normal = vec3f(gn_s)
                    vp.wo = vec3f((-rd).to_simd())
                    vp.alb = RGB(Float32(1))
                    vp.mat_kind = LobeKind.bssrdf          # BSSRDF
                    vp.alpha = mat.albedo.r          # boundary IOR
                    vp.med_idx = inside_idx
                    vp.is_volume = PhotonKind.surface
                    vp.valid = Int32(1)
                    break
            # Entering, leaving or total-internal-reflecting at the boundary of
            # a subsurface interior is part of the ONE BSSRDF event the walk
            # inside it belongs to, so it is not charged to maxdepth. Same rule
            # as the path tracer (Material_C.sss_boundary).
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
            gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes,
                gn, gn, ray_dir, px_scale * cone_len,
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            var (new_dir, new_org, radiance_scale, new_cur_ior, new_prev_ior) = _dielectric_bounce(
                ray_dir, hit.to_simd(), gn, ior, n_events == 1 and Int(cur_med_idx) < 0, pcg, current_dielectric_ior, previous_dielectric_ior, mat.type == MatKind.thin_dielectric)
            current_dielectric_ior = new_cur_ior
            previous_dielectric_ior = new_prev_ior
            vp.beta *= radiance_scale  # camera-path (Radiance mode): apply non-symmetric-scattering correction
            rd = vec3f(new_dir)
            ro = point3f(new_org)
            if has_media:
                var new_idx = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                    cur_med_idx = new_idx

        elif mat.type == MatKind.interface:
            # Transparent boundary — update medium, continue ray
            if has_media:
                var new_idx = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                    cur_med_idx = new_idx
            ro = hit + rd * Float32(0.0002)

        else:
            break  # area_light, etc.

    return vp

def _sppm_camera_pass(
    vps:        Pointer[SPPMPixel, MutUntrackedOrigin],
    n_pix:      Int,
    vp_samples: Int,
    fw:       Int32,
    r2c:      Pointer[Float32, MutUntrackedOrigin],
    c2w:      Pointer[Float32, MutUntrackedOrigin],
    ref sd:       SceneDescriptor2_C,
    init_r2:  Float32,
    seed:     UInt64,
    maxdepth: Int,
    film_filter: FilmFilter,
):
    """Trace `vp_samples` independent primary rays per pixel, ONCE for the
    whole render (not once per SPPM pass — see sppm_render's docstring for
    why a per-pass re-trace breaks SPPM's convergence guarantee). Each sample
    gets its own persistent (r2, tau, N_acc) accumulator, fixed at this one
    surface/position for every subsequent photon pass; sppm_render averages
    the `vp_samples` independently-converged estimates per pixel at the end.
    Having multiple independent samples (rather than one, traced once) is
    what avoids the old black-speckle bug: a pixel where the camera ray
    crosses a dielectric surface makes a genuine random reflect-vs-refract
    choice each sample, so as long as vp_samples is large enough, at least
    some samples land on the diffuse floor even if others reflect away."""
    # One scratch Intersection per worker (indexed by `combined`) instead
    # of one shared slot — same convention the GPU kernel already uses
    # (inter_scratch + combined, one per thread) — needed now that this loop
    # runs across CPU threads too, not just GPU ones.
    var scratch = unsafe_alloc[Intersection](max(n_pix * vp_samples, 1))

    def trace_one(combined: Int) {imm}:
        var pix = combined // vp_samples
        var px = pix % Int(fw)
        var py = pix // Int(fw)
        var pcg = PCG32(seed ^ UInt64(combined * 6364136223846793005 + 1), UInt64(1))
        vps[unsafe_offset=combined] = _sppm_trace_visible_point[False](sd, pcg, r2c, c2w, px, py, Int32(pix), init_r2, scratch.unsafe_offset(combined), maxdepth, film_filter)

    parallelize(trace_one, n_pix * vp_samples)

    scratch.unsafe_free()


# ── Photon pass ───────────────────────────────────────────────────────────────

def _sppm_store_photon[use_gpu: Bool](
    ph:          SPPMPhoton,
    photons:     Pointer[SPPMPhoton, MutUntrackedOrigin],
    max_photons: Int,
    counter:     Pointer[Int32, MutUntrackedOrigin],
):
    """Reserve the next photon slot and store `ph` there, dropping it if the
    buffer is already full. Comptime-branches only on the slot-reservation
    primitive (atomic fetch-add for racing GPU threads vs. a plain
    increment for the serial CPU loop) — mirrors bdpt.mojo's
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
# `cl` (the camera-space direction's length before normalising) varies a little
# across the film; a photon belongs to no pixel, so this takes the film centre
# and uses it for the whole pass. The max(0.125, 1/sqrt(n)) factor is pbrt's
# (cpu/integrators.cpp:251) and is repeated here on purpose: the right filter
# width is the SAMPLE spacing, not the whole pixel.
@always_inline
def _sppm_photon_px_scale(
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    fw: Int, fh: Int,
) -> Float32:
    var (_rd, _ro, cl) = camera_ray_from_film_xy(
        Float32(fw) * Float32(0.5), Float32(fh) * Float32(0.5), r2c, c2w)
    var _pxs = sqrt(r2c[unsafe_offset=0]*r2c[unsafe_offset=0]
                  + r2c[unsafe_offset=1]*r2c[unsafe_offset=1]
                  + r2c[unsafe_offset=2]*r2c[unsafe_offset=2])
    return (_pxs / cl if cl > Float32(0.0) else Float32(0.0)) * max(
        Float32(0.125), Float32(1.0) / sqrt(Float32(_VP_SAMPLES)))

@always_inline
def _sppm_cam_pos(c2w: Pointer[Float32, MutUntrackedOrigin]) -> Vec3f:
    return Vec3f(c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14])

def _sppm_trace_photon[use_gpu: Bool, tex_gpu: Bool](
    ref sd:               SceneDescriptor2_C,
    mut pcg:          PCG32,
    scratch:          Pointer[Intersection, MutUntrackedOrigin],
    n_emit:           Int,
    photons:          Pointer[SPPMPhoton, MutUntrackedOrigin],
    max_photons:      Int,
    counter:          Pointer[Int32, MutUntrackedOrigin],
    default_emit_med: Int32,
    maxdepth: Int,
    # Bump/normal-map footprint reference for this photon's hits. A photon has
    # no ray cone -- the camera-side walk derives one from r2c, and there is no
    # equivalent on the light side -- so this is pbrt's own fallback for a
    # vertex without differentials (Camera::Approximate_dp_dxy): footprint =
    # pixel angular size x distance from the camera. See
    # shading.mojo's _camera_approx_footprint for why zero is NOT an option
    # here, and project_surface_maps_photon_side_gap for the more accurate
    # routes (Photon Differentials, LEADR) deliberately not taken yet.
    cam_pos: Vec3f,
    px_scale: Float32,
    # Decomposed spectral tables rather than reading sd.spectral. `sd` is a
    # SceneDescriptor2_C passed BY VALUE, and it contains a SpectralHandle --
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
    var n_sphere = 0
    for i in range(Int(sd.sphereCount)):
        if sd.spheres[unsafe_offset=i].isAreaLight != Int8(0):
            n_sphere += 1
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

    var light_pick = Int(pcg.next_uint() % UInt32(n_lights))
    if light_pick < n_area:
        # Pick a random area light + triangle + barycentric point on it.
        var light_sample = sample_area_light_uniform(sd.areaLights, sd.meshes, n_area, pcg, sd.curves)
        var al = light_sample.light
        var lp = light_sample.point
        var ln = light_sample.normal

        # Sample cosine-weighted emission direction from the light hemisphere
        # (same Frisvad-frame construction as _cosine_hemisphere_sample).
        var du1 = pcg.next_float()
        var du2 = pcg.next_float()
        var pdir = _cosine_hemisphere_sample(ln, du1, du2)

        # Photon flux: total_light_power / n_emit
        # total_power = emission * pi * total_area * n_lights (uniform light selection)
        var scale = PI * al.total_area * Float32(n_lights) / Float32(n_emit)
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
        var want = light_pick - n_area
        var si_e = 0
        var seen = 0
        for i in range(Int(sd.sphereCount)):
            if sd.spheres[unsafe_offset=i].isAreaLight != Int8(0):
                if seen == want:
                    si_e = i
                    break
                seen += 1
        var sph_e = sd.spheres[unsafe_offset=si_e]
        var us1 = pcg.next_float(); var us2 = pcg.next_float()
        var cz = Float32(1) - Float32(2) * us1
        var sz = sqrt(max(Float32(0), Float32(1) - cz * cz))
        var sphi = Float32(2) * PI * us2
        var sn = Vec3f(sz * cos(sphi), sz * sin(sphi), cz)
        var area_e = Float32(4) * PI * sph_e.radius * sph_e.radius
        var scale_e = PI * area_e * Float32(n_lights) / Float32(n_emit)
        var du1e = pcg.next_float(); var du2e = pcg.next_float()
        flux = spec_illum(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                          sph_e.emission.r * scale_e, sph_e.emission.g * scale_e, sph_e.emission.b * scale_e, ph_wavelengths)
        ro = sph_e.center + sn * sph_e.radius * Float32(1.0001)
        rd = vec3f(_cosine_hemisphere_sample(sn, du1e, du2e))
    elif light_pick < n_area + n_sphere + n_distant:
        var dl = sd.distantLights[unsafe_offset=light_pick - n_area - n_sphere]
        var (center, radius) = _scene_bounding_sphere(sd)
        var dir = Vec3f(dl.direction.x, dl.direction.y, dl.direction.z)
        var disk_pt = _sample_disk_perpendicular(dir, center, radius, Point2f(pcg.next_float(), pcg.next_float()))
        flux = spec_illum(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, dl.emission.r, dl.emission.g, dl.emission.b, ph_wavelengths) * (Float32(n_lights) * PI * radius * radius) / Float32(n_emit)
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
        flux = spec_illum(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, env_rgb.r, env_rgb.g, env_rgb.b, ph_wavelengths) * (Float32(n_lights) * PI * radius * radius / pdf_dir) / Float32(n_emit)
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
        flux = spec_illum(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, pll.intensity.r, pll.intensity.g, pll.intensity.b, ph_wavelengths) * (Float32(4) * PI * Float32(n_lights) / Float32(n_emit))
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
    # scattering events between them, are all ONE BSSRDF event. Charging them
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
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1.0e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                           sd.spheres, Int(sd.sphereCount))
        if scratch[unsafe_offset=0].hit == Int8(0):
            break  # miss

        var inter = scratch[unsafe_offset=0]
        var ray_dir = rd.to_simd()
        var t_hit = inter.tHit

        # ── Volume free-flight ────────────────────────────────────────────
        if has_media and Int(cur_med_idx) >= 0:
            var med = sd.mediums[unsafe_offset=Int(cur_med_idx)]
            # ONE shared sampler for both medium kinds (geometry.mojo):
            # homogeneous closed form, or delta tracking against the real
            # density field. This call site used to be the homogeneous one
            # unconditionally, which rendered every "uniformgrid"/"nanovdb"/
            # "cloud" medium as uniform density-1 fog -- bunny-cloud came out a
            # featureless sphere with no bunny in it. See sample_free_flight.
            var ff = sample_free_flight(
                med, sd.grids, sd.nvdbGrids, Vec3f(ro.x, ro.y, ro.z), rd, t_hit, pcg)
            if ff.collided:
                # A scattering event INSIDE a subsurface interior is a step of
                # the BSSRDF random walk, not a path bounce, so it is not
                # charged to maxdepth -- the same exemption Medium.is_sss
                # buys the path tracer. Skin1 at the scale sssdragon/head use
                # needs tens to hundreds of these before a photon escapes or is
                # absorbed; charging them ends the walk almost immediately.
                if med.is_sss != Int32(0):
                    bounce -= 1
                # Volume scatter — store photon and sample new direction
                var sp = ro + rd * ff.t_free
                # n_events > 1: skip storing at the light's own first
                # segment — that direct contribution is now covered
                # by _sppm_nee_update instead (matches pbrt's own
                # SPPM, which skips photon-grid gathering at depth 0
                # specifically to avoid double-counting with its NEE
                # term).
                if n_events > 1:
                    _sppm_store_photon[use_gpu](
                        SPPMPhoton(pos=sp, flux=flux, nxt=Int32(-1), is_volume=PhotonKind.volume, dir_in=rd, wavelengths=ph_wavelengths),
                        photons, max_photons, counter)
                # Scatter: isotropic phase function, modulate by albedo
                # Single-scattering albedo is a per-channel COEFFICIENT
                # (sigma_s/sigma_t), not a reflectance: band-pick it, never
                # push it through the reflectance upsampler. spec_refl was
                # used here, and RGB(a,a,a) does NOT upsample to a in every
                # lane, so even a GREY medium picked up a spurious D65-shaped
                # tint compounding once per scattering event -- the exact
                # bug class docs/02_spectra_and_color.md documents ("A
                # coefficient is not a color"). Band-picking degenerates to
                # exactly the channel value in every lane.
                flux *= rgb_bands_to_spectral_sample((ff.albedo).r, (ff.albedo).g, (ff.albedo).b, ph_wavelengths)
                # Sample new isotropic direction (uniform sphere)
                var usp1 = pcg.next_float()
                var usp2 = pcg.next_float()
                var cosT = Float32(2.0) * usp1 - Float32(1.0)
                var sinT = sqrt(max(Float32(0), Float32(1) - cosT*cosT))
                var phiS = Float32(2.0) * PI * usp2
                rd = Vec3f(sinT * cos(phiS), sinT * sin(phiS), cosT)
                ro = sp + rd * Float32(0.0001)
                continue
            else:
                # Apply Beer-Lambert transmittance through segment. Spectral:
                # upsample sigma_t to the 4 hero lanes FIRST, exponentiate
                # PER LANE -- see spectral_free_flight_weight's docstring for
                # why this differs from (and replaces) band-picking the
                # already-exponentiated RGB ratio.
                flux *= spectral_free_flight_weight(med, ff, t_hit, ph_wavelengths,
                    spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[unsafe_offset=mat_idx]
        var hit = ro + rd * t_hit

        if mat.type == MatKind.mix:
            var mix_idx1 = Int(mat.tex_idx & Int32(0xFFFF))
            var mix_idx2 = Int((mat.tex_idx >> 16) & Int32(0xFFFF))
            var mix_amount = mat.roughU
            var mix_chosen = mix_idx2 if pcg.next_float() < mix_amount else mix_idx1
            mat = sd.materials[unsafe_offset=mix_chosen]
            if mat.type == MatKind.mix:
                mat.type = MatKind.diffuse

        # See geometry.mojo's TERMINAL_SEGMENT_GRACE_ROUNDS and the matching
        # guard in _sppm_trace_visible_point above. A photon that simply
        # escapes deposits nothing either way, but without this guard a
        # photon could still scatter off a real material one bounce past
        # maxdepth in the grace round -- kept for consistency with the
        # camera side, not from an isolated repro.
        if bounce > max_charged:
            break

        if mat.type == MatKind.coated_diffuse:
            # pbrt's LayeredBxDF (layered.mojo), in importance mode: store the
            # photon for the flux arriving at the coated surface (the diffuse
            # branch's convention), then scatter with a layered sample. Now
            # after the maxdepth guard above, like every other material; the
            # old coat walk sat before it and could scatter one bounce past.
            if n_events > 1:
                _sppm_store_photon[use_gpu](
                    SPPMPhoton(pos=hit, flux=flux, nxt=Int32(-1), is_volume=PhotonKind.surface, dir_in=rd, wavelengths=ph_wavelengths),
                    photons, max_photons, counter)
            var eff_alb_cd = mat.albedo
            var (tm_cd, t0_cd, t1_cd, t2_cd, tok_cd) = _get_tri_verts(inter, sd.meshes)
            if tok_cd:
                eff_alb_cd = _tex_lookup[tex_gpu](mat, inter, t0_cd, t1_cd, t2_cd, tm_cd, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            var gn_cd = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            if dot(gn_cd, ray_dir) > Float32(0.0):
                gn_cd = gn_cd * Float32(-1.0)
            gn_cd = apply_surface_maps_at_hit[tex_gpu](mat, inter, sd.meshes,
                gn_cd, gn_cd, ray_dir, _camera_approx_footprint(hit, cam_pos, px_scale),
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            gn_cd = face_toward(gn_cd, -ray_dir)   # pbrt two-sided reflection, see face_toward
            var n_cd = vec3f(gn_cd)
            var fr_cd = Frame.from_z(n_cd)
            var tx_cd = Vec3f(fr_cd.x.x, fr_cd.x.y, fr_cd.x.z)
            var ty_cd = Vec3f(fr_cd.y.x, fr_cd.y.y, fr_cd.y.z)
            var wo_w = vec3f(-ray_dir)
            var wo_cd = Vec3f(dot(wo_w, tx_cd), dot(wo_w, ty_cd), dot(wo_w, n_cd))
            var R_cd = spec_refl(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, eff_alb_cd.r, eff_alb_cd.g, eff_alb_cd.b, ph_wavelengths)
            var bs_cd = layered_sample(wo_cd, pcg.next_float(), pcg.next_float(), pcg.next_float(), R_cd,
                                       mat.emission.r, max(mat.roughU, mat.roughV), False)
            if not bs_cd.valid or bs_cd.pdf <= Float32(0.0):
                break
            var w_cd = bs_cd.f * (abs(bs_cd.wi.z) / bs_cd.pdf)
            var rr_cd = min(Float32(1.0), w_cd.max_component())
            if rr_cd <= Float32(0.0) or pcg.next_float() >= rr_cd:
                break
            flux *= w_cd * (Float32(1.0) / rr_cd)
            rd = tx_cd * bs_cd.wi.x + ty_cd * bs_cd.wi.y + n_cd * bs_cd.wi.z
            ro = hit + rd * Float32(0.0002)
            continue

        if mat.type == MatKind.diffuse or mat.type == MatKind.diffuse_transmit:
            # n_events > 1: skip storing a photon at a surface directly hit
            # by the light with no intermediate bounce — that direct
            # contribution is now covered by _sppm_nee_update instead
            # (matches pbrt's own SPPM, which skips gathering at depth 0
            # for the same reason: avoid double-counting direct light
            # once via NEE and again via an unfiltered photon density).
            if n_events > 1:
                _sppm_store_photon[use_gpu](
                    SPPMPhoton(pos=hit, flux=flux, nxt=Int32(-1), is_volume=PhotonKind.surface, dir_in=rd, wavelengths=ph_wavelengths),
                    photons, max_photons, counter)
            # Real image-texture reflectance -- see bdpt.mojo's matching
            # light-side comment (task #150/#151). Affects the RR
            # continuation probability AND the flux multiply below, since a
            # wrong (flat-grey) albedo here would corrupt every subsequent
            # bounce's stored photon flux, not just this vertex.
            var eff_alb = mat.albedo
            var (tex_mesh, tv0, tv1, tv2, tex_ok) = _get_tri_verts(inter, sd.meshes)
            if tex_ok:
                eff_alb = _tex_lookup[tex_gpu](mat, inter, tv0, tv1, tv2, tex_mesh, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            # diffusetransmission has a REFLECT lobe (eff_alb, computed
            # above) and a TRANSMIT lobe (mat.emission, or eff_alb again
            # when textured -- "texture reflectance"/"texture transmittance"
            # share one texture slot, see shading.mojo's matching comment),
            # picked stochastically by luminance, same convention as the
            # shared bxdf.mojo:bxdf_sample_diffuse_transmit the path tracer
            # uses and bdpt.mojo's two camera/light-path copies (see
            # project_photon_estimator_energy_gap memory for the 1e-6-clamp
            # bug those two had and its fix, d2a3cd84 -- NOT reproduced
            # here). Before this, the photon pass ignored the transmit lobe
            # entirely and always reflected off the ray-facing side, so a
            # photon could never pass THROUGH a leaf: any indirect light
            # that should reach a shadowed point by transmitting through
            # backlit foliage was simply absent from the photon map (only
            # SPPM's own single-bounce direct NEE could ever see it).
            # `p_sel == 1, bounce_n == gn, lobe_alb == eff_alb` for every
            # other material -- a no-op, no extra random draw.
            var p_sel = Float32(1.0)
            var bounce_side = Float32(1.0)   # flips to -1 for the transmit lobe
            var lobe_alb = eff_alb
            if mat.type == MatKind.diffuse_transmit:
                var trans_dt = eff_alb if Int(mat.tex_idx) != -1 else mat.emission
                var pr_dt = eff_alb.luma()
                var pt_dt = trans_dt.luma()
                # Nothing to scatter -- see bdpt.mojo's matching comment
                # (d2a3cd84) for why this must TERMINATE, not force a lobe.
                if pr_dt + pt_dt <= Float32(1e-9):
                    break
                var tot_dt = pr_dt + pt_dt
                var take_refl = pcg.next_float() < pr_dt / tot_dt
                p_sel = max((pr_dt if take_refl else pt_dt) / tot_dt, Float32(1e-6))
                bounce_side = Float32(1.0) if take_refl else Float32(-1.0)
                lobe_alb = eff_alb if take_refl else trans_dt
            # Russian-roulette continuation for indirect diffuse-diffuse
            # bounces (color bleeding) — without this, photons always
            # terminated at the first diffuse hit, so light could never
            # bounce off one diffuse surface onto another (e.g. a red
            # wall tinting a nearby box's facing side). Uses the CHOSEN
            # lobe's own albedo (== eff_alb, unchanged, for plain diffuse).
            var rr_prob = max(lobe_alb.r, max(lobe_alb.g, lobe_alb.b))
            if rr_prob <= Float32(0.0) or pcg.next_float() >= rr_prob:
                break
            var gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            if dot(gn, ray_dir) > Float32(0.0):
                gn = gn * Float32(-1.0)
            # Bump/normal maps -- see the coateddiffuse branch above.
            gn = apply_surface_maps_at_hit[tex_gpu](mat, inter, sd.meshes,
                gn, gn, ray_dir, _camera_approx_footprint(hit, cam_pos, px_scale),
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            gn = face_toward(gn, -ray_dir)   # pbrt two-sided reflection, see face_toward
            gn = gn * bounce_side   # flip to the transmit side, chosen above
            var new_dir = _cosine_hemisphere_sample(gn, pcg.next_float(), pcg.next_float())
            flux *= spec_refl_unbounded(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, (lobe_alb / (rr_prob * p_sel)).r, (lobe_alb / (rr_prob * p_sel)).g, (lobe_alb / (rr_prob * p_sel)).b, ph_wavelengths)
            rd = vec3f(new_dir)
            ro = hit + rd * Float32(0.0002)   # along rd: gn may face into the surface (face_toward)
            continue

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            # ── Subsurface boundary: deposit on the SURFACE and stop ───────
            # The photon's remaining transport is what the diffusion kernel
            # models, so entering the medium and random-walking would double
            # count it. Flux is attenuated by the boundary's Fresnel
            # TRANSMITTANCE, which is the fraction that actually gets in;
            # R_d then carries it from here to wherever it re-emerges.
            # is_volume = 2 marks a BSSRDF surface photon, which only a
            # BSSRDF visible point gathers.
            # NOT gated on bounce > 0, unlike a surface photon. That gate
            # exists to avoid double counting the light's first segment against
            # the visible point's own NEE -- and a BSSRDF visible point has no
            # NEE (there is no BRDF at it; all of its light arrives as photons).
            # head.pbrt is lit by an environment map, so nearly every photon
            # reaches the skin on its FIRST segment: with the gate, essentially
            # nothing was deposited and the render came out pure black.
            # `has_media` matters: a BSSRDF deposit TERMINATES the photon, so
            # without a subsurface medium to terminate into this branch just
            # eats the photon. The camera side already required a real interior
            # (medium_after_crossing >= 0) and VCM already required `has_med`;
            # this one tested sss_boundary alone, which is what let the
            # uninitialised flag above reach anything at all.
            if has_media and mat.sss_boundary != Int8(0) and Int(cur_med_idx) < 0:
                var gn_b = _shading_normal_at(inter, sd.meshes, sd.instances, sd.spheres, hit)
                var cos_in = dot(gn_b, rd)
                if cos_in > Float32(0.0):
                    cos_in = -cos_in
                var ft = Float32(1.0) - fr_dielectric(-cos_in, mat.albedo.r)
                if ft > Float32(0.0):
                    _sppm_store_photon[use_gpu](
                        SPPMPhoton(pos=hit, flux=flux * ft, nxt=Int32(-1),
                                   is_volume=PhotonKind.bssrdf, dir_in=rd,
                                   wavelengths=ph_wavelengths),
                        photons, max_photons, counter)
                break
            # Entering, leaving or total-internal-reflecting at the boundary of
            # a subsurface interior is part of the ONE BSSRDF event the walk
            # inside it belongs to, so it is not charged to maxdepth. Same rule
            # as the path tracer (Material_C.sss_boundary).
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
            gn = apply_surface_maps_at_hit[tex_gpu](mat, inter, sd.meshes,
                gn, gn, ray_dir, _camera_approx_footprint(hit, cam_pos, px_scale),
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            var (new_dir, new_org, _, new_cur_ior, new_prev_ior) = _dielectric_bounce(
                ray_dir, hit.to_simd(), gn, ior, n_events == 1 and Int(cur_med_idx) < 0, pcg, current_dielectric_ior, previous_dielectric_ior, mat.type == MatKind.thin_dielectric, radiance_mode=False)
            current_dielectric_ior = new_cur_ior
            previous_dielectric_ior = new_prev_ior
            # Light path (TransportMode::Importance): do NOT apply the
            # radiance_scale non-symmetric-scattering correction to flux —
            # it's only for camera/Radiance-mode paths, see
            # _dielectric_bounce's docstring.
            rd = vec3f(new_dir)
            ro = point3f(new_org)
            if has_media:
                var new_idx = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                    cur_med_idx = new_idx

        elif mat.type == MatKind.conductor or mat.type == MatKind.coated_conductor:
            # Rough conductor/coated_conductor: store a gatherable photon
            # (same n_events > 1 depth-0 double-count guard as diffuse) unless
            # the sampled lobe is a perfect-mirror delta, which just
            # continues the path — mirrors bdpt.mojo's light-path treatment.
            var gn_c = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            if dot(gn_c, ray_dir) > Float32(0.0):
                gn_c = gn_c * Float32(-1.0)
            # Bump/normal maps -- see the coateddiffuse branch above.
            gn_c = apply_surface_maps_at_hit[tex_gpu](mat, inter, sd.meshes,
                gn_c, gn_c, ray_dir, _camera_approx_footprint(hit, cam_pos, px_scale),
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            gn_c = face_toward(gn_c, -ray_dir)   # pbrt two-sided reflection, see face_toward
            var wo_c = (-rd).to_simd()
            # Real image-texture F0, exactly as the diffuse branch does for
            # reflectance. Without it a TEXTURED conductor was stored/weighted
            # with its flat `mat.albedo` -- the scaffolding default for many
            # corpus materials -- so it rendered as one colour with no image.
            # kroken's framed wall pictures came out BLACK for this reason
            # (that scene has 6 coatedconductor + 3 conductor materials, and
            # this branch serves both). Same defect the diffuse branch fixed
            # under task #150/#151; the fix never reached conductor. BOTH the
            # camera and photon sides need it, or the two disagree on the
            # surface's own colour.
            var eff_alb_c = mat.albedo
            var (tmc, tvc0, tvc1, tvc2, tex_ok_c) = _get_tri_verts(inter, sd.meshes)
            if tex_ok_c:
                eff_alb_c = _tex_lookup[use_gpu](mat, inter, tvc0, tvc1, tvc2, tmc, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            var frm_c = Frame.from_z(Vec3f(gn_c[0], gn_c[1], gn_c[2]))
            var gc_c = GeomContext(
                normal=gn_c, geo_normal=gn_c, hit_point=hit.to_simd(), wo=wo_c,
                tangent=Vec3f(frm_c.x.x, frm_c.x.y, frm_c.x.z),
                bitangent=Vec3f(frm_c.y.x, frm_c.y.y, frm_c.y.z),
                alb=eff_alb_c, pixel_uv=Float32(0),
            )
            var uc1 = pcg.next_float(); var uc2 = pcg.next_float()
            var bs_c: BxDFSample
            if mat.type == MatKind.conductor:
                bs_c = bxdf_sample_conductor(gc_c, mat, uc1, uc2)
            else:
                var ior_c = mat.emission.r if mat.emission.r > Float32(1) else Float32(1.5)
                var usplit_c = pcg.next_float()
                bs_c = bxdf_sample_coated_conductor(gc_c, mat, ior_c, usplit_c, uc1, uc2)
            # Same ordering as the camera side above, same reason: this
            # photon has ALREADY ARRIVED at the surface, so it must be
            # deposited whether or not its OUTGOING sample is valid. Only
            # the continuation below consumes bs_c.
            if not bxdf_is_delta(bs_c.flags) and n_events > 1:
                _sppm_store_photon[use_gpu](
                    SPPMPhoton(pos=hit, flux=flux, nxt=Int32(-1), is_volume=PhotonKind.surface, dir_in=rd, wavelengths=ph_wavelengths),
                    photons, max_photons, counter)
            if bs_c.is_valid == Int8(0):
                break
            flux *= spec_refl_unbounded(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, (bs_c.f).r, (bs_c.f).g, (bs_c.f).b, ph_wavelengths)
            rd = vec3f(bs_c.wi)
            ro = hit + rd * Float32(0.0002)

        elif mat.type == MatKind.hair:
            # No delta lobe, so always store (subject to the same depth-0
            # guard) and always importance-sample a continuation direction —
            # mirrors bdpt.mojo's light-path hair treatment.
            var curve_idx_h = Int(inter.primId.id1)
            var wo_h = (-rd).to_simd()
            var hc = _hair_precompute(mat, sd.curves, curve_idx_h, inter.v, inter.u, wo_h)
            if n_events > 1:
                _sppm_store_photon[use_gpu](
                    SPPMPhoton(pos=hit, flux=flux, nxt=Int32(-1), is_volume=PhotonKind.surface, dir_in=rd, wavelengths=ph_wavelengths),
                    photons, max_photons, counter)
            var (wi_hs, f_hs, pdf_hs, _) = _hair_sample_dir(hc, pcg)
            flux *= spec_refl_unbounded(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, (f_hs / pdf_hs).r, (f_hs / pdf_hs).g, (f_hs / pdf_hs).b, ph_wavelengths)
            rd = vec3f(wi_hs)
            var hsign = Float32(1) if dot(wi_hs, hc.geo_normal) >= Float32(0) else Float32(-1)
            ro = hit + vec3f(hc.geo_normal) * curve_offset_eps(hc.radius) * hsign

        elif mat.type == MatKind.measured:
            # No delta lobe, so always store (same depth-0 guard as the
            # other branches) and always importance-sample a continuation
            # direction -- mirrors bdpt.mojo's light-path measured
            # treatment, via the same shared bxdf.mojo/measured_bxdf_eval.mojo
            # interface.
            var gn_m = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            if dot(gn_m, ray_dir) > Float32(0.0):
                gn_m = gn_m * Float32(-1.0)
            # Bump/normal maps -- see the coateddiffuse branch above.
            gn_m = apply_surface_maps_at_hit[tex_gpu](mat, inter, sd.meshes,
                gn_m, gn_m, ray_dir, _camera_approx_footprint(hit, cam_pos, px_scale),
                sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            gn_m = face_toward(gn_m, -ray_dir)   # pbrt two-sided reflection, see face_toward
            if mat.measured_idx < Int32(0):
                break
            var wo_m = (-rd).to_simd()
            var frm_m = Frame.from_z(Vec3f(gn_m[0], gn_m[1], gn_m[2]))
            var tangent_m = Vec3f(frm_m.x.x, frm_m.x.y, frm_m.x.z)
            var bitangent_m = Vec3f(frm_m.y.x, frm_m.y.y, frm_m.y.z)
            var mb_m = sd.measuredBrdfs[unsafe_offset=Int(mat.measured_idx)]
            var wo_l_m = Vec3f(dot(wo_m, tangent_m), dot(wo_m, bitangent_m), dot(wo_m, gn_m))
            # Deposited BEFORE sampling the continuation, like the conductor
            # branch: the photon arrived whether or not its outgoing sample
            # is valid.
            if n_events > 1:
                _sppm_store_photon[use_gpu](
                    SPPMPhoton(pos=hit, flux=flux, nxt=Int32(-1), is_volume=PhotonKind.surface, dir_in=rd, wavelengths=ph_wavelengths),
                    photons, max_photons, counter)
            var uml1 = pcg.next_float(); var uml2 = pcg.next_float()
            var (wi_l_m, f_m, pdf_m, valid_m) = bxdf_sample_measured(mb_m, wo_l_m, uml1, uml2, ph_wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
            if not valid_m or pdf_m <= Float32(0):
                break
            var wi_m = tangent_m * wi_l_m[0] + bitangent_m * wi_l_m[1] + gn_m * wi_l_m[2]
            var wilen_m = dot(wi_m, wi_m)
            if wilen_m > Float32(0):
                wi_m = wi_m * (Float32(1.0) / sqrt(wilen_m))
            var cos_wi_m = dot(wi_m, gn_m)
            if cos_wi_m <= Float32(0):
                break
            # The ADJOINT BRDF, as in bdpt.mojo's light path (LobeCtx.adjoint):
            # a photon needs f(toward camera, toward light); f_m is the other
            # order, and this table is not reciprocal near grazing.
            var f_adj_m = bxdf_eval_measured(mb_m, wi_l_m, wo_l_m, ph_wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)[0]
            flux *= f_adj_m * (cos_wi_m / pdf_m)
            rd = vec3f(wi_m)
            ro = hit + rd * Float32(0.0002)

        elif mat.type == MatKind.interface:
            if has_media:
                var new_idx = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                    cur_med_idx = new_idx
            ro = hit + rd * Float32(0.0002)

        else:
            # area_light self-hit, etc.: absorb
            break


@always_inline
def _sppm_has_sphere_lights(ref sd: SceneDescriptor2_C) -> Bool:
    """Whether any analytic sphere is an area light. Cheap O(sphereCount)
    scan — sd.spheres holds ALL analytic spheres (light or not), not a
    pre-filtered lights-only array like every other light type, so this is
    the only way to know without a dedicated count. Used only at
    once-per-render/once-per-pass guard sites (never per-pixel/per-photon) —
    NOT folded into _sppm_photon_pass's own light-selection pool below,
    since sphere-light photon/light-path EMISSION isn't supported (NEE only,
    see _sppm_nee_one) — this only prevents those guards from wrongly
    treating "no emittable lights" as "no lights at all" and skipping NEE
    for a scene lit purely by sphere lights."""
    for i in range(Int(sd.sphereCount)):
        if sd.spheres[unsafe_offset=i].isAreaLight != Int8(0):
            return True
    return False

def _sppm_photon_pass(
    photons:      Pointer[SPPMPhoton, MutUntrackedOrigin],
    n_emit:       Int,
    max_photons:  Int,
    ref sd:           SceneDescriptor2_C,
    seed:         UInt64,
    pass_idx:     Int,
    maxdepth:     Int,
    cam_pos:      Vec3f,
    px_scale:     Float32,
) -> Int:
    """CPU driver: emit n_emit photon paths, returning the number actually
    stored (clamped to max_photons)."""
    var n_lights = Int(sd.areaLightCount) + Int(sd.distantLightCount) + Int(sd.infiniteLightCount) + Int(sd.pointLightCount)
    if n_lights == 0:
        return 0

    # One scratch Intersection per worker (indexed by k), same convention
    # as the GPU kernel's inter_scratch + k — needed now that this loop runs
    # across CPU threads too.
    var scratch = unsafe_alloc[Intersection](max(n_emit, 1))
    var counter = unsafe_alloc[Int32](1)
    counter[unsafe_offset=0] = Int32(0)

    # Determine the "default" starting medium for photons emitted into a medium.
    # Convention: photon is cosine-sampled from ln, so dot(pdir,ln)>0 always.
    # We use the first MediumInterface whose outside_medium_idx is valid.
    var default_emit_med = Int32(-1)
    if Int(sd.mediumCount) > 0 and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[unsafe_offset=mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    # use_gpu=True: parallel CPU workers now need the SAME atomic slot-
    # reservation _sppm_store_photon uses for GPU threads (concurrent
    # racing writers to the shared `photons` buffer/`counter`), regardless
    # of which backend is actually running. tex_gpu=False: this driver IS
    # actually CPU, so _tex_lookup must use the CPU tex_filenames path
    # (see _sppm_trace_photon's docstring for why these are separate params).
    def emit_one(k: Int) {imm}:
        var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))
        _sppm_trace_photon[True, False](sd, pcg, scratch.unsafe_offset(k), n_emit, photons, max_photons, counter, default_emit_med, maxdepth,
            cam_pos, px_scale,
            sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y,
            sd.spectral.cie_z, sd.spectral.d65, pass_wavelengths(pass_idx))

    parallelize(emit_one, n_emit)

    var n_stored = min(Int(counter[unsafe_offset=0]), max_photons)
    counter.unsafe_free()
    scratch.unsafe_free()
    return n_stored


# ── Hash grid ─────────────────────────────────────────────────────────────────

def _sppm_reset_grid_cell(heads: Pointer[Int32, MutUntrackedOrigin], h: Int):
    heads[unsafe_offset=h] = Int32(-1)


def _sppm_insert_photon[use_gpu: Bool](
    k:        Int,
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
):
    """Insert stored photon `k` into the hash grid. Comptime-branches only on
    the bucket-head update primitive (atomic exchange for racing GPU threads
    vs. a plain read-modify-write for the serial CPU loop)."""
    var ix = Int(floor(photons[unsafe_offset=k].pos.x * inv_cell))
    var iy = Int(floor(photons[unsafe_offset=k].pos.y * inv_cell))
    var iz = Int(floor(photons[unsafe_offset=k].pos.z * inv_cell))
    var h = _hash_cell(ix, iy, iz)
    comptime if use_gpu:
        var old = Atomic._xchg(heads.unsafe_offset(h), Int32(k))
        photons[unsafe_offset=k].nxt = old
    else:
        photons[unsafe_offset=k].nxt = heads[unsafe_offset=h]
        heads[unsafe_offset=h] = Int32(k)


def _build_grid(
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    n_phot:   Int,
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
):
    def reset_one(i: Int) {imm}:
        _sppm_reset_grid_cell(heads, i)

    parallelize(reset_one, _HSIZE)

    # [True]: parallel CPU workers race on the same bucket heads a GPU
    # kernel's threads would, so need the same atomic-exchange insert.
    def insert_one(k: Int) {imm}:
        _sppm_insert_photon[True](k, photons, heads, inv_cell)

    parallelize(insert_one, n_phot)


# ── Gather + SPPM update ──────────────────────────────────────────────────────

@always_inline
def gather_disk_contains(e: Vec3f, dist2: Float32, r2: Float32, n: Vec3f) -> Bool:
    """Does a photon at offset `e` from a gather point lie inside the gather
    DISK -- radius sqrt(r2), on the tangent plane with normal `n` -- rather
    than merely inside the BALL of that radius?

    THE photon-acceptance test for a surface density estimate, shared by
    SPPM's gather and VCM's merge. A surface estimate divides the photons it
    finds by one disk's area, pi r^2, so it is only right if the photons came
    from that disk. A ball also sweeps in photons lying on OTHER surfaces
    within reach -- an adjacent wall, a step, the parallel surface below a
    shelf -- and sums them against the same single disk. VCM found this first
    (classroom read 2.5x pbrt with merging on, 0.985x with it off) and grew
    this guard; SPPM never did, and its gather radius in a room-sized scene is
    large (0.51 scene units in barcelona-pavilion, around its pool). A photon
    on this surface sits on the tangent plane to float precision, so a tenth
    of the radius is generous."""
    if dist2 > r2:
        return False
    var off = dot(e, n)
    return off * off <= r2 * Float32(0.01)

@always_inline
def _sppm_photon_on_vp_surface(vp_is_volume: Int32, vp_mat_kind: Int32, vp_normal: Vec3f,
                               e: Vec3f, dist2: Float32, r2: Float32, dir_in: Vec3f,
                               bssrdf_ok: Bool) -> Bool:
    """Whether a photon inside the gather ball may count toward a SURFACE
    visible point at all. Kinds whose estimate is not a flat-disk one are
    exempt: BSSRDF (diffusion reaches across curved skin), hair (a fibre has
    no tangent plane), and volume points (a 3-D estimate by construction).

    An opaque Lambertian point also requires the photon to have ARRIVED from
    the side the point faces -- its f = alb/pi is pulled out of the sum as
    angle-independent, so without this a photon on the back of a thin panel
    lit the front. A transmission lobe legitimately takes both sides, and the
    GGX/measured branches evaluate their BRDF per photon, which is already
    ~0 for a direction below the surface."""
    if vp_is_volume != PhotonKind.surface or bssrdf_ok:
        return True
    if vp_mat_kind == LobeKind.hair or vp_mat_kind == LobeKind.bssrdf:
        return True
    if not gather_disk_contains(e, dist2, r2, vp_normal):
        return False
    if vp_mat_kind == LobeKind.lambertian:
        return dot(dir_in, vp_normal) < Float32(0.0)
    return True

def _sppm_gather_one(
    vps:      Pointer[SPPMPixel, MutUntrackedOrigin],
    i:        Int,
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    ref sd:       SceneDescriptor2_C,
    # Decomposed for exactly the reason the spectral tables below are: `sd` is
    # passed BY VALUE and its struct members do not read back reliably inside
    # the GPU kernel. Reading sd.mediumCount here returned 0 while the driver
    # held the real count, so every BSSRDF visible point silently fell back to
    # looking for SURFACE photons on a surface that only has BSSRDF ones, and
    # gathered nothing at all.
    med_arr:   Pointer[Medium, MutUntrackedOrigin],
    med_count: Int,
    # Density fields, decomposed for the same reason: the gather kernel's own
    # `sd` carries dangling grid pointers, so a heterogeneous medium's density
    # lookup must not go through it.
    grids_arr: Pointer[Grid, MutUntrackedOrigin],
    nvdb_arr:  Pointer[NvdbGrid, MutUntrackedOrigin],
    # Decomposed spectral tables rather than reading sd.spectral. `sd` is a
    # SceneDescriptor2_C passed BY VALUE, and it contains a SpectralHandle --
    # the 6-field TrivialRegisterPassable struct suspected (modular/modular#6759,
    # later retracted by its own author as unreproducible) of corrupting
    # under by-value passing across a real call boundary. Whatever the cause,
    # decomposing fixed a real observed symptom here: measured here, the GPU
    # kernel held sd.spectral.res == 64 immediately before the call and this
    # function read 0 from the same field, silently taking every conversion's
    # table-less path while the photons it was gathering had been built WITH
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
    pass_wl:  SampledWavelengths,
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
    var r2 = vp.r2

    # Accumulate contributions from photons in 3x3x3 neighborhood
    # Gathered spectrally at THIS pass's wavelengths, then converted to RGB
    # exactly once on the way into `tau` (see SPPMPixel.tau).
    var phi = SpectralSample(Float32(0))
    var M = Float32(0)
    # BSSRDF constants, hoisted out of the photon loop.
    var bssrdf_r2 = r2
    var bssrdf_ft_o = Float32(1.0)
    # The medium is read ONCE here, never inside the photon loop: an
    # unguarded sd.mediums[...] per photon is an illegal access on the GPU the
    # moment med_idx is out of range, and it aborted every --sppm run.
    var bssrdf_ok = (vp.mat_kind == LobeKind.bssrdf and Int(vp.med_idx) >= 0
                     and Int(vp.med_idx) < med_count and _is_real_ptr[Medium](med_arr))
    var bssrdf_ss = RGB(Float32(0))
    var bssrdf_sa = RGB(Float32(0))
    var bssrdf_g  = Float32(0)
    if bssrdf_ok:
        var mb0 = med_arr[unsafe_offset=Int(vp.med_idx)]
        bssrdf_ss = mb0.sigma_s
        bssrdf_sa = mb0.sigma_a
        bssrdf_g  = mb0.g
        var rmax = dipole_max_radius(mb0.sigma_s, mb0.sigma_a, mb0.g)
        bssrdf_r2 = rmax * rmax
        # Fresnel TRANSMITTANCE on the way out, for the view direction.
        var cos_o = abs(dot(vp.normal.to_simd(), vp.wo.to_simd()))
        bssrdf_ft_o = Float32(1.0) - fr_dielectric(cos_o, vp.alpha)

    var cix = Int(floor(vp.pos.x * inv_cell))
    var ciy = Int(floor(vp.pos.y * inv_cell))
    var ciz = Int(floor(vp.pos.z * inv_cell))
    for ddx in range(-1, 2):
        for ddy in range(-1, 2):
            for ddz in range(-1, 2):
                var h = _hash_cell(cix + ddx, ciy + ddy, ciz + ddz)
                var k = Int(heads[unsafe_offset=h])
                while k != -1:
                    var ph = photons[unsafe_offset=k]
                    var e = ph.pos - vp.pos
                    var dist2 = e.length_sq()
                    # A BSSRDF visible point gathers only BSSRDF surface
                    # photons (is_volume == 2), and out to the DIFFUSION
                    # truncation distance rather than the shrinking SPPM
                    # radius -- R_d is a normalised kernel, so its reach is a
                    # material property, not a bias parameter.
                    var want_kind = PhotonKind.bssrdf if bssrdf_ok else vp.is_volume
                    var reach2 = bssrdf_r2 if bssrdf_ok else r2
                    if dist2 <= reach2 and ph.is_volume == want_kind and _sppm_photon_on_vp_surface(
                            Int32(vp.is_volume), Int32(vp.mat_kind), vp.geo_normal.to_simd(), e.to_simd(), dist2, r2,
                            ph.dir_in.to_simd(), bssrdf_ok):
                        # Volume VP: isotropic phase f=alb/(4π). Surface VP:
                        # Lambertian f=alb/π (angle-independent, pulled out
                        # of the sum) or, for a conductor VP, the raw GGX
                        # BRDF evaluated per-photon (wi reconstructed from
                        # the photon's stored incoming travel direction) —
                        # NOT multiplied by any extra cosine, since the
                        # photon's flux already encodes the appropriate
                        # cosine-weighted density (same convention the
                        # Lambertian/phase branches already rely on).
                        if vp.mat_kind == LobeKind.bssrdf and bssrdf_ok:
                            # DIFFUSION BSSRDF. L_o(xo,wo) = (Ft(wo)/pi) *
                            # sum_p R_d(|xi_p - xo|) * Phi_p  (Jensen & Buhler
                            # 2002). Note what is NOT here: no 1/(pi r^2), no
                            # kernel volume, no radius at all. R_d already
                            # integrates over the plane to the material's
                            # diffuse albedo, so the photons' flux is carried
                            # exactly -- which is why this path has no
                            # bias/radius tradeoff to lose.
                            var rr = sqrt(dist2)
                            var rd_rgb = dipole_rd(bssrdf_ss, bssrdf_sa, bssrdf_g, vp.alpha, rr)
                            # UNBOUNDED upsampler. R_d is a density with units
                            # of 1/area, not a reflectance: for skin (mfp
                            # ~0.001) it is on the order of 1e4. spec_refl
                            # clamps to [0,1], which silently flattened every
                            # value to 1 and rendered the head ~200x too dark.
                            # Same trap as the throughput weights earlier --
                            # a coefficient is not a colour.
                            phi += spec_refl_unbounded(spectral_coeffs, spectral_res,
                                             spectral_cie_x, spectral_cie_y,
                                             spectral_cie_z, spectral_d65,
                                             rd_rgb.r * bssrdf_ft_o * (Float32(1.0) / PI),
                                             rd_rgb.g * bssrdf_ft_o * (Float32(1.0) / PI),
                                             rd_rgb.b * bssrdf_ft_o * (Float32(1.0) / PI),
                                             ph.wavelengths) * ph.flux
                        elif vp.is_volume == PhotonKind.volume:
                            # VOLUME radiance estimate, which is NOT the surface
                            # one with a different kernel volume.
                            #
                            # Photons stored in a medium sit at SCATTERING
                            # EVENTS, and the density of those events is itself
                            # proportional to sigma_s: flux deposited in dV from
                            # direction w' is sigma_s * L(x,w') dV dw'. Recovering
                            # radiance therefore needs a 1/sigma_s that a surface
                            # gather has no analogue of (Jensen's volumetric
                            # radiance estimate). It was missing, so the estimate
                            # was too bright by exactly sigma_s -- invisible in
                            # thin fog and ruinous in anything dense. Measured on
                            # a plain homogeneous scatterer of albedo 0.9 against
                            # the path tracer, with no subsurface material
                            # anywhere: 1.67x at sigma_s = 1, 10.5x at 10, 55.9x
                            # at 100. That is the whole of the "SPPM subsurface is
                            # 33-142x hot" symptom -- skin is simply a very dense
                            # medium.
                            #
                            # `vp.alb` (= sigma_s/sigma_t) is applied here rather
                            # than folded into vp.beta, mirroring how a surface VP
                            # defers its BRDF, so alb/sigma_s collapses to
                            # 1/sigma_t. Using that form directly is both cheaper
                            # and safer: sigma_t is never zero inside a medium,
                            # and it avoids dividing two INDEPENDENTLY upsampled
                            # spectra, which is exactly what made the subsurface
                            # albedo exceed 1 in 6de092a1.
                            # Guard the index: a VP whose medium is unknown
                            # (or a descriptor with no medium table) must not
                            # index the array -- dereferencing it crashed every
                            # --sppm run here. NOTE: no `continue` in this
                            # guard. The enclosing loop is a `while k != -1`
                            # walk down a photon linked list whose `k` advance
                            # sits at the BOTTOM, so a `continue` never
                            # advances it and the render hangs forever (it did,
                            # for 30 minutes, before this was written as a
                            # plain conditional).
                            var mi_v = Int(vp.med_idx)
                            # med_arr/med_count, NOT sd.mediums/sd.mediumCount:
                            # the GPU gather kernel builds its sd with a
                            # DANGLING medium table and count 0, so reading it
                            # here made ok_med always false and every volume
                            # visible point gathered exactly zero -- SPPM in
                            # any participating medium was its direct-lighting
                            # term alone (measured: 0.29-0.52x of the path
                            # tracer, and completely insensitive to
                            # --sppm-photons, which is what gave it away).
                            # Same decomposition the BSSRDF path above already
                            # had for this exact reason; the volume branch was
                            # missed.
                            var ok_med = (mi_v >= 0 and mi_v < med_count
                                          and _is_real_ptr[Medium](med_arr))
                            var medv = med_arr[unsafe_offset=mi_v] if ok_med else Medium(
                                sigma_a=RGB(Float32(0)), sigma_s=RGB(Float32(1)),
                                g=Float32(0), grid_idx=Int32(-1), nvdb_idx=Int32(-1),
                                nvdb_temp_idx=Int32(-1), le_scale=Float32(0),
                                temp_offset=Float32(0), temp_scale=Float32(1),
                                is_sss=Int32(0))
                            var sig_t_spec = medium_sigma_t_spectral(
                                medv, ph.wavelengths, spectral_coeffs, spectral_res,
                                spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
                            # Heterogeneous media author sigma per UNIT DENSITY,
                            # so the extinction actually in force at the VP is
                            # density(x) * sigma_t (same rule Medium documents).
                            var dens = Float32(1.0)
                            if medium_is_heterogeneous(medv):
                                var gvol = medium_grid_for(medv, grids_arr)
                                var nvol = medium_nvdb_for(medv, nvdb_arr)
                                if medv.nvdb_idx >= Int32(0):
                                    dens = nvdb_sample_density(nvol, vp.pos.to_simd())
                                else:
                                    dens = grid_sample_density(gvol, vp.pos.to_simd())
                            var inv0 = Float32(1.0) / max(sig_t_spec.v0 * dens, Float32(1e-12))
                            var inv1 = Float32(1.0) / max(sig_t_spec.v1 * dens, Float32(1e-12))
                            var inv2 = Float32(1.0) / max(sig_t_spec.v2 * dens, Float32(1e-12))
                            var inv3 = Float32(1.0) / max(sig_t_spec.v3 * dens, Float32(1e-12))
                            if ok_med:
                                # Keep `alb`, and divide by sigma_t (NOT
                                # sigma_s): this implementation stores a photon
                                # at every COLLISION, before the scatter/absorb
                                # decision, so deposit density goes as sigma_t
                                # and the flux is pre-albedo. Jensen's 1/sigma_s
                                # assumes deposits at SCATTERING events; matching
                                # the divisor to where photons are actually
                                # stored is what makes the two conventions agree.
                                phi += spec_refl(spectral_coeffs, spectral_res,
                                                 spectral_cie_x, spectral_cie_y,
                                                 spectral_cie_z, spectral_d65,
                                                 vp.alb.r, vp.alb.g, vp.alb.b, ph.wavelengths) \
                                       * SpectralSample(inv0, inv1, inv2, inv3) * INV_FOUR_PI * ph.flux
                        else:
                            # THE lobe evaluator, bare f (the density
                            # estimate itself carries the cosine) -- one call
                            # for ggx/hair/measured/layered/lambertian/
                            # diffuse_transmit alike, replacing what was a
                            # 5-way dispatch here (bxdf_eval_conductor_ggx,
                            # _hair_eval_lobes, _sppm_vp_brdf, a bespoke
                            # LobeCtx for layered, and a bare vp.alb/PI
                            # fallback for everything else). That fallback
                            # was diffuse_transmit's own path too: every
                            # photon gathered at a diffusetransmission VP was
                            # charged the REFLECT lobe's colour regardless of
                            # which side it arrived from --
                            # _sppm_photon_on_vp_surface already admits
                            # photons from both sides for this kind, but
                            # nothing here picked the transmit colour for the
                            # back one. pre_oriented=True: the accept test
                            # above already enforces the correct side for a
                            # plain Lambertian VP, and every other kind
                            # decides sidedness itself.
                            var le = lobe_eval[want_pdfs=False](
                                LobeCtx(vp.mat_kind, True, False, vp.normal.to_simd(), vp.wo.to_simd(), vp.alb,
                                        vp.mat_idx, vp.alpha, Float32(0), vp.hair_curve_idx,
                                        vp.hair_h, vp.hair_v, True, False),
                                (-ph.dir_in).to_simd(), LobeTables(sd.materials, sd.curves, sd.measuredBrdfs),
                                spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                                ph.wavelengths)
                            if le.cos_used > Float32(1e-6):
                                phi += le.f_cos * (Float32(1.0) / le.cos_used) * ph.flux
                        M += Float32(1.0)
                    k = Int(ph.nxt)

    # SPPM update (only if new photons found)
    if M > Float32(0.0):
        var N = vp.N_acc
        var ratio = (N + _ALPHA * M) / (N + M)
        vps[unsafe_offset=i].r2  = r2 * ratio
        var (phi_r, phi_g, phi_b) = spectral_sample_to_rgb(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, phi, pass_wl)
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
        # A BSSRDF visible point has no kernel normalisation to track, so its
        # history must NOT be rescaled when the radius shrinks -- there is no
        # radius in its estimator at all.
        var tau_scale = Float32(1.0) if vp.mat_kind == LobeKind.bssrdf else (
            ratio * sqrt(ratio) if vp.is_volume == PhotonKind.volume else ratio)
        vps[unsafe_offset=i].tau = (vp.tau + RGB(phi_r, phi_g, phi_b)) * tau_scale
        vps[unsafe_offset=i].N_acc = N + _ALPHA * M


def _gather_update(
    vps:      Pointer[SPPMPixel, MutUntrackedOrigin],
    n_pix:    Int,
    photons:  Pointer[SPPMPhoton, MutUntrackedOrigin],
    heads:    Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    ref sd:       SceneDescriptor2_C,
    pass_wl:  SampledWavelengths,
):
    def gather_one(i: Int) {imm}:
        _sppm_gather_one(vps, i, photons, heads, inv_cell, sd, sd.mediums, Int(sd.mediumCount),
                         sd.grids, sd.nvdbGrids,
                         sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y,
                         sd.spectral.cie_z, sd.spectral.d65, pass_wl)

    parallelize(gather_one, n_pix)


# ── Direct (NEE) lighting update ──────────────────────────────────────────────
# Mirrors pbrt-v4's SPPM "pixel.Ld" term: a shadow-ray light sample taken at
# each visible point EVERY SPPM pass (same cadence as the photon pass),
# summed here and divided by n_passes at finalize time. Needed because the
# photon-density (tau) term alone has to represent BOTH direct and indirect
# lighting through one noisy channel — pbrt keeps them separate, which is
# why its SPPM output is much smoother/less "blotchy" for the same pass
# count. Handles surface VPs (BRDF) and volume VPs (isotropic phase,
# albedo/4pi with no cosine) alike, each attenuated by the transmittance of
# whatever medium the VP sits in.

@always_inline
def _sppm_vp_brdf(
    vp: SPPMPixel,
    ref sd: SceneDescriptor2_C,
    vn: Vec3f,
    wi: Vec3f,
) -> RGB:
    """Raw f_r(wo,wi) at a surface VP for NEE — dispatches on mat_kind, same
    "raw BRDF, cosine supplied externally via the caller's geom/cos_s/
    cos_env factor" convention as _sppm_gather_one. Lambertian is
    angle-independent (alb/π); conductor/hair both need the actual wi."""
    if vp.mat_kind == LobeKind.ggx:
        return bxdf_eval_conductor_ggx(vn, vp.wo.to_simd(), wi, vp.alpha, vp.alb)
    if vp.mat_kind == LobeKind.hair:
        var mat_h = sd.materials[unsafe_offset=Int(vp.mat_idx)]
        var hc = _hair_precompute(mat_h, sd.curves, Int(vp.hair_curve_idx), vp.hair_v, vp.hair_h, vp.wo.to_simd())
        var (_, f_h, _) = _hair_eval_lobes(
            wi, hc.tangent, hc.b_perp, hc.n_perp, hc.phi_o,
            hc.dphi0, hc.dphi1, hc.dphi2,
            hc.cos_tp0_o, hc.sin_tp0_o, hc.cos_tp1_o, hc.sin_tp1_o, hc.cos_tp2_o, hc.sin_tp2_o,
            hc.cos_theta_o, hc.sin_theta_o, hc.inv_vm0, hc.inv_vm1, hc.inv_vm2, hc.mp_c0, hc.mp_c1, hc.mp_c2, hc.s,
            hc.A0, hc.A1, hc.A2, hc.A3, hc.lum0, hc.lum1, hc.lum2, hc.lum3, hc.total_lum,
        )
        return f_h
    if vp.mat_kind == LobeKind.measured:
        # Measured BxDF is inherently spectral (tabulated `spectra` tensor
        # indexed by wavelength) -- does its own spectral eval + RGB
        # conversion internally using vp.wavelengths, same as bdpt.mojo's
        # _eval_vertex mat_kind=3 branch.
        var mat_m = sd.materials[unsafe_offset=Int(vp.mat_idx)]
        var mb_m = sd.measuredBrdfs[unsafe_offset=Int(mat_m.measured_idx)]
        var frm_m = Frame.from_z(Vec3f(vn[0], vn[1], vn[2]))
        var tangent_m = Vec3f(frm_m.x.x, frm_m.x.y, frm_m.x.z)
        var bitangent_m = Vec3f(frm_m.y.x, frm_m.y.y, frm_m.y.z)
        var wo_m = vp.wo.to_simd()
        var wo_l_m = Vec3f(dot(wo_m, tangent_m), dot(wo_m, bitangent_m), dot(wo_m, vn))
        var wi_l_m = Vec3f(dot(wi, tangent_m), dot(wi, bitangent_m), dot(wi, vn))
        var (fr_spec_m, _) = bxdf_eval_measured(mb_m, wo_l_m, wi_l_m, vp.wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
        var (r_m, g_m, b_m) = spectral_sample_to_rgb(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, fr_spec_m, vp.wavelengths)
        return RGB(r_m, g_m, b_m)
    return vp.alb / PI

@always_inline
def _sppm_shadow_transmittance(
    vp: SPPMPixel,
    ref sd: SceneDescriptor2_C,
    org: Point3f,
    wi: Vec3f,
    dist: Float32,
) -> RGB:
    """Beer-Lambert transmittance along a shadow ray leaving a VP that sits
    inside a participating medium. RGB(1) when the VP is in vacuum.

    Mirrors gpu.mojo's `_volume_nee_light` homogeneous branch: the closed
    form applies only over the span the ray actually spends INSIDE the
    medium, which ends at the medium's bounding interface. This function does
    not otherwise know where that is, so it finds it with an ordinary
    closest-hit query -- interface surfaces are invisible to `any_hit` (they
    must not occlude) but ARE visible to `traverse_bvh2_core`, so the first
    hit IS that shell. Without this clamp a light outside the medium would be
    attenuated across vacuum, the defect fixed for the path tracer in
    f79999f4 (see project_volume_area_light_nee_bug)."""
    if Int(vp.med_idx) < 0 or Int(sd.mediumCount) == 0:
        return RGB(Float32(1))
    var med = sd.mediums[unsafe_offset=Int(vp.med_idx)]
    if med.grid_idx >= Int32(0) or med.nvdb_idx >= Int32(0):
        # Heterogeneous: needs ratio tracking, not a closed form. SPPM only
        # ever samples homogeneous free flight today (see
        # sample_homogeneous_free_flight's own call sites), so a VP can only
        # be inside a homogeneous medium -- but fail open rather than
        # silently applying a wrong closed form if that ever changes.
        return RGB(Float32(1))
    var exit_i = Intersection(
        PrimId(Int64(0), Int64(0), Int64(-1), Int32(-1), Int8(0), 0, 0, 0),
        Float32(0), Float32(0), Float32(0), Int8(0), 0, 0, 0)
    traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves,
                       Ray(org, vec3f(wi)), dist, Pointer(to=exit_i),
                       sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                       sd.spheres, Int(sd.sphereCount))
    var span = dist if exit_i.hit == Int8(0) else exit_i.tHit
    var sigma_t = med.sigma_a + med.sigma_s
    return RGB(exp(-sigma_t.r * span), exp(-sigma_t.g * span), exp(-sigma_t.b * span))

@always_inline
@always_inline


def _sppm_nee_weight(
    vp: SPPMPixel,
    ref sd: SceneDescriptor2_C,
    vn: Vec3f,
    wo: Vec3f,
    ls: LightSample,
) -> SpectralSample:
    """Dispatches one LightSample (bvh.mojo's shared distant/point/sphere/
    infinite sampler output — see that struct's docstring) to the right
    per-material NEE weight function for a stored SPPM visible point, via
    bxdf.mojo's _nee_weight_simple/_nee_weight_hair — the shared "BxDF
    interface" half of the Light/BxDF refactor. Supersedes hand-deriving
    direction/pdf/MIS math once per light type inside _sppm_nee_one; area
    lights still go through _sppm_vp_brdf directly (see that function's own
    docstring for why area-light NEE isn't part of this shared interface)."""
    if vp.is_volume == PhotonKind.volume:
        # Isotropic phase function: albedo/4pi, and NO cosine factor -- a
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
        var ph_v = spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65,
                             vp.alb.r * INV_FOUR_PI, vp.alb.g * INV_FOUR_PI, vp.alb.b * INV_FOUR_PI, vp.wavelengths)
        var li_v = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65,
                              ls.Li.r, ls.Li.g, ls.Li.b, vp.wavelengths)
        return ph_v * li_v * (Float32(1.0) / ls.pdf)
    # THE NEE weight (bxdf.mojo), from the SAME LobeCtx the gather and every
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
    return nee_weight_lobe(ls,
        LobeCtx(vp.mat_kind, True, False, vn, wo, vp.alb, vp.mat_idx, vp.alpha,
                Float32(0), vp.hair_curve_idx, vp.hair_h, vp.hair_v, True, False),
        LobeTables(sd.materials, sd.curves, sd.measuredBrdfs),
        sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65,
        vp.wavelengths, mis_policy_sole())

@always_inline
def _sppm_vp_shadow_eps(vp: SPPMPixel, ref sd: SceneDescriptor2_C, wo: Vec3f) -> Float32:
    """Shadow-ray self-intersection offset for a stored SPPM visible point.
    Hair vertices need the same curve-radius-scaled epsilon as the ray-bounce
    offsets in shading.mojo/bdpt.mojo/sppm.mojo's own photon pass (see
    bvh.mojo's curve_offset_eps) -- the fixed 0.0001 that works fine for
    triangle/sphere hits was too coarse relative to a curve's own radius.
    Triangle/sphere-hit VPs are unaffected, unchanged fixed epsilon."""
    if vp.mat_kind == LobeKind.hair:
        var mat_h = sd.materials[unsafe_offset=Int(vp.mat_idx)]
        var hc = _hair_precompute(mat_h, sd.curves, Int(vp.hair_curve_idx), vp.hair_v, vp.hair_h, wo)
        return curve_offset_eps(hc.radius)
    return Float32(0.0001)

@always_inline
def _sppm_simple_light_count(ref sd: SceneDescriptor2_C) -> Int:
    """Sibling of shading.mojo's _nee_simple_light_count / bdpt.mojo's
    _bdpt_simple_light_count -- a third, independent definition rather than
    a shared one, because sppm.mojo is typed against SceneDescriptor2_C (like
    bdpt.mojo) but needs the (LightSample, tmax) PAIRING (like shading.mojo's
    ShadeContext version), and bdpt.mojo already imports from sppm.mojo (for
    _sppm_trace_visible_point etc.), so importing back the other way would be
    circular. distant + point + sphere; area and infinite stay at their own
    call sites, same rationale as the other two files."""
    return Int(sd.distantLightCount) + Int(sd.pointLightCount) + Int(sd.sphereCount)


@always_inline
def _sppm_sample_simple_light(
    ref sd: SceneDescriptor2_C, i: Int, hit_point: Vec3f, mut pcg: PCG32,
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


def _sppm_nee_one(
    vps:     Pointer[SPPMPixel, MutUntrackedOrigin],
    i:       Int,
    ref sd:      SceneDescriptor2_C,
    mut pcg: PCG32,
):
    """Direct (NEE) lighting update for visible point `i`. Shared verbatim
    between the CPU driver (_sppm_nee_update, one call per VP in a loop) and
    the GPU kernel (sppm_nee_gpu, one call per thread) — no divergence, each
    call only ever touches its own vps[i]. Samples area lights (one random
    CDF-uniform pick), distant lights, and infinite lights (all of the
    latter two — few and typically dominant, matching shading.mojo's own
    NEE asymmetry rationale) — no MIS weighting needed for infinite lights
    here, unlike bdpt.mojo's camera-path NEE: a VP is only ever `valid` when
    its ONE traced ray hit a real surface, and _sppm_trace_visible_point's
    own miss-escape env contribution (vp.env) only fires when it didn't —
    mutually exclusive per sample, so there's no competing strategy to
    double-count against."""
    var vp = vps[unsafe_offset=i]
    if vp.valid == Int32(0):
        return
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
    var two_sided_vp = vp.mat_kind == LobeKind.diffuse_transmit
    var shadow_org_back = vp.pos + vp.geo_normal * (-shadow_eps)

    var n_area = Int(sd.areaLightCount)
    if n_area > 0:
        # Pick a random area light + triangle + point on it (same scheme as
        # _sppm_trace_photon's emission sampling).
        var light_sample = sample_area_light_uniform(sd.areaLights, sd.meshes, n_area, pcg, sd.curves)
        var al = light_sample.light
        var lp = light_sample.point
        var ln = light_sample.normal

        var to_light = lp - spos
        var dist2 = dot(to_light, to_light)
        var dist = sqrt(dist2)
        if dist > Float32(0.0):
            var wi = to_light * (Float32(1.0) / dist)
            # No cosine at a volume vertex (it has no normal); the isotropic
            # phase function replaces the BRDF below.
            var cos_surface = Float32(1.0) if is_vol else dot(vn, wi)
            var cos_light = -dot(ln, wi)
            if cos_surface > Float32(0.0) and cos_light > Float32(0.0):
                # Shadow ray, offset from both ends to avoid self-intersection.
                var shadow_ray = Ray(shadow_org, vec3f(wi))
                var t_max = dist * Float32(0.999)
                if not any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, shadow_ray, t_max,
                                      sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                                      sd.spheres, Int(sd.sphereCount),
                                      materials=sd.materials):
                    # pdf_area = 1/(n_area * total_area) — uniform-over-all-lights
                    # assumption, same as the emission-sampling flux scale factor.
                    var inv_pdf_area = Float32(n_area) * al.total_area
                    var geom = cos_surface * cos_light / dist2 * inv_pdf_area
                    # Attenuate across whatever medium the VP sits in. RGB(1)
                    # in vacuum, so this is inert for every media-free scene.
                    var tr_a = _sppm_shadow_transmittance(vp, sd, shadow_org, wi, dist)
                    geom *= tr_a.r
                    # Spectral eval (staged spectral rendering rollout, Stage 4
                    # -- see project_spectral_rendering memory): real
                    # per-wavelength material response x light emission,
                    # multiplied as a SpectralSample product (not each factor
                    # separately, for real product-of-spectra accuracy) then
                    # converted back to RGB, mirroring bdpt.mojo's _connect.
                    # bxdf_eval_any_spectral's conductor branch uses the same
                    # arbitrary-direction (no cosine fused) convention
                    # _sppm_vp_brdf's own bxdf_eval_conductor_ggx call already
                    # relies on, so this is a drop-in spectral replacement for
                    # mat_kind in {diffuse(0), conductor(1)}. Hair (2),
                    # measured (3, always spectral internally regardless --
                    # see _sppm_vp_brdf's own mat_kind=3 branch), and the
                    # no-table-loaded case fall back to the original RGB path.
                    if is_vol:
                        # Isotropic phase (albedo/4pi), matching the photon
                        # gather's own volume-VP convention.
                        vps[unsafe_offset=i].ld += (spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65,
                                                vp.alb.r * INV_FOUR_PI, vp.alb.g * INV_FOUR_PI, vp.alb.b * INV_FOUR_PI, vp.wavelengths)
                                      * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al.emission.r, al.emission.g, al.emission.b, vp.wavelengths)
                                      * geom)
                    elif sd.spectral.res <= 0:
                        # No spectral tables loaded: the one case
                        # lobe_eval's spectral path can't serve, so this
                        # keeps the plain-RGB fallback.
                        var brdf = _sppm_vp_brdf(vp, sd, vn, wi)
                        vps[unsafe_offset=i].ld += (spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, brdf.r, brdf.g, brdf.b, vp.wavelengths)
                                      * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al.emission.r, al.emission.g, al.emission.b, vp.wavelengths)
                                      * geom)
                    else:
                        # THE shared evaluator (bxdf.mojo), the vertex's own
                        # LobeCtx -- covers ggx/hair/measured/layered/
                        # lambertian/diffuse_transmit alike now, where this
                        # used to route hair and measured to a separate RGB
                        # fallback (_sppm_vp_brdf) because lobe_eval's own
                        # dispatch didn't cover them yet. SPPM's NEE uses the
                        # BARE BRDF -- the surface cosine is already inside
                        # `geom`, the convention _sppm_vp_brdf's docstring
                        # states -- so it divides out the lobe's own cosine
                        # rather than assuming which one that is. Connections
                        # want f*cos and merging wants bare f; three consumers,
                        # three conventions, one evaluator that says which it
                        # applied.
                        var le_vp = lobe_eval[want_pdfs=False](
                            LobeCtx(vp.mat_kind, True, False, vn, wo, vp.alb,
                                    vp.mat_idx, vp.alpha, Float32(0), vp.hair_curve_idx,
                                    vp.hair_h, vp.hair_v, True, False),
                            wi, LobeTables(sd.materials, sd.curves, sd.measuredBrdfs),
                            sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x,
                            sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65,
                            vp.wavelengths)
                        var f_spec = (le_vp.f_cos * (Float32(1.0) / le_vp.cos_used)
                                      if le_vp.cos_used > Float32(1e-6)
                                      else SpectralSample(Float32(0.0)))
                        var light_spec = rgb_illuminant_to_spectral_sample(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al.emission.r, al.emission.g, al.emission.b, vp.wavelengths)
                        vps[unsafe_offset=i].ld += f_spec * light_spec * geom

    # Distant/point/sphere/infinite NEE — via the shared Light interface
    # (bvh.mojo's LightSample samplers) + BxDF interface (_sppm_nee_weight
    # above), replacing 4 formerly hand-inlined per-light-type blocks. Area
    # lights are handled separately above (still via _sppm_vp_brdf directly
    # — see that function's docstring).
    # distant/point/sphere via the shared sampler -- pure loop collapse,
    # order was already distant,point,sphere. Sphere-light NEE loops ALL
    # analytic spheres (the sampler itself skips non-emissive ones), since
    # sd.spheres/sphereCount is the raw geometric array, not a pre-filtered
    # lights-only one like every other light type.
    for li in range(_sppm_simple_light_count(sd)):
        var res = _sppm_sample_simple_light(sd, li, spos, pcg)
        var ls = res[0].copy()
        var tmax = res[1]
        var w = _sppm_nee_weight(vp, sd, vn, wo, ls)
        if not w.is_black():
            var s_org = shadow_org_back if (two_sided_vp and dot(vp.geo_normal.to_simd(), ls.wi) < Float32(0)) else shadow_org
            var shadow_ray = Ray(s_org, vec3f(ls.wi))
            if not any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, shadow_ray, tmax,
                                  sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                                  sd.spheres, Int(sd.sphereCount),
                                  materials=sd.materials):
                var tr_s = _sppm_shadow_transmittance(vp, sd, s_org, ls.wi, ls.dist)
                vps[unsafe_offset=i].ld += w * tr_s.r

    for inf_i in range(Int(sd.infiniteLightCount)):
        var ls_e = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_i], Point2f(pcg.next_float(), pcg.next_float()))
        var w_e = _sppm_nee_weight(vp, sd, vn, wo, ls_e)
        if not w_e.is_black():
            var s_org_e = shadow_org_back if (two_sided_vp and dot(vp.geo_normal.to_simd(), ls_e.wi) < Float32(0)) else shadow_org
            var shadow_ray_e = Ray(s_org_e, vec3f(ls_e.wi))
            if not any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, shadow_ray_e, ls_e.dist,
                                  sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                                  sd.spheres, Int(sd.sphereCount),
                                  materials=sd.materials):
                var tr_e = _sppm_shadow_transmittance(vp, sd, s_org_e, ls_e.wi, ls_e.dist)
                vps[unsafe_offset=i].ld += w_e * tr_e.r


def _sppm_nee_update(
    vps:     Pointer[SPPMPixel, MutUntrackedOrigin],
    n_vps:   Int,
    ref sd:      SceneDescriptor2_C,
    seed:    UInt64,
    pass_idx: Int,
):
    var n_lights = Int(sd.areaLightCount) + Int(sd.distantLightCount) + Int(sd.infiniteLightCount) + Int(sd.pointLightCount)
    if n_lights == 0 and not _sppm_has_sphere_lights(sd):
        return

    def nee_one(i: Int) {imm}:
        var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + i), UInt64(11))
        _sppm_nee_one(vps, i, sd, pcg)

    parallelize(nee_one, n_vps)


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
    (unlike bdpt.mojo, which had to add a new return value for this).
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
    max_comp:   Float32,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin],
    spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
) -> RGB:
    """Averages the vp_samples independently-converged samples for pixel i —
    see _sppm_trace_visible_point's docstring for why each sample has its
    own fixed visible point/accumulator rather than sharing one per pixel.
    Shared verbatim between the CPU driver (sppm_render's tail loop) and the
    GPU kernel (sppm_finalize_gpu)."""
    var acc = RGB(Float32(0))
    for vs in range(vp_samples):
        var vp = vps[unsafe_offset=i * vp_samples + vs]
        if vp.valid == Int32(0):
            # No surface hit — either a dead sample, or the traced ray
            # escaped the scene into an infinite (environment) light, whose
            # already beta-weighted radiance _sppm_trace_visible_point
            # stored directly in vp.env (0 if neither happened).
            acc += vp.env
            continue
        # ── Output boundary. The two terms cross it differently, which is
        # forced by SPPM's structure, not a choice:
        #
        #   ld  is gathered at the VP's OWN wavelengths every pass, so the
        #       BSDF x emission product stays spectral across all passes and
        #       converts once, here.
        #   tau sums photon gathers from many passes, each at that pass's
        #       wavelengths, so it is already RGB (see SPPMPixel.tau) and
        #       beta has to meet it in RGB.
        #
        # The per-bounce compounding that RGB transport gets wrong lives
        # ALONG each subpath, and both subpaths are spectral end to end; what
        # remains is one product at the junction.
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
            var denom = PI * vp.r2 * Float32(n_passes)
            if vp.mat_kind == LobeKind.bssrdf:
                # BSSRDF: R_d is already a normalised kernel, so there is NO
                # kernel area or volume to divide by -- only the pass average.
                # Dividing by pi*r^2 here would scale the result by an
                # arbitrary radius that carries no meaning on this path.
                denom = Float32(n_passes)
            elif vp.is_volume == PhotonKind.volume:
                denom = (Float32(4.0) / Float32(3.0)) * PI * vp.r2 * sqrt(vp.r2) * Float32(n_passes)
            acc += vp.beta * (vp.tau / denom)
        # Direct (NEE) term — pbrt's "pixel.Ld", resampled once per
        # pass, averaged over n_passes. Applies to volume VPs too: this
        # was gated on `is_volume == 0` until 2026-09-09, so a volume
        # scatter point's direct lighting was computed and then thrown
        # away -- the third of three independent gates that each had to
        # be opened for fog to receive any direct light at all.
        var (dr, dg, db) = spectral_sample_to_rgb(
            spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
            spectral_cie_z, spectral_d65, vp.ld / Float32(n_passes), vp.wavelengths)
        acc += vp.beta * RGB(dr, dg, db)
    acc = acc / Float32(vp_samples)

    # ISO exposure compensation (matches normalize_film)
    acc *= iso_scale

    # NaN guard and optional max-component clamp
    if acc.r != acc.r or acc.r < Float32(0): acc.r = Float32(0)
    if acc.g != acc.g or acc.g < Float32(0): acc.g = Float32(0)
    if acc.b != acc.b or acc.b < Float32(0): acc.b = Float32(0)
    if max_comp > Float32(0):
        var mx = max(acc.r, max(acc.g, acc.b))
        if mx > max_comp:
            acc *= max_comp / mx
    return acc


# ── Public entry point ────────────────────────────────────────────────────────

def _sppm_render_core(
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneDescriptor2_C,
    n_passes: Int,
    n_photons_per_pass: Int,
    initial_radius: Float32,
    verbose:  Bool,
) -> Tuple[Bool, Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]]:
    """Stochastic Progressive Photon Mapping main loop, factored out of
    `sppm_render` so the CPU driver's aux-buffer/denoise/write tail is a
    separate, reusable step. Returns (ok, pixels, albedo_pixels) --
    `ok=False` (both pointers null/unusable) when the scene has no lights,
    mirroring the old function's `Int32(-1)` early return. Same buffer
    contract as `_bdpt_render_core` (bdpt.mojo): caller-owned `n_pix*3`
    Float32 arrays, iso-scaled/max_comp-clamped, NOT yet denoised."""
    var fw = Int(psc[unsafe_offset=0].film_w)
    var fh = Int(psc[unsafe_offset=0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[unsafe_offset=0].film_iso / Float32(100)
    var max_comp = psc[unsafe_offset=0].film_max_comp

    if Int(sd.areaLightCount) + Int(sd.distantLightCount) + Int(sd.infiniteLightCount) + Int(sd.pointLightCount) == 0 and not _sppm_has_sphere_lights(sd):
        print("SPPM: no lights in scene, cannot emit photons")
        return Tuple[Bool, Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]](
            False, Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Pointer[Float32, MutUntrackedOrigin].unsafe_dangling())

    print("SPPM: " + String(fw) + "x" + String(fh)
          + " " + String(n_passes) + " passes x "
          + String(n_photons_per_pass) + " photons  r=" + String(initial_radius))

    # Allocate visible points (n_pix * _VP_SAMPLES independent samples) and photon buffer.
    #
    # max_photons is sized for the WORST CASE, not n_photons_per_pass: one
    # emitted photon path can store up to min(maxdepth, _MAX_B) - 1 deposits
    # (one per RR diffuse-diffuse bounce past the first), not just one. A
    # buffer sized at n_photons_per_pass (the older code here) meant the
    # shared atomic `counter` in _sppm_store_photon started silently
    # dropping deposits the moment total stored events -- not paths --
    # exceeded n_photons_per_pass, which for any scene with real multi-
    # bounce diffuse interreflection is most of them. Root-caused via a
    # closed-cavity radiative-equilibrium test (reflectance 1.0 everywhere,
    # one wall emitting L=1: every surface must read exactly 1.0, Russian
    # roulette never fires since albedo=1, so this scene hits the worst
    # case on every path) -- SPPM read 0.39 of that 1.0 with the old sizing
    # and 0.87 with this fix, path tracer (unaffected by this buffer) reads
    # 1.00 throughout. See project_photon_estimator_energy_gap memory.
    #
    # An earlier version of this comment claimed storing every bounce
    # event "measurably over-brightened the render" and kept the drop
    # deliberately -- that test predates several other estimator fixes
    # landed this session and was never re-verified against a ground
    # truth; the closed-cavity test above is the ground truth, and it
    # says the drop was strictly wrong, not a deliberate bias/cost
    # tradeoff. If a future regression looks like "over-brightening",
    # re-derive against an equilibrium or analytic scene before
    # reintroducing a cap -- don't just trust the old claim.
    var n_vps    = n_pix * _VP_SAMPLES
    var vps     = unsafe_alloc[SPPMPixel](n_vps)
    var max_bounces_per_photon = min(Int(psc[unsafe_offset=0].max_depth), _MAX_B)
    # A subsurface interior blows this budget wide open: its random-walk steps
    # are deliberately NOT charged to maxdepth (see _sppm_trace_photon's loop
    # header), so one photon entering skin deposits at every scatter for as
    # long as the walk survives -- hundreds of events, not `maxdepth` of them.
    # Sized for maxdepth alone, the shared atomic counter in
    # _sppm_store_photon starts dropping deposits almost immediately, and the
    # few paths that DID fit leave a wildly inflated local photon density that
    # the estimator still divides by the full emitted count: head.pbrt read
    # ~33x the pbrt reference, uniformly, at every pass count. Exactly the
    # failure the comment above describes, with a different cause.
    var has_sss_medium = False
    for mi in range(Int(sd.mediumCount)):
        if sd.mediums[unsafe_offset=mi].is_sss != Int32(0):
            has_sss_medium = True
            break
    if has_sss_medium:
        max_bounces_per_photon += SSS_WALK_ROUNDS
    var max_photons = n_photons_per_pass * max(max_bounces_per_photon, 1)
    var photons = unsafe_alloc[SPPMPhoton](max_photons)
    var heads   = unsafe_alloc[Int32](_HSIZE)
    # A BSSRDF visible point gathers out to the material's DIFFUSION reach,
    # which for skin is several times the SPPM radius this scene would
    # otherwise pick. The photon grid's cells are initial_radius-sized and the
    # gather looks at 3x3x3 of them, so the radius has to cover that reach or
    # the neighbour search silently misses the photons carrying the subsurface
    # signal. Widening it costs nothing on this path: R_d is normalised, so
    # unlike a density estimate the radius does not scale the answer.
    var eff_radius = initial_radius
    for _mi in range(Int(sd.mediumCount)):
        if sd.mediums[unsafe_offset=_mi].is_sss != Int32(0):
            var _rq = dipole_max_radius(sd.mediums[unsafe_offset=_mi].sigma_s, sd.mediums[unsafe_offset=_mi].sigma_a, sd.mediums[unsafe_offset=_mi].g)
            if _rq > eff_radius:
                eff_radius = _rq
    # init_r2 stays on the SCENE's radius -- widening it would blur every
    # ordinary surface visible point in the scene. Only the grid CELLS grow,
    # so the 3x3x3 neighbour search can reach the diffusion distance; a larger
    # cell never changes a surface gather, it only searches more candidates.
    var init_r2 = initial_radius * initial_radius
    var inv_cell = Float32(1.0) / eff_radius  # cell size == effective radius

    # Trace the camera/visible-point samples ONCE for the whole render — see
    # _sppm_camera_pass's docstring for why a per-pass re-trace (the old
    # design) breaks SPPM's convergence guarantee.
    var cam_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(0x9E3779B97F4A7C15 + 7)
    _sppm_camera_pass(
        vps, n_pix, _VP_SAMPLES, psc[unsafe_offset=0].film_w,
        psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world,
        sd, init_r2, cam_seed, Int(psc[unsafe_offset=0].max_depth),
        film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                       psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y),
    )
    if verbose:
        var n_valid = 0
        for i in range(n_vps):
            if vps[unsafe_offset=i].valid != Int32(0): n_valid += 1
        print("SPPM: " + String(n_valid) + "/" + String(n_vps) + " visible points found")

    # Photon passes
    for pass_idx in range(n_passes):
        var pass_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(pass_idx * 2654435761 + 1)
        var n_stored = _sppm_photon_pass(photons, n_photons_per_pass, max_photons, sd, pass_seed, pass_idx, Int(psc[unsafe_offset=0].max_depth),
                                         _sppm_cam_pos(psc[unsafe_offset=0].camera_to_world),
                                         _sppm_photon_px_scale(
                                             psc[unsafe_offset=0].raster_to_camera,
                                             psc[unsafe_offset=0].camera_to_world, fw, fh))
        if n_stored > 0:
            _build_grid(photons, n_stored, heads, inv_cell)
            _gather_update(vps, n_vps, photons, heads, inv_cell, sd, pass_wavelengths(pass_idx))
        var nee_seed = psc[unsafe_offset=0].rng_seed ^ UInt64(pass_idx * 0xBF58476D1CE4E5B9 + 3)
        _sppm_nee_update(vps, n_vps, sd, nee_seed, pass_idx)
        if verbose or (pass_idx + 1) % 10 == 0:
            print("SPPM: pass " + String(pass_idx + 1) + "/" + String(n_passes)
                  + " stored=" + String(n_stored), end="\r")

    print("")  # newline after progress

    # Assemble output image: average the _VP_SAMPLES independently-converged
    # samples per pixel (each sample's own r2/tau/N_acc converges correctly
    # since its position/surface is fixed for the whole render — see
    # _sppm_camera_pass's docstring). A sample that never found a diffuse/
    # volume hit (N_acc == 0, e.g. it reflected off the water into the void)
    # contributes 0 for that sample, same as a path tracer sample that misses
    # everything — the average over all samples is what correctly reproduces
    # the fresnel-weighted reflect/refract blend a real specular interface
    # would show.
    var out_pixels = unsafe_alloc[Float32](n_pix * 3)
    var albedo_pixels = unsafe_alloc[Float32](n_pix * 3)

    def finalize_one(i: Int) {imm}:
        var acc = _sppm_finalize_one_pixel(vps, i, _VP_SAMPLES, Int32(n_passes), iso_scale, max_comp,
                                         sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x,
                                         sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
        out_pixels[unsafe_offset=i * 3 + 0] = acc.r
        out_pixels[unsafe_offset=i * 3 + 1] = acc.g
        out_pixels[unsafe_offset=i * 3 + 2] = acc.b
        var alb = _sppm_finalize_albedo_one_pixel(vps, i, _VP_SAMPLES)
        albedo_pixels[unsafe_offset=i * 3 + 0] = alb.r
        albedo_pixels[unsafe_offset=i * 3 + 1] = alb.g
        albedo_pixels[unsafe_offset=i * 3 + 2] = alb.b

    parallelize(finalize_one, n_pix)

    heads.unsafe_free()
    photons.unsafe_free()
    vps.unsafe_free()
    return Tuple[Bool, Pointer[Float32, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]](
        True, out_pixels, albedo_pixels)

def sppm_render(
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneDescriptor2_C,
    n_passes: Int,
    n_photons_per_pass: Int,
    initial_radius: Float32,
    no_denoise: Bool,
    verbose:  Bool,
) -> Int32:
    """CLI-facing SPPM entry point: run the progressive photon-mapping
    estimator (`_sppm_render_core`), then denoise (SPPMPixel.alb as the
    albedo AOV, normals/depth from the same integrator-agnostic
    render_aux_buffers the plain path tracer and VCM use) and write."""
    var (ok, out_pixels, albedo_pixels) = _sppm_render_core(
        psc, sd, n_passes, n_photons_per_pass, initial_radius, verbose)
    if not ok:
        return Int32(-1)

    var n_pix = Int(psc[unsafe_offset=0].film_w) * Int(psc[unsafe_offset=0].film_h)
    var normals = unsafe_alloc[Float32](n_pix * 3)
    var depth = unsafe_alloc[Float32](n_pix)
    var sd_local = sd
    render_aux_buffers(psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world, Int32(0), Int32(0),
                        psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h, Pointer(to=sd_local), normals, depth)

    var denoised = unsafe_alloc[Float32](n_pix * 3)
    if no_denoise:
        for i in range(n_pix * 3): denoised[unsafe_offset=i] = out_pixels[unsafe_offset=i]
    else:
        denoise(out_pixels, albedo_pixels, normals, depth, psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h,
                denoised, Int32(5), Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05))

    _ = write_image_cropwindow(denoised, psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h,
        psc[unsafe_offset=0].crop_x0, psc[unsafe_offset=0].crop_y0, psc[unsafe_offset=0].crop_x1, psc[unsafe_offset=0].crop_y1,
        psc[unsafe_offset=0].film_filename, Int32(32), Int32(32))

    out_pixels.unsafe_free(); albedo_pixels.unsafe_free(); normals.unsafe_free(); depth.unsafe_free(); denoised.unsafe_free()
    return Int32(0)

