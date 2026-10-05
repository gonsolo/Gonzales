# BDPT/VCM light subpath: state, init and the per-bounce step.
# Part of the BDPT/VCM machinery that used to be one file (bdpt.mojo).

from std.math import sqrt, cos, sin, log, exp, max, abs
from .geometry import face_toward, RGB, Point3f, Point2f, Vec3f, vec3f, point3f, dot, PI, INV_FOUR_PI, INV_PI
from .materials import MatKind, LobeKind, dielectric_is_rough
from .primitives import Ray, Intersection
from .media import sample_free_flight, spectral_free_flight_weight
from .bssrdf import bssrdf_exit_ft
from .vcm_mis import vcm_arrival_carries, vcm_scatter_carries, bssrdf_hop_carries, bssrdf_exit_scatter_carries
from .vcm_camis import (
    CamisLightRecord, CamisLightCarry, camis_light_carry_off, camis_light_origin, camis_light_arrive,
    camis_light_scatter,
)
from .bvh import (
    SceneView, traverse_bvh2_core, test_spheres, _scene_bounding_sphere, _sample_disk_perpendicular,
    _sample_infinite_light_dir, _hair_precompute, curve_offset_eps, sphere_light_cone_pdf, light_path_pick,
)
from .rng import PCG32
from .sppm import (
    _geom_normal, _shading_normal_at, _dielectric_bounce, medium_after_crossing, _cosine_hemisphere_sample,
    sample_area_light_point, sample_sphere_light_emission,
)
from .shading import uv_footprint_at_hit, _tex_lookup, _get_tri_verts, apply_surface_maps_at_hit
from .bxdf import (
    lobe_kind_of, lobe_param_of, lobe_is_delta_of, lobe_is_available_of, bxdf_is_delta,
    bxdf_pdf_conductor_ggx,
)
from .spectrum import SampledWavelengths, SpectralSample, spec_refl, spec_refl_unbounded, spec_illum
from .bdpt_vertex import _BDPT_MAX_DEPTH, _BDPT_MAX_VERTS, BDPTVertex, _null_vertex
from .bdpt_eval import _bdpt_vertex_mis_scoped
from .vcm_grid import _vcm_depth, _vcm_keep, _VCM_CAMIS, _vcm_eta_scale, _vcm_eta_at
from .bdpt_connect import _bdpt_store_lvc_vertex
from .bdpt_bssrdf import _bdpt_sample_bssrdf_exit
from .bdpt_camera import _vcm_scatter, _resolve_mix, _coated_conductor_scatter

@fieldwise_init
struct VCMLightPathState(TrivialRegisterPassable):
    """Task #163 stage 4: persistent per-light-path state carried across
    separate wavefront-staged GPU kernel launches (`_bdpt_light_path_init_gpu`
    then one `_bdpt_light_path_bounce_gpu` call per bounce), the light-path
    counterpart to gpu.mojo's `PathState` for the plain wavefront path
    tracer. `active=0` means the path is done (produced by
    `_null_light_path_state()` or by `_bdpt_light_path_bounce` returning
    False) -- the host loop stops calling the bounce kernel for a lane once
    its `active` flag drops to 0, matching PathState's own convention."""
    var ro: Point3f
    var rd: Vec3f
    var flux: SpectralSample
    var dvcm: Float32
    var dvc: Float32
    var dvm: Float32
    var is_finite_origin: Int8
    # Index into sd.spheres of the sphere light this path left, until its
    # first hit applies the origin's 1/Omega (see the init's sphere branch);
    # -2 once applied; -1 for every other light.
    var origin_sphere: Int32
    var cur_med_idx: Int32
    var n_lbounces: Int32
    var n_verts: Int32
    var active: Int8
    var pcg_state: UInt64
    var pcg_inc: UInt64
    var wl0: Float32
    var wl1: Float32
    var wl2: Float32
    var wl3: Float32
    # Touching-dielectric IOR depth-2 stack for _dielectric_bounce (see that
    # function's docstring, sppm.mojo) -- same role and convention as
    # VCMCameraPathState's matching fields. Both start at vacuum (1.0).
    var current_dielectric_ior: Float32
    var previous_dielectric_ior: Float32
    var n_delta: Int32

def _null_light_path_state() -> VCMLightPathState:
    return VCMLightPathState(
        Point3f(Float32(0), Float32(0), Float32(0)),
        Vec3f(Float32(0), Float32(0), Float32(0)),
        SpectralSample(Float32(0)),
        Float32(0), Float32(0), Float32(0), Int8(0), Int32(-1),
        Int32(-1), Int32(0), Int32(0), Int8(0),
        UInt64(0), UInt64(0),
        Float32(0), Float32(0), Float32(0), Float32(0),
        Float32(1.0), Float32(1.0),   # current_dielectric_ior, previous_dielectric_ior (vacuum)
        Int32(0),                     # n_delta
    )

def _bdpt_light_path_init[use_gpu: Bool](
    ref sd: SceneView,
    mut pcg: PCG32,
    default_emit_med: Int32,
    lp_idx: Int,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    mis_vc_weight_factor: Float32,
    pass_wl: SampledWavelengths,
) -> VCMLightPathState:
    """Task #163 stage 4: light-emission setup half of
    `_bdpt_trace_light_path` (bdpt.mojo:1734-1880), split out to seed a
    `VCMLightPathState` for the wavefront-staged bounce loop instead of
    falling straight into an inline `for` loop. Byte-for-byte copy of that
    function's pre-loop body -- see its own docstring/VCM Stage 2b comments
    for the MIS derivation, not repeated here. The only changes are the two
    bare `return` sites (no lights in the scene; zero-pdf infinite-light dir
    sample) now returning `_null_light_path_state()` (active=0) instead, and
    the end of the function packaging the locals into a returned state
    instead of continuing into a bounce loop."""
    var n_area = Int(sd.areaLightCount)
    var n_sphere = Int(sd.sphereLightCount)
    var n_distant = Int(sd.distantLightCount)
    var n_infinite = Int(sd.infiniteLightCount)
    var n_point = Int(sd.pointLightCount)
    var n_lights = n_area + n_sphere + n_distant + n_infinite + n_point   # == _bdpt_n_lights
    lvc_path_len[unsafe_offset=lp_idx] = Int32(0)
    if n_lights == 0:
        return _null_light_path_state()   # no lights in the scene

    var ro: Point3f
    var rd: Vec3f
    # A light subpath's flux is EMISSION, so it crosses into the spectral
    # domain through the illuminant curve -- the same boundary the camera
    # subpath crosses at an emitter hit.
    var flux: SpectralSample
    var n_verts: Int
    # VCM Stage 2b: real per-vertex MIS state, recursively carried along
    # this light subpath (see project_vcm_stage2_mis_derivation memory for
    # the verified formulas). Only AREA lights get a real, non-zero origin
    # this pass — distant/infinite/point-seeded paths start at 0, a
    # deliberately scoped simplification (their own MIS treatment needs
    # the delta-light formulas Georgiev's tech report's (48)-(50) define,
    # not yet ported). Only diffuse/measured vertices update these
    # meaningfully; every other material resets them as if specular
    # (dVCM=0, dVC/dVM *= cosThetaOut) since they don't have a real
    # separate forward/reverse pdf today — see the memory file.
    var dvcm_carry: Float32
    var dvc_carry: Float32
    var dvm_carry: Float32
    var is_finite_origin = False
    var origin_sphere = Int32(-1)
    # This pass's shared hero wavelengths (see _bdpt_camera_path_init's
    # comment for why every subpath in a pass must agree on them).
    var wavelengths = pass_wl

    # Power-proportional, not uniform (light_path_pick): every factor below
    # that used to read n_lights is 1 / this light's pick probability, and
    # the camera side reads the same probability back (light_path_pick_pdf).
    var (light_pick, p_pick) = light_path_pick(sd, pcg.next_float())
    if p_pick <= Float32(0):
        return _null_light_path_state()   # a light that emits nothing
    var inv_pick = Float32(1) / p_pick
    if light_pick < n_area:
        # Pick a light uniformly + a random triangle + barycentric point on it.
        # The light the pick CHOSE -- not sample_area_light_uniform's own
        # uniform re-pick, which ignored it: the emission then followed 1/2
        # per light while every density assumed the power pick's 0.99/0.01,
        # and a dim panel's paths came out 100x overweighted half the time.
        var light_sample = sample_area_light_point(sd.areaLights[unsafe_offset=light_pick], sd.meshes, pcg, sd.curves)
        var al = light_sample.light
        var lp = light_sample.point
        var ln = light_sample.normal

        # Cosine-weighted emission direction
        var du1 = pcg.next_float(); var du2 = pcg.next_float()
        var pdir = _cosine_hemisphere_sample(ln, du1, du2)

        # Light vertex (the emission point, s=1 strategy connects here directly).
        # beta = 1/p_A = area × n_lights (area sampling PDF correction only).
        # alb  = Le (emission) — used as f_lgt in _connect for the light vertex.
        # G already handles the cos_l factor, so f_lgt = Le (no extra cos multiply).
        var area_weight = al.total_area * inv_pick
        var lv0_vert = _null_vertex()
        lv0_vert.pos = point3f(lp)
        lv0_vert.normal = vec3f(ln)
        lv0_vert.shading_normal = vec3f(ln)
        lv0_vert.beta = SpectralSample(area_weight)
        lv0_vert.alb = al.emission
        lv0_vert.is_surface = Int32(1); lv0_vert.is_light = Int32(1)
        lv0_vert.pdf_fwd = Float32(1) / area_weight
        lv0_vert.med_idx = default_emit_med
        lv0_vert.wavelengths = wavelengths

        # VCM Stage 2b: real MIS origin state for this (finite, area) light
        # -- see project_vcm_stage2_mis_derivation memory for the verified
        # derivation. cos_theta_emit cancels out of the flux formula above
        # (Malley's method) but is needed again here, unrelated to flux.
        is_finite_origin = True
        var cos_theta_emit = max(dot(pdir, ln), Float32(0.0001))
        var direct_pdf_a = Float32(1) / area_weight
        var emission_pdf_w = direct_pdf_a * cos_theta_emit / PI
        dvcm_carry = direct_pdf_a / emission_pdf_w
        dvc_carry = cos_theta_emit / emission_pdf_w
        # SmallVCM vertexcm.hxx:856 -- light-origin dVM uses mMisVcWeightFactor,
        # NOT mMisVmWeightFactor (verified against the reference source; a
        # variable-name mix-up here was silently inert while merge stayed
        # disabled -- lv.dVM was never read by the connect-only weight, only
        # by RangeQuery::Process's real merge weight, added below).
        comptime if _VCM_CAMIS:
            # The slot holds dVC0 instead (vcm_scatter_carries[dvc0]): there
            # is no eta at the emitter, so dVC0 == dVC here (S0's
            # trace_light_records).
            dvm_carry = dvc_carry
        else:
            dvm_carry = dvc_carry * mis_vc_weight_factor
        lv0_vert.dVCM = dvcm_carry
        lv0_vert.dVC = dvc_carry
        lv0_vert.dVM = dvm_carry
        _bdpt_store_lvc_vertex(lv0_vert, lvc, lp_idx, 0)

        # For traced vertices: beta = Le × cos_θ / (p_A × p_ω) where p_ω = cos_θ/π
        # for cosine-weighted emission -- the cos_θ_emitted terms CANCEL exactly
        # (Malley's method: this is the whole point of cosine-weighted emission
        # sampling, see sppm.mojo's photon-emission flux for the same, correctly
        # cos_θ-free formula). β = Le × area × n_lights × π, no cos_θ term.
        # BUG FIX (2026-07-10): this previously divided by cos_θ_emitted instead
        # of letting it cancel, inflating flux by up to 100x on near-grazing
        # emission directions (clamped at cos_θ=0.01) -- root cause of BDPT's
        # pre-existing indirect-bounce fireflies, see project_bdpt_todo memory.
        flux = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al.emission.r, al.emission.g, al.emission.b, wavelengths) * (area_weight * PI)
        ro = point3f(lp) + vec3f(ln)*Float32(0.0001)
        rd = vec3f(pdir)
        n_verts = 1  # vertex 0 is the light point itself
    elif light_pick < n_area + n_sphere:
        # Sphere light: the shared emission sampler (a uniform point on the
        # emitter, a cosine direction -- the mesh-light model with area
        # 4 pi r^2), NO origin vertex stored. Its s=1 strategy is the
        # camera's own NEE, which samples the cone the sphere subtends
        # (_sample_sphere_light_nee) -- better than connecting to a uniform
        # surface point, and the reason these lights never started light
        # paths before: until the densities below existed, a light path
        # could not be weighed against that NEE.
        #
        # SmallVCM's origin carries, with the pick 1/n_lights in both
        # densities (the camera's NEE enumerates lights, no pick):
        #     emissionPdfW = cos0 / (pi * area_weight)
        #     dVC  = cos0 / emissionPdfW          = pi * area_weight
        #     dVCM = directPdfA / emissionPdfW
        # where directPdfA is the NEE's density for THIS point seen from the
        # first vertex x1 -- the cone's 1/Omega(x1) in area measure,
        # cos0 / (Omega(x1) d^2). x1 is not known yet, so dVCM starts at
        # pi * area_weight and the first hit multiplies by 1/Omega(x1) in
        # place of the usual d^2, which that area conversion cancels
        # (_bdpt_light_path_bounce, `origin_sphere`).
        var (si_l, sp_l, sn_l, sdir_l) = sample_sphere_light_emission(sd, light_pick - n_area, pcg)
        var sph_l = sd.spheres[unsafe_offset=si_l]
        var area_weight_s = Float32(4) * PI * sph_l.radius * sph_l.radius * inv_pick
        flux = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, sph_l.emission.r, sph_l.emission.g, sph_l.emission.b, wavelengths) * (area_weight_s * PI)
        ro = point3f(sp_l + sn_l * (sph_l.radius * Float32(0.0001)))
        rd = vec3f(sdir_l)
        n_verts = 0
        is_finite_origin = True
        origin_sphere = Int32(si_l)
        dvcm_carry = PI * area_weight_s
        dvc_carry = PI * area_weight_s
        comptime if _VCM_CAMIS:
            dvm_carry = dvc_carry   # dVC0 == dVC at an origin, see the area branch
        else:
            dvm_carry = dvc_carry * mis_vc_weight_factor
    elif light_pick < n_area + n_sphere + n_distant:
        var dl = sd.distantLights[unsafe_offset=light_pick - n_area - n_sphere]
        var (center, radius) = _scene_bounding_sphere(sd)
        var dir = Vec3f(dl.direction.x, dl.direction.y, dl.direction.z)
        var disk_pt = _sample_disk_perpendicular(dir, center, radius, Point2f(pcg.next_float(), pcg.next_float()))
        # Phi_light = emission(irradiance) × disk_area; p_i = 1/n_lights;
        # pdf_pos = 1/disk_area; pdf_dir = 1 (delta) → flux = emission × disk_area × n_lights.
        flux = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, dl.emission.r, dl.emission.g, dl.emission.b, wavelengths) * (inv_pick * PI * radius * radius)
        ro = disk_pt
        rd = dir
        n_verts = 0  # no finite light point to store as a cache vertex
        # Real MIS origin for a DISTANT light -- the third instance of the
        # zero-carries bug: c1f2e10f fixed the infinite branch and left this
        # one at 0 ("scoped simplification"). Zero carries collapse every
        # merge-with and t=1 weight for a sun photon to near-full credit while
        # the camera's delta-light NEE ALSO takes full weight: classroom (sun +
        # env) read 2x pbrt with merging on, 0.985x with it off, and none of
        # the exact scenes could see it because none has a distant light.
        # SmallVCM's directional case: directPdfW = 1 (delta), emissionPdfW =
        # the disk's area pdf, so dVCM = diskArea and dVC = dVM = 0 (a delta
        # light has no cosine and no BSDF-sampleable direction). n_lights
        # multiplies dVCM because the CAMERA's NEE loops every light with no
        # pick while this path picked one with probability 1/n_lights.
        dvcm_carry = PI * radius * radius * inv_pick
        dvc_carry = Float32(0)
        dvm_carry = Float32(0)
    elif light_pick < n_area + n_sphere + n_distant + n_infinite:
        var il = sd.infiniteLights[unsafe_offset=light_pick - n_area - n_sphere - n_distant]
        var (center, radius) = _scene_bounding_sphere(sd)
        # _sample_infinite_light_dir returns env_dir in the NEE convention
        # ("direction FROM a shading point TOWARD the light" — same as
        # shading.mojo's _nee_infinite_light usage). A photon leaving the
        # light travels the opposite way, arriving FROM that direction
        # INTO the scene — negate it for the emitted ray/disk placement.
        var (env_dir, env_rgb, pdf_dir) = _sample_infinite_light_dir(il, Point2f(pcg.next_float(), pcg.next_float()))
        if pdf_dir <= Float32(0):
            return _null_light_path_state()   # infinite-light dir sample had zero pdf
        var emit_dir = -env_dir
        var disk_pt = _sample_disk_perpendicular(emit_dir, center, radius, Point2f(pcg.next_float(), pcg.next_float()))
        # Same derivation as distant, but pdf_dir is the CDF's solid-angle pdf
        # instead of an implicit delta (=1): flux = radiance × disk_area × n_lights / pdf_dir.
        flux = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, env_rgb.r, env_rgb.g, env_rgb.b, wavelengths) * (inv_pick * PI * radius * radius / pdf_dir)
        ro = disk_pt
        rd = emit_dir
        n_verts = 0
        # VCM Stage 2b MIS origin for a NON-FINITE (environment) light.
        # SmallVCM vertexcm.hxx GenerateLightSample, background branch:
        #     directPdfW   = pdf_dir * lightPickProb
        #     emissionPdfW = directPdfW / diskArea
        #     dVCM = directPdfW / emissionPdfW = diskArea
        #     dVC  = usedCosLight / emissionPdfW, usedCosLight = 1 (not finite)
        #     dVM  = dVC * mis_vc_weight_factor
        # lightPickProb = 1/n_lights cancels out of dVCM and appears in dVC
        # as the n_lights factor, exactly as in `flux` just above.
        #
        # These used to be left at 0 -- the scoped simplification the block
        # comment at the top of this function describes. Zero carries collapse
        # every MIS denominator to 1, so both light-side techniques took FULL
        # credit for illumination the camera side had already reported through
        # its env NEE. Measured on env-furnace (analytic answer 0.5): correct
        # at 0.5037 with both disabled, 0.5676 with t=1 splatting re-enabled
        # (+12.8%), 0.6701 with merging too (+20.5%) -- the whole of VCM's
        # +34% on every env-lit scene. Area lights were never affected (they
        # get real carries above), which is why only env-lit cells showed it.
        #
        # `is_finite_origin` deliberately stays False: that is what suppresses
        # the first-segment dist^2 factor below, correct here because the disk
        # is a sampling device, not a real emitter position.
        var disk_area = PI * radius * radius
        dvcm_carry = disk_area * inv_pick
        dvc_carry = disk_area * inv_pick / pdf_dir
        comptime if _VCM_CAMIS:
            dvm_carry = dvc_carry   # dVC0 == dVC at an origin, see the area branch
        else:
            dvm_carry = dvc_carry * mis_vc_weight_factor
    else:
        # Point light: a real finite position (unlike distant/infinite), but
        # still no NEE-equivalent cache vertex — see this function's own
        # docstring for why (same reasoning as distant/infinite: direct
        # illumination comes from _bdpt_trace_camera_and_connect's own
        # per-vertex point-light NEE instead). Emits uniformly over the
        # sphere (isotropic point light); pdf_dir = 1/(4π), so
        # flux = intensity × 4π × n_lights (pdf_dir cancels).
        var pll = sd.pointLights[unsafe_offset=light_pick - n_area - n_sphere - n_distant - n_infinite]
        var u1p = pcg.next_float(); var u2p = pcg.next_float()
        var cos_p = Float32(1) - Float32(2) * u1p
        var sin_p = sqrt(max(Float32(0), Float32(1) - cos_p * cos_p))
        var phi_p = Float32(2) * PI * u2p
        var pdir_p = Vec3f(sin_p * cos(phi_p), sin_p * sin(phi_p), cos_p)
        flux = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, pll.intensity.r, pll.intensity.g, pll.intensity.b, wavelengths) * (Float32(4) * PI * inv_pick)
        ro = pll.position
        rd = pdir_p
        n_verts = 0
        # Real MIS origin for a POINT light -- the fourth instance of the
        # zero-carries bug (infinite c1f2e10f, distant after it): with zero
        # carries every t=1/merge weight on a point-light path collapsed to
        # full credit while the camera's delta NEE took full weight too.
        # SmallVCM's point light: directPdfA = 1 (delta position),
        # emissionPdfW = 1/(4 pi n_lights), no cosine, so dVCM = 4 pi
        # n_lights and dVC = dVM = 0 (a delta light cannot be hit). A real
        # finite position, so the first segment's d^2 applies.
        is_finite_origin = True
        dvcm_carry = Float32(4) * PI * inv_pick
        dvc_carry = Float32(0)
        dvm_carry = Float32(0)

    var cur_med_idx = default_emit_med
    var n_lbounces = 1  # counts all surface hits (1 = not-the-primary-ray, matches area-light convention — see _dielectric_bounce's bounce==0 special case)

    return VCMLightPathState(
        ro, rd, flux, dvcm_carry, dvc_carry, dvm_carry,
        Int8(1) if is_finite_origin else Int8(0), origin_sphere,
        cur_med_idx, Int32(n_lbounces), Int32(n_verts), Int8(1),
        pcg.state, pcg.inc,
        wavelengths.lambda0, wavelengths.lambda1, wavelengths.lambda2, wavelengths.lambda3,
        Float32(1.0), Float32(1.0),   # current_dielectric_ior, previous_dielectric_ior (vacuum)
        Int32(0),                     # n_delta
    )

def _bdpt_light_path_bounce[use_gpu: Bool](
    ref sd:      SceneView,
    mut pcg: PCG32,
    has_med: Bool,
    inter: Intersection,
    lvc:      Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx:   Int,
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    mut ro: Point3f,
    mut rd: Vec3f,
    mut flux: SpectralSample,
    mut n_verts: Int,
    mut n_delta: Int,   # delta bounces so far; they count toward maxdepth (see _vcm_depth)
    mut dvcm_carry: Float32,
    mut dvc_carry: Float32,
    mut dvm_carry: Float32,
    # Kind of the PREVIOUS vertex on this subpath (True = volume), needed at
    # a medium arrival to pick the right dVC free-flight ratio
    # (Scenes/vcm_volume_mis_derivation.py: sigma_t^(kind_a - kind_b), 1 for
    # a same-kind edge, 1/sigma_t or sigma_t for a kind-changing one).
    mut prev_was_volume: Bool,
    is_finite_origin: Bool,
    mut origin_sphere: Int32,
    mut cur_med_idx: Int32,
    mut n_lbounces: Int,
    mut current_dielectric_ior: Float32,
    mut previous_dielectric_ior: Float32,
    wavelengths: SampledWavelengths,
    # _VCM_CAMIS light state (vcm_camis.CamisLightCarry) and THIS path's
    # slice of the record buffer (lvc_camis + lp_idx * _BDPT_MAX_VERTS, the
    # same slots as `lvc`). Untouched when the hybrid is compiled out.
    mut camis_l: CamisLightCarry,
    camis_recs: Pointer[CamisLightRecord, MutUntrackedOrigin],
) -> Bool:
    """Task #163 stage 4: wavefront-staged variant of ONE bounce iteration of
    `_bdpt_trace_light_path`'s main loop (bdpt.mojo:1882-2383), split out so a
    GPU host loop can interleave a separate batched intersect dispatch
    (traverse_paths_gpu today, Vulkan RT via vulkaninterop later) between
    calls instead of tracing the whole subpath inside one kernel invocation.

    The material-dispatch BODY below (lines mirroring 1892-2380 of
    `_bdpt_trace_light_path`) is a byte-for-byte copy of that function's loop
    body -- see this module's VCM Stage 2b/2c comments there for the MIS
    derivation, not repeated here. The only changes from the original are
    mechanical, control-flow-shape ones:
      - the per-bounce intersect (`traverse_bvh2_core`/`test_spheres`) is
        REMOVED -- the caller supplies `inter` already computed via a
        separate batched dispatch.
      - every `break` that terminated the ORIGINAL function's outer bounce
        loop (7 sites: coat total-internal-reflection, coat walk exit
        failure, conductor invalid sample, 3x measured-BxDF failure, the
        unhandled-material-type catch-all) becomes `return False`.
      - the ORIGINAL function's 2 outer-loop `continue` sites (volume
        free-flight scatter, rough-coat reflect) become `return True`.
      - the coat recycling walk's OWN inner `for depth in
        range(MAX_COAT_DEPTH):` loop keeps its own real `break` statements
        (early-exit on RR kill, on finding a valid exit direction) --
        untouched, since those exit the INNER walk loop, not this function.
      - `n_verts`/loop-carried locals are `mut` PARAMETERS instead of
        function-local variables persisted implicitly across loop
        iterations -- the caller is a persistent per-light-path state
        struct (`VCMLightPathState`) that survives across separate kernel
        launches, one call to this function per bounce.
      - `lvc_path_len[lp_idx]` is NOT written here (unlike the original,
        which wrote it once after the loop) -- the caller writes
        `lvc_path_len[lp_idx] = Int32(n_verts)` unconditionally after every
        call, on both `return True` and `return False`, which is simpler
        and correct on every exit path without needing to track exactly
        which one fired.

    Returns True if the light path should continue to another bounce, False
    if it has terminated (hit nothing, exhausted a material's valid-sample
    conditions, or hit an unhandled material type)."""
        if n_verts >= _BDPT_MAX_VERTS:
            return False   # mirrors the original loop's top-of-iteration guard
        # The next vertex stored would be interior vertex n_verts (after an area
        # light's origin slot) or n_verts + 1 (no origin slot): past d it starts
        # no strategy (see _vcm_depth).
        var next_count = n_verts
        if n_verts == 0 or lvc[unsafe_offset=lp_idx * _BDPT_MAX_VERTS].is_light != Int32(1):
            next_count = n_verts + 1
        if next_count + n_delta > _vcm_depth(sd):
            return False
        if inter.hit == Int8(0):
            return False   # nothing hit -- path escapes the scene
        var t_hit = inter.tHit
        var ray_dir = rd.to_simd()

        # VCM Stage 2b: distance-squared portion of the per-bounce MIS
        # correction (project_vcm_stage2_mis_derivation memory) -- shared
        # across every material branch below, since it only depends on the
        # travel distance, not the hit material. The |cosThetaFix| portion
        # is applied separately inside each branch once its own local
        # normal is known.
        if origin_sphere >= Int32(0):
            # First hit of a sphere light's path: the origin's direct density
            # is the camera NEE's cone pdf seen from HERE, and its area
            # conversion cancels this segment's d^2 (init, sphere branch).
            # Where the cone does not exist (inside the sphere) NEE cannot
            # reach the origin, and dVCM's NEE share is 0.
            dvcm_carry *= sphere_light_cone_pdf(sd.spheres[unsafe_offset=Int(origin_sphere)],
                                                ro.to_simd() + ray_dir * t_hit)
            origin_sphere = Int32(-2)   # applied; still a sphere light's path
        elif n_verts >= 1 or is_finite_origin:
            dvcm_carry *= t_hit * t_hit

        # Volume free-flight
        if has_med and Int(cur_med_idx) >= 0:
            comptime if _VCM_CAMIS:
                camis_l.in_class = False   # a medium segment: see the camera side
            var med = sd.mediums[unsafe_offset=Int(cur_med_idx)]
            # ONE shared sampler for both medium kinds (geometry.mojo):
            # homogeneous closed form, or delta tracking against the real
            # density field. This call site used to be the homogeneous one
            # unconditionally, which rendered every "uniformgrid"/"nanovdb"/
            # "cloud" medium as uniform density-1 fog -- bunny-cloud came out a
            # featureless sphere with no bunny in it. See sample_free_flight.
            var ff = sample_free_flight(
                med, sd.grids, sd.nvdbGrids, Vec3f(ro.x, ro.y, ro.z), rd, t_hit, pcg)
            if ff.collided and origin_sphere != Int32(-1):
                # A sphere light's path ends at its first medium collision.
                # Medium vertices are outside per-vertex MIS: the camera's
                # medium vertex already claims camera -> medium -> sphere
                # paths completely, through its own NEE to the sphere (power
                # heuristic against the phase hit), and a connection to a
                # light-side medium vertex is weighted 1 as well -- the two
                # summed read 1.6x the path tracer on a fog ball lit by a
                # sphere. Area lights never had the overlap: the camera has no
                # area-light NEE of its own at a medium vertex.
                return False
            if ff.collided:
                # Chromatic collision weight; see the camera-side comment.
                flux *= spectral_free_flight_weight(med, ff, t_hit, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                var sp = ro + rd*ff.t_free
                # dVCM's Jacobian above was applied with t_hit (the distance
                # to the SURFACE the ray was cast toward), computed before we
                # knew this segment would collide first -- the real arrival
                # distance is ff.t_free. Correct it before the free-flight
                # factor (Scenes/vcm_volume_mis_derivation.py: dVCM *=
                # d^2/ff_b, d = the vertex's own arrival distance).
                dvcm_carry *= (ff.t_free * ff.t_free) / max(t_hit * t_hit, Float32(1e-20))
                # Free-flight at this VOLUME arrival: ff_b = the collision
                # density actually sampled = ff.pdf. No cos_fix -- a volume
                # vertex has no surface normal to divide by.
                dvcm_carry *= Float32(1.0) / max(ff.pdf, Float32(1e-30))
                # dVC's ff_a/ff_b ratio is 1/sigma_t only when the PREVIOUS
                # vertex was a surface (kind-changing edge); a volume->volume
                # multi-scatter step has ratio 1 (same kind, see the
                # harness's free_flight -- the sigma_t*exp(-sigma_t d) terms
                # are identical on both ends and cancel).
                if not prev_was_volume:
                    dvc_carry *= Float32(1.0) / max(ff.sig_t, Float32(1e-20))
                var v = _null_vertex()
                v.pos = sp
                # `flux`, NOT flux*albedo -- see the matching camera-side
                # comment in _bdpt_camera_path_bounce: a stored vertex's
                # beta/flux must EXCLUDE its own local response, which
                # _eval_vertex_spectral applies fresh at connect/merge time.
                v.beta = flux
                v.alb = ff.albedo
                v.is_surface = Int32(0); v.is_delta = Int32(0)
                v.pdf_fwd = ff.pdf   # the density actually sampled from; under hero-wavelength MIS this is a lane MIXTURE, not sig_t's lone exponential
                v.med_idx = cur_med_idx
                v.wavelengths = wavelengths
                # ARRIVAL carries, before the outgoing scatter step below
                # overwrites them -- the same v.dVCM/dVC/dVM = ... pattern
                # every surface branch uses, so connect/merge (which read
                # these off the stored vertex) see the right state.
                v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                n_verts += 1
                v.n_delta = Int32(n_delta)
                _bdpt_store_lvc_vertex(v, lvc, lp_idx, n_verts - 1)
                comptime if _VCM_CAMIS:
                    # Every stored slot gets a record, out-of-Class ones too:
                    # the buffer outlives the pass, and S3 reads the class bit.
                    camis_light_arrive(camis_l, camis_recs, n_verts - 1, Float32(0), t_hit, Float32(0), False)
                # Distant/infinite/point lights store NO lv0 (n_verts starts at
                # 0 for them -- this function's own docstring: they have no
                # NEE-equivalent cache vertex at all), so THIS volume scatter
                # can land at index 0 and is the ONLY bidirectional-connection
                # pathway those light types have into a medium -- deleting it
                # outright (an earlier version of this fix) broke them (an
                # env-lit slab went from 1.35x to 0.44x pbrt). An AREA-light
                # origin, by contrast, has the light-source vertex ITSELF at
                # index 0, and _bdpt_connect_to_cache restricts a camera
                # VOLUME vertex to connecting to THAT one only -- see its own
                # comment for why connecting to later same-subpath volume
                # vertices too over-counted an n-scatter path (n+1) times
                # (measured 2.7x on an area-lit slab). That restriction is
                # keyed on the light path's own origin, so storing every
                # volume vertex here unconditionally is correct for both.
                # Volume scatter is isotropic (no surface normal, no
                # cosThetaFix): cos_out=1, pdf_fwd=pdf_rev=1/(4pi), so
                # cos_over_pdf = 4pi -- the SAME vcm_scatter_carries every
                # surface branch uses, just with isotropic constants instead
                # of a sampled lobe's (Scenes/vcm_volume_mis_derivation.py:
                # connections+merges exact to 4.4e-16 with this rule).
                var eta_v = _vcm_eta_at(sd, v, mis_vm_weight_factor)
                (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries[dvc0=_VCM_CAMIS](
                    dvcm_carry, dvc_carry, dvm_carry,
                    Float32(4.0) * PI, INV_FOUR_PI, INV_FOUR_PI,
                    mis_vc_weight_factor, eta_v)
                prev_was_volume = True
                # Continuation: flux = prev × alb_s (same as stored vertex beta)
                flux *= spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (ff.albedo).r, (ff.albedo).g, (ff.albedo).b, wavelengths)
                var u1 = pcg.next_float(); var u2 = pcg.next_float()
                var cosT = Float32(2)*u1 - Float32(1)
                var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                var phi  = Float32(2)*PI*u2
                rd = Vec3f(sinT*cos(phi), sinT*sin(phi), cosT)
                ro = sp + rd*Float32(0.0002)
                return True   # volume free-flight scatter: no vertex stored this bounce, path continues
            else:
                flux *= spectral_free_flight_weight(med, ff, t_hit, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                # Same missing free-flight factor on dVCM as the camera path --
                # see _bdpt_camera_path_bounce's matching comment for the
                # derivation and the measured error growth with optical depth.
                var ff_exp_l = -log(max(ff.pdf, Float32(1e-30)))   # optical depth of the density actually sampled from (see FreeFlight.pdf)
                if ff_exp_l > Float32(60.0): ff_exp_l = Float32(60.0)
                dvcm_carry *= exp(ff_exp_l)
                # dVC's ff_a/ff_b ratio is sigma_t only when the PREVIOUS
                # vertex was a volume scatter (kind-changing edge, exiting
                # the medium to survive to this surface); surface->surface
                # ratio is 1, already correct with no change (defect 2,
                # f0a407e4) -- Scenes/vcm_volume_mis_derivation.py.
                if prev_was_volume:
                    dvc_carry *= ff.sig_t
                prev_was_volume = False

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[unsafe_offset=mat_idx]
        var hit = ro + rd*t_hit
        var eta_x = mis_vm_weight_factor * _vcm_eta_scale(sd, hit)   # merging's MIS density HERE

        _resolve_mix(sd, pcg, mat, mat_idx)

        if mat.type == MatKind.diffuse or mat.type == MatKind.diffuse_transmit or mat.type == MatKind.coated_diffuse or mat.type == MatKind.conductor or mat.type == MatKind.measured or mat.type == MatKind.hair or (dielectric_is_rough(mat) and mat.sss_boundary == Int8(0)):
            if not lobe_is_available_of(mat):
                return False   # e.g. a measured table that failed to load
            # GEOMETRY, not material: a curve hit has its own normal and spawn
            # offset (curve_offset_eps), and no texture or bump map.
            var on_curve = inter.primId.type == Int8(5)
            var spawn_eps = Float32(0.0001)
            var gn: Vec3f
            if on_curve:
                var hc_g = _hair_precompute(mat, sd.curves, Int(inter.primId.id1), inter.v, inter.u, (-ray_dir).to_simd())
                gn = hc_g.geo_normal
                spawn_eps = curve_offset_eps(hc_g.radius)
            elif mat.type == MatKind.dielectric:
                gn = _shading_normal_at(inter, sd.meshes, sd.instances, sd.spheres, hit)
            else:
                gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            # A rough dielectric needs the OUTWARD normal: inside vs outside
            # is which side of it wo lies on (lobe_eval's rough_dielectric).
            var outward = mat.type == MatKind.dielectric
            if dot(gn, ray_dir) > Float32(0) and not outward: gn = gn * Float32(-1)
            # VCM Stage 2b: finish the per-bounce MIS correction (the
            # dist² portion was already applied above, shared across
            # branches) -- cos_fix is the incoming ray's cosine against
            # this vertex's own normal, matching SmallVCM's cosThetaFix.
            var cos_fix = abs(dot(-ray_dir, gn))
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_arrival_carries(
                dvcm_carry, dvc_carry, dvm_carry, cos_fix)
            # Real image-texture reflectance -- see the matching comment in
            # _bdpt_trace_camera_and_connect's diffuse branch. Also feeds
            # the continuation flux multiply below, not just the stored
            # vertex, since both must agree on this bounce's actual albedo.
            var eff_alb = mat.albedo
            var (tex_mesh, tv0, tv1, tv2, tex_ok) = _get_tri_verts(inter, sd.meshes)
            if tex_ok and not on_curve:
                eff_alb = _tex_lookup[use_gpu](mat, inter, tv0, tv1, tv2, tex_mesh, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount),
                    uv_footprint_at_hit(inter, sd.meshes, sd.instances, hit.to_simd(), ray_dir, Float32(-1.0), sd.camFp).width)
            # Bump/normal maps -- see the camera-side diffuse branch.
            # Saved BEFORE the perturbation: the stored vertex keeps this as its
            # GEOMETRIC normal, because _connect's solid-angle -> area pdf
            # conversions are built on it. See BDPTVertex.shading_normal.
            var gn_geo = gn
            if not on_curve:
                gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes, sd.instances,
                    gn, gn, hit.to_simd(), ray_dir, Float32(-1.0), sd.camFp,
                    sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            if not outward:
                gn = face_toward(gn, -ray_dir)   # pbrt two-sided reflection, see face_toward
            var v = _null_vertex()
            v.pos = hit
            v.normal = vec3f(gn_geo)
            v.shading_normal = vec3f(gn)
            v.beta = flux
            v.alb = eff_alb
            v.is_surface = Int32(1)
            v.mat_kind = lobe_kind_of(mat.type)   # see the camera subpath
            v.mat_idx = Int32(mat_idx)
            v.pdf_bwd = lobe_param_of(mat)   # LobeCtx.param: a conductor's GGX alpha
            # A smooth conductor is a mirror: it scatters, but stores no
            # vertex and starts no strategy.
            v.is_delta = Int32(1) if lobe_is_delta_of(mat) else Int32(0)
            if on_curve:
                v.hair_curve_idx = Int32(inter.primId.id1)
                v.hair_h = inter.u
                v.hair_v = inter.v
            v.pdf_fwd = Float32(1); v.med_idx = cur_med_idx
            v.wo = vec3f(-ray_dir)  # VCM Stage 2b: needed for _connect's reverse-pdf eval
            v.wavelengths = wavelengths
            v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
            if v.is_delta != Int32(0):
                n_delta += 1
            if v.is_delta == Int32(0):
                n_verts += 1
                v.n_delta = Int32(n_delta)
                _bdpt_store_lvc_vertex(v, lvc, lp_idx, n_verts - 1)
                comptime if _VCM_CAMIS:
                    camis_light_arrive(camis_l, camis_recs, n_verts - 1, cos_fix, t_hit,
                                       log(_vcm_keep(sd, hit)), _bdpt_vertex_mis_scoped(v))
            var sc = _vcm_scatter(v, True, pcg, sd, wavelengths, dvcm_carry, dvc_carry, dvm_carry, mis_vc_weight_factor, eta_x)
            if not sc.valid:
                return False
            comptime if _VCM_CAMIS:
                if v.is_delta != Int32(0) or sc.is_delta:
                    camis_l.in_class = False
                else:
                    camis_light_scatter(camis_l, n_verts - 1, eta_x, sc.cos_out / sc.pdf_fwd,
                                        sc.cos_out, sc.pdf_fwd, sc.pdf_rev)
            rd = vec3f(sc.wi)
            if on_curve:
                ro = hit + vec3f(gn_geo) * (spawn_eps if dot(sc.wi, gn_geo) >= Float32(0) else -spawn_eps)
            else:
                ro = hit + rd*Float32(0.0002)
            flux *= sc.weight

        elif mat.type == MatKind.coated_conductor:
            var wo_c = (-rd).to_simd()
            var (bs_c, gn_c, gn_c_geo) = _coated_conductor_scatter[use_gpu](
                sd, pcg, mat, inter, hit.to_simd(), ray_dir, wo_c, Float32(-1.0))
            if bs_c.is_valid == Int8(0):
                return False   # conductor/coated_conductor sample invalid
            # VCM Stage 2d: rough conductor DOES have a real standalone pdf
            # (bxdf_pdf_conductor_ggx) -- finish the shared dist² correction
            # the same way diffuse does, mirroring
            # _bdpt_trace_camera_and_connect's matching conductor branch.
            var alpha_c = max(mat.roughU, mat.roughV)
            var cos_fix_c = abs(dot(-ray_dir, gn_c))
            if cos_fix_c > Float32(1e-6):
                dvc_carry /= cos_fix_c
                dvm_carry /= cos_fix_c
            if not bxdf_is_delta(bs_c.flags):
                var v = _null_vertex()
                v.pos = hit
                v.normal = vec3f(gn_c_geo)
                v.shading_normal = vec3f(gn_c)
                v.beta = flux
                v.alb = mat.albedo
                v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = LobeKind.ggx
                v.pdf_bwd = alpha_c
                v.wo = vec3f(wo_c)
                v.pdf_fwd = Float32(1)
                v.med_idx = cur_med_idx
                v.wavelengths = wavelengths
                v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                n_verts += 1
                v.n_delta = Int32(n_delta)
                _bdpt_store_lvc_vertex(v, lvc, lp_idx, n_verts - 1)
                comptime if _VCM_CAMIS:
                    camis_light_arrive(camis_l, camis_recs, n_verts - 1, cos_fix_c, t_hit,
                                       log(_vcm_keep(sd, hit)), _bdpt_vertex_mis_scoped(v))
            flux *= spec_refl_unbounded(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (bs_c.f).r, (bs_c.f).g, (bs_c.f).b, wavelengths)
            rd = vec3f(bs_c.wi)
            ro = hit + rd*Float32(0.0002)
            var cos_theta_out_c = abs(dot(bs_c.wi, gn_c))
            if bxdf_is_delta(bs_c.flags):
                # VCM: specular bounce -- same reset SmallVCM's own delta-
                # bounce case uses, matches the camera-side conductor branch.
                dvcm_carry = Float32(0)
                dvc_carry *= cos_theta_out_c
                dvm_carry *= cos_theta_out_c
                comptime if _VCM_CAMIS:
                    camis_l.in_class = False
            else:
                var bsdf_dir_pdf_w_c = bxdf_pdf_conductor_ggx(gn_c, wo_c, bs_c.wi, alpha_c)
                if bsdf_dir_pdf_w_c > Float32(1e-8):
                    var bsdf_rev_pdf_w_c = bxdf_pdf_conductor_ggx(gn_c, bs_c.wi, wo_c, alpha_c)
                    var inv_pdf_c = cos_theta_out_c / bsdf_dir_pdf_w_c
                    (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries[dvc0=_VCM_CAMIS](
                        dvcm_carry, dvc_carry, dvm_carry, inv_pdf_c, bsdf_dir_pdf_w_c, bsdf_rev_pdf_w_c,
                        mis_vc_weight_factor, eta_x)
                    comptime if _VCM_CAMIS:
                        camis_light_scatter(camis_l, n_verts - 1, eta_x, inv_pdf_c,
                                            cos_theta_out_c, bsdf_dir_pdf_w_c, bsdf_rev_pdf_w_c)
                else:
                    dvcm_carry = Float32(0)
                    dvc_carry = Float32(0)
                    dvm_carry = Float32(0)
                    comptime if _VCM_CAMIS:
                        camis_l.in_class = False

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            comptime if _VCM_CAMIS:
                camis_l.in_class = False   # BSSRDF hop or delta dielectric: see the camera side
            # ── Subsurface boundary: the light subpath's half of the hop ───
            # Mirror of the camera side: the light enters at x_i, the exit x_o
            # is drawn by the SAME sampler (the MIS weights require identical
            # densities on both sides; the hop is symmetric), x_o is stored as
            # a connectible light vertex with the exit lobe, and the subpath
            # continues from it. The entry is never stored.
            var did_bssrdf_hop = False
            if mat.sss_boundary != Int8(0) and Int(cur_med_idx) < 0 and has_med:
                var med_in = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                if med_in >= Int32(0) and Int(med_in) < Int(sd.mediumCount):
                    var gn_e = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
                    if dot(gn_e, ray_dir) > Float32(0): gn_e = gn_e * Float32(-1)
                    var cos_e = abs(dot(-ray_dir, gn_e))
                    (dvcm_carry, dvc_carry, dvm_carry) = vcm_arrival_carries(
                        dvcm_carry, dvc_carry, dvm_carry, cos_e)
                    var eta_e = mat.albedo.r
                    var ex = _bdpt_sample_bssrdf_exit(sd, Int(med_in), hit, gn_e, cos_e, eta_e, pcg)
                    if not ex.ok:
                        return False
                    flux *= spec_refl_unbounded(
                        sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x,
                        sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65,
                        ex.weight.r, ex.weight.g, ex.weight.b, wavelengths)
                    var (h0, h1, h2) = bssrdf_hop_carries[dvc0=_VCM_CAMIS](
                        dvcm_carry, dvc_carry, dvm_carry, ex.p_area, cos_e * INV_PI, mis_vc_weight_factor)
                    dvcm_carry = h0
                    dvc_carry = h1
                    dvm_carry = h2
                    var n_o = ex.n_o
                    var v = _null_vertex()
                    v.pos = ex.x_o
                    v.normal = vec3f(n_o)
                    v.shading_normal = vec3f(n_o)
                    v.beta = flux
                    v.alb = RGB(Float32(1))
                    v.is_surface = Int32(1)
                    v.mat_kind = LobeKind.bssrdf
                    v.pdf_fwd = ex.p_area
                    v.pdf_bwd = eta_e
                    v.wo = vec3f(n_o)
                    v.med_idx = cur_med_idx
                    v.wavelengths = wavelengths
                    v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                    n_verts += 1
                    v.n_delta = Int32(n_delta)
                    _bdpt_store_lvc_vertex(v, lvc, lp_idx, n_verts - 1)
                    comptime if _VCM_CAMIS:
                        camis_light_arrive(camis_l, camis_recs, n_verts - 1, Float32(0), t_hit, Float32(0), False)
                    var ux1 = pcg.next_float(); var ux2 = pcg.next_float()
                    rd = vec3f(_cosine_hemisphere_sample(n_o, ux1, ux2))
                    ro = ex.x_o + rd * Float32(0.0002)
                    var cos_out_x = abs(dot(rd.to_simd(), n_o))
                    flux *= SpectralSample(bssrdf_exit_ft(cos_out_x, eta_e))
                    var (c0, c1, c2) = bssrdf_exit_scatter_carries[dvc0=_VCM_CAMIS](
                        dvcm_carry, dvc_carry, dvm_carry, ex.p_area, cos_out_x,
                        mis_vc_weight_factor, mis_vm_weight_factor * _vcm_eta_scale(sd, ex.x_o))
                    dvcm_carry = c0
                    dvc_carry = c1
                    dvm_carry = c2
                    n_lbounces += 1
                    did_bssrdf_hop = True
            if not did_bssrdf_hop:
                n_delta += 1
                # pbrt's outward normal: interpolated N when the mesh has it
                # (_shading_normal_at), which pbrt also refracts through.
                var gn = _shading_normal_at(inter, sd.meshes, sd.instances, sd.spheres, hit)
                # Bump/normal maps. barcelona's water is a `dielectric` with a
                # "texture displacement", so this is THE branch the corpus
                # cares about most -- and VCM refracted through a perfectly
                # flat sheet while the path tracer refracted through a
                # rippled one. `orient_to` is the RAW winding normal, NOT a
                # face-forwarded one: _dielectric_bounce decides entering vs
                # exiting from dot(ray_dir, n) < 0, and flipping the
                # perturbed normal toward the ray would make that test
                # tautological and resurrect the 1/eta^4 transmission loss
                # (same rule as shading.mojo's dielectric site).
                gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes, sd.instances,
                    gn, gn, hit.to_simd(), ray_dir, Float32(-1.0), sd.camFp,
                    sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
                var (new_dir, new_org, _, new_cur_ior, new_prev_ior) = _dielectric_bounce(
                    ray_dir, hit.to_simd(), gn, mat.albedo.r, n_lbounces == 0 and Int(cur_med_idx) < 0, pcg, current_dielectric_ior, previous_dielectric_ior, mat.type == MatKind.thin_dielectric, radiance_mode=False)
                current_dielectric_ior = new_cur_ior
                previous_dielectric_ior = new_prev_ior
                n_lbounces += 1
                # Light path (TransportMode::Importance): do NOT apply the
                # radiance_scale non-symmetric-scattering correction — it's only
                # for camera/Radiance-mode paths, see _dielectric_bounce's
                # docstring.
                if has_med:
                    var new_idx = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                    if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
                rd = vec3f(new_dir)
                ro = point3f(new_org)
                # VCM Stage 2b: dielectric is genuinely delta/specular (true
                # reflect-or-refract, not an approximation like conductor's
                # VNDF sampling) -- matches SmallVCM's own specular-bounce
                # handling directly (vertexcm.hxx:977-982), no LVC vertex
                # stored either way.
                var cos_fix_d = abs(dot(-ray_dir, gn))
                if cos_fix_d > Float32(1e-6):
                    dvc_carry /= cos_fix_d
                    dvm_carry /= cos_fix_d
                var cos_theta_out_d = abs(dot(new_dir, gn))
                dvcm_carry = Float32(0)
                dvc_carry *= cos_theta_out_d
                dvm_carry *= cos_theta_out_d

        elif mat.type == MatKind.interface:
            if has_med:
                var new_idx = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            ro = hit + rd*Float32(0.0002)
            # VCM Stage 2b: pure medium-boundary pass-through, no direction
            # change/BSDF event -- carry state passes through as already
            # dist²-corrected above (the ray genuinely traveled t_hit), no
            # cosFix/reset applied since there's no real "bounce" here.
            comptime if _VCM_CAMIS:
                camis_l.in_class = False   # a split edge: see the camera side

        else:
            return False   # unhandled material type


        return True   # bounce processed normally, path continues

def _bdpt_trace_light_path[use_gpu: Bool](
    ref sd:      SceneView,
    mut pcg: PCG32,
    has_med: Bool,
    default_emit_med: Int32,
    scratch: Pointer[Intersection, MutUntrackedOrigin],
    lvc:      Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx:   Int,
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    pass_wl: SampledWavelengths,
    # _VCM_CAMIS: CAMIS records parallel to `lvc` (vcm_camis.CamisLightRecord),
    # written at every stored vertex -- required, like `lvc`, because with the
    # hybrid compiled in a forgotten buffer is written through, not ignored.
    # Unused (any pointer will do) when it is compiled out.
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin],
):
    """Emit a photon from a random light and trace a light subpath, storing
    every non-delta vertex (including the light-source point itself, the
    s=1/NEE-equivalent strategy) into light path `lp_idx`'s own dedicated
    LVC slice via `_bdpt_store_lvc_vertex` — see the module's LVC-BPT
    docstring above and that function's docstring for the VCM Stage 2b
    per-path-indexed storage layout. `_BDPT_MAX_VERTS` caps how many
    vertices any ONE light path contributes; `lvc_path_len[lp_idx]` records
    how many of those slots this path actually filled (may be fewer than
    _BDPT_MAX_VERTS if the path terminated early), initialized to 0 up
    front so every exit path (including the early returns below) leaves it
    correctly set.

    `lp_idx` is also the pixel index this light path is deterministically
    PAIRED with for connections (n_light_paths == n_pix, one light path per
    pixel) — real VCM's dVCM/dVC/dVM MIS weights assume this pairing (see
    project_vcm_stage2_mis_derivation memory), unlike the old design's
    random draws from a shared pool across all pixels.

    Lights are chosen uniformly across ALL light types (area + distant +
    infinite), not just area lights — `n_lights` below is this combined
    total, so every per-light PDF-correction factor (`area_weight`, the
    distant/infinite flux scale) uses it, not just an area-only count.
    Distant/infinite lights have no finite position of their own, so unlike
    the area-light case, no vertex-0 (`lv0_vert`) is stored for them — a
    disk-sampled point on the scene's bounding sphere (see bvh.mojo's
    `_scene_bounding_sphere`/`_sample_disk_perpendicular`) isn't a real
    scene point and its G(a,b) inverse-square/cosine term would be
    physically wrong for a directional source. Direct (NEE-equivalent)
    illumination from these lights is instead provided by
    `_bdpt_trace_camera_and_connect`'s own per-vertex NEE to distant/
    infinite lights — this function only needs to seed a physically correct
    ray+flux and let the ordinary bounce loop below take over once that ray
    hits a real surface (which DOES get stored/connected normally)."""

    # SHARED WITH THE WAVEFRONT GPU DRIVER. This body used to be a
    # byte-for-byte copy of _bdpt_light_path_init + _bdpt_light_path_bounce
    # -- the two halves task #163 split out for wavefront staging -- so
    # every material/MIS/estimator change had to be made twice and silently
    # drifted whenever it wasn't. That drift is exactly why --vcm-wavefront
    # still lacks the t=1 splat and the Vulkan RT coverage fixes. Both
    # designs now run the SAME step: this one loops over it inline, the
    # wavefront driver launches it once per depth level with the loop-carried
    # state parked in a VCMLightPathState between launches.
    var st = _bdpt_light_path_init[use_gpu](
        sd, pcg, default_emit_med, lp_idx, lvc, lvc_path_len, mis_vc_weight_factor, pass_wl)
    if st.active == Int8(0):
        return
    var ro = st.ro
    var rd = st.rd
    var flux = st.flux
    var n_verts = Int(st.n_verts)
    var n_delta = Int(st.n_delta)
    var dvcm_carry = st.dvcm
    var dvc_carry = st.dvc
    var dvm_carry = st.dvm
    var is_finite_origin = st.is_finite_origin == Int8(1)
    var origin_sphere = st.origin_sphere
    var cur_med_idx = st.cur_med_idx
    var n_lbounces = Int(st.n_lbounces)
    var current_dielectric_ior = st.current_dielectric_ior
    var previous_dielectric_ior = st.previous_dielectric_ior
    var wavelengths = SampledWavelengths(st.wl0, st.wl1, st.wl2, st.wl3)
    var prev_was_volume = False
    # CAMIS light state. Only an AREA light stores an origin vertex (n_verts
    # starts at 1), and only an area light starts a path inside the Class;
    # every other light's first stored vertex writes an out-of-Class record.
    var camis_l = camis_light_carry_off()
    var camis_recs: Pointer[CamisLightRecord, MutUntrackedOrigin]
    comptime if _VCM_CAMIS:
        camis_recs = lvc_camis.unsafe_offset(lp_idx * _BDPT_MAX_VERTS)
        var lv0 = lvc[unsafe_offset=lp_idx * _BDPT_MAX_VERTS]
        if n_verts == 1 and lv0.is_light == Int32(1):
            # lv0.pdf_fwd is the origin's position density P_A (init, area
            # branch), and the emission cosine is clamped as init clamps it.
            camis_l = camis_light_origin(camis_recs, 0, log(lv0.pdf_fwd),
                                         max(dot(rd.to_simd(), lv0.normal.to_simd()), Float32(0.0001)))
    else:
        camis_recs = lvc_camis   # never dereferenced

    for _ in range(_BDPT_MAX_DEPTH):
        # The same intersect step _bdpt_light_path_intersect_gpu performs --
        # kept here rather than inside the bounce because this one step is
        # the eventual Vulkan RT swap point on the wavefront side.
        var ray = Ray(ro, rd)
        scratch[unsafe_offset=0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, scratch)
        # A miss is handled inside the step (returns False), so the old
        # top-of-loop `if scratch[0].hit == 0: break` is no longer needed.
        if not _bdpt_light_path_bounce[use_gpu](
            sd, pcg, has_med, scratch[unsafe_offset=0], lvc, lp_idx,
            mis_vc_weight_factor, mis_vm_weight_factor,
            ro, rd, flux, n_verts, n_delta, dvcm_carry, dvc_carry, dvm_carry,
            prev_was_volume,
            is_finite_origin, origin_sphere, cur_med_idx, n_lbounces,
            current_dielectric_ior, previous_dielectric_ior, wavelengths,
            camis_l, camis_recs):
            break

    lvc_path_len[unsafe_offset=lp_idx] = Int32(n_verts)
    return

# ── BSDF/phase evaluation at a vertex ────────────────────────────────────────
