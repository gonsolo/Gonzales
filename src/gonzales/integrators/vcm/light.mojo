# BDPT/VCM light subpath: state, init and the per-bounce step.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.math import sqrt, cos, sin, log, exp, max
from gonzales.transport.medium import MediumView
from gonzales.transport.sss_walk import sss_entered, sss_walks_below, sss_relative_eta
from gonzales.transport.medium_interface import medium_after
from gonzales.core.geometry import Point3f, Point2f, Vec3f, vec3f, point3f, dot, face_toward, PI, INV_FOUR_PI, RGB
from gonzales.scene.appearance.materials import MatKind, LobeKind, Material, resolve_mix
from gonzales.scene.shapes.primitives import Intersection
from gonzales.core.geometry import Ray
from gonzales.paths.vcm.mis import vcm_scatter_carries, vcm_phase_scatter, vcm_arrival_carries
from gonzales.scene.accel.bvh import traverse_dispatch
from gonzales.scene.accel.bvh import SceneView, test_spheres, _scene_bounding_sphere, _sample_disk_perpendicular, _sample_infinite_light_dir, sphere_light_cone_pdf, light_path_pick
from gonzales.core.rng import PCG32
from gonzales.integrators.sppm.sppm import _cosine_hemisphere_sample, sample_area_light_point, sample_sphere_light_emission, area_light_emit_normal
from gonzales.scene.appearance.lights import area_light_side_pdf
from gonzales.core.spectrum import SampledWavelengths, SpectralSample, spec_refl, spec_illum
from gonzales.scene.shapes.geom_normal import _geom_normal
from gonzales.paths.vcm.vertex import _VCM_SSS_CAMERA_ONLY, _vcm_walk_limit, _BDPT_MAX_VERTS, BDPTVertex, _null_vertex, VCM_GSMS_ARRIVAL
from gonzales.transport.glossy_sms import GLOSSY_SMS_MIN_PDF
from gonzales.paths.vcm.eval import _bdpt_vertex_mis_scoped, _vcm_mnee_light_dvcm, _vcm_gsms_on, _vcm_gsms_light_dvcm
from gonzales.paths.vcm.strategy import VcmStrategies, _VCM_EXIT_MERGE, _vcm_keep, _vcm_eta_scale, _vcm_fibre_scale
from gonzales.paths.vcm.connect import _bdpt_store_lvc_vertex
from gonzales.integrators.vcm.camera import (
    _vcm_scatter, _vcm_lobe_vertex, _vcm_sss_walk, _vcm_sss_exit_vertex,
    _vcm_sss_exit_scatter, _vcm_dielectric_cross, VcmWalkResume, _vcm_sss_walk_suspend,
)
from gonzales.transport.render_state import PathState
from gonzales.transport.sss_walk import SssExit

@fieldwise_init
struct VCMMneeLightTrack(TrivialRegisterPassable):
    """A light path that leaves its light through glass only, for MNEE's share at the first surface it reaches."""
    var y: Vec3f              # the light point
    var ny: Vec3f             # its emitting normal
    var side: Float32         # its emission-face probability; 0: no longer glass only, MNEE has no share
    var first_glass: Vec3f    # the glass vertex nearest the light
    var n_glass: Int32


@always_inline
def vcm_mnee_light_track_none() -> VCMMneeLightTrack:
    var z = Vec3f(Float32(0))
    return VCMMneeLightTrack(z, z, Float32(0), z, Int32(0))


@fieldwise_init
struct VCMLightPathState(TrivialRegisterPassable):
    """Task #163 stage 4: persistent per-light-path state carried across
    separate wavefront-staged GPU kernel launches (`_bdpt_light_path_init_gpu`"""
    var ro: Point3f
    var rd: Vec3f
    var flux: SpectralSample
    var dvcm: Float32
    var dvc: Float32
    var dvm: Float32
    var is_finite_origin: Bool
    # Index into sd.spheres of the sphere light this path left, until its
    # first hit applies the origin's 1/Omega (see the init's sphere branch);
    # -2 once applied; -1 for every other light.
    var origin_sphere: Int32
    var cur_med_idx: Int32
    var n_verts: Int
    var active: Int8
    var pcg_state: UInt64
    var pcg_inc: UInt64
    var wl: SampledWavelengths
    # Touching-dielectric IOR depth-2 stack for _dielectric_bounce (see that
    # function's docstring, sppm.mojo) -- same role and convention as
    # VCMCameraPathState's matching fields. Both start at vacuum (1.0).
    var current_dielectric_ior: Float32
    var previous_dielectric_ior: Float32
    var n_delta: Int
    # Null-interface distance since the last real vertex: the d^2 of a crossing is applied once, over the whole edge.
    var mis_null_dist: Float32
    var prev_sig_t: Float32   # sigma_t at the previous vertex, 0 on a surface: dVC's free-flight ratio (MediumView.ff_ratio)
    var mnee: VCMMneeLightTrack

def _null_light_path_state() -> VCMLightPathState:
    var z = Float32(0)
    return VCMLightPathState(
        ro=Point3f(z, z, z), rd=Vec3f(z, z, z), flux=SpectralSample(z), dvcm=z, dvc=z, dvm=z, is_finite_origin=False,
        origin_sphere=Int32(-1), cur_med_idx=Int32(-1), n_verts=0, active=Int8(0), pcg_state=UInt64(0), pcg_inc=UInt64(0),
        wl=SampledWavelengths(z, z, z, z), current_dielectric_ior=Float32(1.0), previous_dielectric_ior=Float32(1.0),
        n_delta=0, mis_null_dist=z, prev_sig_t=Float32(0), mnee=vcm_mnee_light_track_none(),
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
    """A light subpath's emission: its origin vertex, first ray and MIS carries."""
    var n_area = Int(sd.areaLightCount)
    var n_sphere = Int(sd.sphereLightCount)
    var n_distant = Int(sd.distantLightCount)
    var n_infinite = Int(sd.infiniteLightCount)
    var n_point = Int(sd.pointLightCount)
    var n_lights = n_area + n_sphere + n_distant + n_infinite + n_point   # == light_path_count
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
    var dvcm_carry: Float32
    var dvc_carry: Float32
    var dvm_carry: Float32
    var is_finite_origin = False
    var mnee_l = vcm_mnee_light_track_none()
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
        var light_sample = sample_area_light_point(sd.areaLights[unsafe_offset=light_pick], sd.meshes, pcg, sd.curves)
        var al = light_sample.light
        var lp = light_sample.point
        var ln = area_light_emit_normal(light_sample, pcg)
        var side_pdf = area_light_side_pdf(al)

        # Cosine-weighted emission direction
        var du1 = pcg.next_float(); var du2 = pcg.next_float()
        var pdir = _cosine_hemisphere_sample(ln, du1, du2)

        # Light vertex (the emission point, s=1 strategy connects here directly).
        # beta = 1/p_A = area × n_lights (area sampling PDF correction only).
        var area_weight = al.total_area * inv_pick
        var lv0_vert = _null_vertex()
        lv0_vert.pos = point3f(lp)
        lv0_vert.normal = vec3f(ln)
        lv0_vert.shading_normal = vec3f(ln)
        lv0_vert.pdf_bwd = side_pdf   # a light vertex's pdf_bwd carries its emission-face probability
        mnee_l = VCMMneeLightTrack(lp, ln, side_pdf, Vec3f(Float32(0)), Int32(0)) if al.kind == Int8(0) else mnee_l
        lv0_vert.beta = SpectralSample(area_weight)
        lv0_vert.alb = al.emission
        lv0_vert.is_surface = Int32(1); lv0_vert.is_light = Int32(1)
        lv0_vert.pdf_fwd = Float32(1) / area_weight
        lv0_vert.med_idx = default_emit_med
        lv0_vert.wavelengths = wavelengths

        # VCM Stage 2b: real MIS origin state for this (finite, area) light
        # -- see project_vcm_stage2_mis_derivation memory for the verified
        is_finite_origin = True
        var cos_theta_emit = max(dot(pdir, ln), Float32(0.0001))
        var direct_pdf_a = Float32(1) / area_weight
        var emission_pdf_w = direct_pdf_a * cos_theta_emit * side_pdf / PI
        dvcm_carry = direct_pdf_a / emission_pdf_w
        dvc_carry = cos_theta_emit / emission_pdf_w
        # SmallVCM vertexcm.hxx:856 -- light-origin dVM uses mMisVcWeightFactor,
        # NOT mMisVmWeightFactor (verified against the reference source; a
        dvm_carry = dvc_carry * mis_vc_weight_factor
        lv0_vert.dVCM = dvcm_carry
        lv0_vert.dVC = dvc_carry
        lv0_vert.dVM = dvm_carry
        _bdpt_store_lvc_vertex(lv0_vert, lvc, lp_idx, 0)

        # For traced vertices: beta = Le × cos_θ / (p_A × p_ω) where p_ω = cos_θ/π
        # for cosine-weighted emission -- the cos_θ_emitted terms CANCEL exactly
        flux = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al.emission.r, al.emission.g, al.emission.b, wavelengths) * (area_weight * PI / side_pdf)
        ro = point3f(lp) + vec3f(ln)*Float32(0.0001)
        rd = vec3f(pdir)
        n_verts = 1  # vertex 0 is the light point itself
    elif light_pick < n_area + n_sphere:
        # Sphere light: the shared emission sampler (a uniform point on the
        # emitter, a cosine direction -- the mesh-light model with area
        var (si_l, sp_l, sn_l, sdir_l) = sample_sphere_light_emission(sd, light_pick - n_area, pcg)
        var sph_l = sd.spheres[unsafe_offset=si_l]
        var area_weight_s = Float32(4) * PI * sph_l.radius * sph_l.radius * inv_pick
        flux = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, sph_l.emission.r, sph_l.emission.g, sph_l.emission.b, wavelengths) * (area_weight_s * PI)
        ro = point3f(sp_l + sn_l * (sph_l.radius * Float32(0.0001)))
        rd = vec3f(sdir_l)
        n_verts = 0
        is_finite_origin = True
        origin_sphere = Int32(si_l)
        mnee_l = VCMMneeLightTrack(sp_l, sn_l, Float32(1), Vec3f(Float32(0)), Int32(0))
        dvcm_carry = PI * area_weight_s
        dvc_carry = PI * area_weight_s
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
        dvcm_carry = PI * radius * radius * inv_pick
        dvc_carry = Float32(0)
        dvm_carry = Float32(0)
    elif light_pick < n_area + n_sphere + n_distant + n_infinite:
        var il = sd.infiniteLights[unsafe_offset=light_pick - n_area - n_sphere - n_distant]
        var (center, radius) = _scene_bounding_sphere(sd)
        # _sample_infinite_light_dir returns env_dir in the NEE convention
        # ("direction FROM a shading point TOWARD the light" — same as
        var (env_dir, env_rgb, pdf_dir) = _sample_infinite_light_dir(il, Point2f(pcg.next_float(), pcg.next_float()))
        if pdf_dir <= Float32(0):
            return _null_light_path_state()   # infinite-light dir sample had zero pdf
        var emit_dir = -env_dir
        var disk_pt = _sample_disk_perpendicular(emit_dir, center, radius, Point2f(pcg.next_float(), pcg.next_float()))
        if not il.portal_admits(disk_pt, emit_dir):
            return _null_light_path_state()
        # Same derivation as distant, but pdf_dir is the CDF's solid-angle pdf
        # instead of an implicit delta (=1): flux = radiance × disk_area × n_lights / pdf_dir.
        flux = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, env_rgb.r, env_rgb.g, env_rgb.b, wavelengths) * (inv_pick * PI * radius * radius / pdf_dir)
        ro = disk_pt
        rd = emit_dir
        n_verts = 0
        # VCM Stage 2b MIS origin for a NON-FINITE (environment) light.
        # SmallVCM vertexcm.hxx GenerateLightSample, background branch:
        var disk_area = PI * radius * radius
        dvcm_carry = disk_area * inv_pick
        dvc_carry = disk_area * inv_pick / pdf_dir
        dvm_carry = dvc_carry * mis_vc_weight_factor
    else:
        # Point light: a real finite position (unlike distant/infinite), but
        # still no NEE-equivalent cache vertex — see this function's own
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
        is_finite_origin = True
        dvcm_carry = Float32(4) * PI * inv_pick
        dvc_carry = Float32(0)
        dvm_carry = Float32(0)

    var cur_med_idx = default_emit_med

    return VCMLightPathState(
        ro=ro, rd=rd, flux=flux, dvcm=dvcm_carry, dvc=dvc_carry, dvm=dvm_carry, is_finite_origin=is_finite_origin,
        origin_sphere=origin_sphere, cur_med_idx=cur_med_idx, n_verts=n_verts, active=Int8(1),
        pcg_state=pcg.state, pcg_inc=pcg.inc, wl=wavelengths,
        current_dielectric_ior=Float32(1.0), previous_dielectric_ior=Float32(1.0),   # vacuum
        n_delta=0, mis_null_dist=Float32(0), prev_sig_t=Float32(0), mnee=mnee_l,
    )


def _vcm_skin_arrival(
    ref sd: SceneView, inter: Intersection, mat: Material, hit: Point3f, ray_dir: Vec3f, mis_vm: Float32,
    mut pcg: PCG32, lvc: Pointer[BDPTVertex, MutUntrackedOrigin], lp_idx: Int, mut st: VCMLightPathState,
    mut dvcm_in: Float32, dvc_in: Float32, dvm_in: Float32,
) -> Float32:
    """A photon arriving at skin, stored before the boundary chooses for camera walk exits to merge
    (Scenes/vcm_sss_exit_merge_derivation.py); returns the walk's entry merge share, 0 when no slot is left."""
    var gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
    var cos_a = abs(dot(ray_dir, gn))
    if st.mnee.side > Float32(0) and st.mnee.n_glass > Int32(0):
        # Reached through glass only: the camera exit's MNEE samples this entry too.
        var d_m = _vcm_mnee_light_dvcm(sd, hit.to_simd(), gn, st.mnee.y, st.mnee.ny, st.mnee.side,
                                       Int(st.mnee.n_glass), st.mnee.first_glass, st.cur_med_idx, st.wl, pcg)
        if d_m > Float32(0):
            dvcm_in = d_m * cos_a   # the walk applies the arrival cosine
    comptime if not _VCM_EXIT_MERGE:
        return Float32(0)
    # Stored whenever a slot is left: the camera counts one per skin hit (cam_nd), so every split of a path classifies
    # it alike and the exit merge runs wherever that classification counts it (Scenes/vcm_sss_arrival_slots.py).
    if st.n_verts >= _BDPT_MAX_VERTS:
        return Float32(0)
    var (a_vcm, a_vc, a_vm) = vcm_arrival_carries(dvcm_in, dvc_in, dvm_in, cos_a)
    var a = _null_vertex()
    a.pos = hit
    a.normal = vec3f(face_toward(gn, -ray_dir))
    a.shading_normal = a.normal
    a.beta = st.flux
    a.alb = RGB(Float32(1))
    a.is_surface = Int32(1)
    a.mat_kind = LobeKind.sss_entry
    a.pdf_bwd = sss_relative_eta(mat, st.current_dielectric_ior)
    a.wo = vec3f(-ray_dir)
    a.med_idx = st.cur_med_idx
    a.wavelengths = st.wl
    a.dVCM = a_vcm; a.dVC = a_vc; a.dVM = a_vm
    a.n_delta = Int32(st.n_delta)
    _bdpt_store_lvc_vertex(a, lvc, lp_idx, st.n_verts)
    st.n_verts += 1
    st.n_delta -= 1   # the slot is no path vertex: what happens at the hit counts itself
    return VcmStrategies.walk_merge_density(sd, hit, mis_vm)


def _bdpt_light_path_bounce[use_gpu: Bool, defer_walk: Bool = False](
    ref sd:      SceneView,
    mut pcg: PCG32,
    has_med: Bool,
    inter: Intersection,
    lvc:      Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx:   Int,
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    mut st: VCMLightPathState,
    walkers: Pointer[PathState, MutUntrackedOrigin] = Pointer[PathState, MutUntrackedOrigin].unsafe_dangling(),
    walk_resume: Pointer[VcmWalkResume, MutUntrackedOrigin] = Pointer[VcmWalkResume, MutUntrackedOrigin].unsafe_dangling(),
) -> Bool:
    """One bounce of a light subpath, shared by the per-path loop and the wavefront kernels; False ends the path.
    `defer_walk`: as _bdpt_camera_path_bounce's, _vcm_light_walk_exit ends the bounce."""
        if st.n_verts >= _BDPT_MAX_VERTS:
            return False   # mirrors the original loop's top-of-iteration guard
        # The next vertex stored would be interior vertex n_verts (after an area
        # light's origin slot) or n_verts + 1 (no origin slot): past d it starts
        # no strategy (see VcmStrategies.depth).
        var next_count = st.n_verts
        if st.n_verts == 0 or lvc[unsafe_offset=lp_idx * _BDPT_MAX_VERTS].is_light != Int32(1):
            next_count = st.n_verts + 1
        if next_count + st.n_delta > VcmStrategies.depth(sd) or VcmStrategies.beyond_cache(next_count):
            return False
        if inter.hit == Int8(0):
            return False   # nothing hit -- path escapes the scene
        var t_hit = inter.tHit
        var ray_dir = st.rd.to_simd()
        var d_seg = Float32(0)   # the d^2 applied to dVCM at this hit, 0 if none
        var cone_hit = Float32(0); var sphere_at = Int32(-1)   # a sphere light's first hit: its direct density from there

        # VCM Stage 2b: distance-squared portion of the per-bounce MIS
        # correction (project_vcm_stage2_mis_derivation memory) -- shared
        if st.origin_sphere >= Int32(0):
            # First hit of a sphere light's path: the origin's direct density
            # is the camera NEE's cone pdf seen from HERE, and its area
            cone_hit = sphere_light_cone_pdf(sd.spheres[unsafe_offset=Int(st.origin_sphere)],
                                                st.ro.to_simd() + ray_dir * t_hit)
            sphere_at = st.origin_sphere
            st.dvcm *= cone_hit
            st.origin_sphere = Int32(-2)   # applied; still a sphere light's path
        elif st.n_verts >= 1 or st.is_finite_origin:
            d_seg = t_hit + st.mis_null_dist
            st.dvcm *= d_seg * d_seg

        # Volume free-flight
        if has_med and Int(st.cur_med_idx) >= 0:
            var med = MediumView.at(sd, st.cur_med_idx)
            # ONE shared sampler for both medium kinds (geometry.mojo):
            # homogeneous closed form, or delta tracking against the real
            var ff = med.free_flight(Vec3f(st.ro.x, st.ro.y, st.ro.z), st.rd, t_hit, pcg, st.wl)
            if ff.collided:
                # Chromatic collision weight; see the camera-side comment.
                st.flux *= med.weight(ff, t_hit, st.wl)
                var sp = st.ro + st.rd*ff.t_free
                # dVCM's Jacobian above was applied with t_hit (the distance
                # to the SURFACE the ray was cast toward), computed before we
                if d_seg > Float32(0):
                    st.dvcm *= ((ff.t_free + st.mis_null_dist) * (ff.t_free + st.mis_null_dist)) / (d_seg * d_seg)
                elif sphere_at >= Int32(0):
                    st.dvcm *= sphere_light_cone_pdf(sd.spheres[unsafe_offset=Int(sphere_at)], sp.to_simd()) / max(cone_hit, Float32(1e-30))
                # Free-flight at this VOLUME arrival: ff_b = the collision
                # density actually sampled = ff.pdf. No cos_fix -- a volume
                # vertex has no surface normal to divide by.
                st.dvcm *= Float32(1.0) / max(ff.pdf, Float32(1e-30))
                # dVC's ff_a/ff_b ratio is 1/sigma_t only when the PREVIOUS
                # vertex was a surface (kind-changing edge); a volume->volume
                st.dvc *= med.ff_ratio(st.prev_sig_t > Float32(0), st.prev_sig_t, ff.t_free, ff, st.wl)
                var v = _null_vertex()
                v.pos = sp
                # `flux`, NOT flux*albedo -- see the matching camera-side
                # comment in _bdpt_camera_path_bounce: a stored vertex's
                v.beta = st.flux
                v.alb = ff.albedo
                v.is_surface = Int32(0); v.is_delta = Int32(0)
                v.wo = vec3f(-ray_dir); v.pdf_bwd = med.med.g
                v.pdf_fwd = ff.pdf   # the density actually sampled from; under hero-wavelength MIS this is a lane MIXTURE, not sig_t's lone exponential
                v.med_idx = st.cur_med_idx
                v.wavelengths = st.wl
                # ARRIVAL carries, before the outgoing scatter step below
                # overwrites them -- the same v.dVCM/dVC/dVM = ... pattern
                v.dVCM = st.dvcm; v.dVC = st.dvc; v.dVM = st.dvm
                st.n_verts += 1
                v.n_delta = Int32(st.n_delta)
                _bdpt_store_lvc_vertex(v, lvc, lp_idx, st.n_verts - 1)
                # Distant/infinite/point lights store NO lv0 (n_verts starts at
                # 0 for them -- this function's own docstring: they have no
                var eta_v = VcmStrategies.merge_density(sd, v, mis_vm_weight_factor)
                var (dvcm_v, dvc_v, dvm_v, wi_v) = vcm_phase_scatter(
                    -ray_dir, med.med.g, pcg.next_float(), pcg.next_float(),
                    st.dvcm, st.dvc, st.dvm, mis_vc_weight_factor, eta_v)
                st.dvcm = dvcm_v; st.dvc = dvc_v; st.dvm = dvm_v
                st.prev_sig_t = ff.sig_t
                st.mnee.side = Float32(0)
                # Continuation: flux = prev × alb_s (same as stored vertex beta)
                st.flux *= spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (ff.albedo).r, (ff.albedo).g, (ff.albedo).b, st.wl)
                st.rd = vec3f(wi_v)
                st.ro = sp + st.rd*Float32(0.0002)
                st.mis_null_dist = Float32(0)
                return True   # volume free-flight scatter: no vertex stored this bounce, path continues
            else:
                st.flux *= med.weight(ff, t_hit, st.wl)
                # Same missing free-flight factor on dVCM as the camera path --
                # see _bdpt_camera_path_bounce's matching comment for the
                # derivation and the measured error growth with optical depth.
                var ff_exp_l = -log(max(ff.pdf, Float32(1e-30)))   # optical depth of the density actually sampled from (see FreeFlight.pdf)
                if ff_exp_l > Float32(60.0): ff_exp_l = Float32(60.0)
                st.dvcm *= exp(ff_exp_l)
                # dVC's ff_a/ff_b ratio is sigma_t only when the PREVIOUS
                # vertex was a volume scatter (kind-changing edge, exiting
                st.dvc *= med.ff_ratio(st.prev_sig_t > Float32(0), st.prev_sig_t, t_hit, ff, st.wl)
                st.prev_sig_t = Float32(0)

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[unsafe_offset=mat_idx]
        var hit = st.ro + st.rd*t_hit
        var eta_x = mis_vm_weight_factor * _vcm_eta_scale(sd, hit)   # merging's MIS density HERE

        resolve_mix(sd.materials, pcg, mat, mat_idx)
        if mat.type == MatKind.hair:   # the fibre kernel's density (_vcm_fibre_scale)
            eta_x = mis_vm_weight_factor * _vcm_fibre_scale(sd, hit, Int(inter.primId.id1), inter.v)
        if mat.type != MatKind.interface:
            st.mis_null_dist = Float32(0)

        var walks = False
        var dvcm_in = st.dvcm; var dvc_in = st.dvc; var dvm_in = st.dvm; var med_in = st.cur_med_idx
        var entry_eta = Float32(0)
        comptime if not _VCM_SSS_CAMERA_ONLY:
            if sss_walks_below(mat, inter, med_in):
                entry_eta = _vcm_skin_arrival(sd, inter, mat, hit, ray_dir, mis_vm_weight_factor, pcg, lvc, lp_idx, st,
                                              dvcm_in, dvc_in, dvm_in)
                if st.n_verts >= _BDPT_MAX_VERTS:
                    return False   # the arrival took the last slot: whatever follows is past the cache
        if mat.uses_lobe():
            if not mat.has_lobe():
                return False   # e.g. a measured table that failed to load
            var lh = _vcm_lobe_vertex[use_gpu](sd, inter, mat, mat_idx, hit, ray_dir, Float32(-1.0), st.flux, st.cur_med_idx,
                                               st.wl, st.dvcm, st.dvc, st.dvm)
            var v = lh.v
            var gn_geo = lh.gn_geo; var spawn_eps = lh.spawn_eps; var on_curve = lh.on_curve
            if st.mnee.side > Float32(0) and st.mnee.n_glass > Int32(0) and v.is_delta == Int32(0) \
               and LobeKind.is_diffuse(v.mat_kind):
                # Reached through glass only: MNEE from here samples this path too (_vcm_mnee_light_dvcm).
                var d_m = _vcm_mnee_light_dvcm(sd, hit.to_simd(), gn_geo, st.mnee.y, st.mnee.ny, st.mnee.side,
                                               Int(st.mnee.n_glass), st.mnee.first_glass, st.cur_med_idx, st.wl, pcg)
                if d_m > Float32(0):
                    st.dvcm = d_m
            elif st.mnee.side > Float32(0) and st.mnee.n_glass > Int32(0) and v.is_delta == Int32(0) and _vcm_gsms_on(sd, v):
                # Reached through a chain only: glossy SMS from here samples it too.
                var d_g = _vcm_gsms_light_dvcm(sd, hit.to_simd(), gn_geo, st.mnee.y, st.mnee.ny, st.mnee.side,
                                               Int(st.mnee.n_glass), -ray_dir, st.cur_med_idx, st.wl)
                if d_g > Float32(0):
                    st.dvcm = d_g
                    v.hair_curve_idx = VCM_GSMS_ARRIVAL
            v.dVCM = st.dvcm; v.dVC = st.dvc; v.dVM = st.dvm
            if v.is_delta != Int32(0):
                st.n_delta += 1
            if v.is_delta == Int32(0):
                st.n_verts += 1
                v.n_delta = Int32(st.n_delta)
                _bdpt_store_lvc_vertex(v, lvc, lp_idx, st.n_verts - 1)
            var c0 = (st.dvcm, st.dvc, st.dvm)
            var sc = _vcm_scatter(v, True, pcg, sd, st.wl, st.dvcm, st.dvc, st.dvm, mis_vc_weight_factor, eta_x)
            if not sc.valid:
                return False
            if v.hair_curve_idx == VCM_GSMS_ARRIVAL and not sc.is_delta and sc.pdf_rev < GLOSSY_SMS_MIN_PDF:
                # glossy SMS's dVCM counts only where x0's lobe toward the chain reaches GLOSSY_SMS_MIN_PDF
                (st.dvcm, st.dvc, st.dvm) = vcm_scatter_carries(Float32(0), c0[1], c0[2], sc.cos_out / sc.pdf_fwd,
                    sc.pdf_fwd, sc.pdf_rev, mis_vc_weight_factor, eta_x)
            walks = sc.transmit and sss_walks_below(mat, inter, med_in)
            if not walks:
                st.mnee.side = Float32(0)
            st.rd = vec3f(sc.wi)
            if on_curve:
                st.ro = hit + vec3f(gn_geo) * (spawn_eps if dot(sc.wi, gn_geo) >= Float32(0) else -spawn_eps)
            else:
                st.ro = hit + st.rd*Float32(0.0002)
            st.flux *= sc.weight

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            var ior_cur = st.current_dielectric_ior; var ior_prev = st.previous_dielectric_ior
            _ = _vcm_dielectric_cross[use_gpu, False](sd, inter, mat, hit, ray_dir, Float32(-1.0), has_med, pcg,
                    st.ro, st.rd, st.current_dielectric_ior, st.previous_dielectric_ior, st.cur_med_idx, st.dvcm, st.dvc, st.dvm)
            if sss_entered(mat, inter, ray_dir, st.rd, med_in, sd, hit.to_simd()):
                st.cur_med_idx = med_in
                st.current_dielectric_ior = ior_cur; st.previous_dielectric_ior = ior_prev
                walks = True
            else:
                st.n_delta += 1
                if st.mnee.n_glass == Int32(0):
                    st.mnee.first_glass = hit.to_simd()
                st.mnee.n_glass += 1

        elif mat.passes_rays():
            if has_med:
                st.cur_med_idx = medium_after(ray_dir, st.cur_med_idx, inter, mat, sd, hit.to_simd())
            st.ro = hit + st.rd*Float32(0.0002)
            if d_seg > Float32(0):
                st.dvcm /= max(d_seg * d_seg, Float32(1e-20))
                st.mis_null_dist += t_hit + Float32(0.0002)
            # VCM Stage 2b: pure medium-boundary pass-through, no direction
            # change/BSDF event -- carry state passes through as already

        else:
            return False   # unhandled material type

        if walks:
            comptime if _VCM_SSS_CAMERA_ONLY:
                return False
            comptime if defer_walk:
                _vcm_sss_walk_suspend(sd, inter, mat, mat_idx, hit, ray_dir, st.rd, st.wl, pcg, st.current_dielectric_ior,
                                      dvcm_in, dvc_in, dvm_in, entry_eta, walkers, walk_resume, lp_idx)
                return True
            var ex = _vcm_sss_walk[use_gpu, adjoint=True](sd, inter, mat, hit, ray_dir, st.rd, st.wl, mis_vc_weight_factor, pcg, st.flux,
                                            st.dvcm, st.dvc, st.dvm, dvcm_in, dvc_in, dvm_in, st.current_dielectric_ior, entry_eta)
            return _vcm_light_walk_exit(sd, ex, pcg, lvc, lp_idx, mis_vc_weight_factor, st)

        return True   # bounce processed normally, path continues


def _vcm_light_walk_exit(ref sd: SceneView, ex: SssExit, mut pcg: PCG32, lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
                         lp_idx: Int, mis_vc_weight_factor: Float32, mut st: VCMLightPathState) -> Bool:
    """The light bounce after a walk: its exit vertex stored and the scatter out; False ends the path."""
    if not ex.ok or st.n_verts >= _BDPT_MAX_VERTS:   # rough skin's entry may have taken the last slot
        return False
    var v = _vcm_sss_exit_vertex(ex, ex.eta, st.flux, st.cur_med_idx, st.wl, st.dvcm, st.dvc, st.dvm)
    st.n_verts += 1
    v.n_delta = Int32(st.n_delta)
    _bdpt_store_lvc_vertex(v, lvc, lp_idx, st.n_verts - 1)
    _ = _vcm_sss_exit_scatter(sd, ex, ex.eta, mis_vc_weight_factor, Float32(0), pcg,   # a light exit does not merge
                              st.ro, st.rd, st.flux, st.dvcm, st.dvc, st.dvm)
    st.mnee.side = Float32(0)
    return True

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
):
    """Emit a photon from a random light and trace a light subpath, storing
    every non-delta vertex (including the light-source point itself, the"""

    # SHARED WITH THE WAVEFRONT GPU DRIVER. This body used to be a
    # byte-for-byte copy of _bdpt_light_path_init + _bdpt_light_path_bounce
    var st = _bdpt_light_path_init[use_gpu](
        sd, pcg, default_emit_med, lp_idx, lvc, lvc_path_len, mis_vc_weight_factor, pass_wl)
    if st.active == Int8(0):
        return

    for _ in range(_vcm_walk_limit(sd.vcmMaxDepth)):
        # The same intersect step _bdpt_light_path_intersect_gpu performs --
        # kept here rather than inside the bounce because this one step is
        # the eventual Vulkan RT swap point on the wavefront side.
        var ray = Ray(st.ro, st.rd)
        scratch[unsafe_offset=0].hit = Int8(0)
        traverse_dispatch(sd.bvh4, sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, scratch)
        # A miss is handled inside the step (returns False).
        if not _bdpt_light_path_bounce[use_gpu](
            sd, pcg, has_med, scratch[unsafe_offset=0], lvc, lp_idx,
            mis_vc_weight_factor, mis_vm_weight_factor, st):
            break

    lvc_path_len[unsafe_offset=lp_idx] = Int32(st.n_verts)
    return
