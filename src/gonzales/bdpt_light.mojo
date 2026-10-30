# BDPT/VCM light subpath: state, init and the per-bounce step.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.math import sqrt, cos, sin, log, exp, max
from .gpu_media import medium_after
from .geometry import Point3f, Point2f, Vec3f, vec3f, point3f, dot, PI, INV_FOUR_PI
from .materials import MatKind, LobeKind, resolve_mix
from .primitives import Ray, Intersection
from .media import sample_free_flight, spectral_free_flight_weight
from .vcm_mis import vcm_scatter_carries
from .vcm_camis import (
    CamisLightRecord, CamisLightCarry, camis_light_carry_off, camis_light_origin, camis_light_arrive,
    camis_light_scatter,
)
from .bvh import traverse_dispatch
from .bvh import SceneView, test_spheres, _scene_bounding_sphere, _sample_disk_perpendicular, _sample_infinite_light_dir, sphere_light_cone_pdf, light_path_pick
from .rng import PCG32
from .sppm import _cosine_hemisphere_sample, sample_area_light_point, sample_sphere_light_emission, area_light_emit_normal, area_light_side_pdf
from .spectrum import SampledWavelengths, SpectralSample, spec_refl, spec_illum
from .bdpt_vertex import _vcm_walk_limit, _BDPT_MAX_VERTS, BDPTVertex, _null_vertex
from .bdpt_eval import _bdpt_vertex_mis_scoped, _vcm_mnee_light_dvcm
from .vcm_grid import _vcm_depth, _VCM_MAX_STORED, _vcm_keep, _VCM_CAMIS, _vcm_eta_scale, _vcm_eta_at, _vcm_fibre_scale
from .bdpt_connect import _bdpt_store_lvc_vertex
from .bdpt_camera import (
    _vcm_scatter, _vcm_lobe_vertex, _vcm_bssrdf_hop, _vcm_bssrdf_vertex,
    _vcm_bssrdf_exit_scatter, _vcm_dielectric_cross,
)

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
    var prev_was_volume: Bool   # the previous vertex was a volume scatter: a medium arrival's dVC ratio depends on it
    var mnee: VCMMneeLightTrack

def _null_light_path_state() -> VCMLightPathState:
    var z = Float32(0)
    return VCMLightPathState(
        ro=Point3f(z, z, z), rd=Vec3f(z, z, z), flux=SpectralSample(z), dvcm=z, dvc=z, dvm=z, is_finite_origin=False,
        origin_sphere=Int32(-1), cur_med_idx=Int32(-1), n_verts=0, active=Int8(0), pcg_state=UInt64(0), pcg_inc=UInt64(0),
        wl=SampledWavelengths(z, z, z, z), current_dielectric_ior=Float32(1.0), previous_dielectric_ior=Float32(1.0),
        n_delta=0, mis_null_dist=z, prev_was_volume=False, mnee=vcm_mnee_light_track_none(),
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
        comptime if _VCM_CAMIS:
            dvm_carry = dvc_carry   # dVC0 == dVC at an origin, see the area branch
        else:
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
        n_delta=0, mis_null_dist=Float32(0), prev_was_volume=False, mnee=mnee_l,
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
    mut st: VCMLightPathState,
    # _VCM_CAMIS light state (vcm_camis.CamisLightCarry) and THIS path's
    # slice of the record buffer (lvc_camis + lp_idx * _BDPT_MAX_VERTS, the
    # same slots as `lvc`). Untouched when the hybrid is compiled out.
    mut camis_l: CamisLightCarry,
    camis_recs: Pointer[CamisLightRecord, MutUntrackedOrigin],
) -> Bool:
    """One bounce of a light subpath, shared by the per-path loop and the wavefront kernels; False ends the path."""
        if st.n_verts >= _BDPT_MAX_VERTS:
            return False   # mirrors the original loop's top-of-iteration guard
        # The next vertex stored would be interior vertex n_verts (after an area
        # light's origin slot) or n_verts + 1 (no origin slot): past d it starts
        # no strategy (see _vcm_depth).
        var next_count = st.n_verts
        if st.n_verts == 0 or lvc[unsafe_offset=lp_idx * _BDPT_MAX_VERTS].is_light != Int32(1):
            next_count = st.n_verts + 1
        if next_count + st.n_delta > _vcm_depth(sd) or next_count > _VCM_MAX_STORED:
            return False
        if inter.hit == Int8(0):
            return False   # nothing hit -- path escapes the scene
        var t_hit = inter.tHit
        var ray_dir = st.rd.to_simd()
        var d_seg = Float32(0)   # the d^2 applied to dVCM at this hit, 0 if none

        # VCM Stage 2b: distance-squared portion of the per-bounce MIS
        # correction (project_vcm_stage2_mis_derivation memory) -- shared
        if st.origin_sphere >= Int32(0):
            # First hit of a sphere light's path: the origin's direct density
            # is the camera NEE's cone pdf seen from HERE, and its area
            st.dvcm *= sphere_light_cone_pdf(sd.spheres[unsafe_offset=Int(st.origin_sphere)],
                                                st.ro.to_simd() + ray_dir * t_hit)
            st.origin_sphere = Int32(-2)   # applied; still a sphere light's path
        elif st.n_verts >= 1 or st.is_finite_origin:
            d_seg = t_hit + st.mis_null_dist
            st.dvcm *= d_seg * d_seg

        # Volume free-flight
        if has_med and Int(st.cur_med_idx) >= 0:
            comptime if _VCM_CAMIS:
                camis_l.in_class = False   # a medium segment: see the camera side
            var med = sd.mediums[unsafe_offset=Int(st.cur_med_idx)]
            # ONE shared sampler for both medium kinds (geometry.mojo):
            # homogeneous closed form, or delta tracking against the real
            var ff = sample_free_flight(
                med, sd.grids, sd.nvdbGrids, Vec3f(st.ro.x, st.ro.y, st.ro.z), st.rd, t_hit, pcg)
            if ff.collided and st.origin_sphere != Int32(-1):
                # A sphere light's path ends at its first medium collision.
                # Medium vertices are outside per-vertex MIS: the camera's
                return False
            if ff.collided:
                # Chromatic collision weight; see the camera-side comment.
                st.flux *= spectral_free_flight_weight(med, ff, t_hit, st.wl, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                var sp = st.ro + st.rd*ff.t_free
                # dVCM's Jacobian above was applied with t_hit (the distance
                # to the SURFACE the ray was cast toward), computed before we
                st.dvcm *= ((ff.t_free + st.mis_null_dist) * (ff.t_free + st.mis_null_dist)) / max(d_seg * d_seg, Float32(1e-20))
                # Free-flight at this VOLUME arrival: ff_b = the collision
                # density actually sampled = ff.pdf. No cos_fix -- a volume
                # vertex has no surface normal to divide by.
                st.dvcm *= Float32(1.0) / max(ff.pdf, Float32(1e-30))
                # dVC's ff_a/ff_b ratio is 1/sigma_t only when the PREVIOUS
                # vertex was a surface (kind-changing edge); a volume->volume
                if not st.prev_was_volume:
                    st.dvc *= Float32(1.0) / max(ff.sig_t, Float32(1e-20))
                var v = _null_vertex()
                v.pos = sp
                # `flux`, NOT flux*albedo -- see the matching camera-side
                # comment in _bdpt_camera_path_bounce: a stored vertex's
                v.beta = st.flux
                v.alb = ff.albedo
                v.is_surface = Int32(0); v.is_delta = Int32(0)
                v.pdf_fwd = ff.pdf   # the density actually sampled from; under hero-wavelength MIS this is a lane MIXTURE, not sig_t's lone exponential
                v.med_idx = st.cur_med_idx
                v.wavelengths = st.wl
                # ARRIVAL carries, before the outgoing scatter step below
                # overwrites them -- the same v.dVCM/dVC/dVM = ... pattern
                v.dVCM = st.dvcm; v.dVC = st.dvc; v.dVM = st.dvm
                st.n_verts += 1
                v.n_delta = Int32(st.n_delta)
                _bdpt_store_lvc_vertex(v, lvc, lp_idx, st.n_verts - 1)
                comptime if _VCM_CAMIS:
                    # Every stored slot gets a record, out-of-Class ones too:
                    # the buffer outlives the pass, and S3 reads the class bit.
                    camis_light_arrive(camis_l, camis_recs, st.n_verts - 1, Float32(0), t_hit, Float32(0), False)
                # Distant/infinite/point lights store NO lv0 (n_verts starts at
                # 0 for them -- this function's own docstring: they have no
                var eta_v = _vcm_eta_at(sd, v, mis_vm_weight_factor)
                (st.dvcm, st.dvc, st.dvm) = vcm_scatter_carries[dvc0=_VCM_CAMIS](
                    st.dvcm, st.dvc, st.dvm,
                    Float32(4.0) * PI, INV_FOUR_PI, INV_FOUR_PI,
                    mis_vc_weight_factor, eta_v)
                st.prev_was_volume = True
                st.mnee.side = Float32(0)
                # Continuation: flux = prev × alb_s (same as stored vertex beta)
                st.flux *= spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (ff.albedo).r, (ff.albedo).g, (ff.albedo).b, st.wl)
                var u1 = pcg.next_float(); var u2 = pcg.next_float()
                var cosT = Float32(2)*u1 - Float32(1)
                var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                var phi  = Float32(2)*PI*u2
                st.rd = Vec3f(sinT*cos(phi), sinT*sin(phi), cosT)
                st.ro = sp + st.rd*Float32(0.0002)
                st.mis_null_dist = Float32(0)
                return True   # volume free-flight scatter: no vertex stored this bounce, path continues
            else:
                st.flux *= spectral_free_flight_weight(med, ff, t_hit, st.wl, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                # Same missing free-flight factor on dVCM as the camera path --
                # see _bdpt_camera_path_bounce's matching comment for the
                # derivation and the measured error growth with optical depth.
                var ff_exp_l = -log(max(ff.pdf, Float32(1e-30)))   # optical depth of the density actually sampled from (see FreeFlight.pdf)
                if ff_exp_l > Float32(60.0): ff_exp_l = Float32(60.0)
                st.dvcm *= exp(ff_exp_l)
                # dVC's ff_a/ff_b ratio is sigma_t only when the PREVIOUS
                # vertex was a volume scatter (kind-changing edge, exiting
                if st.prev_was_volume:
                    st.dvc *= ff.sig_t
                st.prev_was_volume = False

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[unsafe_offset=mat_idx]
        var hit = st.ro + st.rd*t_hit
        var eta_x = mis_vm_weight_factor * _vcm_eta_scale(sd, hit)   # merging's MIS density HERE

        resolve_mix(sd.materials, pcg, mat, mat_idx)
        if mat.type == MatKind.hair:   # the fibre kernel's density (_vcm_fibre_scale)
            eta_x = mis_vm_weight_factor * _vcm_fibre_scale(sd, hit, Int(inter.primId.id1), inter.v)
        if mat.type != MatKind.interface:
            st.mis_null_dist = Float32(0)

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
                                               Int(st.mnee.n_glass), st.mnee.first_glass, pcg)
                if d_m > Float32(0):
                    st.dvcm = d_m
            v.dVCM = st.dvcm; v.dVC = st.dvc; v.dVM = st.dvm
            if v.is_delta != Int32(0):
                st.n_delta += 1
            if v.is_delta == Int32(0):
                st.n_verts += 1
                v.n_delta = Int32(st.n_delta)
                _bdpt_store_lvc_vertex(v, lvc, lp_idx, st.n_verts - 1)
                comptime if _VCM_CAMIS:
                    camis_light_arrive(camis_l, camis_recs, st.n_verts - 1, lh.cos_fix, t_hit,
                                       log(_vcm_keep(sd, hit)), _bdpt_vertex_mis_scoped(v))
            var sc = _vcm_scatter(v, True, pcg, sd, st.wl, st.dvcm, st.dvc, st.dvm, mis_vc_weight_factor, eta_x)
            st.mnee.side = Float32(0)
            if not sc.valid:
                return False
            comptime if _VCM_CAMIS:
                if v.is_delta != Int32(0) or sc.is_delta:
                    camis_l.in_class = False
                else:
                    camis_light_scatter(camis_l, st.n_verts - 1, eta_x, sc.cos_out / sc.pdf_fwd,
                                        sc.cos_out, sc.pdf_fwd, sc.pdf_rev)
            st.rd = vec3f(sc.wi)
            if on_curve:
                st.ro = hit + vec3f(gn_geo) * (spawn_eps if dot(sc.wi, gn_geo) >= Float32(0) else -spawn_eps)
            else:
                st.ro = hit + st.rd*Float32(0.0002)
            st.flux *= sc.weight

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            comptime if _VCM_CAMIS:
                camis_l.in_class = False   # BSSRDF hop or delta dielectric: see the camera side
            # ── Subsurface boundary: the light subpath's half of the hop ───
            # Mirror of the camera side: the light enters at x_i, the exit x_o
            # is drawn by the SAME sampler (the MIS weights require identical
            var did_bssrdf_hop = False
            if mat.sss_boundary != Int8(0) and Int(st.cur_med_idx) < 0 and has_med:
                var med_in = medium_after(ray_dir, Int32(-1), inter, mat, sd, hit.to_simd())
                if med_in >= Int32(0) and Int(med_in) < Int(sd.mediumCount):
                    var ex = _vcm_bssrdf_hop(sd, inter, mat, hit, ray_dir, Int(med_in), mis_vc_weight_factor, pcg, st.wl,
                                             st.flux, st.dvcm, st.dvc, st.dvm)
                    if not ex.ok:
                        return False
                    var eta_e = mat.albedo.r
                    var v = _vcm_bssrdf_vertex(ex, eta_e, st.flux, st.cur_med_idx, st.wl, st.dvcm, st.dvc, st.dvm)
                    st.n_verts += 1
                    v.n_delta = Int32(st.n_delta)
                    _bdpt_store_lvc_vertex(v, lvc, lp_idx, st.n_verts - 1)
                    comptime if _VCM_CAMIS:
                        camis_light_arrive(camis_l, camis_recs, st.n_verts - 1, Float32(0), t_hit, Float32(0), False)
                    _ = _vcm_bssrdf_exit_scatter(sd, ex, eta_e, mis_vc_weight_factor, mis_vm_weight_factor, pcg,
                                                 st.ro, st.rd, st.flux, st.dvcm, st.dvc, st.dvm)
                    did_bssrdf_hop = True
                    st.mnee.side = Float32(0)
            if not did_bssrdf_hop:
                st.n_delta += 1
                if st.mnee.n_glass == Int32(0):
                    st.mnee.first_glass = hit.to_simd()
                st.mnee.n_glass += 1
                _ = _vcm_dielectric_cross[use_gpu, False](sd, inter, mat, hit, ray_dir, Float32(-1.0), has_med, pcg,
                    st.ro, st.rd, st.current_dielectric_ior, st.previous_dielectric_ior, st.cur_med_idx, st.dvcm, st.dvc, st.dvm)

        elif mat.passes_rays():
            if has_med:
                st.cur_med_idx = medium_after(ray_dir, st.cur_med_idx, inter, mat, sd, hit.to_simd())
            st.ro = hit + st.rd*Float32(0.0002)
            if d_seg > Float32(0):
                st.dvcm /= max(d_seg * d_seg, Float32(1e-20))
                st.mis_null_dist += t_hit + Float32(0.0002)
            # VCM Stage 2b: pure medium-boundary pass-through, no direction
            # change/BSDF event -- carry state passes through as already
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
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin],
):
    """Emit a photon from a random light and trace a light subpath, storing
    every non-delta vertex (including the light-source point itself, the"""

    # SHARED WITH THE WAVEFRONT GPU DRIVER. This body used to be a
    # byte-for-byte copy of _bdpt_light_path_init + _bdpt_light_path_bounce
    var st = _bdpt_light_path_init[use_gpu](
        sd, pcg, default_emit_med, lp_idx, lvc, lvc_path_len, mis_vc_weight_factor, pass_wl)
    if st.active == Int8(0):
        return
    # CAMIS light state. Only an AREA light stores an origin vertex (n_verts
    # starts at 1), and only an area light starts a path inside the Class;
    # every other light's first stored vertex writes an out-of-Class record.
    var camis_l = camis_light_carry_off()
    var camis_recs: Pointer[CamisLightRecord, MutUntrackedOrigin]
    comptime if _VCM_CAMIS:
        camis_recs = lvc_camis.unsafe_offset(lp_idx * _BDPT_MAX_VERTS)
        var lv0 = lvc[unsafe_offset=lp_idx * _BDPT_MAX_VERTS]
        if st.n_verts == 1 and lv0.is_light == Int32(1):
            # lv0.pdf_fwd is the origin's position density P_A (init, area
            # branch), and the emission cosine is clamped as init clamps it.
            camis_l = camis_light_origin(camis_recs, 0, log(lv0.pdf_fwd),
                                         max(dot(st.rd.to_simd(), lv0.normal.to_simd()), Float32(0.0001)))
    else:
        camis_recs = lvc_camis   # never dereferenced

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
            mis_vc_weight_factor, mis_vm_weight_factor, st, camis_l, camis_recs):
            break

    lvc_path_len[unsafe_offset=lp_idx] = Int32(st.n_verts)
    return
