# BDPT/VCM camera subpath: state, init and the per-bounce step.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.collections import Array
from std.math import sqrt, cos, sin, log, exp, max, abs
from .gpu_media import medium_after
from .geometry import offset_eps, RGB, Point3f, Point2f, Vec3f, vec3f, point3f, dot, cross, PI, INV_FOUR_PI, INV_PI
from .render_state import PDF_DELTA_FULL, PDF_DROP_DIRECT
from .materials import Material, MatKind, LobeKind, resolve_mix
from .primitives import Ray, Intersection, mesh_pos
from .media import sample_free_flight, spectral_free_flight_weight
from .bssrdf import bssrdf_exit_ft
from .vcm_mis import (
    vcm_arrival_carries, vcm_scatter_carries, bssrdf_hop_carries, bssrdf_exit_scatter_carries,
    vcm_env_escape_weight,
)
from .vcm_camis import (
    CamisCamRecord, CamisCamCarry, camis_cam_carry_init, camis_cam_arrive, camis_cam_scatter,
    CamisLightRecord, camis_clamp_log_p, camis_eval_emission_hit,
)
from .bvh import traverse_dispatch
from .bvh import SceneView, test_spheres, _scene_bounding_sphere, _eval_infinite_light_and_pdf, sphere_light_cone_pdf, _sample_infinite_light_nee, light_path_pick_pdf
from .sampling import power_heuristic, camera_ray_from_film_xy, FilmFilter, film_filter_offset, ThinLens
from .rng import PCG32
from .sppm import _geom_normal, _shading_normal_at, _dielectric_bounce, _cosine_hemisphere_sample, area_light_side_pdf
from .shading import _get_tri_verts, apply_surface_maps_at_hit, area_light_hit_cos, curve_light_hit
from .bxdf import BSDF, LobeSample, LobeTables
from .spectrum import (
    SampledWavelengths, SpectralSample, spec_refl, spec_refl_unbounded, spec_illum,
    rgb_bands_to_spectral_sample,
)
from .bdpt_vertex import _VOL_PHASE_HIT, _BDPT_MAX_DEPTH, _BDPT_MAX_VERTS, BDPTVertex, _null_vertex
from .bdpt_nee import _bdpt_simple_light_count, _bdpt_sample_simple_light, _bdpt_nee_contribute
from .bdpt_eval import (
    _vcm_sphere_slot, _vcm_infinite_slot, _vcm_nee_surface, _vertex_ctx, _bdpt_vertex_mis_scoped, _vcm_mnee,
    _vcm_mnee_hit_term,
)
from .vcm_grid import _vcm_depth, _vcm_keep, _VCM_CAMIS, _CAMIS_CAM_RECS, _CAMIS_FORCE_C1, _vcm_eta_scale, _vcm_eta_at, _vcm_fibre_scale
from .bdpt_connect import _bdpt_connect_to_cache, _bdpt_connect_to_cache_deferred
from .vcm_merge import _bdpt_merge_from_cache
from .bdpt_bssrdf import _bdpt_sample_bssrdf_exit, BssrdfExitSample
from .surface import surface_hit

def _bdpt_trace_camera_and_connect[use_gpu: Bool](
    r2c:     Pointer[Float32, MutUntrackedOrigin],
    c2w:     Pointer[Float32, MutUntrackedOrigin],
    px:      Int, py:      Int,
    ref sd:      SceneView,
    mut pcg: PCG32,
    has_med: Bool,
    scratch: Pointer[Intersection, MutUntrackedOrigin],
    lvc:     Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx:  Int,
    path_len: Int,
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    merge_heads: Pointer[Int32, MutUntrackedOrigin],
    merge_inv_cell: Float32,
    merge_r2: Float32,
    merge_norm: Float32,
    px_scale: Float32,
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    n_light_paths_f: Float32,
    pass_wl: SampledWavelengths,
    film_filter: FilmFilter,
    start_med_idx: Int32 = Int32(-1),
    # _VCM_CAMIS: the light subpaths' CAMIS records, parallel to `lvc`
    # (vcm_camis.CamisLightRecord). Not read yet -- stage S3's weight sites
    # will; plumbed now so S3 changes no signature.
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin] = Pointer[CamisLightRecord, MutUntrackedOrigin].unsafe_dangling(),
) -> Tuple[SpectralSample, SpectralSample, RGB, Int32, Int32, Int32]:
    """Trace one camera subpath from pixel (px,py). At each non-delta vertex,
    connect inline/synchronously to the shared Light Vertex Cache via"""

    # SHARED WITH THE WAVEFRONT GPU DRIVER -- see _bdpt_trace_light_path's
    # matching comment. This body was a byte-for-byte copy of
    var st = _bdpt_camera_path_init[use_gpu](
        r2c, c2w, px, py, pcg, px_scale, n_light_paths_f, pass_wl, film_filter, start_med_idx, sd.lens)
    var visit_naive = Int32(0)
    var visit_footprint = Int32(0)
    var visit_thin = Int32(0)
    if st.active == Int8(0):
        return (st.total, st.total_merge, st.first_alb, visit_naive, visit_footprint, visit_thin)
    # CAMIS camera state (vcm_camis.mojo): live across the whole bounce loop,
    # the same `mut` threading dvcm_carry uses. One element and never touched
    # when _VCM_CAMIS is off.
    var camis = camis_cam_carry_init()
    var camis_recs = Array[CamisCamRecord, _CAMIS_CAM_RECS](
        fill=CamisCamRecord(Float32(0), Float32(0), Float32(0), Float32(0), Float32(0)))

    for _ in range(_BDPT_MAX_DEPTH):
        # The same intersect step _bdpt_camera_path_intersect_gpu performs;
        # kept at the call site because it is the Vulkan RT swap point.
        var ray = Ray(st.ro, st.rd)
        scratch[unsafe_offset=0].hit = Int8(0)
        traverse_dispatch(sd.bvh4, sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, scratch)
        # A miss (including its infinite-light escape credit) is handled
        # inside the step, so the old top-of-loop miss block is gone.
        if not _bdpt_camera_path_bounce[use_gpu](
            sd, pcg, has_med, scratch[unsafe_offset=0], scratch, lvc, lp_idx, path_len,
            merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm,
            mis_vc_weight_factor, mis_vm_weight_factor,
            st, visit_naive, visit_footprint, visit_thin, camis, camis_recs,
            lvc_camis=lvc_camis, n_light_paths_f=n_light_paths_f):
            break

    return (st.total, st.total_merge, st.first_alb, visit_naive, visit_footprint, visit_thin)

@fieldwise_init
struct VCMMneeTrack(TrivialRegisterPassable):
    """Where a camera path's glass chain began, for MNEE's share of an emitter hit at its end."""
    var x0: Vec3f        # the last stored surface vertex
    var pdf0: Float32    # its lobe's density for the direction taken; 0: MNEE does not start there
    var n_glass: Int32   # dielectric vertices since


@always_inline
def vcm_mnee_track_none() -> VCMMneeTrack:
    return VCMMneeTrack(Vec3f(Float32(0)), Float32(0), Int32(0))


@fieldwise_init
struct VCMCameraPathState(TrivialRegisterPassable):
    """Task #163 stage 4: persistent per-camera-path state carried across
    separate wavefront-staged GPU kernel launches (`_bdpt_camera_path_init`"""
    var ro: Point3f
    var rd: Vec3f
    var beta: SpectralSample
    var total: SpectralSample
    # Vertex-MERGING contribution only, tracked separately from `total`
    # (which after this split holds connect + t=1 light-tracing splats) --
    var total_merge: SpectralSample
    var first_alb: RGB
    var dvcm: Float32
    var dvc: Float32
    var dvm: Float32
    var n_verts: Int
    var cur_med_idx: Int32
    var last_bsdf_pdf: Float32
    var active: Int8
    var pcg_state: UInt64
    var pcg_inc: UInt64
    var wl: SampledWavelengths
    # Distance travelled through null interfaces since the last real
    # scattering event -- same role as PathState.mis_null_dist. The
    var mis_null_dist: Float32
    # Touching-dielectric IOR depth-2 stack for _dielectric_bounce (see that
    # function's docstring, sppm.mojo) -- same role and convention as
    var current_dielectric_ior: Float32
    var previous_dielectric_ior: Float32
    # Specular-chain path length for the texture/bump footprint, see
    # _bdpt_camera_path_bounce's `cone_len`.
    var cone_len: Float32
    var n_delta: Int
    var prev_was_volume: Bool   # the previous vertex was a volume scatter (see the light subpath's)
    var mnee: VCMMneeTrack

def _bdpt_camera_path_init[use_gpu: Bool](
    r2c:     Pointer[Float32, MutUntrackedOrigin],
    c2w:     Pointer[Float32, MutUntrackedOrigin],
    px:      Int, py:      Int,
    mut pcg: PCG32,
    px_scale: Float32,
    n_light_paths_f: Float32,
    pass_wl: SampledWavelengths,
    film_filter: FilmFilter,
    start_med_idx: Int32 = Int32(-1),
    lens: ThinLens = ThinLens.pinhole(),
) -> VCMCameraPathState:
    """A camera subpath's start: the primary ray and its MIS carries."""
    # JITTER. This was `px + 0.5` -- the pixel CENTRE, identically for every
    # sample. VCM was the only integrator that did not jitter: the path tracer
    var u_fx = pcg.next_float()
    var u_fy = pcg.next_float()
    var (dfx, dfy) = film_filter_offset(u_fx, u_fy, film_filter)
    var fX = Float32(px) + Float32(0.5) + dfx
    var fY = Float32(py) + Float32(0.5) + dfy
    var ul0 = Float32(0.5); var ul1 = Float32(0.5)
    if lens.radius > Float32(0):
        ul0 = pcg.next_float(); ul1 = pcg.next_float()
    var (rd, ro, _cl) = camera_ray_from_film_xy(fX, fY, r2c, c2w, lens, ul0, ul1)

    # VCM Stage 2b: real per-vertex MIS state for the eye subpath (see
    # project_vcm_stage2_mis_derivation memory). cameraPdfW derived by
    var cos_theta_at_camera = abs(
        rd.x * c2w[unsafe_offset=8] + rd.y * c2w[unsafe_offset=9] + rd.z * c2w[unsafe_offset=10])
    var camera_pdf_w = Float32(1) / max(
        px_scale * px_scale * cos_theta_at_camera * cos_theta_at_camera * cos_theta_at_camera,
        Float32(1e-12))
    var dvcm_carry = n_light_paths_f / camera_pdf_w
    var dvc_carry = Float32(0)
    var dvm_carry = Float32(0)

    var n_verts = 0
    var beta = SpectralSample(Float32(1))
    var cur_med_idx = start_med_idx
    var total = SpectralSample(Float32(0))
    var total_merge = SpectralSample(Float32(0))
    var first_alb = RGB(Float32(0))  # denoiser albedo AOV -- set at the first stored vertex, below
    # ONE hero-wavelength set per spp PASS, shared by every camera AND light
    # subpath in it, rather than one per subpath. This is forced by the flip
    var wavelengths = pass_wl
    # pdf (solid angle) of the cosine-weighted diffuse bounce that produced
    # the CURRENT `rd`, used to MIS-weight this ray's eventual infinite-light
    var last_bsdf_pdf = Float32(-1)


    return VCMCameraPathState(
        ro=ro, rd=rd, beta=beta, total=total, total_merge=total_merge, first_alb=first_alb,
        dvcm=dvcm_carry, dvc=dvc_carry, dvm=dvm_carry, n_verts=n_verts, cur_med_idx=cur_med_idx,
        last_bsdf_pdf=last_bsdf_pdf, active=Int8(1), pcg_state=pcg.state, pcg_inc=pcg.inc, wl=wavelengths,
        mis_null_dist=Float32(0.0), current_dielectric_ior=Float32(1.0), previous_dielectric_ior=Float32(1.0),   # vacuum
        cone_len=Float32(0.0), n_delta=0, prev_was_volume=False, mnee=vcm_mnee_track_none(),
    )

def _vcm_scatter(
    v: BDPTVertex, adjoint: Bool, mut pcg: PCG32, ref sd: SceneView,
    wavelengths: SampledWavelengths,
    mut dvcm: Float32, mut dvc: Float32, mut dvm: Float32,
    mis_vc_weight_factor: Float32, eta_x: Float32,
) -> LobeSample:
    """Scatter at a stored vertex through THE lobe sampler (bxdf.mojo's
    lobe_sample) and advance VCM's carries. `adjoint` on the light subpath."""
    var s = (_vertex_ctx(v, adjoint)).sample(pcg.next_float(), pcg.next_float(), pcg.next_float(), pcg.next_float(), LobeTables(sd.materials, sd.curves, sd.measuredBrdfs, sd.spectral), wavelengths)
    if not s.valid:
        return s
    if s.is_delta:
        dvcm = Float32(0)
        dvc *= s.cos_out
        dvm *= s.cos_out
    else:
        (dvcm, dvc, dvm) = vcm_scatter_carries[dvc0=_VCM_CAMIS](
            dvcm, dvc, dvm, s.cos_out / s.pdf_fwd, s.pdf_fwd, s.pdf_rev,
            mis_vc_weight_factor, eta_x)
    return s

@always_inline
def _area_light_hit(ref sd: SceneView, inter: Intersection, mat: Material,
                    ray_dir: Vec3f) -> Tuple[RGB, Float32, Int, Float32]:
    """What a ray that hit an area-light triangle or curve reached:
    (emission, total area, its pick slot -- area lights come first, in index"""
    if inter.primId.type == Int8(5):
        var (al_ci, cos_c) = curve_light_hit(inter, sd.curves, sd.areaLights,
                                             Int(sd.areaLightCount), ray_dir)
        var area = Float32(0)
        var slot = 0
        if al_ci >= 0:
            area = sd.areaLights[unsafe_offset=al_ci].total_area
            slot = al_ci
        return (mat.emission, area, slot, cos_c)   # the curve's own emitter slot
    var al_hit = sd.areaLights[unsafe_offset=Int(inter.primId.id1)]
    return (al_hit.emission, al_hit.total_area, Int(inter.primId.id1),
            area_light_hit_cos(inter, sd.meshes, sd.instances, ray_dir))


@always_inline
def _vcm_bssrdf_hop(
    ref sd: SceneView, inter: Intersection, mat: Material, hit: Point3f, ray_dir: Vec3f, med_in: Int, mis_vc: Float32,
    mut pcg: PCG32, wl: SampledWavelengths, mut beta: SpectralSample, mut dvcm: Float32, mut dvc: Float32, mut dvm: Float32,
) -> BssrdfExitSample:
    """A subsurface hop from the entry at `hit` to an exit both subpaths sample alike: arrival, the hop's weight in
    `beta` and its MIS carries."""
    var gn_e = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
    if dot(gn_e, ray_dir) > Float32(0): gn_e = gn_e * Float32(-1)
    var cos_e = abs(dot(-ray_dir, gn_e))
    (dvcm, dvc, dvm) = vcm_arrival_carries(dvcm, dvc, dvm, cos_e)
    var ex = _bdpt_sample_bssrdf_exit(sd, med_in, hit, gn_e, cos_e, mat.albedo.r, pcg)
    if not ex.ok:
        return ex
    ref s = sd.spectral
    beta *= spec_refl_unbounded(s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65, ex.weight.r, ex.weight.g, ex.weight.b, wl)
    (dvcm, dvc, dvm) = bssrdf_hop_carries[dvc0=_VCM_CAMIS](dvcm, dvc, dvm, ex.p_area, cos_e * INV_PI, mis_vc)
    return ex


@always_inline
def _vcm_bssrdf_vertex(ex: BssrdfExitSample, eta: Float32, beta: SpectralSample, med_idx: Int32, wl: SampledWavelengths,
                       dvcm: Float32, dvc: Float32, dvm: Float32) -> BDPTVertex:
    """The subpath vertex at a hop's exit: a diffuse lobe weighted by the exit Fresnel."""
    var v = _null_vertex()
    v.pos = ex.x_o
    v.normal = vec3f(ex.n_o)
    v.shading_normal = vec3f(ex.n_o)
    v.beta = beta
    v.alb = RGB(Float32(1))
    v.is_surface = Int32(1)
    v.mat_kind = LobeKind.bssrdf
    v.pdf_fwd = ex.p_area   # reverse density toward x_i: the hop is symmetric
    v.pdf_bwd = eta
    v.wo = vec3f(ex.n_o)    # unused by LobeKind.bssrdf
    v.med_idx = med_idx
    v.wavelengths = wl
    v.dVCM = dvcm; v.dVC = dvc; v.dVM = dvm
    return v


@always_inline
def _vcm_bssrdf_exit_scatter(
    ref sd: SceneView, ex: BssrdfExitSample, eta: Float32, mis_vc: Float32, mis_vm: Float32, mut pcg: PCG32,
    mut ro: Point3f, mut rd: Vec3f, mut beta: SpectralSample, mut dvcm: Float32, mut dvc: Float32, mut dvm: Float32,
) -> Float32:
    """Leaves a hop's exit along the cosine-sampled exit lobe, weight Ft(cos_out); returns cos_out."""
    var ux1 = pcg.next_float(); var ux2 = pcg.next_float()
    rd = vec3f(_cosine_hemisphere_sample(ex.n_o, ux1, ux2))
    ro = ex.x_o + rd * Float32(0.0002)
    var cos_out = abs(dot(rd.to_simd(), ex.n_o))
    beta *= SpectralSample(bssrdf_exit_ft(cos_out, eta))
    (dvcm, dvc, dvm) = bssrdf_exit_scatter_carries[dvc0=_VCM_CAMIS](
        dvcm, dvc, dvm, ex.p_area, cos_out, mis_vc, mis_vm * _vcm_eta_scale(sd, ex.x_o))
    return cos_out


@always_inline
def _vcm_dielectric_cross[use_gpu: Bool, radiance: Bool](
    ref sd: SceneView, inter: Intersection, mat: Material, hit: Point3f, ray_dir: Vec3f, cone_w: Float32, has_med: Bool,
    mut pcg: PCG32, mut ro: Point3f, mut rd: Vec3f, mut cur_ior: Float32, mut prev_ior: Float32, mut med_idx: Int32,
    mut dvcm: Float32, mut dvc: Float32, mut dvm: Float32,
) -> Float32:
    """A smooth dielectric's reflect-or-refract, the delta MIS update and the medium change; returns the camera path's
    radiance scale (1 for light paths, `radiance` False)."""
    # pbrt's outward normal: interpolated N when the mesh has it, which pbrt also refracts through. Maps apply too:
    # barcelona's water is a dielectric with a displacement texture.
    var gn = _shading_normal_at(inter, sd.meshes, sd.instances, sd.spheres, hit)
    gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes, sd.instances, gn, gn, hit.to_simd(), ray_dir, cone_w,
                                            sd.camFp, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
    var (new_dir, new_org, radiance_scale, new_cur_ior, new_prev_ior) = _dielectric_bounce(
        ray_dir, hit.to_simd(), gn, mat.albedo.r, False, pcg, cur_ior, prev_ior, mat.type == MatKind.thin_dielectric,
        radiance_mode=radiance)
    var n_in_out = cur_ior / new_cur_ior   # refraction compresses solid angle by (n_in/n_out)^2
    cur_ior = new_cur_ior
    prev_ior = new_prev_ior
    if has_med:
        med_idx = medium_after(new_dir, med_idx, inter, mat, sd, hit.to_simd())
    rd = vec3f(new_dir)
    ro = point3f(new_org)
    # Delta vertex, as in SmallVCM: dVCM restarts, dVC and dVM trade the arrival cosine for the departure one.
    var cos_in = abs(dot(-ray_dir, gn))
    if cos_in > Float32(1e-6):
        dvc /= cos_in
        dvm /= cos_in
    var cos_out = abs(dot(new_dir, gn)) * n_in_out * n_in_out
    dvcm = Float32(0)
    dvc *= cos_out
    dvm *= cos_out
    return radiance_scale if radiance else Float32(1)


@fieldwise_init
struct VcmLobeHit(TrivialRegisterPassable):
    """A lobe material's subpath vertex and the hit's normals, arrival cosine, spawn offset and texture albedo."""
    var v: BDPTVertex
    var gn: Vec3f        # shading normal, maps applied
    var gn_geo: Vec3f
    var cos_fix: Float32
    var spawn_eps: Float32
    var on_curve: Bool
    var eff_alb: RGB


@always_inline
def _vcm_lobe_vertex[use_gpu: Bool](
    ref sd: SceneView, inter: Intersection, mat: Material, mat_idx: Int, hit: Point3f, ray_dir: Vec3f, cone_w: Float32,
    beta: SpectralSample, med_idx: Int32, wl: SampledWavelengths, mut dvcm: Float32, mut dvc: Float32, mut dvm: Float32,
) -> VcmLobeHit:
    """The vertex either subpath stores at a lobe material, with the arrival applied to the MIS carries; its dVCM, dVC
    and dVM are the caller's to set. `cone_w` is the camera ray's footprint width for texture filtering, -1 for none."""
    var h = surface_hit[use_gpu](sd, inter, mat, hit, ray_dir, cone_w)
    (dvcm, dvc, dvm) = vcm_arrival_carries(dvcm, dvc, dvm, h.cos_in)
    var v = _null_vertex()
    v.pos = hit
    v.normal = vec3f(h.gn_geo)
    v.shading_normal = vec3f(h.gn)
    v.beta = beta
    v.alb = h.alb
    v.is_surface = Int32(1)
    # One lobe per material: everything after evaluates and samples the vertex through its BSDF.
    v.mat_kind = mat.lobe()
    v.mat_idx = Int32(mat_idx)
    v.pdf_bwd = mat.lobe_param()   # BSDF.param: a conductor's GGX alpha
    # A smooth conductor is a mirror: it scatters, but stores no vertex and starts no strategy.
    v.is_delta = Int32(1) if mat.is_mirror() else Int32(0)
    v.hair_curve_idx = h.curve_idx
    v.hair_h = h.curve_h
    v.hair_v = h.curve_v
    v.pdf_fwd = Float32(1)   # unused by the uniform-subsample estimator
    v.wo = vec3f(-ray_dir)
    v.med_idx = med_idx
    v.wavelengths = wl
    return VcmLobeHit(v, h.gn, h.gn_geo, h.cos_in, h.spawn_eps, h.on_curve, h.alb)


def _bdpt_camera_path_bounce[use_gpu: Bool](
    ref sd:      SceneView,
    mut pcg: PCG32,
    has_med: Bool,
    inter: Intersection,
    scratch: Pointer[Intersection, MutUntrackedOrigin],
    lvc:     Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx:  Int,
    path_len: Int,
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    merge_heads: Pointer[Int32, MutUntrackedOrigin],
    merge_inv_cell: Float32,
    merge_r2: Float32,
    merge_norm: Float32,
    mis_vc_weight_factor: Float32,
    mis_vm_weight_factor: Float32,
    mut st: VCMCameraPathState,
    # Benchmark instrumentation only: merge visits per estimator (see _bdpt_merge_from_cache).
    mut visit_naive: Int32,
    mut visit_footprint: Int32,
    mut visit_thin: Int32,
    # _VCM_CAMIS camera state (vcm_camis.CamisCamCarry/CamisCamRecord), the
    # records of every vertex stored so far. Untouched when the hybrid is compiled out.
    mut camis: CamisCamCarry,
    mut camis_recs: Array[CamisCamRecord, _CAMIS_CAM_RECS],
    # Task #163 stage 5: when set, the DIFFUSE branch's connect step queues
    # its shadow rays into these buffers (one _BDPT_MAX_VERTS-sized slice
    defer_shadow_rays: Bool = False,
    shadow_rays: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    shadow_pending: Pointer[SpectralSample, MutUntrackedOrigin] = Pointer[SpectralSample, MutUntrackedOrigin].unsafe_dangling(),
    shadow_valid: Pointer[Int8, MutUntrackedOrigin] = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling(),
    shadow_seg_med: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
    # _VCM_CAMIS: see _bdpt_trace_camera_and_connect's matching parameter.
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin] = Pointer[CamisLightRecord, MutUntrackedOrigin].unsafe_dangling(),
    # _VCM_CAMIS: n_t in Eq. 17's clamp c >= 1/(n_t keep); see
    # _bdpt_trace_camera_and_connect's matching parameter.
    n_light_paths_f: Float32 = Float32(0),
) -> Bool:
    """One bounce of a camera subpath, shared by the per-path loop and the wavefront kernels; False ends the path."""
        if st.n_verts >= _BDPT_MAX_VERTS:
            return False   # mirrors the original loop's top-of-iteration guard
        if inter.hit == Int8(0):
            for inf_i in range(Int(sd.infiniteLightCount)):
                var ilight = sd.infiniteLights[unsafe_offset=inf_i]
                var (Le, pdf_light_here) = _eval_infinite_light_and_pdf(ilight, st.rd, st.ro.to_simd())
                var mis_w = Float32(1)
                if st.last_bsdf_pdf == PDF_DROP_DIRECT:
                    # A scatter whose direct term NEE already reported (the
                    # rough-coat exit ray): indirect only. Same contract PT's
                    mis_w = Float32(0)
                elif st.n_verts > 0 and pdf_light_here > Float32(0):
                    # NOT `last_bsdf_pdf >= 0`, which is what this used to be.
                    # That test excluded every path whose last event was DELTA
                    var (_c_esc, r_esc) = _scene_bounding_sphere(sd)
                    var emis_esc = pdf_light_here * light_path_pick_pdf(sd, _vcm_infinite_slot(sd, inf_i)) / max(PI * r_esc * r_esc, Float32(1e-12))
                    mis_w = vcm_env_escape_weight(pdf_light_here, emis_esc,
                                                  st.dvcm, st.dvc)
                st.total += st.beta * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (Le).r, (Le).g, (Le).b, st.wl) * mis_w
            return False   # nothing hit -- path escapes the scene
        var t_hit = inter.tHit
        var ray_dir = st.rd.to_simd()
        if st.n_verts == 0:
            st.cone_len += t_hit
        var vcm_cone_w = sd.camFp.cone_spread * st.cone_len if st.n_verts == 0 else Float32(-1.0)

        # VCM Stage 2b: distance-squared portion of the per-bounce MIS
        # correction -- see _bdpt_trace_light_path's matching comment
        # A null crossing is one edge split in two: its d^2 is undone at the interface and the whole distance applied here.
        var d_seg = t_hit + st.mis_null_dist
        st.dvcm *= d_seg * d_seg

        # Volume free-flight
        if has_med and Int(st.cur_med_idx) >= 0:
            comptime if _VCM_CAMIS:
                # A medium segment (collision or pass-through, whose 1/FF
                # enters dVCM) is outside CAMIS's Class: S0's harness models
                # vacuum edges only. Legacy weights stay exact for it.
                camis.in_class = False
            var med = sd.mediums[unsafe_offset=Int(st.cur_med_idx)]
            # ONE shared sampler for both medium kinds (geometry.mojo):
            # homogeneous closed form, or delta tracking against the real
            var ff = sample_free_flight(
                med, sd.grids, sd.nvdbGrids, Vec3f(st.ro.x, st.ro.y, st.ro.z), st.rd, t_hit, pcg)
            if ff.collided:
                # Chromatic collision weight -- the ratio to the sampled (red)
                # channel. Without it a chromatic medium is biased at every
                st.beta *= spectral_free_flight_weight(med, ff, t_hit, st.wl, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                # Volume scatter vertex
                var sp = st.ro + st.rd*ff.t_free
                # dVCM's Jacobian above was applied with t_hit (distance to
                # the SURFACE the ray was cast toward); the real arrival
                st.dvcm *= ((ff.t_free + st.mis_null_dist) * (ff.t_free + st.mis_null_dist)) / max(d_seg * d_seg, Float32(1e-20))
                st.dvcm *= Float32(1.0) / max(ff.pdf, Float32(1e-30))
                if not st.prev_was_volume:
                    st.dvc *= Float32(1.0) / max(ff.sig_t, Float32(1e-20))
                var v = _null_vertex()
                v.pos = sp
                # `beta`, NOT beta*albedo -- a vertex's beta is the throughput
                # ARRIVING at it; connect time applies the vertex's own
                v.beta = st.beta
                v.alb = ff.albedo
                v.is_surface = Int32(0); v.is_delta = Int32(0)
                v.pdf_fwd = ff.pdf   # the density actually sampled from; under hero-wavelength MIS this is a lane MIXTURE, not sig_t's lone exponential
                v.med_idx = st.cur_med_idx
                v.wavelengths = st.wl
                # ARRIVAL carries -- merge/connect below read these off `v`
                # (it is never stored in the LVC on the camera side, only
                # used transiently for this bounce's own merge/connect).
                v.dVCM = st.dvcm; v.dVC = st.dvc; v.dVM = st.dvm
                if st.n_verts == 0: st.first_alb = ff.albedo
                # NOT `n_verts += 1`. _BDPT_MAX_VERTS is the light-vertex
                # CACHE's per-path slot count, and the camera subpath stores
                var eta_v = _vcm_eta_at(sd, v, mis_vm_weight_factor)
                (st.dvcm, st.dvc, st.dvm) = vcm_scatter_carries[dvc0=_VCM_CAMIS](
                    st.dvcm, st.dvc, st.dvm,
                    Float32(4.0) * PI, INV_FOUR_PI, INV_FOUR_PI,
                    mis_vc_weight_factor, eta_v)
                st.prev_was_volume = True
                st.mnee.pdf0 = Float32(0)
                if path_len > 0:
                    # BUG FIX (found investigating volumetric-caustic's
                    # colored-blob fireflies): merge is only safe to run
                    if _bdpt_vertex_mis_scoped(v):
                        st.total_merge += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor, st.n_verts + st.n_delta, camis, camis_recs, lvc_camis, n_light_paths_f, visit_naive=visit_naive, visit_footprint=visit_footprint, visit_thin=visit_thin)
                    st.total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor, st.n_verts + st.n_delta, camis, camis_recs, lvc_camis, n_light_paths_f)
                # Volume-scatter NEE: distant/point/sphere/infinite lights.
                # _bdpt_connect_to_cache above only reaches AREA lights -- the
                var alb_spec_v = rgb_bands_to_spectral_sample((ff.albedo).r, (ff.albedo).g, (ff.albedo).b, st.wl)
                var phase_alb_v = alb_spec_v * INV_FOUR_PI
                for li_v in range(_bdpt_simple_light_count(sd)):
                    var ls_v = _bdpt_sample_simple_light(sd, li_v, sp.to_simd(), pcg)
                    if ls_v.valid and ls_v.pdf > Float32(0):
                        var li_spec_v = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (ls_v.Li).r, (ls_v.Li).g, (ls_v.Li).b, st.wl)
                        var mis_v = Float32(1) if ls_v.is_delta else power_heuristic(ls_v.pdf, INV_FOUR_PI)
                        var w_v = phase_alb_v * li_spec_v * (mis_v / ls_v.pdf)
                        st.total += _bdpt_nee_contribute(st.beta, w_v, ls_v, sp, Vec3f(Float32(0)), st.cur_med_idx, sd, scratch, st.wl, Float32(0))
                for inf_v in range(Int(sd.infiniteLightCount)):
                    var ls_infv = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_v], Point2f(pcg.next_float(), pcg.next_float()), sp)
                    if ls_infv.valid and ls_infv.pdf > Float32(0):
                        var li_spec_infv = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (ls_infv.Li).r, (ls_infv.Li).g, (ls_infv.Li).b, st.wl)
                        var mis_infv = Float32(1) if ls_infv.is_delta else power_heuristic(ls_infv.pdf, INV_FOUR_PI)
                        var w_infv = phase_alb_v * li_spec_infv * (mis_infv / ls_infv.pdf)
                        st.total += _bdpt_nee_contribute(st.beta, w_infv, ls_infv, sp, Vec3f(Float32(0)), st.cur_med_idx, sd, scratch, st.wl, Float32(0))
                # Continuation beta = prev × alb_s (same as stored vertex beta)
                st.beta *= spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (ff.albedo).r, (ff.albedo).g, (ff.albedo).b, st.wl)
                var u1 = pcg.next_float(); var u2 = pcg.next_float()
                var cosT = Float32(2)*u1 - Float32(1)
                var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                var phi  = Float32(2)*PI*u2
                st.rd = Vec3f(sinT*cos(phi), sinT*sin(phi), cosT)
                st.ro = sp + st.rd*Float32(0.0002)
                st.last_bsdf_pdf = _VOL_PHASE_HIT   # see the sentinel's definition
                st.mis_null_dist = Float32(0)     # this vertex is the new origin
                return True   # volume free-flight scatter: no vertex stored this bounce, path continues
            else:
                # Pass-through weight (a per-channel RATIO -- see
                # sample_homogeneous_free_flight; the sampled channel's
                st.beta *= spectral_free_flight_weight(med, ff, t_hit, st.wl, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                # MIS: dVCM must equal 1/P(prev -> this vertex), and reaching a
                # SURFACE through a medium carries a survival factor
                var ff_exp = -log(max(ff.pdf, Float32(1e-30)))   # optical depth of the density actually sampled from (see FreeFlight.pdf)
                if ff_exp > Float32(60.0): ff_exp = Float32(60.0)  # defensive: e^-60 pass-through never occurs
                st.dvcm *= exp(ff_exp)
                # ...unless the PREVIOUS vertex was itself a volume scatter --
                # then this edge changes kind (volume -> surface) and dVC
                if st.prev_was_volume:
                    st.dvc *= ff.sig_t
                st.prev_was_volume = False

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[unsafe_offset=mat_idx]
        var hit = st.ro + st.rd*t_hit
        var eta_x = mis_vm_weight_factor * _vcm_eta_scale(sd, hit)   # merging's MIS density HERE

        # Direct hit on an emissive analytic sphere — checked BEFORE material
        # dispatch since the sphere's own material is often an inert
        if inter.primId.type == Int8(4):
            var sph_hit = sd.spheres[unsafe_offset=Int(inter.primId.id1)]
            if sph_hit.isAreaLight != Int8(0):
                var mis_w_sph_hit = Float32(1)
                if st.last_bsdf_pdf == PDF_DROP_DIRECT:
                    # Direct term already reported by NEE: indirect only, so
                    # this hit contributes nothing. Was `pass`, which left the
                    # weight at 1 -- see the area-light handler below.
                    mis_w_sph_hit = Float32(0)
                elif st.last_bsdf_pdf >= Float32(0) or (st.n_verts > 0 and st.last_bsdf_pdf == PDF_DELTA_FULL):
                    # The competing NEE was taken at the vertex that GENERATED
                    # this ray, so its cone pdf is measured from there -- not
                    var x_nee = st.ro.to_simd() - ray_dir * st.mis_null_dist
                    var pdf_cone = sphere_light_cone_pdf(sph_hit, x_nee)
                    if st.dvcm > Float32(0) or st.dvc > Float32(0):
                        # SmallVCM's GetLightRadiance, the balance weight over
                        # EVERY strategy now that sphere lights start light
                        var aw_hit = Float32(4) * PI * sph_hit.radius * sph_hit.radius / light_path_pick_pdf(sd, _vcm_sphere_slot(sd, Int(inter.primId.id1)))
                        var w_cam_sph = pdf_cone * st.dvcm / max(t_hit * t_hit, Float32(1e-12)) + st.dvc / (PI * aw_hit)
                        if st.last_bsdf_pdf == PDF_DELTA_FULL and st.mnee.pdf0 > Float32(0):
                            var sn = (hit - sph_hit.center).to_simd() * (Float32(1) / sph_hit.radius)
                            var tng = cross(sn, Vec3f(Float32(1), Float32(0), Float32(0)) if abs(sn[0]) < Float32(0.9) else Vec3f(Float32(0), Float32(1), Float32(0)))
                            w_cam_sph += _vcm_mnee_hit_term(sd, st.mnee.x0, st.mnee.pdf0, Int(st.mnee.n_glass), st.ro.to_simd(), hit.to_simd(),
                                                            tng, cross(sn, tng), Float32(1) / aw_hit, pcg)
                        mis_w_sph_hit = Float32(1) / (Float32(1) + w_cam_sph)
                    elif st.last_bsdf_pdf < Float32(0):
                        pass   # delta bounce off a vertex with placeholder carries: unweighted, as before
                    elif pdf_cone > Float32(0):
                        # A vertex outside real per-vertex MIS (zero
                        # placeholder carries) keeps the two-strategy weight.
                        # No 1/count factor -- see _sample_sphere_light_nee.
                        mis_w_sph_hit = power_heuristic(st.last_bsdf_pdf, pdf_cone)
                elif st.last_bsdf_pdf == _VOL_PHASE_HIT:
                    # An isotropic-phase-sampled ray landing on a sphere light,
                    # weighed against the volume vertex's own NEE (volume
                    # vertices stay outside real per-vertex MIS).
                    var pdf_cone_v = sphere_light_cone_pdf(sph_hit, st.ro.to_simd() - ray_dir * st.mis_null_dist)
                    if pdf_cone_v > Float32(0):
                        mis_w_sph_hit = power_heuristic(INV_FOUR_PI, pdf_cone_v)
                st.total += st.beta * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (sph_hit.emission).r, (sph_hit.emission).g, (sph_hit.emission).b, st.wl) * mis_w_sph_hit
                return False   # direct hit on emissive analytic sphere -- terminates the path

        resolve_mix(sd.materials, pcg, mat, mat_idx)
        if mat.type == MatKind.hair:   # the fibre kernel's density (_vcm_fibre_scale)
            eta_x = mis_vm_weight_factor * _vcm_fibre_scale(sd, hit, Int(inter.primId.id1), inter.v)

        if mat.type == MatKind.area_light or mat.type == MatKind.lamp:
            # Direct hit on a triangle/curve area light — same MIS-against-
            # last_bsdf_pdf treatment as the sphere case above. id1 is the
            var is_curve = inter.primId.type == Int8(5)
            var (al_emission, al_area, al_slot, cos_l_hit) = _area_light_hit(sd, inter, mat, ray_dir)
            # A curve is a closed tube, so an unweighted (primary/delta) hit is
            # always on its outside -- credited regardless of the reconstructed
            if cos_l_hit > Float32(0) or (is_curve and st.last_bsdf_pdf < Float32(0) and st.last_bsdf_pdf != PDF_DROP_DIRECT):
                var mis_w_al_hit = Float32(1)
                if st.last_bsdf_pdf == PDF_DROP_DIRECT:
                    # NEE already reported this ray's direct term (the rough
                    # coat's exit ray) -- geometry.mojo's contract for this
                    mis_w_al_hit = Float32(0)
                elif st.last_bsdf_pdf >= Float32(0) or (st.n_verts > 0 and st.last_bsdf_pdf == PDF_DELTA_FULL):
                    # A DELTA last bounce (glass, mirror: last_bsdf_pdf = -1)
                    # is weighted too once a real vertex is stored -- the
                    if cos_l_hit > Float32(0) and al_area > Float32(0) and (st.dvcm > Float32(0) or st.dvc > Float32(0)):
                        # SmallVCM's GetLightRadiance, the balance weight over
                        # EVERY strategy -- not the 2-strategy power heuristic
                        var p_a = light_path_pick_pdf(sd, al_slot) / al_area
                        var emission_pdf_w = p_a * cos_l_hit * INV_PI * area_light_side_pdf(sd.areaLights[unsafe_offset=al_slot])
                        var mnee_t = Float32(0)   # MNEE through the glass chain behind this hit, if it reproduces it
                        if st.last_bsdf_pdf == PDF_DELTA_FULL and st.mnee.pdf0 > Float32(0) and not is_curve:
                            var (lmesh_h, lv0_h, lv1_h, lv2_h, _) = _get_tri_verts(inter, sd.meshes)
                            var lp0_h = mesh_pos(lmesh_h, lv0_h)
                            mnee_t = _vcm_mnee_hit_term(sd, st.mnee.x0, st.mnee.pdf0, Int(st.mnee.n_glass), st.ro.to_simd(), hit.to_simd(),
                                                        mesh_pos(lmesh_h, lv1_h) - lp0_h, mesh_pos(lmesh_h, lv2_h) - lp0_h, p_a, pcg)
                        comptime if _VCM_CAMIS:
                            if camis.in_class:
                                # This hit is not a stored vertex (the path
                                # terminates), so camis is still the carry
                                var dvcm_arr_al = st.dvcm / max(cos_l_hit, Float32(1e-6))
                                var log_py_arr_al = camis.log_py + camis_clamp_log_p(
                                    camis.log_k, -log(dvcm_arr_al))
                                var log_g_prev_arr_al = log(camis.cos_out_prev / max(t_hit * t_hit, Float32(1e-12)))
                                var cam_arr_al = CamisCamCarry(camis.in_class, camis.cut, camis.log_k,
                                    log_py_arr_al, camis.rc, log_g_prev_arr_al, Float32(0))
                                mis_w_al_hit = camis_eval_emission_hit[_CAMIS_CAM_RECS](
                                    cam_arr_al, camis_recs, st.n_verts,
                                    dvcm_arr_al, st.dvc / max(cos_l_hit, Float32(1e-6)),
                                    st.dvm / max(cos_l_hit, Float32(1e-6)), log(_vcm_keep(sd, hit)),
                                    p_a, emission_pdf_w, log(cos_l_hit * INV_PI),
                                    n_light_paths_f, True, _CAMIS_FORCE_C1)
                            else:
                                var w_cam_hit_off = (p_a * st.dvcm + emission_pdf_w * st.dvc) / cos_l_hit + mnee_t
                                mis_w_al_hit = Float32(1) / (Float32(1) + w_cam_hit_off)
                        else:
                            var w_cam_hit = (p_a * st.dvcm + emission_pdf_w * st.dvc) / cos_l_hit + mnee_t
                            mis_w_al_hit = Float32(1) / (Float32(1) + w_cam_hit)
                    elif st.last_bsdf_pdf < Float32(0):
                        pass   # delta bounce off a vertex with placeholder carries: unweighted, as before
                    elif cos_l_hit > Float32(0) and al_area > Float32(0):
                        # A vertex outside real per-vertex MIS (its carries are
                        # zero placeholders) keeps the old two-strategy weight.
                        var dist2_hit = t_hit * t_hit
                        var n_area_hit = Float32(max(Int(sd.areaLightCount), 1))
                        var pdf_light_al = dist2_hit / (cos_l_hit * n_area_hit * al_area)
                        mis_w_al_hit = power_heuristic(st.last_bsdf_pdf, pdf_light_al)
                    else:
                        mis_w_al_hit = Float32(0)
                elif st.last_bsdf_pdf == _VOL_PHASE_HIT:
                    # The competing strategy is the volume vertex's s=1
                    # connection to the light-source vertex, whose area pdf is
                    var d_vol = t_hit + st.mis_null_dist
                    var n_lights_hit = Float32(1) / light_path_pick_pdf(sd, al_slot)
                    if cos_l_hit > Float32(0) and al_area > Float32(0):
                        var pdf_light_vol = d_vol * d_vol / (cos_l_hit * n_lights_hit * al_area)
                        mis_w_al_hit = power_heuristic(INV_FOUR_PI, pdf_light_vol)
                    else:
                        mis_w_al_hit = Float32(0)
                st.total += st.beta * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al_emission.r, al_emission.g, al_emission.b, st.wl) * mis_w_al_hit
            if mat.type == MatKind.lamp:   # Cycles' lamps let rays through, as a null interface (shade_light.h)
                st.ro = hit + st.rd*Float32(0.0002)
                st.dvcm /= max(d_seg * d_seg, Float32(1e-20))
                st.mis_null_dist += t_hit + Float32(0.0002)
                return True
            return False   # direct hit on an area-light triangle/curve -- terminates the path

        # Every arm of the dispatch below except a null interface is a real
        # scattering event, and the null-interface distance accumulated on
        if mat.type != MatKind.interface:
            st.mis_null_dist = Float32(0)

        if mat.uses_lobe():
            if not mat.has_lobe():
                return False   # e.g. a measured table that failed to load
            var lh = _vcm_lobe_vertex[use_gpu](sd, inter, mat, mat_idx, hit, ray_dir, vcm_cone_w, st.beta, st.cur_med_idx,
                                               st.wl, st.dvcm, st.dvc, st.dvm)
            var v = lh.v
            var gn = lh.gn; var gn_geo = lh.gn_geo
            var spawn_eps = lh.spawn_eps; var on_curve = lh.on_curve; var eff_alb = lh.eff_alb
            v.dVCM = st.dvcm; v.dVC = st.dvc; v.dVM = st.dvm
            if v.is_delta != Int32(0):
                if st.n_verts + st.n_delta >= _vcm_depth(sd):
                    return False   # a delta vertex past d starts no strategy either
                st.n_delta += 1
            if v.is_delta == Int32(0):
                if st.n_verts == 0: st.first_alb = eff_alb
                if st.n_verts + st.n_delta >= _vcm_depth(sd):
                    return False   # a vertex past d starts no strategy (see _vcm_depth)
                st.n_verts += 1
                comptime if _VCM_CAMIS:
                    # Before merge/connect: S3's weights at this vertex read
                    # its log P(y) and the finalised records below it.
                    camis_cam_arrive(camis, camis_recs, st.n_verts - 1, st.dvcm, lh.cos_fix, t_hit,
                                     _bdpt_vertex_mis_scoped(v))
                # Merging queries the GLOBAL photon grid and does not use this
                # pixel's own paired light path, so unlike the connect below it
                st.total_merge += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor, st.n_verts + st.n_delta, camis, camis_recs, lvc_camis, n_light_paths_f, visit_naive=visit_naive, visit_footprint=visit_footprint, visit_thin=visit_thin)
                if path_len > 0:
                    # Task #163 stage 5: the diffuse branch's connect shadow
                    # rays are the single highest-volume, cleanest shadow-ray
                    if defer_shadow_rays:
                        _bdpt_connect_to_cache_deferred(v, sd, lvc, lp_idx, path_len, mis_vm_weight_factor, shadow_rays, shadow_pending, shadow_valid, shadow_seg_med, st.n_verts + st.n_delta, camis, camis_recs, lvc_camis, n_light_paths_f)
                    else:
                        st.total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor, st.n_verts + st.n_delta, camis, camis_recs, lvc_camis, n_light_paths_f)

                # NEE to distant/point/sphere/infinite lights, via the shared
                # Light interface (bvh.mojo's LightSample samplers) + BxDF
                st.total += _vcm_nee_surface(sd, v, _vertex_ctx(v), hit, gn_geo, st.beta, eta_x, st.dvcm, st.dvc,
                                          st.cur_med_idx, scratch, st.wl, pcg, spawn_eps,
                                          v.mat_kind == LobeKind.diffuse_transmit or v.mat_kind == LobeKind.hair or v.mat_kind == LobeKind.rough_dielectric,
                                          Float32(0))
                # MNEE toward a light behind glass, one more strategy in the balance (_vcm_mnee).
                if LobeKind.is_diffuse(v.mat_kind):
                    st.total += _vcm_mnee(sd, v, hit.to_simd(), gn, st.beta, pcg, st.wl, eta_x, st.dvcm, st.dvc)

            # Scatter through THE lobe sampler -- the same BSDF connect,
            # merge and NEE evaluate this vertex with (see _vcm_scatter).
            var sc = _vcm_scatter(v, False, pcg, sd, st.wl, st.dvcm, st.dvc, st.dvm, mis_vc_weight_factor, eta_x)
            if not sc.valid:
                return False
            comptime if _VCM_CAMIS:
                if v.is_delta != Int32(0) or sc.is_delta:
                    camis.in_class = False   # a mirror vertex or a smooth coat's delta lobe
                else:
                    camis_cam_scatter(camis, camis_recs, st.n_verts - 1, eta_x, log(_vcm_keep(sd, hit)),
                                      sc.cos_out / sc.pdf_fwd, sc.cos_out, sc.pdf_fwd, sc.pdf_rev)
            st.rd = vec3f(sc.wi)
            if on_curve:
                st.ro = hit + vec3f(gn_geo) * (spawn_eps if dot(sc.wi, gn_geo) >= Float32(0) else -spawn_eps)
            else:
                st.ro = hit + st.rd*Float32(0.0002)
            st.beta *= sc.weight
            st.last_bsdf_pdf = Float32(-1) if sc.is_delta else sc.pdf_fwd
            if v.is_delta == Int32(0):
                var receiver = LobeKind.is_diffuse(v.mat_kind)
                st.mnee = VCMMneeTrack(hit.to_simd(), sc.pdf_fwd if receiver and not sc.is_delta else Float32(0), Int32(0))
            else:
                st.mnee.pdf0 = Float32(0)

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            comptime if _VCM_CAMIS:
                # A BSSRDF hop and a delta dielectric are both outside the Class.
                camis.in_class = False
            var did_bssrdf_hop = False
            # ── Subsurface boundary: a BSSRDF hop to a real EXIT VERTEX ────
            # The camera arrives at the entry x_i, the exit x_o is sampled from
            # the diffusion profile (shared with the light subpath, see
            if mat.sss_boundary != Int8(0) and Int(st.cur_med_idx) < 0 and has_med:
                var med_in = medium_after(ray_dir, Int32(-1), inter, mat, sd, hit.to_simd())
                if med_in >= Int32(0) and Int(med_in) < Int(sd.mediumCount):
                    var ex = _vcm_bssrdf_hop(sd, inter, mat, hit, ray_dir, Int(med_in), mis_vc_weight_factor, pcg, st.wl,
                                             st.beta, st.dvcm, st.dvc, st.dvm)
                    if not ex.ok:
                        return False
                    var eta_e = mat.albedo.r
                    var x_o = ex.x_o
                    var n_o = ex.n_o
                    var v = _vcm_bssrdf_vertex(ex, eta_e, st.beta, st.cur_med_idx, st.wl, st.dvcm, st.dvc, st.dvm)
                    if st.n_verts == 0: st.first_alb = mat.sss_mean_refl
                    if st.n_verts + st.n_delta >= _vcm_depth(sd):
                        return False   # a vertex past d starts no strategy (see _vcm_depth)
                    st.n_verts += 1
                    # Merging queries the GLOBAL photon grid and does not use this
                    # pixel's own paired light path, so unlike the connect below it
                    st.total_merge += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor, st.n_verts + st.n_delta, camis, camis_recs, lvc_camis, n_light_paths_f, visit_naive=visit_naive, visit_footprint=visit_footprint, visit_thin=visit_thin)
                    if path_len > 0:
                        if defer_shadow_rays:
                            _bdpt_connect_to_cache_deferred(v, sd, lvc, lp_idx, path_len, mis_vm_weight_factor, shadow_rays, shadow_pending, shadow_valid, shadow_seg_med, st.n_verts + st.n_delta, camis, camis_recs, lvc_camis, n_light_paths_f)
                        else:
                            st.total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor, st.n_verts + st.n_delta, camis, camis_recs, lvc_camis, n_light_paths_f)
                    # Direct lighting at the exit: the diffuse vertex's NEE with
                    # the exit lobe's Fresnel factor toward each light.
                    var ctx_x = BSDF(LobeKind.lambertian, True, False, n_o, n_o, RGB(Float32(1)), Int32(-1), Float32(0),
                                        Float32(0), Int32(-1), Float32(0), Float32(0), True, False)
                    st.total += _vcm_nee_surface(sd, v, ctx_x, x_o, n_o, st.beta, mis_vm_weight_factor * _vcm_eta_scale(sd, x_o),
                                              v.dVCM, v.dVC, st.cur_med_idx, scratch, st.wl, pcg,
                                              offset_eps(x_o.x, x_o.y, x_o.z), False, eta_e)
                    # Continue with the exit lobe: cosine-sampled, weight Ft(cos_out).
                    var cos_out_x = _vcm_bssrdf_exit_scatter(sd, ex, eta_e, mis_vc_weight_factor, mis_vm_weight_factor, pcg,
                                                             st.ro, st.rd, st.beta, st.dvcm, st.dvc, st.dvm)
                    st.last_bsdf_pdf = cos_out_x * INV_PI
                    did_bssrdf_hop = True
                    st.mnee.pdf0 = Float32(0)
            # Ordinary specular boundary, only when no hop was taken.
            if not did_bssrdf_hop:
                if st.n_verts + st.n_delta >= _vcm_depth(sd):
                    return False   # a delta vertex past d starts no strategy either
                st.n_delta += 1
                st.mnee.n_glass += 1
                st.beta *= _vcm_dielectric_cross[use_gpu, True](sd, inter, mat, hit, ray_dir, vcm_cone_w, has_med, pcg,
                    st.ro, st.rd, st.current_dielectric_ior, st.previous_dielectric_ior, st.cur_med_idx, st.dvcm, st.dvc, st.dvm)
                st.last_bsdf_pdf = Float32(-1)  # delta bounce: no infinite-light NEE done here

        elif mat.type == MatKind.interface:
            if has_med:
                st.cur_med_idx = medium_after(ray_dir, st.cur_med_idx, inter, mat, sd, hit.to_simd())
            st.ro = hit + st.rd*Float32(0.0002)
            st.dvcm /= max(d_seg * d_seg, Float32(1e-20))
            st.mis_null_dist += t_hit + Float32(0.0002)   # see VCMCameraPathState.mis_null_dist
            # VCM Stage 2b: pure pass-through, carry unchanged (see
            # _bdpt_trace_light_path's matching interface-branch comment).
            comptime if _VCM_CAMIS:
                # A null crossing splits one edge in two (dVCM picks up both
                # segments' d^2), which CAMIS's per-edge P does not model.
                camis.in_class = False

        else:
            return False   # unhandled material type


        return True   # bounce processed normally, path continues
