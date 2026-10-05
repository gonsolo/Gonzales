# BDPT/VCM camera subpath: state, init and the per-bounce step.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.collections import Array
from std.math import sqrt, cos, sin, log, exp, max, abs
from .geometry import face_toward, RGB, Point3f, Point2f, Vec3f, vec3f, point3f, Frame, dot, PI, INV_FOUR_PI, INV_PI
from .render_state import PDF_DELTA_FULL, PDF_DROP_DIRECT
from .materials import Material, MatKind, LobeKind, dielectric_is_rough
from .primitives import Ray, Intersection
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
from .bvh import (
    SceneView, traverse_bvh2_core, test_spheres, _scene_bounding_sphere, _eval_infinite_light_and_pdf,
    _hair_precompute, curve_offset_eps, sphere_light_cone_pdf, _sample_infinite_light_nee,
    light_path_pick_pdf,
)
from .sampling import power_heuristic, camera_ray_from_film_xy, FilmFilter, film_filter_offset
from .rng import PCG32
from .sppm import _geom_normal, _shading_normal_at, _dielectric_bounce, medium_after_crossing, _cosine_hemisphere_sample
from .shading import (
    uv_footprint_at_hit, _tex_lookup, _get_tri_verts, apply_surface_maps_at_hit, area_light_hit_cos,
    curve_light_hit,
)
from .bxdf import (
    LobeCtx, lobe_sample, LobeSample, lobe_kind_of, lobe_param_of, lobe_is_delta_of, lobe_is_available_of,
    GeomContext, BxDFSample, bxdf_sample_coated_conductor, bxdf_is_delta, bxdf_pdf_conductor_ggx, LobeTables,
)
from .spectrum import (
    SampledWavelengths, SpectralSample, spec_refl, spec_refl_unbounded, spec_illum,
    rgb_bands_to_spectral_sample,
)
from .bdpt_vertex import _VOL_PHASE_HIT, _BDPT_MAX_DEPTH, _BDPT_MAX_VERTS, BDPTVertex, _null_vertex
from .bdpt_nee import (
    _bdpt_simple_light_count, _bdpt_sample_simple_light, _bdpt_nee_contribute, _bdpt_mnee_diffuse_area_light,
    _bdpt_mnee_sphere_light,
)
from .bdpt_eval import _vcm_sphere_slot, _vcm_infinite_slot, _vcm_nee_surface, _vertex_ctx, _bdpt_vertex_mis_scoped
from .vcm_grid import _vcm_depth, _vcm_keep, _VCM_CAMIS, _CAMIS_CAM_RECS, _CAMIS_FORCE_C1, _vcm_eta_scale, _vcm_eta_at
from .bdpt_connect import _bdpt_connect_to_cache, _bdpt_connect_to_cache_deferred
from .vcm_merge import _bdpt_merge_from_cache
from .bdpt_bssrdf import _bdpt_sample_bssrdf_exit

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
    connect inline/synchronously to the shared Light Vertex Cache via
    `_bdpt_connect_to_cache` — mirrors how every live GPU shading kernel in
    this codebase already does its shadow ray (any_hit test, straight into
    the thread's own accumulator; gpu.mojo's queued ShadowTask mechanism
    is dead code, never used by the live render loop). Returns (total,
    total_merge, first_alb): this camera path's connect+splat contribution
    and its vertex-MERGING contribution for one spp sample, tracked
    SEPARATELY (see VCMCameraPathState.total_merge's docstring), plus the material
    albedo at its first non-delta (stored) vertex — the same "first hit,
    skipping through mirrors/glass" convention shading.mojo's path.albedo AOV
    already uses, needed for the denoiser's albedo guide buffer (see
    vcm_render's docstring), plus (visit_naive, visit_footprint, visit_thin):
    Benchmark instrumentation only, see
    _bdpt_merge_from_cache's docstring paragraph -- summed over every bounce
    of this one camera subpath. `use_gpu` now genuinely matters: it selects
    _tex_lookup's CPU (tex_filenames/OIIO) vs GPU (GpuTexture array)
    texture-sampling branch for diffuse/coateddiffuse vertex albedo — CPU
    and GPU callers MUST pass the value matching their own reality (the
    CPU driver previously passed [False] here anyway, so this was already
    correct; VCM Stage 2c's texture fix (task #150) is what makes it
    load-bearing) — kept as a parameter so its signature matches
    `_bdpt_trace_light_path`'s and the two thin kernels wrapping this stay
    symmetric in Phase (b).

    `lp_idx`/`path_len` (VCM Stage 2b) identify this pixel's own
    DETERMINISTICALLY PAIRED light path (n_light_paths == n_pix, standard
    Veach BDPT pairing — see _bdpt_store_lvc_vertex's docstring and
    project_vcm_stage2_mis_derivation memory) — `_bdpt_connect_to_cache`
    connects to every one of that path's `path_len` stored vertices, an
    exhaustive sum over its depth-strategies, no random subsampling or
    rescaling needed."""

    # SHARED WITH THE WAVEFRONT GPU DRIVER -- see _bdpt_trace_light_path's
    # matching comment. This body was a byte-for-byte copy of
    # _bdpt_camera_path_init + _bdpt_camera_path_bounce (988 lines against
    # 978, 599 identical non-comment lines). Both designs now run the same
    # step: this one loops over it inline, the wavefront driver launches it
    # once per depth level with the loop-carried state -- including the
    # `total`/`first_alb` accumulators -- parked in a VCMCameraPathState.
    var st = _bdpt_camera_path_init[use_gpu](
        r2c, c2w, px, py, pcg, px_scale, n_light_paths_f, pass_wl, film_filter, start_med_idx)
    var ro = st.ro
    var rd = st.rd
    var beta = st.beta
    var total = st.total
    var total_merge = st.total_merge
    # Benchmark instrumentation only -- see
    # _bdpt_merge_from_cache's docstring paragraph.
    var visit_naive = Int32(0)
    var visit_footprint = Int32(0)
    var visit_thin = Int32(0)
    var first_alb = st.first_alb
    var n_verts = Int(st.n_verts)
    var n_delta = Int(st.n_delta)
    var n_bounces = Int(st.n_bounces)
    var cur_med_idx = st.cur_med_idx
    var dvcm_carry = st.dvcm
    var dvc_carry = st.dvc
    var dvm_carry = st.dvm
    var last_bsdf_pdf = st.last_bsdf_pdf
    var mis_null_dist = st.mis_null_dist
    var current_dielectric_ior = st.current_dielectric_ior
    var previous_dielectric_ior = st.previous_dielectric_ior
    var cone_len = st.cone_len
    var wavelengths = SampledWavelengths(st.wl0, st.wl1, st.wl2, st.wl3)
    if st.active == Int8(0):
        return (total, total_merge, first_alb, visit_naive, visit_footprint, visit_thin)
    # CAMIS camera state (vcm_camis.mojo): live across the whole bounce loop,
    # the same `mut` threading dvcm_carry uses. One element and never touched
    # when _VCM_CAMIS is off.
    var camis = camis_cam_carry_init()
    var camis_recs = Array[CamisCamRecord, _CAMIS_CAM_RECS](
        fill=CamisCamRecord(Float32(0), Float32(0), Float32(0), Float32(0), Float32(0)))
    var prev_was_volume = False

    for _ in range(_BDPT_MAX_DEPTH):
        # The same intersect step _bdpt_camera_path_intersect_gpu performs;
        # kept at the call site because it is the Vulkan RT swap point.
        var ray = Ray(ro, rd)
        scratch[unsafe_offset=0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, scratch)
        # A miss (including its infinite-light escape credit) is handled
        # inside the step, so the old top-of-loop miss block is gone.
        if not _bdpt_camera_path_bounce[use_gpu](
            sd, pcg, has_med, scratch[unsafe_offset=0], scratch, lvc, lp_idx, path_len,
            merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm,
            mis_vc_weight_factor, mis_vm_weight_factor,
            ro, rd, beta, total, total_merge, visit_naive, visit_footprint, visit_thin,
            first_alb, n_verts, n_delta, n_bounces, cur_med_idx,
            dvcm_carry, dvc_carry, dvm_carry, prev_was_volume, last_bsdf_pdf, mis_null_dist,
            current_dielectric_ior, previous_dielectric_ior, wavelengths, cone_len,
            camis, camis_recs, lvc_camis=lvc_camis, n_light_paths_f=n_light_paths_f):
            break

    return (total, total_merge, first_alb, visit_naive, visit_footprint, visit_thin)

@fieldwise_init
struct VCMCameraPathState(TrivialRegisterPassable):
    """Task #163 stage 4: persistent per-camera-path state carried across
    separate wavefront-staged GPU kernel launches (`_bdpt_camera_path_init`
    then one `_bdpt_camera_path_bounce` call per bounce), the camera-path
    counterpart to `VCMLightPathState` (see that struct's docstring for
    the general convention). Unlike the light-path side, `total`/`first_alb`
    are running ACCUMULATORS carried across every bounce, not just
    per-bounce scratch -- the host loop reads them once `active` drops to 0,
    the same value the original single-function loop would have returned."""
    var ro: Point3f
    var rd: Vec3f
    var beta: SpectralSample
    var total: SpectralSample
    # Vertex-MERGING contribution only, tracked separately from `total`
    # (which after this split holds connect + t=1 light-tracing splats) --
    # see project_water_caustic_sppm_gap memory: merging is a kernel-
    # density estimate with the same "real caustic threads read as noise
    # to a spatial-variance denoiser" problem SPPM's photon gather had.
    # Denoised separately in _vcm_finalize_one_pixel/finish_render.
    var total_merge: SpectralSample
    var first_alb: RGB
    var dvcm: Float32
    var dvc: Float32
    var dvm: Float32
    var n_verts: Int32
    var n_bounces: Int32
    var cur_med_idx: Int32
    var last_bsdf_pdf: Float32
    var active: Int8
    var pcg_state: UInt64
    var pcg_inc: UInt64
    var wl0: Float32
    var wl1: Float32
    var wl2: Float32
    var wl3: Float32
    # Distance travelled through null interfaces since the last real
    # scattering event -- same role as PathState.mis_null_dist. The
    # interface branch resets `ro` to the boundary it crossed, so a later
    # emitter hit's t_hit measures from the boundary, not from the vertex
    # whose sample generated the direction; the MIS pdf needs the latter.
    var mis_null_dist: Float32
    # Touching-dielectric IOR depth-2 stack for _dielectric_bounce (see that
    # function's docstring, sppm.mojo) -- same role and convention as
    # PathState.current_dielectric_ior/previous_dielectric_ior
    # (geometry.mojo). Both start at vacuum (1.0).
    var current_dielectric_ior: Float32
    var previous_dielectric_ior: Float32
    # Specular-chain path length for the texture/bump footprint, see
    # _bdpt_camera_path_bounce's `cone_len`.
    var cone_len: Float32
    var n_delta: Int32

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
) -> VCMCameraPathState:
    """Task #163 stage 4: camera-ray generation + MIS-origin setup half of
    `_bdpt_trace_camera_and_connect` (bdpt_*.mojo:830-886), split out to seed
    a `VCMCameraPathState` for the wavefront-staged bounce loop instead of
    falling straight into an inline `for` loop. Byte-for-byte copy of that
    function's pre-loop body -- see its own docstring/VCM Stage 2b comments
    for the cameraPdfW derivation, not repeated here. Unlike
    `_bdpt_light_path_init`, there is no early-inactive case (a camera
    subpath always starts active, even in the degenerate zero-length-ray
    edge case the original code silently tolerates) -- `active` is always 1."""
    # JITTER. This was `px + 0.5` -- the pixel CENTRE, identically for every
    # sample. VCM was the only integrator that did not jitter: the path tracer
    # uses `px + 0.5 + deltaX` through the scene's reconstruction filter
    # (sampling.mojo, gen_primary_ray_state) and SPPM uses
    # `px + pcg.next_float()` (sppm.mojo). So VCM had no anti-aliasing at all
    # and all of its spp were perfectly correlated in the film dimension.
    #
    # It also turned any first-hit-determined black into a PERMANENT black
    # pixel, since all 64 samples traced the identical ray. On
    # barcelona-pavilion that was 1587 pixels (1.98% of the lit image) stuck
    # at exactly 0.0, identical under --seed 1 and --seed 7 -- foliage, where
    # the centre ray lands on a leaf whose alpha cut-out we do not parse (see
    # `Shape "texture alpha"`, still unimplemented) so the quad is opaque
    # black. Rendering at 2x and downsampling drops that count to ZERO, which
    # is what identified the sampling as the amplifier rather than a
    # geometric hole or a NaN.
    #
    # Uniform jitter, matching SPPM. NOT the path tracer's Gaussian
    # reconstruction filter: sharing that needs the filter parameters plumbed
    # into this kernel, and SPPM already boxes, so this leaves VCM consistent
    # with one of the two rather than inventing a third behaviour. Unifying
    # all three on the real filter is the follow-up.
    #
    # FOLLOW-UP DONE: all three integrators now draw the sub-pixel position
    # through the scene's PixelFilter with the same sampler (sampling.mojo's
    # filter_sample_2d, pbrt-v4's kernel shapes). The t=1 splats are spread
    # over the same footprint, see _bdpt_splat_filtered.
    var u_fx = pcg.next_float()
    var u_fy = pcg.next_float()
    var (dfx, dfy) = film_filter_offset(u_fx, u_fy, film_filter)
    var fX = Float32(px) + Float32(0.5) + dfx
    var fY = Float32(py) + Float32(0.5) + dfy
    var (rd, ro, _cl) = camera_ray_from_film_xy(fX, fY, r2c, c2w)

    # VCM Stage 2b: real per-vertex MIS state for the eye subpath (see
    # project_vcm_stage2_mis_derivation memory). cameraPdfW derived by
    # analogy with SmallVCM's pinhole-camera imageToSolidAngleFactor,
    # using px_scale (world-space size of one pixel at unit distance along
    # the camera forward axis — same quantity the plain path tracer's mip
    # LOD already uses, pipeline.mojo) as the "pixel area at distance 1"
    # convention: cameraPdfW = 1/(px_scale² × cosθ³), cosθ = angle between
    # this ray and the camera forward axis. c2w's rotation preserves dot
    # products, so dotting the WORLD-space direction with the camera's own
    # forward axis in world space (c2w's z-column) equals what the raw
    # camera-space z-component would have been before that rotation -- no
    # need to keep the pre-rotation cx/cy/cz around just for this.
    var cos_theta_at_camera = abs(
        rd.x * c2w[unsafe_offset=8] + rd.y * c2w[unsafe_offset=9] + rd.z * c2w[unsafe_offset=10])
    var camera_pdf_w = Float32(1) / max(
        px_scale * px_scale * cos_theta_at_camera * cos_theta_at_camera * cos_theta_at_camera,
        Float32(1e-12))
    var dvcm_carry = n_light_paths_f / camera_pdf_w
    var dvc_carry = Float32(0)
    var dvm_carry = Float32(0)

    var n_verts = 0
    var n_bounces = 0  # total surface hits including glass (for _dielectric_bounce entering logic)
    var beta = SpectralSample(Float32(1))
    var cur_med_idx = start_med_idx
    var total = SpectralSample(Float32(0))
    var total_merge = SpectralSample(Float32(0))
    var first_alb = RGB(Float32(0))  # denoiser albedo AOV -- set at the first stored vertex, below
    # ONE hero-wavelength set per spp PASS, shared by every camera AND light
    # subpath in it, rather than one per subpath. This is forced by the flip
    # to spectral transport: a connection multiplies a camera vertex's beta
    # by a light vertex's flux, and a merge does the same across ARBITRARY
    # light paths -- lane i of one only means the same thing as lane i of the
    # other if both were traced at identical wavelengths. Per-subpath
    # sampling made that false for every connection and every merge. Sharing
    # per pass correlates wavelengths across pixels within one sample, which
    # costs nothing in expectation (the pass index still decorrelates across
    # spp) and is what photon-mapping-family renderers do for exactly this
    # reason.
    var wavelengths = pass_wl
    # pdf (solid angle) of the cosine-weighted diffuse bounce that produced
    # the CURRENT `rd`, used to MIS-weight this ray's eventual infinite-light
    # miss-escape contribution against the NEE-to-infinite-light sample taken
    # at the vertex that generated it (see the diffuse branch below and the
    # miss handler). -1 = no competing NEE strategy exists for whatever
    # generated this ray (primary ray, or a conductor/dielectric/volume
    # bounce — none of which do infinite-light NEE in this function) → full
    # weight, no MIS needed. Mirrors shading.mojo's lastBsdfPdf bookkeeping.
    var last_bsdf_pdf = Float32(-1)


    return VCMCameraPathState(
        ro, rd, beta, total, total_merge, first_alb, dvcm_carry, dvc_carry, dvm_carry,
        Int32(n_verts), Int32(n_bounces), cur_med_idx, last_bsdf_pdf, Int8(1),
        pcg.state, pcg.inc,
        wavelengths.lambda0, wavelengths.lambda1, wavelengths.lambda2, wavelengths.lambda3,
        Float32(0.0),
        Float32(1.0), Float32(1.0),   # current_dielectric_ior, previous_dielectric_ior (vacuum)
        Float32(0.0),                 # cone_len
        Int32(0),                     # n_delta
    )

def _vcm_scatter(
    v: BDPTVertex, adjoint: Bool, mut pcg: PCG32, ref sd: SceneView,
    wavelengths: SampledWavelengths,
    mut dvcm: Float32, mut dvc: Float32, mut dvm: Float32,
    mis_vc_weight_factor: Float32, eta_x: Float32,
) -> LobeSample:
    """Scatter at a stored vertex through THE lobe sampler (bxdf.mojo's
    lobe_sample) and advance VCM's carries. `adjoint` on the light subpath.

    Both subpaths' diffuse, diffusetransmission and coateddiffuse branches
    used to write this out by hand -- four copies of the lobe choice, the
    densities and the carry update, one of which still held its own
    transmittance lookup and a pre-bump cosine in the reverse density.
    Taking the vertex's own LobeCtx means the carries describe exactly the
    densities lobe_eval hands connect, merge and NEE. A delta event (a
    smooth coat's mirror) zeroes dVCM and scales the rest by its cosine."""
    var s = lobe_sample(_vertex_ctx(v, adjoint), pcg.next_float(), pcg.next_float(), pcg.next_float(), pcg.next_float(),
        LobeTables(sd.materials, sd.curves, sd.measuredBrdfs),
        sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y,
        sd.spectral.cie_z, sd.spectral.d65, wavelengths)
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
    (emission, total area, its pick slot -- area lights come first, in index
    order -- and the cosine on the lit side). All of it comes from the SHARED
    resolvers (shading.mojo's area_light_hit_cos / curve_light_hit), not from
    _geom_normal and areaLights[id1]: the raw winding normal disagreed with
    the side the light sampler emits from, and for a curve id1 is not an
    AreaLight index at all."""
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
def _resolve_mix(ref sd: SceneView, mut pcg: PCG32, mut mat: Material, mut mat_idx: Int):
    """Mix material: stochastically resolve to one of its two sub-materials
    (mirrors shading.mojo's shade_mix; a mix of mix collapses to diffuse).
    Draws from `pcg` only for a mix, and keeps `mat_idx` in sync with the
    resolved material (hair needs the real index to re-fetch at connect time)."""
    if mat.type != MatKind.mix:
        return
    var mix_idx1 = Int(mat.tex_idx & Int32(0xFFFF))
    var mix_idx2 = Int((mat.tex_idx >> 16) & Int32(0xFFFF))
    var mix_chosen = mix_idx2 if pcg.next_float() < mat.roughU else mix_idx1
    mat = sd.materials[unsafe_offset=mix_chosen]
    mat_idx = mix_chosen
    if mat.type == MatKind.mix:
        mat.type = MatKind.diffuse


@always_inline
def _coated_conductor_scatter[use_gpu: Bool](
    ref sd: SceneView, mut pcg: PCG32, mat: Material, inter: Intersection,
    hit: Vec3f, ray_dir: Vec3f, wo: Vec3f, cone_w: Float32,
) -> Tuple[BxDFSample, Vec3f, Vec3f]:
    """Geometry and sample shared by both subpaths' coated_conductor branch:
    (sample, perturbed face-forwarded normal, geometric normal before bump/
    normal maps -- the stored vertex keeps the latter, _connect's solid-angle
    -> area conversions are built on it, see BDPTVertex.shading_normal).

    bxdf_sample_coated_conductor's own u_split picks coat-vs-conductor lobe;
    its delta (coat-reflect) branch is treated like a mirror elsewhere (no
    stored vertex), its glossy branch like a plain conductor (approximation:
    connections reuse conductor's GGX eval/f0, ignoring the coat's own
    (1-f_coat) attenuation -- the same one shading.mojo's sampling makes)."""
    var gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit)
    if dot(gn, ray_dir) > Float32(0): gn = gn * Float32(-1)
    var gn_geo = gn
    gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes, sd.instances,
        gn, gn, hit, ray_dir, cone_w, sd.camFp,
        sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
    gn = face_toward(gn, -ray_dir)   # pbrt two-sided reflection, see face_toward
    var frm = Frame.from_z(Vec3f(gn[0], gn[1], gn[2]))
    var gc = GeomContext(
        normal=gn, geo_normal=gn, hit_point=hit, wo=wo,
        tangent=Vec3f(frm.x.x, frm.x.y, frm.x.z),
        bitangent=Vec3f(frm.y.x, frm.y.y, frm.y.z),
        alb=mat.albedo, pixel_uv=Float32(0),
    )
    var uc1 = pcg.next_float(); var uc2 = pcg.next_float()
    var ior = mat.emission.r if mat.emission.r > Float32(1) else Float32(1.5)
    var usplit = pcg.next_float()
    return (bxdf_sample_coated_conductor(gc, mat, ior, usplit, uc1, uc2), gn, gn_geo)


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
    mut ro: Point3f,
    mut rd: Vec3f,
    mut beta: SpectralSample,
    mut total: SpectralSample,
    mut total_merge: SpectralSample,
    # Benchmark instrumentation only -- see
    # _bdpt_merge_from_cache's matching docstring paragraph. Threaded through
    # unconditionally (defaults live below on the params that already had
    # them; adding defaults to `mut` params here would need a mutable
    # lvalue at every call site anyway, so these are required like the
    # other running accumulators above).
    mut visit_naive: Int32,
    mut visit_footprint: Int32,
    mut visit_thin: Int32,
    mut first_alb: RGB,
    mut n_verts: Int,
    mut n_delta: Int,   # delta bounces so far; they count toward maxdepth (see _vcm_depth)
    mut n_bounces: Int,
    mut cur_med_idx: Int32,
    mut dvcm_carry: Float32,
    mut dvc_carry: Float32,
    mut dvm_carry: Float32,
    # Kind of the PREVIOUS vertex on this subpath (True = volume) -- see the
    # light-path bounce function's matching parameter for the derivation.
    mut prev_was_volume: Bool,
    mut last_bsdf_pdf: Float32,
    mut mis_null_dist: Float32,
    mut current_dielectric_ior: Float32,
    mut previous_dielectric_ior: Float32,
    wavelengths: SampledWavelengths,
    # Path length along the camera ray's specular chain, the ray cone standing
    # in for pbrt's camera differentials (footprint.mojo); only meaningful
    # while no vertex is stored (n_verts == 0), exactly when pbrt still has
    # differentials. Same role as PathState.cone_len.
    mut cone_len: Float32,
    # _VCM_CAMIS camera state (vcm_camis.CamisCamCarry/CamisCamRecord), the
    # records of every vertex stored so far; see _bdpt_trace_camera_and_connect.
    # Untouched when the hybrid is compiled out.
    mut camis: CamisCamCarry,
    mut camis_recs: Array[CamisCamRecord, _CAMIS_CAM_RECS],
    # Task #163 stage 5: when set, the DIFFUSE branch's connect step queues
    # its shadow rays into these buffers (one _BDPT_MAX_VERTS-sized slice
    # per pixel, indexed like the LVC itself) instead of resolving them
    # inline via software BVH -- see _bdpt_connect_to_cache_deferred's own
    # docstring. Every OTHER material branch's connect (and every NEE/MNEE
    # shadow ray anywhere) is UNCHANGED regardless of this flag -- a
    # deliberately narrow first slice, not "all shadow rays on Vulkan RT".
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
    """Task #163 stage 4: wavefront-staged variant of ONE bounce iteration of
    `_bdpt_trace_camera_and_connect`'s main loop (bdpt_*.mojo:887-1684), the
    camera-path counterpart to `_bdpt_light_path_bounce` (see that function's
    docstring for the general rationale -- same design, applied to the
    larger of the two subpath loops). `scratch` is STILL a live parameter
    here, unlike the light-path bounce function: shadow rays (NEE/connect/
    merge/MNEE, the majority of ray casts per sample) stay on the existing
    software-BVH `any_hit`-style tests inside `_bdpt_nee_contribute`/
    `_bdpt_connect_to_cache`/`_bdpt_merge_from_cache`/
    `_bdpt_mnee_diffuse_area_light`, unchanged -- only this function's own
    PRIMARY/bounce ray intersect is externalized to the caller, per the
    user-confirmed stage 4 scope (AskUserQuestion, 2026-07-12: "shadow rays
    stay on software BVH in this first cut").

    The body below is a byte-for-byte copy of the original loop body
    (mechanically extracted, not retyped), with the same class of
    control-flow-shape changes `_bdpt_light_path_bounce` used: every
    outer-loop `break` (9 sites: direct hit on an emissive analytic sphere,
    direct hit on an area-light triangle/curve, coat total-internal-
    reflection-like case, coat recycling walk absorbed, conductor invalid
    sample, 3x measured-BxDF failure, the unhandled-material-type
    catch-all) becomes `return False`; the loop's 2 outer `continue` sites
    (volume free-flight scatter; rough-coat reflect) become `return True`;
    the coat recycling walk's OWN inner `for depth in
    range(MAX_COAT_DEPTH):` loop keeps its own real `break` statements
    (RR kill, exit-direction found) untouched, since those exit the INNER
    walk loop, not this function. The original miss-handling block (evaluate
    infinite-light MIS against `last_bsdf_pdf`, add to `total`) is preserved
    verbatim as this function's own top-of-body miss check, since the
    caller now supplies `inter` pre-computed instead of this function
    calling `traverse_bvh2_core`/`test_spheres` itself.

    `total`/`first_alb`/`beta`/`n_verts`/`n_bounces`/`cur_med_idx`/
    `dvcm_carry`/`dvc_carry`/`dvm_carry`/`last_bsdf_pdf` are `mut`
    PARAMETERS instead of function-local variables persisted implicitly
    across loop iterations -- the caller is a persistent per-camera-path
    state struct that survives across separate kernel launches, one call to
    this function per bounce. `total`/`first_alb` in particular ACCUMULATE
    across calls (unlike the light-path side, which has no equivalent
    running accumulator) -- the host loop reads them from the state after
    the path goes inactive, exactly like it would read a local variable
    after the original single-function loop returned.

    Returns True if the camera path should continue to another bounce,
    False if it has terminated (escaped the scene, hit an emitter, exhausted
    a material's valid-sample conditions, or hit an unhandled material
    type)."""
        if n_verts >= _BDPT_MAX_VERTS:
            return False   # mirrors the original loop's top-of-iteration guard
        if inter.hit == Int8(0):
            for inf_i in range(Int(sd.infiniteLightCount)):
                var ilight = sd.infiniteLights[unsafe_offset=inf_i]
                var (Le, pdf_light_here) = _eval_infinite_light_and_pdf(ilight, rd, ro.to_simd())
                var mis_w = Float32(1)
                if last_bsdf_pdf == PDF_DROP_DIRECT:
                    # A scatter whose direct term NEE already reported (the
                    # rough-coat exit ray): indirect only. Same contract PT's
                    # handlers keep; bdpt's gate below let this through as a
                    # zero pdf at near-full weight, double-counting against
                    # the coat's full-weight per-iteration NEE.
                    mis_w = Float32(0)
                elif n_verts > 0 and pdf_light_here > Float32(0):
                    # NOT `last_bsdf_pdf >= 0`, which is what this used to be.
                    # That test excluded every path whose last event was DELTA
                    # (the dielectric branch sets last_bsdf_pdf = -1), handing
                    # the escape a hard-coded weight of 1 -- while the diffuse
                    # vertex further back had already reported the same
                    # transport through connect/merge. Both counted in full:
                    # Scenes/furnace/dielectric-inert.pbrt read 1.1708 at the
                    # default light-path count and converged to 1.9665, against
                    # an analytic 1.0.
                    #
                    # No specular special case is needed, and SmallVCM does not
                    # have one either (GetLightRadiance weights unconditionally
                    # past the first hit): a delta bounce sets dvcm_carry = 0,
                    # and a zero dVCM already removes the NEE term from the
                    # weight below on its own, while dVC keeps discounting
                    # connect and merge -- which is the part the old gate threw
                    # away. `n_verts > 0` now carries the one case that IS
                    # unweighted: NO REAL VERTEX HAS BEEN STORED YET, so no
                    # other strategy can have reported this transport -- a
                    # primary ray straight into the environment, or one that
                    # only ever hit DELTA surfaces (mirror, glass), neither of
                    # which stores a vertex. That is a weight-1 case, not a
                    # "no pdf" case, and conflating the two is what hid the bug.
                    #
                    # It must be n_verts and NOT n_bounces: the diffuse branch
                    # stores a vertex without incrementing n_bounces, so an
                    # n_bounces test reads an ordinary diffuse path as a
                    # primary ray and hands every furnace escape weight 1 --
                    # measured, every VCM env-lit furnace cell went ~1.01 to
                    # ~1.45 before this was corrected.
                    # Balance heuristic over EVERY strategy, matching the rest
                    # of this file -- vcm_env_escape_weight, derived exactly in
                    # Scenes/vcm_env_mis_derivation.py. The power heuristic it
                    # replaces gave the escape 0.80 where balance over all
                    # strategies gives 0.54, leaving no share for merging or
                    # t=1. dvcm/dvc_carry are the POST-SCATTER carries from the
                    # last real vertex (this handler runs before the per-hit
                    # arrival update), which is exactly what that weight wants.
                    var (_c_esc, r_esc) = _scene_bounding_sphere(sd)
                    var emis_esc = pdf_light_here * light_path_pick_pdf(sd, _vcm_infinite_slot(sd, inf_i)) / max(PI * r_esc * r_esc, Float32(1e-12))
                    mis_w = vcm_env_escape_weight(pdf_light_here, emis_esc,
                                                  dvcm_carry, dvc_carry)
                total += beta * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (Le).r, (Le).g, (Le).b, wavelengths) * mis_w
            return False   # nothing hit -- path escapes the scene
        var t_hit = inter.tHit
        var ray_dir = rd.to_simd()
        if n_verts == 0:
            cone_len += t_hit
        var vcm_cone_w = sd.camFp.cone_spread * cone_len if n_verts == 0 else Float32(-1.0)

        # VCM Stage 2b: distance-squared portion of the per-bounce MIS
        # correction -- see _bdpt_trace_light_path's matching comment
        # (project_vcm_stage2_mis_derivation memory). The eye subpath's
        # origin is always "finite" (a real camera position), so the
        # correction applies unconditionally here (unlike the light side's
        # is_finite_origin check).
        dvcm_carry *= t_hit * t_hit

        # Volume free-flight
        if has_med and Int(cur_med_idx) >= 0:
            comptime if _VCM_CAMIS:
                # A medium segment (collision or pass-through, whose 1/FF
                # enters dVCM) is outside CAMIS's Class: S0's harness models
                # vacuum edges only. Legacy weights stay exact for it.
                camis.in_class = False
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
                # Chromatic collision weight -- the ratio to the sampled (red)
                # channel. Without it a chromatic medium is biased at every
                # scattering event; exactly 1 for a grey medium. Applied to
                # `beta` BEFORE the vertex stores it, so the stored throughput
                # is the one arriving at the vertex.
                beta *= spectral_free_flight_weight(med, ff, t_hit, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                # Volume scatter vertex
                var sp = ro + rd*ff.t_free
                # dVCM's Jacobian above was applied with t_hit (distance to
                # the SURFACE the ray was cast toward); the real arrival
                # distance here is ff.t_free -- see the light-side bounce
                # function's matching comment for the full derivation.
                dvcm_carry *= (ff.t_free * ff.t_free) / max(t_hit * t_hit, Float32(1e-20))
                dvcm_carry *= Float32(1.0) / max(ff.pdf, Float32(1e-30))
                if not prev_was_volume:
                    dvc_carry *= Float32(1.0) / max(ff.sig_t, Float32(1e-20))
                var v = _null_vertex()
                v.pos = sp
                # `beta`, NOT beta*albedo -- a vertex's beta is the throughput
                # ARRIVING at it; connect time applies the vertex's own
                # response separately (v.alb/(4*pi)). Baking albedo in here
                # too double-counted it (see docs/09_volumetric_media.md,
                # "VCM/BDPT volume connections": 0.785x -> 0.983x fix).
                v.beta = beta
                v.alb = ff.albedo
                v.is_surface = Int32(0); v.is_delta = Int32(0)
                v.pdf_fwd = ff.pdf   # the density actually sampled from; under hero-wavelength MIS this is a lane MIXTURE, not sig_t's lone exponential
                v.med_idx = cur_med_idx
                v.wavelengths = wavelengths
                # ARRIVAL carries -- merge/connect below read these off `v`
                # (it is never stored in the LVC on the camera side, only
                # used transiently for this bounce's own merge/connect).
                v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                if n_verts == 0: first_alb = ff.albedo
                # NOT `n_verts += 1`. _BDPT_MAX_VERTS is the light-vertex
                # CACHE's per-path slot count, and the camera subpath stores
                # nothing in that cache (only _bdpt_light_path_bounce calls
                # _bdpt_store_lvc_vertex). Counting volume scatters against it
                # capped a camera walk at 10 scatter events -- far too few for
                # a dense medium, where a diffusive walk needs tens of orders
                # to converge (the path tracer needs maxdepth ~33 on the same
                # scene). Volume scatters are bounded by _BDPT_MAX_DEPTH loop
                # iterations instead, which is what this branch's own "no
                # vertex stored this bounce" return comment already implied.
                # Volume scatter is isotropic: cos_out=1, pdf_fwd=pdf_rev=
                # 1/(4pi), cos_over_pdf=4pi -- the same vcm_scatter_carries
                # every surface branch uses (Scenes/vcm_volume_mis_derivation.py).
                var eta_v = _vcm_eta_at(sd, v, mis_vm_weight_factor)
                (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries[dvc0=_VCM_CAMIS](
                    dvcm_carry, dvc_carry, dvm_carry,
                    Float32(4.0) * PI, INV_FOUR_PI, INV_FOUR_PI,
                    mis_vc_weight_factor, eta_v)
                prev_was_volume = True
                if path_len > 0:
                    # BUG FIX (found investigating volumetric-caustic's
                    # colored-blob fireflies): merge is only safe to run
                    # unconditionally alongside connect when BOTH already
                    # carry a real per-candidate MIS weight that makes their
                    # SUM correct (see this file's opening VCM comment).
                    # Volume vertices are out of _bdpt_vertex_mis_scoped's
                    # scope, so _bdpt_merge_from_cache's own weight defaults
                    # to 1 -- summing merge(w=1) + connect(w=1) here would
                    # double the correct answer (E[merge]+E[connect] = 2I,
                    # not I), exactly the double-counting the module's
                    # original Stage-1 stochastic-pick design was built to
                    # avoid. Only call merge when this vertex is actually
                    # MIS-scoped; otherwise connect alone (unweighted) is
                    # already the complete, correct estimate -- Stage 2b's
                    # original, verified behavior for out-of-scope kinds.
                    if _bdpt_vertex_mis_scoped(v):
                        total_merge += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor, n_verts + n_delta, camis, camis_recs, lvc_camis, n_light_paths_f, visit_naive=visit_naive, visit_footprint=visit_footprint, visit_thin=visit_thin)
                    total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor, n_verts + n_delta, camis, camis_recs, lvc_camis, n_light_paths_f)
                # Volume-scatter NEE: distant/point/sphere/infinite lights.
                # _bdpt_connect_to_cache above only reaches AREA lights -- the
                # only kind _bdpt_light_path_init seeds the light-vertex cache
                # from (its own `n_lights` count is area+distant+infinite+point,
                # but only area lights get a real lv0 origin vertex placed in
                # the cache; see its docstring). Every other light type
                # therefore illuminated a scatter vertex only through the
                # tiny-probability event of an isotropic-scattered ray
                # randomly re-hitting it (the sphere case of which was ALSO
                # unweighted until the fix just above this block) -- both far
                # too dark and far too noisy, the same failure this file's own
                # docs/09_volumetric_media.md chapter describes for the
                # wavefront integrator's pre-fix state. Mirrors gpu.mojo's
                # _volume_nee_light (that integrator's equivalent): phase
                # (isotropic, INV_FOUR_PI -- this file's volume scatter is
                # hardcoded isotropic, see the uniform-sphere sample just
                # below, so g plays no role here yet, a separate pre-existing
                # gap) x albedo x Li x reciprocal power-heuristic MIS / pdf.
                # See project_sphere_light_nee_bug memory, "STILL OPEN: sphere
                # lights + media under --vcm read 0.66x".
                # Albedo is a per-channel COEFFICIENT (sigma_s/sigma_t), not a
                # reflectance -- band-pick it. spec_refl was used here, and
                # RGB(a,a,a) does not upsample to a in every lane, so even a
                # grey medium picked up a D65-shaped tint (see
                # docs/02_spectra_and_color.md, "A coefficient is not a color").
                var alb_spec_v = rgb_bands_to_spectral_sample((ff.albedo).r, (ff.albedo).g, (ff.albedo).b, wavelengths)
                var phase_alb_v = alb_spec_v * INV_FOUR_PI
                for li_v in range(_bdpt_simple_light_count(sd)):
                    var ls_v = _bdpt_sample_simple_light(sd, li_v, sp.to_simd(), pcg)
                    if ls_v.valid and ls_v.pdf > Float32(0):
                        var li_spec_v = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (ls_v.Li).r, (ls_v.Li).g, (ls_v.Li).b, wavelengths)
                        var mis_v = Float32(1) if ls_v.is_delta else power_heuristic(ls_v.pdf, INV_FOUR_PI)
                        var w_v = phase_alb_v * li_spec_v * (mis_v / ls_v.pdf)
                        total += _bdpt_nee_contribute(beta, w_v, ls_v, sp, Vec3f(Float32(0)), cur_med_idx, sd, scratch, wavelengths, Float32(0))
                for inf_v in range(Int(sd.infiniteLightCount)):
                    var ls_infv = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_v], Point2f(pcg.next_float(), pcg.next_float()))
                    if ls_infv.valid and ls_infv.pdf > Float32(0):
                        var li_spec_infv = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (ls_infv.Li).r, (ls_infv.Li).g, (ls_infv.Li).b, wavelengths)
                        var mis_infv = Float32(1) if ls_infv.is_delta else power_heuristic(ls_infv.pdf, INV_FOUR_PI)
                        var w_infv = phase_alb_v * li_spec_infv * (mis_infv / ls_infv.pdf)
                        total += _bdpt_nee_contribute(beta, w_infv, ls_infv, sp, Vec3f(Float32(0)), cur_med_idx, sd, scratch, wavelengths, Float32(0))
                # Continuation beta = prev × alb_s (same as stored vertex beta)
                beta *= spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (ff.albedo).r, (ff.albedo).g, (ff.albedo).b, wavelengths)
                var u1 = pcg.next_float(); var u2 = pcg.next_float()
                var cosT = Float32(2)*u1 - Float32(1)
                var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                var phi  = Float32(2)*PI*u2
                rd = Vec3f(sinT*cos(phi), sinT*sin(phi), cosT)
                ro = sp + rd*Float32(0.0002)
                last_bsdf_pdf = _VOL_PHASE_HIT   # see the sentinel's definition
                mis_null_dist = Float32(0)     # this vertex is the new origin
                return True   # volume free-flight scatter: no vertex stored this bounce, path continues
            else:
                # Pass-through weight (a per-channel RATIO -- see
                # sample_homogeneous_free_flight; the sampled channel's
                # transmittance is already carried by the pass-through
                # probability, so multiplying the FULL Beer-Lambert factor
                # here double-counted it).
                beta *= spectral_free_flight_weight(med, ff, t_hit, wavelengths, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                # MIS: dVCM must equal 1/P(prev -> this vertex), and reaching a
                # SURFACE through a medium carries a survival factor
                # FF = exp(-sigma_t * d) that the vacuum recursion above
                # (`dvcm_carry *= t_hit*t_hit`) does not include. So dVCM takes
                # 1/FF. dVC needs nothing here: its second-order term carries
                # FF_a/FF_b, and for a surface->surface edge both endpoints
                # contribute the same survival factor, so the ratio is exactly 1.
                # Verified against a brute-force enumeration of every strategy's
                # full path pdf -- see Scenes/vcm_surface_ff_gap.py, where the
                # uncorrected recursion's worst relative error runs 8e-16 in
                # vacuum but 0.27 at sigma_t=0.1, 12 at 1.0 and 1e5 at 4.0.
                # Identically 1 when sigma_t = 0, so this is inert outside media.
                var ff_exp = -log(max(ff.pdf, Float32(1e-30)))   # optical depth of the density actually sampled from (see FreeFlight.pdf)
                if ff_exp > Float32(60.0): ff_exp = Float32(60.0)  # defensive: e^-60 pass-through never occurs
                dvcm_carry *= exp(ff_exp)
                # ...unless the PREVIOUS vertex was itself a volume scatter --
                # then this edge changes kind (volume -> surface) and dVC
                # picks up a factor of sigma_t, not 1 (see the light-side
                # bounce function's matching comment).
                if prev_was_volume:
                    dvc_carry *= ff.sig_t
                prev_was_volume = False

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[unsafe_offset=mat_idx]
        var hit = ro + rd*t_hit
        var eta_x = mis_vm_weight_factor * _vcm_eta_scale(sd, hit)   # merging's MIS density HERE

        # Direct hit on an emissive analytic sphere — checked BEFORE material
        # dispatch since the sphere's own material is often an inert
        # placeholder (e.g. pbrt's "Null" material on AreaLightSource
        # spheres), so mat.type never reflects that this primitive emits;
        # Sphere.isAreaLight is the only way to know. MIS-weighted against
        # last_bsdf_pdf — same bookkeeping already used for infinite-light
        # miss-escape, since a competing NEE strategy toward this SAME
        # sphere may have already been taken at the PREVIOUS vertex (see
        # this function's own sphere-light NEE blocks). No facing check —
        # an analytic sphere is only ever hit from outside, always the
        # front (emitting) face, unlike a one-sided triangle area light.
        if inter.primId.type == Int8(4):
            var sph_hit = sd.spheres[unsafe_offset=Int(inter.primId.id1)]
            if sph_hit.isAreaLight != Int8(0):
                var mis_w_sph_hit = Float32(1)
                if last_bsdf_pdf == PDF_DROP_DIRECT:
                    # Direct term already reported by NEE: indirect only, so
                    # this hit contributes nothing. Was `pass`, which left the
                    # weight at 1 -- see the area-light handler below.
                    mis_w_sph_hit = Float32(0)
                elif last_bsdf_pdf >= Float32(0) or (n_verts > 0 and last_bsdf_pdf == PDF_DELTA_FULL):
                    # The competing NEE was taken at the vertex that GENERATED
                    # this ray, so its cone pdf is measured from there -- not
                    # from `hit`, ON the sphere, where the cone is the whole
                    # hemisphere (that bug once counted every sphere light
                    # twice), and not from `ro` either, which is the last null
                    # interface crossed rather than the vertex.
                    var x_nee = ro.to_simd() - ray_dir * mis_null_dist
                    var pdf_cone = sphere_light_cone_pdf(sph_hit, x_nee)
                    if dvcm_carry > Float32(0) or dvc_carry > Float32(0):
                        # SmallVCM's GetLightRadiance, the balance weight over
                        # EVERY strategy now that sphere lights start light
                        # paths too (t=1 and merging compete here, not just
                        # NEE). Same form as the area-light hit below:
                        #     wCamera = (directPdfA * dVCM + emissionPdfW * dVC) / cos_l
                        # with directPdfA the cone pdf in area measure,
                        # pdf_cone * cos_l / t^2 -- its t^2 cancels the one
                        # already in dVCM -- and emissionPdfW the light
                        # path's cos_l / (pi * 4 pi r^2 * n_lights). After a
                        # delta bounce dVCM is 0 and only dVC speaks, as for
                        # the area light (17ccebfc).
                        var aw_hit = Float32(4) * PI * sph_hit.radius * sph_hit.radius / light_path_pick_pdf(sd, _vcm_sphere_slot(sd, Int(inter.primId.id1)))
                        var w_cam_sph = pdf_cone * dvcm_carry / max(t_hit * t_hit, Float32(1e-12)) + dvc_carry / (PI * aw_hit)
                        mis_w_sph_hit = Float32(1) / (Float32(1) + w_cam_sph)
                    elif last_bsdf_pdf < Float32(0):
                        pass   # delta bounce off a vertex with placeholder carries: unweighted, as before
                    elif pdf_cone > Float32(0):
                        # A vertex outside real per-vertex MIS (zero
                        # placeholder carries) keeps the two-strategy weight.
                        # No 1/count factor -- see _sample_sphere_light_nee.
                        mis_w_sph_hit = power_heuristic(last_bsdf_pdf, pdf_cone)
                elif last_bsdf_pdf == _VOL_PHASE_HIT:
                    # An isotropic-phase-sampled ray landing on a sphere light,
                    # weighed against the volume vertex's own NEE (volume
                    # vertices stay outside real per-vertex MIS).
                    var pdf_cone_v = sphere_light_cone_pdf(sph_hit, ro.to_simd() - ray_dir * mis_null_dist)
                    if pdf_cone_v > Float32(0):
                        mis_w_sph_hit = power_heuristic(INV_FOUR_PI, pdf_cone_v)
                total += beta * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (sph_hit.emission).r, (sph_hit.emission).g, (sph_hit.emission).b, wavelengths) * mis_w_sph_hit
                return False   # direct hit on emissive analytic sphere -- terminates the path

        _resolve_mix(sd, pcg, mat, mat_idx)

        if mat.type == MatKind.area_light:
            # Direct hit on a triangle/curve area light — same MIS-against-
            # last_bsdf_pdf treatment as the sphere case above. id1 is the
            # AreaLight index directly for a type==3 (area-light-triangle)
            # hit, per pbrt_parser.mojo's own PrimId encoding.
            # Which light, what it emits, and which side of it is lit all come
            # from the SHARED resolvers (shading.mojo's area_light_hit_cos /
            # curve_light_hit), not from _geom_normal and areaLights[id1]. See
            # their header: the raw winding normal disagreed with the side the
            # light sampler emits from, and for a curve id1 is not an
            # AreaLight index at all.
            var is_curve = inter.primId.type == Int8(5)
            var (al_emission, al_area, al_slot, cos_l_hit) = _area_light_hit(sd, inter, mat, ray_dir)
            # A curve is a closed tube, so an unweighted (primary/delta) hit is
            # always on its outside -- credited regardless of the reconstructed
            # radial cosine, exactly as the path tracer does. A triangle light
            # is one-sided.
            if cos_l_hit > Float32(0) or (is_curve and last_bsdf_pdf < Float32(0) and last_bsdf_pdf != PDF_DROP_DIRECT):
                var mis_w_al_hit = Float32(1)
                if last_bsdf_pdf == PDF_DROP_DIRECT:
                    # NEE already reported this ray's direct term (the rough
                    # coat's exit ray) -- geometry.mojo's contract for this
                    # sentinel is "contribute indirect only", and an emitter
                    # hit IS the direct term. This used to `pass`, leaving the
                    # weight at 1: a full double count on every coat exit that
                    # landed on an emitter, which the env miss handler had
                    # already been fixed for.
                    mis_w_al_hit = Float32(0)
                elif last_bsdf_pdf >= Float32(0) or (n_verts > 0 and last_bsdf_pdf == PDF_DELTA_FULL):
                    # A DELTA last bounce (glass, mirror: last_bsdf_pdf = -1)
                    # is weighted too once a real vertex is stored -- the
                    # env escape's n_verts rule, same reason: that vertex
                    # already reported this transport through t=1 and
                    # merging, and the delta bounce's zero dVCM with a live
                    # dVC is exactly what discounts them. Gating on
                    # `>= 0` alone handed every caustic seen through glass
                    # weight 1: a mesh lamp over a glass sphere read 2.00x
                    # pbrt in the caustic.
                    if cos_l_hit > Float32(0) and al_area > Float32(0) and (dvcm_carry > Float32(0) or dvc_carry > Float32(0)):
                        # SmallVCM's GetLightRadiance, the balance weight over
                        # EVERY strategy -- not the 2-strategy power heuristic
                        # this was, which knew nothing of merging, t=1 or the
                        # light-path connections and so could not partition
                        # unity with them (Scenes/vcm_area_mis_derivation.py:
                        # as-coded weights summed to 0.95-0.98, dark wherever
                        # area-light direct light matters). The carries are the
                        # post-scatter ones with d^2 already applied; arriving
                        # at the light divides by its cosine.
                        #     wCamera = directPdfA * dVCM + emissionPdfW * dVC
                        # with the light path's own densities: a uniform pick
                        # over every light, then uniform area (area_weight).
                        var p_a = light_path_pick_pdf(sd, al_slot) / al_area
                        var emission_pdf_w = p_a * cos_l_hit * INV_PI
                        comptime if _VCM_CAMIS:
                            if camis.in_class:
                                # This hit is not a stored vertex (the path
                                # terminates), so camis is still the carry
                                # left by the LAST scatter -- arrive at the
                                # emitter locally, without mutating it (no
                                # further bounce reads it): the same
                                # cos/t_hit divide camis_cam_arrive itself
                                # does, matching dvcm_carry/dvc_carry's own
                                # /cos_l_hit below.
                                var dvcm_arr_al = dvcm_carry / max(cos_l_hit, Float32(1e-6))
                                var log_py_arr_al = camis.log_py + camis_clamp_log_p(
                                    camis.log_k, -log(dvcm_arr_al))
                                var log_g_prev_arr_al = log(camis.cos_out_prev / max(t_hit * t_hit, Float32(1e-12)))
                                var cam_arr_al = CamisCamCarry(camis.in_class, camis.cut, camis.log_k,
                                    log_py_arr_al, camis.rc, log_g_prev_arr_al, Float32(0))
                                mis_w_al_hit = camis_eval_emission_hit[_CAMIS_CAM_RECS](
                                    cam_arr_al, camis_recs, n_verts,
                                    dvcm_arr_al, dvc_carry / max(cos_l_hit, Float32(1e-6)),
                                    dvm_carry / max(cos_l_hit, Float32(1e-6)), log(_vcm_keep(sd, hit)),
                                    p_a, emission_pdf_w, log(cos_l_hit * INV_PI),
                                    n_light_paths_f, True, _CAMIS_FORCE_C1)
                            else:
                                var w_cam_hit_off = (p_a * dvcm_carry + emission_pdf_w * dvc_carry) / cos_l_hit
                                mis_w_al_hit = Float32(1) / (Float32(1) + w_cam_hit_off)
                        else:
                            var w_cam_hit = (p_a * dvcm_carry + emission_pdf_w * dvc_carry) / cos_l_hit
                            mis_w_al_hit = Float32(1) / (Float32(1) + w_cam_hit)
                    elif last_bsdf_pdf < Float32(0):
                        pass   # delta bounce off a vertex with placeholder carries: unweighted, as before
                    elif cos_l_hit > Float32(0) and al_area > Float32(0):
                        # A vertex outside real per-vertex MIS (its carries are
                        # zero placeholders) keeps the old two-strategy weight.
                        var dist2_hit = t_hit * t_hit
                        var n_area_hit = Float32(max(Int(sd.areaLightCount), 1))
                        var pdf_light_al = dist2_hit / (cos_l_hit * n_area_hit * al_area)
                        mis_w_al_hit = power_heuristic(last_bsdf_pdf, pdf_light_al)
                    else:
                        mis_w_al_hit = Float32(0)
                elif last_bsdf_pdf == _VOL_PHASE_HIT:
                    # The competing strategy is the volume vertex's s=1
                    # connection to the light-source vertex, whose area pdf is
                    # 1/(total_area * n_lights) with n_lights counting EVERY
                    # light type (that is how _bdpt_light_path_init picks a
                    # light) -- spelled the same way here, since MIS is only
                    # right when both halves agree on the pdf. The distance is
                    # from the scattering vertex, not the current ray origin.
                    var d_vol = t_hit + mis_null_dist
                    var n_lights_hit = Float32(1) / light_path_pick_pdf(sd, al_slot)
                    if cos_l_hit > Float32(0) and al_area > Float32(0):
                        var pdf_light_vol = d_vol * d_vol / (cos_l_hit * n_lights_hit * al_area)
                        mis_w_al_hit = power_heuristic(INV_FOUR_PI, pdf_light_vol)
                    else:
                        mis_w_al_hit = Float32(0)
                total += beta * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al_emission.r, al_emission.g, al_emission.b, wavelengths) * mis_w_al_hit
            return False   # direct hit on an area-light triangle/curve -- terminates the path

        # Every arm of the dispatch below except a null interface is a real
        # scattering event, and the null-interface distance accumulated on
        # the way here has served its purpose. A plain (non-chained) `if`,
        # deliberately: folding this into the elif chain below as its own
        # first arm silently swallowed every other branch (diffuse, conductor,
        # dielectric, interface's own logic, ...) -- an `if` that matches
        # takes over the ENTIRE following elif sequence, which is exactly
        # backwards from "reset unless interface". Caught by the smoketest
        # regressing (cpu-vcm 0.130 -> 0.106 on cornell-box, no medium at
        # all), which is why this comment is here: the failure mode is
        # completely silent otherwise -- it still compiles and still returns.
        if mat.type != MatKind.interface:
            mis_null_dist = Float32(0)

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
            # VCM Stage 2b: finish the per-bounce MIS correction (dist²
            # portion already applied above) -- see
            # _bdpt_trace_light_path's matching comment.
            var cos_fix = abs(dot(-ray_dir, gn))
            (dvcm_carry, dvc_carry, dvm_carry) = vcm_arrival_carries(
                dvcm_carry, dvc_carry, dvm_carry, cos_fix)
            # Real image-texture reflectance (e.g. "texture reflectance" on
            # coateddiffuse) — before this, bdpt_*.mojo always used the flat
            # mat.albedo fallback (material_builder.mojo's own 0.5 grey
            # default for any texture-backed material), silently washing
            # out any textured diffuse/coateddiffuse surface. _tex_lookup
            # itself returns mat.albedo unchanged when mat.tex_idx == -1
            # (no texture), so this is a strict improvement, never a
            # regression, for flat-color materials.
            var eff_alb = mat.albedo
            var (tex_mesh, tv0, tv1, tv2, tex_ok) = _get_tri_verts(inter, sd.meshes)
            if tex_ok and not on_curve:
                eff_alb = _tex_lookup[use_gpu](mat, inter, tv0, tv1, tv2, tex_mesh, sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount),
                    uv_footprint_at_hit(inter, sd.meshes, sd.instances, hit.to_simd(), ray_dir, vcm_cone_w, sd.camFp).width)
            # Bump/normal maps. bdpt_*.mojo applied NONE of them, on either
            # subpath, while the path tracer has since 2026-09 -- a textbook
            # instance of project_pt_only_feature_gaps, and the one that left
            # VCM's two halves standing on different geometry: the camera
            # vertex on a perturbed surface, the light vertex it merges with
            # on a flat one. Footprint: pbrt's, see footprint.mojo.
            # Saved BEFORE the perturbation: the stored vertex keeps this as its
            # GEOMETRIC normal, because _connect's solid-angle -> area pdf
            # conversions are built on it. See BDPTVertex.shading_normal.
            var gn_geo = gn
            if not on_curve:
                gn = apply_surface_maps_at_hit[use_gpu](mat, inter, sd.meshes, sd.instances,
                    gn, gn, hit.to_simd(), ray_dir, vcm_cone_w, sd.camFp,
                    sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
            if not outward:
                gn = face_toward(gn, -ray_dir)   # pbrt two-sided reflection, see face_toward
            var v = _null_vertex()
            v.pos = hit
            v.normal = vec3f(gn_geo)
            v.shading_normal = vec3f(gn)
            v.beta = beta
            v.alb = eff_alb
            v.is_surface = Int32(1)
            # One lobe per material, from bxdf.mojo's lobe_kind_of: everything
            # below evaluates and samples the vertex through lobe_eval and
            # lobe_sample, so this branch never asks which material it is.
            v.mat_kind = lobe_kind_of(mat.type)
            v.mat_idx = Int32(mat_idx)
            v.pdf_bwd = lobe_param_of(mat)   # LobeCtx.param: a conductor's GGX alpha
            # A smooth conductor is a mirror: it scatters, but stores no
            # vertex and starts no strategy.
            v.is_delta = Int32(1) if lobe_is_delta_of(mat) else Int32(0)
            if on_curve:
                v.hair_curve_idx = Int32(inter.primId.id1)
                v.hair_h = inter.u
                v.hair_v = inter.v
            v.pdf_fwd = Float32(1)  # unused by the uniform-subsample estimator
            v.wo = vec3f(-ray_dir)  # VCM Stage 2b: needed for _connect's reverse-pdf eval
            v.med_idx = cur_med_idx
            v.wavelengths = wavelengths
            v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
            if v.is_delta != Int32(0):
                if n_verts + n_delta >= _vcm_depth(sd):
                    return False   # a delta vertex past d starts no strategy either
                n_delta += 1
            if v.is_delta == Int32(0):
                if n_verts == 0: first_alb = eff_alb
                if n_verts + n_delta >= _vcm_depth(sd):
                    return False   # a vertex past d starts no strategy (see _vcm_depth)
                n_verts += 1
                comptime if _VCM_CAMIS:
                    # Before merge/connect: S3's weights at this vertex read
                    # its log P(y) and the finalised records below it.
                    camis_cam_arrive(camis, camis_recs, n_verts - 1, dvcm_carry, cos_fix, t_hit,
                                     _bdpt_vertex_mis_scoped(v))
                # Merging queries the GLOBAL photon grid and does not use this
                # pixel's own paired light path, so unlike the connect below it
                # must NOT be gated on that path having stored anything. It was,
                # and in a white furnace only ~32% of light paths hit the quad at
                # all, so ~68% of pixels skipped merging entirely: the estimator
                # delivered 0.109 against an analytic 0.5.
                total_merge += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor, n_verts + n_delta, camis, camis_recs, lvc_camis, n_light_paths_f, visit_naive=visit_naive, visit_footprint=visit_footprint, visit_thin=visit_thin)
                if path_len > 0:
                    # Task #163 stage 5: the diffuse branch's connect shadow
                    # rays are the single highest-volume, cleanest shadow-ray
                    # call site in VCM (always exactly one ray per stored
                    # light-path vertex, present in every diffuse-heavy scene
                    # regardless of light types used) -- when defer_shadow_rays
                    # is set, queue them into the shadow-ray buffers instead of
                    # resolving inline, for a later batched Vulkan RT dispatch.
                    # Every other material branch's connect (and all NEE/MNEE
                    # shadow rays) stays on the unchanged, inline,
                    # software-BVH _bdpt_connect_to_cache path either way.
                    if defer_shadow_rays:
                        _bdpt_connect_to_cache_deferred(v, sd, lvc, lp_idx, path_len, mis_vm_weight_factor, shadow_rays, shadow_pending, shadow_valid, shadow_seg_med, n_verts + n_delta, camis, camis_recs, lvc_camis, n_light_paths_f)
                    else:
                        total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor, n_verts + n_delta, camis, camis_recs, lvc_camis, n_light_paths_f)

                # NEE to distant/point/sphere/infinite lights, via the shared
                # Light interface (bvh.mojo's LightSample samplers) + BxDF
                # interface (bxdf.mojo's _nee_weight_simple) + this file's own
                # _bdpt_nee_contribute glue (media-aware transmittance test) —
                # replacing 4 formerly hand-inlined blocks also duplicated in
                # the conductor branch below and in _bdpt_trace_light_path. Area
                # lights are NOT covered: the LVC cache-connection strategy
                # above already handles them — see _bdpt_trace_light_path's
                # docstring for why distant/infinite/point/sphere need this
                # separate direct term instead.
                # distant/point/sphere via the shared sampler (pure loop collapse --
                # order was already distant,point,sphere, matching the iterator).
                total += _vcm_nee_surface(sd, v, _vertex_ctx(v), hit, gn_geo, beta, eta_x, dvcm_carry, dvc_carry,
                                          cur_med_idx, scratch, wavelengths, pcg, spawn_eps,
                                          v.mat_kind == LobeKind.diffuse_transmit or v.mat_kind == LobeKind.hair or v.mat_kind == LobeKind.rough_dielectric,
                                          Float32(0))
                # MNEE's receiver evaluates albedo/pi itself (see
                # _bdpt_mnee_diffuse_area_light), so only Lambertian lobes can
                # host it -- a limitation of MNEE, not a material branch.
                if v.mat_kind == LobeKind.lambertian or v.mat_kind == LobeKind.diffuse_transmit:
                    # Real MNEE for area lights behind glass (task #161) -- see
                    # _bdpt_mnee_diffuse_area_light's docstring. Ordinary (non-glass)
                    # area lights are deliberately left to connect/merge, unchanged.
                    total += _bdpt_mnee_diffuse_area_light(sd, hit, gn, eff_alb, beta, pcg, wavelengths)
                    # Sphere-shaped area lights (task #161 follow-up, 2026-07-13):
                    # a completely separate list from sd.areaLights (see
                    # _bdpt_mnee_sphere_light's docstring). Deliberately NOT a
                    # `for` loop -- see that function's own investigation note for
                    # the real GPU codegen bug (CUDA_ERROR_ILLEGAL_ADDRESS) a loop
                    # here triggers on this task's own target scene. Manually
                    # unrolled instead, capped at _MNEE_MAX_SPHERES (currently 4
                    # -- update both together if that constant ever changes).
                    var n_sph_mnee = Int(sd.sphereCount)
                    if 0 < n_sph_mnee:
                        total += _bdpt_mnee_sphere_light(sd, hit, gn, eff_alb, beta, pcg, 0, n_sph_mnee, wavelengths)
                    if 1 < n_sph_mnee:
                        total += _bdpt_mnee_sphere_light(sd, hit, gn, eff_alb, beta, pcg, 1, n_sph_mnee, wavelengths)
                    if 2 < n_sph_mnee:
                        total += _bdpt_mnee_sphere_light(sd, hit, gn, eff_alb, beta, pcg, 2, n_sph_mnee, wavelengths)
                    if 3 < n_sph_mnee:
                        total += _bdpt_mnee_sphere_light(sd, hit, gn, eff_alb, beta, pcg, 3, n_sph_mnee, wavelengths)

            # Scatter through THE lobe sampler -- the same LobeCtx connect,
            # merge and NEE evaluate this vertex with (see _vcm_scatter).
            var sc = _vcm_scatter(v, False, pcg, sd, wavelengths, dvcm_carry, dvc_carry, dvm_carry, mis_vc_weight_factor, eta_x)
            if not sc.valid:
                return False
            comptime if _VCM_CAMIS:
                if v.is_delta != Int32(0) or sc.is_delta:
                    camis.in_class = False   # a mirror vertex or a smooth coat's delta lobe
                else:
                    camis_cam_scatter(camis, camis_recs, n_verts - 1, eta_x, log(_vcm_keep(sd, hit)),
                                      sc.cos_out / sc.pdf_fwd, sc.cos_out, sc.pdf_fwd, sc.pdf_rev)
            rd = vec3f(sc.wi)
            if on_curve:
                ro = hit + vec3f(gn_geo) * (spawn_eps if dot(sc.wi, gn_geo) >= Float32(0) else -spawn_eps)
            else:
                ro = hit + rd*Float32(0.0002)
            beta *= sc.weight
            last_bsdf_pdf = Float32(-1) if sc.is_delta else sc.pdf_fwd

        elif mat.type == MatKind.coated_conductor:
            var wo_c = (-rd).to_simd()
            var (bs_c, gn_c, gn_c_geo) = _coated_conductor_scatter[use_gpu](
                sd, pcg, mat, inter, hit.to_simd(), ray_dir, wo_c, vcm_cone_w)
            if bs_c.is_valid == Int8(0):
                return False   # conductor/coated_conductor sample invalid
            var alpha_c = max(mat.roughU, mat.roughV)
            # VCM Stage 2d: rough conductor DOES have a real standalone pdf
            # (bxdf_pdf_conductor_ggx, the same Heitz 2018 VNDF density
            # bxdf_sample_conductor's glossy branch itself samples from) --
            # finish the per-bounce MIS correction the same way diffuse
            # does (dist² portion already applied above).
            var cos_fix_c = abs(dot(-ray_dir, gn_c))
            if cos_fix_c > Float32(1e-6):
                dvc_carry /= cos_fix_c
                dvm_carry /= cos_fix_c
            if not bxdf_is_delta(bs_c.flags):
                var v = _null_vertex()
                v.pos = hit
                v.normal = vec3f(gn_c_geo)
                v.shading_normal = vec3f(gn_c)
                v.beta = beta
                v.alb = mat.albedo
                v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = LobeKind.ggx
                v.pdf_bwd = alpha_c
                v.wo = vec3f(wo_c)
                v.pdf_fwd = Float32(1)
                v.med_idx = cur_med_idx
                v.wavelengths = wavelengths
                v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                if n_verts == 0: first_alb = mat.albedo
                if n_verts + n_delta >= _vcm_depth(sd):
                    return False   # a vertex past d starts no strategy (see _vcm_depth)
                n_verts += 1
                comptime if _VCM_CAMIS:
                    camis_cam_arrive(camis, camis_recs, n_verts - 1, dvcm_carry, cos_fix_c, t_hit,
                                     _bdpt_vertex_mis_scoped(v))
                # Merging queries the GLOBAL photon grid and does not use this
                # pixel's own paired light path, so unlike the connect below it
                # must NOT be gated on that path having stored anything. It was,
                # and in a white furnace only ~32% of light paths hit the quad at
                # all, so ~68% of pixels skipped merging entirely: the estimator
                # delivered 0.109 against an analytic 0.5.
                total_merge += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor, n_verts + n_delta, camis, camis_recs, lvc_camis, n_light_paths_f, visit_naive=visit_naive, visit_footprint=visit_footprint, visit_thin=visit_thin)
                if path_len > 0:
                    total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor, n_verts + n_delta, camis, camis_recs, lvc_camis, n_light_paths_f)

                # Distant/point/sphere/infinite NEE, via the shared Light
                # interface + BxDF interface + _bdpt_nee_contribute glue —
                # replacing 4 formerly hand-inlined blocks also duplicated
                # in the diffuse branch above and _bdpt_trace_light_path.
                # NOTE: the old inline sphere-light block here was missing
                # its cosine factor (computed shadow_dir but never took
                # dot(gn_c, shadow_dir) before dividing by pdf) — a real
                # overbrightness bug, fixed as a side effect of routing
                # through the shared, already-correct _nee_weight_simple.
                var ctx_c = LobeCtx(LobeKind.ggx, True, False, gn_c, wo_c, mat.albedo, Int32(-1), alpha_c,
                                    Float32(0), Int32(-1), Float32(0), Float32(0), True, False)
                total += _vcm_nee_surface(sd, v, ctx_c, hit, gn_c_geo, beta, eta_x, dvcm_carry, dvc_carry,
                                          cur_med_idx, scratch, wavelengths, pcg, Float32(0.0001), False, Float32(0))

            beta *= spec_refl_unbounded(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, (bs_c.f).r, (bs_c.f).g, (bs_c.f).b, wavelengths)
            rd = vec3f(bs_c.wi)
            ro = hit + rd*Float32(0.0002)
            var cos_theta_out_c = abs(dot(bs_c.wi, gn_c))
            if bxdf_is_delta(bs_c.flags):
                last_bsdf_pdf = Float32(-1)  # mirror bounce: no infinite-light NEE done at this vertex
                # VCM: specular bounce -- same dVCM=0/cosine-rescale reset
                # SmallVCM's own delta-bounce case uses (vertexcm.hxx's
                # SampleScattering, kSpecular branch); matches this file's
                # diffuse/dielectric specular handling elsewhere.
                dvcm_carry = Float32(0)
                dvc_carry *= cos_theta_out_c
                dvm_carry *= cos_theta_out_c
                comptime if _VCM_CAMIS:
                    camis.in_class = False
            else:
                # VCM Stage 2d: real non-specular recursive update using the
                # actual VNDF sampling density -- see
                # _bdpt_trace_light_path's matching comment for the general
                # (cosThetaOut/bsdfDirPdfW) form (doesn't simplify to a
                # constant like diffuse's PI does, since GGX's pdf isn't
                # proportional to cos_theta_out).
                var bsdf_dir_pdf_w_c = bxdf_pdf_conductor_ggx(gn_c, wo_c, bs_c.wi, alpha_c)
                last_bsdf_pdf = bsdf_dir_pdf_w_c
                if bsdf_dir_pdf_w_c > Float32(1e-8):
                    var bsdf_rev_pdf_w_c = bxdf_pdf_conductor_ggx(gn_c, bs_c.wi, wo_c, alpha_c)
                    var inv_pdf_c = cos_theta_out_c / bsdf_dir_pdf_w_c
                    (dvcm_carry, dvc_carry, dvm_carry) = vcm_scatter_carries[dvc0=_VCM_CAMIS](
                        dvcm_carry, dvc_carry, dvm_carry, inv_pdf_c, bsdf_dir_pdf_w_c, bsdf_rev_pdf_w_c,
                        mis_vc_weight_factor, eta_x)
                    comptime if _VCM_CAMIS:
                        camis_cam_scatter(camis, camis_recs, n_verts - 1, eta_x, log(_vcm_keep(sd, hit)),
                                          inv_pdf_c, cos_theta_out_c, bsdf_dir_pdf_w_c, bsdf_rev_pdf_w_c)
                else:
                    dvcm_carry = Float32(0)
                    dvc_carry = Float32(0)
                    dvm_carry = Float32(0)
                    comptime if _VCM_CAMIS:
                        camis.in_class = False

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            comptime if _VCM_CAMIS:
                # A BSSRDF hop and a delta dielectric are both outside the Class.
                camis.in_class = False
            var did_bssrdf_hop = False
            # ── Subsurface boundary: a BSSRDF hop to a real EXIT VERTEX ────
            # The camera arrives at the entry x_i, the exit x_o is sampled from
            # the diffusion profile (shared with the light subpath, see
            # _bdpt_sample_bssrdf_exit), and x_o then becomes an ordinary
            # connectible vertex: merge, connect, direct lighting, continue.
            # Its BSDF is the exit lobe Ft(cos)/pi (LobeKind.bssrdf), whose pdfs are
            # exactly Lambertian. The MIS carries across the hop follow the
            # rule machine-checked in Scenes/vcm_bssrdf_mis_derivation.py.
            # The entry itself is never connectible.
            if mat.sss_boundary != Int8(0) and Int(cur_med_idx) < 0 and has_med:
                var med_in = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                if med_in >= Int32(0) and Int(med_in) < Int(sd.mediumCount):
                    var gn_e = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
                    if dot(gn_e, ray_dir) > Float32(0): gn_e = gn_e * Float32(-1)
                    var cos_e = abs(dot(-ray_dir, gn_e))
                    # Arrival at the entry, as at any surface vertex.
                    (dvcm_carry, dvc_carry, dvm_carry) = vcm_arrival_carries(
                        dvcm_carry, dvc_carry, dvm_carry, cos_e)
                    var eta_e = mat.albedo.r
                    var ex = _bdpt_sample_bssrdf_exit(sd, Int(med_in), hit, gn_e, cos_e, eta_e, pcg)
                    if not ex.ok:
                        return False
                    beta *= spec_refl_unbounded(
                        sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x,
                        sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65,
                        ex.weight.r, ex.weight.g, ex.weight.b, wavelengths)
                    var (h0, h1, h2) = bssrdf_hop_carries[dvc0=_VCM_CAMIS](
                        dvcm_carry, dvc_carry, dvm_carry, ex.p_area, cos_e * INV_PI, mis_vc_weight_factor)
                    dvcm_carry = h0
                    dvc_carry = h1
                    dvm_carry = h2
                    var x_o = ex.x_o
                    var n_o = ex.n_o
                    var v = _null_vertex()
                    v.pos = x_o
                    v.normal = vec3f(n_o)
                    v.shading_normal = vec3f(n_o)
                    v.beta = beta
                    v.alb = RGB(Float32(1))
                    v.is_surface = Int32(1)
                    v.mat_kind = LobeKind.bssrdf
                    v.pdf_fwd = ex.p_area      # reverse density toward x_i (hop is symmetric)
                    v.pdf_bwd = eta_e
                    v.wo = vec3f(n_o)          # unused by LobeKind.bssrdf
                    v.med_idx = cur_med_idx
                    v.wavelengths = wavelengths
                    v.dVCM = dvcm_carry; v.dVC = dvc_carry; v.dVM = dvm_carry
                    if n_verts == 0: first_alb = mat.sss_mean_refl
                    if n_verts + n_delta >= _vcm_depth(sd):
                        return False   # a vertex past d starts no strategy (see _vcm_depth)
                    n_verts += 1
                    # Merging queries the GLOBAL photon grid and does not use this
                    # pixel's own paired light path, so unlike the connect below it
                    # must NOT be gated on that path having stored anything. It was,
                    # and in a white furnace only ~32% of light paths hit the quad at
                    # all, so ~68% of pixels skipped merging entirely: the estimator
                    # delivered 0.109 against an analytic 0.5.
                    total_merge += _bdpt_merge_from_cache(v, sd, lvc, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, mis_vc_weight_factor, n_verts + n_delta, camis, camis_recs, lvc_camis, n_light_paths_f, visit_naive=visit_naive, visit_footprint=visit_footprint, visit_thin=visit_thin)
                    if path_len > 0:
                        if defer_shadow_rays:
                            _bdpt_connect_to_cache_deferred(v, sd, lvc, lp_idx, path_len, mis_vm_weight_factor, shadow_rays, shadow_pending, shadow_valid, shadow_seg_med, n_verts + n_delta, camis, camis_recs, lvc_camis, n_light_paths_f)
                        else:
                            total += _bdpt_connect_to_cache(v, sd, has_med, scratch, lvc, lp_idx, path_len, mis_vm_weight_factor, n_verts + n_delta, camis, camis_recs, lvc_camis, n_light_paths_f)
                    # Direct lighting at the exit: the diffuse vertex's NEE with
                    # the exit lobe's Fresnel factor toward each light.
                    var ctx_x = LobeCtx(LobeKind.lambertian, True, False, n_o, n_o, RGB(Float32(1)), Int32(-1), Float32(0),
                                        Float32(0), Int32(-1), Float32(0), Float32(0), True, False)
                    total += _vcm_nee_surface(sd, v, ctx_x, x_o, n_o, beta, mis_vm_weight_factor * _vcm_eta_scale(sd, x_o),
                                              v.dVCM, v.dVC, cur_med_idx, scratch, wavelengths, pcg,
                                              Float32(0.0001), False, eta_e)
                    # Continue with the exit lobe: cosine-sampled, weight Ft(cos_out).
                    var ux1 = pcg.next_float(); var ux2 = pcg.next_float()
                    rd = vec3f(_cosine_hemisphere_sample(n_o, ux1, ux2))
                    ro = x_o + rd * Float32(0.0002)
                    var cos_out_x = abs(dot(rd.to_simd(), n_o))
                    last_bsdf_pdf = cos_out_x * INV_PI
                    beta *= SpectralSample(bssrdf_exit_ft(cos_out_x, eta_e))
                    var (c0, c1, c2) = bssrdf_exit_scatter_carries[dvc0=_VCM_CAMIS](
                        dvcm_carry, dvc_carry, dvm_carry, ex.p_area, cos_out_x,
                        mis_vc_weight_factor, mis_vm_weight_factor * _vcm_eta_scale(sd, x_o))
                    dvcm_carry = c0
                    dvc_carry = c1
                    dvm_carry = c2
                    n_bounces += 1
                    did_bssrdf_hop = True
            # Ordinary specular boundary, only when no hop was taken.
            if not did_bssrdf_hop:
                if n_verts + n_delta >= _vcm_depth(sd):
                    return False   # a delta vertex past d starts no strategy either
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
                    gn, gn, hit.to_simd(), ray_dir, vcm_cone_w, sd.camFp,
                    sd.textures, sd.gpuTextures, Int(sd.gpuTextureCount))
                var (new_dir, new_org, radiance_scale, new_cur_ior, new_prev_ior) = _dielectric_bounce(
                    ray_dir, hit.to_simd(), gn, mat.albedo.r, n_bounces == 0 and Int(cur_med_idx) < 0, pcg, current_dielectric_ior, previous_dielectric_ior, mat.type == MatKind.thin_dielectric)
                current_dielectric_ior = new_cur_ior
                previous_dielectric_ior = new_prev_ior
                n_bounces += 1
                last_bsdf_pdf = Float32(-1)  # delta bounce: no infinite-light NEE done here
                # Specular vertex: no BSDF record needed, just track throughput.
                # Camera path (Radiance mode): apply the non-symmetric-scattering
                # correction (see _dielectric_bounce's docstring).
                beta *= radiance_scale
                if has_med:
                    var new_idx = medium_after_crossing(ray_dir, inter, sd.meshes, mat, sd, hit)
                    if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
                rd = vec3f(new_dir)
                ro = point3f(new_org)
                # VCM Stage 2b: genuinely delta/specular, matches SmallVCM's own
                # specular-bounce handling directly.
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
            mis_null_dist += t_hit + Float32(0.0002)   # see VCMCameraPathState.mis_null_dist
            # VCM Stage 2b: pure pass-through, carry unchanged (see
            # _bdpt_trace_light_path's matching interface-branch comment).
            comptime if _VCM_CAMIS:
                # A null crossing splits one edge in two (dVCM picks up both
                # segments' d^2), which CAMIS's per-edge P does not model.
                camis.in_class = False

        else:
            return False   # unhandled material type


        return True   # bounce processed normally, path continues

# ── Trace one light subpath, storing its vertices into the shared cache ─────
