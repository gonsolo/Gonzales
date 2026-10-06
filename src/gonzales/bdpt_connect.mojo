# BDPT/VCM connections: to the camera (t=1 splat), to the light-vertex cache, and between vertices.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.collections import Array
from std.math import sqrt, floor, log, max, min, abs
from std.atomic import Atomic
from .geometry import Point3f, Vec3f, dot, PI, INV_FOUR_PI
from .primitives import Intersection
from .vcm_camis import (
    CamisCamRecord, CamisCamCarry, CamisLightRecord, CAMIS_IN_CLASS, CAMIS_TAN_1DEG, camis_eval_connect,
    camis_eval_connect_s1, camis_eval_splat,
)
from .bvh import SceneView
from .sampling import power_heuristic, FilmFilter, filter_eval_2d, filter_integral_2d
from .spectrum import SpectralSample, spec_illum
from .bdpt_vertex import _BDPT_MAX_VERTS, BDPTVertex
from .bdpt_nee import _visible_transmittance
from .bdpt_eval import _eval_vertex_spectral, _bdpt_vertex_pdfs, _bdpt_vertex_mis_scoped, _bdpt_connect_pair_weighted
from .vcm_grid import (
    _vcm_depth, _vcm_light_count, _vcm_keep, _VCM_CAMIS, _CAMIS_CAM_RECS, _CAMIS_FORCE_C1, _vcm_eta_scale,
    _vcm_eta_at, _vcm_cos_at,
)
from .vcm_merge_weight import _camis_gather_light

@always_inline
def _pdf_solid_to_area(pdf_solid: Float32, cos_theta: Float32, dist2: Float32) -> Float32:
    """Convert solid-angle PDF to area PDF: p_A = p_ω * |cosθ| / r²."""
    if dist2 < Float32(1e-8): return Float32(0)
    return pdf_solid * (cos_theta if cos_theta > Float32(0) else -cos_theta) / dist2

# ── Store a vertex in the shared Light Vertex Cache ──────────────────────────

@always_inline
def _bdpt_store_lvc_vertex(
    v: BDPTVertex,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx: Int,
    local_idx: Int,
):
    """Store vertex `local_idx` (0-indexed, always < _BDPT_MAX_VERTS by
    construction of the caller's loop bound) of light path `lp_idx` into"""
    lvc[unsafe_offset=lp_idx * _BDPT_MAX_VERTS + local_idx] = v

# ── LVC connection scale factor ──────────────────────────────────────────────

@always_inline
@always_inline
def _bdpt_world_to_raster(
    p_world: Vec3f,
    w2c: Pointer[Float32, MutUntrackedOrigin],     # inverse(cameraToWorld), col-major
    c2r: Pointer[Float32, MutUntrackedOrigin],     # inverse of the 3x3 raster->camera map, row-major
    fw: Int32, fh: Int32,
) -> Tuple[Bool, Float32, Float32, Float32]:
    """Project a world point onto the film. Returns (ok, filmX, filmY,
    cos_theta), where cos_theta is the angle between the camera's forward"""
    # world -> camera
    var px = p_world[0]; var py = p_world[1]; var pz = p_world[2]
    var cx = w2c[unsafe_offset=0]*px + w2c[unsafe_offset=4]*py + w2c[unsafe_offset=8]*pz  + w2c[unsafe_offset=12]
    var cy = w2c[unsafe_offset=1]*px + w2c[unsafe_offset=5]*py + w2c[unsafe_offset=9]*pz  + w2c[unsafe_offset=13]
    var cz = w2c[unsafe_offset=2]*px + w2c[unsafe_offset=6]*py + w2c[unsafe_offset=10]*pz + w2c[unsafe_offset=14]
    if cz <= Float32(1e-6):
        return (False, Float32(0), Float32(0), Float32(0))   # behind the lens
    var clen = sqrt(cx*cx + cy*cy + cz*cz)
    if clen <= Float32(1e-12):
        return (False, Float32(0), Float32(0), Float32(0))
    var cos_theta = cz / clen                                # forward axis is +z
    # camera-space direction -> (filmX, filmY): q = C2R . c, then divide
    var q0 = c2r[unsafe_offset=0]*cx + c2r[unsafe_offset=1]*cy + c2r[unsafe_offset=2]*cz
    var q1 = c2r[unsafe_offset=3]*cx + c2r[unsafe_offset=4]*cy + c2r[unsafe_offset=5]*cz
    var q2 = c2r[unsafe_offset=6]*cx + c2r[unsafe_offset=7]*cy + c2r[unsafe_offset=8]*cz
    if abs(q2) <= Float32(1e-12):
        return (False, Float32(0), Float32(0), Float32(0))
    var fx = q0 / q2
    var fy = q1 / q2
    if fx < Float32(0) or fy < Float32(0) or fx >= Float32(fw) or fy >= Float32(fh):
        return (False, Float32(0), Float32(0), Float32(0))
    return (True, fx, fy, cos_theta)

@always_inline
def _bdpt_splat_filtered[use_atomics: Bool](
    accum: Pointer[Float32, MutUntrackedOrigin],   # 3 floats per pixel
    fx: Float32, fy: Float32,                     # continuous raster position
    rgb_r: Float32, rgb_g: Float32, rgb_b: Float32,
    fw: Int, fh: Int,
    ff: FilmFilter,
):
    """Spread one t=1 splat over every pixel its PixelFilter footprint covers,
    weighted f(p - pixel centre) / integral(f) -- pbrt-v4's RGBFilm::AddSplat,"""
    var ftype = ff[0].cast[DType.int32]()
    var rx = ff[2]
    var ry = ff[3]
    var inv_int = Float32(1.0) / max(filter_integral_2d(ftype, ff[1], rx, ry), Float32(1e-12))
    var x0 = max(0, Int(floor(fx + Float32(0.5) - rx)))
    var x1 = min(fw - 1, Int(floor(fx + Float32(0.5) + rx)))
    var y0 = max(0, Int(floor(fy + Float32(0.5) - ry)))
    var y1 = min(fh - 1, Int(floor(fy + Float32(0.5) + ry)))
    for py in range(y0, y1 + 1):
        for px in range(x0, x1 + 1):
            var w = filter_eval_2d(fx - (Float32(px) + Float32(0.5)), fy - (Float32(py) + Float32(0.5)),
                                   ftype, ff[1], rx, ry) * inv_int
            if w <= Float32(0.0):
                continue
            var o = (py * fw + px) * 3
            comptime if use_atomics:
                _ = Atomic[Float32].fetch_add(accum.unsafe_offset(o + 0), rgb_r * w)
                _ = Atomic[Float32].fetch_add(accum.unsafe_offset(o + 1), rgb_g * w)
                _ = Atomic[Float32].fetch_add(accum.unsafe_offset(o + 2), rgb_b * w)
            else:
                accum[unsafe_offset=o + 0] += rgb_r * w
                accum[unsafe_offset=o + 1] += rgb_g * w
                accum[unsafe_offset=o + 2] += rgb_b * w

def _bdpt_connect_to_camera(
    lv: BDPTVertex,
    ref sd: SceneView,
    scratch: Pointer[Intersection, MutUntrackedOrigin],
    cam_pos: Vec3f,
    w2c: Pointer[Float32, MutUntrackedOrigin],
    c2r: Pointer[Float32, MutUntrackedOrigin],
    fw: Int32, fh: Int32,
    px_scale: Float32,
    n_light_paths_f: Float32,
    mis_vm_weight_factor: Float32,
    # _VCM_CAMIS: lv's own light-path records, gathered via
    # _camis_gather_light from `lvc_camis` at slot `k` -- see
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin] = Pointer[BDPTVertex, MutUntrackedOrigin].unsafe_dangling(),
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin] = Pointer[CamisLightRecord, MutUntrackedOrigin].unsafe_dangling(),
    k: Int = 0,
) -> Tuple[Bool, Float32, Float32, SpectralSample]:
    """The t=1 strategy: connect a LIGHT-subpath vertex directly to the
    camera and return the continuous raster position it lands on plus its"""
    if lv.is_delta != Int32(0) or lv.is_surface == Int32(0):
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))
    # The light-source vertex itself is NEVER splatted. Connecting the
    # emission point straight to the camera builds the length-1 path
    if lv.is_light == Int32(1) or not _bdpt_vertex_mis_scoped(lv):
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))

    var lp = lv.pos.to_simd()
    var d3 = cam_pos - lp
    var dist2 = dot(d3, d3)
    if dist2 < Float32(1e-8):
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))
    var dist = sqrt(dist2)
    var dir_to_cam = d3 * (Float32(1) / dist)

    var pr = _bdpt_world_to_raster(lp, w2c, c2r, fw, fh)
    if not pr[0]:
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))
    var cos_at_camera = pr[3]
    if cos_at_camera <= Float32(1e-6):
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))

    var wl = lv.wavelengths
    var cos_to_camera = abs(dot(lv.normal.to_simd(), dir_to_cam))
    var f = _eval_vertex_spectral(lv, Vec3f(dir_to_cam[0], dir_to_cam[1], dir_to_cam[2]), sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wl, adjoint=True)
    if f.is_black():
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))
    if cos_to_camera <= Float32(1e-8):
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))

    var Tr = _visible_transmittance(
        lv.pos, Point3f(cam_pos[0], cam_pos[1], cam_pos[2]), lv.med_idx, sd, scratch, wl)
    if Tr.is_black():
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))

    var image_plane_dist = Float32(1) / max(px_scale, Float32(1e-12))
    var ipcd = image_plane_dist / cos_at_camera
    var image_to_solid_angle = (ipcd * ipcd) / cos_at_camera
    # `_eval_vertex` already carries |cos| at the light vertex, so it is
    # deliberately NOT reapplied here (see the docstring).
    var geom = image_to_solid_angle / dist2
    var image_to_surface = image_to_solid_angle * cos_to_camera / dist2

    var inv_n = Float32(1) / max(n_light_paths_f, Float32(1))
    var contrib = (lv.beta * f * Tr
                   * (geom * inv_n))

    # MIS, SmallVCM ConnectToCamera:
    #   wLight = (cameraPdfA / lightSubPathCount)
    var (_dp, rev_pdf_w) = _bdpt_vertex_pdfs(lv, Vec3f(dir_to_cam[0], dir_to_cam[1], dir_to_cam[2]), sd)
    var camera_pdf_a = image_to_surface
    var eta_lv_splat = mis_vm_weight_factor * _vcm_eta_scale(sd, lv.pos)
    var mis_weight: Float32
    comptime if _VCM_CAMIS:
        var light_arr = lvc_camis[unsafe_offset=k]
        if (light_arr.flags & CAMIS_IN_CLASS) != Int32(0):
            var lp_idx = k // _BDPT_MAX_VERTS
            var base = lp_idx * _BDPT_MAX_VERTS
            var lam = k % _BDPT_MAX_VERTS
            var origin_in_class = (lvc_camis[unsafe_offset=base].flags & CAMIS_IN_CLASS) != Int32(0)
            var lr = _camis_gather_light[_CAMIS_CAM_RECS](lvc, lvc_camis, base, lam)
            # t=1's own "r": there is no stored camera subpath, so Eq. 17's
            # radius uses the splatted light vertex's OWN distance to the
            var r_splat = dist * CAMIS_TAN_1DEG
            var log_k_splat = log(PI * r_splat * r_splat)
            mis_weight = camis_eval_splat[_CAMIS_CAM_RECS](
                light_arr, lr[0], lr[2],
                lr[1], -log(lv.dVCM),
                lr[3], origin_in_class,
                lv.dVCM, lv.dVC, lv.dVM, eta_lv_splat,
                log_k_splat, camera_pdf_a, rev_pdf_w, n_light_paths_f,
                n_light_paths_f, True, _CAMIS_FORCE_C1)
        else:
            var w_light = (camera_pdf_a * inv_n) * (eta_lv_splat + lv.dVCM + lv.dVC * rev_pdf_w)
            mis_weight = Float32(1) / (w_light + Float32(1))
    else:
        var w_light = (camera_pdf_a * inv_n) * (eta_lv_splat + lv.dVCM + lv.dVC * rev_pdf_w)
        mis_weight = Float32(1) / (w_light + Float32(1))
    contrib = contrib * mis_weight

    return (True, pr[1], pr[2], contrib)

def _bdpt_connect_to_cache(
    cv: BDPTVertex,
    ref sd: SceneView,
    has_med: Bool,
    scratch: Pointer[Intersection, MutUntrackedOrigin],
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx: Int,
    path_len: Int,
    mis_vm_weight_factor: Float32,
    cam_count: Int,   # cv's non-delta interior-vertex count (see _vcm_depth)
    # _VCM_CAMIS: see _bdpt_merge_from_cache's matching parameters.
    camis: CamisCamCarry = CamisCamCarry(False, Int32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0)),
    camis_recs: Array[CamisCamRecord, _CAMIS_CAM_RECS] = Array[CamisCamRecord, _CAMIS_CAM_RECS](
        fill=CamisCamRecord(Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))),
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin] = Pointer[CamisLightRecord, MutUntrackedOrigin].unsafe_dangling(),
    n_light_paths_f: Float32 = Float32(0),
) -> SpectralSample:
    """VCM Stage 2b (2026-07-10): connect eye vertex `cv` to EVERY vertex of
    its deterministically PAIRED light path (`lp_idx` — standard Veach BDPT"""
    var sum = SpectralSample(Float32(0))
    # Sum every MIS-WEIGHTED pair, but take at most ONE UNWEIGHTED pair.
    # Volume vertices have no dVCM/dVC (no surface pdf), so any (s,t) split
    var took_unweighted = False
    var d = _vcm_depth(sd)
    for local in range(path_len):
        if _vcm_light_count(lvc, lp_idx, local) + cam_count > d:
            break   # see _vcm_depth: a path longer than d has no strategy at all
        var lv = lvc[unsafe_offset=lp_idx * _BDPT_MAX_VERTS + local]
        if not _bdpt_connect_pair_weighted(cv, lv):
            if took_unweighted:
                continue
            took_unweighted = True
        sum += _connect(cv, lv, sd, has_med, scratch, mis_vm_weight_factor,
            camis, camis_recs, cam_count - 1, lvc, lvc_camis, lp_idx * _BDPT_MAX_VERTS + local, n_light_paths_f)
    return sum

def _bdpt_connect_to_cache_deferred(
    cv: BDPTVertex,
    ref sd: SceneView,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lp_idx: Int,
    path_len: Int,
    mis_vm_weight_factor: Float32,
    shadow_rays: Pointer[Float32, MutUntrackedOrigin],
    shadow_pending: Pointer[SpectralSample, MutUntrackedOrigin],
    shadow_valid: Pointer[Int8, MutUntrackedOrigin],
    shadow_seg_med: Pointer[Int32, MutUntrackedOrigin],
    cam_count: Int,
    # _VCM_CAMIS: see _bdpt_merge_from_cache's matching parameters.
    camis: CamisCamCarry = CamisCamCarry(False, Int32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0)),
    camis_recs: Array[CamisCamRecord, _CAMIS_CAM_RECS] = Array[CamisCamRecord, _CAMIS_CAM_RECS](
        fill=CamisCamRecord(Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))),
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin] = Pointer[CamisLightRecord, MutUntrackedOrigin].unsafe_dangling(),
    n_light_paths_f: Float32 = Float32(0),
):
    """Task #163 stage 5: Vulkan-RT-batched counterpart to
    _bdpt_connect_to_cache -- instead of resolving each connection's shadow"""
    var base = lp_idx * _BDPT_MAX_VERTS
    # Same "every weighted pair, at most one unweighted pair" rule as
    # _bdpt_connect_to_cache -- see its comment for the derivation and the
    var took_unweighted = False
    for local in range(_BDPT_MAX_VERTS):
        if local < path_len and _vcm_light_count(lvc, lp_idx, local) + cam_count > _vcm_depth(sd):
            break   # see _vcm_depth
        if local >= path_len:
            shadow_valid[unsafe_offset=base + local] = Int8(0)
            continue
        var lv = lvc[unsafe_offset=base + local]
        if not _bdpt_connect_pair_weighted(cv, lv):
            if took_unweighted:
                shadow_valid[unsafe_offset=base + local] = Int8(0)
                continue
            took_unweighted = True
        var (contrib, valid) = _connect_unweighted(cv, lv, sd, mis_vm_weight_factor,
            camis, camis_recs, cam_count - 1, lvc, lvc_camis, base + local, n_light_paths_f)
        if not valid:
            shadow_valid[unsafe_offset=base + local] = Int8(0)
            continue
        var d3 = lv.pos - cv.pos
        var dist = sqrt(d3.length_sq())
        var dir = d3.to_simd() / dist
        var idx8 = (base + local) * 8
        shadow_rays[unsafe_offset=idx8 + 0] = cv.pos.x
        shadow_rays[unsafe_offset=idx8 + 1] = cv.pos.y
        shadow_rays[unsafe_offset=idx8 + 2] = cv.pos.z
        shadow_rays[unsafe_offset=idx8 + 3] = Float32(1e-4)
        shadow_rays[unsafe_offset=idx8 + 4] = dir[0]
        shadow_rays[unsafe_offset=idx8 + 5] = dir[1]
        shadow_rays[unsafe_offset=idx8 + 6] = dir[2]
        shadow_rays[unsafe_offset=idx8 + 7] = dist * Float32(0.9995)
        shadow_pending[unsafe_offset=base + local] = contrib
        shadow_seg_med[unsafe_offset=base + local] = cv.med_idx
        shadow_valid[unsafe_offset=base + local] = Int8(1)

# ── VCM vertex merging: spatial hash grid over the LVC ───────────────────────
# THE thinned hash grid SPPM's photons use (sppm.mojo, grid_reset_cell ..
# grid_weight), keyed on BDPTVertex.pos instead of SPPMPhoton.pos, and using a

# Stochastic merge thinning: the shared thinned grid (sppm.mojo, grid_keep and
# friends). What is VCM's own is the MIS: merging's density in a thinned

def _connect(
    cv: BDPTVertex,  # camera-subpath vertex
    lv: BDPTVertex,  # light-subpath vertex (including light point itself)
    ref sd: SceneView,
    has_med: Bool,
    scratch: Pointer[Intersection, MutUntrackedOrigin],
    mis_vm_weight_factor: Float32,
    # _VCM_CAMIS: a default of in_class=False (NOT camis_cam_carry_init(),
    # whose default is True) so a call site that forgets to pass real state
    camis: CamisCamCarry = CamisCamCarry(False, Int32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0)),
    camis_recs: Array[CamisCamRecord, _CAMIS_CAM_RECS] = Array[CamisCamRecord, _CAMIS_CAM_RECS](
        fill=CamisCamRecord(Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))),
    num_cam_scat: Int = 0,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin] = Pointer[BDPTVertex, MutUntrackedOrigin].unsafe_dangling(),
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin] = Pointer[CamisLightRecord, MutUntrackedOrigin].unsafe_dangling(),
    k: Int = 0,
    n_light_paths_f: Float32 = Float32(0),
) -> SpectralSample:
    """Evaluate the contribution of connecting cv to lv via a shadow ray.
    Each connection is already a complete, self-normalized estimator of its"""
    # THE connection estimator lives in _connect_unweighted; this is that
    # times visibility. They used to be two byte-for-byte copies (one for the
    var (contrib, valid) = _connect_unweighted(cv, lv, sd, mis_vm_weight_factor,
        camis, camis_recs, num_cam_scat, lvc, lvc_camis, k, n_light_paths_f)
    if not valid or contrib.is_black():
        return SpectralSample(Float32(0))
    # Medium for the shadow segment: the camera vertex's (both endpoints agree
    # in a well-defined scene).
    var Tr = _visible_transmittance(cv.pos, lv.pos, cv.med_idx, sd, scratch, cv.wavelengths)
    if Tr.is_black():
        return SpectralSample(Float32(0))
    return contrib * Tr

def _connect_unweighted(
    cv: BDPTVertex,  # camera-subpath vertex
    lv: BDPTVertex,  # light-subpath vertex (including light point itself)
    ref sd: SceneView,
    mis_vm_weight_factor: Float32,
    camis: CamisCamCarry = CamisCamCarry(False, Int32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0)),
    camis_recs: Array[CamisCamRecord, _CAMIS_CAM_RECS] = Array[CamisCamRecord, _CAMIS_CAM_RECS](
        fill=CamisCamRecord(Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))),
    num_cam_scat: Int = 0,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin] = Pointer[BDPTVertex, MutUntrackedOrigin].unsafe_dangling(),
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin] = Pointer[CamisLightRecord, MutUntrackedOrigin].unsafe_dangling(),
    k: Int = 0,
    n_light_paths_f: Float32 = Float32(0),
) -> Tuple[SpectralSample, Bool]:
    """Task #163 stage 5: byte-for-byte copy of _connect's math (see that
    function's own docstring for the MIS derivation, not repeated here),"""
    if cv.is_delta != Int32(0) or lv.is_delta != Int32(0):
        return (SpectralSample(Float32(0)), False)

    var d3 = lv.pos - cv.pos
    var dist2 = d3.length_sq()
    if dist2 < Float32(1e-8):
        return (SpectralSample(Float32(0)), False)
    var dist = sqrt(dist2)

    var dir = d3.to_simd() / dist
    var neg_dir = -dir

    # Both endpoints evaluate in the spectral domain and multiply there --
    # the product of two spectra, not the product of two RGB triples, which
    var wl = cv.wavelengths
    var f_cam_spec = _eval_vertex_spectral(cv, dir, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wl)
    var f_lgt_spec: SpectralSample
    if lv.is_light == Int32(1):
        # Light emission: Le carries no cosine of its own, so the emitting
        # surface's cosine is applied HERE, explicitly -- see the geometry
        # note below for why it can no longer come from a shared G.
        var ln = lv.normal.to_simd()
        var cos_l = dot(neg_dir, ln)
        if lv.pdf_bwd > Float32(0) and lv.pdf_bwd < Float32(1):
            cos_l = abs(cos_l)   # two-sided light: the connection is face-free (pdf_bwd = face probability)
        if cos_l <= Float32(0):
            return (SpectralSample(Float32(0)), False)
        f_lgt_spec = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, lv.alb.r, lv.alb.g, lv.alb.b, wl) * cos_l
    else:
        f_lgt_spec = _eval_vertex_spectral(lv, neg_dir, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wl, adjoint=True)
    var f_combined = f_cam_spec * f_lgt_spec

    # GEOMETRY: 1/d^2 ONLY. Each endpoint's cosine is already inside its f_cos.
    #
    var contrib = cv.beta * lv.beta * f_combined * (Float32(1) / dist2)
    contrib *= _bdpt_connect_mis_weight(cv, lv, sd, dir, dist2, mis_vm_weight_factor,
        camis, camis_recs, num_cam_scat, lvc, lvc_camis, k, n_light_paths_f)
    return (contrib, True)


@always_inline
def _bdpt_connect_mis_weight(
    cv: BDPTVertex,
    lv: BDPTVertex,
    ref sd: SceneView,
    dir: Vec3f,        # unit direction cv -> lv
    dist2: Float32,    # |lv - cv|^2
    mis_vm_weight_factor: Float32,
    camis: CamisCamCarry,
    camis_recs: Array[CamisCamRecord, _CAMIS_CAM_RECS],
    num_cam_scat: Int,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin],
    k: Int,
    n_light_paths_f: Float32,
) -> Float32:
    """Balance-heuristic connection weight (Georgiev et al. 2012 / SmallVCM's
    ConnectVertices), factored out of _connect_unweighted exactly as"""
    var neg_dir = -dir
    # VCM Stage 2b/2d: real MIS weight for diffuse/conductor/light-source
    # connections (see _connect's docstring + _bdpt_vertex_pdfs'/
    if _bdpt_vertex_mis_scoped(cv) and (lv.is_light == Int32(1) or _bdpt_vertex_mis_scoped(lv)):
        var cos_cv = _vcm_cos_at(cv, dir)
        var cos_lv = _vcm_cos_at(lv, neg_dir)
        var (camera_bsdf_dir_pdf_w, camera_bsdf_rev_pdf_w) = _bdpt_vertex_pdfs(cv, dir, sd)
        # Light-source vertex: forward and reverse pdf are the SAME
        # cosine-weighted-emission formula (no real "wo" to distinguish a
        var light_bsdf_dir_pdf_w: Float32
        var light_bsdf_rev_pdf_w: Float32
        if lv.is_light == Int32(1):
            var side_pdf = lv.pdf_bwd if lv.pdf_bwd > Float32(0) else Float32(1)
            light_bsdf_dir_pdf_w = side_pdf * cos_lv / PI
            light_bsdf_rev_pdf_w = side_pdf * cos_lv / PI
        else:
            var (ldp, lrp) = _bdpt_vertex_pdfs(lv, neg_dir, sd)
            light_bsdf_dir_pdf_w = ldp
            light_bsdf_rev_pdf_w = lrp
        var camera_bsdf_dir_pdf_a = camera_bsdf_dir_pdf_w * cos_lv / dist2
        var light_bsdf_dir_pdf_a = light_bsdf_dir_pdf_w * cos_cv / dist2
        var eta_cv = _vcm_eta_at(sd, cv, mis_vm_weight_factor)
        comptime if _VCM_CAMIS:
            if camis.in_class:
                if lv.is_light == Int32(1):
                    var origin = lvc_camis[unsafe_offset=k]
                    if (origin.flags & CAMIS_IN_CLASS) != Int32(0):
                        return camis_eval_connect_s1[_CAMIS_CAM_RECS](
                            camis, camis_recs, num_cam_scat,
                            cv.dVCM, cv.dVC, cv.dVM, eta_cv, log(_vcm_keep(sd, cv.pos)),
                            origin.log_pa_rev, lv.pdf_fwd, True,
                            camera_bsdf_dir_pdf_a, camera_bsdf_rev_pdf_w, light_bsdf_dir_pdf_a,
                            n_light_paths_f, True, _CAMIS_FORCE_C1)
                else:
                    var light_arr = lvc_camis[unsafe_offset=k]
                    if (light_arr.flags & CAMIS_IN_CLASS) != Int32(0):
                        var lp_idx = k // _BDPT_MAX_VERTS
                        var base = lp_idx * _BDPT_MAX_VERTS
                        var lam = k % _BDPT_MAX_VERTS
                        var origin_in_class = (lvc_camis[unsafe_offset=base].flags & CAMIS_IN_CLASS) != Int32(0)
                        var lr = _camis_gather_light[_CAMIS_CAM_RECS](lvc, lvc_camis, base, lam)
                        var eta_lv_c = _vcm_eta_at(sd, lv, mis_vm_weight_factor)
                        return camis_eval_connect[_CAMIS_CAM_RECS, _CAMIS_CAM_RECS](
                            camis, camis_recs, num_cam_scat,
                            cv.dVCM, cv.dVC, cv.dVM, eta_cv, log(_vcm_keep(sd, cv.pos)),
                            light_arr, lr[0], lr[2],
                            lr[1], -log(lv.dVCM),
                            lr[3], origin_in_class,
                            lv.dVCM, lv.dVC, lv.dVM, eta_lv_c,
                            camera_bsdf_dir_pdf_a, camera_bsdf_rev_pdf_w, light_bsdf_dir_pdf_a, light_bsdf_rev_pdf_w,
                            n_light_paths_f, True, _CAMIS_FORCE_C1)
        var w_light: Float32
        if lv.is_light == Int32(1):
            # s=1: this connection IS VCM's direct-light strategy for an area
            # light (there is no separate area-light NEE), so its light-side
            w_light = camera_bsdf_dir_pdf_a / max(lv.pdf_fwd, Float32(1e-30))
        else:
            var eta_lv = _vcm_eta_at(sd, lv, mis_vm_weight_factor)
            w_light = camera_bsdf_dir_pdf_a * (eta_lv + lv.dVCM + lv.dVC * light_bsdf_rev_pdf_w)
        var w_camera = light_bsdf_dir_pdf_a * (eta_cv + cv.dVCM + cv.dVC * camera_bsdf_rev_pdf_w)
        return Float32(1) / (w_light + Float32(1) + w_camera)
    elif cv.is_surface == Int32(0) and lv.is_light == Int32(1) and lv.pdf_fwd > Float32(0):
        # Volume vertex -> light source: MIS against the phase-hit strategy
        # (the camera path continuing by uniform-sphere sampling and landing
        var cos_lv_vol = abs(dot(neg_dir, lv.normal.to_simd()))
        if cos_lv_vol > Float32(1e-8):
            var pdf_light_w_vol = lv.pdf_fwd * dist2 / cos_lv_vol
            return power_heuristic(pdf_light_w_vol, INV_FOUR_PI)
    return Float32(1)

# ── Main BDPT render ──────────────────────────────────────────────────────────
