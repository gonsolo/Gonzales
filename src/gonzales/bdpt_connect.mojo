# BDPT/VCM connections: to the camera (t=1 splat), to the light-vertex cache, and between vertices.
# Part of the BDPT/VCM machinery that used to be one file (bdpt.mojo).

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
    construction of the caller's loop bound) of light path `lp_idx` into
    its own dedicated slice of the LVC: light path `lp_idx` owns exactly
    the slots [lp_idx*_BDPT_MAX_VERTS, (lp_idx+1)*_BDPT_MAX_VERTS).
    VCM Stage 2b (2026-07-10): replaced the old shared-global-cache +
    atomic-slot-reservation design (every light path competing for slots in
    one flat array) with this per-path-indexed layout, needed so the
    camera side can deterministically pair each pixel with its OWN light
    path (real Georgiev/SmallVCM-style VCM's dVCM/dVC/dVM MIS weights
    assume that pairing, not a random shared-pool draw — see
    project_vcm_stage2_mis_derivation memory). Bonus: since each light
    path now owns a non-contended slice, no atomics are needed here at
    all, on CPU or GPU."""
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
    cos_theta), where cos_theta is the angle between the camera's forward
    axis and the direction to the point.

    This is the inverse of sampling.mojo's `gen_primary_ray_state` camera
    transform, which builds a ray direction as `M . (filmX, filmY, 1)` with
    M the 3x3 taken from rasterToCamera's columns 0, 1 and 3 (its z column
    is unused because the film sits at z=0). Inverting that map and
    dividing through by the third component recovers (filmX, filmY) for any
    camera-space direction, which is exactly what the t=1 light-tracing
    strategy needs and what nothing in this renderer could do before.

    ok=False when the point is behind the camera or lands off-film."""
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
    weighted f(p - pixel centre) / integral(f) -- pbrt-v4's RGBFilm::AddSplat,
    which normalises the splat by the filter integral at output time.

    Splats used to land on the single pixel containing p. That is a box
    filter, so while the camera half of VCM now reconstructs through the
    scene's PixelFilter, its light-traced half would have stayed box-sharp --
    the two halves of one image filtered differently. For a half-pixel box this
    is exactly the old behaviour: weight 1, one pixel. Energy is conserved in
    expectation, since the weights over the footprint sum to integral(f)."""
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
    # _bdpt_merge_mis_weight's matching parameters. `k` is lv's own LVC slot
    # (base + local), needed since lv is already dereferenced here.
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin] = Pointer[BDPTVertex, MutUntrackedOrigin].unsafe_dangling(),
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin] = Pointer[CamisLightRecord, MutUntrackedOrigin].unsafe_dangling(),
    k: Int = 0,
) -> Tuple[Bool, Float32, Float32, SpectralSample]:
    """The t=1 strategy: connect a LIGHT-subpath vertex directly to the
    camera and return the continuous raster position it lands on plus its
    contribution. The caller spreads it over the filter footprint
    (_bdpt_splat_filtered), as pbrt's RGBFilm::AddSplat does.
    Ported from SmallVCM's `ConnectToCamera` (vertexcm.hxx), the same
    reference the rest of this file's MIS quantities came from.

    This strategy did not exist here before, and measurement says it is
    20.8% of a cornell-box image (pbrt's own per-strategy BDPT output).
    Since `_connect`'s MIS weight already RESERVES its share through the
    dVC/dVCM recursion, leaving it out did not merely omit those paths --
    it under-weighted every other strategy by that share, which is what
    made --vcm come out at 0.774 of gonzales's own path tracer.

    Radiometry, following SmallVCM exactly:

        imagePlaneDist       = 1 / px_scale        (pixels per world unit
                                                    at unit distance -- the
                                                    same convention the
                                                    camera-path dVCM's
                                                    cameraPdfW already uses)
        imageToSolidAngle    = (imagePlaneDist/cosAtCamera)^2 / cosAtCamera
        imageToSurface       = imageToSolidAngle * |cosToCamera| / dist^2
        contrib              = beta * f * imageToSurface / lightSubPathCount

    with one deliberate difference in bookkeeping: SmallVCM's
    `BSDF::Evaluate` returns the BSDF WITHOUT the outgoing cosine and picks
    it up again inside imageToSurface, whereas this file's `_eval_vertex`
    returns BSDF x cos. So the cosine is taken from `_eval_vertex` and
    imageToSolidAngle is used here WITHOUT it -- multiplying both would
    square the cosine and darken grazing geometry.

    Only MIS-scoped vertices are connected, and never the light-source
    vertex (see the gate below for why that one is excluded). An unscoped
    vertex gets `weight = 1` from `_connect`, which that function documents
    as already being the complete estimate for its kind; splatting such a
    vertex too would double-count it.

    Returns (ok, pixel_index, contribution)."""
    if lv.is_delta != Int32(0) or lv.is_surface == Int32(0):
        return (False, Float32(-1), Float32(-1), SpectralSample(Float32(0)))
    # The light-source vertex itself is NEVER splatted. Connecting the
    # emission point straight to the camera builds the length-1 path
    # "camera sees the emitter", which the camera path already credits in
    # full: its primary ray starts with last_bsdf_pdf = -1, so a direct hit on
    # an area light takes weight 1. Both claimed the whole emitter -- every
    # directly visible light rendered at exactly 2x (cornell-box's emitter
    # pixels read 1.978x the path tracer while the rest of the image matched).
    #
    # SmallVCM makes the same choice for the same reason: its light paths
    # connect to the camera only from their first BOUNCE onward, which is
    # precisely why its GetLightRadiance can return weight 1 at path length 1.
    # This keeps that pairing instead of re-weighting the camera hit, and it
    # is also the lower-variance estimator of the two -- every pixel that sees
    # the emitter sees it on its own camera ray, with no noise from where
    # light paths happened to start.
    #
    # lv0 stays in the light-vertex cache: CAMERA-VERTEX connections to it are
    # the s=1 strategy at path length >= 2, a different thing entirely.
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
    #            * (misVmWeightFactor + dVCM + dVC * bsdfRevPdfW)
    #   weight = 1 / (wLight + 1)
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
            # lens -- the same primaryDistance convention as y1's, just with
            # this vertex standing in for y1 (SmallVCM/PdfRatioVcm agree).
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
    its deterministically PAIRED light path (`lp_idx` — standard Veach BDPT
    pairing: n_light_paths == n_pix, one dedicated light path per pixel,
    see _bdpt_store_lvc_vertex's docstring). This REPLACES the old
    K-uniformly-random-draws-from-a-shared-global-pool estimator (and its
    avg_light_path_len/K rescaling — no longer needed, since every vertex
    of the ONE paired path is visited exactly once, an exhaustive sum, not
    a subsample). Real VCM's dVCM/dVC/dVM MIS weights (see
    project_vcm_stage2_mis_derivation memory) assume exactly this pairing;
    the old random-subsample design was not verified compatible with them.
    No RNG needed here anymore — the set of light vertices to connect to is
    now fully determined by which pixel `cv`'s eye subpath belongs to."""
    var sum = SpectralSample(Float32(0))
    # Sum every MIS-WEIGHTED pair, but take at most ONE UNWEIGHTED pair.
    # Volume vertices have no dVCM/dVC (no surface pdf), so any (s,t) split
    # touching one comes back unweighted -- summing every such split
    # over-counts (see docs/09_volumetric_media.md, "VCM/BDPT volume
    # connections": the 1.027/1.388/1.651/1.969x non-decaying-increment
    # signature). Keeping only the lowest-index unweighted pair restores
    # exactly one strategy per path length, unbiased. Keyed on the PAIR, not
    # on which side is in a medium -- a surface cv into several volume light
    # vertices is the same over-count from the other side.
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
    _bdpt_connect_to_cache -- instead of resolving each connection's shadow
    ray inline via software BVH (_visible_transmittance), writes the ray +
    _connect_unweighted's unweighted contribution into `lp_idx`'s own
    dedicated _BDPT_MAX_VERTS-sized slice of the shadow-ray queue buffers
    (same per-path-slice indexing convention as the LVC itself, see
    _bdpt_store_lvc_vertex's docstring), for a later batched Vulkan RT
    dispatch + resolve pass to fill in. ALWAYS writes exactly
    _BDPT_MAX_VERTS slots, marking unused/invalid ones so stale data from a
    previous bounce or sample never leaks through a reused buffer."""
    var base = lp_idx * _BDPT_MAX_VERTS
    # Same "every weighted pair, at most one unweighted pair" rule as
    # _bdpt_connect_to_cache -- see its comment for the derivation and the
    # measurement. Slots skipped by the rule are marked invalid, exactly like
    # slots past path_len, so no stale contribution leaks through.
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
# SEPARATE parallel `merge_next` array for chaining rather than a field inside
# BDPTVertex itself (avoids touching BDPTVertex's layout/every other
# construction site in this file).

# Stochastic merge thinning: the shared thinned grid (sppm.mojo, grid_keep and
# friends). What is VCM's own is the MIS: merging's density in a thinned
# bucket is keep * eta, not eta, and the other strategies are told so
# (_vcm_keep, fed the previous pass's counts from the second, alternating
# table). Scenes/vcm_area_mis_derivation.py checks the weights stay a
# partition of unity with a per-vertex eta.

def _connect(
    cv: BDPTVertex,  # camera-subpath vertex
    lv: BDPTVertex,  # light-subpath vertex (including light point itself)
    ref sd: SceneView,
    has_med: Bool,
    scratch: Pointer[Intersection, MutUntrackedOrigin],
    mis_vm_weight_factor: Float32,
    # _VCM_CAMIS: a default of in_class=False (NOT camis_cam_carry_init(),
    # whose default is True) so a call site that forgets to pass real state
    # silently falls back to the exact legacy weight instead of computing
    # CAMIS math from garbage (log_k=0 etc.) that happens to look in-Class.
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
    Each connection is already a complete, self-normalized estimator of its
    own depth-strategy's contribution (see the module's LVC-BPT docstring);
    the caller sums over every vertex of the paired light path with no
    further scaling.

    VCM Stage 2b/2d/153/hair (2026-07-10/11): real per-vertex MIS weighting
    (Georgiev et al. 2012 / SmallVCM, see project_vcm_stage2_mis_derivation
    memory) is applied when both endpoints have a genuine standalone pdf --
    diffuse (mat_kind=0), light-source vertices (is_light=1, whose
    cosine-weighted emission profile is mathematically the same shape as
    diffuse), rough conductor/coated_conductor (mat_kind=1, GGX-VNDF pdf),
    hair (mat_kind=2, Marschner 3-lobe pdf via _hair_eval_lobes -- the same
    pdf machinery _nee_weight_hair already trusts for NEE MIS), and
    measured (mat_kind=3, tabulated-BRDF pdf via a reconstructed local
    frame). Only volume (isotropic phase, no surface normal) falls through
    to `weight=1`, today's plain unweighted behavior; dielectric/
    thin_dielectric are genuinely delta/specular and never even reach here
    at all (never stored as LVC vertices, see this file's opening VCM
    comment) -- both deliberately scoped boundaries, not silent
    omissions."""
    # THE connection estimator lives in _connect_unweighted; this is that
    # times visibility. They used to be two byte-for-byte copies (one for the
    # deferred Vulkan-RT shadow-ray path, which resolves visibility later),
    # and a copy is how a fix lands in one and not the other -- see
    # feedback_unify_while_fixing. Evaluating the BSDFs first also skips the
    # shadow ray entirely for a connection that is zero anyway.
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
    function's own docstring for the MIS derivation, not repeated here),
    with the visibility test (_visible_transmittance) REMOVED -- returns
    (contrib_unweighted, valid) instead of the final Tr-weighted
    contribution. `valid=False` means _connect's own early-exit conditions
    (delta endpoint, coincident points, light facing away) already prove
    the contribution is exactly zero regardless of visibility -- the
    caller should NOT bother queuing a shadow ray for these. `valid=True`
    means the caller must still resolve visibility (Tr) and multiply it
    into contrib_unweighted -- this function intentionally does not do
    that itself, so its result can be used to fill a batched Vulkan RT
    shadow-ray queue instead of resolving inline per-thread. Used ONLY by
    the wavefront-staged GPU path's deferred-connect kernels
    (_bdpt_connect_diffuse_deferred_gpu et al.) when use_vk=True -- _connect
    itself is UNCHANGED and still drives the CPU renderer, the
    non-wavefront GPU renderer, and the wavefront GPU renderer's own
    non-deferred (use_vk=False) path.

    VCM Stage 2b/2d/153/hair (2026-07-10/11): real per-vertex MIS weighting
    (Georgiev et al. 2012 / SmallVCM, see project_vcm_stage2_mis_derivation
    memory) is applied when both endpoints have a genuine standalone pdf --
    diffuse (mat_kind=0), light-source vertices (is_light=1, whose
    cosine-weighted emission profile is mathematically the same shape as
    diffuse), rough conductor/coated_conductor (mat_kind=1, GGX-VNDF pdf),
    hair (mat_kind=2, Marschner 3-lobe pdf via _hair_eval_lobes -- the same
    pdf machinery _nee_weight_hair already trusts for NEE MIS), and
    measured (mat_kind=3, tabulated-BRDF pdf via a reconstructed local
    frame). Only volume (isotropic phase, no surface normal) falls through
    to `weight=1`, today's plain unweighted behavior; dielectric/
    thin_dielectric are genuinely delta/specular and never even reach here
    at all (never stored as LVC vertices, see this file's opening VCM
    comment) -- both deliberately scoped boundaries, not silent
    omissions."""
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
    # is the entire point. This used to be a dual RGB/spectral path whose
    # spectral branch evaluated at the LIGHT vertex's own wavelengths and
    # discarded the camera subpath's, because LVC vertices each carried an
    # independent wavelength draw and there was no consistent basis to
    # multiply in. Every subpath in a pass now shares one wavelength set
    # (see _bdpt_pass_wavelengths), so cv and lv are guaranteed to agree and
    # the fallback is gone -- including for hair and measured, which the old
    # spectral branch had to route around.
    var wl = cv.wavelengths
    var f_cam_spec = _eval_vertex_spectral(cv, dir, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wl)
    var f_lgt_spec: SpectralSample
    if lv.is_light == Int32(1):
        # Light emission: Le carries no cosine of its own, so the emitting
        # surface's cosine is applied HERE, explicitly -- see the geometry
        # note below for why it can no longer come from a shared G.
        var ln = lv.normal.to_simd()
        var cos_l = dot(neg_dir, ln)
        if cos_l <= Float32(0):
            return (SpectralSample(Float32(0)), False)
        f_lgt_spec = spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, lv.alb.r, lv.alb.g, lv.alb.b, wl) * cos_l
    else:
        f_lgt_spec = _eval_vertex_spectral(lv, neg_dir, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wl, adjoint=True)
    var f_combined = f_cam_spec * f_lgt_spec

    # GEOMETRY: 1/d^2 ONLY. Each endpoint's cosine is already inside its f_cos.
    #
    # This used to be G = |cos_cv| |cos_lv| / d^2, on top of two
    # _eval_vertex_spectral values that are f*cos -- so every connection
    # applied each endpoint's surface cosine TWICE and delivered roughly half
    # its MIS share. It hid for a long time because the white-furnace cells
    # cannot see it: there the camera and light vertices lie on one plane, G
    # is ~0 and connections contribute nothing at all. The closed cavity,
    # where connections carry most of the answer, read 0.638 at the default
    # light-path count against an analytic 1.0 -- and the per-strategy split
    # showed the deficit tracking connect's own contribution almost exactly
    # (0.356 delivered / 0.35 missing at N=1024, 0.261 / 0.25 at N=4096).
    #
    # Same mistake LobeEval's docstring records merging having made (f*cos
    # where the estimator wanted bare f); this is its connection-side twin.
    # Putting the cosines in f_cos rather than dividing them back out keeps
    # the lobe's OWN cosine at each end, which is not always |cos(dir, n)| --
    # hair carries the fibre cosine and a volume none -- so this is right for
    # every lobe kind, where `f_cos / cos_used * cos*cos` would not be. It is
    # exactly the form _bdpt_connect_to_camera (t=1) already uses.
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
    ConnectVertices), factored out of _connect_unweighted exactly as
    _bdpt_merge_mis_weight was factored out of the merge loop (8c934ec5).

    `_VCM_CAMIS` (plan deep-hugging-locket stage S3): when both endpoints
    are in CAMIS's Class, returns camis_eval_connect_s1's (lv is the light
    source, s=1) or camis_eval_connect's (s>=2, gathering lv's light-side
    records via _camis_gather_light) exact correlation-aware weight instead
    -- partition of unity breaks unless connect is converted together with
    merge, which is why this is the SAME comptime flag and the SAME
    in-Class gate as _bdpt_merge_mis_weight. Returns 1 (the prior unweighted
    behavior) for a pair outside both branches below."""
    var neg_dir = -dir
    # VCM Stage 2b/2d: real MIS weight for diffuse/conductor/light-source
    # connections (see _connect's docstring + _bdpt_vertex_pdfs'/
    # project_vcm_stage2_mis_derivation memory for the full derivation and
    # its "not independently verified" caveats).
    #
    # GEOMETRIC normals here, unlike the shading normal the two f_cos in
    # _connect_unweighted were evaluated against: these are the solid-angle
    # -> area density conversion (pbrt's Vertex::ConvertDensity, which reads
    # ng()), not a BSDF cosine. A perturbed normal in a density is a bias.
    if _bdpt_vertex_mis_scoped(cv) and (lv.is_light == Int32(1) or _bdpt_vertex_mis_scoped(lv)):
        var cos_cv = _vcm_cos_at(cv, dir)
        var cos_lv = _vcm_cos_at(lv, neg_dir)
        var (camera_bsdf_dir_pdf_w, camera_bsdf_rev_pdf_w) = _bdpt_vertex_pdfs(cv, dir, sd)
        # Light-source vertex: forward and reverse pdf are the SAME
        # cosine-weighted-emission formula (no real "wo" to distinguish a
        # direction from, unlike a genuine BSDF bounce) -- REASONED, not
        # independently verified against a reference light-source-specific
        # connect path.
        var light_bsdf_dir_pdf_w: Float32
        var light_bsdf_rev_pdf_w: Float32
        if lv.is_light == Int32(1):
            light_bsdf_dir_pdf_w = cos_lv / PI
            light_bsdf_rev_pdf_w = cos_lv / PI
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
            # competitor is only the camera BSDF hitting the same point:
            #     wLight = bsdfDirPdfA / directPdfA     (SmallVCM DirectIllumination)
            # The generic formula below ran on the light origin's stored
            # carries instead -- an extra merge-at-the-light term (eta), an
            # extra connection term built from the light path's OWN emission
            # cosine, and a stray cos_l on the 1/p_A term. Weights summed to
            # 0.95-0.98 (Scenes/vcm_area_mis_derivation.py); w_camera below
            # was already exact.
            w_light = camera_bsdf_dir_pdf_a / max(lv.pdf_fwd, Float32(1e-30))
        else:
            var eta_lv = _vcm_eta_at(sd, lv, mis_vm_weight_factor)
            w_light = camera_bsdf_dir_pdf_a * (eta_lv + lv.dVCM + lv.dVC * light_bsdf_rev_pdf_w)
        var w_camera = light_bsdf_dir_pdf_a * (eta_cv + cv.dVCM + cv.dVC * camera_bsdf_rev_pdf_w)
        return Float32(1) / (w_light + Float32(1) + w_camera)
    elif cv.is_surface == Int32(0) and lv.is_light == Int32(1) and lv.pdf_fwd > Float32(0):
        # Volume vertex -> light source: MIS against the phase-hit strategy
        # (the camera path continuing by uniform-sphere sampling and landing
        # on this emitter), whose pdf is the isotropic phase pdf 1/(4pi).
        # lv.pdf_fwd is the light point's area pdf; convert to solid angle at
        # cv. The hit side computes the same pdf in the same measure.
        var cos_lv_vol = abs(dot(neg_dir, lv.normal.to_simd()))
        if cos_lv_vol > Float32(1e-8):
            var pdf_light_w_vol = lv.pdf_fwd * dist2 / cos_lv_vol
            return power_heuristic(pdf_light_w_vol, INV_FOUR_PI)
    return Float32(1)

# ── Main BDPT render ──────────────────────────────────────────────────────────
