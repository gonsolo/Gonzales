"""Which of VCM's strategies exist at a vertex and with what merge density."""
from std.math import sqrt, floor, max, min
from gonzales.core.geometry import Point3f, PI
from gonzales.scene.materials import LobeKind
from gonzales.scene.bvh import SceneView, _is_real_ptr, curve_radius
from gonzales.estimators.photon_grid import _hash_cell, _PHOTON_BUCKET_CAP, grid_keep
from gonzales.paths.bdpt_vertex import _BDPT_MAX_VERTS, BDPTVertex
from gonzales.scene.media import medium_kernel_radius


@always_inline
def _vcm_depth(ref sd: SceneView) -> Int:
    """VCM's full-path length limit: the scene's maxdepth, glass bounces included."""
    var d = Int(sd.vcmMaxDepth)
    return d if d > 0 else _VCM_MAX_STORED


comptime _VCM_MAX_STORED = _BDPT_MAX_VERTS - 1   # non-delta vertices of a full path, or a strategy would not fit the cache


@always_inline
def _vcm_past_depth(ref sd: SceneView, lvc: Pointer[BDPTVertex, MutUntrackedOrigin], lp_idx: Int, local: Int, cam_count: Int) -> Bool:
    return _vcm_light_count(lvc, lp_idx, local) + cam_count > _vcm_depth(sd)


@always_inline
def _vcm_past_cache(lvc: Pointer[BDPTVertex, MutUntrackedOrigin], lp_idx: Int, local: Int, cam_count: Int) -> Bool:
    """A path the light cache cannot hold whole: only the camera strategies (hit, NEE) exist for it, and only they share its weight."""
    return _vcm_light_count(lvc, lp_idx, local) - Int(lvc[unsafe_offset=lp_idx * _BDPT_MAX_VERTS + local].n_delta) + cam_count > _VCM_MAX_STORED


@always_inline
def _vcm_too_long(ref sd: SceneView, lvc: Pointer[BDPTVertex, MutUntrackedOrigin], lp_idx: Int, local: Int, cam_count: Int) -> Bool:
    """Light slot `local` plus `cam_count` camera vertices is no light strategy."""
    return _vcm_past_depth(sd, lvc, lp_idx, local, cam_count) or _vcm_past_cache(lvc, lp_idx, local, cam_count)


@always_inline
def _vcm_light_count(lvc: Pointer[BDPTVertex, MutUntrackedOrigin], lp_idx: Int, local: Int) -> Int:
    """Interior-vertex count of light-path slot `local`, delta bounces included:
    slot 0 is the light point itself for an area light (count 0), but already
    the first surface hit for a point / distant / environment light, which
    stores no origin."""
    var n = local if lvc[unsafe_offset=lp_idx * _BDPT_MAX_VERTS].is_light == Int32(1) else local + 1
    return n + Int(lvc[unsafe_offset=lp_idx * _BDPT_MAX_VERTS + local].n_delta)


@always_inline
def _vcm_keep(ref sd: SceneView, p: Point3f) -> Float32:
    """Variance-aware merge MIS: the probability that thinning keeps a light
    vertex at `p`, estimated from the PREVIOUS pass's bucket counts"""
    if not _is_real_ptr(sd.vcmKeepCounts):
        return Float32(1)
    var h = _hash_cell(Int(floor(p.x * sd.vcmKeepInvCell)), Int(floor(p.y * sd.vcmKeepInvCell)),
                       Int(floor(p.z * sd.vcmKeepInvCell)))
    var n = Float32(sd.vcmKeepCounts[unsafe_offset=h]) * sd.vcmKeepScale
    if _vcm_budget_active(sd):
        # Per-cell budget: the keep probability that minimises merge variance
        # sum V/k for a fixed merge work sum Q n k -- k = lambda sqrt(V/(Q n)),
        if n <= Float32(0):
            return Float32(1)
        var q = sd.vcmStatIn[unsafe_offset=2 * h] * sd.vcmKeepScale
        var v = sd.vcmStatIn[unsafe_offset=2 * h + 1] * sd.vcmKeepScale
        var k_min = min(Float32(1), _VCM_BUDGET_MIN_KEEP / n)
        if q <= Float32(0) or v <= Float32(0):
            return k_min
        return max(k_min, min(Float32(1), sd.vcmLambda * sqrt(v / (q * n))))
    var cap = Float32(_vcm_cap(sd))
    if n <= cap:
        return Float32(1)
    return cap / n


@always_inline
def _vcm_cap(ref sd: SceneView) -> Int32:
    """VCM's merge-bucket cap: --vcm-cap, else _PHOTON_BUCKET_CAP."""
    return sd.vcmBucketCap if sd.vcmBucketCap > Int32(0) else _PHOTON_BUCKET_CAP


comptime _VCM_BUDGET_MIN_KEEP = Float32(4.0)


# Divide every merge by its gather disk's surface coverage (sppm.mojo's
# gather_disk_coverage). A comptime switch for A/B builds only.
comptime _MERGE_COVERAGE = True


# Per-candidate merge-visit counters (visit_naive/footprint/thin) for the
# candidate-reduction figure. Each costs two extra disk tests and a grid_keep
comptime _VCM_VISIT_INSTRUMENT = False


comptime _VCM_CAMIS = False


# Camera CAMIS records per path: one per stored vertex (_BDPT_MAX_VERTS bounds
# those), or a 1-element stand-in when the hybrid is compiled out.
comptime _CAMIS_CAM_RECS = _BDPT_MAX_VERTS if _VCM_CAMIS else 1


# Light CAMIS records per `lvc` slot: 1, or 0 when compiled out (the drivers
# then allocate a single dummy element).
comptime _CAMIS_LVC_PER_SLOT = 1 if _VCM_CAMIS else 0


# Stage S3 debug identity check: force every CAMIS c to 1 at every weight
# site (camis_eval_*'s own force_c1 argument). With this True, a
comptime _CAMIS_FORCE_C1 = False


@always_inline
def _vcm_budget_active(ref sd: SceneView) -> Bool:
    """Whether this pass thins by the per-cell budget (--vcm-budget, from the
    second pass on) rather than the fixed cap."""
    return _is_real_ptr(sd.vcmStatIn) and _is_real_ptr(sd.vcmKeepCounts) and sd.vcmLambda > Float32(0)


# Merge radius per camera vertex, in pixels of image footprint. A global
# radius sized to the scene (SmallVCM's 0.003 * scene radius) is ~0.24 m on
comptime _VCM_FOOTPRINT_PIXELS = Float32(2.0)


@always_inline
def _vcm_merge_radius_at(ref sd: SceneView, p: Point3f, is_volume: Bool = False) -> Float32:
    """Merge radius at `p`: _VCM_FOOTPRINT_PIXELS of image footprint at that
    point's distance from the camera, never more than this pass's global"""
    var scale = _VCM_RADIUS_VOLUME_SCALE if is_volume else Float32(1.0)
    if sd.vcmFootprint <= Float32(0) or sd.vcmMergeR <= Float32(0):
        return sd.vcmMergeR * scale
    var ceil = sd.vcmMergeR * scale
    var d = p - Point3f(sd.vcmCamX, sd.vcmCamY, sd.vcmCamZ)
    var r = sd.vcmFootprint * scale * sqrt(d.x * d.x + d.y * d.y + d.z * d.z)
    return min(ceil, max(r, ceil * Float32(1e-3)))


@always_inline
def _vcm_volume_radius_at(ref sd: SceneView, v: BDPTVertex) -> Float32:
    """Merge radius at volume vertex `v`."""
    var r = _vcm_merge_radius_at(sd, v.pos, is_volume=True)
    if Int(v.med_idx) < 0:
        return r
    return min(r, medium_kernel_radius(sd.mediums[unsafe_offset=Int(v.med_idx)], sd.grids, sd.nvdbGrids, v.pos.to_simd()))


@always_inline
def _vcm_eta_scale(ref sd: SceneView, p: Point3f) -> Float32:
    """eta(x) / eta: merging's MIS density at `p` relative to the global
    N pi r_pass^2 -- thinning's keep probability times the radius shrink."""
    var s = _vcm_keep(sd, p)
    if sd.vcmFootprint > Float32(0) and sd.vcmMergeR > Float32(0):
        var q = _vcm_merge_radius_at(sd, p) / sd.vcmMergeR
        s *= q * q
    return s


@always_inline
def _vcm_eta_scale_vol(ref sd: SceneView, v: BDPTVertex) -> Float32:
    """Same ratio as _vcm_eta_scale but for the BALL kernel measure a volume
    vertex merges in: keep(x) * (r(x)/r_pass)^3, cubed rather than squared,
    since (4/3)pi r^3 scales with the radius CUBED
    (Scenes/vcm_volume_mis_derivation.py, kernel_measure)."""
    var s = _vcm_keep(sd, v.pos)
    if sd.vcmMergeR > Float32(0):
        var q = _vcm_volume_radius_at(sd, v) / (sd.vcmMergeR * _VCM_RADIUS_VOLUME_SCALE)
        s *= q * q * q
    return s


@always_inline
def _vcm_on_fibre(v: BDPTVertex) -> Bool:
    return v.is_surface == Int32(1) and v.mat_kind == LobeKind.hair


@always_inline
def _vcm_fibre_scale(ref sd: SceneView, p: Point3f, curve_idx: Int, v_global: Float32) -> Float32:
    """eta(x) / eta on a hair fibre: the fibre kernel's measure 2 r(x) w(x) -- the strand within r, a ribbon of its
    width w facing every direction -- over pi r_pass^2, times thinning's keep probability. 0 without a pass radius."""
    if sd.vcmMergeR <= Float32(0):
        return Float32(0)
    var r = _vcm_merge_radius_at(sd, p) if sd.vcmFootprint > Float32(0) else sd.vcmMergeR
    var w = Float32(2) * curve_radius(sd.curves, curve_idx, v_global)
    return _vcm_keep(sd, p) * Float32(2) * r * w / (PI * sd.vcmMergeR * sd.vcmMergeR)


@always_inline
def _vcm_mergeable(ref sd: SceneView, v: BDPTVertex) -> Bool:
    """Whether merging may estimate a density here: a fibre only with its kernel's pass radius."""
    return not _vcm_on_fibre(v) or sd.vcmMergeR > Float32(0)


@always_inline
def _vcm_eta_at(ref sd: SceneView, v: BDPTVertex, mis_vm_weight_factor: Float32) -> Float32:
    """This vertex's own eta(x) = N * (kernel measure at x), fully scaled --
    a DISK area on a surface, a BALL volume in a medium; 0 where it cannot merge. `mis_vm_weight_factor`"""
    if v.mat_kind == LobeKind.sss_exit:
        return Float32(0)
    if _vcm_on_fibre(v):
        return mis_vm_weight_factor * _vcm_fibre_scale(sd, v.pos, Int(v.hair_curve_idx), v.hair_v)
    if v.is_surface == Int32(0):
        var s3 = _VCM_RADIUS_VOLUME_SCALE * _VCM_RADIUS_VOLUME_SCALE * _VCM_RADIUS_VOLUME_SCALE
        return mis_vm_weight_factor * (Float32(4.0 / 3.0) * sd.vcmMergeR * s3) * _vcm_eta_scale_vol(sd, v)
    return mis_vm_weight_factor * _vcm_eta_scale(sd, v.pos)


@always_inline
def _vcm_inv_eta_at(ref sd: SceneView, v: BDPTVertex, mis_vc_weight_factor: Float32) -> Float32:
    """1/eta(x) at this vertex, in the mis_vc_weight_factor=1/eta_pass
    convention _bdpt_merge_mis_weight uses (the merge weight's dVM = dVC/eta
    rule divides by eta rather than multiplying, see vcm_mis.mojo's
    vcm_scatter_carries docstring)."""
    if _vcm_on_fibre(v):
        var fs = _vcm_fibre_scale(sd, v.pos, Int(v.hair_curve_idx), v.hair_v)
        return mis_vc_weight_factor / fs if fs > Float32(0) else Float32(0)
    if v.is_surface == Int32(0):
        var s3 = _VCM_RADIUS_VOLUME_SCALE * _VCM_RADIUS_VOLUME_SCALE * _VCM_RADIUS_VOLUME_SCALE
        return mis_vc_weight_factor / ((Float32(4.0 / 3.0) * sd.vcmMergeR * s3) * _vcm_eta_scale_vol(sd, v))
    return mis_vc_weight_factor / _vcm_eta_scale(sd, v.pos)


# Volume merges get their own, much larger, fraction: a medium's own extent
# (Scenes/thinning/*.pbrt's rooms, Bitterli's volumetric-caustic) sets the
comptime _VCM_RADIUS_VOLUME_SCALE = Float32(6.667)
