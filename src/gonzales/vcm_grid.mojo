# VCM parameters (depth, keep, radius) and the merge grid.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from max.algorithm import parallelize
from std.math import sqrt, floor, max, min, abs, pow
from std.memory.alloc import unsafe_alloc
from .geometry import Point3f, Vec3f, dot
from .materials import LobeKind
from .primitives import Ray, Intersection, PrimId
from .bvh import SceneView, traverse_bvh2_core, test_spheres, _is_real_ptr
from .sppm import (
    _HSIZE, _hash_cell, _PHOTON_BUCKET_CAP, grid_reset_cell, grid_count, grid_keep, grid_coin_bits, grid_push,
)
from .bdpt_vertex import _BDPT_MAX_VERTS, BDPTVertex
from .bdpt_eval import _bdpt_vertex_mis_scoped

@always_inline
def _vcm_depth(ref sd: SceneView) -> Int:
    """VCM's full-path length limit d: min(scene maxdepth, _BDPT_MAX_VERTS - 1).

    The balance weights assume every strategy that COULD produce a path does
    run. That only holds if every strategy is limited by the same FULL-path
    length -- SmallVCM's mMaxPathLength. Before 2026-09-24 each subpath was
    capped at _BDPT_MAX_VERTS on its own, so for a path longer than that only
    some of its strategies existed and the weights reserved shares for the
    rest: closed-cavity.vcm read 0.927 once the emission-hit weight became
    exact (the old two-strategy weight had been over-crediting emission hits
    and hid it). Counted in non-delta interior vertices on both subpaths (a
    delta bounce is no strategy endpoint), so a path reads the same length
    whichever strategy made it. The cap keeps t=1 splats possible: a finite
    light path stores its origin plus up to d interior vertices."""
    var d = Int(sd.vcmMaxDepth)
    if d <= 0 or d > _BDPT_MAX_VERTS - 1:
        d = _BDPT_MAX_VERTS - 1
    return d


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
    vertex at `p`, estimated from the PREVIOUS pass's bucket counts
    (sd.vcmKeep*). Merging there really runs on keep * N photons, so its MIS
    density is keep * eta rather than eta, and the balance heuristic should
    lean on the other strategies exactly where photons were thinned.

    This only has to be a CONSISTENT function of position -- every strategy
    of a path asks it about the same vertices -- for the weights to stay a
    partition of unity; it does not have to equal this pass's exact keep
    probability, which is what the thinning itself (and so unbiasedness)
    uses. That is why a pass may read the previous pass's table: this pass's
    counts do not exist yet while its light paths are being traced and their
    dVC carries built. 1 when there is no previous pass."""
    if not _is_real_ptr(sd.vcmKeepCounts):
        return Float32(1)
    var h = _hash_cell(Int(floor(p.x * sd.vcmKeepInvCell)), Int(floor(p.y * sd.vcmKeepInvCell)),
                       Int(floor(p.z * sd.vcmKeepInvCell)))
    var n = Float32(sd.vcmKeepCounts[unsafe_offset=h]) * sd.vcmKeepScale
    if _vcm_budget_active(sd):
        # Per-cell budget: the keep probability that minimises merge variance
        # sum V/k for a fixed merge work sum Q n k -- k = lambda sqrt(V/(Q n)),
        # lambda meeting the budget (vcm_render_gpu). Q: merge queries in the
        # cell, V: their contribution's second moment at full keep, n: its
        # light vertices, all from the previous pass. This SAME value thins
        # the insert and weights the gather (1/k), so a cell at k = 0 simply
        # has no merging, and the MIS gives the other strategies its share.
        # Floored at ~_VCM_BUDGET_MIN_KEEP expected vertices, so no cell goes
        # dark for good: at k = 0 it would never produce statistics again.
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

# CAMIS-prefix class-gated hybrid (plan deep-hugging-locket; vcm_camis.mojo;
# Scenes/vcm_camis_hybrid_derivation.py). ON: the carries' third slot (dVM,
# never read) becomes dVC0, and both subpaths write the per-vertex CAMIS
# records -- camera ones into a local array threaded through the bounce loop,
# light ones into `lvc_camis`, parallel to `lvc`. Stage S1 only BUILDS that
# state; no weight reads it yet (S2 evaluates, S3 converts every weight
# site). OFF (the default) must stay byte-identical to the pre-CAMIS
# renderer: every CAMIS statement sits under `comptime if _VCM_CAMIS`, the
# carry functions take it as a comptime parameter, and the state arrays
# shrink to one element, so a normal build dead-code-eliminates all of it.
# Not supported by the wavefront GPU driver yet (it refuses to run).
# Multi-level merge index. The thinning grid's cells are sized to the pass
# radius r_pass, but a footprint-scaled query's disk is often 10-1000x smaller
# in area, so a query walking its 27 coarse cells tests hundreds of photons to
# accept a handful (measured: 0.02-0.2% accepted) and neighbouring threads walk
# wildly different chain lengths. The KEPT photons are therefore also linked
# into `_VCM_FINE_LEVELS` finer hash grids (cell r_pass/2^l, level l); a query
# walks the finest level whose cell still covers its radius. Thinning, its
# keep probabilities and the MIS densities stay on the coarse cells, and the
# acceptance test is unchanged, so the accepted photon set -- and the estimate
# -- are the same; only the lookup is cheaper.
#   merge_heads: [coarse heads | coarse counts | level 1 heads | ... | level L]
#   merge_next:  _VCM_MN_STRIDE links per slot, k * stride + level
comptime _VCM_FINE_LEVELS = 4
comptime _VCM_MN_STRIDE = 1 + _VCM_FINE_LEVELS
comptime _VCM_HEADS_SIZE = (2 + _VCM_FINE_LEVELS) * _HSIZE

# Per-candidate merge-visit counters (visit_naive/footprint/thin) for the
# candidate-reduction figure. Each costs two extra disk tests and a grid_keep
# hash lookup per candidate in the hottest loop of the register-saturated
# merge kernel, so they are off unless a figure needs them.
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
# _VCM_CAMIS=True build must reproduce the _VCM_CAMIS=False image up to
# float noise -- confirms the gather/threading plumbing (right LVC slot,
# right vertex, right per-path indices) independent of whether the actual
# correlation correction is right. Never True together with _VCM_CAMIS=False
# (nothing reads it then).
comptime _CAMIS_FORCE_C1 = False


@always_inline
def _vcm_budget_active(ref sd: SceneView) -> Bool:
    """Whether this pass thins by the per-cell budget (--vcm-budget, from the
    second pass on) rather than the fixed cap."""
    return _is_real_ptr(sd.vcmStatIn) and _is_real_ptr(sd.vcmKeepCounts) and sd.vcmLambda > Float32(0)

# Merge radius per camera vertex, in pixels of image footprint. A global
# radius sized to the scene (SmallVCM's 0.003 * scene radius) is ~0.24 m on
# barcelona-pavilion, as wide as the candle lanterns: the gather disk hangs off
# the surface, the photons on the neighbouring faces are rightly rejected, and
# the estimate -- normalised by the whole disk -- reads low. Merging then gets
# the MOST weight exactly where photons are densest.
comptime _VCM_FOOTPRINT_PIXELS = Float32(2.0)


@always_inline
def _vcm_merge_radius_at(ref sd: SceneView, p: Point3f, is_volume: Bool = False) -> Float32:
    """Merge radius at `p`: _VCM_FOOTPRINT_PIXELS of image footprint at that
    point's distance from the camera, never more than this pass's global
    radius (which sets the grid cells) and shrinking with it pass by pass.

    A function of POSITION only, like _vcm_keep, so every strategy of a path
    agrees on merging's density at each vertex: eta(x) = N pi r(x)^2.

    `is_volume=True` scales BOTH the footprint term and the ceiling by
    _VCM_RADIUS_VOLUME_SCALE -- a volume merge's true feature scale (light
    scattering in a medium, found e.g. investigating Bitterli's
    volumetric-caustic) is set by the medium's own extent, not by the same
    fraction of the scene bounding sphere that works for surface merges
    (a lantern panel, a wall). Keeping the two fractions separate protects
    every already-measured surface number (shade/barcelona/etc.) from this
    change -- is_volume=False (the default) is byte-identical to before."""
    var scale = _VCM_RADIUS_VOLUME_SCALE if is_volume else Float32(1.0)
    if sd.vcmFootprint <= Float32(0) or sd.vcmMergeR <= Float32(0):
        return sd.vcmMergeR * scale
    var ceil = sd.vcmMergeR * scale
    var d = p - Point3f(sd.vcmCamX, sd.vcmCamY, sd.vcmCamZ)
    var r = sd.vcmFootprint * scale * sqrt(d.x * d.x + d.y * d.y + d.z * d.z)
    return min(ceil, max(r, ceil * Float32(1e-3)))


@always_inline
def _vcm_grid_inv_cell(ref sd: SceneView, radius_i: Float32) -> Float32:
    """Coarse merge-grid cell: r_pass, or the volume radius in a scene with media
    (the 3x3x3 search must cover it; the fine levels keep surface lookups cheap)."""
    var scale = _VCM_RADIUS_VOLUME_SCALE if Int(sd.mediumCount) > 0 else Float32(1.0)
    return Float32(1.0) / max(radius_i * scale, Float32(1e-6))


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
def _vcm_eta_scale_vol(ref sd: SceneView, p: Point3f) -> Float32:
    """Same ratio as _vcm_eta_scale but for the BALL kernel measure a volume
    vertex merges in: keep(x) * (r(x)/r_pass)^3, cubed rather than squared,
    since (4/3)pi r^3 scales with the radius CUBED
    (Scenes/vcm_volume_mis_derivation.py, kernel_measure)."""
    var s = _vcm_keep(sd, p)
    if sd.vcmFootprint > Float32(0) and sd.vcmMergeR > Float32(0):
        var q = _vcm_merge_radius_at(sd, p, is_volume=True) / (sd.vcmMergeR * _VCM_RADIUS_VOLUME_SCALE)
        s *= q * q * q
    return s

@always_inline
def _vcm_eta_at(ref sd: SceneView, v: BDPTVertex, mis_vm_weight_factor: Float32) -> Float32:
    """This vertex's own eta(x) = N * (kernel measure at x), fully scaled --
    a DISK area on a surface, a BALL volume in a medium. `mis_vm_weight_factor`
    is the pass-level N*pi*r_pass^2 (disk) constant every call site already
    carries; converting it to the equivalent ball constant at the same
    r_pass needs one extra factor of (4/3)*r_pass, since
    N*(4/3)*pi*r_pass^3 = (N*pi*r_pass^2) * (4/3)*r_pass."""
    if v.is_surface == Int32(0):
        var s3 = _VCM_RADIUS_VOLUME_SCALE * _VCM_RADIUS_VOLUME_SCALE * _VCM_RADIUS_VOLUME_SCALE
        return mis_vm_weight_factor * (Float32(4.0 / 3.0) * sd.vcmMergeR * s3) * _vcm_eta_scale_vol(sd, v.pos)
    return mis_vm_weight_factor * _vcm_eta_scale(sd, v.pos)

@always_inline
def _vcm_inv_eta_at(ref sd: SceneView, v: BDPTVertex, mis_vc_weight_factor: Float32) -> Float32:
    """1/eta(x) at this vertex, in the mis_vc_weight_factor=1/eta_pass
    convention _bdpt_merge_mis_weight uses (the merge weight's dVM = dVC/eta
    rule divides by eta rather than multiplying, see vcm_mis.mojo's
    vcm_scatter_carries docstring)."""
    if v.is_surface == Int32(0):
        var s3 = _VCM_RADIUS_VOLUME_SCALE * _VCM_RADIUS_VOLUME_SCALE * _VCM_RADIUS_VOLUME_SCALE
        return mis_vc_weight_factor / ((Float32(4.0 / 3.0) * sd.vcmMergeR * s3) * _vcm_eta_scale_vol(sd, v.pos))
    return mis_vc_weight_factor / _vcm_eta_scale(sd, v.pos)

@always_inline
def _vcm_cos_at(v: BDPTVertex, w: Vec3f) -> Float32:
    """|cos| between direction `w` and this vertex's own geometric normal --
    1.0 at a volume vertex, which has none (Scenes/vcm_volume_mis_derivation.py's
    geom_to_measure: no cosine in the solid-angle -> volume Jacobian, versus
    cos/d^2 -> area at a surface). v.normal is never set on a stored volume
    vertex (left at _null_vertex()'s (0,1,0) placeholder), so reading it
    directly without this guard is a real bug, not just an approximation."""
    if v.is_surface == Int32(0):
        return Float32(1.0)
    return abs(dot(w, v.normal.to_simd()))


@always_inline
def _bdpt_merge_slot_bucket(
    k: Int,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
) -> Int:
    """Hash bucket of LVC slot `k`, or -1 for an unused tail slot of its
    light path's per-path slice (VCM Stage 2b storage layout, see
    _bdpt_store_lvc_vertex's docstring -- `k` ranges over the full
    `n_light_paths * _BDPT_MAX_VERTS` capacity, not just the vertices
    actually stored)."""
    var lp_idx = k // _BDPT_MAX_VERTS
    var local_idx = k % _BDPT_MAX_VERTS
    if local_idx >= Int(lvc_path_len[unsafe_offset=lp_idx]):
        return -1
    # Only vertices a gather can use go into the grid -- the same static test
    # _bdpt_merge_from_cache applies per photon. The rest (light-source
    # points above all: every light path contributes one, so a tiny emitter
    # stacks hundreds of thousands into a few cells) were walked and rejected
    # by every nearby query, and inflated the counts the thinning reads.
    var lv = lvc[unsafe_offset=k]
    # is_surface is NOT required here any more: a volume light vertex is a
    # legitimate merge candidate too (Scenes/vcm_volume_mis_derivation.py) --
    # kind matching against the camera vertex happens at gather time
    # (_bdpt_merge_from_cache), since this grid is shared by both kinds.
    if not (lv.is_delta == Int32(0) and lv.is_light == Int32(0)
            and lv.mat_kind != LobeKind.bssrdf and _bdpt_vertex_mis_scoped(lv)):
        return -1
    var ix = Int(floor(lvc[unsafe_offset=k].pos.x * inv_cell))
    var iy = Int(floor(lvc[unsafe_offset=k].pos.y * inv_cell))
    var iz = Int(floor(lvc[unsafe_offset=k].pos.z * inv_cell))
    return _hash_cell(ix, iy, iz)

@always_inline
def _bdpt_count_merge_vertex(
    k: Int,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    heads: Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
):
    """First pass of the grid build: tally slot `k` into its bucket's count."""
    var h = _bdpt_merge_slot_bucket(k, lvc, lvc_path_len, inv_cell)
    if h < 0:
        return
    grid_count(heads, h)

def _bdpt_insert_merge_vertex[use_gpu: Bool](
    k: Int,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    heads: Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    ref sd: SceneView,
):
    """Second pass of the grid build: insert LVC slot `k` into the merge hash
    grid, unless it's an unused tail slot or thinned out of an over-full
    bucket (see _PHOTON_BUCKET_CAP). Comptime-branches only on the
    bucket-head update primitive -- identical pattern to sppm.mojo's
    _sppm_insert_photon[use_gpu]."""
    var h = _bdpt_merge_slot_bucket(k, lvc, lvc_path_len, inv_cell)
    if h < 0:
        return
    var pos = lvc[unsafe_offset=k].pos
    if _vcm_budget_active(sd):
        # The per-cell budget's keep probability -- the one the gather
        # weights by and the MIS reads (_vcm_keep).
        var kp = _vcm_keep(sd, pos)
        if kp < Float32(1) and Float32(grid_coin_bits(k, pos)) * Float32(2.3283064e-10) >= kp:
            return
    elif not grid_keep(heads, h, k, pos, _vcm_cap(sd)):
        return
    merge_next[unsafe_offset=k * _VCM_MN_STRIDE] = grid_push[use_gpu](heads, h, k)
    comptime for l in range(1, _VCM_FINE_LEVELS + 1):
        var ic = inv_cell * Float32(1 << l)
        var hl = _hash_cell(Int(floor(pos.x * ic)), Int(floor(pos.y * ic)), Int(floor(pos.z * ic)))
        merge_next[unsafe_offset=k * _VCM_MN_STRIDE + l] = grid_push[use_gpu](heads.unsafe_offset((1 + l) * _HSIZE), hl, k)

@always_inline
def _vcm_reset_cell(heads: Pointer[Int32, MutUntrackedOrigin], h: Int):
    """Reset bucket h of the coarse grid and of every fine level."""
    grid_reset_cell(heads, h)
    comptime for l in range(1, _VCM_FINE_LEVELS + 1):
        heads[unsafe_offset=(1 + l) * _HSIZE + h] = Int32(-1)


def _bdpt_build_merge_grid(
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    n_light_paths: Int,
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    heads: Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    ref sd: SceneView,
):
    """CPU-only grid build (mirrors sppm.mojo's _build_grid): reset all
    buckets, then insert every LVC vertex via the SAME atomic-exchange
    insert the GPU kernel uses (parallel CPU workers race on bucket heads
    exactly like GPU threads would). Iterates the full
    `n_light_paths * _BDPT_MAX_VERTS` capacity;
    `_bdpt_insert_merge_vertex` itself skips each path's unused tail slots
    via `lvc_path_len`."""
    def reset_one(i: Int) {imm}:
        _vcm_reset_cell(heads, i)
    parallelize(reset_one, _HSIZE)

    def count_one(k: Int) {imm}:
        _bdpt_count_merge_vertex(k, lvc, lvc_path_len, heads, inv_cell)
    parallelize(count_one, n_light_paths * _BDPT_MAX_VERTS)

    def insert_one(k: Int) {imm}:
        _bdpt_insert_merge_vertex[True](k, lvc, lvc_path_len, merge_next, heads, inv_cell, sd)
    parallelize(insert_one, n_light_paths * _BDPT_MAX_VERTS)

# Initial merge radius as a fraction of the scene bounding sphere: SmallVCM's
# own default (config.hxx, mRadiusFactor = 0.003). It was 0.03 here -- 100x the
# gather AREA -- which in a large scene makes the disk far wider than the
# surface under it (barcelona: r ~ 1-2.5 m over a 0.5 m^2 chair seat), and a
# density estimate normalised by the whole disk reads low there. Harmless while
# merging carried little weight; once NEE is weighted correctly the balance
# heuristic hands direct sun to merging and that bias IS the image: the chairs
# read 0.54 (shadowed) / 0.33 (sunlit) of pbrt at 0.03, 1.02 / 0.97 at 0.003.
comptime _VCM_RADIUS_FRACTION = Float32(0.003)
# Volume merges get their own, much larger, fraction: a medium's own extent
# (Scenes/thinning/*.pbrt's rooms, Bitterli's volumetric-caustic) sets the
# right feature scale, not the scene's whole bounding sphere the way a
# surface lantern/panel does. 0.02, ~6.7x, is the value that first made
# Bitterli's volumetric-caustic beam visible (2026-09-29 investigation);
# not yet auto-derived from anything scene-specific -- see
# _vcm_merge_radius_at's docstring.
comptime _VCM_RADIUS_VOLUME_SCALE = Float32(6.667)
comptime _VCM_RADIUS_ALPHA = Float32(2.0) / Float32(3.0)  # Georgiev 2012's typical choice


def _camera_typical_distance(
    rasterToCamera: Pointer[Float32, MutUntrackedOrigin],
    cameraToWorld: Pointer[Float32, MutUntrackedOrigin],
    film_w: Int, film_h: Int,
    ref sd: SceneView,
    percentile: Float32 = Float32(0.5),
) -> Float32:
    """--vcm-radius-from-camera (EXPERIMENTAL, opt-in only): median
    primary-ray hit distance across a coarse 16x16 grid of the film, as an
    alternative to _scene_bounding_sphere's whole-scene radius for
    vcm_merge_radius's `scene_radius` argument -- both are just "the basis
    _VCM_RADIUS_FRACTION scales to get the merge-radius ceiling", so this
    can feed the SAME unchanged vcm_merge_radius unmodified.

    The bounding sphere is disconnected from where the camera is actually
    looking: on a scene viewed from far away relative to its own extent,
    the whole-scene radius can already sit BELOW what the footprint
    mechanism would ask for at every visible point, so the per-vertex clamp
    in _vcm_merge_radius_at binds everywhere and footprint scaling never
    engages at all -- measured on veach-bidir this session: naive and
    footprint-only candidate counts were bit-identical at all 262144 lit
    pixels, 0% showing any reduction. Using median viewing depth as the
    ceiling's basis instead ties it to what the camera sees rather than
    the scene's whole extent.

    Host-side, run once before the sample loop -- same cost class as
    camera_footprint's own 512-sample diagonal scan (footprint.mojo), but
    tracing real rays through the BVH/spheres for DEPTH, not differential
    directions for footprint spread. Returns 0 when nothing is hit (e.g.
    the camera looks entirely at background/infinite lights); the caller
    must fall back to _scene_bounding_sphere's radius in that case."""
    var org = Point3f(cameraToWorld[unsafe_offset=12], cameraToWorld[unsafe_offset=13], cameraToWorld[unsafe_offset=14])
    comptime GRID = 16
    var hits = List[Float32]()
    var isects = unsafe_alloc[Intersection](1)
    for gy in range(GRID):
        for gx in range(GRID):
            var filmX = (Float32(gx) + Float32(0.5)) / Float32(GRID) * Float32(film_w)
            var filmY = (Float32(gy) + Float32(0.5)) / Float32(GRID) * Float32(film_h)
            # rasterToCamera (column-major 4x4), no filter offset -- same
            # transform as render_aux_buffers' trace_pixel (bvh.mojo).
            var cx = rasterToCamera[unsafe_offset=0]*filmX + rasterToCamera[unsafe_offset=4]*filmY + rasterToCamera[unsafe_offset=12]
            var cy = rasterToCamera[unsafe_offset=1]*filmX + rasterToCamera[unsafe_offset=5]*filmY + rasterToCamera[unsafe_offset=13]
            var cz = rasterToCamera[unsafe_offset=2]*filmX + rasterToCamera[unsafe_offset=6]*filmY + rasterToCamera[unsafe_offset=14]
            var cw = rasterToCamera[unsafe_offset=3]*filmX + rasterToCamera[unsafe_offset=7]*filmY + rasterToCamera[unsafe_offset=15]
            if cw != Float32(0.0) and cw != Float32(1.0):
                cx /= cw; cy /= cw; cz /= cw
            var cl = sqrt(cx*cx + cy*cy + cz*cz)
            if cl > Float32(0): cx /= cl; cy /= cl; cz /= cl
            var dir = Vec3f(
                cameraToWorld[unsafe_offset=0]*cx + cameraToWorld[unsafe_offset=4]*cy + cameraToWorld[unsafe_offset=8]*cz,
                cameraToWorld[unsafe_offset=1]*cx + cameraToWorld[unsafe_offset=5]*cy + cameraToWorld[unsafe_offset=9]*cz,
                cameraToWorld[unsafe_offset=2]*cx + cameraToWorld[unsafe_offset=6]*cy + cameraToWorld[unsafe_offset=10]*cz,
            )
            var dl = dir.length()
            if dl > Float32(0): dir = dir / dl
            var ray = Ray(org, dir)
            isects[unsafe_offset=0] = Intersection(PrimId(-1, -1, 0, -1, 0, 0, 0, 0), Float32(1e38), 0.0, 0.0, Int8(0), 0, 0, 0)
            if Int(sd.meshCount) > 0 or Int(sd.curveCount) > 0 or Int(sd.instanceCount) > 0:
                traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), isects.unsafe_offset(0),
                                   sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
            if Int(sd.sphereCount) > 0:
                test_spheres(sd.spheres, Int(sd.sphereCount), ray, isects.unsafe_offset(0))
            if isects[unsafe_offset=0].hit != Int8(0):
                hits.append(isects[unsafe_offset=0].tHit)
    isects.unsafe_free()
    if len(hits) == 0:
        return Float32(0)
    # <=256 samples: a plain insertion sort avoids pulling in a sort dependency
    # for what is a one-time, host-side, pre-render cost.
    for i in range(1, len(hits)):
        var v = hits[i]
        var j = i - 1
        while j >= 0 and hits[j] > v:
            hits[j + 1] = hits[j]
            j -= 1
        hits[j + 1] = v
    # percentile=0.5 (default) is the median; a higher percentile pushes the
    # basis outward, toward the farther end of what the camera sees -- see
    # --vcm-radius-cam-percentile.
    var pidx = Int(percentile * Float32(len(hits) - 1) + Float32(0.5))
    if pidx < 0: pidx = 0
    if pidx >= len(hits): pidx = len(hits) - 1
    return hits[pidx]


@always_inline
def vcm_merge_radius(scene_radius: Float32, si: Int) -> Float32:
    """The progressive VCM merge radius for sample `si` (0-based).

        r_i = _VCM_RADIUS_FRACTION * scene_radius / (i+1)^(0.5*(1-alpha))

    Hachisuka & Jensen 2008 via Georgiev et al. 2012 Eq. 11: ONE global
    radius shared by every pixel this sample, shrinking monotonically --
    distinct from sppm.mojo's per-pixel Knaus-Zwicker scheme.

    Shared because this was written out THREE times (the CPU driver and
    two GPU drivers) with the fraction and the exponent hand-copied into
    each. That is not hypothetical drift: an experiment that changed only
    the CPU copy produced a perfect no-op and nearly sent the 2026-09-18
    merge-leak investigation down the wrong path, because the --gpu render
    under test was reading a different constant entirely."""
    return (scene_radius * _VCM_RADIUS_FRACTION
            / pow(Float32(si + 1), Float32(0.5) * (Float32(1) - _VCM_RADIUS_ALPHA)))
