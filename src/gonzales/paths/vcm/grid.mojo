"""VCM's merge grid over the light vertex cache, and the pass radius."""
from max.algorithm import parallelize
from std.atomic import Atomic
from std.math import sqrt, floor, max, pow
from std.memory.alloc import unsafe_alloc
from gonzales.core.geometry import Point3f, Vec3f
from gonzales.scene.appearance.materials import LobeKind
from gonzales.scene.shapes.primitives import Intersection, PrimId
from gonzales.core.geometry import Ray
from gonzales.scene.accel.bvh import traverse_dispatch
from gonzales.scene.accel.bvh import SceneView, test_spheres
from gonzales.estimators.photon_grid import _HSIZE, _hash_cell, grid_reset_cell, grid_count, grid_keep, grid_coin_bits, grid_push
from gonzales.paths.vcm.vertex import _BDPT_MAX_VERTS, BDPTVertex
from gonzales.paths.vcm.eval import _bdpt_vertex_mis_scoped
from gonzales.paths.vcm.strategy import _vcm_budget_active, _vcm_cap, _vcm_keep


comptime _VCM_FINE_LEVELS = 4


# A photon's merge record: its position (3 float32), then its link on each level; 32 bytes, so a gather step reads one
# record and the vertex only for photons within reach.
comptime _VCM_MN_LINK = 3
comptime _VCM_MN_STRIDE = _VCM_MN_LINK + 1 + _VCM_FINE_LEVELS


# After the lists, each level's buckets are copied into contiguous ranges of records (pos, slot) for the gather. heads
# holds, past the lists' heads and counts, the LVC capacity, each level's fill cursor, then (start, count) per level and
# bucket; merge_next holds, past the link records, each level's sorted records.
comptime _VCM_RANGES = (2 + _VCM_FINE_LEVELS) * _HSIZE
comptime _VCM_RANGE_HDR = 8
comptime _VCM_HEADS_SIZE = _VCM_RANGES + _VCM_RANGE_HDR + 2 * (1 + _VCM_FINE_LEVELS) * _HSIZE
comptime _VCM_SORTED = 4
comptime _VCM_MN_SIZE = _VCM_MN_STRIDE + (1 + _VCM_FINE_LEVELS) * _VCM_SORTED   # merge_next int32s per LVC slot


# The coarse cell in a scene with media: the power of two at or above the volume radius scale, so that a fine level is
# r_pass itself (with 6.667 a surface query searched cells of 1.67 r).
comptime _VCM_GRID_VOLUME_SCALE = Float32(8.0)


@always_inline
def _vcm_grid_inv_cell(ref sd: SceneView, radius_i: Float32) -> Float32:
    """Coarse merge-grid cell: r_pass, or 8 r_pass in a scene with media (the 3x3x3 search must cover the volume
    radius; the fine levels keep surface lookups cheap)."""
    var scale = _VCM_GRID_VOLUME_SCALE if Int(sd.mediumCount) > 0 else Float32(1.0)
    return Float32(1.0) / max(radius_i * scale, Float32(1e-6))


@always_inline
def _bdpt_merge_slot_bucket(
    k: Int,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_path_len: Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
) -> Int:
    """Hash bucket of LVC slot `k`, or -1 for an unused tail slot of its
    light path's per-path slice (VCM Stage 2b storage layout, see"""
    var lp_idx = k // _BDPT_MAX_VERTS
    var local_idx = k % _BDPT_MAX_VERTS
    if local_idx >= Int(lvc_path_len[unsafe_offset=lp_idx]):
        return -1
    # Only vertices a gather can use go into the grid -- the same static test
    # _bdpt_merge_from_cache applies per photon. The rest (light-source
    var lv = lvc[unsafe_offset=k]
    # is_surface is NOT required here any more: a volume light vertex is a
    # legitimate merge candidate too (Scenes/vcm_volume_mis_derivation.py) --
    if not (lv.is_delta == Int32(0) and lv.is_light == Int32(0)
            and lv.mat_kind != LobeKind.sss_exit and _bdpt_vertex_mis_scoped(lv)):
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
    grid, unless it's an unused tail slot or thinned out of an over-full"""
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
    var rec = merge_next.unsafe_offset(k * _VCM_MN_STRIDE)
    var rp = rec.unsafe_bitcast[Float32]()
    rp[unsafe_offset=0] = pos.x
    rp[unsafe_offset=1] = pos.y
    rp[unsafe_offset=2] = pos.z
    rec[unsafe_offset=_VCM_MN_LINK] = grid_push[use_gpu](heads, h, k)
    comptime for l in range(1, _VCM_FINE_LEVELS + 1):
        var ic = inv_cell * Float32(1 << l)
        var hl = _hash_cell(Int(floor(pos.x * ic)), Int(floor(pos.y * ic)), Int(floor(pos.z * ic)))
        rec[unsafe_offset=_VCM_MN_LINK + l] = grid_push[use_gpu](heads.unsafe_offset((1 + l) * _HSIZE), hl, k)


@always_inline
def _vcm_reset_cell(heads: Pointer[Int32, MutUntrackedOrigin], h: Int):
    """Reset bucket h of the coarse grid and of every fine level, and its sorted ranges."""
    grid_reset_cell(heads, h)
    comptime for l in range(1, _VCM_FINE_LEVELS + 1):
        heads[unsafe_offset=(1 + l) * _HSIZE + h] = Int32(-1)
    comptime for l in range(1 + _VCM_FINE_LEVELS):
        heads[unsafe_offset=_VCM_RANGES + _VCM_RANGE_HDR + 2 * (l * _HSIZE + h) + 1] = Int32(0)
    if h == 0:
        for i in range(_VCM_RANGE_HDR):
            heads[unsafe_offset=_VCM_RANGES + i] = Int32(0)


@always_inline
def _vcm_sort_bucket(t: Int, merge_next: Pointer[Int32, MutUntrackedOrigin], heads: Pointer[Int32, MutUntrackedOrigin],
                     cap: Int):
    """Bucket t % _HSIZE of level t // _HSIZE: its list copied, in list order, into one range of the level's records."""
    var l = t // _HSIZE
    var h = t % _HSIZE
    if t == 0:
        heads[unsafe_offset=_VCM_RANGES] = Int32(cap)
    var head = Int(heads[unsafe_offset=h if l == 0 else (1 + l) * _HSIZE + h])
    var n = 0
    var k = head
    while k != -1:
        n += 1
        k = Int(merge_next[unsafe_offset=k * _VCM_MN_STRIDE + _VCM_MN_LINK + l])
    if n == 0:
        return
    var start = Int(Atomic.fetch_add(heads.unsafe_offset(_VCM_RANGES + 1 + l), Int32(n)))
    heads[unsafe_offset=_VCM_RANGES + _VCM_RANGE_HDR + 2 * t] = Int32(start)
    heads[unsafe_offset=_VCM_RANGES + _VCM_RANGE_HDR + 2 * t + 1] = Int32(n)
    var out = merge_next.unsafe_offset(cap * _VCM_MN_STRIDE + (l * cap + start) * _VCM_SORTED)
    k = head
    var i = 0
    while k != -1:
        var rec = merge_next.unsafe_offset(k * _VCM_MN_STRIDE)
        out[unsafe_offset=i * _VCM_SORTED] = rec[unsafe_offset=0]
        out[unsafe_offset=i * _VCM_SORTED + 1] = rec[unsafe_offset=1]
        out[unsafe_offset=i * _VCM_SORTED + 2] = rec[unsafe_offset=2]
        out[unsafe_offset=i * _VCM_SORTED + 3] = Int32(k)
        k = Int(rec[unsafe_offset=_VCM_MN_LINK + l])
        i += 1


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
    buckets, then insert every LVC vertex via the SAME atomic-exchange"""
    def reset_one(i: Int) {imm}:
        _vcm_reset_cell(heads, i)
    parallelize(reset_one, _HSIZE)

    def count_one(k: Int) {imm}:
        _bdpt_count_merge_vertex(k, lvc, lvc_path_len, heads, inv_cell)
    parallelize(count_one, n_light_paths * _BDPT_MAX_VERTS)

    def insert_one(k: Int) {imm}:
        _bdpt_insert_merge_vertex[True](k, lvc, lvc_path_len, merge_next, heads, inv_cell, sd)
    parallelize(insert_one, n_light_paths * _BDPT_MAX_VERTS)

    def sort_one(t: Int) {imm}:
        _vcm_sort_bucket(t, merge_next, heads, n_light_paths * _BDPT_MAX_VERTS)
    parallelize(sort_one, (1 + _VCM_FINE_LEVELS) * _HSIZE)


# Initial merge radius as a fraction of the scene bounding sphere: SmallVCM's
# own default (config.hxx, mRadiusFactor = 0.003). It was 0.03 here -- 100x the
comptime _VCM_RADIUS_FRACTION = Float32(0.003)


comptime _VCM_RADIUS_ALPHA = Float32(2.0) / Float32(3.0)  # Georgiev 2012's typical choice


def _camera_typical_distance(
    rasterToCamera: Pointer[Float32, MutUntrackedOrigin],
    cameraToWorld: Pointer[Float32, MutUntrackedOrigin],
    film_w: Int, film_h: Int,
    ref sd: SceneView,
    percentile: Float32 = Float32(0.5),
) -> Float32:
    """--vcm-radius-from-camera (EXPERIMENTAL, opt-in only): median
    primary-ray hit distance across a coarse 16x16 grid of the film, as an"""
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
                traverse_dispatch(sd.bvh4, sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), isects.unsafe_offset(0),
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
    """The progressive VCM merge radius for sample `si` (0-based)."""
    return (scene_radius * _VCM_RADIUS_FRACTION
            / pow(Float32(si + 1), Float32(0.5) * (Float32(1) - _VCM_RADIUS_ALPHA)))
