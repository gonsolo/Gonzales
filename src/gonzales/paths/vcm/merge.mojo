# VCM vertex merging against the light-path vertex cache.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.collections import Array
from std.math import sqrt, floor, max, abs
from std.atomic import Atomic
from gonzales.core.geometry import dot, PI, Vec3f
from gonzales.scene.appearance.materials import LobeKind
from gonzales.scene.accel.bvh import SceneView, _is_real_ptr, curve_radius
from gonzales.estimators.photon_grid import disk_measure, ribbon_measure, _HSIZE, _hash_cell, NEIGHBOUR_CELLS, neighbour_buckets, _PHOTON_BUCKET_CAP, grid_keep, grid_weight, gather_disk_coverage, gather_disk_contains, gather_ball_reach
from gonzales.transport.medium import MediumView
from gonzales.scene.appearance.media import hg_phase
from gonzales.core.spectrum import SpectralSample
from gonzales.paths.vcm.vertex import _BDPT_MAX_VERTS, BDPTVertex
from gonzales.paths.vcm.eval import _lobe_eval, _bdpt_vertex_mis_scoped
from gonzales.paths.vcm.grid import _VCM_FINE_LEVELS, _VCM_MN_STRIDE, _VCM_RANGES, _VCM_RANGE_HDR, _VCM_SORTED
from gonzales.paths.vcm.strategy import VcmStrategies, _vcm_keep, _vcm_cap, _MERGE_COVERAGE, _VCM_VISIT_INSTRUMENT, _vcm_budget_active, _vcm_merge_radius_at, _VCM_RADIUS_VOLUME_SCALE, _vcm_volume_radius_at, _vcm_on_fibre
from gonzales.paths.vcm.merge_weight import _bdpt_merge_mis_weight


@fieldwise_init
struct MergeKernel(TrivialRegisterPassable):
    """The density estimate at one camera vertex: a disk on a surface, a ball in a medium, a ribbon on a fibre."""
    var r2: Float32
    var norm: Float32   # 1 / (N * kernel measure)

    @staticmethod
    @always_inline
    def at(ref sd: SceneView, cv: BDPTVertex, r2_pass: Float32, norm_pass: Float32) -> MergeKernel:
        """`norm_pass` is always the disk constant 1/(N*pi*r_pass^2); the grid cells are sized to that radius."""
        var r2 = r2_pass
        var norm = norm_pass
        if cv.is_surface == Int32(0):
            var r_pass_len = sqrt(max(r2_pass, Float32(1e-20)))
            var s3 = _VCM_RADIUS_VOLUME_SCALE * _VCM_RADIUS_VOLUME_SCALE * _VCM_RADIUS_VOLUME_SCALE
            var r_pass_vol = r_pass_len * _VCM_RADIUS_VOLUME_SCALE
            norm = norm_pass / (Float32(4.0 / 3.0) * r_pass_len * s3)
            r2 = r_pass_vol * r_pass_vol
            if sd.vcmMergeR > Float32(0):
                var rq = _vcm_volume_radius_at(sd, cv)
                r2 = rq * rq
                norm = norm * (r_pass_vol * r_pass_vol * r_pass_vol) / max(rq * rq * rq, Float32(1e-20))
            # The ball stays inside the medium: past its boundary there are no photons (gather_ball_reach).
            var r_vol = sqrt(r2)
            var reach = max(gather_ball_reach(sd, cv.pos, r_vol), Float32(0.05) * r_vol)
            if reach < r_vol:
                var q = r_vol / reach
                norm = norm * (q * q * q)
                r2 = reach * reach
        elif sd.vcmFootprint > Float32(0) and sd.vcmMergeR > Float32(0):
            var rq = _vcm_merge_radius_at(sd, cv.pos)
            r2 = rq * rq
            norm = norm_pass * (r2_pass / max(r2, Float32(1e-20)))
        if _vcm_on_fibre(cv):
            # N * 2r * w, the strand within r as a ribbon of its own width.
            var w = Float32(2) * curve_radius(sd.curves, Int(cv.hair_curve_idx), cv.hair_v)
            norm = norm_pass * disk_measure(r2_pass) / max(ribbon_measure(r2, w), Float32(1e-20))
        return MergeKernel(r2, norm)

    @always_inline
    def accepts(self, cv: BDPTVertex, lv: BDPTVertex, e: Vec3f, dist2: Float32) -> Bool:
        if _vcm_on_fibre(cv):   # photons on the same strand only
            return lv.hair_curve_idx == cv.hair_curve_idx and dist2 <= self.r2
        if cv.is_surface == Int32(0):
            return dist2 <= self.r2 and lv.med_idx == cv.med_idx   # a photon across a boundary is another medium's
        # A photon on another surface within reach must not count against this disk's area.
        return dot(cv.normal.to_simd(), lv.normal.to_simd()) > Float32(0.7) and gather_disk_contains(e.to_simd(), dist2, self.r2, cv.normal.to_simd())


@always_inline
def _bdpt_merge_from_cache(
    cv: BDPTVertex,
    ref sd: SceneView,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    merge_next: Pointer[Int32, MutUntrackedOrigin],
    heads: Pointer[Int32, MutUntrackedOrigin],
    inv_cell: Float32,
    r2_pass: Float32,
    norm_pass: Float32,
    mis_vc_weight_factor: Float32,
    cam_count: Int,   # cv's interior-vertex count, delta bounces included (see VcmStrategies.depth)
    cam_nd: Int,      # of them non-delta (see VcmStrategies.past_cache)
    # Benchmark instrumentation only (not part of the real
    # estimator, never read by anything that affects `result`): three
    *,
    mut visit_naive: Int32,
    mut visit_footprint: Int32,
    mut visit_thin: Int32,
) -> SpectralSample:
    """Vertex MERGING (photon-mapping-style density estimation) against the
    shared Light Vertex Cache -- the "M" in VCM, run UNCONDITIONALLY"""
    if cv.is_delta != Int32(0) or VcmStrategies.beyond_cache(cam_nd):   # past the cache every merge is too long
        return SpectralSample(Float32(0))
    # The merge GRID is what this needs, so test the grid -- not, as the call
    # sites used to, whether THIS pixel's own paired light path happened to
    if not (_is_real_ptr(heads) and _is_real_ptr(merge_next) and _is_real_ptr(lvc)):
        return SpectralSample(Float32(0))
    # A vertex kind with no real pdf has no real MIS weight either -- the
    # weight below falls back to 1, and an unweighted merge summed with an
    if not _bdpt_vertex_mis_scoped(cv) or not VcmStrategies.can_merge_at(sd, cv):
        return SpectralSample(Float32(0))
    var total = SpectralSample(Float32(0))
    var vol_cv = cv.is_surface == Int32(0) and Int(cv.med_idx) >= 0
    var kernel = MergeKernel.at(sd, cv, r2_pass, norm_pass)
    var r2 = kernel.r2
    var inv_eta_cv = VcmStrategies.inv_merge_density(sd, cv, mis_vc_weight_factor)
    var cix = Int(floor(cv.pos.x * inv_cell))
    var ciy = Int(floor(cv.pos.y * inv_cell))
    var ciz = Int(floor(cv.pos.z * inv_cell))
    var budget = _vcm_budget_active(sd)
    # The finest level whose cell still covers this query's radius (level 0 --
    # the coarse grid -- when it is wider than a coarse cell, as before).
    var lvl = 0
    var ic = inv_cell
    comptime for l in range(1, _VCM_FINE_LEVELS + 1):
        var icl = inv_cell * Float32(1 << l)
        if lvl == l - 1 and r2 * icl * icl <= Float32(1):
            lvl = l
            ic = icl
    var cap = Int(heads[unsafe_offset=_VCM_RANGES])
    var lrec = merge_next.unsafe_offset(cap * _VCM_MN_STRIDE + lvl * cap * _VCM_SORTED)
    var lrange = heads.unsafe_offset(_VCM_RANGES + _VCM_RANGE_HDR + 2 * lvl * _HSIZE)
    var lcix = Int(floor(cv.pos.x * ic))
    var lciy = Int(floor(cv.pos.y * ic))
    var lciz = Int(floor(cv.pos.z * ic))
    var buckets = neighbour_buckets(lcix, lciy, lciz)
    for b in range(NEIGHBOUR_CELLS):
        var hl = buckets[b]
        if hl < 0:
            break
        var start = Int(lrange[unsafe_offset=2 * hl])
        for i in range(start, start + Int(lrange[unsafe_offset=2 * hl + 1])):
            # Every acceptance needs dist2 <= r2: test it on the position alone before reading the rest of the vertex.
            var rec = lrec.unsafe_offset(i * _VCM_SORTED)
            comptime if not _VCM_VISIT_INSTRUMENT:
                var rp = rec.unsafe_bitcast[Float32]()
                var dx = rp[unsafe_offset=0] - cv.pos.x
                var dy = rp[unsafe_offset=1] - cv.pos.y
                var dz = rp[unsafe_offset=2] - cv.pos.z
                if dx * dx + dy * dy + dz * dz > r2:
                    continue
            var k = Int(rec[unsafe_offset=3])
            # The merged path shares cv and the photon, so its interior
            # length is light count + camera count - 1 (see VcmStrategies.depth).
            if VcmStrategies.too_long(sd, lvc, k // _BDPT_MAX_VERTS, k % _BDPT_MAX_VERTS, cam_count - 1, cam_nd - 1):
                continue
            var lv = lvc[unsafe_offset=k]
            # is_light==1 vertices are the light SOURCE's own point
            # (the s=1 connection strategy): their beta is 1/pdf_area
            if VcmStrategies.can_merge(cv, lv):
                var e = lv.pos - cv.pos
                var dist2 = e.length_sq()
                comptime if _VCM_VISIT_INSTRUMENT:
                    if cv.is_surface == Int32(1) and dot(cv.normal.to_simd(), lv.normal.to_simd()) > Float32(0.7):
                        if gather_disk_contains(e.to_simd(), dist2, r2_pass, cv.normal.to_simd()):
                            visit_naive += Int32(1)
                        if gather_disk_contains(e.to_simd(), dist2, r2, cv.normal.to_simd()):
                            visit_footprint += Int32(1)
                            if grid_keep(heads, _hash_cell(Int(floor(lv.pos.x * inv_cell)), Int(floor(lv.pos.y * inv_cell)), Int(floor(lv.pos.z * inv_cell))), k, lv.pos, _PHOTON_BUCKET_CAP):
                                visit_thin += Int32(1)
                if kernel.accepts(cv, lv, e, dist2):
                    var f_cv: SpectralSample
                    if vol_cv:   # the phase alone; photon_scale carries albedo / sigma_t, after the sum
                        f_cv = SpectralSample(hg_phase(dot(cv.wo.to_simd(), lv.wo.to_simd()), cv.pdf_bwd))
                    else:
                        var le_cv = _lobe_eval[want_pdfs=False](cv, lv.wo.to_simd(), sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, cv.wavelengths)
                        f_cv = le_cv.f_cos
                        # MERGING TAKES THE BARE BSDF, NOT f*cos.
                        # _eval_vertex_spectral returns f*|cos| because
                        var cos_div = le_cv.cos_used
                        if cv.is_surface == Int32(1) and cv.mat_kind != LobeKind.hair:
                            cos_div = abs(dot(lv.wo.to_simd(), cv.normal.to_simd()))
                        if cos_div > Float32(1e-6):
                            f_cv = f_cv * (Float32(1.0) / cos_div)
                        else:
                            f_cv = SpectralSample(Float32(0.0))
                    var w = _bdpt_merge_mis_weight(cv, lv, sd, inv_eta_cv)
                    # 1 / keep: the bucket's under the fixed cap; under
                    # the per-cell budget the vertex's own (_vcm_keep,
                    # the probability its insert survived).
                    var keep_w = grid_weight(heads, _hash_cell(Int(floor(lv.pos.x * inv_cell)), Int(floor(lv.pos.y * inv_cell)), Int(floor(lv.pos.z * inv_cell))), _vcm_cap(sd))
                    if budget:
                        keep_w = Float32(1) / _vcm_keep(sd, lv.pos)
                    total += f_cv * lv.beta * (w * keep_w)
    var result = total * cv.beta * kernel.norm
    if vol_cv:
        result *= MediumView.at(sd, cv.med_idx).photon_scale(cv.pos.to_simd(), cv.wavelengths)
    # Truncation: the photons came only from the part of the disk that is
    # surface (gather_disk_coverage). Queries that found nothing have nothing
    # to correct, and a disk the probes find empty keeps the plain estimate.
    comptime if _MERGE_COVERAGE:
        if cv.is_surface == Int32(1) and cv.mat_kind != LobeKind.hair and total.v0 + total.v1 + total.v2 + total.v3 > Float32(0):
            var cov = gather_disk_coverage(sd, cv.pos, cv.normal, sqrt(r2))
            if cov > Float32(0):
                result = result * (Float32(1) / cov)

    if _is_real_ptr(sd.vcmStatOut):
        # The per-cell budget's statistics for the NEXT pass: one merge query
        # in this cell, and its contribution's second moment, scaled by this
        # cell's keep to undo the thinning's 1/k (V is wanted at full keep).
        var h0 = _hash_cell(cix, ciy, ciz)
        var lum = (result.v0 + result.v1 + result.v2 + result.v3) * Float32(0.25)
        _ = Atomic[Float32].fetch_add(sd.vcmStatOut.unsafe_offset(2 * h0), Float32(1))
        _ = Atomic[Float32].fetch_add(sd.vcmStatOut.unsafe_offset(2 * h0 + 1), lum * lum * _vcm_keep(sd, cv.pos))
    return result
