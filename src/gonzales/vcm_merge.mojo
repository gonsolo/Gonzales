# VCM vertex merging against the light-path vertex cache.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.collections import Array
from std.math import sqrt, floor, max, abs
from std.atomic import Atomic
from .geometry import dot
from .materials import LobeKind
from .vcm_camis import CamisCamRecord, CamisCamCarry, CamisLightRecord
from .bvh import SceneView, _is_real_ptr
from .sppm import (
    _HSIZE, _hash_cell, _PHOTON_BUCKET_CAP, grid_keep, grid_weight, gather_disk_coverage,
    gather_disk_contains,
)
from .spectrum import SpectralSample
from .bdpt_vertex import _BDPT_MAX_VERTS, BDPTVertex
from .bdpt_eval import _lobe_eval, _bdpt_vertex_mis_scoped
from .vcm_grid import (
    _vcm_depth, _vcm_light_count, _vcm_keep, _vcm_cap, _MERGE_COVERAGE, _VCM_FINE_LEVELS, _VCM_MN_STRIDE,
    _VCM_VISIT_INSTRUMENT, _CAMIS_CAM_RECS, _vcm_budget_active, _vcm_merge_radius_at,
    _VCM_RADIUS_VOLUME_SCALE,
)
from .vcm_merge_weight import _bdpt_merge_mis_weight

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
    cam_count: Int,   # cv's non-delta interior-vertex count (see _vcm_depth)
    # _VCM_CAMIS: cv's own running camera-side CAMIS state (untouched when
    # the hybrid is compiled out) and the light subpaths' records, parallel
    # to `lvc` -- see _bdpt_merge_mis_weight. Default is in_class=False (NOT
    # camis_cam_carry_init(), whose default is True), so a forgotten call
    # site falls back to the exact legacy weight instead of computing CAMIS
    # math from garbage (log_k=0 etc.) that happens to look in-Class.
    camis: CamisCamCarry = CamisCamCarry(False, Int32(0), Float32(0), Float32(0), Float32(0), Float32(0), Float32(0)),
    camis_recs: Array[CamisCamRecord, _CAMIS_CAM_RECS] = Array[CamisCamRecord, _CAMIS_CAM_RECS](
        fill=CamisCamRecord(Float32(0), Float32(0), Float32(0), Float32(0), Float32(0))),
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin] = Pointer[CamisLightRecord, MutUntrackedOrigin].unsafe_dangling(),
    n_light_paths_f: Float32 = Float32(0),
    # Benchmark instrumentation only (not part of the real
    # estimator, never read by anything that affects `result`): three
    # per-query candidate-visit counters, incremented once per candidate this
    # cell walk considers that passes the SAME normal/disk-membership gate
    # the real merge uses, at three nested radii -- naive (the pass's plain
    # global r2_pass, no footprint scaling), footprint (this query's actual
    # r2, footprint-scaled when sd.vcmFootprint > 0), and thinning (footprint
    # AND would additionally have survived _bdpt_insert_merge_vertex's own
    # keep decision, reconstructed via grid_keep -- see that function's
    # non-budget branch, mirrored exactly here). A single render with
    # insertion thinning disabled (huge --vcm-cap) makes every candidate
    # naive/footprint could ever want available to walk; grid_keep's inputs
    # (the count pass's per-bucket `n`, and grid_coin_bits's pure hash of
    # (k, pos)) don't depend on whether insertion itself thinned, so this
    # reconstruction is exact, not an approximation.
    *,
    mut visit_naive: Int32,
    mut visit_footprint: Int32,
    mut visit_thin: Int32,
) -> SpectralSample:
    """Vertex MERGING (photon-mapping-style density estimation) against the
    shared Light Vertex Cache -- the "M" in VCM, run UNCONDITIONALLY
    alongside _bdpt_connect_to_cache's vertex CONNECTION for every non-delta
    camera vertex (real VCM does both every time, not a stochastic either/or
    -- see _bdpt_trace_camera_and_connect's call sites).

    For each LVC vertex `lv` within radius sqrt(r2) of `cv`, treats `lv` as
    a stored photon: evaluates cv's own BSDF toward lv's stored `wo`
    (the direction the light path arrived from at lv -- exactly the
    `-photon.dir_in` convention sppm.mojo's _sppm_gather_one already uses,
    since BDPTVertex.wo IS that same "direction back toward the light"
    quantity for a light-subpath vertex), multiplies by lv's beta (already
    a valid, unbiased single-light-path throughput estimate -- same
    quantity _bdpt_connect_to_cache's _connect already uses for the light
    side), and divides by (n_light_paths * pi * r2): the standard photon-
    density-estimation normalization, where n_light_paths independent light
    subpaths are the "N emitted photons" and each contributes AT MOST the
    non-delta vertices it stored (mirroring _bdpt_lvc_connection_scale's
    own 1/n_light_paths derivation for connections -- see that function's
    docstring). No shadow ray, no geometry term: merging assumes cv and lv
    are close enough to be treated as the same point, so lv's own light
    path having reached lv unoccluded already implies the segment is
    clear.

    VCM Stage 2c/2d/153/hair: real per-candidate MIS weight (Georgiev et
    al. 2012 / SmallVCM's RangeQuery::Process, vertexcm.hxx:129-166,
    verified against the reference source) is applied when both cv and lv
    are `_bdpt_vertex_mis_scoped` (diffuse/conductor/coated_conductor/
    hair/measured) -- the same scope _connect uses for its own connection
    weight, for the same reason (see that function's docstring)."""
    if cv.is_delta != Int32(0):
        return SpectralSample(Float32(0))
    # The merge GRID is what this needs, so test the grid -- not, as the call
    # sites used to, whether THIS pixel's own paired light path happened to
    # store a vertex. Those are different questions, and conflating them cost
    # merging most of its energy (see the call sites). A caller that never
    # intends to merge passes dangling sentinels here, the _is_real_ptr
    # convention from geometry.mojo, and the old `path_len > 0` guard was
    # shielding them by accident.
    if not (_is_real_ptr(heads) and _is_real_ptr(merge_next) and _is_real_ptr(lvc)):
        return SpectralSample(Float32(0))
    # A vertex kind with no real pdf has no real MIS weight either -- the
    # weight below falls back to 1, and an unweighted merge summed with an
    # unweighted connect estimates 2I, not I. So such a vertex must not merge
    # AT ALL; connect alone is already its complete, correct estimate. The
    # volume branch of _bdpt_camera_path_bounce has always said exactly this
    # and gated its own call; the other call sites relied on `path_len > 0`
    # to do it by accident, and hoisting that gate (correctly, merging does
    # not depend on the paired light path) exposed them. coateddiffuse is the
    # one that bites: its coat-walk vertex is stored with dVCM = dVC = dVM = 0
    # and pdf_fwd = 1 placeholders, so it is deliberately out of scope --
    # furnace-coateddiffuse.vcm read 1.64 against an analytic 1.0 with it
    # merging unweighted. Giving that vertex REAL carries is its own task
    # (see the elegance backlog's item 9); until then it does not merge.
    if not _bdpt_vertex_mis_scoped(cv):
        return SpectralSample(Float32(0))
    var total = SpectralSample(Float32(0))
    var d_len = _vcm_depth(sd)
    # This query's own radius (see _vcm_merge_radius_at); r2/norm arrive as the
    # pass's global values and the grid cells are sized to that radius.
    var r2 = r2_pass
    var norm = norm_pass
    if cv.is_surface == Int32(0):
        # Ball kernel measure at a volume merge vertex: norm_pass is always
        # the DISK constant 1/(N*pi*r_pass^2) (computed once, unconditionally,
        # by every caller); convert to the ball constant
        # 1/(N*(4/3)*pi*R^3) at R = r_pass * _VCM_RADIUS_VOLUME_SCALE (the
        # volume-specific fraction, not the surface one -- see
        # _vcm_merge_radius_at's docstring), footprint-scaled or not
        # (Scenes/vcm_volume_mis_derivation.py, kernel_measure).
        var r_pass_len = sqrt(max(r2_pass, Float32(1e-20)))
        var s3 = _VCM_RADIUS_VOLUME_SCALE * _VCM_RADIUS_VOLUME_SCALE * _VCM_RADIUS_VOLUME_SCALE
        var r_pass_vol = r_pass_len * _VCM_RADIUS_VOLUME_SCALE
        norm = norm_pass / (Float32(4.0 / 3.0) * r_pass_len * s3)
        r2 = r_pass_vol * r_pass_vol
        if sd.vcmFootprint > Float32(0) and sd.vcmMergeR > Float32(0):
            var rq = _vcm_merge_radius_at(sd, cv.pos, is_volume=True)
            r2 = rq * rq
            norm = norm * (r_pass_vol * r_pass_vol * r_pass_vol) / max(rq * rq * rq, Float32(1e-20))
    elif sd.vcmFootprint > Float32(0) and sd.vcmMergeR > Float32(0):
        var rq = _vcm_merge_radius_at(sd, cv.pos)
        r2 = rq * rq
        norm = norm_pass * (r2_pass / max(r2, Float32(1e-20)))
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
    var lheads = heads.unsafe_offset(0 if lvl == 0 else (1 + lvl) * _HSIZE)
    var lcix = Int(floor(cv.pos.x * ic))
    var lciy = Int(floor(cv.pos.y * ic))
    var lciz = Int(floor(cv.pos.z * ic))
    for ddx in range(-1, 2):
        for ddy in range(-1, 2):
            for ddz in range(-1, 2):
                var hl = _hash_cell(lcix + ddx, lciy + ddy, lciz + ddz)
                var k = Int(lheads[unsafe_offset=hl])
                while k != -1:
                    # The merged path shares cv and the photon, so its interior
                    # length is light count + camera count - 1 (see _vcm_depth).
                    if _vcm_light_count(lvc, k // _BDPT_MAX_VERTS, k % _BDPT_MAX_VERTS) + cam_count - 1 > d_len:
                        k = Int(merge_next[unsafe_offset=k * _VCM_MN_STRIDE + lvl])
                        continue
                    var lv = lvc[unsafe_offset=k]
                    # is_light==1 vertices are the light SOURCE's own point
                    # (the s=1 connection strategy): their beta is 1/pdf_area
                    # ONLY, with the actual emitted radiance held separately
                    # in lv.alb (see _connect's own is_light special case,
                    # which multiplies the two together). Merging with them
                    # using the generic "beta = flux" assumption below would
                    # silently drop that emission factor -- exactly matching
                    # why sppm.mojo's _sppm_trace_photon never stores a
                    # photon at bounce==0 either (its own docstring: "that
                    # direct contribution is now covered by NEE instead").
                    # Every OTHER stored vertex's beta already has emission
                    # folded in via the light path's own flux computation.
                    # BSSRDF exits excluded: a light-side exit vertex has no
                    # incoming ray (it was reached by a hop), so there is no
                    # photon direction to evaluate the camera vertex against.
                    # ... and the LIGHT vertex must be MIS-scoped too, for the
                    # reason this function's own docstring already gives about
                    # the CAMERA vertex: a kind with no real pdf has no real
                    # MIS weight, the weight below falls back to 1, and an
                    # unweighted merge summed with a weighted connect
                    # estimates more than I. That gate was enforced on cv (an
                    # early return) but not on lv, so the asymmetry let an
                    # unscoped light vertex merge at FULL weight. Reachable in
                    # practice: lobe_scoped's list is lambertian/ggx/hair/
                    # measured/coated_walk, so a diffuse_transmit photon is
                    # unscoped -- and barcelona-pavilion's foliage is 5
                    # diffusetransmission materials, sitting exactly over the
                    # shadowed regions that measured 2-3x too bright.
                    if lv.is_delta == Int32(0) and lv.is_surface == cv.is_surface and lv.is_light == Int32(0) and lv.mat_kind != LobeKind.bssrdf and _bdpt_vertex_mis_scoped(lv):
                        var e = lv.pos - cv.pos
                        var dist2 = e.length_sq()
                        var accept: Bool
                        if cv.is_surface == Int32(1):
                            # Surface-compatibility guard: a distance-only gather
                            # counts a photon lying on a DIFFERENT surface (the
                            # adjacent wall, the far side of a thin panel) as if
                            # it were on this one. The leak grows with the gather
                            # radius (_VCM_RADIUS_FRACTION of the scene bounding
                            # sphere).
                            var _ncmp = dot(cv.normal.to_simd(), lv.normal.to_simd())
                            # Gather in the tangent DISK, not the ball. The normal
                            # test above rejects a photon on a differently-oriented
                            # surface; it cannot reject one on a PARALLEL surface
                            # inside the radius -- a desk top over a shelf, a sill
                            # over a floor. Two lit parallel surfaces in one ball
                            # sum both their photons and normalise by ONE disk,
                            # pi r^2: up to 2x, radius-dependent, and impossible on
                            # a single flat quad, which is why the white furnace
                            # stayed exact while classroom read 2.5x pbrt with
                            # merging on and 0.985x with it off. A photon on THIS
                            # surface sits on its tangent plane to float precision;
                            # a tenth of the radius is generous.
                            # Benchmark instrumentation only -- see
                            # this function's own visit_naive/footprint/thin
                            # docstring paragraph. Pure reads, no effect on
                            # `total`/`result` below.
                            comptime if _VCM_VISIT_INSTRUMENT:
                                if _ncmp > Float32(0.7):
                                    if gather_disk_contains(e.to_simd(), dist2, r2_pass, cv.normal.to_simd()):
                                        visit_naive += Int32(1)
                                    if gather_disk_contains(e.to_simd(), dist2, r2, cv.normal.to_simd()):
                                        visit_footprint += Int32(1)
                                        if grid_keep(heads, _hash_cell(Int(floor(lv.pos.x * inv_cell)), Int(floor(lv.pos.y * inv_cell)), Int(floor(lv.pos.z * inv_cell))), k, lv.pos, _PHOTON_BUCKET_CAP):
                                            visit_thin += Int32(1)
                            # Disk-not-ball: the SHARED test, sppm.mojo's
                            # gather_disk_contains -- SPPM's gather now uses it too.
                            accept = _ncmp > Float32(0.7) and gather_disk_contains(e.to_simd(), dist2, r2, cv.normal.to_simd())
                        else:
                            # Volume-volume merge: a genuine BALL acceptance
                            # test. Neither endpoint has a surface normal, so
                            # there is no tangent-plane concept to guard with --
                            # dist2 <= r2 is exactly what
                            # Scenes/vcm_volume_mis_derivation.py's kernel_measure
                            # assumes (no paper-figure visit instrumentation for
                            # this case; that's a surface-scene-only probe).
                            accept = dist2 <= r2
                        if accept:
                            var le_cv = _lobe_eval[want_pdfs=False](cv, lv.wo.to_simd(), sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, cv.wavelengths)
                            var f_cv = le_cv.f_cos
                            # MERGING TAKES THE BARE BSDF, NOT f*cos.
                            # _eval_vertex_spectral returns f*|cos| because
                            # that is what a CONNECTION needs -- the cosine
                            # belongs to its geometry term. A photon-density
                            # estimate must not apply it: the incident cosine
                            # is already carried by the photon AREAL DENSITY,
                            # grazing arrivals being proportionally rarer per
                            # unit area. Applying it again integrates cos^2
                            # where the reflection integral wants cos -- over
                            # a hemisphere exactly (2pi/3)/pi = 2/3, and
                            # merging alone measured 0.685 of the white
                            # furnace's analytic answer. SmallVCM splits it
                            # the same way: RangeQuery::Process uses the bare
                            # bsdfFactor, connections multiply by cosThetaGen.
                            # ... and THIS lobe's own cosine, which is not
                            # always |cos(dir, n)|: hair carries the FIBRE
                            # cosine and a volume carries none. Guessing it
                            # here -- as this did before _lobe_eval existed --
                            # divides hair by the wrong quantity entirely.
                            #
                            # ... and on a surface the density's cosine is the
                            # GEOMETRIC one: photons land per unit of actual
                            # surface area. f_cos carries the SHADING cosine
                            # (that is the reflection integral pbrt's and our
                            # path tracers evaluate), so dividing by the
                            # geometric cosine leaves f*cos_s/cos_g -- the
                            # same ns-for-the-BSDF, ng-for-the-density split
                            # _connect makes. Dividing by cos_s instead made
                            # merging disagree with NEE wherever a bump map
                            # tilts ns: a --vcm-photons sweep drifted 1.17 ->
                            # 1.41 (grazing bump) and 1.07 -> 0.68 (top-down)
                            # while the flat control held 1.008.
                            var cos_div = le_cv.cos_used
                            if cv.is_surface == Int32(1) and cv.mat_kind != LobeKind.hair:
                                cos_div = abs(dot(lv.wo.to_simd(), cv.normal.to_simd()))
                            if cos_div > Float32(1e-6):
                                f_cv = f_cv * (Float32(1.0) / cos_div)
                            else:
                                f_cv = SpectralSample(Float32(0.0))
                            var w = _bdpt_merge_mis_weight(cv, lv, sd, mis_vc_weight_factor,
                                camis, camis_recs, cam_count - 1, lvc, lvc_camis, k, n_light_paths_f)
                            # 1 / keep: the bucket's under the fixed cap; under
                            # the per-cell budget the vertex's own (_vcm_keep,
                            # the probability its insert survived).
                            var keep_w = grid_weight(heads, _hash_cell(Int(floor(lv.pos.x * inv_cell)), Int(floor(lv.pos.y * inv_cell)), Int(floor(lv.pos.z * inv_cell))), _vcm_cap(sd))
                            if budget:
                                keep_w = Float32(1) / _vcm_keep(sd, lv.pos)
                            total += f_cv * lv.beta * (w * keep_w)
                    k = Int(merge_next[unsafe_offset=k * _VCM_MN_STRIDE + lvl])
    var result = total * cv.beta * norm
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

# ── Trace one camera subpath, connecting to the shared cache inline ─────────
