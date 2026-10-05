# VCM merge MIS weight.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.collections import Array
from std.math import log
from .vcm_camis import CamisCamRecord, CamisCamCarry, CamisLightRecord, CAMIS_IN_CLASS, camis_eval_merge
from .bvh import SceneView
from .bdpt_vertex import _BDPT_MAX_VERTS, BDPTVertex
from .bdpt_eval import _bdpt_vertex_pdfs, _bdpt_vertex_mis_scoped
from .vcm_grid import _vcm_keep, _VCM_CAMIS, _CAMIS_CAM_RECS, _CAMIS_FORCE_C1, _vcm_eta_scale, _vcm_inv_eta_at

@always_inline
def _camis_gather_light[N: Int](
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin],
    base: Int, lam: Int,
) -> Tuple[Array[CamisLightRecord, N], Array[Float32, N], Int, Float32]:
    """The light-side records a CAMIS evaluation at light-path-local index
    `lam` needs, walked from the stored LVC slice starting at `base`:"""
    var scat = Array[CamisLightRecord, N](fill=CamisLightRecord(
        Float32(0), Float32(0), Float32(0), Float32(0), Float32(0), Int32(0), Int32(0)))
    var log_pa_fwd = Array[Float32, N](fill=Float32(0))
    var num_scat = lam - 1
    for i in range(num_scat):
        scat[i] = lvc_camis[unsafe_offset=base + i + 1]
        log_pa_fwd[i] = -log(lvc[unsafe_offset=base + i + 1].dVCM)
    var log_pa0 = lvc_camis[unsafe_offset=base].log_pa_rev
    return (scat^, log_pa_fwd^, num_scat, log_pa0)


@always_inline
def _bdpt_merge_mis_weight(
    cv: BDPTVertex,
    lv: BDPTVertex,
    ref sd: SceneView,
    mis_vc_weight_factor: Float32,
    camis: CamisCamCarry,
    camis_recs: Array[CamisCamRecord, _CAMIS_CAM_RECS],
    num_cam_scat: Int,
    lvc: Pointer[BDPTVertex, MutUntrackedOrigin],
    lvc_camis: Pointer[CamisLightRecord, MutUntrackedOrigin],
    k: Int,
    n_light_paths_f: Float32,
) -> Float32:
    """Balance-heuristic merge weight (Georgiev et al. 2012 / SmallVCM's
    RangeQuery::Process), factored out of _bdpt_merge_from_cache's hot loop"""
    if not (_bdpt_vertex_mis_scoped(cv) and _bdpt_vertex_mis_scoped(lv)):
        return Float32(1)
    var (camera_bsdf_dir_pdf_w, camera_bsdf_rev_pdf_w) = _bdpt_vertex_pdfs(cv, lv.wo.to_simd(), sd)
    # dVM is dVC / eta at the merge vertex. With one global eta that held by
    # construction (the carried dVM); with the variance-aware eta(x) it has
    var inv_eta_x = _vcm_inv_eta_at(sd, cv, mis_vc_weight_factor)
    comptime if _VCM_CAMIS:
        var light_arr = lvc_camis[unsafe_offset=k]
        if camis.in_class and (light_arr.flags & CAMIS_IN_CLASS) != Int32(0):
            var lp_idx = k // _BDPT_MAX_VERTS
            var base = lp_idx * _BDPT_MAX_VERTS
            var lam = k % _BDPT_MAX_VERTS
            var origin_in_class = (lvc_camis[unsafe_offset=base].flags & CAMIS_IN_CLASS) != Int32(0)
            var lr = _camis_gather_light[_CAMIS_CAM_RECS](lvc, lvc_camis, base, lam)
            return camis_eval_merge[_CAMIS_CAM_RECS, _CAMIS_CAM_RECS](
                camis, camis_recs, num_cam_scat,
                cv.dVCM, cv.dVC, cv.dVM, Float32(1) / inv_eta_x, log(_vcm_keep(sd, cv.pos)),
                light_arr, lr[0], lr[2],
                lr[1], -log(lv.dVCM),
                lr[3], origin_in_class,
                lv.dVCM, lv.dVC, lv.dVM, _vcm_eta_scale(sd, lv.pos) / mis_vc_weight_factor,
                camera_bsdf_dir_pdf_w, camera_bsdf_rev_pdf_w,
                n_light_paths_f, True, _CAMIS_FORCE_C1)
    var w_light = (lv.dVCM + lv.dVC * camera_bsdf_dir_pdf_w) * inv_eta_x
    var w_camera = (cv.dVCM + cv.dVC * camera_bsdf_rev_pdf_w) * inv_eta_x
    return Float32(1) / (w_light + Float32(1) + w_camera)
