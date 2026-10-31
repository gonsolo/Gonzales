# VCM merge MIS weight.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.collections import Array
from std.math import log
from gonzales.scene.accel.bvh import SceneView
from gonzales.paths.bdpt_vertex import _BDPT_MAX_VERTS, BDPTVertex
from gonzales.paths.bdpt_eval import _bdpt_vertex_pdfs, _bdpt_vertex_mis_scoped
from gonzales.paths.vcm_strategy import _vcm_keep, _vcm_eta_scale

@always_inline
def _bdpt_merge_mis_weight(
    cv: BDPTVertex,
    lv: BDPTVertex,
    ref sd: SceneView,
    inv_eta_x: Float32,   # 1/eta at cv (_vcm_inv_eta_at), which a volume vertex pays a density lookup for
) -> Float32:
    """Balance-heuristic merge weight (Georgiev et al. 2012 / SmallVCM's
    RangeQuery::Process), factored out of _bdpt_merge_from_cache's hot loop"""
    if not (_bdpt_vertex_mis_scoped(cv) and _bdpt_vertex_mis_scoped(lv)):
        return Float32(1)
    var (camera_bsdf_dir_pdf_w, camera_bsdf_rev_pdf_w) = _bdpt_vertex_pdfs(cv, lv.wo.to_simd(), sd)
    # dVM is dVC / eta at the merge vertex. With one global eta that held by
    # construction (the carried dVM); with the variance-aware eta(x) it has
    var w_light = (lv.dVCM + lv.dVC * camera_bsdf_dir_pdf_w) * inv_eta_x
    var w_camera = (cv.dVCM + cv.dVC * camera_bsdf_rev_pdf_w) * inv_eta_x
    return Float32(1) / (w_light + Float32(1) + w_camera)
