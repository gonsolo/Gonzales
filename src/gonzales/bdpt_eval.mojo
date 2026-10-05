# BDPT/VCM per-vertex lobe evaluation, MIS scoping and the light-slot policy.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.math import max, abs
from .geometry import Point3f, Point2f, Vec3f, dot, PI
from .materials import LobeKind
from .primitives import Intersection
from .bssrdf import bssrdf_exit_ft
from .vcm_mis import MisPolicy
from .bvh import (
    SceneView, _scene_bounding_sphere, LightSample, _sample_infinite_light_nee, light_path_count,
    light_path_pick_pdf,
)
from .rng import PCG32
from .bxdf import LobeCtx, LobeEval, lobe_eval, lobe_scoped, nee_weight_lobe, LobeTables
from .spectrum import SampledWavelengths, SpectralSample
from .bdpt_vertex import BDPTVertex
from .bdpt_nee import _bdpt_simple_light_count, _bdpt_sample_simple_light, _bdpt_nee_contribute

@always_inline
def _vcm_sphere_slot(ref sd: SceneView, si: Int) -> Int:
    """Pick slot of the emitting sphere sd.spheres[si]: after the area
    lights, spheres counted in index order, emitters only."""
    var k = 0
    for i in range(si):
        if sd.spheres[unsafe_offset=i].isAreaLight != Int8(0):
            k += 1
    return Int(sd.areaLightCount) + k


@always_inline
def _vcm_distant_slot(ref sd: SceneView, j: Int) -> Int:
    return Int(sd.areaLightCount) + Int(sd.sphereLightCount) + j


@always_inline
def _vcm_infinite_slot(ref sd: SceneView, j: Int) -> Int:
    return Int(sd.areaLightCount) + Int(sd.sphereLightCount) + Int(sd.distantLightCount) + j


@always_inline
def _vcm_point_slot(ref sd: SceneView, j: Int) -> Int:
    return (Int(sd.areaLightCount) + Int(sd.sphereLightCount) + Int(sd.distantLightCount)
            + Int(sd.infiniteLightCount) + j)


@always_inline
def _bdpt_n_lights(ref sd: SceneView) -> Float32:
    """The light-pick denominator the light path used, needed on the camera
    side because its NEE loops every light with no pick: the MIS densities
    must describe the same experiment on both subpaths."""
    return Float32(light_path_count(sd))

def _vcm_simple_light_policy(
    ref sd: SceneView, i: Int, ls: LightSample, v: BDPTVertex, eta_x: Float32,
    dvcm: Float32, dvc: Float32, cos_geo: Float32, wavelengths: SampledWavelengths,
) -> MisPolicy:
    """VCM's weight for NEE from camera vertex `v` to the i-th distant/point/
    sphere light (_bdpt_sample_simple_light's order): the balance over every
    strategy, since each of these lights also starts light paths whose t=1
    and merges reach the same paths. The power heuristic that used to stand
    here for point and sphere lights -- and for every one of them at coat and
    subsurface-exit vertices -- partitions unity with BSDF sampling alone.

    vcm_env_nee_weight wants SmallVCM's emissionPdfW * cosToLight /
    (directPdfW * cosAtLight) with directPdfW = ls.pdf, so the light-side
    factors fold into `emission_pdf_w` (p = the light path's pick
    probability for this light, light_path_pick_pdf; the camera side picks none):
        distant  p / (pi R^2)            the bounding disk, delta direction
        point    p / (4 pi d^2)          SmallVCM's directPdfW is d^2; ours
                                         reports 1 with Li = I / d^2
        sphere   p / (pi 4 pi r^2)       cos0 p / (pi A) over cos0; ls.pdf is
                                         the cone's, as the light path's
                                         first-hit dVCM assumes"""
    var le = _lobe_eval[want_pdfs=True](v, ls.wi, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
    var nd = Int(sd.distantLightCount)
    var np_ = Int(sd.pointLightCount)
    var emission: Float32
    if i < nd:
        var (_c, r) = _scene_bounding_sphere(sd)
        emission = light_path_pick_pdf(sd, _vcm_distant_slot(sd, i)) / max(PI * r * r, Float32(1e-12))
    elif i < nd + np_:
        emission = light_path_pick_pdf(sd, _vcm_point_slot(sd, i - nd)) / max(Float32(4.0) * PI * ls.dist * ls.dist, Float32(1e-12))
    else:
        var sph = sd.spheres[unsafe_offset=i - nd - np_]
        emission = light_path_pick_pdf(sd, _vcm_sphere_slot(sd, i - nd - np_)) / max(PI * Float32(4.0) * PI * sph.radius * sph.radius, Float32(1e-12))
    return MisPolicy(le.scoped, eta_x, dvcm, dvc, emission, le.pdf_rev, False, cos_geo)


@always_inline
def _vcm_nee_surface(
    ref sd: SceneView, v: BDPTVertex, ctx: LobeCtx, pos: Point3f, gn_geo: Vec3f,
    beta: SpectralSample, eta_x: Float32, dvcm: Float32, dvc: Float32,
    cur_med_idx: Int32, scratch: Pointer[Intersection, MutUntrackedOrigin],
    wavelengths: SampledWavelengths, mut pcg: PCG32,
    spawn_eps: Float32, two_sided: Bool, exit_eta: Float32,
) -> SpectralSample:
    """NEE from one camera surface vertex to every distant/point/sphere and infinite
    light, weighted by VCM's balance policy. `ctx` is the lobe, `gn_geo` the geometric
    normal for the MIS density, and `exit_eta` > 0 multiplies a BSSRDF exit's Fresnel
    factor. Replaces three near-identical copies, one of which had drifted (the BSSRDF
    exit's infinite-light loop passed no policy)."""
    var total = SpectralSample(Float32(0))
    var tab = LobeTables(sd.materials, sd.curves, sd.measuredBrdfs)
    for li in range(_bdpt_simple_light_count(sd)):
        var ls = _bdpt_sample_simple_light(sd, li, pos.to_simd(), pcg)
        if not ls.valid:
            continue   # a non-emitting sphere
        var pol = _vcm_simple_light_policy(sd, li, ls, v, eta_x, dvcm, dvc, abs(dot(ls.wi, gn_geo)), wavelengths)
        var w = nee_weight_lobe(ls, ctx, tab, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths, pol)
        if exit_eta > Float32(0):
            w = w * bssrdf_exit_ft(dot(gn_geo, ls.wi), exit_eta)
        total += _bdpt_nee_contribute(beta, w, ls, pos, gn_geo, cur_med_idx, sd, scratch, wavelengths, spawn_eps, two_sided)
    for inf_i in range(Int(sd.infiniteLightCount)):
        var ls = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_i], Point2f(pcg.next_float(), pcg.next_float()))
        var (_c, r) = _scene_bounding_sphere(sd)
        var le = _lobe_eval[want_pdfs=True](v, ls.wi, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
        # `scoped` decides: a kind with real densities gets the balance weight over all four strategies.
        var pol = MisPolicy(le.scoped, eta_x, dvcm, dvc,
                            ls.pdf * light_path_pick_pdf(sd, _vcm_infinite_slot(sd, inf_i)) / max(PI * r * r, Float32(1e-12)),
                            le.pdf_rev, False, abs(dot(ls.wi, gn_geo)))
        var w = nee_weight_lobe(ls, ctx, tab, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths, pol)
        if exit_eta > Float32(0):
            w = w * bssrdf_exit_ft(dot(gn_geo, ls.wi), exit_eta)
        total += _bdpt_nee_contribute(beta, w, ls, pos, gn_geo, cur_med_idx, sd, scratch, wavelengths, spawn_eps, two_sided)
    return total

@always_inline
def _vertex_ctx(v: BDPTVertex, adjoint: Bool = False) -> LobeCtx:
    """A stored VCM vertex, as the shared BxDF interface sees it. `adjoint`
    for a light-subpath vertex -- see LobeCtx.adjoint."""
    return LobeCtx(v.mat_kind, v.is_surface == Int32(1), v.is_delta != Int32(0),
                   v.shading_normal.to_simd(), v.wo.to_simd(), v.alb, v.mat_idx,
                   v.pdf_bwd, v.pdf_fwd, v.hair_curve_idx, v.hair_h, v.hair_v, False,
                   adjoint)


@always_inline
def _lobe_eval[want_pdfs: Bool = True](
    v:   BDPTVertex,
    dir_to_other:  Vec3f,
    ref sd:  SceneView,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    wavelengths: SampledWavelengths,
    adjoint: Bool = False,
) -> LobeEval:
    """VCM's view of the shared lobe evaluator (bxdf.mojo)."""
    var le = lobe_eval[want_pdfs](_vertex_ctx(v, adjoint), dir_to_other,
        LobeTables(sd.materials, sd.curves, sd.measuredBrdfs),
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, wavelengths)
    if adjoint and v.is_surface == Int32(1) and v.mat_kind != LobeKind.hair:
        # Veach's adjoint shading-normal correction (pbrt-v3 Vertex::f in
        # Importance mode, CorrectShadingNormal): at a light-subpath vertex
        # the photon ARRIVED with a density in the GEOMETRIC cosine of wo,
        # and the edge it leaves on is a density in the GEOMETRIC cosine of
        # wi, while the BSDF value here carries shading cosines. Without it a
        # bump map made light tracing disagree with NEE and merging: the
        # grazing-bump reproducer drifted with --vcm-photons as the weights
        # shifted between them. Flat surfaces (ns == ng) are unaffected.
        var ns = v.shading_normal.to_simd()
        var ng = v.normal.to_simd()
        var wo = v.wo.to_simd()
        var den = abs(dot(wo, ng)) * abs(dot(dir_to_other, ns))
        if den > Float32(1e-8):
            le.f_cos = le.f_cos * (abs(dot(wo, ns)) * abs(dot(dir_to_other, ng)) / den)
        else:
            le.f_cos = SpectralSample(Float32(0))
    return le

@always_inline
def _eval_vertex_spectral(
    v:   BDPTVertex,
    dir_to_other:  Vec3f,
    ref sd:  SceneView,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    wavelengths: SampledWavelengths,
    adjoint: Bool = False,
) -> SpectralSample:
    """Throughput at `v` toward `dir_to_other`: the BSDF (or phase function)
    times this lobe's OWN cosine. A thin view onto _lobe_eval -- see
    LobeEval for why the dispatch it used to duplicate now lives in one
    place. Pass `adjoint=True` for a light-subpath vertex."""
    return _lobe_eval[want_pdfs=False](
        v, dir_to_other, sd, spectral_coeffs, spectral_res, spectral_cie_x,
        spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths, adjoint).f_cos

@always_inline
def _bdpt_vertex_pdfs(
    v: BDPTVertex, dir_to_other: Vec3f, ref sd: SceneView,
) -> Tuple[Float32, Float32]:
    """Forward/reverse solid-angle densities at `v` toward `dir_to_other`.

    A thin view onto _lobe_eval, which is now the single dispatch over
    LobeKind. This was one of three hand-maintained copies of that dispatch
    -- see LobeEval. Wavelengths come off the vertex because the density half
    never uses them; only the throughput half does, and that is what this
    call discards."""
    var le = _lobe_eval[want_pdfs=True](
        v, dir_to_other, sd, sd.spectral.coeffs, sd.spectral.res,
        sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z,
        sd.spectral.d65, v.wavelengths)
    return (le.pdf_fwd, le.pdf_rev)

@always_inline
def _lobe_scoped(v: BDPTVertex) -> Bool:
    """VCM's view of the shared scope test (bxdf.mojo)."""
    return lobe_scoped(_vertex_ctx(v))

@always_inline
def _bdpt_vertex_mis_scoped(v: BDPTVertex) -> Bool:
    """Alias kept for call sites; see _lobe_scoped."""
    return _lobe_scoped(v)


@always_inline
def _bdpt_connect_pair_weighted(cv: BDPTVertex, lv: BDPTVertex) -> Bool:
    """True when _connect applies a real per-pair MIS weight to (cv, lv).

    MUST stay identical to the condition guarding _connect's own dVCM/dVC
    weight block -- the caller uses this to decide which connections may be
    summed freely (weighted ones) and which must be limited to one per camera
    vertex (unweighted ones), so a mismatch here silently reintroduces the
    over-count this predicate exists to prevent."""
    return _bdpt_vertex_mis_scoped(cv) and (lv.is_light == Int32(1) or _bdpt_vertex_mis_scoped(lv))

# ── Connect one camera vertex to one light vertex ─────────────────────────────
