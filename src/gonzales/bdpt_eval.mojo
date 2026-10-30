# BDPT/VCM per-vertex lobe evaluation, MIS scoping and the light-slot policy.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.math import max, abs, sqrt, cos, sin
from std.atomic import Atomic
from .geometry import Point3f, Point2f, Vec3f, RGB, dot, cross, PI, _is_real_ptr
from .materials import LobeKind
from .primitives import Intersection, mesh_vi, mesh_pos
from .lights import area_light_pick_triangle
from .sppm import area_light_side_pdf
from .shading import (
    _shade_context, _mnee_first_probe, _mnee_chain, _mnee_chain_t_eta, _mnee_reproduce, _emitter_face_normal,
)
from .bssrdf import bssrdf_exit_ft
from .vcm_mis import MisPolicy
from .bvh import SceneView, _scene_bounding_sphere, LightSample, _sample_infinite_light_nee, light_path_pick, light_path_pick_pdf
from .rng import PCG32
from .bxdf import BSDF, LobeEval, LobeTables
from .spectrum import SampledWavelengths, SpectralSample, spec_illum
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


def _vcm_simple_light_policy(
    ref sd: SceneView, i: Int, ls: LightSample, v: BDPTVertex, eta_x: Float32,
    dvcm: Float32, dvc: Float32, cos_geo: Float32, wavelengths: SampledWavelengths,
) -> MisPolicy:
    """VCM's weight for NEE from camera vertex `v` to the i-th distant/point/
    sphere light (_bdpt_sample_simple_light's order): the balance over every"""
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
    ref sd: SceneView, v: BDPTVertex, ctx: BSDF, pos: Point3f, gn_geo: Vec3f,
    beta: SpectralSample, eta_x: Float32, dvcm: Float32, dvc: Float32,
    cur_med_idx: Int32, scratch: Pointer[Intersection, MutUntrackedOrigin],
    wavelengths: SampledWavelengths, mut pcg: PCG32,
    spawn_eps: Float32, two_sided: Bool, exit_eta: Float32,
) -> SpectralSample:
    """NEE from one camera surface vertex to every distant/point/sphere and infinite
    light, weighted by VCM's balance policy. `ctx` is the lobe, `gn_geo` the geometric"""
    var total = SpectralSample(Float32(0))
    var tab = LobeTables(sd.materials, sd.curves, sd.measuredBrdfs, sd.spectral)
    for li in range(_bdpt_simple_light_count(sd)):
        var ls = _bdpt_sample_simple_light(sd, li, pos.to_simd(), pcg)
        if not ls.valid:
            continue   # a non-emitting sphere
        var pol = _vcm_simple_light_policy(sd, li, ls, v, eta_x, dvcm, dvc, abs(dot(ls.wi, gn_geo)), wavelengths)
        var w = ctx.nee_weight_mis(ls, tab, wavelengths, pol)
        if exit_eta > Float32(0):
            w = w * bssrdf_exit_ft(dot(gn_geo, ls.wi), exit_eta)
        total += _bdpt_nee_contribute(beta, w, ls, pos, gn_geo, cur_med_idx, sd, scratch, wavelengths, spawn_eps, two_sided)
    for inf_i in range(Int(sd.infiniteLightCount)):
        var ls = _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_i], Point2f(pcg.next_float(), pcg.next_float()), pos)
        var (_c, r) = _scene_bounding_sphere(sd)
        var le = _lobe_eval[want_pdfs=True](v, ls.wi, sd, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wavelengths)
        # `scoped` decides: a kind with real densities gets the balance weight over all four strategies.
        var pol = MisPolicy(le.scoped, eta_x, dvcm, dvc,
                            ls.pdf * light_path_pick_pdf(sd, _vcm_infinite_slot(sd, inf_i)) / max(PI * r * r, Float32(1e-12)),
                            le.pdf_rev, False, abs(dot(ls.wi, gn_geo)))
        var w = ctx.nee_weight_mis(ls, tab, wavelengths, pol)
        if exit_eta > Float32(0):
            w = w * bssrdf_exit_ft(dot(gn_geo, ls.wi), exit_eta)
        total += _bdpt_nee_contribute(beta, w, ls, pos, gn_geo, cur_med_idx, sd, scratch, wavelengths, spawn_eps, two_sided)
    return total

@always_inline
def _vertex_ctx(v: BDPTVertex, adjoint: Bool = False) -> BSDF:
    """A stored VCM vertex, as the shared BxDF interface sees it. `adjoint`
    for a light-subpath vertex -- see BSDF.adjoint."""
    return BSDF(v.mat_kind, v.is_surface == Int32(1), v.is_delta != Int32(0),
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
    var le = (_vertex_ctx(v, adjoint)).f[want_pdfs](dir_to_other, LobeTables(sd.materials, sd.curves, sd.measuredBrdfs, sd.spectral), wavelengths)
    if adjoint and v.is_surface == Int32(1) and v.mat_kind != LobeKind.hair:
        # Veach's adjoint shading-normal correction (pbrt-v3 Vertex::f in
        # Importance mode, CorrectShadingNormal): at a light-subpath vertex
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
    """Forward/reverse solid-angle densities at `v` toward `dir_to_other`."""
    var le = _lobe_eval[want_pdfs=True](
        v, dir_to_other, sd, sd.spectral.coeffs, sd.spectral.res,
        sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z,
        sd.spectral.d65, v.wavelengths)
    return (le.pdf_fwd, le.pdf_rev)

@always_inline
def _lobe_scoped(v: BDPTVertex) -> Bool:
    """VCM's view of the shared scope test (bxdf.mojo)."""
    return _vertex_ctx(v).scoped()

@always_inline
def _bdpt_vertex_mis_scoped(v: BDPTVertex) -> Bool:
    """Alias kept for call sites; see _lobe_scoped."""
    return _lobe_scoped(v)


@always_inline
def _bdpt_connect_pair_weighted(cv: BDPTVertex, lv: BDPTVertex) -> Bool:
    """True when _connect applies a real per-pair MIS weight to (cv, lv)."""
    return _bdpt_vertex_mis_scoped(cv) and (lv.is_light == Int32(1) or _bdpt_vertex_mis_scoped(lv))

# ── MNEE as one more VCM strategy ───────────────────────────────────────────
# Scenes/vcm_mnee_mis_derivation.py derives every term below. MNEE draws its light point exactly like a light path's
# origin (light_path_pick, then uniform on the light), so both strategies share that point's area density p_A.

@always_inline
def _vcm_sphere_of_slot(ref sd: SceneView, slot: Int) -> Int:
    """The sphere index of sphere-light slot `slot` (_vcm_sphere_slot's inverse), -1 if there is none."""
    var k = slot - Int(sd.areaLightCount)
    for i in range(Int(sd.sphereCount)):
        if sd.spheres[unsafe_offset=i].isAreaLight != Int8(0):
            if k == 0:
                return i
            k -= 1
    return -1


@always_inline
def _vcm_mnee_light_point(
    ref sd: SceneView, slot: Int, mut pcg: PCG32,
) -> Tuple[Bool, Vec3f, Vec3f, Vec3f, Vec3f, Float32, Float32, RGB]:
    """A uniform point on the mesh or sphere light in light-path slot `slot`:
    (ok, point, two tangents, unit emitting normal, area, emission-face probability, emission)."""
    var z = Vec3f(Float32(0))
    var none = (False, z, z, z, z, Float32(0), Float32(0), RGB(Float32(0)))
    if slot < Int(sd.areaLightCount):
        var al = sd.areaLights[unsafe_offset=slot]
        if al.kind != Int8(0) or al.total_area <= Float32(0):
            return none
        var lmesh = sd.meshes[unsafe_offset=Int(al.meshIdx)]
        var lb = area_light_pick_triangle(al, pcg.next_float()) * 3
        var lv0 = Int(mesh_vi(lmesh, lb)); var lv1 = Int(mesh_vi(lmesh, lb + 1)); var lv2 = Int(mesh_vi(lmesh, lb + 2))
        var lp0 = mesh_pos(lmesh, lv0); var lp1 = mesh_pos(lmesh, lv1); var lp2 = mesh_pos(lmesh, lv2)
        var sr1 = sqrt(pcg.next_float()); var ru2 = pcg.next_float()
        var bu = sr1 * (Float32(1) - ru2); var bv = sr1 * ru2
        var ln = _emitter_face_normal(lmesh, lv0, lv1, lv2, bu, bv, lp0, lp1, lp2)
        var lnl = sqrt(dot(ln, ln))
        if lnl <= Float32(0):
            return none
        return (True, lp0 * (Float32(1) - bu - bv) + lp1 * bu + lp2 * bv, lp1 - lp0, lp2 - lp0, ln * (Float32(1) / lnl),
                al.total_area, area_light_side_pdf(al), al.emission)
    var si = _vcm_sphere_of_slot(sd, slot)
    if si < 0:
        return none
    var sph = sd.spheres[unsafe_offset=si]
    if sph.radius <= Float32(0):
        return none
    var cos_t = Float32(1) - Float32(2) * pcg.next_float()
    var sin_t = sqrt(max(Float32(0), Float32(1) - cos_t * cos_t))
    var phi = Float32(2) * PI * pcg.next_float()
    var ln = Vec3f(sin_t * cos(phi), sin_t * sin(phi), cos_t)
    return (True, sph.center.to_simd() + ln * sph.radius,
            Vec3f(-sin_t * sin(phi), sin_t * cos(phi), Float32(0)) * sph.radius,
            Vec3f(cos_t * cos(phi), cos_t * sin(phi), -sin_t) * sph.radius,
            ln, Float32(4) * PI * sph.radius * sph.radius, Float32(1), sph.emission)


@no_inline   # one copy of the solver in the camera bounce
def _vcm_mnee(
    ref sd: SceneView, v: BDPTVertex, hit_v: Vec3f, gn: Vec3f, beta: SpectralSample, mut pcg: PCG32,
    wl: SampledWavelengths, eta_x: Float32, dvcm: Float32, dvc: Float32,
) -> SpectralSample:
    """MNEE from camera vertex `v` to a light behind glass, balance-weighted against BSDF sampling through the chain
    and against every light-subpath strategy that reaches v through it. dvcm/dvc are v's arrival carries."""
    var z = SpectralSample(Float32(0))
    if sd.hasGlass == Int32(0):
        return z
    var (slot, p_pick) = light_path_pick(sd, pcg.next_float())
    if p_pick <= Float32(0) or slot < 0 or slot >= Int(sd.areaLightCount) + Int(sd.sphereLightCount):   # -1: no lights
        return z
    var (ok, light_point, l_du, l_dv, ln, area, side, emission) = _vcm_mnee_light_point(sd, slot, pcg)
    if not ok:
        return z
    var p_a = p_pick / area
    var to_light = light_point - hit_v
    var dist = sqrt(dot(to_light, to_light))
    if dist < Float32(1e-4):
        return z
    var shadow_dir = to_light * (Float32(1) / dist)
    var ctx = _shade_context(sd, Pointer[UInt32, MutUntrackedOrigin].unsafe_dangling())
    var (glass, probe) = _mnee_first_probe(ctx, hit_v, shadow_dir, dist)
    if _is_real_ptr(sd.vcmMneeStats):
        _ = Atomic.fetch_add(sd.vcmMneeStats, Int32(1))
        if glass:
            _ = Atomic.fetch_add(sd.vcmMneeStats.unsafe_offset(1), Int32(1))
    if not glass:
        return z
    var c = _mnee_chain(ctx, gn, hit_v, shadow_dir, dist, light_point, l_du, l_dv, pcg, probe)
    if not c[1]:
        return z
    var n = c[2]
    var verts = c[3].copy()
    var wy = verts[n-1].pos - light_point
    var cos_y = dot(ln, wy) / sqrt(max(dot(wy, wy), Float32(1e-30)))
    if side < Float32(1):
        cos_y = abs(cos_y)
    if cos_y <= Float32(0):
        return z
    var w0 = verts[0].pos - hit_v
    var le = _lobe_eval[want_pdfs=True](v, w0 * (Float32(1) / sqrt(dot(w0, w0))), sd, sd.spectral.coeffs, sd.spectral.res,
        sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, wl)
    var (t_ch, eta_ch) = _mnee_chain_t_eta(hit_v, verts, n)
    var g = c[5]
    var w_light = le.pdf_fwd * t_ch * g / p_a
    var w_camera = side / PI / (eta_ch * eta_ch) * c[6] * g * t_ch * (eta_x + dvcm + dvc * le.pdf_rev)
    var w = Float32(1) / (w_light + Float32(1) + w_camera)
    return (beta * le.f_cos
        * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, emission.r, emission.g, emission.b, wl)
        * (g * c[4] / p_a * w))


@no_inline
def _vcm_mnee_hit_term(
    ref sd: SceneView, x0: Vec3f, pdf_x0: Float32, n_glass: Int, last_glass: Vec3f, light_hit: Vec3f,
    l_du: Vec3f, l_dv: Vec3f, p_a: Float32, mut pcg: PCG32,
) -> Float32:
    """MNEE's share in the weight of a camera path that reached light_hit from vertex x0 through n_glass glass
    vertices: p_A / (p_bsdf (1 - F) g) when MNEE's solve reproduces the path, else 0."""
    if sd.hasGlass == Int32(0) or n_glass <= 0 or pdf_x0 <= Float32(0):
        return Float32(0)
    var seg = light_hit - x0
    var seg_len = sqrt(dot(seg, seg))
    if seg_len < Float32(1e-4):
        return Float32(0)
    var ctx = _shade_context(sd, Pointer[UInt32, MutUntrackedOrigin].unsafe_dangling())
    var (glass, probe) = _mnee_first_probe(ctx, x0, seg * (Float32(1) / seg_len), seg_len)
    if not glass:
        return Float32(0)
    var rp = _mnee_reproduce(ctx, x0, n_glass, last_glass, light_hit, l_du, l_dv, probe, pcg)
    if rp[0] != 1 or rp[3] <= Float32(0):
        return Float32(0)
    return p_a / (pdf_x0 * _mnee_chain_t_eta(x0, rp[2], rp[1])[0] * rp[3])


@no_inline
def _vcm_mnee_light_dvcm(
    ref sd: SceneView, x0: Vec3f, n_geo: Vec3f, y: Vec3f, ny: Vec3f, side: Float32, n_glass: Int, first_glass: Vec3f,
    mut pcg: PCG32,
) -> Float32:
    """dVCM on arriving at x0 for a light path that left light point y through n_glass glass vertices, the first at
    first_glass: 1 / (p_emit (1 - F) J_light) when MNEE's solve from x0 reproduces that chain, else 0 (no share).
    J_light = (n_x0 / n_y)^2 cos_x0 g / cos_y by etendue, and p_emit's cos_y / pi cancels its cos_y."""
    if sd.hasGlass == Int32(0) or n_glass <= 0 or side <= Float32(0):
        return Float32(0)
    var seg = y - x0
    var seg_len = sqrt(dot(seg, seg))
    if seg_len < Float32(1e-4):
        return Float32(0)
    var ctx = _shade_context(sd, Pointer[UInt32, MutUntrackedOrigin].unsafe_dangling())
    var (glass, probe) = _mnee_first_probe(ctx, x0, seg * (Float32(1) / seg_len), seg_len)
    if not glass:
        return Float32(0)
    var a = Vec3f(Float32(1), Float32(0), Float32(0)) if abs(ny[0]) < Float32(0.9) else Vec3f(Float32(0), Float32(1), Float32(0))
    var tu = cross(ny, a)
    var rp = _mnee_reproduce(ctx, x0, n_glass, first_glass, y, tu, cross(ny, tu), probe, pcg)
    if rp[0] != 1 or rp[3] <= Float32(0):
        return Float32(0)
    var (t_ch, eta_ch) = _mnee_chain_t_eta(x0, rp[2], rp[1])
    var w0 = rp[2][0].pos - x0
    var cos0 = abs(dot(n_geo, w0)) / sqrt(max(dot(w0, w0), Float32(1e-30)))
    var den = side * cos0 * rp[3] * t_ch
    return PI * eta_ch * eta_ch / den if den > Float32(0) else Float32(0)

