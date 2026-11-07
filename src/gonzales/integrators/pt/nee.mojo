"""The path tracer's next-event estimation: lights, MNEE and SMS."""
from std.atomic import Atomic
from std.collections import Array
from std.math import sqrt
from gonzales.core.geometry import offset_point, RGB, Point3f, Point2f, Vec3f, dot, cross, fast_sincos, TWO_PI, _is_real_ptr
from gonzales.core.geometry import point3f
from gonzales.scene.appearance.materials import sss_boundary_lobe
from gonzales.transport.render_state import PathState
from gonzales.scene.shapes.primitives import Intersection, mesh_vi
from gonzales.core.geometry import Ray
from gonzales.scene.shapes.primitives import mesh_pos
from gonzales.scene.appearance.lights import AreaLight, InfiniteLight, light_sampler_sample, light_sampler_sample_at, light_sampler_pdf_at, area_light_pick_triangle
from gonzales.scene.shapes.curves import curve_at, curve_piece_endpoints, _curve_perp_axis
from gonzales.transport.bxdf import BSDF, NeeLobe, bxdf_pdf_diffuse, _nee_weight_simple_spectral, LobeTables
from gonzales.core.rng import PCG32
from gonzales.scene.accel.bvh import any_hit_dispatch, LightSample, _sample_distant_light_nee, _sample_point_light_nee, _sample_sphere_light_nee, _sample_infinite_light_nee, _sample_infinite_light_textured
from gonzales.core.sampling import power_heuristic, sample_cosine_hemisphere_world
from gonzales.transport.guide import GuideGrid, null_guide
from gonzales.core.reservoir import reservoir_update, reservoir_finalize, reservoir_combine, reservoir_cap_confidence
from gonzales.transport.sms import MAX_SMS_VERTICES, SMSVertex, sms_vertex_init
from gonzales.integrators.pt.restir_sms import SMSReservoir, sms_reservoir_init, sms_target_pdf, SMSReservoirIO, sms_reservoir_io_null, sms_spatial_combine
from gonzales.transport.texturing import _get_tri_verts
from gonzales.transport.shade_context import ShadeContext
from gonzales.transport.manifold import _emitter_face_normal, _mnee_chain, _mnee_bsdf_over_light, _mnee_chain_t_eta, _mnee_first_probe, _mnee_reproduce, _sms_solve_from_probe, _mnee_chain_transmittance, _mnee_pass
from gonzales.core.spectrum import SpectralSample
from gonzales.transport.glossy_sms import GSMS_DEBUG, glossy_sms_stage, glossy_sms_solve, glossy_sms_mis, glossy_sms_reproduce, glossy_sms_t_chain
from gonzales.transport.sms import mnee_orthonormal_basis
from gonzales.integrators.pt.common import _to_spec_illum, _clamped, _to_spec_diffuse_lit, _emitter_invisible, _shadow_inline, _shadow_blocked, _shadow_contribute


@always_inline
def _nee_simple_light_count(ctx: ShadeContext) -> Int:
    """Number of lights reachable through _nee_sample_simple_light: the three
    types every material samples the SAME way. Area and infinite lights are
    deliberately excluded -- each material samples those with a genuinely
    different strategy (MNEE-capable area sampling, env-map CDF vs cosine
    fallback), which is why they stay written out at their call sites."""
    return ctx.lights.distant_count + ctx.lights.point_count + ctx.lights.sphere_count


@always_inline
def _nee_sample_simple_light(
    ctx: ShadeContext, i: Int, hit_point: Vec3f, mut pcg: PCG32,
) -> Tuple[LightSample, Float32]:
    """The i-th distant/point/sphere light, plus the shadow-ray tmax to use for
    it. One place that knows which sampler each type needs and which tmax
    factor it takes -- that pairing used to be re-typed in every sweep, and
    getting it wrong is precisely how sphere lights ended up shadowing
    themselves (bba82627: tmax from the CENTRE distance).

    ORDER IS LOAD-BEARING: distant, then point, then sphere. Of the three only
    SPHERE draws from `pcg`, so any sweep that visited them in some other
    order still consumed randomness in the same sequence, and switching it to
    this one is byte-identical. A sweep that interleaves an INFINITE light
    (which also draws) between them is NOT, and must keep its own order."""
    var nd = ctx.lights.distant_count
    var np_ = ctx.lights.point_count
    if i < nd:
        var ls_d = _sample_distant_light_nee(ctx.lights.distant_lights[unsafe_offset=i])
        var d = ls_d.dist
        return (ls_d^, d)
    if i < nd + np_:
        var ls_p = _sample_point_light_nee(ctx.lights.point_lights[unsafe_offset=i - nd], hit_point)
        var d = ls_p.dist * Float32(0.9999)
        return (ls_p^, d)
    var si = i - nd - np_
    var ls_s = _sample_sphere_light_nee(ctx.lights.spheres[unsafe_offset=si], ctx.lights.sphere_count, hit_point, pcg)
    var d = ls_s.dist * Float32(0.9999)
    return (ls_s^, d)


@always_inline
def _nee_loop_simple[enqueue_shadow: Bool, defer_only: Bool = False](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    normal: Vec3f,
    hit_point: Vec3f,
    alb: RGB,
    alpha: Float32,
    mat_kind: Int32,
    wo: Vec3f,
    mut pcg: PCG32,
    guide_write: GuideGrid = null_guide(),
    lobe_w: Float32 = Float32(1.0),
):
    """Distant + point + sphere-light NEE for a 'simple' BxDF (flat
    mat_kind/alb/alpha, evaluable via bxdf_eval_any/_nee_weight_simple_spectral)
    -- the one light-loop shape shared VERBATIM between diffuse (mat_kind=0)
    and conductor (mat_kind=1), previously hand-copy-pasted once per material
    (3 loops x 2 materials). Compile-time specialized per call site (no
    runtime indirection), so this is safe to use from GPU-compiled code too.

    Area and infinite lights deliberately stay OUTSIDE this loop: each
    material samples them with a genuinely different strategy (diffuse's
    MNEE-capable _nee_area_lights / cosine-hemisphere _nee_infinite_light
    fallback vs conductor's plain uniform-sphere _sample_area_light_nee/
    _sample_infinite_light_nee) — folding those in here would be a behavior
    change, not a refactor. See project_light_bxdf_interfaces memory."""
    for li in range(_nee_simple_light_count(ctx)):
        var res = _nee_sample_simple_light(ctx, li, hit_point, pcg)
        var ls = res[0].copy()
        var tmax = res[1]
        var w = _nee_weight_simple_spectral(ls, mat_kind, alb, alpha, normal, wo, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65, path_ptr[].wavelengths, LobeTables(ctx.materials, ctx.curves, ctx.measured_brdfs, ctx.spectral)) * lobe_w
        if not w.is_black():
            var contrib = path_ptr[].throughput * w
            _shadow_contribute[enqueue_shadow, defer_only](path_ptr, ctx, hit_point, ls.wi, tmax, contrib, guide_write)


def _lobe_nee[enqueue_shadow: Bool, defer_only: Bool = False](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    bsdf: BSDF,
    hit_point: Vec3f,
    mut pcg: PCG32,
):
    """NEE through a BSDF: one area-light sample, every simple light, every environment."""
    var ls_area = _sample_area_light_nee(ctx, hit_point, pcg)
    _lobe_nee_with[BSDF, enqueue_shadow, defer_only](path_ptr, ctx, bsdf, hit_point, ls_area, Point2f(Float32(0), Float32(0)),
                                                     False, pcg)


@always_inline
def _lobe_nee_with[L: NeeLobe, enqueue_shadow: Bool, defer_only: Bool = False](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    bsdf: L,
    hit_point: Vec3f,
    ls_area: LightSample,
    u_env: Point2f,
    has_u_env: Bool,
    mut pcg: PCG32,
    side_n: Vec3f = Vec3f(Float32(0)),
):
    """_lobe_nee with the area-light sample drawn, and the first environment's numbers when `has_u_env` (Sobol).
    A transmitting lobe passes its normal as `side_n`: each shadow ray then leaves to the light's side of it."""
    var tab = LobeTables(ctx.materials, ctx.curves, ctx.measured_brdfs, ctx.spectral)
    _lobe_nee_one[L, enqueue_shadow, defer_only](path_ptr, ctx, bsdf, tab, hit_point, ls_area, ls_area.dist * Float32(0.9999), side_n)
    for li in range(_nee_simple_light_count(ctx)):
        var res = _nee_sample_simple_light(ctx, li, hit_point, pcg)
        _lobe_nee_one[L, enqueue_shadow, defer_only](path_ptr, ctx, bsdf, tab, hit_point, res[0].copy(), res[1], side_n)
    for inf_i in range(ctx.lights.infinite_count):
        var u = u_env if inf_i == 0 and has_u_env else Point2f(pcg.next_float(), pcg.next_float())
        var ls_e = _sample_infinite_light_nee(ctx.lights.infinite_lights[unsafe_offset=inf_i], u, point3f(hit_point))
        _lobe_nee_one[L, enqueue_shadow, defer_only](path_ptr, ctx, bsdf, tab, hit_point, ls_e, ls_e.dist, side_n)


@always_inline
def _lobe_nee_one[L: NeeLobe, enqueue_shadow: Bool, defer_only: Bool = False](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    bsdf: L,
    tab: LobeTables,
    hit_point: Vec3f,
    ls: LightSample,
    tmax: Float32,
    side_n: Vec3f,
):
    if not bsdf.faces(ls):
        return
    var org = hit_point
    if dot(side_n, side_n) > Float32(0):
        org = offset_point(hit_point, side_n if dot(ls.wi, side_n) > Float32(0) else -side_n)
    # A costly lobe tests visibility first: a blocked sample then needs no evaluation.
    var inline_shadow = False
    comptime if not defer_only:
        if bsdf.costly():
            inline_shadow = _shadow_inline(ctx, null_guide())
            if inline_shadow and _shadow_blocked(ctx, org, ls.wi, tmax):
                return
    var w = bsdf.nee_weight(ls, tab, path_ptr[].wavelengths)
    if w.is_black():
        return
    if inline_shadow:
        path_ptr[].estimate += _clamped(ctx, path_ptr[].throughput * w, Int(path_ptr[].bounce))
    else:
        _shadow_contribute[enqueue_shadow, defer_only](path_ptr, ctx, org, ls.wi, tmax, path_ptr[].throughput * w)


@always_inline
def _nee_infinite_light[enqueue_shadow: Bool, defer_only: Bool = False](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    ilight: InfiniteLight,
    normal: Vec3f,
    hit_point: Vec3f,
    alb: RGB,
    u_env1: Float32,
    u_env2: Float32,
    mut pcg: PCG32,
    guide_write: GuideGrid,
):
    var env_dir: Vec3f
    var env_rgb: RGB
    var pdf_light: Float32

    if ilight.tex_idx >= Int32(0) and _is_real_ptr(ilight.pixels_ptr) and _is_real_ptr(ilight.cdf_ptr) and ilight.cdf_w > Int32(0):
        # ── CDF importance sampling, via the shared bvh.mojo implementation ──
        # (A separate "sample a cosine-hemisphere direction, look up the env
        # map there, add it unweighted" block used to live here as an extra
        # "BSDF-sampling" MIS strategy. It's redundant AND wrong: the real
        # BSDF-sampled continuation ray already delivers that same strategy
        # correctly — shade_nee_core's miss handler (below in this file)
        # applies power_heuristic(pdf_bsdf, pdf_light) when a bounce escapes
        # to this same light. The removed block added its full contribution
        # with NO such weight, double-counting unoccluded sky light and
        # washing out shadow contrast — see [[project_object_instancing]] for
        # the barcelona-pavilion/bunny-fur symptom this caused.)
        var (dir_v, rgb_v, pdf_v) = _sample_infinite_light_textured(ilight, Point2f(u_env1, u_env2))
        env_dir = dir_v.to_simd()
        env_rgb = rgb_v
        pdf_light = pdf_v
    else:
        # ── Fallback: cosine-weighted hemisphere (no CDF) ─────────────────
        var _env_s = sample_cosine_hemisphere_world(pcg.next_float(), pcg.next_float(), normal)
        env_dir = _env_s[0]
        pdf_light = _env_s[1]
        env_rgb = ilight.scale

    # ── MIS + shadow ray ──────────────────────────────────────────────────
    var cos_env = dot(normal, env_dir)
    if cos_env > Float32(0.0) and not env_rgb.is_black() and pdf_light > Float32(0.0):
        var pdf_bsdf_nee = bxdf_pdf_diffuse(cos_env)
        var mis_w = power_heuristic(pdf_light, pdf_bsdf_nee)
        var contrib = path_ptr[].throughput * _to_spec_diffuse_lit(ctx, alb, env_rgb, path_ptr[].wavelengths) * (cos_env / pdf_light) * mis_w
        var t_max_env = Float32(100000.0)
        _shadow_contribute[enqueue_shadow, defer_only](path_ptr, ctx, hit_point, env_dir, t_max_env, contrib, guide_write)


@always_inline
def _sample_light_point_and_normal(
    ctx: ShadeContext, al: AreaLight, u1: Float32, u2: Float32, mut pcg: PCG32,
    ref_point: Vec3f,
) -> Tuple[Vec3f, Vec3f, Vec3f, Vec3f]:
    """Point + outward normal on an area light's surface for NEE, plus the
    surface's own (dp_du, dp_dv) tangent basis (used only by
    _nee_area_lights' MNEE glass-refraction focusing — meaningless for a
    curve's round tube, always zero for kind==1 since MNEE is gated off
    there): a random triangle on a mesh light (kind==0, u1/u2 = barycentric
    coords), or a random point on a curve's swept tube (kind==1, u1 =
    position along the picked piece, u2 = angle around the tube). Which
    triangle/piece is picked is its own uniform draw from pcg, independent
    of (u1, u2) — mirrors the pre-existing mesh-light behaviour of picking
    a uniform random triangle rather than area-weighting by triangle."""
    if al.kind == Int8(1):
        var curve = curve_at(ctx.curves, Int(al.meshIdx))
        var piece = Int(pcg.next_uint() % UInt32(max(Int(curve.n_pieces), 1)))
        var (q0, q1, r0, r1) = curve_piece_endpoints(curve, piece)
        var axis = q1 - q0
        var axis_len = sqrt(dot(axis, axis))
        var axis_dir = Vec3f(Float32(0.0), Float32(0.0), Float32(1.0))
        if axis_len > Float32(1e-8):
            axis_dir = axis * (Float32(1.0) / axis_len)
        var r = r0 + (r1 - r0) * u1
        var u_perp = _curve_perp_axis(axis_dir)
        var v_perp = cross(axis_dir, u_perp)
        var theta = u2 * TWO_PI
        var (stheta, ctheta) = fast_sincos(theta)
        var radial = u_perp * ctheta + v_perp * stheta
        var point = q0 + axis_dir * (axis_len * u1) + radial * r
        var zero3 = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
        return (point, radial, zero3, zero3)
    else:
        var lmesh = ctx.meshes[unsafe_offset=Int(al.meshIdx)]
        var lti = area_light_pick_triangle(al, pcg.next_float())
        var lb = lti * 3
        var lv0 = Int(mesh_vi(lmesh, lb))
        var lv1 = Int(mesh_vi(lmesh, lb + 1))
        var lv2 = Int(mesh_vi(lmesh, lb + 2))
        var lp0 = mesh_pos(lmesh, lv0)
        var lp1 = mesh_pos(lmesh, lv1)
        var lp2 = mesh_pos(lmesh, lv2)
        var sqrt_r1 = sqrt(u1)
        var point = lp0 * (Float32(1.0) - sqrt_r1) + lp1 * (sqrt_r1 * (Float32(1.0) - u2)) + lp2 * (sqrt_r1 * u2)
        # Winding oriented to the mesh's supplied normals (see
        # _emitter_face_normal) -- this normal gates NEE via cos_l > 0, so
        # getting it backwards makes the light illuminate nothing at all.
        # Barycentrics of the sampled point, matching `point` above.
        var b_u = sqrt_r1 * (Float32(1.0) - u2)
        var b_v = sqrt_r1 * u2
        var lcross = _emitter_face_normal(lmesh, lv0, lv1, lv2, b_u, b_v, lp0, lp1, lp2)
        var normal = lcross
        var lcross_len = dot(lcross, lcross)
        if lcross_len > Float32(0.0):
            normal = lcross * (Float32(1.0) / sqrt(lcross_len))
        if al._pad0 != Int8(0) and dot(normal, point - ref_point) > Float32(0.0):
            normal = -normal   # twosided: emit toward the shading point
        return (point, normal, lp1 - lp0, lp2 - lp0)


@always_inline
def _sample_area_light_nee(
    ctx: ShadeContext,
    hit_point: Vec3f,
    mut pcg: PCG32,
) -> LightSample:
    """Uniform LightSample wrapper around the mesh/curve area-light picking
    machinery (light_sampler_sample + _sample_light_point_and_normal) — lets
    ANY NEE call site built on the Light/BxDF interface (bvh.mojo's
    LightSample + bxdf.mojo's _nee_weight_simple/_nee_weight_simple_spectral/
    _nee_weight_coated_*) consume area lights too, not just the diffuse-
    specific _nee_area_lights MNEE path (which additionally does glass-
    refraction focusing — irrelevant here, this is the plain solid-angle
    pdf every other light type already uses). valid=False when there are no
    area lights or the sampled point is degenerate (zero distance/area, or
    facing away from the shading point)."""
    var invalid = LightSample(Vec3f(Float32(0.0), Float32(0.0), Float32(0.0)),
                               RGB(Float32(0.0)), Float32(0.0), Float32(0.0), False, False)
    if ctx.lights.area_light_count == 0:
        return invalid.copy()
    var u_light = pcg.next_float()
    var r1 = pcg.next_float()
    var r2 = pcg.next_float()
    return _sample_area_light_nee_u(ctx, hit_point, u_light, r1, r2, pcg)


def _sample_area_light_nee_u(ctx: ShadeContext, hit_point: Vec3f, u_light: Float32, r1: Float32, r2: Float32,
                             mut pcg: PCG32) -> LightSample:
    """_sample_area_light_nee with the light pick and the point on it given (Sobol dimensions)."""
    var invalid = LightSample(Vec3f(Float32(0.0), Float32(0.0), Float32(0.0)),
                               RGB(Float32(0.0)), Float32(0.0), Float32(0.0), False, False)
    if ctx.lights.area_light_count == 0:
        return invalid.copy()
    var ls_result = light_sampler_sample(ctx.lights.light_sampler, u_light)
    var light_idx = ls_result[0]
    var light_sel_pdf = ls_result[1]
    var al = ctx.lights.area_lights[unsafe_offset=light_idx]
    var (light_point, light_normal, _, _) = _sample_light_point_and_normal(ctx, al, r1, r2, pcg, hit_point)
    var to_light = light_point - hit_point
    var dist_sq = dot(to_light, to_light)
    var dist = sqrt(dist_sq)
    if dist <= Float32(0.0001) or al.total_area <= Float32(0.0):
        return invalid.copy()
    var wi = to_light * (Float32(1.0) / dist)
    var cos_l = -dot(light_normal, wi)
    if cos_l <= Float32(0.0):
        return invalid.copy()
    var pdf = dist_sq * light_sel_pdf / (cos_l * al.total_area)
    if pdf <= Float32(0.0):
        return invalid.copy()
    return LightSample(wi, al.emission, pdf, dist, False, True)


@always_inline
def _sms_probe_and_solve(
    ctx: ShadeContext,
    hit_point: Vec3f,
    shadow_dir: Vec3f,
    dist: Float32,
    light_point: Vec3f,
    ldp_du_v: Vec3f,
    ldp_dv_v: Vec3f,
    mut pcg: PCG32,
) -> Tuple[Bool, Bool, Int, Array[SMSVertex, MAX_SMS_VERTICES], Float32, Float32, Float32]:
    var (probe_glass, probe_inter) = _mnee_first_probe(ctx, hit_point, shadow_dir, dist)
    if not probe_glass:
        return (False, False, 0, Array[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init()), Float32(0.0), Float32(0.0), Float32(0.0))
    return _sms_solve_from_probe(ctx, hit_point, shadow_dir, dist, light_point, ldp_du_v, ldp_dv_v, pcg, probe_inter)


@always_inline
def _mnee_mis_weight(
    normal: Vec3f, x0: Vec3f, verts: Array[SMSVertex, MAX_SMS_VERTICES], n: Int, g: Float32, pdf_area: Float32,
    t_pass: Float32,
) -> Float32:
    """Power-heuristic weight of MNEE against BSDF sampling for one solved chain; the BSDF path gets one minus it.
    The solve is deterministic, so MNEE's density is the light's area density over g, and both sides evaluate this on
    the same chain: the weights sum to one. BSDF sampling: diffuse at x0, then refraction picked with 1 - F, then the chain's media crossed unscattered (t_pass)."""
    if g <= Float32(0.0) or pdf_area <= Float32(0.0):
        return Float32(1.0)
    var d = verts[0].pos - x0
    var p_bsdf = bxdf_pdf_diffuse(dot(normal, d) / sqrt(max(dot(d, d), Float32(1e-30))))
    var q = _mnee_bsdf_over_light(p_bsdf, _mnee_chain_t_eta(x0, verts, n)[0] * t_pass, g, pdf_area)
    return Float32(1.0) / (Float32(1.0) + q * q)


@no_inline
def _mnee_bsdf_weight(
    ctx: ShadeContext, path_ptr: Pointer[PathState, MutUntrackedOrigin], inter: Intersection, light_hit: Vec3f,
) -> Float32:
    """The weight of this BSDF path to the emitter against MNEE, which may sample the very same path.

    _mnee_reproduce repeats the manifold solve that last_ns_p's NEE ran for this light point. If it ends at the vertex
    this ray left from, through as many specular vertices, MNEE samples this path too and the two share it by MIS
    (_mnee_mis_weight); any other path (another topology, another root, no root) is the BSDF's alone."""
    var (lmesh, lv0, lv1, lv2, _) = _get_tri_verts(inter, ctx.meshes)
    var lp0 = mesh_pos(lmesh, lv0)
    var x0 = path_ptr[].last_ns_p
    var o = path_ptr[].ray.origin
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var rp = _mnee_reproduce(ctx, x0, Int(path_ptr[].ns_spec), Vec3f(o.x, o.y, o.z), light_hit,
                             mesh_pos(lmesh, lv1) - lp0, mesh_pos(lmesh, lv2) - lp0, pcg)
    if not rp[0]:
        return Float32(1.0)
    var al_idx = Int(inter.primId.id1)
    var al = ctx.lights.area_lights[unsafe_offset=al_idx]
    var pdf_area = light_sampler_pdf_at(ctx.lights.light_sampler, Int32(al_idx), x0) / max(al.total_area, Float32(1e-30))
    var t_med = _mnee_chain_transmittance(ctx, x0, rp[2], rp[1], light_hit, path_ptr[].current_medium_idx, path_ptr[].wavelengths)
    return Float32(1.0) - _mnee_mis_weight(path_ptr[].last_ns_n, x0, rp[2], rp[1], rp[3], pdf_area, _mnee_pass(t_med))


@always_inline
def _mnee_live(ctx: ShadeContext) -> Bool:
    """MNEE (and the emitter-hit suppression that goes with it) runs only while the scene has smooth glass and the
    glass probes have not shown it to be rare. Cycles' sessions leave it off: Cycles runs MNEE only between objects
    flagged as caustic caster and receiver."""
    if not ctx.has_glass or ctx.cycles[3] != 0:
        return False
    if _is_real_ptr(ctx.mnee_stats) and ctx.mnee_stats[unsafe_offset=2] != Int32(0):
        return False
    return True


@always_inline
def _mnee_area_light_contribute(
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    normal: Vec3f,
    hit_point: Vec3f,
    alb: RGB,
    shadow_dir: Vec3f,
    dist: Float32,
    light_point: Vec3f,
    ldp_du_v: Vec3f,
    ldp_dv_v: Vec3f,
    al: AreaLight,
    inv_pdf_area: Float32,
    lobe_w: Float32,
    exit_eta: Float32 = Float32(0),   # a walk's exit: its lobe (sss_boundary_lobe) on top of `alb`
) -> Bool:
    """MNEE glass-caustic probing for one already-chosen area-light sample,
    factored out of `_nee_area_lights` so ReSTIR's `di_resolve` can use the
    identical transport (it previously could not, which lost ALL light
    reaching a surface through a dielectric on the ReSTIR path -- ~9% of
    staircase2, concentrated behind its glass railing).

    Probes for up to MAX_SMS_VERTICES glass surfaces between `hit_point` and
    `light_point`; on a hit, runs the 1-/2-vertex MNEE fast path (Phase
    5.4) or, for a genuinely longer chain, sms.mojo's general N-vertex
    manifold walk (Phase 5.1), adds the refracted contribution directly to
    `path_ptr[].estimate`, and returns True. The caller MUST then skip its
    own straight shadow ray -- MNEE/SMS REPLACES it
    (that ray is occluded by the glass by construction, so tracing it would
    contribute nothing anyway, but the `used_mnee` bookkeeping is what keeps
    the two strategies from being double-counted if that ever stops holding).
    Returns False when no dielectric intervenes, leaving the caller to do
    ordinary shadow-ray NEE.

    `inv_pdf_area` is 1/p_A(y), the reciprocal AREA-measure density of the
    light sample -- the one quantity that legitimately differs between the
    two callers, hence a parameter rather than a recomputation:
      * plain NEE  passes `al.total_area / light_sel_pdf` (the sample really
        was drawn with that area density);
      * ReSTIR DI  passes `W * dist^2 / cos_l`, converting its RIS weight
        (which estimates 1/q in SOLID-ANGLE measure, since
        `_di_sample_candidate`'s gen_pdf carries the dist^2/cos_l factor)
        back into area measure via q_sa = q_A * dist^2/cos_l. Getting this
        conversion wrong is silent: it only shows up as a scene-dependent
        brightness error on glass, exactly the class of bug this whole
        area is prone to.

    No MIS weight is applied here, matching the pre-existing plain-NEE
    behaviour: the refracted path is a specular chain that BSDF sampling
    effectively never reproduces, so there is no competing strategy to
    balance against."""
    # MNEE's glass-refraction focusing needs the light's own surface
    # tangents (ldp_du_v/ldp_dv_v) — well-defined for a flat mesh triangle,
    # not for a curve's swept tube. Curve lights (kind==1) fall back to
    # plain (non-MNEE) shadow-ray NEE through glass.
    if al.kind != Int8(0) or not _mnee_live(ctx):
        return False

    # Most samples see no glass: skip the solver's heavy setup for them (it would return not-found untouched).
    var (probe_glass, probe_inter) = _mnee_first_probe(ctx, hit_point, shadow_dir, dist)
    if _is_real_ptr(ctx.mnee_stats) and ((path_ptr[].pcgState >> 20) & UInt64(15)) == UInt64(0):
        var probes = Int(Atomic.fetch_add(ctx.mnee_stats, Int32(1))) + 1
        if probe_glass:
            _ = Atomic.fetch_add(ctx.mnee_stats.unsafe_offset(1), Int32(1))
        if probes == 4096 and Int(ctx.mnee_stats[unsafe_offset=1]) * 20 < probes:
            ctx.mnee_stats[unsafe_offset=2] = Int32(1)   # glass on under 5% of probes: MNEE is not worth its cost in this scene
    if not probe_glass:
        return False

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var c = _mnee_chain(ctx, normal, hit_point, shadow_dir, dist, light_point, ldp_du_v, ldp_dv_v, pcg, probe_inter)
    path_ptr[].pcgState = pcg.state
    if not c[0]:
        return False
    if not c[1]:
        return True
    # MIS against the BSDF path through the same chain (_mnee_bsdf_weight gives it the rest).
    var t_med = _mnee_chain_transmittance(ctx, hit_point, c[3], c[2], light_point, path_ptr[].current_medium_idx, path_ptr[].wavelengths)
    var w_mnee = _mnee_mis_weight(normal, hit_point, c[3], c[2], c[5], Float32(1.0) / inv_pdf_area, _mnee_pass(t_med))
    var mnee_wt = _to_spec_diffuse_lit(ctx, alb, al.emission, path_ptr[].wavelengths) * t_med * (c[6] * c[5] * c[4] * lobe_w * inv_pdf_area * w_mnee)
    if exit_eta > Float32(0):
        mnee_wt *= sss_boundary_lobe(c[6], exit_eta)
    path_ptr[].estimate += _clamped(ctx, path_ptr[].throughput * mnee_wt, Int(path_ptr[].bounce))
    return True


def _mnee_at_walk_exit(
    path_ptr: Pointer[PathState, MutUntrackedOrigin], ctx: ShadeContext, n: Vec3f, hit_point: Vec3f, eta: Float32,
    mut pcg: PCG32,
):
    """MNEE from a walk's exit, for skin seen through glass: _nee_area_lights' draw, without its plain shadow ray."""
    if ctx.lights.area_light_count == 0:
        return
    path_ptr[].last_ns_n = n
    path_ptr[].last_ns_p = hit_point
    path_ptr[].ns_spec = Int8(0)
    var (idx, sel_pdf) = light_sampler_sample_at(ctx.lights.light_sampler, hit_point, pcg.next_float())
    if idx < 0:
        return
    var al = ctx.lights.area_lights[unsafe_offset=idx]
    var (light_point, _, du, dv) = _sample_light_point_and_normal(ctx, al, pcg.next_float(), pcg.next_float(), pcg, hit_point)
    var to_light = light_point - hit_point
    var dist = sqrt(dot(to_light, to_light))
    if dist <= Float32(0.0001) or al.total_area <= Float32(0) or sel_pdf <= Float32(0):
        return
    path_ptr[].pcgState = pcg.state
    _ = _mnee_area_light_contribute(path_ptr, ctx, n, hit_point, RGB(Float32(1)), to_light * (Float32(1) / dist), dist,
                                    light_point, du, dv, al, al.total_area / sel_pdf, Float32(1), eta)
    pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)


def _glossy_sms_nee(
    path_ptr: Pointer[PathState, MutUntrackedOrigin], ctx: ShadeContext, bsdf: BSDF, hit_point: Vec3f, mut pcg: PCG32,
):
    """Glossy SMS at a glossy vertex: one light point, the specular chain a lobe seed solves to it, the reciprocal root
    estimate, and MIS against reaching the light through the same chain by sampling the lobe."""
    var u_l = pcg.next_float(); var u1 = pcg.next_float(); var u2 = pcg.next_float()
    var (idx, sel_pdf) = light_sampler_sample(ctx.lights.light_sampler, u_l)
    var al = ctx.lights.area_lights[unsafe_offset=idx]
    if al.kind != Int8(0) or al.total_area <= Float32(0):
        return
    var (y, ln, du, dv) = _sample_light_point_and_normal(ctx, al, u1, u2, pcg, hit_point)
    var (ldu, ldv) = mnee_orthonormal_basis(du, dv)
    var tab = LobeTables(ctx.materials, ctx.curves, ctx.measured_brdfs, ctx.spectral)
    var wl = path_ptr[].wavelengths
    comptime if GSMS_DEBUG == 6:
        if path_ptr[].specularBounce == Int8(1):   # reached through the camera's glass only
            path_ptr[].estimate += SpectralSample(Float32(1)) * glossy_sms_stage(ctx, hit_point, bsdf, tab, wl, y, ldu, ldv, pcg)
        return
    var verts = Array[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init())
    var r = glossy_sms_solve(ctx, hit_point, bsdf, tab, wl, y, ldu, ldv, pcg, verts)
    if not r[0]:
        return
    var n = r[1]
    if al._pad0 == Int8(0) and dot(ln, verts[n-1].pos - y) <= Float32(0):
        return   # a one-sided lamp turned away from the chain
    var d0 = verts[0].pos - hit_point
    var ev = bsdf.f(d0 * (Float32(1) / sqrt(dot(d0, d0))), tab, wl)
    if ev.f_cos.is_black():
        return
    var p_area = sel_pdf / al.total_area
    var t_med = _mnee_chain_transmittance(ctx, hit_point, verts, n, y, path_ptr[].current_medium_idx, wl)
    var w = glossy_sms_mis(ev.pdf_fwd, glossy_sms_t_chain(hit_point, verts, n) * _mnee_pass(t_med), r[3], p_area)
    var c = path_ptr[].throughput * ev.f_cos * _to_spec_illum(ctx, al.emission, wl) * t_med * (r[2] * r[3] * r[4] * w / p_area)
    comptime if GSMS_DEBUG == 1:
        path_ptr[].estimate += c
    path_ptr[].estimate += _clamped(ctx, c, Int(path_ptr[].bounce))


@no_inline
def _glossy_sms_bsdf_weight(
    ctx: ShadeContext, path_ptr: Pointer[PathState, MutUntrackedOrigin], inter: Intersection, light_hit: Vec3f,
) -> Float32:
    """This BSDF path's share against glossy SMS from the glossy vertex it left (PathState.gl_pdf): the rest of MIS
    when re-tracing its first direction by the chain rule lands on this light point, everything otherwise."""
    var x0 = path_ptr[].last_ns_p
    var n = Int(path_ptr[].ns_spec)
    var verts = Array[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init())
    var rp = glossy_sms_reproduce(ctx, x0, path_ptr[].gl_dir, n, light_hit, inter, verts)
    if not rp[0]:
        return Float32(1.0)
    var al_idx = Int(inter.primId.id1)
    var al = ctx.lights.area_lights[unsafe_offset=al_idx]
    var ls = ctx.lights.light_sampler
    var sel = max(ls.cdf[unsafe_offset=al_idx + 1] - ls.cdf[unsafe_offset=al_idx], Float32(1e-6))
    var t_med = _mnee_chain_transmittance(ctx, x0, verts, n, light_hit, path_ptr[].current_medium_idx, path_ptr[].wavelengths)
    var t_chain = glossy_sms_t_chain(x0, verts, n) * _mnee_pass(t_med)
    return Float32(1.0) - glossy_sms_mis(path_ptr[].gl_pdf, t_chain, rp[1], sel / max(al.total_area, Float32(1e-30)))


# ── ReSTIR SMS (Phase 6's remaining first piece: candidate generation) ──────
# Reuses _sms_probe_and_solve for the identical probe+Newton-solve logic
# _mnee_area_light_contribute's direct-contribution path already uses --
# see that function's docstring and project_sms_restir_phase6 memory for
# why this is a SEPARATE function rather than a third calling convention
# threaded through the already-validated direct-contribution one.
#
# Unlike ReSTIR DI's di_generate_reservoir (M=16 cheap light-sampler
# draws streamed via RIS), SMS has effectively M=1 per pixel per frame --
# generating even ONE admissible chain is the expensive part (a full
# Newton solve). ReSTIR's value here
# is almost entirely about reusing that one expensive solve across frames
# (temporal) and pixels (spatial) via sms_shift, not about resampling
# among many cheap candidates. Unlike ReSTIR GI's candidate generation
# (split across two bounces, target pdf deferred to a later call because
# it needs data unavailable until then), SMS has every quantity
# sms_target_pdf needs available immediately at generation time -- so,
# like di_generate_reservoir, this returns an ALREADY-STREAMED reservoir,
# not an unstreamed candidate.
def sms_generate_reservoir(
    ctx: ShadeContext,
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    shadow_dir: Vec3f, dist: Float32, light_point: Vec3f,
    ldp_du_v: Vec3f, ldp_dv_v: Vec3f,
    al: AreaLight, inv_pdf_area: Float32, mut pcg: PCG32,
) -> Tuple[Bool, SMSReservoir]:
    """Phase 6's candidate generation: probe+solve for an admissible
    specular chain toward the given area-light sample (same scope
    restriction as _mnee_area_light_contribute -- mesh lights only, kind
    0; curve lights fall back to ordinary shadow-ray NEE, no SMS
    candidate generated) and, on success, stream it into a fresh
    reservoir with RIS weight p̂/q where p̂ is sms_target_pdf and q folds
    in the light's own area-measure generation density (`inv_pdf_area`
    is 1/q, matching _mnee_area_light_contribute's own parameter of the
    same name) and `trials` (always 1.0).

    Returns (dielectric_found, reservoir) -- mirrors
    _sms_probe_and_solve's own split: `dielectric_found` alone tells the
    caller whether to skip its own straight shadow ray (the light IS
    occluded by glass either way), separate from whether the reservoir
    actually holds a candidate (n_vertices=0 covers "no dielectric",
    "solve failed to converge", and "target function evaluates to zero"
    identically -- all "no candidate this frame", matching
    di_generate_reservoir's own zero-weight handling). Conflating the two
    would be a real bug: reporting `dielectric_found=True` whenever
    `al.kind==0` (i.e. for every mesh light, glass or not) would make a
    temporal-reuse caller skip ordinary shadow-ray NEE for every mesh
    light in every scene, not just ones with glass in the way."""
    var res = sms_reservoir_init()
    if al.kind != Int8(0):
        return (False, res^)
    var _probe_res2 = _sms_probe_and_solve(
        ctx, hit_point, shadow_dir, dist, light_point, ldp_du_v, ldp_dv_v, pcg)
    var dielectric_found = _probe_res2[0]
    var solve_ok = _probe_res2[1]
    var n = _probe_res2[2]
    var verts = _probe_res2[3].copy()
    var bsdf_product = _probe_res2[4]
    var dx1_dxlight = _probe_res2[5]
    var trials = _probe_res2[6]
    if not dielectric_found or not solve_ok:
        return (dielectric_found, res^)
    var p_hat = sms_target_pdf(hit_point, normal, alb, verts[0].pos, verts[0].normal, al.emission, bsdf_product, dx1_dxlight)
    var weight = p_hat * inv_pdf_area * trials
    var accept = reservoir_update(res.state, weight, pcg.next_float())
    if accept:
        res.n_vertices = Int32(n)
        res.verts = verts.copy()
        res.light_point = light_point
        res.ldp_du = ldp_du_v
        res.ldp_dv = ldp_dv_v
        res.le = al.emission
        res.bsdf_product = bsdf_product
        res.dx1_dxlight = dx1_dxlight
    return (dielectric_found, res^)


# Defensive cap on a finalized SMS reservoir's own state.w -- same fix,
# same root cause, as DI_MAX_FINALIZED_WEIGHT/GI_MAX_FINALIZED_WEIGHT
# (see DI_MAX_FINALIZED_WEIGHT's own comment for the full mechanism):
# reservoir_combine's weight formula multiplies in a source reservoir's
# own already-finalized state.w, so one anomalous value compounds into
# unbounded growth across many frames of temporal reuse without a bound.
# Applied proactively here, not discovered via a live bug this time --
# both DI and GI needed this fix independently before it was understood
# to be a structural property of reservoir_combine itself, not a
# payload-specific quirk (see project_restir_migration.md memory).
comptime SMS_MAX_FINALIZED_WEIGHT: Float32 = Float32(10.0)


comptime SMS_TEMPORAL_M_CAP: Float32 = Float32(64.0)


def sms_resolve(
    path_ptr: Pointer[PathState, MutUntrackedOrigin], ctx: ShadeContext,
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    mut res: SMSReservoir,
):
    """Finalizes the reservoir's RIS weight (Bitterli et al. 2020 Eq. 6,
    via reservoir_finalize), then traces one shadow ray for the winning
    chain's last vertex -> light segment and adds its weighted
    contribution -- mirrors di_resolve/_mnee_area_light_contribute's own
    "use the result" step (same wi/cos_s_x0/visibility-ray pattern), the
    one piece sms_target_pdf itself doesn't cover since it only computes
    the scalar magnitude, not the per-channel Le-weighted contribution
    or the occlusion test. No MIS weight, same reasoning as
    _mnee_area_light_contribute's own (module docstring there): an SMS
    chain is a specular path BSDF sampling effectively never reproduces."""
    if res.n_vertices <= Int32(0):
        return
    var p_hat = sms_target_pdf(hit_point, normal, alb, res.verts[0].pos, res.verts[0].normal, res.le, res.bsdf_product, res.dx1_dxlight)
    reservoir_finalize(res.state, p_hat)
    if res.state.w > SMS_MAX_FINALIZED_WEIGHT:
        res.state.w = SMS_MAX_FINALIZED_WEIGHT
    if res.state.w <= Float32(0.0):
        return
    var n = Int(res.n_vertices)
    var first_vertex = res.verts[0].pos
    var last_vertex = res.verts[n - 1].pos
    var wi_f = hit_point - first_vertex
    var wi_len = sqrt(dot(wi_f, wi_f))
    if wi_len <= Float32(1e-8):
        return
    var wi_fn = wi_f * (Float32(1.0) / wi_len)
    var cos_s_x0 = dot(normal, -wi_fn)
    if cos_s_x0 <= Float32(0.0):
        return
    var wo_f = res.light_point - last_vertex
    var wo_len = sqrt(dot(wo_f, wo_f))
    if wo_len <= Float32(1e-8):
        return
    var wo_fn = wo_f * (Float32(1.0) / wo_len)
    var vis_org = last_vertex + wo_fn * Float32(0.001)
    var vis_ray = Ray(Point3f(vis_org[0], vis_org[1], vis_org[2]), Vec3f(wo_fn[0], wo_fn[1], wo_fn[2]))
    # See _mnee_area_light_contribute's identical fix: a sphere caustic
    # caster's outgoing leg necessarily re-crosses its OWN far side, so
    # exclude just that one sphere from this occlusion test.
    var ign_center = Vec3f(Float32(0.0))
    var ign_radius = Float32(-1.0)
    if res.verts[n-1].is_sphere != Int8(0):
        ign_center = res.verts[n-1].sphere_center
        ign_radius = res.verts[n-1].sphere_radius
    if any_hit_dispatch(ctx.bvh4, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, vis_ray, wo_len * Float32(0.999),
                          ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                          ctx.lights.spheres, ctx.lights.sphere_count,
                          ign_center, ign_radius, materials=ctx.materials):
        return
    var contrib = _to_spec_diffuse_lit(ctx, alb, res.le, path_ptr[].wavelengths) * (cos_s_x0 * res.state.w)
    path_ptr[].estimate += _clamped(ctx, path_ptr[].throughput * contrib, Int(path_ptr[].bounce))


def sms_temporal_step(
    path_ptr: Pointer[PathState, MutUntrackedOrigin], ctx: ShadeContext,
    hit_point: Vec3f, normal: Vec3f, alb: RGB,
    shadow_dir: Vec3f, dist: Float32, light_point: Vec3f,
    ldp_du_v: Vec3f, ldp_dv_v: Vec3f,
    al: AreaLight, inv_pdf_area: Float32, mut pcg: PCG32,
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
    pixel_idx: Int = -1,
) -> Bool:
    """Phase 6's temporal-only driver (no spatial reuse yet -- see
    project_sms_restir_phase6.md memory for the remaining scope):
    generate this frame's fresh candidate (sms_generate_reservoir),
    temporally combine with the previous frame's stored reservoir at the
    SAME pixel via IDENTITY reprojection (gonzales has no motion support,
    so x0/light_point are unchanged frame to frame -- no shift/Jacobian
    needed, matching di_temporal_step's own reasoning for the DI case),
    resolve one shadow ray for the combined winner, then M-cap and store
    for the next frame. Returns True iff a dielectric was found this
    frame (mirrors _mnee_area_light_contribute's own return convention --
    the caller must skip its ordinary shadow ray in that case, regardless
    of whether resolve produced a visible contribution). Falls back to
    plain single-frame candidate generation + resolve (no persistence)
    when sms_io isn't real -- matches di_temporal_step's own fallback for
    the batch (non-interactive) path."""
    var gen_result = sms_generate_reservoir(ctx, hit_point, normal, alb, shadow_dir, dist, light_point, ldp_du_v, ldp_dv_v, al, inv_pdf_area, pcg)
    var dielectric_found = gen_result[0]
    var res = gen_result[1].copy()
    var has_temporal = pixel_idx >= 0 and _is_real_ptr(sms_io.read)
    if has_temporal:
        var prev = sms_io.read[unsafe_offset=pixel_idx].copy()
        if prev.n_vertices > Int32(0):
            dielectric_found = True
            var p_hat_prev_here = sms_target_pdf(hit_point, normal, alb, prev.verts[0].pos, prev.verts[0].normal, prev.le, prev.bsdf_product, prev.dx1_dxlight)
            var accept = reservoir_combine(res.state, prev.state, p_hat_prev_here, pcg.next_float())
            if accept:
                res.n_vertices = prev.n_vertices
                res.verts = prev.verts.copy()
                res.light_point = prev.light_point
                res.ldp_du = prev.ldp_du
                res.ldp_dv = prev.ldp_dv
                res.le = prev.le
                res.bsdf_product = prev.bsdf_product
                res.dx1_dxlight = prev.dx1_dxlight

    # Spatial reuse (Phase 6's second driver). Runs after the temporal
    # combine and before resolve, matching gi_temporal_spatial_combine's
    # ordering: every neighbour's chain is carried across by the manifold
    # shift, so a tap that cannot be shifted into this pixel's domain
    # contributes nothing rather than a wrong chain. No-op unless the
    # caller supplied a real G-buffer.
    sms_spatial_combine(
        res, hit_point, normal, alb, light_point, ldp_du_v, ldp_dv_v,
        pcg, sms_io, pixel_idx)

    sms_resolve(path_ptr, ctx, hit_point, normal, alb, res)
    if has_temporal:
        reservoir_cap_confidence(res.state, SMS_TEMPORAL_M_CAP)
        sms_io.write[unsafe_offset=pixel_idx] = res^
    return dielectric_found


def _nee_area_lights[enqueue_shadow: Bool, mnee: Bool = True, defer_only: Bool = False](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    normal: Vec3f,
    hit_point: Vec3f,
    alb: RGB,
    u_light: Float32,
    u_bary1: Float32,
    u_bary2: Float32,
    mut pcg: PCG32,
    guide_write: GuideGrid,
    # Selection-probability compensation for compound BxDFs with more than one
    # lobe (e.g. diffuse_transmit's reflect/transmit split) — see bxdf_sample_
    # diffuse_transmit. 1.0 for a plain single-lobe Lambertian surface.
    lobe_w: Float32 = Float32(1.0),
    # Phase 6 (ReSTIR SMS): real only from _shade_diffuse_nee's bounce-0
    # call site when --sms-restir is active -- every other call site
    # (diffuse_transmit's own NEE, deeper bounces) leaves this at the
    # dangling default, getting exactly today's plain-MNEE-every-bounce
    # behavior. sms_temporal_step has no lobe_w parameter (assumes 1.0,
    # single-lobe diffuse only) -- fine today since no compound-BxDF call
    # site ever passes a real sms_io, but would need extending before one
    # could.
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
    pixel_idx: Int = -1,
):
    if ctx.lights.area_light_count == 0:
        return
    # Record the vertex, not the outcome: whether MNEE covers a given emitter
    # point is a property of that POINT (would this NEE reach it, and is
    # there glass on the way?), not of which point this sample happened to
    # draw -- so the decision is deferred to the emitter-hit site, which
    # knows the point actually reached. Marking only when THIS draw passed
    # the cosine tests below left the vertex unmarked whenever the draw
    # landed on a face turned away (most of a cube lamp seen from below),
    # and the BSDF ray through the glass was then counted on top of MNEE:
    # a mesh-cube lamp over a glass slab read 1.25x pbrt, 1.5x under it.
    # The ReSTIR SMS branch never marked, and still does not.
    if not (pixel_idx >= 0 and _is_real_ptr(sms_io.read)):
        path_ptr[].last_ns_n = normal
        path_ptr[].last_ns_p = hit_point
        path_ptr[].ns_spec = Int8(0)
    # The importance pick at this vertex; the emitter-hit site repeats it from last_ns_p, which this vertex marked.
    var at_point = not (pixel_idx >= 0 and _is_real_ptr(sms_io.read))
    var ls_result_nee = light_sampler_sample_at(ctx.lights.light_sampler, hit_point, u_light) if at_point else light_sampler_sample(ctx.lights.light_sampler, u_light)
    var light_idx = ls_result_nee[0]
    var light_sel_pdf_nee = ls_result_nee[1]
    if light_idx < 0:
        return
    var al = ctx.lights.area_lights[unsafe_offset=light_idx]
    var (light_point, light_normal, ldp_du_v, ldp_dv_v) = _sample_light_point_and_normal(ctx, al, u_bary1, u_bary2, pcg, hit_point)
    var to_light = light_point - hit_point
    var dist_sq = dot(to_light, to_light)
    var dist = sqrt(dist_sq)
    if dist > Float32(0.0001) and al.total_area > Float32(0.0):
        var shadow_dir = to_light * (Float32(1.0) / dist)
        var cos_s = dot(normal, shadow_dir)
        var cos_l = -dot(light_normal, shadow_dir)
        if cos_s > Float32(0.0) and cos_l > Float32(0.0):
            var pdf_light = dist_sq * light_sel_pdf_nee / (cos_l * al.total_area)
            var pdf_bsdf_nee = bxdf_pdf_diffuse(cos_s)
            var w_nee = Float32(1.0) if _emitter_invisible(ctx, al) else power_heuristic(pdf_light, pdf_bsdf_nee)
            var spec_w = _to_spec_diffuse_lit(ctx, alb, al.emission, path_ptr[].wavelengths) * (cos_s * w_nee * lobe_w / pdf_light)
            comptime if not defer_only:
                if pixel_idx >= 0 and _is_real_ptr(sms_io.read):
                    var contrib = path_ptr[].throughput * spec_w
                    # ReSTIR SMS (Phase 6): temporal-reused glass-caustic probing
                    # in place of plain per-frame MNEE. Same used/skip-shadow-ray
                    # convention as the plain-MNEE branch below.
                    var used_sms = sms_temporal_step(
                        path_ptr, ctx, hit_point, normal, alb, shadow_dir, dist,
                        light_point, ldp_du_v, ldp_dv_v, al,
                        al.total_area / light_sel_pdf_nee, pcg, sms_io, pixel_idx)
                    if not used_sms:
                        _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, shadow_dir, dist * Float32(0.9999), contrib, guide_write)
                    return
            # MNEE glass-caustic probing, shared verbatim with ReSTIR DI's
            # di_resolve via _mnee_area_light_contribute above. Returns True
            # when a dielectric intervened and it has already added the
            # refracted contribution -- in that case the straight shadow ray
            # below is deliberately skipped (MNEE replaces it, it does not
            # supplement it).
            # The vertex was marked at the top of this function (see
            # there and PathState.last_ns_n), so a later BSDF-sampled
            # arrival at the emitter through a specular chain -- the SAME
            # path family this strategy samples -- is not counted twice.
            # Keying that on `used_mnee` instead would make the
            # suppression depend on the NEE light draw, which is
            # independent of where the BSDF ray goes.
            var used_mnee = False
            comptime if mnee:
                if _shadow_inline(ctx, guide_write) and _mnee_live(ctx) and al.kind == Int8(0):
                    # A clear straight ray means no glass is in the way, so the MNEE probe is only needed when
                    # something blocks it (the probe's segment lies inside the shadow ray's).
                    if not _shadow_blocked(ctx, hit_point, shadow_dir, dist * Float32(0.9999)):
                        path_ptr[].estimate += _clamped(ctx, path_ptr[].throughput * spec_w, Int(path_ptr[].bounce))
                    else:
                        _ = _mnee_area_light_contribute(
                            path_ptr, ctx, normal, hit_point, alb, shadow_dir, dist,
                            light_point, ldp_du_v, ldp_dv_v, al,
                            al.total_area / light_sel_pdf_nee, lobe_w)
                    used_mnee = True
                else:
                    used_mnee = _mnee_area_light_contribute(
                        path_ptr, ctx, normal, hit_point, alb, shadow_dir, dist,
                        light_point, ldp_du_v, ldp_dv_v, al,
                        al.total_area / light_sel_pdf_nee, lobe_w)
            if not used_mnee:
                var contrib = path_ptr[].throughput * spec_w
                _shadow_contribute[enqueue_shadow, defer_only](path_ptr, ctx, hit_point, shadow_dir, dist * Float32(0.9999), contrib, guide_write)
