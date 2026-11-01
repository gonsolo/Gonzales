"""The path tracer's material shaders."""
from std.math import sqrt, atan, exp, pow
from gonzales.core.geometry import offset_point, offset_eps, RGB, Point3f, Point2f, Vec3f, dot, face_toward, cross, Frame, PI, INV_PI, _is_real_ptr
from gonzales.core.geometry import point3f
from gonzales.scene.appearance.materials import Material, MatKind, LobeKind, sss_boundary_lobe
from gonzales.transport.sss_walk import sss_leave, sss_entry_lobe, sss_walk_start
from gonzales.transport.render_state import PathState, SssWalk
from gonzales.scene.shapes.primitives import Intersection, TriangleMesh, Sphere, mesh_uv, mesh_has_uvs
from gonzales.core.geometry import Ray
from gonzales.scene.shapes.primitives import mesh_pos
from gonzales.transport.bxdf import BSDF, HairBSDF, LobeSample, BxDFFlags, bxdf_sample_dielectric, bxdf_sample_thin_dielectric, bxdf_sample_diffuse_transmit, bxdf_eval_conductor_ggx, ggx_albedo, bxdf_pdf_conductor_ggx, _nee_weight_simple_spectral, LobeTables
from gonzales.core.rng import PCG32
from gonzales.scene.accel.bvh import curve_hit_frame, curve_offset_eps, LightSample, _sample_infinite_light_nee
from gonzales.core.sampling import power_heuristic, sample_cosine_hemisphere_world, sample_ggx_vndf
from gonzales.core.transform import transform_normal, Mat4
from gonzales.transport.guide import GuideGrid, guide_pos_to_cell, guide_pdf, guide_sample, guide_cell_has_data, null_guide, guide_is_active
from gonzales.core.spectrum import SpectralSample, spectral_sample_to_rgb, rgb_bands_to_spectral_sample
from gonzales.core.reservoir import reservoir_update
from gonzales.transport.restir_di import ReservoirIO, reservoir_io_null
from gonzales.transport.restir_gi import gi_target_pdf, gi_temporal_spatial_combine
from gonzales.integrators.pt.restir_sms import SMSReservoirIO, sms_reservoir_io_null
from gonzales.transport.texturing import _apply_surface_maps, _tex_lookup, sample_texture
from gonzales.transport.shade_context import GIPendingX1, ShadeContext
from gonzales.transport.manifold import _interp_vertex_normal
from gonzales.integrators.pt.common import _shading_normal, _curve_extra_offset, _hit_geom, _to_spec_refl, _clamped, _to_spec_illum, _shadow_inline, _shadow_blocked, _shadow_contribute, RR_THROUGHPUT_CLAMP, BOUNCE_DIFFUSE, BOUNCE_GLOSSY, BOUNCE_TRANSMIT, _count_bounce, _apply_russian_roulette, spawn_origin, _finish_delta_bounce, _apply_normal_map_sphere, _pt_hit_footprint, _build_geom_context_full, _draw_sobol_8
from gonzales.integrators.pt.nee import _nee_simple_light_count, _nee_sample_simple_light, _nee_loop_simple, _lobe_nee, _lobe_nee_with, _nee_infinite_light, _sample_area_light_nee_u, _mnee_at_walk_exit, _nee_area_lights
from gonzales.integrators.pt.restir import _gi_generate_recon_candidate, gi_resolve, di_temporal_step


# ── DiffuseTransmission branch ────────────────────────────────────────────────
@always_inline
def shade_diffuse_transmission[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    ctx: ShadeContext,
):
    var mat = ctx.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var (ok, is_sphere, normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(
        path_ptr, inter, ctx.meshes, ctx.lights.spheres, inter.primId.instanceIdx, ctx.instances, ctx.curves)
    if not ok:
        path_ptr[].active = 0
        return
    var geo_n = normal   # face-forwarded geometric normal, for the ray offset
    var dt_fp_width = Float32(0.0)
    if not is_sphere:
        # 146 corpus diffusetransmission materials carry "texture displacement"
        # (foliage cards in sanmiguel, etc).
        var (tri_dt, fp_dt) = _pt_hit_footprint(path_ptr, mat, ctx.cam_fp, ctx.instances, mesh, v0, v1, v2, inter, ray_org, ray_dir, ctx.textures, ctx.n_textures)
        dt_fp_width = fp_dt.width
        normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geo_n, inter.primId.instanceIdx, ctx.instances)
        normal = _apply_surface_maps[use_gpu](mat, v0, v1, v2, mesh, inter, normal, geo_n,
            tri_dt, fp_dt, ctx.tex_filenames, ctx.textures, ctx.n_textures)
        normal = face_toward(normal, -ray_dir)   # reflect lobe on wo's side

    # "texture reflectance"/"texture transmittance" (e.g. a leaf.tga imagemap)
    # both resolve to the same mat.tex_idx (Material has one texture slot,
    # shared across kinds) — sample it once and use it for both lobes. Leaving
    # this at the flat mat.albedo/mat.emission default (as before) rendered
    # textured diffusetransmission foliage as a dull, wall-coloured grey blob
    # instead of the actual leaf texture, effectively invisible against a
    # similarly-toned wall. No UV space exists on a sphere, so it always
    # keeps the flat default.
    var refl = mat.albedo
    var trans = mat.emission
    if not is_sphere and Int(mat.tex_idx) != -1:
        var tex_rgb = _tex_lookup[use_gpu](mat, inter, v0, v1, v2, mesh, ctx.tex_filenames, ctx.textures, ctx.n_textures, dt_fp_width)
        refl = mat.dt_reflectance(tex_rgb)
        trans = mat.dt_transmittance(tex_rgb)

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var (bs, bounce_normal, lobe_alb, lobe_w, _) = bxdf_sample_diffuse_transmit(
        normal, refl, trans, pcg.next_float(), pcg.next_float(), pcg.next_float())
    if bs.is_valid == Int8(0):
        path_ptr[].active = 0
        path_ptr[].pcgState = pcg.state
        return

    # Offset along the GEOMETRIC normal, to the chosen lobe's side (pbrt's
    # OffsetRayOrigin): the shading normal can point into the surface.
    var off_n = geo_n if dot(bounce_normal, -ray_dir) > Float32(0.0) else -geo_n
    var extra_off = _curve_extra_offset(inter, ctx.curves, ctx.instances)
    if dot(bounce_normal, -ray_dir) <= Float32(0.0):
        extra_off *= Float32(2.05)   # a transmitted ray must clear the far wall of the tube
    var hit_point = offset_point(ray_org + ray_dir * inter.tHit, off_n) + off_n * extra_off

    # ── NEE direct light sampling (MIS weighted, with MNEE glass caustics) ─────
    # Shares _nee_area_lights with plain diffuse — including its MNEE probe for
    # up to 2 glass surfaces between hit_point and the light — parameterized by
    # the chosen lobe's normal/albedo and its selection-weight compensation.
    _nee_area_lights[enqueue_shadow](path_ptr, ctx, bounce_normal, hit_point, lobe_alb,
        pcg.next_float(), pcg.next_float(), pcg.next_float(), pcg, null_guide(), lobe_w)

    # ── Distant/point/sphere/infinite NEE, via the shared Light interface +
    # BxDF interface (_nee_weight_simple, mat_kind=0=diffuse — this lobe's
    # BRDF is Lambertian) ────────────────────────────────────────────────────
    # This material previously had NO direct lighting from these light types
    # at all (only area lights, above) — a severe gap for infinite-light-only
    # scenes with diffusetransmission-heavy foliage (e.g. sanmiguel-courtyard:
    # every transmissive leaf/vine surface relied purely on noisy BSDF-escape
    # rays to ever see the env map's sun, producing systematic under-lighting
    # — darker, blue-sky-dominated average — plus severe fireflies from the
    # rare lucky hits). lobe_w compensates for the stochastic reflect/
    # transmit lobe selection, same as _nee_area_lights above.
    var wo_dt = -ray_dir
    # Distant + point + sphere via the shared sweep. This was three verbatim
    # copies of _nee_loop_simple's own body -- same weight function, same
    # order, differing only by the `lobe_w` compensation, which the helper now
    # takes. Order matters and is preserved: of these light types only SPHERE
    # draws from `pcg`, so the sequence is identical and so is every image.
    _nee_loop_simple[enqueue_shadow](path_ptr, ctx, bounce_normal, hit_point,
                                     lobe_alb, Float32(0.0), Int32(0), wo_dt, pcg,
                                     null_guide(), lobe_w)
    for inf_i in range(ctx.lights.infinite_count):
        var ls_e = _sample_infinite_light_nee(ctx.lights.infinite_lights[unsafe_offset=inf_i], Point2f(pcg.next_float(), pcg.next_float()), point3f(hit_point))
        var w_e = _nee_weight_simple_spectral(ls_e, LobeKind.lambertian, lobe_alb, Float32(0), bounce_normal, wo_dt, ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65, path_ptr[].wavelengths, LobeTables(ctx.materials, ctx.curves, ctx.measured_brdfs, ctx.spectral)) * lobe_w
        if not w_e.is_black():
            var contrib_e = path_ptr[].throughput * w_e
            _shadow_contribute[enqueue_shadow](path_ptr, ctx, hit_point, ls_e.wi, ls_e.dist, contrib_e)

    # ── BSDF scatter ───────────────────────────────────────────────────────────
    path_ptr[].ray = Ray(Point3f(hit_point[0], hit_point[1], hit_point[2]), Vec3f(bs.wi[0], bs.wi[1], bs.wi[2]))

    # Throughput: lobe_alb / pdf_bsdf * lobe_selection_weight
    # = lobe_alb / (cos/π) * (total/p_lobe) → lobe_alb * π/cos * lobe_w
    # But cosine-hemisphere importance sampling gives cos/π cancel:
    # f * cos / pdf = (lobe_alb/π) * cos / (cos/π) = lobe_alb
    # Then multiply by lobe_w to compensate for stochastic lobe selection.
    path_ptr[].throughput *= _to_spec_refl(ctx, lobe_alb, path_ptr[].wavelengths) * lobe_w

    # Store BSDF pdf for next-bounce MIS (cosine hemisphere: cos/π)
    path_ptr[].lastBsdfPdf = bs.pdf
    path_ptr[].specularBounce = Int8(0)

    if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
        path_ptr[].albedo = refl
    path_ptr[].bounce += 1

    var u_rr = pcg.next_float()
    _apply_russian_roulette(path_ptr, ctx.cycles, pcg, u_rr, BOUNCE_TRANSMIT if dot(bounce_normal, -ray_dir) <= Float32(0.0) else BOUNCE_DIFFUSE)


@always_inline
def _shade_principled[enqueue_shadow: Bool, defer_only: Bool](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    mat: Material,
    inter: Intersection,
    alb: RGB, normal: Vec3f, geo_normal: Vec3f, hit_point: Vec3f, wo: Vec3f, ior: Float32, alpha: Float32,
    mut pcg: PCG32,
):
    """Cycles' Principled BSDF: a GGX dielectric specular layer over a diffuse or random-walk base, the base scaled
    by one minus the layer's directional albedo (closure_layering_weight). pbrt's layered coat traps light by total
    internal reflection and renders a mid grey at about half of Cycles' brightness."""
    var r0 = (ior - Float32(1)) / (ior + Float32(1))
    var f0 = r0 * r0
    var a = max(alpha, Float32(1e-3))
    var mu_o = max(dot(normal, wo), Float32(1e-4))
    var schlick = (Float32(1) - mu_o) * (Float32(1) - mu_o)
    schlick = schlick * schlick * (Float32(1) - mu_o)
    var transmit = Float32(1) - ggx_albedo(mu_o, a) * (f0 + (Float32(1) - f0) * schlick)
    var walk_w = mat.sss_walk_weight()
    var kd = transmit * (Float32(1) - walk_w)
    var Rd = _to_spec_refl(ctx, alb, path_ptr[].wavelengths) * (kd * INV_PI)
    var spec_e = Float32(1) - transmit
    var ps = min(max(spec_e / max(spec_e + kd * alb.luma(), Float32(1e-6)), Float32(0.1)), Float32(0.9))

    var ss = _draw_sobol_8(path_ptr, ctx.sobol_matrices)
    var ls_area = _sample_area_light_nee_u(ctx, hit_point, ss.light, ss.bary1, ss.bary2, pcg)
    _principled_nee[enqueue_shadow, defer_only](path_ptr, ctx, hit_point, ls_area, ls_area.dist * Float32(0.9999), normal, wo, Rd, f0, a, ps)
    for li in range(_nee_simple_light_count(ctx)):
        var res = _nee_sample_simple_light(ctx, li, hit_point, pcg)
        _principled_nee[enqueue_shadow, defer_only](path_ptr, ctx, hit_point, res[0].copy(), res[1], normal, wo, Rd, f0, a, ps)
    for inf_i in range(ctx.lights.infinite_count):
        var ls_inf = _sample_infinite_light_nee(ctx.lights.infinite_lights[unsafe_offset=inf_i], Point2f(ss.env1, ss.env2) if inf_i == 0 else Point2f(pcg.next_float(), pcg.next_float()), point3f(hit_point))
        _principled_nee[enqueue_shadow, defer_only](path_ptr, ctx, hit_point, ls_inf, ls_inf.dist, normal, wo, Rd, f0, a, ps)

    if walk_w > Float32(0):
        # The surface lobes keep a share even at full weight: they alone sample the highlight NEE's MIS assumes.
        var q = walk_w * Float32(0.8)
        if pcg.next_float() < q:
            path_ptr[].throughput *= walk_w / q
            _sss_walk_enter(path_ptr, ctx, mat, inter, alb, normal, geo_normal, hit_point, wo, ior, alpha, transmit, pcg)
            return
        path_ptr[].throughput *= Float32(1) / (Float32(1) - q)

    var wi: Vec3f
    var spec = ss.rr < ps
    if spec:
        var frame = Frame.from_z(normal)
        var tx = Vec3f(frame.x.x, frame.x.y, frame.x.z)
        var ty = Vec3f(frame.y.x, frame.y.y, frame.y.z)
        var hl = sample_ggx_vndf(Vec3f(dot(wo, tx), dot(wo, ty), mu_o), a, a, ss.scat1, ss.scat2)
        var h = tx * hl.x + ty * hl.y + normal * hl.z
        wi = h * (Float32(2) * dot(wo, h)) - wo
    else:
        var cs = sample_cosine_hemisphere_world(ss.scat1, ss.scat2, normal)
        wi = cs[0]
    var cos_i = dot(normal, wi)
    if cos_i <= Float32(0):
        path_ptr[].active = 0
        path_ptr[].pcgState = pcg.state
        return
    var pdf = ps * bxdf_pdf_conductor_ggx(normal, wo, wi, a) + (Float32(1) - ps) * cos_i * INV_PI
    var f_spec = bxdf_eval_conductor_ggx(normal, wo, wi, a, RGB(f0)).r
    var org = spawn_origin(hit_point, geo_normal, wi)
    path_ptr[].ray = Ray(Point3f(org[0], org[1], org[2]), Vec3f(wi[0], wi[1], wi[2]))
    path_ptr[].throughput *= (Rd + SpectralSample(f_spec)) * (cos_i / max(pdf, Float32(1e-12)))
    path_ptr[].specularBounce = Int8(0)
    path_ptr[].lastBsdfPdf = pdf
    path_ptr[].bounce += 1
    var u_rr = pcg.next_float()
    _apply_russian_roulette(path_ptr, ctx.cycles, pcg, u_rr, BOUNCE_GLOSSY if spec else BOUNCE_DIFFUSE)


@always_inline
def _principled_nee[enqueue_shadow: Bool, defer_only: Bool](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    hit_point: Vec3f,
    ls: LightSample,
    tmax: Float32,
    n: Vec3f, wo: Vec3f, Rd: SpectralSample, f0: Float32, alpha: Float32, ps: Float32,
):
    if not ls.valid:
        return
    var cos_i = dot(n, ls.wi)
    if cos_i <= Float32(0) or (not ls.is_delta and ls.pdf <= Float32(0)):
        return
    var inline_shadow = False
    comptime if not defer_only:
        inline_shadow = _shadow_inline(ctx, null_guide())
        if inline_shadow and _shadow_blocked(ctx, hit_point, ls.wi, tmax):
            return
    var w = cos_i
    if not ls.is_delta:
        var pdf = ps * bxdf_pdf_conductor_ggx(n, wo, ls.wi, alpha) + (Float32(1) - ps) * cos_i * INV_PI
        w *= power_heuristic(ls.pdf, pdf) / ls.pdf
    var f_spec = bxdf_eval_conductor_ggx(n, wo, ls.wi, alpha, RGB(f0)).r
    var contrib = path_ptr[].throughput * (Rd + SpectralSample(f_spec)) * _to_spec_illum(ctx, ls.Li, path_ptr[].wavelengths) * w
    if inline_shadow:
        path_ptr[].estimate += _clamped(ctx, contrib, Int(path_ptr[].bounce))
    else:
        _shadow_contribute[enqueue_shadow, defer_only](path_ptr, ctx, hit_point, ls.wi, tmax, contrib)


def _sss_walk_enter(
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    mat: Material,
    inter: Intersection,
    alb: RGB, normal: Vec3f, geo_normal: Vec3f, hit_point: Vec3f, wo: Vec3f, ior: Float32, alpha_coat: Float32,
    transmit: Float32,
    mut pcg: PCG32,
):
    """Start a random walk under the surface as Cycles does (subsurface_bounce, subsurface_random_walk_coefficients):
    refract through a GGX microfacet of the surface's roughness, with its single-scattering albedo fitted to the
    textured base colour."""
    var vdh = mat.sss_walk_van_de_hulst()
    var g = min(max(mat.sss_walk_g(), Float32(-0.99)), Float32(0.99) if vdh else Float32(0.9))
    var mfp = mat.sss_walk_mfp()
    var wl = path_ptr[].wavelengths
    # Each wavelength's albedo comes from the base colour's reflectance spectrum there, so the walk reflects that
    # spectrum; the radius, a length per RGB channel, is band-picked.
    var c = _to_spec_refl(ctx, alb, wl)
    var r = rgb_bands_to_spectral_sample(mfp.r, mfp.g, mfp.b, wl)
    var k = Float32(1) if vdh else Float32(1) / (Float32(1) - g)
    var a = SpectralSample(_cycles_sss_alpha(c.v0, g, vdh), _cycles_sss_alpha(c.v1, g, vdh),
                           _cycles_sss_alpha(c.v2, g, vdh), _cycles_sss_alpha(c.v3, g, vdh))
    # Cycles raises a low albedo to 0.2 and takes the difference out of the throughput.
    var lift = SpectralSample(min(a.v0 * Float32(5), Float32(1)), min(a.v1 * Float32(5), Float32(1)),
                              min(a.v2 * Float32(5), Float32(1)), min(a.v3 * Float32(5), Float32(1)))
    a = SpectralSample(max(a.v0, Float32(0.2)), max(a.v1, Float32(0.2)), max(a.v2, Float32(0.2)), max(a.v3, Float32(0.2)))
    var sigma_t = SpectralSample(k / max(r.v0, Float32(1e-7)), k / max(r.v1, Float32(1e-7)),
                                 k / max(r.v2, Float32(1e-7)), k / max(r.v3, Float32(1e-7)))
    # Refraction through a sampled microfacet; the specular layer above already took its share.
    var frame = Frame.from_z(normal)
    var tx = Vec3f(frame.x.x, frame.x.y, frame.x.z)
    var ty = Vec3f(frame.y.x, frame.y.y, frame.y.z)
    var cos_ni = dot(normal, wo)
    var gn = face_toward(geo_normal, wo)
    var a_ggx = max(alpha_coat, Float32(1e-4))
    var hl = sample_ggx_vndf(Vec3f(dot(wo, tx), dot(wo, ty), cos_ni), a_ggx, a_ggx, pcg.next_float(), pcg.next_float())
    path_ptr[].pcgState = pcg.state
    var h = tx * hl.x + ty * hl.y + normal * hl.z
    var neta = Float32(1) / min(max(ior, Float32(1.01)), Float32(3.8))
    var cos_hi = dot(h, wo)
    var arg = Float32(1) - neta * neta * (Float32(1) - cos_hi * cos_hi)
    if cos_ni <= Float32(0) or arg < Float32(0):
        path_ptr[].active = 0
        return
    var d = -(wo * neta) + h * (neta * cos_hi - max(sqrt(arg), Float32(1e-7)))
    var dl = sqrt(dot(d, d))
    d = d * (Float32(1) / dl)
    if dot(d, gn) >= Float32(0):
        path_ptr[].active = 0
        return
    # Just under the surface: spawn_origin's fixed 2e-4 is most of a mean free path for thin skin.
    var raw = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z) + \
              Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z) * inter.tHit
    var org = offset_point(raw, -gn)
    path_ptr[].ray = Ray(Point3f(org[0], org[1], org[2]), Vec3f(d[0], d[1], d[2]))
    path_ptr[].throughput *= lift * transmit
    path_ptr[].sss = SssWalk(sigma_t, a, g, Int32(inter.primId.id1), inter.primId.instanceIdx, Int32(0), Int32(0),
                             raw, normal, Float32(-1), Float32(1), Float32(1), Float32(1), Float32(0), Float32(0), Float32(0))
    path_ptr[].specularBounce = Int8(0)
    path_ptr[].lastBsdfPdf = Float32(0)
    path_ptr[].bounce += 1


def _sss_walk_enter_interior(
    path_ptr: Pointer[PathState, MutUntrackedOrigin], mat: Material, inter: Intersection, hit: Vec3f, geo_normal: Vec3f,
    wi: Vec3f,
):
    """Start the walk below a pbrt subsurface boundary along the refracted direction wi, with the interior's own
    extinction and albedo (Material.sss_walk_interior). It leaves through the walk's Lambertian exit, which carries all
    the energy out as pbrt's normalized Fresnel exit lobe does."""
    var (ray, walk) = sss_walk_start(mat, inter, hit, geo_normal, wi, path_ptr[].wavelengths)
    path_ptr[].ray = ray
    path_ptr[].sss = walk
    if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
        var mr = mat.sss_mean_refl
        path_ptr[].albedo = mr if mr.r + mr.g + mr.b > Float32(0) else RGB(Float32(1))
    path_ptr[].specularBounce = Int8(0)
    path_ptr[].lastBsdfPdf = Float32(0)
    path_ptr[].bounce += 1


@always_inline
def _cycles_sss_alpha(color: Float32, g: Float32, van_de_hulst: Bool) -> Float32:
    """Cycles' single-scattering albedo for a subsurface colour (subsurface_random_walk_remap): d'Eon's van de Hulst
    mapping, or the Legacy walk's fit in the anisotropy."""
    var c = min(max(color, Float32(0)), Float32(1))
    if van_de_hulst:
        var q = Float32(4.20863) * c - sqrt(Float32(9.59217) + Float32(41.6808) * c + Float32(17.7126) * c * c) + Float32(4.09712)
        var s2 = q * q
        return min(max((Float32(1) - s2) / (Float32(1) - g * s2), Float32(0)), Float32(0.999999))
    var g2 = g * g; var g3 = g2 * g; var g4 = g3 * g; var g5 = g4 * g; var g6 = g5 * g; var g7 = g6 * g
    var A = Float32(1.8260523782) - Float32(1.28451056436) * g - Float32(1.79904629312) * g2 + Float32(9.19393289202) * g3 - Float32(22.8215585862) * g4 + Float32(32.0234874259) * g5 - Float32(23.6264803333) * g6 + Float32(7.21067002658) * g7
    var B = Float32(4.98511194385) + Float32(0.127355959438) * exp(Float32(31.1491581433) * g - Float32(201.847017512) * g2 + Float32(841.576016723) * g3 - Float32(2018.09288505) * g4 + Float32(2731.71560286) * g5 - Float32(1935.41424244) * g6 + Float32(559.009054474) * g7)
    var C = Float32(1.09686102424) - Float32(0.394704063468) * g + Float32(1.05258115941) * g2 - Float32(8.83963712726) * g3 + Float32(28.8643230661) * g4 - Float32(46.8802913581) * g5 + Float32(38.5402837518) * g6 - Float32(12.7181042538) * g7
    var D = Float32(0.496310210422) + Float32(0.360146581622) * g - Float32(2.15139309747) * g2 + Float32(17.8896899217) * g3 - Float32(55.2984010333) * g4 + Float32(82.065982243) * g5 - Float32(58.5106008578) * g6 + Float32(15.8478295021) * g7
    var E = Float32(4.23190299701) + Float32(0.00310603949088) * exp(Float32(76.7316253952) * g - Float32(594.356773233) * g2 + Float32(2448.8834203) * g3 - Float32(5576.68528998) * g4 + Float32(7116.60171912) * g5 - Float32(4763.54467887) * g6 + Float32(1303.5318055) * g7)
    var F = Float32(2.40602999408) - Float32(2.51814844609) * g + Float32(9.18494908356) * g2 - Float32(79.2191708682) * g3 + Float32(259.082868209) * g4 - Float32(403.613804597) * g5 + Float32(302.85712436) * g6 - Float32(87.4370473567) * g7
    var blend = pow(c, Float32(0.25))
    var alpha = (Float32(1) - blend) * A * pow(atan(B * c), C) + blend * D * pow(atan(E * c), F)
    return min(max(alpha, Float32(0)), Float32(0.999999))


def _shade_sss_exit[enqueue_shadow: Bool](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    ctx: ShadeContext,
):
    """Where a random walk leaves the surface: a white Lambertian facing out, as Cycles' legacy walk exits."""
    path_ptr[].sss.exiting = Int32(0)
    var (ok, is_sphere, geo_normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(
        path_ptr, inter, ctx.meshes, ctx.lights.spheres, inter.primId.instanceIdx, ctx.instances, ctx.curves)
    if not ok or is_sphere:
        path_ptr[].active = 0
        return
    var gn = face_toward(geo_normal, ray_dir)
    var n = face_toward(_shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geo_normal, inter.primId.instanceIdx, ctx.instances), ray_dir)
    var hit_point = offset_point(ray_org + ray_dir * inter.tHit, gn)
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var exit_mat = ctx.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var interior = exit_mat.sss_walk_interior()   # pbrt's subsurface; Principled keeps Cycles' plain Lambertian exit
    var white = BSDF(LobeKind.sss_exit if interior else LobeKind.lambertian, True, False, n, n, RGB(Float32(1)), Int32(-1),
                     exit_mat.albedo.r if interior else Float32(0), Float32(0), Int32(-1), Float32(0), Float32(0), True, False)
    _lobe_nee[enqueue_shadow](path_ptr, ctx, white, hit_point, pcg)
    if interior:
        _mnee_at_walk_exit(path_ptr, ctx, n, hit_point, exit_mat.albedo.r, pcg)
    var out = sss_leave(n, exit_mat.albedo.r, pcg.next_float(), pcg.next_float())
    path_ptr[].pcgState = pcg.state
    if out.pdf <= Float32(0) or dot(out.wi, gn) <= Float32(0):
        path_ptr[].active = 0
        return
    if interior:
        path_ptr[].throughput *= SpectralSample(out.weight)
    var org = spawn_origin(hit_point, gn, out.wi)
    path_ptr[].ray = Ray(Point3f(org[0], org[1], org[2]), Vec3f(out.wi[0], out.wi[1], out.wi[2]))
    path_ptr[].specularBounce = Int8(0)
    path_ptr[].lastBsdfPdf = out.pdf
    path_ptr[].bounce += 1


def shade_coated_diffuse[use_gpu: Bool, enqueue_shadow: Bool, defer_only: Bool = False](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    ctx: ShadeContext,
    mat: Material,
):
    var (ok, is_sphere, geo_normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(
        path_ptr, inter, ctx.meshes, ctx.lights.spheres, inter.primId.instanceIdx, ctx.instances, ctx.curves)
    if not ok:
        path_ptr[].active = 0
        return

    var alb: RGB
    var normal: Vec3f
    if is_sphere:
        # Analytic sphere: exact normal IS the shading normal, no UV space
        # so alb falls back to the material's flat albedo.
        alb = mat.albedo
        normal = geo_normal
    else:
        var (tri_cd, fp_cd) = _pt_hit_footprint(path_ptr, mat, ctx.cam_fp, ctx.instances, mesh, v0, v1, v2, inter, ray_org, ray_dir, ctx.textures, ctx.n_textures)
        alb = _tex_lookup[use_gpu](mat, inter, v0, v1, v2, mesh, ctx.tex_filenames, ctx.textures, ctx.n_textures, fp_cd.width)
        # Use interpolated shading normal (geometric normal still drives hit-point offset)
        normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geo_normal, inter.primId.instanceIdx, ctx.instances)
        normal = _apply_surface_maps[use_gpu](mat, v0, v1, v2, mesh, inter, normal, geo_normal,
            tri_cd, fp_cd, ctx.tex_filenames, ctx.textures, ctx.n_textures)
        # pbrt's LayeredBxDF is twoSided: see face_toward.
        normal = face_toward(normal, -ray_dir)

    var hit_point = offset_point(ray_org + ray_dir * inter.tHit, geo_normal) + geo_normal * _curve_extra_offset(inter, ctx.curves, ctx.instances)
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
        path_ptr[].albedo = alb

    var ior = mat.emission.r            # coat IOR (eta_coat/eta_air), set at parse
    var coat_alpha = max(mat.roughU, mat.roughV)
    var wo = Vec3f(-ray_dir[0], -ray_dir[1], -ray_dir[2])  # toward viewer
    if mat.emission.b > Float32(0.5):
        _shade_principled[enqueue_shadow, defer_only](path_ptr, ctx, mat, inter, alb, normal, geo_normal, hit_point, wo,
                                                      ior, coat_alpha, pcg)
        return
    # pbrt's LayeredBxDF through the BSDF. param forces the coat alpha: `mat` may be Filter-Glossy widened.
    var bsdf = BSDF(LobeKind.layered, True, False, normal, wo, alb, Int32(inter.primId.materialIndex), coat_alpha,
                    Float32(0), Int32(-1), Float32(0), Float32(0), True, False)
    _lobe_nee[enqueue_shadow, defer_only](path_ptr, ctx, bsdf, hit_point, pcg)
    var s = bsdf.sample[want_rev=False](pcg.next_float(), pcg.next_float(), pcg.next_float(), Float32(0),
                                        LobeTables(ctx.materials, ctx.curves, ctx.measured_brdfs, ctx.spectral), path_ptr[].wavelengths)
    _lobe_bounce[False](path_ptr, ctx, s, hit_point, geo_normal, alb, pcg, diffuse=True)


# ── Dielectric (glass) branch ─────────────────────────────────────────────────
@always_inline
def shade_dielectric[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    ctx: ShadeContext,
    mat: Material,
):
    var (ok, is_sphere, geom_normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(path_ptr, inter, ctx.meshes, ctx.lights.spheres, instances=ctx.instances, curves=ctx.curves)
    if not ok:
        path_ptr[].active = 0
        return
    if not is_sphere:
        # The winding normal, NOT face-forwarded to the ray. `_hit_geom` (via
        # _geom_normal_and_ray) flips its geometric normal to oppose the
        # incoming ray, which is right for diffuse/conductor shading and fatal
        # here: `bxdf_sample_dielectric` decides entering-vs-exiting with
        # `dot(ray_dir, geom_normal) < 0`, and against a pre-flipped normal
        # that test is tautologically TRUE. Every crossing then reads as an
        # entry, so eta = 1/ior is applied on the way in AND on the way out
        # instead of 1/ior then ior -- a full traversal loses
        # (1/ior^2)*(1/ior^2) = 1/ior^4 of the transmitted radiance instead of
        # the correct factor of exactly 1. Measured on a backlit parallel slab
        # against the closed form L*(1-R)/(1+R): 0.679x at eta=1.1, 0.321x at
        # 1.33, 0.204x at 1.5, 0.061x at 2.0 -- tracking 1/eta^4 (0.683,
        # 0.320, 0.198, 0.063) across the whole range.
        #
        # This is the SAME defect the sphere branch below already documents
        # and fixes; it was believed not to affect meshes because
        # `_shading_normal` re-orients to the winding, but that orients to the
        # geo_normal it is HANDED, which arrives already flipped -- so the
        # flip survived. Recompute the raw winding normal from the vertices
        # (cross(p1-p0, p2-p0), the same expression _geom_normal_and_ray uses
        # before its flip) and orient the interpolated shading normal to THAT.
        # ReverseOrientation is already baked into the winding at parse time,
        # so this stays the authoritative outside direction.
        var dp0 = mesh_pos(mesh, v0)
        var dp1 = mesh_pos(mesh, v1)
        var dp2 = mesh_pos(mesh, v2)
        var raw_gn = cross(dp1 - dp0, dp2 - dp0)
        if inter.primId.instanceIdx >= Int32(0):
            raw_gn = transform_normal(Mat4(ctx.instances[unsafe_offset=Int(inter.primId.instanceIdx)].world_to_obj()), raw_gn)
        var raw_len = dot(raw_gn, raw_gn)
        if raw_len > Float32(0.0):
            raw_gn = raw_gn * (Float32(1.0) / sqrt(raw_len))
        else:
            raw_gn = geom_normal
        # pbrt's outward normal: with per-vertex normals, N itself, and the
        # winding normal flipped to AGREE with N (FaceForward(n, ns) in
        # Triangle::InteractionFromIntersection) -- not N flipped toward the
        # winding. A mesh wound inward but with outward N (lte-orb's glass
        # shell) otherwise reads every entry as an exit.
        var ns_out = _interp_vertex_normal(mesh, v0, v1, v2, inter.u, inter.v, inter.primId.instanceIdx, ctx.instances)
        if dot(ns_out, ns_out) > Float32(0.0):
            if dot(raw_gn, ns_out) < Float32(0.0):
                raw_gn = -raw_gn
            geom_normal = ns_out
        else:
            geom_normal = raw_gn
        # barcelona-pavilion's water is a `dielectric` carrying
        # "texture displacement" -- before this, that map was silently ignored
        # (measured: high-frequency structure in the water was identical with
        # the displacement present and stripped, ratio 0.999). Orient to
        # `raw_gn`, the WINDING normal, NOT to a ray-face-forwarded one: the
        # entering/exiting test below is `dot(ray_dir, n) < 0`, so flipping the
        # perturbed normal toward the ray would make it tautologically true and
        # bring back the 1/eta^4 loss this branch exists to prevent.
        var (tri_de, fp_de) = _pt_hit_footprint(path_ptr, mat, ctx.cam_fp, ctx.instances, mesh, v0, v1, v2, inter, ray_org, ray_dir, ctx.textures, ctx.n_textures)
        geom_normal = _apply_surface_maps[use_gpu](mat, v0, v1, v2, mesh, inter, geom_normal, raw_gn,
            tri_de, fp_de, ctx.tex_filenames, ctx.textures, ctx.n_textures)
    else:
        # `_hit_geom`/`_sphere_geom_normal_and_ray` returns a FACE-FORWARDED
        # normal (always flipped to oppose the incoming ray) -- correct for
        # diffuse/conductor shading, but it destroys the entering/exiting
        # distinction dielectric needs: `bxdf_sample_dielectric`'s `facing`
        # check against a normal that's ALWAYS pre-flipped toward the ray is
        # tautologically always true, so every hit (both the true entry AND
        # the true exit on the far side of a solid sphere) got read as an
        # "entering" event -- eta=1/ior applied TWICE instead of 1/ior then
        # ior, losing (1/ior)^2/ior^2 = 1/ior^4 of the transmitted radiance
        # for a full traversal instead of the correct factor of 1. Recompute
        # the TRUE, un-flipped outward normal (mirrors the mesh branch's own
        # "dielectric needs the raw pre-faceforward normal" rule, see the
        # GeomContext-builders comment a few hundred lines down) before any
        # normal-map perturbation, which itself also needs the true outward
        # direction as its base frame.
        var sph = ctx.lights.spheres[unsafe_offset=Int(inter.primId.id1)]
        var center = Vec3f(sph.center.x, sph.center.y, sph.center.z)
        var hit_point_raw = ray_org + ray_dir * inter.tHit
        var true_normal = hit_point_raw - center
        var tn_len2 = dot(true_normal, true_normal)
        if tn_len2 > Float32(0.0):
            true_normal = true_normal * (Float32(1.0) / sqrt(tn_len2))
        geom_normal = true_normal
        if mat.normal_tex_idx >= Int32(0):
            # Mitsuba "normalmap" wrapping a dielectric (e.g. the SMS paper's
            # own sphere_sms.xml bumpy glass sphere): perturb the shading
            # normal used for refraction itself, not just NEE shading --
            # this is what turns a smooth-lens refraction into the swirly
            # caustic pattern the reference renderer shows. See
            # _apply_normal_map_sphere's own docstring for the analytic
            # sphere UV parameterization this needs (Sphere has none
            # built in).
            geom_normal = _apply_normal_map_sphere[use_gpu](mat, center, hit_point_raw, geom_normal, ctx.tex_filenames, ctx.textures, ctx.n_textures)

    var ior = mat.albedo.r
    if mat.is_rough_dielectric() and mat.sss_boundary == Int8(0):
        # A real BSDF, not a delta: pbrt's rough DielectricBxDF, with NEE.
        _shade_rough_dielectric[enqueue_shadow](path_ptr, ctx, mat, Int(inter.primId.materialIndex), geom_normal, ray_dir,
                                                ray_org, inter.tHit, ior)
        return
    # Entering vs exiting is the mesh's own orientation, as in pbrt (ReverseOrientation negates the
    # normals at parse time); a pane whose normal faces away from the camera totally reflects.
    var force_entering = False

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var (bs, normal, new_dielectric_ior, new_previous_dielectric_ior) = bxdf_sample_dielectric(
        geom_normal, ray_dir, ior, force_entering, _draw_sobol_8(path_ptr, ctx.sobol_matrices).scat1,
        path_ptr[].current_dielectric_ior, path_ptr[].previous_dielectric_ior)
    var is_reflect = (Int(bs.flags) & Int(BxDFFlags.reflect)) != 0
    # ── Spatially varying subsurface reflectance ──────────────────────────
    # The interior is ONE homogeneous medium, so a textured `reflectance` has
    # to be collapsed to a mean to build it (Material.sss_mean_refl). But
    # the diffuse reflectance a given point should show is its OWN texel, and
    # with the mean alone the head renders flat -- measured 23% less spatial
    # detail and 36% less red-channel hue variation than pbrt, with the
    # eyebrows, lips and skin variation all collapsed to one tone.
    #
    # The inversion in material_builder solves alpha so the walk REPRODUCES
    # the reflectance it was given, i.e. R(alpha(A)) = A. So a point whose
    # texel is A, rendered through a medium built for Abar, is off by exactly
    # A/Abar to first order -- apply that on the way IN, once per BSSRDF
    # event. (`_tex_lookup` already folds this material's whole scale/mix
    # texture graph, so this picks up nested graphs for free.)
    if mat.sss_boundary != Int8(0) and mat.tex_idx >= Int32(0) and not is_reflect:
        var entering = force_entering or dot(ray_dir, geom_normal) < Float32(0.0)
        if entering:
            var a_pt = _tex_lookup[use_gpu](mat, inter, v0, v1, v2, mesh,
                                            ctx.tex_filenames, ctx.textures, ctx.n_textures)
            var mr = mat.sss_mean_refl
            # Guard a degenerate mean, and cap the correction: a texel far
            # above the mean must brighten its patch, never manufacture an
            # unbounded throughput spike the way an unclamped ratio would.
            var kr = min(a_pt.r / mr.r, Float32(4.0)) if mr.r > Float32(1e-4) else Float32(1.0)
            var kg = min(a_pt.g / mr.g, Float32(4.0)) if mr.g > Float32(1e-4) else Float32(1.0)
            var kb = min(a_pt.b / mr.b, Float32(4.0)) if mr.b > Float32(1e-4) else Float32(1.0)
            # Band-picked rather than smoothly upsampled: the spectral table
            # is not threaded into the dielectric kernel, and unlike the
            # sigma_s ratio that had to be made consistent with sigma_t (see
            # medium_sigma_s_spectral), this tint is applied ONCE per BSSRDF
            # event -- only on the way in, since TIR events are reflections --
            # so its error is bounded instead of compounding over hundreds of
            # scatters.
            path_ptr[].throughput *= rgb_bands_to_spectral_sample(
                max(kr, Float32(0.0)), max(kg, Float32(0.0)), max(kb, Float32(0.0)),
                path_ptr[].wavelengths)
    if mat.sss_walk_interior() and not is_reflect and inter.primId.type == Int8(0) and \
       (force_entering or dot(ray_dir, geom_normal) < Float32(0.0)):
        var wi_lobe = sss_entry_lobe(face_toward(geom_normal, bs.wi), mat.albedo.r, pcg)
        _sss_walk_enter_interior(path_ptr, mat, inter, ray_org + ray_dir * inter.tHit, geom_normal, wi_lobe)
        path_ptr[].pcgState = pcg.state
        return
    if not is_reflect:
        # bxdf_sample_dielectric's radiance_transmit factor is eta*eta, and
        # eta (its internal entering/exiting-relative ratio) always equals
        # old_current_ior/new_current_ior here -- entering: eta =
        # current_ior/ior = old/new (new_dielectric_ior becomes `ior`);
        # exiting: eta = ior/previous_ior = old/new too, since old current_ior
        # always equals this surface's own `ior` on an exit (it was set to
        # `ior` by the matching entry) and new_dielectric_ior becomes the OLD
        # previous_ior. So this reconstructs eta from the ior transition
        # alone in both cases, no extra return value needed. See
        # PathState.eta_scale's docstring for why RR needs this decoupled
        # from the real throughput.
        # new/current, NOT current/new: this has to UNDO the eta^2 the
        # throughput just took, so it is the RECIPROCAL of that factor
        # (pbrt's `etaScale *= Sqr(bs->eta)` with bs->eta = eta_t/eta_i,
        # against an f that was divided by the same square). Written the
        # other way up it compounded the compression instead: inside glass
        # RR read 0.356 * 0.356 = 0.127 for ior 1.675 and killed ~87% of the
        # paths that were mid-transit, which is why dambreak0's glass tank
        # and transparent-machines rendered as flat opaque slabs under the
        # path tracer while VCM/SPPM -- which do not run this RR -- showed
        # the glass. It cancels over a complete enter+exit pair, so only
        # paths still INSIDE a dielectric when RR fires were affected.
        var eta_for_rr = new_dielectric_ior / path_ptr[].current_dielectric_ior
        path_ptr[].eta_scale *= eta_for_rr * eta_for_rr
    path_ptr[].current_dielectric_ior = new_dielectric_ior
    path_ptr[].previous_dielectric_ior = new_previous_dielectric_ior
    var hit_point = offset_point(ray_org + ray_dir * inter.tHit, normal if is_reflect else -normal)
    _finish_delta_bounce(path_ptr, ctx.cycles, pcg, bs, SpectralSample(bs.f.r), hit_point, RGB(Float32(1)),
                         mat.sss_boundary == Int8(0))


# Rough dielectric: pbrt's DielectricBxDF over a Trowbridge-Reitz
# microsurface (layered.mojo's diel_*, the same functions LayeredBxDF's coat
# uses). Unlike smooth glass it is a real BSDF with a density, so it gets NEE
# and MIS like any glossy surface. It used to render as smooth glass: bistro
# and kroken ask for roughness 0.25, lte-orb for frosted glass, and all of it
# came out clear.
@always_inline
def _shade_rough_dielectric[enqueue_shadow: Bool](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    mat: Material,
    mat_idx: Int,
    n: Vec3f,          # the OUTWARD normal (not face-forwarded): inside/outside is wo's side of it
    ray_dir: Vec3f,
    ray_org: Vec3f,
    t_hit: Float32,
    ior: Float32,
):
    var hit_raw = ray_org + ray_dir * t_hit
    var wo = -ray_dir
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    # param is `mat`'s alpha: Filter Glossy may have widened it. Shadow and scatter rays leave to their side of n.
    var bsdf = BSDF(LobeKind.rough_dielectric, True, False, n, wo, RGB(Float32(1)), Int32(mat_idx),
                    max(mat.roughU, mat.roughV), Float32(0), Int32(-1), Float32(0), Float32(0), False, False)
    var ss = _draw_sobol_8(path_ptr, ctx.sobol_matrices)
    var ls_area = _sample_area_light_nee_u(ctx, hit_raw, ss.light, ss.bary1, ss.bary2, pcg)
    _lobe_nee_with[BSDF, enqueue_shadow](path_ptr, ctx, bsdf, hit_raw, ls_area, Point2f(ss.env1, ss.env2), True, pcg, n)
    var s = bsdf.sample[want_rev=False](ss.rr, ss.scat1, ss.scat2, Float32(0),
                                        LobeTables(ctx.materials, ctx.curves, ctx.measured_brdfs, ctx.spectral), path_ptr[].wavelengths)
    if s.valid and s.transmit:
        # The radiance f carried 1/etap^2, which eta_scale undoes for Russian roulette (pbrt's etaScale); the
        # touching-dielectric IOR stack pushes on the way in, pops on the way out.
        var entering = dot(wo, n) > Float32(0.0)
        var etap = ior if entering else Float32(1.0) / ior
        path_ptr[].eta_scale *= etap * etap
        if entering:
            path_ptr[].previous_dielectric_ior = path_ptr[].current_dielectric_ior
            path_ptr[].current_dielectric_ior = ior
        else:
            path_ptr[].current_dielectric_ior = path_ptr[].previous_dielectric_ior
            path_ptr[].previous_dielectric_ior = Float32(1.0)
    _lobe_bounce[False](path_ptr, ctx, s, hit_raw, n, RGB(Float32(1)), pcg, offset_eps(hit_raw.x, hit_raw.y, hit_raw.z))


# Thin dielectric (type 9): one-sided glass — Fresnel selects reflect or transmit,
# but transmitted ray is NOT refracted (direction unchanged). Models window glass,
# soap bubbles, thin films. IOR stored in mat.albedo.r like regular dielectric.
@always_inline
def shade_thin_dielectric(
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    mat: Material,
    spheres: Pointer[Sphere, MutUntrackedOrigin],
    cycles: SIMD[DType.int32, 4],
):
    var (ok, _, geom_normal, ray_dir, ray_org, _, _, _, _) = _hit_geom(path_ptr, inter, meshes, spheres)
    if not ok:
        path_ptr[].active = 0
        return

    var ior = mat.albedo.r

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var (bs, normal) = bxdf_sample_thin_dielectric(geom_normal, ray_dir, ior, pcg.next_float())

    var is_reflect = (Int(bs.flags) & Int(BxDFFlags.reflect)) != 0
    var hit_point = offset_point(ray_org + ray_dir * inter.tHit, normal if is_reflect else -normal)
    _finish_delta_bounce(path_ptr, cycles, pcg, bs, SpectralSample(bs.f.r), hit_point, RGB(Float32(1)))


# ── Conductor (mirror + GGX microfacet) branch ────────────────────────────────


@always_inline
def shade_conductor[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    ctx: ShadeContext,
    mat: Material,
):
    # Texture-driven roughness ("texture roughness"/"uroughness" — see
    # material_builder.mojo): resolve to a local mutable copy of `mat` here,
    # once, before anything reads roughU/V below, rather than threading a
    # texture lookup through every downstream GGX-alpha call site. Isotropic
    # only (see Material.rough_tex_idx's docstring); falls back to the
    # parsed scalar roughU/V when there's no texture or no UVs.
    var mat_eff = mat
    var (ok, is_sphere, geo_normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(path_ptr, inter, ctx.meshes, ctx.lights.spheres, instances=ctx.instances, curves=ctx.curves)
    if not ok:
        path_ptr[].active = 0
        return

    var normal = geo_normal
    if not is_sphere:
        if mat_eff.rough_tex_idx >= Int32(0) and mesh_has_uvs(mesh):
            var bw0 = Float32(1.0) - inter.u - inter.v
            var uv_u = bw0*mesh_uv(mesh, v0)[0] + inter.u*mesh_uv(mesh, v1)[0] + inter.v*mesh_uv(mesh, v2)[0]
            var uv_v = bw0*mesh_uv(mesh, v0)[1] + inter.u*mesh_uv(mesh, v1)[1] + inter.v*mesh_uv(mesh, v2)[1]
            var found = False
            var rtex = sample_texture[use_gpu](Int(mat_eff.rough_tex_idx), uv_u, uv_v, True, Float32(0.0),
                ctx.tex_filenames, ctx.textures, ctx.n_textures, found)
            if found:
                # Perceptual roughness -> GGX alpha, the default remaproughness=true.
                var r = sqrt(max(rtex.luma(), Float32(0.0)))
                mat_eff.roughU = r
                mat_eff.roughV = r
        normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geo_normal)
        var (tri_co, fp_co) = _pt_hit_footprint(path_ptr, mat, ctx.cam_fp, ctx.instances, mesh, v0, v1, v2, inter, ray_org, ray_dir, ctx.textures, ctx.n_textures)
        normal = _apply_surface_maps[use_gpu](mat_eff, v0, v1, v2, mesh, inter, normal, geo_normal,
            tri_co, fp_co, ctx.tex_filenames, ctx.textures, ctx.n_textures)
        normal = face_toward(normal, -ray_dir)

    var hit_point = offset_point(ray_org + ray_dir * inter.tHit, geo_normal)
    var wo = Vec3f(-ray_dir[0], -ray_dir[1], -ray_dir[2])
    var alpha = max(max(mat_eff.roughU, mat_eff.roughV), Float32(0.0001))
    var coated = mat.type == MatKind.coated_conductor   # pbrt's layered coat over this metal
    var bsdf = BSDF(LobeKind.layered if coated else LobeKind.ggx, True, not coated and alpha <= Float32(0.001), normal, wo,
                    mat_eff.albedo, Int32(inter.primId.materialIndex) if coated else Int32(-1), Float32(0) if coated else alpha,
                    Float32(0), Int32(-1), Float32(0), Float32(0), True, False)
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    if not bsdf.is_delta:
        _lobe_nee[enqueue_shadow](path_ptr, ctx, bsdf, hit_point, pcg)
    var s = bsdf.sample(pcg.next_float(), pcg.next_float(), pcg.next_float(), Float32(0),
                        LobeTables(ctx.materials, ctx.curves, ctx.measured_brdfs, ctx.spectral), path_ptr[].wavelengths)
    _lobe_bounce[False](path_ptr, ctx, s, hit_point, geo_normal, mat_eff.albedo, pcg)


@always_inline
def _lobe_bounce[aov_from_f: Bool](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    s: LobeSample,
    hit_point: Vec3f,
    geo_n: Vec3f,
    albedo: RGB,
    mut pcg: PCG32,
    side_eps: Float32 = Float32(0),
    diffuse: Bool = False,
):
    """Continue a path along a BSDF sample; `aov_from_f` takes the albedo AOV from f instead of `albedo`.
    A curve or rough-glass hit passes its offset `side_eps`: the ray leaves along geo_n to whichever side wi points."""
    if not s.valid:
        path_ptr[].active = 0
        path_ptr[].pcgState = pcg.state
        return
    var org = spawn_origin(hit_point, geo_n, s.wi)
    if side_eps > Float32(0):
        org = hit_point + geo_n * (side_eps if dot(s.wi, geo_n) >= Float32(0) else -side_eps)
    path_ptr[].ray = Ray(Point3f(org[0], org[1], org[2]), Vec3f(s.wi[0], s.wi[1], s.wi[2]))
    if path_ptr[].bounce == 0 or path_ptr[].specularBounce == Int8(1):
        comptime if aov_from_f:
            var (ar, ag, ab) = spectral_sample_to_rgb(
                ctx.spectral.coeffs, ctx.spectral.res, ctx.spectral.cie_x,
                ctx.spectral.cie_y, ctx.spectral.cie_z, ctx.spectral.d65,
                s.weight * (s.pdf_fwd / s.cos_out), path_ptr[].wavelengths)
            path_ptr[].albedo = RGB(max(ar, Float32(0.0)), max(ag, Float32(0.0)), max(ab, Float32(0.0)))
        else:
            path_ptr[].albedo = albedo
    path_ptr[].throughput *= s.weight
    if s.is_delta:
        path_ptr[].specularBounce = Int8(1)
        path_ptr[].lastBsdfPdf = Float32(0.0)
        path_ptr[].ns_spec = min(path_ptr[].ns_spec + Int8(1), Int8(100))
    else:
        path_ptr[].specularBounce = Int8(0)
        path_ptr[].lastBsdfPdf = s.pdf_fwd
    path_ptr[].bounce += 1
    var u_rr = pcg.next_float()
    # Cycles' bounce kinds: a delta event is glossy, a non-delta one diffuse when the caller says so.
    var kind = BOUNCE_TRANSMIT if s.transmit else (BOUNCE_DIFFUSE if diffuse and not s.is_delta else BOUNCE_GLOSSY)
    _apply_russian_roulette(path_ptr, ctx.cycles, pcg, u_rr, kind)


# ── Measured (tabulated Dupuy & Jakob BxDF) ──────────────────────────────────
# The real MeasuredBxDF, not an approximation — see measured_bsdf.mojo's
# loader + measured_bxdf_eval.mojo's f/Sample_f/PDF port. Always glossy
# reflection (never delta, never transmissive), isotropic in practice (see
# the loader's isotropic-only scope note), so an arbitrary Frisvad tangent
# frame is fine — no UV alignment needed (measured materials aren't
# anisotropic in this scene corpus).
@always_inline
def shade_measured[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    ctx: ShadeContext,
    mat: Material,
):
    var (ok, is_sphere, geo_normal, ray_dir, ray_org, mesh, v0, v1, v2) = _hit_geom(path_ptr, inter, ctx.meshes, ctx.lights.spheres, instances=ctx.instances, curves=ctx.curves)
    if not ok:
        path_ptr[].active = 0
        return

    var normal: Vec3f
    var hit_normal = geo_normal

    if is_sphere:
        # Analytic sphere: exact normal, never tilted past wo.
        normal = geo_normal
    else:
        normal = _shading_normal(mesh, v0, v1, v2, inter.u, inter.v, geo_normal)
        var (tri_me, fp_me) = _pt_hit_footprint(path_ptr, mat, ctx.cam_fp, ctx.instances, mesh, v0, v1, v2, inter, ray_org, ray_dir, ctx.textures, ctx.n_textures)
        normal = _apply_surface_maps[use_gpu](mat, v0, v1, v2, mesh, inter, normal, geo_normal,
            tri_me, fp_me, ctx.tex_filenames, ctx.textures, ctx.n_textures)

        # A shading normal tilted past wo (a silhouette of a low-poly mesh, or
        # a bump map at a grazing view) used to fall back to geo_normal here,
        # on the claim that pbrt picks the side from the geometric normal. It
        # does not: pbrt-v4's MeasuredBxDF mirrors wo/wi into the upper
        # hemisphere of the SHADING frame -- see face_toward. The black
        # car-silhouette pixels that fallback fixed are fixed by this too:
        # wo is never below the frame.
        normal = face_toward(normal, -ray_dir)

    var hit_point = offset_point(ray_org + ray_dir * inter.tHit, hit_normal)
    var wo = Vec3f(-ray_dir[0], -ray_dir[1], -ray_dir[2])

    if mat.measured_idx < Int32(0):
        # Load failure fallback (see material_builder.mojo) -- render as flat
        # black rather than dereferencing a nonexistent measured_brdfs entry.
        path_ptr[].active = 0
        return

    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)

    # NEE runs unconditionally, BEFORE the BSDF-sample validity checks below
    # -- matches pbrt's own design, where direct lighting is independent of
    # whether BSDF sampling succeeds (MeasuredBxDF::f() only bails out on a
    # degenerate half-vector, not on "the self-sampled reflection direction
    # happened to land below the horizon"). The tabulated lobe frequently
    # samples a sub-horizon wi near grazing angles (e.g. the front of a car
    # body), and gating NEE on that sample's validity was discarding real,
    # correctly-lit direct illumination at exactly those pixels -- confirmed
    # against the pbrt reference showing they should stay lit, not black.
    var bsdf = BSDF(LobeKind.measured, True, False, normal, wo, RGB(Float32(0)), Int32(inter.primId.materialIndex), Float32(0),
                    Float32(0), Int32(-1), Float32(0), Float32(0), True, False)
    _lobe_nee[enqueue_shadow](path_ptr, ctx, bsdf, hit_point, pcg)

    var s = bsdf.sample(Float32(0), pcg.next_float(), pcg.next_float(), Float32(0),
                        LobeTables(ctx.materials, ctx.curves, ctx.measured_brdfs, ctx.spectral), path_ptr[].wavelengths)
    _lobe_bounce[True](path_ptr, ctx, s, hit_point, hit_normal, RGB(Float32(0)), pcg)


# ── Marschner/Chiang hair BSDF (3-lobe: R, TT, TRT) ─────────────────────────
def shade_hair[use_gpu: Bool, enqueue_shadow: Bool](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    ctx: ShadeContext,
    mat: Material,
):
    """Hair (Marschner R/TT/TRT) through its BSDF, with the Sobol dimensions for light and scatter."""
    if inter.primId.type != 5:
        path_ptr[].active = 0
        return
    var curve_idx = Int(inter.primId.id1)
    var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var ray_org = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var wo = -ray_dir
    var cf = curve_hit_frame(ctx.curves, curve_idx, inter.v, inter.u, wo)
    var hit_base = ray_org + ray_dir * inter.tHit
    var tab = LobeTables(ctx.materials, ctx.curves, ctx.measured_brdfs, ctx.spectral)
    var bsdf = HairBSDF.at(BSDF(LobeKind.hair, True, False, cf[3], wo, mat.albedo, Int32(inter.primId.materialIndex),
                                Float32(0), Float32(0), Int32(curve_idx), inter.u, inter.v, False, False), mat, tab)
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var ss = _draw_sobol_8(path_ptr, ctx.sobol_matrices)
    var ls_area = _sample_area_light_nee_u(ctx, hit_base, ss.light, ss.bary1, ss.bary2, pcg)
    _lobe_nee_with[HairBSDF, enqueue_shadow](path_ptr, ctx, bsdf, hit_base, ls_area, Point2f(ss.env1, ss.env2), True, pcg)
    var s = bsdf.sample[want_rev=False](ss.rr, ss.scat1, ss.scat2, pcg.next_float(), tab, path_ptr[].wavelengths)
    # Albedo AOV: the TT lobe's colour, the closest hair has to a diffuse one.
    _lobe_bounce[False](path_ptr, ctx, s, hit_base, cf[3], bsdf.hc.A1, pcg, curve_offset_eps(cf[4]))


def _shade_diffuse_nee[use_gpu: Bool, enqueue_shadow: Bool, mnee: Bool = True, defer_only: Bool = False](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    ctx: ShadeContext,
    normal: Vec3f,
    hit_point: Vec3f,
    alb: RGB,
    wo: Vec3f,
    u_light: Float32,
    u_bary1: Float32,
    u_bary2: Float32,
    u_env1: Float32,
    u_env2: Float32,
    mut pcg: PCG32,
    guide_write: GuideGrid = null_guide(),
    restir_io: ReservoirIO = reservoir_io_null(),
    pixel_idx: Int = -1,
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
):
    # NEE sampling asymmetry: area lights use CDF-weighted selection (one light per
    # bounce, weight = power), while infinite/env lights are ALL sampled every bounce.
    # Rationale: area lights are finite and numerous (N can be large), so stochastic
    # selection with MIS is necessary. Infinite lights are typically 1-2 env maps,
    # and their contribution is often dominant — sampling all of them every bounce
    # costs little and avoids the variance of single-sample env selection. At N>2
    # env lights this asymmetry would need revisiting.

    # ── Area light NEE ────────────────────────────────────────────────────────
    # Phase 2 (docs/A2_restir_migration_plan.md): ReSTIR DI's RIS reservoir
    # replaces plain one-sample NEE for area lights, primary bounce only
    # (bounce 0 -- matches every real ReSTIR DI implementation's scope; it's
    # a screen-space technique over the primary G-buffer, not a per-bounce
    # one). Deliberately does NOT get MNEE's glass-caustic probing that
    # _nee_area_lights below has -- restir_di.mojo's target function assumes
    # an unoccluded straight shadow ray, same simplifying assumption as
    # every other non-diffuse material's NEE in this file.
    #
    # ReSTIR SMS (Phase 6) is threaded into _nee_area_lights below, scoped
    # to bounce 0 ONLY (sms_io is only passed through when bounce==0) --
    # same screen-space-G-buffer reasoning as ReSTIR DI's own bounce-0
    # scope: sms_temporal_step's per-pixel reservoir reuse only makes sense
    # where hit_point is deterministic per pixel under a static camera,
    # true at the primary hit and NOT at deeper bounces (whose hit points
    # vary sample to sample even with a fixed camera). Known, documented
    # gap: when ctx.use_restir is ALSO active, di_temporal_step handles
    # bounce 0 instead of _nee_area_lights, so ReSTIR SMS's bounce-0 reuse
    # does not run in that combination yet -- same class of gap as MNEE's
    # own pre-existing "no glass probing under ReSTIR DI's bounce 0"
    # limitation noted above, not a new one. --sms-restir without --restir
    # is unaffected.
    var sms_io_this_bounce = sms_io if path_ptr[].bounce == Int32(0) else sms_reservoir_io_null()
    comptime if defer_only:
        _nee_area_lights[enqueue_shadow, mnee, True](path_ptr, ctx, normal, hit_point, alb, u_light, u_bary1, u_bary2, pcg, guide_write)
    else:
        if ctx.use_restir and path_ptr[].bounce == Int32(0):
            di_temporal_step(path_ptr, ctx, hit_point, normal, alb, pcg, restir_io, pixel_idx)
        else:
            _nee_area_lights[enqueue_shadow, mnee](path_ptr, ctx, normal, hit_point, alb, u_light, u_bary1, u_bary2, pcg, guide_write,
                sms_io=sms_io_this_bounce, pixel_idx=pixel_idx)

        # ── ReSTIR GI (Phase 4.1's remaining half) ───────────────────────────────
        # x1 is diffuse (this function only runs for diffuse hits) and bounce 0:
        # mark this path as awaiting a possible reconnection vertex at bounce 1,
        # carrying x1's own shading data forward (gi_target_pdf needs it later,
        # and it's out of scope by the time bounce 1's shade call runs). Real
        # only when ctx.gi_pending is a live buffer -- dangling on every call
        # site except shade_core_cpu_nee's, so this is a no-op everywhere else,
        # including every GPU kernel and every non-restir CPU render.
        if ctx.use_restir and path_ptr[].bounce == Int32(0) and _is_real_ptr(ctx.gi_pending):
            ctx.gi_pending[unsafe_offset=ctx.path_idx] = GIPendingX1(active=Int8(1), hit_point=hit_point, normal=normal, alb=alb, throughput=path_ptr[].throughput)
        elif (ctx.use_restir and path_ptr[].bounce == Int32(1) and _is_real_ptr(ctx.gi_pending)
              and ctx.gi_pending[unsafe_offset=ctx.path_idx].active == Int8(1)):
            # x2 is ALSO diffuse (same reasoning: this function only runs for
            # diffuse hits) -- exactly the scope this increment supports. Any
            # other bounce-1 material silently never reaches here at all (that
            # material's own shade_* function has no gi_pending-consuming logic),
            # so gi_pending[tid].active is simply left at 1 and never read again
            # -- harmless, see GIPendingX1's own docstring.
            var snap = ctx.gi_pending[unsafe_offset=ctx.path_idx]
            ctx.gi_pending[unsafe_offset=ctx.path_idx].active = Int8(0)
            var raw = _gi_generate_recon_candidate(ctx, hit_point, normal, alb, pcg)
            if raw.valid != Int8(0):
                var w = gi_target_pdf(snap.hit_point, snap.normal, snap.alb, raw.recon_point, raw.recon_normal, raw.lo)
                _ = reservoir_update(raw.state, w, pcg.next_float())
            # Combine with history/neighbors (Phase 4.2 -- a no-op fallback to a
            # plain single-candidate finalize when ctx.gi_io/pixel_idx aren't
            # real, matching di_temporal_step's own null-safety contract) then
            # resolve: one shadow ray between x1 (snap's own data) and the
            # combined winner, injected into path_ptr[].estimate if visible.
            gi_temporal_spatial_combine(raw, snap.hit_point, snap.normal, snap.alb, pcg, ctx.gi_io, pixel_idx)
            gi_resolve(path_ptr, ctx, snap.hit_point, snap.normal, snap.alb, snap.throughput, raw)

    # ── Distant/point/sphere light NEE, via the shared Light interface
    # (bvh.mojo's LightSample samplers) + BxDF interface (bxdf.mojo's
    # _nee_weight_simple) — replacing 3 formerly hand-inlined blocks also
    # duplicated in _shade_conductor_nee/shade_hair. Infinite lights keep
    # their own dedicated _nee_infinite_light call below: its no-CDF
    # fallback deliberately uses cosine-weighted (not uniform-sphere)
    # sampling, a diffuse-specific optimization that must NOT be replaced
    # by the material-agnostic sampler. Area lights stay on _nee_area_lights
    # above (MNEE glass-refraction probing, also diffuse-specific).
    _nee_loop_simple[enqueue_shadow, defer_only](path_ptr, ctx, normal, hit_point, alb, Float32(0.0), Int32(0), wo, pcg, guide_write)

    # ── Infinite (env-map) light NEE ──────────────────────────────────────────
    for inf_i in range(ctx.lights.infinite_count):
        _nee_infinite_light[enqueue_shadow, defer_only](path_ptr, ctx, ctx.lights.infinite_lights[unsafe_offset=inf_i], normal, hit_point, alb, u_env1, u_env2, pcg, guide_write)


# Unified NEE core — comptime-specialized for CPU (use_gpu=False) and GPU (use_gpu=True).
# Texture lookup samples the in-memory GpuTexture table on both backends (OIIO only as the CPU's no-table fallback).
@always_inline
def shade_diffuse[use_gpu: Bool, enqueue_shadow: Bool, mnee: Bool = True, defer_only: Bool = False](
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
    ctx: ShadeContext,
    mat: Material,
    guide_write: GuideGrid = null_guide(),
    restir_io: ReservoirIO = reservoir_io_null(),
    pixel_idx: Int = -1,
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
):
    var (gc, ok) = _build_geom_context_full[use_gpu](path_ptr, inter, mat, ctx)
    if not ok:
        path_ptr[].active = 0
        return
    var normal = gc.normal   # already turned toward wo, see face_toward
    var hit_point = gc.hit_point
    var alb = gc.alb

    # Sampler design: diffuse uses Sobol for key visible decisions (light selection,
    # barycentrics, env direction, scatter, RR) and PCG for auxiliary draws that
    # don't benefit from low-discrepancy sequences (area-light triangle index,
    # sphere-light direction, infinite-light fallback, indirect-bounce scatter).
    # Specular materials (conductor, dielectric, coated_conductor, thin_dielectric)
    # use PCG only — their scatter decisions are near-deterministic at low roughness,
    # and they don't do NEE, so stratification yields negligible variance reduction.
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    # Read ahead of the NEE (whose shadow-ray traversal evicts the path's cache line); neither changes before the scatter below.
    var bounce_now = path_ptr[].bounce
    var specular_now = path_ptr[].specularBounce

    # Pre-draw 8 Z-Sobol samples for this bounce's key decisions.
    # Dims are consecutive starting at path_ptr[].sampler_dim (which begins at 2).
    # Per-dimension scrambling is derived from pcgInc (unique per path).
    var ss = _draw_sobol_8(path_ptr, ctx.sobol_matrices)
    var u_light = ss.light
    var u_bary1 = ss.bary1
    var u_bary2 = ss.bary2
    var u_env1  = ss.env1
    var u_env2  = ss.env2
    var u_scat1 = ss.scat1
    var u_scat2 = ss.scat2
    var u_rr    = ss.rr

    _shade_diffuse_nee[use_gpu, enqueue_shadow, mnee, defer_only](path_ptr, ctx, normal, hit_point, alb, gc.wo,
        u_light, u_bary1, u_bary2, u_env1, u_env2, pcg, guide_write, restir_io, pixel_idx, sms_io)

    # ── Scatter direction: 50/50 mixture of guide and cosine-weighted BSDF ──────
    # The guide is active when its energy pointer is a real allocation (Int > 1).
    # MIS balance heuristic: weight = f·cosθ / pdf_mix, where
    #   pdf_mix = 0.5·pdf_guide + 0.5·pdf_bsdf  and  f = alb/π (Lambertian).
    var dir: Vec3f
    var cos_theta: Float32
    var pdf_mix: Float32

    if guide_is_active(ctx.guide):
        var cell = guide_pos_to_cell(ctx.guide, Point3f(hit_point[0], hit_point[1], hit_point[2]))
        var bsdf_s = sample_cosine_hemisphere_world(u_scat1, u_scat2, normal)
        var bsdf_dir = bsdf_s[0]
        var pdf_b = bsdf_s[1]  # cos(theta)/π at bsdf_dir
        # Only apply 50/50 mixture when this cell has training data.
        # Empty cells return guide_pdf = 1/4π which is less than typical bsdf pdf,
        # inflating pdf_mix above pdf_b and increasing variance for no benefit.
        var has_guide = guide_cell_has_data(ctx.guide, cell)
        # α=0.25: 25% guide samples, 75% BSDF. MIS weights: α·pg + (1-α)·pb.
        # Low α limits the variance increase when the guide distribution is wrong.
        comptime GUIDE_ALPHA = Float32(0.25)
        comptime GUIDE_BETA  = Float32(0.75)
        if has_guide and pcg.next_float() < GUIDE_ALPHA:
            # Guide sample path
            var (gdx, gdy, gdz, pdf_g, guide_ok) = guide_sample(ctx.guide, cell, pcg.next_float(), pcg.next_float())
            var cos_g = gdx*normal[0] + gdy*normal[1] + gdz*normal[2]
            if guide_ok and cos_g > Float32(0.001):
                dir = Vec3f(gdx, gdy, gdz)
                cos_theta = cos_g
                pdf_mix = GUIDE_ALPHA*pdf_g + GUIDE_BETA*(cos_g / PI)
            else:
                # The guide drew a direction below the surface: that sample contributes nothing, so the path ends
                # here (weight 0). Falling back to a BSDF sample instead (as this branch used to) made the real
                # sampling density larger than pdf_mix and brightened the image by ~4% on the Cornell box.
                dir = bsdf_dir
                cos_theta = Float32(0)
                pdf_mix = Float32(1)
        elif has_guide:
            # BSDF sample with guide MIS correction
            dir = bsdf_dir
            cos_theta = pdf_b * PI
            var pg = guide_pdf(ctx.guide, cell, bsdf_dir[0], bsdf_dir[1], bsdf_dir[2])
            pdf_mix = GUIDE_ALPHA*pg + GUIDE_BETA*pdf_b
        else:
            # Cell outside AABB or empty — pure BSDF
            dir = bsdf_dir
            cos_theta = pdf_b * PI
            pdf_mix = pdf_b
    else:
        var bsdf_s = sample_cosine_hemisphere_world(u_scat1, u_scat2, normal)
        dir = bsdf_s[0]
        cos_theta = bsdf_s[1] * PI  # cos_theta = pdf * π
        pdf_mix = bsdf_s[1]

    var org_d = spawn_origin(hit_point, gc.geo_normal, dir)
    path_ptr[].ray = Ray(Point3f(org_d[0], org_d[1], org_d[2]), Vec3f(dir[0], dir[1], dir[2]))
    if bounce_now == 0 or specular_now == Int8(1):
        path_ptr[].albedo = alb
    # Store mixture PDF for next-bounce MIS (area light hit, env light miss)
    path_ptr[].lastBsdfPdf = pdf_mix
    # `diffuse` is the ONE material whose env NEE samples cosine-weighted
    # (_nee_infinite_light's no-CDF fallback, a deliberate diffuse-only
    # optimization), so its MIS partner for an escape is cos(theta)/pi --
    # NOT pdf_mix, which also carries the guide lobe, and not the
    # INV_FOUR_PI the dispatch defaulted it to for every other material.
    path_ptr[].lastEnvNeePdf = cos_theta * INV_PI
    path_ptr[].specularBounce = Int8(0)
    # Weight = f·cosθ / pdf_mix = (alb/π)·cosθ / pdf_mix.
    # MIS balance heuristic is unbiased — no cap needed here.
    var _w = cos_theta / (PI * pdf_mix)
    # Throughput and bounce stay in registers through the scatter and the
    # Russian roulette, and are stored once: the same arithmetic as updating
    # path_ptr[] in place and then re-reading it, minus the store-to-load
    # round trip through global memory that the roulette used to wait on.
    var thr = path_ptr[].throughput
    thr *= _to_spec_refl(ctx, alb, path_ptr[].wavelengths) * _w
    var bounce_next = bounce_now + Int32(1)
    path_ptr[].bounce = bounce_next
    _count_bounce(path_ptr, ctx.cycles, BOUNCE_DIFFUSE)
    if bounce_next > 1 and ctx.cycles[3] == 0:
        # See _apply_russian_roulette for why rr_lum carries eta_scale.
        var rr_lum = thr.luma() * path_ptr[].eta_scale
        var q = Float32(1.0) - (rr_lum if rr_lum < Float32(0.95) else Float32(0.95))
        if u_rr < q:
            path_ptr[].active = 0
        else:
            thr *= Float32(1.0) / (Float32(1.0) - q)
            var new_lum = thr.luma()
            if new_lum > RR_THROUGHPUT_CLAMP:
                thr *= RR_THROUGHPUT_CLAMP / new_lum
    path_ptr[].throughput = thr
    path_ptr[].pcgState = pcg.state


@always_inline
def shade_interface(
    path_ptr: Pointer[PathState, MutUntrackedOrigin],
    inter: Intersection,
):
    """Interface (null/passthrough) material: advance the ray through the surface.
    No scattering, no throughput change. Medium transition is handled externally:
    on CPU by rendering.mojo's medium-interface loop; on GPU by shade_interface_gpu.

    Deliberately does NOT touch specularBounce / lastBsdfPdf. A null interface
    is not a scattering event, so it must leave the MIS bookkeeping of the last
    REAL one intact. Setting specularBounce=1 here (as this used to) told the
    miss handler "no NEE competed for this direction", so a path that scattered
    in the medium and then crossed the bounding shell on its way out collected
    the environment at FULL weight while its NEE partner had already been
    correctly weighted 0.5 -- - 0.5 + 1.0 = 1.5x too much light. Measured
    exactly that: a conservative (albedo 1) medium in a uniform field, which
    must render at exactly 1.0, came out at 1.498. A camera ray still gets full
    weight on a genuine miss because the miss handler's own `bounce == 0` test
    covers that case, so nothing needs to be faked here."""
    var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var ray_org = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var hit_point = ray_org + ray_dir * inter.tHit + ray_dir * Float32(0.0002)
    # Crossing a null interface moves the ray ORIGIN without being a scattering
    # event, so remember how far it moved -- the emitter-hit MIS downstream
    # needs the distance back to the real scattering vertex, not to here.
    # See PathState.mis_null_dist.
    path_ptr[].mis_null_dist += inter.tHit + Float32(0.0002)
    path_ptr[].ray = Ray(Point3f(hit_point[0], hit_point[1], hit_point[2]), path_ptr[].ray.direction)
