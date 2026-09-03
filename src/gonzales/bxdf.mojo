from std.math import sqrt
from .geometry import RGB, MatKind, Material_C, Vec3f, dot, INV_PI, PI, fr_dielectric
from .sampling import sample_ggx_vndf, sample_cosine_hemisphere_world, power_heuristic
from .bvh import LightSample, HairLobeConstants, _hair_eval_lobes
from .spectrum import SampledWavelengths, SpectralSample, rgb_to_spectral_sample, rgb_illuminant_to_spectral_sample, spectral_sample_to_rgb

# ── Isotropic GGX (Trowbridge-Reitz) evaluation ───────────────────────────────
# Companion to sample_ggx_vndf: that function only *samples* a half-vector: NEE
# against a light-chosen direction needs to *evaluate* the distribution at an
# arbitrary half-vector, plus the matching VNDF sampling pdf, for MIS. All
# angles here are cosines against the shading normal (local-frame z, or
# equivalently a dot product in world space — the formulas only need cosines).
@always_inline
def ggx_D(cos_theta_h: Float32, alpha: Float32) -> Float32:
    """Trowbridge-Reitz microfacet distribution at a half-vector whose angle
    to the normal has cosine cos_theta_h."""
    var alpha2 = alpha * alpha
    var cos2 = cos_theta_h * cos_theta_h
    var denom = cos2 * (alpha2 - Float32(1.0)) + Float32(1.0)
    if denom <= Float32(0.0):
        return Float32(0.0)
    return alpha2 / (PI * denom * denom)

@always_inline
def ggx_lambda(cos_theta: Float32, alpha: Float32) -> Float32:
    """Smith Lambda(w) for a direction with cosine cos_theta to the normal."""
    if cos_theta <= Float32(0.0):
        return Float32(0.0)
    var cos2 = cos_theta * cos_theta
    var tan2 = max(Float32(0.0), Float32(1.0) - cos2) / cos2
    return (Float32(-1.0) + sqrt(Float32(1.0) + alpha * alpha * tan2)) * Float32(0.5)

@always_inline
def ggx_G1(cos_theta: Float32, alpha: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + ggx_lambda(cos_theta, alpha))

@always_inline
def ggx_G2(cos_o: Float32, cos_i: Float32, alpha: Float32) -> Float32:
    """Height-correlated Smith masking-shadowing for reflection."""
    return Float32(1.0) / (Float32(1.0) + ggx_lambda(cos_o, alpha) + ggx_lambda(cos_i, alpha))

@always_inline
def ggx_vndf_pdf(cos_o: Float32, cos_wm: Float32, d: Float32, alpha: Float32) -> Float32:
    """PDF (solid angle, over wi) of a reflection direction produced by
    reflecting a VNDF-sampled half-vector: pdf(wm)/(4|wo.wm|), with
    pdf(wm) = G1(wo)*D(wm)*max(0,wo.wm)/cos_o (Heitz 2018)."""
    if cos_o <= Float32(0.0) or cos_wm <= Float32(0.0):
        return Float32(0.0)
    var pdf_wm = ggx_G1(cos_o, alpha) * d * cos_wm / cos_o
    return pdf_wm / (Float32(4.0) * cos_wm)

@always_inline
def bxdf_eval_conductor_ggx(
    n:     Vec3f,
    wo:    Vec3f,
    wi:    Vec3f,
    alpha: Float32,
    f0:    RGB,
) -> RGB:
    """Isotropic GGX (Trowbridge-Reitz) conductor f_r(wo,wi) — the raw BRDF
    value for an arbitrary (not self-sampled) direction pair, NOT multiplied
    by any cosine. Used by sppm.mojo's photon-density gather (whose stored
    photon flux already encodes the appropriate cosine-weighted density) and
    NEE (whose caller applies its own cos_surface factor externally) — unlike
    bdpt.mojo's own connection-formula variant, which folds cos_i in for its
    path-throughput weighting. Schlick Fresnel at the half-vector,
    height-correlated Smith G2."""
    var (valid, k, schlick) = _ggx_conductor_shape_terms(n, wo, wi, alpha)
    if not valid:
        return RGB(Float32(0))
    var fr = f0 + (RGB(Float32(1)) - f0) * schlick
    return RGB(k * fr.r, k * fr.g, k * fr.b)

@always_inline
def _ggx_conductor_shape_terms(
    n:     Vec3f,
    wo:    Vec3f,
    wi:    Vec3f,
    alpha: Float32,
) -> Tuple[Bool, Float32, Float32]:
    """The wavelength-independent half of bxdf_eval_conductor_ggx's formula
    (microfacet D/G2 + the Schlick blend factor), shared by both the RGB
    evaluator above and the spectral one (bxdf_eval_any_spectral) so the two
    don't duplicate the GGX math itself, only how they combine it with a
    color (f0 directly for RGB, a spectrally-converted f0 for spectral).
    Returns (valid, k, schlick) where the full BRDF is
    k * (f0 + (1-f0)*schlick) for whichever color representation f0 is in."""
    var cos_o = dot(wo, n)
    var cos_i = dot(wi, n)
    if cos_o <= Float32(0) or cos_i <= Float32(0):
        return (False, Float32(0), Float32(0))
    var wh = wo + wi
    var whl = dot(wh, wh)
    if whl <= Float32(0):
        return (False, Float32(0), Float32(0))
    wh = wh * (Float32(1) / sqrt(whl))
    var cos_h = dot(wh, n)
    var cos_wo_h = dot(wo, wh)
    if cos_wo_h < Float32(0): cos_wo_h = -cos_wo_h
    var d = ggx_D(cos_h, alpha)
    var g = ggx_G2(cos_o, cos_i, alpha)
    var one_m = Float32(1) - cos_wo_h
    var one_m2 = one_m * one_m
    var schlick = one_m2 * one_m2 * one_m
    var k = d * g / (Float32(4) * cos_o * cos_i)
    return (True, k, schlick)

@always_inline
def bxdf_pdf_conductor_ggx(
    n:     Vec3f,
    wo:    Vec3f,
    wi:    Vec3f,
    alpha: Float32,
) -> Float32:
    """Solid-angle PDF of VNDF-sampled GGX reflection landing at an arbitrary
    wi (isotropic approximation — see bxdf_eval_conductor_ggx's own docstring
    for why this is a reasoned simplification for anisotropic materials).
    This is the competing-strategy density for MIS against NEE: it matches
    the sampling density bxdf_sample_conductor's glossy branch itself uses
    (Heitz 2018 VNDF sampling), just evaluated at a caller-chosen direction
    instead of the self-sampled one."""
    var cos_o = dot(wo, n)
    var cos_i = dot(wi, n)
    if cos_o <= Float32(0) or cos_i <= Float32(0):
        return Float32(0)
    var wh = wo + wi
    var whl = dot(wh, wh)
    if whl <= Float32(0):
        return Float32(0)
    wh = wh * (Float32(1) / sqrt(whl))
    var cos_wm = dot(wh, n)
    var d = ggx_D(cos_wm, alpha)
    return ggx_vndf_pdf(cos_o, cos_wm, d, alpha)

# ── BxDF flags ────────────────────────────────────────────────────────────────
struct BxDFFlags:
    comptime delta    = Int8(1)   # Dirac delta (perfect mirror / glass)
    comptime diffuse  = Int8(2)   # cosine lobe
    comptime glossy   = Int8(4)   # GGX microfacet lobe
    comptime reflect  = Int8(8)
    comptime transmit = Int8(16)

@always_inline
def bxdf_is_delta(flags: Int8) -> Bool:
    return (Int(flags) & Int(BxDFFlags.delta)) != 0

# ── BxDF sample result ────────────────────────────────────────────────────────
# Convention:
#   Delta BSDFs:     f = throughput multiplier, pdf = 1.0
#                    integrator does: throughput *= f  (no cos/pdf division)
#   Non-delta BSDFs: f = BxDF value f(wo,wi), pdf = sampling pdf
#                    integrator does: throughput *= f * cos_wi / pdf
@fieldwise_init
struct BxDFSample(TrivialRegisterPassable):
    var wi:       Vec3f  # sampled incident direction (world-space)
    var f:        RGB                      # BxDF value or throughput multiplier
    var pdf:      Float32                  # sampling PDF; 1.0 for delta BSDFs
    var flags:    Int8                     # BxDFFlags bitmask
    var is_valid: Int8                     # 0 = degenerate (TIR, back-face, etc.)
    var _pad0:    Int8
    var _pad1:    Int8

# ── Local geometry at a surface hit ──────────────────────────────────────────
# Populated once per bounce — _build_geom_context_full[use_gpu] for NEE materials
# (shading.mojo), or built inline where the delta BSDFs' geometry needs diverge
# from that shared builder (see the comment above shade_conductor's GeomContext
# construction). Passed by value to all BxDF and NEE functions — LLVM/NVPTX
# eliminates the struct when all fields are inlined at @always_inline call sites.
@fieldwise_init
struct GeomContext(TrivialRegisterPassable):
    var normal:     Vec3f  # shading normal, faceforward to geo_normal
    var geo_normal: Vec3f  # geometric normal, faceforward to wo
    var hit_point:  Vec3f  # world-space surface hit point
    var wo:         Vec3f  # outgoing direction = -ray.direction
    var tangent:    Vec3f  # shading tangent (Frisvad frame)
    var bitangent:  Vec3f  # shading bitangent
    var alb:        RGB                      # surface albedo (texture or mat.albedo)
    var pixel_uv:   Float32                  # mip LOD footprint (0 on CPU path)

# ── Pre-drawn Sobol samples for one bounce ────────────────────────────────────
# Drawn by _draw_sobol_8 at the start of each non-delta bounce.
# Named fields replace the current unnamed u_light, u_bary1, ... parameters.
@fieldwise_init
struct SobolSamples8(TrivialRegisterPassable):
    var light: Float32   # light CDF selection
    var bary1: Float32   # area light barycentric r1
    var bary2: Float32   # area light barycentric r2
    var env1:  Float32   # env-map u1 (BSDF-MIS cosine sample / CDF u)
    var env2:  Float32   # env-map u2 (BSDF-MIS cosine sample / CDF v)
    var scat1: Float32   # BSDF scatter u1
    var scat2: Float32   # BSDF scatter u2
    var rr:    Float32   # Russian roulette

# ── Conductor (mirror + GGX microfacet) ──────────────────────────────────────
# Perfect mirror when roughU/roughV ~ 0, else anisotropic GGX VNDF (Heitz 2018).
# F0 = mat.albedo, brightened to white at grazing via Schlick 5th-power.
# Anisotropy is aligned to gc.tangent/gc.bitangent — the caller is responsible
# for choosing a UV-gradient tangent frame there when the mesh has UVs and the
# material is anisotropic (GeomContext's default Frisvad frame is arbitrary
# and would rotate the highlight incorrectly otherwise).
@always_inline
def bxdf_sample_conductor(
    gc: GeomContext,
    mat: Material_C,
    u1: Float32, u2: Float32,
) -> BxDFSample:
    # roughU/V already hold the resolved GGX alpha (see _psc_handle_make_named_material's
    # remaproughness handling) — no squaring here.
    var alpha_x = max(mat.roughU, Float32(0.0001))
    var alpha_y = max(mat.roughV, Float32(0.0001))
    var is_rough = mat.roughU > Float32(0.001) or mat.roughV > Float32(0.001)
    var white = RGB(Float32(1.0))

    if not is_rough:
        var wi = gc.normal * (Float32(2.0) * dot(gc.wo, gc.normal)) - gc.wo
        var wlen = dot(wi, wi)
        if wlen > Float32(0.0):
            wi = wi * (Float32(1.0) / sqrt(wlen))
        var cos_i = max(Float32(0.0), dot(gc.wo, gc.normal))
        var one_m = Float32(1.0) - cos_i
        var schlick = one_m * one_m * one_m * one_m * one_m
        var fresnel_rgb = mat.albedo + (white - mat.albedo) * schlick
        return BxDFSample(wi, fresnel_rgb, Float32(1.0), BxDFFlags.delta | BxDFFlags.reflect, Int8(1), Int8(0), Int8(0))

    var wo_l = Vec3f(dot(gc.wo, gc.tangent), dot(gc.wo, gc.bitangent), dot(gc.wo, gc.normal))
    var wh_l = sample_ggx_vndf(wo_l, alpha_x, alpha_y, u1, u2)
    var wh = gc.tangent * wh_l.x + gc.bitangent * wh_l.y + gc.normal * wh_l.z
    var whlen = dot(wh, wh)
    if whlen > Float32(0.0):
        wh = wh * (Float32(1.0) / sqrt(whlen))
    var wo_dot_wh = dot(gc.wo, wh)
    var wi = wh * (Float32(2.0) * wo_dot_wh) - gc.wo
    var wilen = dot(wi, wi)
    if wilen > Float32(0.0):
        wi = wi * (Float32(1.0) / sqrt(wilen))
    if dot(wi, gc.normal) <= Float32(0.0):
        return BxDFSample(wi, RGB(Float32(0.0)), Float32(1.0), BxDFFlags.glossy | BxDFFlags.reflect, Int8(0), Int8(0), Int8(0))
    var cos_wh = max(Float32(0.0), wo_dot_wh)
    var one_m = Float32(1.0) - cos_wh
    var one_m2 = one_m * one_m
    var schlick = one_m2 * one_m2 * one_m
    var fresnel_rgb = mat.albedo + (white - mat.albedo) * schlick
    return BxDFSample(wi, fresnel_rgb, Float32(1.0), BxDFFlags.glossy | BxDFFlags.reflect, Int8(1), Int8(0), Int8(0))

# ── CoatedConductor: dielectric clearcoat over GGX conductor ─────────────────
# Schlick Fresnel at the air/coat interface selects: specular reflection off
# the coat, or the isotropic GGX conductor lobe beneath. Energy-conserving
# two-lobe approximation of pbrt's LayeredBxDF. Isotropic only (single alpha)
# so gc.tangent/gc.bitangent (Frisvad) need no UV alignment.
@always_inline
def bxdf_sample_coated_conductor(
    gc: GeomContext,
    mat: Material_C,
    ior: Float32,
    u_split: Float32, u1: Float32, u2: Float32,
) -> BxDFSample:
    var cos_theta = max(Float32(0.0), dot(gc.wo, gc.normal))
    var r0 = (ior - Float32(1.0)) / (ior + Float32(1.0))
    r0 = r0 * r0
    var one_m = Float32(1.0) - cos_theta
    var one_m2 = one_m * one_m
    var f_coat = r0 + (Float32(1.0) - r0) * one_m2 * one_m2 * one_m
    var white = RGB(Float32(1.0))

    if u_split < f_coat:
        var wi = gc.normal * (Float32(2.0) * dot(gc.wo, gc.normal)) - gc.wo
        var wlen = dot(wi, wi)
        if wlen > Float32(0.0):
            wi = wi * (Float32(1.0) / sqrt(wlen))
        return BxDFSample(wi, white, Float32(1.0), BxDFFlags.delta | BxDFFlags.reflect, Int8(1), Int8(0), Int8(0))

    # roughU/V already hold the resolved GGX alpha — no squaring here.
    var alpha_u = max(mat.roughU, Float32(0.0001))
    var alpha_v = max(mat.roughV, Float32(0.0001))
    var alpha = (alpha_u + alpha_v) * Float32(0.5)
    var wo_l = Vec3f(dot(gc.wo, gc.tangent), dot(gc.wo, gc.bitangent), dot(gc.wo, gc.normal))
    var wh_l = sample_ggx_vndf(wo_l, alpha, alpha, u1, u2)
    var wh = gc.tangent * wh_l.x + gc.bitangent * wh_l.y + gc.normal * wh_l.z
    var whlen = dot(wh, wh)
    if whlen > Float32(0.0):
        wh = wh * (Float32(1.0) / sqrt(whlen))
    var wo_dot_wh = dot(gc.wo, wh)
    var wi = wh * (Float32(2.0) * wo_dot_wh) - gc.wo
    var wilen = dot(wi, wi)
    if wilen > Float32(0.0):
        wi = wi * (Float32(1.0) / sqrt(wilen))
    if dot(wi, gc.normal) <= Float32(0.0):
        return BxDFSample(wi, RGB(Float32(0.0)), Float32(1.0), BxDFFlags.glossy | BxDFFlags.reflect, Int8(0), Int8(0), Int8(0))
    var cos_wh = max(Float32(0.0), wo_dot_wh)
    var one_m3 = Float32(1.0) - cos_wh
    var one_m4 = one_m3 * one_m3
    var schlick = one_m4 * one_m4 * one_m3
    var f0_luma = mat.albedo.luma()
    var f_metal = f0_luma + (Float32(1.0) - f0_luma) * schlick
    var tput = mat.albedo * f_metal * (Float32(1.0) - f_coat)
    return BxDFSample(wi, tput, Float32(1.0), BxDFFlags.glossy | BxDFFlags.reflect, Int8(1), Int8(0), Int8(0))

# ── Dielectric (smooth glass) ─────────────────────────────────────────────────
# Needs the RAW winding/shading normal (not faceforward to the ray) to tell
# entering from exiting — GeomContext's geo_normal is already faceforward to
# wo, so this takes geometry directly instead of a GeomContext. Returns the
# facing normal alongside the sample so the caller can offset hit_point
# (+normal for reflect, -normal for transmit; BxDFSample.flags says which).
@always_inline
def bxdf_sample_dielectric(
    geom_normal: Vec3f,
    ray_dir: Vec3f,
    ior: Float32,
    force_entering: Bool,   # bounce==0: trust physics (camera ray always from air)
    u_reflect: Float32,
) -> Tuple[BxDFSample, Vec3f]:
    var facing = dot(ray_dir, geom_normal) < Float32(0.0)
    var entering = facing or force_entering
    var normal = geom_normal if facing else -geom_normal
    var eta = (Float32(1.0) / ior) if entering else ior

    var cos_i = -dot(ray_dir, normal)
    var sin2_t = eta * eta * (Float32(1.0) - cos_i * cos_i)
    var tir = sin2_t > Float32(1.0)
    # eta here is η_i/η_t; fr_dielectric wants its reciprocal as the relative IOR.
    var fresnel = fr_dielectric(cos_i, Float32(1.0) / eta)
    var white = RGB(Float32(1.0))

    if tir or u_reflect < fresnel:
        var refl = ray_dir + normal * (Float32(2.0) * cos_i)
        var rlen = dot(refl, refl)
        if rlen > Float32(0.0):
            refl = refl * (Float32(1.0) / sqrt(rlen))
        return (BxDFSample(refl, white, Float32(1.0), BxDFFlags.delta | BxDFFlags.reflect, Int8(1), Int8(0), Int8(0)), normal)

    var cos_t = sqrt(Float32(1.0) - sin2_t)
    var refr = ray_dir * eta + normal * (eta * cos_i - cos_t)
    var rlen = dot(refr, refr)
    if rlen > Float32(0.0):
        refr = refr * (Float32(1.0) / sqrt(rlen))
    # Radiance-transport (camera-path) non-symmetry correction: transmitting
    # through a change of IOR compresses/expands solid angle, so radiance is
    # NOT conserved across the interface the way importance/flux is (PBRT
    # SpecularTransmission's `mode == Radiance` factor). This function is only
    # ever reached from camera-path contexts (the plain path tracer's wavefront
    # shading kernels), never from a light-emission/photon path, so the eta²
    # factor applies unconditionally here.
    #
    # BUG FIX (found via a real firefly repro, staircase2 scene): this was
    # `white / (eta * eta)` -- the RECIPROCAL of the correct factor. Veach's
    # radiance non-symmetry law says physical (forward, light-traced)
    # radiance scales by (eta_transmitted/eta_incident)^2 when crossing into
    # a medium of different IOR; a camera/importance path needs the INVERSE
    # of that forward scaling as its correction, i.e. (eta_incident/eta_
    # transmitted)^2 = eta*eta (since `eta` here IS eta_incident/eta_
    # transmitted, per the comment above). The bug was invisible for the
    # common case (a ray straight through one flat pane: one entering event
    # at eta, one exiting event at 1/eta, and the two -- wrong or right --
    # factors always cancel to ~1 either way). It only shows up as a real
    # error once a path's entering/exiting transmission events go
    # unbalanced (multiple glass surfaces, geometry with more one-directional
    # crossings than the other), where the inverted factor compounds
    # multiplicatively bounce over bounce -- observed as throughput
    # inflating past 1e11 by bounce ~30 on a maxdepth=65 scene with several
    # glass surfaces, producing extreme, denoiser-smeared fireflies.
    var radiance_transmit = white * (eta * eta)
    return (BxDFSample(refr, radiance_transmit, Float32(1.0), BxDFFlags.delta | BxDFFlags.transmit, Int8(1), Int8(0), Int8(0)), normal)

# ── Thin dielectric (one-sided glass slab: window, soap film) ────────────────
# Transmitted ray keeps its original direction (no refraction) — models a thin
# slab whose entry/exit refractions cancel. Fresnel compounded across both
# slab interfaces: R' = 2R/(1+R) (PBRT thin-glass formula).
@always_inline
def bxdf_sample_thin_dielectric(
    geom_normal: Vec3f,
    ray_dir: Vec3f,
    ior: Float32,
    u_reflect: Float32,
) -> Tuple[BxDFSample, Vec3f]:
    var entering = dot(ray_dir, geom_normal) < Float32(0.0)
    var normal = geom_normal if entering else -geom_normal
    var cos_i = max(Float32(0.0), -dot(ray_dir, normal))

    var r_single = fr_dielectric(cos_i, ior)
    var fresnel = r_single
    if r_single < Float32(1.0):
        fresnel = Float32(2.0) * r_single / (Float32(1.0) + r_single)
    var white = RGB(Float32(1.0))

    if u_reflect < fresnel:
        var refl = ray_dir + normal * (Float32(2.0) * cos_i)
        var rlen = dot(refl, refl)
        if rlen > Float32(0.0):
            refl = refl * (Float32(1.0) / sqrt(rlen))
        return (BxDFSample(refl, white, Float32(1.0), BxDFFlags.delta | BxDFFlags.reflect, Int8(1), Int8(0), Int8(0)), normal)

    return (BxDFSample(ray_dir, white, Float32(1.0), BxDFFlags.delta | BxDFFlags.transmit, Int8(1), Int8(0), Int8(0)), normal)

# ── Diffuse (Lambertian) ──────────────────────────────────────────────────────
@always_inline
def bxdf_eval_diffuse(alb: RGB) -> RGB:
    """f(wo,wi) = albedo/π, independent of direction (ideal Lambertian)."""
    return alb * INV_PI

@always_inline
def bxdf_pdf_diffuse(cos_wi: Float32) -> Float32:
    """Cosine-hemisphere sampling pdf = cos(θ)/π."""
    return max(Float32(0.0), cos_wi) * INV_PI

@always_inline
def bxdf_sample_diffuse(
    gc: GeomContext,
    alb: RGB,
    u1: Float32, u2: Float32,
) -> BxDFSample:
    var s = sample_cosine_hemisphere_world(u1, u2, gc.normal)
    var wi = s[0]
    var pdf = s[1]
    var is_valid = Int8(1) if pdf > Float32(0.0) else Int8(0)
    return BxDFSample(wi, alb * INV_PI, pdf, BxDFFlags.diffuse | BxDFFlags.reflect, is_valid, Int8(0), Int8(0))

# ── DiffuseTransmission ───────────────────────────────────────────────────────
# Two Lambertian lobes (reflect / transmit) selected stochastically by
# luminance. f bakes in the selection-weight compensation so the integrator
# convention (throughput *= f * cos_wi / pdf) reproduces lobe_alb * lobe_w
# exactly. NEE for this material stays in shading.mojo: it needs bounce_normal
# and lobe_alb per chosen lobe, which don't fit GeomContext's single normal —
# so this returns them alongside the sample instead of folding NEE in here.
@always_inline
def bxdf_sample_diffuse_transmit(
    normal: Vec3f,   # faceforward shading normal (to the ray)
    refl: RGB, trans: RGB,
    u_lobe: Float32, u1: Float32, u2: Float32,
) -> Tuple[BxDFSample, Vec3f, RGB, Float32, Bool]:
    var pr = refl.luma()
    var pt = trans.luma()
    var total = pr + pt
    if total <= Float32(0.0):
        var z = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
        return (BxDFSample(z, RGB(Float32(0.0)), Float32(0.0), BxDFFlags.diffuse, Int8(0), Int8(0), Int8(0)),
            normal, RGB(Float32(0.0)), Float32(0.0), True)

    var choose_reflect = u_lobe < pr / total
    var bounce_normal = normal if choose_reflect else -normal
    var lobe_alb = refl if choose_reflect else trans
    var lobe_w = total / (pr if choose_reflect else pt)

    var s = sample_cosine_hemisphere_world(u1, u2, bounce_normal)
    var wi = s[0]
    var pdf = s[1]
    var is_valid = Int8(1) if pdf > Float32(0.0) else Int8(0)
    var f = lobe_alb * INV_PI * lobe_w
    return (BxDFSample(wi, f, pdf, BxDFFlags.diffuse | BxDFFlags.transmit, is_valid, Int8(0), Int8(0)),
        bounce_normal, lobe_alb, lobe_w, choose_reflect)

# ── Generic BxDF interface (the "materials" half of the Light/BxDF interface
# refactor — see bvh.mojo's LightSample for the "lights" half) ───────────────
# One dispatch point mirroring sppm.mojo's own (now superseded) private
# _sppm_vp_brdf, promoted to a shared function so shading.mojo/bdpt.mojo/
# sppm.mojo can all evaluate "this material's raw f(wo,wi) + its matching
# sampling pdf" through a single call instead of each re-deriving the
# per-material formula inline. mat_kind: 0 = diffuse (default/else branch),
# 1 = conductor/coated_conductor (isotropic GGX approximation — same
# simplification bxdf_eval_conductor_ggx's own docstring already documents).
# Hair is NOT dispatched here: it needs a whole precomputed HairLobeConstants
# (built once per hit, not per light sample) rather than a flat (alb, alpha)
# pair — see bxdf_eval_any_hair below, which takes that struct directly.
@always_inline
def bxdf_eval_any(
    mat_kind: Int32,
    alb:      RGB,             # diffuse albedo, or conductor f0
    alpha:    Float32,         # conductor GGX roughness; unused for diffuse
    n:        Vec3f,
    wo:       Vec3f,
    wi:       Vec3f,
) -> Tuple[RGB, Float32]:
    if mat_kind == Int32(1):
        return (bxdf_eval_conductor_ggx(n, wo, wi, alpha, alb), bxdf_pdf_conductor_ggx(n, wo, wi, alpha))
    var cos_wi = dot(n, wi)
    return (bxdf_eval_diffuse(alb), bxdf_pdf_diffuse(cos_wi))

@always_inline
def _nee_weight_simple(
    ls:    LightSample,
    mat_kind: Int32,
    alb:   RGB,
    alpha: Float32,
    n:     Vec3f,
    wo:    Vec3f,
) -> RGB:
    """NEE contribution weight (throughput NOT yet applied — caller does
    `throughput * result`) for one LightSample against a diffuse or
    conductor/coated_conductor surface, via the generic bxdf_eval_any
    dispatch above. Delta lights (ls.is_delta) get MIS weight 1; real-pdf
    lights (sphere/infinite) are weighted via the power heuristic against
    this material's own sampling pdf at wi — exactly the formula every
    per-material NEE function in shading.mojo/bdpt.mojo/sppm.mojo already
    used, just written once instead of once per (material, light-type)
    pair."""
    if not ls.valid:
        return RGB(Float32(0.0))
    var cos_s = dot(n, ls.wi)
    if cos_s <= Float32(0.0):
        return RGB(Float32(0.0))
    var (f, pdf_bsdf) = bxdf_eval_any(mat_kind, alb, alpha, n, wo, ls.wi)
    if f.r <= Float32(0.0) and f.g <= Float32(0.0) and f.b <= Float32(0.0):
        return RGB(Float32(0.0))
    if ls.is_delta:
        return f * ls.Li * cos_s
    var mis_w = power_heuristic(ls.pdf, pdf_bsdf)
    return f * ls.Li * (cos_s * mis_w / ls.pdf)

@always_inline
def _nee_weight_coated_coat_lobe(
    ls:         LightSample,
    ior:        Float32,
    coat_alpha: Float32,
    n:          Vec3f,
    wo:         Vec3f,
) -> RGB:
    """NEE contribution weight (throughput NOT applied) for ONE LightSample
    against a coateddiffuse/coated_conductor coat's own glossy dielectric
    GGX reflection lobe (D*G2*F/(4*cos_o), same microfacet term
    shade_coated_diffuse's coat block used to hand-inline once per light
    type). Delta lights get MIS weight 1 (no competing BSDF-sampling
    strategy exists for a delta direction); real-pdf lights (sphere/
    infinite/area) use the power heuristic against this lobe's own
    ggx_vndf_pdf. Caller should skip calling this for a smooth coat
    (coat_alpha below the is_rough_coat threshold) — a delta reflection can
    never land on a stochastically-sampled light direction, so evaluating
    this would just waste the shadow-ray test."""
    if not ls.valid:
        return RGB(Float32(0.0))
    var cos_o = dot(wo, n)
    var cos_s = dot(n, ls.wi)
    if cos_o <= Float32(0.0) or cos_s <= Float32(0.0):
        return RGB(Float32(0.0))
    var wm = wo + ls.wi
    var wm_len = dot(wm, wm)
    if wm_len <= Float32(0.0):
        return RGB(Float32(0.0))
    wm = wm * (Float32(1.0) / sqrt(wm_len))
    var cos_wm = dot(wo, wm)
    if cos_wm <= Float32(0.0):
        return RGB(Float32(0.0))
    var d = ggx_D(dot(n, wm), coat_alpha)
    var g2 = ggx_G2(cos_o, cos_s, coat_alpha)
    var f = fr_dielectric(cos_wm, ior)
    var f_cos = d * g2 * f / (Float32(4.0) * cos_o)
    if ls.is_delta:
        return ls.Li * f_cos
    if ls.pdf <= Float32(0.0):
        return RGB(Float32(0.0))
    var pdf_bsdf = ggx_vndf_pdf(cos_o, cos_wm, d, coat_alpha)
    var w = power_heuristic(ls.pdf, pdf_bsdf)
    return ls.Li * (f_cos * w / ls.pdf)

@always_inline
def _nee_weight_coated_diffuse_base(
    ls:  LightSample,
    alb: RGB,
    ior: Float32,
    n:   Vec3f,
) -> RGB:
    """NEE contribution weight (throughput AND the walk's `beta` NOT
    applied — caller multiplies by both) for ONE LightSample against a
    coateddiffuse's Lambertian base layer, reached by transmitting through
    the coat (attenuated by 1 - Fresnel at the light's own incidence
    angle). Delta lights get MIS weight 1; real-pdf lights use the power
    heuristic against the base's own cosine-weighted pdf — same shape as
    _nee_weight_simple, kept separate because the coat's transmittance
    term and the caller's `beta` state don't fit bxdf_eval_any's flat
    (mat_kind, alb, alpha) signature."""
    if not ls.valid:
        return RGB(Float32(0.0))
    var cos_s = dot(n, ls.wi)
    if cos_s <= Float32(0.0):
        return RGB(Float32(0.0))
    var t_light = Float32(1.0) - fr_dielectric(cos_s, ior)
    if ls.is_delta:
        return alb * ls.Li * (cos_s * t_light / PI)
    if ls.pdf <= Float32(0.0):
        return RGB(Float32(0.0))
    var pdf_bsdf = cos_s / PI
    var w = power_heuristic(ls.pdf, pdf_bsdf)
    return alb * ls.Li * (cos_s * t_light * w / (ls.pdf * PI))

# ── Spectral siblings (staged rollout, see project_spectral_rendering memory
# / lovely-dazzling-meteor plan) ────────────────────────────────────────────
# Added ALONGSIDE bxdf_eval_any/_nee_weight_simple above rather than mutating
# them in place: those two are also called from bdpt.mojo (Stage 3) and
# sppm.mojo (Stage 4), which aren't wavelength-aware yet — changing their
# signature now would force-couple this (Stage 2, plain-path-tracer-only)
# change into BDPT/SPPM ahead of their own stages, defeating the point of
# staging. shading.mojo (Stage 2) calls these new spectral versions instead;
# bdpt.mojo/sppm.mojo keep calling the RGB originals unchanged until their
# own stage migrates them. Hair is NOT covered here (Marschner lobe color
# comes from sigma_a absorption, a genuinely more involved conversion) —
# _nee_weight_hair stays RGB-only for now, a deliberate scoped exclusion.
# The `spectral_coeffs/_res/_cie_x/_cie_y/_cie_z/_d65` sextuple below replaces
# a single `ctx: SpectralHandle` parameter -- see spectrum.mojo's long
# comment above rgb_to_spectral_sample for why: passing that 6-field struct
# BY VALUE across a real Mojo function-call boundary is a confirmed,
# reproducible miscompilation (one field comes back corrupted on a random
# subset of runs). Callers hold a SpectralHandle (e.g. shading.mojo's
# ctx.spectral) and pass its fields individually: ctx.spectral.coeffs,
# ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z,
# ctx.spectral.d65.
@always_inline
def bxdf_eval_any_spectral(
    mat_kind: Int32,
    alb:      RGB,             # diffuse albedo, or conductor f0
    alpha:    Float32,         # conductor GGX roughness; unused for diffuse
    n:        Vec3f,
    wo:       Vec3f,
    wi:       Vec3f,
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin],
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin],
    wavelengths: SampledWavelengths,
) -> Tuple[SpectralSample, Float32]:
    var alb_spectral = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, alb.r, alb.g, alb.b, wavelengths)
    if mat_kind == Int32(1):
        var (valid, k, schlick) = _ggx_conductor_shape_terms(n, wo, wi, alpha)
        if not valid:
            return (SpectralSample(Float32(0.0)), bxdf_pdf_conductor_ggx(n, wo, wi, alpha))
        # fr = f0 + (1-f0)*schlick, expressed without SpectralSample.__sub__
        # (not defined): fr = f0*(1-schlick) + 1*schlick.
        var fr_spectral = alb_spectral * (Float32(1.0) - schlick) + SpectralSample(schlick)
        return (fr_spectral * k, bxdf_pdf_conductor_ggx(n, wo, wi, alpha))
    var cos_wi = dot(n, wi)
    return (alb_spectral * INV_PI, bxdf_pdf_diffuse(cos_wi))

@always_inline
def _nee_weight_simple_spectral(
    ls:    LightSample,
    mat_kind: Int32,
    alb:   RGB,
    alpha: Float32,
    n:     Vec3f,
    wo:    Vec3f,
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin],
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin],
    wavelengths: SampledWavelengths,
) -> SpectralSample:
    """Spectral counterpart of _nee_weight_simple — same formula, but the
    material color and light color are each converted to a SpectralSample at
    this path's hero wavelengths (rgb_to_spectral_sample for the reflectance,
    rgb_illuminant_to_spectral_sample for the light's unbounded radiance/
    intensity) before multiplying, instead of multiplying plain RGB
    triples."""
    if not ls.valid:
        return SpectralSample(Float32(0.0))
    var cos_s = dot(n, ls.wi)
    if cos_s <= Float32(0.0):
        return SpectralSample(Float32(0.0))
    var (f, pdf_bsdf) = bxdf_eval_any_spectral(mat_kind, alb, alpha, n, wo, ls.wi, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
    if f.v0 <= Float32(0.0) and f.v1 <= Float32(0.0) and f.v2 <= Float32(0.0) and f.v3 <= Float32(0.0):
        return SpectralSample(Float32(0.0))
    var li_spectral = rgb_illuminant_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, ls.Li.r, ls.Li.g, ls.Li.b, wavelengths)
    if ls.is_delta:
        return f * li_spectral * cos_s
    var mis_w = power_heuristic(ls.pdf, pdf_bsdf)
    return (f * li_spectral) * (cos_s * mis_w / ls.pdf)

@always_inline
def _nee_weight_simple_via_spectral(
    ls:    LightSample,
    mat_kind: Int32,
    alb:   RGB,
    alpha: Float32,
    n:     Vec3f,
    wo:    Vec3f,
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin],
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin],
    wavelengths: SampledWavelengths,
) -> RGB:
    """Stage 2c-3's actual behavior-changing entry point: evaluates the NEE
    term spectrally (real Jakob-Hanika material/light color conversion at
    this path's 4 hero wavelengths) then immediately converts back to RGB,
    rather than flipping PathState_C.throughput/estimate to a spectral type
    for the whole path (that's a separate, much larger future change — see
    project_spectral_rendering memory). This still captures the real
    per-wavelength product-of-spectra behavior for the direct-lighting term,
    just re-expressed in RGB before accumulation, exactly the approach
    validated by this session's product-of-spectra tests
    (test_bxdf_spectral.mojo's _nee_weight_simple_spectral round-trip test)."""
    # An INVALID LightSample (bvh.mojo's _invalid_light_sample / a
    # non-emissive analytic sphere picked up by _nee_loop_simple's
    # sphere-light loop) must short-circuit BEFORE the spectral round-trip
    # below: `_nee_weight_simple_spectral` itself returns black for it, but
    # `spectral_sample_to_rgb` is still evaluated on that black sample, and
    # the RATIO_CAP clamp at the bottom is deliberately bypassed whenever
    # `plain` is black -- so any non-finite value the round-trip produces
    # (a null/dangling SpectralHandle is a documented "never dereferenced"
    # sentinel, and the zero-wavelength non-spectral case divides 0/0) is
    # returned RAW. A NaN then defeats the caller's `is_black()` guard (NaN
    # != 0), reaching _shadow_contribute as a NaN contribution on a
    # zero-length shadow ray, and one such sample poisons its whole pixel
    # (NaN + anything = NaN). normalize_film's NaN guard then rewrites the
    # pixel to 0, so the symptom is a SILENT, nondeterministic all-black
    # render rather than any error -- see the sphere_sms.xml diagnosis.
    if not ls.valid:
        return RGB(Float32(0.0))
    var spec = _nee_weight_simple_spectral(ls, mat_kind, alb, alpha, n, wo, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
    var (r, g, b) = spectral_sample_to_rgb(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, spec, wavelengths)

    # Variance guard: a hero-wavelength MC estimate of a COLORED light's
    # contribution uses only N_SPECTRAL_SAMPLES (4) wavelength samples per
    # path -- for a genuinely colored emitter (not a flat/white light) this
    # has real per-channel variance around the correct converged color,
    # visible as red/blue speckle fireflies on nearly-neutral (e.g. chrome)
    # conductor surfaces at low sample counts. Found via a real artifact the
    # user spotted: staircase2's rear light is strongly reddish
    # (L=[4.575,3.59,1.55]), reflected by the scene's Metal railing.
    # _nee_weight_simple (the plain, non-spectral RGB path) is a cheap,
    # always-available, correctly-colored reference for the right ballpark
    # -- clamp the spectral result to within RATIO_CAP x of it per channel.
    # This bounds the EXTRA variance spectral evaluation introduces without
    # discarding genuine, converged color shift (the reason to use spectral
    # evaluation here at all): a well-converged spectral estimate should
    # already agree with the plain-RGB one to within a modest factor, so
    # RATIO_CAP only ever clips the rare, high-variance outlier samples.
    var plain = _nee_weight_simple(ls, mat_kind, alb, alpha, n, wo)
    comptime RATIO_CAP: Float32 = 4.0
    var cr = min(r, plain.r * RATIO_CAP) if plain.r > Float32(0.0) else r
    var cg = min(g, plain.g * RATIO_CAP) if plain.g > Float32(0.0) else g
    var cb = min(b, plain.b * RATIO_CAP) if plain.b > Float32(0.0) else b
    # Belt-and-braces: never hand a non-finite weight back to a caller that
    # gates only on is_black() -- see the invalid-sample comment above for
    # why a single NaN here silently blacks out the entire render.
    if not (cr == cr and cg == cg and cb == cb):
        return plain
    return RGB(cr, cg, cb)

@always_inline
def _nee_weight_hair(ls: LightSample, hc: HairLobeConstants) -> RGB:
    """Hair's own version of _nee_weight_simple above — hair's BRDF needs
    the full precomputed per-hit HairLobeConstants (Marschner R/TT/TRT lobe
    state) rather than a flat (alb, alpha) pair, so it can't share
    bxdf_eval_any's signature, but presents the same LightSample-in,
    weight-out shape."""
    if not ls.valid:
        return RGB(Float32(0.0))
    var (cos_ti, f_val, pdf_over_cos) = _hair_eval_lobes(
        ls.wi, hc.tangent, hc.b_perp, hc.n_perp, hc.phi_o,
        hc.dphi0, hc.dphi1, hc.dphi2,
        hc.cos_tp0_o, hc.sin_tp0_o, hc.cos_tp1_o, hc.sin_tp1_o, hc.cos_tp2_o, hc.sin_tp2_o,
        hc.cos_theta_o, hc.sin_theta_o, hc.inv_vm0, hc.inv_vm1, hc.inv_vm2, hc.mp_c0, hc.mp_c1, hc.mp_c2, hc.s,
        hc.A0, hc.A1, hc.A2, hc.A3, hc.lum0, hc.lum1, hc.lum2, hc.lum3, hc.total_lum,
    )
    if ls.is_delta:
        return f_val * cos_ti * ls.Li
    var pdf_bsdf = max(cos_ti * pdf_over_cos, Float32(1e-6))
    var mis_w = power_heuristic(ls.pdf, pdf_bsdf)
    return f_val * cos_ti * ls.Li * (mis_w / ls.pdf)
