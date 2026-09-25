from std.collections import Array
from std.math import sqrt
from .layered import layered_f, layered_pdf, layered_sample
from .geometry import RGB, MatKind, LobeKind, Material_C, Vec3f, dot, INV_PI, PI, fr_dielectric, coat_beer_lambert_tr, cos_theta_t_dielectric, DEFAULT_COAT_THICKNESS, Frame, refract, INV_FOUR_PI, MeasuredBRDF_C
from .curves import Curve_C
from .bssrdf import fdr_moment, bssrdf_exit_ft
from .sampling import sample_ggx_vndf, sample_cosine_hemisphere_world, power_heuristic
from .vcm_mis import MisPolicy, mis_policy_power, mis_policy_sole, nee_mis_weight
from .rng import PCG32
from .measured_bxdf_eval import bxdf_eval_measured, bxdf_pdf_measured, bxdf_sample_measured
from .bvh import LightSample, HairLobeConstants, _hair_eval_lobes, SceneDescriptor2_C, _hair_precompute, _hair_sample_dir_u
from .spectrum import SampledWavelengths, SpectralSample, rgb_to_spectral_sample, rgb_illuminant_to_spectral_sample, spectral_sample_to_rgb, rgb_bands_to_spectral_sample

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
    var ms = ggx_ms_lobe(dot(wo, n), dot(wi, n), alpha, f0)
    return RGB(k * fr.r + ms.r, k * fr.g + ms.g, k * fr.b + ms.b)

# ── Kulla-Conty multiple-scattering energy compensation ─────────────────────
# Single-scattering GGX models light that strikes ONE microfacet and leaves.
# Light that bounces between microfacets before escaping is simply dropped, so
# a rough conductor loses energy that a real one keeps, and the loss grows
# with roughness. Measured here on the white furnace before this existed:
# 0.987 / 0.944 / 0.794 / 0.542 / 0.327 at alpha 0.1 / 0.2 / 0.4 / 0.7 / 1.0,
# where the answer is 1.0 at every roughness.
#
# Kulla & Conty (Revisiting Physically Based Shading, SIGGRAPH 2017 course)
# add the missing energy back as a second, reciprocal lobe:
#
#     f_ms(mu_o, mu_i) = (1 - E(mu_o)) (1 - E(mu_i)) / (pi (1 - E_avg))
#
# with E the single-scattering directional albedo and E_avg its
# cosine-weighted mean. f_ss + f_ms integrates to exactly 1 for a white
# surface: each angle's deficit becomes that lobe's strength.


@always_inline
def ggx_albedo(mu: Float32, alpha: Float32) -> Float32:
    """E(mu, alpha): directional albedo of single-scattering GGX at F = 1 --
    the fraction of incident energy one GGX bounce actually carries away, and
    therefore 1 - E is exactly what the compensation lobe has to put back.

    Fitted to a DETERMINISTIC QUADRATURE of the renderer's own f_ss, computed
    in the half-vector domain through the exact GGX NDF inverse CDF, 3000 x
    600 cells per (mu, alpha) over a 41 x 50 grid. Deterministic matters more
    than it sounds: the first version of this table was generated as
    E[G2/G1] over VNDF SAMPLES, which silently inherited a bug in
    sample_ggx_vndf and produced a table that was right only at mu = 1 -- so
    the compensation was correct head-on and wrong everywhere else, and the
    furnace SCENES could not see it because they all view their quad head-on.
    A quadrature cannot inherit a sampler's bug because it never calls one.

    E is NOT monotone in mu: it dips around mu ~ 0.4 and rises again toward
    grazing (0.307 / 0.379 / 0.499 / 0.760 at alpha = 1 for mu = 1 / 0.7 /
    0.4 / 0.1). A plain polynomial in mu cannot hold that shape, and forcing
    one to try is what produced coefficients in the millions with catastrophic
    fp32 cancellation. This is a degree 7x7 CHEBYSHEV fit in (2mu - 1,
    2sqrt(alpha) - 1) instead: every coefficient is below 0.24, the recurrence
    is stable, and the error at the angles that matter is under 0.13%. sqrt is
    the right variable for alpha because the deficit grows like sqrt(alpha)
    near zero.

    Accuracy is the ONLY thing standing between this and an exact furnace:
    with E_avg defined as the cosine-weighted mean of THIS fit (see
    ggx_albedo_avg), the compensation lobe's normalisation is exact, so the
    white furnace reads 1 + (E_true(mu_o) - E_fit(mu_o)) -- the pointwise fit
    error, nothing else. Measured worst case 0.4%.
    Pinned by Tests/unit/test_conductor_energy.mojo."""
    var m = min(max(mu, Float32(0.0)), Float32(1.0))
    var a = min(max(alpha, Float32(0.0)), Float32(1.0))
    var x = Float32(2.0) * m - Float32(1.0)
    var y = Float32(2.0) * sqrt(a) - Float32(1.0)
    var x0 = Float32(1.0)
    var x1 = x
    var x2 = Float32(2.0) * x * x1 - x0
    var x3 = Float32(2.0) * x * x2 - x1
    var x4 = Float32(2.0) * x * x3 - x2
    var x5 = Float32(2.0) * x * x4 - x3
    var x6 = Float32(2.0) * x * x5 - x4
    var x7 = Float32(2.0) * x * x6 - x5
    var y0 = Float32(1.0)
    var y1 = y
    var y2 = Float32(2.0) * y * y1 - y0
    var y3 = Float32(2.0) * y * y2 - y1
    var y4 = Float32(2.0) * y * y3 - y2
    var y5 = Float32(2.0) * y * y4 - y3
    var y6 = Float32(2.0) * y * y5 - y4
    var y7 = Float32(2.0) * y * y6 - y5
    var s0 = Float32(0.1799031) * y0 + Float32(0.2346308) * y1 + Float32(0.0651005) * y2 + -Float32(0.0026610) * y3 + -Float32(0.0104297) * y4 + Float32(0.0017012) * y5 + -Float32(0.0003267) * y6 + Float32(0.0005651) * y7
    var s1 = Float32(0.0824428) * y0 + Float32(0.1673271) * y1 + Float32(0.0655683) * y2 + -Float32(0.0152530) * y3 + -Float32(0.0057621) * y4 + -Float32(0.0042224) * y5 + Float32(0.0044000) * y6 + -Float32(0.0012006) * y7
    var s2 = -Float32(0.0340425) * y0 + -Float32(0.0627857) * y1 + Float32(0.0015742) * y2 + Float32(0.0126882) * y3 + -Float32(0.0133245) * y4 + Float32(0.0033451) * y5 + -Float32(0.0003205) * y6 + Float32(0.0006203) * y7
    var s3 = Float32(0.0118549) * y0 + Float32(0.0273742) * y1 + -Float32(0.0102168) * y2 + Float32(0.0000819) * y3 + Float32(0.0118832) * y4 + -Float32(0.0103083) * y5 + Float32(0.0034849) * y6 + -Float32(0.0001592) * y7
    var s4 = -Float32(0.0025948) * y0 + -Float32(0.0158656) * y1 + Float32(0.0094135) * y2 + -Float32(0.0040777) * y3 + -Float32(0.0064109) * y4 + Float32(0.0107764) * y5 + -Float32(0.0072839) * y6 + Float32(0.0022995) * y7
    var s5 = -Float32(0.0000513) * y0 + Float32(0.0094167) * y1 + -Float32(0.0064788) * y2 + Float32(0.0039060) * y3 + Float32(0.0024465) * y4 + -Float32(0.0073229) * y5 + Float32(0.0067780) * y6 + -Float32(0.0032536) * y7
    var s6 = Float32(0.0002302) * y0 + -Float32(0.0045547) * y1 + Float32(0.0033563) * y2 + -Float32(0.0024166) * y3 + -Float32(0.0004979) * y4 + Float32(0.0035747) * y5 + -Float32(0.0040240) * y6 + Float32(0.0023914) * y7
    var s7 = Float32(0.0001859) * y0 + Float32(0.0015979) * y1 + -Float32(0.0013145) * y2 + Float32(0.0013160) * y3 + -Float32(0.0003611) * y4 + -Float32(0.0011990) * y5 + Float32(0.0018070) * y6 + -Float32(0.0013354) * y7
    var deficit = s0 * x0 + s1 * x1 + s2 * x2 + s3 * x3 + s4 * x4 + s5 * x5 + s6 * x6 + s7 * x7
    return min(max(Float32(1.0) - deficit, Float32(1e-3)), Float32(1.0))


@always_inline
def ggx_albedo_avg(alpha: Float32) -> Float32:
    """E_avg(alpha) = 2 * integral of E(mu, alpha) mu dmu over [0,1].

    Deliberately fitted to the cosine-weighted mean of ggx_albedo's FIT, not
    of the quadrature truth. That sounds backwards and is the whole trick:
    the compensation lobe is normalised by 1/(pi(1 - E_avg)), so f_ms
    integrates to exactly 1 - E(mu_o) only when E_avg is the mean of the very
    same E the lobe's shape uses. Fitting both independently to truth leaves
    a residual that shows up as a furnace error twice the size of either.
    Degree 6 in alpha, max error 0.0004 against that mean.
    test_ggx_albedo_table_is_consistent pins the relationship itself."""
    var a = min(max(alpha, Float32(0.0)), Float32(1.0))
    var e = Float32(1.0003924) + a * (-Float32(0.0735147) + a * (-Float32(2.5279152) + a * (Float32(4.8253712) + a * (-Float32(5.0972079) + a * (Float32(3.0991620) + a * (-Float32(0.8173428)))))))
    return min(max(e, Float32(1e-3)), Float32(1.0))


@always_inline
def ggx_ms_shape(cos_o: Float32, cos_i: Float32, alpha: Float32) -> Float32:
    """The white, uncolored Kulla-Conty lobe

        (1 - E(mu_o)) (1 - E(mu_i)) / (pi (1 - E_avg))

    -- reciprocal by construction, and normalised so that for a perfectly
    reflective surface f_ss + f_ms integrates to exactly 1 at every mu_o."""
    var eavg = ggx_albedo_avg(alpha)
    var denom = max(Float32(1.0) - eavg, Float32(1e-4))
    return ((Float32(1.0) - ggx_albedo(cos_o, alpha))
            * (Float32(1.0) - ggx_albedo(cos_i, alpha)) * INV_PI / denom)


@always_inline
def ggx_ms_tint(f0: Float32, eavg: Float32) -> Float32:
    """Turquin (2019) colour term for one channel: the multiply-scattered
    lobe has bounced off the conductor more than once, so it is tinted by
    more than one Fresnel reflection. F_avg is the Fresnel average under
    Schlick (F0 + (1-F0)/21 in closed form); the geometric series over the
    unknown number of bounces sums to

        k = F_avg^2 E_avg / (1 - F_avg (1 - E_avg))

    which is 1 for a white conductor -- so this never changes the energy the
    white furnace measures, only the hue of a coloured one."""
    var favg = f0 + (Float32(1.0) - f0) * (Float32(1.0) / Float32(21.0))
    return favg * favg * eavg / max(Float32(1.0) - favg * (Float32(1.0) - eavg), Float32(1e-4))


@always_inline
def ggx_ms_lobe(cos_o: Float32, cos_i: Float32, alpha: Float32, f0: RGB) -> RGB:
    """The Kulla-Conty compensation lobe f_ms(wo, wi), cosine NOT applied.

    The colour term is Turquin's (Practical multiple scattering compensation
    for microfacet models, 2019): a multi-bounce path inside the microsurface
    is tinted once per bounce, so it cannot simply reuse F0.

        k = F_avg^2 E_avg / (1 - F_avg (1 - E_avg)),  F_avg = F0 + (1-F0)/21

    At F0 = 1 this is exactly 1, which is what makes the white furnace the
    honest test of the energy half on its own."""
    var eavg = ggx_albedo_avg(alpha)
    var shape = ggx_ms_shape(cos_o, cos_i, alpha)
    return RGB(shape * ggx_ms_tint(f0.r, eavg),
               shape * ggx_ms_tint(f0.g, eavg),
               shape * ggx_ms_tint(f0.b, eavg))


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
    # Two lobes are now sampled (see bxdf_sample_conductor), so the MIS
    # partner density is their mixture. The split probability is the energy
    # the single-scattering lobe fails to carry at this mu_o, which is what
    # makes the mixture spend samples where the compensation actually is.
    var p_ms = Float32(1.0) - ggx_albedo(cos_o, alpha)
    return (p_ms * cos_i * INV_PI
            + (Float32(1.0) - p_ms) * ggx_vndf_pdf(cos_o, cos_wm, d, alpha))

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

    # Two lobes: the single-scattering GGX one, and the Kulla-Conty
    # compensation lobe that carries the energy GGX drops. u1 selects between
    # them and is then RESCALED back to [0,1) rather than a third random
    # number being drawn -- the sampler hands out a fixed Sobol pair per
    # bounce (SobolSamples8.scat1/scat2), and stratification survives the
    # affine remap.
    var cos_o_s = dot(gc.wo, gc.normal)
    var alpha_iso = max(alpha_x, alpha_y)
    var p_ms = Float32(1.0) - ggx_albedo(cos_o_s, alpha_iso)
    var u_sel = u1
    var wi: Vec3f
    if u_sel < p_ms:
        var ur = u_sel / max(p_ms, Float32(1e-6))
        var cs = sample_cosine_hemisphere_world(min(ur, Float32(0.9999)), u2, gc.normal)
        wi = cs[0]
    else:
        var ur = (u_sel - p_ms) / max(Float32(1.0) - p_ms, Float32(1e-6))
        var wo_l = Vec3f(dot(gc.wo, gc.tangent), dot(gc.wo, gc.bitangent), dot(gc.wo, gc.normal))
        var wh_l = sample_ggx_vndf(wo_l, alpha_x, alpha_y, min(ur, Float32(0.9999)), u2)
        var wh_s = gc.tangent * wh_l.x + gc.bitangent * wh_l.y + gc.normal * wh_l.z
        var whlen = dot(wh_s, wh_s)
        if whlen > Float32(0.0):
            wh_s = wh_s * (Float32(1.0) / sqrt(whlen))
        wi = wh_s * (Float32(2.0) * dot(gc.wo, wh_s)) - gc.wo
        var wilen = dot(wi, wi)
        if wilen > Float32(0.0):
            wi = wi * (Float32(1.0) / sqrt(wilen))
    var cos_i_s = dot(wi, gc.normal)
    if cos_i_s <= Float32(0.0) or cos_o_s <= Float32(0.0):
        return BxDFSample(wi, RGB(Float32(0.0)), Float32(1.0), BxDFFlags.glossy | BxDFFlags.reflect, Int8(0), Int8(0), Int8(0))

    # f_ss * cos_i and its own density, from the half-vector this pair
    # implies -- computed rather than carried, because the cosine branch has
    # no half-vector of its own.
    var wh = gc.wo + wi
    var whl2 = dot(wh, wh)
    if whl2 <= Float32(0.0):
        return BxDFSample(wi, RGB(Float32(0.0)), Float32(1.0), BxDFFlags.glossy | BxDFFlags.reflect, Int8(0), Int8(0), Int8(0))
    wh = wh * (Float32(1.0) / sqrt(whl2))
    var cos_wm = dot(wh, gc.normal)
    var cos_wh = max(Float32(0.0), dot(gc.wo, wh))
    var d = ggx_D(cos_wm, alpha_iso)
    var g2 = ggx_G2(cos_o_s, cos_i_s, alpha_iso)
    var one_m = Float32(1.0) - cos_wh
    var one_m2 = one_m * one_m
    var schlick = one_m2 * one_m2 * one_m
    var fresnel_rgb = mat.albedo + (white - mat.albedo) * schlick
    # NOTE the 1/(4 cos_o) and NOT 1/(4 cos_o cos_i): this is f_ss * cos_i.
    # The old code returned the bare Fresnel here, i.e. f_ss*cos_i/pdf_vndf
    # with the G2/G1 DROPPED, so a rough conductor's BSDF-sampled bounce was
    # too bright by G1/G2 -- 1.6x at alpha=1. Measured on the white furnace
    # with NEE disabled and MIS forced to 1: the strategy read P(wi above the
    # horizon) (0.99/0.96/0.86/0.67/0.50 over alpha 0.1..1.0) where the GGX
    # directional albedo it should read is 0.99/0.95/0.79/0.50/0.31. The coat
    # walk in this same file always had the factor (coat_walk_scatter's
    # `w.beta *= ggx_G2(...) / ggx_G1(...)`); the plain conductor never did.
    var f_ss_cos = d * g2 / (Float32(4.0) * cos_o_s)
    var eavg = ggx_albedo_avg(alpha_iso)
    var ms_shape_cos = ggx_ms_shape(cos_o_s, cos_i_s, alpha_iso) * cos_i_s
    var pdf_mix = (p_ms * cos_i_s * INV_PI
                   + (Float32(1.0) - p_ms) * ggx_vndf_pdf(cos_o_s, cos_wm, d, alpha_iso))
    if pdf_mix <= Float32(0.0):
        return BxDFSample(wi, RGB(Float32(0.0)), Float32(1.0), BxDFFlags.glossy | BxDFFlags.reflect, Int8(0), Int8(0), Int8(0))
    var inv_pdf = Float32(1.0) / pdf_mix
    var w = RGB((f_ss_cos * fresnel_rgb.r + ms_shape_cos * ggx_ms_tint(mat.albedo.r, eavg)) * inv_pdf,
                (f_ss_cos * fresnel_rgb.g + ms_shape_cos * ggx_ms_tint(mat.albedo.g, eavg)) * inv_pdf,
                (f_ss_cos * fresnel_rgb.b + ms_shape_cos * ggx_ms_tint(mat.albedo.b, eavg)) * inv_pdf)
    return BxDFSample(wi, w, Float32(1.0), BxDFFlags.glossy | BxDFFlags.reflect, Int8(1), Int8(0), Int8(0))

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
@fieldwise_init
struct DielectricInterface(TrivialRegisterPassable):
    """The decision every dielectric interaction makes before it picks a
    lobe: which way the surface faces, the relative IOR, and whether the
    transmitted direction exists at all. Sampling (the Fresnel coin flip)
    and the refracted/reflected directions are the CALLER's, so this one
    function serves the path tracer's sampler, SPPM/VCM's own bounce, and
    the --pixel tracer, which each used to re-derive it (and had drifted:
    the debug tracer twice went stale against the real formula, and
    SPPM's copy still flips the normal by `entering` rather than by which
    way the surface faces the ray)."""
    var normal:   Vec3f     # geometric normal, flipped to face the incoming ray
    var entering: Bool      # crossing INTO this surface's material
    var eta:      Float32   # relative IOR n_i / n_t
    var cos_i:    Float32
    var sin2_t:   Float32
    var tir:      Bool      # no transmitted direction exists
    var fresnel:  Float32   # reflectance at this angle

@always_inline
def dielectric_interface(
    geom_normal: Vec3f,
    ray_dir: Vec3f,
    ior: Float32,
    force_entering: Bool,   # bounce==0: trust physics (camera ray always from air)
    current_ior: Float32 = Float32(1.0),
    previous_ior: Float32 = Float32(1.0),
) -> DielectricInterface:
    var facing = dot(ray_dir, geom_normal) < Float32(0.0)
    var entering = facing or force_entering
    var normal = geom_normal if facing else -geom_normal
    # Entering: relative IOR is (medium the ray is coming FROM) / (this
    # surface's own IOR) -- current_ior, not a hardcoded vacuum. Exiting:
    # relative IOR is (this surface's own IOR) / (medium one level below,
    # i.e. what's really outside) -- previous_ior, not a hardcoded vacuum
    # either. Both directions used to assume vacuum on the far side; for a
    # simple isolated pane previous_ior defaults to 1.0 so this is
    # unchanged, but for touching same-material CAD parts (bolt threaded
    # through a bracket, etc.) the old unconditional `eta = ior` on exit
    # spuriously triggered TOTAL INTERNAL REFLECTION on a boundary that
    # should have been invisible (eta=1), trapping rays in a runaway TIR
    # cascade instead of letting them escape -- confirmed via --pixel trace
    # on transparent-machines (Scenes/dielectric-touching-same-ior.pbrt's
    # simple 2-sphere repro didn't expose this because it never chains
    # enough touching interfaces to matter at the whole-image mean).
    var eta = (current_ior / ior) if entering else (ior / previous_ior)
    var cos_i = -dot(ray_dir, normal)
    var sin2_t = eta * eta * (Float32(1.0) - cos_i * cos_i)
    # eta here is eta_i/eta_t; fr_dielectric wants its reciprocal.
    return DielectricInterface(normal, entering, eta, cos_i, sin2_t,
                               sin2_t > Float32(1.0), fr_dielectric(cos_i, Float32(1.0) / eta))

@always_inline
def bxdf_sample_dielectric(
    geom_normal: Vec3f,
    ray_dir: Vec3f,
    ior: Float32,
    force_entering: Bool,   # bounce==0: trust physics (camera ray always from air)
    u_reflect: Float32,
    current_ior: Float32 = Float32(1.0),    # IOR of the medium the ray is ALREADY in; 1.0 = vacuum
    previous_ior: Float32 = Float32(1.0),   # IOR one level below current_ior (what exiting restores)
    # pbrt's TransportMode, made explicit. A CAMERA/radiance path takes the
    # eta^2 non-symmetry correction on transmission; a LIGHT/photon path does
    # not. This used to be an unwritten assumption ("this function is only
    # ever reached from camera-path contexts"), which is exactly what let
    # SPPM keep a second, inverted copy of the factor. See _dielectric_bounce.
    radiance_mode: Bool = True,
) -> Tuple[BxDFSample, Vec3f, Float32, Float32]:
    """Third/fourth return values are the CALLER'S new current_ior/
    previous_ior to store (path state) for the next dielectric interaction
    along this path -- unchanged on reflect/TIR (still in the same medium).
    On a transmitted ENTRY: current_ior <- `ior` (now inside this surface's
    material), previous_ior <- the OLD current_ior (pushed, so the matching
    exit can restore it). On a transmitted EXIT: current_ior <- the OLD
    previous_ior (popped -- what's really outside, not always vacuum),
    previous_ior <- 1.0 (this is only a depth-2 stack: a third level of
    nesting loses the level below the one just popped -- a scoped,
    documented limitation, not an oversight; see
    PathState_C.previous_dielectric_ior's docstring).

    `current_ior`/`previous_ior` default to vacuum so every OTHER caller
    (BDPT/SPPM's own separate _dielectric_bounce, and any test that doesn't
    care about touching-dielectric seams) is unaffected -- this only changes
    behavior when a caller actually threads non-vacuum values through."""
    var di = dielectric_interface(geom_normal, ray_dir, ior, force_entering, current_ior, previous_ior)
    var normal = di.normal
    var entering = di.entering
    var eta = di.eta
    var cos_i = di.cos_i
    var sin2_t = di.sin2_t
    var tir = di.tir
    var fresnel = di.fresnel
    var white = RGB(Float32(1.0))

    if tir or u_reflect < fresnel:
        var refl = ray_dir + normal * (Float32(2.0) * cos_i)
        var rlen = dot(refl, refl)
        if rlen > Float32(0.0):
            refl = refl * (Float32(1.0) / sqrt(rlen))
        return (BxDFSample(refl, white, Float32(1.0), BxDFFlags.delta | BxDFFlags.reflect, Int8(1), Int8(0), Int8(0)), normal, current_ior, previous_ior)

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
    var radiance_transmit = (white * (eta * eta)) if radiance_mode else white
    var new_current_ior = ior if entering else previous_ior
    var new_previous_ior = current_ior if entering else Float32(1.0)
    return (BxDFSample(refr, radiance_transmit, Float32(1.0), BxDFFlags.delta | BxDFFlags.transmit, Int8(1), Int8(0), Int8(0)), normal, new_current_ior, new_previous_ior)

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
    if mat_kind == LobeKind.ggx:
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
    mis: MisPolicy = mis_policy_power(),
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
    var mis_w = nee_mis_weight(mis, ls.pdf, pdf_bsdf, cos_s)
    return f * ls.Li * (cos_s * mis_w / ls.pdf)

# ── Coateddiffuse: THE layered-BSDF walk, shared by every integrator ─────────
#
# A coated diffuse surface has no closed-form BSDF: evaluating it means
# stochastically walking the coat (refract in, bounce off the Lambertian base,
# try to refract back out, repeat). Before this existed, EVERY integrator
# re-derived that walk inline in its own bounce loop -- shading.mojo's
# shade_coated_diffuse, bdpt.mojo's camera AND light path branches, and
# (eventually) sppm.mojo. Each copy then drifted independently, which is
# exactly how 2026-09-15 found: a missing 1/eta^2 in one, absent coat-thickness
# attenuation in another, no rough-coat G2/G1 anywhere but the path tracer, and
# -- worst -- sppm.mojo having no coat model AT ALL (it aliased coateddiffuse
# to plain Lambertian, rendering 2.02x too bright). See
# project_elegance_backlog_2026_09_10 items 4/9 and
# project_bistro_brightness_gap.
#
# WHY A STEPPER AND NOT ONE `walk()` CALL: the walk's RNG draws are interleaved
# with each integrator's OWN next-event-estimation draws -- the coat-lobe NEE
# fires between the half-vector sample and the entry coin flip, and the base
# NEE fires between the Russian-roulette check and the cosine-hemisphere
# sample, at every recycle depth. A black-box `walk()` would have to hoist
# those out, changing every existing render's random sequence. Splitting the
# walk at exactly those two seams keeps each caller's RNG stream bit-identical
# while still leaving ONE copy of the physics. Integrator bookkeeping (NEE,
# vertex/VP/photon storage, MIS carries, transport-mode eta^2) stays with the
# integrator, where it belongs; only the material's own behaviour lives here.
#
# Call sequence:
#     var w = coat_walk_begin(gn, wo, alb, ior, alpha, pcg)
#     <caller's coat-lobe NEE, using w.alpha/w.gn/w.wo>
#     coat_walk_enter(w, pcg)
#     if w.event == COAT_REFLECT:  <caller uses w.wi / w.pdf / w.beta>
#     else:
#         while w.event == COAT_WALKING:
#             if not coat_walk_at_base(w, pcg): break
#             <caller's base NEE, using w.beta>
#             coat_walk_scatter(w, pcg)
#         if w.event == COAT_EXIT:  <caller uses w.wi / w.beta>

comptime COAT_WALKING = Int32(0)   # inside the coat, more base bounces possible
comptime COAT_REFLECT = Int32(1)   # sampled the coat's own top-interface lobe
comptime COAT_EXIT    = Int32(2)   # refracted back out after >= 1 base bounce
comptime COAT_ABSORB  = Int32(3)   # terminated (Russian roulette or depth cap)

comptime COAT_MAX_DEPTH = 10

# Per-channel chrominance floor -- see "Numerical hygiene" in
# docs/05_reflection_models.md. Without it, up to COAT_MAX_DEPTH applications
# of `beta *= alb` against one cached texel drive the weakest channel toward
# zero while another stays large. Deliberately part of the SHARED walk: before
# this, shading.mojo applied it and bdpt.mojo did not -- a silent divergence
# between two copies of "the same" model, which is the whole reason this
# function exists.
comptime COAT_BETA_CHROMA_FLOOR: Float32 = 0.1

@fieldwise_init
struct CoatWalk(TrivialRegisterPassable):
    """State of one coateddiffuse layered-BSDF walk. See the block comment
    above for the call sequence and why this is a stepper."""
    var gn:        Vec3f      # face-forwarded geometric/shading normal
    var tangent:   Vec3f
    var bitangent: Vec3f
    var wo:        Vec3f      # toward the viewer (camera path) or the previous vertex
    var wm:        Vec3f      # sampled top-interface microfacet (== gn when smooth)
    var alb:       RGB        # base-layer reflectance at this hit
    var ior:       Float32    # coat IOR (eta_coat / eta_outside)
    var inv_ior:   Float32
    var alpha:     Float32    # coat GGX roughness
    var is_rough:  Bool
    var cos_o:     Float32    # dot(wo, gn)
    var cos_wm:    Float32    # dot(wo, wm)
    var f_entry:   Float32    # Fresnel reflectance at the top interface
    # Accumulated walk throughput, NOT including the transport-mode-dependent
    # eta^2 (that is the caller's call: radiance transport needs it, importance
    # transport does not -- see bdpt.mojo's light-path exit). On the REFLECT
    # path this is ACHROMATIC by construction: only scalar G2/G1 weights are
    # applied and the coloured base layer is never reached, so a caller whose
    # throughput is spectral may use `beta.r` as a plain scalar rather than
    # pushing a bare weight through the reflectance upsampler (which would not
    # come back unchanged per lane).
    var beta:      RGB
    var wi:        Vec3f      # outgoing direction once event is REFLECT or EXIT
    var pdf:       Float32    # REFLECT: VNDF pdf (rough) / -1 (smooth delta). EXIT: 0.
    var event:     Int32
    var depth:     Int32

@always_inline
@always_inline
def coat_exit_norm(ior: Float32) -> Float32:
    """Z = the fraction of base-sampled directions that escape the coat, i.e.
    the dielectric interface's directional albedo for transmission:

        Z = int_H (cos/pi) (1 - F(cos)) dw = 2 int_0^1 mu (1 - F(mu)) dmu

    This is the normalizer of `bxdf_pdf_coated_exit`; see that function for
    why the exit distribution needs it. Midpoint quadrature (dielectric
    Fresnel has no closed form), but restricted to [mu_c, 1] where mu_c is
    the total-internal-reflection critical angle cos, sqrt(1 - 1/ior^2):
    below it F == 1 exactly, so the integrand is identically zero and the
    full-interval integrand has a JUMP at mu_c. Quadrature across that jump
    converges at O(1/N) and left the pdf ~0.6% off normalization at
    ior 1.2 -- integrating only the live interval restores fast convergence
    and is why test_pdf_integrates_to_one holds to 2e-3."""
    comptime N = 64
    var inv_ior = Float32(1.0) / ior
    var mu_c = sqrt(max(Float32(0.0), Float32(1.0) - inv_ior * inv_ior))
    var span = Float32(1.0) - mu_c
    if span <= Float32(0.0):
        return Float32(0.0)
    var acc = Float32(0.0)
    for i in range(N):
        var mu = mu_c + span * (Float32(i) + Float32(0.5)) / Float32(N)
        acc += mu * (Float32(1.0) - fr_dielectric(mu, inv_ior))
    return Float32(2.0) * span * acc / Float32(N)

@fieldwise_init
struct LobeTables(TrivialRegisterPassable):
    """The three scene tables a lobe evaluation can read, and nothing else.

    lobe_eval took these as three loose pointers because the integrators do
    not share a context type: VCM has SceneDescriptor2_C (39 fields, passed
    by ref), the path tracer has ShadeContext (23 fields, passed by value),
    and they carry ELEVEN of the same scene pointers between them.

    Merging those two is a ~979-site rename across by-value GPU structs with
    a documented crash history (vcm_mat_kind_and_sd_by_value_traps), so this
    takes the narrow win instead: one named thing to pass, produced from
    either context by a one-line accessor. If the two contexts are ever
    reconciled, this is what they should agree on first."""
    var materials:      Pointer[Material_C, MutUntrackedOrigin]
    var curves:         Pointer[Curve_C, MutUntrackedOrigin]
    var measured_brdfs: Pointer[MeasuredBRDF_C, MutUntrackedOrigin]


@fieldwise_init
struct LobeCtx(TrivialRegisterPassable):
    """Everything the ONE lobe evaluator needs about a shading point, and
    nothing about which integrator is asking.

    This is the BxDF interface's input. A path tracer builds one from its
    local shading variables, VCM from a stored BDPTVertex, SPPM from a
    visible point -- and they all then get the same f, the same densities and
    the same answer to "is this lobe MIS-scoped". Before it existed each
    integrator carried its own dispatch over LobeKind and they drifted; every
    VCM defect found on 2026-09-20 lived in that drift.

    `param` is the per-kind scalar, the way BDPTVertex already overloaded
    pdf_bwd: GGX alpha for a conductor, eta for a BSSRDF exit, coat alpha for
    a coat walk."""
    var kind:           Int32
    var is_surface:     Bool
    var is_delta:       Bool
    var n:              Vec3f
    var wo:             Vec3f
    var alb:            RGB
    var mat_idx:        Int32
    var param:          Float32
    var pdf_fwd:        Float32
    var hair_curve_idx: Int32
    var hair_h:         Float32
    var hair_v:         Float32
    # True when the caller has already oriented `n` for the lobe it wants, so
    # `wo` may legitimately sit on the far side -- which is how
    # shade_diffuse_transmission addresses its TRANSMITTED lobe. The opaque
    # sidedness test below must not fire there: it exists for stored vertices,
    # where `wo` really is the direction the subpath arrived from.
    var pre_oriented:   Bool
    # True at a LIGHT-subpath vertex, where `wo` points back toward the light
    # and the direction evaluated points toward the camera side. Radiance
    # needs f(toward camera, toward light) there too -- the ADJOINT
    # f*(a, b) = f(b, a). Every analytic lobe here is reciprocal, so only the
    # tabulated measured BRDF reads it: its table is indexed by the FIRST
    # argument (pbrt's MeasuredBxDF::f), and near grazing it is far from
    # reciprocal -- cm_white_spec.bsdf at 83 deg: f(cam,light) 20.2 vs
    # f(light,cam) 1.7. Mixing the two orders across VCM's strategies made
    # MIS blend estimates of two different integrands.
    var adjoint:        Bool


@fieldwise_init
struct LobeEval(TrivialRegisterPassable):
    """One vertex lobe, evaluated ONCE: throughput, the cosine that throughput
    already contains, both densities, and whether it has real densities at all.

    This exists because the same dispatch over LobeKind was written out THREE
    times -- `_eval_vertex_spectral` (f*cos), `_bdpt_vertex_pdfs`
    (forward/reverse densities) and `_bdpt_vertex_mis_scoped` (which kinds
    have densities) -- and the three lists drifted apart. Every VCM defect
    found on 2026-09-20 lived in that drift:

      * `coated_walk` appeared in the scope test and had NO branch in the
        evaluator, so a coateddiffuse vertex was silently evaluated as bare
        diffuse -- no entry Fresnel, no coat absorption, no 1/eta^2.
      * the evaluator returned f*cos while photon-density estimation needs the
        BARE f, and nothing in the signature said which -- merging applied the
        cosine twice and lost exactly (2pi/3)/pi = 2/3 of its energy.
      * the opaque-Lambertian fallback had no sidedness test, so a photon that
        landed on the BACK of a surface reflected out of the front; t=1 light
        tracing alone came out 2.04x its analytic answer.

    `cos_used` answers the second one structurally: a density estimator
    divides it out EXPLICITLY instead of a caller guessing the convention. It
    is not always |cos(dir,n)| -- hair carries the FIBRE cosine and a volume
    carries none -- which a caller cannot know and kept getting wrong.
    `scoped` answers the first: one list, not three."""
    var f_cos:    SpectralSample   # BSDF (or phase) * the cosine this lobe uses
    var cos_used: Float32          # that cosine; 1 where the lobe has none
    var pdf_fwd:  Float32          # solid-angle density INTO dir_to_other
    var pdf_rev:  Float32          # ... and back toward v.wo (adjoint)
    var scoped:   Bool             # real densities exist -> may take part in MIS


@always_inline
def lobe_scoped(c: LobeCtx) -> Bool:
    """Does this lobe have REAL forward/reverse densities, i.e. may it take
    part in MIS? THE one list -- lobe_eval reports it as LobeEval.scoped and
    every integrator asks this, so a kind cannot be in scope for one consumer
    and out of it for another.

    coateddiffuse's coat walk is IN scope when the coat is SMOOTH, and out when
    rough. Smooth: bxdf_pdf_coated_exit is its real exit density, coat_eval_smooth
    is its real f(wo,wi), both subpaths' coat vertices carry real MIS state
    (6ec79292 camera side, and the light side in the same change as this
    docstring), and its NEE is single-shot so every strategy sees ONE model of
    the vertex. With all of that -- and the sampled exit ray no longer carrying
    a 1/eta^2 the refraction Jacobian already cancels -- the white furnace reads
    0.9188 against PT's 0.9128. Every one of those pieces was needed; the
    earlier attempts that scoped it with any of them missing measured worse
    than unscoped, and the numbers are in project_coateddiffuse_eta2_bug.
    Rough: no closed form for the exit density, so it stays a sole-strategy
    NEE vertex (its exit ray drops direct via PDF_DROP_DIRECT, like PT).
    Dielectrics are genuinely delta and never will be in scope."""
    if c.kind == LobeKind.layered:
        return c.is_surface
    if c.kind == LobeKind.coated_walk:
        # Rough coats are in scope too, on an APPROXIMATE density:
        # bxdf_pdf_coated_exit is derived for a smooth coat and takes no
        # alpha, so for a rough one it is the right shape with the wrong
        # width. That is legitimate -- an approximated pdf used for MIS does
        # not bias the estimator (pbrt, Reflection Models / Further Reading),
        # it only moves variance between strategies. Leaving rough coats OUT
        # was the worse option: their NEE then takes sole-strategy weight 1
        # while merging and t=1 splat at the same vertices with fabricated
        # zero-carry weights, which is a real double count rather than a
        # variance trade.
        return c.is_surface
    if not c.is_surface or c.is_delta:
        return False
    # coated_reflect is the coat's own glossy GGX lobe, so unlike the
    # coated_walk EXIT above it has a genuinely EXACT density -- the same
    # ggx_vndf_pdf that sampled it in coat_walk_enter, no approximation and
    # no missing alpha. It is only ever STORED for a rough coat (a smooth
    # coat's reflect branch is a delta mirror and stores no vertex at all),
    # so reaching here already implies the lobe is non-delta.
    # diffuse_transmit: lobe_eval returns its real two-lobe densities (the
    # luminance split times each side's cosine). It was missing here while
    # lobe_eval hard-coded True for it, so VCM's NEE at a leaf weighted
    # itself against connections that _connect then ran UNWEIGHTED: floor
    # lit only through a diffusetransmission panel read 1.8x pbrt.
    return (c.kind == LobeKind.lambertian or c.kind == LobeKind.ggx
            or c.kind == LobeKind.hair or c.kind == LobeKind.measured
            or c.kind == LobeKind.coated_reflect
            or c.kind == LobeKind.diffuse_transmit)


@always_inline
def lobe_eval[want_pdfs: Bool = True](
    c:   LobeCtx,
    dir_to_other:  Vec3f,
    tab: LobeTables,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    wavelengths: SampledWavelengths,
) -> LobeEval:
    """THE lobe dispatch.

    Takes the three tables it actually reads rather than a scene descriptor,
    because the integrators do not agree on a descriptor type: VCM has
    SceneDescriptor2_C, the path tracer has ShadeContext. Depending on one of
    them would have locked this to one integrator again, which is the whole
    condition this interface exists to remove.

    `want_pdfs=False` skips the density half, which for
    hair is an entire second `_hair_precompute` -- not a micro-optimisation."""
    var ZERO = SpectralSample(Float32(0))
    if c.is_delta:
        return LobeEval(ZERO, Float32(1), Float32(0), Float32(0), False)

    if not c.is_surface:
        # Volume: isotropic phase, no surface, and so NO cosine at all.
        var alb_v = rgb_bands_to_spectral_sample(c.alb.r, c.alb.g, c.alb.b, wavelengths)
        return LobeEval(alb_v * INV_FOUR_PI, Float32(1), INV_FOUR_PI, INV_FOUR_PI, False)

    var vn = c.n
    var vwo = c.wo

    if c.kind == LobeKind.layered:
        # pbrt's LayeredBxDF (layered.mojo). Everything is real: f, and the
        # pdf estimator both ways, which is a deterministic function of the
        # directions (hash-seeded), so every strategy that asks gets one value.
        # Light-side vertices evaluate the adjoint (importance transport).
        var mat_ly = tab.materials[unsafe_offset=Int(c.mat_idx)]
        var ior_ly = mat_ly.emission.r
        var alpha_ly = max(mat_ly.roughU, mat_ly.roughV)
        var fr_ly = Frame.from_z(Vec3f(vn[0], vn[1], vn[2]))
        var tx_ly = Vec3f(fr_ly.x.x, fr_ly.x.y, fr_ly.x.z)
        var ty_ly = Vec3f(fr_ly.y.x, fr_ly.y.y, fr_ly.y.z)
        var wo_ly = Vec3f(dot(vwo, tx_ly), dot(vwo, ty_ly), dot(vwo, vn))
        var wi_ly = Vec3f(dot(dir_to_other, tx_ly), dot(dir_to_other, ty_ly), dot(dir_to_other, vn))
        var rad_ly = not c.adjoint
        var R_ly = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, c.alb.r, c.alb.g, c.alb.b, wavelengths)
        var f_ly = layered_f(wo_ly, wi_ly, R_ly, ior_ly, alpha_ly, rad_ly)
        var cos_ly = abs(wi_ly.z)
        var fwd_ly = Float32(0)
        var rev_ly = Float32(0)
        comptime if want_pdfs:
            fwd_ly = layered_pdf(wo_ly, wi_ly, ior_ly, alpha_ly, rad_ly)
            rev_ly = layered_pdf(wi_ly, wo_ly, ior_ly, alpha_ly, not rad_ly)
        return LobeEval(f_ly * cos_ly, cos_ly, fwd_ly, rev_ly, True)
    if c.kind == LobeKind.coated_walk:
        # coateddiffuse's coat-walk EXIT lobe. Same factorization
        # _nee_weight_coated_diffuse_base uses on the NEE side, reading ior
        # from the material so the two cannot drift. The view-side
        # transmittance is deliberately absent -- implicit in the walk having
        # reached the base at all (project_coateddiffuse_eta2_bug).
        var mat_cw = tab.materials[unsafe_offset=Int(c.mat_idx)]
        var ior_cw = mat_cw.emission.r
        var cos_cw = dot(dir_to_other, vn)
        if cos_cw <= Float32(0) or dot(vwo, vn) <= Float32(0):
            return LobeEval(ZERO, Float32(1), Float32(0), Float32(0), False)
        # THE evaluator: f(wo, wi) for this lobe, both transmissions and the
        # whole TIR recycling series in closed form (coat_eval_smooth). This
        # used to be a single-scatter approximation that disagreed with the
        # per-iteration NEE beside it.
        # Upsample the true REFLECTANCE and carry the rest as a scalar:
        # rgb_to_spectral_sample's domain is a bounded reflectance, and a BSDF
        # value is not one ("a coefficient is not a color",
        # docs/02_spectra_and_color.md). The series is averaged over channels
        # here -- exact for a grey base, approximate for a saturated one.
        var f_cw = coat_eval_smooth(vn, vwo, dir_to_other, RGB(Float32(1.0)), ior_cw)
        var avg_cw = (c.alb.r + c.alb.g + c.alb.b) * Float32(1.0 / 3.0)
        var gain_cw = f_cw.r / max(Float32(1.0) - avg_cw * fdr_moment(ior_cw), Float32(1e-4)) * max(Float32(1.0) - fdr_moment(ior_cw), Float32(1e-4))
        var alb_cw = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, c.alb.r, c.alb.g, c.alb.b, wavelengths) * gain_cw
        # A SMOOTH coat's exit density IS tractable -- bxdf_pdf_coated_exit,
        # histogram-verified against the real walk in 27152093 -- and is
        # symmetric in the exit cosine, so the reverse density is the same
        # function of wo. A rough coat has no such closed form and stays out
        # of scope.
        var scoped_cw = lobe_scoped(c)
        var fwd_cw = Float32(0)
        var rev_cw = Float32(0)
        comptime if want_pdfs:
            if scoped_cw:
                fwd_cw = bxdf_pdf_coated_exit(cos_cw, ior_cw)
                rev_cw = bxdf_pdf_coated_exit(abs(dot(vwo, vn)), ior_cw)
        return LobeEval(alb_cw * cos_cw, cos_cw,
                        fwd_cw, rev_cw, scoped_cw)

    if c.kind == LobeKind.coated_reflect:
        # The coat's OWN glossy reflection off the top interface. THE SAME
        # model _nee_weight_coated_coat_lobe uses for this lobe's NEE --
        # D*G2*F/(4 cos_o), a dielectric-Fresnel GGX -- so the evaluated and
        # the NEE side cannot drift, exactly the reason lobe_eval exists.
        #
        # Until coated_reflect had its own LobeKind this fell into the
        # coated_walk branch above and was evaluated with coat_eval_smooth:
        # the BASE TRANSMISSION model, carrying the base's albedo and both
        # coat crossings, for a bounce that never enters the coat at all.
        # Achromatic by construction -- a dielectric coat's Fresnel tints
        # nothing -- hence the scalar SpectralSample rather than an upsample
        # of c.alb, which belongs to the base and must NOT appear here.
        var mat_cr = tab.materials[unsafe_offset=Int(c.mat_idx)]
        var ior_cr = mat_cr.emission.r
        var cos_o_cr = dot(vwo, vn)
        var cos_i_cr = dot(dir_to_other, vn)
        if cos_o_cr <= Float32(0) or cos_i_cr <= Float32(0):
            return LobeEval(ZERO, Float32(1), Float32(0), Float32(0), False)
        var wm_cr = vwo + dir_to_other
        var wmlen_cr = dot(wm_cr, wm_cr)
        if wmlen_cr <= Float32(0):
            return LobeEval(ZERO, Float32(1), Float32(0), Float32(0), False)
        wm_cr = wm_cr * (Float32(1.0) / sqrt(wmlen_cr))
        var cos_wm_cr = dot(vwo, wm_cr)
        if cos_wm_cr <= Float32(0):
            return LobeEval(ZERO, Float32(1), Float32(0), Float32(0), False)
        var d_cr = ggx_D(dot(vn, wm_cr), c.param)
        var g2_cr = ggx_G2(cos_o_cr, cos_i_cr, c.param)
        var fr_cr = fr_dielectric(cos_wm_cr, ior_cr)
        var f_cr = d_cr * g2_cr * fr_cr / (Float32(4.0) * cos_o_cr)
        var fwd_cr = Float32(0)
        var rev_cr = Float32(0)
        comptime if want_pdfs:
            # EXACT, unlike the coated_walk exit's approximation: this is the
            # very density coat_walk_enter sampled the bounce with.
            fwd_cr = ggx_vndf_pdf(cos_o_cr, cos_wm_cr, d_cr, c.param)
            rev_cr = ggx_vndf_pdf(cos_i_cr, dot(dir_to_other, wm_cr), d_cr, c.param)
        return LobeEval(SpectralSample(f_cr * cos_i_cr), cos_i_cr,
                        fwd_cr, rev_cr, True)

    if c.kind == LobeKind.bssrdf:
        var cos_x = abs(dot(dir_to_other, vn))
        return LobeEval(SpectralSample(bssrdf_exit_ft(cos_x, c.param) * INV_PI * cos_x),
                        cos_x, cos_x * INV_PI, c.pdf_fwd, False)

    if c.kind == LobeKind.ggx:
        # _eval_conductor_ggx_spectral folds in cos(dir_to_other, n) via its
        # `k * cos_i`, so that -- not 1 -- is this lobe's cosine.
        var f_g = _eval_conductor_ggx_spectral(vn, vwo, dir_to_other, c.param, c.alb, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
        var cos_g = abs(dot(dir_to_other, vn))
        var fwd_g = Float32(0)
        var rev_g = Float32(0)
        comptime if want_pdfs:
            fwd_g = bxdf_pdf_conductor_ggx(vn, vwo, dir_to_other, c.param)
            rev_g = bxdf_pdf_conductor_ggx(vn, dir_to_other, vwo, c.param)
        return LobeEval(f_g, cos_g, fwd_g, rev_g, True)

    if c.kind == LobeKind.hair:
        # Hair's cosine is the FIBRE's cos_ti, not |n.wi| -- the case a caller
        # dividing by |cos(dir,n)| gets silently wrong.
        var mat_h = tab.materials[unsafe_offset=Int(c.mat_idx)]
        var hc = _hair_precompute(mat_h, tab.curves, Int(c.hair_curve_idx), c.hair_v, c.hair_h, vwo)
        var (cos_ti, f_val, pdf_oc_fwd) = _hair_eval_lobes(
            dir_to_other, hc.tangent, hc.b_perp, hc.n_perp, hc.phi_o,
            hc.dphi0, hc.dphi1, hc.dphi2,
            hc.cos_tp0_o, hc.sin_tp0_o, hc.cos_tp1_o, hc.sin_tp1_o, hc.cos_tp2_o, hc.sin_tp2_o,
            hc.cos_theta_o, hc.sin_theta_o, hc.inv_vm0, hc.inv_vm1, hc.inv_vm2, hc.mp_c0, hc.mp_c1, hc.mp_c2, hc.s,
            hc.A0, hc.A1, hc.A2, hc.A3, hc.lum0, hc.lum1, hc.lum2, hc.lum3, hc.total_lum,
        )
        var hair_spec = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, f_val.r, f_val.g, f_val.b, wavelengths)
        var rev_h = Float32(0)
        comptime if want_pdfs:
            var hc_rev = _hair_precompute(mat_h, tab.curves, Int(c.hair_curve_idx), c.hair_v, c.hair_h, dir_to_other)
            var (cos_ti_rev, _, pdf_oc_rev) = _hair_eval_lobes(
                vwo, hc_rev.tangent, hc_rev.b_perp, hc_rev.n_perp, hc_rev.phi_o,
                hc_rev.dphi0, hc_rev.dphi1, hc_rev.dphi2,
                hc_rev.cos_tp0_o, hc_rev.sin_tp0_o, hc_rev.cos_tp1_o, hc_rev.sin_tp1_o, hc_rev.cos_tp2_o, hc_rev.sin_tp2_o,
                hc_rev.cos_theta_o, hc_rev.sin_theta_o, hc_rev.inv_vm0, hc_rev.inv_vm1, hc_rev.inv_vm2, hc_rev.mp_c0, hc_rev.mp_c1, hc_rev.mp_c2, hc_rev.s,
                hc_rev.A0, hc_rev.A1, hc_rev.A2, hc_rev.A3, hc_rev.lum0, hc_rev.lum1, hc_rev.lum2, hc_rev.lum3, hc_rev.total_lum,
            )
            rev_h = cos_ti_rev * pdf_oc_rev
        return LobeEval(hair_spec * cos_ti, cos_ti, cos_ti * pdf_oc_fwd, rev_h, True)

    if c.kind == LobeKind.measured:
        var vmat = tab.materials[unsafe_offset=Int(c.mat_idx)]
        var mb = tab.measured_brdfs[unsafe_offset=Int(vmat.measured_idx)]
        var frm = Frame.from_z(Vec3f(vn[0], vn[1], vn[2]))
        var tangent = Vec3f(frm.x.x, frm.x.y, frm.x.z)
        var bitangent = Vec3f(frm.y.x, frm.y.y, frm.y.z)
        var wo_l = Vec3f(dot(vwo, tangent), dot(vwo, bitangent), dot(vwo, vn))
        var wi_l = Vec3f(dot(dir_to_other, tangent), dot(dir_to_other, bitangent), dot(dir_to_other, vn))
        var fr_spec = SpectralSample(Float32(0))
        if c.adjoint:
            fr_spec = bxdf_eval_measured(mb, wi_l, wo_l, wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)[0]
        else:
            fr_spec = bxdf_eval_measured(mb, wo_l, wi_l, wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)[0]
        var cos_m = abs(dot(dir_to_other, vn))
        var fwd_m = Float32(0)
        var rev_m = Float32(0)
        comptime if want_pdfs:
            fwd_m = bxdf_pdf_measured(mb, wo_l, wi_l)
            rev_m = bxdf_pdf_measured(mb, wi_l, wo_l)
        return LobeEval(fr_spec * cos_m, cos_m, fwd_m, rev_m, True)

    if c.kind == LobeKind.diffuse_transmit:
        # TWO cosine lobes, one on each side of the surface. Until this
        # existed there was no LobeKind for diffuse transmission at all, so
        # every stored vertex of a `diffusetransmission` material fell
        # through to the opaque Lambertian branch below and lost its
        # transmit lobe outright -- exactly half the energy, in SPPM and VCM
        # alike, while the path tracer stayed correct because it shades via
        # bxdf_sample_diffuse_transmit directly and never round-trips a
        # stored vertex. The textbook PT-only feature gap.
        #
        # Crossing the surface is the POINT here, so the opaque sidedness
        # test below must NOT apply: `same_side` SELECTS the lobe rather
        # than rejecting the direction.
        var cos_dt = dot(dir_to_other, vn)
        var cos_wo_dt = dot(vwo, vn)
        if abs(cos_dt) <= Float32(1e-9) or abs(cos_wo_dt) <= Float32(1e-9):
            return LobeEval(ZERO, Float32(1), Float32(0), Float32(0), lobe_scoped(c))
        var same_side = cos_dt * cos_wo_dt > Float32(0)
        var trans_dt = _dt_transmittance(c, tab)
        var lobe_alb_dt = c.alb if same_side else trans_dt
        # The SAME luminance split bxdf_sample_diffuse_transmit samples with.
        # Deriving the density any other way lets MIS drift against the
        # sampler, which is the drift this interface exists to prevent.
        var pr_dt = c.alb.luma()
        var pt_dt = trans_dt.luma()
        var tot_dt = max(pr_dt + pt_dt, Float32(1e-9))
        var p_lobe_dt = (pr_dt if same_side else pt_dt) / tot_dt
        var cos_a_dt = abs(cos_dt)
        var alb_dt = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, lobe_alb_dt.r, lobe_alb_dt.g, lobe_alb_dt.b, wavelengths)
        var fwd_dt = Float32(0)
        var rev_dt = Float32(0)
        comptime if want_pdfs:
            fwd_dt = p_lobe_dt * cos_a_dt * INV_PI
            rev_dt = p_lobe_dt * abs(cos_wo_dt) * INV_PI
        return LobeEval(alb_dt * (INV_PI * cos_a_dt), cos_a_dt, fwd_dt, rev_dt, lobe_scoped(c))

    # Opaque Lambertian, and the coated base's fallback. ZERO across the
    # surface: `c.wo` is the direction the subpath ARRIVED from, so an opaque
    # lobe transports only to wo's own side of the normal. Returning |cos|
    # unconditionally made diffuse two-sided and TRANSMISSIVE. A degenerate wo
    # carries no sidedness information, so it falls back rather than silently
    # zeroing every contribution.
    var cos_o = dot(dir_to_other, vn)
    if (not c.pre_oriented) and dot(vwo, vwo) > Float32(1e-8) and cos_o * dot(vwo, vn) <= Float32(0):
        return LobeEval(ZERO, Float32(1), Float32(0), Float32(0), True)
    var cos_l = abs(cos_o)
    var alb_l = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, c.alb.r, c.alb.g, c.alb.b, wavelengths)
    return LobeEval(alb_l * (INV_PI * cos_l), cos_l,
                    cos_l * INV_PI, abs(dot(vwo, vn)) * INV_PI, True)


@always_inline
def lobe_kind_of(mat_type: Int8) -> Int32:
    """THE material -> lobe map for the kinds whose whole scattering goes
    through lobe_eval/lobe_sample. An integrator stores this on its vertex
    and never branches on the material again."""
    if mat_type == MatKind.diffuse_transmit:
        return LobeKind.diffuse_transmit
    if mat_type == MatKind.coated_diffuse:
        return LobeKind.layered
    if mat_type == MatKind.conductor:
        return LobeKind.ggx
    if mat_type == MatKind.measured:
        return LobeKind.measured
    if mat_type == MatKind.hair:
        return LobeKind.hair
    return LobeKind.lambertian


@always_inline
def lobe_is_available_of(mat: Material_C) -> Bool:
    """False when lobe_kind_of(mat.type) has no data to evaluate: a measured
    material whose .bsdf table did not load (measured_idx -1, which lobe_eval
    would otherwise index)."""
    return not (mat.type == MatKind.measured and mat.measured_idx < Int32(0))


@always_inline
def lobe_param_of(mat: Material_C) -> Float32:
    """LobeCtx.param for lobe_kind_of(mat.type): the GGX alpha of a conductor
    (isotropic, as lobe_eval evaluates it), 0 for kinds that take none."""
    if mat.type == MatKind.conductor:
        return max(mat.roughU, mat.roughV)
    return Float32(0)


@always_inline
def lobe_is_delta_of(mat: Material_C) -> Bool:
    """A smooth conductor is a mirror: lobe_sample returns a delta event and
    no vertex may be stored for it (the same threshold as
    bxdf_sample_conductor)."""
    return mat.type == MatKind.conductor and max(mat.roughU, mat.roughV) <= Float32(0.001)


@always_inline
def _dt_transmittance(c: LobeCtx, tab: LobeTables) -> RGB:
    """A diffusetransmission lobe's transmittance. It lives in the MATERIAL
    (Material_C.emission, see shade_diffuse_transmission), so it needs a real
    mat_idx; callers without one pass -1, which falls back to a symmetric
    lobe rather than index the table out of bounds. One texture slot serves
    both lobes when textured, matching shade_diffuse_transmission; c.alb is
    already the resolved one. Shared by lobe_eval and lobe_sample so the
    density one reports is the split the other draws from."""
    if Int(c.mat_idx) >= 0:
        var mat_dt = tab.materials[unsafe_offset=Int(c.mat_idx)]
        if Int(mat_dt.tex_idx) == -1:
            return mat_dt.emission
    return c.alb


@fieldwise_init
struct LobeSample(TrivialRegisterPassable):
    """One scattering decision at a vertex -- lobe_eval's sampling half.

    `weight` is what the throughput is multiplied by, f*|cos|/pdf. `pdf_fwd`
    and `pdf_rev` are the densities MIS needs (solid angle, both 0 for a delta
    event). For an analytic lobe they come FROM lobe_eval at the sampled
    direction, and so does the weight: sampling only chooses the direction,
    so a sampler and its evaluator cannot drift apart. A stochastic lobe
    (layered) is the exception: its f is itself an estimate, so the weight
    is the random walk's own throughput while the densities still come from
    lobe_eval, the value every other strategy sees."""
    var valid:    Bool
    var wi:       Vec3f
    var weight:   SpectralSample
    var is_delta: Bool
    var pdf_fwd:  Float32
    var pdf_rev:  Float32
    var cos_out:  Float32   # the lobe's own |cos| at wi
    var scoped:   Bool      # lobe_scoped(c), and nothing else


@always_inline
def _lobe_sample_invalid() -> LobeSample:
    return LobeSample(False, Vec3f(Float32(0)), SpectralSample(Float32(0)), False,
                      Float32(0), Float32(0), Float32(0), False)


def lobe_sample(
    c:   LobeCtx,
    uc:  Float32,
    u0:  Float32,
    u1:  Float32,
    u2:  Float32,
    tab: LobeTables,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    wavelengths: SampledWavelengths,
) -> LobeSample:
    """THE lobe sampler: the same LobeCtx lobe_eval takes, so the two see one
    vertex. `uc` picks among lobes, (u0, u1) the direction, and u2 is a
    fourth number only hair's logistic azimuth draws. Kinds not yet
    covered return valid=False; callers must not fall back to a sampler of
    their own, which is the drift this exists to end."""
    if not c.is_surface:
        return _lobe_sample_invalid()
    var vn = c.n
    var vwo = c.wo

    if c.kind == LobeKind.ggx:
        var cos_o = dot(vwo, vn)
        if cos_o <= Float32(0):
            return _lobe_sample_invalid()
        if c.is_delta or c.param <= Float32(0.001):
            # Smooth conductor: a mirror, with the same Schlick-on-f0 Fresnel
            # the rough lobe uses.
            var wi_m = vn * (Float32(2) * cos_o) - vwo
            var one_m = Float32(1) - cos_o
            var sch = one_m * one_m * one_m * one_m * one_m
            var f0 = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, c.alb.r, c.alb.g, c.alb.b, wavelengths)
            return LobeSample(True, wi_m, f0 * (Float32(1) - sch) + SpectralSample(sch), True,
                              Float32(0), Float32(0), cos_o, lobe_scoped(c))
        # Rough: the multiple-scattering cosine lobe or a VNDF reflection,
        # mixed exactly as bxdf_pdf_conductor_ggx (lobe_eval's density) says.
        var p_ms = Float32(1) - ggx_albedo(cos_o, c.param)
        var wi_g: Vec3f
        if uc < p_ms:
            wi_g = sample_cosine_hemisphere_world(u0, u1, vn)[0]
        else:
            var fr_g = Frame.from_z(Vec3f(vn[0], vn[1], vn[2]))
            var tx_g = Vec3f(fr_g.x.x, fr_g.x.y, fr_g.x.z)
            var ty_g = Vec3f(fr_g.y.x, fr_g.y.y, fr_g.y.z)
            var wo_l = Vec3f(dot(vwo, tx_g), dot(vwo, ty_g), cos_o)
            var wh_l = sample_ggx_vndf(wo_l, c.param, c.param, min(u0, Float32(0.9999)), u1)
            var wh = tx_g * wh_l.x + ty_g * wh_l.y + vn * wh_l.z
            wh = wh * (Float32(1) / sqrt(max(dot(wh, wh), Float32(1e-20))))
            wi_g = wh * (Float32(2) * dot(vwo, wh)) - vwo
        var le_g = lobe_eval[want_pdfs=True](c, wi_g, tab, spectral_coeffs, spectral_res,
            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
        if le_g.pdf_fwd <= Float32(0):
            return _lobe_sample_invalid()
        return LobeSample(True, wi_g, le_g.f_cos * (Float32(1) / le_g.pdf_fwd), False,
                          le_g.pdf_fwd, le_g.pdf_rev, le_g.cos_used, lobe_scoped(c))

    if c.is_delta:
        return _lobe_sample_invalid()

    if c.kind == LobeKind.hair:
        # The 3-lobe Marschner sampler (bvh.mojo). Its density is the fibre
        # cosine times pdf_over_cos, which is lobe_eval's pdf_fwd exactly.
        var mat_h = tab.materials[unsafe_offset=Int(c.mat_idx)]
        var hc = _hair_precompute(mat_h, tab.curves, Int(c.hair_curve_idx), c.hair_v, c.hair_h, vwo)
        var sh = _hair_sample_dir_u(hc, uc, u0, u1, u2)
        var le_h = lobe_eval[want_pdfs=True](c, sh[0], tab, spectral_coeffs, spectral_res,
            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
        if le_h.pdf_fwd <= Float32(0):
            return _lobe_sample_invalid()
        return LobeSample(True, sh[0], le_h.f_cos * (Float32(1) / le_h.pdf_fwd), False,
                          le_h.pdf_fwd, le_h.pdf_rev, le_h.cos_used, lobe_scoped(c))

    if c.kind == LobeKind.measured:
        # pbrt's MeasuredBxDF::Sample_f, in the same local frame lobe_eval
        # evaluates the table in. The weight divides by the density the
        # sampler actually drew from; f comes from lobe_eval, which applies
        # the adjoint argument order on the light subpath.
        var mb = tab.measured_brdfs[unsafe_offset=Int(tab.materials[unsafe_offset=Int(c.mat_idx)].measured_idx)]
        var fr_m = Frame.from_z(Vec3f(vn[0], vn[1], vn[2]))
        var tx_m = Vec3f(fr_m.x.x, fr_m.x.y, fr_m.x.z)
        var ty_m = Vec3f(fr_m.y.x, fr_m.y.y, fr_m.y.z)
        var wo_l = Vec3f(dot(vwo, tx_m), dot(vwo, ty_m), dot(vwo, vn))
        var (wi_l, _f_m, pdf_m, ok_m) = bxdf_sample_measured(mb, wo_l, u0, u1, wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
        if not ok_m or pdf_m <= Float32(0):
            return _lobe_sample_invalid()
        var wi_m = tx_m * wi_l[0] + ty_m * wi_l[1] + vn * wi_l[2]
        wi_m = wi_m * (Float32(1) / sqrt(max(dot(wi_m, wi_m), Float32(1e-20))))
        if dot(wi_m, vn) <= Float32(0):
            return _lobe_sample_invalid()
        var le_m = lobe_eval[want_pdfs=True](c, wi_m, tab, spectral_coeffs, spectral_res,
            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
        return LobeSample(True, wi_m, le_m.f_cos * (Float32(1) / pdf_m), False,
                          pdf_m, le_m.pdf_rev, le_m.cos_used, lobe_scoped(c))

    if c.kind == LobeKind.layered:
        var mat_ly = tab.materials[unsafe_offset=Int(c.mat_idx)]
        var fr = Frame.from_z(Vec3f(vn[0], vn[1], vn[2]))
        var tx = Vec3f(fr.x.x, fr.x.y, fr.x.z)
        var ty = Vec3f(fr.y.x, fr.y.y, fr.y.z)
        var wo_l = Vec3f(dot(vwo, tx), dot(vwo, ty), dot(vwo, vn))
        var R = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, c.alb.r, c.alb.g, c.alb.b, wavelengths)
        var bs = layered_sample(wo_l, uc, u0, u1, R, mat_ly.emission.r,
                                max(mat_ly.roughU, mat_ly.roughV), not c.adjoint)
        if not bs.valid or bs.pdf <= Float32(0):
            return _lobe_sample_invalid()
        var wi = tx * bs.wi.x + ty * bs.wi.y + vn * bs.wi.z
        var cos_out = abs(bs.wi.z)
        var w = bs.f * (cos_out / bs.pdf)
        if bs.specular:
            return LobeSample(True, wi, w, True, Float32(0), Float32(0), cos_out, lobe_scoped(c))
        var le = lobe_eval[want_pdfs=True](c, wi, tab, spectral_coeffs, spectral_res,
            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
        return LobeSample(True, wi, w, False, le.pdf_fwd, le.pdf_rev, cos_out, lobe_scoped(c))

    var bounce_n = vn
    if c.kind == LobeKind.lambertian:
        # The lobe lives on wo's side; lobe_eval zeroes the other one.
        if dot(vwo, vn) < Float32(0):
            bounce_n = -vn
    elif c.kind == LobeKind.diffuse_transmit:
        # The SAME luminance split lobe_eval reports as the lobe probability.
        var pr = c.alb.luma()
        var pt = _dt_transmittance(c, tab).luma()
        if pr + pt <= Float32(1e-9):
            return _lobe_sample_invalid()   # nothing to scatter; do not invent a lobe
        var wo_side = vn if dot(vwo, vn) >= Float32(0) else -vn
        bounce_n = wo_side if uc < pr / (pr + pt) else -wo_side
    else:
        return _lobe_sample_invalid()

    var s = sample_cosine_hemisphere_world(u0, u1, bounce_n)
    var wi = s[0]
    var le = lobe_eval[want_pdfs=True](c, wi, tab, spectral_coeffs, spectral_res,
        spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
    if le.pdf_fwd <= Float32(0):
        return _lobe_sample_invalid()
    return LobeSample(True, wi, le.f_cos * (Float32(1) / le.pdf_fwd), False,
                      le.pdf_fwd, le.pdf_rev, le.cos_used, lobe_scoped(c))


# Moved here from bdpt.mojo so that the ONE lobe evaluator can live below
# every integrator rather than inside one of them. It was the only piece
# of the dispatch that still lived above bxdf.
@always_inline
def _eval_conductor_ggx_spectral(
    n:     Vec3f,
    wo:    Vec3f,
    wi:    Vec3f,
    alpha: Float32,
    f0: RGB,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    wavelengths: SampledWavelengths,
) -> SpectralSample:
    """Spectral counterpart of _eval_conductor_ggx — same GGX/Schlick math,
    f0 converted to a SpectralSample (reflectance convention) before the
    Schlick blend instead of blending plain RGB triples."""
    var cos_o = dot(wo, n)
    var cos_i = dot(wi, n)
    if cos_o <= Float32(0) or cos_i <= Float32(0):
        return SpectralSample(Float32(0))
    var wh = wo + wi
    var whl = dot(wh, wh)
    if whl <= Float32(0):
        return SpectralSample(Float32(0))
    wh = wh * (Float32(1) / sqrt(whl))
    var cos_h = dot(wh, n)
    var cos_wo_h = dot(wo, wh)
    if cos_wo_h < Float32(0): cos_wo_h = -cos_wo_h
    var d = ggx_D(cos_h, alpha)
    var g = ggx_G2(cos_o, cos_i, alpha)
    var one_m = Float32(1) - cos_wo_h
    var one_m2 = one_m * one_m
    var schlick = one_m2 * one_m2 * one_m
    var f0_spec = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, f0.r, f0.g, f0.b, wavelengths)
    var fr_spec = f0_spec * (Float32(1) - schlick) + SpectralSample(schlick)
    var k = d * g / (Float32(4) * cos_o * cos_i) * cos_i
    # Kulla-Conty, tinted per hero wavelength rather than per RGB channel --
    # ggx_ms_lobe's own body, with f0_spec's four components in place of f0's
    # three. Multiplied by cos_i because this evaluator returns f*cos.
    var eavg = ggx_albedo_avg(alpha)
    var shape = ggx_ms_shape(cos_o, cos_i, alpha) * cos_i
    var ms = SpectralSample(shape * ggx_ms_tint(f0_spec.v0, eavg),
                            shape * ggx_ms_tint(f0_spec.v1, eavg),
                            shape * ggx_ms_tint(f0_spec.v2, eavg),
                            shape * ggx_ms_tint(f0_spec.v3, eavg))
    return fr_spec * k + ms


@always_inline
def coat_eval_smooth(n: Vec3f, wo: Vec3f, wi: Vec3f, alb: RGB, ior: Float32) -> RGB:
    """f(wo, wi) for a SMOOTH coat over a Lambertian base -- the evaluator
    gonzales never had.

    CoatWalk is a SAMPLER: `wi` is an output, and there is no way to ask it
    what the BSDF is for a given pair of directions. pbrt's LayeredBxDF is a
    full BxDF -- f(), Sample_f() AND PDF() -- and its f() runs a stochastic
    walk to answer exactly that question (Guo, Hasan & Zhao 2018). gonzales
    ported the sampling and not the evaluation, so every consumer that needs
    f(wo, wi) -- NEE, connections, vertex merging -- had to approximate it,
    and each approximated it differently.

    For THIS stack the stochastic walk is unnecessary: one smooth dielectric
    interface over a Lambertian base has a closed form, and it is exact
    rather than approximate, because a Lambertian base re-randomises to
    cosine at every bounce -- precisely the assumption the closed form makes.
    Scenes/coateddiffuse_analytic_check.py already validates the renderer
    against it:

        f = rho (1 - F(cos_o)) (1 - F(cos_i)) / (pi eta^2 (1 - rho F_di))

    The 1/(1 - rho*F_di) factor IS the TIR recycling series, summed in closed
    form. A merge sees one vertex and cannot sum that series by repetition
    the way per-iteration NEE does, which is why merging at a coated vertex
    measured 3x short without it.

    BOTH transmissions are explicit here. The sampled walk leaves the
    view-side one implicit because reaching the base at all costs an entry
    coin flip whose probability IS that factor (project_coateddiffuse_eta2_bug
    -- applying it twice there cost 56% at grazing exit). An evaluator has no
    coin flip, so it must carry both."""
    var cos_o = abs(dot(wo, n))
    var cos_i = abs(dot(wi, n))
    if cos_o <= Float32(1e-6) or cos_i <= Float32(1e-6):
        return RGB(Float32(0.0))
    var t_o = (Float32(1.0) - fr_dielectric(cos_o, ior)) * coat_beer_lambert_tr(
        cos_theta_t_dielectric(cos_o, ior), DEFAULT_COAT_THICKNESS)
    var t_i = (Float32(1.0) - fr_dielectric(cos_i, ior)) * coat_beer_lambert_tr(
        cos_theta_t_dielectric(cos_i, ior), DEFAULT_COAT_THICKNESS)
    var f_di = fdr_moment(ior)
    var k = t_o * t_i * INV_PI / max(ior * ior, Float32(1e-6))
    # The series is per-channel: a saturated base recycles more in the
    # channel it reflects most.
    return RGB(alb.r * k / max(Float32(1.0) - alb.r * f_di, Float32(1e-4)),
               alb.g * k / max(Float32(1.0) - alb.g * f_di, Float32(1e-4)),
               alb.b * k / max(Float32(1.0) - alb.b * f_di, Float32(1e-4)))



@always_inline
def bxdf_pdf_coated_exit(cos_out: Float32, ior: Float32) -> Float32:
    """Solid-angle pdf of a SMOOTH coateddiffuse walk's exit direction.

    `coat_walk_scatter` was annotated "layered exit pdf is intractable" and
    stores pdf=0, which is why coateddiffuse is excluded from VCM's real MIS
    scope (_bdpt_vertex_mis_scoped) and gets one unweighted connection
    instead. For a SMOOTH coat it is not intractable, because the exit
    direction's distribution does not depend on how many internal recycles
    happened: every recycle ends in another INDEPENDENT cosine sample of the
    Lambertian base, so the exit is always "cosine-sample the base, accept
    with probability 1-F(cos_up), refract out". Hence

        p_up(w_up)  = [cos_up/pi * (1 - F(cos_up))] / Z
        p_out(w_out) = p_up(w_up) * |dw_up/dw_out|

    with the etendue Jacobian |dw_up/dw_out| = cos_out / (ior^2 cos_up),
    in which cos_up cancels:

        p_out(w_out) = (1 - F(cos_up)) * cos_out / (pi * Z * ior^2)

    cos_up is recovered from cos_out by Snell (sin_up = sin_out / ior).
    Integrates to exactly 1 over the outside hemisphere -- substitute the
    same Jacobian back and it collapses to (1/Z) * int (cos/pi)(1-F) = 1.

    It is also INDEPENDENT of wo, so the forward and reverse pdfs a BDPT
    connection needs are the same function evaluated at the two directions.

    Rough coats additionally convolve with the exit microfacet's VNDF and
    are NOT covered: callers must check the coat is smooth first."""
    if cos_out <= Float32(0.0):
        return Float32(0.0)
    var sin2_out = max(Float32(0.0), Float32(1.0) - cos_out * cos_out)
    var sin2_up = sin2_out / (ior * ior)
    if sin2_up >= Float32(1.0):
        return Float32(0.0)
    var cos_up = sqrt(Float32(1.0) - sin2_up)
    var f_exit = fr_dielectric(cos_up, Float32(1.0) / ior)
    var z = coat_exit_norm(ior)
    if z <= Float32(1e-9):
        return Float32(0.0)
    return (Float32(1.0) - f_exit) * cos_out / (PI * z * ior * ior)

def coat_walk_begin(
    gn:    Vec3f,
    wo:    Vec3f,
    alb:   RGB,
    ior:   Float32,
    alpha: Float32,
    mut pcg: PCG32,
) -> CoatWalk:
    """Build the tangent frame and sample the coat's top-interface microfacet.
    Consumes 2 RNG draws when the coat is rough, none when smooth. Does NOT
    decide reflect-vs-transmit yet -- call the caller's own coat-lobe NEE
    first, then coat_walk_enter, to keep the historical draw order."""
    var is_rough = alpha > Float32(0.001)
    var frm = Frame.from_z(Vec3f(gn[0], gn[1], gn[2]))
    var tangent = Vec3f(frm.x.x, frm.x.y, frm.x.z)
    var bitangent = Vec3f(frm.y.x, frm.y.y, frm.y.z)
    var wm = gn
    if is_rough:
        var wo_l = Vec3f(dot(wo, tangent), dot(wo, bitangent), dot(wo, gn))
        var wm_l = sample_ggx_vndf(wo_l, alpha, alpha, pcg.next_float(), pcg.next_float())
        wm = tangent * wm_l.x + bitangent * wm_l.y + gn * wm_l.z
        var wmlen = dot(wm, wm)
        if wmlen > Float32(0.0):
            wm = wm * (Float32(1.0) / sqrt(wmlen))
    var cos_wm = dot(wo, wm)
    return CoatWalk(
        gn=gn, tangent=tangent, bitangent=bitangent, wo=wo, wm=wm, alb=alb,
        ior=ior, inv_ior=Float32(1.0) / ior, alpha=alpha, is_rough=is_rough,
        cos_o=dot(wo, gn), cos_wm=cos_wm, f_entry=fr_dielectric(cos_wm, ior),
        beta=RGB(Float32(1.0)), wi=Vec3f(Float32(0.0), Float32(0.0), Float32(0.0)),
        pdf=Float32(0.0), event=COAT_WALKING, depth=Int32(0),
    )

@always_inline
def coat_walk_enter(mut w: CoatWalk, mut pcg: PCG32):
    """The entry coin flip: reflect off the coat, or refract into it.

    Consumes 1 RNG draw. On REFLECT the throughput weight carries the rough
    lobe's G2/G1 masking-shadowing (weight 1 for a smooth coat, which is a
    delta lobe); the Fresnel factor itself is NOT applied -- the coin flip IS
    its estimator. On transmit, `beta` starts at the entry crossing's
    Beer-Lambert attenuation times that same G2/G1.

    TRAP, and the reason the two internal angles below differ: Beer-Lambert
    follows the SAMPLED FACET's refraction (cos_wm), because that is the path
    light actually takes through the coat; G2/G1 uses the MACRO normal's
    refraction (cos_o), matching how the light-side NEE weight models the same
    crossing with no sampled facet available. Conflating them is wrong in both
    directions."""
    if pcg.next_float() < w.f_entry:
        var refl = w.wm * (Float32(2.0) * w.cos_wm) - w.wo
        var rlen = dot(refl, refl)
        if rlen > Float32(0.0):
            refl = refl * (Float32(1.0) / sqrt(rlen))
        if dot(refl, w.gn) <= Float32(0.0):
            w.event = COAT_ABSORB      # reflected below the surface
            return
        w.wi = refl
        w.event = COAT_REFLECT
        if w.is_rough:
            w.beta *= ggx_G2(w.cos_o, dot(refl, w.gn), w.alpha) / ggx_G1(w.cos_o, w.alpha)
            w.pdf = ggx_vndf_pdf(w.cos_o, w.cos_wm, ggx_D(dot(w.gn, w.wm), w.alpha), w.alpha)
        else:
            w.pdf = Float32(-1.0)      # smooth mirror coat: delta lobe, no MIS
        return
    # Transmitted into the coat.
    w.beta = RGB(coat_beer_lambert_tr(cos_theta_t_dielectric(w.cos_wm, w.ior), DEFAULT_COAT_THICKNESS))
    if w.is_rough:
        w.beta *= ggx_G2(w.cos_o, cos_theta_t_dielectric(w.cos_o, w.ior), w.alpha) / ggx_G1(w.cos_o, w.alpha)
        # ... and put back the multiple-scattering energy that G2/G1, being a
        # single-scattering model, just dropped. SQRT because this is ONE of
        # the TWO crossings every complete path makes (in, then out -- whether
        # the way out is the walk's own exit or a light-side NEE crossing), and
        # coat_rough_ms_boost measures the deficit of the PAIR. Two sqrts at
        # the same angle multiply back to exactly the measured factor.
        w.beta *= sqrt(coat_rough_ms_boost(w.cos_o, w.alpha))

# ── Rough-coat multiple-scattering compensation ─────────────────────────────
# A rough coat's interface crossings are SINGLE-scattering: coat_walk_enter
# and coat_walk_scatter each weight a crossing by Smith G2/G1, which drops the
# light a microfacet masks. In reality that light bounces among the
# microfacets and mostly still gets through, so dropping it loses energy that
# grows with roughness -- on the uniform-env furnace with a WHITE base, where
# the coat can only absorb a few percent, BOTH integrators read:
#
#     roughness   alpha    PT       VCM
#     0.0         0.000    0.9149   0.9166      <- the coat's real absorption
#     0.1         0.316    0.7853   0.8305
#     1.0         1.000    0.6292   0.6608      <- a THIRD of the energy gone
#
# (alpha = sqrt(roughness), pbrt's remaproughness -- material_builder.mojo.)
# That is the same defect, and very nearly the same curve, that GGX
# conductors had before ggx_ms_lobe above: 0.794 at alpha 0.4, 0.327 at 1.0.
# Kulla-Conty restores it there as a second lobe; a WALK is a sampler and has
# no lobe to add, so the restoration is a weight on beta instead.
#
# The factor is E_walk(mu, 0) / E_walk(mu, alpha) -- the measured deficit of
# the ROUGH walk against the SMOOTH one at the same angle, so a rough coat
# ends up transporting what a smooth one does, which is the right target: the
# coat's absorption is a Beer-Lambert path-length effect that roughness
# barely moves, so roughness must not change the total. Exactly 1.0 at
# alpha 0, hence a true no-op for a smooth coat.
#
# FOOTGUN: Tools/coat_energy_table.mojo measures the RAW walk, so the table
# below is only valid while the walk is UNcompensated. Regenerating it with
# this compensation live measures a walk that already has it and converges to
# all-ones. Disable the call in coat_walk_enter before running the generator.
#
# Same uniform 11x11 grid and the same mu >= 0.1 clamp as elsewhere: below
# that the alpha=0 baseline collapses toward zero (near-total Fresnel
# reflection at grazing for a smooth coat) and the ratio to it is a division
# by nearly nothing. Clamped to [0.25, 4] for the same reason.
comptime _COAT_MS_GRID = 11


@always_inline
def _coat_ms_table() -> Array[Float32, 121]:
    """E_walk(mu,0)/E_walk(mu,alpha), mu-major (row i = mu = i/10, col
    j = alpha = j/10). Stack-built per call -- 121 stores, negligible beside
    the walk itself, and needs no mutable global state in a GPU kernel."""
    var t = Array[Float32, 121](fill=Float32(1.0))
    t[0] = Float32(1.000000)
    t[1] = Float32(0.250000)
    t[2] = Float32(0.250000)
    t[3] = Float32(0.250000)
    t[4] = Float32(0.250000)
    t[5] = Float32(0.250000)
    t[6] = Float32(0.250000)
    t[7] = Float32(0.250000)
    t[8] = Float32(0.250000)
    t[9] = Float32(0.250000)
    t[10] = Float32(0.250000)
    t[11] = Float32(1.000000)
    t[12] = Float32(0.717138)
    t[13] = Float32(0.625403)
    t[14] = Float32(0.609905)
    t[15] = Float32(0.616769)
    t[16] = Float32(0.631687)
    t[17] = Float32(0.648426)
    t[18] = Float32(0.667088)
    t[19] = Float32(0.685697)
    t[20] = Float32(0.705562)
    t[21] = Float32(0.724739)
    t[22] = Float32(1.000000)
    t[23] = Float32(0.946205)
    t[24] = Float32(0.911996)
    t[25] = Float32(0.918184)
    t[26] = Float32(0.940042)
    t[27] = Float32(0.968321)
    t[28] = Float32(1.000965)
    t[29] = Float32(1.034693)
    t[30] = Float32(1.068275)
    t[31] = Float32(1.101814)
    t[32] = Float32(1.134503)
    t[33] = Float32(1.000000)
    t[34] = Float32(1.006469)
    t[35] = Float32(1.029312)
    t[36] = Float32(1.065174)
    t[37] = Float32(1.106075)
    t[38] = Float32(1.149551)
    t[39] = Float32(1.194478)
    t[40] = Float32(1.240522)
    t[41] = Float32(1.286249)
    t[42] = Float32(1.329670)
    t[43] = Float32(1.374753)
    t[44] = Float32(1.000000)
    t[45] = Float32(1.024713)
    t[46] = Float32(1.074430)
    t[47] = Float32(1.131595)
    t[48] = Float32(1.189044)
    t[49] = Float32(1.245468)
    t[50] = Float32(1.300614)
    t[51] = Float32(1.355156)
    t[52] = Float32(1.405805)
    t[53] = Float32(1.459711)
    t[54] = Float32(1.512809)
    t[55] = Float32(1.000000)
    t[56] = Float32(1.028966)
    t[57] = Float32(1.089427)
    t[58] = Float32(1.156497)
    t[59] = Float32(1.224540)
    t[60] = Float32(1.289931)
    t[61] = Float32(1.353489)
    t[62] = Float32(1.410740)
    t[63] = Float32(1.468193)
    t[64] = Float32(1.528492)
    t[65] = Float32(1.586866)
    t[66] = Float32(1.000000)
    t[67] = Float32(1.028664)
    t[68] = Float32(1.091026)
    t[69] = Float32(1.162684)
    t[70] = Float32(1.235521)
    t[71] = Float32(1.304995)
    t[72] = Float32(1.369158)
    t[73] = Float32(1.432779)
    t[74] = Float32(1.496579)
    t[75] = Float32(1.553246)
    t[76] = Float32(1.616279)
    t[77] = Float32(1.000000)
    t[78] = Float32(1.027641)
    t[79] = Float32(1.089746)
    t[80] = Float32(1.162199)
    t[81] = Float32(1.235738)
    t[82] = Float32(1.305877)
    t[83] = Float32(1.372563)
    t[84] = Float32(1.437023)
    t[85] = Float32(1.499389)
    t[86] = Float32(1.557651)
    t[87] = Float32(1.623460)
    t[88] = Float32(1.000000)
    t[89] = Float32(1.026828)
    t[90] = Float32(1.088567)
    t[91] = Float32(1.159786)
    t[92] = Float32(1.231028)
    t[93] = Float32(1.299952)
    t[94] = Float32(1.366017)
    t[95] = Float32(1.431906)
    t[96] = Float32(1.492814)
    t[97] = Float32(1.552165)
    t[98] = Float32(1.611242)
    t[99] = Float32(1.000000)
    t[100] = Float32(1.026108)
    t[101] = Float32(1.085727)
    t[102] = Float32(1.153428)
    t[103] = Float32(1.223461)
    t[104] = Float32(1.290412)
    t[105] = Float32(1.353791)
    t[106] = Float32(1.414750)
    t[107] = Float32(1.476199)
    t[108] = Float32(1.534264)
    t[109] = Float32(1.594724)
    t[110] = Float32(1.000000)
    t[111] = Float32(1.025541)
    t[112] = Float32(1.082160)
    t[113] = Float32(1.149588)
    t[114] = Float32(1.217673)
    t[115] = Float32(1.282852)
    t[116] = Float32(1.345987)
    t[117] = Float32(1.405809)
    t[118] = Float32(1.466820)
    t[119] = Float32(1.523606)
    t[120] = Float32(1.580431)
    return t^


@always_inline
def coat_rough_ms_boost(mu: Float32, alpha: Float32) -> Float32:
    """Bilinear lookup into the compensation grid. Exactly 1.0 for a smooth
    coat, so applying it unconditionally costs one multiply and changes
    nothing there."""
    var t = _coat_ms_table()
    var mc = min(max(mu, Float32(0.1)), Float32(1.0))
    var ac = min(max(alpha, Float32(0.0)), Float32(1.0))
    var fm = mc * Float32(_COAT_MS_GRID - 1)
    var fa = ac * Float32(_COAT_MS_GRID - 1)
    var i0 = Int(fm)
    var j0 = Int(fa)
    var i1 = min(i0 + 1, _COAT_MS_GRID - 1)
    var j1 = min(j0 + 1, _COAT_MS_GRID - 1)
    var tm = fm - Float32(i0)
    var ta = fa - Float32(j0)
    var v0 = t[i0 * _COAT_MS_GRID + j0] * (Float32(1.0) - ta) + t[i0 * _COAT_MS_GRID + j1] * ta
    var v1 = t[i1 * _COAT_MS_GRID + j0] * (Float32(1.0) - ta) + t[i1 * _COAT_MS_GRID + j1] * ta
    return v0 * (Float32(1.0) - tm) + v1 * tm


@always_inline
def coat_walk_at_base(mut w: CoatWalk, mut pcg: PCG32) -> Bool:
    """Russian-roulette gate at the start of one recycle iteration. Returns
    True when the walk is sitting on the base layer and the caller may run its
    own NEE against `w.beta`; False when the walk has terminated (event is then
    COAT_ABSORB). Consumes 1 RNG draw only past depth 3 and only once the
    throughput has decayed, exactly as every hand-rolled copy of this loop
    did."""
    if w.depth >= Int32(COAT_MAX_DEPTH):
        w.event = COAT_ABSORB
        return False
    if w.depth > Int32(3):
        var beta_max = max(w.beta.r, max(w.beta.g, w.beta.b))
        if beta_max < Float32(0.25):
            var q_rr = max(Float32(0.0), Float32(1.0) - beta_max)
            if pcg.next_float() < q_rr:
                w.event = COAT_ABSORB
                return False
            w.beta = w.beta * (Float32(1.0) / (Float32(1.0) - q_rr))
    return True

@always_inline
def coat_walk_scatter(mut w: CoatWalk, mut pcg: PCG32):
    """One base bounce plus the attempt to leave the coat: cosine-sample the
    Lambertian base, attenuate by its albedo, then sample the underside
    microfacet and either refract out (event becomes COAT_EXIT, `wi` is the
    exit direction) or total-internally reflect and recycle for another
    iteration. Consumes 2 draws for the base direction, 2 more when rough for
    the exit facet, and 1 for the exit coin flip.

    The exit crossing costs ONE coat traversal; an internal reflection costs
    TWO (up to the underside, then back down to the base for the next
    iteration's NEE) -- hence tr_leg vs tr_leg squared."""
    var w_up = sample_cosine_hemisphere_world(pcg.next_float(), pcg.next_float(), w.gn)[0]
    w.beta *= w.alb
    var beta_max_c = max(w.beta.r, max(w.beta.g, w.beta.b))
    var beta_floor = beta_max_c * COAT_BETA_CHROMA_FLOOR
    if w.beta.r < beta_floor: w.beta.r = beta_floor
    if w.beta.g < beta_floor: w.beta.g = beta_floor
    if w.beta.b < beta_floor: w.beta.b = beta_floor
    var wm_e = w.gn
    if w.is_rough:
        var wup_l = Vec3f(dot(w_up, w.tangent), dot(w_up, w.bitangent), dot(w_up, w.gn))
        var wm_e_l = sample_ggx_vndf(wup_l, w.alpha, w.alpha, pcg.next_float(), pcg.next_float())
        wm_e = w.tangent * wm_e_l.x + w.bitangent * wm_e_l.y + w.gn * wm_e_l.z
        var wmelen = dot(wm_e, wm_e)
        if wmelen > Float32(0.0):
            wm_e = wm_e * (Float32(1.0) / sqrt(wmelen))
    var cos_up = dot(w_up, wm_e)
    var tr_leg = coat_beer_lambert_tr(cos_up, DEFAULT_COAT_THICKNESS)
    var f_exit = fr_dielectric(cos_up, w.inv_ior)
    w.depth += Int32(1)
    if pcg.next_float() < (Float32(1.0) - f_exit):
        var rr = refract(Vec3f(-w_up[0], -w_up[1], -w_up[2]),
                         Vec3f(-wm_e[0], -wm_e[1], -wm_e[2]), w.ior)
        if rr[0]:
            var wt = rr[1]
            var exit_dir = Vec3f(wt.x, wt.y, wt.z)
            var elen = dot(exit_dir, exit_dir)
            if elen > Float32(0.0):
                exit_dir = exit_dir * (Float32(1.0) / sqrt(elen))
            # A rough microfacet can refract below the surface; recycle then.
            if dot(exit_dir, w.gn) > Float32(0.0):
                w.beta *= tr_leg
                if w.is_rough:
                    var cos_up_n = dot(w_up, w.gn)
                    w.beta *= ggx_G2(cos_up_n, dot(exit_dir, w.gn), w.alpha) / ggx_G1(cos_up_n, w.alpha)
                    # The second crossing -- see coat_walk_enter's sqrt note.
                    w.beta *= sqrt(coat_rough_ms_boost(dot(exit_dir, w.gn), w.alpha))
                w.wi = exit_dir
                w.pdf = Float32(0.0)   # layered exit pdf is intractable -- NEE-only
                w.event = COAT_EXIT
                return
    # Internal reflection: recycled, two crossings this bounce.
    w.beta *= tr_leg * tr_leg
    if w.is_rough:
        var cos_up_r = dot(w_up, w.gn)
        w.beta *= ggx_G2(cos_up_r, cos_up_r, w.alpha) / ggx_G1(cos_up_r, w.alpha)

@always_inline
def _nee_weight_coated_coat_lobe(
    ls:         LightSample,
    ior:        Float32,
    coat_alpha: Float32,
    n:          Vec3f,
    wo:         Vec3f,
    mis: MisPolicy = mis_policy_power(),
) -> RGB:
    """NEE weight (throughput not applied) for a coateddiffuse/coated_conductor
    coat's own glossy GGX lobe (D*G2*F/(4*cos_o)) against ONE LightSample --
    see docs/05_reflection_models.md for the model. Delta lights: MIS weight
    1. Real-pdf lights: power heuristic against ggx_vndf_pdf. Caller must
    skip this for a smooth coat (delta reflection can't land on a
    stochastic light sample)."""
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
    # This lobe's REVERSE density, toward wo -- the same VNDF density with the
    # two directions swapped. Only VCM's policy reads it (it is the dVC term's
    # multiplier); the path tracer's power heuristic never looks.
    var mis_l = mis
    if mis.is_vcm:
        mis_l.pdf_rev_w = ggx_vndf_pdf(cos_s, dot(ls.wi, wm), d, coat_alpha)
    if ls.is_delta:
        # A delta light cannot be found by BSDF sampling, so for a PATH TRACER
        # this is the sole strategy and weight 1 is right -- and stays right,
        # since the default policy makes nee_mis_weight return
        # power_heuristic(1, 0) = 1. It is NOT right for VCM: merging and t=1
        # light tracing reach this same vertex and compete for the same
        # photons, exactly as the coat EXIT vertex's own simple-light NEE
        # already documents ("The sun is delta ... so it takes a balance share
        # rather than weight 1"). This lobe returned before ever consulting
        # `mis`, so it took FULL weight beside strategies that had already
        # reserved their share -- and it fires ONLY for a rough coat, which is
        # the roughness-gated excess measured on barcelona-pavilion.
        return ls.Li * (f_cos * nee_mis_weight(mis_l, Float32(1.0), Float32(0.0), cos_s))
    if ls.pdf <= Float32(0.0):
        return RGB(Float32(0.0))
    var pdf_bsdf = ggx_vndf_pdf(cos_o, cos_wm, d, coat_alpha)
    var w = nee_mis_weight(mis_l, ls.pdf, pdf_bsdf, cos_s)
    return ls.Li * (f_cos * w / ls.pdf)

@always_inline
def _nee_weight_coated_diffuse_base[nee_is_sole_strategy: Bool = False](
    ls:  LightSample,
    alb: RGB,
    ior: Float32,
    n:   Vec3f,
    coat_alpha: Float32 = Float32(0.0),
    mis: MisPolicy = mis_policy_power(),
) -> RGB:
    """NEE weight (throughput AND the walk's `beta` not applied) for a
    coateddiffuse base against ONE LightSample -- see
    docs/05_reflection_models.md for the eta^2/Fresnel derivation. Kept
    separate from _nee_weight_simple because the coat transmittance and
    `beta` walk state don't fit bxdf_eval_any's flat signature.

    `nee_is_sole_strategy` (comptime) states whether a BSDF-sampling
    strategy competes with NEE for this light AT THIS VERTEX -- a property
    of the calling integrator's own bookkeeping, statically known per call
    site, so it compiles the branch out rather than costing a runtime test:

    - True  -- the caller guarantees its continuation ray carries
      lastBsdfPdf = 0, so every miss/emitter handler drops that ray's
      direct-light contribution (`power_heuristic(0, pdf_light)` = 0, and
      the area-light handler's own `if pdf_bsdf > 0` skips it outright).
      NEE is then the ONLY strategy contributing this light and must carry
      the full weight. shading.mojo's `shade_coated_diffuse` is exactly
      this case: the layered exit ray's true pdf is intractable, so it
      zeroes it deliberately. Splitting the weight against a competing pdf
      that no longer exists silently discards a real fraction of the light
      with nothing else picking it up.
    - False -- ordinary two-strategy MIS against a cosine-lobe pdf, the
      historical behaviour, kept as the default so bdpt.mojo's own call
      sites (which have their own separate dVCM/dVC MIS story) are
      unchanged.

    TRAP: applies the LIGHT-side coat transmittance only. The VIEW-side one
    is already supplied, in expectation, by the caller's entry coin flip
    (`if u < f_entry: reflect; return`) -- reaching this walk at all costs
    exactly one factor of `1 - F(cos_o)`. Applying it here too squares it,
    which is invisible at normal incidence (0.96 -> 0.92 at eta 1.5) and
    catastrophic at grazing exit, where F -> 1: it cost 56% of the energy at
    a 4-degree view. Same double-count the coat-lobe NEE's own docstring
    warns about, and the env-map NEE in the same loop already gets this
    right ("view-side coat transmittance is implicit in reaching this
    branch")."""
    if not ls.valid:
        return RGB(Float32(0.0))
    var cos_s = dot(n, ls.wi)
    if cos_s <= Float32(0.0):
        return RGB(Float32(0.0))
    var t_light = Float32(1.0) - fr_dielectric(cos_s, ior)
    # Light-side Fresnel transmission AND 1/eta^2 -- see
    # docs/05_reflection_models.md. TRAP: 1/eta^2 lives HERE, not folded into
    # the caller's walk `beta`, because beta is what the coat loop's RR and
    # chrominance floor threshold against (`beta_max < 0.25`) -- scaling it
    # by 1/eta^2 (0.25 at eta 2) would fire RR before the walk even starts.
    # Coat-thickness Beer-Lambert attenuation for the light's descent through
    # the coat to reach the base -- see coat_beer_lambert_tr's docstring
    # (geometry.mojo) and pbrt's LayeredBxDF::Tr, applied via the light's
    # REFRACTED internal angle, not its external one.
    var cos_s_internal = cos_theta_t_dielectric(cos_s, ior)
    var tr_light = coat_beer_lambert_tr(cos_s_internal, DEFAULT_COAT_THICKNESS)
    if coat_alpha > Float32(0.001):
        # Rough transmission weight G2(wo,wi)/G1(wo), same as every other
        # rough crossing in shade_coated_diffuse.
        t_light *= ggx_G2(cos_s, cos_s_internal, coat_alpha) / ggx_G1(cos_s, coat_alpha)
        # The light-side crossing is the OTHER of the two a NEE path makes --
        # the walk's entry is the first. Same sqrt split; see
        # coat_walk_enter's note.
        t_light *= sqrt(coat_rough_ms_boost(cos_s, coat_alpha))
    var t_both = t_light * tr_light / max(ior * ior, Float32(1e-6))
    if ls.is_delta:
        return alb * ls.Li * (cos_s * t_both / PI)
    if ls.pdf <= Float32(0.0):
        return RGB(Float32(0.0))
    # MIS weight. `nee_is_sole_strategy` is the CALLER's guarantee that no
    # BSDF-sampling strategy competes for this light at this vertex -- see
    # the parameter's docstring above. Only the weight is affected; the
    # 1/pdf of the light sample itself is always required.
    var w = Float32(1.0)
    comptime if not nee_is_sole_strategy:
        # The competing BSDF strategy here is the coat walk's EXIT, whose
        # density is bxdf_pdf_coated_exit -- not the bare cosine lobe of the
        # base underneath it. Passing cos/pi understates the competitor and
        # so overstates this sample's share.
        w = nee_mis_weight(mis, ls.pdf, bxdf_pdf_coated_exit(cos_s, ior), cos_s)
    return alb * ls.Li * (cos_s * t_both * w / (ls.pdf * PI))

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
# BY VALUE across a real Mojo function-call boundary was suspected of a
# miscompilation (modular/modular#6759, one field observed corrupted on a
# random subset of runs; later retracted by its own author as
# unreproducible -- kept decomposed defensively regardless). Callers hold a
# SpectralHandle (e.g. shading.mojo's
# ctx.spectral) and pass its fields individually: ctx.spectral.coeffs,
# ctx.spectral.res, ctx.spectral.cie_x, ctx.spectral.cie_y, ctx.spectral.cie_z,
# ctx.spectral.d65.
@always_inline
def nee_weight_lobe(
    ls:    LightSample,
    c:     LobeCtx,
    tab:   LobeTables,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    wavelengths: SampledWavelengths,
    mis: MisPolicy = mis_policy_power(),
) -> SpectralSample:
    """THE NEE weight: f*cos*Li/pdf times the MIS weight, for any lobe, from
    the same LobeCtx lobe_eval and lobe_sample take -- so a stored vertex's
    NEE, connections and merges all see one vertex (hair's curve fields
    included, which a flat (kind, alb, alpha) signature cannot carry).

    Opaque lobes reject a light behind the shading normal; diffuse_transmit
    and hair transport to both sides, so for them lobe_eval decides."""
    if not ls.valid:
        return SpectralSample(Float32(0.0))
    var cos_s = dot(c.n, ls.wi)
    var two_sided = c.kind == LobeKind.diffuse_transmit or c.kind == LobeKind.hair
    if (cos_s <= Float32(0.0) and not two_sided) or (two_sided and abs(cos_s) <= Float32(0.0) and c.kind != LobeKind.hair):
        return SpectralSample(Float32(0.0))
    var le = lobe_eval[want_pdfs=True](c, ls.wi, tab,
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, wavelengths)
    var f = le.f_cos
    var pdf_bsdf = le.pdf_fwd
    if f.v0 <= Float32(0.0) and f.v1 <= Float32(0.0) and f.v2 <= Float32(0.0) and f.v3 <= Float32(0.0):
        return SpectralSample(Float32(0.0))
    var li_spectral = rgb_illuminant_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, ls.Li.r, ls.Li.g, ls.Li.b, wavelengths)
    # `f` is f*cos already (LobeEval.f_cos), so the cosine is NOT applied
    # again here -- that double application is exactly the 2/3 energy loss
    # vertex merging had before cos_used made the convention explicit.
    # The MIS cosine is the lobe's own (hair: the fibre cosine).
    var cos_mis = le.cos_used if c.kind == LobeKind.hair else abs(cos_s)
    if ls.is_delta:
        # A delta light cannot be found by BSDF sampling, so a path tracer
        # gives its NEE full weight. VCM cannot: merging and t=1 light tracing
        # still compete for that photon, so the sample takes its balance
        # share with the BSDF term absent -- SmallVCM's DirectIllumination has
        # wLight = 0 for a delta light and wCamera intact. Weight 1 here was a
        # straight double count against every sun photon in the cache.
        if not mis.is_vcm:
            return f * li_spectral
        return f * li_spectral * nee_mis_weight(mis, Float32(1.0), Float32(0.0), cos_mis)
    var mis_w = nee_mis_weight(mis, ls.pdf, pdf_bsdf, cos_mis)
    return (f * li_spectral) * (mis_w / ls.pdf)


@always_inline
def _nee_weight_simple_spectral(
    ls:    LightSample,
    mat_kind: Int32,
    alb:   RGB,
    alpha: Float32,
    n:     Vec3f,
    wo:    Vec3f,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    wavelengths: SampledWavelengths,
    tab: LobeTables,
    mis: MisPolicy = mis_policy_power(),
    mat_idx: Int32 = Int32(-1),
) -> SpectralSample:
    """Spectral counterpart of _nee_weight_simple — same formula, but the
    material color and light color are each converted to a SpectralSample at
    this path's hero wavelengths (rgb_to_spectral_sample for the reflectance,
    rgb_illuminant_to_spectral_sample for the light's unbounded radiance/
    intensity) before multiplying, instead of multiplying plain RGB
    triples."""
    return nee_weight_lobe(ls,
        LobeCtx(mat_kind, True, False, n, wo, alb, mat_idx, alpha,
                Float32(0), Int32(-1), Float32(0), Float32(0), True, False),
        tab, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, wavelengths, mis)


@always_inline
def _nee_weight_hair(
    ls: LightSample, hc: HairLobeConstants,
    mis: MisPolicy = mis_policy_power(),
) -> RGB:
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
    # `cos_ti` is the FIBRE cosine, not |n.wi| -- hair's lobe has no surface
    # normal to take one against. Same distinction LobeEval.cos_used exists
    # for in bdpt.mojo, and the reason a caller must never guess it.
    var mis_w = nee_mis_weight(mis, ls.pdf, pdf_bsdf, cos_ti)
    return f_val * cos_ti * ls.Li * (mis_w / ls.pdf)
