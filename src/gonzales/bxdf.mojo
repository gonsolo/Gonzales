from std.math import sqrt
from .layered import diel_f, diel_pdf, diel_sample, layered_f, layered_pdf, layered_sample
from .geometry import RGB, Vec3f, dot, INV_PI, PI, Frame, INV_FOUR_PI
from .materials import LobeKind, MatKind, Material, fr_dielectric, MeasuredBRDF, sss_boundary_lobe
from .curves import Curve
from .sampling import sample_ggx_vndf, sample_cosine_hemisphere_world
from .vcm_mis import MisPolicy, mis_policy_power, nee_mis_weight
from .measured_bxdf_eval import bxdf_eval_measured, bxdf_pdf_measured, bxdf_sample_measured
from .bvh import LightSample, HairLobeConstants, _hair_eval_lobes, _hair_precompute, _hair_sample_dir_u
from .spectrum import SampledWavelengths, SpectralHandle, SpectralSample, rgb_to_spectral_sample, spec_refl_unbounded, rgb_illuminant_to_spectral_sample, rgb_bands_to_spectral_sample

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
    bdpt_*.mojo's own connection-formula variant, which folds cos_i in for its
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
    mat: Material,
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
    PathState.previous_dielectric_ior's docstring).

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
# _sppm_vp_brdf, promoted to a shared function so shading.mojo/bdpt_*.mojo/
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
    per-material NEE function in shading.mojo/bdpt_*.mojo/sppm.mojo already
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

@fieldwise_init
struct LobeTables(TrivialRegisterPassable):
    """The three scene tables a lobe evaluation can read, and nothing else.

    _lobe_eval took these as three loose pointers because the integrators do
    not share a context type: VCM has SceneView (39 fields, passed
    by ref), the path tracer has ShadeContext (23 fields, passed by value),
    and they carry ELEVEN of the same scene pointers between them.

    Merging those two is a ~979-site rename across by-value GPU structs with
    a documented crash history (vcm_mat_kind_and_sd_by_value_traps), so this
    takes the narrow win instead: one named thing to pass, produced from
    either context by a one-line accessor. If the two contexts are ever
    reconciled, this is what they should agree on first."""
    var materials:      Pointer[Material, MutUntrackedOrigin]
    var curves:         Pointer[Curve, MutUntrackedOrigin]
    var measured_brdfs: Pointer[MeasuredBRDF, MutUntrackedOrigin]
    var spectral:       SpectralHandle   # RGB to spectrum for the lobes' colours


trait NeeLobe(TrivialRegisterPassable):
    """What light sampling asks of a shading point: a light sample's contribution, f*cos*Li/pdf times its MIS weight."""
    def nee_weight(self, ls: LightSample, tab: LobeTables, wl: SampledWavelengths) -> SpectralSample: ...
    def faces(self, ls: LightSample) -> Bool: ...      # the light sample can contribute at all
    def costly(self) -> Bool: ...                      # evaluating costs more than a shadow ray


@fieldwise_init
struct BSDF(NeeLobe, TrivialRegisterPassable):
    """Everything the ONE lobe evaluator needs about a shading point, and
    nothing about which integrator is asking.

    This is the BxDF interface's input. A path tracer builds one from its
    local shading variables, VCM from a stored BDPTVertex, SPPM from a
    visible point -- and they all then get the same f, the same densities and
    the same answer to "is this lobe MIS-scoped". Before it existed each
    integrator carried its own dispatch over LobeKind and they drifted; every
    VCM defect found on 2026-09-20 lived in that drift.

    `param` is the per-kind scalar, the way BDPTVertex already overloaded
    pdf_bwd: GGX alpha for a conductor, coat alpha for
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

    @always_inline
    def f[want_pdfs: Bool = True](self, wi: Vec3f, tab: LobeTables, wl: SampledWavelengths) -> LobeEval:
        """f*cos toward `wi`, the densities both ways and whether they may take part in MIS."""
        var s = tab.spectral
        return _lobe_eval[want_pdfs](self, wi, tab, s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65, wl)

    @always_inline
    def sample[want_rev: Bool = True](self, uc: Float32, u0: Float32, u1: Float32, u2: Float32, tab: LobeTables,
               wl: SampledWavelengths) -> LobeSample:
        """A scattering direction and its weight f*cos/pdf; `uc` picks the lobe, u2 only hair draws."""
        var s = tab.spectral
        return _lobe_sample[want_rev](self, uc, u0, u1, u2, tab, s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65, wl)

    @always_inline
    def scoped(self) -> Bool:
        """Real forward and reverse densities exist, so the vertex may take part in MIS."""
        return _lobe_scoped(self)

    @always_inline
    def nee_weight(self, ls: LightSample, tab: LobeTables, wl: SampledWavelengths) -> SpectralSample:
        """A light sample's contribution through this BSDF, f*cos*Li/pdf times its power-heuristic weight."""
        return self.nee_weight_mis(ls, tab, wl, mis_policy_power())

    @always_inline
    def faces(self, ls: LightSample) -> Bool:
        return _nee_faces(ls, self)

    @always_inline
    def costly(self) -> Bool:
        """The layered lobe's f and pdf are random walks."""
        return self.kind == LobeKind.layered

    @always_inline
    def nee_weight_mis(self, ls: LightSample, tab: LobeTables, wl: SampledWavelengths, mis: MisPolicy) -> SpectralSample:
        """nee_weight under another integrator's MIS policy."""
        var s = tab.spectral
        return _nee_weight_lobe(ls, self, tab, s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65, wl, mis)


@fieldwise_init
struct HairBSDF(NeeLobe, TrivialRegisterPassable):
    """A hair BSDF with its fibre setup made once for wo, for a shading point that lights and samples it repeatedly."""
    var bsdf: BSDF
    var hc:   HairLobeConstants

    @staticmethod
    @always_inline
    def at(bsdf: BSDF, mat: Material, tab: LobeTables) -> HairBSDF:
        """`mat` as the caller shades it, e.g. after Filter Glossy widened its lobes."""
        return HairBSDF(bsdf, _hair_precompute(mat, tab.curves, Int(bsdf.hair_curve_idx), bsdf.hair_v, bsdf.hair_h, bsdf.wo))

    @always_inline
    def faces(self, ls: LightSample) -> Bool:
        return _nee_faces(ls, self.bsdf)

    @always_inline
    def costly(self) -> Bool:
        return False

    @always_inline
    def nee_weight(self, ls: LightSample, tab: LobeTables, wl: SampledWavelengths) -> SpectralSample:
        if not _nee_faces(ls, self.bsdf):
            return SpectralSample(Float32(0.0))
        var s = tab.spectral
        var le = _hair_lobe_eval[True, False](self.bsdf, self.hc, ls.wi, tab, s.coeffs, s.res, s.cie_x, s.cie_y,
                                             s.cie_z, s.d65, wl)
        return _nee_weight_of(ls, self.bsdf, le, s.coeffs, s.res, s.cie_x, s.cie_y, s.cie_z, s.d65, wl, mis_policy_power())

    @always_inline
    def sample[want_rev: Bool = True](self, uc: Float32, u0: Float32, u1: Float32, u2: Float32, tab: LobeTables,
                                      wl: SampledWavelengths) -> LobeSample:
        var s = tab.spectral
        return _hair_lobe_sample[want_rev](self.bsdf, self.hc, uc, u0, u1, u2, tab, s.coeffs, s.res, s.cie_x,
                                           s.cie_y, s.cie_z, s.d65, wl)


@fieldwise_init
struct LobeEval(TrivialRegisterPassable):
    """One vertex lobe, evaluated ONCE: throughput, the cosine that throughput
    already contains, both densities, and whether it has real densities at all.

    This exists because the same dispatch over LobeKind was written out THREE
    times -- `_eval_vertex_spectral` (f*cos), `_bdpt_vertex_pdfs`
    (forward/reverse densities) and `_bdpt_vertex_mis_scoped` (which kinds
    have densities) -- and the three lists drifted apart. Every VCM defect
    found on 2026-09-20 lived in that drift:

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
def _lobe_scoped(c: BSDF) -> Bool:
    """Does this lobe have REAL forward/reverse densities, i.e. may it take
    part in MIS? THE one list -- _lobe_eval reports it as LobeEval.scoped and
    every integrator asks this, so a kind cannot be in scope for one consumer
    and out of it for another.

    coateddiffuse is the layered lobe; dielectrics are delta and never in scope.

    Volume (isotropic phase): always in scope. Unlike a surface lobe there is
    no kind dispatch -- every medium here is hardcoded isotropic (1/4pi),
    never delta -- and the forward/reverse densities are always real and
    exact. See Scenes/vcm_volume_mis_derivation.py (connections+merges exact
    to 4.4e-16 including every mixed surface/volume path)."""
    if not c.is_surface:
        return True
    if c.kind == LobeKind.layered:
        return c.is_surface
    if not c.is_surface or c.is_delta:
        return False
    # diffuse_transmit: _lobe_eval returns its real two-lobe densities (the
    # luminance split times each side's cosine). It was missing here while
    # _lobe_eval hard-coded True for it, so VCM's NEE at a leaf weighted
    # itself against connections that _connect then ran UNWEIGHTED: floor
    # lit only through a diffusetransmission panel read 1.8x pbrt.
    return (c.kind == LobeKind.lambertian or c.kind == LobeKind.ggx
            or c.kind == LobeKind.hair or c.kind == LobeKind.measured
            or c.kind == LobeKind.diffuse_transmit
            or c.kind == LobeKind.rough_dielectric or c.kind == LobeKind.sss_exit)


@always_inline
def _layered_pdfs[want_rev: Bool](c: BSDF, mat_ly: Material, wo_l: Vec3f, wi_l: Vec3f, alpha: Float32) -> Tuple[Float32, Float32]:
    """A layered lobe's densities toward wi_l and, when `want_rev`, back toward wo_l; local frame."""
    var ior = mat_ly.emission.r
    var rad = not c.adjoint
    var ab = max(mat_ly.roughU, mat_ly.roughV)
    var fwd: Float32
    var rev = Float32(0)
    if mat_ly.type == MatKind.coated_conductor:
        fwd = layered_pdf[True](wo_l, wi_l, ior, alpha, rad, ab)
        comptime if want_rev:
            rev = layered_pdf[True](wi_l, wo_l, ior, alpha, not rad, ab)
    else:
        fwd = layered_pdf(wo_l, wi_l, ior, alpha, rad)
        comptime if want_rev:
            rev = layered_pdf(wi_l, wo_l, ior, alpha, not rad)
    return (fwd, rev)


@always_inline
def _hair_lobe_eval[want_pdfs: Bool, want_rev: Bool](
    c: BSDF, hc: HairLobeConstants, dir_to_other: Vec3f, tab: LobeTables,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    wavelengths: SampledWavelengths,
) -> LobeEval:
    """_lobe_eval's hair lobe from a fibre setup already made for wo. Its cosine is the fibre's cos_ti, not |n.wi|."""
    var (cos_ti, f_val, pdf_oc_fwd) = _hair_eval_lobes(
        dir_to_other, hc.tangent, hc.b_perp, hc.n_perp, hc.phi_o,
        hc.dphi0, hc.dphi1, hc.dphi2,
        hc.cos_tp0_o, hc.sin_tp0_o, hc.cos_tp1_o, hc.sin_tp1_o, hc.cos_tp2_o, hc.sin_tp2_o,
        hc.cos_theta_o, hc.sin_theta_o, hc.inv_vm0, hc.inv_vm1, hc.inv_vm2, hc.mp_c0, hc.mp_c1, hc.mp_c2, hc.s,
        hc.A0, hc.A1, hc.A2, hc.A3, hc.lum0, hc.lum1, hc.lum2, hc.lum3, hc.total_lum,
    )
    # f, not a reflectance: a narrow R or TT lobe peaks far above 1, and the [0,1] reflectance upsampler cut it there.
    var hair_spec = spec_refl_unbounded(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, f_val.r, f_val.g, f_val.b, wavelengths)
    var fwd_h = Float32(0)
    var rev_h = Float32(0)
    comptime if want_pdfs:
        fwd_h = _hair_mis_pdf(c, tab, c.wo, dir_to_other)
        comptime if want_rev:
            rev_h = _hair_mis_pdf(c, tab, dir_to_other, c.wo)
    _ = pdf_oc_fwd
    return LobeEval(hair_spec * cos_ti, cos_ti, fwd_h, rev_h, True)


@always_inline
def _lobe_eval[want_pdfs: Bool = True, want_rev: Bool = True](
    c:   BSDF,
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
    SceneView, the path tracer has ShadeContext. Depending on one of
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

    if c.kind == LobeKind.rough_dielectric:
        # pbrt's DielectricBxDF over a Trowbridge-Reitz microsurface
        # (layered.mojo's diel_*). `vn` is the OUTWARD normal -- inside vs
        # outside is which side of it wo lies on, so a caller must not
        # face-forward it. Reflection and transmission both, untinted; the
        # IOR lives in the material (albedo.r, like smooth glass). A light
        # vertex evaluates importance transport: no 1/eta^2.
        var ior_rd = tab.materials[unsafe_offset=Int(c.mat_idx)].albedo.r
        var fr_rd = Frame.from_z(Vec3f(vn[0], vn[1], vn[2]))
        var tx_rd = Vec3f(fr_rd.x.x, fr_rd.x.y, fr_rd.x.z)
        var ty_rd = Vec3f(fr_rd.y.x, fr_rd.y.y, fr_rd.y.z)
        var wo_rd = Vec3f(dot(vwo, tx_rd), dot(vwo, ty_rd), dot(vwo, vn))
        var wi_rd = Vec3f(dot(dir_to_other, tx_rd), dot(dir_to_other, ty_rd), dot(dir_to_other, vn))
        var f_rd = diel_f(wo_rd, wi_rd, ior_rd, c.param, not c.adjoint)
        var cos_rd = abs(wi_rd.z)
        var fwd_rd = Float32(0)
        var rev_rd = Float32(0)
        comptime if want_pdfs:
            fwd_rd = diel_pdf(wo_rd, wi_rd, ior_rd, c.param, True, True)
            comptime if want_rev:
                rev_rd = diel_pdf(wi_rd, wo_rd, ior_rd, c.param, True, True)
        return LobeEval(SpectralSample(f_rd * cos_rd), cos_rd, fwd_rd, rev_rd, True)

    if c.kind == LobeKind.layered:
        # pbrt's LayeredBxDF (layered.mojo). Everything is real: f, and the
        # pdf estimator both ways, which is a deterministic function of the
        # directions (hash-seeded), so every strategy that asks gets one value.
        # Light-side vertices evaluate the adjoint (importance transport).
        var mat_ly = tab.materials[unsafe_offset=Int(c.mat_idx)]
        var ior_ly = mat_ly.emission.r
        var alpha_ly = c.param if c.param > Float32(0) else mat_ly.coat_alpha()   # param>0: caller forces the top alpha
        var cond_ly = mat_ly.type == MatKind.coated_conductor
        var ab_ly = max(mat_ly.roughU, mat_ly.roughV)
        var fr_ly = Frame.from_z(Vec3f(vn[0], vn[1], vn[2]))
        var tx_ly = Vec3f(fr_ly.x.x, fr_ly.x.y, fr_ly.x.z)
        var ty_ly = Vec3f(fr_ly.y.x, fr_ly.y.y, fr_ly.y.z)
        var wo_ly = Vec3f(dot(vwo, tx_ly), dot(vwo, ty_ly), dot(vwo, vn))
        var wi_ly = Vec3f(dot(dir_to_other, tx_ly), dot(dir_to_other, ty_ly), dot(dir_to_other, vn))
        var rad_ly = not c.adjoint
        var R_ly = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, c.alb.r, c.alb.g, c.alb.b, wavelengths)
        var f_ly = layered_f[True](wo_ly, wi_ly, R_ly, ior_ly, alpha_ly, rad_ly, mat_ly.emission.g, ab_ly) if cond_ly \
            else layered_f(wo_ly, wi_ly, R_ly, ior_ly, alpha_ly, rad_ly, mat_ly.emission.g)
        var cos_ly = abs(wi_ly.z)
        var fwd_ly = Float32(0)
        var rev_ly = Float32(0)
        comptime if want_pdfs:
            var pd = _layered_pdfs[want_rev](c, mat_ly, wo_ly, wi_ly, alpha_ly)
            fwd_ly = pd[0]
            rev_ly = pd[1]
        return LobeEval(f_ly * cos_ly, cos_ly, fwd_ly, rev_ly, True)
    if c.kind == LobeKind.sss_exit:
        var cos_x = abs(dot(dir_to_other, vn))
        return LobeEval(SpectralSample(sss_boundary_lobe(cos_x, c.param) * INV_PI * cos_x),
                        cos_x, cos_x * INV_PI, Float32(1) - fr_dielectric(cos_x, c.param), _lobe_scoped(c))

    if c.kind == LobeKind.ggx:
        # _eval_conductor_ggx_spectral folds in cos(dir_to_other, n) via its
        # `k * cos_i`, so that -- not 1 -- is this lobe's cosine.
        var f_g = _eval_conductor_ggx_spectral(vn, vwo, dir_to_other, c.param, c.alb, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
        var cos_g = abs(dot(dir_to_other, vn))
        var fwd_g = Float32(0)
        var rev_g = Float32(0)
        comptime if want_pdfs:
            fwd_g = bxdf_pdf_conductor_ggx(vn, vwo, dir_to_other, c.param)
            comptime if want_rev:
                rev_g = bxdf_pdf_conductor_ggx(vn, dir_to_other, vwo, c.param)
        return LobeEval(f_g, cos_g, fwd_g, rev_g, True)

    if c.kind == LobeKind.hair:
        var mat_h = tab.materials[unsafe_offset=Int(c.mat_idx)]
        var hc = _hair_precompute(mat_h, tab.curves, Int(c.hair_curve_idx), c.hair_v, c.hair_h, vwo)
        return _hair_lobe_eval[want_pdfs, want_rev](c, hc, dir_to_other, tab, spectral_coeffs, spectral_res,
            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)

    if c.kind == LobeKind.measured:
        var vmat = tab.materials[unsafe_offset=Int(c.mat_idx)]
        var mb = tab.measured_brdfs[unsafe_offset=Int(vmat.measured_idx)]
        var frm = Frame.from_z(Vec3f(vn[0], vn[1], vn[2]))
        var tangent = Vec3f(frm.x.x, frm.x.y, frm.x.z)
        var bitangent = Vec3f(frm.y.x, frm.y.y, frm.y.z)
        var wo_l = Vec3f(dot(vwo, tangent), dot(vwo, bitangent), dot(vwo, vn))
        var wi_l = Vec3f(dot(dir_to_other, tangent), dot(dir_to_other, bitangent), dot(dir_to_other, vn))
        var fr_spec: SpectralSample
        if c.adjoint:
            fr_spec = bxdf_eval_measured(mb, wi_l, wo_l, wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)[0]
        else:
            fr_spec = bxdf_eval_measured(mb, wo_l, wi_l, wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)[0]
        var cos_m = abs(dot(dir_to_other, vn))
        var fwd_m = Float32(0)
        var rev_m = Float32(0)
        comptime if want_pdfs:
            fwd_m = bxdf_pdf_measured(mb, wo_l, wi_l)
            comptime if want_rev:
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
            return LobeEval(ZERO, Float32(1), Float32(0), Float32(0), _lobe_scoped(c))
        var same_side = cos_dt * cos_wo_dt > Float32(0)
        var trans_dt = _dt_transmittance(c, tab)
        var refl_dt = _dt_reflectance(c, tab)
        var lobe_alb_dt = refl_dt if same_side else trans_dt
        # The SAME luminance split bxdf_sample_diffuse_transmit samples with.
        # Deriving the density any other way lets MIS drift against the
        # sampler, which is the drift this interface exists to prevent.
        var pr_dt = refl_dt.luma()
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
        return LobeEval(alb_dt * (INV_PI * cos_a_dt), cos_a_dt, fwd_dt, rev_dt, _lobe_scoped(c))

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
def _dt_transmittance(c: BSDF, tab: LobeTables) -> RGB:
    """A diffusetransmission lobe's transmittance. It lives in the MATERIAL
    (Material.emission, see shade_diffuse_transmission), so it needs a real
    mat_idx; callers without one pass -1, which falls back to a symmetric
    lobe rather than index the table out of bounds. One texture slot serves
    both lobes when textured, matching shade_diffuse_transmission; c.alb is
    already the resolved one. Shared by _lobe_eval and _lobe_sample so the
    density one reports is the split the other draws from."""
    if Int(c.mat_idx) >= 0:
        return tab.materials[unsafe_offset=Int(c.mat_idx)].dt_transmittance(c.alb)
    return c.alb

@always_inline
def _dt_reflectance(c: BSDF, tab: LobeTables) -> RGB:
    """The reflect lobe's colour: c.alb, unless the material's texture drives the transmittance only."""
    if Int(c.mat_idx) >= 0:
        return tab.materials[unsafe_offset=Int(c.mat_idx)].dt_reflectance(c.alb)
    return c.alb


@fieldwise_init
struct LobeSample(TrivialRegisterPassable):
    """One scattering decision at a vertex -- _lobe_eval's sampling half.

    `weight` is what the throughput is multiplied by, f*|cos|/pdf. `pdf_fwd`
    and `pdf_rev` are the densities MIS needs (solid angle, both 0 for a delta
    event). For an analytic lobe they come FROM _lobe_eval at the sampled
    direction, and so does the weight: sampling only chooses the direction,
    so a sampler and its evaluator cannot drift apart. A stochastic lobe
    (layered) is the exception: its f is itself an estimate, so the weight
    is the random walk's own throughput while the densities still come from
    _lobe_eval, the value every other strategy sees."""
    var valid:    Bool
    var wi:       Vec3f
    var weight:   SpectralSample
    var is_delta: Bool
    var pdf_fwd:  Float32
    var pdf_rev:  Float32
    var cos_out:  Float32   # the lobe's own |cos| at wi
    var scoped:   Bool      # _lobe_scoped(c), and nothing else
    var transmit: Bool      # a transmission event, for Cycles' per-kind bounce limits


@always_inline
def _lobe_sample_invalid() -> LobeSample:
    return LobeSample(False, Vec3f(Float32(0)), SpectralSample(Float32(0)), False,
                      Float32(0), Float32(0), Float32(0), False, False)


@always_inline
def _hair_mis_pdf(c: BSDF, tab: LobeTables, wo: Vec3f, wi: Vec3f) -> Float32:
    """Hair's density of wi given wo for MIS, at h = 0. The true density depends on the offset h the ARRIVING ray
    hit the fibre at, which the camera and light subpaths see differently at one vertex; MIS needs one value."""
    var hc0 = _hair_precompute(tab.materials[unsafe_offset=Int(c.mat_idx)], tab.curves, Int(c.hair_curve_idx), c.hair_v,
                               Float32(0), wo)
    var (cos_ti, _, pdf_oc) = _hair_eval_lobes(
        wi, hc0.tangent, hc0.b_perp, hc0.n_perp, hc0.phi_o,
        hc0.dphi0, hc0.dphi1, hc0.dphi2,
        hc0.cos_tp0_o, hc0.sin_tp0_o, hc0.cos_tp1_o, hc0.sin_tp1_o, hc0.cos_tp2_o, hc0.sin_tp2_o,
        hc0.cos_theta_o, hc0.sin_theta_o, hc0.inv_vm0, hc0.inv_vm1, hc0.inv_vm2, hc0.mp_c0, hc0.mp_c1, hc0.mp_c2, hc0.s,
        hc0.A0, hc0.A1, hc0.A2, hc0.A3, hc0.lum0, hc0.lum1, hc0.lum2, hc0.lum3, hc0.total_lum,
    )
    return cos_ti * pdf_oc


@always_inline
def _hair_lobe_sample[want_rev: Bool](
    c: BSDF, hc: HairLobeConstants, uc: Float32, u0: Float32, u1: Float32, u2: Float32, tab: LobeTables,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    wavelengths: SampledWavelengths,
) -> LobeSample:
    """The 3-lobe Marschner sampler; its density is the fibre cosine times pdf_over_cos, _lobe_eval's pdf_fwd."""
    var (wi, f_rgb, pdf_oc, cos_ti) = _hair_sample_dir_u(hc, uc, u0, u1, u2)
    var transmit = uc * hc.total_lum >= hc.lum0   # Cycles counts the R lobe as glossy, TT and TRT as transmission
    if pdf_oc * cos_ti <= Float32(0):
        return _lobe_sample_invalid()
    # The weight is the sampler's own f/pdf; the densities MIS sees are the h-free ones (_hair_mis_pdf).
    var f = spec_refl_unbounded(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
                                f_rgb.r, f_rgb.g, f_rgb.b, wavelengths)
    var rev = Float32(0)
    comptime if want_rev:
        rev = _hair_mis_pdf(c, tab, wi, c.wo)
    return LobeSample(True, wi, f * (Float32(1) / pdf_oc), False, _hair_mis_pdf(c, tab, c.wo, wi), rev, cos_ti,
                      _lobe_scoped(c), transmit)


def _lobe_sample[want_rev: Bool = True](
    c:   BSDF,
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
    """THE lobe sampler: the same BSDF _lobe_eval takes, so the two see one
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
                              Float32(0), Float32(0), cos_o, _lobe_scoped(c), False)
        # Rough: the multiple-scattering cosine lobe or a VNDF reflection,
        # mixed exactly as bxdf_pdf_conductor_ggx (_lobe_eval's density) says.
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
        var le_g = _lobe_eval[want_pdfs=True, want_rev=want_rev](c, wi_g, tab, spectral_coeffs, spectral_res,
            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
        if le_g.pdf_fwd <= Float32(0):
            return _lobe_sample_invalid()
        return LobeSample(True, wi_g, le_g.f_cos * (Float32(1) / le_g.pdf_fwd), False,
                          le_g.pdf_fwd, le_g.pdf_rev, le_g.cos_used, _lobe_scoped(c), False)

    if c.is_delta:
        return _lobe_sample_invalid()

    if c.kind == LobeKind.rough_dielectric:
        # diel_sample draws the direction (reflect or transmit by Fresnel,
        # a VNDF microfacet); weight and densities come back from _lobe_eval,
        # as for every analytic lobe, so sampler and evaluator cannot drift.
        var ior_rs = tab.materials[unsafe_offset=Int(c.mat_idx)].albedo.r
        var fr_rs = Frame.from_z(Vec3f(vn[0], vn[1], vn[2]))
        var tx_rs = Vec3f(fr_rs.x.x, fr_rs.x.y, fr_rs.x.z)
        var ty_rs = Vec3f(fr_rs.y.x, fr_rs.y.y, fr_rs.y.z)
        var wo_rs = Vec3f(dot(vwo, tx_rs), dot(vwo, ty_rs), dot(vwo, vn))
        var ds = diel_sample(wo_rs, uc, u0, u1, ior_rs, c.param, not c.adjoint, True, True)
        if not ds.valid:
            return _lobe_sample_invalid()
        var wi_rs = tx_rs * ds.wi.x + ty_rs * ds.wi.y + vn * ds.wi.z
        var le_rs = _lobe_eval[want_pdfs=True, want_rev=want_rev](c, wi_rs, tab, spectral_coeffs, spectral_res,
            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
        if le_rs.pdf_fwd <= Float32(0):
            return _lobe_sample_invalid()
        return LobeSample(True, wi_rs, le_rs.f_cos * (Float32(1) / le_rs.pdf_fwd), False,
                          le_rs.pdf_fwd, le_rs.pdf_rev, le_rs.cos_used, _lobe_scoped(c), ds.wi.z * wo_rs.z < Float32(0))

    if c.kind == LobeKind.hair:
        # The 3-lobe Marschner sampler (bvh.mojo). Its density is the fibre
        # cosine times pdf_over_cos, which is _lobe_eval's pdf_fwd exactly.
        var hc = _hair_precompute(tab.materials[unsafe_offset=Int(c.mat_idx)], tab.curves, Int(c.hair_curve_idx),
                                  c.hair_v, c.hair_h, vwo)
        return _hair_lobe_sample[want_rev](c, hc, uc, u0, u1, u2, tab, spectral_coeffs, spectral_res,
            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)

    if c.kind == LobeKind.measured:
        # pbrt's MeasuredBxDF::Sample_f, in the same local frame _lobe_eval
        # evaluates the table in. The weight divides by the density the
        # sampler actually drew from; f comes from _lobe_eval, which applies
        # the adjoint argument order on the light subpath.
        var mb = tab.measured_brdfs[unsafe_offset=Int(tab.materials[unsafe_offset=Int(c.mat_idx)].measured_idx)]
        var fr_m = Frame.from_z(Vec3f(vn[0], vn[1], vn[2]))
        var tx_m = Vec3f(fr_m.x.x, fr_m.x.y, fr_m.x.z)
        var ty_m = Vec3f(fr_m.y.x, fr_m.y.y, fr_m.y.z)
        var wo_l = Vec3f(dot(vwo, tx_m), dot(vwo, ty_m), dot(vwo, vn))
        var (wi_l, f_m, pdf_m, ok_m) = bxdf_sample_measured(mb, wo_l, u0, u1, wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
        if not ok_m or pdf_m <= Float32(0):
            return _lobe_sample_invalid()
        var wi_m = tx_m * wi_l[0] + ty_m * wi_l[1] + vn * wi_l[2]
        wi_m = wi_m * (Float32(1) / sqrt(max(dot(wi_m, wi_m), Float32(1e-20))))
        if dot(wi_m, vn) <= Float32(0):
            return _lobe_sample_invalid()
        var cos_m = abs(dot(wi_m, vn))
        if not c.adjoint:   # the sampler's f is already f(wo, wi)
            var wi_lm = Vec3f(dot(wi_m, tx_m), dot(wi_m, ty_m), dot(wi_m, vn))
            return LobeSample(True, wi_m, f_m * (cos_m / pdf_m), False, pdf_m,
                              bxdf_pdf_measured(mb, wi_lm, wo_l) if want_rev else Float32(0), cos_m, _lobe_scoped(c), False)
        var le_m = _lobe_eval[want_pdfs=True, want_rev=want_rev](c, wi_m, tab, spectral_coeffs, spectral_res,
            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
        return LobeSample(True, wi_m, le_m.f_cos * (Float32(1) / pdf_m), False,
                          pdf_m, le_m.pdf_rev, le_m.cos_used, _lobe_scoped(c), False)

    if c.kind == LobeKind.layered:
        var mat_ly = tab.materials[unsafe_offset=Int(c.mat_idx)]
        var fr = Frame.from_z(Vec3f(vn[0], vn[1], vn[2]))
        var tx = Vec3f(fr.x.x, fr.x.y, fr.x.z)
        var ty = Vec3f(fr.y.x, fr.y.y, fr.y.z)
        var wo_l = Vec3f(dot(vwo, tx), dot(vwo, ty), dot(vwo, vn))
        var R = rgb_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, c.alb.r, c.alb.g, c.alb.b, wavelengths)
        var alpha_s = c.param if c.param > Float32(0) else mat_ly.coat_alpha()   # as _lobe_eval: param>0 forces the top alpha
        var bs = layered_sample[True](wo_l, uc, u0, u1, R, mat_ly.emission.r, alpha_s, not c.adjoint,
                                      mat_ly.emission.g, max(mat_ly.roughU, mat_ly.roughV)) \
            if mat_ly.type == MatKind.coated_conductor \
            else layered_sample(wo_l, uc, u0, u1, R, mat_ly.emission.r, alpha_s, not c.adjoint, mat_ly.emission.g)
        if not bs.valid or bs.pdf <= Float32(0):
            return _lobe_sample_invalid()
        var wi = tx * bs.wi.x + ty * bs.wi.y + vn * bs.wi.z
        var cos_out = abs(bs.wi.z)
        var w = bs.f * (cos_out / bs.pdf)
        if bs.specular:
            return LobeSample(True, wi, w, True, Float32(0), Float32(0), cos_out, _lobe_scoped(c), False)
        # The densities only: f here is the walk's own estimate, and a second layered_f would be thrown away.
        var (fwd, rev) = _layered_pdfs[want_rev](c, mat_ly, wo_l, bs.wi, alpha_s)
        return LobeSample(True, wi, w, False, fwd, rev, cos_out, _lobe_scoped(c), False)

    var bounce_n = vn
    if c.kind == LobeKind.lambertian:
        # The lobe lives on wo's side; _lobe_eval zeroes the other one.
        if dot(vwo, vn) < Float32(0):
            bounce_n = -vn
    elif c.kind == LobeKind.diffuse_transmit:
        # The SAME luminance split _lobe_eval reports as the lobe probability.
        var pr = _dt_reflectance(c, tab).luma()
        var pt = _dt_transmittance(c, tab).luma()
        if pr + pt <= Float32(1e-9):
            return _lobe_sample_invalid()   # nothing to scatter; do not invent a lobe
        var wo_side = vn if dot(vwo, vn) >= Float32(0) else -vn
        bounce_n = wo_side if uc < pr / (pr + pt) else -wo_side
    else:
        return _lobe_sample_invalid()

    var s = sample_cosine_hemisphere_world(u0, u1, bounce_n)
    var wi = s[0]
    var le = _lobe_eval[want_pdfs=True, want_rev=want_rev](c, wi, tab, spectral_coeffs, spectral_res,
        spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, wavelengths)
    if le.pdf_fwd <= Float32(0):
        return _lobe_sample_invalid()
    return LobeSample(True, wi, le.f_cos * (Float32(1) / le.pdf_fwd), False,
                      le.pdf_fwd, le.pdf_rev, le.cos_used, _lobe_scoped(c), dot(wi, vn) * dot(vwo, vn) < Float32(0))


# Moved here from bdpt_*.mojo so that the ONE lobe evaluator can live below
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


# ── Spectral siblings (staged rollout, see project_spectral_rendering memory
# / lovely-dazzling-meteor plan) ────────────────────────────────────────────
# Added ALONGSIDE bxdf_eval_any/_nee_weight_simple above rather than mutating
# them in place: those two are also called from bdpt_*.mojo (Stage 3) and
# sppm.mojo (Stage 4), which aren't wavelength-aware yet — changing their
# signature now would force-couple this (Stage 2, plain-path-tracer-only)
# change into BDPT/SPPM ahead of their own stages, defeating the point of
# staging. shading.mojo (Stage 2) calls these new spectral versions instead;
# bdpt_*.mojo/sppm.mojo keep calling the RGB originals unchanged until their
# own stage migrates them.
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
def _nee_weight_lobe(
    ls:    LightSample,
    c:     BSDF,
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
    the same BSDF _lobe_eval and _lobe_sample take -- so a stored vertex's
    NEE, connections and merges all see one vertex (hair's curve fields
    included, which a flat (kind, alb, alpha) signature cannot carry).

    Opaque lobes reject a light behind the shading normal; diffuse_transmit
    and hair transport to both sides, so for them _lobe_eval decides."""
    if not _nee_faces(ls, c):
        return SpectralSample(Float32(0.0))
    var le = _lobe_eval[want_pdfs=True, want_rev=False](c, ls.wi, tab,
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, wavelengths)
    return _nee_weight_of(ls, c, le, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z,
                          spectral_d65, wavelengths, mis)


@always_inline
def _nee_faces(ls: LightSample, c: BSDF) -> Bool:
    """A valid light sample the lobe can see: opaque lobes reject one behind the normal, two-sided ones do not."""
    if not ls.valid or (not ls.is_delta and ls.pdf <= Float32(0)):
        return False
    var cos_s = dot(c.n, ls.wi)
    # Lobes that transmit see lights on both sides; rough glass was missing, so VCM and SPPM lost its transmitted NEE.
    var two_sided = c.kind == LobeKind.diffuse_transmit or c.kind == LobeKind.hair or c.kind == LobeKind.rough_dielectric
    return not ((cos_s <= Float32(0.0) and not two_sided) or (two_sided and abs(cos_s) <= Float32(0.0) and c.kind != LobeKind.hair))


@always_inline
def _nee_weight_of(
    ls: LightSample, c: BSDF, le: LobeEval,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    wavelengths: SampledWavelengths,
    mis: MisPolicy,
) -> SpectralSample:
    """_nee_weight_lobe from the lobe's evaluation toward the light."""
    var cos_s = dot(c.n, ls.wi)
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
    return _nee_weight_lobe(ls,
        BSDF(mat_kind, True, False, n, wo, alb, mat_idx, alpha,
                Float32(0), Int32(-1), Float32(0), Float32(0), True, False),
        tab, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y,
        spectral_cie_z, spectral_d65, wavelengths, mis)

