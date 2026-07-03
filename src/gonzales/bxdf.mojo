from std.math import sqrt
from .geometry import RGB, MatKind, Material_C, Vec3f, dot, INV_PI, fr_dielectric
from .sampling import sample_ggx_vndf, sample_cosine_hemisphere_world

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
    var wi:       SIMD[DType.float32, 3]  # sampled incident direction (world-space)
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
    var normal:     SIMD[DType.float32, 3]  # shading normal, faceforward to geo_normal
    var geo_normal: SIMD[DType.float32, 3]  # geometric normal, faceforward to wo
    var hit_point:  SIMD[DType.float32, 3]  # world-space surface hit point
    var wo:         SIMD[DType.float32, 3]  # outgoing direction = -ray.direction
    var tangent:    SIMD[DType.float32, 3]  # shading tangent (Frisvad frame)
    var bitangent:  SIMD[DType.float32, 3]  # shading bitangent
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
    var white = RGB(Float32(1.0), Float32(1.0), Float32(1.0))

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
        return BxDFSample(wi, RGB(Float32(0.0), Float32(0.0), Float32(0.0)), Float32(1.0), BxDFFlags.glossy | BxDFFlags.reflect, Int8(0), Int8(0), Int8(0))
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
    var white = RGB(Float32(1.0), Float32(1.0), Float32(1.0))

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
        return BxDFSample(wi, RGB(Float32(0.0), Float32(0.0), Float32(0.0)), Float32(1.0), BxDFFlags.glossy | BxDFFlags.reflect, Int8(0), Int8(0), Int8(0))
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
    geom_normal: SIMD[DType.float32, 3],
    ray_dir: SIMD[DType.float32, 3],
    ior: Float32,
    force_entering: Bool,   # bounce==0: trust physics (camera ray always from air)
    u_reflect: Float32,
) -> Tuple[BxDFSample, SIMD[DType.float32, 3]]:
    var facing = dot(ray_dir, geom_normal) < Float32(0.0)
    var entering = facing or force_entering
    var normal = geom_normal if facing else -geom_normal
    var eta = (Float32(1.0) / ior) if entering else ior

    var cos_i = -dot(ray_dir, normal)
    var sin2_t = eta * eta * (Float32(1.0) - cos_i * cos_i)
    var tir = sin2_t > Float32(1.0)
    # eta here is η_i/η_t; fr_dielectric wants its reciprocal as the relative IOR.
    var fresnel = fr_dielectric(cos_i, Float32(1.0) / eta)
    var white = RGB(Float32(1.0), Float32(1.0), Float32(1.0))

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
    return (BxDFSample(refr, white, Float32(1.0), BxDFFlags.delta | BxDFFlags.transmit, Int8(1), Int8(0), Int8(0)), normal)

# ── Thin dielectric (one-sided glass slab: window, soap film) ────────────────
# Transmitted ray keeps its original direction (no refraction) — models a thin
# slab whose entry/exit refractions cancel. Fresnel compounded across both
# slab interfaces: R' = 2R/(1+R) (PBRT thin-glass formula).
@always_inline
def bxdf_sample_thin_dielectric(
    geom_normal: SIMD[DType.float32, 3],
    ray_dir: SIMD[DType.float32, 3],
    ior: Float32,
    u_reflect: Float32,
) -> Tuple[BxDFSample, SIMD[DType.float32, 3]]:
    var entering = dot(ray_dir, geom_normal) < Float32(0.0)
    var normal = geom_normal if entering else -geom_normal
    var cos_i = max(Float32(0.0), -dot(ray_dir, normal))

    var r_single = fr_dielectric(cos_i, ior)
    var fresnel = r_single
    if r_single < Float32(1.0):
        fresnel = Float32(2.0) * r_single / (Float32(1.0) + r_single)
    var white = RGB(Float32(1.0), Float32(1.0), Float32(1.0))

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
    normal: SIMD[DType.float32, 3],   # faceforward shading normal (to the ray)
    refl: RGB, trans: RGB,
    u_lobe: Float32, u1: Float32, u2: Float32,
) -> Tuple[BxDFSample, SIMD[DType.float32, 3], RGB, Float32, Bool]:
    var pr = refl.luma()
    var pt = trans.luma()
    var total = pr + pt
    if total <= Float32(0.0):
        var z = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(0.0))
        return (BxDFSample(z, RGB(Float32(0.0), Float32(0.0), Float32(0.0)), Float32(0.0), BxDFFlags.diffuse, Int8(0), Int8(0), Int8(0)),
            normal, RGB(Float32(0.0), Float32(0.0), Float32(0.0)), Float32(0.0), True)

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
