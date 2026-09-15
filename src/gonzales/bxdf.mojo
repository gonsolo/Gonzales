from std.math import sqrt
from .geometry import RGB, MatKind, Material_C, Vec3f, dot, INV_PI, PI, fr_dielectric, coat_beer_lambert_tr, cos_theta_t_dielectric, DEFAULT_COAT_THICKNESS, Frame, refract
from .sampling import sample_ggx_vndf, sample_cosine_hemisphere_world, power_heuristic
from .rng import PCG32
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
    current_ior: Float32 = Float32(1.0),    # IOR of the medium the ray is ALREADY in; 1.0 = vacuum
    previous_ior: Float32 = Float32(1.0),   # IOR one level below current_ior (what exiting restores)
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
    var tir = sin2_t > Float32(1.0)
    # eta here is η_i/η_t; fr_dielectric wants its reciprocal as the relative IOR.
    var fresnel = fr_dielectric(cos_i, Float32(1.0) / eta)
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
    var radiance_transmit = white * (eta * eta)
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
    if ls.is_delta:
        return ls.Li * f_cos
    if ls.pdf <= Float32(0.0):
        return RGB(Float32(0.0))
    var pdf_bsdf = ggx_vndf_pdf(cos_o, cos_wm, d, coat_alpha)
    var w = power_heuristic(ls.pdf, pdf_bsdf)
    return ls.Li * (f_cos * w / ls.pdf)

@always_inline
def _nee_weight_coated_diffuse_base[nee_is_sole_strategy: Bool = False](
    ls:  LightSample,
    alb: RGB,
    ior: Float32,
    n:   Vec3f,
    coat_alpha: Float32 = Float32(0.0),
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
        w = power_heuristic(ls.pdf, cos_s / PI)
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
