"""pbrt-v4's LayeredBxDF, specialised to coateddiffuse: a (rough) dielectric
coat over a Lambertian base, twoSided, no medium albedo.

A line-by-line port of pbrt's bxdfs.h (LayeredBxDF::f / Sample_f / PDF) and
bxdfs.cpp (DielectricBxDF), with util/scattering.h's TrowbridgeReitz
distribution. Everything is in the LOCAL shading frame, z up.

Why a port and not an approximation: the analytic coat model this replaces
(coat walk + a smooth-interface Fresnel transmission scaled by G2/G1 and a
multiple-scattering boost) was exact for a smooth coat but not for a rough one.
Its angular distribution was wrong: +16% on a coated floor under a 45-degree
sun, yet 0.87 in the uniform-light furnace. pbrt's layered BSDF is itself a
STOCHASTIC estimator -- each f() runs a short random walk between the two
interfaces -- so matching pbrt means running the same walk.

pbrt seeds that walk from a hash of the directions, making f(wo, wi) a
deterministic function; the same is done here, so MIS weights that call f and
PDF twice for one direction pair see one value.

One deliberate departure: layered_f's exit-NEE MIS weight. pbrt pairs
bs.pdf with exitInterface.PDF(-w, wi), a density over OUTSIDE directions;
the competing wis strategy's density of the same inside direction is
PDF(wi, -w). With pbrt's weight f integrates to 0.698 where its own
Sample_f gives 0.646 (white base, rough coat); corrected, 0.6453. See
docs/05_reflection_models.md.
"""
from std.math import sqrt, cos, sin, exp, abs, min, max
from std.memory import bitcast
from .geometry import Vec3f, dot, cross, fr_dielectric, safe_sqrt, PI, INV_PI
from .spectrum import SpectralSample
from .rng import PCG32

comptime LAYERED_THICKNESS = Float32(0.01)   # pbrt coateddiffuse default
comptime LAYERED_MAX_DEPTH = 10              # pbrt coateddiffuse default
comptime _ONE_MINUS_EPS = Float32(0.99999994)


# ── Trowbridge-Reitz (isotropic) ─────────────────────────────────────────────

@always_inline
def tr_effectively_smooth(alpha: Float32) -> Bool:
    return alpha < Float32(1e-3)

@always_inline
def tr_D(wm: Vec3f, alpha: Float32) -> Float32:
    var cos2 = wm.z * wm.z
    var sin2 = max(Float32(0), Float32(1) - cos2)
    if cos2 <= Float32(0):
        return Float32(0)
    var tan2 = sin2 / cos2
    var cos4 = cos2 * cos2
    if cos4 < Float32(1e-16):
        return Float32(0)
    var e = tan2 / (alpha * alpha)
    return Float32(1) / (PI * alpha * alpha * cos4 * (Float32(1) + e) * (Float32(1) + e))

@always_inline
def tr_lambda(w: Vec3f, alpha: Float32) -> Float32:
    var cos2 = w.z * w.z
    if cos2 <= Float32(0):
        return Float32(0)
    var tan2 = max(Float32(0), Float32(1) - cos2) / cos2
    return (sqrt(Float32(1) + alpha * alpha * tan2) - Float32(1)) * Float32(0.5)

@always_inline
def tr_G1(w: Vec3f, alpha: Float32) -> Float32:
    return Float32(1) / (Float32(1) + tr_lambda(w, alpha))

@always_inline
def tr_G(wo: Vec3f, wi: Vec3f, alpha: Float32) -> Float32:
    return Float32(1) / (Float32(1) + tr_lambda(wo, alpha) + tr_lambda(wi, alpha))

@always_inline
def tr_pdf(w: Vec3f, wm: Vec3f, alpha: Float32) -> Float32:
    """Visible-normal density D_w(wm)."""
    if w.z == Float32(0):
        return Float32(0)
    return tr_G1(w, alpha) / abs(w.z) * tr_D(wm, alpha) * abs(dot(w, wm))

@always_inline
def tr_sample_wm(w: Vec3f, u0: Float32, u1: Float32, alpha: Float32) -> Vec3f:
    var wh = Vec3f(alpha * w.x, alpha * w.y, w.z).normalize()
    if wh.z < Float32(0):
        wh = -wh
    var t1 = Vec3f(Float32(1), Float32(0), Float32(0))
    if wh.z < Float32(0.99999):
        t1 = cross(Vec3f(Float32(0), Float32(0), Float32(1)), wh).normalize()
    var t2 = cross(wh, t1)
    var r = sqrt(u0)
    var th = Float32(2) * PI * u1
    var px = r * cos(th)
    var py = r * sin(th)
    var h = sqrt(max(Float32(0), Float32(1) - px * px))
    var s = (Float32(1) + wh.z) * Float32(0.5)
    py = h + s * (py - h)      # Lerp(s, h, py)
    var pz = sqrt(max(Float32(0), Float32(1) - px * px - py * py))
    var nh = t1 * px + t2 * py + wh * pz
    return Vec3f(alpha * nh.x, alpha * nh.y, max(Float32(1e-6), nh.z)).normalize()


# ── Geometry helpers (pbrt's Refract / Reflect / FaceForward) ───────────────

@always_inline
def _refract(wi: Vec3f, n_in: Vec3f, eta_in: Float32) -> Tuple[Bool, Vec3f, Float32]:
    """(ok, wt, etap). pbrt's Refract: flips to the other side when wi is below n."""
    var n = n_in
    var eta = eta_in
    var cos_i = dot(n, wi)
    if cos_i < Float32(0):
        eta = Float32(1) / eta
        cos_i = -cos_i
        n = -n
    var sin2_i = max(Float32(0), Float32(1) - cos_i * cos_i)
    var sin2_t = sin2_i / (eta * eta)
    if sin2_t >= Float32(1):
        return (False, Vec3f(Float32(0)), eta)
    var cos_t = safe_sqrt(Float32(1) - sin2_t)
    var wt = (-wi) / eta + n * (cos_i / eta - cos_t)
    return (True, wt, eta)

@always_inline
def _reflect(wo: Vec3f, n: Vec3f) -> Vec3f:
    return -wo + n * (Float32(2) * dot(wo, n))


# ── One interface sample: the top dielectric or the bottom diffuse ──────────

@fieldwise_init
struct ISample(TrivialRegisterPassable):
    var valid: Bool
    var f: SpectralSample
    var wi: Vec3f
    var pdf: Float32
    var specular: Bool
    var reflection: Bool

@always_inline
def _no_sample() -> ISample:
    return ISample(False, SpectralSample(Float32(0)), Vec3f(Float32(0)), Float32(0), False, False)


@always_inline
def diel_sample(wo: Vec3f, uc: Float32, u0: Float32, u1: Float32, eta: Float32, alpha: Float32,
                radiance: Bool, allow_r: Bool, allow_t: Bool) -> ISample:
    """DielectricBxDF::Sample_f."""
    if eta == Float32(1) or tr_effectively_smooth(alpha):
        var R = fr_dielectric(wo.z, eta)
        var T = Float32(1) - R
        var pr = R if allow_r else Float32(0)
        var pt = T if allow_t else Float32(0)
        if pr == Float32(0) and pt == Float32(0):
            return _no_sample()
        if uc < pr / (pr + pt):
            var wi = Vec3f(-wo.x, -wo.y, wo.z)
            return ISample(True, SpectralSample(R / abs(wi.z)), wi, pr / (pr + pt), True, True)
        var (ok, wi, etap) = _refract(wo, Vec3f(Float32(0), Float32(0), Float32(1)), eta)
        if not ok:
            return _no_sample()
        var ft = T / abs(wi.z)
        if radiance:
            ft /= etap * etap
        return ISample(True, SpectralSample(ft), wi, pt / (pr + pt), True, False)
    var wm = tr_sample_wm(wo, u0, u1, alpha)
    var R = fr_dielectric(dot(wo, wm), eta)
    var T = Float32(1) - R
    var pr = R if allow_r else Float32(0)
    var pt = T if allow_t else Float32(0)
    if pr == Float32(0) and pt == Float32(0):
        return _no_sample()
    if uc < pr / (pr + pt):
        var wi = _reflect(wo, wm)
        if wo.z * wi.z <= Float32(0):
            return _no_sample()
        var pdf = tr_pdf(wo, wm, alpha) / (Float32(4) * abs(dot(wo, wm))) * pr / (pr + pt)
        var f = tr_D(wm, alpha) * tr_G(wo, wi, alpha) * R / (Float32(4) * wi.z * wo.z)
        return ISample(True, SpectralSample(f), wi, pdf, False, True)
    var (ok, wi, etap) = _refract(wo, wm, eta)
    if not ok or wo.z * wi.z > Float32(0) or wi.z == Float32(0):
        return _no_sample()
    var denom = (dot(wi, wm) + dot(wo, wm) / etap) * (dot(wi, wm) + dot(wo, wm) / etap)
    var dwm_dwi = abs(dot(wi, wm)) / denom
    var pdf = tr_pdf(wo, wm, alpha) * dwm_dwi * pt / (pr + pt)
    var ft = T * tr_D(wm, alpha) * tr_G(wo, wi, alpha) * abs(dot(wi, wm) * dot(wo, wm) / (wi.z * wo.z * denom))
    if radiance:
        ft /= etap * etap
    return ISample(True, SpectralSample(ft), wi, pdf, False, False)


@always_inline
def _diel_half(wo: Vec3f, wi: Vec3f, eta: Float32) -> Tuple[Bool, Vec3f, Float32, Bool]:
    """(ok, wm, etap, reflect) shared by DielectricBxDF::f and ::PDF."""
    var reflect = wi.z * wo.z > Float32(0)
    var etap = Float32(1)
    if not reflect:
        etap = eta if wo.z > Float32(0) else Float32(1) / eta
    var wm = wi * etap + wo
    if wi.z == Float32(0) or wo.z == Float32(0) or wm.length_sq() == Float32(0):
        return (False, wm, etap, reflect)
    wm = wm.normalize()
    if wm.z < Float32(0):
        wm = -wm
    if dot(wm, wi) * wi.z < Float32(0) or dot(wm, wo) * wo.z < Float32(0):
        return (False, wm, etap, reflect)
    return (True, wm, etap, reflect)


@always_inline
def diel_f(wo: Vec3f, wi: Vec3f, eta: Float32, alpha: Float32, radiance: Bool) -> Float32:
    """DielectricBxDF::f (0 for a smooth interface: delta lobes are sampled only)."""
    if eta == Float32(1) or tr_effectively_smooth(alpha):
        return Float32(0)
    var (ok, wm, etap, reflect) = _diel_half(wo, wi, eta)
    if not ok:
        return Float32(0)
    var F = fr_dielectric(dot(wo, wm), eta)
    if reflect:
        return tr_D(wm, alpha) * tr_G(wo, wi, alpha) * F / abs(Float32(4) * wi.z * wo.z)
    var denom = (dot(wi, wm) + dot(wo, wm) / etap) * (dot(wi, wm) + dot(wo, wm) / etap) * wi.z * wo.z
    var ft = tr_D(wm, alpha) * (Float32(1) - F) * tr_G(wo, wi, alpha) * abs(dot(wi, wm) * dot(wo, wm) / denom)
    if radiance:
        ft /= etap * etap
    return ft


@always_inline
def diel_pdf(wo: Vec3f, wi: Vec3f, eta: Float32, alpha: Float32, allow_r: Bool, allow_t: Bool) -> Float32:
    """DielectricBxDF::PDF."""
    if eta == Float32(1) or tr_effectively_smooth(alpha):
        return Float32(0)
    var (ok, wm, etap, reflect) = _diel_half(wo, wi, eta)
    if not ok:
        return Float32(0)
    var R = fr_dielectric(dot(wo, wm), eta)
    var T = Float32(1) - R
    var pr = R if allow_r else Float32(0)
    var pt = T if allow_t else Float32(0)
    if pr == Float32(0) and pt == Float32(0):
        return Float32(0)
    if reflect:
        return tr_pdf(wo, wm, alpha) / (Float32(4) * abs(dot(wo, wm))) * pr / (pr + pt)
    var denom = (dot(wi, wm) + dot(wo, wm) / etap) * (dot(wi, wm) + dot(wo, wm) / etap)
    var dwm_dwi = abs(dot(wi, wm)) / denom
    return tr_pdf(wo, wm, alpha) * dwm_dwi * pt / (pr + pt)


@always_inline
def diffuse_sample(wo: Vec3f, u0: Float32, u1: Float32, R: SpectralSample) -> ISample:
    """DiffuseBxDF::Sample_f (reflection only)."""
    var r = sqrt(u0)
    var th = Float32(2) * PI * u1
    var z = sqrt(max(Float32(0), Float32(1) - u0))
    var wi = Vec3f(r * cos(th), r * sin(th), z)
    if wo.z < Float32(0):
        wi.z = -wi.z
    return ISample(True, R * INV_PI, wi, abs(wi.z) * INV_PI, False, True)

@always_inline
def diffuse_f(wo: Vec3f, wi: Vec3f, R: SpectralSample) -> SpectralSample:
    if wo.z * wi.z <= Float32(0):
        return SpectralSample(Float32(0))
    return R * INV_PI

@always_inline
def diffuse_pdf(wo: Vec3f, wi: Vec3f) -> Float32:
    if wo.z * wi.z <= Float32(0):
        return Float32(0)
    return abs(wi.z) * INV_PI


@always_inline
def _power(a: Float32, b: Float32) -> Float32:
    var aa = a * a
    var bb = b * b
    if aa + bb <= Float32(0):
        return Float32(0)
    return aa / (aa + bb)

@always_inline
def _tr(w: Vec3f) -> Float32:
    """pbrt's LayeredBxDF::Tr(thickness, w): the coat's own absorption."""
    return exp(-abs(LAYERED_THICKNESS / w.z))


@always_inline
def _hash_dirs(a: Vec3f, b: Vec3f, salt: UInt64) -> PCG32:
    """pbrt seeds the layered walk from a hash of the directions, so f(wo, wi)
    is a deterministic function of its arguments."""
    var ab = bitcast[DType.uint32, 4](SIMD[DType.float32, 4](a.x, a.y, a.z, Float32(0)))
    var bb = bitcast[DType.uint32, 4](SIMD[DType.float32, 4](b.x, b.y, b.z, Float32(0)))
    var h = salt * UInt64(0x9E3779B97F4A7C15)
    comptime for k in range(3):
        h = (h ^ UInt64(ab[k])) * UInt64(0xBF58476D1CE4E5B9)
        h ^= h >> 31
        h = (h ^ UInt64(bb[k])) * UInt64(0x94D049BB133111EB)
        h ^= h >> 29
    return PCG32(h, (h >> 17) | UInt64(1))

@always_inline
def _r(mut rng: PCG32) -> Float32:
    return min(rng.next_float(), _ONE_MINUS_EPS)


# ── LayeredBxDF<Dielectric, Diffuse, twoSided=true>, albedo = 0 ─────────────

def layered_f(wo_in: Vec3f, wi_in: Vec3f, R: SpectralSample, eta: Float32, alpha: Float32,
              radiance: Bool) -> SpectralSample:
    """LayeredBxDF::f. A coat over an opaque base never transmits, so paths in
    opposite hemispheres are zero outright (pbrt reaches the same zero through
    the bottom's refusal to sample transmission)."""
    var wo = wo_in
    var wi = wi_in
    if wo.z < Float32(0):
        wo = -wo
        wi = -wi
    if wo.z * wi.z <= Float32(0):
        return SpectralSample(Float32(0))
    # Entered at the top and leaves through it again: exitZ = thickness.
    var f = SpectralSample(diel_f(wo, wi, eta, alpha, radiance))
    var rng = _hash_dirs(wo, wi, UInt64(1))
    var top_specular = tr_effectively_smooth(alpha)

    var wos = diel_sample(wo, _r(rng), _r(rng), _r(rng), eta, alpha, radiance, False, True)
    if not wos.valid or wos.f.is_black() or wos.pdf == Float32(0) or wos.wi.z == Float32(0):
        return f
    var wis = diel_sample(wi, _r(rng), _r(rng), _r(rng), eta, alpha, not radiance, False, True)
    if not wis.valid or wis.f.is_black() or wis.pdf == Float32(0) or wis.wi.z == Float32(0):
        return f

    var beta = wos.f * (abs(wos.wi.z) / wos.pdf)
    var z = LAYERED_THICKNESS
    var w = wos.wi
    for depth in range(LAYERED_MAX_DEPTH):
        if depth > 3 and beta.max_component() < Float32(0.25):
            var q = max(Float32(0), Float32(1) - beta.max_component())
            if _r(rng) < q:
                break
            beta = beta / (Float32(1) - q)
        # No medium: go straight to the other interface.
        z = Float32(0) if z == LAYERED_THICKNESS else LAYERED_THICKNESS
        beta *= _tr(w)
        if z == LAYERED_THICKNESS:
            # Reflection back down at the (exit) top interface.
            var bs = diel_sample(-w, _r(rng), _r(rng), _r(rng), eta, alpha, radiance, True, False)
            if not bs.valid or bs.f.is_black() or bs.pdf == Float32(0) or bs.wi.z == Float32(0):
                break
            beta = beta * bs.f * (abs(bs.wi.z) / bs.pdf)
            w = bs.wi
        else:
            # The diffuse base (non-exit, non-specular): NEE along wis...
            var wt = Float32(1)
            if not top_specular:
                wt = _power(wis.pdf, diffuse_pdf(-w, -wis.wi))
            f += beta * diffuse_f(-w, -wis.wi, R) * (abs(wis.wi.z) * wt * _tr(wis.wi) / wis.pdf) * wis.f
            # ...then sample the base for the next direction...
            var bs = diffuse_sample(-w, _r(rng), _r(rng), R)
            if not bs.valid or bs.f.is_black() or bs.pdf == Float32(0) or bs.wi.z == Float32(0):
                break
            beta = beta * bs.f * (abs(bs.wi.z) / bs.pdf)
            w = bs.wi
            # ...and its own NEE through the exit interface.
            if not top_specular:
                var fexit = diel_f(-w, wi, eta, alpha, radiance)
                if fexit > Float32(0):
                    # wis's density of this inside direction, not pbrt's
                    # PDF(-w, wi) -- see the module docstring.
                    var wis_pdf_here = diel_pdf(wi, -w, eta, alpha, False, True)
                    f += beta * (_tr(bs.wi) * fexit * _power(bs.pdf, wis_pdf_here))
    return f


@fieldwise_init
struct LayeredSample(TrivialRegisterPassable):
    var valid: Bool
    var f: SpectralSample      # BSDF value (no cos(wi) -- the caller applies it)
    var wi: Vec3f
    var pdf: Float32           # PROPORTIONAL only (pbrt pdfIsProportional) unless specular reflection
    var specular: Bool
    var pdf_is_proportional: Bool


def layered_sample(wo_in: Vec3f, uc: Float32, u0: Float32, u1: Float32, R: SpectralSample,
                   eta: Float32, alpha: Float32, radiance: Bool) -> LayeredSample:
    """LayeredBxDF::Sample_f. The returned pdf is only proportional (pbrt sets
    pdfIsProportional): throughput uses f*cos/pdf, MIS must call layered_pdf."""
    var wo = wo_in
    var flip = False
    if wo.z < Float32(0):
        wo = -wo
        flip = True
    var bs = diel_sample(wo, uc, u0, u1, eta, alpha, radiance, True, True)
    if not bs.valid or bs.f.is_black() or bs.pdf == Float32(0) or bs.wi.z == Float32(0):
        return LayeredSample(False, SpectralSample(Float32(0)), Vec3f(Float32(0)), Float32(0), False, False)
    if bs.reflection:
        var wr = -bs.wi if flip else bs.wi
        return LayeredSample(True, bs.f, wr, bs.pdf, bs.specular, True)
    var w = bs.wi
    var specular_path = bs.specular
    var rng = _hash_dirs(wo, Vec3f(uc, u0, u1), UInt64(2))
    var f = bs.f * abs(bs.wi.z)
    var pdf = bs.pdf
    var z = LAYERED_THICKNESS
    for depth in range(LAYERED_MAX_DEPTH):
        var rr_beta = f.max_component() / pdf
        if depth > 3 and rr_beta < Float32(0.25):
            var q = max(Float32(0), Float32(1) - rr_beta)
            if _r(rng) < q:
                break
            pdf *= Float32(1) - q
        if w.z == Float32(0):
            break
        z = Float32(0) if z == LAYERED_THICKNESS else LAYERED_THICKNESS
        f = f * _tr(w)
        var s: ISample
        if z == Float32(0):
            s = diffuse_sample(-w, _r(rng), _r(rng), R)
        else:
            var c0 = _r(rng)
            var c1 = _r(rng)
            var c2 = _r(rng)
            s = diel_sample(-w, c0, c1, c2, eta, alpha, radiance, True, True)
        if not s.valid or s.f.is_black() or s.pdf == Float32(0) or s.wi.z == Float32(0):
            break
        f = f * s.f
        pdf *= s.pdf
        specular_path = specular_path and s.specular
        w = s.wi
        if not s.reflection:
            # Left the layers (only the top transmits).
            var wr = -w if flip else w
            return LayeredSample(True, f, wr, pdf, specular_path, True)
        f = f * abs(s.wi.z)
    return LayeredSample(False, SpectralSample(Float32(0)), Vec3f(Float32(0)), Float32(0), False, False)


def layered_pdf(wo_in: Vec3f, wi_in: Vec3f, eta: Float32, alpha: Float32, radiance: Bool) -> Float32:
    """LayeredBxDF::PDF: a stochastic estimate, mixed 90/10 with a uniform
    sphere density exactly as pbrt does."""
    var wo = wo_in
    var wi = wi_in
    if wo.z < Float32(0):
        wo = -wo
        wi = -wi
    var rng = _hash_dirs(wi, wo, UInt64(3))
    var pdf_sum = Float32(0)
    if wo.z * wi.z > Float32(0):
        pdf_sum += diel_pdf(wo, wi, eta, alpha, True, False)
        # TRT: transmit through the top, reflect off the base, transmit back.
        var wos = diel_sample(wo, _r(rng), _r(rng), _r(rng), eta, alpha, radiance, False, True)
        var wis = diel_sample(wi, _r(rng), _r(rng), _r(rng), eta, alpha, not radiance, False, True)
        if wos.valid and not wos.f.is_black() and wos.pdf > Float32(0) and wis.valid and not wis.f.is_black() and wis.pdf > Float32(0):
            if tr_effectively_smooth(alpha):
                pdf_sum += diffuse_pdf(-wos.wi, -wis.wi)
            else:
                var R1 = SpectralSample(Float32(1))
                var rs = diffuse_sample(-wos.wi, _r(rng), _r(rng), R1)
                if rs.valid and rs.pdf > Float32(0):
                    var r_pdf = diffuse_pdf(-wos.wi, -wis.wi)
                    pdf_sum += _power(wis.pdf, r_pdf) * r_pdf
                    var t_pdf = diel_pdf(-rs.wi, wi, eta, alpha, True, True)
                    pdf_sum += _power(rs.pdf, t_pdf) * t_pdf
    # (TT is always zero here: the diffuse base does not transmit.)
    return Float32(0.9) * pdf_sum + Float32(0.1) * (Float32(1) / (Float32(4) * PI))
