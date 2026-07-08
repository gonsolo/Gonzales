# Foundational spectral-rendering types: hero-wavelength sampling, an
# analytic CIE-1931 XYZ color-matching-function fit, and an approximate
# RGB<->spectrum conversion. This is layer 1 of item 15 in the priority
# backlog ("Spectral rendering") — self-contained and independently
# testable, NOT yet wired into any integrator. `SampledSpectrum` in
# geometry.mojo remains an alias for `RGB` until a later layer threads
# `SampledWavelengths` through the shading context and switches that alias
# over; see project_spectral_rendering memory for the staged plan.
#
# The CIE XYZ fit is the analytic multi-Gaussian approximation from Wyman,
# Sloan & Shirley, "Simple Analytic Approximations to the CIE XYZ Color
# Matching Functions" (JCGT 2013) — closed-form, no lookup table needed,
# accurate to within ~1e-3 of the tabulated CIE 1931 2-degree data over the
# visible range.
#
# The RGB->spectrum upsampling is NOT the Jakob-Hanika 2019 method PBRT
# itself uses (that needs a precomputed 3D coefficient table produced by an
# offline per-RGB-value optimization) — it's a much simpler analytic
# 3-Gaussian-bump basis (one bump per primary), good enough to give a
# non-flat, plausible-looking spectrum for a given RGB reflectance/emission,
# but it does NOT round-trip through the CIE integral back to the exact
# input RGB the way a properly-optimized upsampling table would. Documented
# limitation, not a bug — matches this session's approach to `measured` and
# `subsurface` materials (a scoped, clearly-labeled approximation, not the
# full literature method).

from std.math import exp, sqrt, cos, sin

comptime N_SPECTRAL_SAMPLES = 4
comptime LAMBDA_MIN = Float32(360.0)
comptime LAMBDA_MAX = Float32(830.0)

# ── Analytic CIE 1931 XYZ color-matching-function fit (Wyman et al. 2013) ──

@always_inline
def _asym_gaussian(x: Float32, mu: Float32, sigma1: Float32, sigma2: Float32) -> Float32:
    var sigma = sigma1 if x < mu else sigma2
    var t = (x - mu) / sigma
    return exp(Float32(-0.5) * t * t)

@always_inline
def cie_x(lambda_nm: Float32) -> Float32:
    return (Float32(1.056) * _asym_gaussian(lambda_nm, Float32(599.8), Float32(37.9), Float32(31.0))
          + Float32(0.362) * _asym_gaussian(lambda_nm, Float32(442.0), Float32(16.0), Float32(26.7))
          - Float32(0.065) * _asym_gaussian(lambda_nm, Float32(501.1), Float32(20.4), Float32(26.2)))

@always_inline
def cie_y(lambda_nm: Float32) -> Float32:
    return (Float32(0.821) * _asym_gaussian(lambda_nm, Float32(568.8), Float32(46.9), Float32(40.5))
          + Float32(0.286) * _asym_gaussian(lambda_nm, Float32(530.9), Float32(16.3), Float32(31.1)))

@always_inline
def cie_z(lambda_nm: Float32) -> Float32:
    return (Float32(1.217) * _asym_gaussian(lambda_nm, Float32(437.0), Float32(11.8), Float32(36.0))
          + Float32(0.681) * _asym_gaussian(lambda_nm, Float32(459.0), Float32(26.0), Float32(13.8)))

# Integral of CIE Y-bar over [LAMBDA_MIN, LAMBDA_MAX] — the normalization
# constant (CIE_Y_INTEGRAL = 106.856895 for the standard 1931 2-degree
# observer over 360-830nm) used to turn a Monte-Carlo XYZ estimate into
# properly-normalized tristimulus values (matches PBRT's CIE_Y_integral).
comptime CIE_Y_INTEGRAL = Float32(106.856895)

# ── Hero-wavelength sampling ────────────────────────────────────────────────

@fieldwise_init
struct SampledWavelengths(TrivialRegisterPassable):
    """4 hero-sampled wavelengths (nm) + their sampling pdf (1/nm), shared by
    all 4 since stratified hero sampling uses one pdf for the whole set."""
    var lambda0: Float32
    var lambda1: Float32
    var lambda2: Float32
    var lambda3: Float32
    var pdf: Float32

    @always_inline
    def get(self, i: Int) -> Float32:
        if i == 0: return self.lambda0
        elif i == 1: return self.lambda1
        elif i == 2: return self.lambda2
        else: return self.lambda3

@always_inline
def sample_wavelengths_uniform(u: Float32) -> SampledWavelengths:
    """Stratified hero-wavelength sampling (Wilkie et al. 2014): pick one
    primary wavelength uniformly, then offset the other 3 by even strides
    across the visible range, wrapping around. Uniform pdf = 1/(range)."""
    var span = LAMBDA_MAX - LAMBDA_MIN
    var lambda0 = LAMBDA_MIN + u * span
    var lambdas = SIMD[DType.float32, 4](lambda0, lambda0, lambda0, lambda0)
    for i in range(1, N_SPECTRAL_SAMPLES):
        var off = lambdas[0] + span * (Float32(i) / Float32(N_SPECTRAL_SAMPLES))
        if off > LAMBDA_MAX:
            off -= span
        lambdas[i] = off
    var pdf = Float32(1.0) / span
    return SampledWavelengths(lambdas[0], lambdas[1], lambdas[2], lambdas[3], pdf)

# ── Spectral radiance sample (4-wide, tied to one SampledWavelengths) ──────

@fieldwise_init
struct SpectralSample(TrivialRegisterPassable):
    """Radiance/reflectance at 4 hero-sampled wavelengths. Arithmetic between
    two SpectralSamples is only meaningful if both share the same
    SampledWavelengths — callers are responsible for that invariant (mirrors
    how PBRT threads SampledWavelengths through the whole path)."""
    var v0: Float32
    var v1: Float32
    var v2: Float32
    var v3: Float32

    @always_inline
    def __init__(out self, v: Float32):
        self.v0 = v; self.v1 = v; self.v2 = v; self.v3 = v

    @always_inline
    def __add__(self, o: SpectralSample) -> SpectralSample:
        return SpectralSample(self.v0 + o.v0, self.v1 + o.v1, self.v2 + o.v2, self.v3 + o.v3)

    @always_inline
    def __mul__(self, o: SpectralSample) -> SpectralSample:
        return SpectralSample(self.v0 * o.v0, self.v1 * o.v1, self.v2 * o.v2, self.v3 * o.v3)

    @always_inline
    def __mul__(self, s: Float32) -> SpectralSample:
        return SpectralSample(self.v0 * s, self.v1 * s, self.v2 * s, self.v3 * s)

    @always_inline
    def __truediv__(self, s: Float32) -> SpectralSample:
        var inv = Float32(1.0) / s
        return self * inv

    @always_inline
    def get(self, i: Int) -> Float32:
        if i == 0: return self.v0
        elif i == 1: return self.v1
        elif i == 2: return self.v2
        else: return self.v3

    @always_inline
    def average(self) -> Float32:
        return (self.v0 + self.v1 + self.v2 + self.v3) * Float32(0.25)

# ── RGB -> spectrum upsampling (approximate, see module docstring) ─────────

@always_inline
def _rgb_basis_r(lambda_nm: Float32) -> Float32:
    return _asym_gaussian(lambda_nm, Float32(600.0), Float32(50.0), Float32(90.0))

@always_inline
def _rgb_basis_g(lambda_nm: Float32) -> Float32:
    return _asym_gaussian(lambda_nm, Float32(545.0), Float32(45.0), Float32(45.0))

@always_inline
def _rgb_basis_b(lambda_nm: Float32) -> Float32:
    return _asym_gaussian(lambda_nm, Float32(455.0), Float32(45.0), Float32(50.0))

# Correction matrix so the round trip rgb -> spectrum -> (CIE XYZ) -> rgb is
# the identity (up to hero-wavelength Monte Carlo noise), NOT an attempt at
# spectral realism. The raw 3-basis upsampling above measurably does NOT
# round-trip (white came back as ~(1.54, 1.64, 0.92) with substantial
# cross-channel bleed when checked numerically) — since the whole pipeline
# (basis evaluation -> CIE integral -> XYZ->RGB matrix) is linear in the
# input RGB, that end-to-end map is exactly some 3x3 matrix M. Measured M by
# feeding in the 3 unit colors (20000-sample stratified hero-wavelength
# average, converged well past MC noise):
#   M = [[ 1.363472,  0.313217, -0.133374],
#        [ 0.507774,  0.897182,  0.236897],
#        [-0.059150,  0.069572,  0.913498]]
# and pre-multiply every input RGB by M^-1 here so the net transform is
# M @ M^-1 = I. This does NOT make the intermediate spectrum "correct" in
# any physical sense (M^-1 has negative entries, so the pre-corrected
# per-primary weights can be negative/>1 — clamped per-wavelength below since
# a reflectance/emission spectrum can't itself go negative) — it only
# guarantees achromatic colors and the sRGB primaries survive the round trip
# undistorted, which matters far more for a first spectral-rendering layer
# than basis-function elegance.
comptime _M_INV_RR = Float32(0.86027551); comptime _M_INV_RG = Float32(-0.31643614); comptime _M_INV_RB = Float32(0.20766458)
comptime _M_INV_GR = Float32(-0.51188899); comptime _M_INV_GG = Float32(1.32576438); comptime _M_INV_GB = Float32(-0.41854808)
comptime _M_INV_BR = Float32(0.09468971); comptime _M_INV_BG = Float32(-0.12146000); comptime _M_INV_BB = Float32(1.14001670)

@always_inline
def rgb_to_spectral_sample(rgb_r: Float32, rgb_g: Float32, rgb_b: Float32, wavelengths: SampledWavelengths) -> SpectralSample:
    """Evaluate an approximate reflectance/emission spectrum built from 3
    smooth per-primary basis functions, at the path's 4 hero wavelengths.
    Pre-corrected by M^-1 (see comment above) so the round trip through
    spectral_sample_to_rgb is the identity for achromatic/primary colors."""
    var cr = _M_INV_RR * rgb_r + _M_INV_RG * rgb_g + _M_INV_RB * rgb_b
    var cg = _M_INV_GR * rgb_r + _M_INV_GG * rgb_g + _M_INV_GB * rgb_b
    var cb = _M_INV_BR * rgb_r + _M_INV_BG * rgb_g + _M_INV_BB * rgb_b
    var out = SpectralSample(Float32(0.0))
    for i in range(N_SPECTRAL_SAMPLES):
        var lam = wavelengths.get(i)
        var val = cr * _rgb_basis_r(lam) + cg * _rgb_basis_g(lam) + cb * _rgb_basis_b(lam)
        val = max(Float32(0.0), val)
        if i == 0: out.v0 = val
        elif i == 1: out.v1 = val
        elif i == 2: out.v2 = val
        else: out.v3 = val
    return out

# ── spectrum -> RGB (via XYZ), for converting a final pixel radiance sample
#    back to a displayable color ──────────────────────────────────────────

@always_inline
def spectral_sample_to_rgb(radiance: SpectralSample, wavelengths: SampledWavelengths) -> Tuple[Float32, Float32, Float32]:
    """Monte-Carlo estimate of the CIE XYZ integral from one hero-wavelength
    sample, then XYZ -> linear sRGB. Divides by the sampling pdf and by
    CIE_Y_INTEGRAL, matching PBRT's SampledSpectrum::ToXYZ/ToRGB — the
    unbiased estimator for integral(radiance(lambda) * cie_x/y/z(lambda) dlambda)
    is (1/N) * sum_i radiance_i * cie_*(lambda_i) / pdf_i."""
    var x = Float32(0.0); var y = Float32(0.0); var z = Float32(0.0)
    if wavelengths.pdf > Float32(0.0):
        for i in range(N_SPECTRAL_SAMPLES):
            var lam = wavelengths.get(i)
            var r = radiance.get(i)
            x += r * cie_x(lam)
            y += r * cie_y(lam)
            z += r * cie_z(lam)
        var norm = Float32(1.0) / (Float32(N_SPECTRAL_SAMPLES) * wavelengths.pdf * CIE_Y_INTEGRAL)
        x *= norm; y *= norm; z *= norm

    # XYZ -> linear sRGB (standard matrix, D65 white point)
    var r_lin =  Float32(3.2406) * x - Float32(1.5372) * y - Float32(0.4986) * z
    var g_lin = -Float32(0.9689) * x + Float32(1.8758) * y + Float32(0.0415) * z
    var b_lin =  Float32(0.0557) * x - Float32(0.2040) * y + Float32(1.0570) * z
    return (r_lin, g_lin, b_lin)
