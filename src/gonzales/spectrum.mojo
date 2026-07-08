# Foundational spectral-rendering types: hero-wavelength sampling and
# spectral-value arithmetic, plus the actual RGB<->spectrum conversion —
# which is the REAL Jakob & Hanika 2019 method (via rgb2spec.mojo's prebaked
# table + exact tabulated CIE curves), not an approximation. This used to
# hold a hand-designed 3-Gaussian-basis approximation (layer 1); that was
# retired once rgb2spec.mojo's real table was built and verified (layer 2)
# — see project_spectral_rendering memory for the staged history. This is
# layer 1+2 combined: self-contained and independently testable, NOT yet
# wired into any integrator. `SampledSpectrum` in geometry.mojo remains an
# alias for `RGB` until a later layer threads `SampledWavelengths` through
# the shading context and switches that alias over.

from gonzales.rgb2spec import (
    SpectrumTable, CieXyzTables, RGBSigmoidCoeffs,
    rgb_to_coeffs_table_lookup, eval_sigmoid_spectrum,
    rgb_illuminant_to_coeffs, eval_illuminant_spectrum,
    build_cie_xyz_tables, cie_xyz_at, xyz_to_srgb,
    load_default_spectrum_table, CIE_Y_INTEGRAL,
)

comptime N_SPECTRAL_SAMPLES = 4
comptime LAMBDA_MIN = Float32(360.0)
comptime LAMBDA_MAX = Float32(830.0)

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

# ── Spectral context: the loaded table + CIE data, built once per render ───

@fieldwise_init
struct SpectralContext(Copyable, Movable):
    """Bundles everything RGB<->spectrum conversion needs, built ONCE (e.g.
    at scene load, mirroring how gonzales already builds its sobol matrices
    once and threads a pointer through the whole render) and passed down to
    every conversion call — never reloaded/rebuilt per-sample."""
    var table: SpectrumTable
    var cie: CieXyzTables

def load_spectral_context(data_dir: String) -> Tuple[Bool, SpectralContext]:
    var loaded = load_default_spectrum_table(data_dir + "/rgb2spectrum_table.bin")
    var ok = loaded[0]
    var table = loaded[1].copy()
    if not ok:
        var empty = SpectralContext(table^, CieXyzTables(List[Float64](), List[Float64](), List[Float64]()))
        return (False, empty^)
    var cie = build_cie_xyz_tables()
    var ctx = SpectralContext(table^, cie^)
    return (True, ctx^)

# ── RGB -> spectrum upsampling (real Jakob-Hanika, via rgb2spec.mojo) ──────

@always_inline
def rgb_to_spectral_sample(ctx: SpectralContext, rgb_r: Float32, rgb_g: Float32, rgb_b: Float32, wavelengths: SampledWavelengths) -> SpectralSample:
    """Reflectance/albedo conversion — values are expected in [0,1] (clamped
    here defensively) since the table's domain is a bounded reflectance."""
    var r = Float64(rgb_r); var g = Float64(rgb_g); var b = Float64(rgb_b)
    if r < Float64(0.0): r = Float64(0.0)
    if r > Float64(1.0): r = Float64(1.0)
    if g < Float64(0.0): g = Float64(0.0)
    if g > Float64(1.0): g = Float64(1.0)
    if b < Float64(0.0): b = Float64(0.0)
    if b > Float64(1.0): b = Float64(1.0)
    var coeffs = rgb_to_coeffs_table_lookup(ctx.table.coeffs, ctx.table.res, r, g, b)
    return SpectralSample(
        eval_sigmoid_spectrum(coeffs, wavelengths.lambda0),
        eval_sigmoid_spectrum(coeffs, wavelengths.lambda1),
        eval_sigmoid_spectrum(coeffs, wavelengths.lambda2),
        eval_sigmoid_spectrum(coeffs, wavelengths.lambda3),
    )

@always_inline
def rgb_illuminant_to_spectral_sample(ctx: SpectralContext, rgb_r: Float32, rgb_g: Float32, rgb_b: Float32, wavelengths: SampledWavelengths) -> SpectralSample:
    """Light-emission conversion — PBRT's RGBIlluminantSpectrum convention
    (values are NOT bounded to [0,1], tints the D65 illuminant shape rather
    than standing alone as a bare reflectance)."""
    var (coeffs, scale) = rgb_illuminant_to_coeffs(ctx.table, Float64(rgb_r), Float64(rgb_g), Float64(rgb_b))
    return SpectralSample(
        eval_illuminant_spectrum(coeffs, scale, wavelengths.lambda0),
        eval_illuminant_spectrum(coeffs, scale, wavelengths.lambda1),
        eval_illuminant_spectrum(coeffs, scale, wavelengths.lambda2),
        eval_illuminant_spectrum(coeffs, scale, wavelengths.lambda3),
    )

# ── spectrum -> RGB (via XYZ), for converting a final pixel radiance sample
#    back to a displayable color ──────────────────────────────────────────

@always_inline
def spectral_sample_to_rgb(ctx: SpectralContext, radiance: SpectralSample, wavelengths: SampledWavelengths) -> Tuple[Float32, Float32, Float32]:
    """Monte-Carlo estimate of the CIE XYZ integral from one hero-wavelength
    sample (exact tabulated CIE curves, same data the table itself was
    fitted against), then XYZ -> linear sRGB. Divides by the sampling pdf and
    by CIE_Y_INTEGRAL, matching PBRT's SampledSpectrum::ToXYZ/ToRGB — the
    unbiased estimator for integral(radiance(lambda) * cie_x/y/z(lambda) dlambda)
    is (1/N) * sum_i radiance_i * cie_*(lambda_i) / pdf_i."""
    var x = Float64(0.0); var y = Float64(0.0); var z = Float64(0.0)
    if wavelengths.pdf > Float32(0.0):
        for i in range(N_SPECTRAL_SAMPLES):
            var lam = Float64(wavelengths.get(i))
            var r = Float64(radiance.get(i))
            var (xv, yv, zv) = cie_xyz_at(ctx.cie, lam)
            x += r * xv; y += r * yv; z += r * zv
        var norm = Float64(1.0) / (Float64(N_SPECTRAL_SAMPLES) * Float64(wavelengths.pdf) * Float64(CIE_Y_INTEGRAL))
        x *= norm; y *= norm; z *= norm

    var (r_lin, g_lin, b_lin) = xyz_to_srgb(x, y, z)
    return (Float32(r_lin), Float32(g_lin), Float32(b_lin))
