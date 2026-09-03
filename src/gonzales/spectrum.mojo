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
    rgb_to_coeffs_table_lookup_ptr, eval_sigmoid_spectrum,
    rgb_illuminant_to_coeffs_ptr, eval_illuminant_spectrum,
    build_cie_xyz_tables, cie_xyz_at_ptr, xyz_to_srgb,
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
    once and threads a pointer through the whole render). OWNS its buffers
    (SpectrumTable.coeffs / CieXyzTables.*_tbl are List[...]) — do NOT thread
    this struct itself into per-bounce code (see rgb2spec.mojo's "owning
    table types" docstring for why). Call spectral_handle(ctx) ONCE after
    loading to get a cheap, TrivialRegisterPassable SpectralHandle, and
    thread THAT through the render instead. The owning SpectralContext value
    must simply stay alive (in scope) for the whole render, since the
    handle's raw pointers point into its buffers."""
    var table: SpectrumTable
    var cie: CieXyzTables

def load_spectral_context(data_dir: String) -> Tuple[Bool, SpectralContext]:
    var loaded = load_default_spectrum_table(data_dir + "/rgb2spectrum_table.bin")
    var ok = loaded[0]
    var table = loaded[1].copy()
    if not ok:
        var empty = SpectralContext(table^, CieXyzTables(List[Float32](), List[Float32](), List[Float32](), List[Float32]()))
        return (False, empty^)
    var cie = build_cie_xyz_tables()
    var ctx = SpectralContext(table^, cie^)
    return (True, ctx^)

@fieldwise_init
struct SpectralHandle(TrivialRegisterPassable):
    """Cheap-to-copy view into a SpectralContext's buffers — raw pointers
    only, safe to reconstruct/pass every bounce/every NEE sample (unlike
    SpectralContext itself, which owns the underlying List buffers). The
    SpectralContext this was built from must outlive every use of the
    handle."""
    var coeffs: UnsafePointer[Float32, MutExternalOrigin]
    var res:    Int
    var cie_x:  UnsafePointer[Float32, MutExternalOrigin]
    var cie_y:  UnsafePointer[Float32, MutExternalOrigin]
    var cie_z:  UnsafePointer[Float32, MutExternalOrigin]
    var d65:    UnsafePointer[Float32, MutExternalOrigin]

@always_inline
def spectral_handle(mut ctx: SpectralContext) -> SpectralHandle:
    return SpectralHandle(
        ctx.table.coeffs.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](), ctx.table.res,
        ctx.cie.x_tbl.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
        ctx.cie.y_tbl.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
        ctx.cie.z_tbl.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
        ctx.cie.d65_tbl.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
    )

@always_inline
def null_spectral_handle() -> SpectralHandle:
    """Dangling-pointer sentinel (mirrors bvh.mojo's SceneDescriptor2_C
    dangling-texture convention and guide.mojo's null_guide()) for call
    sites that don't have a real spectral table loaded yet (BDPT/SPPM,
    Stage 3/4; test fixtures) — never dereferenced by code that doesn't
    consume it, same as those other sentinels."""
    return SpectralHandle(
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(), 0,
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    )

# ── RGB -> spectrum upsampling (real Jakob-Hanika, via rgb2spec.mojo) ──────

# NOTE on the functions below: they take SpectralHandle's fields DECOMPOSED
# into individual pointer/int parameters, NOT `handle: SpectralHandle` as a
# single by-value struct argument. This mirrors gpu.mojo's GPU-kernel
# parameter convention (forced there by DevicePassable), but here it's for a
# different, CPU-only reason: passing the 6-field SpectralHandle struct BY
# VALUE across a real (non-inlined) Mojo function-call boundary is a
# confirmed, reproducible MISCOMPILATION -- one field (observed on the `d65`
# pointer specifically, via cie_d65_runtime's result) comes back corrupted
# (near-zero, sign-flipped, or NaN) on a random subset of otherwise-identical
# process runs, in both `mojo run` and a `mojo build`-compiled binary. Fully
# root-caused via scratch diagnostics: the table data, the CIE data, and
# every leaf function are 100% deterministic in isolation; only wrapping the
# same calls in a function that receives SpectralHandle BY VALUE reproduces
# it. Passing the same 6 fields as separate scalar arguments (or a pointer
# TO a SpectralHandle) is 100% stable across 20+ repeated runs each. Given
# SpectralHandle is threaded through several nested calls here
# (bxdf_eval_any_spectral -> rgb_to_spectral_sample, etc.), every function on
# the path needed the same treatment -- see bxdf.mojo's spectral siblings and
# shading.mojo's call sites, which pass ctx.spectral.coeffs/.res/.cie_x/etc.
# instead of ctx.spectral.
#
# SECOND CONFIRMED INSTANCE (2026-07-09, task #130): gpu.mojo's
# gpu_upload_scene/pipeline.mojo's _gpu_upload_scene took `spectral:
# SpectralHandle` by value too -- in the GPU-enabled (--target-accelerator)
# compilation unit this reproduced the exact same corruption class
# (spectral_res read back as 0, coeffs pointer read back as a tiny garbage
# address), making the whole spectral NEE pipeline evaluate against a
# 1-element dummy table and every --gpu render using it come back black.
# Same fix applied there. Two independent reproductions of the same failure
# mode across different call chains and compilation targets -- worth filing
# as a Mojo compiler bug upstream rather than assuming more instances won't
# turn up; any NEW function that takes a TrivialRegisterPassable struct
# containing pointer fields by value should be treated as suspect until
# proven otherwise.
@always_inline
def rgb_to_spectral_sample(
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin],
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin],
    rgb_r: Float32, rgb_g: Float32, rgb_b: Float32, wavelengths: SampledWavelengths,
) -> SpectralSample:
    """Reflectance/albedo conversion — values are expected in [0,1] (clamped
    here defensively) since the table's domain is a bounded reflectance."""
    var r = rgb_r; var g = rgb_g; var b = rgb_b
    if r < Float32(0.0): r = Float32(0.0)
    if r > Float32(1.0): r = Float32(1.0)
    if g < Float32(0.0): g = Float32(0.0)
    if g > Float32(1.0): g = Float32(1.0)
    if b < Float32(0.0): b = Float32(0.0)
    if b > Float32(1.0): b = Float32(1.0)
    var coeffs = rgb_to_coeffs_table_lookup_ptr(spectral_coeffs, spectral_res, r, g, b)
    var v0 = eval_sigmoid_spectrum(coeffs, wavelengths.lambda0)
    var v1 = eval_sigmoid_spectrum(coeffs, wavelengths.lambda1)
    var v2 = eval_sigmoid_spectrum(coeffs, wavelengths.lambda2)
    var v3 = eval_sigmoid_spectrum(coeffs, wavelengths.lambda3)
    return SpectralSample(v0, v1, v2, v3)

@always_inline
def rgb_illuminant_to_spectral_sample(
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin],
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin],
    rgb_r: Float32, rgb_g: Float32, rgb_b: Float32, wavelengths: SampledWavelengths,
) -> SpectralSample:
    """Light-emission conversion — PBRT's RGBIlluminantSpectrum convention
    (values are NOT bounded to [0,1], tints the D65 illuminant shape rather
    than standing alone as a bare reflectance)."""
    var (coeffs, scale) = rgb_illuminant_to_coeffs_ptr(spectral_coeffs, spectral_res, rgb_r, rgb_g, rgb_b)
    var v0 = eval_illuminant_spectrum(coeffs, scale, spectral_d65, wavelengths.lambda0)
    var v1 = eval_illuminant_spectrum(coeffs, scale, spectral_d65, wavelengths.lambda1)
    var v2 = eval_illuminant_spectrum(coeffs, scale, spectral_d65, wavelengths.lambda2)
    var v3 = eval_illuminant_spectrum(coeffs, scale, spectral_d65, wavelengths.lambda3)
    return SpectralSample(v0, v1, v2, v3)

# ── spectrum -> RGB (via XYZ), for converting a final pixel radiance sample
#    back to a displayable color ──────────────────────────────────────────

@always_inline
def spectral_sample_to_rgb(
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin],
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin],
    radiance: SpectralSample, wavelengths: SampledWavelengths,
) -> Tuple[Float32, Float32, Float32]:
    """Monte-Carlo estimate of the CIE XYZ integral from one hero-wavelength
    sample (exact tabulated CIE curves, same data the table itself was
    fitted against), then XYZ -> linear sRGB. Divides by the sampling pdf and
    by CIE_Y_INTEGRAL, matching PBRT's SampledSpectrum::ToXYZ/ToRGB — the
    unbiased estimator for integral(radiance(lambda) * cie_x/y/z(lambda) dlambda)
    is (1/N) * sum_i radiance_i * cie_*(lambda_i) / pdf_i."""
    var x = Float32(0.0); var y = Float32(0.0); var z = Float32(0.0)
    if wavelengths.pdf > Float32(0.0):
        for i in range(N_SPECTRAL_SAMPLES):
            var lam = wavelengths.get(i)
            var r = radiance.get(i)
            var (xv, yv, zv) = cie_xyz_at_ptr(spectral_cie_x, spectral_cie_y, spectral_cie_z, lam)
            x += r * xv; y += r * yv; z += r * zv
        var norm = Float32(1.0) / (Float32(N_SPECTRAL_SAMPLES) * wavelengths.pdf * CIE_Y_INTEGRAL)
        x *= norm; y *= norm; z *= norm

    return xyz_to_srgb(x, y, z)
