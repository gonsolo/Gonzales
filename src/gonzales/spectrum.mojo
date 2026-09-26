# Foundational spectral-rendering types: hero-wavelength sampling and
# spectral-value arithmetic, plus the actual RGB<->spectrum conversion —
# which is the REAL Jakob & Hanika 2019 method (via rgb2spec.mojo's prebaked
# table + exact tabulated CIE curves), not an approximation. This used to
# hold a hand-designed 3-Gaussian-basis approximation (layer 1); that was
# retired once rgb2spec.mojo's real table was built and verified (layer 2)
# — see project_spectral_rendering memory for the staged history. This is
# layer 1+2 combined. It IS now wired into every integrator: all three
# transport spectrally (see project_spectral_throughput_flip). geometry.mojo's
# `SampledSpectrum = RGB` alias, which this comment used to point at as the
# thing a later layer would "switch over", has been deleted -- the flip did
# not happen by swapping it, and afterwards a type named SampledSpectrum that
# was three floats only misled. RGB is now named RGB wherever a quantity
# genuinely has three authored channels.

from std.math import exp, log, min, max
from gonzales.sampling import mix_bits_u64
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
    """4 hero-sampled wavelengths (nm). Their sampling density is pbrt's
    visible-wavelength pdf, a pure function of each wavelength -- see pdf()."""
    var lambda0: Float32
    var lambda1: Float32
    var lambda2: Float32
    var lambda3: Float32

    @always_inline
    def get(self, i: Int) -> Float32:
        if i == 0: return self.lambda0
        elif i == 1: return self.lambda1
        elif i == 2: return self.lambda2
        else: return self.lambda3

    @always_inline
    def pdf(self, i: Int) -> Float32:
        """Sampling density (1/nm) of lane i; 0 outside [LAMBDA_MIN,
        LAMBDA_MAX], which is what the all-zero placeholder sets read."""
        return visible_wavelength_pdf(self.get(i))

@always_inline
def visible_wavelength_pdf(lam: Float32) -> Float32:
    """pbrt-v4's VisibleWavelengthsPDF: a sech^2 lobe centred on 538 nm,
    normalised over [360, 830]."""
    if lam < LAMBDA_MIN or lam > LAMBDA_MAX:
        return Float32(0)
    var e = exp(Float32(0.0072) * (lam - Float32(538)))
    var c = Float32(0.5) * (e + Float32(1) / e)
    return Float32(0.0039398042) / (c * c)

@always_inline
def _sample_visible_wavelength(u: Float32) -> Float32:
    """pbrt-v4's SampleVisibleWavelengths, the inverse CDF of the pdf above:
    538 - 138.888889 * atanh(0.85691062 - 1.82750197 u)."""
    var t = Float32(0.85691062) - Float32(1.82750197) * u
    var lam = Float32(538) - Float32(138.888889) * (Float32(0.5) * log((Float32(1) + t) / (Float32(1) - t)))
    # u = 1 exactly (a lane at u + 3/4 with u = 1/4) lands a hair past 830 in
    # Float32, where the pdf is 0; keep it inside the range.
    return min(max(lam, LAMBDA_MIN), LAMBDA_MAX)

@always_inline
def sample_wavelengths(u: Float32) -> SampledWavelengths:
    """pbrt-v4's SampledWavelengths::SampleVisible: stratified hero sampling
    (Wilkie et al. 2014) in the CDF's u, lanes at u + i/4 wrapped, each
    mapped through the visible-importance inverse CDF.

    Importance matters beyond variance. This used to draw uniformly over
    [360, 830], an unbiased estimator whose single-sample XYZ swings far
    above its mean at the spectrum's ends -- and a per-sample clamp
    (`maxcomponentvalue`) cuts exactly those swings, so the uniform
    estimator lost energy and saturation that pbrt keeps: a directly seen
    L = (5, 150, 10) lamp at limit 50 read G = 60 against pbrt's 68.8, and
    even a white L = 30 lamp, whose mean is well inside the limit, lost 5-13%
    of blue."""
    var lv = SIMD[DType.float32, 4](0)
    for i in range(N_SPECTRAL_SAMPLES):
        var up = u + Float32(i) / Float32(N_SPECTRAL_SAMPLES)
        if up > Float32(1):
            up -= Float32(1)
        lv[i] = _sample_visible_wavelength(up)
    return SampledWavelengths(lv[0], lv[1], lv[2], lv[3])

@always_inline
def pass_wavelengths(pass_idx: Int) -> SampledWavelengths:
    """The hero wavelengths for one progressive pass (a VCM spp sample, an
    SPPM photon pass), shared by EVERY camera and light subpath in it -- see
    docs/02_spectra_and_color.md ("Spectral Transport") for why a
    connection/merge requires this and the measured numbers behind the two
    traps below.

    TRAP 1: must be a pure function of the PASS INDEX alone, not the RNG
    seed -- light and camera kernels are launched with different seeds
    (`pass_seed` vs `base_seed`), so deriving wavelengths from either one
    gives the two subpaths of the SAME pass different wavelength sets.
    Measured: 4% CPU/GPU chroma mismatch on cornell-box, collapsing to 0.3%
    (ordinary atomics-ordering noise) once fixed.

    TRAP 2: a plain low-discrepancy sequence over u aliases:
    sample_wavelengths is itself a lattice (Wilkie et al. hero sampling at
    fixed 1/4 strides in u), so u and u + 1/4 give the same set,
    and e.g. a base-2 radical inverse of u repeats each set for 4 passes.
    A hash avoids that but leaves an O(1/sqrt(N)) error that is the SAME
    for every seed and scales the whole image: -3.7% luminance, -8% blue
    at 128 spp on the white furnace. The sequence goes on frac(4u)
    instead: within 0.1% per channel at every spp from 32 to 512.

    Consequence: the wavelength schedule is the same for every --seed,
    which is fine -- everything else in the render is still seeded, and a
    fixed schedule is one fewer thing that could differ between backends."""
    # The 4 lanes are a 1/4 lattice in u, so u and u + 1/4 give the same SET
    # of wavelengths: only frac(4u) matters. It is a base-2 radical inverse
    # of the pass index, so N passes cover the sets at O(1/N); the hashed
    # quarter only rotates which lane is the hero.
    var r = UInt32(pass_idx)
    r = (r << 16) | (r >> 16)
    r = ((r & 0x00FF00FF) << 8) | ((r & 0xFF00FF00) >> 8)
    r = ((r & 0x0F0F0F0F) << 4) | ((r & 0xF0F0F0F0) >> 4)
    r = ((r & 0x33333333) << 2) | ((r & 0xCCCCCCCC) >> 2)
    r = ((r & 0x55555555) << 1) | ((r & 0xAAAAAAAA) >> 1)
    var h = mix_bits_u64(UInt64(pass_idx) + UInt64(0x9E3779B97F4A7C15))
    var quarter = Float32(h & UInt32(3))
    var u = (Float32(r >> UInt32(8)) * Float32(1.0 / 16777216.0) + quarter) * Float32(0.25)
    return sample_wavelengths(u)

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

    # ── Operator surface matching RGB ─────────────────────────────────────
    # Path transport carries SpectralSample, so the arithmetic the integrators
    # already write against RGB (`throughput *= f`, `estimate += c`) has to
    # exist here too, or the flip becomes a rewrite instead of a type change.

    @always_inline
    def __sub__(self, o: SpectralSample) -> SpectralSample:
        return SpectralSample(self.v0 - o.v0, self.v1 - o.v1, self.v2 - o.v2, self.v3 - o.v3)

    @always_inline
    def __rmul__(self, s: Float32) -> SpectralSample:
        return self * s

    @always_inline
    def __iadd__(mut self, o: SpectralSample):
        self.v0 += o.v0; self.v1 += o.v1; self.v2 += o.v2; self.v3 += o.v3

    @always_inline
    def __isub__(mut self, o: SpectralSample):
        self.v0 -= o.v0; self.v1 -= o.v1; self.v2 -= o.v2; self.v3 -= o.v3

    @always_inline
    def __imul__(mut self, o: SpectralSample):
        self.v0 *= o.v0; self.v1 *= o.v1; self.v2 *= o.v2; self.v3 *= o.v3

    @always_inline
    def __imul__(mut self, s: Float32):
        self.v0 *= s; self.v1 *= s; self.v2 *= s; self.v3 *= s

    @always_inline
    def __itruediv__(mut self, s: Float32):
        var inv = Float32(1.0) / s
        self.v0 *= inv; self.v1 *= inv; self.v2 *= inv; self.v3 *= inv

    @always_inline
    def luma(self) -> Float32:
        """Scalar magnitude for Russian roulette / firefly tests. RGB.luma()
        is a CIE luminance; the hero-wavelength analogue is the mean over the
        sampled wavelengths (what pbrt uses for the same purpose), NOT a
        luminance -- these 4 values are radiance at arbitrary wavelengths and
        have no fixed luminous weighting."""
        return self.average()

    @always_inline
    def max_component(self) -> Float32:
        var m = self.v0
        if self.v1 > m: m = self.v1
        if self.v2 > m: m = self.v2
        if self.v3 > m: m = self.v3
        return m

    @always_inline
    def is_black(self) -> Bool:
        return (self.v0 <= Float32(0.0) and self.v1 <= Float32(0.0)
                and self.v2 <= Float32(0.0) and self.v3 <= Float32(0.0))

@always_inline
def _band_pick(r: Float32, g: Float32, b: Float32, lam: Float32) -> Float32:
    """Which sRGB primary owns wavelength `lam`, at the usual crossovers."""
    if lam < Float32(490.0): return b
    elif lam < Float32(580.0): return g
    return r

@always_inline
def rgb_bands_to_spectral_sample(
    r: Float32, g: Float32, b: Float32, wl: SampledWavelengths
) -> SpectralSample:
    """Evaluate a per-CHANNEL COEFFICIENT (not a colour) at the 4 hero
    wavelengths, by picking whichever of r/g/b owns each wavelength's band
    (usual sRGB-primary crossover: blue below 490nm, green to 580nm, red
    above) -- NOT rgb_to_spectral_sample / _illuminant_, which reconstruct a
    reflectance/emission spectrum and are meaningless fed a ratio, since
    RGB(1,1,1) doesn't come back as 1 in every lane (see
    docs/02_spectra_and_color.md, "A coefficient is not a color"). A grey
    ratio (every channel exactly 1) must come back as exactly 1 in every
    lane; band-picking gives that, the reflectance/illuminant upsamplers
    don't -- measured on the slab harness at tau=8, a grey medium read 3.5x
    (homogeneous) / 10.7x (NanoVDB) its analytic answer before this fix. A
    real chromatic-extinction fit needs hero-wavelength free-flight
    sampling with MIS across wavelengths, which is separate work (see
    _sample_medium_core's own note)."""
    return SpectralSample(
        _band_pick(r, g, b, wl.lambda0), _band_pick(r, g, b, wl.lambda1),
        _band_pick(r, g, b, wl.lambda2), _band_pick(r, g, b, wl.lambda3))

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
    var coeffs: Pointer[Float32, MutUntrackedOrigin]
    var res:    Int
    var cie_x:  Pointer[Float32, MutUntrackedOrigin]
    var cie_y:  Pointer[Float32, MutUntrackedOrigin]
    var cie_z:  Pointer[Float32, MutUntrackedOrigin]
    var d65:    Pointer[Float32, MutUntrackedOrigin]

# ── Boundary conversions, in DECOMPOSED-pointer form ────────────────────────
# These take SpectralHandle's fields as individual params rather than the
# handle itself. The first version of these helpers took `h: SpectralHandle`
# by value across a real Mojo call boundary and produced a BDPT/VCM
# CPU-vs-GPU mismatch of 4% on cornell-box that survived to 256 spp, on code
# both backends share; decomposing the parameters made it go away. This was
# suspected as a compiler miscompilation (modular/modular#6759), but the
# report was later retracted by its own author as unreproducible -- treat it
# as an unexplained anomaly with a working code-level fix, not a confirmed
# bug. Keep them decomposed anyway, defensively.

@always_inline
def spec_refl(
    coeffs: Pointer[Float32, MutUntrackedOrigin], res: Int,
    cie_x: Pointer[Float32, MutUntrackedOrigin],
    cie_y: Pointer[Float32, MutUntrackedOrigin],
    cie_z: Pointer[Float32, MutUntrackedOrigin],
    d65: Pointer[Float32, MutUntrackedOrigin],
    r: Float32, g: Float32, b: Float32, wl: SampledWavelengths,
) -> SpectralSample:
    """RGB REFLECTANCE -> spectral, at the material boundary."""
    return rgb_to_spectral_sample(coeffs, res, cie_x, cie_y, cie_z, d65, r, g, b, wl)

@always_inline
def spec_refl_unbounded(
    coeffs: Pointer[Float32, MutUntrackedOrigin], res: Int,
    cie_x: Pointer[Float32, MutUntrackedOrigin],
    cie_y: Pointer[Float32, MutUntrackedOrigin],
    cie_z: Pointer[Float32, MutUntrackedOrigin],
    d65: Pointer[Float32, MutUntrackedOrigin],
    r: Float32, g: Float32, b: Float32, wl: SampledWavelengths,
) -> SpectralSample:
    """A reflectance-shaped WEIGHT that may exceed 1 (a Russian-roulette-
    compensated throughput, an f/pdf ratio), upsampled without clamping --
    spec_refl's plain path clamps to [0,1], which destroys energy per
    channel (so it shifts colour, not just brightness) for a legitimately
    >1 weight. Splits off twice the largest RGB component as a scalar and
    upsamples only the normalised remainder, keeping chromaticity in the
    well-conditioned middle of the table's domain while preserving the exact
    magnitude (PBRT's RGBUnboundedSpectrum); the ILLUMINANT
    curve isn't a substitute here since it's the wrong spectral shape for a
    reflectance. See docs/02_spectra_and_color.md ("RGB <-> Spectrum
    Conversion")."""
    var m = max(r, max(g, b))
    if m <= Float32(0.0):
        return SpectralSample(Float32(0.0))
    # PBRT's RGBUnboundedSpectrum: normalise by 2*max, ALWAYS -- not by max,
    # and not only when max > 1. Both halves matter. Dividing by max puts the
    # largest component at exactly 1.0, the saturated EDGE of the sigmoid
    # fit's domain, where the polynomial has to blow up to reach 1 and the
    # fitted shape is at its least accurate; 2*max puts it at 0.5, mid-domain
    # where the fit is well conditioned. And skipping the normalisation for
    # max <= 1 silently switched conventions at m == 1, so a medium with
    # "rgb sigma_a" [0.25 0.5 1.0] was fitted with blue pinned to that edge --
    # measured 0.41x PBRT's blue on Scenes/media-chromatic-absorb.pbrt.
    # The grey invariant is unaffected: an achromatic fit is exactly flat, so
    # 2m * fit(0.5) == m == fit(m).
    var scale = Float32(2.0) * m
    var inv = Float32(1.0) / scale
    return rgb_to_spectral_sample(coeffs, res, cie_x, cie_y, cie_z, d65,
                                  r * inv, g * inv, b * inv, wl) * scale

@always_inline
def spec_illum(
    coeffs: Pointer[Float32, MutUntrackedOrigin], res: Int,
    cie_x: Pointer[Float32, MutUntrackedOrigin],
    cie_y: Pointer[Float32, MutUntrackedOrigin],
    cie_z: Pointer[Float32, MutUntrackedOrigin],
    d65: Pointer[Float32, MutUntrackedOrigin],
    r: Float32, g: Float32, b: Float32, wl: SampledWavelengths,
) -> SpectralSample:
    """RGB EMISSION/RADIANCE -> spectral, at the light boundary. Uses the
    ILLUMINANT upsampling, a DIFFERENT curve from spec_refl's -- see
    _to_spec_illum in shading.mojo for what mixing them up costs."""
    return rgb_illuminant_to_spectral_sample(coeffs, res, cie_x, cie_y, cie_z, d65, r, g, b, wl)

@always_inline
def spectral_handle(mut ctx: SpectralContext) -> SpectralHandle:
    return SpectralHandle(
        ctx.table.coeffs.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), ctx.table.res,
        ctx.cie.x_tbl.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        ctx.cie.y_tbl.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        ctx.cie.z_tbl.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        ctx.cie.d65_tbl.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
    )

@always_inline
def null_spectral_handle() -> SpectralHandle:
    """Dangling-pointer sentinel (mirrors bvh.mojo's SceneView
    dangling-texture convention and guide.mojo's null_guide()) for call
    sites that don't have a real spectral table loaded yet (BDPT/SPPM,
    Stage 3/4; test fixtures) — never dereferenced by code that doesn't
    consume it, same as those other sentinels."""
    return SpectralHandle(
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), 0,
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    )

# ── RGB -> spectrum upsampling (real Jakob-Hanika, via rgb2spec.mojo) ──────

# NOTE on the functions below: they take SpectralHandle's fields DECOMPOSED
# into individual pointer/int parameters, NOT `handle: SpectralHandle` as a
# single by-value struct argument. This mirrors gpu.mojo's GPU-kernel
# parameter convention (forced there by DevicePassable), but here it's a
# separate, CPU-only precaution: passing the 6-field SpectralHandle struct BY
# VALUE across a real (non-inlined) Mojo function-call boundary appeared to
# corrupt one field (observed on the `d65` pointer specifically, via
# cie_d65_runtime's result -- near-zero, sign-flipped, or NaN) on a random
# subset of otherwise-identical process runs, in both `mojo run` and a
# `mojo build`-compiled binary, while every leaf function was 100%
# deterministic in isolation and passing the same 6 fields as separate
# scalar arguments was 100% stable across 20+ repeated runs each. A second,
# similar-looking case turned up in gpu.mojo's gpu_upload_scene /
# pipeline.mojo's _gpu_upload_scene (2026-07-09, task #130), which took
# `spectral: SpectralHandle` by value in the GPU-enabled compilation unit and
# made every --gpu render using it come back black; fixed the same way.
#
# This was filed upstream as modular/modular#6759; the reporter later
# retracted it themselves as unreproducible, after 250+ further trials at
# the exact historical commit/toolchain came back clean. So this is NOT a
# confirmed compiler bug -- treat both instances as unexplained anomalies
# with a working code-level fix. SpectralHandle is threaded through several
# nested calls here (bxdf_eval_any_spectral -> rgb_to_spectral_sample, etc.),
# so every function on the path got the same decomposed-parameter treatment
# defensively -- see bxdf.mojo's spectral siblings and shading.mojo's call
# sites, which pass ctx.spectral.coeffs/.res/.cie_x/etc. instead of
# ctx.spectral.
@always_inline
def rgb_to_spectral_sample(
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    rgb_r: Float32, rgb_g: Float32, rgb_b: Float32, wavelengths: SampledWavelengths,
) -> SpectralSample:
    """Reflectance/albedo conversion — values are expected in [0,1] (clamped
    here defensively) since the table's domain is a bounded reflectance."""
    # No table loaded (null_spectral_handle): carry R/G/B on lanes 0/1/2 --
    # spectral_sample_to_rgb's matching fallback reads them straight back, so
    # transport degrades to the old per-channel RGB renderer rather than
    # dereferencing the sentinel's dangling table pointers.
    if spectral_res <= 0:
        return SpectralSample(rgb_r, rgb_g, rgb_b, Float32(0.0))
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
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    rgb_r: Float32, rgb_g: Float32, rgb_b: Float32, wavelengths: SampledWavelengths,
) -> SpectralSample:
    """Light-emission conversion — PBRT's RGBIlluminantSpectrum convention
    (values are NOT bounded to [0,1], tints the D65 illuminant shape rather
    than standing alone as a bare reflectance)."""
    # No table loaded (null_spectral_handle): carry R/G/B on lanes 0/1/2 --
    # spectral_sample_to_rgb's matching fallback reads them straight back, so
    # transport degrades to the old per-channel RGB renderer rather than
    # dereferencing the sentinel's dangling table pointers.
    if spectral_res <= 0:
        return SpectralSample(rgb_r, rgb_g, rgb_b, Float32(0.0))
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
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    radiance: SpectralSample, wavelengths: SampledWavelengths,
) -> Tuple[Float32, Float32, Float32]:
    """Monte-Carlo estimate of the CIE XYZ integral from one hero-wavelength
    sample (exact tabulated CIE curves, same data the table itself was
    fitted against), then XYZ -> linear sRGB. Divides by the sampling pdf and
    by CIE_Y_INTEGRAL, matching PBRT's RGB::ToXYZ/ToRGB — the
    unbiased estimator for integral(radiance(lambda) * cie_x/y/z(lambda) dlambda)
    is (1/N) * sum_i radiance_i * cie_*(lambda_i) / pdf_i."""
    # No table loaded (null_spectral_handle): lanes 0/1/2 carry plain R/G/B,
    # see rgb_to_spectral_sample's matching fallback. The two compose to the
    # identity, so a table-less build degrades cleanly to the old per-channel
    # RGB renderer instead of dereferencing the sentinel's dangling pointers.
    if spectral_res <= 0:
        return (radiance.v0, radiance.v1, radiance.v2)
    var x = Float32(0.0); var y = Float32(0.0); var z = Float32(0.0)
    for i in range(N_SPECTRAL_SAMPLES):
        var p = wavelengths.pdf(i)
        if p > Float32(0.0):
            var lam = wavelengths.get(i)
            var r = radiance.get(i) / p
            var (xv, yv, zv) = cie_xyz_at_ptr(spectral_cie_x, spectral_cie_y, spectral_cie_z, lam)
            x += r * xv; y += r * yv; z += r * zv
    var norm = Float32(1.0) / (Float32(N_SPECTRAL_SAMPLES) * CIE_Y_INTEGRAL)
    x *= norm; y *= norm; z *= norm

    return xyz_to_srgb(x, y, z)
