# Spectra and Color

Physically based rendering requires an accurate model of light color. In the
real world, light is a continuous spectrum of wavelengths, and a real
material's or light's spectral response can do things no 3-value color space
can represent exactly — a spike in a fluorescent tube's emission, a metal's
reflectance sitting outside the sRGB gamut. All three of gonzales's
integrators (path tracer, VCM, SPPM) transport genuine spectral radiance —
`SpectralSample`, four hero-sampled wavelengths per path — converting to and
from RGB only at the two boundaries that need it: reading a material's
albedo from an image texture, and turning a finished pixel's radiance into a
displayable color. `RGB` survives as the type for those boundaries, for
denoiser AOVs (which are color-space quantities, not radiometric ones), and
as the plain, table-less fallback when no spectral data is loaded.

## The RGB Type

The workhorse boundary type is `RGB` in `geometry.mojo`:

```mojo
struct RGB(TrivialRegisterPassable):
    var r: Float32
    var g: Float32
    var b: Float32

    fn luma(self) -> Float32:
        return Float32(0.2126)*self.r + Float32(0.7152)*self.g + Float32(0.0722)*self.b
```

`TrivialRegisterPassable` ensures values land in registers and in
GPU-friendly flat path buffers with no indirection — the same reason
`SpectralSample` (below) uses it too. Global sentinel values (`RGB(0,0,0)`
for black, `RGB(1,1,1)` for white) appear throughout the shading code.

## Spectral Transport: Hero-Wavelength Sampling

Tracing a full continuous spectrum per path is intractable — instead,
gonzales follows Wilkie et al. 2014's *hero-wavelength* scheme, stratified
sampling of just four wavelengths per path (`SampledWavelengths`,
`spectrum.mojo`). One "hero" wavelength λ₀ is drawn uniformly over the
visible range (360–830 nm); the other three are placed at fixed strides of
one quarter of that range from it, wrapping around at the top end. All four
share one sampling pdf (`1/range`), since they're really one stratified draw,
not four independent ones. `SpectralSample` then carries a value at each of
the four — radiance, reflectance, or transmittance, depending on context —
and ordinary path-tracing arithmetic (`throughput *= f`, `estimate += c`)
runs identically to the RGB code it replaced, just widened to four lanes.

**A subtlety only bidirectional methods expose:** a BDPT/VCM connection
multiplies a camera vertex's throughput by a light vertex's flux, and a
photon-mapping merge does the same across independently-generated light
paths. That's only physically meaningful if lane *i* means the same
wavelength on both sides — so every subpath in one progressive pass (one VCM
sample, one SPPM photon pass) must share the *identical* four hero
wavelengths, derived from nothing but the pass index (`pass_wavelengths`).
Deriving wavelengths from a per-kernel RNG seed instead — the natural thing
to reach for, since camera and light kernels are seeded differently for
other reasons — silently breaks that invariant: measured, it disagreed
between backends by 4% on a saturated test scene, entirely chromatically,
and pinning one shared wavelength set collapsed that to 0.3% residual
(ordinary atomic-accumulation ordering noise).

**A second subtlety: how the pass-to-wavelength schedule is generated
matters more than it looks.** The natural choice — a low-discrepancy
sequence (golden-ratio or Halton-style strides) over the pass index — turns
out to alias against the hero-sampling stride itself: hero sampling is
already a lattice (four wavelengths at fixed span/4 offsets), so stacking a
second regular lattice on top of it produces a doubly-regular pattern that
systematically under- or over-samples parts of the CIE curves rather than
covering them. A hash of the pass index avoids this. Measured chroma error
against a reference, on a saturated test scene: 0.0135 for a golden-ratio
sequence, 0.0120 for a Halton-style one, 0.0026 for a hash — an order of
magnitude better, from simply not being regular. One consequence worth
knowing: the wavelength schedule is therefore the same across every
`--seed`, which is fine (everything else in the render is still seeded, and
a fixed schedule is one fewer thing that could differ between CPU and GPU).

## RGB ↔ Spectrum Conversion

Where a genuine spectrum isn't available — a texture's albedo authored as
RGB, a light's color given as `"rgb L"` in a scene file — gonzales upsamples
it to a plausible spectrum using the real Jakob & Hanika 2019 method: a
prebaked table (`rgb2spec.mojo`) maps any RGB triple to sigmoid-spectrum
coefficients, evaluated at the four hero wavelengths on demand. This is not
an approximation layered on top of RGB transport; it's the boundary
conversion for spectral transport, chosen because it's smooth, invertible,
and reproduces the input RGB exactly when re-integrated against the CIE
curves.

**Two different upsampling curves, and mixing them up is a real bug
class.** A material's reflectance and a light's emission are physically
different kinds of quantity, and pbrt's convention — which gonzales
follows — upsamples them differently:

- **Reflectance** (`spec_refl`) is bounded to [0, 1] and clamped there,
  since the table's domain is a physical reflectance and nothing legitimate
  exceeds it.
- **Illuminant/emission** (`spec_illum`) is *not* bounded — light intensity
  can be arbitrarily large — and instead tints the shape of the D65
  illuminant rather than standing alone as a bare reflectance curve.

Using the reflectance curve for an emitter, or vice versa, produces a
plausible-looking but physically wrong spectrum. A related, easy-to-miss
case: a Russian-roulette-compensated throughput or an `f/pdf` sampling
weight is reflectance-*shaped* (it multiplies onto a path's throughput the
same way an albedo does) but is legitimately allowed to exceed 1 — clamping
it the way ordinary reflectance is clamped silently destroys energy, and
does so per-channel, so it shifts color as well as brightness. gonzales
handles this (`spec_refl_unbounded`) by splitting off the largest RGB
component as a scalar multiplier and upsampling only the normalized,
in-range remainder, keeping chromaticity in the table's valid domain while
preserving the exact magnitude.

**A coefficient is not a color.** A participating medium's per-channel
extinction σₜ is a plain multiplier, not a reflectance or an emission, and
feeding one through the *illuminant* upsampler is genuinely meaningless —
`spec_illum` tints the D65 shape by construction, so a coefficient acquires
a spurious daylight-shaped colour that compounds once per scattering event.
Measured on a validation harness, a grey homogeneous medium read 3.5× its
known-correct analytic answer, and a heterogeneous (NanoVDB) one 10.7×.
The fix in place today is `rgb_bands_to_spectral_sample`, which picks
whichever of R, G, or B "owns" each hero wavelength by the usual sRGB-primary
crossover (blue below 490 nm, green to 580 nm, red above).

**Correction (2026-09-10).** This passage previously justified band-picking
by asserting that `RGB(1,1,1)` "does not upsample to exactly 1 in every
wavelength lane (the table has no reason to be flat there)". **That is
false, and it was measured.** `Tests/unit/test_coefficient_upsampling.mojo`
checks the actual table across four hero-wavelength sets: `spec_refl(1,1,1)`
returns exactly 1.0 in all four lanes, and `spec_refl_unbounded(2,2,2)`
exactly 2.0. The grey invariant is preserved by the smooth Jakob-Hanika
upsampler, not only by band-picking. Note also that the recorded symptom —
a *D65-shaped* tint — is the fingerprint of `spec_illum`, not of
`spec_refl`; the original fix was very likely right for a different reason
than the one written down.

This matters because it removes the stated blocker on matching pbrt, whose
medium coefficients use `RGBUnboundedSpectrum` — a smooth sigmoid, the
direct analogue of `spec_refl_unbounded`. Band-picking remains a defensible
choice, but it is a **3-band step function**: it carries no more chromatic
information than the original RGB had, and it is not the same spectral model
pbrt uses. See "Where this still falls short" below.

Band-picking is now used consistently for every medium *coefficient*: the
free-flight weights, and the single-scattering albedo at each of the three
places that consume it. Three of those used to push the albedo through the
reflectance upsampler instead, so a grey medium acquired a D65-shaped tint
once per scattering event — the same defect this section describes, just in
a spot nobody had checked.

### Where this still falls short, measured

It is tempting to grade chromatic media against the obvious closed form:
for a pure absorber of depth *d*, each channel should read `exp(-σ_c·d)`.
**That target is wrong, and real pbrt misses it too.** On
`Scenes/media-chromatic-absorb.pbrt` — σ_a = (0.25, 0.5, 1.0) across a
sphere of diameter 4, so τ = (1, 2, 4) against an L = 1 backdrop:

| centre 16×16 | R | G | B |
|---|---|---|---|
| naive `exp(-τ)` | 0.3679 | 0.1353 | 0.0183 |
| real pbrt-v4, 512 spp | 0.3321 | 0.1314 | 0.0093 |
| gonzales PT, 512 spp | 0.4332 | 0.1606 | 0.0068 |
| **pbrt / naive** | 0.90 | 0.97 | **0.51** |
| **gonzales / pbrt** | 1.30 | 1.22 | **0.73** |

`exp(-σ_c·d)` is an *RGB-renderer* expectation. Any genuinely spectral
renderer upsamples the authored RGB coefficient to a spectrum, exponentiates
per wavelength, and integrates against the CIE curves — and
`∫exp(-σ(λ)d)·x̄(λ)dλ` is not `exp(-σ_R·d)`. pbrt lands 2× off the naive
answer in blue for exactly this reason, and that is correct behaviour, not a
defect. The deviation is largest in blue because blue carries the largest τ,
where the exponential is most nonlinear and within-band variation of σ(λ)
matters most.

So the meaningful figure is the last row: **gonzales sits 1.30/1.22/0.73
against pbrt**, a chromatic spread of about 1.8×. That is a real gap and
worth closing, but it is roughly half the size that grading against the
naive analytic suggests. Two candidate causes, in order of likelihood:
band-picking models σ(λ) as a 3-band step function where pbrt uses a smooth
sigmoid, and gonzales band-picks the *transmittance* (already exponentiated)
rather than upsampling σ and exponentiating per lane — and for a smooth
upsampler those two orderings differ, since `upsample(exp(-σd))` ≠
`exp(-upsample(σ)·d)`. Band-picking is the one case where they coincide,
because selection commutes with `exp`.

**What remains, and it is measurable.** Sampling still draws the free-flight
distance from one channel (red) and reweights the others by their ratio to
it. That is unbiased, and a chromatic scattering test now shows VCM agreeing
with the path tracer to about 3% where it was 2.8x apart. But the *round
trip* — three authored RGB numbers, band-picked into four hero lanes,
reconstructed through the CIE curves — is lossy in a way no estimator can
undo. On a pure absorber with `sigma_a = (0.25, 0.5, 1.0)`, where the
analytic answer is exactly `exp(-tau)` per channel, every integrator
including the path tracer reads blue at roughly a quarter of it
(`Scenes/media-chromatic-absorb.pbrt`). A grey medium of the same optical
depth is correct to 3 decimal places, which locates the error in the
representation rather than in the transport.

Closing that needs a genuinely spectral `sigma_t(lambda)` rather than three
numbers, plus hero-wavelength free-flight sampling with MIS across
wavelengths. The MIS half is derivable — sample from one lane, weight by the
balance heuristic over all four lanes' free-flight densities — but it is
blocked on representation: `Medium_C` stores RGB, and SPPM's `VisiblePoint`
carries a deliberately RGB throughput (its `tau` accumulates across passes
whose hero wavelengths differ, so it *cannot* be spectral). Both are real
architectural commitments, not oversights.

**Converting back:** a finished pixel's spectral radiance sample is turned
back into a displayable color by a direct Monte Carlo estimate of the CIE
XYZ integral — `sum_i radiance_i * cie_xyz(lambda_i) / pdf_i`, using the
same tabulated CIE curves the upsampling table was fit against — followed by
the ordinary XYZ→linear-sRGB matrix.

## Metal Optical Constants

Metals like silver, aluminium, copper, and gold have wavelength-dependent
refractive indices and extinction coefficients. These are stored as arrays of
(wavelength, value) pairs sampled from measured data and looked up at render
time in `shading.mojo`.

## Black-Body Radiation

Light sources like incandescent bulbs emit radiation whose color depends on
temperature. The renderer approximates black-body color using a polynomial fit
that maps temperatures from candlelight (~1800 K, warm orange) through daylight
(~6500 K, neutral white) to overcast sky (~10000 K, bluish white).

## Gamma Correction

Linear-to-sRGB and sRGB-to-linear conversions are applied at texture load time
and image output time respectively. Getting this wrong is one of the most common
sources of washed-out or overly dark renders — all rendering happens in linear
light, displays expect sRGB.
