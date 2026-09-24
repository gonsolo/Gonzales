# Reflection Models

When a ray hits a surface, the renderer needs to know how light scatters
from that point. This is described by the Bidirectional Scattering
Distribution Function (BSDF), which tells us the ratio of reflected (or
transmitted) light for any pair of incoming and outgoing directions.
All material shading lives in `shading.mojo`.

## The Shading Framework

Rather than a protocol or virtual dispatch, gonzales uses compile-time
`@parameter` dispatch on a material type integer:

```mojo
fn shade_nee_core[use_gpu: Bool](path_ptr, mat, ...):
    if mat.type == 1:   # diffuse
        shade_diffuse(path_ptr, mat, ...)
    elif mat.type == 3: # conductor
        shade_conductor(path_ptr, mat, ...)
    elif mat.type == 4: # dielectric
        shade_dielectric(path_ptr, mat, ...)
    elif mat.type == 5: # coated diffuse
        shade_coated_diffuse(path_ptr, mat, ...)
```

The `use_gpu` compile-time parameter selects between GPU texture sampling
and CPU texture sampling within the same function body, keeping CPU and GPU
paths in sync with no code duplication.

## Diffuse Reflection

The simplest reflection model: light scatters equally in all directions above
the surface. The BSDF value is constant — just the reflectance divided by π:

```
f_diffuse(ω_i, ω_o) = albedo / π
```

The factor of 1/π comes from energy conservation: integrating a constant
BSDF over the hemisphere with the cosine weight must not exceed one.
Sampling is cosine-weighted hemisphere sampling, which matches the
distribution of the integrand and reduces variance.

## Dielectric Materials

Glass and water are dielectrics — they both reflect and transmit light.
`shade_dielectric` uses the Fresnel equations to determine the split
between reflection and refraction based on the angle of incidence and the
refractive index ratio.

For smooth surfaces, the BSDF is purely specular: light reflects in exactly
one direction. For rough surfaces, the Trowbridge-Reitz microfacet distribution
spreads the reflection into a lobe.

## Microfacet Reflection

The Trowbridge-Reitz (GGX) distribution models rough surfaces as a
collection of tiny flat mirrors (microfacets) oriented according to a
statistical distribution. The density function `D(ωh)` gives the density of
microfacets with half-vector ωh. Combined with the Fresnel term and the
Smith masking-shadowing function `G`, this produces physically plausible
glossy reflections.

## Coated and Layered BSDFs

Real materials often have multiple layers — a clear coat over diffuse paint,
lacquer over wood, a wet or oxidized metal surface. `coateddiffuse` is a
smooth or rough dielectric coat over a Lambertian base, and
`src/gonzales/layered.mojo` is a line-by-line port of PBRT's `LayeredBxDF`
for it: the rough `DielectricBxDF`, the Trowbridge-Reitz distribution, coat
thickness 0.01 with `exp(−thickness/|cos θ|)` attenuation per crossing, and
a random walk of at most ten interface events. All three integrators use it
as one lobe (`LobeKind.layered`) with a real `f` and forward and reverse
densities, so a coated vertex takes part in every MIS decision.

### The random walk

The layered BSDF has no closed form, so both halves of its interface are
estimators:

- **`layered_sample`** follows one analog path: sample the coat, and if the
  ray transmits, bounce between base and coat until it leaves through the
  top. The path's throughput is exactly `f·|cos|/pdf`, and its pdf is only
  proportional, so MIS weights call `layered_pdf` instead.
- **`layered_f`** evaluates `f(wo, wi)` for one given pair, by a short walk
  that at each base bounce does NEE in both directions: along a
  pre-sampled inside direction `wis` (from `wi` through the coat), and
  along the base's own sampled direction through the exit. The two
  estimates are combined with the power heuristic. The walk is seeded from
  a hash of the two directions, so `f` is a deterministic function and
  every strategy that asks for one pair gets one value.
- **`layered_pdf`** is a similar stochastic estimate, mixed 90/10 with a
  uniform density. It integrates to about 2, not 1, which is harmless for
  MIS because only ratios of densities enter it.

**Where gonzales departs from PBRT: the exit-NEE MIS weight.** PBRT weights
the base-sampled exit term with `PowerHeuristic(bs.pdf,
exitInterface.PDF(-w, wi))`. The competing strategy is `wis`, whose density
lives on *inside* directions and is `exitInterface.PDF(wi, -w)` (sampled
from `wi` toward `-w`). PBRT's second argument is a density over *outside*
directions given the inside one, so the two weights for one path don't sum
to 1. The two densities differ by the refraction Jacobian, which is never 1
for a rough coat. As a result, PBRT's `f` disagrees with its own `Sample_f`.
White base, rough coat (roughness 0.1), view near normal:

| Estimate | Albedo |
|---|---|
| `∫ f cos` with PBRT's weight | 0.698 |
| Analog walk (`Sample_f`) | 0.646 |
| `∫ f cos` with the corrected weight | 0.6453 |

PBRT's own `simplepath` integrator with `samplelights false` never calls
`f`, so it is an independent referee. On a coated floor under an area light,
it reads 0.0517. PBRT's `volpath` (which evaluates `f` in NEE) reads 0.0540
(+4%), `randomwalk` (`f` only) 0.0544, and gonzales 0.0522. The error only
matters where `f` carries the estimate: NEE from small or delta lights, and
VCM's connections and merges. Under uniform light, BSDF sampling holds most
of the MIS weight, so the white furnace can't see it. In VCM it produced a
brightness that grew with the light-path count (1.028 → 1.048 of the true
value from 4k to 512k paths), because merging evaluates `f` and gains MIS
weight as paths are added.

A consequence for comparisons: where sunlight or a point light hits a rough
coat directly, gonzales is now a few percent darker than PBRT, by design.

### Radiance compression: the η² factor

A ray crossing from a dense medium (the coat, index of refraction η) into
air doesn't just lose energy to reflection at the interface — the *solid
angle* it's confined to expands by η², and radiance (power per unit solid
angle per unit area) is diluted by exactly that factor. Combined with the
ordinary Fresnel transmission term `(1 − F(cosθ))` at both the light's
incidence angle and the viewer's angle (light has to get both *into* the
coat toward the base and *back out* toward the eye), the base's
contribution through the coat is:

```
f = albedo/π · (1 − F(cos θᵢ)) · (1 − F(cos θₒ)) / η²
```

Every term here — both Fresnel factors and the `1/η²` — is easy to miss
independently, and gonzales did for a while: without them a coated surface
measurably renders *brighter* than the same albedo left uncoated, which is
backwards (a dielectric coat can only ever attenuate the base, never
amplify it). At the default coat index of 1.5, the missing factor is worth
2.3× — large enough to be obviously wrong once checked against a reference,
but the kind of error that a spot check with a single test light angle can
still miss, since the two Fresnel terms partially cancel except near
grazing angles.

**Where that formula meets a stochastic walk, though, only *one* of the two
Fresnel factors may be written down.** The expression above is the BSDF as
an analytic function. Gonzales resolves the coat as a random walk, and a
ray reaches the base at all only by losing the entry coin flip against
`F(cos θₒ)` — so arriving there has *already* cost a factor of
`1 − F(cos θₒ)`, in expectation, with nothing dividing it back out. An NEE
weight evaluated at the base must therefore supply the light-side factor
only. Applying the view-side one again squares it.

That mistake is nearly invisible where it is cheapest to test — at normal
incidence and η = 1.5 it turns 0.96 into 0.92 — and ruinous where nobody
looks: as the view approaches grazing, `F → 1`, and the squared term
destroyed 56% of the energy at a 4° view. It is worth stating the general
form, because it recurs whenever an analytic BSDF is grafted onto a
sampling procedure: **a factor already paid by a sampling decision must not
be written into the estimator as well.** The reliable way to catch it is a
sweep, not a spot check — a single near-normal probe reports everything is
fine.

### Which reference is "correct"? Neither, exactly

It's tempting to treat any one renderer as ground truth when validating
against it, but this material is a good illustration of why that's not
quite right. Mitsuba's `plastic` BSDF is the *same* physical configuration
— smooth dielectric coat over Lambertian base — solved in closed form under
one specific assumption: that light re-randomizes to a uniform cosine
distribution on every internal bounce. That assumption lets it sum the
entire `TRT + TRTRT + ...` series analytically:

```
f = albedo/(1 − albedo·Fᵈᵢ) · (1 − F(cosθᵢ))(1 − F(cosθₒ)) / (π η²)
```

where `Fᵈᵢ` is the hemispherically-averaged internal Fresnel reflectance
(the fraction of light bouncing off the coat's underside from *any*
direction, not one specific angle). PBRT's `LayeredBxDF`, and gonzales's
walk above, instead track the *actual* directional distribution through
each bounce — a more expensive but more accurate model when the coat has
real (non-zero) thickness and the base isn't perfectly diffuse-Lambertian
in its own right.

There is one configuration where that closed form stops being an
approximation and becomes *exact*, which makes it a genuine ground truth
rather than a third opinion: a **smooth** coat over a **Lambertian** base.
The formula's whole assumption is that light re-randomizes to a cosine
distribution on each internal bounce — and a Lambertian base does exactly
that, by definition, at every bounce. `Scenes/coateddiffuse-grazing-probe.pbrt`
is built to sit in that configuration, and
`Scenes/coateddiffuse_analytic_check.py` evaluates the formula per pixel
(at grazing incidence `F` varies so steeply across the patch that the mean
of `f` and `f` of the mean angle differ by ~7%). Gonzales matches it to
0.1% across a 60°→4° view sweep. Reaching for an exact special case beats
arguing about which renderer to trust.

That same check settles a standing 4.5% gap against PBRT: PBRT's
`coateddiffuse` defaults to `thickness 0.01` and attenuates by
`exp(−thickness/cos θ)` on entry, exit, and every internal bounce. Set
PBRT's thickness to zero and it lands within 0.24% of the closed form.
The old gonzales coat model had no thickness at all, so that gap was a
missing *parameter*, not a wrong *transport*. It is worth knowing which of
those you are looking at before trying to "fix" a number. The
`LayeredBxDF` port now models the thickness exactly as PBRT does.

Evaluated on the same test geometry, PBRT and Mitsuba disagree with each
other by up to 8% at η = 2 — a real difference between two published,
peer-reviewed models, not a bug in either. So a discrepancy between
gonzales and any one reference of a similar size is not automatically a
defect; it may simply mean gonzales's stochastic walk is closer to one
model's assumptions than the other's. The practical approach is to target
whichever renderer's *specific algorithm* gonzales's own code most closely
mirrors (here, PBRT's directional walk) rather than chase exact numeric
agreement with a structurally different closed-form model.

### Numerical hygiene in a stochastic recycling loop

Two unrelated numerical-stability techniques recur throughout gonzales
wherever a value compounds over an unbounded number of Monte Carlo steps —
worth naming once here rather than re-deriving at each site:

- **Clamp the compounding *result*, not the individual step.** Each single
  operation (a Russian-roulette throughput compensation, one `beta *=
  albedo` recycle bounce) is an unbiased estimator on its own. The problem
  is a *streak*: many legitimate individual steps compounding into an
  extreme value — 2× per bounce is unremarkable, but thirty bounces of it
  is not. Clamping after each step, rather than only at the very end,
  trades a small bias for a large variance reduction and stops runaway
  values from ever being computed, rather than painting over them
  afterward the way an image-space firefly filter does.
- **A per-channel floor prevents color starvation, not just brightness
  blowup.** Raising an ordinary texture's albedo to a high power (many
  recycle bounces at one texel) can drive the weakest color channel toward
  zero while another channel stays large — an ordinary, mild per-channel
  imbalance becomes an extreme, visible color-saturated artifact once
  compounded. Flooring each channel at a small fraction of the strongest
  channel bounds how extreme a single texel's saturation can compound to,
  while still allowing real, order-of-magnitude color saturation through
  for textures that are actually strongly tinted.

## Mix BSDF

Materials can mix two BSDFs by a scalar weight. This enables partially
oxidized metal or wet surfaces without dedicated material models.

## Hair (Marschner)

Curve primitives use the Marschner model: a longitudinal lobe `Mp` (a von
Mises–Fisher distribution over the deviation from the perfect-cone
reflection angle, `bvh.mojo::_hair_Mp`) times an azimuthal term summing the
R, TT, and TRT light paths (single reflection off the cuticle, and the two
paths that refract through the fiber once or twice, each attenuated by an
absorption term `A0`/`A1`/`A2`/`A3` — `_hair_eval_lobes`). Unlike every other
material in gonzales, hair's NEE weight (`_nee_weight_hair`, `bxdf.mojo`)
stays RGB-only rather than spectral: converting Marschner's per-path
absorption to a genuine spectral quantity needs the fiber's `sigma_a`
threaded through as an absorption coefficient rather than a color, which is
a separate, larger piece of work than the plain reflectance/illuminant
upsampling every other material uses (see `docs/02_spectra_and_color.md`) —
a deliberate scope cut, not an oversight.

Curves can themselves be area lights (`AreaLight_C.kind == 1`, sampled as a
random point on the tube: a random piece, arc-position, and angle around the
circle). NEE, BDPT, and SPPM all reach curve lights through the same
generic `sample_area_light_uniform` codepath mesh lights use, needing no
curve-specific logic beyond the sampling branch itself — MNEE is the one
exception, staying mesh-only since it needs a flat `(dp_du, dp_dv)` tangent
basis a round tube doesn't have; curve lights behind glass fall back to
plain shadow-ray NEE.

## Measured BRDFs

The `measured` material type is a real port of pbrt-v4's Dupuy & Jakob
tabulated BRDF representation (`measured_bsdf.mojo`'s loader,
`measured_bxdf_eval.mojo`'s piecewise-linear-2D evaluation, ported directly
from pbrt-v4's `bxdfs.{h,cpp}`) — a `.bsdf` tensor file measured from a real
material (e.g. sportscar's car paint), not a fitted analytic model. Two
traps worth knowing if this code is touched again:

- **Composite before converting to RGB, not after.** Converting a bare
  reflectance spectrum to RGB in isolation implicitly assumes an
  equal-energy illuminant, whose white point doesn't match sRGB's D65
  reference — a flat 0.8 reflectance round-tripped to a visibly tinted
  RGB (0.96, 0.76, 0.73) instead of neutral gray. `bxdf_eval_measured`
  returns the raw spectral value and lets the caller composite it with the
  light's actual illuminant spectrum (or a neutral D65 shape, at sample
  time when no specific light is known yet) before the one-time
  `spectral_sample_to_rgb` conversion — the same reflectance-then-
  illuminant discipline every other material follows.
- **Fall back to the geometric normal, not the shading normal, for the
  same-hemisphere gate.** At near-edge-on viewing angles on a curved,
  low-poly mesh, an interpolated shading normal can land on the opposite
  side of `wo` from the geometric normal even when the geometric normal is
  correctly face-forwarded — a classic silhouette artifact. Gating
  `bxdf_eval_measured`'s reflect/transmit test on the shading-normal frame
  hard-zeroes every light sample at exactly those pixels; `shade_measured`
  falls back to the geometric normal whenever `dot(shading_normal, wo) <=
  0`, matching pbrt's own convention that the geometric normal gates which
  side is valid while the shading normal only shapes the lobe.
