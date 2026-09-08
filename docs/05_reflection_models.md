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
lacquer over wood, a wet or oxidized metal surface. `shade_coated_diffuse`
models the common case: a smooth or rough dielectric coat (car paint, gloss
varnish) over a Lambertian base. It follows the same shape as PBRT's
`LayeredBxDF`, but resolves it as one continuous stochastic random walk
rather than a closed-form integral.

### The random walk

At the coat's air interface, the exact dielectric Fresnel term splits the
ray probabilistically: reflect off the coat as a glossy (or, for a smooth
coat like fresh lacquer, mirror) lobe, or transmit into the coat toward the
base. A transmitted ray enters a loop, capped at ten iterations:

1. **Scatter off the diffuse base.** NEE samples every light type against
   the base's Lambertian response, attenuated by how much of that light's
   own incoming and outgoing directions actually make it through the coat
   (see below). Sample a new, cosine-weighted outgoing direction.
2. **Hit the coat's underside from inside.** The dielectric Fresnel term
   (now evaluated from inside the denser medium, hence `1/η`) again splits
   probabilistically: escape through the coat into air, or total-internally
   reflect and recycle back down to the base for another bounce.

Each recycled bounce multiplies an accumulator `beta` by the base albedo, so
after `n` bounces the light carries `albedo^n` — this is what saturates a
coated material's color relative to the same albedo left uncoated: light
that would have escaped after one bounce on a bare diffuse surface instead
gets a second, third, or further chance to pick up the base's tint before
it finally exits. A textured base makes this compounding sensitive to a
single texel's own color imbalance (see "Numerical hygiene" below).

Where gonzales's walk differs deliberately from PBRT's: `LayeredBxDF::f()`
reuses *one* correlated light sample across every recycled bounce to
evaluate the whole `TRT`, `TRTRT`, ... series in one pass. Gonzales instead
draws an independent, fresh light sample at every iteration and fires NEE
every time — the same expected energy, decorrelated, and simpler to reason
about at the cost of one shadow ray per recycle bounce (bounded by a
Russian-roulette gate once `beta` has decayed past a few bounces, so a
low-albedo coat's walk terminates quickly and a high-albedo one doesn't
flood the renderer with shadow rays).

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
