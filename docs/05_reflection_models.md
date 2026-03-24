# Reflection Models

When a ray hits a surface, the renderer needs to know how light scatters
from that point. This is described by the Bidirectional Scattering
Distribution Function (BSDF), which tells us the ratio of reflected (or
transmitted) light for any pair of incoming and outgoing directions.

## The BSDF Framework

Gonzales defines a `FramedBsdf` protocol that all BSDF models conform to.
Each BSDF operates in a local coordinate system defined by a `BsdfFrame` —
the surface's shading normal and tangent vectors. The `ShadingFrame` type
handles the world-to-local and local-to-world transformations.

Every BSDF must implement three operations:

- **evaluate** — given outgoing and incident directions, return the BSDF value
- **sample** — given an outgoing direction and random numbers, generate a new
  incident direction with its PDF
- **probabilityDensity** — return the PDF for a given direction pair

## Diffuse Reflection

The simplest reflection model: light scatters equally in all directions above
the surface. The BSDF value is constant — just the reflectance divided by π:

{{snippet:Sources/libgonzales/Bsdf/DiffuseBsdf.swift:diffuse-bsdf}}

The factor of 1/π comes from energy conservation: integrating a constant
BSDF over the hemisphere with the cosine weight must not exceed one.
Sampling is cosine-weighted hemisphere sampling, which matches the
distribution of the integrand and reduces variance.

## Dielectric Materials

Glass and water are dielectrics — they both reflect and transmit light.
The `DielectricBsdf` uses the Fresnel equations to determine the split
between reflection and refraction based on the angle of incidence and the
refractive index ratio.

For smooth surfaces (low roughness), the BSDF is purely specular: light
reflects in exactly one direction and the PDF is a delta function. For
rough surfaces, the Trowbridge-Reitz microfacet distribution spreads the
reflection into a lobe.

## Microfacet Reflection

The Trowbridge-Reitz (GGX) distribution models rough surfaces as a
collection of tiny flat mirrors (microfacets) oriented according to a
statistical distribution. The key function `D(ωh)` gives the density of
microfacets with half-vector ωh. Combined with the Fresnel term and the
Smith masking-shadowing function `G`, this produces physically plausible
glossy reflections.

## Coated and Layered BSDFs

Real materials often have multiple layers — a clear coat over metallic paint,
or skin with subsurface scattering. Gonzales supports:

- **CoatedConductorBsdf** — a dielectric layer over a metallic substrate
- **CoatedDiffuseBsdf** — a dielectric layer over a diffuse substrate  
- **LayeredBsdf** — a general multi-layer model using random walks between
  interface boundaries

## Hair BSDF

The `HairBsdf` implements the Marschner hair scattering model, which
treats each hair fiber as a dielectric cylinder. It models three primary
scattering modes: R (surface reflection), TT (transmission through the
fiber), and TRT (internal reflection). This is essential for rendering
realistic hair and fur.

## Mix BSDF

The `MixBsdf` linearly blends two BSDFs using a scalar amount parameter.
This enables materials like partially oxidized metal (mix of conductor and
diffuse) without needing a dedicated material model for every combination.
