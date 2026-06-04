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

Real materials often have multiple layers — a clear coat over diffuse paint.
`shade_coated_diffuse` models a dielectric layer over a diffuse substrate
and includes NEE (next-event estimation) for direct lighting at both the
surface and the coat interface.

## Mix BSDF

Materials can mix two BSDFs by a scalar weight. This enables partially
oxidized metal or wet surfaces without dedicated material models.
