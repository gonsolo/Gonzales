# Path Tracing

Path tracing is the algorithm at the heart of gonzales. It estimates the
rendering equation by tracing random paths of light through the scene,
accumulating contributions from light sources at each surface interaction.

The implementation lives in `VolumePathIntegrator.swift` — a single 470-line
file that handles both surface and volume scattering.

## The Bounce Loop

Each pixel's color is estimated by tracing one or more paths. A path starts
at the camera, bounces off surfaces (or through volumes), and accumulates
light. The main loop iterates over bounces until the path terminates:

{{snippet:Sources/libgonzales/Integrator/VolumePathIntegrator.swift:bounce-loop}}

At each bounce, the integrator:

1. Finds the nearest surface intersection via the BVH
2. Adds direct emission if this is the first bounce or follows a specular bounce
3. Samples one light for direct illumination
4. Samples the BSDF to choose the next bounce direction
5. Applies Russian roulette to decide whether to continue

The `BounceState` struct carries all per-path state — the current ray,
accumulated estimate, throughput weight, and albedo for denoising.

## Multiple Importance Sampling

Direct lighting uses Multiple Importance Sampling (MIS) to combine two
sampling strategies:

1. **Light sampling** — sample a point on the light source, evaluate the BSDF
   for that direction
2. **BSDF sampling** — sample a direction from the BSDF, trace a ray to see
   if it hits a light

Neither strategy alone is optimal for all materials. MIS combines them using
the power heuristic:

```swift
let lightWeight = powerHeuristic(pdfF: lightPdf, pdfG: brdfPdf)
let lightContribution = lightEstimate * lightWeight / lightPdf

let brdfWeight = powerHeuristic(pdfF: brdfPdf, pdfG: lightPdf)
let brdfContribution = brdfEstimate * brdfWeight / brdfPdf

return lightContribution + brdfContribution
```

The power heuristic weights each sample by `pdf² / (pdf₁² + pdf₂²)`,
giving near-optimal variance in practice.

## Russian Roulette

Without termination, paths would bounce forever. Russian roulette provides
an unbiased way to stop paths probabilistically. When the throughput weight
drops below 1.0, the path is terminated with probability proportional to
the weight loss. Surviving paths are boosted to compensate:

{{snippet:Sources/libgonzales/Integrator/VolumePathIntegrator.swift:russian-roulette}}

This elegantly concentrates computation on paths that carry significant
energy while maintaining an unbiased estimate. The `bounce > 1` guard
ensures that at least two bounces are always computed, preserving direct
and first-indirect illumination quality.

## Volume Scattering

When a ray passes through a participating medium (fog, smoke, clouds),
it may scatter before reaching a surface. The integrator samples the
medium for a scattering event, evaluates direct lighting at the scattering
point using the phase function, and continues the path in a new direction
sampled from the phase function.
