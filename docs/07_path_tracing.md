# Path Tracing

Path tracing is the algorithm at the heart of gonzales. It estimates the
rendering equation by tracing random paths of light through the scene,
accumulating contributions from light sources at each surface interaction.

The implementation lives in `shading.mojo` (material shaders) and
`rendering.mojo` / `gpu.mojo` (the bounce loop).

## The Bounce Loop

Each pixel's color is estimated by tracing one or more paths. A path starts
at the camera, bounces off surfaces, and accumulates light. The main loop
iterates over bounces until the path terminates. In `PathState_C`:

```mojo
# Per-path state carried across all bounces:
var throughput: RGB   # product of BSDF weights so far
var estimate: RGB     # accumulated radiance
var bounce: Int32     # current depth
var active: Int8      # 0 = terminated
```

At each bounce, the integrator:

1. Finds the nearest surface intersection via the BVH
2. Adds direct emission if this is the first bounce or a specular bounce
3. Samples one light for direct illumination (NEE)
4. Samples the BSDF to choose the next bounce direction
5. Updates throughput and applies Russian roulette

## Multiple Importance Sampling

Direct lighting uses Multiple Importance Sampling (MIS) to combine two
sampling strategies:

1. **Light sampling** — sample a point on the light source, evaluate the BSDF
2. **BSDF sampling** — sample a direction from the BSDF, check if it hits a light

Neither strategy alone is optimal for all materials. MIS combines them using
the power heuristic:

```
w_light = pdf_light² / (pdf_light² + pdf_bsdf²)
estimate = light_radiance * w_light / pdf_light
         + bsdf_radiance  * w_bsdf  / pdf_bsdf
```

The power heuristic gives near-optimal variance across a wide range of
material roughnesses.

## Russian Roulette

Without termination, paths would bounce forever. Russian roulette provides
an unbiased way to stop paths probabilistically. When the throughput weight
drops below 1.0, the path is terminated with probability `1 - luma(throughput)`.
Surviving paths are boosted to compensate:

```mojo
if path_ptr[].bounce > 1:
    var q = max(Float32(0.05), Float32(1) - path_ptr[].throughput.luma())
    if pcg_next_float(state, inc) < q:
        path_ptr[].active = Int8(0)
        return
    path_ptr[].throughput *= Float32(1) / (Float32(1) - q)
```

The `bounce > 1` guard ensures at least two bounces are always computed,
preserving direct and first-indirect illumination quality.

## CPU vs GPU Path

The same shading functions (`shade_nee_core[use_gpu: Bool]`) run on both CPU
and GPU. The `use_gpu` compile-time parameter selects GPU texture sampling vs
CPU OpenImageIO texture lookup inside the same function body. On the GPU, the
entire bounce loop is unrolled at compile time across a fixed `maxDepth`.
