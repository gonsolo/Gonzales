# Introduction

Gonzales is a physically based renderer written in Mojo. It implements
a path tracer capable of rendering production-quality scenes — including
Disney's Moana Island — with both CPU and GPU execution via Mojo's unified
compute model.

## Why Mojo?

Most renderers are written in C++. Mojo offers an alternative with Python-like
syntax, zero-cost abstractions, and first-class GPU support. The same kernel
code targets CPU SIMD and GPU warps without separate codepaths — Mojo's
`@parameter if has_accelerator()` compile-time dispatch selects the right
backend at build time. This is what allows gonzales to run identical path
tracing logic on both CPU and NVIDIA GPU from a single ~7,800-line codebase.

On the CPU side, `std.algorithm.parallelize` distributes tiles across all
logical cores. On the GPU side, Mojo's `DeviceContext` and `@gpu` functions
drive wavefront path tracing with no CUDA boilerplate.

## Architecture

The renderer is organized into focused Mojo modules:

| Module | Lines | Responsibility |
|--------|------:|---------------|
| `parsing.mojo` | 1,959 | PBRT-v4 scene parser |
| `gpu.mojo` | 1,294 | GPU kernels: wavefront path tracing, à-trous denoiser |
| `shading.mojo` | 770 | Material shading — diffuse, coated, conductor, dielectric |
| `bvh.mojo` | 535 | BVH construction (SAH) and traversal |
| `pipeline.mojo` | 457 | Batch and interactive rendering pipelines |
| `rendering.mojo` | 407 | CPU tile renderer and film accumulation |
| `ply.mojo` | 316 | PLY mesh loader |
| `geometry.mojo` | 191 | Ray, intersection, and path state structs |
| `sampling.mojo` | 173 | Z-Sobol sampler with Owen scrambling |

External C/C++ libraries (OpenImageIO, Ptex, Vulkan) are called via Mojo's
C FFI — not reimplemented.

## The Rendering Equation

At its core, gonzales solves the rendering equation first formulated by
James Kajiya (DOI: 10.1145/15922.15902):

> L_o(p, ω_o) = L_e(p, ω_o) + ∫ f(p, ω_o, ω_i) L_i(p, ω_i) |cos θ_i| dω_i

Each chapter of this book walks through a piece of the machinery needed to
evaluate this integral numerically — from the geometric primitives that define
surfaces, through the material models that describe how light scatters, to the
Monte Carlo estimator that ties everything together.

## What This Book Covers

1. **Geometry** — the vector, ray, and bounding box types
2. **Spectra** — how color is represented and manipulated
3. **Shapes and Acceleration** — scene intersection via BVH
4. **Sampling** — low-discrepancy sequences for variance reduction
5. **Reflection Models** — the BSDF framework
6. **Lights and Materials** — light sources and surface descriptions
7. **Path Tracing** — the integrator: MIS, Russian roulette, volumes
8. **Rendering Pipeline** — GPU wavefront and CPU tile-based rendering
