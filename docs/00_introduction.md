# Introduction

Gonzales is a physically based renderer written in Swift. It implements
a volume path tracer capable of rendering production-quality scenes — including
Disney's Moana Island — using modern Swift 6.2 features like strict concurrency,
structured task groups, and SIMD vector types.

## Why Swift?

Most renderers are written in C++. Swift offers an alternative with value
semantics by default, memory safety without garbage collection, and first-class
concurrency. Gonzales explores how far these advantages carry in a
performance-critical domain where every nanosecond in the inner loop matters.

## Architecture

The renderer is organized into 18 focused modules that mirror the established
architecture of PBRT (Physically Based Rendering: From Theory to Implementation):

| Module | Responsibility |
|--------|---------------|
| **Geometry** | Vectors, points, normals, transforms, bounding boxes |
| **Core** | Ray, spectrum, film, scene, distributions, rendering loop |
| **Accelerators** | BVH construction and traversal |
| **Sampler** | Z-Sobol quasi-random sequences |
| **Bsdf** | Diffuse, dielectric, hair, coated, layered, mix |
| **Reflection** | Fresnel equations, microfacet distributions, BSDF framework |
| **Light** | Area, point, distant, and infinite (environment) lights |
| **Material** | Material dispatch and parameter evaluation |
| **Texture** | Image textures, Ptex support via C++ interop |
| **Shape** | Spheres, triangles, PLY meshes, curves |
| **Camera** | Perspective camera with depth of field |
| **Filter** | Pixel reconstruction filters |
| **Integrator** | Volume path tracing with MIS |
| **Image** | EXR output via OpenImageIO C++ interop |

The codebase is split into a library (`libgonzales`) and an executable
(`gonzales`), enabling both standalone rendering and unit testing.

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
8. **Rendering Pipeline** — tile-based concurrency and image output
