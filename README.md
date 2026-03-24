[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build](https://github.com/gonsolo/gonzales/actions/workflows/main.yml/badge.svg)](https://github.com/gonsolo/gonzales/actions/workflows/main.yml)
[![Test](https://github.com/gonsolo/gonzales/actions/workflows/test.yml/badge.svg)](https://github.com/gonsolo/gonzales/actions/workflows/test.yml)
[![Book](https://github.com/gonsolo/gonzales/actions/workflows/book.yaml/badge.svg)](https://github.com/gonsolo/gonzales/actions/workflows/book.yaml)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/4672/badge?v=1)](https://www.bestpractices.dev/projects/4672)

# Gonzales — Physically Based Renderer

A production-capable Monte Carlo path tracer written in **Swift 6.3**, designed
for high-end light transport simulation. Gonzales renders complex scenes —
including Disney's Moana Island and all 32 Bitterli benchmark scenes — using
modern Swift features: strict concurrency, structured task groups, and SIMD
vector types.

📖 Read the [Gonzales Book](https://gonsolo.github.io/gonzales/) for detailed
documentation with annotated source code.

![Moana Island rendered by Gonzales](Images/moana.png)

## Architecture

The renderer is organized into 18 modules mirroring the architecture of
[PBRT](https://www.pbr-book.org/):

| Module | Responsibility |
|--------|---------------|
| **Integrator** | Volume path tracing with multiple importance sampling |
| **Accelerators** | BVH with Surface Area Heuristic |
| **Sampler** | Z-Sobol quasi-random sequences with Owen scrambling |
| **Bsdf** | Diffuse, dielectric, microfacet, coated, layered, hair, mix |
| **Reflection** | Fresnel equations, Trowbridge-Reitz microfacet distribution |
| **Light** | Area, point, distant, and infinite (environment map) lights |
| **Core** | Ray, spectrum, film, scene, distributions, tile renderer |
| **Geometry** | Vectors, points, normals, transforms, bounding boxes (SIMD4) |
| **Shape** | Spheres, triangle meshes, PLY, curves |
| **Texture** | Image textures, Ptex via C++ interop |
| **Camera** | Perspective camera with depth of field |
| **Image** | EXR output via OpenImageIO C++ interop |

## Key Features

- **Veach-style MIS** — Power heuristic balancing light and BSDF sampling
- **Pure Swift BVH** — Native acceleration with SAH, no Embree dependency
- **Z-Sobol sampling** — Low-discrepancy sequences for fast convergence
- **Russian roulette** — Unbiased path termination for efficiency
- **PBRT-v4 format** — Full scene file compatibility
- **Structured concurrency** — Tile-based parallel rendering via `withThrowingTaskGroup`
- **Ptex & OpenImageIO** — C++ interop for professional texture and image handling

## Rendering Moana

| Version | Resolution | SPP | Time | Notes |
|---------|-----------|-----|------|-------|
| v0.0 (2021) | 2048×858 | 64 | 26h | GCE 8 CPU, 64 GB |
| v0.1 (2023) | 1920×800 | 64 | 78 min | Threadripper 1920X, with Embree |
| v0.2 (2026) | — | — | — | ARC cleanup, Embree removed |
| v0.3 (2026) | — | — | — | [Release Notes](Documentation/ReleaseNotes/0.3.md) |

## Installation

### Arch Linux (AUR)

```
yay gonzales-git
```

https://aur.archlinux.org/packages/gonzales-git

### Building from Source

```
swift build -c release
```

> **Note (Swift 6.1.2+ on Arch Linux):** An [incompatibility with GCC 15](https://github.com/swiftlang/swift/issues/81774) requires patching `/usr/lib/swift/lib/swift/_FoundationCShims/_CStdlib.h` line 55 to wrap the `#if __has_include(<math.h>)` block in `#if 0 ... #endif`.

## Getting Started

1. Download scenes from [Bitterli](https://benedikt-bitterli.me/resources) (PBRT-v4 format) or [pbrt-v4-scenes](https://github.com/mmp/pbrt-v4-scenes)
2. Render: `.build/release/gonzales path/to/scene.pbrt`
3. View the output EXR with any HDR image viewer

## Acknowledgments

[Physically Based Rendering: From Theory to Implementation](https://www.pbr-book.org/) has been an inspiration since the project was called *lrt*.

© Andreas Wendleder 2019–2026
