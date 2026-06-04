[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build](https://github.com/gonsolo/gonzales/actions/workflows/main.yml/badge.svg)](https://github.com/gonsolo/gonzales/actions/workflows/main.yml)
[![Test](https://github.com/gonsolo/gonzales/actions/workflows/test.yml/badge.svg)](https://github.com/gonsolo/gonzales/actions/workflows/test.yml)
[![Book](https://github.com/gonsolo/gonzales/actions/workflows/book.yaml/badge.svg)](https://github.com/gonsolo/gonzales/actions/workflows/book.yaml)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/4672/badge?v=1)](https://www.bestpractices.dev/projects/4672)

# Gonzales — Physically Based Renderer

A production-capable Monte Carlo path tracer written in **Mojo**, designed
for high-end light transport simulation. Gonzales renders complex scenes —
including Disney's Moana Island and all 32 Bitterli benchmark scenes — with
GPU-accelerated wavefront path tracing and an à-trous wavelet denoiser.

📖 Read the [Gonzales Book](https://gonsolo.github.io/gonzales/) for detailed
documentation with annotated source code.

![Moana Island rendered by Gonzales](Images/moana.png)

## Architecture

The renderer is written entirely in **Mojo** (~7,800 lines) and organized
into focused modules:

| Module | Lines | Responsibility |
|--------|------:|---------------|
| `parsing.mojo` | 1,959 | PBRT-v4 scene parser — geometry, materials, lights, textures |
| `gpu.mojo` | 1,294 | GPU kernels: wavefront path tracing, à-trous denoiser, film |
| `shading.mojo` | 770 | Material shading — diffuse, coated, conductor, dielectric |
| `bvh.mojo` | 535 | BVH construction (SAH) and traversal |
| `pipeline.mojo` | 457 | Batch and interactive rendering pipelines |
| `rendering.mojo` | 407 | CPU tile renderer, film accumulation, bilateral denoiser |
| `ply.mojo` | 316 | PLY mesh loader |
| `sampling.mojo` | 173 | Z-Sobol sampler with Owen scrambling |
| `geometry.mojo` | 191 | Ray, intersection, path state structs |
| `transform.mojo` | — | 4×4 matrix math |
| `rng.mojo` | — | PCG32 random number generator |
| `postprocess.mojo` | — | Joint bilateral denoiser (CPU) |
| `viewer.mojo` | — | Interactive Vulkan viewer bridge |

External C/C++ libraries (OpenImageIO, Ptex, Vulkan) are called via Mojo's
C FFI — not reimplemented.

## Key Features

- **Wavefront GPU path tracing** — Batches 8 samples per bounce loop; NVIDIA GPU via Mojo's GPU API
- **À-trous wavelet denoiser** — Variance-adaptive 5-pass GPU denoiser (Dammertz 2010)
- **Veach-style MIS** — Power heuristic balancing NEE and BSDF sampling
- **Pure Mojo BVH** — SAH construction and traversal, no Embree dependency
- **Z-Sobol sampling** — Low-discrepancy sequences for fast convergence
- **Russian roulette** — Unbiased path termination for efficiency
- **PBRT-v4 format** — Full scene file compatibility
- **Ptex & OpenImageIO** — C FFI interop for professional texture and image handling
- **Interactive viewer** — GPU-accelerated real-time preview with progressive refinement

## Rendering Moana

| Version | Resolution | SPP | Time | Notes |
|---------|-----------|-----|------|-------|
| v0.0 (2021) | 2048×858 | 64 | 26h | GCE 8 CPU, 64 GB |
| v0.1 (2023) | 1920×800 | 64 | 78 min | Threadripper 1920X, with Embree |
| v0.2 (2026) | — | — | — | ARC cleanup, Embree removed |
| v0.3 (2026) | — | — | — | [Release Notes](Documentation/ReleaseNotes/0.3.md) |

## Performance

Benchmark: [Bitterli bathroom](https://benedikt-bitterli.me/resources/) scene, 1024×1024, 64 spp.
Hardware: AMD Ryzen 9 7950X (24 cores), NVIDIA RTX 3060 12 GB.

| Renderer | Mode | Wall time | Notes |
|---|---|---|---|
| **Gonzales** | GPU | **6.9s** | Wavefront path tracing + à-trous denoiser |
| **pbrt-v4** | GPU (OptiX) | 5.4s | Hardware RT cores |
| **Embree pathtracer** | CPU | 11.0s | Hardware AVX2 BVH, no textures/materials |
| **Gonzales** | CPU | 29.9s | Full materials and textures |
| **pbrt-v4** | CPU | 53.2s | Full materials and textures |

Gonzales CPU is **1.8× faster** than pbrt CPU. Gonzales GPU trails pbrt GPU by 1.3× — the
gap is OptiX RT cores, which are inaccessible outside of OptiX. Embree's CPU number is
not directly comparable since the benchmark scene contains no texture lookups or material
evaluation (geometry traversal only).

### Lines of code

| Project | Lines (own code) |
|---|---|
| **Gonzales** | **~7,800** |
| pbrt-v4 | ~84,000 (excluding bundled data tables and third-party libs) |
| Embree kernel | ~96,000 (BVH/traversal only, no rendering) |

## Prerequisites

| Dependency | Description | Install (Arch) |
| --- | --- | --- |
| [Swift 6.3](https://swift.org) | Compiler with C++20 interop | `pacman -S swift` |
| [OpenImageIO](https://github.com/AcademySoftwareFoundation/OpenImageIO) | EXR/HDR image I/O | `pacman -S openimageio` |
| [Ptex](https://github.com/wdas/ptex) | Per-face texture mapping (Disney) | `pacman -S ptex` |
| [Loupe](https://gitlab.gnome.org/GNOME/loupe) | EXR image viewer (for `make view_release`) | `pacman -S loupe` |

## Installation

### Arch Linux (AUR)

```bash
yay gonzales-git
```

<https://aur.archlinux.org/packages/gonzales-git>

### Building from Source

```bash
make debug    # debug build
make release  # optimized release build
```

> **Note (Swift 6.1.2+ on Arch Linux):** An [incompatibility with GCC 15](https://github.com/swiftlang/swift/issues/81774) requires patching `/usr/lib/swift/lib/swift/_FoundationCShims/_CStdlib.h` line 55 to wrap the `#if __has_include(<math.h>)` block in `#if 0 ... #endif`.

## Getting Started

1. Download scenes from [Bitterli](https://benedikt-bitterli.me/resources) (PBRT-v4 format) or [pbrt-v4-scenes](https://github.com/mmp/pbrt-v4-scenes)
2. Quick test — render and view a Cornell Box:
   ```bash
   make view_release
   ```
3. Or render any scene directly: `.build/release/gonzales path/to/scene.pbrt`

## Acknowledgments

[Physically Based Rendering: From Theory to Implementation](https://www.pbr-book.org/) has been an inspiration since the project was called *lrt*.

© Andreas Wendleder 2019–2026
