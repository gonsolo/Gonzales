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

The renderer is written entirely in **Mojo** (~10,700 lines) and organized
into focused modules:

| Module | Lines | Responsibility |
|--------|------:|---------------|
| `parsing.mojo` | 3,254 | PBRT-v4 scene parser — geometry, materials, lights, textures |
| `shading.mojo` | 2,225 | Material shading — diffuse, coated, conductor, dielectric |
| `gpu.mojo` | 1,553 | GPU kernels: wavefront path tracing, à-trous denoiser, film |
| `pipeline.mojo` | 697 | Batch and interactive rendering pipelines |
| `bvh.mojo` | 569 | BVH construction (SAH) and traversal |
| `geometry.mojo` | 566 | Ray, intersection, path state structs |
| `rendering.mojo` | 443 | CPU tile renderer, film accumulation, bilateral denoiser |
| `sampling.mojo` | 362 | Z-Sobol sampler with Owen scrambling |
| `ply.mojo` | 346 | PLY mesh loader |
| `scene.mojo` | 177 | Scene helpers |
| `transform.mojo` | 140 | 4×4 matrix math |
| `__init__.mojo` | 124 | CLI entry point |
| `postprocess.mojo` | 114 | Joint bilateral denoiser (CPU) |
| `viewer.mojo` | 75 | Interactive Vulkan viewer bridge |
| `rng.mojo` | 22 | PCG32 random number generator |

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
| **Gonzales** | **~10,700** |
| pbrt-v4 | ~84,000 (excluding bundled data tables and third-party libs) |
| Embree kernel | ~96,000 (BVH/traversal only, no rendering) |

## Prerequisites

| Dependency | Description | Install (Arch) |
| --- | --- | --- |
| [Mojo](https://www.modular.com/mojo) (via `uv`) | Compiler; GPU target requires `--target-accelerator sm_XX` | `uv run mojo` |
| [OpenImageIO](https://github.com/AcademySoftwareFoundation/OpenImageIO) | EXR/HDR image I/O | `pacman -S openimageio` |
| [Ptex](https://github.com/wdas/ptex) | Per-face texture mapping (Disney) | `pacman -S ptex` |
| [Vulkan](https://vulkan.lunarg.com/) | Interactive viewer | `pacman -S vulkan-icd-loader` |
| [Loupe](https://gitlab.gnome.org/GNOME/loupe) | EXR image viewer (for `make view_release`) | `pacman -S loupe` |

For GPU rendering, set `--target-accelerator` to match your GPU's compute
capability (e.g. `sm_86` for RTX 3060, `sm_89` for RTX 4090).

## Installation

### Building from Source

```bash
make debug    # debug build
make release  # optimized release build
```

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
