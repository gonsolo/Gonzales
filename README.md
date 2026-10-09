[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build](https://github.com/gonsolo/gonzales/actions/workflows/main.yml/badge.svg)](https://github.com/gonsolo/gonzales/actions/workflows/main.yml)
[![Test](https://github.com/gonsolo/gonzales/actions/workflows/test.yml/badge.svg)](https://github.com/gonsolo/gonzales/actions/workflows/test.yml)
[![Book](https://github.com/gonsolo/gonzales/actions/workflows/book.yaml/badge.svg)](https://github.com/gonsolo/gonzales/actions/workflows/book.yaml)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/4672/badge?v=1)](https://www.bestpractices.dev/projects/4672)

# Gonzales — Physically Based Renderer

A production-capable spectral Monte Carlo renderer written in **Mojo**, designed
for high-end light transport simulation. Gonzales renders complex scenes —
including Disney's Moana Island, all 32 Bitterli benchmark scenes and a 65-scene
pbrt/Bitterli corpus — with path tracing, SPPM, BDPT and VCM on the CPU and on
NVIDIA GPUs, plus an à-trous wavelet denoiser.

📖 Read the [Gonzales Book](https://gonsolo.github.io/gonzales/) for detailed
documentation with annotated source code.

![Moana Island rendered by Gonzales](Images/moana.png)

## Architecture

The renderer is written entirely in **Mojo** (~54,600 lines in `src/gonzales/`, plus
~12,700 lines of unit tests) and organized into focused groups of modules:

| Group | Lines | Responsibility |
|-------|------:|---------------|
| Shading and BxDFs | 12,700 | Diffuse, coated (pbrt's LayeredBxDF), conductor, dielectric, hair, measured, BSSRDF; spectral upsampling and sensors |
| Path tracer | 9,900 | CPU per-tile wavefront and CUDA wavefront kernels, Z-Sobol sampler, film and outputs |
| Scene parsing | 9,400 | pbrt-v4 and Mitsuba parsers, PLY loader (threaded prefetch), materials, lights |
| BDPT / VCM | 6,900 | Bidirectional path tracing and vertex connection and merging (CPU and GPU) |
| Acceleration | 4,600 | SAH BVH2 (GPU) and BVH4 (CPU), instancing, native curve primitives |
| SPPM | 3,700 | Stochastic progressive photon mapping (CPU and GPU) |
| ReSTIR and guiding | 1,900 | Reservoir resampling (DI, GI, volumes, SMS) and path guiding |
| MNEE / SMS | 1,800 | Manifold next-event estimation and specular manifold sampling for glass caustics |
| Media | 1,800 | Homogeneous, uniform-grid, NanoVDB and cloud media |
| Denoising | 650 | GPU à-trous and CPU joint bilateral denoisers |
| Vulkan RT, RT cores, viewer | 460 | Second GPU backend, `--rt-hardware`, interactive viewer |

External C/C++ libraries (OpenImageIO, Ptex, Vulkan) are called via Mojo's
C FFI — not reimplemented.

## Key Features

- **Spectral transport** — Four hero wavelengths per path in every integrator, pbrt-style RGB-to-spectrum upsampling and sensor models
- **Wavefront GPU path tracing** — CUDA kernels in Mojo with live-path compaction, compile-time kernel variants, deferred software shadow rays and an optional RT-core backend (`--rt-hardware`)
- **Fast CPU renderer** — Per-tile wavefront, 4-wide BVH with triangle records, vectorised shading
- **Four light-transport algorithms** — Path tracing, SPPM (`--sppm`), BDPT/VCM (`--vcm`) and ReSTIR variants, sharing materials, lights and MIS code
- **Specular caustics** — MNEE and SMS for glass and mirror caustics, with path-identity MIS so the result matches pbrt on the test cubes
- **Veach-style MIS** — Power heuristic balancing NEE and BSDF sampling; Russian roulette
- **Full material set** — Coated diffuse/conductor, rough and smooth dielectrics, hair, subsurface, measured BRDFs, alpha cutouts, bump and normal maps, Ptex
- **Volumes** — Heterogeneous media, NanoVDB, clouds, chromatic media
- **Scene formats** — PBRT-v4 (including object instancing and curves) and Mitsuba
- **Denoising** — Variance-adaptive à-trous GPU denoiser (Dammertz 2010) with albedo and normal guides
- **Interactive viewer** — Vulkan viewer with progressive refinement

## Testing

`make unittest` runs ~60 unit-test files concurrently; `make smoketest` runs an
integrator × feature matrix, and `make analytictest` checks renders against closed-form
furnace and plane solutions.

## Rendering Moana

| Version | Resolution | SPP | Time | Notes |
|---------|-----------|-----|------|-------|
| v0.0 (2021) | 2048×858 | 64 | 26h | GCE 8 CPU, 64 GB |
| v0.1 (2023) | 1920×800 | 64 | 78 min | Threadripper 1920X, with Embree |
| v0.2 (2026) | — | — | — | ARC cleanup, Embree removed |
| v0.3 (2026) | — | — | — | [Release Notes](Documentation/ReleaseNotes/0.3.md) |

## Performance

Benchmark: [Bitterli bathroom](https://benedikt-bitterli.me/resources/) scene, 1024×1024, 64 spp, no denoiser.
Hardware: AMD Ryzen Threadripper 1920X (12 cores, 24 threads), NVIDIA RTX 3060 12 GB.

| Renderer | Mode | Wall time | Notes |
|---|---|---|---|
| **Gonzales** | GPU | **4.4s** | Wavefront path tracing, software BVH traversal |
| **Gonzales** | GPU (`--rt-hardware`) | 5.3s | RT cores driven from CUDA |
| **pbrt-v4** | GPU (OptiX) | 4.9s | Hardware RT cores |
| **Embree pathtracer** | CPU | 10.3s / 15.0s | Bare geometry / approximate materials, no textures |
| **Gonzales** | CPU | **24.2s** | Full materials and textures |
| **pbrt-v4** | CPU | 50.1s | Full materials and textures |

Gonzales CPU is **2.1× faster** than pbrt CPU, and Gonzales GPU is on par with pbrt GPU.
Embree's tutorial path tracer has no textures and only approximate materials, so its
numbers are not directly comparable.

### Lines of code

| Project | Lines (own code) |
|---|---|
| **Gonzales** | **~54,600** |
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
