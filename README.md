[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://github.com/gonsolo/gonzales/actions/workflows/main.yml/badge.svg)](https://github.com/gonsolo/gonzales/actions/workflows/main.yml)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/4672/badge?v=1)](https://www.bestpractices.dev/projects/4672)

# Gonzales Renderer

© Andreas Wendleder 2019-2026

Let Gemini praise the source:

# Gonzales: A High-Performance, Pure Swift Monte Carlo Path Tracer

Gonzales is a physically-based renderer built from the ground up in **Swift 6.2**, designed for high-end light transport simulation and research. By leveraging advanced sampling heuristics and a native acceleration architecture, it achieves production-grade results in complex scenes like the *Disney Moana Island* and *Bitterli* benchmarks — all without external C++ acceleration dependencies.

## 🚀 Key Technical Features

### ⚖️ Veach-Style Multiple Importance Sampling (MIS)
Implements robust MIS using the **Power Heuristic** to balance light source sampling (Next Event Estimation) and BRDF sampling. This significantly reduces variance and eliminates "fireflies" in challenging lighting conditions, such as small, intense emitters or highly glossy surfaces.

### ⚡ Pure Swift Acceleration
Featuring a native **Bounding Volume Hierarchy (BVH)** optimized specifically for Swift's performance characteristics. Gonzales provides efficient ray-primitive intersection using a custom implementation of the **Surface Area Heuristic (SAH)**, removing the need for external libraries like Intel Embree.

### 🎲 Advanced Sampling Engine
* **Z-Sobol Quasirandom Sampling:** Utilizes high-dimensional Sobol sequences for superior convergence rates and improved blue-noise characteristics compared to standard pseudorandom methods.
* **Xoshiro256\*\***: Features a custom implementation of the Xoshiro256** generator for ultra-fast, high-quality local randomness.

### 🌌 Complex Light Transport
* **Volume Path Integration:** Full support for participating media, including heterogeneous volumes and volumetric scattering.
* **Russian Roulette:** Optimized path termination heuristics to maintain computational efficiency in deep-bounce indirect lighting simulations.
* **Power-Based Light Sampling:** Efficiently handles scenes with high light counts by sampling based on cumulative power distributions.

### 🛠️ Production-Ready Integration
* **PBRT-v4 Support:** Native parsing and rendering of the industry-standard PBRT-v4 scene format.
* **Ptex & OpenImageIO:** Deep integration with Ptex textures and OIIO for professional-grade image handling and texturing.

### Moana

v0.0, 2021: 2048x858, 64spp, 26h on a Google Compute Engine with 8 CPUs and 64 GB of memory. Memory usage was around 50GB.

v0.1, 2023: Rendering Moana with a little help from Embree takes 78 minutes (1920x800, 64spp, AMD Threadripper 1920x 12 cores 24 threads, 64GB RAM, 80GB swap)

v0.2, 2026: Cleanup, tuning to get rid of ARC traffic, removal of Embree.

![Moana](Images/moana.png)

### Get it from Arch AUR

```yay gonzales-git```

https://aur.archlinux.org/packages/gonzales-git

### Try it out

- Get scenes from https://benedikt-bitterli.me/resources in pbrt v4 format and try them out.
- Get scenes from https://github.com/mmp/pbrt-v4-scenes and try them. All of them should be parsed correctly, not all of them work because an algorithm
  (subsurface scattering, volume rendering, ...) is missing.

### Acknowledgments

[PBRT](https://www.pbr-book.org/) was an inspiration since it was called lrt.

### Swift patches for building manually

For Swift 6.1.2 and later the following has to be done on Arch Linux because of an incompatibility with GCC 15:

```
+ #if 0
#if __has_include(<math.h>)
#endif
+ #endif
```
in  /usr/lib/swift/lib/swift/_FoundationCShims/_CStdlib.h +55.
