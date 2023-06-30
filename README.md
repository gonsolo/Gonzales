[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://github.com/gonsolo/gonzales/actions/workflows/main.yml/badge.svg)](https://github.com/gonsolo/gonzales/actions/workflows/main.yml)

# Gonzales Renderer

© Andreas Wendleder 2019-2023

Brute-force path tracing written in Swift.

### Moana

~~2048x858, 64spp, 26h on a Google Compute Engine with 8 CPUs and 64 GB of memory.
Memory usage was around 50GB.~~ This was 2021.

With version 0.1.0 rendering Moana takes 78 minutes (1920x800, 64spp, AMD Threadripper 1920x 12 cores 24 threads, 64GB RAM, 80GB swap)

![Moana](Images/moana.png)

### Build on Ubuntu

- Install swift from https://www.swiftlang.xyz.
- Install libopenimageio-dev and libembree-dev.
- Install ptex from https://mentors.debian.net/package/ptex.
- Type make

### Try it out

- Get scenes from https://benedikt-bitterli.me/resources in pbrt v4 format and try them out.
- Get scenes from https://github.com/mmp/pbrt-v4-scenes and try them. All of them should be parsed correctly, not all of them work because an algorithm
  (subsurface scattering, volume rendering, ...) is missing.

### Acknowledgments

[PBRT](https://www.pbr-book.org/) was an inspiration since it was called lrt.
