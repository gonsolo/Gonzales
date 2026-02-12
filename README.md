[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://github.com/gonsolo/gonzales/actions/workflows/main.yml/badge.svg)](https://github.com/gonsolo/gonzales/actions/workflows/main.yml)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/4672/badge?v=1)](https://www.bestpractices.dev/projects/4672)

# Gonzales Renderer

© Andreas Wendleder 2019-2026

Brute-force path tracing written in Swift.

### Moana

~~2048x858, 64spp, 26h on a Google Compute Engine with 8 CPUs and 64 GB of memory.
Memory usage was around 50GB.~~ This was 2021.

With version 0.1.0 rendering Moana takes 78 minutes (1920x800, 64spp, AMD Threadripper 1920x 12 cores 24 threads, 64GB RAM, 80GB swap)

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

For Swift 6.1.2 and later the following has to be done on Arch Linux because of incompatibility with GCC 15:

```
+ #if 0
#if __has_include(<math.h>)
#endif
+ #endif
```
in  /usr/lib/swift/lib/swift/_FoundationCShims/_CStdlib.h +55.
