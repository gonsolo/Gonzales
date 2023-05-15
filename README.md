[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://github.com/gonsolo/gonzales/actions/workflows/main.yml/badge.svg)](https://github.com/gonsolo/gonzales/actions/workflows/main.yml)

# Gonzales Renderer

© Andreas Wendleder 2019-2023

Brute-force path tracing written in Swift.

### Moana

~~2048x858, 64spp, 26h on a Google Compute Engine with 8 CPUs and 64 GB of memory.
Memory usage was around 50GB.~~

With version 0.1.0 rendering Moana takes 78 minutes (1920x800, 64spp, AMD Threadripper 1920x 12 cores 24 threads, 64GB RAM, 80GB swap)

![Moana](Images/moana.png)

### Build

Type `make` and wait for error messages. :)

### Acknowledgments

[PBRT](https://www.pbr-book.org/) was an inspiration since it was called lrt.
