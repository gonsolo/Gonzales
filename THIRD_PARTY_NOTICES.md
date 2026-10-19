# Third-party material

Gonzales' own code is under the licence in `LICENSE`. The following material comes from other projects.

* **pbrt-v4** (Copyright 1998-2020 Matt Pharr, Wenzel Jakob and Greg Humphreys; Apache License 2.0, see
  `third_party/LICENSE-pbrt-v4-Apache-2.0.txt`). These modules are ports of pbrt-v4 code: `layered.mojo` (layered BxDF),
  `measured_bsdf.mojo`, `sensors.mojo` (pixel sensors), `rgb2spec.mojo` (the RGB-to-spectrum optimiser of
  `rgb2spec_opt.cpp` and the sigmoid-polynomial lookup) and `spd.mojo` (spectrum files). Other modules follow pbrt-v4's
  algorithms (visible-wavelength sampling, light sampling, film clamping, ray offsets, pixel filters) without being
  line-by-line ports. The Apache-2.0 terms apply to the ported parts in addition to the licence of this project.
* **RGB-to-spectrum table** `src/gonzales/data/rgb2spectrum_table.bin`: generated with `rgb2spec.mojo` (resolution 64,
  sRGB) following Jakob and Hanika, "A Low-Dimensional Function Space for Efficient Spectral Upsampling" (2019); the
  generator is a port of pbrt-v4's, see above.
* **Sobol direction numbers** `src/gonzales/data/new-joe-kuo-6.21201` and `Sources/SobolGenerator/`: Copyright (c) 2008,
  Frances Y. Kuo and Stephen Joe, BSD-style licence, see `third_party/LICENSE-Joe-Kuo.txt`.
* **Mitsuba 3** (BSD 3-Clause) and **Cycles** (Apache 2.0) were used as references for algorithms and for the Mitsuba
  scene file format that `mitsuba_parser.mojo` reads; their source code is not included.
* **Libraries linked at build time and not included:** OpenImageIO, Ptex, NanoVDB (file reader), Vulkan and the NVIDIA
  CUDA driver interface, each under its own licence.
* **The Moana Island Scene** is Disney's data set. It is not included and is used under its own licence (research and
  software development only; it may only be called the "Moana Island Scene").
