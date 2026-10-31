# A5: Library Plan

Plan for splitting `src/gonzales` into layered subpackages with shared transport interfaces, so that new integrators
(UPBP, photon planes, guiding, Metropolis) are thin modules and a feature cannot exist in one integrator only.

**Status (2026-10-31): draft for review, nothing moved.** Counts come from a scan of `from .x import` lines. Interfaces
below are proposals and have not been compiled for the GPU.

## Why

Most VCM bugs of 2026-10-30 had one of two shapes. A weight reserved a share for a strategy that never ran (volume splat,
walks past the 9-vertex cache). A feature lived in one integrator (PT honoured `g`, VCM and SPPM hardcoded isotropic).
Both come from strategies and media being implicit in each integrator's control flow.

## Layers

Each layer imports only from layers above it. 97 modules, 61,774 lines.

| Layer | Holds | Lines today |
|---|---|---|
| `core` | geometry, spectra, sampling, random numbers | 4,672 |
| `scene` | primitives, materials, media, lights, BVH, parsers | 22,251 |
| `transport` | lobes, phase functions, light sampling, walks, MNEE, SMS | 4,627 |
| `paths` | subpaths, vertices, connections, MIS | 2,885 |
| `estimators` | photon stores and density estimates | 663 |
| `integrators` | PT, VCM, SPPM, ReSTIR, then the new ones | 12,684 |
| `gpu` | kernels and drivers | 9,527 |
| `app` | pipeline, command line, viewer | 4,465 |

`scene` is large enough to need sublayers: `scene.geometry`, `scene.accel`, `scene.materials`, `scene.parse`.

## What is in the way

22 of 590 imports point the wrong way, plus one cycle.

| Imports | Symbols | Move |
|---|---|---|
| `spectrum` <-> `sampling` | `mix_bits_u64`; `sample_wavelengths` | hash to `core.rng`; breaks the only cycle |
| `sampling` -> `primitives` | `Ray` | `Ray` to `core` |
| `bvh`, `host_textures`, `pbrt_parser`, `scene_cache`, `sms` -> `render_state` | `GpuTexture`, `NormalSlopeMap` | those two types to `scene.materials` |
| `bxdf` -> `vcm_mis` | `MisPolicy`, `nee_mis_weight` | policy to `transport`; the recursion stays in `paths` |
| `vcm_grid`, `vcm_merge` -> `sppm` | grid, `gather_disk_*` | new `estimators.grid` |
| `bdpt_nee`, `bdpt_eval` -> `sppm` | `_geom_normal`, `area_light_side_pdf` | already elsewhere or `transport` |
| 7 modules -> `gpu_media` | `medium_after`, `sss_*`, `_sample_medium_core` | new `transport.medium` and `transport.walk` |
| `bdpt_connect` -> `vcm_grid` | `_vcm_too_long`, `_vcm_eta_*` | strategy set, below |
| `bdpt_eval`, `surface` -> `shading` | `_mnee_*`, `_shade_context`, texture lookup | `transport.manifold`, `transport.context` |

## Interfaces

Compile-time traits, so kernels stay monomorphized. Compiler problems get fixed in the Mojo sources, not designed around.

**`Medium`** owns free flight and the phase function: `sample(ray) -> FreeFlight` with its real density, `transmittance`,
`sigma_t_at(p)`, `phase(wo, wi)`, `sample_phase(wo, u)`, `kernel_radius(p)`. No integrator touches sigma or `g`.

**`Hop`** is a walk from entry to exit: `begin(hit)`, `walk(...) -> HopExit` with weight, exit and normal, and the boundary
densities `T(cos)` and exit lobe. Shared by PT, VCM, SPPM, and the BSSRDF photon map as a second implementation.

**`StrategySet`** declares what exists at a vertex: `can_merge(v)`, `can_connect(v, lv)`, `can_splat(lv)`,
`has_manifold(v)`, `past_cache(n)`. Carries take `eta` from it, and the weights are computed from it, so a share without a
running strategy becomes impossible. This replaces `_vcm_eta_at`, `_vcm_too_long`, `_vcm_past_cache` and `mnee_rival`.

**`PhotonEstimator`** is what merging needs: `store(vertex)`, `query(cv) -> radiance, eta`, `kernel_measure(v)`. Points on a
surface, points in a medium, fibres, beams, planes and volumes are implementations. Integrators see only this.

**`Integrator`** is a thin driver: one camera step plus its strategy set, against one wavefront kernel skeleton.

## Order

Each step is one or a few commits. Unit tests stay green and renders stay identical, checked against stored images.

1. Layer check script, reporting today's violations; part of `make unittest`, allowed list shrinks to empty.
2. Break the cycle and the small moves: `Ray`, hash, `GpuTexture`, `MisPolicy`.
3. `estimators.grid`, out of `sppm`.
4. `transport.medium` and `transport.walk`, out of `gpu_media`.
5. `transport.manifold` and `context`, out of `shading`.
6. Subpackages: move files, absolute imports, tests mirrored.
7. `Medium` and `Hop` traits, PT, VCM and SPPM on them.
8. `StrategySet`, VCM on it. The riskiest step: weights must not change except where a test shows a bug.
9. `PhotonEstimator`, VCM and SPPM on it.
10. Split `shading.mojo` (5,643 lines) by material and integrator.
11. New integrators, one module each.

## Open questions

- Whether kernel compile time stays acceptable with traits across packages. Step 7 measures it on the B580.
- How much of the ReSTIR code stays; it is large and little used.
- CAMIS is compiled out by default and threads through every weight. Keep it behind `StrategySet` or drop it.
