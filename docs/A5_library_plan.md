# A5: Library Plan

Plan for splitting `src/gonzales` into layered subpackages with shared transport interfaces, so that new integrators
(UPBP, photon planes, guiding, Metropolis) are thin modules and a feature cannot exist in one integrator only.

**Status (2026-10-31): steps 1 to 6b done on branch `library`, not merged yet.** `src/gonzales/README.md` has the layer order, and `Tests/check_layers.py` enforces it with no exceptions. The interfaces below are in progress.

## Why

Most VCM bugs of 2026-10-30 had one of two shapes. A weight reserved a share for a strategy that never ran (volume splat,
walks past the 9-vertex cache). A feature lived in one integrator (PT honoured `g`, VCM and SPPM hardcoded isotropic).
Both come from strategies and media being implicit in each integrator's control flow.

## Interfaces

Compile-time traits, so kernels stay monomorphized. Compiler problems get fixed in the Mojo sources, not designed around.

**`Medium`** owns free flight and the phase function: `sample(ray) -> FreeFlight` with its real density, `transmittance`,
`sigma_t_at(p)`, `phase(wo, wi)`, `sample_phase(wo, u)`, `kernel_radius(p)`. No integrator touches sigma or `g`.

**`Hop`** is a walk from entry to exit: `begin(hit)`, `walk(...) -> HopExit` with weight, exit and normal, and the boundary
densities `T(cos)` and exit lobe. Shared by PT, VCM, SPPM, and the BSSRDF photon map as a second implementation.

**`StrategySet`** declares what exists at a vertex: `can_merge`, `merge_density`, `has_manifold`, `too_long`, `past_cache`.
Carries take `eta` from it, and the weights are computed from it, so a share without a running strategy becomes impossible.

**`PhotonEstimator`** is what merging needs: `store(vertex)`, `query(cv) -> radiance, eta`, `kernel_measure(v)`. Points on a
surface, points in a medium, fibres, beams, planes and volumes are implementations. Integrators see only this.

**`Integrator`** is a thin driver: one camera step plus its strategy set, against one wavefront kernel skeleton.

## Order

Each step is one or a few commits. Unit tests stay green and renders stay identical, checked against stored images.

Done: layer check; `Ray`, hash, `GpuTexture` and `MisPolicy` moves; `estimators.photon_grid`; `transport.medium` and `sss_walk`; `transport.manifold`; subpackages with absolute imports; `cpu` beside `gpu`; `shading.mojo` split; CAMIS deleted.

1. Tests mirrored to the package layout.
2. `Medium` and `Hop`: `MediumView` and `sss_hop` serve VCM and SPPM. PT still has its own step-by-step walk.
3. `StrategySet`: `VcmStrategies` gathers the rules in one place. It becomes a trait with its second user, UPBP, as a trait with one implementation only decorates.
4. `PhotonEstimator`, VCM and SPPM on it.
5. New integrators, one module each.

## Open questions

- Whether kernel compile time stays acceptable with traits across packages. The `Medium` and `Hop` step measures it on the B580.
- ReSTIR stays for now (user, 2026-10-31): it is woven through 18 modules, so it would be a re-port from history, not a revert.
