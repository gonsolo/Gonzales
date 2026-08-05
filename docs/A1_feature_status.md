# Appendix 2: Feature Status & Remaining Gaps

> **Last updated: 2026-08-05**

This document records which renderer features are implemented and which are still
missing, in priority order. It is kept as a living document — check commits for
the implementing commit hash. Detailed investigation history and measurements for
most of these live in the Claude session memory (`project_*.md` files), not here;
this doc is the quick-scan summary.

---

## Implemented Features

### Light Transport
| Feature | Commit | Notes |
|---|---|---|
| Unidirectional path tracing | — | CPU multithreaded + GPU wavefront + GPU interactive (1 spp/dispatch) |
| NEE + MIS — area lights | `b237f91` | Power heuristic, solid-angle pdf |
| NEE — distant lights | `07befc5` | Delta light, MIS weight = 1 |
| NEE — point lights | `07befc5` | Delta light, MIS weight = 1 |
| Env-map (infinite) light — BSDF path | `07befc5` | HDRI lookup on path miss |
| Env-map MIS — importance sampling | `aec8cc4` | 2D marginal/conditional CDF at load time; NEE + correct BSDF-side pdf |
| Path guiding (SD-tree) | — | Adaptive kd-tree/quadtree, replaces earlier fixed grid |
| ReSTIR DI | `0966a85`+ | RIS + temporal + spatial reservoir reuse; primary bounce, area lights, diffuse material only; CPU interactive/batch + GPU interactive; GPU batch renders 1 spp/dispatch for full reuse (see `f529ca7`) |
| ReSTIR GI | `f2294eab`+ | One-bounce reconnection path reuse (diffuse x1 and x2 only), real temporal+spatial reservoir combine, interactive mode; both DI and GI's temporal reservoirs had a real energy-explosion bug (`reservoir_combine` feedback), fixed via `*_MAX_FINALIZED_WEIGHT` clamps |
| MNEE (manifold caustics) | `46789d4`+ | Glass/coateddiffuse-base caustics; also wired into ReSTIR's resolve step |
| SMS (specular manifold sampling) | — | Generalizes MNEE's 1-/2-vertex manifold walk to N specular vertices (`sms.mojo`), block-tridiagonal Newton solve, random seeding + Bernoulli-trial reciprocal estimator for chains where the manifold solution isn't unique; 1-/2-vertex cases stay on MNEE's original fast path unchanged |
| SMS-ReSTIR (`--sms-restir`) | `2f3455e4`+ | Temporal-only reservoir reuse for glass-caustic MNEE probing (manifold shift + bijectivity check, `restir_sms.mojo`); interactive mode only, independent of `--restir`; validated stable at 256 frames (no energy explosion); no spatial reuse yet |
| VCM (connect + merge) | — | `bdpt.mojo`; real Georgiev MIS, verified vs SmallVCM; CPU/GPU/wavefront/Vulkan-RT-assisted |
| SPPM | — | Progressive photon mapping, CPU+GPU, water-caustic scenes |
| Volumetric media (uniformgrid) | — | Delta/ratio tracking heterogeneous media |
| Spectral rendering | — | Full rollout: `SpectralSample`-based NEE/BSDF/CIE conversion path |
| Measured BRDF | — | Tabulated pbrt-v4 `.bsdf` format, CPU+GPU, all 3 integrators |

### Materials
| Feature | Commit | Notes |
|---|---|---|
| Diffuse (Lambertian) | — | Type 1 |
| Emissive | — | Type 2 |
| Conductor — GGX VNDF | `4cad301` | Types 3; Schlick Fresnel |
| Conductor — true anisotropic GGX | `aec8cc4` | Separate α_x/α_y; UV-gradient tangent frame |
| Dielectric | — | Type 4; Fresnel + Snell refraction |
| Thin dielectric | `aec8cc4` | Type 9; Fresnel reflect or pass-through (no refraction) |
| CoatedDiffuse | `555b35d` | Type 5; GGX coat over Lambertian; full stochastic recycling walk |
| DiffuseTransmission | `555b35d` | Type 6; Lambertian transmission |
| CoatedConductor | `619c28c` | Type 7; Schlick coat over GGX conductor |
| Mix material | `619c28c` | Type 8; stochastic sub-material selection |
| Normal mapping | `aec8cc4` | Tangent-space normal texture; UV-gradient frame; CPU only |
| UV-mapped textures + mip-mapping | — | OIIO texture system; GPU mip pyramid + LOD footprint (see `bxdf.mojo`'s `pixel_uv`) |
| Hair/fiber BSDF | `7cf6ed8` | Native curve BVH primitive (not tessellated); Marschner 3-lobe (R/TT/TRT); real per-vertex MIS in VCM |
| "measured" material type | — | Approximated as rough conductor from the `.bsdf` tensor's luminance — explicitly not the real algorithm |
| "subsurface" material type | — | Approximated as coateddiffuse (Skin1/Skin2 albedo) — explicitly NOT real random-walk SSS, see gap below |

### Infrastructure
| Feature | Commit | Notes |
|---|---|---|
| BVH2 (SAH) traversal | `step2` | CPU + GPU, 32-byte nodes |
| Sobol QMC sampler | — | Morton-curve tile ordering |
| À-trous denoiser (unified) | `397efb3` | CPU + GPU now share ONE algorithm (multi-pass à-trous, firefly pre-clamp); GPU previously had a separate, buggier design — see `4ab0aed` |
| Interactive viewer | `step7` | GLFW / Vulkan |
| PBRT v4 scene parser | — | `Include`/`Import` (incl. `.pbrt.gz`), most shapes (`trianglemesh`/`plymesh`/`curve`/`sphere`/`disk`/`loopsubdiv`) — see gaps below for what's still missing |
| GPU wavefront rendering | `step5-6` | Mojo GPU kernels, WAVEFRONT_BATCH=8 concurrent samples/pixel/dispatch |
| Object instancing (`ObjectInstance`) | `ab7a56f`, `11ddf08` | Two-level BVH (BLAS+TLAS), CPU + GPU software traversal |
| Vulkan RT — 2nd GPU intersection backend | `a9eb370`+ | `VK_KHR_ray_query`; primary rays + diffuse-branch shadow rays, scene-adaptive on/off by light-path occupancy. Full scene coverage (2026-08-03, plain `--gpu --vulkan-rt-shade` batch path): meshes, object instancing (multi-geometry-per-template BLAS + real per-instance transforms), spheres (reuses the CUDA path's own analytic `test_spheres` pass unchanged), and curves all supported now. 2026-08-04: the extreme-density `vkAllocateMemory` OOM (curly-hair/bunny-fur, 1M+ curve leaves) is FIXED — the curve BLAS now builds in 250K-leaf chunks instead of one monolithic build. Fixing that surfaced a stale-`tHit` black-render bug (fixed) and, after that, a real correctness ceiling: curve candidates were deferred into a finite buffer of some kind (first a fixed-per-ray cap, then a capacity-bounded shared pool), and no buffer size closed the gap on the densest scenes without either dropping candidates (bald patches) or running out of VRAM. **Final fix**: `intersect_batch.comp` now does the real ray-vs-curve narrow-phase test itself (a GLSL port of `intersect_curve`) the moment it sees a candidate, and calls `rayQueryGenerateIntersectionEXT` to commit a valid hit inline — the way procedural geometry is meant to work with `VK_KHR_ray_query`. This needs no candidate buffer of any kind, so there is no capacity ceiling at all: curly-hair now matches the CUDA baseline to within ~0.1% mean radiance (previously ~5.2x too dark at the tightest safe buffer size), runs faster than the tuned buffer-based approach ever did (~65s vs ~148s at 512x512/32spp), and has zero known density limit. The separate VCM Vulkan RT path (bdpt.mojo) still excludes all of instancing/spheres/curves, not yet extended (see gap below). |

---

## Remaining Gaps — Priority Order

### Quick Wins (Low Effort)

| # | Feature | Why it matters |
|---|---|---|
| 1 | **PBRT parser completeness** | Alpha masking, `Texture "mix"`, `Camera "environment"`, portal/projection/goniometric lights still unhandled; blocks loading some scenes |
| 2 | **Multi-channel EXR AOVs** | Only beauty + albedo export today; normal/depth/object-ID passes exist internally (denoiser G-buffer) but aren't written out as separate AOV files |
| 3 | **PathState compression** | FP16 throughput/albedo; ~2× GPU memory reduction on 4K frames — not started |

### Medium Effort

| # | Feature | Why it matters |
|---|---|---|
| 4 | **Random-walk SSS** | Skin, marble, wax, candles — current "subsurface" is a coateddiffuse approximation, not real volumetric random-walk transport |
| 5 | **ReSTIR beyond one-bounce direct+indirect lighting** | ReSTIR DI (bounce 0), ReSTIR GI (one-bounce reconnection), and SMS-ReSTIR (temporal-only glass-caustic reuse) are all done. Phases 7-8 of `docs/A2_restir_migration_plan.md` (volumetric ReSTIR, BDPT-ReSTIR) would extend reach further — not started. SMS-ReSTIR's own spatial reuse also remains. Phase 9 (unifying ReSTIR's reservoir formalism with VCM's connect/merge into one joint structure) is explicitly marked **open research** in the plan doc itself, not a scoped deliverable |
| 6 | **VCM's Vulkan RT wiring lacks instancing/spheres/curves** | Its primary/bounce interop (`vulkaninterop_rt_traverse_light_paths_gpu`/`_camera_gpu`, bdpt.mojo) is separate from the plain wavefront path's, and none of the 2026-08-03/04 Vulkan RT coverage fixes were ported there — still falls back to CUDA for any instanced/sphere/curve scene |

### High Effort

| # | Feature | Why it matters |
|---|---|---|
| 7 | **NanoVDB / OpenVDB grids** | `disney-cloud` and similar volumetric scenes still unsupported (uniformgrid media works; sparse VDB grids don't) |
| 8 | **Motion blur** | Time-sampled BVH; static world only right now, no animation support at all |
| 9 | **Displacement mapping** | On-the-fly tessellation + height field for geometric detail |

### Explicitly Out of Scope
| Feature | Reason |
|---|---|
| OIDN / OptiX denoiser | Using custom à-trous denoiser instead |
| OptiX as a 3rd GPU backend | Same RT-core hardware as Vulkan ray query, no capability gain — see `project_vulkan_rt_backend.md` memory |
