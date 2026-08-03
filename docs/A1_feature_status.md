# Appendix 2: Feature Status & Remaining Gaps

> **Last updated: 2026-08-03**

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
| MNEE (manifold caustics) | `46789d4`+ | Glass/coateddiffuse-base caustics; also wired into ReSTIR's resolve step |
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
| Vulkan RT — 2nd GPU intersection backend | `a9eb370`+ | `VK_KHR_ray_query`; primary rays + diffuse-branch shadow rays, scene-adaptive on/off by light-path occupancy; **mesh-only** — see gap below |

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
| 4 | **Vulkan RT scene coverage** | Object instancing DONE (2026-08-03, plain `--gpu --vulkan-rt-shade` batch path only — multi-geometry-per-template BLAS + real per-instance transforms, verified on barcelona-pavilion's tree instancing vs the CUDA baseline). Still hard-falls-back to CUDA for curves or spheres — excludes hair scenes and sphere-light scenes from the Vulkan RT speedup. Also still falls back for ANY instanced scene on the separate VCM Vulkan RT path (bdpt.mojo), not yet extended. Remaining plan in `project_vulkan_rt_backend.md` memory: spheres (tessellate for traversal only) → curves (hardest; needs either a custom intersection shader or a broad/narrow two-phase reuse of the existing CUDA curve-candidate pattern) |
| 5 | **Random-walk SSS** | Skin, marble, wax, candles — current "subsurface" is a coateddiffuse approximation, not real volumetric random-walk transport |
| 6 | **ReSTIR beyond primary-bounce direct lighting** | ReSTIR DI only covers bounce 0, area lights, diffuse material. Phases 4-8 of `docs/A2_restir_migration_plan.md` (ReSTIR GI, SMS-ReSTIR, volumetric ReSTIR, BDPT-ReSTIR) would extend its reach — mostly not started. Phase 9 (unifying ReSTIR's reservoir formalism with VCM's connect/merge into one joint structure) is explicitly marked **open research** in the plan doc itself, not a scoped deliverable |

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
