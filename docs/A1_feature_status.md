# Appendix 2: Feature Status & Remaining Gaps

> **Last updated: 2026-06-05**

This document records which renderer features are implemented and which are still
missing, in priority order. It is kept as a living document — check commits for
the implementing commit hash.

---

## Implemented Features

### Light Transport
| Feature | Commit | Notes |
|---|---|---|
| Unidirectional path tracing | — | CPU multithreaded + GPU wavefront |
| NEE + MIS — area lights | `b237f91` | Power heuristic, solid-angle pdf |
| NEE — distant lights | `07befc5` | Delta light, MIS weight = 1 |
| NEE — point lights | `07befc5` | Delta light, MIS weight = 1 |
| Env-map (infinite) light — BSDF path | `07befc5` | HDRI lookup on path miss |
| Env-map MIS — importance sampling | `aec8cc4` | 2D marginal/conditional CDF at load time; NEE + correct BSDF-side pdf |

### Materials
| Feature | Commit | Notes |
|---|---|---|
| Diffuse (Lambertian) | — | Type 1 |
| Emissive | — | Type 2 |
| Conductor — GGX VNDF | `4cad301` | Types 3; Schlick Fresnel |
| Conductor — true anisotropic GGX | `aec8cc4` | Separate α_x/α_y; UV-gradient tangent frame |
| Dielectric | — | Type 4; Fresnel + Snell refraction |
| Thin dielectric | `aec8cc4` | Type 9; Fresnel reflect or pass-through (no refraction) |
| CoatedDiffuse | `555b35d` | Type 5; GGX coat over Lambertian |
| DiffuseTransmission | `555b35d` | Type 6; Lambertian transmission |
| CoatedConductor | `619c28c` | Type 7; Schlick coat over GGX conductor |
| Mix material | `619c28c` | Type 8; stochastic sub-material selection |
| Normal mapping | `aec8cc4` | Tangent-space normal texture; UV-gradient frame; CPU only |
| UV-mapped textures | — | Via OIIO texture system |

### Infrastructure
| Feature | Commit | Notes |
|---|---|---|
| BVH2 (SAH) traversal | `step2` | CPU + GPU, 32-byte nodes |
| Sobol QMC sampler | — | Morton-curve tile ordering |
| À-trous bilateral denoiser | `step7c` | CPU + GPU |
| Interactive viewer | `step7` | GLFW / Vulkan |
| PBRT v4 scene parser | — | Partial — see gaps below |
| GPU wavefront rendering | `step5-6` | Mojo GPU kernels |

---

## Remaining Gaps — Priority Order

### Quick Wins (Low Effort)

| # | Feature | Why it matters |
|---|---|---|
| 1 | **PBRT parser completeness** | `Include`/`Import`, alpha masking, `Texture "scale"/"mix"`, `Camera "environment"`, portal/projection lights; blocks loading new scenes |
| 2 | **Sphere / quadric primitives** | Analytic intersection; avoids tessellating spheres/disks into triangles |
| 3 | **Multi-channel EXR AOVs** | Normal, depth, object-ID passes for compositing pipeline |

### Medium Effort

| # | Feature | Why it matters |
|---|---|---|
| 4 | **Instancing** (`ObjectInstance`) | Forest/city scenes OOM without it; requires TLAS/BLAS BVH split |
| 5 | **PathState compression** | FP16 throughput/albedo; ~2× GPU memory reduction on 4K frames |
| 6 | **Texture mip-mapping on GPU** | Auto mip + trilinear sampling; eliminates minification aliasing |
| 7 | **Spectral / hero-wavelength** | Dispersion, iridescence, correct dielectric/conductor Fresnel; RGB is systematically wrong for chromatic effects |
| 8 | **Random-walk SSS** | Skin, marble, wax, candles — entire class of translucent materials |

### High Effort

| # | Feature | Why it matters |
|---|---|---|
| 9 | **Vulkan RT cores** | `VK_KHR_ray_tracing_pipeline`; 3–5× GPU traversal speedup |
| 10 | **ReSTIR DI** | Many-light scenes on GPU; temporal + spatial reservoir resampling |
| 11 | **Hair / fiber BSDF** | Bézier curve primitive + Marschner 3-lobe (R/TT/TRT) |
| 12 | **Participating media** | Delta tracking; VDB grid loading; fog, smoke, volumetric glass |
| 13 | **BDPT / caustics** | Glass and mirror caustics are invisible in unidirectional PT |
| 14 | **Motion blur** | Time-sampled BVH; static world only right now |
| 15 | **Displacement mapping** | On-the-fly tessellation + height field for geometric detail |

### Explicitly Out of Scope
| Feature | Reason |
|---|---|
| OIDN / OptiX denoiser | Using custom à-trous denoiser instead |
