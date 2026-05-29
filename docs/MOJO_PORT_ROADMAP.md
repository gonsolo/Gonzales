# Gonzales → Pure Mojo Port Roadmap

Goal: port the entire gonzales renderer to **pure Mojo** — no Swift, no
hand-written C++. The current architecture is a strangler-fig in progress:
Swift is the application shell (CLI → parse → build scene → drive render →
write image) and calls into `libmojo.so` over a C ABI. Steps 1–10 already
moved the render hot path into Mojo behind that ABI. The same pattern carries
us the rest of the way: keep Swift as a verification harness, hollow out one
subsystem at a time, then flip the shell to Mojo last.

## Verification bar

Every step keeps the build green and is verified by re-rendering the Cornell
box and comparing to the golden image:

```
make
compare -metric RMSE cornell-box.exr cornell-box-golden.exr diff.png
```

RMSE ≈ 0.01 is the established pass bar (the residual is random-seed variance,
not error). Swift remains the driver until Phase C.

## Decision: external libraries

"No C++" is interpreted pragmatically: stop writing *our own* Swift/C++, but
keep linking the external C/C++ libraries and call them through Mojo's C FFI
instead of Swift bridges:

- **OpenImageIO** — EXR/PNG image I/O
- **Ptex** — texture access
- **OpenImageDenoise** — denoising (CUDA)
- **Vulkan / GLFW** — interactive viewer

Reimplementing these in pure Mojo (EXR codec, Ptex, a denoiser, a Vulkan
renderer) is a separate, much larger effort per library and is explicitly out
of scope unless decided otherwise.

## Status snapshot

**Phases A–C complete — pure Mojo CLI renderer shipped and cleaned up.**

`Sources/mojoKernel/kernel.mojo` (~4800 lines) now owns the entire pipeline:
CLI entry point, `.pbrt` parsing, scene construction, BVH2 build (SAH),
path tracing loop, 6 material shaders (diffuse, conductor, dielectric,
coatedDiffuse, diffuseTransmission, arealight), NEE, Russian roulette,
ZSobol sampling, Gaussian filter, per-pixel film accumulation, parallel tile
scheduling (all logical cores via `std.algorithm.parallelize`),
joint-bilateral denoising, EXR output via OpenImageIO FFI, and three texture
types: constant, checkerboard (world-space 3D), and imagemap (via OIIO FFI).

`Sources/SobolGenerator/gen_sobol.mojo` generates the Sobol matrix binary
at build time.

Swift (`libgonzales`, ~17k lines) and all stale C bridges (`ptexBridge`,
`vulkanViewer`, `openimageio`, `ptex` raw headers, `mojo_kernel.h`,
`dummy.c`) have been deleted. `Sources/` now contains only three directories:
`mojoKernel/`, `openImageIOBridge/`, and `SobolGenerator/`.

1024×1024 / 64 spp Cornell box with imagemap textures renders in ~20 s on
8 logical cores (was ~79 s single-threaded).

## Phase A — Parsing & scene construction into Mojo

Mojo learns to turn a `.pbrt` file into render-ready data.

| Step | Scope | Verify |
|------|-------|--------|
| 11a | `PbrtScanner` → whole-file in-memory buffer (Swift-only refactor, enabler) | identical render |
| 11b | `scanFloat`/`scanInt` numeric conversion → Mojo, over `(bytes, cursor)` | identical render |
| 11c | Bulk array scan (`parseReals`/`parseIntegers`) → one Mojo call per `[ … ]` run | identical render + parse speedup on big scenes |
| 11d | Token/keyword scanning (`skipWhitespace`, `scanString`, `scanUpToCharactersList`) → Mojo | identical render |
| 12 | pbrt directive interpreter (`parse()` state machine, `ParameterDictionary`) → Mojo | identical render |
| 13 | Transforms/CTM stack + shape/mesh construction → Mojo | identical render |
| 14 | Materials, lights, camera, sampler, filter *construction* → Mojo (shaders already there) | identical render |

## Phase B — Render orchestration & output into Mojo

Mojo can run the whole pipeline behind the ABI.

| Step | Scope | Verify |
|------|-------|--------|
| 15 | Tile generation + parallel scheduling → Mojo | identical render |
| 16 | Film image buffer + normalize → Mojo | identical render |
| 17 | Denoise: call OIDN C API directly from Mojo (drop Swift `Denoiser`) | identical render |
| 18 | Image write: call OpenImageIO C API from Mojo (drop `openImageIOBridge`) | identical `.exr` on disk |

## Phase C — The flip (remove Swift)

| Step | Scope | Verify |
|------|-------|--------|
| 19 | `PbrtScanner` ported to Mojo (handle-based API; gzip decompression stays in Swift) | identical render |
| 20 | `fn main()` in kernel.mojo (argv, sobol load, timing); `gen_sobol.mojo` replaces Swift tool; Makefile drops Swift build; delete `gonzales`, `DevirtualizeMacro`, `SobolGenerator/main.swift`, `Plugins`, stale bridges — `libgonzales` kept as reference | cornell-box renders from pure Mojo binary |

## Phase D — Remaining / optional subsystems

| Step | Scope | Notes |
|------|-------|-------|
| 21a | Constant textures | done |
| 21b | Checkerboard texture (world-space 3D) | done |
| 21c | Imagemap texture via OIIO FFI | done |
| 21d | Ptex (FFI) | for subdivision-surface scenes |
| 22 | Volumetric media / `Medium` | not in kernel yet |
| 23 | GPU path consolidation | a GPU traversal path already exists |
| 24 | Vulkan interactive viewer (FFI) | largest C++ chunk; lowest priority |

## Sequencing notes

- **A → B → C is the critical path** to a pure-Mojo CLI renderer for untextured
  scenes. Phase D is independent and can come whenever those features are needed.
- Steps stay independent because each hides behind the existing C ABI. The only
  hard ordering is within Phase A: 11a enables 11b/c; parse (11–12) before
  scene-build (13–14) before render-orchestration (15+).
- The flip (19–20) is the only step without an incremental fallback, but by then
  the Swift shell is just glue, so it is small.

## Progress

- [x] 11a — whole-file scanner buffer
- [x] 11b — numeric scan in Mojo
- [x] 11c — bulk array scan in Mojo
- [x] 11d — token/keyword scan in Mojo
- [x] 12 — directive interpreter in Mojo
- [x] 13 — transforms + shape construction in Mojo
- [x] 14 — material/light/camera/sampler/filter construction in Mojo
- [x] 15 — tile generation + scheduling in Mojo
- [x] 16 — film buffer + normalize in Mojo
- [x] 17 — denoise via Mojo joint bilateral filter (OIDN removed)
- [x] 18 — image write via OpenImageIO bridge from Mojo (mojo_write_exr)
- [x] 19 — PbrtScanner ported to Mojo (handle-based API)
- [x] 20 — pure Mojo binary: fn main(), gen_sobol.mojo, Makefile rewrite, Swift build tooling deleted
- [x] cleanup — libgonzales Swift reference deleted; stale bridges removed; Sources/ down to 3 directories
- [x] 21a — constant textures
- [x] 21b — checkerboard texture (world-space 3D XOR)
- [x] 21c — imagemap texture via OpenImageIO FFI
- [x] parallelism — tile dispatch parallelized over all logical cores (~4× on 8-core machine)
- [ ] 21d — Ptex (FFI)
- [ ] 22 — volumetric media
- [ ] 23 — GPU path consolidation
- [ ] 24 — Vulkan viewer
