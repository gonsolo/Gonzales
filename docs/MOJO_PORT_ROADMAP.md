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

Already in Mojo (`Sources/mojoKernel/kernel.mojo`, ~2170 lines): BVH2 traversal
& triangle intersection, path tracing loop, 6 material shaders (diffuse,
conductor, dielectric, coatedDiffuse, diffuseTransmission, arealight), NEE,
Russian roulette, ZSobol sampling, Gaussian filter importance sampling,
per-pixel film accumulation, BVH2 construction (SAH).

Swift is ~15k lines across 18 `libgonzales` subsystems plus C/C++ glue
(`openImageIOBridge`, `ptexBridge`, `exr`, `vulkanViewer`) and build tooling
(`DevirtualizeMacro`, `SobolGenerator`).

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
| 19 | CLI/arg parsing → Mojo; add a thin Mojo `main` (executable target, not just shared lib) | binary renders cornell-box identically |
| 20 | Delete `libgonzales`, the Swift `gonzales` target, `DevirtualizeMacro`; replace `SobolGenerator` with a Mojo/data step | regression: cornell-box + several pbrt-v4 scenes match |

## Phase D — Remaining / optional subsystems

| Step | Scope | Notes |
|------|-------|-------|
| 21 | Textures + Ptex (FFI) | needed for textured scenes |
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
- [ ] 12 — directive interpreter in Mojo
- [ ] 13 — transforms + shape construction in Mojo
- [ ] 14 — material/light/camera/sampler/filter construction in Mojo
- [ ] 15 — tile generation + scheduling in Mojo
- [ ] 16 — film buffer + normalize in Mojo
- [ ] 17 — denoise via OIDN C API from Mojo
- [ ] 18 — image write via OpenImageIO C API from Mojo
- [ ] 19 — Mojo CLI + `main`, Mojo executable target
- [ ] 20 — remove Swift, regression suite
- [ ] 21 — textures + Ptex
- [ ] 22 — volumetric media
- [ ] 23 — GPU path consolidation
- [ ] 24 — Vulkan viewer
