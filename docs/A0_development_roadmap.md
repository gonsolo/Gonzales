# Appendix 1: Development Roadmap

**Design principle:** hardware-agnostic compute kernels. The same Mojo code
targets CPU SIMD, GPU warps, and future wide-vector architectures (RISC-V V,
ARM SVE). Write once, run anywhere the hardware is wide.

## Current Architecture

Gonzales uses a **megakernel** integrator: each tile traces one ray at a time
through all its bounces before starting the next ray. The BVH uses 8-wide
nodes (BVH8) with SIMD intersection tests. The scene graph uses Swift
reference types, which introduces ARC (Automatic Reference Counting) overhead
in the traversal hot path.

Profiling on the Barcelona Pavilion scene (16 spp) shows approximately:

- **33%** BVH traversal
- **22%** `swift_retain` / `swift_release` (ARC overhead)
- **15%** triangle intersection and shading
- **30%** materials, textures, and integrator logic

## Phase 1: CPU Performance

Eliminate the Swift language tax and improve single-ray traversal. Each step
produces a measurable speedup and can be benchmarked against the Barcelona
Pavilion scene.

### Step 1: Mojo Traversal Kernel ✅ (2026-03-25)

Write the BVH traversal and triangle intersection in Mojo, exported via
`@export` with C calling convention and called from Swift via a bridging
header. Eliminates all ARC traffic from the hot path (~22% of CPU time).

Mojo's built-in `SIMD[f32, 8]` maps directly to AVX2 for the AABB test,
and the same kernel can later target GPU via `@gpu` (Phase 3) without a
rewrite. A working Mojo CPU+GPU raytracer prototype exists at
`/home/gonsolo/work/mojo_gpu_raytracer/`.

### Step 2: BVH2 with Compact Nodes ✅ (2026-03-28)

Replace BVH8 (288-byte nodes, 4.5 cache lines) with a binary BVH using
32-byte cache-line-sized nodes. For incoherent secondary rays that dominate
path tracing, smaller nodes mean fewer cache misses per traversal step.

Use a linearized depth-first layout with skip pointers for stackless
traversal — this same layout works on GPUs.

### Step 3: OpenImageDenoise Integration ✅ (2026-03-28)

Integrate Intel OIDN for AI-powered denoising. Produces clean images from
noisy renders, dramatically reducing sample counts for production quality.
Already used by PBRT-v4 and Blender Cycles.

## Phase 2: Wavefront Architecture

Restructure the integrator to process rays in batches — the foundation for
GPU compute and future wide-vector CPU architectures.

### Step 4a: Extract PathState (Refactor) ✅ (2026-03-28)

Make the integrator's path state explicit and public. The existing
`BounceState` struct is already close — promote it to a standalone
`PathState` type that can later be stored in flat buffers:

- ray, throughput, accumulated radiance, bounce depth, pixel index
- albedo, first normal (for AOVs)
- specularBounce flag

Verify: **pixel-perfect match** with current output.

### Step 4b: Batch Processing ✅ (2026-03-28)

Collect all rays at the same bounce depth into a buffer and process them
together:

1. Generate all primary rays for the tile into a buffer
2. Trace all primary rays (batch BVH traversal)
3. Shade all hits, generate shadow rays
4. Trace all shadow rays
5. Generate bounce rays, repeat

The tile structure stays intact — only the inner loop changes.

## Phase 3: GPU Wavefront

Port the traversal kernel to GPU compute. The Mojo kernel from Step 1,
compact BVH2 from Step 2, and wavefront architecture from Step 4 provide
the foundation.

### Step 5: GPU BVH Traversal

Add `@gpu` to the existing Mojo traversal kernel. The flat BVH2 node
array and primitive ID arrays are already GPU-friendly (contiguous,
fixed-size structs). The wavefront batch dispatch from Step 4b provides
the host-side launch structure. SOA conversion happens here as needed
for GPU-coalesced memory access.

### Step 6: GPU Shading and Material Evaluation

Port material and texture evaluation to GPU compute. The wavefront
architecture naturally separates trace and shade phases, so each can be a
distinct compute dispatch.

### Step 7: Interactive Viewer ✅ (2026-03-28)

Implement an interactive viewer using GLFW/SDL2 with OpenGL/Vulkan display:

- **Camera navigation**: WASD + mouse look, orbit, pan, zoom
- **Progressive rendering**: start with 1 spp, refine while idle
- **Fullscreen support**: toggle between windowed and fullscreen modes
- **Tone mapping**: real-time exposure and gamma controls

PBRT-v4 has an interactive GUI mode using GLFW and Dear ImGui. The
fullscreen rendering support was contributed upstream (PBRT PR #307,
merged Nov 2022) — the PBRT codebase at `/home/gonsolo/src/pbrt-v4/`
serves as a working reference.

### Step 8: GPU BVH Construction

For dynamic scenes, build the BVH on the GPU using Linear BVH (LBVH) with
Morton codes. Use a TLAS/BLAS two-level structure for instanced geometry.

## Phase 4: Ecosystem Integration

Connect gonzales to production pipelines and tools.

### Step 9: USD Scene Support

OpenUSD (Pixar's Universal Scene Description) integration:

- **Scene abstraction layer** — common interface for PBRT + USD readers
- **OpenUSD C++ bridging** — via Swift C++ interop (Swift 5.9+) or C wrapper
- **Hydra render delegate** — USD's rendering abstraction, used by Cycles,
  Arnold, and RenderMan. Works in any Hydra application (Blender, Solaris,
  usdview, NVIDIA Omniverse)
- **MaterialX support** — USD's material description format

### Step 10: Blender Integration

Gonzales as a render engine inside Blender, via two paths:

- **Python Render Engine API** — `bpy.types.RenderEngine` with a Python
  extension module (similar to Cycles' `_cycles` in
  `intern/cycles/blender/python.cpp`)
- **Hydra render delegate** — reuses the Hydra delegate from Step 9, plugs
  into Blender's Hydra viewport (see `intern/cycles/hydra/`). More
  future-proof: works in any Hydra-capable application

## Optional: CPU-Specific Optimizations

These optimizations improve CPU rendering performance but are not on the
critical path to GPU. They can be added independently at any time.

### SOA Scene Data Layout

Convert scene data from Array-of-Structures to Structure-of-Arrays:
separate arrays for vertex X, Y, Z coordinates. Benefits SIMD ray packets
and GPU coalesced access. Do as part of GPU work (Step 5) or standalone.

### CPU Ray Packets

Implement N-ray packet traversal for coherent rays using SIMD (AVX2, RVV,
SVE). The wavefront batch structure from Step 4b provides the ray buffers.
Width-agnostic with Mojo's `SIMD[DType, N]`. Worth revisiting when
wide-vector CPUs (RISC-V V) become mainstream.

## Step Dependencies

```text
Step 1: Mojo Kernel ───────┐
                           ▼
Step 2: BVH2 ────────────► Step 5: GPU Traversal ──► Step 8: GPU BVH Build
                           ▲
Step 4: Wavefront ─────────┤
                           ▼
                    Step 6: GPU Shading
                           ▼
                    Step 7: Interactive Viewer
                           ▼
                    Step 9: USD ──► Step 10: Blender

Step 3: OIDN (Independent)
Optional: SOA, CPU Ray Packets (independent, any time)
```

## VFX Reference Platform Ecosystem

Production rendering relies on open-source libraries maintained under the
ASWF (Academy Software Foundation). All are C++ and follow the same Swift
bridging pattern already used for OIIO and Ptex.

| Library                 | Status  | Purpose                                                    |
| ----------------------- | ------- | ---------------------------------------------------------- |
| **OpenImageIO**         | ✅ Used | Image I/O (EXR, PNG, JPEG) and texture system              |
| **Ptex**                | ✅ Used | Per-face texture mapping (Disney scenes)                   |
| **OpenImageDenoise**    | ✅ Used | AI denoiser for noise-free images at low sample counts     |
| **MaterialX**           | Step 9  | Industry-standard material descriptions (required for USD) |
| **OpenColorIO**         | Future  | Color management (ACES, scene-linear workflows)            |
| **OpenShadingLanguage** | Future  | Runtime-compiled shaders (used by Cycles, Arnold)          |
| **OpenVDB**             | Future  | Sparse volumetric data (clouds, smoke, fire)               |
| **Alembic**             | Future  | Animation cache format (animated geometry)                 |
