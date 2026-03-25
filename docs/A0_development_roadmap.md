# Appendix 1: Development Roadmap

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
`/home/gonsolo/work/mojo_gpu_raytracer/`. Estimate: 1–2 weeks.

### Step 2: BVH2 with Compact Nodes

Replace BVH8 (288-byte nodes, 4.5 cache lines) with a binary BVH using
32-byte cache-line-sized nodes. For incoherent secondary rays that dominate
path tracing, smaller nodes mean fewer cache misses per traversal step.

Use a linearized depth-first layout with skip pointers for stackless
traversal — this same layout works on GPUs. Estimate: 1–2 weeks.

### Step 3: OpenImageDenoise Integration

Integrate Intel OIDN for AI-powered denoising. Produces clean images from
noisy renders, dramatically reducing sample counts for production quality.
Already used by PBRT-v4 and Blender Cycles. Estimate: 2–3 days.

## Phase 2: Wavefront Architecture

Restructure the integrator to process rays in batches — the foundation for
both CPU ray packets and GPU compute.

### Step 4: Wavefront Integrator

Restructure `Tile.render()` from *one ray through all bounces* to *all rays
at bounce N, then bounce N+1*:

1. Generate all primary rays for the tile into a buffer
2. Trace all primary rays (batch BVH traversal)
3. Shade all hits, generate shadow rays
4. Trace all shadow rays
5. Generate bounce rays, repeat

The tile structure stays intact — only the inner loop changes.
Estimate: 2–3 weeks.

### Step 5: SOA Scene Data Layout

Convert scene data from Array-of-Structures to Structure-of-Arrays:

- **Meshes**: separate arrays for vertex X, Y, Z coordinates instead of
  `[Point3]`
- **Materials/textures**: flat indexed arrays instead of class references
- **Primitive IDs**: flat index + type tag instead of enum dispatch

SOA maps directly to both SIMD lanes (CPU) and coalesced memory access
(GPU). Estimate: 1–2 weeks.

### Step 6: CPU Ray Packets

Implement 8-ray packet traversal for coherent rays. Within each tile,
process 2×4 pixel blocks as a single packet through the BVH. Primary and
shadow rays benefit most. Fall back to single-ray when coherence drops
(Embree's hybrid approach). Estimate: 2–3 weeks.

## Phase 3: GPU Wavefront

Port the traversal kernel to GPU compute. The Mojo kernel from Step 1 and
SOA data from Step 5 provide the foundation.

### Step 7: GPU BVH Traversal

Implement BVH traversal as a GPU compute kernel. The Mojo traversal from
Step 1 targets GPU via `@gpu`. SOA ray buffers from Step 5 and flat BVH
from Step 2 provide the data layout. The wavefront integrator from Step 4
provides the batch dispatch structure. Estimate: 3–4 weeks.

### Step 8: GPU Shading and Material Evaluation

Port material and texture evaluation to GPU compute. The wavefront
architecture naturally separates trace and shade phases, so each can be a
distinct compute dispatch. Estimate: 3–4 weeks.

### Step 9: Interactive Viewer

Implement an interactive viewer using GLFW/SDL2 with OpenGL/Vulkan display:

- **Camera navigation**: WASD + mouse look, orbit, pan, zoom
- **Progressive rendering**: start with 1 spp, refine while idle
- **Fullscreen support**: toggle between windowed and fullscreen modes
- **Tone mapping**: real-time exposure and gamma controls

PBRT-v4 has an interactive GUI mode using GLFW and Dear ImGui. The
fullscreen rendering support was contributed upstream (PBRT PR #307,
merged Nov 2022) — the PBRT codebase at `/home/gonsolo/src/pbrt-v4/`
serves as a working reference. Estimate: 2–3 weeks.

### Step 10: GPU BVH Construction

For dynamic scenes, build the BVH on the GPU using Linear BVH (LBVH) with
Morton codes. Use a TLAS/BLAS two-level structure for instanced geometry.
Estimate: 2–3 weeks.

## Phase 4: Ecosystem Integration

Connect gonzales to production pipelines and tools.

### Step 11: USD Scene Support

OpenUSD (Pixar's Universal Scene Description) integration:

- **Scene abstraction layer** — common interface for PBRT + USD readers,
  feeding SOA data from Step 5
- **OpenUSD C++ bridging** — via Swift C++ interop (Swift 5.9+) or C wrapper
- **Hydra render delegate** — USD's rendering abstraction, used by Cycles,
  Arnold, and RenderMan. Works in any Hydra application (Blender, Solaris,
  usdview, NVIDIA Omniverse)
- **MaterialX support** — USD's material description format

Estimate: 4–6 weeks.

### Step 12: Blender Integration

Gonzales as a render engine inside Blender, via two paths:

- **Python Render Engine API** — `bpy.types.RenderEngine` with a Python
  extension module (similar to Cycles' `_cycles` in
  `intern/cycles/blender/python.cpp`)
- **Hydra render delegate** — reuses the Hydra delegate from Step 11, plugs
  into Blender's Hydra viewport (see `intern/cycles/hydra/`). More
  future-proof: works in any Hydra-capable application

Estimate: 3–4 weeks.

## Step Dependencies

```text
Step 1: Mojo Kernel ───────┐
                           ▼
Step 2: BVH2 ────────────► Step 7: GPU Traversal ──────► Step 10: GPU BVH Build
                           ▲
Step 4: Wavefront ─────────┤
  ├─► Step 6: CPU Packets  │
  └─► Step 9: Viewer       │
                           │
Step 5: SOA Data ──────────┤
  ├─► Step 11: USD ──────► Step 12: Blender Integration
  └─► Step 8: GPU Shading

Step 3: OIDN (Independent)
```

## VFX Reference Platform Ecosystem

Production rendering relies on open-source libraries maintained under the
ASWF (Academy Software Foundation). All are C++ and follow the same Swift
bridging pattern already used for OIIO and Ptex.

| Library                 | Status  | Purpose                                                    |
| ----------------------- | ------- | ---------------------------------------------------------- |
| **OpenImageIO**         | ✅ Used | Image I/O (EXR, PNG, JPEG) and texture system              |
| **Ptex**                | ✅ Used | Per-face texture mapping (Disney scenes)                   |
| **OpenImageDenoise**    | Step 3  | AI denoiser for noise-free images at low sample counts     |
| **MaterialX**           | Step 11 | Industry-standard material descriptions (required for USD) |
| **OpenColorIO**         | Future  | Color management (ACES, scene-linear workflows)            |
| **OpenShadingLanguage** | Future  | Runtime-compiled shaders (used by Cycles, Arnold)          |
| **OpenVDB**             | Future  | Sparse volumetric data (clouds, smoke, fire)               |
| **Alembic**             | Future  | Animation cache format (animated geometry)                 |
