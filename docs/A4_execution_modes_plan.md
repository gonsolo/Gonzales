# A4: Execution Modes Plan

Plan for keeping every combination of **where** a render runs, **how** rays
are traced and **which** integrator, without duplicating feature code.

**Status (2026-10-03): draft, nothing started.** Every claim about current
behaviour below was read from the source on this date; items marked
*verify* were not run and must be checked before a step starts.

## 1. The matrix

| Axis | Choices |
|---|---|
| Where | CPU, GPU megakernel, GPU wavefront |
| Traversal | software BVH, hardware RT (`--rt-hardware`, RT cores via `GONZALES_RTCORE`) |
| Integrator | PT, SPPM, VCM |

Hardware RT exists only on the GPU. The CPU always uses the software BVH.

### What exists today

| Integrator | CPU | GPU megakernel | GPU wavefront |
|---|---|---|---|
| PT | yes | no | yes (`gpu_render_wavefront`, `gpu.mojo`) |
| VCM | `vcm_render` | `vcm_render_gpu` | `vcm_render_gpu_wavefront` (`--vcm-wavefront`, implied by `--rt-hardware`); same features and the same image as the megakernel |
| SPPM | yes | yes | not applicable (*verify*) |

Measured 2026-10-03, pavilion night 640x340, 64 spp, seed 1, whole-image
ratio to the PT reference / total time: megakernel VCM 1.040 / 54.6 s;
staged VCM, software traversal, 1.040 / 56.9 s (same image as the megakernel);
staged VCM on the RT cores 1.042 / 69.2 s, of which about 10 s is
acceleration-structure setup. VCM is not trace bound, so the RT cores do not
speed it up (`docs/rtcore/NOTES.md` reports the same). An earlier run that
looked 3.4x slower and darker had silently fallen back to the Vulkan ray-query
path: `build/rt_trace.cubin` is resolved relative to the working directory, so
`--rt-hardware` run from another directory prints one "could not load" line
and continues. Set `GONZALES_RTCORE_CUBIN` to an absolute path, or fix the
lookup (see step 1).

## 2. Principle

One implementation of every feature, in shared `@always_inline` functions
(shading, materials, light sampling, MIS weights, merge and connect
estimators). Each mode is a thin driver. This is the project's
unify-while-fixing rule: a feature added to a driver instead of to the shared
function is the bug shape that left the wavefront VCM driver without the
footprint radius.

What stays per mode, deliberately:

- scheduling and where path state lives (registers, global arrays, tile loops);
- memory operations (GPU atomics for film splats and guide recording, CPU
  shards);
- the traversal call.

## 3. Steps

Each step has a pass criterion. A step that fails its criterion stops the
plan and is reported, not worked around.

### Step 1 -- Traversal interface

Put closest-hit and any-hit behind one comptime-selected interface, so shading
code asks for a hit and does not know the source. Existing entry points:
`traverse_bvh2_core`, `any_hit_bvh2_core` (`bvh.mojo`), and
`rtcore_shadow_rays_gpu` (`gpu_wavefront.mojo`).

Current state of the RT-core path (`docs/rtcore/NOTES.md`, 2026-10-02):
instancing through the TLAS, spheres, alpha cutouts and VCM work; curves do not
(`pipeline.mojo` falls back to the Vulkan ray-query path with a note). Only
sm_86 and one driver version were tested, and the acceleration structure is
built through Vulkan. The cubin lookup is cwd-relative and falls back silently;
make it resolve next to the executable and make the fallback loud.

Pass: with the interface in place and software traversal selected, every
render is bit-identical to before (same image, same `make unittest`, same
smoke matrix). No performance regression with `ncu` on the shade and merge
kernels.

### Step 2 -- Hardware versus software, measured

Measured 2026-10-03, path tracer, 64 spp, seed 1, `--no-denoise`, render time
only (the "Done:" line; hardware setup is extra: about 10 s on the pavilion).
Software BVH against `--rt-hardware` (cubin found next to the executable):

| scene | software [s] | hardware [s] | hardware speed-up | mean hw/sw |
|---|---|---|---|---|
| Cornell box | 0.1 | 0.1 | 1.00 | 1.0000 |
| pavilion (zz_bias) | 0.8 | 4.7 | 0.17 | 1.135 (unclamped, firefly-dominated mean; not yet checked) |
| bathroom | 2.4 | 28.6 | 0.08 | 1.0000 |
| kitchen | 2.2 | 7.7 | 0.29 | 1.0000 |
| staircase | 8.6 | 14.7 | 0.59 | 1.0001 |
| classroom | 2.6 | 8.0 | 0.33 | 1.0004 |
| veach-ajar | 1.0 | 3.0 | 0.33 | 1.0000 |
| bistro vespa | 5.9 | 1.9 | 3.11 | not compared |

The RT cores win clearly only on the heaviest geometry (vespa, 1.4 M triangles).
On the other scenes hardware tracing is slower, by up to 12x, with images that
agree. The slowdown is large in absolute terms on small scenes, which points at
per-launch cost (an ncu run of the pavilion shows 180 launches each of the
trace and "convert" kernels at 2 spp) and at the alpha re-trace passes, not at
traversal. Next: profile where hardware time goes on bathroom and pavilion
before deciding anything. Do not make hardware the default on this evidence.

Follow-up, same day. The alpha passes now stop as soon as no ray is pending
(one 4-byte read-back per pass instead of always `RT_ALPHA_PASSES = 10` traces);
bathroom hardware 28.6 s -> 19.1 s, identical image, no change elsewhere
(the other scenes have no alpha). What remains is not the alpha passes: with
`ncu` on the kitchen (no alpha) the per-round full-grid passes around the trace
cost about as much as the trace itself: pack rays 8.0 s, result convert 6.5 s,
unpack 5.3 s, pack/reset shadow tasks 10.3 s, each over all n_pix x 8 ray
slots, live or dead, 18 rounds per sample. Software traversal fuses all of it
into the shade kernels. Crown (3.5 M triangles) is also slower in hardware
(8.3 s against 5.2 s at 16 spp), so triangle count is not a usable
switch. Bistro cafe and boulangerie run out of memory in hardware mode on the 12 GB
card, so they could not be compared. Closing the gap needs path compaction or
fusing pack/trace/unpack, a larger change than this step.

### Step 3 -- Shared bounce for VCM (check, probably already done)

The staged driver renders the same image as the megakernel (measured above),
so the features are shared. Verify by reading `vcm_render_gpu_wavefront`: if it
calls the same `_bdpt_*` functions as `vcm_render_gpu`, no extraction is
needed. If it duplicates logic anywhere, move that into the shared function.

Pass: the two modes agree on the 4-seed binned ratio against the BDPT referee
within one standard deviation, and the staged driver's duplicated logic (if
any) is listed and removed.

### Step 4 -- Staged-driver memory

The staged driver keeps per-path state for every pixel and runs out of memory
at 1920x1080 on a 12 GB card, with and without the RT cores; the megakernel
does not (`docs/rtcore/NOTES.md`). Measure the per-path state size, and tile
the pixels if that is what it takes to render the teaser at full resolution.

Pass: staged VCM renders 1920x1080 on the 12 GB card.

### Step 5 -- Registers

Profile both GPU modes with `ncu` (recipe in
`project_gpu_speed_vs_pbrt_cycles` memory). Record registers per thread,
achieved occupancy and spill sectors per kernel. Decide per kernel whether
splitting helps. Register caps and occupancy tuning were dead ends before;
do not assume this changes.

Pass: a table; a split or cap is kept only if the measured time improves.

### Step 6 -- Path guiding on the GPU

Behind a comptime flag so unguided kernels keep their register count.
Needed: device-resident SD-tree, recording through the deferred shadow tasks
(`ShadowTask` in `render_state.mojo`) with atomic adds, and a host iteration
loop (render, merge, refine, re-upload). Guiding applies to diffuse materials
only and learns from direct light seen from the second vertex.

Gate before this step: equal-time CPU comparison, guided against unguided, on
4 scenes. If there is no clear gain, guiding stays opt-in and the port stops.

Pass: guided and unguided means agree on several scenes; unguided kernels
are bit-identical to before.

### Step 7 -- ReSTIR on the GPU

Today ReSTIR DI and GI are CPU only. Do not start before steps 1-4 are done
and the gain of CPU ReSTIR has been measured on the corpus.

## 4. Test matrix

An identity check per mode and traversal pair, run automatically, on at least
one scene per feature: Cornell box (closed cavity, known reference),
pavilion night (truncation bias), and one instanced scene. Extend
`make smoketest`'s integrator x feature matrix with mode and traversal
columns instead of adding a separate harness.

## 5. Non-goals

- Making hardware RT the default before step 2 shows it wins. For VCM it
  currently does not (69.2 s against 54.6 s).
- Retiring the megakernel or the wavefront. Both stay; a flag and a measured
  heuristic choose.
- Any promise that a unified code path is also fast. Each mode is profiled
  on its own.
