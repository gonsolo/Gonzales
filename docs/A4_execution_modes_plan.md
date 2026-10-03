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

### Step 5 -- Registers (path tracer done 2026-10-03)

`ncu` on the pavilion path tracer, before: shade_diffuse and shade_measured 255
registers, conductor 212, dielectric 168, traversal 124, nee 113 (one block of
256 threads per SM, 16.7% theoretical occupancy). Mojo exposes ptxas's
minimum-blocks-per-SM hint as `@__llvm_metadata(`nvvm.minctasm`=N)` together
with `MAX_THREADS_PER_BLOCK_METADATA`; N = 2 caps a 256-thread kernel at 128
registers. Applied to the five hottest shade kernels and the traversal kernel
(`gpu_tuning.mojo`), PT render time at 64 spp:

| scene | N=1 (no cap) | N=2 | N=3 |
|---|---|---|---|
| pavilion | 0.8 s | 0.6 s | 0.6 s |
| kitchen | 2.2 s | 1.5 s | 1.6-1.7 s |
| staircase | 8.5 s | 6.2 s | 6.9-7.0 s |

N = 2 is 25-33% faster, N = 3 less so, and mixing 2 and 3 across shade and
traversal kernels is no better than 2 everywhere. Images are bit-identical
(max abs difference 0.0 on all three scenes). The earlier note that register
caps are a dead end was wrong for the path tracer: it was measured on the VCM
merge/connect kernel. Open: the VCM kernels (`bdpt.mojo`, 255 registers) and
the remaining PT kernels (gen_primary, thin dielectric, coated conductor,
hair, interface, mix).

VCM kernels (pavilion, 16 spp, 3 runs each): megakernel 12.9 s -> 12.2 s (-5%),
staged driver 13.5 s -> 11.0 s (-18%) with N = 2 on the emit, splat, connect,
light-bounce, camera-bounce and shadow-resolve kernels; N = 3 gives no gain.
Images agree to 3e-5 (atomic splats are not reproducible to the bit).

Smoke matrix: `chromatic-medium.vcm` (-7.7%) and `subsurface-coated.pt`
(-3.7%) fail their pins, identically on the commit before this session
(64287c1b), so they are not caused by this step.

### Step 6 -- Path guiding on the GPU (done 2026-10-03; result: no gain)

Ported: the SD-tree lives on the host between iterations and in device memory
during one (`gpu_guide.mojo`); `guide_record` adds atomically, so one shared shard
replaces the CPU's 16; the training schedule is one function, `train_guided`
(`guide.mojo`), driven by a CPU and a GPU renderer. Two bugs found on the way, both
in code the CPU path shares:

- A guide sample below the surface fell back to a BSDF sample while keeping the
  mixture pdf, so the real sampling density was larger than the weight assumed:
  Cornell box 3.9% too bright. Now the sample contributes zero (ratio 1.0003).
- `guide_refine` gave each new spatial leaf an empty directional tree, so with
  many samples per leaf (every leaf splits every refine) the guide never had data
  when it was read, and guided renders were bit-identical to unguided ones.
  Children now start from a half-energy copy of the parent's tree.

Equal-time evaluation, 7 scenes, 2 seeds, relMSE against a 512 spp unguided
reference (efficiency = 1/(relMSE x render time), guided over plain):

| scene | 16 spp | 64 spp |
|---|---|---|
| cornell | 0.32x | 0.10x |
| kitchen | 0.43x | 0.13x |
| staircase | 0.50x | 0.72x |
| classroom | 0.43x | 0.11x |
| living-room | 0.47x | 0.82x |
| bathroom | 0.46x | 0.11x |
| dining-room | 0.38x | 0.13x |

Guided relMSE is the same or worse everywhere and guided renders take 2-10x
longer (the atomic adds contend at the tree roots, and the inline shadow rays
that teach the tree replace the cheaper path). Means agree to 0.2% (bathroom
0.6% low). Verdict: guiding stays opt-in; do not make it the default. The
guide learns only from direct light seen from the second vertex and only on
diffuse surfaces, which may be why it finds nothing to exploit in these scenes.

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
