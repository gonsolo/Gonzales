# GPU profiling & optimization findings

Tooling and the data-backed conclusions from profiling the GPU path
(NVIDIA RTX 3060, Nsight Compute 2026.2, CUDA 13.3).

## How to profile

```
make profile            # full: hot kernels on cornell-box -> build/gonzales.ncu-rep
make profile-shade      # just shade_nee_gpu: regs/occupancy/duration
```

Nsight Compute needs admin access to GPU performance counters
(`RmProfilingAdminOnly: 1`), so the targets run `ncu` under `sudo -E`
(`-E` preserves the environment so the Mojo runtime libraries load). Import a
report with `ncu --import <file>.ncu-rep --page details`.

Wall-clock timing on this machine is **too noisy** to measure small kernel
wins (bathroom GPU varies ~7.5–9.8 s run to run from clock/thermal/scheduler
noise). Use Nsight Compute's deterministic per-kernel metrics (registers,
occupancy, duration, memory throughput) to evaluate changes, not wall time.

## Baseline (cornell-box, one wavefront batch)

| Kernel | Duration | DRAM % | Compute % | Regs | Occupancy |
|---|---|---|---|---|---|
| `gen_primary_ray` | 3.8 ms | 23% | 23% | 34 | 92% |
| `traverse_paths`  | 2.3–2.5 ms | **72–92%** | 25–60% | 48 | 64–67% |
| `shade_nee_gpu`   | 4.9–6.4 ms | 44–56% | 13–18% | **96** | **28%** |

`shade_nee_gpu` is the dominant kernel. Note it does NEE for area + envmap +
sphere lights, each with an **inline any-hit BVH traversal** for the shadow
ray — so it is both compute- and memory-heavy, not a pure-shading kernel.

## Findings (what the profiler proved)

### #1 — shade-kernel occupancy via register cap: NOT a win (disproven)
Hypothesis: 96 regs → 28% occupancy is the bottleneck. Capping registers with
`@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=...)` (= `nvvm.maxntid` /
`__launch_bounds__`) does raise occupancy, but **the kernel gets slower** —
the register spills cost more than the occupancy gains:

| maxntid | Regs | Occupancy | shade duration |
|---|---|---|---|
| (none)  | 96 | 28% | **4.91 ms** (best) |
| 768     | 80 | 40% | 5.14 ms |
| 1024    | 64 | 54% | 5.63 ms |

The trend is monotonic: the kernel is optimal at its natural 96 registers. Its
low occupancy is fine because each thread does substantial independent work and
register reuse beats more resident warps. `@no_inline` on the cold material
BRDFs also left registers at 96 (the compiler counts the whole call graph), so
that doesn't help either. **Conclusion: don't cap registers.**

### #3 — SoA path state: weak expected benefit
`traverse_paths` is DRAM-bound, but the traffic is dominated by **BVH node
reads** (~32 B × dozens of nodes/ray), not the one-time ray read from
`PathState`. `gen_primary_ray` (the main path-state writer) is **not**
DRAM-bound (23%). So converting `PathState_C` (AoS, 96 B) to SoA — a large,
invasive refactor touching every shade function — would mostly help
`gen_primary_ray`'s once-per-batch writes, a small slice of the frame.
Low priority until the bigger items are done.

### #2 — BVH node bandwidth: the real lever (large rewrite)
`traverse_paths` is genuinely bandwidth-bound (DRAM 72–92%), and the shade
kernel's inline shadow rays add more of the same traffic. `BVH2Node` is already
a tight 32 B (no padding to trim), so reducing bandwidth needs either:
- **Quantized/compressed nodes** (parent-relative 16-bit AABBs → ~16 B/node,
  halving traversal bandwidth). Conservative quantization keeps the image
  identical (verify: render mean unchanged); risk is a non-conservative bug
  (caught by the mean).
- **Wider BVH (BVH4/BVH8)** — fewer nodes traversed per ray.

Both are multi-hundred-line rewrites of the builder + traversal across the CPU
and GPU paths. This is the highest-value optimization but should be done
attended, verifying the render mean is unchanged at each step against this
baseline.

### Measured: narrow-node compression does NOT help (sector granularity)
Tried compressing `BVH2Node` 32 B → 24 B (global 16-bit quantized AABB, header
in node[0]). Output stayed byte-identical (cornell 0.1144, car 0.1051,
bathroom 2.426 — verified), but **traverse was not faster** (2.26–2.52 ms,
DRAM still 72–91%, same as 32 B) and registers rose 48→53. Reason: NVIDIA DRAM
transactions are **32-byte sectors**. Incoherent BVH2 traversal reads one node
per step from a scattered address ⇒ one 32 B sector per node *regardless* of
whether the node is 16/24/32 B (24 B is actually worse — its stride straddles
sector boundaries). So shrinking a 2-wide node below 32 B cannot reduce sector
traffic. Reverted.

**Implication:** the bandwidth win requires **wide nodes**, and even an
*uncompressed* BVH4 node (~112 B ≈ 4 sectors for 4 children) barely beats the
~3 sectors of the BVH2 subtree it replaces. The real win is a **compressed**
wide node: CWBVH-style — parent bounds (24 B) + N children quantized 8-bit
relative to the parent + child refs, sized to span few sectors (BVH4 ≈ 64 B ≈
2 sectors for 4 children vs ~3 for BVH2; BVH8/CWBVH ≈ 80 B ≈ 3 sectors for 8
children vs ~7). Width amortizes the parent-bounds storage and cuts steps. So:
**compressed BVH4/BVH8 is the only lever that actually moves traverse
bandwidth** — narrow compression and register caps are dead ends (both
measured).

### Measured: compressed BVH4 is correct but SLOWER (dequant ALU > bandwidth)
Implemented a full compressed BVH4 (64 B/node: parent AABB + 4 children with
8-bit parent-relative quantized bounds + packed child meta; collapse the SAH
BVH2 greedily into 4-wide; conservative ±1-cell quantization). It is correct —
render means byte-identical (cornell 0.1145, car 0.1006, bathroom 2.408 in that
build env) — and cuts node count ~4× (bathroom 1.03M→0.25M). But it is
**~26% slower**: bathroom best-of-5 9.3 s vs 7.4 s baseline. Three traversal
variants tried (gather+sort 10.2 s, scalar nearest-first 9.3 s, full SIMD-4
13 s — lane extraction spills to local memory). The per-node 8-bit
dequantization ALU outweighs the saved node-fetch bandwidth on this software
GPU traverser. Preserved on branch `bvh4-compressed-experiment` (NOT merged).

### Overall conclusion
Every software lever tried — register cap (#1), SoA (#3, weak premise), narrow
node compression, and compressed BVH4 (#2) — yields **no speedup**; combined
with the project's prior failures (material sorting, separate shadow kernel,
path compaction), the GPU path is at a **software-traversal local optimum**.
The remaining gap to pbrt is pbrt's use of **hardware RT cores (OptiX)**, which
Mojo/CUDA cannot access (see project memory). Without hardware RT, the
realistic wins left are algorithmic (fewer rays/samples, better sampling) or
quality-vs-time trade-offs — not traversal micro-optimization.
