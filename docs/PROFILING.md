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
