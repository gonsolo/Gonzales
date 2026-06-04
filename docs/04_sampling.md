# Sampling

Monte Carlo integration is only as good as the samples that drive it.
Poorly distributed samples produce noisy images; well-distributed ones
converge faster. Gonzales uses two main sampling strategies: piecewise
constant distributions for light selection, and Z-Sobol sequences for
low-discrepancy path tracing samples. All sampling code lives in
`sampling.mojo`.

## Piecewise Constant Distributions

Many rendering decisions require sampling from a discrete distribution
defined by a table of weights — for example, choosing which light to
sample proportionally to its emitted power. The implementation builds a
CDF (cumulative distribution function) from an array of non-negative
values. Binary search via `findInterval` locates the correct bin in O(log n).

The 2D extension uses one 1D distribution per row plus a marginal distribution
over rows, enabling efficient importance sampling of environment maps.

## Z-Sobol Sampler

For primary sampling dimensions (pixel position, lens, BSDF directions),
gonzales uses a Z-ordered Sobol sequence. Unlike purely random samples,
Sobol sequences are *low-discrepancy* — they fill the sample space more
evenly, reducing variance without increasing sample count.

### Owen Scrambling

Raw Sobol sequences can exhibit visible structure in 2D projections. Owen
scrambling randomizes the sequence while preserving its low-discrepancy
properties. Gonzales implements a fast approximation using bit reversal
and hash mixing:

```mojo
fn fast_owen_scramble(value_in: UInt32, seed: UInt32) -> UInt32:
    var v = reverse_bits32(value_in)
    v ^= v * UInt32(0x3d20adea)
    v += seed
    v *= (seed >> 16) | UInt32(1)
    v ^= v * UInt32(0x05526c56)
    v ^= v * UInt32(0x53a22864)
    return reverse_bits32(v)
```

The scrambler reverses the bits of the input, applies a sequence of
multiplicative hash steps, and reverses again — effectively applying a
random tree-based permutation to the Sobol points.

### Generating Samples

`sobol_sample` combines Sobol matrix multiplication with Owen scrambling
to produce a single float in [0, 1):

```mojo
fn sobol_sample(index: Int, dim: Int, seed: UInt32,
                matrices: UnsafePointer[UInt32, MutAnyOrigin]) -> Float32:
    var acc: UInt32 = 0
    var cur = index
    var base = dim * 52
    for bit in range(52):
        if cur & 1 != 0:
            acc ^= matrices[base + bit]
        cur >>= 1
        if cur == 0: break
    var scrambled = fast_owen_scramble(acc, seed)
    return min(Float32(scrambled) * Float32(2.32830643653869628906e-10),
               Float32(0.9999999))
```

The Sobol matrices are a flat array of 52 × 32 `UInt32` values loaded from a
binary file generated at build time. Each bit of the sample index selects
whether to XOR the corresponding matrix row.

### Z-Ordering

The "Z" in Z-Sobol refers to the Morton curve (Z-order curve) used to
map 2D pixel coordinates to a 1D sample index:

```mojo
fn encode_morton2(x: UInt32, y: UInt32) -> UInt64:
    # Interleave bits of x and y using standard bit-spreading masks
    var x64 = UInt64(x)
    x64 = (x64 | (x64 << 16)) & UInt64(0x0000FFFF0000FFFF)
    # ... (continued spreading)
    return x64 | (y64 << 1)
```

This ensures that neighboring pixels use nearby portions of the Sobol sequence,
improving cache coherence during rendering and producing visually smoother
noise patterns.
