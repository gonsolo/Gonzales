# Sampling

Monte Carlo integration is only as good as the samples that drive it.
Poorly distributed samples produce noisy images; well-distributed ones
converge faster. Gonzales uses two main sampling strategies: piecewise
constant distributions for light selection, and Z-Sobol sequences for
the low-discrepancy samples that drive path tracing.

## Piecewise Constant Distributions

Many rendering decisions require sampling from a discrete or continuous
distribution defined by a table of weights — for example, choosing which
row of an environment map to sample proportionally to its brightness.

The `PiecewiseConstant1D` type builds a CDF (cumulative distribution
function) from an array of non-negative values:

{{snippet:Sources/libgonzales/Core/Distribution1D.swift:cdf-construction}}

The CDF starts at zero and ends at one (or becomes uniform if all values
are zero). Binary search via `findInterval` then locates the correct bin
in O(log n) time.

Continuous sampling maps a uniform random variable [0, 1) to a value in
[0, 1) with probability proportional to the function values:

{{snippet:Sources/libgonzales/Core/Distribution1D.swift:sample-continuous}}

The 2D extension `PiecewiseConstant2D` uses this as a building block:
it constructs one 1D distribution per row plus a marginal distribution
over rows, enabling efficient importance sampling of environment maps.

## Z-Sobol Sampler

For the primary sampling dimensions (pixel position, lens, time, BSDF
directions), gonzales uses a Z-ordered Sobol sequence. Unlike purely
random samples, Sobol sequences are *low-discrepancy* — they fill the
sample space more evenly, reducing variance without increasing sample count.

### Owen Scrambling

Raw Sobol sequences can exhibit visible structure in 2D projections. Owen
scrambling randomizes the sequence while preserving its low-discrepancy
properties. Gonzales implements a fast approximation:

{{snippet:Sources/libgonzales/Sampler/ZSobolSampler.swift:owen-scramble}}

The scrambler reverses the bits of the input, applies a hash mixing
function, and reverses again. This effectively applies a random
tree-based permutation to the Sobol points — the same technique
described in PBRT 4th edition, Section 8.7.

### Generating Samples

The `sobolSample` function combines the Sobol matrix multiplication with
Owen scrambling to produce a single float in [0, 1):

{{snippet:Sources/libgonzales/Sampler/ZSobolSampler.swift:sobol-sample}}

The Sobol matrices themselves are stored as a flattened array of 52 × 32
`UInt32` values (the `sobolDataAccessor`), accessed directly without
intermediate allocations. Each bit of the sample index selects whether to
XOR the corresponding matrix row, building the result one bit at a time.

### Z-Ordering

The "Z" in Z-Sobol refers to the Morton curve (Z-order curve) used to
map 2D pixel coordinates to a 1D sample index. This ensures that
neighboring pixels in the image use nearby portions of the Sobol sequence,
which improves cache coherence during rendering and produces visually
smoother noise patterns.
