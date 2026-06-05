# Sampling

Monte Carlo integration is only as good as the samples that drive it.
Poorly distributed samples produce noisy images; well-distributed ones
converge faster. Gonzales uses two main strategies: piecewise-constant
distributions for light selection, and Z-Sobol sequences for primary
path samples. All sampling code lives in `sampling.mojo`.

## Multiple Importance Sampling

When two sampling strategies both contribute to the same integral, the
**power heuristic** (Veach 1997) combines them optimally:

<!-- <<listing: power_heuristic>> -->

With β = 2 the heuristic reduces variance compared to the balance heuristic
while remaining easy to evaluate.  The weight for strategy *f* is:

```
w(f) = f² / (f² + g²)
```

## Directional Sampling

### Cosine-Weighted Hemisphere

Lambertian surfaces scatter light proportionally to cosine of the angle
from the normal. Sampling directly proportional to that cosine — rather
than uniformly — reduces variance by a factor of π:

<!-- <<listing: sample_cosine_hemisphere>> -->

Malley's method: project a uniform disk sample (radius √u₁, angle 2πu₂)
onto the hemisphere. The vertical component follows automatically from
the unit-sphere constraint.  PDF = cos θ / π.

### GGX Visible Normal Distribution (VNDF)

Rough conductors and dielectrics are modelled with the GGX distribution.
Sampling the *visible* NDF (Heitz 2018) rather than the full NDF halves
variance by only proposing microfacet normals that face the outgoing
direction:

<!-- <<listing: sample_ggx_vndf>> -->

The four-step algorithm:

1. **Stretch** the outgoing direction by (αₓ, αᵧ) to transform the
   anisotropic problem into an isotropic one.
2. **Build a local frame** around the stretched direction using
   Duff et al. 2017 (same branchless formula as `Frame::from_z`).
3. **Sample a disk** on the visible hemisphere, then project onto the
   sphere.
4. **Unstretch** to recover the half-vector in the original frame.

## Piecewise Constant Distributions

Light selection and environment-map importance sampling require drawing
from a discrete distribution defined by a table of weights.  The
implementation builds a CDF and binary-searches it in O(log n).  The 2D
extension samples environment maps via one 1D distribution per row plus a
marginal distribution over rows.

## Z-Sobol Sampler

For primary dimensions (pixel position, lens, BSDF directions), Gonzales
uses a Z-ordered Sobol sequence.  Sobol sequences are *low-discrepancy* —
they fill the sample space more evenly than pseudo-random numbers,
reducing variance without increasing sample count.

### Owen Scrambling

Raw Sobol sequences exhibit visible structure in 2D projections. Owen
scrambling randomises the sequence while preserving its low-discrepancy
properties via bit reversal and hash mixing:

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

### Z-Ordering

The "Z" in Z-Sobol refers to the Morton curve used to map 2D pixel
coordinates to a 1D sample index.  Neighbouring pixels use nearby
Sobol points, improving cache coherence and producing visually smoother
noise patterns.
