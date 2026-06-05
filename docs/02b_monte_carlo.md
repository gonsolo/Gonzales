# Monte Carlo Integration

Path tracing is Monte Carlo integration. Everything else — BVH traversal,
material shaders, MIS — exists to serve one core identity:

```
L(p, ωo) ≈ (1/N) Σ f(p, ωo, ωi) Lᵢ(p, ωi) |cos θᵢ| / p(ωᵢ)
```

This chapter establishes the theory before the implementation chapters fill
in the terms.

## The Rendering Equation

Kajiya (1986) showed that the color of a surface point is:

```
L(p, ωo) = Le(p, ωo) + ∫_Ω f(p, ωo, ωi) Li(p, ωi) |cos θᵢ| dωᵢ
```

- `L(p, ωo)` — radiance leaving point `p` toward the camera direction `ωo`
- `Le(p, ωo)` — self-emission (non-zero for light sources only)
- `f(p, ωo, ωi)` — BSDF: fraction of light from `ωi` scattered toward `ωo`
- `Li(p, ωi)` — incoming radiance from direction `ωi` (which is itself an `L`)
- `|cos θᵢ|` — Lambert's law: light hits at a glancing angle with less power
- `Ω` — the hemisphere above the surface normal

The equation is recursive (`Li` contains another `L`) so it has no closed form.
Monte Carlo is the standard numerical solution.

## The Monte Carlo Estimator

For any integral `I = ∫ f(x) dx`, the estimator

```
Î = (1/N) Σᵢ f(Xᵢ) / p(Xᵢ)     where Xᵢ ~ p
```

is **unbiased**: `E[Î] = I` regardless of how `p` is chosen, as long as
`p(x) > 0` wherever `f(x) ≠ 0`.

Variance decreases as `O(1/N)`, so error decreases as `O(1/√N)`. Doubling
sample count halves the error — the noise every renderer user has seen.

Applied to the rendering equation: each traced ray is one sample. The pixel
color is the average over all samples for that pixel.

## Importance Sampling

Variance is minimized when `p(x) ∝ |f(x)|`. Choosing a distribution that
matches the integrand reduces the noise per sample — effectively doing more
with fewer rays.

In practice, exact proportionality is impossible. Gonzales uses two
approximations:

- **Cosine-weighted hemisphere** for diffuse surfaces: `p(ωᵢ) = cos θᵢ / π`,
  matching the `|cos θᵢ|` term in the rendering equation.
- **GGX VNDF** for rough conductors/dielectrics: `p(ωᵢ) ∝ D(ωh) |ωh·ωo|`,
  matching the microfacet distribution peak.

Both cancel most of the integrand, leaving a near-constant quotient and
dramatically lower variance than uniform hemisphere sampling.

## Multiple Importance Sampling

When a scene has both very bright small lights (best sampled directly) and
very glossy materials (best sampled from the BSDF), neither strategy alone
works well. Veach (1997) proved that combining strategies with the
**power heuristic** achieves near-optimal variance:

<!-- <<listing: power_heuristic>> -->

Weight `w(f) = f² / (f² + g²)` gives full credit to whichever strategy has
the higher PDF at the sampled point. See [04_sampling.md](04_sampling.md)
for the full MIS implementation.

## Russian Roulette

Paths must terminate. Truncating at fixed depth `n` introduces bias (dark
scenes). Russian roulette terminates paths probabilistically and **corrects**
surviving paths to remain unbiased.

At bounce `b`, terminate with probability `q`. If the path survives,
multiply throughput by `1/(1-q)`:

```
E[f / (1-q) · 𝟏(survive)] = f/(1-q) · (1-q) = f  ✓
```

Gonzales chooses `q = max(0.05, 1 - luma(throughput))` — paths that have
already lost most of their energy are most likely to be terminated, which
is where truncation would have caused the least visible error anyway.

## Convergence in Practice

| Samples/pixel | Expected noise (relative) |
|---|---|
| 1 | 1.00 (reference) |
| 4 | 0.50 |
| 16 | 0.25 |
| 64 | 0.125 |
| 256 | 0.0625 |

The O(1/√N) rate is why high sample counts matter and why good importance
sampling — which effectively multiplies the sample count — is so valuable.
Low-discrepancy sequences (Z-Sobol, see [04_sampling.md](04_sampling.md))
achieve O(1/N) convergence in smooth integrands, a substantial practical win.
