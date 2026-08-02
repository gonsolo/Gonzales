# ReSTIR / GRIS Theory: MIS, RIS, Shift Mappings, and the Common Currency Problem

This is the companion theory document `A2_restir_migration_plan.md` refers to.
It exists so that phased engineering work (Phases 0-8) and the open-research
phases (9-10) don't require re-deriving MIS/RIS/GRIS from the literature each
time — read this once, then treat `A2` as the source of truth for
gonzales-specific decisions, file locations, and research status.

Nothing here is gonzales-specific except where noted; it is the general
theory a competent implementation of Phases 0-8 needs, plus the precise
formal statement of the problem Phase 9 is stuck on.

---

## 1. Multiple Importance Sampling (MIS)

**The problem.** Often no single sampling technique matches the integrand
well everywhere. NEE (light sampling) is great for large, nearby lights and
terrible for glossy reflections of small lights; BSDF sampling is the
opposite. Using only one wastes the other's strength; averaging both naively
double-counts the integral and can *increase* variance.

**The fix (Veach 1997).** Sample each technique `i` some number of times,
weight each sample by a **MIS weight** `w_i(x)` chosen so the weights sum to
1 for any `x` that could have come from multiple techniques, and combine:

```
I ≈ Σᵢ (1/nᵢ) Σⱼ w_i(X_ij) f(X_ij) / p_i(X_ij)
```

**Balance heuristic** (provably good, what gonzales uses throughout
`bdpt.mojo` and `shading.mojo`'s NEE+BSDF combination):

```
w_i(x) = nᵢ p_i(x) / Σₖ nₖ p_k(x)
```

Every technique's weight at `x` depends on **every other technique's density
at that same `x`** — not just its own. This is the seed of the "common
currency" requirement: MIS only works if you can evaluate `p_k(x)` for every
technique `k`, at the point `x` a *different* technique produced.

**Power heuristic** (`β > 1` in `w_i ∝ (nᵢp_i)^β`) sharpens the balance
heuristic, further favoring the technique that was actually likely to
produce `x`. Gonzales's VCM uses `β = 2` (see `bdpt.mojo`'s MIS weight code);
ReSTIR literature mostly uses `β = 1` (plain balance heuristic) because RIS
needs it for unbiasedness — see below.

---

## 2. Resampled Importance Sampling (RIS)

**The problem.** Sometimes you can't sample from a good `p(x)` directly —
you can only *evaluate* how good a candidate is (a **target function**
`p̂(x)`, usually proportional to the integrand `f`), and you can cheaply draw
candidates from some other, easier distribution.

**RIS** (Talbot et al. 2005): draw `M` i.i.d. candidates `X_1..X_M` from a
source distribution `p`, compute a **resampling weight** for each,

```
w_i = p̂(X_i) / p(X_i)
```

and pick one candidate `Y`, with probability proportional to its `w_i`. `Y`
approximates a draw from `p̂` without ever needing to normalize or invert
`p̂`. The chosen sample's contribution is corrected by an **unbiased
contribution weight** (UCW):

```
W_Y = (1/p̂(Y)) · (1/M) Σᵢ w_i
```

so that `⟨I⟩ = f(Y) · W_Y` is unbiased for `∫ f dμ`, for any `M ≥ 1`, even
though `p̂` was never normalized.

**Multiple source distributions.** If the `M` candidates come from different
techniques `p_1..p_M` (not all the same `p`), the plain `1/p_i(X_i)` weight
must become an MIS-style weight `m_i(X_i)` (generalizing the balance
heuristic to RIS candidates):

```
w_i = m_i(X_i) · p̂(X_i) / p_i(X_i),      m_i(x) = p_i(x) / Σⱼ p_j(x)
```

**Why this matters for real-time rendering:** `M` candidates can be streamed
through **weighted reservoir sampling** (Chao's algorithm) in O(1) memory —
you never need to store all `M` at once. A **reservoir** holds one running
sample plus enough statistics (`w_sum`, candidate count `m`) to update in
O(1) per new candidate and combine two reservoirs in O(1). This is Phase
0.1's `Reservoir` struct.

---

## 3. Generalized RIS (GRIS) — the foundation of ReSTIR

**The problem RIS alone doesn't solve.** ReSTIR's whole idea is *reuse*:
treat a neighboring pixel's (or the previous frame's) already-resampled
output as one more RIS candidate for the current pixel. But that sample was
drawn for a **different target distribution**, in a **different domain**
(different pixel ⇒ different visibility, different shading point, sometimes
a different-length path). Plain RIS assumes all candidates already live in
the current domain; this doesn't.

**GRIS** (Lin et al. 2022, "Generalized Resampled Importance Sampling:
Foundations of ReSTIR") generalizes RIS to candidates from **different
domains**, connected by a **shift mapping**.

A shift mapping `T_i : Ω_i → Ω` takes a sample `x̄` from a *source* domain
`Ω_i` and deterministically maps it to a candidate `T_i(x̄)` in the *current*
domain `Ω` (e.g., "reconnect this neighbor's found light sample to my own
shading point"). Not every `x̄` is shiftable — a shift can be **undefined**
(occlusion, a failed reconnection, hitting a delta BSDF where continuation
isn't defined) — but where defined, it must be **bijective onto its image**.

Reusing a shifted sample changes its probability density by the **Jacobian
determinant** of the shift, exactly as with a change-of-variables in any
integral:

```
p_{T_i(x̄)}(y) = p_{x̄}(T_i⁻¹(y)) · |∂T_i⁻¹/∂y|
```

GRIS folds this into the resampling weight. For a candidate `X_i` from
domain `i`, shifted into the current domain as `T_i(X_i)`:

```
w_i = m_i(T_i(X_i)) · p̂(T_i(X_i)) · W_{X_i} · |∂T_i/∂X_i|
```

where `W_{X_i}` is `X_i`'s own UCW from *its* domain (this is what lets you
chain reuse across many frames/pixels without re-deriving from scratch each
time), and the **generalized balance heuristic** with **confidence weights**
`c_i` (how many effective samples a candidate represents — Kettunen et al.
2023's connection to Veach's multi-sample MIS) is:

```
m_i(x̄) = cᵢ · p̂←i(x̄) / Σⱼ cⱼ · p̂←j(x̄)
```

`p̂←i(x̄)` is `p̂` evaluated by mapping `x̄` *back* into domain `i` via
`T_i⁻¹` and correcting by the reverse Jacobian — this is the "evaluate every
technique's density at every candidate" requirement from Section 1, now
generalized across domains via shifts.

**This is the actual "common currency" mechanism.** GRIS doesn't require
every technique's raw density to already be expressed in a shared unit — it
requires that, for any candidate found by technique `i`, you can compute
`p̂←j` for every *other* technique `j` at the (shifted) same physical
outcome. Reconnection shifts, random-replay shifts, and VCM's kernel
density estimate all satisfy this. **Section 6 below is about the one
technique gonzales plans to use that doesn't.**

---

## 4. Shift mappings, concretely

Three shift mappings cover essentially everything in the migration plan:

**Reconnection shift.** Deterministically connect the current domain's
prefix to a suffix vertex from the source domain (e.g., ReSTIR DI: keep the
camera ray, reconnect to the source's chosen light point). Requires both
endpoints to be non-delta (`bxdf_is_delta`), or the connection is undefined
— the standard reason ReSTIR DI/GI can't do caustics/SDS on their own
(`A2` §4, Phase 4.3). Jacobian is a ratio of geometry terms:
`|cos θ_y| / ‖x - y‖²` at each end, already used throughout gonzales's
`_nee_area_lights`.

**Random replay shift.** Keep the *random numbers*, not the path, and
retrace from a new starting point — the shifted path is whatever those same
random numbers produce at the new location. Always defined (no bijectivity
failure), but only "local" — it doesn't converge two originally-different
paths back together the way reconnection does. Its Jacobian is the ratio of
consecutive forward/reverse solid-angle sampling PDFs at each replayed
vertex (see ReSTIR BDPT's Eq. 28-29, `A2` §8).

**Half-vector / manifold shift** (Kettunen et al. 2015's gradient-domain
rendering; Hachisuka's manifold exploration). Perturbs a *specular chain*
by holding the half-vector (or another local constraint) fixed and
re-solving via Newton iteration for the neighboring path's specular
vertices, computing the shift's Jacobian from the **implicit function
theorem** applied to the specular constraint. **Gonzales already has the
core of this**: `_mnee_walk`/`_mnee_walk2` (`shading.mojo:1944`, `:1821`)
already do Newton iteration on the half-vector constraint, the
constraint-Jacobian determinant (`det_b`), and the transfer matrix
(`dx1_dxlight`) needed here. This is why `A2` calls SMS "a generalization
of code you have" (§1, fact 5) — Phase 5/6 extend this machinery to N
vertices and random seeding rather than building a manifold shift from
scratch.

---

## 5. VCM's weight derivation — the working template

Before GRIS existed, Georgiev et al. (2012) solved a structurally identical
problem for **vertex connection and merging (VCM)**: unify BDPT's
connection strategies (which have closed-form path-space densities) with
**vertex merging** (photon density estimation, whose "density" is a kernel
estimate, not a path-space PDF) inside one balance-heuristic MIS weight.

The trick: express the merging estimator's density in the **same
area-measure units** BDPT's connection strategies already use. A merge at a
vertex effectively integrates over a disc of radius `r` around it; treating
that disc as "the density with which this vertex would have been sampled"
gives a kernel term proportional to `1/(π r²)`, scaled by how many photons
`N` are available to land in it:

```
η_vcm = π · r² · N
```

This is a real, working example of the "common currency" derivation Phase
9.2 needs for SMS — and it's not hypothetical, it's **implemented and
verified in gonzales today**, at `bdpt.mojo:5352`
(`eta_vcm = PI * merge_r2 * n_light_paths_merge`), feeding directly into
the MIS weight-combination constants (`mis_vm_weight_factor`,
`mis_vc_weight_factor`) used every sample. If you need a template for "how
do I even start deriving a density for a technique that doesn't obviously
have one," this is the answer: **find the region of path space the
technique implicitly integrates over, and express its measure in the same
units the rest of the reservoir uses.**

The reason this template doesn't immediately transfer to SMS (Phase 9.2) is
the subject of Section 6.

---

## 6. The common currency problem, formally stated

To fuse `N` techniques into one reservoir — one joint resampling/MIS
combination — **every technique needs an evaluable density `p_i(x)`, in a
shared measure, at any candidate `x` any technique produced** (Sections 1
and 3). "Evaluable" doesn't require closed-form; VCM's kernel estimate
(Section 5) isn't closed-form in the sense of a formula you write down once
— but it *is* a deterministic function of `(x, r, N)` you can compute
exactly, every time, for the same inputs.

Most techniques gonzales plans to fuse satisfy this:

| Technique | Density | Nature |
|---|---|---|
| Path tracing / BSDF sampling | BSDF-derived solid-angle PDF, converted to area measure via geometry term | Closed-form |
| NEE | Light-sampling PDF (area or solid-angle, by light type) | Closed-form |
| BDPT connections | Product of forward/reverse subpath PDFs (`p_{s,t}`, Veach 1997 Eq. 4-5) | Closed-form |
| VCM merging | Kernel density estimate, `η_vcm = πr²N` | Deterministic given `(r, N)` |
| ReSTIR reconnection/replay shifts | Jacobian of the shift, applied to the source domain's already-known density | Closed-form given the shift |

**SMS breaks this.** Specular Manifold Sampling (Zeltner/Georgiev/Jakob
2020) finds a specular chain by Newton-solving from a random seed. The
"density" of finding that particular solution, `q_SMS(x)`, is the measure
of the **basin of attraction** — the set of seeds whose Newton iteration
converges to that same solution — under the seed distribution. This basin
has no closed form: it depends on the nonlinear specular constraint's local
curvature in a way that generally requires actually running Newton's method
from many seeds to characterize.

SMS sidesteps needing `q_SMS(x)` directly with a **Bernoulli-trial
reciprocal estimator**: re-seed and re-solve until the same solution
recurs; the trial count `T` is geometrically distributed with
`E[T] = 1/q_SMS(x)`, giving an **unbiased estimator of the reciprocal**,
`1/q_SMS`, not of `q_SMS` itself, and not a fixed value — a fresh run
gives a different `T`. This is exactly what a balance-heuristic MIS weight
(Section 1) or a GRIS resampling weight (Section 3) needs *not* to be: a
noisy random variable standing in for one term in a denominator shared with
every other technique's exact density.

This is gonzales's actual Phase 9.2 gap. `A2_restir_migration_plan.md` §9.2
and §5 record, in full technical detail (verified against primary sources),
why the two obvious published toolkits don't close it — **Marginal MIS**
(West/Georgiev/Hachisuka 2022) requires a smooth, evaluable *conditional*
density given an auxiliary variable, which SMS's Newton solve doesn't have
(the conditional is a Dirac delta, not a density); **Misso et al.'s
debiasing framework** (2022) only offers alternative single-quantity
estimators of `1/q_SMS`, never addressing how to combine that estimate with
other techniques' *exact* densities in one joint weight — plus a derived
reduction that was toy-simulated and empirically refuted, and the pragmatic
fallback (bias-bounded plug-in, falling back to a separate reservoir) that
came out of that same simulation. Read `A2` directly for that log; it isn't
duplicated here since it's an ongoing research record, not settled theory.

---

## 7. Where this leaves gonzales

**Phases 0-8** are ordinary engineering against the theory above: reservoirs
(Section 2), shift mappings (Section 4), and GRIS's generalized balance
heuristic (Section 3) are all that's needed, plus the specific file:line
architectural facts `A2` records. None of them require solving Section 6's
problem — they each fuse at most one non-trivial technique with plain path
tracing, the same shape as every published GRIS extension to date (ReSTIR
DI/GI, SMS-ReSTIR, Volumetric ReSTIR, ReSTIR-SSS, ReSTIR BDPT — none of
which cross-combine two non-trivial techniques' reservoirs together
either, per `A2`'s memory-tracked literature survey).

**Phase 9** needs a genuinely new idea for Section 6's problem, or
acceptance of the pragmatic fallback `A2` §9.2 already describes. This
document gives the formal shape of the gap; it does not close it.

**Phase 10** (cost-awareness) is a different axis entirely — not a density
problem, a decision-theoretic one (whether to *invoke* an expensive
technique at all, independent of how its weight is computed once you have).
It doesn't depend on Section 6 being solved; `A2` §10 records why, and that
the mechanism (unlike Phase 9) has already been validated by simulation.
