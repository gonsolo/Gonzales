# A2: ReSTIR / SMS Migration Plan

Plan for moving gonzales from VCM + SPPM + lightweight guiding to a
GRIS/ReSTIR-based sampling framework. **Status: plan only, nothing
implemented.**

Theory (MIS, RIS/GRIS, shift mappings, VCM's weight derivation, the
"common currency" problem) lives in a companion document and is not
repeated here.

---

## 1. Starting point

| Component | File | Lines | State |
|---|---|---|---|
| Unidirectional NEE path tracer | `shading.mojo` | 2856 | Production; shared CPU/GPU |
| BxDF abstraction | `bxdf.mojo` | 697 | `BxDFSample`/`GeomContext`/`BxDFFlags` |
| MNEE | `shading.mojo:1769-2351` | ~580 | 1- and 2-vertex manifold walks |
| VCM (connect + merge) | `bdpt.mojo` | 7180 | Production; real Georgiev MIS, verified vs SmallVCM; CPU/GPU/wavefront/Vulkan-RT |
| SPPM | `sppm.mojo` | 2383 | Production, standalone |
| Path guiding | `guide.mojo` | 169 | Fixed 16³ × 64 bins, no adaptivity |
| GPU wavefront | `gpu.mojo` | 3580 | 18-kernel per-bounce dispatch (`:2731`) |

### Architectural facts (verified by reading the code)

Several of these contradict what a generic ReSTIR plan would assume.

1. **No temporal state at all.** `grep -niE "motion|shutter|animat|keyframe"`
   over `src/` → zero hits. No motion vectors, no history, no reprojection.
   `PathState_C` (`geometry.mojo:410`) has no usable temporal key.

2. **Interactive mode is nearly-free ReSTIR.** `render_interactive`
   (`pipeline.mojo:1232`) accumulates 1 spp/frame and clears the film on
   camera move (`:1358-1361`). Static camera ⇒ temporal reprojection is the
   **identity function**, and film-clear-on-move gives correct reservoir
   invalidation for free. Cheapest correct deployment; target it first.

3. **Final-frame mode must keep working with zero temporal input.** There
   "temporal" reuse means reuse **across spp passes** — purely internal.
   A host app (Blender) may later supply motion vectors, so make
   reprojection a **pluggable strategy**, not hard-coded:

   | Mode | Correspondence source | Phase |
   |---|---|---|
   | Interactive, static camera | identity | 2 |
   | Final-frame, standalone | none (across-spp only) | 3 |
   | Final-frame, host-driven | host motion vectors | 3.5, optional |

4. **Deferred shadow-ray infrastructure exists but is dead.**
   `shade_enqueue_shadow_gpu` (`gpu.mojo:2057`), `traverse_shadow_rays_gpu`
   (`:2112`), `shadow_buf` (`:114`) are allocated but never enqueued; NEE
   shadow rays are traced inline. `shading.mojo`'s NEE functions are already
   parameterized on `[enqueue_shadow: Bool]`. ReSTIR DI needs exactly this —
   revive, don't rewrite.

5. **SMS is a generalization of code you have.** `_mnee_walk`
   (`shading.mojo:1944`) / `_mnee_walk2` (`:1821`) already do Newton
   iteration on the half-vector constraint, the constraint-Jacobian
   determinant (`det_b`), a block-tridiagonal 2-vertex solve, and the
   transfer matrix (`dx1_dxlight`). SMS adds N-vertex chains, random
   seeding, and the Bernoulli estimator.

6. **G-buffer is insufficient.** `gen_aux_buffers_gpu` (`gpu.mojo:2567`)
   gives primary-hit normals, depth, curve mask; no world position, no
   material/instance ID — both needed for shift validity tests. CPU
   `render_aux_buffers` (`bvh.mojo:1860`) writes only normals + depth.

7. **Integrator selection is CLI-only.** The scene's `Integrator "..."`
   name is parsed and **discarded** (`pbrt_parser.mojo:246-254`); only
   `maxdepth`/`radius`/`photonsperiteration` survive. Dispatch is a flat
   `if/elif` ladder (`pipeline.mojo:786-1047`). Add `--restir` alongside
   `--vcm`/`--sppm`.

8. **GPU wavefront runs 8 samples/pixel in flight** (`WAVEFRONT_BATCH = 8`,
   `gpu.mojo:23`; `path_buf[si_local * n_pixels + px_flat]`). Reservoirs are
   per-*pixel* — that mapping is a real design decision (see 2.4).

---

## 2. Validation

**Harness exists:** `compare_bitterli.sh` already renders all six target
scenes at 64 spp against pbrt-v4 (`glass-of-water`, `veach-bidir`,
`volumetric-caustic`, `water-caustic`, `bathroom`, plus 26 others).

**Per-change:** render before/after at matched spp, then compare the diff
**against the renderer's own noise floor** — gonzales is not
bit-deterministic across runs (parallel tiles + adaptive guide grid); the
cornell-box floor at 64-256 spp is mean ≈ 0.0003-0.0005, max ≈ 0.008-0.03.
A refactor is behavior-preserving iff its diff sits at that floor. RNG
reordering is acceptable — verify the diff *shrinks* with more spp.

**Per-phase:** additionally equal-time or equal-quality vs the integrator
being replaced, on the scenes it currently handles best.

---

## 3. SDS paths: the options are not just VCM

Specular-Diffuse-Specular paths (diffuse vertex between two speculars —
`bathroom`'s mirror reflections) defeat NEE and BDPT connection alike, and
are the gap ReSTIR BDPT leaves open. VCM's merge is one answer, not the only:

| Technique | Mechanism | Status |
|---|---|---|
| VCM vertex merging | density estimation at the diffuse vertex | **already have** |
| SPPM | standalone photon density estimation | **already have** |
| SMS | solves the specular chain; abstract states it "samples SDS paths" | Phase 5 |
| SMS-ReSTIR | SMS + reservoir reuse | Phase 6 — **the in-framework answer** |
| Specular Polynomials (TOG 2024) | Newton-free polynomial roots; beats SMS/MPG equal-time, but ~10× slower than Newton **on GPU**, approximate for refraction, 1-bounce-exact | weak fit, see §5.1b |
| Photon-Driven Manifold Sampling | photon-seeded manifold exploration | not planned |
| Manifold Path Guiding | guiding specialized to specular manifolds | not planned |
| Caustics path reuse (Xu et al. CGF 2023) | ReSTIR-style caustics reuse | not planned |

**Consequence:** gonzales already ships two SDS-capable integrators, so it
can never regress on SDS. The only question is which technique eventually
carries that load — most likely Phase 6, not a retained merge term.

---

## 4. Phases

Phases 1-8 each deliver standalone value and do **not** depend on Phase 9.

| # | Goal | Depends on | Kind |
|---|---|---|---|
| 0 | Reservoir + G-buffer infrastructure | — | Engineering |
| 1 | Path guiding upgrade (SD-tree) | — (parallel) | Engineering |
| 2 | ReSTIR DI, interactive | 0 | Engineering |
| 3 | ReSTIR DI, offline (no temporal input) | 2 | Engineering |
| 3.5 | Host-supplied temporal data | 3 | Optional, deferred |
| 4 | ReSTIR GI (path reuse) | 3 | Engineering |
| 5 | SMS (generalize MNEE) | — (parallel from 0) | Eng. + some research |
| 6 | SMS-ReSTIR (manifold shift reservoir) | 4, 5 | Research-flavored |
| 7 | Volumetric ReSTIR → retire SPPM | 4 | Research-flavored |
| 8 | ReSTIR BDPT → retire VCM | 4 | Hard |
| 9 | Common currency: joint reservoir | 6, 7, 8 | **Open research** |
| 10 | Cost-aware weights + throttling | 9 | **Open research** |

### Phase 0 — Infrastructure

New file: `src/gonzales/reservoir.mojo`. No behavior change.

- **0.1** `Reservoir` struct (`sample`, `w_sum`, `m`, `w`) +
  `reservoir_update` (weighted reservoir sampling) + `reservoir_combine`.
  `TrivialRegisterPassable`. If a parametric payload fights the type
  checker, start concrete (`DIReservoir`) and generalize later.
- **0.2** M-cap on `m` (~20× per-frame candidate count) to bound
  correlation under repeated reuse.
- **0.2b** Pluggable reprojection hook (fact #3). Only identity and none
  need implementing now. Cheap now, expensive to retrofit.
- **0.3** Extend G-buffer with world position + material ID, GPU
  (`gen_aux_buffers_gpu`, `gpu.mojo:2567`) and CPU (`render_aux_buffers`,
  `bvh.mojo:1860`). Normals+depth alone accept too many invalid shifts.
- **0.4** Revive the deferred shadow path (fact #4): wire the existing
  kernels into `_gpu_bounce_kernels` (`gpu.mojo:2731`), verify identical
  output to inline NEE.
- **0.5** Plumb a no-op `--restir` flag: `__init__.mojo` (~:129) →
  `parse_and_render` (`pipeline.mojo:724`) → dispatch ladder.

### Phase 1 — Path guiding upgrade (independent track)

Replace the fixed 16³×64 grid with an adaptive **SD-tree** (Müller et al.,
Practical Path Guiding): kd-tree over space, quadtree over direction per
leaf. Keep the existing `guide_sample`/`guide_pdf`/`guide_record` API shape
and the `guide_cell_has_data` gating (`guide.mojo:102`) — its purpose
(don't let an uninformative distribution inflate MIS weights) still holds.
The two-pass driver (`pipeline.mojo:1056-1158`) extends to N iterations.

Validate on `pavillon-night`, `bathroom`, `veach-ajar`; no regression on
cornell-box. Genuinely optional relative to the ReSTIR track, but the
lowest-risk item here.

### Phase 2 — ReSTIR DI, interactive

New file: `src/gonzales/restir_di.mojo`. Reconnection shift only.

- **2.1** Payload: light index, sampled point, light normal, `Le`. Target
  `p̂` = unshadowed contribution (BSDF × G × Le) — visibility deliberately
  excluded, resolved once for the winner.
- **2.2** Initial RIS: M ≈ 8-32 candidates from the existing light sampler
  (`_nee_area_lights:2180`), weighted `p̂/q`, streamed into the reservoir.
- **2.3** Temporal reuse via identity reprojection (fact #2). Clear
  reservoirs where the film is cleared (`pipeline.mojo:1358-1361`).
- **2.4** Resolve the 8-in-flight question (fact #8). Options: (a) slot 0
  only participates; (b) 8 reservoirs/pixel; (c) stream all 8 as extra
  candidates. **Recommend (c)** — closest to plain RIS, no extra storage —
  but verify unbiasedness, since the 8 are correlated in the Sobol
  sequence. Moot in Phase 2 (1 spp/frame); real in Phase 3.
- **2.5** Spatial reuse: k ≈ 3-5 neighbors, reconnection shift (Jacobian
  `|cosθ_y| / ‖x-y‖²`, already used throughout `_nee_area_lights`). Reject
  on G-buffer mismatch (normal dot < ~0.9, depth delta > ~10%, different
  material).
- **2.6** One shadow ray for the winner via the 0.4 path.
- **2.7** **MIS with BSDF sampling.** ReSTIR replaces the light-sampling
  half of NEE; the BSDF half and its `power_heuristic` weighting must still
  combine correctly. Get this wrong and expect systematic energy error.

Exit: matches reference at convergence, beats plain NEE at equal frame
count on many-light scenes.

### Phase 3 — ReSTIR DI, offline

**Hard constraint: must be fully correct with no temporal input.**

- **3.1** Persist reservoirs between `gpu_render_wavefront` batch calls
  (`pipeline.mojo:976-996`). This is the only "temporal" reuse available
  standalone.
- **3.2** **Confidence weighting is now load-bearing.** Unlike interactive
  mode (independent frames averaged by the film), across-pass reuse
  correlates the samples averaged into one image. The M-cap bounds it —
  verify empirically against a non-ReSTIR reference. **Most likely source
  of subtle systematic error in the whole plan.**
- **3.3** CPU path: **recommend GPU-only first.** Tile-parallel structure
  (`rendering.mojo:226`) makes cross-tile spatial reuse awkward for smaller
  payoff.
- **3.4** Resolve 2.4 for real.

Validate with a full `compare_bitterli.sh` run.

### Phase 3.5 — Host-supplied temporal data (optional, deferred)

Strictly additive; Phase 3 must already work without it. Define an external
interface (per-frame camera transform and/or motion-vector buffer) into the
0.2b hook; add reservoir persistence across `parse_and_render` calls (needs
a session handle). Must handle **disocclusion** and **moving
specular/caustic features**, which don't follow diffuse motion vectors —
flagged in ReSTIR BDPT's own limitations for animated caustics. Needs an
animated scene to validate; gonzales has no animation support today.

### Phase 4 — ReSTIR GI

- **4.1** Payload becomes a path suffix: reconnection vertex + accumulated
  suffix radiance + enough state to re-evaluate the BSDF there.
- **4.2** Reconnection shift one bounce deeper; same Jacobian.
- **4.3** **Delta-BSDF rejection** — reconnection is invalid if either
  endpoint is delta (`bxdf_is_delta` already exists). Fall back to the
  canonical sample. This is exactly why ReSTIR GI can't do caustics and why
  Phases 5-6 exist.
- **4.4** Always keep the canonical sample in the combination.

Validate on `bathroom`, `pavillon-night`, `living-room`. **Do not expect
improvement on `glass-of-water`/`veach-bidir`** — out of reconnection's
reach by construction.

Exit: matches or beats the current path tracer at equal time on
indirect-heavy scenes. This is where ReSTIR starts carrying real weight and
later phases become worth their risk.

### Phase 5 — SMS: generalize MNEE

New file: `src/gonzales/sms.mojo`. Independent of the ReSTIR track.

- **5.1** Generalize the manifold walk to N vertices — `_mnee_walk2`'s
  block-tridiagonal solve extends naturally. Keep the existing convergence
  criterion (`max|c| < 1e-3`, 20 iters) and bail-outs.
- **5.1b** *Specular Polynomials* (Fan et al., TOG 2024,
  arXiv:2405.13409, `github.com/mollnn/spoly`) — read in full 2026-07-27.
  Deterministic, Newton-free, beats SMS and Manifold Path Guiding in
  equal-time caustics comparisons, avoids SPPM's blur, derived for
  triangles with interpolated normals. **But a weak fit here**, per its
  own §6: ~10× *slower* than Newton **on GPU** (4.936 vs 0.306 µs,
  two-bounce; the 2.5-3.3× win is CPU-only); refraction uses a first-order
  rational approximation whose recommended fix is *one Newton iteration*;
  accurate for one bounce only; 3+ specular vertices combinatorially
  infeasible. Plausible niche: CPU-side single-bounce reflective
  caustics/glints. The interesting angle is the **hybrid** the paper itself
  suggests — polynomial seed + one Newton refinement — cheap here because
  MNEE's Newton machinery already exists.
- **5.2** Random seeding over the specular surface's sample space (UV for
  well-parameterized meshes, directional otherwise).
- **5.3** Bernoulli-trial reciprocal estimator: re-seed and re-solve until
  the same solution recurs; trial count `t` is an unbiased estimator of
  `1/p`. Needs a solution-equality test. This is what makes SMS unbiased —
  and what makes it expensive.
- **5.4** Keep MNEE as a fast path for the common 1-2 interface case;
  invoke SMS for longer/unknown chains. Decide on measured cost.
- **5.5** MIS with existing NEE — ordinary single-pixel MIS, does **not**
  need Phase 9.

Validate on `glass-of-water`, `water-caustic`, plus
`Scenes/bxdf-mnee-transmit-test.pbrt` and `Scenes/bxdf-smoketest.pbrt`.

### Phase 6 — SMS-ReSTIR

Follows Hong et al., SIGGRAPH Asia 2025 (`github.com/Utah-Graphics-Lab/PSMS-ReSTIR`).

Payload: the found specular chain + light sample. **Manifold shift**:
forward Newton solve at the current pixel seeded from the neighbor's
solution, then a **backward solve** verifying bijectivity within a
uniqueness threshold; reject on failure (`Shift.slang::shiftPathSMS` is the
model). Keep it a **separate reservoir** per the published design — fusing
with DI/GI is Phase 9. Optionally adopt their tile-based sample-space
partitioning, which is what makes it interactive-viable.

### Phase 7 — Volumetric ReSTIR → retire SPPM

Check whether **Ghost ReSTIR** (null-scattering primary sample space,
"ghost vertices" for variable null-vertex counts, targeting SIGGRAPH 2026)
has published in full before implementing the older Volumetric ReSTIR
formulation — it specifically fixes reconnection validity under variable
null-vertex counts. Gonzales already has homogeneous media (`Medium_C`,
`sample_homogeneous_free_flight`, `sample_medium_gpu`). Retire `sppm.mojo`
only after parity on `volumetric-caustic`.

### Phase 8 — ReSTIR BDPT → retire VCM

**Highest-risk phase.**

- **8.1** Bidirectional hybrid shift with technique-aware extended path
  space — the reservoir domain must carry *which strategy* produced a path.
  Concretely (re-verified full text 2026-08-02): pair each path with its
  technique index, `X̂ = (X, τ)`, over `Ω̂ = ∪_τ Ω_τ`; target function
  `p̂(x̂) = ω_τ(x̄)q̂(x̄)`; resampling MIS weight is the generalized balance
  heuristic *with confidence weights* (their Eq. 17,
  `m_i = c_i p̂←i / Σ_j c_j p̂←j`); BDPT's own strategy MIS weight `ω_τ` is
  then recovered cheaply during reconnection via a recursive formulation
  (van Antwerpen 2011, extended here) rather than retracing the whole
  subpath. Reusable machinery even before Phase 9.2 is solved.
- **8.2** Caustics reservoirs. Confidence weights for these must not be
  updated based on whether a caustic sample actually landed on the pixel
  (that would correlate the weight with the realized sample and bias the
  result) — ReSTIR BDPT updates via a *proxy*: the prior frame's
  motion-vector-mapped reservoir weight (their §5.1, Eq. 27). Worth the
  same care if Phase 6's SMS reservoir ever gains temporal reuse.
- **8.3** **ReSTIR BDPT does not cover everything your VCM does.** Its
  Limitations section states SDS paths remain unsolved (suggesting vertex
  merging or manifold shifts), and it has **no participating-media support
  at all** (verified by full-text read).
- **8.4** **Parity checklist before deleting `bdpt.mojo`:** homogeneous
  media, specular chains, hair, measured BRDFs, coated materials — across
  CPU, GPU, GPU-wavefront, and Vulkan-RT. All matched, or explicitly
  accepted as documented regressions.
- **8.5** Plan for a companion SDS technique, not a clean deletion.
  Preference: Phase 6 (SMS-ReSTIR) first — in-framework, and what the
  ReSTIR BDPT authors themselves suggest; residual VCM merge second;
  keeping `--sppm` alive third.

Validate on `veach-bidir` plus an SDS regression check on `bathroom`
against the retained VCM baseline. **Keep `--vcm` working for at least one
release after `--restir` lands.**

### Phase 9 — Common currency (**open research**)

Goal: one reservoir where DI/GI, SMS, volumetric, and bidirectional
candidates compete, so technique selection emerges from resampling weights.

- **9.1** Derive SMS's density in **area-measure units**, the way VCM
  expresses its merge kernel as `η_vcm = π r² N` — already implemented
  correctly at `bdpt.mojo:5352` and the working template.
- **9.2** Handle SMS's density being a **random variable**, not a fixed
  formula. Naive substitution into a balance-heuristic weight biases the
  result. **Checked 2026-07-28, neither published toolkit solves this**
  (see §5): Marginal MIS requires a smooth, evaluable conditional PDF
  `p(x|t)` that SMS's Newton basin-of-attraction doesn't have; Misso et
  al. 2022 only offers alternatives to Bernoulli trials for `1/q_SMS`
  alone, not a way to combine it with other techniques' densities.
  **A candidate derivation was toy-tested (§5) and empirically refuted**:
  reducing `w = p̂/(n_SMS·q_SMS + C)` to Misso et al.'s single-variable
  reciprocal case, fed by a direct hit-rate estimator of `q_SMS` instead
  of Bernoulli trials, is unbiased in principle but has *worse* RMSE than
  doing nothing (biased plug-in) everywhere it was meant to help, and is
  numerically catastrophic near the `q_SMS`-dominant regime.
  **Reframed, pragmatic direction (untested against gonzales, but
  actionable — bounded engineering, not open research):** the same toy
  data shows the *naive* plug-in's bias is tiny and computable in closed
  form (`≈ Var(q̂_SMS)/F³`, verified numerically) whenever `q_SMS` is
  small relative to the rest of the sum — the common case, most pixels
  aren't on a specular chain. So: use the naive plug-in when that bound
  is below a chosen threshold; fall back to Phase 6's separate reservoir
  (no joint weight, already published) when it isn't. Needs: verifying
  the bound holds with gonzales's actual `n_i`/`M` scales, and picking a
  threshold empirically against the validation scenes.
  **Second independent confirmation (re-verified full text 2026-08-02):**
  ReSTIR BDPT's own caustics reservoirs use exactly this strategy for its
  `t≤1` caustic paths — it does not attempt a joint MIS weight against an
  intractable density at all; it sets `f = p̂ = 0` on the "wrong" reservoir
  per path type and simply **sums the two reservoirs' final estimates**
  (their §5.1). Same move as SMS-ReSTIR's separate `Reservoir`, from a
  different paper solving a different technique's density problem — this
  looks like the field's actual consensus practice, not a gonzales-specific
  workaround.
- **9.3** Only then unify the Phase 4/6/7/8 reservoirs into a tagged union.

**Do not start before Phases 4-8 work independently.**

### Phase 10 — Cost-awareness (**open research, but a viable direction —
checked 2026-07-28**)

Weight by contribution-*per-compute*; add a bandit-style layer throttling
*invocation* of low-yield techniques — a correct weight only discards bad
candidates after paying to generate them.

**Literature check:** Kondapaneni et al. 2019, *Optimal MIS*, read in full —
does **not** give cost-aware weighting despite being cited for it. It derives
variance-optimal weights for a *fixed* technique roster and fixed sample
counts (useful for Phase 9's weight math); its own §9 explicitly leaves
"whether some techniques should be included in the mix at all" as future
work. No canonical paper found for bandit-driven per-pixel technique
throttling in rendering specifically — likely genuinely open, not just
unread.

**Toy-simulated the actual mechanism** (many pixels, most gain nothing from
an expensive technique, a rare few gain a lot; compared Always/Never/
Adaptive-two-phase/Oracle at matched total budget). Unlike 9.2, **this one
came back positive**: a budget-safe adaptive policy (spend a small,
budget-*proportional* — not absolute — slice exploring every pixel, then
invest further only where a hit was actually observed) beat both naive
extremes at every budget and every scene sparsity tested (0.5-10% of
pixels benefiting), by up to 9-24x on mean image error, with zero false
positives throughout. Real lesson from a first-pass bug: a *fixed absolute*
exploration budget can itself exceed the total available budget and blow
up the estimator — exploration cost must scale with what's actually
available, not be assumed affordable.

**Honest gap:** even the fixed version never got within an order of
magnitude of Oracle, and missed most true rare-technique pixels outright at
tight budgets (single independent per-pixel trials can't reliably detect a
rare event) — it still won because avoiding waste on the far more numerous
zero-yield pixels mattered more than catching every rare one.

**Spatial pooling, tested (2026-07-28):** the original toy scattered
"special" pixels i.i.d. — no spatial structure to exploit, so this was
untested. Rebuilt the scene as a grid with contiguous caustic-like blobs,
and added a pooling policy: spend exploration probes per *block* instead
of per pixel; if any probe anywhere in a block hits, the whole block
qualifies for extra investment (ReSTIR's own move — propagate evidence
found anywhere in a neighborhood to the whole neighborhood). **Result:
substantially better than independent-per-pixel at every sparsity level
tested (0.5-10%), 1.4-4.2x lower mean image error**, by turning many
individually-unreliable per-pixel exploration decisions into fewer,
far-more-reliable per-block ones (false negatives dropped from 38/49 to
7/49 at 0.5% sparsity, same total exploration budget).

Two real limits, not a clean unconditional win:
- **Block size has a sweet spot, not "bigger is better."** Too small
  (2×2): barely improves on independent, not enough pixels pooled to
  concentrate trials. Too large (16×16): one hit anywhere in the block
  triggers extra investment across the whole thing — false positives
  explode (876/3948 ordinary pixels wrongly flagged, vs. 0 for
  independent). Best result was at block size ≈ the blob size used to
  generate the scene, not a coincidence — mirrors needing to tune
  ReSTIR's spatial-reuse radius to the actual feature scale.
- **Pooling can lose to independent exploration once the budget is
  generous.** At the most generous budget tested, independent alone
  already reached zero false negatives and beat pooling — pooling's
  false positives dilute the shared follow-up budget across pixels that
  don't need it, while independent exploration never has false positives
  to dilute with (ordinary pixels have exactly-zero success probability
  by construction). So the advantage concentrates exactly in the
  tight-budget, sparse-feature regime Phase 10 actually cares about —
  nobody needs throttling once the budget is already generous — but it
  isn't unconditional.

**Net:** unlike 9.2's two dead ends, this mechanism is empirically viable
and worth prototyping for real, with two concrete, honestly-earned
caveats (block-size tuning, budget-regime dependence) a real
implementation would need to navigate. Toy sim (`phase10_toy.py`, scratch
only, not committed to the repo) — see chat history 2026-07-28 to
reproduce.

**Checked 2026-08-02:** Bálint et al., *Forget Superresolution, Sample
Adaptively (when Path Tracing)* (SIGGRAPH 2026), read in full — **does not
bear on Phase 10 despite the title.** It adaptively allocates uniform
path-tracing samples-per-pixel via a learned network (a differentiable
relaxed-stochastic-rounding trick enabling gradients through a discrete
sample-count decision); there is no technique selection, no per-technique
cost model, and no MIS interaction anywhere in it — a strictly different
problem from "should this pixel invoke an expensive technique at all."
Filed as a related-but-orthogonal technique, not a lead.

---

## 5. A recorded dead end

An earlier draft proposed that Specular Polynomials might dissolve Phase
9.2 by enumerating all specular solutions deterministically, making the
density closed-form. **The paper rules this out itself** (§5.1b): accurate
for one bounce only; ≥2 bounces need an eigenvalue solver or unbounded
bisection; ≥3 specular vertices combinatorially infeasible; and it
explicitly recommends *"integrating a stochastic approach ... if one wishes
to guarantee unbiasedness (Zeltner et al. 2020)"* — pointing back at SMS.

The lesson generalizes: **the stochastic-density problem is not an artifact
of Newton being a weak solver.** It is intrinsic to specular chains of
nontrivial length. Don't go looking for a deterministic solver that makes
Phase 9 easy.

Two follow-up reads (2026-07-28), both verified in full text, neither
sufficient — see 9.2 for the precise gap they leave open:

- **Marginal MIS** (West/Georgiev/Hachisuka, SIGGRAPH Asia 2022) generalizes
  MIS to techniques whose density is a marginal over a smooth auxiliary
  variable with a *readily computable conditional PDF*. SMS's seed→root map
  via Newton's method doesn't have one — the conditional given a seed is a
  delta, and the basin measure that would make it a proper density is
  exactly the intractable quantity SMS's Bernoulli trick works around.
- **Misso et al. 2022**, "Unbiased and consistent rendering using biased
  estimators," generalizes reciprocal/exponential debiasing (Taylor or
  telescoping series in a bias parameter `k`) and explicitly notes SMS's
  Bernoulli-trial reciprocal estimator is already a special case of it.
  Their own experiment applying it to specular manifold sampling (their
  Fig. 13) only found alternative single-quantity estimators of `1/q_SMS`,
  performing similarly to or not-yet-competitive with Bernoulli trials —
  it does not address combining that estimate with other techniques'
  densities in a joint weight, which is what Phase 9.2 actually needs.
- **Derived-and-toy-tested reduction (2026-07-28):** since only `q_SMS`
  is noisy in the denominator `F = n_SMS·q_SMS + C` (other techniques'
  densities are exact constants), `g(F) = p̂/F` is a single-variable
  reciprocal, not genuinely multivariate — reducible to Misso et al.'s
  already-solved Eq. 5/6 case, *if* fed an unbiased direct estimator of
  `q_SMS` (a plain hit-rate: fire `M` seeds, count the fraction landing
  in the target root's basin — unbiased for any `M`, unlike Bernoulli
  trials which target `1/q_SMS`). A Python toy simulation (synthetic
  basin probability, no gonzales code involved) tested this: at matched
  sample budget the debiased estimator's RMSE is 2-10x *worse* than the
  naive plug-in when `C` dominates (the realistic regime), and its
  variance is unbounded near `C=0` (the regime it would matter most) —
  same instability the Bernoulli-trial method was specifically designed
  to avoid. **Empirically refuted as a practical estimator.** What it did
  produce: an analytic, numerically-confirmed bias bound for the naive
  plug-in (`≈ Var(q̂_SMS)/F³`) small enough in the `C`-dominant regime to
  motivate the pragmatic bias-bounded-plug-in-plus-fallback direction
  recorded in 9.2.

---

## 6. Cross-cutting

**Mojo:** `def` not `fn`, `comptime` not `alias`, `mut` not `inout`,
`@fieldwise_init`, explicit `Copyable`/`Movable` for structs nested as
fields in other fieldwise-init structs (this bit us on the `LightContext`
split). Use keyword args at construction sites with same-typed fields —
positional args hide transposition bugs from the type checker.

**CPU/GPU sharing:** `shading.mojo` is shared (GPU per-material kernels are
thin wrappers). Put reservoir logic in shared code so it isn't written
twice; the drivers (wavefront loop, tile loop) do need separate work.

**Keep old integrators alive.** `--vcm`, `--sppm`, and the plain path
tracer must keep working throughout; the flat dispatch ladder already
supports this.

**Commits:** one logical step each, render-diff validated before committing.

---

## 7. Risk

- **Phases 0-4** are ordinary engineering with published references.
- **Phase 5** is mostly engineering on existing MNEE machinery, plus one
  new piece (the Bernoulli estimator).
- **Phases 6-8** implement very recent papers (2025-2026); verify primary
  sources before coding.
- **Phases 9-10 are open research.** Phase 9 is plausibly publishable.
- **Largest project risk:** attempting Phase 8 before Phase 4 has proven
  the reservoir infrastructure. VCM works today and is reference-verified;
  replacing it is the one step that can make the renderer worse.

## 8. First session

Phase 0.1 + 0.5: create `reservoir.mojo` with the struct,
`reservoir_update`, `reservoir_combine`, M-capping, and unit tests; plumb a
no-op `--restir` flag. No behavior change, fully verifiable.
