# A2: ReSTIR / SMS Migration Plan

Plan for moving gonzales from VCM + SPPM + lightweight guiding to a
GRIS/ReSTIR-based sampling framework.

**Status (2026-09-09): Phases 0-5 done; Phase 6 core done (`--sms-restir`,
temporal-only reuse — its spatial half runs but finds mostly-empty
neighbour reservoirs, so no win yet); Phase 7 machinery complete (light RIS
at volume scatter vertices, distance resampling **on by default**, temporal
reuse opt-in via `--vol-restir-reuse`, spatial reuse measured and shipped
disabled); Phase 8.1's MIS-weight recovery derived and machine-checked
(`restir_bdpt.mojo`, `Tests/unit/test_restir_bdpt_mis.mojo`) with the rest
of Phase 8 not started; Phases 9-10 open research.**

**Neither SPPM nor VCM has been retired, and completing Phases 7 and 8 does
not by itself permit it** — those are separate coverage decisions, written
as explicit gates in §4a. Read that before concluding a phase's completion
means an integrator can go.

A caution learned the hard way (2026-09-09): a phase's *machinery* being
done, an integrator being *replaceable*, and a measured *variance win* are
three different claims, and this document previously blurred all three.
Several recorded figures also turned out to be measured against baselines
that were themselves buggy — always re-measure after a correctness fix
lands rather than trusting a number recorded earlier in this file.

Theory (MIS, RIS/GRIS, shift mappings, VCM's weight derivation, the
"common currency" problem) lives in a companion document,
[`A3_restir_theory.md`](A3_restir_theory.md), and is not repeated here.

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

> **Correction (2026-09-09): "ships it" is not "it works", and this
> paragraph was measured false.** On `water-caustic` — surface SDS, no
> media — the caustic web is produced by **SPPM only**. Plain PT scores
> 0.05× the reference and the entire submerged volume renders **black**.
>
> Two clarifications the table above invites the reader to get wrong:
>
> 1. **SMS/MNEE is not interactive-gated.** `_mnee_area_light_contribute`
>    runs unconditionally in the batch path tracer's NEE (disabling it
>    changes 95% of pixels on `water-caustic`). `--sms-restir` is a
>    *reuse* layer on top, and its interactive-only gate is correct: a
>    lone candidate with no reservoir combine is algebraically identical
>    to per-frame MNEE. So "SMS-ReSTIR is the in-framework answer" is
>    about *variance*, while the SDS *capability* is plain MNEE/SMS, which
>    is always on.
> 2. **A failed manifold solve is worse than no MNEE at all.** MNEE
>    suppresses the straight shadow ray as soon as it finds a dielectric
>    (correct — glass occludes it), then returns without contributing if
>    the Newton solve does not converge. On a wavy water surface that path
>    is taken almost everywhere, so the receiver goes black rather than
>    merely noisy. Verified by painting failed solves magenta: the whole
>    submerged volume lights up.
>
> So the honest statement is: **SDS coverage in batch mode rests on SPPM
> today.** Phase 6 cannot take that load until the underlying solver finds
> the multiple admissible refraction points a caustic web is made of —
> reuse cannot rescue candidates that never converge. See §4a Gate S.

---

## 4. Phases

Phases 1-8 each deliver standalone value and do **not** depend on Phase 9.

| # | Goal | Depends on | Kind | Status |
|---|---|---|---|---|
| 0 | Reservoir + G-buffer infrastructure | — | Engineering | Done |
| 1 | Path guiding upgrade (SD-tree) | — (parallel) | Engineering | Done (`guide.mojo`) |
| 2 | ReSTIR DI, interactive | 0 | Engineering | Done |
| 3 | ReSTIR DI, offline (no temporal input) | 2 | Engineering | Done |
| 3.5 | Host-supplied temporal data | 3 | Optional, deferred | Not started |
| 4 | ReSTIR GI (path reuse) | 3 | Engineering | Done (diffuse x1/x2 only) |
| 5 | SMS (generalize MNEE) | — (parallel from 0) | Eng. + some research | Done (`sms.mojo`) |
| 6 | SMS-ReSTIR (manifold shift reservoir) | 4, 5 | Research-flavored | Core done, `--sms-restir` (temporal only, no spatial) |
| 7 | Volumetric ReSTIR machinery (SPPM retirement is gated separately — §4a) | 4 | Research-flavored | Done: light RIS (7.1/7.2) wired CPU+GPU; distance resampling (default **on**); temporal reuse (opt-in, `--vol-restir-reuse`); spatial reuse measured as no consistent win, shipped disabled |
| 8 | ReSTIR BDPT machinery (VCM retirement is gated separately — §4a) | 4 | Hard | 8.1 MIS-weight recovery derived from gonzales's own dVCM/dVC and machine-checked (`restir_bdpt.mojo`); reconnection Jacobian + reservoir plumbing not started; 8.2/8.3 not started |
| 9 | Common currency: joint reservoir | 6, 7, 8 | **Open research** | Investigated, not implemented |
| 10 | Cost-aware weights + throttling | 9 | **Open research** | Investigated, not implemented |

## 4a. Retirement gates (added 2026-09-09)

Phases 7 and 8 were originally titled "→ retire SPPM" and "→ retire VCM".
That framing was **misleading and cost real time**: it repeatedly led both
the author and later contributors to read "Phase 7 complete" as "SPPM can
go", when the two are not the same claim. What those phases deliver is
*resampling machinery*. Retiring an integrator is a separate decision, and
it is gated on **coverage** — whether every effect the old integrator
uniquely reaches has an in-framework substitute — not on the machinery
existing.

The plan already said as much in two places, easy to miss: §3's closing
line ("gonzales already ships two SDS-capable integrators, so it can never
regress on SDS"), and §8.3's admission that ReSTIR BDPT leaves SDS
unsolved and has **no participating-media support at all**.

State the gates explicitly, so they can be checked rather than assumed:

**Gate S — retire SPPM.** Measured 2026-09-09 (`Scenes/caustic_presence_check.py`);
**all four conditions currently FAIL.** The metric is scale-free —
luminance ÷ its own row median along the caustic's trajectory — because the
Tungsten references carry a scene-conversion scale (real pbrt-v4 is equally
1.8× off on `glass-of-water`, so absolute error cannot answer "is the
caustic there"). Reference scores 5.53; a caustic-free render 1.54.

1. `volumetric-caustic` scores excess > 2.0 **without** `--sppm`, with fog
   and glass sphere visibly rendered. *Today: still no integrator passes* —
   PT scores 1.54 (absent, 2.0× too bright overall) and VCM did not finish
   in 40 min. **This condition is about retiring SPPM, so SPPM's own score
   does not satisfy it** — but the "SPPM renders an almost-empty dark box"
   half of the original note was a bug in SPPM, now fixed (2026-09-09,
   commit `4c1a5c1e`): it scores **2.11 (CAUSTIC PRESENT)** with fog, sphere
   and beam rendered and a mean within 5% of the reference. Four independent
   defects were involved — analytic spheres invisible to SPPM's traversals
   entirely, a placeholder sphere normal, a dropped `hit` argument, and
   volume visible points receiving no direct lighting through three separate
   gates — plus a volume photon-density estimator normalising by disk area
   instead of sphere volume. See `project_sppm.md`.

   Residual, not closed: gonzales scores 2.11 against real pbrt-v4's 3.96 and
   the Tungsten reference's 5.53 on the same file, so the beam is present but
   materially less concentrated. It does not improve with sample budget
   (2.03/2.06/2.11 at 8/16/32 passes), so it is systematic.
2. `water-caustic` scores > 2.0 without `--sppm`, **in batch mode**.
   *Today: only SPPM passes* (present, structurally correct, 5.4× bright).
   PT and `--sms-restir` both score 0.05× the reference — caustic absent.
   The blocker is **MNEE's Newton solve failing below the waterline**, not
   the `--sms-restir` gate (see the bullets below); the submerged volume
   renders black because a failed solve contributes nothing while still
   suppressing the straight shadow ray.
3. Both within a stated tolerance of **real pbrt-v4 on the same file** —
   pbrt is the arbiter, not the Tungsten `.exr`.
4. The replacement finishes in time comparable to SPPM (< ~560 s for
   `water-caustic`). *Today: VCM could not, on an idle GPU.*

**Attempted and REFUTED (2026-09-09): multi-root SMS re-probe for mesh
casters.** Since a typical submerged point on `water-caustic` has **≥8**
distinct specular solutions (instrumented: 843,579 points at the counter's
cap of 8, single-root points rare) while the mesh path takes exactly one at
weight 1.0, the obvious fix was a re-probe proposal — perturb the shadow
direction, re-traverse, rebuild the vertex from the surface actually hit
(which does fix the wrong-plane problem that tangent-plane jitter has) —
plus reciprocal counting. The cone half-angle was *derived*, not tuned:
displacing the specular vertex by arc length `d` rotates the required
normal at `|dn*/dd| ≈ (1/r_i + 1/r_o)/2`, and a physical surface supplies a
normal tilted at most π/2 from the view direction, confining roots to
`d_max = π·r_i·r_o/(r_i+r_o)`, i.e. `θ = π·r_o/(r_i+r_o)`.

**The covering sweep disqualified it.** An unbiased estimator must be
invariant to widening the cone — a wider cone lowers each root's hit
probability and raises the trial count to compensate exactly. Measured
submerged-band ratio vs the reference:

| widen | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|
| ratio | 0.482 | 0.109 | 0.0255 | 0.0151 | 0.0145 |

It **falls 33×** and flattens low; it is also sample-count dependent
(0.0255 at 32 spp → 0.153 at 128 spp), which is disqualifying on its own.
Mechanism, measured with a trial-count-as-radiance diagnostic (mean 2.7,
p99 19, max 72): the cap is not binding on average but truncates exactly
the rare filament roots whose proposal probability is smallest and
contribution largest, and widening makes more roots rare. The heavy tail
lives in `f(X*)·T` — this emitter is `L = 541126` and the geometric term
spikes — not in transmittance. At the brightest setting the submerged
volume was much brighter than baseline but had **no web structure at all**:
a broad diffuse wash under salt-and-pepper noise. Energy recovered, feature
not. Cost was ~60× baseline. Implementation preserved at commit `bf3fd541`
on `worktree-agent-aba36ff2b7a0cd103`, then reverted; nothing shipped.

**Consequence for Gate S:** deterministic root-finding plus reciprocal
counting is not the route to batch SDS on this scene class. A viable fix
must attack the contribution's heavy tail directly — i.e. photon density
estimation, which is precisely what SPPM does and why it remains the only
integrator that renders this caustic. That materially strengthens the case
for keeping SPPM rather than retiring it.

Three findings from that measurement change the picture:

- **`--sms-restir` is interactive-only — but that is NOT why batch mode
  lacks the caustic** (corrected 2026-09-09, after the flag was initially
  and wrongly blamed). SMS/MNEE glass-caustic probing runs **unconditionally
  in batch**: `_mnee_area_light_contribute` is called from
  `shading.mojo`'s plain NEE path with no flag guarding it, and disabling it
  changes 95% of pixels on `water-caustic`. `--sms-restir` only swaps that
  call for `sms_temporal_step`, a *reuse* layer. Its gate is sound and
  documented at `pipeline.mojo`'s `use_sms_restir` parameter: a lone SMS
  candidate with no reservoir combine is mathematically identical to
  per-frame MNEE (`W` collapses to exactly `inv_pdf_area*trials`), so batch
  wiring would add nothing. **Wiring it would not fix `water-caustic`:
  reuse cannot rescue a candidate that never converges** — see the next
  bullet.
- **The real cause of `water-caustic`'s missing caustic: the Newton solve
  fails essentially everywhere below the waterline.** MNEE detects the
  water dielectric and correctly suppresses the straight shadow ray (glass
  occludes it either way), then `shading.mojo`'s `if not solve_ok: return
  True` returns having contributed **nothing** — so a failed solve renders
  *black*, not merely noisy. Diagnostic: painting failed solves magenta
  turns the entire submerged volume magenta, matching exactly the region
  that should carry the caustic web. Scene is a tiny mesh area light above
  a wavy `dielectric` plymesh (`Mesh001.ply`, eta 1.8) — many admissible
  refraction points per shading point, which is what makes the web, and
  what a single Newton solve per (point, light-sample) pair does not find.
  This is the actual SDS gap, and it is in the solver, not the flag.
- **SPPM is broken on `volumetric-caustic`**, despite having medium code
  (`sample_homogeneous_free_flight`, `sppm.mojo:571`/`:993`) — likely the
  camera path not starting inside the enclosing medium
  (`MediumInterface "gas" ""` on the FrontWall). A failure, not a gap.
- Therefore **volumetric caustics have no working producer at all**. That
  is a live capability gap, independent of any retirement decision, and
  outranks retirement as work.

Also: **SPPM had zero smoketest coverage** (`SMOKE_MODES` = cpu-pt,
cpu-vcm, gpu-pt, gpu-vcm, gpu-vcm-wf), which is why the above went
unnoticed. **Fixed 2026-09-09** (`4c1a5c1e`): `SMOKE_MODES` gained
`cpu-sppm` and `gpu-sppm` rows (both mean 0.11571, exact CPU/GPU
agreement), and a new `make causticstest` target runs the scale-free
presence metric on `volumetric-caustic` — because a mean-based smoketest
structurally cannot catch a missing caustic (an unlit box still has a
plausible mean, and cornell-box has neither media nor an analytic sphere).
It needs those rows far more than it needs deleting. Cost of keeping
it, for the other side of the ledger: 1995 lines, plus 620 lines exiled
into `bdpt.mojo` (6440–7060) by the per-file Mojo `enqueue_function`
defect, 11 referencing files, one 386-line test.

**Gate V — retire VCM.** All of Gate S, plus §8.3's own limitations
resolved: SDS paths covered by something (Phase 6, per §3's table), and
participating media handled, which ReSTIR BDPT does not do.

Until a gate is met, "Phase N is complete" means the machinery shipped —
never that the integrator can be removed.

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

**Scope, measured 2026-09-09.** `--sms-restir` is interactive-only, and
that gate is **correct, not a shortfall**: it reuses SMS candidates across
frames, and a lone candidate with no reservoir combine is algebraically
identical to the per-frame MNEE that already runs unconditionally in batch
(`W` collapses to exactly `inv_pdf_area*trials`; see `pipeline.mojo`'s
`use_sms_restir` parameter comment). Batch wiring would therefore be a
no-op, and was deliberately not done.

**What this phase does NOT deliver, and cannot until the solver improves:**
SDS *capability*. On `water-caustic` the underlying MNEE solve fails
essentially everywhere below the waterline, and because a failed solve
suppresses the straight shadow ray without contributing, the submerged
volume renders black. Reuse cannot rescue a candidate that never
converges, so no amount of Phase 6 work will make that scene pass Gate S
condition 2. The prerequisite is a solver that finds the *multiple*
admissible refraction points a caustic web consists of — one Newton solve
per (shading point, light sample) pair finds at most one. Hong et al.'s
tile-based partitioning is about interactive viability, not about this.

### Phase 7 — Volumetric ReSTIR machinery

*(Originally titled "→ retire SPPM". Retiring SPPM is Gate S in §4a, not a
deliverable of this phase — see that section for why the rename.)*

**Ghost ReSTIR check RESOLVED (2026-09-06): it published, but only as a
SIGGRAPH 2026 *Poster* — a two-page extended abstract, no full paper, no
supplemental, no code.** "Ghost ReSTIR: Volumetric Resampling with Ghost
Vertices in Null-Scattering Space", Zhang, Lin, Hong, Kettunen, Yuksel and
Wyman, presented 19 July 2026 (ACM DL 10.1145/3799825.3818719; Chris
Wyman's own publication page categorises it as Poster with only a PDF
abstract). The gate below said "published in full", and it has not been —
so **implement the older, fully-published Volumetric ReSTIR formulation**,
but architect for ghost vertices, because the abstract already gives the
shape of the fix even without the derivation:

- formulated in **null-scattering primary sample space**;
- **target function containing no intermediate transmittance**, so it
  evaluates consistently across all resampling stages (the older
  formulation's inconsistent-transmittance problem is precisely what this
  fixes, along with poor scaling in volume resolution);
- **ghost vertices** = auxiliary infinite tails appended to each path
  segment, giving a well-defined **bijection for reconnection shifts
  regardless of null-vertex count or density change**.

Concretely that means: keep transmittance evaluation behind one seam so the
target function can later drop intermediate transmittance, and keep the
reconnection shift's domain mapping pluggable rather than assuming a fixed
null-vertex count. Expect a full paper to follow — that author list is the
core GRIS/ReSTIR group — so treat the older formulation as scaffold rather
than sinking effort into its internals.

**Sequencing note:** Volumetric ReSTIR's payoff is largest on heterogeneous
and sparse volumes, but gonzales has homogeneous media plus `uniformgrid`
only; NanoVDB/OpenVDB is Gap 7 in `A1_feature_status.md`. So VDB support
partly gates this phase's *validation surface* — `volumetric-caustic` still
works as a test case, but `disney-cloud`-class scenes cannot be used to
exercise it until VDB lands. Consider pulling VDB earlier than its priority
suggests if Phase 7 is to be properly validated.

**Implemented (2026-09-08):** the RIS candidate side is done and wired into
the GPU wavefront medium sampler -- `restir_vol.mojo` (`VolReservoir`
payload, `vol_target_pdf`, `VOL_RIS_CANDIDATES = 8`) and its call site in
`gpu.mojo`'s `_sample_medium_core` (~:2320-2400). A volume scattering vertex
draws `VOL_RIS_CANDIDATES` (light, point) pairs, streams them through a
single-frame reservoir via `reservoir_update`, and resolves the winner with
one shadow ray + transmittance march. With `VOL_RIS_CANDIDATES = 1` this
reduces exactly to the pre-ReSTIR single-sample estimator (`W = 1/q`), kept
deliberately as the cheapest available correctness check.

**Not yet wired:** `vol_temporal_spatial_combine` -- temporal reuse via
identity reprojection plus spatial reuse over `VOL_SPATIAL_NEIGHBORS = 4`
neighbours, with Bitterli et al. 2020 Algorithm 6 Z-normalization -- is
fully implemented and unit-tested (`Tests/unit/test_restir_vol.mojo`), but
has no caller in `gpu.mojo`/`pipeline.mojo`: no persistent per-pixel
`VolReservoirIO` buffers are allocated, and no depth/world-position
G-buffer data is threaded to it. Wiring this in (persistent reservoir
buffers across wavefront batches, the G-buffer plumbing, and a call site
replacing the current single-frame resolve) is the remaining Phase 7 work.

**2026-09-08: TEMPORAL reuse shipped, CPU and GPU (commits 1685154c,
21c3cfb2), spatial still deferred.** `--vol-restir-reuse` (off by default)
wires `vol_temporal_spatial_combine` into `_sample_medium_core` via a new
`restir_vol_a_buf`/`restir_vol_b_buf` reservoir pair on `GpuSceneHandle`
(GPU) and a matching ping-ponged pair in `render_interactive`'s CPU branch
(`rendering.mojo`'s `render_tile`/`render_all_tiles`), both mirroring DI's
own reservoir-pair pattern exactly. Batch `--gpu` joins `--restir`'s
existing dispatch-mode switch (1 sample/pixel/dispatch) to get real
cross-sample persistence; CPU persistence only exists in
`render_interactive` (`--interactive-frames`), matching `--restir`'s own
CPU scope — plain CPU batch rendering has no reuse either.

**2026-09-08: spatial reuse wired and ACTUALLY MEASURED (commit
ee9e4370), not just deferred by DI analogy.** G-buffer pointers
(CPU: `depth_int`/`world_pos_int`; GPU: the same `atrous_depth_buf`/
`gbuf_worldpos_buf` DI's own spatial reuse already uses) are now wired
through, so `vol_temporal_spatial_combine`'s spatial pass is live
whenever `VOL_SPATIAL_NEIGHBORS > 0`. Matched-cap measurement (5 seeds,
`Scenes/vol-restir-mesh-light.pbrt`, same methodology as DI's own
verdict): both temporal-only and temporal+spatial are unbiased (mean
within ~0.15% of a 16384spp reference either way), but spatial shows no
consistent variance win — average MSE across seeds is a near-wash
(slightly worse on average, highly variable per seed: 76% worse to 54%
better depending on seed). Shipped `VOL_SPATIAL_NEIGHBORS=0` (disabled)
— DI's exact verdict, now independently confirmed for the volumetric
case rather than assumed. See the `project_restir_migration` memory's
"Spatial reuse for volumes" section for the full numbers and a real
git-workflow lesson from verifying this on a working directory another
concurrent session was also committing to.

Verified on a new test scene, `Scenes/vol-restir-mesh-light.pbrt`: GPU's
flag-on MSE is 3–10x lower than flag-off's against a 16384spp reference,
averaged over 5 seeds, no systematic bias. CPU verification caught and
fixed a REAL bug present in the original GPU-only commit too (not
CPU-specific in its root cause, just not exposed by that commit's
verification methodology): a single path can scatter many times inside a
dense medium within one frame, and each scatter independently called the
combine and overwrote the persisted reservoir, so only the last one
survived — a "stalled convergence = bias" pattern, fixed by only letting a
path's first in-frame scatter touch the persisted reservoir. Post-fix, CPU
shows the same qualitative pattern as GPU (unbiased mean, real if more
modest variance win) and CPU/GPU agree with each other within ~0.6% at
matched frame counts. `make smoketest` unaffected (flag defaults off).
See the `project_restir_migration` memory's "7.3 (temporal reuse)" section
for the full verification writeup and two real debugging lessons worth
reading before touching this code again: (1) the *existing*
`Scenes/vcm-media-sphere-light.pbrt` test scene turned out to exercise a
completely different, sphere-native light code path and was useless for
verifying this feature — the new mesh-light scene exists because of that;
(2) `--seed` has no effect on CPU interactive rendering, so verifying
CPU-side reservoir work needs a convergence-rate check (low frame count
vs. high) rather than a multi-seed MSE average.

**2026-09-09: distance resampling IMPLEMENTED and measured — Phase 7 is
now feature-complete.** Shipped behind `VOL_RIS_DISTANCE`
(`restir_vol.mojo`), default OFF; homogeneous, achromatic media only.

Two earlier passes concluded this was blocked. Both were reasoning about
the wrong proposal. The obstruction they hit — delta tracking's accepted
distance has no closed-form marginal density — only bites if you try to
use *that* distribution as the RIS proposal, or approximate it with a
majorant-rate exponential and then correct for the mismatch. Draw the
candidates from the **exact conditional collision density** instead,

```
q(t) = sigma_t e^{-sigma_t t} / (1 - e^{-sigma_t t_surf})
```

(analytic for a homogeneous medium), take the target
`p_hat(t,y) = q(t) c_hat(t,y)` with `c_hat` the unshadowed target 7.2
already uses, and `q(t)` cancels out of both the RIS weight and the
resolve. Nothing is left to correct: no transmittance march, no
`1/P(collided)` divisor, no throughput correction. The weight formula and
resolve are unchanged from 7.2; the only difference is that each
candidate evaluates its target at its own vertex and the winner's vertex
is what gets shadowed.

That matters beyond simplicity, because the previously-recorded design
(majorant proposal + `h(t) = T(0,t)/P(collided)`, two ratio-tracking
marches per scatter event) is not merely costlier — it is **biased** for
heterogeneous media. `P(collided) = 1 - T(0,t_surf)` has no closed form
there, so `T` must be estimated stochastically, and the estimator then
divides by `1 - T_hat`; since `x -> 1/(1-x)` is strictly convex, Jensen
gives `E[1/(1-T_hat)] > 1/(1-E[T_hat])` — a systematic over-estimate,
worst exactly where the medium is optically thin. Choosing the exact
conditional removes the offending factors rather than estimating them.

**2026-09-09, later the same day: the two now COMPOSE, everything was
re-measured after the sphere-boundary shadow-ray fix, and the conclusions
changed. `VOL_RIS_DISTANCE` now defaults ON; `--vol-restir-reuse` stays
opt-in.**

Every Phase 7 measurement before that fix is void. A shadow ray leaving a
sphere-bounded medium was Beer-Lambert'd across the vacuum all the way to
the light, because the exit-point search walked the mesh/curve BVH and
never tested analytic spheres — and *both* volumetric test scenes bound
their medium with `Shape "sphere"`. The dense scene's mean was 5.7e-5
where it should be 0.0302. Ratios measured against a baseline 574x too
dark cannot be rescued by reinterpretation.

Re-measured (64x64 GPU, `--no-denoise`, MSE at a matched 64spp budget
over 5 seeds, against **65536spp** references):

| config | thin fog MSE | vs 7.2 | dense MSE | vs 7.2 |
|---|---|---|---|---|
| 7.2 baseline | 7.040e-3 | — | 8.412e-5 | — |
| temporal only | 7.245e-3 | +2.9% | 8.355e-5 | −0.7% |
| distance only | 5.664e-3 | **−19.5%** | 8.250e-5 | **−1.9%** |
| both | 5.927e-3 | −15.8% | 8.686e-5 | +3.3% |

All four are unbiased: 4096spp ratios to reference land in 0.99936 ..
1.00076 across both scenes. The stronger check is that the two
independent 65536spp references — the plain 7.2 estimator and the
distance-resampled one — agree to 0.018% (thin) and 0.006% (dense); two
structurally different estimators converging to the same answer is much
harder to fake than either matching itself.

So: **distance resampling survived re-measurement and is now on by
default.** **Temporal reuse's 3-10x win did not survive** — it is a wash
on dense and mildly worse on thin. It stays wired, correct and opt-in
(a genuinely coherent interactive sequence is a different regime from
these batch runs), but nothing currently justifies defaulting it on, and
its tuning constants should be re-derived rather than trusted. The
combination is unbiased but slightly worse than distance alone on both
scenes, so composing them is now *permitted* rather than *required*.

### How they compose, and the bias the old guard was hiding

The `not dist_ris` guard that made the two mutually exclusive rested on a
premise that was backwards. Its comment said `VolShiftMode.identity`
"re-targets a previous frame's sample at THIS pixel's vertex", so
distance resampling would break it. `identity` does the opposite: it
returns the **donor's** vertex verbatim, and the combine then stores it
as the winner's. So the configuration the guard *permitted* was the
broken one and the configuration it *forbade* was the sound one.

Concretely, with distance resampling off and temporal reuse on:
`reservoir_finalize` sets `W = w_sum / (m · p̂(winner))` with `p̂`
evaluated at `res.scatter_point` — the donor's vertex when a donor won —
while the resolve in `_sample_medium_core` shadow-rayed from
`scatter_pt_s`, this frame's own vertex. `F` and `W` were evaluated at
different points, which is not a valid RIS estimator. It stayed invisible
because both points lie on the same camera ray in a homogeneous medium,
so the two targets are close and the error is a quiet scale factor with
no visual signature.

The fix is one line at the resolve — always shadow from
`res.scatter_point`, the point `p̂` was evaluated at — plus choosing the
shift by what actually produced the vertex:

- **distance resampling ON → `identity`.** The vertex came from
  `q(t)`, which depends only on `sigma_t` and `t_surf`, identical across
  frames at a pixel. A donor's vertex is a draw from precisely the
  receiver's own proposal: domains match, Jacobian 1. This is the
  *better*-founded case, not the broken one.
- **distance resampling OFF → `retarget`** (new mode). The vertex is
  delta tracking's single `t_free`, a point mass that differs every
  frame, so the receiver's proposal could never have produced the
  donor's. Import only the light sample and keep our own vertex — plain
  ReSTIR DI reuse, where the shading point is fixed by the pixel.

`retarget` also keeps the receiver's `sigma_s`/`phase_g`, since those
describe the vertex rather than the light sample and differ within a
heterogeneous medium (the `medium_idx` gate only guarantees the same
medium, not the same density in it).

Heterogeneous media remain out of scope but are no longer *blocked*: the
same construction works there if each candidate's distance comes from its
own exact conditional, i.e. an independent delta-tracking walk
rejection-conditioned on collision — unbiased, still no marches, but
~M walks per segment in expectation. Worth paying only if the cheap
homogeneous case proves valuable in practice.

Gonzales already has homogeneous media (`Medium_C`,
`sample_homogeneous_free_flight`, `sample_medium_gpu`). Retire `sppm.mojo`
only after parity on `volumetric-caustic`.

### Phase 8 — ReSTIR BDPT machinery

*(Originally titled "→ retire VCM". Retiring VCM is Gate V in §4a. Note
8.3 below states plainly that ReSTIR BDPT does not cover everything VCM
does, so this phase cannot on its own justify the removal.)*

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

**2026-09-09: 8.1's derivation is DONE and machine-checked; media scoped
OUT.** The blocker below ("8.1's own recursive formulation isn't derived
yet") is resolved — see "8.1 derivation" immediately after this note. The
other two items (media, Phase 6 spatial reuse) stand.

**2026-09-08: scoped, not yet tractable for a file-level implementation
plan.** Not blocked on Phase 9 (the dependency runs the other way), but
bundles three separate open items: 8.1's own "recursive formulation (van
Antwerpen 2011, extended here)" isn't derived yet (no local copy of the
paper found; gonzales's existing `dVCM`/`dVC` machinery is a genuine
template to extend, unlike Phase 9.2's SMS density, which has no closed
form at all); correct bidirectional MIS through participating media
doesn't exist yet even in gonzales's own VCM (`_bdpt_vertex_mis_scoped`,
bdpt.mojo:4909, excludes every volume vertex today — "a known-hard
problem, research not porting", per the project_restir_migration
memory), and base ReSTIR BDPT has zero media support to inherit instead;
and retiring VCM also needs Phase 6's spatial reuse to start paying off,
which it currently doesn't. See the `project_restir_migration` memory's
"Phase 8" section for the full scoping writeup and a recommended
resolution ordering.

#### 8.1 derivation — recovering `ω_τ` in O(1), and why it survives a shift

Derived 2026-09-09 from gonzales's own `dVCM`/`dVC` recursion rather than
from van Antwerpen 2011 (still not sourced; it turned out not to be
needed). Implemented as executable, dependency-free reference math in
`src/gonzales/restir_bdpt.mojo`, checked against a brute-force sum over
every strategy in `Tests/unit/test_restir_bdpt_mis.mojo`.

Index a full path's vertices `x_0` (light) … `x_k` (camera). Strategy `s`
means the light subpath supplied `x_0..x_{s-1}`, so the connection crosses
edge `s-1`. Per edge `e`, write `pf[e]`/`pr[e]` for the forward/reverse
solid-angle pdfs and `gl[e]`/`gc[e]` for the solid-angle→area conversions
at `x_{e+1}`/`x_e`. Then the balance-heuristic weight is

```
ω_s = 1 / ( 1 + (pf[s-1]·gl[s-1])·B_C(s) + (pr[s-1]·gc[s-1])·B_L(s) )

B_L(0)   = 0                        B_C(k+1) = 0
B_L(1)   = 1 / pa_light             B_C(k)   = 1 / pa_cam
B_L(s+1) = (1 + pr[s-1]·gc[s-1]·B_L(s)) / (pf[s-1]·gl[s-1])
B_C(s)   = (1 + pf[s]  ·gl[s]  ·B_C(s+1)) / (pr[s]·gc[s])
```

Both tails of the strategy sum telescope, and each factors into *one
subpath's* accumulator times *the connecting edge's* own pdfs. This is the
same shape `bdpt.mojo` already ships at its connect sites (`w_light`/
`w_camera`, `bdpt.mojo:3946`), with `η_VM = 0` (merging off — ReSTIR
BDPT's setting) and the vertex's reverse pdf folded into the accumulator
instead of split across `dVCM`/`dVC`.

**The property Phase 8 actually needs — subpath locality.** `B_L(s)` reads
only `pa_light` and edges `≤ s-2`; `B_C(s)` reads only `pa_cam` and edges
`≥ s`. Neither touches the other side. Under a reconnection shift that
keeps a candidate's light subpath and swaps in the receiving pixel's
camera prefix:

- `B_L(s)` is **shift-invariant** — reuse the stored value verbatim.
- `B_C(s)` is exactly what the *receiving* pixel already carries for its
  own base path.
- The connecting edge's four pdfs must be re-evaluated, but they are
  needed anyway to evaluate the shifted path's contribution.

So `ω_τ(T(x̄))` costs **O(1)** with no subpath retracing — which is what
8.1 asked for. `test_shifted_path_mis_weight_reuses_light_accumulator`
demonstrates this against a from-scratch evaluation of the shifted path.

**The trap, locked down by a test.** The target must be evaluated *at the
shifted path*: reusing the donor's `B_C` as well — the tempting shortcut,
since the candidate already carries it — evaluates `ω` at the *unshifted*
path. It is wrong, and silently so (a plausible image, quietly wrong
weights). `test_reusing_donor_camera_accumulator_is_wrong` asserts the two
disagree, so the mistake cannot pass unnoticed.

**Why this composes with GRIS.** Two independent MIS layers: `ω_τ` is a
partition of unity over strategies for each fixed path (verified by
`test_strategy_weights_partition_unity`); GRIS's `m_i` is a partition of
unity over candidate domains for each fixed sample. GRIS needs only that
`p̂ = ω_τ·q̂` be evaluable and that the shift be a bijection with a correct
Jacobian, so correctness reduces to evaluating `ω` at the shifted path
(above) plus the standard reconnection Jacobian — **which is separate work
and not covered by this derivation.**

**Limits, inherited not introduced.** Delta vertices reset
`dVCM`/`dVC` to 0, which is correct (they cannot be connected to) and
consistent with reconnection shifts needing a rough vertex anyway.
Distant/infinite/point-seeded light paths start at 0 — a documented
scoped simplification in gonzales today, which Phase 8 would inherit.
Volume vertices are excluded entirely (`_bdpt_vertex_mis_scoped`), which
is the media decision below.

#### 8.6 Media scope decision (2026-09-09): **surfaces only**

Phase 8 is scoped to non-volumetric transport; `--vcm`/`--sppm` stay alive
for volumetric caustics indefinitely. Reasoning, not assertion:

1. The `dVCM`/`dVC` recursion the 8.1 derivation extends **does not cover
   volume vertices at all** — `_bdpt_vertex_mis_scoped` (`bdpt.mojo:4909`)
   excludes them, so there is no accumulator to reuse and nothing to shift.
2. Base ReSTIR BDPT has **zero** media support (§8.3, full-text verified),
   so porting it would inherit this gap, not close it.
3. Volumetric scenes are already served by Phase 7's separate volumetric
   reservoir (light resampling + temporal reuse, shipped and verified).
   Keeping them in a separate reservoir and summing estimates is the
   field-consensus move §9.2 already documents for intractable densities.
4. Closing it properly means extending `dVCM`/`dVC` to the phase
   function's directional/reverse pdfs — a standalone research task, and
   one being probed concurrently from the VCM+media side.

### Phase 9 — Common currency (**open research**)

Goal: one reservoir where DI/GI, SMS, volumetric, and bidirectional
candidates compete, so technique selection emerges from resampling weights.

> **Status note (2026-09-10).** 9.2's last actionable idea — the
> bias-bounded naive plug-in — was measured against 158,824 real SMS solves
> and **refuted**; see 9.2. That removes the remaining hope of a *single*
> joint weight covering SMS, and leaves separate-reservoir-and-sum as the
> answer for the specular technique, matching ReSTIR BDPT's own practice.
> The realistic endpoint of this migration is therefore **one framework
> with several reservoirs**, not one weight over everything: resampling
> does the work where densities are tractable, and estimates are summed
> where they are not. 9.1 and 9.3 are unaffected and still worth doing.

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
  **MEASURED 2026-09-10 — the bound is correct, its premise is not. The
  pragmatic direction is REFUTED; use the fallback unconditionally.**
  Instrumented `sms_solve_bernoulli` with a hit-rate probe
  (`SMS_QHAT_PROBE_M`, default 0; the existing trial count `T ~ Geom(q)`
  is unbiased for `1/q`, and `1/T` is *not* an unbiased estimator of `q`,
  which is what the denominator needs). 158,824 real SMS solves on
  `sphere_sms.xml`; analysis reproducible via `Scenes/sms_qhat_analysis.py`.
  1. **The algebra simplifies and `n_SMS` cancels entirely**: with
     `Var(q̂)=q(1-q)/M` and `r = C/(n_SMS·q)`, the relative bias is
     `(1-q)/(M·q·(1+r)²)`. So it is governed by `q` and `M`, *not* by the
     `n_i` scales this entry expected to be the deciding factor.
  2. **The closed form is accurate where it is finite** — within 1–7% of
     the exact expectation over `k ~ Binom(M,q)` across the whole measured
     `q` range. The math was right.
  3. **But at `r = 0` the true bias is infinite, not small.** `q̂ = k/M`
     is exactly zero for **0.40%** of solves, making `F = 0`. The Taylor
     bound predicts `8e-4` where the truth is unbounded, because the
     expansion assumes `δ ≪ F₀` and that fails in precisely this tail.
     `P(q̂=0)` plateaus at ~4e-3 no matter how large `M` grows — those
     solves have genuinely tiny `q` (the `SMS_BERNOULLI_MAX_TRIALS` cap
     cases), so no sample budget rescues them.
  4. **`r = 0` *is* the SMS regime**, which inverts this entry's premise.
     The premise was that `q_SMS` is small relative to the rest of the sum
     because "most pixels aren't on a specular chain" — but pixels not on a
     specular chain never invoke SMS at all. SMS only produces a candidate
     where it is the *only* technique with support: the caster is a smooth
     (delta) `dielectric`, so NEE and BSDF sampling have zero density on
     that path. Measured corroboration from the same session: a dielectric
     blocks the straight shadow ray, and the fallback ray contributes
     exactly 0 (`_shadow_is_null_material` exempts only interface/null
     materials). So `C = 0` exactly where the joint weight would be used.
  5. **Cost, at `r = 0`, from the real `q` distribution** (median 0.70,
     p5 0.156): reaching 1% bias needs `M ≥ 40` at median `q`, `≥ 121` at
     p25, `≥ 540` at p5 — extra Newton solves *per SMS solve*, against the
     existing Bernoulli estimator's median `T = 1`. Even at `M = 64`, 41%
     of solves exceed 1% bias and route to the fallback; `M = 1024` still
     leaves 2.8%.
  So the threshold would send most SMS work down the fallback path anyway,
  at a ~500× cost multiplier, while leaving a hard division-by-zero tail no
  budget removes. **Do not implement the switch.** Phase 6's separate
  reservoir is not a fallback here — it is the answer, which is exactly
  what the independent confirmation below already indicates.
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
