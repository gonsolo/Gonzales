# Manifold Caustics: Specular Manifold Sampling

A dielectric caustic — light bent by glass or water onto a diffuse
surface — is invisible to both of the sampling strategies chapter 5's
dielectric section describes. NEE can't hit it: a shadow ray toward the
light has probability zero of also landing exactly on the point where a
specular chain happens to focus. BSDF sampling can't either, for the same
reason in reverse: a randomly sampled reflection/refraction direction
essentially never lands back on the light. What's needed is a solver that
starts from "this shading point and this light" and finds the specular
path connecting them directly. `shading.mojo`'s `_mnee_walk`/`_mnee_walk2`
already do this for one or two specular vertices (MNEE — Manifold
Next-Event Estimation). `sms.mojo` generalizes it to an arbitrary-length
chain, and adds what a fixed 1-/2-vertex solve never needed: curved
casters and an unbiased correction for multiple solutions. All code
snippets below are current source, verified against `src/gonzales/sms.mojo`
while writing this chapter — not transcribed from development notes.

## The manifold walk

Given a light point and a shading point, a specular chain between them
must satisfy one tangential constraint per intermediate vertex: the
generalized half-vector (the bisector of the incoming/outgoing directions,
refraction-weighted by the local IOR ratio) must align with the vertex's
own shading normal. For a chain of `n` vertices this is `n` coupled 2D
constraints, solved by Newton's method. Because each vertex's constraint
only involves its immediate neighbors, the linearized system is
block-tridiagonal:

```
[b0 c0      ] [dx0]   [cv0]
[a1 b1 c1   ] [dx1] = [cv1]
[   a2 b2 c2] [dx2]   [cv2]
[      .. ..] [.. ]   [.. ]
```

solved with the standard block Thomas algorithm — forward elimination then
back-substitution on 2×2 blocks, no general linear solver needed. This is
not a new formulation: `_mnee_walk`/`_mnee_walk2`'s own `(a, b, c)`
coupling matrices (themselves ported from Cycles' `mnee.h`) are exactly
the n=1/n=2 special cases of this same system, and `sms_walk(n=2)` was
verified to reduce to `_mnee_walk2`'s formulas by hand and confirmed
bit-for-bit on a synthetic two-glass-pane case — a genuine generalization,
not a rewrite, of code chapter 5 already describes. The two fast paths
stay in `shading.mojo` untouched for n≤2; only chains of three or more
specular vertices dispatch into `sms_walk`.

gonzales's own scenes default to the same **half-vector** formulation
MNEE uses. `sms.mojo` also carries a second, independent formulation —
**angle-difference constraints**, which refract the direction to the
shading point through the vertex and compare the result to the light
direction as a difference of spherical angles — because the reference
scene used to validate this code (`sphere_sms.xml`, from Zeltner et al.
2020's own paper) sets `caustics_halfvector_constraints = false` and
selects this formulation. The two give different roots, basins, and
Jacobians for the same geometry; they are not interchangeable, and a
curved single-vertex chain specifically uses the angle-difference form to
match the reference.

## Curved casters: reprojection is the hard part

A flat triangle's tangent plane *is* its surface everywhere, so a Newton
step just moves within it. A sphere's tangent plane only agrees with the
true surface at the point of tangency — continuing a Newton step naively
in it drifts off the actual constraint. `sms_walk` reprojects onto the
true sphere after every step for any vertex flagged `is_sphere`, and this
reprojection is where two of the six real bugs in this section originate.

The naive fix — snap the raw step to the nearest point on the sphere,
`normalize(x_raw - center) * radius` — has no notion of which side of the
sphere is actually reachable from the rest of the chain. On the reference
scene (a half-buried glass sphere over a floor) this reliably converged to
points on the sphere's *underside*, physically embedded below the floor,
silently discarded by the downstream visibility check. The fix,
`_sms_reproject_onto_sphere_anchored`, ports the real SMS reference
renderer's own strategy: instead of snapping, cast a ray from a **fixed
anchor** (the shading point for the first vertex, the previous vertex's
position for any later one) through the raw proposal and take the actual
intersection. A ray from a real anchor can only ever hit the surface
visible from that anchor, which forecloses the wrong-hemisphere failure
by construction — not a bias tweak, a structural fix. The anchor argument
also disambiguates entry from exit: an off-sphere anchor (`x0`) takes the
ray's *first* crossing, while an anchor already sitting on the same
sphere (the previous chain vertex) skips past its own surface and takes
the *far* crossing, matching how a straight line actually continues
through a solid sphere's interior.

A second, independent bug lived in the same neighborhood: even a
correctly-reprojected single-curved-vertex solution routinely turned out
to be unusable, because its *outgoing* leg (vertex → light) re-entered the
same sphere's far side before reaching the light — a real, unmodeled
consequence of the model only solving for one bend, not the sphere's
actual exit refraction. The fix isn't a stricter Newton step; it's telling
the final shadow ray to ignore exactly the one sphere the chain just
solved a bend at (`any_hit_bvh2_core` grew optional
`ignore_sphere_center`/`ignore_sphere_radius` parameters, a no-op for
every other caller). Measured on the exact floor-radius band identified as
the real caustic zone: 0/180 → 197/197 accepted contributions before vs.
after.

Separately from either of these — and a much larger, structural gap — is
that `any_hit_bvh2_core` and `traverse_bvh2_core`, the shadow-ray and
closest-hit BVH walks used by *every* render mode's occlusion and NEE
queries (path tracer, BDPT, SPPM, every GPU shadow kernel), never tested
analytic spheres at all; only a handful of primary-ray call sites called a
separate `test_spheres()` explicitly. This meant analytic sphere lights
and casters had never cast a shadow anywhere in gonzales, in any render
mode, independent of SMS. Fixed by folding sphere testing directly into
both BVH walks as optional trailing parameters (dangling/zero defaults, so
every pre-existing caller is unaffected unless it opts in), wired at all
17 call sites that needed it.

## The Bernoulli-trial estimator

For a flat single-triangle refraction — MNEE's own scope — the manifold
solution given a probe is essentially unique, so one deterministic Newton
solve is enough. That stops being true for a chain of three or more
specular vertices, or any normal-mapped caster: a single seed's solve is
not guaranteed to be the *only* solution contributing along that light
direction. `sms_solve_bernoulli` (Zeltner et al. 2020) handles this
without a separate probability estimate: solve once from a jittered seed
to fix a primary solution `X*`, then draw fresh seeds and count trials
until one reconverges to `X*`. The trial count `T` is itself an unbiased
estimator of `1/q` where `q` is the probability a random seed finds that
root (`E[T] = 1/q` for a geometric distribution) — no separate division
step needed. Two seeds count as the *same* root when the directions from
the shading point agree to within `cos ≈ 1 − 1e-5` (comparing directions,
not positions, matters: `sms_walk`'s own Newton stop tolerance is looser
than that, so two seeds in the same basin can land measurably apart in
position without being different roots). The trial count is capped at
`SMS_BERNOULLI_MAX_TRIALS = 512`; truncation biases the result *darker*,
never brighter, so the cap is a cost/darkness trade-off, not a
correctness switch.

## Six real bugs, one measured end to end

Validating this code needed an actual ground truth, not just "looks
plausible": a brute-force path-traced render of the same scene with no
manifold solver at all (16384 spp, no shortcuts) gave 0.2343 on the
sphere-caustic region, and an independently published SMS reference
renderer agreed (0.2278–0.2281). Against that fixed target, six real bugs
were found and fixed, in order:

1. **Eta orientation** — the solid-angle compression factor for a delta
   refraction takes the material IOR oriented from the *emitter* side; the
   half-vector constraint takes it oriented from the *shading-point* side.
   Those orientations are provably opposite for a refraction, and one
   shared, single-oriented eta was used for both — an η⁴ error (~5×). It
   cancels exactly in a 2-vertex enter+exit chain (the two etas are
   reciprocal), which is why only a single-refraction sphere path exposed
   it. 0.18× → 0.86× of reference.
2. **Real Bernoulli estimator + tighter tolerance** — the sphere path was
   using one deterministic seed at weight 1 instead of the Bernoulli
   estimator above, and the Newton stop tolerance (1e-3) was loose enough
   to make one basin look like several, inflating the trial count and
   biasing the image dark. Tightened to 1e-5, matching the reference.
3. **Angle-difference constraint for curved vertices** — porting the
   reference's actual formulation (rather than reusing half-vector
   everywhere) raised the fraction of caustic pixels receiving any
   contribution from 62.7% to 85.0%.
4. **Mixed measures in the light-side Jacobian** — the light-sampling
   helper returns raw triangle edge vectors, not an orthonormal basis, so
   a Jacobian computed from it was in parametric-unit measure while the
   inverse-pdf it was multiplied against was in area measure. Both bases
   need `make_orthonormal()`-equivalent treatment, matching the
   reference. Caught by dumping per-factor distributions from gonzales
   and a clean-room reference port side by side and finding one factor
   (the geometric term) off by 0.611× while every other factor agreed to
   0.4%. Percentile ratios vs. reference: p25/p50/p75/p90 went from
   0.744/0.703/0.794/0.818 to 1.003/0.995/1.061/1.203.
5. **Per-emitter-point suppression** — the ordinary NEE path needs to know
   when SMS already covers a light so it doesn't also fire an ordinary
   shadow ray at it; the original check keyed on "did MNEE fire for the
   light this NEE sample drew," independent of where the BSDF ray
   actually goes — biased at a vertex with glass toward some lights and
   open sky toward others. Fixed to probe the specific segment to that
   emitter point directly, so NEE and SMS partition the emitter by
   construction rather than by a proxy.
6. **The big one — double counting.** MNEE/SMS samples the specular chain
   directly and replaces the shadow ray for it; ordinary BSDF sampling can
   *also* reach the same emitter along the same physical family (diffuse
   bounce → refract through the glass → hit the light), and gonzales was
   adding both with no MIS between them. The assumption that "BSDF
   sampling effectively never reproduces the specular chain" — true for a
   small light — is false when the caster is a sphere large enough that
   the shading point sits inside its solid angle. Proof this was the
   whole story: zeroing the SMS contribution entirely *still* rendered
   the caustic at 0.2297, matching the 0.2343 truth. A new
   `PathState_C.sms_covered` flag now suppresses the duplicate at the
   emitter-hit site. Result against the path-traced ground truth:
   mean 0.4403 → 0.2400 (truth 0.2343 — 2.4% residual, including the
   firefly tail), p99 6.105 → 1.078 (truth 1.020). This was also the
   entire firefly source, which is why nothing inside the estimator
   itself had ever visibly changed the noise level while it was present.

Every other SMS ingredient checked out fine along the way and stayed
untouched: Snell's law held to a median residual of 0.0000 at every trial
count, Fresnel behaved correctly toward grazing angles, the light-side
Jacobian matched finite differences to 0.1%, and the *set* of solutions
found matched the reference port's own solution-count statistics closely
(convergence rate 40% vs. the reference's 17.96%, mean distinct roots
1.95–2.00 either way). The lesson generalizes: when a per-sample quantity
matches a reference almost exactly (here, total SMS energy per sample
matched to 0.07%) while the *image* is still wrong by a large factor,
the bug is almost certainly not in the estimator itself — it's something
outside it double-counting or misweighting the same correct samples.

## Where the caustic actually is

The validation scene (`sphere_sms.xml`) centers its sphere on the ground
plane with a radius large enough that the sphere is half-buried. The
caustic lands on the floor *underneath* the sphere and is only visible
through the glass — it's the swirl pattern seen on the sphere itself, not
a bright patch on the open floor nearby. A dense scan of the manifold
constraint over the whole visible floor, outside that region, never finds
a root. This sounds obvious once stated, but cost real debugging time
before it was: any instrumentation measuring "primary-hit (bounce 0)
caustic contributions" on this scene is measuring something that's zero
by geometry, since a camera ray hits the floor directly, not through a
refraction.

**Validation tooling**: `Tools/sms_mitsuba_ref.mojo` is a self-contained
clean-room transcription of the reference single-scatter estimator plus a
minimal path tracer sharing nothing with gonzales's renderer but `Vec3f`
and the OIIO bridge — it reproduces the published reference render to
within a few percent and is the tool used above to isolate the Jacobian
bug via per-factor comparison. **Measurement discipline that mattered**:
always pass `--seed N` — without it two runs of the same scene differ in
95% of pixels with a ~19% mean swing from RNG alone, large enough to hide
or fabricate any smaller effect being measured; and never judge a caustic
by a median-filtered mean alone, since that statistic is specifically
insensitive to the rare, heavily-weighted samples that turned out to be
where bug 6 actually lived.

## SMS and ReSTIR

`docs/A2_restir_migration_plan.md`'s Phase 6 reuses this chapter's
`sms_walk` as the shift mapping for a GRIS reservoir over specular
chains — generate a candidate via the ordinary probe-and-solve above,
store it in an `SMSReservoir`, and temporally/spatially combine reservoirs
across pixels by re-walking a neighbor's stored solution into the current
pixel's domain and checking bijectivity. Core generation and temporal
reuse are done and validated (stable over many accumulated frames, no
energy growth). Spatial reuse runs but currently finds almost every
neighbor reservoir empty and so reaches the shift function itself
essentially never — see A2 for the measured rejection breakdown and the
tile-based sample-space partitioning (Hong et al. 2025) that's the
documented next step to fix that.
