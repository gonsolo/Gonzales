# Volumetric Media

Fog, smoke, clouds, and translucent liquids are not surfaces — light
scatters and is absorbed continuously as it travels through a volume. This
chapter covers gonzales's participating-media system: how it represents a
medium's density, how it samples a free-flight distance through one, and a
family of real bugs the volumetric NEE and bidirectional-connection paths
went through before they matched a reference. All code lives in
`gpu.mojo` (`_sample_medium_core`, shared verbatim by the CPU tile loop and
the GPU kernel), `sppm.mojo`/`bdpt.mojo` (their own simpler
`sample_homogeneous_free_flight`), and `nanovdb.mojo`.

## Three representations

| Medium | Density | Path tracer | BDPT/VCM, SPPM |
|---|---|---|---|
| `homogeneous` | constant | closed-form transmittance | closed-form transmittance |
| `uniformgrid` | dense `nx*ny*nz` float array | delta tracking | not supported |
| `nanovdb` | sparse `.nvdb` grid | delta tracking, point-sampled or trilinear | not supported |

The asymmetry is deliberate, not an oversight: BDPT/VCM and SPPM only ever
exercise homogeneous media (glass-of-water and volumetric-caustic scenes)
today, so they keep their own smaller `sample_homogeneous_free_flight`
rather than sharing the path tracer's richer, heterogeneous-capable
`_sample_medium_core`. Extending them to `uniformgrid`/`nanovdb` is future
work, not a known bug.

## Free-flight sampling: delta and ratio tracking

A homogeneous medium's free-flight distance has a closed-form exponential
pdf, `sigma_t * exp(-sigma_t * t)`, sampled directly by inverse-CDF. A
heterogeneous medium has no such closed form — its extinction varies point
to point — so gonzales uses **delta tracking** (Woodcock tracking): sample
candidate distances at a constant *majorant* rate `sigma_maj` (an upper
bound on the medium's true extinction anywhere), and at each candidate,
accept it as a real collision with probability `sigma_t(x)/sigma_maj`,
otherwise continue (a "null collision"). This is unbiased for any
majorant that actually bounds the density, and it needs no closed form for
the true density's integral.

The transmittance a shadow ray needs (for NEE) uses the same trick in
reverse — **ratio tracking**: at each candidate, multiply an accumulated
weight by `1 - sigma_t(x)/sigma_maj` instead of stochastically
accepting/rejecting, giving a continuous, unbiased transmittance estimate
with no need to know whether the ray actually left the medium.

### Local majorants: 6.7x

A single global majorant forces the delta-tracking step size to whatever
the *densest voxel anywhere in the grid* demands, even while the ray is
crossing empty space — on `disney-cloud`, a fixed step across a 596-unit
bounding sphere took ~2400 candidates per segment, almost all null
collisions in vacuum. NanoVDB already stores a max-density value per tree
node (leaf, lower, upper), so `_sample_medium_core` instead walks the ray
one *node* at a time and uses that node's own max as the local majorant —
an empty upper node (spanning 4096³ voxels) is skipped in a single step
instead of thousands. Unbiased by the memorylessness of the exponential
distribution: when a sampled distance overshoots the current node's
extent, tracking simply resumes from the node boundary under the next
node's majorant. `uniformgrid` keeps its prior single-segment,
whole-ray-majorant behavior unchanged (it has no per-node structure to
exploit). Trilinear-interpolation leaf caching (reusing one leaf lookup
for the 7 neighboring stencil taps) adds a further, smaller win. Measured
on `disney-cloud`, 160×90/32spp, GPU: 34.2s (global majorant) → 6.1s
(local majorants) → 5.14s (+ leaf caching) — 6.7x cumulative.

A too-small majorant biases delta tracking *silently* — the medium simply
looks thinner, with no crash or obvious visual artifact — so every change
to the majorant descent is checked with `make nvdb_diff`, which validates
that the claimed bound is `>=` the true max over the node it covers, not
just that density lookups themselves are correct.

## Volumetric NEE bugs

Direct lighting from a volume scatter vertex has to attenuate the shadow
ray through whatever's between the scatter point and the light — media,
vacuum, and any surfaces in between — and weight the result correctly
against BSDF/phase-function sampling. Three separate, real bugs surfaced
here, each with a distinct symptom:

**Attenuating vacuum, not just the medium.** The homogeneous branch of
volume NEE computed `T = exp(-sigma_t * dist)` over the *whole* distance
to the light, including any vacuum beyond the medium's boundary — while
`uniformgrid`'s ratio-tracking branch correctly stopped attenuating once
the shadow ray left the grid's bounding box. A far light read 0.117x pbrt
(predicted by `e^2`, since the true optical depth to the boundary is half
what was computed); a control with an environment light, unaffected by
this path, read 1.001x, confirming the medium sampling itself was fine.
Fixed by bounding the shadow-ray segment at the first interface surface
it crosses — occlusion is already ruled out by the time this attenuation
runs, so any hit found along the way must be a non-opaque medium boundary.

**MIS distance measured from the wrong vertex.** `shade_interface`
advances a path's ray *origin* forward at every null (interface) surface
it crosses, without bending the ray. A path that scattered inside a medium
and then crossed such an interface on its way to an emitter therefore had
its emitter-hit MIS weight computed from `inter.tHit` — measured from the
interface crossing, not from the volume scattering vertex where the
competing NEE sample was actually taken. With an emitter close to the
interface, the two distances imply wildly different light pdfs, and both
the NEE and the emitter-hit strategy ended up taking a weight near 1 —
summing instead of partitioning one unit of weight, and reading 1.80x
pbrt. The fix (`PathState_C.mis_null_dist`, `a9ba7df1`) accumulates
distance crossed through null interfaces since the last real scatter; the
emitter-hit weight adds it back onto `t_hit` before computing a pdf, and
it resets to zero at every real scattering event. This is not actually a
volumetric bug — any path crossing a null interface between its last real
scatter and an emitter hit was affected — media just make that geometry
common. After the fix: 0.90x, with the diagnostic technique worth keeping:
measuring each MIS strategy *in isolation* (NEE-only, emitter-hit-only)
showed each was individually plausible (0.861x, 0.975x) and their naive
sum was exactly the observed 1.80x error.

**Shadow-ray self-intersection for close lights.** A volume-scatter NEE
shadow ray had its *origin* offset forward by a small absolute epsilon,
but its `tmax` was still measured from the original, un-offset scatter
point — so the ray overshot and clipped the emitter itself whenever the
light was within about 0.4 units. Every unoccluded sample inside that
radius was reported occluded and contributed nothing (`f79999f4`). Fixed
by measuring `tmax` from the point the ray actually starts at, not the
point it was conceptually cast from — the same fix pattern applies
anywhere an origin is nudged by an epsilon: keep the distance measurement
consistent with the actual ray, not the ideal one.

## VCM/BDPT volume connections

Bidirectional connections through a medium hit a related but distinct
family of bugs, all rooted in one fact: **volume vertices have no
`dVCM`/`dVC`/`dVM` MIS weights** (Veach/Georgiev's derivation assumes a
surface pdf that a phase function doesn't have in the same form), so any
code path that implicitly relied on those weights to avoid double-counting
volume vertices needed its own explicit handling instead.

- **Albedo double-counted.** A stored volume light-vertex-cache (LVC)
  entry baked its own albedo into `v.beta`, but the connection code's
  volume branch *also* multiplied by the vertex's albedo (isotropic phase
  = `albedo/4π`) at connect time — double-counting once per volume vertex
  touched. Isolated by a direct camera-to-light-source connection alone:
  0.785x with the double count, 0.983x without, exactly a factor of
  `1/albedo`.
- **An n-scatter photon connected (n+1) times.** For an area light, LVC
  index 0 *is* the light-source vertex; every later index is one more real
  scatter along the same photon. A camera volume vertex connecting to
  every stored index therefore summed the same physical path once per
  scatter along it. Fixed by restricting a volume connection to index 0
  only when that index is itself light-originated — gated on the light
  path's own origin flag, not on the light *type*, since distant/point/
  infinite lights store no light-source vertex at all and a naive "stop
  storing volume vertices at index 0" fix broke those paths entirely
  (an env-lit slab dropped from 1.35x to 0.44x during one wrong attempt).
- **Unweighted `(s,t)` splits summed without limit.** Connecting a camera
  vertex to light-cache vertex `s` and to `s+1` reaches paths of different
  total length — both legitimate on their own — but they are also two of
  several `(s,t)` splits that reach a path of any *given* length, and
  those splits are competing strategies that must share one unit of MIS
  weight. The `dVCM`/`dVC` machinery enforces that for the pairs it
  covers; it does not cover volume vertices, so those pairs came back at
  full weight and simply summed. The diagnostic signature: capping the
  light-vertex count at 1/2/3/all on an env-lit slab gave 1.027 / 1.388 /
  1.651 / 1.969x pbrt — each additional split adding a roughly *constant*
  amount that never decays with more samples. That non-decaying-increment
  shape is the general signature of an unweighted sum over competing
  strategies, worth recognizing in any similar sweep. Fixed
  (`edf77c3e`) by summing every MIS-weighted pair but taking at most one
  *unweighted* pair (lowest index) per camera vertex, restoring exactly
  one strategy per path length. Keyed on the *pair*, not on which side
  originates in a medium — a surface camera vertex connecting to several
  volume light vertices is the identical over-count from the other side,
  and an earlier, light-type-keyed version of the fix missed it.

Combined effect of the three fixes: a far-light control went from 2.68x to
0.98x pbrt, a close light from 3.50x to 0.92x, a three-light scene to
0.98x. The proper long-term fix is real MIS weights for volume vertices —
extending `dVCM`/`dVC`/`dVM` to the isotropic phase function's own
(trivial, constant) directional/reverse pdfs — not attempted, because the
existing strategy set does no explicit env/distant/point NEE at volume
vertices, and textbook VCM weights assume NEE exists everywhere; applying
them unmodified would silently under-weight in a different way.

## Validating without a reference renderer

The most useful check for volumetric correctness needs no reference image
at all: a **conservative medium (single-scattering albedo exactly 1) in a
spatially uniform lighting environment must be exactly invisible** —
radiative equilibrium means the outgoing radiance equals the environment's
radiance everywhere, independent of the medium's density, thickness, or
phase function. Every pixel should read exactly 1.0× the environment,
whatever the scene. This analytic case caught several bugs whose
symptoms were suspiciously round numbers: a missing volume-NEE partner
left the conservative case reading exactly 0.500 (one MIS strategy
missing its complement), and a null-interface bug that clobbered MIS
state left it at exactly 1.498 (one strategy taking full weight on top of
an already-complete 1.0 estimate). An exact simple ratio against a known
analytic answer is a strong hint to look at *strategy weights*, not at the
sampler.

Two traps worth remembering when using this test: it is only meaningful
once the medium is confirmed to actually be entered (an inverted bounding
volume's winding can make "invisible because energy is conserved" and
"invisible because nothing was ever sampled" look identical — instrument
the in-medium event count first); and a 1×1×1 density grid interpolates
degenerately in both gonzales and pbrt, disagreeing with each other by
30%+ even though each independently matches the analytic answer — use a
grid several voxels across before trusting any cross-renderer ratio.
