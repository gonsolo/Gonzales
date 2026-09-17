#!/usr/bin/env python3
"""Phase 8.2: why a caustic reservoir's confidence weight may not be updated
from whether the sample landed.

docs/A2_restir_migration_plan.md, Phase 8.2: "Confidence weights for these
must not be updated based on whether a caustic sample actually landed on the
pixel (that would correlate the weight with the realized sample and bias the
result) -- ReSTIR BDPT updates via a proxy: the prior frame's
motion-vector-mapped reservoir weight."

That is a claim about an estimator, so this checks it by running the
estimator, in the same spirit as Scenes/vcm_bssrdf_mis_derivation.py. No
renderer, no dependencies beyond the standard library.

The model is the smallest thing that has the structure of a caustic pixel
under temporal reuse:

  * a discrete path space of N candidates; f is zero for most of them and
    large for a few (a caustic: rare, high-energy, found by luck)
  * each frame draws ONE fresh candidate uniformly, and combines it with the
    previous frame's reservoir, weighted by confidence -- ReSTIR's streaming
    combine, the same shape reservoir.mojo implements
  * the quantity being estimated is I = sum_x f(x), known exactly, so bias is
    measurable rather than argued about

Three confidence rules are compared:

  count   c = number of frames accumulated (capped). Independent of what was
          sampled. This is what gonzales ships today.
  landed  c = number of frames whose selected sample was NONZERO. This is the
          rule Phase 8.2 warns against: it reads the realized sample.
  weight  c scaled by the carried reservoir's OWN weight. This is the
          obvious reading of "update via the prior frame's reservoir
          weight", and it is measured here because it is the mistake a
          reader of the plan is most likely to make: the weight is itself a
          function of the realized sample, so this biases too (+88% here).
          Modelling ReSTIR BDPT's actual Eq. 27 faithfully needs the paper,
          which this repo does not have; until then the only rule shown to
          be safe is a realization-independent count.

Run: python3 Scenes/restir_caustic_confidence_check.py
Exit status is nonzero if a rule that should be unbiased is not, or if the
rule that should be biased is not -- an always-passing check is worthless.
"""
import random

N = 64                 # candidates in the path space
N_CAUSTIC = 3          # how many of them carry the caustic
CAUSTIC_VALUE = 40.0   # its energy: rare and bright, which is the hard case
BACKGROUND = 0.05      # the dim non-caustic paths
FRAMES = 24            # temporal chain length
CHAINS = 200000        # independent chains averaged
M_CAP = 20.0           # confidence cap, as every ReSTIR implementation has


def make_f():
    f = [BACKGROUND] * N
    for i in range(N_CAUSTIC):
        f[i] = CAUSTIC_VALUE
    return f


class Reservoir:
    """The streaming reservoir of reservoir.mojo: y, w_sum, m, W."""
    __slots__ = ("y", "w_sum", "m", "W")

    def __init__(self):
        self.y, self.w_sum, self.m, self.W = None, 0.0, 0.0, 0.0

    def stream(self, y, w, rng):
        if w <= 0.0:
            return
        self.w_sum += w
        if rng.random() * self.w_sum < w:
            self.y = y

    def finalize(self, target):
        # W = w_sum / (m * p̂(y)); m is the confidence weight (1/M-style
        # weights, ReSTIR DI's own form).
        p = target(self.y) if self.y is not None else 0.0
        self.W = self.w_sum / (self.m * p) if (self.m > 0.0 and p > 0.0) else 0.0


def run_chain(f, rule, rng):
    """One temporal chain; returns the final frame's estimate of sum(f)."""
    target = lambda x: f[x]
    prev = None
    prev_conf = 0.0
    for _ in range(FRAMES):
        # Fresh canonical candidate: uniform over the domain, so its
        # resampling weight is p̂/p = f(x) * N.
        x = rng.randrange(N)
        cand = Reservoir()
        cand.stream(x, f[x] * N, rng)
        cand.m = 1.0
        cand.finalize(target)

        res = Reservoir()
        if prev is not None and prev.y is not None:
            res.stream(prev.y, prev_conf * target(prev.y) * prev.W, rng)
        res.stream(cand.y, 1.0 * target(cand.y) * cand.W, rng)
        res.m = prev_conf + 1.0
        res.finalize(target)

        # ── the rule under test: what confidence does the NEXT frame give
        # this reservoir?
        if rule == "count":
            prev_conf = min(res.m, M_CAP)
        elif rule == "landed":
            # Realization-dependent: only count frames that actually landed
            # on a caustic path. This is the rule Phase 8.2 forbids.
            landed = res.y is not None and f[res.y] >= CAUSTIC_VALUE
            prev_conf = min(res.m, M_CAP) if landed else 0.0
        elif rule == "weight":
            # Scales confidence by the carried reservoir's own weight. Reads
            # the realized sample through W and f(y), which is exactly what
            # makes it biased -- see this file's header.
            scale = min(1.0, res.W * target(res.y) / max(sum(f), 1e-9)) if res.y is not None else 0.0
            prev_conf = min(res.m, M_CAP) * (0.5 + 0.5 * scale)
        else:
            raise SystemExit("unknown rule " + rule)
        prev = res

    return (f[prev.y] * prev.W) if (prev is not None and prev.y is not None) else 0.0


def main():
    f = make_f()
    truth = sum(f)
    print(f"path space {N} candidates, {N_CAUSTIC} carrying the caustic "
          f"({CAUSTIC_VALUE} vs {BACKGROUND}); sum(f) = {truth:.4f}")
    print(f"{FRAMES} frames per chain, {CHAINS} chains\n")

    results = {}
    for rule in ("count", "landed", "weight"):
        rng = random.Random(12345)          # common random numbers across rules
        total = sum(run_chain(f, rule, rng) for _ in range(CHAINS))
        est = total / CHAINS
        # Standard error of the mean, so "unbiased" is judged against noise
        # rather than against a hopeful eyeball.
        rng2 = random.Random(12345)
        sq = 0.0
        for _ in range(CHAINS):
            v = run_chain(f, rule, rng2)
            sq += v * v
        var = max(sq / CHAINS - est * est, 0.0)
        sem = (var / CHAINS) ** 0.5
        err = est / truth - 1.0
        sigmas = abs(est - truth) / sem if sem > 0 else float("inf")
        results[rule] = (est, err, sigmas)
        print(f"  {rule:7s} estimate {est:9.4f}   error {err:+7.2%}   "
              f"{sigmas:6.1f} sigma from truth")

    print()
    ok = True
    if results["count"][2] > 4.0:
        print(f"FAIL: 'count' should be unbiased, but sits "
              f"{results['count'][2]:.1f} sigma from truth")
        ok = False
    if results["weight"][2] < 4.0:
        print("FAIL: 'weight' was biased when this was written (+88%); if it "
              "is clean now the model changed -- re-derive before trusting it.")
        ok = False
    if results["landed"][2] < 4.0:
        print("FAIL: 'landed' should be measurably biased here, but is not -- "
              "either the model no longer exercises the trap, or the claim "
              "does not hold in this formulation. Do not weaken the test to "
              "make it pass.")
        ok = False
    if ok:
        print(f"PASS: realization-independent confidence (count) is unbiased "
              f"({results['count'][1]:+.2%});\n      confidence read from "
              f"whether the sample landed is biased {results['landed'][1]:+.1%};"
              f"\n      confidence scaled by the carried weight is biased "
              f"{results['weight'][1]:+.1%} -- both read the realized sample.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
