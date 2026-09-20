#!/usr/bin/env python3
"""Render the scenes whose answer is ARITHMETIC, and compare against it.

Every other render test in this repo compares against something measured: a
pinned value (smoke_matrix.py), a converged reference (compare_bitterli.sh), or
a feature's presence (caustic_presence_check.py). All of those detect DRIFT.
None of them can tell you the renderer is wrong today, because the thing they
compare against was produced by the same renderer.

These scenes are different: their radiance is known in closed form, so a
disagreement is a defect, full stop. That distinction is not academic -- the
smoke matrix pins closed-cavity's VCM cell at 0.63 and reports green, while
the correct answer is 1.0.

    analytic_check.py            # check every case
    analytic_check.py --update   # rewrite the recorded per-integrator gaps

Renders are GPU-only, by project rule.

WHY EACH SCENE IS ANALYTIC -- all three work by killing the Neumann series
with symmetry rather than by solving it:

  closed-cavity   A sealed box, walls rho=1, emitter radiance L. Nothing can
                  absorb, so it equilibrates at uniform radiance L everywhere
                  (Kirchhoff). Geometry drops out of the answer entirely.
                  Tests UNBOUNDED interreflection: a per-bounce energy error
                  compounds instead of cancelling, which is what makes this
                  the strictest of the three.
  env-furnace     One diffuse quad under a constant environment: irradiance
                  pi*L, so Lo = rho*L. Tests ONE bounce plus the NEE/BSDF MIS
                  pair -- and it was this scene that exposed VCM's merge
                  double count, which no corpus bisection had found.
"""
import argparse, json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytic_gaps.json")
MODES = {"pt": [], "vcm": ["--vcm"], "sppm": ["--sppm"]}

# scene -> (path, expected radiance, crop, extra flags, why)
CASES = {
    "closed-cavity": dict(
        scene="Scenes/closed-cavity-equilibrium.pbrt", expect=1.0, res="32x32",
        crop=None, strict=True,
        why="sealed rho=1 cavity, emitter L=1 -> equilibrium radiance L"),
    "env-furnace": dict(
        scene="Scenes/env-furnace-analytic.pbrt", expect=0.5, res="64x64",
        crop=(16, 48), strict=True,
        why="diffuse quad rho=0.5 under constant env L=1 -> Lo = rho*L"),
}
# Per-material furnace tests: a quad of each material under a constant
# environment, set to absorb nothing, so energy conservation forces Lo = L.
# `strict` marks the materials that MUST conserve exactly -- for those a
# deviation is unambiguously an implementation bug.
for _m in ("diffuse", "dielectric", "thindielectric", "coateddiffuse",
           "diffusetransmission", "mix"):
    CASES["furnace-" + _m] = dict(
        scene="Scenes/furnace/%s.pbrt" % _m, expect=1.0, res="64x64",
        crop=(16, 48), strict=True,
        why="%s, nothing absorbed -> Lo = L" % _m)
# Conductor is swept over roughness and NOT asserted at 1.0: single-scattering
# GGX is lossy by construction (it drops multi-bounce microfacet paths), so the
# deviation is the model, not the code, until a Kulla-Conty/Turquin
# compensation term exists. The SHAPE of the curve is the diagnostic -- a
# monotone falloff growing with alpha is GGX behaving as theory predicts;
# anything else is ours.
for _a in ("00", "01", "02", "04", "07", "10"):
    CASES["furnace-conductor-a" + _a] = dict(
        scene="Scenes/furnace/conductor-a%s.pbrt" % _a, expect=1.0, res="64x64",
        crop=(16, 48), strict=False,
        why="conductor alpha=%s.%s; GGX single-scattering loss is EXPECTED" % (_a[0], _a[1]))

# A cell may legitimately miss the analytic answer today. Record the gap so the
# suite catches a REGRESSION without pretending the renderer is correct: the
# recorded number is a known defect, never a target. TOL is how much worse a
# cell may get before it fails.
TOL = 0.04


def render(case, mode):
    """Render one case and return its mean radiance over the crop."""
    import numpy as np
    import OpenImageIO as oiio
    c = CASES[case]
    for f in os.listdir(REPO):
        if f.endswith(".exr"):
            os.remove(os.path.join(REPO, f))
    cmd = [os.path.join(REPO, "build", "gonzales"), "--gpu", "--no-denoise",
           "--spp", "64", "--resolution", c["res"], "--seed", "1",
           *MODES[mode], os.path.join(REPO, c["scene"])]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    written = [f for f in os.listdir(REPO) if f.endswith(".exr") and "albedo" not in f]
    try:
        if proc.returncode != 0 or len(written) != 1:
            return None, (proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["no output"]
        img = oiio.ImageInput.open(os.path.join(REPO, written[0])).read_image("float")[..., :3]
        if c["crop"]:
            a, b = c["crop"]
            img = img[a:b, a:b]
        return float(img.mean()), []
    finally:
        for f in written:
            p = os.path.join(REPO, f)
            if os.path.exists(p):
                os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()
    gaps = json.load(open(GAPS)) if os.path.exists(GAPS) else {}
    failures, known = [], []

    for case in sorted(CASES):
        c = CASES[case]
        print(f"{case}: expect {c['expect']:.4f}  ({c['why']})")
        for mode in MODES:
            key = f"{case}.{mode}"
            got, why = render(case, mode)
            if got is None:
                print(f"  FAIL {key:24s} {'; '.join(why)}")
                failures.append(key)
                continue
            ratio = got / c["expect"]
            if args.update:
                gaps[key] = round(ratio, 4)
                print(f"  set  {key:24s} {got:.4f}  ratio {ratio:.4f}")
                continue
            recorded = gaps.get(key)
            if recorded is None:
                print(f"  FAIL {key:24s} ratio {ratio:.4f} -- no recorded gap; run --update")
                failures.append(key)
            elif abs(ratio - 1.0) <= TOL:
                print(f"  ok   {key:24s} {got:.4f}  ratio {ratio:.4f}  CORRECT")
            elif ratio < recorded - TOL or ratio > recorded + TOL:
                print(f"  FAIL {key:24s} ratio {ratio:.4f}, was {recorded:.4f} -- REGRESSED")
                failures.append(key)
            else:
                pct = (ratio - 1.0) * 100.0
                print(f"  GAP  {key:24s} {got:.4f}  ratio {ratio:.4f}  "
                      f"({pct:+.1f}% vs the analytic answer -- KNOWN DEFECT)")
                known.append(key)

    if args.update:
        json.dump(dict(sorted(gaps.items())), open(GAPS, "w"), indent=2)
        open(GAPS, "a").write("\n")
        print(f"wrote {GAPS}")
        return 0
    if known:
        print(f"\n{len(known)} cell(s) do NOT match the analytic answer and are "
              f"tracked as known defects: {', '.join(known)}")
    if failures:
        print(f"analytic check FAILED: {len(failures)} cell(s): {', '.join(failures)}")
        return 1
    print("analytic check passed (no regressions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
