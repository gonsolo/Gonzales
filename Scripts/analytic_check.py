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
for _m in ("diffuse", "dielectric", "thindielectric",
           "diffusetransmission", "mix"):
    CASES["furnace-" + _m] = dict(
        scene="Scenes/furnace/%s.pbrt" % _m, expect=1.0, res="64x64",
        crop=(16, 48), strict=True,
        why="%s, nothing absorbed -> Lo = L" % _m)

# coateddiffuse is the one material here that does NOT have to reach 1.0, and
# the reason is physics rather than a defect: the coat is a Beer-Lambert
# absorbing slab of thickness DEFAULT_COAT_THICKNESS (0.01, pbrt's default,
# hardcoded -- gonzales does not parse `thickness`). Every coat traversal
# attenuates, and the TIR recycle costs two, so a rho=1 base still loses a few
# percent. Verified rather than assumed: setting DEFAULT_COAT_THICKNESS to 0
# and rebuilding gives 0.9934, i.e. the whole 0.0872 shortfall is the coat,
# and only ~0.7% is the COAT_MAX_DEPTH truncation. Keep the recorded gap as a
# REGRESSION guard -- it caught the +30% double count that used to sit here --
# but do not read it as an energy-conservation failure.
CASES["furnace-coateddiffuse"] = dict(
    scene="Scenes/furnace/coateddiffuse.pbrt", expect=1.0, res="64x64",
    crop=(16, 48), strict=False,
    why="coateddiffuse; coat absorbs at thickness 0.01, so Lo < L is CORRECT")
# The ROUGH sibling, and the reason it exists: furnace-coateddiffuse above is
# roughness 0, so until 2026-09-22 the suite had NEVER tested a rough coat --
# the mechanical reason "rough coats" stayed its named remainder. The rough
# case is far worse than the smooth one, and the loss grows with roughness on
# a material whose coat can only absorb a few percent:
#
#     roughness   alpha    PT       VCM
#     0.0         0.000    0.9149   0.9166
#     0.1         0.316    0.7853   0.8305
#     1.0         1.000    0.6292   0.6608
#
# (alpha = sqrt(roughness): pbrt's remaproughness, material_builder.mojo.)
# That shape -- fine at alpha 0, ~35% gone by alpha 1 -- is the signature of
# uncompensated single-scattering loss at the coat's rough interface, the
# same defect ggx_ms_lobe already fixes for conductors, whose numbers before
# that existed were strikingly similar (0.794 at alpha 0.4, 0.327 at 1.0;
# see bxdf.mojo's Kulla-Conty header). It is NOT a VCM defect: PT is worse
# than VCM at every roughness, so any "make VCM match PT" comparison on a
# rough coat is calibrating against the more wrong of the two.
CASES["furnace-coateddiffuse-rough"] = dict(
    scene="Scenes/furnace/coateddiffuse-rough.pbrt", expect=1.0, res="64x64",
    crop=(16, 48), strict=False,
    why="coateddiffuse at roughness 0.1; a rough coat loses far more than the "
        "coat's own absorption -- tracked as a KNOWN DEFECT, see the header "
        "above and project_vcm_rough_coat_mis")
# Conductor is swept over roughness and IS asserted at 1.0 -- as of
# 2026-09-21 it conserves energy at every roughness. It did not used to, and
# the reason it now does is four separate defects deep; the scene headers in
# Scenes/furnace/conductor-a*.pbrt carry the full account. The two worth
# repeating here, because they are about THIS SUITE rather than about GGX:
#
#   * The curve's SHAPE was the diagnostic that cracked the first one. It
#     read 0.50 at alpha=0.002, rose to 0.62, then fell -- a step at the
#     delta/rough threshold plus a non-monotonicity, where real GGX albedo
#     falls monotonically. That convicted the renderer without any reference
#     image, which is the whole point of an analytic scene.
#
#   * But the last one, a broken sample_ggx_vndf costing 30 percent of a
#     GRAZING rough conductor's energy, this suite structurally COULD NOT
#     see: every furnace scene in this directory views its quad head-on, and
#     the bug vanishes at normal incidence. It took a unit test sweeping mu_o
#     (Tests/unit/test_conductor_energy.mojo) to find it. Treat that as a
#     standing limitation of these scenes, not a one-off -- an analytic scene
#     only proves the renderer correct on the configuration it renders, and
#     every scene here renders one flat quad seen head-on.
for _a in ("00", "01", "02", "04", "07", "10"):
    CASES["furnace-conductor-a" + _a] = dict(
        scene="Scenes/furnace/conductor-a%s.pbrt" % _a, expect=1.0, res="64x64",
        crop=(16, 48), strict=True,
        why="conductor alpha=%s.%s; Kulla-Conty compensated, must conserve" % (_a[0], _a[1]))

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
    # A per-render timeout, because without one a single hung render stalls
    # the suite forever and is indistinguishable from "still going" -- which
    # cost 15 minutes of staring at a blank log once. 300s is ~600x the
    # slowest cell here, so it can only fire on a genuine hang.
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              timeout=300)
    except subprocess.TimeoutExpired:
        return None, ["TIMED OUT after 300s"]
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
