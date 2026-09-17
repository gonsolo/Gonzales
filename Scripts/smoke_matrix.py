#!/usr/bin/env python3
"""The {integrator} x {feature} smoke matrix.

`make smoketest`'s per-mode rows only ever grew when somebody remembered to
add one: SPPM had no row at all until three real bugs in it had already
shipped. This runs every feature scene under every integrator instead, and
treats a MISSING cell as a failure, so a combination cannot go untested
quietly. Each cell pins two statistics: the image mean, and the median
luminance. The median alone misses a localized change (one caustic, one
object); the mean alone is swamped in a scene whose emitters fill the frame
(sss-backlit-slab's every pixel is brighter than 1). Together they catch
both.

The integrators legitimately disagree on absolute level in some scenes (VCM
reads well under the path tracer in a closed box at low maxdepth, the
documented light-path-budget x depth-truncation gap), so each cell has its
OWN pinned value. A cell can also be marked unsupported with a reason, which
is a recorded decision rather than a silent hole.

    smoke_matrix.py            # check every cell (what make smoketest runs)
    smoke_matrix.py --update   # re-render and rewrite the pinned values
    smoke_matrix.py --only sss # just the features whose name contains "sss"

Renders are GPU-only, by project rule.
"""
import argparse, json, os, re, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_matrix.json")
MODES = {"pt": [], "vcm": ["--vcm"], "sppm": ["--sppm"]}
ARGS = ["--gpu", "--no-denoise", "--spp", "64", "--resolution", "48x48", "--seed", "1"]
# feature -> the scene that exercises it. One line per thing that can break on
# its own; a scene may legitimately cover more than one.
FEATURES = {
    "baseline-diffuse":   "Scenes/cornell-box.pbrt",
    "subsurface-coated":  "Scenes/cornell-box-subsurface.pbrt",
    "normal-map":         "Scenes/cornell-box-normalmap.pbrt",
    "analytic-sphere":    "Scenes/diffuse-sphere-vcm.pbrt",
    "chromatic-medium":   "Scenes/media-chromatic-scatter.pbrt",
    "medium-sphere-light":"Scenes/vcm-media-sphere-light.pbrt",
    "closed-cavity":      "Scenes/closed-cavity-equilibrium.pbrt",
    "curves-hair":        "Scenes/glowing_hair.pbrt",
    "many-lights":        "Scenes/restir-manylights.pbrt",
    "mnee-through-glass": "Scenes/mnee-two-lights.pbrt",
    "sss-slab":           "Scenes/sss-backlit-slab.pbrt",
}
DEFAULT_TOL = 0.03


def render_stats(scene, mode):
    """Render one cell and return (mean, median luminance), or None if it died."""
    import numpy as np
    import OpenImageIO as oiio
    start = time.time()
    cmd = [os.path.join(REPO, "build", "gonzales"), *ARGS, *MODES[mode], scene]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    written = [f for f in os.listdir(REPO)
               if f.endswith(".exr") and os.path.getmtime(os.path.join(REPO, f)) >= start]
    beauty = [f for f in written if "albedo" not in f]
    try:
        if proc.returncode != 0:
            return None, (proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["exited %d" % proc.returncode]
        if len(beauty) != 1:
            return None, ["expected one EXR, got %s" % written]
        img = oiio.ImageInput.open(os.path.join(REPO, beauty[0])).read_image("float")[..., :3]
        if not np.isfinite(img).all():
            return None, ["%d non-finite samples" % int((~np.isfinite(img)).sum())]
        return (float(img.mean()), float(np.median(img.mean(axis=2)))), []
    finally:
        for f in written:
            path = os.path.join(REPO, f)
            if os.path.exists(path):
                os.remove(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
    features = {k: v for k, v in FEATURES.items() if args.only in k}
    failures = []
    for feature, scene in sorted(features.items()):
        for mode in MODES:
            cell = f"{feature}.{mode}"
            pin = pins.get(cell)
            if isinstance(pin, dict) and "unsupported" in pin:
                print(f"  --   {cell:34s} unsupported: {pin['unsupported']}")
                continue
            got, why = render_stats(os.path.join(REPO, scene), mode)
            if got is None:
                print(f"  FAIL {cell:34s} {'; '.join(why)}")
                failures.append(cell)
                continue
            mean, median = got
            if args.update:
                pins[cell] = {"mean": round(mean, 6), "median": round(median, 6),
                              "tol": (pin or {}).get("tol", DEFAULT_TOL)}
                print(f"  set  {cell:34s} mean {mean:.6f}  median {median:.6f}")
                continue
            if pin is None:
                print(f"  FAIL {cell:34s} no pinned value -- this combination is untested. "
                      f"Run --update, or pin it unsupported with a reason.")
                failures.append(cell)
                continue
            drift = []
            for name, value in (("mean", mean), ("median", median)):
                want = pin[name]
                off = value / want - 1.0 if want else (0.0 if value == 0 else float("inf"))
                if abs(off) > pin["tol"]:
                    drift.append(f"{name} {value:.6f}, expected {want:.6f} ({off:+.1%})")
            if drift:
                print(f"  FAIL {cell:34s} {'; '.join(drift)} (tol {pin['tol']:.0%})")
                failures.append(cell)
            else:
                print(f"  ok   {cell:34s} mean {mean:.6f}  median {median:.6f}")

    if args.update:
        json.dump(dict(sorted(pins.items())), open(PINS, "w"), indent=2)
        open(PINS, "a").write("\n")
        print(f"wrote {PINS}")
    if failures:
        print(f"smoke matrix FAILED: {len(failures)} cell(s): {', '.join(failures)}")
        return 1
    print(f"smoke matrix passed ({len(features)} features x {len(MODES)} integrators)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
