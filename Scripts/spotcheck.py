#!/usr/bin/env python3
"""Render a few random corpus scenes and say whether a change HELPED or HURT.

The analytic suite (analytic_check.py) measures ENERGY on flat, untextured
quads seen head-on. It is structurally blind to material fidelity -- textures,
gloss, angular BRDF variation, emissive media -- which is precisely the class
of defect a human notices immediately in a render. This closes that gap
cheaply: a handful of real scenes, after every change.

THREE-WAY, and that matters. Comparing only against the reference cannot tell
you whether a change helped; comparing only against the previous render calls
every intended change a regression. The 2026-09-21 water-caustic case is the
worked example: SPPM went 0.6028 -> 0.0355 against a pbrt reference of 0.0413,
i.e. it moved hard TOWARD the reference while looking, to the eye, like it had
lost its caustics. Only before/after/reference separates "fixed" from "broke".

    Scripts/spotcheck.py --save              # cache the current renders as BEFORE
    <make your change>
    Scripts/spotcheck.py                     # render AFTER, compare three ways

    Scripts/spotcheck.py -n 6 --mode vcm     # more scenes, a different integrator
    Scripts/spotcheck.py --scenes kroken,teapot

Renders are GPU-only, by project rule.
"""
import argparse, json, os, random, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, ".spotcheck")          # BEFORE renders + the map
MAP = os.path.join(CACHE, "scene_map.json")
REF = os.path.join(CACHE, "ref")                  # pbrt reference tiles
MODES = {"pt": [], "sppm": ["--sppm", "--sppm-photons", "600000"], "vcm": ["--vcm"]}


def tile_height(scene, width):
    xr = yr = 512
    try:
        t = open(scene, errors="replace").read()
        mx = re.search(r'"integer xresolution"\s*\[?\s*(\d+)', t)
        my = re.search(r'"integer yresolution"\s*\[?\s*(\d+)', t)
        if mx: xr = int(mx.group(1))
        if my: yr = int(my.group(1))
    except Exception:
        pass
    return max(1, yr * width // xr)


def render(scene, mode, width, out_exr, timeout):
    for f in os.listdir(REPO):
        if f.lower().endswith((".exr", ".png")) and "albedo" not in f:
            os.remove(os.path.join(REPO, f))
    h = tile_height(scene, width)
    try:
        p = subprocess.run(
            [os.path.join(REPO, "build", "gonzales"), "--gpu", "--no-denoise",
             "--spp", "64", "--resolution", f"{width}x{h}", "--seed", "1",
             *MODES[mode], scene],
            cwd=REPO, capture_output=True, text=True, timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        return None
    got = [f for f in os.listdir(REPO)
           if f.lower().endswith((".exr", ".png")) and "albedo" not in f]
    if rc != 0 or not got:
        return None
    src = os.path.join(REPO, got[0])
    subprocess.run(["oiiotool", src, "-o", out_exr], capture_output=True)
    for f in got:
        q = os.path.join(REPO, f)
        if os.path.exists(q):
            os.remove(q)
    return out_exr if os.path.exists(out_exr) else None


def stats(path):
    """Cheap, human-meaningful summary: brightness and colour balance."""
    import numpy as np, OpenImageIO as oiio
    i = oiio.ImageInput.open(path)
    if not i:
        return None
    a = i.read_image("float")[..., :3]
    return dict(mean=float(a.mean()),
                r=float(a[..., 0].mean()), g=float(a[..., 1].mean()),
                b=float(a[..., 2].mean()), shape=a.shape[:2])


def dist(x, y):
    """How far apart two renders are: relative brightness + colour-ratio error."""
    if x is None or y is None or x["shape"] != y["shape"]:
        return None
    db = abs(x["mean"] - y["mean"]) / max(y["mean"], 1e-6)
    rg = abs(x["r"] / max(x["g"], 1e-6) - y["r"] / max(y["g"], 1e-6))
    bg = abs(x["b"] / max(x["g"], 1e-6) - y["b"] / max(y["g"], 1e-6))
    return db + rg + bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="cache the current renders as the BEFORE baseline")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--mode", default="sppm", choices=sorted(MODES))
    ap.add_argument("--scenes", default="", help="comma-separated names")
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=None,
                    help="pick the same random scenes again")
    a = ap.parse_args()

    if not os.path.isfile(MAP):
        print(f"no scene map at {MAP}\n"
              f"create it with a name->scene-file JSON, e.g. from the gallery.",
              file=sys.stderr)
        return 2
    smap = json.load(open(MAP))
    if a.scenes:
        names = [s.strip() for s in a.scenes.split(",") if s.strip() in smap]
    else:
        rng = random.Random(a.seed)
        names = rng.sample(sorted(smap), min(a.n, len(smap)))

    before_dir = os.path.join(CACHE, a.mode)
    os.makedirs(before_dir, exist_ok=True)
    tmp = os.path.join(CACHE, "tmp")
    os.makedirs(tmp, exist_ok=True)

    if a.save:
        for n in names:
            sc = smap[n] if os.path.isabs(smap[n]) else os.path.join(REPO, smap[n])
            ok = render(sc, a.mode, a.width, os.path.join(before_dir, n + ".exr"), a.timeout)
            print(f"  cached {n}" if ok else f"  FAILED {n}")
        print(f"baseline cached in {before_dir}")
        return 0

    print(f"spot check ({a.mode}, {a.width}px) -- three-way: after vs BEFORE vs pbrt ref\n")
    print("  %-24s %9s %9s   %s" % ("scene", "d(before)", "d(ref)", "verdict"))
    worse = []
    for n in names:
        sc = smap[n] if os.path.isabs(smap[n]) else os.path.join(REPO, smap[n])
        now = render(sc, a.mode, a.width, os.path.join(tmp, n + ".exr"), a.timeout)
        if now is None:
            print("  %-24s %9s %9s   RENDER FAILED" % (n, "-", "-"))
            worse.append(n)
            continue
        s_now = stats(now)
        b = os.path.join(before_dir, n + ".exr")
        r = os.path.join(REF, n + ".exr")
        s_bef = stats(b) if os.path.exists(b) else None
        s_ref = stats(r) if os.path.exists(r) else None
        d_bef = dist(s_now, s_bef) if s_bef else None
        d_ref_now = dist(s_now, s_ref) if s_ref else None
        d_ref_bef = dist(s_bef, s_ref) if (s_bef and s_ref) else None
        if d_bef is None:
            verdict = "no baseline (run --save first)"
        elif d_bef < 1e-4:
            verdict = "unchanged"
        elif d_ref_now is None or d_ref_bef is None:
            verdict = "CHANGED, no reference to judge against"
        elif d_ref_now < d_ref_bef:
            verdict = "changed, TOWARD reference  (good)"
        else:
            verdict = "changed, AWAY from reference  (LOOK)"
            worse.append(n)
        print("  %-24s %9s %9s   %s" % (
            n,
            "-" if d_bef is None else f"{d_bef:.4f}",
            "-" if d_ref_now is None else f"{d_ref_now:.4f}",
            verdict))
    if worse:
        print("\n  inspect: " + ", ".join(worse))
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main())
