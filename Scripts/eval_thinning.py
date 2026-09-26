#!/usr/bin/env python3
"""Equal-time evaluation of VCM merge thinning with and without the keep-aware
MIS -- step 1 of the "bounded-cost vertex merging" study.

Variants (one build, switched on the command line):
  A  no thinning        --vcm-cap 1073741824
  B  thinning only      --vcm-no-keep-mis   (Kern-style cap, MIS unaware)
  C  thinning + MIS     (default)
"Fewer photons" needs no variant of its own: every sample is a pass of light
paths, so A at fewer samples IS uniformly fewer photons, and the equal-time
curves compare it directly.

All three are unbiased with the same expectation, so the reference is C at many
samples with its own seed -- our renderer, not pbrt, whose unclamped
path-traced references are firefly-dominated exactly where this matters
(barcelona-pavilion-night). Error is relMSE; time is the RENDER time from
gonzales's progress line, without scene load.

    Scripts/eval_thinning.py              render what is missing, then report
    Scripts/eval_thinning.py --report     report only
Outputs go to ~/renders/thinning-eval/ (outside the repo: no images in git).
"""

import csv, os, re, subprocess, sys, time
import numpy as np
import OpenImageIO as oiio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.expanduser("~/renders/thinning-eval")
GONZ = os.path.join(REPO, "build", "gonzales")
ENV = dict(os.environ, LD_LIBRARY_PATH=os.path.join(REPO, "build"),
           GONZALES_DATA_DIR=os.path.join(REPO, "src", "gonzales", "data"))
PAV = os.path.expanduser("~/src/pbrt-v4-scenes/barcelona-pavilion")

# name -> (scene file, working dir, reference spp); concentrated-cost scenes
# first, then the neutral controls.
SCENES = {
    "lantern":        ("lantern.pbrt",     os.path.join(REPO, "Scenes/thinning"), 1024),
    "lampshade":      ("lampshade.pbrt",   os.path.join(REPO, "Scenes/thinning"), 1024),
    "candle-jar":     ("candle-jar.pbrt",  os.path.join(REPO, "Scenes/thinning"), 1024),
    "chandelier":     ("chandelier.pbrt",  os.path.join(REPO, "Scenes/thinning"), 1024),
    "headlight":      ("headlight.pbrt",   os.path.join(REPO, "Scenes/thinning"), 1024),
    "pavilion-night": ("zz_night_640.pbrt", PAV, 512),
    "twolights":      ("twolights.pbrt",   os.path.join(REPO, "Scenes/thinning"), 1024),
    "cornell-box":    ("cornell-box.pbrt", os.path.join(REPO, "Scenes"), 1024),
}
VARIANTS = {"A": ["--vcm-cap", "1073741824"], "B": ["--vcm-no-keep-mis"], "C": []}
SPP = {"A": [4, 16], "B": [4, 16, 64], "C": [4, 16, 64]}
SEEDS = [1, 2]
REF_SEED = 99


def fmt_seconds(s):
    m = re.match(r"(?:(\d+)m )?([\d.]+)s", s)
    return (int(m.group(1)) * 60 if m.group(1) else 0) + float(m.group(2))


def render(scene, variant, spp, seed, out):
    path, cwd, _ = SCENES[scene]
    produced = os.path.join(cwd, re.search(r'"string filename" \[ "([^"]+)"',
                                           open(os.path.join(cwd, path)).read()).group(1))
    cmd = [GONZ, "--gpu", "--no-denoise", "--vcm", "--spp", str(spp), "--seed", str(seed),
           *VARIANTS[variant], path]
    t0 = time.time()
    log = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True).stdout
    wall = time.time() - t0
    done = re.findall(r"Done: ([\dms. ]+?)\s{2,}", log.replace("\r", "\n"))
    rtime = fmt_seconds(done[-1].strip()) if done else wall
    os.replace(produced, out)
    stem = os.path.splitext(produced)[0]
    for side in [os.path.join(cwd, "albedo.exr")] + [f"{stem}.{s}.exr" for s in ("albedo", "normal", "depth", "noisy")]:
        if os.path.exists(side):
            os.remove(side)
    return rtime, wall


def load(p):
    i = oiio.ImageInput.open(p); s = i.spec()
    a = np.array(i.read_image(0, 0, 0, 3, "float")).reshape(s.height, s.width, 3); i.close()
    return a


def run_all():
    os.makedirs(OUT, exist_ok=True)
    times_path = os.path.join(OUT, "times.csv")
    times = {}
    if os.path.exists(times_path):
        for r in csv.DictReader(open(times_path)):
            times[r["file"]] = (float(r["render_s"]), float(r["wall_s"]))
    # Warm the GPU kernel cache once so no measured run pays the JIT.
    render("cornell-box", "C", 1, 1, os.path.join(OUT, "_warmup.exr"))
    jobs = []
    for sc, (_, _, ref_spp) in SCENES.items():
        jobs.append((sc, "C", ref_spp, REF_SEED, f"{sc}_ref.exr"))
        for v, spps in SPP.items():
            for spp in spps:
                for seed in SEEDS:
                    jobs.append((sc, v, spp, seed, f"{sc}_{v}_{spp}_{seed}.exr"))
    for n, (sc, v, spp, seed, fn) in enumerate(jobs):
        out = os.path.join(OUT, fn)
        if os.path.exists(out) and fn in times:
            continue
        rt, wall = render(sc, v, spp, seed, out)
        times[fn] = (rt, wall)
        with open(times_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file", "render_s", "wall_s"])
            for k, (a, b) in sorted(times.items()):
                w.writerow([k, f"{a:.3f}", f"{b:.3f}"])
        print(f"[{n + 1}/{len(jobs)}] {fn}: render {rt:.1f} s", flush=True)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    times = {r["file"]: float(r["render_s"]) for r in csv.DictReader(open(os.path.join(OUT, "times.csv")))}
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, sc in zip(axes.ravel(), SCENES):
        refp = os.path.join(OUT, f"{sc}_ref.exr")
        if not os.path.exists(refp):
            ax.set_title(sc + " (no reference yet)"); continue
        ref = load(refp); lum = ref.mean(2)
        for v, spps in SPP.items():
            xs, ys = [], []
            for spp in spps:
                es, ts = [], []
                for seed in SEEDS:
                    fn = f"{sc}_{v}_{spp}_{seed}.exr"
                    p = os.path.join(OUT, fn)
                    if not os.path.exists(p) or fn not in times:
                        continue
                    a = load(p)
                    es.append((((a - ref) ** 2).mean(2) / (lum ** 2 + 1e-2)).mean())
                    ts.append(times[fn])
                if es:
                    e, t = float(np.mean(es)), float(np.mean(ts))
                    xs.append(t); ys.append(e)
                    rows.append([sc, v, spp, f"{t:.3f}", f"{e:.5f}", f"{1.0 / (e * t):.2f}"])
            if xs:
                ax.loglog(xs, ys, "o-", label={"A": "A no thinning", "B": "B thinning only",
                                                 "C": "C thinning + MIS"}[v])
        ax.set_title(sc); ax.set_xlabel("render time [s]"); ax.set_ylabel("relMSE"); ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "curves.png"), dpi=110)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "variant", "spp", "render_s", "relMSE", "efficiency"]); w.writerows(rows)
    for r in rows:
        print("%-15s %s %4s spp  %8s s  relMSE %s  eff %s" % tuple(r))


if __name__ == "__main__":
    if "--report" not in sys.argv:
        run_all()
    report()
