#!/usr/bin/env python3
"""Sweep coateddiffuse's eta, gonzales vs real pbrt-v4.

project_coateddiffuse_eta2_bug left a "~4-14% deficit that grows with eta"
open (0.955 at eta 1.2, 0.903 at 1.5, 0.861 at 2.0). This re-measures that
sweep so the residual can be confirmed, re-scoped, or closed.

There is no closed form for a layered BSDF, so pbrt renders the SAME scene
file and the comparison is a mean ratio over the central region of the quad
(edges excluded so the emitter's falloff and any silhouette pixels do not
enter the mean).

Run from the repo root:  python3 Scenes/coateddiffuse_eta_sweep.py
"""
import re
import subprocess
import sys

import numpy as np
import OpenImageIO as oiio

PBRT = "/home/gonsolo/src/pbrt-v4/gonsolo/pbrt"
SCENE = "Scenes/coateddiffuse-eta-probe.pbrt"
ETAS = (1.2, 1.5, 2.0)


def read_center(path):
    buf = oiio.ImageBuf(path)
    if buf.has_error:
        raise RuntimeError(f"{path}: {buf.geterror()}")
    a = np.asarray(buf.get_pixels(oiio.FLOAT), dtype=np.float64)[:, :, :3]
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    h, w = a.shape[:2]
    # central half only -- avoids the quad's edges entirely
    return a[h // 4:3 * h // 4, w // 4:3 * w // 4].mean()


src = open(SCENE).read()
print(f"{'eta':>5} {'gonzales':>10} {'pbrt':>10} {'ratio':>8}")
rows = []
for eta in ETAS:
    scene = re.sub(r'"float eta" \[ [^\]]* \]',
                   f'"float eta" [ {eta} ]', src)
    open("/tmp/_cd_eta.pbrt", "w").write(scene)

    subprocess.run(["./build/gonzales", "/tmp/_cd_eta.pbrt", "--no-denoise"],
                   check=True, capture_output=True)
    ours = read_center("coateddiffuse-eta-probe.exr")

    subprocess.run([PBRT, "--outfile", "/tmp/_cd_eta_pbrt.exr", "/tmp/_cd_eta.pbrt"],
                   check=True, capture_output=True)
    theirs = read_center("/tmp/_cd_eta_pbrt.exr")

    ratio = ours / theirs if theirs else float("nan")
    rows.append((eta, ours, theirs, ratio))
    print(f"{eta:5.2f} {ours:10.5f} {theirs:10.5f} {ratio:8.4f}")

if any(r[1] <= 1e-9 for r in rows):
    print("\nWARNING: a gonzales render was black -- check scene winding "
          "before trusting any ratio (testing_scene_winding_trap).")
