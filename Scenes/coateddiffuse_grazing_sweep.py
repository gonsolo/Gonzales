#!/usr/bin/env python3
"""Sweep the camera's elevation over a coateddiffuse plane, gonzales vs pbrt-v4.

Companion to coateddiffuse_eta_sweep.py. That sweep varies eta at near-normal
incidence and finds a ~5% deficit at eta 1.5. staircase2's floor is the same
material and the same eta but reads 0.62-0.71 of pbrt at the bottom of frame,
so something other than eta is carrying most of that gap. This sweep varies
only the VIEW elevation -- illumination is a distant light straight down, so
irradiance is constant and the light-side geometry never changes.

Run from the repo root:  python3 Scenes/coateddiffuse_grazing_sweep.py
"""
import math
import re
import subprocess
import sys

import numpy as np
import OpenImageIO as oiio

PBRT = "/home/gonsolo/src/pbrt-v4/gonsolo/pbrt"
SCENE = "Scenes/coateddiffuse-grazing-probe.pbrt"
OUT = "coateddiffuse-grazing-probe.exr"
DIST = 6.0
# Elevation of the camera above the plane, degrees. 60 is near-normal viewing,
# 4 is close to edge-on.
ELEVATIONS = (60.0, 30.0, 15.0, 8.0, 4.0)


def read_center(path):
    buf = oiio.ImageBuf(path)
    if buf.has_error:
        raise RuntimeError(f"{path}: {buf.geterror()}")
    a = buf.get_pixels(oiio.FLOAT)
    a = np.array(a)[:, :, :3]
    h, w, _ = a.shape
    # Central half only: keeps horizon pixels and any silhouette out of the mean.
    return a[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]


def write_scene(elev_deg):
    src = open(SCENE).read()
    a = math.radians(elev_deg)
    eye_y = DIST * math.sin(a)
    eye_z = DIST * math.cos(a)
    src = src.replace("EYEY", f"{eye_y:.6f}").replace("EYEZ", f"{eye_z:.6f}")
    tmp = "/tmp/g2/grazing_tmp.pbrt"
    open(tmp, "w").write(src)
    return tmp


def main():
    print(f"{'elev':>6} {'gonzales':>10} {'pbrt':>10} {'ratio':>8}")
    rows = []
    for elev in ELEVATIONS:
        tmp = write_scene(elev)
        subprocess.run(
            ["./build/gonzales", "--spp", "512", "--no-denoise", tmp],
            check=True, capture_output=True,
        )
        g = read_center(OUT).mean()
        subprocess.run(
            [PBRT, "--outfile", "/tmp/g2/grazing_pbrt.exr", "--spp", "512", tmp],
            check=True, capture_output=True,
        )
        p = read_center("/tmp/g2/grazing_pbrt.exr").mean()
        # A black render makes any ratio meaningless (testing_scene_winding_trap).
        if g <= 1e-6 or p <= 1e-6:
            sys.exit(f"elevation {elev}: black render (gonzales {g:g}, pbrt {p:g})")
        print(f"{elev:6.0f} {g:10.5f} {p:10.5f} {g / p:8.4f}")
        rows.append((elev, g, p, g / p))
    return rows


if __name__ == "__main__":
    main()
