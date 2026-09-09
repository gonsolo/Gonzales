#!/usr/bin/env python3
"""Sweep sigma_a on the Beer-Lambert slab and compare against the closed form.

Validates the subsurface boundary + interior transport: with sigma_s = 0 the
answer is L * (1-R)/(1+R) * exp(-sigma_a * thickness), no free parameters.
Run from the repo root:  python3 Scenes/sss_sweep.py
"""
import math
import re
import subprocess
import sys

import numpy as np
import OpenImageIO as oiio

SCENE = "Scenes/sss-beer-lambert.pbrt"
OUT = "sss-beer-lambert.exr"
L = 60.0
THICK = 0.1
ETA = 1.33
R = ((ETA - 1.0) / (ETA + 1.0)) ** 2
FRESNEL = (1.0 - R) / (1.0 + R)

src = open(SCENE).read()
print(f"  R={R:.6f}  Fresnel slab factor=(1-R)/(1+R)={FRESNEL:.5f}")
print(f"{'sigma_a':>8} {'predicted':>10} {'rendered':>10} {'ratio':>7}")

worst = 0.0
for sa in (0.0, 2.0, 5.0, 10.0, 20.0):
    scene = re.sub(r'"rgb sigma_a" \[ [^\]]* \]',
                   f'"rgb sigma_a" [ {sa} {sa} {sa} ]', src)
    open("/tmp/_sss_sweep.pbrt", "w").write(scene)
    subprocess.run(["./build/gonzales", "/tmp/_sss_sweep.pbrt", "--no-denoise"],
                   check=True, capture_output=True)
    img = np.array(oiio.ImageInput.open(OUT).read_image())
    # Central 10x10 only: those rays hit the slab at near-normal incidence,
    # which is what the closed form above assumes.
    got = float(img[45:55, 45:55].mean())
    want = L * FRESNEL * math.exp(-sa * THICK)
    ratio = got / want
    worst = max(worst, abs(ratio - 1.0))
    print(f"{sa:8.1f} {want:10.3f} {got:10.3f} {ratio:7.4f}")

print(f"\nworst deviation from the analytic answer: {worst*100:.2f}%")
sys.exit(0 if worst < 0.05 else 1)
