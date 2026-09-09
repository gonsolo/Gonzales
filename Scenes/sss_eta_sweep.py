#!/usr/bin/env python3
"""Sweep eta on a PLAIN DIELECTRIC slab (no medium) against the closed form.

A parallel-faced slab must transmit L * (1-R)/(1+R) regardless of eta: the
eta^2 radiance compression on entry is undone by the expansion on exit. If the
renderer instead applies the same-direction factor at both crossings the
result is short by exactly 1/eta^4, which this sweep detects by shape, not by
a single number.

Run from the repo root:  python3 Scenes/sss_eta_sweep.py [material]
where `material` is "dielectric" (default) or "subsurface".
"""
import re
import subprocess
import sys

import numpy as np
import OpenImageIO as oiio

MAT = sys.argv[1] if len(sys.argv) > 1 else "dielectric"
src = open("Scenes/sss-beer-lambert.pbrt").read()
if MAT == "dielectric":
    # Strip the subsurface-only params so this is a bare glass slab.
    src = src.replace('Material "subsurface"', 'Material "dielectric"')
    src = re.sub(r'"float scale" \[ [^\]]* \]', "", src)
    src = re.sub(r'"rgb sigma_s" \[ [^\]]* \]', "", src)
    src = re.sub(r'"rgb sigma_a" \[ [^\]]* \]', "", src)
else:
    # Absorption off, so the interior is transparent and only the boundary
    # transport is under test -- same closed form as the dielectric case.
    src = re.sub(r'"rgb sigma_a" \[ [^\]]* \]', '"rgb sigma_a" [ 0 0 0 ]', src)
    src = re.sub(r'"rgb sigma_s" \[ [^\]]* \]', '"rgb sigma_s" [ 0 0 0 ]', src)

print(f"material: {MAT}")
print(f"{'eta':>6} {'rendered':>10} {'analytic':>10} {'ratio':>8} {'1/eta^4':>9}")
for eta in (1.1, 1.33, 1.5, 2.0):
    scene = re.sub(r'"float eta" \[ [^\]]* \]', f'"float eta" [ {eta} ]', src)
    open("/tmp/_sss_eta.pbrt", "w").write(scene)
    subprocess.run(["./build/gonzales", "/tmp/_sss_eta.pbrt", "--no-denoise"],
                   check=True, capture_output=True)
    img = np.array(oiio.ImageInput.open("sss-beer-lambert.exr").read_image())
    got = float(img[45:55, 45:55].mean())
    R = ((eta - 1.0) / (eta + 1.0)) ** 2
    want = 60.0 * (1.0 - R) / (1.0 + R)
    print(f"{eta:6.2f} {got:10.3f} {want:10.3f} {got/want:8.4f} {1/eta**4:9.4f}")
