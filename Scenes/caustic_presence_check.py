#!/usr/bin/env python3
"""Scale-free test for whether a caustic FEATURE is present in a render.

Why this exists: a whole-image mean can look perfectly reasonable while the
caustic it is supposed to contain is entirely missing, and the bitterli
scenes' Tungsten references sit at a different absolute scale than pbrt's
(volumetric-caustic reads ~2x, glass-of-water ~1.8x -- see
project_dielectric_radiance_transmission_bug follow-up 8: real pbrt is
equally "off" there, so the scale difference is a scene-conversion artefact,
not a renderer bug). So absolute error cannot answer "is the caustic there".

The metric: normalise each row by its own median, giving a per-pixel
"structural excess" that is invariant to any global scale factor, then read
that excess along the caustic's known trajectory. A coherent caustic shows a
large sustained excess; noise does not.

TRAJ below is measured from volumetric-caustic's Tungsten reference (the
beam under the glass sphere, drifting left as it descends). Re-measure it
for any other scene: for each row, argmax of the row-median-normalised
luminance in the region the caustic occupies.

Usage:  python3 caustic_presence_check.py ref.exr candidate1.exr ...
"""
import os
import sys

import numpy as np
import OpenImageIO as oiio

# (row, column) samples along volumetric-caustic's beam, from the reference.
TRAJ = [(580, 366), (640, 348), (700, 331), (760, 313),
        (820, 295), (880, 277), (940, 260)]

# Excess over the row median at which a feature is judged real. The reference
# scores 5.5 and a caustic-free path-traced render 1.5, so 2.0 separates them
# with margin; 1.4 flags "something, but not the feature".
PRESENT, WEAK = 2.0, 1.4


def load(path):
    img = oiio.ImageInput.open(path)
    spec = img.spec()
    a = np.array(img.read_image(format="float"))
    a = a.reshape(spec.height, spec.width, spec.nchannels)[:, :, :3]
    img.close()
    return a


def row_excess(a):
    """Luminance divided by its own row median -- invariant to global scale."""
    lum = a.mean(2)
    return lum / (np.median(lum, axis=1, keepdims=True) + 1e-9)


def verdict(score):
    if score > PRESENT:
        return "CAUSTIC PRESENT"
    return "weak" if score > WEAK else "ABSENT"


def main():
    print("%-14s %8s %8s  %s" % ("image", "beamExc", "mean", "verdict"))
    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print("%-14s  MISSING" % os.path.basename(path))
            continue
        a = load(path)
        exc = row_excess(a)
        # widen slightly: the beam drifts, and a candidate may place it a few
        # pixels off without that meaning it is absent.
        score = float(np.mean([exc[y, x - 6:x + 7].max() for y, x in TRAJ]))
        print("%-14s %8.2f %8.4f  %s"
              % (os.path.basename(path), score, a.mean(), verdict(score)))


if __name__ == "__main__":
    main()
