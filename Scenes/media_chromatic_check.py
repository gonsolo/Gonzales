#!/usr/bin/env python3
"""Per-channel centre check for Scenes/media-chromatic-absorb.pbrt.

The medium is a pure absorber (sigma_s = 0) with sigma_a = (0.25, 0.5, 1.0).
A central ray crosses a chord of 2R = 4, so per-channel optical depth is
tau = (1, 2, 4) and the analytic centre value is exp(-tau), exact, with no
free parameters. Because sigma_s = 0, the free-flight sampler's pass-through
branch is the ONLY way light reaches the camera -- nothing can mask an error.

This is the chromatic counterpart of media_passthrough_check.py, which uses a
grey medium and therefore cannot detect a chromatic error at all: every
per-channel ratio there is 1 by construction.

Usage: media_chromatic_check.py <a.exr> [b.exr ...]
"""
import math
import sys

import numpy as np
import OpenImageIO as oiio

SIGMA_A = (0.25, 0.5, 1.0)
CHORD = 4.0
EXPECT = tuple(math.exp(-s * CHORD) for s in SIGMA_A)


def centre_rgb(path, half=8):
    src = oiio.ImageInput.open(path)
    if src is None:
        raise SystemExit(f"cannot open {path}")
    spec = src.spec()
    pix = np.array(src.read_image(format="float"))
    src.close()
    pix = pix.reshape(spec.height, spec.width, spec.nchannels)[:, :, :3]
    cy, cx = spec.height // 2, spec.width // 2
    patch = pix[cy - half:cy + half, cx - half:cx + half]
    return [float(patch[:, :, c].mean()) for c in range(3)]


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print(f"analytic exp(-tau), tau = {tuple(s * CHORD for s in SIGMA_A)}:")
    print(f"  R {EXPECT[0]:.5f}   G {EXPECT[1]:.5f}   B {EXPECT[2]:.5f}\n")
    worst_all = 0.0
    for path in sys.argv[1:]:
        got = centre_rgb(path)
        ratios = [g / e for g, e in zip(got, EXPECT)]
        worst = max(abs(r - 1.0) for r in ratios)
        worst_all = max(worst_all, worst)
        # A chromatic error shows as the ratios DIVERGING from each other; a
        # uniform miss is geometric (off-axis chords) and hits all three alike.
        spread = max(ratios) / min(ratios)
        print(f"{path}")
        print(f"  got    R {got[0]:.5f}   G {got[1]:.5f}   B {got[2]:.5f}")
        print(f"  ratio  R {ratios[0]:.4f}   G {ratios[1]:.4f}   B {ratios[2]:.4f}"
              f"   | worst {worst * 100:5.1f}%   chroma spread {spread:.4f}")
    return 0 if worst_all < 0.25 else 1


if __name__ == "__main__":
    sys.exit(main())
