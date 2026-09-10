#!/usr/bin/env python3
"""Centre-region check for Scenes/media-passthrough-absorb.pbrt.

The medium is a pure absorber (sigma_s = 0), so the only way light reaches the
camera is the free-flight sampler's PASS-THROUGH branch. A central ray crosses
a chord of 2R = 4 through sigma_a = 0.5, i.e. tau = 2, in front of an L = 1
emitter, so the analytic centre value is exp(-tau) with no free parameters.

Double-counting the sampled channel's transmittance (the pre-2026-09-10 bug in
sppm.mojo::sample_homogeneous_free_flight) instead yields exp(-2*tau), i.e.
too dark by exactly exp(tau).

Usage: media_passthrough_check.py <a.exr> [b.exr ...]
"""
import math
import sys

import numpy as np
import OpenImageIO as oiio

TAU = 2.0


def centre_mean(path, half=8):
    src = oiio.ImageInput.open(path)
    if src is None:
        raise SystemExit(f"cannot open {path}")
    spec = src.spec()
    pix = np.array(src.read_image(format="float"))
    src.close()
    pix = pix.reshape(spec.height, spec.width, spec.nchannels)[:, :, :3]
    cy, cx = spec.height // 2, spec.width // 2
    return float(pix[cy - half:cy + half, cx - half:cx + half].mean())


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    good, bad = math.exp(-TAU), math.exp(-2 * TAU)
    print(f"analytic: correct exp(-tau) = {good:.5f}   "
          f"double-counted exp(-2tau) = {bad:.5f}   ratio = {math.exp(TAU):.3f}x\n")
    for path in sys.argv[1:]:
        val = centre_mean(path)
        verdict = "CORRECT" if abs(val / good - 1.0) < 0.05 else (
            "DOUBLE-COUNTED" if abs(val / bad - 1.0) < 0.10 else "?")
        print(f"  {path:36} centre={val:.5f}  "
              f"vs correct {val / good:6.3f}x  vs double-counted {val / bad:8.3f}x  {verdict}")


if __name__ == "__main__":
    main()
