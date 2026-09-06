#!/usr/bin/env python3
"""Validate a rendered EXR: exists, finite, and actually has light in it.

Used by `make smoketest`. Catches the failure mode unit tests structurally
cannot: an entire render mode dying (crash, all-NaN, or all-black) while
every pure-function test still passes. --gpu --vcm was dead for months
exactly that way.
"""
import sys
import numpy as np
import OpenImageIO as oiio

def main() -> int:
    if len(sys.argv) < 3:
        print("usage: check_render.py <label> <file.exr>", file=sys.stderr)
        return 2
    label, path = sys.argv[1], sys.argv[2]
    src = oiio.ImageInput.open(path)
    if src is None:
        print(f"FAIL {label}: cannot open {path}")
        return 1
    spec = src.spec()
    a = np.array(src.read_image(0, 0, 0, 3, "float")).reshape(
        spec.height, spec.width, 3)
    src.close()
    n_nan = int(np.isnan(a).sum())
    n_inf = int(np.isinf(a).sum())
    mean = float(np.nan_to_num(a).mean())
    if n_nan or n_inf:
        print(f"FAIL {label}: {n_nan} NaN, {n_inf} Inf")
        return 1
    if not mean > 1e-6:
        print(f"FAIL {label}: image is black (mean {mean:.3g})")
        return 1
    print(f"  ok   {label:22s} mean={mean:.5f}  {spec.width}x{spec.height}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
