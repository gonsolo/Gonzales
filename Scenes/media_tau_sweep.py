#!/usr/bin/env python3
"""PT-vs-VCM agreement across an optical-depth sweep on a media scene.

Reference-free by construction: the plain path tracer and VCM are two
independent estimators of the same integral, so their ratio should be 1 at
every density. Sweeping matters because a previous VCM+media bug looked like
it scaled with tau and turned out to be a clean factor of 2 at first
scattering order.

Usage: media_tau_sweep.py <scene.pbrt> [sigma_s values ...]
"""
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import OpenImageIO as oiio

REPO = pathlib.Path(__file__).resolve().parent.parent
BIN = REPO / "build" / "gonzales"
ENV = dict(os.environ, LD_LIBRARY_PATH=str(REPO / "build"))


def mean_of(path):
    src = oiio.ImageInput.open(str(path))
    if src is None:
        return None
    spec = src.spec()
    pix = np.array(src.read_image(format="float"))
    src.close()
    return float(pix.reshape(spec.height, spec.width, spec.nchannels)[:, :, :3].mean())


def render(scene, mode):
    out = REPO / (re.search(r'"string filename"\s*\[\s*"([^"]+)"',
                            scene.read_text()).group(1))
    if out.exists():
        out.unlink()
    cmd = [str(BIN)] + ([mode] if mode else []) + [str(scene)]
    subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, timeout=2400)
    if not out.exists():
        return None
    val = mean_of(out)
    out.unlink()
    return val


def main():
    base = pathlib.Path(sys.argv[1])
    sigmas = [float(x) for x in sys.argv[2:]] or [None]
    text = base.read_text()
    work = REPO / "build" / "tau_sweep.pbrt"
    print(f"{'sigma_s':>9} {'PT':>10} {'VCM':>10} {'VCM/PT':>8}")
    for s in sigmas:
        if s is None:
            work.write_text(text)
            label = "scene"
        else:
            work.write_text(re.sub(r'"rgb sigma_s"\s*\[[^\]]*\]',
                                   f'"rgb sigma_s" [ {s} {s} {s} ]', text))
            label = f"{s:g}"
        pt = render(work, "")
        vcm = render(work, "--vcm")
        if pt is None or vcm is None or pt == 0:
            print(f"{label:>9} {'--':>10} {'--':>10} {'FAILED':>8}")
            continue
        print(f"{label:>9} {pt:10.6f} {vcm:10.6f} {vcm / pt:8.4f}")


if __name__ == "__main__":
    main()
