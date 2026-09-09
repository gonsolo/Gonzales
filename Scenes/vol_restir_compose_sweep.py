#!/usr/bin/env python3
"""Measure volumetric ReSTIR: temporal reuse and distance resampling, alone
and composed.

Distance resampling is a COMPTIME flag (restir_vol.mojo's VOL_RIS_DISTANCE),
so the two "distance on" configurations need a separate build. This script
measures whichever pair the current binary supports and writes JSON; run it
once per build and diff the two files.

Usage:
    vol_restir_compose_sweep.py <tag> [--ref-spp N] [--spp N] [--seeds N]
"""
import argparse, json, os, subprocess, sys
import numpy as np
import OpenImageIO as oiio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "gonzales")
SCENE = ""
OUT = ""


def render(spp, seed, reuse, extra_env=None):
    cmd = [BIN, "--gpu", "--no-denoise", "--spp", str(spp), "--seed", str(seed)]
    if reuse:
        cmd.append("--vol-restir-reuse")
    cmd.append(SCENE)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.join(ROOT, "build") + ":" + env.get("LD_LIBRARY_PATH", "")
    if extra_env:
        env.update(extra_env)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=7200)
    if r.returncode != 0:
        sys.exit(f"render failed ({' '.join(cmd)}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    # Surface any renderer warning rather than letting it scroll past --
    # a silently-dropped asset has bitten this project five times.
    for line in (r.stdout + r.stderr).splitlines():
        low = line.lower()
        if ("fail" in low or "warn" in low or "unsupported" in low) and "0 texture" not in low:
            print(f"    [renderer] {line.strip()}", file=sys.stderr)
    img = oiio.ImageInput.open(os.path.join(ROOT, OUT))
    px = np.array(img.read_image(format="float"))
    img.close()
    return px.reshape(-1, px.shape[-1])[:, :3]


def main():
    global SCENE, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--scene", default="vol-restir-thin-fog")
    ap.add_argument("--ref-spp", type=int, default=65536)
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--bias-spp", type=int, default=4096)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    SCENE = os.path.join(ROOT, "Scenes", args.scene + ".pbrt")
    OUT = args.scene + ".exr"

    print(f"[{args.tag}] reference at {args.ref_spp} spp (features OFF -- the plain 7.2 estimator)")
    ref = render(args.ref_spp, 1, reuse=False)
    ref_mean = float(ref.mean())
    print(f"    reference mean {ref_mean:.6g}")

    out = {"tag": args.tag, "ref_spp": args.ref_spp, "ref_mean": ref_mean,
           "spp": args.spp, "configs": {}}

    for reuse in (False, True):
        name = "reuse" if reuse else "base"
        mses, means = [], []
        for s in range(1, args.seeds + 1):
            img = render(args.spp, s, reuse=reuse)
            mses.append(float(((img - ref) ** 2).mean()))
            means.append(float(img.mean()))
        bias_img = render(args.bias_spp, 1, reuse=reuse)
        bias_ratio = float(bias_img.mean() / ref_mean)
        out["configs"][name] = {
            "mse_per_seed": mses,
            "mse_mean": float(np.mean(mses)),
            "mean_per_seed": means,
            "bias_spp": args.bias_spp,
            "bias_ratio": bias_ratio,
        }
        print(f"    {name:5s} MSE {np.mean(mses):.6g}  "
              f"[{min(mses):.6g} .. {max(mses):.6g}]  "
              f"bias@{args.bias_spp} {bias_ratio:.5f}")

    path = os.path.join(ROOT, f"vol_compose_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"    wrote {path}")


if __name__ == "__main__":
    main()
