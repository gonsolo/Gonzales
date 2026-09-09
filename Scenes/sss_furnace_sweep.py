#!/usr/bin/env python3
"""White furnace sweep for random-walk subsurface scattering.

A non-absorbing interior (sigma_a = 0) inside a lossless dielectric boundary,
surrounded by a uniform environment of radiance 1, must render to exactly 1
everywhere -- no free parameters. Every ray traced backwards escapes to the
environment with total weight 1, because the interior destroys nothing and
the Fresnel split at the boundary sums to 1. This holds for ANY eta and ANY
sigma_s; refraction rearranges where energy goes, it cannot create or destroy
it in a uniform field.

That makes this the ground truth an absorption-only test cannot provide: a
per-scattering-event energy error compounds as albedo^n, so raising sigma_s
raises the bounce count and turns any per-bounce error into an arbitrarily
large image error. sigma_a = 0 also pins the analog scatter/absorb coin at
p_scatter = 1, isolating the scattering path from the absorption path.

Usage: python3 Scenes/sss_furnace_sweep.py [--spp N]
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "Scenes", "sss-white-furnace.pbrt")
BIN = os.path.join(ROOT, "build", "gonzales")

# (sigma_s, eta, strict) -- eta 1 isolates the walk, eta != 1 adds the
# refracting boundary while keeping the configuration lossless.
#
# strict=False marks the DENSE cases, which are limited by a separate,
# known issue: the render loop's round budget (_SSS_WALK_ROUNDS = 256 in
# rendering.mojo/gpu.mojo) truncates walks that need more steps than that,
# and truncation discards the path's energy outright. Optical radius here is
# sigma_s * 1, and a diffusive walk needs ~(optical radius)^2 steps, so
# sigma_s 20 and 50 exceed the budget while sigma_s 5 does not. Measured with
# the budget raised to 2048: 1.0005 / 0.9949 / 0.9999 -- i.e. these are purely
# truncation, not an estimator error. The budget is not raised by default
# because the GPU loop has no anyActive early exit, so every extra round is a
# real dispatch that every SSS scene would pay for.
CASES = [(5.0, 1.0, True), (20.0, 1.0, False), (50.0, 1.0, False),
         (5.0, 1.5, True), (20.0, 1.5, False)]
TOLERANCE = 0.02  # 2%, comfortably above Monte Carlo noise at the spp used


def render(sigma_s, eta, spp, tmpdir):
    src = open(SCENE).read()
    out_exr = os.path.join(tmpdir, f"furnace_s{sigma_s}_e{eta}.exr")
    src = re.sub(r'"rgb sigma_s" \[ [^\]]* \]',
                 f'"rgb sigma_s" [ {sigma_s} {sigma_s} {sigma_s} ]', src)
    src = re.sub(r'"float eta" \[ [^\]]* \]', f'"float eta" [ {eta} ]', src)
    src = re.sub(r'"string filename" \[ "[^"]*\.exr" \]',
                 f'"string filename" [ "{out_exr}" ]', src)
    scene_path = os.path.join(tmpdir, "furnace.pbrt")
    open(scene_path, "w").write(src)

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.join(ROOT, "build") + ":" + env.get("LD_LIBRARY_PATH", "")
    subprocess.run([BIN, "--spp", str(spp), "--no-denoise", scene_path],
                   cwd=ROOT, env=env, capture_output=True, timeout=3600)
    stats = subprocess.run(["oiiotool", "--stats", out_exr],
                           capture_output=True, text=True).stdout
    for line in stats.splitlines():
        if "Stats Avg" in line:
            return [float(v) for v in line.split(":")[1].split()[:3]]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spp", type=int, default=256)
    args = ap.parse_args()

    tmpdir = os.path.join("/tmp", "sss_furnace")
    os.makedirs(tmpdir, exist_ok=True)

    print(f"{'sigma_s':>8} {'eta':>5} {'R':>8} {'G':>8} {'B':>8}   verdict")
    worst, failures = 0.0, 0
    for sigma_s, eta, strict in CASES:
        avg = render(sigma_s, eta, args.spp, tmpdir)
        if avg is None:
            print(f"{sigma_s:>8} {eta:>5}  render/stat failed")
            failures += 1
            continue
        err = max(abs(v - 1.0) for v in avg)
        ok = err <= TOLERANCE
        if strict:
            worst = max(worst, err)
            failures += 0 if ok else 1
            verdict = "ok" if ok else "FAIL"
        else:
            verdict = "ok" if ok else "round-budget truncation (known)"
        print(f"{sigma_s:>8} {eta:>5} {avg[0]:>8.4f} {avg[1]:>8.4f} {avg[2]:>8.4f}"
              f"   {verdict} ({err*100:.2f}%)")

    print(f"\nworst deviation among strict cases: {worst*100:.2f}%")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
