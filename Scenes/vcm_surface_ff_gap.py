#!/usr/bin/env python3
"""Quantify the free-flight gap in the SHIPPED surface-only MIS recursion.

`Scenes/vcm_volume_mis_derivation.py` verifies the CORRECT recursion (free
flight in the carries) to ~1e-16. This script isolates what the shipped
bdpt.mojo actually does, which is the same recursion with every free-flight
factor dropped: surface arrival is `dVCM *= d*d` (bdpt.mojo:1798) followed by
`/= cos_fix`, and the scatter update carries no `ff_a/ff_b` term.

For an ALL-SURFACE path the two differ in exactly one place. `ff_a/ff_b` is
`exp(-sigma_t d)/exp(-sigma_t d) = 1`, so dVC is untouched; but dVCM takes
`1/ff_b = exp(+sigma_t d)` per edge, which the shipped code omits. The error
is therefore identically zero in vacuum and grows exponentially with optical
depth -- and it is invisible today only because volume vertices are excluded
from MIS entirely, so a path must reach a *surface through a medium* with
several strategies competing before it is expressed.

Usage: vcm_surface_ff_gap.py
"""
import importlib.util
import math
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "vcmderiv", HERE / "vcm_volume_mis_derivation.py")
D = importlib.util.module_from_spec(spec)
sys.modules["vcmderiv"] = D
spec.loader.exec_module(D)


def carries_no_ff(xs, sigma_t, p_start, upto, from_camera):
    """The shipped recursion: identical to the reference, minus free flight."""
    n = len(xs) - 1
    dVCM, dVC = 1.0 / p_start, 0.0
    idxs = range(n, upto, -1) if from_camera else range(0, upto)
    for i in idxs:
        j = i - 1 if from_camera else i + 1
        a, b = xs[i], xs[j]
        dvec = D.sub(b.pos, a.pos)
        d = D.norm(dvec)
        w = D.unit(dvec)
        pdf_dir = D.p_dir(a, w)
        first = (i == n) if from_camera else (i == 0)
        if first:
            pdf_rev = 0.0
        else:
            k = i + 1 if from_camera else i - 1
            pdf_rev = D.p_dir(a, D.unit(D.sub(xs[k].pos, a.pos)))
        cos_out = abs(D.dot(a.n, w)) if a.is_surface else 1.0
        dVC = (cos_out / pdf_dir) * (dVC * pdf_rev + dVCM)
        dVCM = 1.0 / pdf_dir
        # --- the omission: no `/ ff_b` on dVCM, no `* ff_a/ff_b` on dVC ---
        dVCM *= d * d
        if b.is_surface:
            cos_fix = abs(D.dot(b.n, w))
            dVCM /= cos_fix
            dVC /= cos_fix
    return dVCM, dVC


def weight_no_ff(xs, s, sigma_t, p_light_area, p_cam_area):
    n = len(xs) - 1
    lv, cv = xs[s - 1], xs[s]
    dvec = D.sub(cv.pos, lv.pos)
    d = D.norm(dvec)
    w_l2c = D.unit(dvec)
    w_c2l = (-w_l2c[0], -w_l2c[1], -w_l2c[2])
    lv_dVCM, lv_dVC = carries_no_ff(xs, sigma_t, p_light_area, s - 1, False)
    cv_dVCM, cv_dVC = carries_no_ff(xs, sigma_t, p_cam_area, s, True)
    cam_dir_pdf_w = D.p_dir(cv, w_c2l)
    lig_dir_pdf_w = D.p_dir(lv, w_l2c)
    lig_rev = 0.0 if s - 1 == 0 else D.p_dir(lv, D.unit(D.sub(xs[s - 2].pos, lv.pos)))
    cam_rev = 0.0 if s == n else D.p_dir(cv, D.unit(D.sub(xs[s + 1].pos, cv.pos)))
    # connect-time conversion, also without free flight
    cam_dir_pdf_a = cam_dir_pdf_w * D.geom_to_measure(lv, d, w_c2l)
    lig_dir_pdf_a = lig_dir_pdf_w * D.geom_to_measure(cv, d, w_l2c)
    w_light = cam_dir_pdf_a * (lv_dVCM + lv_dVC * lig_rev)
    w_camera = lig_dir_pdf_a * (cv_dVCM + cv_dVC * cam_rev)
    return 1.0 / (w_light + 1.0 + w_camera)


def main():
    kinds_list = [("S", "S", "S", "S"), ("S", "S", "S", "S", "S"),
                  ("S", "S", "S", "S", "S", "S")]
    print("ALL-SURFACE paths: shipped (no free flight) vs brute-force ground truth\n")
    print(f"{'sigma_t':>8} {'worst rel err: SHIPPED':>24} {'REFERENCE (with ff)':>22}")
    for sigma_t in (0.0, 0.1, 0.5, 1.0, 2.0, 4.0):
        worst_ship = worst_ref = 0.0
        rng = random.Random(20260910)
        for kinds in kinds_list:
            for _ in range(40):
                xs = D.make_path(kinds, rng)
                pl, pc = 0.37, 0.53
                w_bf, _ = D.brute_force_weights(xs, sigma_t, pl, pc)
                for s in range(1, len(xs)):
                    ref = D.recursion_weight(xs, s, sigma_t, pl, pc)
                    ship = weight_no_ff(xs, s, sigma_t, pl, pc)
                    denom = max(w_bf[s], 1e-30)
                    worst_ref = max(worst_ref, abs(ref - w_bf[s]) / denom)
                    worst_ship = max(worst_ship, abs(ship - w_bf[s]) / denom)
        print(f"{sigma_t:8g} {worst_ship:24.3e} {worst_ref:22.3e}")
    print("\nThe reference stays at machine epsilon at every density; the shipped")
    print("recursion is exact only in vacuum. dVC is unaffected for an all-surface")
    print("path (ff_a/ff_b = 1); the entire gap is dVCM's missing 1/ff_b per edge.")


if __name__ == "__main__":
    main()
