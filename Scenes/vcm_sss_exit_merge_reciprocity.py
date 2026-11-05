#!/usr/bin/env python3
"""Is VCM with exit merging unbiased when the walk's densities, or its contributions, depend on direction? Exact sums.

1D slab (rod model): sites 1..D between faces 0 (top) and D+1 (bottom), 4 lanes with per-site collision probability
c_i(s) and albedo a_i(s) (heterogeneous, chromatic), keep/reverse phase (1 +- g)/2, internal Fresnel F at the faces.
The sampler is sss_walk.mojo's in miniature: per segment a hero lane with probability ~ throughput * albedo and weights
f_i / p_bar, a guide toward the walker's own entry face, Russian roulette, a cap of M segments. Every walk is enumerated
with its exact sampling probability, so each strategy's expected estimate is a finite sum: no noise.
Path: light l -> face A -> walk -> face B -> camera. Strategies: the camera walks B -> A and then hits the light (hit),
connects to it (NEE) or merges at A with photons that arrived there (exit merge); or the light walks A -> B and connects
to the camera (trace). Reflection: A = B = top. Transmission: A = top, B = bottom.
Reported: E[sum of estimates] / I - 1, worst lane, I the camera model's integral (the path tracer's answer).
Walk models: reciprocal (Gonzales since a8db4d03), similarity (sss_walk.mojo's switch to isotropic after K segments,
g != 0), tint (a texture tint at the walker's entry, sss_hop), refract (refract in, Lambert out: the model before a8db4d03).
Weights: dropped (r := 1, Gonzales), exact (true walk densities); adjoint: the light evaluates the camera's model.
Usage: vcm_sss_exit_merge_reciprocity.py    (exit status 1 on failure)
"""
import math, random, sys

D, M, ETA = 2, 6, 1.33
C = [[0.5, 0.2], [0.6, 0.35], [0.7, 0.5], [0.8, 0.65]]       # collision probability per lane, site
ALB = [[0.95, 0.9], [0.9, 0.8], [0.8, 0.6], [0.6, 0.4]]
ABAR = [sum(a) / D for a in ALB]
G, GF, F_FACE, SIM = 0.6, 0.5, 0.4, 2
TINT = {0: [1.0, 0.8, 0.6, 0.9], D + 1: [0.7, 1.0, 1.2, 0.5]}
# outside: two lights, one camera pixel
PL, H, CL, GL, LE = [0.6, 0.4], [0.7, 0.5], [0.9, 0.3], [1.0, 0.7], [[1.0, 1.2, 0.8, 1.5], [2.0, 0.6, 1.1, 0.9]]
E_DIR, S_V, C_V, WE, ETA_M, Z, LAMBERT = [0.25, 0.15], 0.8, 0.6, 1.0, 0.9, 2.0, 0.45


def fresnel(c):
    s2 = (1 - c * c) / (ETA * ETA); ct = math.sqrt(1 - s2)
    rs = (c - ETA * ct) / (c + ETA * ct); rp = (ETA * c - ct) / (ETA * c + ct)
    return 0.5 * (rs * rs + rp * rp)


def Tr(c): return 1 - fresnel(c)
def mu_t(c): return math.sqrt(1 - (1 - c * c) / (ETA * ETA))


def walks(f0, fB):
    """Every walk from face f0 to face fB with at most M segments, as its list of event points."""
    out = []
    def rec(pts, pos, d, nseg):
        if nseg == M:
            return
        s = pos + d
        while 1 <= s <= D:
            for nd in (d, -d):
                rec(pts + [s], s, nd, nseg + 1)
            s += d
        if s == fB:
            out.append(tuple(pts + [s]))
        rec(pts + [s], s, -d, nseg + 1)
    rec([f0], f0, 1 if f0 == 0 else -1, 0)
    return out


def coeffs(j, s, k, model, cin):
    c, a = C[j][s - 1], ALB[j][s - 1]
    if model == "similarity" and k >= SIM:
        c, a = 1 - (1 - c) ** (1 - a * G), a * (1 - G) / (1 - a * G)
    if model == "refract" and k == 0:
        c = 1 - (1 - c) ** (1 / mu_t(cin))
    return c, a


def evaluate(pts, model, cin):
    """(sampling probability, per-lane contribution, per-lane throughput) of a walk, walked from pts[0]."""
    d0 = 1 if pts[0] == 0 else -1
    tp = [1.0] * 4; f = [1.0] * 4; P = 1.0
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]; d = 1 if b > a else -1
        coll = 1 <= b <= D
        pj = []
        for j in range(4):
            p = 1.0
            for s in range(a + d, b, d):
                p *= 1 - coeffs(j, s, k, model, cin)[0]
            pj.append(p * coeffs(j, b, k, model, cin)[0] if coll else p)
        fj = [pj[j] * coeffs(j, b, k, model, cin)[1] if coll else pj[j] for j in range(4)]
        q = [tp[j] * ABAR[j] for j in range(4)]
        pbar = sum(q[j] * pj[j] for j in range(4)) / sum(q)
        tp = [tp[j] * fj[j] / pbar for j in range(4)]; f = [f[j] * fj[j] for j in range(4)]; P *= pbar
        if coll:
            nd = 1 if pts[k + 2] > b else -1
            g = 0.0 if model == "similarity" and k + 1 >= SIM else G
            ph = (1 + g) / 2 if nd == d else (1 - g) / 2
            pd = (1 - GF) * ph + GF * (0.75 if nd == -d0 else 0.25)
            f = [x * ph for x in f]; tp = [x * ph / pd for x in tp]; P *= pd
        else:
            e = (1 - F_FACE) if k == len(pts) - 2 else F_FACE
            f = [x * e for x in f]; P *= e
        m = max(tp)
        if m < 0.1 and k < len(pts) - 2:     # roulette after a scatter, not after the exit
            rr = max(10 * m, 0.05)
            P *= rr; tp = [x / rr for x in tp]
    if model == "tint":
        f = [f[j] * TINT[pts[0]][j] for j in range(4)]; tp = [tp[j] * TINT[pts[0]][j] for j in range(4)]
    return P, f, tp


def exit_lobe(c, model): return LAMBERT if model == "refract" else Tr(c) / Z


def expectation(geom, model, variant, adjoint=False):
    """E[estimate] per lane, I per lane, worst |sum of weights - 1| per path, the exit merge's share of E."""
    A, B = 0, (0 if geom == "reflection" else D + 1)
    E = [0.0] * 4; I = [0.0] * 4; Em = [0.0] * 4; worst_sum = 0.0
    for w in walks(A, B):
        PC, fC, tpC = evaluate(w[::-1], model, C_V)
        for l in range(2):
            PLw, fL, tpL = evaluate(w, model, CL[l])
            ones = (1.0, 1.0); ex = (PLw, PC)
            sets = {"dropped": (ones, ones, ones), "exact": (ex, ex, ex), "M3": (ones, ones, ex), "M4": (ex, ones, ex),
                    "M1": (ones, ones, (PLw, 1.0))}[variant]   # (camera hit/NEE, camera exit merge, light trace)
            def q(W):
                Wl, Wc = W
                cam = S_V * Tr(C_V) * Wc
                return {"hit": cam * E_DIR[l], "nee": cam * PL[l], "mrg": cam * ETA_M * PL[l] * H[l],
                        "trc": PL[l] * H[l] * Tr(CL[l]) * Wl}
            qa, qm, qt = q(sets[0]), q(sets[1]), q(sets[2])
            wt = {"hit": qa["hit"] / sum(qa.values()), "nee": qa["nee"] / sum(qa.values()),
                  "mrg": qm["mrg"] / sum(qm.values()), "trc": qt["trc"] / sum(qt.values())}
            worst_sum = max(worst_sum, abs(sum(wt.values()) - 1))
            p_cam = S_V * Tr(C_V) * PC
            lobe = exit_lobe(CL[l], model)
            p_lig = PL[l] * H[l] * Tr(CL[l]) * PLw
            fix = [fC[j] * Tr(C_V) * exit_lobe(CL[l], model) / (fL[j] * Tr(CL[l]) * exit_lobe(C_V, model))
                   if adjoint else 1.0 for j in range(4)]
            for j in range(4):
                beta = WE / S_V * tpC[j]
                hit = p_cam * E_DIR[l] * wt["hit"] * beta * lobe * GL[l] * LE[l][j] / E_DIR[l]
                nee = p_cam * PL[l] * wt["nee"] * beta * lobe * GL[l] * LE[l][j] / PL[l]
                mrg = p_cam * PL[l] * H[l] * wt["mrg"] * beta * lobe * LE[l][j] * GL[l] / (PL[l] * H[l])
                trc = p_lig * wt["trc"] * LE[l][j] * GL[l] / (PL[l] * H[l]) * tpL[j] * fix[j] * exit_lobe(C_V, model) * WE
                E[j] += hit + nee + mrg + trc; Em[j] += mrg
                I[j] += WE * Tr(C_V) * fC[j] * lobe * GL[l] * LE[l][j]
    bias = max(abs(E[j] / I[j] - 1) for j in range(4))
    return bias, worst_sum, sum(Em) / sum(E)


def mc_sampler_check(n, rng):
    """The sampler with explicit random lane draws: E[throughput, exit at the bottom] against the enumerated sum."""
    truth = [0.0] * 4
    for w in walks(0, D + 1):
        _, f, _ = evaluate(w, "reciprocal", 1.0)
        truth = [truth[j] + f[j] for j in range(4)]
    acc = [0.0] * 4; acc2 = [0.0] * 4
    for _ in range(n):
        pos, d, tp = 0, 1, [1.0] * 4
        for k in range(M):
            q = [tp[j] * ABAR[j] for j in range(4)]
            u = rng.random() * sum(q); lane = 0
            while u > q[lane]:
                u -= q[lane]; lane += 1
            s, stop = pos + d, None
            while 1 <= s <= D:
                if rng.random() < C[lane][s - 1]:
                    stop = s; break
                s += d
            b = s if stop is None else stop
            pj = []
            for j in range(4):
                p = 1.0
                for t in range(pos + d, b, d): p *= 1 - C[j][t - 1]
                pj.append(p * C[j][b - 1] if stop is not None else p)
            fj = [pj[j] * ALB[j][b - 1] if stop is not None else pj[j] for j in range(4)]
            pbar = sum(q[j] * pj[j] for j in range(4)) / sum(q)
            tp = [tp[j] * fj[j] / pbar for j in range(4)]
            pos = b
            if stop is not None:
                toward = -1
                ph_keep = (1 + G) / 2
                p_keep = (1 - GF) * ph_keep + GF * (0.75 if d == toward else 0.25)
                keep = rng.random() < p_keep
                ph = ph_keep if keep else 1 - ph_keep
                tp = [x * ph / (p_keep if keep else 1 - p_keep) for x in tp]
                d = d if keep else -d
            else:
                if rng.random() < F_FACE:
                    d = -d
                else:
                    if b == D + 1:
                        acc = [acc[j] + tp[j] for j in range(4)]; acc2 = [acc2[j] + tp[j] ** 2 for j in range(4)]
                    break
            m = max(tp)
            if m < 0.1:
                rr = max(10 * m, 0.05)
                if rng.random() >= rr: break
                tp = [x / rr for x in tp]
    z = [abs(acc[j] / n - truth[j]) / math.sqrt(max(acc2[j] / n - (acc[j] / n) ** 2, 1e-300) / n) for j in range(4)]
    return max(z)


def main():
    exact_ok = []; must_fail = []
    print(f"  {'geometry':12s} {'walk model':11s} {'dropped':>9s} {'exact':>9s} {'adjoint':>9s}   exit-merge share")
    for geom in ("reflection", "transmission"):
        for model in ("reciprocal", "similarity", "tint", "refract"):
            bd, sd, share = expectation(geom, model, "dropped")
            be, se, _ = expectation(geom, model, "exact")
            ba, _, _ = expectation(geom, model, "dropped", adjoint=True)
            print(f"  {geom:12s} {model:11s} {bd:9.2e} {be:9.2e} {ba:9.2e}   {share:.3f}")
            exact_ok += [sd, se, ba]
            if model == "reciprocal":
                exact_ok += [bd, be]
            elif not (geom == "reflection" and model == "tint"):
                must_fail.append(be)
    print("  anti-vacuity on the reciprocal walk, transmission:")
    for v, what in (("M3", "exact r on the light side only"), ("M4", "exact r except the camera exit's own strategies"),
                    ("M1", "the light divides by its walk density, nothing reverses it")):
        b, s, _ = expectation("transmission", "reciprocal", v)
        print(f"    {v} {what:50s} bias {b:.2e}  worst |sum w - 1| {s:.2e}")
        must_fail.append(b)
    z = mc_sampler_check(100000, random.Random(20261105))
    print(f"  sampler with random lane draws vs the enumerated sum: worst z = {z:.2f}  (noise: must be < 4)")
    worst = max(exact_ok)
    print(f"\nworst error where the method must be exact (reciprocal model, sums of weights, adjoint): {worst:.3e}")
    print(f"anti-vacuity, smallest bias where it must fail: {min(must_fail):.3e}  (must be >> 1e-9)")
    sys.exit(0 if worst < 1e-9 and min(must_fail) > 1e-6 and z < 4 else 1)


if __name__ == "__main__":
    main()
