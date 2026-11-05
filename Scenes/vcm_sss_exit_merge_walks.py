#!/usr/bin/env python3
"""Exit merging's VCM weights when the walk is heterogeneous, anisotropic and chromatic.

vcm_sss_exit_merge_derivation.py treats the walk {hop, hop+1} as one edge with a symmetric density p_hop. Here the walk
has explicit interior vertices y_1..y_m and the TRUE sampling densities of a Gonzales-style sampler: entry lobe
c (1 - F_int(c)), free flights from 4 hero lanes with per-lane, position-dependent sigma_t (heterogeneous), HG phase,
an optional Dwivedi guide toward the walker's own entry, and the lane mixture of sss_walk.mojo (per segment, lane
probabilities proportional to throughput times albedo). P_L(w) is the light walk's density, P_C(rev w) the camera's.

In the recursion the walk density appears only as the ratio r = P_rev / P_fwd at the side that sampled it. Variants:
  dropped  r := 1 on both sides (Gonzales today)                  -> must be a partition of unity for every walk
  exact    r = P_rev/P_fwd, both sides, all strategies           -> must equal the true balance heuristic
  M3       exact on the light side only, dropped on the camera's -> anti-vacuity
  M4       exact, but the strategies AT the camera walk's exit (its connections, its exit merge) use r := 1
Usage: vcm_sss_exit_merge_walks.py    (exit status 1 on failure)
"""
import math, random, sys
from vcm_sss_hop_mis_derivation import T, p_dir, make_path, unit, sub, norm, dot
from vcm_sss_exit_merge_derivation import eta_at

ETA = 1.33
GL = [(-0.9602898564975363, 0.1012285362903763), (-0.7966664774136267, 0.2223810344533745),
      (-0.5255324099163290, 0.3137066458778873), (-0.1834346424956498, 0.3626837833783620),
      (0.1834346424956498, 0.3626837833783620), (0.5255324099163290, 0.3137066458778873),
      (0.7966664774136267, 0.2223810344533745), (0.9602898564975363, 0.1012285362903763)]


def fresnel(c, eta):
    s2 = (1 - c * c) / (eta * eta)
    if s2 >= 1:
        return 1.0
    ct = math.sqrt(1 - s2)
    rs = (c - eta * ct) / (c + eta * ct); rp = (eta * c - ct) / (eta * c + ct)
    return 0.5 * (rs * rs + rp * rp)


def f_int(c): return fresnel(c, 1 / ETA)
R_INT = sum(w * 0.5 * 2 * f_int(0.5 * (x + 1)) * 0.5 * (x + 1) for x, w in GL)   # 2 * int_0^1 F_int(c) c dc


def entry_lobe(c, capped):
    """sss_entry_lobe's density over the hemisphere. Exact: c (1-F_int)/(pi (1-R)). Capped: its 8-try rejection loop."""
    if not capped:
        return c * (1 - f_int(c)) / (math.pi * (1 - R_INT))
    return c / math.pi * ((1 - f_int(c)) * (1 - R_INT ** 8) / (1 - R_INT) + R_INT ** 8)


def hg(mu, g):
    return (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * mu) ** 1.5)


DL = 1.6
PLOG = math.log((DL + 1) / (DL - 1))
def guide(c): return 1 / (2 * math.pi * (DL - c) * PLOG)


class Cfg:
    def __init__(self, name, chroma, hetero, g, lanes, gf, capped=False):
        self.name, self.chroma, self.hetero, self.g, self.lanes, self.gf, self.capped = name, chroma, hetero, g, lanes, gf, capped
        self.s = [30.0, 45.0, 70.0, 110.0] if chroma else [55.0] * 4
        self.a = [0.97, 0.9, 0.75, 0.5] if chroma else [0.9] * 4

    def sigma(self, j, x):
        j = j if self.chroma else 0
        m = 1 + 0.7 * math.sin(31 * x[0] + 17 * x[1] + 3 * j) * math.cos(23 * x[2] - j) if self.hetero else 1.0
        return self.s[j] * m

    def albedo(self, j, x):
        j = j if self.chroma else 0
        return self.a[j] * ((1 - 0.15 * math.sin(41 * x[0] - 13 * x[2] + j) ** 2) if self.hetero else 1.0)

    def tau(self, j, a, b):
        t = norm(sub(b, a))
        return 0.5 * t * sum(w * self.sigma(j, tuple(a[q] + 0.5 * (x + 1) * (b[q] - a[q]) for q in range(3))) for x, w in GL)


def walk_density(pts, n_in, n_out, cfg):
    """Sampling density and per-lane contribution of the walk pts[0] (entry) -> ... -> pts[-1] (exit), in area/volume
    measure, as the walker entering at pts[0] samples it."""
    m = len(pts) - 2
    w = unit(sub(pts[1], pts[0]))
    common = entry_lobe(abs(dot(n_in, w)), cfg.capped)    # lane-independent factors, the same in f and p ...
    pdir = common
    f = [1.0] * 4; tp = [1.0] * 4; pmix = 1.0; plane = [1.0] * 4
    for k in range(m + 1):
        a, b = pts[k], pts[k + 1]
        t2 = dot(sub(b, a), sub(b, a))
        tr = [math.exp(-cfg.tau(j, a, b)) for j in range(4)]
        if k < m:
            pj = [cfg.sigma(j, b) * tr[j] / t2 for j in range(4)]
            fi = [cfg.albedo(j, b) * pj[j] for j in range(4)]
        else:
            cb = abs(dot(n_out, w))
            pj = [tr[j] * cb / t2 * (1 - f_int(cb)) for j in range(4)]
            fi = pj[:]
        if cfg.lanes == "walk":
            plane = [plane[j] * pj[j] for j in range(4)]
        else:
            q = [tp[j] * cfg.a[j] for j in range(4)] if cfg.lanes == "throughput" else [1.0] * 4
            pbar = sum(q[j] * pj[j] for j in range(4)) / sum(q)
            tp = [tp[j] * fi[j] / pbar for j in range(4)]
            pmix *= pbar
        f = [f[j] * fi[j] for j in range(4)]
        if k < m:
            wn = unit(sub(pts[k + 2], b))
            ph = hg(dot(w, wn), cfg.g)
            pd = (1 - cfg.gf) * ph + cfg.gf * guide(dot(wn, n_in))
            common *= ph; pdir *= pd
            tp = [x * ph / pd for x in tp]
            w = wn
    p = pdir * (0.25 * sum(plane) if cfg.lanes == "walk" else pmix)
    return p, [common * x for x in f]


def make_walk(xs, hop, rng):
    """Interior vertices below the skin; the first below the entry, the last below the exit, so neither boundary
    segment is past the critical angle (the exit's 1 - F_int would be 0)."""
    a, b = xs[hop], xs[hop + 1]
    def below(v, d, spread):
        return tuple(v.pos[q] + spread * d * (rng.random() - 0.5) * (q < 2) - v.n[q] * d for q in range(3))
    m = rng.randint(1, 6)
    if m == 1:
        mid = type(a)(tuple(0.5 * (a.pos[q] + b.pos[q]) for q in range(3)), a.n)
        return [a.pos, below(mid, 0.15, 0.2), b.pos]
    inner = [below(a, 0.02 + 0.05 * rng.random(), 0.6)]
    inner += [below(a, 0.02 + 0.06 * rng.random(), 3.0) for _ in range(m - 2)]
    inner.append(below(b, 0.02 + 0.05 * rng.random(), 0.6))
    return [a.pos] + inner + [b.pos]


# ------------------------------------------------------------- the recursion, with the walk's density per side
# Pfn(i): interior density of the walk that starts at boundary vertex i (hop: the light's direction, hop+1: the camera's).

def carries(xs, hop, Pfn, p_start, stop, from_camera, base):
    n = len(xs) - 1
    dVCM, dVC = 1.0 / p_start, 0.0
    for i in (range(n, stop, -1) if from_camera else range(0, stop)):
        j = i - 1 if from_camera else i + 1
        k = i + 1 if from_camera else i - 1
        a, b = xs[i], xs[j]
        if (i == n if from_camera else i == 0):
            pdf_rev = 0.0
        elif {i, k} == {hop, hop + 1}:
            pdf_rev = T(a, b) * Pfn(i)
        else:
            pdf_rev = p_dir(a.n, unit(sub(xs[k].pos, a.pos)))
        eta = eta_at(xs, i, hop, base)
        if {i, j} == {hop, hop + 1}:
            dVC = (1.0 / (T(a, xs[k]) * Pfn(i))) * (dVC * pdf_rev + dVCM + eta)
            dVCM = 0.0
        else:
            d = sub(b.pos, a.pos); dist = norm(d); w = unit(d)
            pdf = p_dir(a.n, w)
            dVC = (abs(dot(a.n, w)) / pdf) * (dVC * pdf_rev + dVCM + eta)
            dVCM = dist * dist / pdf
            cf = abs(dot(b.n, w))
            dVCM /= cf; dVC /= cf
    return dVCM, dVC


def conn_weight(xs, s, hop, PL, PC, pL, pC, base):
    n = len(xs) - 1
    lvi, cvi = s - 1, s
    lv, cv = xs[lvi], xs[cvi]
    d = sub(cv.pos, lv.pos); dist = norm(d); wl = unit(d); wc = (-wl[0], -wl[1], -wl[2])
    lm, lc = carries(xs, hop, PL, pL, lvi, False, base)
    cm, cc = carries(xs, hop, PC, pC, cvi, True, base)
    def rev(v, other, partner, P):
        if {v, other} == {hop, hop + 1}:
            return T(xs[v], xs[partner]) * P(v)
        return p_dir(xs[v].n, unit(sub(xs[other].pos, xs[v].pos)))
    lr = 0.0 if lvi == 0 else rev(lvi, lvi - 1, cvi, PL)
    cr = 0.0 if cvi == n else rev(cvi, cvi + 1, lvi, PC)
    cam_a = p_dir(cv.n, wc) * abs(dot(lv.n, wc)) / (dist * dist)
    lig_a = p_dir(lv.n, wl) * abs(dot(cv.n, wl)) / (dist * dist)
    return 1.0 / (cam_a * (eta_at(xs, lvi, hop, base) + lm + lc * lr) + 1.0
                  + lig_a * (eta_at(xs, cvi, hop, base) + cm + cc * cr))


def merge_weight(xs, k, hop, PL, PC, pL, pC, base):
    lm, lc = carries(xs, hop, PL, pL, k, False, base)
    cm, cc = carries(xs, hop, PC, pC, k, True, base)
    eta = eta_at(xs, k, hop, base)
    toward_light = p_dir(xs[k].n, unit(sub(xs[k - 1].pos, xs[k].pos)))
    if k == hop:
        toward_camera = T(xs[k], xs[k - 1]) * PC(k)      # the light's density of the walk the camera took
    else:
        toward_camera = p_dir(xs[k].n, unit(sub(xs[k + 1].pos, xs[k].pos)))
    return 1.0 / ((lm + lc * toward_light) / eta + 1.0 + (cm + cc * toward_camera) / eta)


def edge_pdf(xs, i, j, hop, P):
    a, b = xs[i], xs[j]
    if {i, j} == {hop, hop + 1}:
        return T(a, xs[i - 1 if j > i else i + 1]) * P(i)
    d = sub(b.pos, a.pos); dist = norm(d); w = unit(d)
    return p_dir(a.n, w) * abs(dot(b.n, w)) / (dist * dist)


def brute(xs, hop, P, pL, pC, base):
    n = len(xs) - 1
    def hl(k):
        p = pL
        for i in range(1, k + 1): p *= edge_pdf(xs, i - 1, i, hop, P)
        return p
    def hc(k):
        p = pC
        for i in range(n - 1, k - 1, -1): p *= edge_pdf(xs, i + 1, i, hop, P)
        return p
    conn = {s: (hl(s - 1) if s >= 1 else 1.0) * (hc(s) if s <= n else 1.0) for s in range(0, n + 2) if s - 1 != hop}
    merge = {k: eta_at(xs, k, hop, base) * hl(k) * hc(k) for k in range(1, n) if eta_at(xs, k, hop, base) > 0}
    tot = sum(conn.values()) + sum(merge.values())
    return {s: p / tot for s, p in conn.items()}, {k: p / tot for k, p in merge.items()}


def weights(xs, hop, pL, pC, base, pick):
    """Every strategy's weight as the recursion computes it; pick(kind, idx) -> (PL, PC) for that strategy."""
    n = len(xs) - 1
    wc = {s: conn_weight(xs, s, hop, *pick("c", s), pL, pC, base) for s in range(1, n + 1) if s - 1 != hop}
    wm = {k: merge_weight(xs, k, hop, *pick("m", k), pL, pC, base) for k in range(1, n) if eta_at(xs, k, hop, base) > 0}
    return wc, wm


def run(case, cfg, rng):
    nv, hop, _, base = case
    xs = make_path(nv, hop, rng)
    n = len(xs) - 1
    pts = make_walk(xs, hop, rng)
    A, B = xs[hop], xs[hop + 1]
    PLw, fL = walk_density(pts, A.n, B.n, cfg)
    PCw, fC = walk_density(pts[::-1], B.n, A.n, cfg)
    drop = lambda i: 1.0
    exact = lambda i: PLw if i == hop else PCw
    pL, pC = 0.37, 0.53
    out = {"r": abs(math.log10(PCw / PLw)), "recip": max(abs(fL[j] / fC[j] - 1) for j in range(4))}
    for name, P in (("dropped", drop), ("exact", exact)):
        bc, bm = brute(xs, hop, P, pL, pC, base)
        wc, wm = weights(xs, hop, pL, pC, base, lambda kind, i: (P, P))
        out[name] = max([abs(wc[s] - bc[s]) / bc[s] for s in wc] + [abs(wm[k] - bm[k]) / bm[k] for k in wm])
    def total(pick, tail_L, tail_C):
        # s = n+1 (the light subpath hits the lens) and s = 0 (the camera's hits the light): whole-path strategies of one side
        wc, wm = weights(xs, hop, pL, pC, base, pick)
        return sum(wc.values()) + sum(wm.values()) + brute(xs, hop, tail_L, pL, pC, base)[0][n + 1] \
            + brute(xs, hop, tail_C, pL, pC, base)[0][0]
    out["sum"] = max(abs(total(lambda kind, i: (P, P), P, P) - 1) for P in (drop, exact))
    out["M3"] = abs(total(lambda kind, i: (exact, drop), exact, drop) - 1)
    at_exit = lambda kind, i: kind in "cm" and i == hop
    out["M4"] = abs(total(lambda kind, i: (drop, drop) if at_exit(kind, i) else (exact, exact), exact, exact) - 1)
    return out


def main():
    cases = [(5, 1, 0, 3.0), (5, 2, 0, 0.1), (6, 2, 0, 30.0), (6, 3, 0, 1.0), (7, 1, 0, 0.01), (7, 4, 0, 5.0), (8, 3, 0, 2.0)]
    cfgs = [Cfg("grey homogeneous, g=0, unguided", False, False, 0.0, "throughput", 0.0),
            Cfg("grey heterogeneous, HG g=0.8", False, True, 0.8, "throughput", 0.0),
            Cfg("chromatic heterogeneous HG, lane fixed per walk", True, True, 0.8, "walk", 0.0),
            Cfg("chromatic, uniform lane mixture per segment", True, True, 0.0, "uniform", 0.0),
            Cfg("chromatic HG, throughput mixture (sss_walk.mojo)", True, True, 0.8, "throughput", 0.0),
            Cfg("grey homogeneous, Dwivedi guide 0.5", False, False, 0.0, "throughput", 0.5),
            Cfg("chromatic HG + guide (sss_walk.mojo)", True, True, 0.8, "throughput", 0.5),
            Cfg("same, 8-try capped entry lobe", True, True, 0.8, "throughput", 0.5, capped=True)]
    worst = 0.0; anti = 0.0
    print(f"{'walk':50s} {'max|log10 r|':>12s} {'f_L/f_C-1':>9s} {'dropped':>9s} {'exact':>9s} {'|sum-1|':>9s} {'M3':>9s} {'M4':>9s}")
    for cfg in cfgs:
        rng = random.Random(20261105)
        rows = [run(c, cfg, rng) for c in cases for _ in range(3)]
        agg = {k: max(r[k] for r in rows) for k in rows[0]}
        print(f"{cfg.name:50s} {agg['r']:12.3e} {agg['recip']:9.1e} {agg['dropped']:9.1e} {agg['exact']:9.1e} {agg['sum']:9.1e} {agg['M3']:9.1e} {agg['M4']:9.1e}")
        worst = max(worst, agg["dropped"], agg["exact"], agg["sum"])
        if agg["r"] > 1e-6:
            anti = max(anti, min(agg["M3"], agg["M4"]))
    print(f"\nworst error, dropped and exact (recursion vs brute force, and |sum of weights - 1|): {worst:.3e}")
    print(f"anti-vacuity, smaller of M3/M4 on the non-reciprocal-density walks: {anti:.3e}  (must be >> 1e-9)")
    sys.exit(0 if worst < 1e-9 and anti > 1e-6 else 1)


if __name__ == "__main__":
    main()
