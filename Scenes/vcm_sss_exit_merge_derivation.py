#!/usr/bin/env python3
"""VCM's weights with a merge at a subsurface walk's boundary, the strategy a smooth surface in front of the skin needs.

A camera that walks from x[hop+1] to x[hop] merges at that exit with a photon arriving at x[hop] that has not walked:
  merge at x[hop]: N eta(x[hop]) * p_light(... -> x[hop]) * p_camera(... -> x[hop+1] -hop-> x[hop])
It is one more strategy of the same path, and it is the only one at the hop: x[hop+1] does not merge, nor does a connection
cross the hop. Ordinary surface vertices merge as usual. Enumerates every connection and merge strategy's full path pdf and
checks that the local recursion reproduces the balance weights. See vcm_sss_hop_mis_derivation.py for the hop itself.
Usage: vcm_sss_exit_merge_derivation.py    (exit status 1 on failure)
"""
import math, random, sys
from vcm_sss_hop_mis_derivation import V, T, p_dir, p_hop, edge_pdf, make_path, unit, sub, norm, dot

ETA_ENTRY_TERM = True     # anti-vacuity switch: the recursion's share for the new strategy


def eta_at(xs, i, hop, base):
    """N * kernel area at x[i], different per vertex as footprint radii are; 0 where nothing merges."""
    n = len(xs) - 1
    if i == 0 or i == n or (hop is not None and i == hop + 1):
        return 0.0
    return base * (0.5 + 1.5 * abs(math.sin(xs[i].pos[0] * 37.0 + xs[i].pos[1] * 11.0)))


def half_light(xs, k, hop, st, pL):
    p = pL
    for i in range(1, k + 1):
        p *= edge_pdf(xs, i - 1, i, hop, st)
    return p


def half_camera(xs, k, hop, st, pC):
    p = pC
    for i in range(len(xs) - 2, k - 1, -1):
        p *= edge_pdf(xs, i + 1, i, hop, st)
    return p


def brute(xs, hop, st, pL, pC, base):
    n = len(xs) - 1
    conn = {}
    for s in range(0, n + 2):
        if hop is not None and s - 1 == hop:
            continue
        p = 1.0
        if s >= 1:
            p *= half_light(xs, s - 1, hop, st, pL)
        if s <= n:
            p *= half_camera(xs, s, hop, st, pC)
        conn[s] = p
    merge = {k: eta_at(xs, k, hop, base) * half_light(xs, k, hop, st, pL) * half_camera(xs, k, hop, st, pC)
             for k in range(1, n) if eta_at(xs, k, hop, base) > 0}
    tot = sum(conn.values()) + sum(merge.values())
    return {s: p / tot for s, p in conn.items()}, {k: p / tot for k, p in merge.items()}


def carries(xs, hop, st, p_start, stop, from_camera, base):
    n = len(xs) - 1
    dVCM, dVC = 1.0 / p_start, 0.0
    for i in (range(n, stop, -1) if from_camera else range(0, stop)):
        j = i - 1 if from_camera else i + 1
        a, b = xs[i], xs[j]
        k = i + 1 if from_camera else i - 1
        is_hop = hop is not None and {i, j} == {hop, hop + 1}
        first = i == n if from_camera else i == 0
        if first:
            pdf_rev = 0.0
        elif hop is not None and {i, k} == {hop, hop + 1}:
            pdf_rev = T(a, b) * p_hop(a, xs[k], st)
        else:
            pdf_rev = p_dir(a.n, unit(sub(xs[k].pos, a.pos)))
        eta = eta_at(xs, i, hop, base)
        if not ETA_ENTRY_TERM and hop is not None and i == hop:
            eta = 0.0
        if is_hop:
            pdf = T(a, xs[k]) * p_hop(a, b, st)
            dVC = (1.0 / pdf) * (dVC * pdf_rev + dVCM + eta)
            dVCM = 0.0
        else:
            d = sub(b.pos, a.pos); dist = norm(d); w = unit(d)
            pdf = p_dir(a.n, w)
            dVC = (abs(dot(a.n, w)) / pdf) * (dVC * pdf_rev + dVCM + eta)
            dVCM = dist * dist / pdf
            cf = abs(dot(b.n, w))
            dVCM /= cf; dVC /= cf
    return dVCM, dVC


def conn_weight(xs, s, hop, st, pL, pC, base):
    n = len(xs) - 1
    lvi, cvi = s - 1, s
    lv, cv = xs[lvi], xs[cvi]
    d = sub(cv.pos, lv.pos); dist = norm(d); wl = unit(d); wc = (-wl[0], -wl[1], -wl[2])
    lm, lc = carries(xs, hop, st, pL, lvi, False, base)
    cm, cc = carries(xs, hop, st, pC, cvi, True, base)
    def rev(v, other, partner):
        if hop is not None and {v, other} == {hop, hop + 1}:
            return T(xs[v], xs[partner]) * p_hop(xs[v], xs[other], st)
        return p_dir(xs[v].n, unit(sub(xs[other].pos, xs[v].pos)))
    lr = 0.0 if lvi == 0 else rev(lvi, lvi - 1, cvi)
    cr = 0.0 if cvi == n else rev(cvi, cvi + 1, lvi)
    cam_a = p_dir(cv.n, wc) * abs(dot(lv.n, wc)) / (dist * dist)
    lig_a = p_dir(lv.n, wl) * abs(dot(cv.n, wl)) / (dist * dist)
    return 1.0 / (cam_a * (eta_at(xs, lvi, hop, base) + lm + lc * lr) + 1.0
                  + lig_a * (eta_at(xs, cvi, hop, base) + cm + cc * cr))


def merge_weight(xs, k, hop, st, pL, pC, base):
    lm, lc = carries(xs, hop, st, pL, k, False, base)
    cm, cc = carries(xs, hop, st, pC, k, True, base)
    eta = eta_at(xs, k, hop, base)
    toward_light = p_dir(xs[k].n, unit(sub(xs[k - 1].pos, xs[k].pos)))
    if hop is not None and k == hop:
        # The camera's last edge is the hop: its reverse leaves through the photon's arrival direction, T of that.
        toward_camera = T(xs[k], xs[k - 1]) * p_hop(xs[k], xs[k + 1], st)
    else:
        toward_camera = p_dir(xs[k].n, unit(sub(xs[k + 1].pos, xs[k].pos)))
    return 1.0 / ((lm + lc * toward_light) / eta + 1.0 + (cm + cc * toward_camera) / eta)


def run(nv, hop, st, base, rng, verbose):
    xs = make_path(nv, hop, rng)
    pL, pC = 0.37, 0.53
    wc, wm = brute(xs, hop, st, pL, pC, base)
    n = len(xs) - 1
    worst = 0.0
    for s in range(1, n + 1):
        if s in wc:
            worst = max(worst, abs(conn_weight(xs, s, hop, st, pL, pC, base) - wc[s]) / max(wc[s], 1e-300))
    worst_merge = 0.0
    for k, w in wm.items():
        r = merge_weight(xs, k, hop, st, pL, pC, base)
        e = abs(r - w) / max(w, 1e-300)
        worst_merge = max(worst_merge, e)
        if verbose:
            print(f"    merge k={k}{' (walk exit)' if k == hop else ''}  brute={w:.10e}  recursion={r:.10e}  relerr={e:.2e}")
    return max(worst, worst_merge)


def main():
    global ETA_ENTRY_TERM
    cases = [(5, 1, 20.0, 3.0), (5, 2, 20.0, 0.1), (6, 2, 60.0, 30.0), (6, 3, 5.0, 1.0), (7, 1, 120.0, 0.01), (7, 4, 40.0, 5.0), (8, 3, 15.0, 2.0)]
    rng = random.Random(20261031)
    worst = 0.0
    for nv, hop, st, base in cases:
        print(f"  {nv} vertices, hop at {hop}, sigma_tr={st}, eta base {base}")
        worst = max(worst, run(nv, hop, st, base, rng, True))
    print(f"\nworst relative error: {worst:.3e}")
    ETA_ENTRY_TERM = False
    rng = random.Random(20261031)
    bad = max(run(nv, hop, st, base, rng, False) for nv, hop, st, base in cases)
    ETA_ENTRY_TERM = True
    print(f"anti-vacuity, the recursion without the walk exit's merge share: worst {bad:.3e}  (must be >> 1e-9)")
    sys.exit(0 if worst < 1e-9 and bad > 1e-6 else 1)


if __name__ == "__main__":
    main()
