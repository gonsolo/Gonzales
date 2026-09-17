#!/usr/bin/env python3
"""Numerical check of VCM's dVCM/dVC recursion across a BSSRDF HOP.

A subsurface event splits into two path vertices joined by a non-ray edge:
the path enters at one point and leaves at another, the exit sampled from the
diffusion profile with an AREA density p_A(r). This harness enumerates every
connection strategy's full path pdf directly and checks that a local recursion
carried along each subpath reproduces the balance-heuristic weights.

Model (surface-only path, vacuum, so free flight is 1 and only the hop is new):
  ray edge a->b   p(b) = p_dir(a, w) * cos_b / d^2          (Lambertian p_dir)
  hop edge A<->B  p(B|A) = p(A|B) = p_A(|A-B|)               (area, symmetric)

The hop is oriented. Along the CAMERA subpath (high index -> low) the path
enters at B and exits at A; along the LIGHT subpath it enters at A and exits
at B. Only an EXIT point is a connectible vertex. A connection ACROSS the hop
edge itself (a deterministic BSSRDF connection) is not implemented, so that
one strategy has zero probability and is excluded from the ground truth.

Usage: vcm_bssrdf_mis_derivation.py    (exit status 1 on failure)
"""
import math, random, sys


def sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def norm(a): return math.sqrt(dot(a, a))
def unit(a):
    n = norm(a); return (a[0]/n, a[1]/n, a[2]/n)


def p_dir(n, w):
    return abs(dot(n, w)) / math.pi


def p_hop(a, b, sigma_tr):
    """Area density of the exit point: exponential radius, uniform azimuth.
    Symmetric in a and b (depends only on the distance) -- the property the
    bidirectional treatment rests on, proven for normal-axis probing in
    bssrdf.mojo; the probe cosine |n_a . n_b| is symmetric too and is folded
    into this factor here."""
    r = norm(sub(a.pos, b.pos))
    return sigma_tr * math.exp(-sigma_tr * r) / (2.0 * math.pi * r) * abs(dot(a.n, b.n))


class V:
    def __init__(self, pos, n):
        self.pos, self.n = pos, n


def edge_pdf(xs, i, j, hop, sigma_tr):
    """Density of sampling xs[j] from xs[i] (i, j adjacent)."""
    a, b = xs[i], xs[j]
    if hop is not None and {i, j} == {hop, hop + 1}:
        return p_hop(a, b, sigma_tr)
    d = sub(b.pos, a.pos); dist = norm(d); w = unit(d)
    return p_dir(a.n, w) * abs(dot(b.n, w)) / (dist * dist)


def path_pdf(xs, s, hop, sigma_tr, pL, pC):
    n = len(xs) - 1
    p = 1.0
    if s >= 1:
        p *= pL
        for i in range(1, s):
            p *= edge_pdf(xs, i - 1, i, hop, sigma_tr)
    if n + 1 - s >= 1:
        p *= pC
        for i in range(n - 1, s - 1, -1):
            p *= edge_pdf(xs, i + 1, i, hop, sigma_tr)
    return p


def excluded(s, hop):
    # connecting ACROSS the hop edge (xs[hop] -- xs[hop+1]) is not implemented
    return hop is not None and s - 1 == hop


def brute_force(xs, hop, sigma_tr, pL, pC):
    n = len(xs) - 1
    ps = [0.0 if excluded(s, hop) else path_pdf(xs, s, hop, sigma_tr, pL, pC)
          for s in range(n + 2)]
    tot = sum(ps)
    return [p / tot for p in ps]


# ------------------------------------------------------------- the recursion
# Exactly the shape bdpt.mojo uses for a ray edge:
#   scatter at a:  dVC = (cos_out/pdf_dir)*(dVC*pdf_rev + dVCM);  dVCM = 1/pdf_dir
#   arrive  at b:  dVCM *= d^2;  dVCM, dVC /= cos_fix
# and for a HOP edge, the rule this script establishes:
#   scatter:  dVC = (1/p_A)*(dVC*pdf_rev + dVCM)    (area measure: no cos_out)
#             dVCM = 0                               (no connection ACROSS the hop)
#   arrive:   no d^2, no cos_fix                     (no ray, no Jacobian)
#   pdf_rev at the vertex AFTER the hop := p_A       (the reverse of the hop)
# The entry vertex's own pdf_rev is its ordinary exit-lobe direction pdf.
#
# dVCM = 0 is the load-bearing part. Keeping dVCM = 1/p_A (the obvious port,
# and what the first VCM implementation shipped) still counts the connection
# across the hop as a live strategy; that strategy does not exist, so every
# real strategy's weight comes out too small -- a uniform relative bias of
# ~1e-4..1e-3 per case here, which compounds into a visibly dark render.
# Candidates tried and measured: keep dVCM (bias above), drop dVCM from dVC
# instead (off by 33x), drop both (off by 33x). Only this one is exact.
RULE = "exact"

def carries(xs, hop, sigma_tr, p_start, stop, from_camera):
    n = len(xs) - 1
    dVCM, dVC = 1.0 / p_start, 0.0
    order = list(range(n, stop, -1)) if from_camera else list(range(0, stop))
    for i in order:
        j = i - 1 if from_camera else i + 1
        a, b = xs[i], xs[j]
        is_hop = hop is not None and {i, j} == {hop, hop + 1}
        k = i + 1 if from_camera else i - 1          # previous vertex
        first = (i == n) if from_camera else (i == 0)
        if first:
            pdf_rev = 0.0
        elif hop is not None and {i, k} == {hop, hop + 1}:
            pdf_rev = p_hop(a, xs[k], sigma_tr)     # reverse of the hop just taken
        else:
            pdf_rev = p_dir(a.n, unit(sub(xs[k].pos, a.pos)))
        if is_hop:
            pdf_dir = p_hop(a, b, sigma_tr)
            dVC = (1.0 / pdf_dir) * (dVC * pdf_rev + dVCM)
            dVCM = 0.0 if RULE == "exact" else 1.0 / pdf_dir
        else:
            d = sub(b.pos, a.pos); dist = norm(d); w = unit(d)
            pdf_dir = p_dir(a.n, w)
            dVC = (abs(dot(a.n, w)) / pdf_dir) * (dVC * pdf_rev + dVCM)
            dVCM = 1.0 / pdf_dir
            dVCM *= dist * dist
            cf = abs(dot(b.n, w))
            dVCM /= cf; dVC /= cf
    return dVCM, dVC


def rev_pdf_at(xs, v_idx, other_idx, hop, sigma_tr):
    """Reverse density at an endpoint toward its subpath predecessor."""
    if hop is not None and {v_idx, other_idx} == {hop, hop + 1}:
        return p_hop(xs[v_idx], xs[other_idx], sigma_tr)
    return p_dir(xs[v_idx].n, unit(sub(xs[other_idx].pos, xs[v_idx].pos)))


def recursion_weight(xs, s, hop, sigma_tr, pL, pC):
    n = len(xs) - 1
    lvi, cvi = s - 1, s
    lv, cv = xs[lvi], xs[cvi]
    d = sub(cv.pos, lv.pos); dist = norm(d); wl2c = unit(d)
    wc2l = (-wl2c[0], -wl2c[1], -wl2c[2])
    lv_dVCM, lv_dVC = carries(xs, hop, sigma_tr, pL, lvi, from_camera=False)
    cv_dVCM, cv_dVC = carries(xs, hop, sigma_tr, pC, cvi, from_camera=True)
    lig_rev = 0.0 if lvi == 0 else rev_pdf_at(xs, lvi, lvi - 1, hop, sigma_tr)
    cam_rev = 0.0 if cvi == n else rev_pdf_at(xs, cvi, cvi + 1, hop, sigma_tr)
    cam_dir_a = p_dir(cv.n, wc2l) * abs(dot(lv.n, wc2l)) / (dist * dist)
    lig_dir_a = p_dir(lv.n, wl2c) * abs(dot(cv.n, wl2c)) / (dist * dist)
    w_light = cam_dir_a * (lv_dVCM + lv_dVC * lig_rev)
    w_camera = lig_dir_a * (cv_dVCM + cv_dVC * cam_rev)
    return 1.0 / (w_light + 1.0 + w_camera)


def make_path(n_verts, hop, rng):
    xs = []
    for i in range(n_verts):
        pos = (rng.uniform(-2, 2), rng.uniform(-2, 2), i * 1.7 + rng.uniform(0.1, 0.4))
        nrm = unit((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.3, 1)))
        xs.append(V(pos, nrm))
    if hop is not None:
        # put the exit near the entry, on a similar surface, as a real hop is
        a = xs[hop]
        xs[hop + 1] = V((a.pos[0] + rng.uniform(-.05, .05), a.pos[1] + rng.uniform(-.05, .05),
                         a.pos[2] + rng.uniform(-.02, .02)),
                        unit((a.n[0] + rng.uniform(-.2, .2), a.n[1] + rng.uniform(-.2, .2), a.n[2])))
    return xs


def run(n_verts, hop, sigma_tr, rng, verbose):
    xs = make_path(n_verts, hop, rng)
    pL, pC = 0.37, 0.53
    wbf = brute_force(xs, hop, sigma_tr, pL, pC)
    n = len(xs) - 1
    worst = 0.0
    for s in range(1, n + 1):
        if excluded(s, hop):
            continue
        wr = recursion_weight(xs, s, hop, sigma_tr, pL, pC)
        err = abs(wr - wbf[s]) / max(wbf[s], 1e-300)
        worst = max(worst, err)
        if verbose:
            print(f"    s={s}  brute={wbf[s]:.12f}  recursion={wr:.12f}  relerr={err:.2e}")
    return worst


def main():
    rng = random.Random(20260917)
    cases = [(4, None, 20.0), (5, 1, 20.0), (5, 2, 20.0), (6, 2, 60.0), (6, 3, 5.0), (7, 1, 120.0), (7, 4, 40.0), (8, 3, 15.0), (8, 5, 0.8)]
    worst = 0.0
    for nv, hop, st in cases:
        print(f"  {nv} vertices, hop at {hop}, sigma_tr={st}")
        worst = max(worst, run(nv, hop, st, rng, True))
    print(f"\nworst relative error: {worst:.3e}")
    ok = worst < 1e-9
    # anti-vacuity: the RECURSION alone uses the naive rule (dVCM = 1/p_A);
    # the ground truth is untouched. The test is only meaningful if this fails.
    global RULE
    RULE = "naive"
    rng2 = random.Random(20260917)
    bad = 0.0
    for nv, hop, st in cases:
        w = run(nv, hop, st, rng2, False)
        if hop is not None:
            bad = max(bad, w)
    RULE = "exact"
    print(f"anti-vacuity, recursion with dVCM = 1/p_A: worst {bad:.3e}  (must be >> 1e-9)")
    sys.exit(0 if ok and bad > 1e-6 else 1)


if __name__ == "__main__":
    main()
