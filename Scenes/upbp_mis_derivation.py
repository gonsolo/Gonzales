#!/usr/bin/env python3
"""UPBP's MIS in VCM's recursive form: numerical check against the balance heuristic over every technique.

Techniques on a path of volume vertices in a homogeneous medium (Krivanek et al. 2014, SmallUPBP PathWeight.hxx):
  BPT    every connection (s, t)
  PP3D   photon point x query point at interior vertex k:  N * (4/3) pi r^3
  PB2D   photon point x camera query beam into x_k:          N * pi r^2           / p_ray_cam(x_k)
  BB1D   photon beam out of x_(k-1) x query beam into x_k:   Nb * 2r * sin(theta) / (p_ray_cam(x_k) p_ray_light(x_k))
each relative to the two subpaths both reaching x_k by distance sampling (p_ray = sigma_t exp(-sigma_t d), the free
flight of that side's last segment). The claim tested: VCM's dVCM/dVC recursion is exact for UPBP when the merge
density at a volume vertex is the SUM eta_k = eta_PP3D + eta_PB2D + eta_BB1D, and technique T at x_k weighs
    w_T = eta_T / (eta_k + lv.dVCM + lv.dVC * pdf_fwd + cv.dVCM + cv.dVC * pdf_rev).
The beam terms need both segments at x_k; a subpath knows only the one it arrived along when it scatters there, so
the code must defer that factor to its next arrival. Exit 1 on a mismatch.
"""
import math, random, sys

SIGMA = 2.0
R = 0.02
N = 4000.0          # light subpaths (photon points)
NB = 4000.0         # photon beams
INV_FOUR_PI = 1.0 / (4.0 * math.pi)


def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def norm(a): return math.sqrt(dot(a, a))
def unit(a):
    n = norm(a); return (a[0] / n, a[1] / n, a[2] / n)


class V:
    def __init__(self, pos, n=None):
        self.pos, self.n = pos, n          # n None: volume vertex; else surface endpoint (light or lens)


def p_dir(v, w): return INV_FOUR_PI if v.n is None else abs(dot(v.n, w)) / math.pi
def ff(b, d): return SIGMA * math.exp(-SIGMA * d) if b.n is None else math.exp(-SIGMA * d)
def geom(b, d, w): return 1.0 / (d * d) if b.n is None else abs(dot(b.n, w)) / (d * d)


def edge(a, b):
    d = norm(sub(b.pos, a.pos)); w = unit(sub(b.pos, a.pos))
    return p_dir(a, w) * geom(b, d, w) * ff(b, d)


def light_half(xs, k, pL):
    p = pL
    for i in range(1, k + 1): p *= edge(xs[i - 1], xs[i])
    return p


def cam_half(xs, k, pC):
    p = pC
    for i in range(len(xs) - 2, k - 1, -1): p *= edge(xs[i + 1], xs[i])
    return p


def p_ray(a, b): return SIGMA * math.exp(-SIGMA * norm(sub(b.pos, a.pos)))     # free flight landing on volume b


def sin_theta(xs, k):
    a = unit(sub(xs[k].pos, xs[k - 1].pos)); b = unit(sub(xs[k].pos, xs[k + 1].pos))
    c = dot(a, b); return math.sqrt(max(0.0, 1.0 - c * c))


def etas(xs, k):
    """Per-technique merge densities at interior volume vertex k, relative to both halves distance-sampling x_k."""
    pp = N * (4.0 / 3.0) * math.pi * R ** 3
    pb = N * math.pi * R * R / p_ray(xs[k + 1], xs[k])
    bb = NB * 2.0 * R * sin_theta(xs, k) / (p_ray(xs[k + 1], xs[k]) * p_ray(xs[k - 1], xs[k]))
    return {'pp3d': pp, 'pb2d': pb, 'bb1d': bb}


def eta_total(xs, k):
    n = len(xs) - 1
    if k <= 0 or k >= n or xs[k].n is not None:
        return 0.0
    return sum(etas(xs, k).values())


def brute(xs, pL, pC):
    n = len(xs) - 1; out = {}
    for s in range(0, n + 2):
        p = 1.0
        if s >= 1: p *= light_half(xs, s - 1, pL)
        if s <= n: p *= cam_half(xs, s, pC)
        out[('bpt', s)] = p
    for k in range(1, n):
        base = light_half(xs, k, pL) * cam_half(xs, k, pC)
        for t, e in etas(xs, k).items():
            out[(t, k)] = base * e
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}


def carries(xs, upto, from_light, p0):
    """VCM's recursion with eta_total at every scattering vertex (full-path knowledge stands in for the deferral)."""
    n = len(xs) - 1
    order = list(range(0, upto + 1)) if from_light else list(range(n, upto - 1, -1))
    dVCM, dVC = 1.0 / p0, 0.0
    for t in range(len(order) - 1):
        ia, ib = order[t], order[t + 1]
        a, b = xs[ia], xs[ib]
        w = unit(sub(b.pos, a.pos)); d = norm(sub(b.pos, a.pos))
        pdf = p_dir(a, w)
        rev = 0.0 if t == 0 else p_dir(a, unit(sub(xs[order[t - 1]].pos, a.pos)))
        cos_out = 1.0 if a.n is None else abs(dot(a.n, w))
        dVC = (cos_out / pdf) * (dVC * rev + dVCM + eta_total(xs, ia))
        dVCM = 1.0 / pdf
        ffa = (SIGMA * math.exp(-SIGMA * d)) if a.n is None else math.exp(-SIGMA * d)
        dVCM *= d * d / ff(b, d)
        dVC *= ffa / ff(b, d)
        if b.n is not None:
            c = abs(dot(b.n, w)); dVCM /= c; dVC /= c
    return dVCM, dVC


def rec(xs, pL, pC):
    n = len(xs) - 1; out = {}
    for s in range(1, n + 1):
        lv, cv = xs[s - 1], xs[s]
        lm, lc = carries(xs, s - 1, True, pL); cm, cc = carries(xs, s, False, pC)
        d = norm(sub(cv.pos, lv.pos)); w = unit(sub(cv.pos, lv.pos)); wr = (-w[0], -w[1], -w[2])
        cam_a = p_dir(cv, wr) * geom(lv, d, wr) * ff(lv, d)
        lig_a = p_dir(lv, w) * geom(cv, d, w) * ff(cv, d)
        lr = 0.0 if s - 1 == 0 else p_dir(lv, unit(sub(xs[s - 2].pos, lv.pos)))
        cr = 0.0 if s == n else p_dir(cv, unit(sub(xs[s + 1].pos, cv.pos)))
        out[('bpt', s)] = 1.0 / (cam_a * (eta_total(xs, s - 1) + lm + lc * lr) + 1.0
                                 + lig_a * (eta_total(xs, s) + cm + cc * cr))
    for k in range(1, n):
        v = xs[k]
        lm, lc = carries(xs, k, True, pL); cm, cc = carries(xs, k, False, pC)
        fwd = p_dir(v, unit(sub(xs[k - 1].pos, v.pos))); rv = p_dir(v, unit(sub(xs[k + 1].pos, v.pos)))
        den = eta_total(xs, k) + lm + lc * fwd + cm + cc * rv
        for t, e in etas(xs, k).items():
            out[(t, k)] = e / den
    return out


def make_path(nv, rng):
    xs = [V((rng.uniform(-.3, .3), 2.0, rng.uniform(-.3, .3)), (0.0, -1.0, 0.0))]
    for i in range(nv):
        xs.append(V((rng.uniform(-.5, .5), 1.4 - 0.35 * i + rng.uniform(-.1, .1), rng.uniform(-.5, .5))))
    cam = V((rng.uniform(1.0, 1.5), rng.uniform(-1.0, 0.0), rng.uniform(1.0, 1.5)))
    cam.n = unit(sub(xs[-1].pos, cam.pos)); xs.append(cam)
    return xs


def main():
    rng = random.Random(20261104)
    worst = 0.0
    for trial in range(60):
        xs = make_path(rng.randint(1, 6), rng)
        b = brute(xs, 0.7, 0.9); r = rec(xs, 0.7, 0.9)
        worst = max(worst, max(abs(r[k] - b[k]) / max(b[k], 1e-300) for k in r))
        if trial < 3:
            print('  %d volume vertices: ' % (len(xs) - 2) + ' '.join('%s%d %.2e' % (k[0], k[1], b[k]) for k in sorted(b) if b[k] > 1e-3))
    print('worst relative error over 60 paths: %.3e' % worst)
    # anti-vacuity: the recursion with PP3D's eta alone must miss the beam techniques' shares
    global eta_total
    full = eta_total
    eta_total = lambda xs, k: etas(xs, k)['pp3d'] if (0 < k < len(xs) - 1) else 0.0
    rng = random.Random(20261104); bad = 0.0
    for trial in range(60):
        xs = make_path(rng.randint(1, 6), rng); b = brute(xs, 0.7, 0.9); r = rec(xs, 0.7, 0.9)
        bad = max(bad, max(abs(r[k] - b[k]) / max(b[k], 1e-300) for k in r))
    eta_total = full
    print('anti-vacuity, PP3D-only eta in the recursion: worst %.3e (must be >> 1e-9)' % bad)
    sys.exit(0 if worst < 1e-9 and bad > 1e-6 else 1)


if __name__ == '__main__':
    main()
