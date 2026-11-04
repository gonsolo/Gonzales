#!/usr/bin/env python3
"""Numerical check of the VCM recursion for a medium CLOSED by delta boundaries.

A path enters a homogeneous medium through a delta vertex D (eta 1: the direction goes on, transmission is certain),
scatters at volume vertices V, and leaves through another D. Nothing connects across a D: no NEE from inside, no light
tracing from inside, no connection whose edge crosses the boundary; D itself neither connects nor merges. Ground truth
enumerates every remaining strategy (connections, both endpoint strategies, merges at V) with its full path pdf; the
recursion carries dVCM/dVC as the code does, per segment: arrival d^2/FF, the FF_a/FF_b ratio for dVC, the cosine at a
surface, and at D the SmallVCM delta rule (dVCM = 0, dVC through). Exit 1 on a mismatch.
"""
import math, random, sys

SIGMA = 3.0          # medium sigma_t; outside is vacuum
R_MERGE = 0.03
N_LIGHT = 5000.0
INV_FOUR_PI = 1.0 / (4.0 * math.pi)


def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def norm(a): return math.sqrt(dot(a, a))
def unit(a):
    n = norm(a); return (a[0] / n, a[1] / n, a[2] / n)


class V:
    def __init__(self, pos, kind, n=None):
        self.pos, self.kind, self.n = pos, kind, n     # kind: L light, C camera, D delta boundary, V volume


def is_surface(v): return v.kind in ('L', 'C', 'D')
def eta(v): return N_LIGHT * (4.0 / 3.0) * math.pi * R_MERGE ** 3     # volume merges only


def seg_sigma(a, b):
    """The medium between two consecutive path vertices: inside when either is a volume vertex or both are boundaries."""
    return SIGMA if (a.kind == 'V' or b.kind == 'V' or (a.kind == 'D' and b.kind == 'D')) else 0.0


def p_dir(v, w):
    if v.kind == 'V':
        return INV_FOUR_PI
    return abs(dot(v.n, w)) / math.pi          # emitter, lens: cosine lobes


def ff(b, d, sigma):
    """Free-flight factor of landing on b at distance d."""
    return sigma * math.exp(-sigma * d) if b.kind == 'V' else math.exp(-sigma * d)


def geom(b, d, w):
    return 1.0 / (d * d) if b.kind == 'V' else abs(dot(b.n, w)) / (d * d)


def reals(xs):
    return [i for i, v in enumerate(xs) if v.kind != 'D']


def sample_pdf(xs, i, j):
    """Density of sampling real vertex xs[j] from real vertex xs[i] (adjacent among reals), through any D between."""
    step = 1 if j > i else -1
    a, b = xs[i], xs[j]
    w = unit(sub(xs[i + step].pos, a.pos))
    p = p_dir(a, w)
    k = i
    while k != j:
        u, v = xs[k], xs[k + step]
        d = norm(sub(v.pos, u.pos))
        p *= ff(v, d, seg_sigma(u, v)) if v.kind != 'D' else math.exp(-seg_sigma(u, v) * d)
        k += step
    # geometry of the whole straight chain (eta 1): solid angle at a -> measure at b
    dtot = norm(sub(b.pos, a.pos))
    return p * geom(b, dtot, unit(sub(b.pos, a.pos)))


def half(xs, R, k, from_light, p0):
    """Density of the subpath reaching real index R[k] from its end."""
    p = p0
    idx = range(1, k + 1) if from_light else range(len(R) - 2, k - 1, -1)
    for m in idx:
        p *= sample_pdf(xs, R[m - 1], R[m]) if from_light else sample_pdf(xs, R[m + 1], R[m])
    return p


def brute(xs, pL, pC):
    R = reals(xs); n = len(R) - 1
    strat = {}
    for s in range(0, n + 2):            # s real light vertices
        if 1 <= s <= n:                  # connecting edge R[s-1] - R[s] must not cross a boundary
            if R[s] - R[s - 1] != 1:
                continue
        p = 1.0
        if s >= 1: p *= half(xs, R, s - 1, True, pL)
        if s <= n: p *= half(xs, R, s, False, pC)
        strat[('c', s)] = p
    for k in range(1, n):
        if xs[R[k]].kind == 'V':
            strat[('m', k)] = eta(xs[R[k]]) * half(xs, R, k, True, pL) * half(xs, R, k, False, pC)
    tot = sum(strat.values())
    return {k: v / tot for k, v in strat.items()}


def carries(xs, upto, from_light, p0):
    """dVCM/dVC walked over every vertex (D included) from one end to real vertex `upto`, as the code does."""
    n = len(xs) - 1
    order = list(range(0, upto + 1)) if from_light else list(range(n, upto - 1, -1))
    dVCM, dVC = 1.0 / p0, 0.0
    for t in range(len(order) - 1):
        a, b = xs[order[t]], xs[order[t + 1]]
        w = unit(sub(b.pos, a.pos)); d = norm(sub(b.pos, a.pos)); sg = seg_sigma(a, b)
        prev = xs[order[t - 1]] if t > 0 else None
        if a.kind == 'D':
            dVCM = 0.0                      # delta scatter: dVC through (pdfs cancel, cos_out at eta 1 = cos_in)
            dVC *= abs(dot(a.n, w))
        else:
            pdf = p_dir(a, w)
            rev = 0.0 if prev is None else p_dir(a, unit(sub(prev.pos, a.pos)))
            cos_out = 1.0 if a.kind == 'V' else abs(dot(a.n, w))
            e = eta(a) if (a.kind == 'V' and prev is not None) else 0.0
            dVC = (cos_out / pdf) * (dVC * rev + dVCM + e)
            dVCM = 1.0 / pdf
        ffa = (sg * math.exp(-sg * d)) if a.kind == 'V' else math.exp(-sg * d)
        ffb = ff(b, d, sg)
        dVCM *= d * d / ffb
        dVC *= ffa / ffb
        if is_surface(b):
            c = abs(dot(b.n, w)); dVCM /= c; dVC /= c
    return dVCM, dVC


def rec_weights(xs, pL, pC):
    R = reals(xs); n = len(R) - 1; out = {}
    for s in range(1, n + 1):
        if R[s] - R[s - 1] != 1:
            continue
        lv, cv = xs[R[s - 1]], xs[R[s]]
        lm, lc = carries(xs, R[s - 1], True, pL)
        cm, cc = carries(xs, R[s], False, pC)
        d = norm(sub(cv.pos, lv.pos)); w = unit(sub(cv.pos, lv.pos)); wr = (-w[0], -w[1], -w[2]); sg = seg_sigma(lv, cv)
        cam_a = p_dir(cv, wr) * geom(lv, d, wr) * ff(lv, d, sg)
        lig_a = p_dir(lv, w) * geom(cv, d, w) * ff(cv, d, sg)
        lr = 0.0 if s - 1 == 0 else p_dir(lv, unit(sub(xs[R[s - 1] - 1].pos, lv.pos)))
        cr = 0.0 if s == n else p_dir(cv, unit(sub(xs[R[s] + 1].pos, cv.pos)))
        el = eta(lv) if (lv.kind == 'V') else 0.0
        ec = eta(cv) if (cv.kind == 'V') else 0.0
        out[('c', s)] = 1.0 / (cam_a * (el + lm + lc * lr) + 1.0 + lig_a * (ec + cm + cc * cr))
    for k in range(1, n):
        v = xs[R[k]]
        if v.kind != 'V':
            continue
        lm, lc = carries(xs, R[k], True, pL)
        cm, cc = carries(xs, R[k], False, pC)
        fwd = p_dir(v, unit(sub(xs[R[k] - 1].pos, v.pos)))
        rev = p_dir(v, unit(sub(xs[R[k] + 1].pos, v.pos)))
        out[('m', k)] = 1.0 / ((lm + lc * fwd) / eta(v) + 1.0 + (cm + cc * rev) / eta(v))
    return out


def make_path(n_inside, rng):
    """Light above, D entry, n_inside volume vertices inside a ball of radius 1, D exit, camera; straight D chains."""
    def on_sphere(dirn):
        return dirn
    xs = [V((rng.uniform(-.5, .5), 3.0, rng.uniform(-.5, .5)), 'L', (0, -1, 0))]
    inside = [(rng.uniform(-.6, .6), rng.uniform(-.6, .6), rng.uniform(-.6, .6)) for _ in range(n_inside)]
    def hit_sphere(o, p):     # where the segment from outside point o to inside point p crosses the unit sphere
        dvec = sub(p, o); a = dot(dvec, dvec); b = 2 * dot(o, dvec); c = dot(o, o) - 1.0
        t = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
        q = (o[0] + t * dvec[0], o[1] + t * dvec[1], o[2] + t * dvec[2]); return q
    cam = V((rng.uniform(1.5, 2.5), rng.uniform(1.0, 2.0), rng.uniform(1.5, 2.5)), 'C', None)
    e1 = hit_sphere(xs[0].pos, inside[0]); xs.append(V(e1, 'D', unit(e1)))
    for p in inside: xs.append(V(p, 'V'))
    e2 = hit_sphere(cam.pos, inside[-1]); xs.append(V(e2, 'D', unit(e2)))
    cam.n = unit(sub(e2, cam.pos)); xs.append(cam)
    return xs


def main():
    rng = random.Random(20261104)
    worst = 0.0
    for trial in range(40):
        xs = make_path(rng.randint(1, 7), rng)
        b = brute(xs, 0.7, 0.9); r = rec_weights(xs, 0.7, 0.9)
        for k in r:
            worst = max(worst, abs(r[k] - b[k]) / max(b[k], 1e-300))
        if trial < 4:
            print('  %d inside: ' % (len(xs) - 4) + ' '.join('%s%d %.3e/%.3e' % (k[0], k[1], b[k], r[k]) for k in sorted(r)))
    print('worst relative error over 40 paths: %.3e' % worst)
    sys.exit(0 if worst < 1e-9 else 1)


if __name__ == '__main__':
    main()
