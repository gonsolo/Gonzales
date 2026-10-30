#!/usr/bin/env python3
"""VCM across a subsurface walk: the interior vertices are path vertices (UPBP), so both directions have exact densities.
A camera walk x_a -> x_b and a light walk x_b -> x_a produce the same extended path. With a reciprocal boundary lobe the
contribution is the same, and the two strategies differ only by r = p_rev / p_fwd of the interior. Half-space, g = 0, one
wavelength, the path tracer's guiding: distance stretch and Dwivedi direction mix, unguided at guided_fraction 0.
Reports how far r is from 1, and the variance of the balance-weighted estimate with exact r against the assumption r = 1."""
import numpy as np

def walk(alb, rng, gf_early=0.5, gf_late=0.75, sim=9, cap=3000):
    dl = 1 / np.sqrt(max(1 - alb ** (2.44294 - 0.0215813 * alb + 0.578637 / alb), 1e-12)) if alb > 0 else 1.0
    plog = np.log((dl + 1) / max(dl - 1, 1e-7))
    def dir_mix(c, gf):      # direction density over the unit sphere; c = cos against the outward normal
        return (1 - gf) / (4 * np.pi) + gf / (2 * np.pi * (dl - c) * plog)
    def seg(t, c, gf, hit):  # distance density with the same stretch mixture as the renderer; sigma_t = 1
        s1 = np.exp(-t) if hit else np.exp(-t)
        sg = 1 - c / dl
        s2 = np.exp(-sg * t) if hit else sg * np.exp(-sg * t)
        return (1 - gf) * s1 + gf * s2
    z = 0.0; c = -np.sqrt(rng.random()); w = 1.0; lf = 0.0; lr = 0.0; segs = []; i = 0
    while True:
        gf = gf_late if i > sim else gf_early
        if i == 0: gf = 0.0  # the first segment's distance is unguided: its direction is the entry lobe's
        u = rng.random() < gf
        sg = 1 - c / dl if u else 1.0
        t = -np.log(1 - rng.random()) / max(sg, 1e-9)
        hit = c > 0 and z + c * t >= 0
        if hit: t = -z / c
        segs.append((t, c, hit, i))
        p = seg(t, c, gf, hit)
        w *= (np.exp(-t) if hit else alb * np.exp(-t)) / p
        lf += np.log(p)
        if hit: break
        z += c * t; i += 1
        if i >= cap: return None
        gf = gf_late if i > sim else gf_early
        if rng.random() < gf:
            xi = rng.random()
            nc = dl - (dl + 1) * np.exp(-xi * plog)
        else:
            nc = 2 * rng.random() - 1
        lf += np.log(dir_mix(nc, gf))
        w *= (1 / (4 * np.pi)) / dir_mix(nc, gf)
        c = nc
    n = len(segs)
    for k in range(n):                                  # the reversed walk meets the same segments in opposite order
        t, cf, hit, _ = segs[n - 1 - k]
        gf = gf_late if k > sim else gf_early
        if k == 0: gf = 0.0
        lr += np.log(seg(t, -cf, gf, k == n - 1))
        if k < n - 1:
            gfd = gf_late if k + 1 > sim else gf_early
            lr += np.log(dir_mix(-segs[n - 2 - k][1], gfd))
    return w, lr - lf, n

rng = np.random.default_rng(3)
print("albedo guided  walks mean_steps | log10 r: median |r| p95 p99.9 | mean      rel.var(all)  rel.var(exact r)  rel.var(r=1)")
for alb in (0.5, 0.8, 0.95, 0.99):
    for gfe, gfl, name in ((0.0, 0.0, "off"), (0.5, 0.75, "PT ")):
        V = []; R = []; S = []
        for _ in range(6000):
            o = walk(alb, rng, gfe, gfl)
            if o is None: continue
            V.append(o[0]); R.append(o[1]); S.append(o[2])
        V = np.array(V); R = np.array(R)
        r = np.exp(R)
        a = np.abs(R) / np.log(10)
        # A pair of equally sampled strategies: each walk weighs itself by the balance heuristic, 2x because each strategy
        # contributes half the samples. Mean must stay at V.mean(); only the variance differs.
        ex = V * 2 / (1 + r)
        print(f"{alb:5.2f}  {name}  {len(V):6d} {np.mean(S):8.1f}   | {np.median(a):8.3f} {np.percentile(a,95):7.3f} {np.percentile(a,99.9):7.3f} | "
              f"{V.mean():8.4f}  {V.var()/V.mean()**2:9.3f}  {ex.var()/ex.mean()**2:12.3f}  {V.var()/V.mean()**2:12.3f}   exact-mean/mean {ex.mean()/V.mean():.4f}")
