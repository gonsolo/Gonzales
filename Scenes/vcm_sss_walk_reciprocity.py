#!/usr/bin/env python3
"""Do camera and light subsurface walks estimate the same radiance? Half-space, isotropic, distant light, far camera.

Model: refract in, internal Fresnel at the exit, Lambertian out (PT's walk). Camera estimate: enters along wc, ends at NEE
toward wl. Light estimate: enters along wl, ends at the connection toward wc. MIS over the two needs them to agree."""
import numpy as np
ETA = 1.33

def fresnel(ci, eta):
    s2 = (1 - ci * ci) / (eta * eta)
    tir = s2 >= 1
    ct = np.sqrt(np.maximum(1 - s2, 0))
    rs = (ci - eta * ct) / (ci + eta * ct)
    rp = (eta * ci - ct) / (eta * ci + ct)
    return np.where(tir, 1.0, 0.5 * (rs * rs + rp * rp))

def escape(mu0, albedo, rng, n, internal_fresnel=True, steps=4000):
    """Weight of walks from z=0 going down with cosine mu0; absorption by albedo weighting."""
    z = np.zeros(n); mu = np.full(n, mu0); w = np.ones(n); out = np.zeros(n); alive = np.ones(n, bool)
    for _ in range(steps):
        i = np.nonzero(alive)[0]
        if i.size == 0: break
        t = -np.log(1 - rng.random(i.size))
        nz = z[i] + mu[i] * t       # z grows downward, mu > 0 goes down
        hit = nz <= 0
        if hit.any():
            h = i[hit]
            if internal_fresnel:    # from inside: F reflects back in
                ci = np.abs(mu[h])
                F = fresnel(ci, 1 / ETA)
                refl = rng.random(h.size) < F
                out[h[~refl]] = w[h[~refl]]; alive[h[~refl]] = False
                z[h[refl]] = 0; mu[h[refl]] = np.abs(mu[h[refl]])
            else:
                out[h] = w[h]; alive[h] = False
        s = i[~hit]
        z[s] = nz[~hit]; w[s] *= albedo; mu[s] = 2 * rng.random(s.size) - 1
        d = (w < 1e-3) & alive      # Russian roulette would change nothing in the mean; drop the negligible tail
        alive[d] = False
    return out.mean(), out.std() / np.sqrt(n)

def refr_mu(ci): return np.sqrt(max(1 - (1 - ci * ci) / (ETA * ETA), 0))

rng = np.random.default_rng(1)
N = 400000
print("albedo  cos_c cos_l |  ours: camera   light    light/camera")
for albedo in (0.8, 0.95, 0.99):
    for cc, cl in ((1.0, 1.0), (1.0, 0.5), (1.0, 0.2), (0.5, 0.2), (0.2, 1.0)):
        ec, _ = escape(refr_mu(cc), albedo, rng, N)
        el, _ = escape(refr_mu(cl), albedo, rng, N)
        Fc = float(fresnel(np.array(cc), ETA)); Fl = float(fresnel(np.array(cl), ETA))
        # L_o / (E/pi): E = cos_l from the distant light; Lambertian exit in both directions.
        cam = (1 - Fc) * ec
        lig = (1 - Fl) * el
        print(f"{albedo:5.2f}  {cc:4.1f} {cl:4.1f}  | {cam:12.4f} {lig:8.4f} {lig / cam:10.3f}")
