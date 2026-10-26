#!/usr/bin/env python3
"""Numerical check of VCM's MIS weights with MNEE as one more strategy.

Sibling of vcm_area_mis_derivation.py, same method: enumerate every strategy's
density over the path space (y0, x_0..x_{n-1}) directly, form balance weights,
and compare them with the local dVCM/dVC formulas. Weights must sum to 1.

Scene: a pool. The light y0 is above a flat water surface (z = 0, index N_W);
floor x_0, wall x_1 and the lens are under water. The edge y0 -> x_0 crosses
the surface at one refraction point v, so:
  * the straight connection y0 -> x_0 (s = 1) is impossible,
  * MNEE samples y0 by area at x_0 and solves for v (deterministic: density
    P_A, as in Hanika et al. 2015),
  * the camera reaches y0 by BSDF sampling at x_0 and refracting at v (s = 0),
  * a light path reaches x_0 through v; connections, splats and merges follow.

Findings (2026-10-26):
  * Today MNEE is added at full weight on top of VCM's own strategies: the weights sum to 2.
  * MNEE's weight is SmallVCM's DirectIllumination with the chain's densities: p_bsdf (1 - F) J_cam on the light
    side of the ratio, p_emit (1 - F) J_light on the camera side.
  * A light vertex reached through delta vertices only gets dVCM = 1 / (p_emit (1 - F) J_light) instead of 0, and
    the camera's emission hit through the chain gets MNEE's ratio P_A / (p_bsdf (1 - F) J_cam).
  * SmallVCM's delta rule dVC *= cos_out misses refraction's solid-angle compression: it needs (n_in / n_out)^2.
    Without MNEE the omission cancels (camera and light carries share it, today's weights still sum to 1); against
    MNEE's exact chain densities it does not.

The chain's Jacobians come from finite differences of the refraction solve:
  J_cam   = |d w_0 / dA_y|   solid angle at x_0 per area at y0
  J_light = |d w_y / dA_x0|  solid angle at y0 per area at x_0
and etendue relates them: n_0^2 cos_0 J_cam = n_y^2 cos_y J_light.
Selecting refraction has probability 1 - F on both sides (Fresnel is symmetric).
"""
import math

R_MERGE = 0.03
N_LIGHT = 20000.0
KERNEL = math.pi * R_MERGE * R_MERGE
ETA_VCM = N_LIGHT * KERNEL
P_CAM_AREA = 1.0
LIGHT_AREA = 0.25
P_A = 1.0 / LIGHT_AREA
N_SPLAT = 1.0
N_W = 1.33                      # water below z = 0, air above


def add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def norm(a):   return math.sqrt(dot(a, a))
def unit(a):   return mul(a, 1.0 / norm(a))
def neg(a):    return (-a[0], -a[1], -a[2])


class V:
    def __init__(self, pos, normal):
        self.pos = pos
        self.n = normal


def tangents(n):
    a = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    t = unit(cross(n, a))
    return t, cross(n, t)


def refraction_point(y, x):
    """v on z = 0 minimising the optical path |y - v| + N_W |v - x| (Fermat)."""
    py, px = (y[0], y[1], 0.0), (x[0], x[1], 0.0)
    dv = sub(px, py)
    v = lambda t: add(py, mul(dv, t))
    df = lambda t: dot(sub(v(t), y), dv) / norm(sub(v(t), y)) + N_W * dot(sub(v(t), x), dv) / norm(sub(v(t), x))
    lo, hi = 0.0, 1.0                                   # Snell: the optical path's derivative is monotonic in t
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if df(m) < 0.0: lo = m
        else: hi = m
    return v(0.5 * (lo + hi))


def fresnel(cos_i, eta):
    """Unpolarised dielectric Fresnel, eta = n_t / n_i."""
    s2 = max(0.0, 1.0 - cos_i * cos_i) / (eta * eta)
    if s2 >= 1.0: return 1.0
    c_t = math.sqrt(1.0 - s2)
    rs = (cos_i - eta * c_t) / (cos_i + eta * c_t)
    rp = (eta * cos_i - c_t) / (eta * cos_i + c_t)
    return 0.5 * (rs * rs + rp * rp)


def solid_angle_jacobian(fixed, moving):
    """|d w / dA| at `fixed` (direction toward the refraction point) per area of `moving` in its tangent plane."""
    h = 1e-5
    t, b = tangents(moving.n)
    w = lambda p: unit(sub(refraction_point(*((fixed.pos, p) if fixed.pos[2] > 0 else (p, fixed.pos))), fixed.pos))
    du = mul(sub(w(add(moving.pos, mul(t, h))), w(add(moving.pos, mul(t, -h)))), 0.5 / h)
    dv = mul(sub(w(add(moving.pos, mul(b, h))), w(add(moving.pos, mul(b, -h)))), 0.5 / h)
    return norm(cross(du, dv))


class Chain:
    """The refraction y0 -> v -> x_0 and everything both sides need from it."""
    def __init__(self, y0, x0):
        self.v = refraction_point(y0.pos, x0.pos)
        self.w0 = unit(sub(self.v, x0.pos))             # at x_0, toward the surface
        self.wy = unit(sub(self.v, y0.pos))             # at y0, toward the surface
        self.T = 1.0 - fresnel(abs(self.wy[2]), N_W)
        self.J_cam = solid_angle_jacobian(x0, y0)
        self.J_light = solid_angle_jacobian(y0, x0)
        self.cos0 = abs(dot(x0.n, self.w0))
        self.cosy = abs(dot(y0.n, self.wy))


def p_dir(v, w):  return abs(dot(v.n, w)) / math.pi
def p_emit(y0, w): return max(0.0, dot(y0.n, w)) / math.pi


def g_to_area(a, b):
    d = norm(sub(b.pos, a.pos))
    return abs(dot(b.n, unit(sub(b.pos, a.pos)))) / (d * d)


def edge_pdf(a, b):    return p_dir(a, unit(sub(b.pos, a.pos))) * g_to_area(a, b)
def cam_edge_pdf(lens, b): return g_to_area(lens, b)


# ── ground truth ───────────────────────────────────────────────────────────

def strategy_densities(y0, xs, lens, etas, ch):
    n = len(xs)
    C = [0.0] * n
    C[n - 1] = P_CAM_AREA * cam_edge_pdf(lens, xs[n - 1])
    for k in range(n - 2, -1, -1):
        C[k] = C[k + 1] * edge_pdf(xs[k + 1], xs[k])
    L = [0.0] * n
    L[0] = P_A * p_emit(y0, ch.wy) * ch.T * ch.J_light
    for k in range(1, n):
        L[k] = L[k - 1] * edge_pdf(xs[k - 1], xs[k])
    d = {("s", 0): C[0] * p_dir(xs[0], ch.w0) * ch.T * ch.J_cam,
         ("mnee",): P_A * C[0]}
    for s in range(2, n + 1):
        d[("s", s)] = L[s - 2] * C[s - 1]
    d[("t1", n - 1)] = N_SPLAT * L[n - 1] * P_CAM_AREA
    for k in range(n):
        d[("merge", k)] = etas[k] * L[k] * C[k]
    return d


# ── carries ────────────────────────────────────────────────────────────────

def scatter(dvcm, dvc, cos_out, pdf_fwd, pdf_rev, eta):
    return 1.0 / pdf_fwd, (cos_out / pdf_fwd) * (dvc * pdf_rev + dvcm + eta)


def arrive(dvcm, dvc, a, b):
    d = norm(sub(b.pos, a.pos))
    cf = abs(dot(b.n, unit(sub(b.pos, a.pos))))
    return dvcm * d * d / cf, dvc / cf


def light_carries(y0, xs, etas, upto, ch, mnee):
    """Arrival carries at xs[upto]. The light origin's carries go through the delta vertex v (SmallVCM: arrive at v,
    dVCM = 0, dVC *= cos_out) to x_0; `mnee` gives x_0 MNEE's term instead of that 0."""
    cos_emit = dot(y0.n, ch.wy)
    emission_pdf_w = P_A * cos_emit / math.pi
    dvcm, dvc = P_A / emission_pdf_w, cos_emit / emission_pdf_w
    dy = norm(sub(ch.v, y0.pos))
    cos_v_in = abs(ch.wy[2])
    dvcm, dvc = dvcm * dy * dy / cos_v_in, dvc / cos_v_in                # arrive at v
    cos_v_out = abs(ch.w0[2])
    dvcm, dvc = 0.0, dvc * cos_v_out * (1.0 / (N_W * N_W) if mnee else 1.0)  # delta at v, air -> water
    dx = norm(sub(xs[0].pos, ch.v))
    dvcm, dvc = dvcm * dx * dx / ch.cos0, dvc / ch.cos0                  # arrive at x_0
    if mnee:
        dvcm = 1.0 / (p_emit(y0, ch.wy) * ch.T * ch.J_light)
    for i in range(0, upto):
        a, b = xs[i], xs[i + 1]
        w = unit(sub(b.pos, a.pos))
        wprev = ch.w0 if i == 0 else unit(sub(xs[i - 1].pos, a.pos))
        dvcm, dvc = scatter(dvcm, dvc, abs(dot(a.n, w)), p_dir(a, w), p_dir(a, wprev), etas[i])
        dvcm, dvc = arrive(dvcm, dvc, a, b)
    return dvcm, dvc


def camera_carries(xs, lens, etas, upto):
    n = len(xs)
    dvcm, dvc = arrive(N_SPLAT / P_CAM_AREA, 0.0, lens, xs[n - 1])
    for i in range(n - 1, upto, -1):
        a, b = xs[i], xs[i - 1]
        w = unit(sub(b.pos, a.pos))
        prev = lens if i == n - 1 else xs[i + 1]
        dvcm, dvc = scatter(dvcm, dvc, abs(dot(a.n, w)), p_dir(a, w), p_dir(a, unit(sub(prev.pos, a.pos))), etas[i])
        dvcm, dvc = arrive(dvcm, dvc, a, b)
    return dvcm, dvc


def cam_pred(xs, lens, k): return lens if k == len(xs) - 1 else xs[k + 1]


# ── weights ────────────────────────────────────────────────────────────────

def w_connect(y0, xs, lens, etas, s, ch, mnee):
    """s >= 2: light vertex xs[s-2], camera vertex xs[s-1]."""
    lv, cv = xs[s - 2], xs[s - 1]
    lvcm, lvc = light_carries(y0, xs, etas, s - 2, ch, mnee)
    ccvcm, ccvc = camera_carries(xs, lens, etas, s - 1)
    dir_lc = unit(sub(cv.pos, lv.pos))
    light_rev_w = p_dir(lv, ch.w0 if s == 2 else unit(sub(xs[s - 3].pos, lv.pos)))
    cam_rev_w = p_dir(cv, unit(sub(cam_pred(xs, lens, s - 1).pos, cv.pos)))
    d = norm(sub(cv.pos, lv.pos))
    cam_dir_a = p_dir(cv, neg(dir_lc)) * abs(dot(lv.n, dir_lc)) / (d * d)
    light_dir_a = p_dir(lv, dir_lc) * abs(dot(cv.n, dir_lc)) / (d * d)
    w_light = cam_dir_a * (etas[s - 2] + lvcm + lvc * light_rev_w)
    w_camera = light_dir_a * (etas[s - 1] + ccvcm + ccvc * cam_rev_w)
    return 1.0 / (w_light + 1.0 + w_camera)


def w_splat(y0, xs, lens, etas, ch, mnee):
    n = len(xs)
    lv = xs[n - 1]
    lvcm, lvc = light_carries(y0, xs, etas, n - 1, ch, mnee)
    rev = p_dir(lv, ch.w0 if n == 1 else unit(sub(xs[n - 2].pos, lv.pos)))
    w_light = (P_CAM_AREA * cam_edge_pdf(lens, lv) / N_SPLAT) * (etas[n - 1] + lvcm + lvc * rev)
    return 1.0 / (w_light + 1.0)


def w_merge(y0, xs, lens, etas, k, ch, mnee):
    lvcm, lvc = light_carries(y0, xs, etas, k, ch, mnee)
    ccvcm, ccvc = camera_carries(xs, lens, etas, k)
    x = xs[k]
    w_to_lpred = ch.w0 if k == 0 else unit(sub(xs[k - 1].pos, x.pos))
    w_to_cpred = unit(sub(cam_pred(xs, lens, k).pos, x.pos))
    w_light = (lvcm + lvc * p_dir(x, w_to_lpred)) / etas[k]
    w_camera = (ccvcm + ccvc * p_dir(x, w_to_cpred)) / etas[k]
    return 1.0 / (w_light + 1.0 + w_camera)


def w_emission_hit(y0, xs, lens, etas, ch, mnee):
    """Camera: scatter at x_0 toward v, delta at v, arrive at y0; SmallVCM's 1 / (1 + directPdfA dVCM + emissionPdfW dVC)
    with dVCM = 0 after the delta, plus MNEE's ratio to this strategy."""
    x0 = xs[0]
    dvcm, dvc = camera_carries(xs, lens, etas, 0)
    prev = cam_pred(xs, lens, 0)
    dvcm, dvc = scatter(dvcm, dvc, ch.cos0, p_dir(x0, ch.w0), p_dir(x0, unit(sub(prev.pos, x0.pos))), etas[0])
    dv = norm(sub(ch.v, x0.pos))
    dvcm, dvc = dvcm * dv * dv / abs(ch.w0[2]), dvc / abs(ch.w0[2])     # arrive at v
    dvcm, dvc = 0.0, dvc * abs(ch.wy[2]) * (N_W * N_W if mnee else 1.0)       # delta at v, water -> air
    dy = norm(sub(y0.pos, ch.v))
    dvcm, dvc = dvcm * dy * dy / ch.cosy, dvc / ch.cosy                 # arrive at y0
    inv = 1.0 + P_A * dvcm + P_A * p_emit(y0, ch.wy) * dvc
    if mnee:
        inv += P_A / (p_dir(x0, ch.w0) * ch.T * ch.J_cam)
    return 1.0 / inv


def w_mnee(y0, xs, lens, etas, ch):
    """SmallVCM's DirectIllumination with the chain: the generalized geometry replaces cos/d^2 on both sides."""
    x0 = xs[0]
    dvcm, dvc = camera_carries(xs, lens, etas, 0)
    bsdf_rev = p_dir(x0, unit(sub(cam_pred(xs, lens, 0).pos, x0.pos)))
    w_light = p_dir(x0, ch.w0) * ch.T * ch.J_cam / P_A
    w_camera = p_emit(y0, ch.wy) * ch.T * ch.J_light * (etas[0] + dvcm + dvc * bsdf_rev)
    return 1.0 / (w_light + 1.0 + w_camera)


# ── report ─────────────────────────────────────────────────────────────────

def run(name, y0, xs, lens, etas):
    n = len(xs)
    ch = Chain(y0, xs[0])
    etendue = (N_W * N_W * ch.cos0 * ch.J_cam) / (ch.cosy * ch.J_light)
    d = strategy_densities(y0, xs, lens, etas, ch)
    tot = sum(d.values())
    truth = {k: v / tot for k, v in d.items()}
    weights = {}
    for mnee in (False, True):
        w = {("s", 0): w_emission_hit(y0, xs, lens, etas, ch, mnee), ("t1", n - 1): w_splat(y0, xs, lens, etas, ch, mnee)}
        for s in range(2, n + 1):
            w[("s", s)] = w_connect(y0, xs, lens, etas, s, ch, mnee)
        for k in range(n):
            w[("merge", k)] = w_merge(y0, xs, lens, etas, k, ch, mnee)
        w[("mnee",)] = w_mnee(y0, xs, lens, etas, ch) if mnee else 1.0   # today: MNEE added at full weight
        weights[mnee] = w
    print(f"\n=== {name} ===   etendue n0^2 cos0 J_cam / (cosy J_light) = {etendue:.6f} (want 1)")
    print(f"  {'strategy':12s} {'truth':>10s} {'today':>10s} {'with MNEE':>10s}")
    for k in sorted(truth, key=str):
        print(f"  {str(k):12s} {truth[k]:10.6f} {weights[False][k]:10.6f} {weights[True][k]:10.6f}")
    print(f"  {'SUM':12s} {sum(truth.values()):10.6f} {sum(weights[False].values()):10.6f} {sum(weights[True].values()):10.6f}")
    return max(abs(weights[True][k] - truth[k]) for k in truth)


def scene():
    y0 = V((0.4, 0.2, 1.5), unit((-0.1, 0.05, -1.0)))     # lamp over the pool, facing down
    lens = V((0.3, -2.5, -0.4), None)                    # camera under water
    x_a = V((0.1, 0.3, -1.0), unit((0.02, 0.0, 1.0)))     # pool floor
    x_b = V((-1.2, 0.0, -0.5), unit((1.0, 0.05, 0.0)))    # pool wall
    return y0, lens, x_a, x_b


if __name__ == "__main__":
    y0, lens, x_a, x_b = scene()
    err = 0.0
    for n_light in (2.0e4, 2.0e6):
        N_LIGHT = n_light
        ETA_VCM = N_LIGHT * KERNEL
        print(f"\n################ {n_light:.0e} light paths, eta = {ETA_VCM:.4g} ################")
        err = max(err, run("1 vertex", y0, [x_a], lens, [ETA_VCM]))
        err = max(err, run("2 vertices", y0, [x_a, x_b], lens, [ETA_VCM, ETA_VCM]))
        err = max(err, run("2 vertices, eta(x) varies", y0, [x_a, x_b], lens, [0.1 * ETA_VCM, ETA_VCM]))
    print(f"\nmax |with MNEE - truth| = {err:.2e}")
