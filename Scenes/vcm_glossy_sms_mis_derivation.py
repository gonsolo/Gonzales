#!/usr/bin/env python3
"""Numerical check of VCM's MIS weights with glossy SMS: MNEE from a GLOSSY vertex through a chain that REFLECTS.

Sibling of vcm_mnee_mis_derivation.py (same method, same scene layout): every strategy's density over the path space
(y0, x_0..x_{n-1}), balance weights from it, compared with the local dVCM/dVC formulas the renderer uses.

What changes against MNEE from a diffuse vertex:
  * x_0 is glossy: its lobe is not the cosine and not symmetric, so every place that used |cos|/pi at x_0 now takes
    the lobe density with its own forward and reverse directions (scatter carries, connections, merges, the emission
    hit through the chain and glossy SMS's own weight).
  * CHAIN = "reflect": the lamp is seen in the water surface, reflected once (probability F), both ends in air, so no
    refraction compression; CHAIN = "refract" is the pool of the sibling script.
  * Glossy SMS's density in these weights is the deterministic solve's, P_A / J_cam. The renderer finds the root from
    a random lobe seed with chance q and multiplies by an unbiased estimate of 1/q; any density that every strategy
    evaluates alike gives weights that sum to 1, so the partition stays exact (the check below), only no longer the
    balance heuristic of the true densities.
Result: the sibling's formulas carry over with p_bsdf the lobe's density, T the chain's product of F (reflection) and
1 - F (refraction), and the compression only at refractions.
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
CHAIN = "refract"               # or "reflect": lamp, x_0 and the lens above the water, seen in its surface
K_LOBE = 12.0                   # glossy x_0: a lobe around a tilted mirror direction, deliberately not reciprocal


def add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def norm(a):   return math.sqrt(dot(a, a))
def unit(a):   return mul(a, 1.0 / norm(a))
def neg(a):    return (-a[0], -a[1], -a[2])


class V:
    def __init__(self, pos, normal, glossy=False):
        self.pos = pos
        self.n = normal
        self.glossy = glossy


def tangents(n):
    a = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
    t = unit(cross(n, a))
    return t, cross(n, t)


def chain_point(y, x):
    if CHAIN == "reflect":
        ym = (y[0], y[1], -y[2])                         # unfold the mirror
        t = x[2] / (x[2] - ym[2])
        return add(x, mul(sub(ym, x), t))
    return refraction_point(y, x)


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
    """|d w / dA| at `fixed` (direction toward the chain point) per area of `moving` in its tangent plane."""
    return _jac(fixed, moving)


def _jac(fixed, moving):
    """|d w / dA| at `fixed` (direction toward the refraction point) per area of `moving` in its tangent plane."""
    h = 1e-5
    t, b = tangents(moving.n)
    w = lambda p: unit(sub(chain_point(*((fixed.pos, p) if fixed.is_light else (p, fixed.pos))), fixed.pos))
    du = mul(sub(w(add(moving.pos, mul(t, h))), w(add(moving.pos, mul(t, -h)))), 0.5 / h)
    dv = mul(sub(w(add(moving.pos, mul(b, h))), w(add(moving.pos, mul(b, -h)))), 0.5 / h)
    return norm(cross(du, dv))


class Chain:
    """The refraction y0 -> v -> x_0 and everything both sides need from it."""
    def __init__(self, y0, x0):
        y0.is_light, x0.is_light = True, False
        self.v = chain_point(y0.pos, x0.pos)
        self.w0 = unit(sub(self.v, x0.pos))             # at x_0, toward the surface
        self.wy = unit(sub(self.v, y0.pos))             # at y0, toward the surface
        F = fresnel(abs(self.wy[2]), N_W)
        self.T = F if CHAIN == "reflect" else 1.0 - F
        self.n2 = 1.0 if CHAIN == "reflect" else N_W * N_W     # (n_x0 / n_y)^2: compression only at a refraction
        self.J_cam = solid_angle_jacobian(x0, y0)
        self.J_light = solid_angle_jacobian(y0, x0)
        self.cos0 = abs(dot(x0.n, self.w0))
        self.cosy = abs(dot(y0.n, self.wy))


def p_dir(v, w, w_other=None):
    """Density of scattering toward w at v, having come from (or going on toward) w_other."""
    if not v.glossy:
        return abs(dot(v.n, w)) / math.pi
    m = sub(mul(v.n, 2.0 * dot(v.n, w_other)), w_other)        # mirror of w_other
    axis = unit(add(m, mul(v.n, 0.3)))
    return (K_LOBE + 1.0) / (2.0 * math.pi) * ((1.0 + dot(axis, w)) * 0.5) ** K_LOBE
def p_emit(y0, w): return max(0.0, dot(y0.n, w)) / math.pi


def g_to_area(a, b):
    d = norm(sub(b.pos, a.pos))
    return abs(dot(b.n, unit(sub(b.pos, a.pos)))) / (d * d)


def edge_pdf(a, b, prev=None):
    return p_dir(a, unit(sub(b.pos, a.pos)), None if prev is None else unit(sub(prev.pos, a.pos))) * g_to_area(a, b)
def cam_edge_pdf(lens, b): return g_to_area(lens, b)


# ── ground truth ───────────────────────────────────────────────────────────

def strategy_densities(y0, xs, lens, etas, ch):
    n = len(xs)
    C = [0.0] * n
    C[n - 1] = P_CAM_AREA * cam_edge_pdf(lens, xs[n - 1])
    for k in range(n - 2, -1, -1):
        C[k] = C[k + 1] * edge_pdf(xs[k + 1], xs[k], cam_pred(xs, lens, k + 1))
    L = [0.0] * n
    L[0] = P_A * p_emit(y0, ch.wy) * ch.T * ch.J_light
    for k in range(1, n):
        L[k] = L[k - 1] * edge_pdf(xs[k - 1], xs[k], V(ch.v, None) if k == 1 else xs[k - 2])
    wcp = unit(sub(cam_pred(xs, lens, 0).pos, xs[0].pos))
    d = {("s", 0): C[0] * p_dir(xs[0], ch.w0, wcp) * ch.T * ch.J_cam,
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
    dvcm, dvc = 0.0, dvc * cos_v_out * (1.0 / ch.n2 if mnee else 1.0)  # delta at v, air -> water
    dx = norm(sub(xs[0].pos, ch.v))
    dvcm, dvc = dvcm * dx * dx / ch.cos0, dvc / ch.cos0                  # arrive at x_0
    if mnee:
        dvcm = 1.0 / (p_emit(y0, ch.wy) * ch.T * ch.J_light)
    for i in range(0, upto):
        a, b = xs[i], xs[i + 1]
        w = unit(sub(b.pos, a.pos))
        wprev = ch.w0 if i == 0 else unit(sub(xs[i - 1].pos, a.pos))
        dvcm, dvc = scatter(dvcm, dvc, abs(dot(a.n, w)), p_dir(a, w, wprev), p_dir(a, wprev, w), etas[i])
        dvcm, dvc = arrive(dvcm, dvc, a, b)
    return dvcm, dvc


def camera_carries(xs, lens, etas, upto):
    n = len(xs)
    dvcm, dvc = arrive(N_SPLAT / P_CAM_AREA, 0.0, lens, xs[n - 1])
    for i in range(n - 1, upto, -1):
        a, b = xs[i], xs[i - 1]
        w = unit(sub(b.pos, a.pos))
        prev = lens if i == n - 1 else xs[i + 1]
        wp = unit(sub(prev.pos, a.pos))
        dvcm, dvc = scatter(dvcm, dvc, abs(dot(a.n, w)), p_dir(a, w, wp), p_dir(a, wp, w), etas[i])
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
    w_lp = ch.w0 if s == 2 else unit(sub(xs[s - 3].pos, lv.pos))
    w_cp = unit(sub(cam_pred(xs, lens, s - 1).pos, cv.pos))
    light_rev_w = p_dir(lv, w_lp, dir_lc)
    cam_rev_w = p_dir(cv, w_cp, neg(dir_lc))
    d = norm(sub(cv.pos, lv.pos))
    cam_dir_a = p_dir(cv, neg(dir_lc), w_cp) * abs(dot(lv.n, dir_lc)) / (d * d)
    light_dir_a = p_dir(lv, dir_lc, w_lp) * abs(dot(cv.n, dir_lc)) / (d * d)
    w_light = cam_dir_a * (etas[s - 2] + lvcm + lvc * light_rev_w)
    w_camera = light_dir_a * (etas[s - 1] + ccvcm + ccvc * cam_rev_w)
    return 1.0 / (w_light + 1.0 + w_camera)


def w_splat(y0, xs, lens, etas, ch, mnee):
    n = len(xs)
    lv = xs[n - 1]
    lvcm, lvc = light_carries(y0, xs, etas, n - 1, ch, mnee)
    rev = p_dir(lv, ch.w0 if n == 1 else unit(sub(xs[n - 2].pos, lv.pos)), unit(sub(lens.pos, lv.pos)))
    w_light = (P_CAM_AREA * cam_edge_pdf(lens, lv) / N_SPLAT) * (etas[n - 1] + lvcm + lvc * rev)
    return 1.0 / (w_light + 1.0)


def w_merge(y0, xs, lens, etas, k, ch, mnee):
    lvcm, lvc = light_carries(y0, xs, etas, k, ch, mnee)
    ccvcm, ccvc = camera_carries(xs, lens, etas, k)
    x = xs[k]
    w_to_lpred = ch.w0 if k == 0 else unit(sub(xs[k - 1].pos, x.pos))
    w_to_cpred = unit(sub(cam_pred(xs, lens, k).pos, x.pos))
    w_light = (lvcm + lvc * p_dir(x, w_to_lpred, w_to_cpred)) / etas[k]
    w_camera = (ccvcm + ccvc * p_dir(x, w_to_cpred, w_to_lpred)) / etas[k]
    return 1.0 / (w_light + 1.0 + w_camera)


def w_emission_hit(y0, xs, lens, etas, ch, mnee):
    """Camera: scatter at x_0 toward v, delta at v, arrive at y0; SmallVCM's 1 / (1 + directPdfA dVCM + emissionPdfW dVC)
    with dVCM = 0 after the delta, plus MNEE's ratio to this strategy."""
    x0 = xs[0]
    dvcm, dvc = camera_carries(xs, lens, etas, 0)
    prev = cam_pred(xs, lens, 0)
    wp = unit(sub(prev.pos, x0.pos))
    dvcm, dvc = scatter(dvcm, dvc, ch.cos0, p_dir(x0, ch.w0, wp), p_dir(x0, wp, ch.w0), etas[0])
    dv = norm(sub(ch.v, x0.pos))
    dvcm, dvc = dvcm * dv * dv / abs(ch.w0[2]), dvc / abs(ch.w0[2])     # arrive at v
    dvcm, dvc = 0.0, dvc * abs(ch.wy[2]) * (ch.n2 if mnee else 1.0)       # delta at v, water -> air
    dy = norm(sub(y0.pos, ch.v))
    dvcm, dvc = dvcm * dy * dy / ch.cosy, dvc / ch.cosy                 # arrive at y0
    inv = 1.0 + P_A * dvcm + P_A * p_emit(y0, ch.wy) * dvc
    if mnee:
        inv += P_A / (p_dir(x0, ch.w0, wp) * ch.T * ch.J_cam)
    return 1.0 / inv


def w_mnee(y0, xs, lens, etas, ch):
    """SmallVCM's DirectIllumination with the chain: the generalized geometry replaces cos/d^2 on both sides."""
    x0 = xs[0]
    dvcm, dvc = camera_carries(xs, lens, etas, 0)
    wp = unit(sub(cam_pred(xs, lens, 0).pos, x0.pos))
    bsdf_rev = p_dir(x0, wp, ch.w0)
    w_light = p_dir(x0, ch.w0, wp) * ch.T * ch.J_cam / P_A
    w_camera = p_emit(y0, ch.wy) * ch.T * ch.J_light * (etas[0] + dvcm + dvc * bsdf_rev)
    return 1.0 / (w_light + 1.0 + w_camera)


# ── report ─────────────────────────────────────────────────────────────────

def run(name, y0, xs, lens, etas):
    n = len(xs)
    ch = Chain(y0, xs[0])
    etendue = (ch.n2 * ch.cos0 * ch.J_cam) / (ch.cosy * ch.J_light)
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


def scene_reflect():
    y0 = V((0.5, 0.2, 1.5), unit((-0.1, 0.05, -1.0)))     # lamp over the water, facing down
    lens = V((-0.6, -2.5, 1.0), None)                    # camera over the water
    x_a = V((-1.2, 0.1, 0.6), unit((1.0, 0.1, 0.3)), glossy=True)   # glossy wall seeing the lamp in the water
    x_b = V((0.2, 1.4, 0.8), unit((0.0, -1.0, 0.1)))     # diffuse wall
    return y0, lens, x_a, x_b


if __name__ == "__main__":
    err = 0.0
    for chain, sc in (("refract", scene), ("reflect", scene_reflect)):
        CHAIN = chain
        for glossy in (False, True):
            y0, lens, x_a, x_b = sc()
            x_a.glossy = glossy
            for n_light in (2.0e4, 2.0e6):
                N_LIGHT = n_light
                ETA_VCM = N_LIGHT * KERNEL
                tag = f"{chain}, x_0 {'glossy' if glossy else 'diffuse'}, {n_light:.0e} light paths"
                err = max(err, run(f"{tag}: 1 vertex", y0, [x_a], lens, [ETA_VCM]))
                err = max(err, run(f"{tag}: 2 vertices", y0, [x_a, x_b], lens, [ETA_VCM, ETA_VCM]))
                err = max(err, run(f"{tag}: 2 vertices, eta(x) varies", y0, [x_a, x_b], lens, [0.1 * ETA_VCM, ETA_VCM]))
    print(f"\nmax |with glossy SMS - truth| = {err:.2e}")
