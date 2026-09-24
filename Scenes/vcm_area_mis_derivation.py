#!/usr/bin/env python3
"""Numerical check of VCM's MIS weights for a FINITE AREA light.

Sibling of vcm_env_mis_derivation.py (environment light) and
vcm_volume_mis_derivation.py. Same method: enumerate every strategy's density
over the path space (y0, x_0..x_{n-1}) directly, form the balance-heuristic
weights, and compare them with the weights bdpt.mojo's local dVCM/dVC formulas
produce. A correct estimator's weights sum to 1 for every path.

Why this file exists (2026-09-24): two symptoms pointed at area-light MIS.
  * Cornell-box VCM read 0.9675 of the path tracer with 65k light paths and
    0.888 with 2M. A consistent VCM is invariant to the light-path count.
  * Making merging's MIS density position-dependent (eta(x) = keep(x) * eta,
    for merge-grid thinning) brightened the candle lanterns 1.3-2.1x. Correct
    weights cannot move the mean, whatever eta(x) is.

bdpt.mojo runs, for an area light:
  s=0   camera BSDF-samples the light      weighted by a 2-strategy POWER
                                           heuristic (bdpt.mojo, "area-light
                                           emission hit")
  s=1   camera vertex x_0 connects to the paired light path's ORIGIN vertex y0
        (there is no separate area-light NEE -- see the camera bounce's NEE
        comment), weighted by _connect's generic formula with y0's stored
        carries
  s>=2  connections between light vertex x_{s-2} and camera vertex x_{s-1}
  t=1   the light vertex seen by the lens splats (_bdpt_splat...)
  merge at every x_k                       (_bdpt_merge_from_cache)

Every `renderer_*` function below mirrors the formula AS CODED; `smallvcm_*`
mirrors SmallVCM's vertexcm.hxx for the same strategy, the candidate fix.
"""
import math

# ── scene constants ────────────────────────────────────────────────────────
R_MERGE = 0.03
N_LIGHT = 20000.0
KERNEL = math.pi * R_MERGE * R_MERGE
ETA_VCM = N_LIGHT * KERNEL
P_CAM_AREA = 1.0
LIGHT_AREA = 0.25
P_A = 1.0 / LIGHT_AREA          # one light: pick probability 1
N_SPLAT = 1.0                   # see vcm_env_mis_derivation.py: any value, both sides


def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def norm(a):   return math.sqrt(dot(a, a))
def unit(a):
    n = norm(a)
    return (a[0] / n, a[1] / n, a[2] / n)
def neg(a):    return (-a[0], -a[1], -a[2])


class V:
    def __init__(self, pos, normal):
        self.pos = pos
        self.n = normal


def p_dir(v, w):
    """Lambertian directional pdf (two-sided |cos|, as face_toward makes it)."""
    return abs(dot(v.n, w)) / math.pi


def p_emit(y0, w):
    """Cosine-weighted emission from the light's front side."""
    return max(0.0, dot(y0.n, w)) / math.pi


def g_to_area(a, b):
    """Solid angle at a toward b -> area at b."""
    d = norm(sub(b.pos, a.pos))
    w = unit(sub(b.pos, a.pos))
    return abs(dot(b.n, w)) / (d * d)


def edge_pdf(a, b):
    return p_dir(a, unit(sub(b.pos, a.pos))) * g_to_area(a, b)


def cam_edge_pdf(lens, b):
    return g_to_area(lens, b)


# ── ground truth ───────────────────────────────────────────────────────────

def strategy_densities(y0, xs, lens, etas):
    """xs[0] is the first vertex after the light, xs[-1] the one the lens sees.
    etas[k] is merging's MIS density factor at xs[k]."""
    n = len(xs)
    C = [0.0] * n
    C[n - 1] = P_CAM_AREA * cam_edge_pdf(lens, xs[n - 1])
    for k in range(n - 2, -1, -1):
        C[k] = C[k + 1] * edge_pdf(xs[k + 1], xs[k])
    C_y0 = C[0] * edge_pdf(xs[0], y0)
    L = [0.0] * n
    L[0] = P_A * p_emit(y0, unit(sub(xs[0].pos, y0.pos))) * g_to_area(y0, xs[0])
    for k in range(1, n):
        L[k] = L[k - 1] * edge_pdf(xs[k - 1], xs[k])
    d = {}
    d[("s", 0)] = C_y0
    d[("s", 1)] = P_A * C[0]
    for s in range(2, n + 1):
        d[("s", s)] = L[s - 2] * C[s - 1]
    d[("t1", n - 1)] = N_SPLAT * L[n - 1] * P_CAM_AREA
    for k in range(n):
        d[("merge", k)] = etas[k] * L[k] * C[k]
    return d


# ── the renderer's carries ─────────────────────────────────────────────────

def scatter(dvcm, dvc, cos_out, pdf_fwd, pdf_rev, eta):
    """vcm_mis.vcm_scatter_carries (dVM is formed lazily as dVC / eta)."""
    return 1.0 / pdf_fwd, (cos_out / pdf_fwd) * (dvc * pdf_rev + dvcm + eta)


def arrive(dvcm, dvc, a, b):
    """bdpt.mojo's arrival: dVCM *= d^2, then both / cos at b."""
    d = norm(sub(b.pos, a.pos))
    w = unit(sub(b.pos, a.pos))
    cf = abs(dot(b.n, w))
    return dvcm * d * d / cf, dvc / cf


def light_carries(y0, xs, etas, upto):
    """Arrival carries at xs[upto] of the light subpath, plus y0's own."""
    w0 = unit(sub(xs[0].pos, y0.pos))
    cos_emit = max(dot(y0.n, w0), 1e-4)
    direct_pdf_a = P_A
    emission_pdf_w = direct_pdf_a * cos_emit / math.pi
    y0_carries = (direct_pdf_a / emission_pdf_w, cos_emit / emission_pdf_w)
    dvcm, dvc = arrive(*y0_carries, y0, xs[0])
    for i in range(0, upto):
        a, b = xs[i], xs[i + 1]
        w = unit(sub(b.pos, a.pos))
        prev = y0 if i == 0 else xs[i - 1]
        wprev = unit(sub(prev.pos, a.pos))
        dvcm, dvc = scatter(dvcm, dvc, abs(dot(a.n, w)), p_dir(a, w), p_dir(a, wprev), etas[i])
        dvcm, dvc = arrive(dvcm, dvc, a, b)
    return (dvcm, dvc), y0_carries


def camera_carries(xs, lens, etas, upto):
    """Arrival carries at xs[upto] of the camera subpath (dVC starts at 0)."""
    n = len(xs)
    dvcm, dvc = N_SPLAT / P_CAM_AREA, 0.0
    dvcm, dvc = arrive(dvcm, dvc, lens, xs[n - 1])
    for i in range(n - 1, upto, -1):
        a, b = xs[i], xs[i - 1]
        w = unit(sub(b.pos, a.pos))
        prev = lens if i == n - 1 else xs[i + 1]
        wprev = unit(sub(prev.pos, a.pos))
        dvcm, dvc = scatter(dvcm, dvc, abs(dot(a.n, w)), p_dir(a, w), p_dir(a, wprev), etas[i])
        dvcm, dvc = arrive(dvcm, dvc, a, b)
    return dvcm, dvc


def cam_pred(xs, lens, k):
    return lens if k == len(xs) - 1 else xs[k + 1]


def light_pred(y0, xs, k):
    return y0 if k == 0 else xs[k - 1]


# ── renderer weights, as coded ─────────────────────────────────────────────

def renderer_connect(y0, xs, lens, etas, s, eta_at_light_vertex):
    """_connect: lv = light vertex, cv = camera vertex."""
    n = len(xs)
    cv_k = s - 1 if s >= 1 else None
    cv = xs[s - 1] if s >= 2 else xs[0]
    ccvcm, ccvc = camera_carries(xs, lens, etas, s - 1 if s >= 2 else 0)
    if s == 1:
        lv = y0
        (lvcm, lvc) = light_carries(y0, xs, etas, 0)[1]      # y0's own stored carries
        dir_lc = unit(sub(cv.pos, lv.pos))
        cos_lv = abs(dot(lv.n, dir_lc))
        light_dir_w = cos_lv / math.pi                       # "REASONED" in bdpt.mojo
        light_rev_w = cos_lv / math.pi
        eta_l = eta_at_light_vertex
    else:
        lv = xs[s - 2]
        (lvcm, lvc), _ = light_carries(y0, xs, etas, s - 2)
        dir_lc = unit(sub(cv.pos, lv.pos))
        light_dir_w = p_dir(lv, dir_lc)
        light_rev_w = p_dir(lv, unit(sub(light_pred(y0, xs, s - 2).pos, lv.pos)))
        eta_l = etas[s - 2]
    cam_dir_w = p_dir(cv, neg(dir_lc))
    cam_rev_w = p_dir(cv, unit(sub(cam_pred(xs, lens, s - 1 if s >= 2 else 0).pos, cv.pos)))
    d = norm(sub(cv.pos, lv.pos))
    cam_dir_a = cam_dir_w * abs(dot(lv.n, dir_lc)) / (d * d)
    light_dir_a = light_dir_w * abs(dot(cv.n, dir_lc)) / (d * d)
    eta_c = etas[s - 1] if s >= 2 else etas[0]
    w_light = cam_dir_a * (eta_l + lvcm + lvc * light_rev_w)
    w_camera = light_dir_a * (eta_c + ccvcm + ccvc * cam_rev_w)
    return 1.0 / (w_light + 1.0 + w_camera)


def renderer_emission_hit(y0, xs, lens, etas):
    """Area-light emission hit: power_heuristic(last_bsdf_pdf, pdf_light)."""
    w = unit(sub(y0.pos, xs[0].pos))
    d = norm(sub(y0.pos, xs[0].pos))
    cos_l = abs(dot(y0.n, w))
    last_bsdf_pdf = p_dir(xs[0], w)
    pdf_light = d * d / (cos_l * LIGHT_AREA)
    a, b = last_bsdf_pdf ** 2, pdf_light ** 2
    return a / (a + b)


def renderer_splat(y0, xs, lens, etas):
    n = len(xs)
    lv = xs[n - 1]
    (lvcm, lvc), _ = light_carries(y0, xs, etas, n - 1)
    rev = p_dir(lv, unit(sub(light_pred(y0, xs, n - 1).pos, lv.pos)))
    cam_pdf_a = P_CAM_AREA * cam_edge_pdf(lens, lv)
    w_light = (cam_pdf_a / N_SPLAT) * (etas[n - 1] + lvcm + lvc * rev)
    return 1.0 / (w_light + 1.0)


def renderer_merge(y0, xs, lens, etas, k):
    (lvcm, lvc), _ = light_carries(y0, xs, etas, k)
    ccvcm, ccvc = camera_carries(xs, lens, etas, k)
    x = xs[k]
    w_to_lpred = unit(sub(light_pred(y0, xs, k).pos, x.pos))
    w_to_cpred = unit(sub(cam_pred(xs, lens, k).pos, x.pos))
    cam_dir_w = p_dir(x, w_to_lpred)      # camera BSDF sampling the photon's origin
    cam_rev_w = p_dir(x, w_to_cpred)
    w_light = (lvcm + lvc * cam_dir_w) / etas[k]
    w_camera = (ccvcm + ccvc * cam_rev_w) / etas[k]
    return 1.0 / (w_light + 1.0 + w_camera)


# ── SmallVCM's formulas for s=0 and s=1 (candidate fix) ────────────────────

def smallvcm_emission_hit(y0, xs, lens, etas):
    """GetLightRadiance: camera carries AFTER scattering at x_0 toward y0 and
    ARRIVING at y0 (d^2, cos at the light), then
        wCamera = directPdfA * dVCM + emissionPdfW * dVC."""
    n = len(xs)
    dvcm, dvc = camera_carries(xs, lens, etas, 0)
    x0 = xs[0]
    w = unit(sub(y0.pos, x0.pos))
    prev = cam_pred(xs, lens, 0)
    dvcm, dvc = scatter(dvcm, dvc, abs(dot(x0.n, w)), p_dir(x0, w),
                        p_dir(x0, unit(sub(prev.pos, x0.pos))), etas[0])
    dvcm, dvc = arrive(dvcm, dvc, x0, y0)
    direct_pdf_a = P_A
    emission_pdf_w = P_A * p_emit(y0, neg(w))
    return 1.0 / (1.0 + direct_pdf_a * dvcm + emission_pdf_w * dvc)


def smallvcm_direct(y0, xs, lens, etas):
    """DirectIllumination at x_0 toward light point y0:
        wLight  = bsdfDirPdfW / directPdfW
        wCamera = emissionPdfW * cosToLight / (directPdfW * cosAtLight)
                  * (eta(x_0) + dVCM + dVC * bsdfRevPdfW)"""
    x0 = xs[0]
    dvcm, dvc = camera_carries(xs, lens, etas, 0)
    w = unit(sub(y0.pos, x0.pos))
    d = norm(sub(y0.pos, x0.pos))
    cos_at_light = abs(dot(y0.n, w))
    cos_to_light = abs(dot(x0.n, w))
    direct_pdf_w = P_A * d * d / cos_at_light
    emission_pdf_w = P_A * p_emit(y0, neg(w))
    bsdf_dir = p_dir(x0, w)
    bsdf_rev = p_dir(x0, unit(sub(cam_pred(xs, lens, 0).pos, x0.pos)))
    w_light = bsdf_dir / direct_pdf_w
    w_camera = emission_pdf_w * cos_to_light / (direct_pdf_w * cos_at_light) * (etas[0] + dvcm + dvc * bsdf_rev)
    return 1.0 / (w_light + 1.0 + w_camera)


# ── report ─────────────────────────────────────────────────────────────────

def run(name, y0, xs, lens, etas, eta_y0):
    n = len(xs)
    d = strategy_densities(y0, xs, lens, etas)
    tot = sum(d.values())
    bw = {k: v / tot for k, v in d.items()}
    rw = {("s", 0): renderer_emission_hit(y0, xs, lens, etas),
          ("s", 1): renderer_connect(y0, xs, lens, etas, 1, eta_y0),
          ("t1", n - 1): renderer_splat(y0, xs, lens, etas)}
    for s in range(2, n + 1):
        rw[("s", s)] = renderer_connect(y0, xs, lens, etas, s, eta_y0)
    for k in range(n):
        rw[("merge", k)] = renderer_merge(y0, xs, lens, etas, k)
    fw = dict(rw)
    fw[("s", 0)] = smallvcm_emission_hit(y0, xs, lens, etas)
    fw[("s", 1)] = smallvcm_direct(y0, xs, lens, etas)
    print(f"\n=== {name} ===")
    print(f"  {'strategy':12s} {'truth':>11s} {'as coded':>11s} {'SmallVCM s0/s1':>15s}")
    for k in sorted(bw, key=lambda k: (k[0], k[1])):
        print(f"  {str(k):12s} {bw[k]:11.6f} {rw[k]:11.6f} {fw[k]:15.6f}")
    print(f"  {'SUM':12s} {sum(bw.values()):11.6f} {sum(rw.values()):11.6f} {sum(fw.values()):15.6f}")
    return sum(rw.values()), sum(fw.values())


def scene():
    y0 = V((0.0, 1.0, 0.0), unit((0.05, -1.0, 0.1)))       # ceiling light, facing down
    lens = V((0.1, 0.4, -3.0), None)
    x_a = V((0.3, -1.0, 0.2), unit((0.0, 1.0, 0.05)))      # floor
    x_b = V((-1.0, 0.1, 0.4), unit((1.0, 0.1, -0.05)))     # left wall
    return y0, lens, x_a, x_b


if __name__ == "__main__":
    y0, lens, x_a, x_b = scene()
    for label, n_light in [("N = 2e4", 2.0e4), ("N = 2e6", 2.0e6)]:
        N_LIGHT = n_light
        ETA_VCM = N_LIGHT * KERNEL
        print(f"\n################ {label} light paths, eta = {ETA_VCM:.4g} ################")
        run("1 vertex, constant eta", y0, [x_a], lens, [ETA_VCM], ETA_VCM)
        run("2 vertices, constant eta", y0, [x_a, x_b], lens, [ETA_VCM, ETA_VCM], ETA_VCM)
        run("2 vertices, eta(x) = keep * eta  (keep 0.1 at x_a, 1 at x_b)",
            y0, [x_a, x_b], lens, [0.1 * ETA_VCM, ETA_VCM], ETA_VCM)
