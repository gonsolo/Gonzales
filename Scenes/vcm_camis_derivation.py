#!/usr/bin/env python3
"""Numerical check of the CAMIS-prefix correction for VCM vertex merging.

Sibling of vcm_area_mis_derivation.py, whose scene helpers (edge_pdf,
g_to_area, the dVCM/dVC scatter/arrive rules, the strategy enumeration) it
imports rather than copies. Same method as every harness in this family: say
what the quantity IS, straight from its definition, then reproduce it a
second, independent way from the local quantities the renderer actually
carries, and demand that the two agree.

CAMIS (Grittmann, Georgiev, Slusallek, "Correlation-Aware Multiple Importance
Sampling", EG 2021), Sec. 3. For a path x = y z split at a merge vertex into
the camera PREFIX y (shared by every photon that merge query reuses -- the
correlation) and the light SUFFIX z:

  Eq. 11  w_t(x) = c_t(x) n_t p_t(x) / sum_k c_k(x) n_k p_k(x)
  Eq. 12  1/n_t <= c_t(x) <= 1
  Eq. 13  c_t(x) = max(P(y)/P(x), 1/n_t)
  Eq. 14  P(y) = prod_i P(y_{i-1} -> y_i),   P(z) = P(z_0) prod_i P(z_{i-1} -> z_i)
  Eq. 15  P(x) = P(z) + P(y) - P(z) P(y)
  Eq. 16  P(x_i -> x_j) = int_{D_r} p(x_i -> x) dx  ~=  min(pi r^2 p(x_i -> x_j), 1)
  Eq. 17  r = d tan(pi/180),  d = |y_1 - y_0|

Two readings of those equations the rest of this file depends on:
  * Eq. 16 integrates p over a DISK OF AREA around x_j, so p is the AREA-
    measure density p_w |cos_j| / d^2. That is what makes pi r^2 p unitless.
  * Eq. 17 is ONE radius per path, set by the primary camera hit distance,
    used for every edge on both sides. Not a per-edge d_i.
Since P(z)(1 - P(y)) >= 0, P(x) >= P(y), so Eq. 13's ratio never exceeds 1;
the upper clamp of Eq. 12 only guards rounding.

In VCM only merging is a correlated (splitting) technique: n_t = the number of
light paths for merges, and connections have n = 1, whence c = 1 by Eq. 12.
So CAMIS amounts to replacing merging's MIS density eta(x) = k(x) N pi r(x)^2 by
c(x) eta(x). In bdpt_*.mojo that means multiplying c into the RESULT of
_vcm_eta_scale (merging's density shrinks, its weight falls). It must NOT be
multiplied into inv_eta_x directly: that divides merging's density by c and
RAISES its weight -- the opposite of CAMIS and of the validated "scale merging
by a constant 1/4" ablation. Checked numerically below.

THE 3-DAY CUT (plan deep-hugging-locket.md) -- an APPROXIMATION, not CAMIS:
  * P(y) is exact: the camera prefix product, accumulated in log space along
    the camera walk (cheap, thread-local).
  * P(z) is replaced by ITS TERMINAL EDGE ALONE, P(z_{s-1} -> x_merge). The
    real P(z) also multiplies P(z_0) and every earlier light edge; carrying
    that needs a persisted per-light-vertex accumulator (the ~14-day exact
    design in project_camis_design). Every factor is <= 1, so the proxy
    OVERESTIMATES P(z), and c falls as P(z) grows: the proxy discounts
    merging at least as hard as exact CAMIS would (asserted below).
  * c is applied ONLY in the merge's own weight (_bdpt_merge_mis_weight), not
    in the eta every other strategy's carries hold. Eq. 11's weights then no
    longer sum to 1 -- merging's lost share is not handed to the others -- so
    the 3-day cut is BIASED DARK wherever merging competes with connections.
    The exact design's partition of unity needs c at every vertex in every
    strategy's denominator; the size of this deficit is printed below, so the
    Day-3 ablation can be read with it in mind.
  * MEASURED BELOW, and worse than the plan assumed: the proxy has one O(r^2)
    factor against P(y)'s one-per-camera-edge, so behind 2+ camera bounces it
    drives c to ~1/n_t almost regardless of geometry (0.00033 vs exact 0.83
    on photon a; within 2x of exact on 0% of 2000 random paths), and the
    non-partition loses ~45% of the toy path's energy (~21% of it from
    locality alone, with exact c). An exact-P(z) variant is cheap on the
    light side IF the per-edge clamp is relaxed to one clamp on the product:
    store sum log p_i (incl. log P_A) and the edge count m in the light
    vertex (e.g. the dead dVM slot) and form m log(pi r^2) + sum at merge
    time -- the "light product w/ one clamp" column, equal to exact here.

What the renderer can use, with no new pdf evaluations: bdpt_*.mojo stores
v.dVCM AFTER `dvcm *= t_hit^2` and the arrival cos divide, i.e.
v.dVCM = d^2 / (pdf_fwd_w |cos|) = 1 / p_A(edge that arrived at v). So
    P(edge into v) = pi r^2 / max(v.dVCM, pi r^2)
for camera AND light vertices alike -- a delta bounce (dVCM reset to 0) gives
exactly 1, as a Dirac density must. The camera's FIRST arrival is the
exception: its dVCM holds n_light_paths / cameraPdfW, not an edge density.
The primary edge's P is 1 by construction (another sample of the same pixel
lands within any 1-degree footprint -- a pixel subtends far less), so the
accumulator starts at log P(y) = 0 at the primary hit.

CORRECTIONS TO THE PLAN'S SKETCH (every one exercised below):
  1. The angle is 1 degree, tan(pi/180) ~= 0.0175 (Eq. 17). The sketch's
     tan(pi/720) is 0.25 degree.
  2. The sketch's edge term pi (d_i tan theta)^2 pdf_fwd_w multiplies an AREA
     (m^2) by a SOLID-ANGLE density: it is not unitless and changes when the
     scene is rescaled. The paper's is pi r^2 p_A with one r per path.
  3. The plan's P(z) proxy pi r2_query camera_bsdf_dir_pdf_w has the same
     unit error, uses the merge radius instead of the CAMIS radius, and reads
     the CAMERA BSDF's density toward the photon (the reverse direction), not
     the light's forward density into x. The terminal light edge is
     min(pi r^2 / lv.dVCM, 1) and is already in scope.
  4. "c -> 1 as the disk radius -> 0" is backwards. The paper (Sec. 3.3):
     a LARGE radius makes every P -> 1, P(x) -> 1, c -> 1 (the balance
     heuristic). As r -> 0 each unclamped edge is O(r^2), so c tracks
     (r^2)^(#prefix edges - #suffix edges): with the 3-day proxy's single
     suffix edge and a 2-edge prefix, c -> 1/n_t; with the exact 3-factor
     suffix, c -> 1.
  5. "c falls monotonically as n_t grows" holds as NON-INCREASING: c falls
     only while the 1/n_t clamp binds, then sits at P(y)/P(x).
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcm_area_mis_derivation as area                       # noqa: E402
from vcm_area_mis_derivation import (V, sub, dot, norm, unit, neg,  # noqa: E402
                                     p_dir, p_emit, g_to_area, edge_pdf)

# ── constants ──────────────────────────────────────────────────────────────
TAN_1DEG = math.tan(math.pi / 180.0)        # Eq. 17
TAN_SKETCH = math.tan(math.pi / 720.0)      # the plan sketch's angle, for comparison only
N_LIGHT = 2.0e4                             # n_t for merging (the renderer's n_light_paths)
ETA = N_LIGHT * area.KERNEL                 # merging's MIS density N pi R_MERGE^2
EPS = sys.float_info.epsilon

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))


def light_edge_pdf(z0, x):
    """Area density of the light's emission sampling x from z0 (cosine lobe)."""
    return p_emit(z0, unit(sub(x.pos, z0.pos))) * g_to_area(z0, x)


def light_edges(light_path):
    """Area densities of every edge of a light subpath [z0, z1, ..., x]."""
    z0 = light_path[0]
    out = [light_edge_pdf(z0, light_path[1])]
    for a, b in zip(light_path[1:], light_path[2:]):
        out.append(edge_pdf(a, b))
    return out


# ── ground truth: Eq. 13-17 as written ─────────────────────────────────────

def edge_P(r2, p_area):
    """Eq. 16."""
    return min(math.pi * r2 * p_area, 1.0)


def c_from_P(py, pz, n_t):
    """Eq. 13 with Eq. 15 substituted, clamped to Eq. 12's bounds. P(z) = 0
    (nothing on the suffix side resembles this path) leaves P(x) = P(y) and
    the balance heuristic, c = 1 -- also when P(y) = 0, a limit the formula
    leaves open and the renderer form below resolves the same way."""
    if pz <= 0.0:
        return 1.0
    px = py + pz - py * pz
    return min(max(py / px, 1.0 / n_t), 1.0)


def direct_camis(lens, cam_vs, light_path, n_t, light_area=area.LIGHT_AREA,
                 r_scale=1.0, cam_edge_p=None):
    """c from the definitions. cam_vs = [y1, ..., x_merge] in camera order;
    light_path = [z0, ..., x_photon] in light order. cam_edge_p optionally
    overrides the camera edges' area densities (glossy/delta lobes).
    Returns (c_proxy, c_exact, P(y), P(z) proxy, P(z) exact)."""
    r = norm(sub(cam_vs[0].pos, lens.pos)) * TAN_1DEG * r_scale     # Eq. 17
    r2 = r * r
    if cam_edge_p is None:
        cam_edge_p = [edge_pdf(a, b) for a, b in zip(cam_vs, cam_vs[1:])]
    py = 1.0                                     # primary edge: P = 1 (see docstring)
    for p in cam_edge_p:
        py *= edge_P(r2, p)
    le = light_edges(light_path)
    pz_proxy = edge_P(r2, le[-1])                # 3-DAY CUT: terminal light edge only
    pz_exact = edge_P(r2, 1.0 / light_area)      # P(z_0): light position sampling
    for p in le:
        pz_exact *= edge_P(r2, p)
    return (c_from_P(py, pz_proxy, n_t), c_from_P(py, pz_exact, n_t),
            py, pz_proxy, pz_exact)


# ── the renderer's form: log-space accumulator on the carries ──────────────

def log_edge_from_carry(r2, dvcm_arrival):
    """log min(pi r^2 p_A, 1) with p_A = 1/dVCM_arrival, as bdpt_*.mojo would
    form it after vcm_arrival_carries. dVCM = 0 (delta) gives log 1 = 0."""
    k = math.pi * r2
    return math.log(k / max(dvcm_arrival, k))


def log_edge_from_pdf(r2, pdf_fwd_w, cos_arrival, t_hit):
    """The same term formed at the scatter site from pdf_fwd_w, as the plan
    places it -- corrected to the AREA density pdf_fwd_w |cos| / t_hit^2."""
    return math.log(min(math.pi * r2 * pdf_fwd_w * cos_arrival / (t_hit * t_hit), 1.0))


def renderer_c(log_py, pz, n_t):
    """c = P(y)/P(x) rewritten as 1 / (1 + P(z) (1/P(y) - 1)), with
    1/P(y) - 1 = expm1(-log P(y)). The Float32 form to use on Day 2: it
    never forms the (underflowing) product P(y), never divides by P(x), and
    a prefix too diffuse to represent (expm1 -> inf) lands on 1/n_t."""
    if pz <= 0.0:
        return 1.0
    e = -log_py
    ratio = 0.0 if e > 709.0 else 1.0 / (1.0 + pz * math.expm1(e))
    return min(max(ratio, 1.0 / n_t), 1.0)


def renderer_camera_walk(lens, cam_vs, etas, r_scale=1.0, pdf_fwd=None, delta=None):
    """The camera subpath exactly as vcm_area_mis_derivation.camera_carries
    walks it (dVC starts at 0, scatter then arrive), plus the CAMIS
    accumulator. Returns (dVCM, dVC at x_merge, log P(y), r^2, max |carry
    form - pdf form| over the edges)."""
    t_hit = norm(sub(cam_vs[0].pos, lens.pos))          # the primary t_hit
    r = t_hit * TAN_1DEG * r_scale
    r2 = r * r
    dvcm, dvc = area.N_SPLAT / area.P_CAM_AREA, 0.0
    dvcm, dvc = area.arrive(dvcm, dvc, lens, cam_vs[0])
    log_py = 0.0                                         # NOT read from this dVCM
    worst = 0.0
    for i in range(len(cam_vs) - 1):
        a, b = cam_vs[i], cam_vs[i + 1]
        prev = lens if i == 0 else cam_vs[i - 1]
        w = unit(sub(b.pos, a.pos))
        pf = p_dir(a, w) if pdf_fwd is None else pdf_fwd[i]
        if delta is not None and delta[i]:
            dvcm, dvc = 0.0, dvc * abs(dot(a.n, w))      # SmallVCM's specular carry rule
        else:
            dvcm, dvc = area.scatter(dvcm, dvc, abs(dot(a.n, w)), pf,
                                     p_dir(a, unit(sub(prev.pos, a.pos))), etas[i])
        dvcm, dvc = area.arrive(dvcm, dvc, a, b)
        le_c = log_edge_from_carry(r2, dvcm)
        if not (delta is not None and delta[i]):
            le_p = log_edge_from_pdf(r2, pf, abs(dot(b.n, w)), norm(sub(b.pos, a.pos)))
            worst = max(worst, abs(le_c - le_p) / max(abs(le_p), 1.0))
        log_py += le_c
    return dvcm, dvc, log_py, r2, worst


def alt_proxies(lens, cam_vs, light_path, n_t, light_area=area.LIGHT_AREA):
    """Two cheaper-than-exact alternatives to the 3-day P(z) proxy, for the
    Day-2 decision only (not asserted):
      sym   terminal edge on BOTH sides: c = P(y_last)/P(x) with P(y) cut to
            its own last edge, so the r^2 powers balance (1 vs 1 factor).
      prod  exact light product, but ONE clamp on the whole product instead
            of one per edge: log P(z) = m log(pi r^2) + sum log p_i, which
            the light walk CAN store (sum log p_i incl. log P_A, and m) even
            though r is the camera's -- e.g. in the dead dVM slot."""
    r = norm(sub(cam_vs[0].pos, lens.pos)) * TAN_1DEG
    r2 = r * r
    py_last = edge_P(r2, edge_pdf(cam_vs[-2], cam_vs[-1]))
    le = light_edges(light_path)
    c_sym = c_from_P(py_last, edge_P(r2, le[-1]), n_t)
    py = 1.0
    for a, b in zip(cam_vs, cam_vs[1:]):
        py *= edge_P(r2, edge_pdf(a, b))
    log_pz = math.log(math.pi * r2 / light_area) + sum(math.log(math.pi * r2 * p) for p in le)
    c_prod = c_from_P(py, math.exp(min(log_pz, 0.0)), n_t)
    return c_sym, c_prod


def renderer_light_carries(light_path, etas):
    """Arrival carries of the photon at light_path[-1] (light_carries)."""
    z0, xs = light_path[0], light_path[1:]
    (dvcm, dvc), _ = area.light_carries(z0, xs, etas, len(xs) - 1)
    return dvcm, dvc


def renderer_camis(lens, cam_vs, light_path, n_t, r_scale=1.0, **kw):
    etas = [ETA] * 8
    _, _, log_py, r2, worst = renderer_camera_walk(lens, cam_vs, etas, r_scale, **kw)
    lvcm, _ = renderer_light_carries(light_path, etas)
    pz = math.exp(log_edge_from_carry(r2, lvcm))        # the 3-day proxy: pi r^2 / lv.dVCM
    return renderer_c(log_py, pz, n_t), log_py, pz, worst


def merge_weight(lv, cv, cam_dir_w, cam_rev_w, eta_div):
    """_bdpt_merge_mis_weight: 1 / (w_light + 1 + w_camera), each carry sum
    divided by merging's density at x (eta_div = mis_vm * eta_scale)."""
    (lvcm, lvc), (cvcm, cvc) = lv, cv
    return 1.0 / ((lvcm + lvc * cam_dir_w) / eta_div + 1.0 + (cvcm + cvc * cam_rev_w) / eta_div)


# ── the plan's sketch, as written, for the unit check only ────────────────

def sketch_c(lens, cam_vs, light_path, n_t, s=1.0):
    """pi (d_i tan(pi/720))^2 pdf_fwd_w per camera edge; P(z) =
    pi r2_query camera_bsdf_dir_pdf_w (r_query = s R_MERGE)."""
    py = 1.0
    for i in range(len(cam_vs) - 1):
        a, b = cam_vs[i], cam_vs[i + 1]
        d = norm(sub(b.pos, a.pos))
        py *= min(math.pi * (d * TAN_SKETCH) ** 2 * p_dir(a, unit(sub(b.pos, a.pos))), 1.0)
    x, zp = cam_vs[-1], light_path[-2]
    pz = min(math.pi * (s * area.R_MERGE) ** 2 * p_dir(x, unit(sub(zp.pos, x.pos))), 1.0)
    return c_from_P(py, pz, n_t)


# ── scene: 2 camera bounces, 2 photons in one gather disk ──────────────────

def scene(s=1.0):
    """Box-ish room, light on the ceiling, merge on the floor. Every length is
    multiplied by s (the scale-invariance check)."""
    P = lambda x, y, z: (s * x, s * y, s * z)
    lens = V(P(0.1, 0.4, -3.0), None)
    y1 = V(P(0.2, -0.3, 1.0), unit((0.05, 0.02, -1.0)))       # back wall
    y2 = V(P(-1.0, 0.2, 0.3), unit((1.0, 0.1, -0.05)))        # left wall
    xm = V(P(0.3, -1.0, 0.2), unit((0.0, 1.0, 0.05)))         # floor: merge vertex
    # photon a: light -> right wall -> floor, lands 1.4 cm from xm
    z0a = V(P(0.0, 1.0, 0.0), unit((0.05, -1.0, 0.1)))
    z1a = V(P(1.0, -0.1, 0.4), unit((-1.0, 0.05, 0.02)))
    xa = V(P(0.312, -1.0, 0.192), xm.n)
    # photon b: straight from another point of the light, lands 1.8 cm away
    z0b = V(P(0.15, 1.0, -0.1), unit((0.05, -1.0, 0.1)))
    xb = V(P(0.29, -1.0, 0.215), xm.n)
    return lens, [y1, y2, xm], [z0a, z1a, xa], [z0b, xb]


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# ── 1. the two photons: direct vs renderer, sign of the placement ──────────

def section_photons():
    lens, cam, pa, pb = scene()
    etas = [ETA] * 8
    cdvcm, cdvc, _, _, _ = renderer_camera_walk(lens, cam, etas)
    xm, y2 = cam[-1], cam[-2]
    print("\n=== 2 camera bounces (lens->y1->y2->x), 2 photons in x's gather disk ===")
    print(f"  CAMIS radius r = |y1-lens| tan(1 deg) = {norm(sub(cam[0].pos, lens.pos)) * TAN_1DEG:.6f}"
          f"   (merge radius {area.R_MERGE}),  n_t = {N_LIGHT:g}")
    print(f"  {'photon':8s} {'P(y)':>11s} {'P(z) proxy':>11s} {'P(z) exact':>11s} "
          f"{'c proxy':>11s} {'c exact':>11s} {'c renderer':>11s} {'rel err':>9s}")
    worst = 0.0
    for name, lp in (("a", pa), ("b", pb)):
        cp, ce, py, pzp, pze = direct_camis(lens, cam, lp, N_LIGHT)
        cr, log_py, pzr, wedge = renderer_camis(lens, cam, lp, N_LIGHT)
        e = max(rel(cr, cp), rel(math.exp(log_py), py), rel(pzr, pzp))
        worst = max(worst, e)
        print(f"  {name:8s} {py:11.4e} {pzp:11.4e} {pze:11.4e} {cp:11.6f} {ce:11.6f} {cr:11.6f} {e:9.1e}")
        c_sym, c_prod = alt_proxies(lens, cam, lp, N_LIGHT)
        print(f"  {'':8s} alternatives: terminal-edge-both-sides c {c_sym:.6f}, "
              f"light product w/ one clamp c {c_prod:.6f}   (exact {ce:.6f})")
        check(f"photon {name}: carry-form edge term == pdf-form edge term", wedge < 4 * EPS, f"{wedge:.1e}")
        check(f"photon {name}: proxy discounts at least as hard as exact (c_proxy <= c_exact)", cp <= ce)

        # the placement: c into eta_scale, never into inv_eta_x
        lv = renderer_light_carries(lp, etas)
        cam_dir_w = p_dir(xm, unit(sub(lp[-2].pos, lp[-1].pos)))   # toward the photon's origin
        cam_rev_w = p_dir(xm, unit(sub(y2.pos, xm.pos)))
        w0 = merge_weight(lv, (cdvcm, cdvc), cam_dir_w, cam_rev_w, ETA)
        w_ok = merge_weight(lv, (cdvcm, cdvc), cam_dir_w, cam_rev_w, ETA * cp)
        w_bad = merge_weight(lv, (cdvcm, cdvc), cam_dir_w, cam_rev_w, ETA / cp)
        w_q = merge_weight(lv, (cdvcm, cdvc), cam_dir_w, cam_rev_w, ETA * 0.25)
        print(f"  {'':8s} merge w: balance {w0:.6f}  c*eta_scale {w_ok:.6f}  "
              f"(c into inv_eta_x: {w_bad:.6f})  constant 1/4: {w_q:.6f}")
        check(f"photon {name}: c into eta_scale LOWERS merging's weight", w_ok < w0 if cp < 1 else w_ok == w0)
        check(f"photon {name}: c into inv_eta_x would RAISE it (the sign trap)", w_bad > w0 if cp < 1 else True)
        check(f"photon {name}: constant-1/4 ablation moves w the same way", w_q < w0)
    check("photons a,b: direct == renderer (c, P(y), P(z))", worst < 1e-14, f"max rel err {worst:.1e}")
    return worst


# ── 2. sanity bounds ───────────────────────────────────────────────────────

def section_bounds():
    lens, cam, pa, pb = scene()
    print("\n=== bounds and limits ===")

    # c in [1/n_t, 1], and non-increasing in n_t
    ok_b, ok_m = True, True
    print(f"  n_t sweep (photon a):  ", end="")
    prev = 2.0
    ratio = direct_camis(lens, cam, pa, 1e300)[0]
    for n_t in (1.0, 2.0, 10.0, 100.0, 1e3, 1e4, 2e4, 1e6, 1e9):
        c = direct_camis(lens, cam, pa, n_t)[0]
        ok_b &= 1.0 / n_t <= c <= 1.0
        ok_m &= c <= prev
        if 1.0 / n_t > ratio:
            ok_m &= c == 1.0 / n_t and c < prev
        else:
            ok_m &= c == ratio
        prev = c
        print(f"{n_t:g}:{c:.4g} ", end="")
    print(f"\n  P(y)/P(x) = {ratio:.6g}: c = 1/n_t while that clamp binds, then flat at the ratio")
    check("c in [1/n_t, 1] over the n_t sweep", ok_b)
    check("c non-increasing in n_t; strictly falling while 1/n_t binds", ok_m)

    # large disk: every P -> 1, c -> 1 exactly (balance)
    cp, ce, py, pzp, pze = direct_camis(lens, cam, pa, N_LIGHT, r_scale=1e4)
    cr = renderer_camis(lens, cam, pa, N_LIGHT, r_scale=1e4)[0]
    print(f"  r x 1e4:  P(y)={py:g} P(z)={pzp:g}/{pze:g}  c proxy {cp:g} exact {ce:g} renderer {cr:g}")
    check("large disk: c == 1 (proxy, exact, renderer)", cp == 1.0 and ce == 1.0 and cr == 1.0)

    # small disk: depends on the edge counts, NOT -> 1 in general
    print("  small disk (photon a: 2 prefix edges; proxy 1 suffix factor, exact 3):")
    for rs in (1.0, 1e-1, 1e-2, 1e-3):
        cp, ce, *_ = direct_camis(lens, cam, pa, N_LIGHT, r_scale=rs)
        print(f"    r x {rs:<6g} c proxy {cp:.6g}   c exact {ce:.6g}")
    cp, ce, *_ = direct_camis(lens, cam, pa, N_LIGHT, r_scale=1e-3)
    check("small disk: proxy c -> 1/n_t (2 prefix vs 1 suffix factor)", cp == 1.0 / N_LIGHT)
    check("small disk: exact c -> 1 (2 prefix vs 3 suffix factors)", ce > 0.999)

    # glossy -> delta prefix: P(y) -> 1, c -> 1
    print("  camera lobe -> delta (Phong exponent m at y1, y2, peak along the path):")
    prev, ok = 0.0, True
    for m in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6):
        pw = (m + 1.0) / (2.0 * math.pi)
        pa_edges = [pw * g_to_area(cam[i], cam[i + 1]) for i in range(2)]
        c = direct_camis(lens, cam, pa, N_LIGHT, cam_edge_p=pa_edges)[0]
        cr = renderer_camis(lens, cam, pa, N_LIGHT, pdf_fwd=[pw, pw])[0]
        ok &= c >= prev and rel(cr, c) < 1e-14
        prev = c
        print(f"    m = {m:<8g} c {c:.6g}   renderer {cr:.6g}")
    check("glossier prefix: c non-decreasing, reaches 1", ok and prev == 1.0)
    cr = renderer_camis(lens, cam, pa, N_LIGHT, delta=[True, True])[0]
    print(f"    delta y1, y2 (dVCM reset to 0): renderer c = {cr:g}")
    check("delta prefix: renderer c == 1 exactly (dVCM = 0 edge -> P = 1)", cr == 1.0)


# ── 3. units: the paper's form is scale-free, the sketch's is not ─────────

def section_scale():
    print("\n=== rescale the whole scene by s (light area by s^2) ===")
    base = None
    ok = True
    for s in (1.0, 10.0, 0.01):
        lens, cam, pa, pb = scene(s)
        la = area.LIGHT_AREA * s * s
        cp, ce, *_ = direct_camis(lens, cam, pa, N_LIGHT, light_area=la)
        cr = renderer_camis(lens, cam, pa, N_LIGHT)[0]
        sk = sketch_c(lens, cam, pa, N_LIGHT, s)
        if base is None:
            base = (cp, ce)
        ok &= rel(cp, base[0]) < 1e-13 and rel(ce, base[1]) < 1e-13 and rel(cr, cp) < 1e-13
        print(f"  s = {s:<5g}  c proxy {cp:.9f}  exact {ce:.9f}  renderer {cr:.9f}   plan sketch {sk:.6g}")
    check("paper form is scale-invariant (the sketch's is not)", ok)


# ── 4. partition of unity over the whole merge path ────────────────────────

def section_partition():
    """The idealized path (photon a landing exactly on x): y0=z0a, z1a, x,
    y2, y1, lens. Truth: Eq. 11 with c_k at EVERY merge vertex; the 3-day
    cut: c only in the evaluated merge's own weight."""
    lens, cam, pa, _ = scene()
    y1, y2, xm = cam
    y0, z1 = pa[0], pa[1]
    xs = [z1, xm, y2, y1]                       # area harness order: light -> lens
    n = len(xs)
    r = norm(sub(y1.pos, lens.pos)) * TAN_1DEG
    r2 = r * r
    c_ex, c_px = [], []
    for k in range(n):
        py = 1.0
        for j in range(k, n - 1):               # camera edges xs[j+1] -> xs[j]
            py *= edge_P(r2, edge_pdf(xs[j + 1], xs[j]))
        le = [light_edge_pdf(y0, xs[0])] + [edge_pdf(xs[j - 1], xs[j]) for j in range(1, k + 1)]
        pz = edge_P(r2, 1.0 / area.LIGHT_AREA)
        for p in le:
            pz *= edge_P(r2, p)
        c_ex.append(c_from_P(py, pz, N_LIGHT))
        c_px.append(c_from_P(py, edge_P(r2, le[-1]), N_LIGHT))
    etas = [ETA] * n
    d_bal = area.strategy_densities(y0, xs, lens, etas)
    d_cam = area.strategy_densities(y0, xs, lens, [c * ETA for c in c_ex])
    wb = {k: v / sum(d_bal.values()) for k, v in d_bal.items()}
    wc = {k: v / sum(d_cam.values()) for k, v in d_cam.items()}

    def local_merge(k, c):
        """Carries built with the uncorrected eta; c only at x_k's divisor."""
        (lvcm, lvc), _ = area.light_carries(y0, xs, etas, k)
        cvcm, cvc = area.camera_carries(xs, lens, etas, k)
        x = xs[k]
        cam_dir_w = p_dir(x, unit(sub(area.light_pred(y0, xs, k).pos, x.pos)))
        cam_rev_w = p_dir(x, unit(sub(area.cam_pred(xs, lens, k).pos, x.pos)))
        return merge_weight((lvcm, lvc), (cvcm, cvc), cam_dir_w, cam_rev_w, c * etas[k])

    print("\n=== partition of unity, path z0 z1 x y2 y1 lens (photon a on x) ===")
    print(f"  {'strategy':12s} {'balance':>10s} {'CAMIS':>10s} {'renderer':>10s} "
          f"{'local ex.':>10s} {'3-day':>10s}   {'c exact':>9s} {'c proxy':>9s}")
    s_local, s_lex, worst_b, worst_c = 0.0, 0.0, 0.0, 0.0
    for key in sorted(wb, key=lambda k: (k[0], k[1])):
        if key[0] == "merge":
            k = key[1]
            rb = local_merge(k, 1.0)                                # today's weight
            rc = area.renderer_merge(y0, xs, lens, [c * ETA for c in c_ex], k)
            wle = local_merge(k, c_ex[k])       # exact c, but only in its own weight
            w3 = local_merge(k, c_px[k])
            worst_b = max(worst_b, rel(rb, wb[key]))
            worst_c = max(worst_c, rel(rc, wc[key]))
            s_local += w3
            s_lex += wle
            print(f"  {str(key):12s} {wb[key]:10.6f} {wc[key]:10.6f} {rc:10.6f} {wle:10.6f} {w3:10.6f}"
                  f"   {c_ex[k]:9.4g} {c_px[k]:9.4g}")
        else:
            s_local += wb[key]
            s_lex += wb[key]
            print(f"  {str(key):12s} {wb[key]:10.6f} {wc[key]:10.6f} {'':10s} {wb[key]:10.6f} {wb[key]:10.6f}")
    print(f"  {'SUM':12s} {sum(wb.values()):10.6f} {sum(wc.values()):10.6f} {'':10s} {s_lex:10.6f} {s_local:10.6f}")
    print(f"  3-day cut keeps {s_local:.4f} of the path's energy ({100 * (1 - s_local):.2f}% dark);"
          f" locality alone (exact c) keeps {s_lex:.4f}")
    check("renderer merge weight (c=1) == balance truth", worst_b < 1e-13, f"{worst_b:.1e}")
    check("c*eta at every vertex in the carries == CAMIS truth (Eq. 11)", worst_c < 1e-13, f"{worst_c:.1e}")
    check("exact CAMIS weights sum to 1", abs(sum(wc.values()) - 1.0) < 1e-14)
    check("3-day cut (c in own weight only) sums BELOW 1 -- biased dark, as documented",
          s_local < 1.0 and all(c <= 1.0 for c in c_px))
    return s_local, s_lex


# ── 5. random geometry ─────────────────────────────────────────────────────

def section_random(count=2000):
    rng = random.Random(2021)

    def rv():
        return (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))

    def facing(p, others):
        """A normal at p facing every neighbour (no back-facing edges)."""
        acc = (0.0, 0.0, 0.0)
        for o in others:
            u = unit(sub(o, p))
            acc = (acc[0] + u[0], acc[1] + u[1], acc[2] + u[2])
        jit = rv()
        return unit((acc[0] + 0.3 * jit[0], acc[1] + 0.3 * jit[1], acc[2] + 0.3 * jit[2]))

    done, worst, ok_b, ok_p = 0, 0.0, True, True
    ratios = []                                  # c_proxy / c_exact at the paper's radius
    while done < count:
        lens_p, y1p, y2p, xp, z1p = rv(), rv(), rv(), rv(), rv()
        z0p = (rng.uniform(-1, 1), 1.5, rng.uniform(-1, 1))
        pts = [lens_p, y1p, y2p, xp, z1p, z0p]
        if min(norm(sub(a, b)) for i, a in enumerate(pts) for b in pts[i + 1:]) < 0.1:
            continue
        lens = V(lens_p, None)
        y1 = V(y1p, facing(y1p, [lens_p, y2p]))
        y2 = V(y2p, facing(y2p, [y1p, xp]))
        x = V(xp, facing(xp, [y2p, z1p]))
        z1 = V(z1p, facing(z1p, [z0p, xp]))
        z0 = V(z0p, unit(sub(z1p, z0p)))
        cam, lp = [y1, y2, x], [z0, z1, x]
        cos_ok = all(abs(dot(v.n, unit(sub(o.pos, v.pos)))) > 0.05
                     for v, o in ((y1, lens), (y1, y2), (y2, y1), (y2, x), (x, y2), (x, z1), (z1, x), (z1, z0)))
        if not cos_ok:
            continue
        rs = 10.0 ** rng.uniform(-2.0, 2.0)
        n_t = 10.0 ** rng.uniform(0.0, 7.0)
        cp, ce, *_ = direct_camis(lens, cam, lp, n_t, r_scale=rs)
        cr, *_ = renderer_camis(lens, cam, lp, n_t, r_scale=rs)
        worst = max(worst, rel(cr, cp))
        ok_b &= 1.0 / n_t <= cp <= 1.0 and 1.0 / n_t <= ce <= 1.0
        ok_p &= cp <= ce
        c1p, c1e, *_ = direct_camis(lens, cam, lp, N_LIGHT)
        ratios.append(c1p / c1e)
        done += 1
    print(f"\n=== {count} random configurations (r x 1e-2..1e2, n_t 1..1e7) ===")
    print(f"  max rel err direct vs renderer: {worst:.2e}")
    ratios.sort()
    within2 = sum(1 for q in ratios if q > 0.5) / len(ratios)
    print(f"  at r x 1, n_t {N_LIGHT:g}: c_proxy/c_exact median {ratios[len(ratios) // 2]:.3g}, "
          f"10th pct {ratios[len(ratios) // 10]:.3g}; within 2x of exact on {100 * within2:.1f}% of paths")
    check(f"random: direct == renderer over {count} paths", worst < 1e-13, f"{worst:.1e}")
    check("random: c in [1/n_t, 1] (proxy and exact)", ok_b)
    check("random: c_proxy <= c_exact", ok_p)
    return worst, ratios[len(ratios) // 2], within2


if __name__ == "__main__":
    w_ph = section_photons()
    section_bounds()
    section_scale()
    kept, kept_ex = section_partition()
    w_rand, med, within2 = section_random()
    print("\n=== checks ===")
    n_fail = 0
    for name, ok, detail in CHECKS:
        n_fail += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    print(f"""
WHAT THIS SETTLES
  * c_VM as implemented (paper Eq. 13-17, 3-day proxy for P(z)):
        r^2   = (t_hit_primary * tan(pi/180))^2               one per camera path
        logPy = sum over camera arrivals AFTER the first of
                log(pi r^2 / max(cv.dVCM, pi r^2))            (= log min(pi r^2 p_A, 1))
        Pz    = pi r^2 / max(lv.dVCM, pi r^2)                 terminal light edge only
        c     = clamp(1 / (1 + Pz * expm1(-logPy)), 1/n_t, 1)
        eta_scale(x) *= c                                     (NOT inv_eta_x *= c)
  * The log-space renderer form reproduces the direct definition to
    {max(w_ph, w_rand):.1e} relative. That is the floor for a log/exp round trip:
    the error scales with |log P(y)| * eps, so ~1e-16 is only reached when
    P(y) is near 1; the per-edge terms themselves agree to <= 4 eps.
  * The single-terminal-edge P(z) proxy is NOT ablation-quality at the
    paper's radius: P(y) keeps one O(r^2) factor per camera edge, the proxy
    only one, so with 2+ camera bounces c collapses toward 1/n_t. Random
    paths: c_proxy/c_exact median {med:.3g}, within 2x on only
    {100 * within2:.1f}%. See the "alternatives" lines of the photon table.
  * The 3-day cut is biased: c in the merge's own weight only kept
    {kept:.4f} of this path's energy ({kept_ex:.4f} with exact c -- the rest
    of the loss is the proxy). Expect Day-3 renders to read darker wherever
    merging competes with connections, and compare means, not just relMSE,
    before crediting E/D with a variance win.
""")
    sys.exit(1 if n_fail else 0)
