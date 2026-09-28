#!/usr/bin/env python3
"""Numerical proof that CAMIS for VCM can be evaluated TRACE-BLIND.

Sibling of vcm_camis_derivation.py (the CAMIS factor c itself, and the
rejected 3-day cut) and vcm_area_mis_derivation.py (the dVCM/dVC carries and
the strategy enumeration). Both are imported, not copied. Same discipline as
every harness in this family: say what the quantity IS straight from its
definition, then reproduce it a second, independent way from ONLY what the
renderer can have at hand, and demand that the two agree.

THE QUESTION (plan deep-hugging-locket.md, stage S0)
  vcm_camis_derivation.section_partition() proves that feeding c_k(x) eta_k
  into the ordinary carry recursion at every vertex reproduces CAMIS
  (Grittmann et al. 2021, Eq. 11) and keeps its partition of unity. But it
  forms c POST HOC, with the whole path in hand. The renderer cannot: the
  camera subpath is walked and its state stored before it is ever paired
  with a photon or a light vertex, and the light subpath is walked before
  any camera path exists. c_k(x) depends on BOTH halves (Eq. 13-15:
  c = clamp(P(y)/P(x), 1/n_t, 1), P(x) = P(y) + P(z) - P(y) P(z)), with a
  different pole per vertex, so no single running scalar can hold it.

THE ARCHITECTURE BEING PROVEN (the plan's "class-gated exact hybrid")
  The dVC recursion is AFFINE in the per-vertex eta:
      dVC_T(eta) = dVC0_T + sum_{i<T} eta_i mu_{i,T}
      mu_{i,T}   = B_i prod_{l=i+1}^{T-1} A_l
      B_l = cos_out_l / (pdf_fwd_l cos_{l+1}),   A_l = B_l pdf_rev_l
  (delta vertex: A = cos_out/cos_{l+1}, B = 0). dVC0 is the same recursion
  with eta = 0. Neither dVC0 nor mu depends on the pairing, so each subpath
  can store, at trace time,
      running:    dVCM, dVC0, log A_T (= sum log A_l), cut, g_prev, r^2, class
      per vertex: u_i = log(eta_i) - log(pdf_rev_i) - log A_i,  log keep_i,
                  log P(y)_i  (camera: prefix product, from the dVCM carry)
                  RC_i        (camera: sum of clamped reverse-edge logs)
                  log p_A fwd / rev  (light: UNCLAMPED -- r is the camera's)
  and at evaluation time, once both halves are known, form every c_i and
      dVC'_T = dVC0_T + sum_i c_i exp(u_i + log A_T)
  which is dVC_T with eta_i -> c_i eta_i at every vertex: exactly the
  post-hoc computation, regrouped. Paths outside the Class (here: any delta
  vertex) keep c = 1 and the stored legacy dVC -- today's exact weights.

  This file builds exactly that: trace_camera_records / trace_light_records
  see ONLY their own subpath; eval_merge / eval_connect / eval_splat /
  eval_emission_hit see ONLY two record views plus junction data (the BSDF
  and geometry evaluations the renderer performs at the evaluation site).
  Every strategy of every test path builds its OWN records from its OWN
  subpaths, so no evaluator can peek at the other half.

GROUND TRUTH
  strategy_densities_general(): vcm_area_mis_derivation.strategy_densities
  generalised to per-vertex sampling pdfs that depend on BOTH directions
  (glossy lobes, so pdf_rev != pdf_fwd; a one-sided sampler, so pdf_rev = 0;
  a delta vertex). It is checked equal to the original on Lambertian paths.
  truth_c(): Eq. 13-17 straight from vcm_camis_derivation.edge_P / c_from_P,
  checked equal to vcm_camis_derivation.direct_camis. Eq. 11 is then the
  balance heuristic over those densities with eta_k -> c_k eta_k.

CONVENTIONS FROZEN HERE (plan S0: "freeze every convention")
  r      = |y1 - lens| tan(1 deg) (Eq. 17), y1 = the CAMERA's primary hit;
           for the t=1 splat that is the light vertex the lens sees.
  P(edge) = min(pi r^2 p_A, 1) per edge (Eq. 16), p_A in AREA measure.
  P(primary lens edge) = 1, so merging at the primary hit always has c = 1.
  P(z_0) = min(pi r^2 P_A, 1), P_A = pick prob / light area.
  lb_i   = 1 / (n_t keep_i)  -- the plan's lower clamp with thinning.
  c      = logistic form 1 / (1 + exp(log Pz - log Py + log(-expm1(log Py))))
           = P(y)/P(x), overflow-safe; P(z) = 0 or P(y) = 1 gives c = 1.
  Class  = area light AND no delta vertex anywhere in the full path; each
           subpath carries a class bit, a strategy uses CAMIS iff both are set.

WHAT THIS FILE CHANGES OR PINS DOWN IN THE PLAN'S STATE LIST (all exercised below)
  1. pdf_rev = 0 (the "cut"). The plan's u_i = log eta - log pdf_rev - log A_i
     is +inf at the cut vertex itself. The rule that works: at a cut at
     vertex s set cut = s and reset BOTH log A and RC to 0; the term AT s
     survives (u_s = log(eta_s B_s)), every term before s is dead (its mu
     contains A_s = 0). Where pdf_rev > 0 the stored u equals the plan's u
     exactly (checked). The LIGHT side needs its own cut index as well (the
     plan lists only "flags" there) -- light-side pdf_rev = 0 is exercised.
  2. The junction's OWN reverse edge is pairing-dependent on both sides:
     camera T -> T-1 (light direction) = cam_rev_w g_prev; light L -> L-1
     (camera direction) = cam_dir_w g_rev (merge) or light_rev_w g_rev
     (connect/splat). g_rev is |cos at the PREDECESSOR| / d^2 -- so log g_j
     must be kept per light vertex, as the plan has it, even though the
     non-junction uses only need log pdf_rev_w + log g_j combined.
  3. A light vertex's forward area density is -log(its own arrival dVCM),
     which the LVC already stores: log p_fwd_A need not be a new field.
  4. Light-side clamps need the camera's r, so the light records are walked
     at every evaluation -- twice (P(z) prefix forward, P(y) suffix backward
     from the junction). Camera-side clamps are known at trace time (r is
     set by the camera's own primary hit), so logPy_i and RC_i are prefix
     sums and the camera walk is one pass.
  5. s = 0 (emission hit) needs the position density direct_pdf_a (for
     P(z_0)) and the DIRECTIONAL emission pdf (for the z_0 -> x_0 edge)
     separately, not only their product emission_pdf_w.
  6. Float32: the plan's log-form u works (partition error ~1e-7 here), but
     its error grows with the drift of the running log A; the Horner form
     (see section 4) does not have that failure mode.
"""
import math
import os
import random
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcm_area_mis_derivation as area                             # noqa: E402
import vcm_camis_derivation as camis                               # noqa: E402
from vcm_area_mis_derivation import (V, sub, dot, norm, unit, neg,  # noqa: E402
                                     p_dir, p_emit, g_to_area, cam_pred, light_pred)
from vcm_camis_derivation import (TAN_1DEG, N_LIGHT, edge_P, c_from_P,  # noqa: E402
                                  log_edge_from_carry, rel)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))


# ── arithmetic back ends: Python double, and numpy float32 (the GPU's) ─────

class _F64:
    name = "float64"
    inf = math.inf

    @staticmethod
    def f(x):
        return float(x)

    @staticmethod
    def log(x):
        return math.log(x) if x > 0.0 else -math.inf

    @staticmethod
    def exp(x):
        return math.inf if x > 709.0 else math.exp(x)

    @staticmethod
    def expm1(x):
        return math.inf if x > 709.0 else math.expm1(x)


class _F32:
    """Every value that enters the record/eval arithmetic is rounded to
    float32 once, where the renderer would compute it (a cos, a pdf, a d^2),
    and every operation after that is float32. Transcendentals go through
    numpy so they stay float32 (math.log would silently promote)."""
    name = "float32"
    inf = np.float32(np.inf)

    @staticmethod
    def f(x):
        return np.float32(x)

    @staticmethod
    def log(x):
        with np.errstate(divide="ignore"):
            return np.log(np.float32(x))

    @staticmethod
    def exp(x):
        with np.errstate(over="ignore"):
            return np.exp(np.float32(x))

    @staticmethod
    def expm1(x):
        with np.errstate(over="ignore"):
            return np.expm1(np.float32(x))


F64, F32 = _F64(), _F32()


# ── geometry and sampling pdfs ─────────────────────────────────────────────

PHONG_M = 20.0          # glossy lobe exponent
GLOSSY_DIFFUSE = 0.15   # diffuse share of the glossy sampler (keeps pdf > 0)


def dir_to(a, b):
    return unit(sub(b.pos, a.pos))


def dist2(a, b):
    d = sub(b.pos, a.pos)
    return dot(d, d)


def kind(v):
    return getattr(v, "kind", "lambert")


def keep(v):
    """Merge-grid thinning's keep probability at v (eta(x) = keep(x) eta)."""
    return getattr(v, "keep", 1.0)


def is_delta(v):
    return kind(v) == "delta"


def is_plain(v):
    """The MVP Class's per-vertex rule: a plain, non-delta surface vertex."""
    return not is_delta(v)


def reflect(w, n):
    d = 2.0 * dot(w, n)
    return (d * n[0] - w[0], d * n[1] - w[1], d * n[2] - w[2])


def bsdf_pdf(v, w_other, w_samp):
    """Solid-angle density of sampling w_samp at v, given the path's other
    direction w_other (both point AWAY from v). pdf_fwd = bsdf_pdf(v, w_in,
    w_out), pdf_rev = bsdf_pdf(v, w_out, w_in).
      lambert  |cos_samp| / pi                       (== area.p_dir)
      glossy   diffuse + Phong lobe about the mirror of w_other, times
               |cos_samp|: the lobe is symmetric, the cos is not, so
               pdf_rev != pdf_fwd
      cut      one-sided: 2|cos|/pi on the side of v.t, else 0 -- sampling
               can cross v one way only, so pdf_rev = 0 (the plan's "cut")
      delta    a Dirac of equal mass both ways; only ratios matter, and the
               carries use SmallVCM's specular rule (pdf_rev/pdf_fwd = 1)"""
    k = kind(v)
    if k == "lambert":
        return p_dir(v, w_samp)
    cs = abs(dot(v.n, w_samp))
    if k == "glossy":
        lobe = max(0.0, dot(w_samp, reflect(w_other, v.n))) ** PHONG_M
        return (GLOSSY_DIFFUSE / math.pi + (1.0 - GLOSSY_DIFFUSE) * (PHONG_M + 2.0)
                / (2.0 * math.pi) * lobe) * cs
    if k == "cut":
        return 2.0 * cs / math.pi if dot(w_samp, v.t) > 0.0 else 0.0
    if k == "delta":
        return 1.0
    raise ValueError(k)


# ── ground truth: every strategy's density, straight from the definitions ──

def truth_edges(y0, xs, lens):
    """AREA densities of every edge, both directions.
    cam[k]  camera sampling xs[k] from its camera-side neighbour
            (cam[n-1] is the lens edge: P_CAM_AREA g)
    cam_y0  camera sampling y0 from xs[0]
    lig[k]  light sampling xs[k] from its light-side neighbour (lig[0] from y0)."""
    n = len(xs)
    cam = [0.0] * n
    cam[n - 1] = area.P_CAM_AREA * area.cam_edge_pdf(lens, xs[n - 1])
    for k in range(n - 2, -1, -1):
        a = xs[k + 1]
        cam[k] = bsdf_pdf(a, dir_to(a, cam_pred(xs, lens, k + 1)), dir_to(a, xs[k])) * g_to_area(a, xs[k])
    a = xs[0]
    cam_y0 = bsdf_pdf(a, dir_to(a, cam_pred(xs, lens, 0)), dir_to(a, y0)) * g_to_area(a, y0)
    lig = [0.0] * n
    lig[0] = p_emit(y0, dir_to(y0, xs[0])) * g_to_area(y0, xs[0])
    for k in range(1, n):
        a = xs[k - 1]
        lig[k] = bsdf_pdf(a, dir_to(a, light_pred(y0, xs, k - 1)), dir_to(a, xs[k])) * g_to_area(a, xs[k])
    return cam, cam_y0, lig


def strategy_densities_general(y0, xs, lens, etas):
    """area.strategy_densities with direction-dependent pdfs and delta
    vertices (a strategy that connects or merges AT a delta vertex has no
    density). Same keys: ("s", s), ("t1", n-1), ("merge", k)."""
    n = len(xs)
    cam, cam_y0, lig = truth_edges(y0, xs, lens)
    C = [0.0] * n
    C[n - 1] = cam[n - 1]
    for k in range(n - 2, -1, -1):
        C[k] = C[k + 1] * cam[k]
    L = [0.0] * n
    L[0] = area.P_A * lig[0]
    for k in range(1, n):
        L[k] = L[k - 1] * lig[k]
    dl = [is_delta(x) for x in xs]
    d = {("s", 0): C[0] * cam_y0,
         ("s", 1): 0.0 if dl[0] else area.P_A * C[0]}
    for s in range(2, n + 1):
        d[("s", s)] = 0.0 if (dl[s - 2] or dl[s - 1]) else L[s - 2] * C[s - 1]
    d[("t1", n - 1)] = 0.0 if dl[n - 1] else area.N_SPLAT * L[n - 1] * area.P_CAM_AREA
    for k in range(n):
        d[("merge", k)] = 0.0 if dl[k] else etas[k] * L[k] * C[k]
    return d


def camis_r2(lens, primary, r_scale=1.0):
    return (norm(sub(primary.pos, lens.pos)) * TAN_1DEG * r_scale) ** 2   # Eq. 17


def truth_c(y0, xs, lens, n_t=N_LIGHT, r_scale=1.0):
    """c_k(x) for merging at every xs[k] of the FULL path (Eq. 13-17), 1 for
    a path outside the Class."""
    n = len(xs)
    if not all(is_plain(x) for x in xs):
        return [1.0] * n
    r2 = camis_r2(lens, xs[n - 1], r_scale)
    cam, _, lig = truth_edges(y0, xs, lens)
    out = []
    for k in range(n):
        py = 1.0                                   # primary edge: P = 1
        for j in range(k, n - 1):
            py *= edge_P(r2, cam[j])
        pz = edge_P(r2, area.P_A)                  # P(z_0)
        for j in range(k + 1):
            pz *= edge_P(r2, lig[j])
        out.append(c_from_P(py, pz, n_t * keep(xs[k])))
    return out


def truth_weights(y0, xs, lens, n_t=N_LIGHT, r_scale=1.0, camis_on=True):
    """Eq. 11: balance heuristic over the densities with eta_k -> c_k eta_k."""
    n = len(xs)
    c = truth_c(y0, xs, lens, n_t, r_scale) if camis_on else [1.0] * n
    etas = [c[k] * keep(xs[k]) * n_t * area.KERNEL for k in range(n)]
    d = strategy_densities_general(y0, xs, lens, etas)
    tot = sum(d.values())
    return {k: v / tot for k, v in d.items()}, d, c


# ── trace time: records, built from ONE subpath only ───────────────────────

@dataclass(frozen=True)
class CamScatter:
    """Written when the camera leaves vertex tau (finalised on arriving at
    tau+1, whose cos B_tau needs). Never read by an evaluation AT tau."""
    tau: int
    u: float          # log(eta B) - log A_{tau+1}  (== the plan's u where pdf_rev > 0)
    u_plan: float     # log eta - log pdf_rev - log A_tau, as the plan writes it
    log_keep: float   # lb = 1 / (n_t keep)
    log_py: float     # log P(y) for merging AT tau: prefix product, 0 at the primary hit
    rc: float         # sum_{s=1..tau} log P(reverse edge s -> s-1), reset at a cut
    a_lin: float      # A_tau        (linear/Horner alternative to u)
    etab_lin: float   # eta_tau B_tau


@dataclass(frozen=True)
class CamArrival:
    """The running registers on arriving at camera vertex T."""
    tau: int
    dvcm: float
    dvc: float        # LEGACY dVC (base eta): the out-of-Class fallback
    dvc0: float       # dVC with eta = 0 (the plan's repurposed dVM slot)
    log_a: float      # sum log A_l, l in (cut, T-1]
    cut: int          # terms tau >= cut are live (a pdf_rev = 0 at `cut` kills earlier ones)
    log_py: float
    rc_prev: float    # rc of the last scattered vertex (T-1)
    log_g_prev: float  # log |cos at T-1| / d^2 of the arrival edge (reverse-edge area factor)
    eta: float
    log_keep: float
    r2: float
    log_k: float      # log(pi r^2)
    in_class: bool


@dataclass(frozen=True)
class LightOrigin:
    log_pa0: float    # log P_A: P(z_0) = min(pi r^2 P_A, 1)
    direct_pdf_a: float
    in_class: bool    # area emitter


@dataclass(frozen=True)
class LightScatter:
    lam: int          # 1 = xs[0]
    u: float
    u_plan: float
    log_keep: float
    log_pa_fwd: float  # log p_A of the edge INTO lam (unclamped: r unknown here)
    log_pa_rev: float  # log p_A of the camera-direction edge lam -> lam-1 (unclamped)
    a_lin: float
    etab_lin: float


@dataclass(frozen=True)
class LightArrival:
    lam: int
    dvcm: float
    dvc: float
    dvc0: float
    log_a: float
    cut: int
    log_pa_fwd: float
    log_g_rev: float  # log |cos at lam-1| / d^2 (camera-direction reverse edge factor)
    eta: float
    log_keep: float
    in_class: bool


def _eta_of(M, v, n_t, eta_mult):
    m = keep(v) * (eta_mult.get(id(v), 1.0) if eta_mult else 1.0)
    return M.f(n_t * area.KERNEL) * M.f(m)


def _log_edge_from_carry(M, log_k, dvcm):
    """camis.log_edge_from_carry in log form: log min(pi r^2 / dVCM, 1)."""
    return min(log_k - M.log(dvcm), M.f(0.0))


def _clampP(M, log_k, log_p):
    """log min(pi r^2 p, 1), Eq. 16."""
    return min(log_k + log_p, M.f(0.0))


def _scatter(M, v, prev, nxt, dvcm, dvc, dvc0, eta):
    """One scatter at v (arrived from prev, leaving toward nxt), then the
    arrival cos at nxt folded into A/B. Returns the new (dvcm, dvc, dvc0)
    BEFORE the arrival divide (the loop divides), and (A, eta B, pdf_rev)."""
    f = M.f
    w_in, w_out = dir_to(v, prev), dir_to(v, nxt)
    cos_out = f(abs(dot(v.n, w_out)))
    cos_next = f(abs(dot(nxt.n, w_out)))
    if is_delta(v):                                   # SmallVCM's specular rule
        return (f(0.0), dvc * cos_out, dvc0 * cos_out), (cos_out / cos_next, f(0.0), f(1.0))
    pf = f(bsdf_pdf(v, w_in, w_out))
    pr = f(bsdf_pdf(v, w_out, w_in))
    q = cos_out / pf                                  # vcm_scatter_carries' cos_over_pdf
    carries = (f(1.0) / pf, q * (dvc * pr + dvcm + eta), q * (dvc0 * pr + dvcm))
    b = q / cos_next
    return carries, (b * pr, eta * b, pr)


def trace_camera_records(lens, cam_vs, n_t=N_LIGHT, M=F64, r_scale=1.0, eta_mult=None, log_a_gauge=0.0):
    """The camera walk of vcm_area_mis_derivation.camera_carries (dVC starts at
    0, scatter then arrive), plus the CAMIS records. cam_vs = [y1, y2, ...] in
    CAMERA order -- the only vertices this function sees.
    Returns (scatters, arrivals): scatters[tau] for every vertex it left,
    arrivals[T] for every vertex it reached.
    log_a_gauge: u_i + log A_T is invariant under log A -> log A + K,
    u -> u - K; a nonzero K emulates a path whose log A register has drifted
    far from 0 (the float32 precision test, section 4)."""
    f = M.f
    zero = f(0.0)
    gauge = f(log_a_gauge)
    r2 = f(camis_r2(lens, cam_vs[0], r_scale))
    log_k = M.log(f(math.pi) * r2)
    dvcm, dvc, dvc0 = f(area.N_SPLAT / area.P_CAM_AREA), zero, zero
    log_a, cut, log_py, rc, log_g_prev = gauge, 0, zero, zero, zero
    in_class = True
    scat, arr = [], []
    prev = lens
    for tau, v in enumerate(cam_vs):
        w = dir_to(prev, v)
        cos_v = f(abs(dot(v.n, w)))
        dvcm = dvcm * f(dist2(prev, v)) / cos_v
        dvc, dvc0 = dvc / cos_v, dvc0 / cos_v
        if tau > 0:
            log_py = log_py + _log_edge_from_carry(M, log_k, dvcm)
            log_g_prev = M.log(f(abs(dot(prev.n, w)) / dist2(prev, v)))
        in_class = in_class and is_plain(v)
        eta = _eta_of(M, v, n_t, eta_mult)
        log_keep = M.log(f(keep(v)))
        arr.append(CamArrival(tau, dvcm, dvc, dvc0, log_a, cut, log_py, rc, log_g_prev,
                              eta, log_keep, r2, log_k, in_class))
        if tau == len(cam_vs) - 1:
            break
        (dvcm, dvc, dvc0), (a_lin, etab, pr) = _scatter(M, v, prev, cam_vs[tau + 1], dvcm, dvc, dvc0, eta)
        u_plan = M.log(eta) - M.log(pr) - log_a
        if a_lin == 0.0:                   # pdf_rev = 0: every earlier eta term is dead
            cut, log_a = tau, gauge
        else:
            log_a = log_a + M.log(a_lin)
        u = M.log(etab) - log_a            # -inf at a delta vertex: never read (out of Class)
        if tau == 0 or pr == 0.0:          # no reverse edge to the lens; reset at a cut
            rc = zero
        else:
            rc = rc + _clampP(M, log_k, M.log(pr) + log_g_prev)
        scat.append(CamScatter(tau, u, u_plan, log_keep, log_py, rc, a_lin, etab))
        prev = v
    return scat, arr


def trace_light_records(z0, light_vs, n_t=N_LIGHT, M=F64, eta_mult=None, log_a_gauge=0.0):
    """The light walk of vcm_area_mis_derivation.light_carries plus the CAMIS
    records. z0 = the emitter point, light_vs = [xs[0], ...] in LIGHT order.
    Edge densities are stored UNCLAMPED: the clamp needs r, which is set by
    the camera's primary hit and unknown here.
    Returns (origin, scatters, arrivals); scatters[i] / arrivals[i] are
    light vertex lam = i + 1."""
    f = M.f
    zero = f(0.0)
    origin = LightOrigin(M.log(f(area.P_A)), f(area.P_A), True)
    scat, arr = [], []
    if not light_vs:
        return origin, scat, arr
    w0 = dir_to(z0, light_vs[0])
    cos_emit = max(dot(z0.n, w0), 1e-4)
    emission_pdf_w = f(area.P_A * cos_emit / math.pi)
    dvcm, dvc = f(area.P_A) / emission_pdf_w, f(cos_emit) / emission_pdf_w
    dvc0 = dvc                             # no eta at the emitter: dVC0 == dVC here
    gauge = f(log_a_gauge)
    log_a, cut = gauge, 1
    in_class = origin.in_class
    prev = z0
    for i, v in enumerate(light_vs):
        lam = i + 1
        w = dir_to(prev, v)
        cos_v = f(abs(dot(v.n, w)))
        dvcm = dvcm * f(dist2(prev, v)) / cos_v
        dvc, dvc0 = dvc / cos_v, dvc0 / cos_v
        log_pa_fwd = -M.log(dvcm)          # dVCM on arrival IS 1 / p_A(edge in)
        log_g_rev = M.log(f(abs(dot(prev.n, w)) / dist2(prev, v)))
        in_class = in_class and is_plain(v)
        eta = _eta_of(M, v, n_t, eta_mult)
        log_keep = M.log(f(keep(v)))
        arr.append(LightArrival(lam, dvcm, dvc, dvc0, log_a, cut, log_pa_fwd, log_g_rev,
                                eta, log_keep, in_class))
        if i == len(light_vs) - 1:
            break
        (dvcm, dvc, dvc0), (a_lin, etab, pr) = _scatter(M, v, prev, light_vs[i + 1], dvcm, dvc, dvc0, eta)
        u_plan = M.log(eta) - M.log(pr) - log_a
        if a_lin == 0.0:
            cut, log_a = lam, gauge
        else:
            log_a = log_a + M.log(a_lin)
        u = M.log(etab) - log_a
        scat.append(LightScatter(lam, u, u_plan, log_keep, log_pa_fwd, M.log(pr) + log_g_rev, a_lin, etab))
        prev = v
    return origin, scat, arr


@dataclass(frozen=True)
class CamView:
    scat: tuple       # vertices 0..T-1
    arr: CamArrival   # vertex T


@dataclass(frozen=True)
class LightView:
    origin: LightOrigin
    scat: tuple       # lam = 1..Lambda-1
    arr: object       # LightArrival at Lambda, or None when the junction is the emitter


def cam_view(recs, T):
    scat, arr = recs
    return CamView(tuple(scat[:T]), arr[T])


def light_view(recs, lam):
    origin, scat, arr = recs
    return LightView(origin, tuple(scat[:max(lam - 1, 0)]), arr[lam - 1] if lam >= 1 else None)


# ── junction data: what the renderer evaluates at the evaluation site ──────

@dataclass(frozen=True)
class Junction:
    vals: dict

    def __getattr__(self, k):
        try:
            return self.vals[k]
        except KeyError:
            raise AttributeError(k)


def junction_connect(lv, lv_pred, cv, cv_pred, M=F64):
    """lv_pred None: lv is the emitter itself (s = 1)."""
    f = M.f
    w = dir_to(lv, cv)
    d2 = dist2(lv, cv)
    cos_lv, cos_cv = abs(dot(lv.n, w)), abs(dot(cv.n, w))
    if lv_pred is None:
        light_dir_w, light_rev_w = p_emit(lv, w), 0.0
    else:
        wl = dir_to(lv, lv_pred)
        light_dir_w, light_rev_w = bsdf_pdf(lv, wl, w), bsdf_pdf(lv, w, wl)
    wc = dir_to(cv, cv_pred)
    cam_dir_w, cam_rev_w = bsdf_pdf(cv, wc, neg(w)), bsdf_pdf(cv, neg(w), wc)
    return Junction(dict(light_dir_a=f(light_dir_w * cos_cv / d2), cam_dir_a=f(cam_dir_w * cos_lv / d2),
                         light_rev_w=f(light_rev_w), cam_rev_w=f(cam_rev_w)))


def junction_merge(x, cv_pred, lv_pred, M=F64):
    """The photon landed on x; its stored incoming direction points at lv_pred."""
    wc, wl = dir_to(x, cv_pred), dir_to(x, lv_pred)
    return Junction(dict(cam_dir_w=M.f(bsdf_pdf(x, wc, wl)), cam_rev_w=M.f(bsdf_pdf(x, wl, wc))))


def junction_splat(lv, lv_pred, lens, M=F64, r_scale=1.0):
    """t = 1: r is formed HERE, from the light vertex the lens sees."""
    f = M.f
    r2 = f(camis_r2(lens, lv, r_scale))
    return Junction(dict(cam_pdf_a=f(area.P_CAM_AREA * area.cam_edge_pdf(lens, lv)),
                         rev=f(bsdf_pdf(lv, dir_to(lv, lens), dir_to(lv, lv_pred))),
                         log_k=M.log(f(math.pi) * r2)))


def junction_emission(y0, x0, M=F64):
    """s = 0: the camera ray from x0 hit the emitter at y0. CAMIS needs the
    position density (P(z_0)) and the DIRECTIONAL emission density (the
    y0 -> x0 edge) separately, not only their product emission_pdf_w."""
    f = M.f
    pe = p_emit(y0, dir_to(y0, x0))
    return Junction(dict(direct_pdf_a=f(area.P_A), emission_pdf_w=f(area.P_A * pe), log_p_emit=M.log(f(pe))))


# ── evaluation time: c at every vertex, from two record views ──────────────

def camis_c(M, log_py, log_pz, log_lb):
    """Eq. 13 + 15 in the logistic form: P(y)/P(x) = 1/(1 + Pz (1/Py - 1)),
    log(1/Py - 1) = -log Py + log(-expm1(log Py)). Never forms Py or Px,
    never overflows (exp -> inf gives c = 0 -> the 1/n_t clamp)."""
    one = M.f(1.0)
    if log_pz == -M.inf or log_py >= 0.0:
        return one
    z = log_pz - log_py + M.log(-M.expm1(log_py))
    c = one / (one + M.exp(z))
    return min(max(c, M.exp(log_lb)), one)


def _camera_side(M, cam, n_t, log_pz_T, log_rev_T, form, force_c1, info):
    """dVC'_T = dVC0_T + sum_tau c_tau eta_tau mu_{tau,T}, and c_T.
    log_pz_T  log P(z) of the path merged AT T (light prefix + junction edge)
    log_rev_T log P of T's own reverse edge T -> T-1 in LIGHT direction:
              pairing-dependent (the junction fixes T's incoming direction)."""
    arr = cam.arr
    one = M.f(1.0)
    log_nt = M.log(M.f(n_t))
    S = M.f(0.0)
    cs = []
    for rec in cam.scat:
        live = rec.tau >= arr.cut
        if force_c1 or not live:
            c = one
        else:
            c = camis_c(M, rec.log_py, log_pz_T + log_rev_T + (arr.rc_prev - rec.rc), -log_nt - rec.log_keep)
        cs.append(c)
        if form == "horner":
            S = S * rec.a_lin + c * rec.etab_lin
        elif live:
            S = S + c * M.exp(rec.u + arr.log_a)
    c_T = one if force_c1 else camis_c(M, arr.log_py, log_pz_T, -log_nt - arr.log_keep)
    if info is not None:
        info["c_cam"] = cs + [c_T]
    return arr.dvc0 + S, c_T


def _light_side(M, light, n_t, log_k, log_py_base, log_rev_L, form, force_c1, info):
    """dVC'_Lambda, log P(z) of the whole light prefix, and c_Lambda.
    log_py_base  log P(y) of the path merged AT Lambda (camera prefix +
                 junction edge); log_rev_L  Lambda's own reverse edge
                 Lambda -> Lambda-1 in CAMERA direction (pairing-dependent).
    The per-edge clamps need the camera's r, so the light prefix is walked
    here -- this is the O(s) record walk the plan budgets for."""
    origin, scat, arr = light.origin, light.scat, light.arr
    one = M.f(1.0)
    log_nt = M.log(M.f(n_t))
    lpz = [_clampP(M, log_k, origin.log_pa0)]
    for rec in scat:
        lpz.append(lpz[-1] + _clampP(M, log_k, rec.log_pa_fwd))
    lpz.append(lpz[-1] + _clampP(M, log_k, arr.log_pa_fwd))
    Lam = arr.lam
    lpy = [None] * Lam
    acc = log_py_base + log_rev_L
    for lam in range(Lam - 1, 0, -1):
        lpy[lam] = acc
        acc = acc + _clampP(M, log_k, scat[lam - 1].log_pa_rev)
    S = M.f(0.0)
    cs = []
    for rec in scat:
        live = rec.lam >= arr.cut
        if force_c1 or not live:
            c = one
        else:
            c = camis_c(M, lpy[rec.lam], lpz[rec.lam], -log_nt - rec.log_keep)
        cs.append(c)
        if form == "horner":
            S = S * rec.a_lin + c * rec.etab_lin
        elif live:
            S = S + c * M.exp(rec.u + arr.log_a)
    c_L = one if force_c1 else camis_c(M, log_py_base, lpz[Lam], -log_nt - arr.log_keep)
    if info is not None:
        info["c_light"] = cs + [c_L]
    return arr.dvc0 + S, lpz[Lam], c_L


def _uses_camis(camis_on, *flags):
    return camis_on and all(flags)


def eval_merge(cam, light, j, n_t=N_LIGHT, M=F64, camis_on=True, form="log", force_c1=False, info=None):
    """_bdpt_merge_mis_weight: merge the photon `light` at camera vertex cam.arr."""
    arr, larr = cam.arr, light.arr
    if _uses_camis(camis_on, arr.in_class, larr.in_class, light.origin.in_class):
        dvc_l, lpz, _ = _light_side(M, light, n_t, arr.log_k, arr.log_py,
                                    _clampP(M, arr.log_k, M.log(j.cam_dir_w) + larr.log_g_rev),
                                    form, force_c1, info)
        dvc_c, c_T = _camera_side(M, cam, n_t, lpz,
                                  _clampP(M, arr.log_k, M.log(j.cam_rev_w) + arr.log_g_prev),
                                  form, force_c1, info)
    else:
        dvc_l, dvc_c, c_T = larr.dvc, arr.dvc, M.f(1.0)
    eta = c_T * arr.eta                    # c into eta_scale, never into inv_eta_x
    w_light = (larr.dvcm + dvc_l * j.cam_dir_w) / eta
    w_camera = (arr.dvcm + dvc_c * j.cam_rev_w) / eta
    return M.f(1.0) / (w_light + M.f(1.0) + w_camera)


def eval_connect(cam, light, j, n_t=N_LIGHT, M=F64, camis_on=True, form="log", force_c1=False, info=None):
    """_connect between light vertex light.arr (or the emitter, s = 1) and
    camera vertex cam.arr."""
    arr, larr, origin = cam.arr, light.arr, light.origin
    use = _uses_camis(camis_on, arr.in_class, origin.in_class, larr is None or larr.in_class)
    log_k = arr.log_k
    if larr is None:                       # s = 1: SmallVCM's DirectIllumination form
        w_light = j.cam_dir_a / origin.direct_pdf_a
        lpz = _clampP(M, log_k, origin.log_pa0)
    elif use:
        dvc_l, lpz, c_L = _light_side(M, light, n_t, log_k,
                                      arr.log_py + _clampP(M, log_k, M.log(j.cam_dir_a)),
                                      _clampP(M, log_k, M.log(j.light_rev_w) + larr.log_g_rev),
                                      form, force_c1, info)
        w_light = j.cam_dir_a * (c_L * larr.eta + larr.dvcm + dvc_l * j.light_rev_w)
    else:
        w_light = j.cam_dir_a * (larr.eta + larr.dvcm + larr.dvc * j.light_rev_w)
    if use:
        dvc_c, c_T = _camera_side(M, cam, n_t, lpz + _clampP(M, log_k, M.log(j.light_dir_a)),
                                  _clampP(M, log_k, M.log(j.cam_rev_w) + arr.log_g_prev),
                                  form, force_c1, info)
    else:
        dvc_c, c_T = arr.dvc, M.f(1.0)
    w_camera = j.light_dir_a * (c_T * arr.eta + arr.dvcm + dvc_c * j.cam_rev_w)
    return M.f(1.0) / (w_light + M.f(1.0) + w_camera)


def eval_splat(light, j, n_t=N_LIGHT, M=F64, camis_on=True, form="log", force_c1=False, info=None):
    """t = 1: the light vertex light.arr is seen by the lens. P(y) of every
    light vertex starts from the primary edge's P = 1 (log 0)."""
    larr = light.arr
    if _uses_camis(camis_on, larr.in_class, light.origin.in_class):
        dvc_l, _, c_L = _light_side(M, light, n_t, j.log_k, M.f(0.0),
                                    _clampP(M, j.log_k, M.log(j.rev) + larr.log_g_rev),
                                    form, force_c1, info)
    else:
        dvc_l, c_L = larr.dvc, M.f(1.0)
    w_light = (j.cam_pdf_a / M.f(area.N_SPLAT)) * (c_L * larr.eta + larr.dvcm + dvc_l * j.rev)
    return M.f(1.0) / (w_light + M.f(1.0))


def eval_emission_hit(cam, j, n_t=N_LIGHT, M=F64, camis_on=True, form="log", force_c1=False, info=None):
    """s = 0: cam.arr is the ARRIVAL at the emitter (after scattering at
    xs[0]), SmallVCM's GetLightRadiance. The light 'subpath' is z_0 alone:
    P(z) of camera vertex tau = P(z_0) P(z_0 -> xs[0]) prod(reverse edges)."""
    arr = cam.arr
    if _uses_camis(camis_on, arr.in_class):
        log_k = arr.log_k
        dvc_c, _ = _camera_side(M, cam, n_t, _clampP(M, log_k, M.log(j.direct_pdf_a)),
                                _clampP(M, log_k, j.log_p_emit + arr.log_g_prev),
                                form, force_c1, info)
    else:
        dvc_c = arr.dvc
    return M.f(1.0) / (M.f(1.0) + j.direct_pdf_a * arr.dvcm + j.emission_pdf_w * dvc_c)


# ── all strategies of one path, each from its OWN subpath records ──────────

def trace_blind_weights(y0, xs, lens, n_t=N_LIGHT, M=F64, r_scale=1.0, camis_on=True,
                        form="log", force_c1=False, eta_mult=None, log_a_gauge=0.0):
    """Every strategy with nonzero density (the only ones a renderer can
    sample). Each builds records from its own camera subpath and its own light
    subpath, in separate calls: neither side can see the other."""
    n = len(xs)
    dens = strategy_densities_general(y0, xs, lens, [1.0] * n)
    cam_order = xs[::-1]
    ev = dict(n_t=n_t, M=M, camis_on=camis_on, form=form, force_c1=force_c1)

    def cam_recs(vs):
        return trace_camera_records(lens, vs, n_t, M, r_scale, eta_mult, log_a_gauge)

    def light_recs(vs):
        return trace_light_records(y0, vs, n_t, M, eta_mult, log_a_gauge)

    W = {}
    for key, d in dens.items():
        if d <= 0.0:
            continue
        what, idx = key
        if what == "s" and idx == 0:
            cam = cam_view(cam_recs(cam_order + [y0]), n)
            W[key] = eval_emission_hit(cam, junction_emission(y0, xs[0], M), **ev)
        elif what == "s":
            s = idx
            T = n - s
            cam = cam_view(cam_recs(cam_order[:T + 1]), T)
            light = light_view(light_recs(xs[:s - 1]), s - 1)
            lv, lv_pred = (y0, None) if s == 1 else (xs[s - 2], light_pred(y0, xs, s - 2))
            j = junction_connect(lv, lv_pred, xs[s - 1], cam_pred(xs, lens, s - 1), M)
            W[key] = eval_connect(cam, light, j, **ev)
        elif what == "t1":
            light = light_view(light_recs(xs), n)
            W[key] = eval_splat(light, junction_splat(xs[n - 1], light_pred(y0, xs, n - 1), lens, M, r_scale), **ev)
        else:
            k = idx
            T = n - 1 - k
            cam = cam_view(cam_recs(cam_order[:T + 1]), T)
            light = light_view(light_recs(xs[:k + 1]), k + 1)
            W[key] = eval_merge(cam, light, junction_merge(xs[k], cam_pred(xs, lens, k), light_pred(y0, xs, k), M), **ev)
    return W, dens


def weight_errors(W, truth):
    err = max(abs(float(W[k]) - truth[k]) for k in W)
    pou = abs(math.fsum(float(v) for v in W.values()) - 1.0)
    return err, pou


# ── 1. reconstruction on the toy scene of vcm_camis_derivation ─────────────

def toy_paths():
    """camis.scene(): 2 camera bounces lens -> y1 -> y2 -> x; photon a
    (light -> right wall -> floor) and photon b (straight from the light),
    both idealised to land exactly on x."""
    lens, cam, pa, pb = camis.scene()
    y1, y2, xm = cam
    return lens, (pa[0], [pa[1], xm, y2, y1]), (pb[0], [xm, y2, y1])


def section_reconstruction():
    lens, (ya, xsa), (yb, xsb) = toy_paths()
    print("\n=== 1. toy scene: every strategy, trace-blind records vs Eq. 11 truth ===")
    worst = {"truth-vs-area": 0.0, "c-vs-direct_camis": 0.0, "legacy-carries": 0.0,
             "linearity": 0.0, "u_plan": 0.0, "posthoc": 0.0, "trace-blind": 0.0,
             "horner": 0.0, "pou": 0.0, "c1-identity": 0.0}
    for name, y0, xs in (("a", ya, xsa), ("b", yb, xsb)):
        n = len(xs)
        # (i) the generalised truth is the area harness's truth on Lambertian paths
        etas = [N_LIGHT * area.KERNEL] * n
        d0, d1 = strategy_densities_general(y0, xs, lens, etas), area.strategy_densities(y0, xs, lens, etas)
        worst["truth-vs-area"] = max(worst["truth-vs-area"], max(rel(d0[k], d1[k]) for k in d1))
        # (ii) truth_c is vcm_camis_derivation.direct_camis's exact c at every vertex
        tc = truth_c(y0, xs, lens)
        for k in range(n):
            ce = camis.direct_camis(lens, xs[k:][::-1], [y0] + xs[:k + 1], N_LIGHT)[1]
            worst["c-vs-direct_camis"] = max(worst["c-vs-direct_camis"], rel(tc[k], ce))
        # (iii) the record builders' legacy carries are area.*_carries
        crec = trace_camera_records(lens, xs[::-1])
        lrec = trace_light_records(y0, xs)
        for T in range(n):
            a = crec[1][T]
            want = area.camera_carries(xs, lens, etas, n - 1 - T)
            worst["legacy-carries"] = max(worst["legacy-carries"], rel(a.dvcm, want[0]), rel(a.dvc, want[1]) if want[1] else abs(a.dvc))
            la = lrec[2][T]
            want = area.light_carries(y0, xs, etas, T)[0]
            worst["legacy-carries"] = max(worst["legacy-carries"], rel(la.dvcm, want[0]), rel(la.dvc, want[1]))
            # (iv) linearity: dVC == dVC0 + sum eta mu, both from the records
            v = cam_view(crec, T)
            s_log = a.dvc0 + math.fsum(math.exp(r.u + a.log_a) for r in v.scat if r.tau >= a.cut)
            worst["linearity"] = max(worst["linearity"], rel(s_log, a.dvc) if a.dvc else abs(s_log))
            for r in v.scat:
                worst["u_plan"] = max(worst["u_plan"], abs(r.u - r.u_plan))
        truth, _, c = truth_weights(y0, xs, lens)
        # (v) section_partition's post-hoc method: c_k eta_k fed into the carries
        post, _ = trace_blind_weights(y0, xs, lens, camis_on=False, eta_mult={id(xs[k]): c[k] for k in range(n)})
        W, dens = trace_blind_weights(y0, xs, lens)
        Wh, _ = trace_blind_weights(y0, xs, lens, form="horner")
        W1, _ = trace_blind_weights(y0, xs, lens, force_c1=True)
        Wleg, _ = trace_blind_weights(y0, xs, lens, camis_on=False)
        worst["posthoc"] = max(worst["posthoc"], weight_errors(post, truth)[0])
        e, p = weight_errors(W, truth)
        worst["trace-blind"] = max(worst["trace-blind"], e)
        worst["pou"] = max(worst["pou"], p)
        worst["horner"] = max(worst["horner"], weight_errors(Wh, truth)[0])
        worst["c1-identity"] = max(worst["c1-identity"], max(abs(W1[k] - Wleg[k]) for k in W1))
        bal = truth_weights(y0, xs, lens, camis_on=False)[0]
        print(f"\n  path {name}: z0 {'z1 ' if n == 4 else ''}x y2 y1 lens   (c at xs[k]: "
              + ", ".join(f"{q:.4g}" for q in c) + ")")
        print(f"  {'strategy':12s} {'balance':>10s} {'CAMIS truth':>12s} {'post-hoc':>12s} "
              f"{'trace-blind':>12s} {'|err|':>8s}")
        for key in sorted(truth, key=lambda k: (k[0], k[1])):
            print(f"  {str(key):12s} {bal[key]:10.6f} {truth[key]:12.8f} {post[key]:12.8f} "
                  f"{W[key]:12.8f} {abs(W[key] - truth[key]):8.1e}")
        print(f"  {'SUM':12s} {sum(bal.values()):10.6f} {sum(truth.values()):12.8f} "
              f"{sum(post.values()):12.8f} {sum(W.values()):12.8f}")
    check("generalised truth == area.strategy_densities on Lambertian paths", worst["truth-vs-area"] < 1e-14,
          f"{worst['truth-vs-area']:.1e}")
    check("truth_c == vcm_camis_derivation.direct_camis c_exact at every vertex", worst["c-vs-direct_camis"] < 1e-14,
          f"{worst['c-vs-direct_camis']:.1e}")
    check("record builders' legacy dVCM/dVC == area.camera_carries/light_carries", worst["legacy-carries"] < 1e-14,
          f"{worst['legacy-carries']:.1e}")
    check("linearity: dVC == dVC0 + sum exp(u_i + log A_T) at every camera vertex", worst["linearity"] < 1e-13,
          f"{worst['linearity']:.1e}")
    check("stored u == the plan's u = log eta - log pdf_rev - log A_i (pdf_rev > 0)", worst["u_plan"] < 1e-13,
          f"{worst['u_plan']:.1e}")
    check("post-hoc c*eta carries (section_partition's method) == truth", worst["posthoc"] < 1e-13,
          f"{worst['posthoc']:.1e}")
    check("toy: trace-blind evaluators == Eq. 11 truth, every strategy", worst["trace-blind"] < 1e-12,
          f"{worst['trace-blind']:.1e}")
    check("toy: Horner (linear) record form == truth", worst["horner"] < 1e-12, f"{worst['horner']:.1e}")
    check("toy: trace-blind weights sum to 1", worst["pou"] < 1e-12, f"{worst['pou']:.1e}")
    check("toy: records with c forced to 1 == legacy weights (G2's identity check)", worst["c1-identity"] < 1e-13,
          f"{worst['c1-identity']:.1e}")
    return worst


# ── random geometry ────────────────────────────────────────────────────────

def _rv(rng):
    return (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0))


def _facing(rng, p, others, jitter=0.3):
    acc = (0.0, 0.0, 0.0)
    for o in others:
        u = unit(sub(o, p))
        acc = (acc[0] + u[0], acc[1] + u[1], acc[2] + u[2])
    j = _rv(rng)
    return unit((acc[0] + jitter * j[0], acc[1] + jitter * j[1], acc[2] + jitter * j[2]))


def _cos_ok(a, b, lo=0.05):
    w = dir_to(a, b)
    return (a.n is None or abs(dot(a.n, w)) > lo) and (b.n is None or abs(dot(b.n, w)) > lo)


def random_light_chain(rng, target, m, avoid=(), kinds=("lambert",)):
    """A light subpath z0 -> z1..z_m -> target (target is NOT included).
    Returns (z0, [z1..z_m]) or None."""
    pts = [_rv(rng) for _ in range(m)]
    z0p = (rng.uniform(-1, 1), 1.5, rng.uniform(-1, 1))
    allp = [z0p] + pts + [target.pos]
    chk = allp + [a for a in avoid if a != target.pos]
    if min(norm(sub(a, b)) for i, a in enumerate(chk) for b in chk[i + 1:]) < 0.15:
        return None
    zs = []
    for i, p in enumerate(pts):
        v = V(p, _facing(rng, p, [allp[i], allp[i + 2]]))
        v.kind = rng.choice(kinds)
        zs.append(v)
    z0 = V(z0p, _facing(rng, z0p, [allp[1]], 0.3))
    chain = [z0] + zs + [target]
    if dot(z0.n, dir_to(z0, chain[1])) < 0.05:
        return None
    if not all(_cos_ok(a, b) for a, b in zip(chain, chain[1:])):
        return None
    return z0, zs


def random_path(rng, n, kinds=("lambert",), keep_lo=1.0, special=None):
    """y0, xs[0..n-1], lens with every edge non-grazing. special = (kind,
    orientation) puts one 'cut' or 'delta' vertex at a random interior index.
    'cut' orientation 'cam': the camera crosses it, the light cannot (camera
    pdf_rev = 0); 'light': the reverse."""
    while True:
        pts = [_rv(rng) for _ in range(n)]
        lens_p = _rv(rng)
        y0p = (rng.uniform(-1, 1), 1.5, rng.uniform(-1, 1))
        allp = [y0p] + pts + [lens_p]
        if min(norm(sub(a, b)) for i, a in enumerate(allp) for b in allp[i + 1:]) < 0.15:
            continue
        xs = [V(pts[k], _facing(rng, pts[k], [allp[k], allp[k + 2]])) for k in range(n)]
        y0 = V(y0p, _facing(rng, y0p, [pts[0]], 0.3))
        lens = V(lens_p, None)
        chain = [y0] + xs + [lens]
        if dot(y0.n, dir_to(y0, xs[0])) < 0.05 or not all(_cos_ok(a, b) for a, b in zip(chain, chain[1:])):
            continue
        for x in xs:
            x.kind = rng.choice(kinds)
            x.keep = rng.uniform(keep_lo, 1.0)
        if special is not None:
            k = rng.randrange(n)
            what, orient = special
            xs[k].kind = what
            if what == "cut":
                w_l, w_c = dir_to(xs[k], light_pred(y0, xs, k)), dir_to(xs[k], cam_pred(xs, lens, k))
                t = sub(w_l, w_c) if orient == "cam" else sub(w_c, w_l)
                xs[k].t = unit(t)
        return y0, xs, lens


def _cut_exercised(y0, xs, lens, dens):
    """Does any sampled strategy's record set actually contain a pdf_rev = 0?"""
    for k, x in enumerate(xs):
        if kind(x) != "cut":
            continue
        w_l, w_c = dir_to(x, light_pred(y0, xs, k)), dir_to(x, cam_pred(xs, lens, k))
        cam_rev0 = bsdf_pdf(x, w_l, w_c) == 0.0      # camera crossing x has pdf_rev = 0
        light_rev0 = bsdf_pdf(x, w_c, w_l) == 0.0
        return cam_rev0, light_rev0
    return False, False


CATEGORIES = (
    # name, kinds, keep_lo, special
    ("lambert",   ("lambert",),                   1.0, None),
    ("glossy",    ("glossy", "glossy", "lambert"), 0.1, None),
    ("cut (cam)", ("glossy", "lambert"),          0.1, ("cut", "cam")),
    ("cut (lgt)", ("glossy", "lambert"),          0.1, ("cut", "light")),
    ("delta",     ("glossy", "lambert"),          0.1, ("delta", None)),
)


def random_suite(count=200, seed=2027):
    """The same path set for the double and float32 sweeps."""
    rng = random.Random(seed)
    out = []
    for cat, kinds, keep_lo, special in CATEGORIES:
        for _ in range(count):
            n = rng.randint(2, 7)
            y0, xs, lens = random_path(rng, n, kinds, keep_lo, special)
            r_scale = 10.0 ** rng.uniform(-2.0, 2.0)
            n_t = 10.0 ** rng.uniform(0.0, 7.0)
            out.append((cat, y0, xs, lens, r_scale, n_t))
    return out


def section_random(suite):
    print(f"\n=== 3. random paths: lengths 2-7, r x 1e-2..1e2, n_t 1..1e7, keep 0.1..1 (double) ===")
    print(f"  {'category':10s} {'paths':>5s} {'max|w-truth|':>13s} {'horner':>9s} {'max|sum-1|':>11s} "
          f"{'c=1 ident':>10s} {'min c':>9s} {'pr!=pf':>7s}")
    worst_all = {}
    exercised = {"cam_cut": 0, "light_cut": 0, "glossy_asym": 0.0, "clamp_bind": 0, "c_lt_1": 0}
    for cat, *_ in CATEGORIES:
        paths = [p for p in suite if p[0] == cat]
        w_e = w_h = w_p = w_i = 0.0
        min_c = 1.0
        asym = 0.0
        ok_class = True
        for _, y0, xs, lens, rs, nt in paths:
            truth, dens, c = truth_weights(y0, xs, lens, nt, rs)
            W, _ = trace_blind_weights(y0, xs, lens, nt, r_scale=rs)
            Wh, _ = trace_blind_weights(y0, xs, lens, nt, r_scale=rs, form="horner")
            e, p = weight_errors(W, truth)
            w_e, w_p = max(w_e, e), max(w_p, p)
            w_h = max(w_h, weight_errors(Wh, truth)[0])
            if cat == "delta":
                # out of the Class: c == 1, and trace-blind == the legacy balance truth
                bal = truth_weights(y0, xs, lens, nt, rs, camis_on=False)[0]
                w_i = max(w_i, weight_errors(W, bal)[0])
                ok_class &= all(q == 1.0 for q in c)
                # and every strategy's own records say so
                crec = trace_camera_records(lens, xs[::-1] + [y0], nt, r_scale=rs)
                lrec = trace_light_records(y0, xs, nt)
                kd = next(k for k, x in enumerate(xs) if is_delta(x))
                ok_class &= all((not a.in_class) == (T >= len(xs) - 1 - kd) for T, a in enumerate(crec[1]))
                ok_class &= all((not a.in_class) == (a.lam - 1 >= kd) for a in lrec[2])
            else:
                W1, _ = trace_blind_weights(y0, xs, lens, nt, r_scale=rs, force_c1=True)
                Wl, _ = trace_blind_weights(y0, xs, lens, nt, r_scale=rs, camis_on=False)
                w_i = max(w_i, max(abs(W1[k] - Wl[k]) for k in W1))
                min_c = min(min_c, min(c))
                exercised["c_lt_1"] += sum(1 for q in c if q < 1.0)
                r2 = camis_r2(lens, xs[-1], rs)
                cam_e, _, lig_e = truth_edges(y0, xs, lens)
                exercised["clamp_bind"] += sum(1 for q in cam_e[:-1] + lig_e if math.pi * r2 * q >= 1.0)
            if cat.startswith("cut"):
                cr, lr = _cut_exercised(y0, xs, lens, dens)
                exercised["cam_cut"] += cr
                exercised["light_cut"] += lr
            for k, x in enumerate(xs):
                if kind(x) == "glossy":
                    w_l, w_c = dir_to(x, light_pred(y0, xs, k)), dir_to(x, cam_pred(xs, lens, k))
                    pf, pr = bsdf_pdf(x, w_c, w_l), bsdf_pdf(x, w_l, w_c)
                    asym = max(asym, abs(pr / pf - 1.0))
        exercised["glossy_asym"] = max(exercised["glossy_asym"], asym)
        worst_all[cat] = (w_e, w_h, w_p, w_i)
        print(f"  {cat:10s} {len(paths):5d} {w_e:13.1e} {w_h:9.1e} {w_p:11.1e} {w_i:10.1e} "
              f"{min_c:9.2e} {asym:7.2f}")
        check(f"random {cat}: trace-blind == Eq. 11 truth", w_e < 1e-12, f"{w_e:.1e}")
        check(f"random {cat}: Horner record form == truth", w_h < 1e-12, f"{w_h:.1e}")
        check(f"random {cat}: weights sum to 1", w_p < 1e-12, f"{w_p:.1e}")
        if cat == "delta":
            check("random delta: out of Class -> c == 1, weights == legacy balance truth, class bits agree",
                  ok_class and w_i < 1e-12, f"{w_i:.1e}")
        else:
            check(f"random {cat}: c forced to 1 == legacy weights", w_i < 1e-12, f"{w_i:.1e}")
    print(f"  exercised: camera-side pdf_rev = 0 on {exercised['cam_cut']} paths, light-side on "
          f"{exercised['light_cut']}; glossy |pdf_rev/pdf_fwd - 1| up to {exercised['glossy_asym']:.2f}; "
          f"{exercised['clamp_bind']} edges with the Eq. 16 clamp binding; {exercised['c_lt_1']} vertices with c < 1")
    check("random: every degenerate branch actually exercised (both cut sides, clamps, c < 1, glossy asymmetry)",
          exercised["cam_cut"] > 0 and exercised["light_cut"] > 0 and exercised["clamp_bind"] > 0
          and exercised["c_lt_1"] > 0 and exercised["glossy_asym"] > 0.1, str(exercised))
    return worst_all, exercised


# ── 2. pairing reuse: ONE camera record set, many different partners ───────

def _pairing_suite(rng, lens, cam, cam_kinds, light_kinds, label):
    """cam = [y1, y2, x, y3] traced ONCE (y3: the camera keeps walking past x,
    so scatter record 2 exists and must not leak into evaluations at x)."""
    for v, k in zip(cam, cam_kinds):
        v.kind = k
    recs = trace_camera_records(lens, cam)
    snapshot = repr(recs)
    # prefix stability: a walk stopped at T has the same records
    stable = all(repr(trace_camera_records(lens, cam[:T + 1])) == repr((recs[0][:T], recs[1][:T + 1]))
                 for T in range(len(cam)))
    worst, n_eval, n_merge, n_conn = 0.0, 0, 0, 0
    # merges at each camera vertex T = 0..2 with many photons landing exactly on it
    for T in range(3):
        x = cam[T]
        cam_part = cam[:T + 1][::-1]                      # light order: x, ..., y1
        got = 0
        while got < 12:
            ch = random_light_chain(rng, x, rng.randint(0, 3), [lens.pos] + [c.pos for c in cam], light_kinds)
            if ch is None:
                continue
            z0, zs = ch
            xs = zs + cam_part
            if not all(_cos_ok(a, b) for a, b in zip(xs, xs[1:])):
                continue
            k = len(zs)
            truth = truth_weights(z0, xs, lens)[0]
            lrec = trace_light_records(z0, zs + [x])
            lv_pred = zs[-1] if zs else z0
            w = eval_merge(cam_view(recs, T), light_view(lrec, k + 1), junction_merge(x, cam_pred(xs, lens, k), lv_pred))
            worst = max(worst, abs(w - truth[("merge", k)]))
            got += 1
            n_merge += 1
    # connections from each camera vertex T to many light vertices (s >= 1)
    for T in range(3):
        cv = cam[T]
        got = 0
        while got < 12:
            m = rng.randint(0, 3)
            if m == 0:                                    # s = 1: connect to an emitter point
                z0p = (rng.uniform(-1, 1), 1.5, rng.uniform(-1, 1))
                z0 = V(z0p, _facing(rng, z0p, [cv.pos]))
                zs = []
                if dot(z0.n, dir_to(z0, cv)) < 0.05 or not _cos_ok(z0, cv):
                    continue
            else:
                lvp = _rv(rng)
                lv = V(lvp, _facing(rng, lvp, [cv.pos]))  # placeholder normal, fixed below
                ch = random_light_chain(rng, lv, m - 1, [lens.pos] + [c.pos for c in cam], light_kinds)
                if ch is None:
                    continue
                z0, zs = ch
                pred = zs[-1] if zs else z0
                lv.n = _facing(rng, lvp, [pred.pos, cv.pos])
                lv.kind = rng.choice(light_kinds)
                zs = zs + [lv]
            xs = zs + cam[:T + 1][::-1]
            chain = [z0] + xs + [lens]
            if dot(z0.n, dir_to(z0, xs[0])) < 0.05 or not all(_cos_ok(a, b) for a, b in zip(chain, chain[1:])):
                continue
            s = len(zs) + 1
            truth, dens, _ = truth_weights(z0, xs, lens)
            if dens[("s", s)] <= 0.0:
                continue
            lrec = trace_light_records(z0, zs)
            lvx, lv_pred = (z0, None) if s == 1 else (zs[-1], light_pred(z0, xs, s - 2))
            j = junction_connect(lvx, lv_pred, cv, cam_pred(xs, lens, s - 1))
            w = eval_connect(cam_view(recs, T), light_view(lrec, s - 1), j)
            worst = max(worst, abs(w - truth[("s", s)]))
            got += 1
            n_conn += 1
    unchanged = repr(recs) == snapshot
    print(f"  {label:34s} {n_merge:3d} merges + {n_conn:3d} connections against ONE camera record set: "
          f"max |w - truth| {worst:.1e}; records unchanged: {unchanged}; prefix-stable: {stable}")
    return worst, unchanged, stable, n_merge + n_conn


def section_pairing():
    print("\n=== 2. pairing reuse: camera records traced once, evaluated against many partners ===")
    rng = random.Random(1717)
    lens, (ya, xsa), (yb, xsb) = toy_paths()
    _, xm, y2, y1 = xsa                                               # the SAME vertex objects
    y3 = V((0.1, 1.0, 0.6), unit((0.02, -1.0, 0.03)))                 # ceiling, past x
    # the camis toy photons a and b first, explicitly
    recs = trace_camera_records(lens, [y1, y2, xm, y3])
    worst0 = 0.0
    for y0, xs in ((ya, xsa), (yb, xsb)):
        k = xs.index(xm)
        truth = truth_weights(y0, xs, lens)[0]
        lrec = trace_light_records(y0, xs[:k + 1])
        w = eval_merge(cam_view(recs, 2), light_view(lrec, k + 1), junction_merge(xm, y2, light_pred(y0, xs, k)))
        worst0 = max(worst0, abs(w - truth[("merge", k)]))
    print(f"  toy photons a, b merged at x from the SAME camera records: max |w - truth| {worst0:.1e}")
    check("pairing: toy photons a and b from one camera record set == truth", worst0 < 1e-12, f"{worst0:.1e}")
    res = []
    for label, ck, lk in (("Lambertian camera, Lambertian light", ("lambert",) * 4, ("lambert",)),
                          ("glossy camera, mixed light", ("glossy",) * 4, ("glossy", "lambert"))):
        cam4 = [V(v.pos, v.n) for v in (y1, y2, xm, y3)]
        res.append(_pairing_suite(rng, lens, cam4, ck, lk, label))
    worst = max(r[0] for r in res)
    check("pairing: every merge/connect partner exact from one unmodified camera record set",
          worst < 1e-12 and all(r[1] for r in res), f"{worst:.1e} over {sum(r[3] for r in res)} pairings")
    check("pairing: camera records are prefix-stable (walking past T never changes records <= T)",
          all(r[2] for r in res))
    return max(worst, worst0), 2 + sum(r[3] for r in res)


# ── 4. float32 ─────────────────────────────────────────────────────────────

def section_float32(suite):
    print("\n=== 4. float32 emulation (numpy.float32 arithmetic end to end), same paths ===")
    print(f"  {'category':10s} {'form':7s} {'max|sum-1|':>11s} {'max|w-truth|':>13s} {'max|sum-1| legacy':>18s}")
    res = {}
    with np.errstate(all="ignore"):
        for cat, *_ in CATEGORIES:
            paths = [p for p in suite if p[0] == cat]
            for form in ("log", "horner"):
                w_p = w_e = w_l = 0.0
                bad_type = False
                for _, y0, xs, lens, rs, nt in paths:
                    truth = truth_weights(y0, xs, lens, nt, rs)[0]
                    W, _ = trace_blind_weights(y0, xs, lens, nt, M=F32, r_scale=rs, form=form)
                    bad_type |= any(not isinstance(v, np.float32) for v in W.values())
                    e, p = weight_errors(W, truth)
                    if not (math.isfinite(e) and math.isfinite(p)):
                        e = p = math.inf
                    w_e, w_p = max(w_e, e), max(w_p, p)
                    Wl, _ = trace_blind_weights(y0, xs, lens, nt, M=F32, r_scale=rs, camis_on=False)
                    w_l = max(w_l, abs(math.fsum(float(v) for v in Wl.values()) - 1.0))
                res[(cat, form)] = (w_p, w_e, w_l, bad_type)
                print(f"  {cat:10s} {form:7s} {w_p:11.1e} {w_e:13.1e} {w_l:18.1e}")
    worst_log = max(v[0] for k, v in res.items() if k[1] == "log")
    worst_hor = max(v[0] for k, v in res.items() if k[1] == "horner")
    worst_leg = max(v[2] for v in res.values())
    check("float32: every weight stays float32 (no silent promotion)", not any(v[3] for v in res.values()))
    check("float32, plan's log-form records: partition error < 1e-4", worst_log < 1e-4, f"{worst_log:.1e}")
    check("float32, Horner-form records: partition error < 1e-4", worst_hor < 1e-4, f"{worst_hor:.1e}")
    print(f"  worst |sum - 1|: log form {worst_log:.1e}, Horner form {worst_hor:.1e}, "
          f"today's legacy weights {worst_leg:.1e} (the float32 floor this is judged against)")

    # 4b. at the renderer's _BDPT_MAX_VERTS = 10, with the whole scene rescaled
    rng = random.Random(99)
    worst_long = 0.0
    with np.errstate(all="ignore"):
        for scale in (1e-2, 1.0, 1e2):
            for _ in range(60):
                n = rng.randint(8, 10)
                y0, xs, lens = random_path(rng, n, ("glossy", "lambert"), 0.1,
                                           rng.choice([None, ("cut", "cam"), ("cut", "light")]))
                for v in [y0, lens] + xs:
                    v.pos = tuple(scale * q for q in v.pos)
                rs, nt = 10.0 ** rng.uniform(-2.0, 2.0), 10.0 ** rng.uniform(0.0, 7.0)
                truth = truth_weights(y0, xs, lens, nt, rs)[0]
                for form in ("log", "horner"):
                    e, p = weight_errors(trace_blind_weights(y0, xs, lens, nt, M=F32, r_scale=rs, form=form)[0], truth)
                    worst_long = max(worst_long, p if math.isfinite(p) else math.inf)
    print(f"  8-10 vertices, scene scale 1e-2 / 1 / 1e2, both forms: max |sum - 1| {worst_long:.1e}")
    check("float32 at 8-10 vertices and scene scale 1e-2..1e2: partition error < 1e-4", worst_long < 1e-4,
          f"{worst_long:.1e}")

    # 4c. the log form's one float32 weakness: exp(u_i + log A_T) is a difference
    # of two running sums, so its error grows with how far log A has drifted
    # from 0. Emulate drift with the gauge offset (exact in double).
    sub_suite = [q for q in suite if q[0] != "delta"][::4]
    drift = {}
    with np.errstate(all="ignore"):
        for K in (0.0, 100.0, -100.0, 1000.0):
            wd = 0.0
            for _, y0, xs, lens, rs, nt in sub_suite:
                truth = truth_weights(y0, xs, lens, nt, rs)[0]
                p = weight_errors(trace_blind_weights(y0, xs, lens, nt, M=F32, r_scale=rs, log_a_gauge=K)[0], truth)[1]
                wd = max(wd, p if math.isfinite(p) else math.inf)
            drift[K] = wd
    print("  log form vs log A drift K (gauge offset): "
          + ", ".join(f"K={K:+g}: {v:.1e}" for K, v in drift.items())
          + "   -- grows ~linearly in |K|; the Horner form has no such register")
    check("float32 log form: still < 1e-4 with log A drifted by 1000 (grows with |drift|, see 4c)",
          drift[1000.0] < 1e-4 and drift[100.0] > drift[0.0], f"{drift[1000.0]:.1e}")
    return worst_log, worst_hor, worst_leg, worst_long, drift


# ── c formula: the logistic form is Eq. 13 ─────────────────────────────────

def section_c_form():
    rng = random.Random(13)
    worst = 0.0
    cases = [(1.0, 0.3, 10.0), (0.2, 0.0, 10.0), (0.0, 0.4, 1e4), (1e-300, 1e-3, 1e4), (1.0 - 1e-12, 0.5, 1e3)]
    for _ in range(5000):
        cases.append((10.0 ** rng.uniform(-30, 0), 10.0 ** rng.uniform(-30, 0), 10.0 ** rng.uniform(0, 7)))
    for py, pz, nt in cases:
        want = c_from_P(py, pz, nt)
        got = camis_c(F64, F64.log(py), F64.log(pz), -math.log(nt))
        worst = max(worst, rel(got, want))
        if py > 0.0:
            worst = max(worst, rel(camis.renderer_c(math.log(py), pz, nt), want))
    check("logistic c form == Eq. 13 c_from_P == camis.renderer_c (incl. P(z)=0, P(y)=0, P(y)->1)",
          worst < 1e-10, f"{worst:.1e}")
    print(f"\n=== 0. c formula: logistic log-space form vs Eq. 13, {len(cases)} cases: max rel err {worst:.1e} ===")


# ── 5. the pinned table for Tests/unit/test_vcm_camis.mojo ─────────────────

def _c(name, v):
    return f"comptime {name} = Float32({float(v)!r})"


def section_pinned():
    lens, (y0, xs), _ = toy_paths()
    n = len(xs)
    k = 1                                       # merge at x (= xs[1]); camera y1 y2 x, light z0 z1 x
    T = n - 1 - k
    crec = trace_camera_records(lens, xs[::-1][:T + 1])
    lrec = trace_light_records(y0, xs[:k + 1])
    cam, light = cam_view(crec, T), light_view(lrec, k + 1)
    j = junction_merge(xs[k], cam_pred(xs, lens, k), light_pred(y0, xs, k))
    info = {}
    w = eval_merge(cam, light, j, info=info)
    w_leg = eval_merge(cam, light, j, camis_on=False)
    truth, _, c = truth_weights(y0, xs, lens)
    lines = ["# vcm_camis_hybrid_derivation.py pinned table -- toy path a of",
             "# vcm_camis_derivation.scene(): z0 -> z1 -> x -> y2 -> y1 -> lens, merge at x.",
             f"# n_t = {N_LIGHT:g}, R_MERGE = {area.R_MERGE}, eta = n_t pi R^2, keep = 1, CAMIS r = |y1 - lens| tan(1 deg)",
             _c("N_T", N_LIGHT), _c("ETA", N_LIGHT * area.KERNEL),
             _c("CAMIS_R2", cam.arr.r2), _c("LOG_PI_R2", cam.arr.log_k)]
    lines.append("# camera per-vertex records {u, log_keep, log_py, rc} (+ Horner a, eta*b), tau = 0 (y1), 1 (y2)")
    for r in cam.scat:
        t = r.tau
        lines += [_c(f"CAM{t}_U", r.u), _c(f"CAM{t}_LOG_KEEP", r.log_keep), _c(f"CAM{t}_LOG_PY", r.log_py),
                  _c(f"CAM{t}_RC", r.rc), _c(f"CAM{t}_A", r.a_lin), _c(f"CAM{t}_ETAB", r.etab_lin)]
    a = cam.arr
    lines.append(f"# camera running registers on arrival at x (tau = {a.tau}); cut = {a.cut}")
    lines += [_c("CAM_DVCM", a.dvcm), _c("CAM_DVC_LEGACY", a.dvc), _c("CAM_DVC0", a.dvc0),
              _c("CAM_LOG_A", a.log_a), _c("CAM_LOG_PY", a.log_py), _c("CAM_RC_PREV", a.rc_prev),
              _c("CAM_LOG_G_PREV", a.log_g_prev), _c("CAM_ETA", a.eta), _c("CAM_LOG_KEEP", a.log_keep)]
    lines.append("# light: origin, per-vertex record lam = 1 (z1), arrival at x (lam = 2)")
    lines.append(_c("LIGHT_LOG_PA0", light.origin.log_pa0))
    for r in light.scat:
        t = r.lam
        lines += [_c(f"LIGHT{t}_U", r.u), _c(f"LIGHT{t}_LOG_KEEP", r.log_keep), _c(f"LIGHT{t}_LOG_PA_FWD", r.log_pa_fwd),
                  _c(f"LIGHT{t}_LOG_PA_REV", r.log_pa_rev), _c(f"LIGHT{t}_A", r.a_lin), _c(f"LIGHT{t}_ETAB", r.etab_lin)]
    la = light.arr
    lines.append(f"# light arrival registers; cut = {la.cut}")
    lines += [_c("LIGHT_DVCM", la.dvcm), _c("LIGHT_DVC_LEGACY", la.dvc), _c("LIGHT_DVC0", la.dvc0),
              _c("LIGHT_LOG_A", la.log_a), _c("LIGHT_LOG_PA_FWD", la.log_pa_fwd), _c("LIGHT_LOG_G_REV", la.log_g_rev),
              _c("LIGHT_ETA", la.eta)]
    lines.append("# junction (evaluated at the merge site)")
    lines += [_c("CAM_DIR_W", j.cam_dir_w), _c("CAM_REV_W", j.cam_rev_w)]
    lines.append("# expected: c per vertex (camera y1, y2, x; light z1), merge weight CAMIS / legacy")
    lines += [_c(f"EXPECT_C_CAM{i}", v) for i, v in enumerate(info["c_cam"])]
    lines += [_c(f"EXPECT_C_LIGHT{i + 1}", v) for i, v in enumerate(info["c_light"][:-1])]
    lines += [_c("EXPECT_MERGE_W_CAMIS", w), _c("EXPECT_MERGE_W_LEGACY", w_leg)]
    lines.append("# all strategies of this path, Eq. 11 truth (sum = 1):")
    for key in sorted(truth, key=lambda q: (q[0], q[1])):
        lines.append(_c(f"EXPECT_W_{key[0].upper()}_{key[1]}", truth[key]))
    print("\n=== 5. pinned table for Tests/unit/test_vcm_camis.mojo (float64 values, test in float32) ===")
    for ln in lines:
        print("  " + ln)
    ok = abs(w - truth[("merge", k)]) < 1e-12
    ok &= all(abs(info["c_cam"][i] - c[n - 1 - i]) < 1e-12 for i in range(T + 1))
    ok &= all(abs(info["c_light"][i] - c[i]) < 1e-12 for i in range(k))
    check("pinned table: merge weight and every per-vertex c == truth", ok)


if __name__ == "__main__":
    section_c_form()
    rec = section_reconstruction()
    w_pair, n_pair = section_pairing()
    suite = random_suite()
    w_rand, ex = section_random(suite)
    f32_log, f32_hor, f32_leg, f32_long, drift = section_float32(suite)
    section_pinned()
    print("\n=== checks ===")
    n_fail = 0
    for name, ok, detail in CHECKS:
        n_fail += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    worst_rand = max(max(v[0], v[2]) for v in w_rand.values())
    print(f"""
WHAT THIS SETTLES
  * The class-gated exact hybrid IS computable trace-blind. Every strategy of
    every path above built its camera records from its camera subpath alone
    and its light records from its light subpath alone; the evaluators saw
    only those plus junction data, and reproduced Eq. 11 to
    {max(rec['trace-blind'], worst_rand):.1e} with weights summing to 1 within {max(rec['pou'], worst_rand):.1e}
    (toy scene, and {len(suite)} random paths of 2-7 vertices incl. glossy,
    pdf_rev = 0 on either side, thinning keep(x), and delta paths).
  * Pairing reuse: {n_pair} merges/connections evaluated against camera record sets
    traced ONCE (max err {w_pair:.1e}); the records were bit-identical afterwards
    and prefix-stable: they never depend on the pairing.
  * Record layout that works (per vertex / running):
      camera  {{u, log keep, log P(y), RC}}  +  {{dVCM, dVC0, dVC, log A, cut,
              log g_prev, r^2}}; the record at T is NOT read by evaluations at T.
      light   {{u, log keep, log p_A fwd, log p_A rev}}  +  {{dVCM, dVC0, dVC,
              log A, cut, log p_A fwd, log g_rev}}; clamps applied at eval time.
    Evaluation: c_i = clamp(logistic(log Pz_i - log Py_i), 1/(n_t keep_i), 1),
      camera tau<T: log Pz = light prefix + junction edge + P(T->T-1) + RC_(T-1) - RC_tau
      light lam<L:  log Py = camera log P(y)_T + junction edge + P(L->L-1) + suffix
      dVC' = dVC0 + sum c_i exp(u_i + log A_T), live terms tau >= cut only.
  * Float32: partition error {f32_log:.1e} (log-form u) vs {f32_hor:.1e} (Horner form);
    today's legacy weights sit at {f32_leg:.1e} on the same paths; {f32_long:.1e} at
    8-10 vertices across scene scales 1e-2..1e2. The log form degrades with
    log A drift ({drift[100.0]:.1e} at |drift| 100, {drift[1000.0]:.1e} at 1000) --
    harmless in range, but the Horner form (store A and eta*B, S = S*A + c*eta*B)
    is immune, needs no cut special case, and costs one float per record more.
  * Merging at the primary hit always has c = 1 (P(y) = 1 by Eq. 17's
    convention); c departs from 1 where the camera prefix has MORE unclamped
    edges than the light suffix has factors (toy path a: 5e-5 at z1, 0.83 at x).
""")
    sys.exit(1 if n_fail else 0)
