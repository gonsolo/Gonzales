#!/usr/bin/env python3
"""Numerical check of the dVCM/dVC recursion extended to VOLUME vertices.

Ground truth: enumerate every BDPT connection strategy's full path pdf
directly, in the mixed area/volume measure, and form the balance-heuristic
weight from those pdfs.  Test: reproduce the same weights from the local
dVCM/dVC recursion carried along each subpath.  If the recursion is right the
two agree for every strategy and the weights sum to 1.

Conventions match bdpt.mojo:
  surface vertex:  directional pdf cos/pi (Lambertian), measure = area
  volume  vertex:  directional pdf 1/(4pi) (isotropic),  measure = volume
  edge a->b:       p_measure(b) = p_dir(a, w_ab) * Gb(a,b) * FF(b, d)
                     Gb = cos_b/d^2   (b surface)
                        =      1/d^2   (b volume)     <-- no cosine, the crux
                     FF = sigma_t*exp(-sigma_t*d)      (b volume)
                        =          exp(-sigma_t*d)     (b surface, survival)
"""
import itertools, math, random

INV_FOUR_PI = 1.0 / (4.0 * math.pi)


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    return (a[0] / n, a[1] / n, a[2] / n)


class V:
    """A path vertex.  normal=None marks a volume-scattering vertex."""

    def __init__(self, pos, normal):
        self.pos = pos
        self.n = normal

    @property
    def is_surface(self):
        return self.n is not None


def p_dir(v, w):
    """Directional pdf of scattering from v into unit direction w."""
    if v.is_surface:
        return max(0.0, abs(dot(v.n, w))) / math.pi
    return INV_FOUR_PI


def geom_to_measure(b, d, wab):
    """Solid-angle -> (area | volume) Jacobian at the RECEIVING vertex b."""
    if b.is_surface:
        return abs(dot(b.n, wab)) / (d * d)
    return 1.0 / (d * d)          # volume: no cosine


def free_flight(b, d, sigma_t):
    """Distance-sampling factor for landing on b at distance d."""
    if b.is_surface:
        return math.exp(-sigma_t * d)          # survived to the surface
    return sigma_t * math.exp(-sigma_t * d)    # collided in the medium


def edge_pdf(a, b, sigma_t):
    """Density of sampling b from a, in b's own measure."""
    dvec = sub(b.pos, a.pos)
    d = norm(dvec)
    w = unit(dvec)
    return p_dir(a, w) * geom_to_measure(b, d, w) * free_flight(b, d, sigma_t)


def path_pdf(xs, s, sigma_t, p_light_area, p_cam_area):
    """Full path pdf under the strategy with s light-subpath vertices.

    xs[0] is on the light, xs[-1] is on the camera.  The connecting edge is
    xs[s-1] -- xs[s]; it is formed deterministically and so contributes no
    density.  s == 0 means the whole path is camera-sampled and the light
    vertex is hit by chance; s == n+1 means the whole path is light-sampled.
    """
    n = len(xs) - 1
    p = 1.0
    if s >= 1:
        p *= p_light_area
        for i in range(1, s):
            p *= edge_pdf(xs[i - 1], xs[i], sigma_t)
    t = n + 1 - s
    if t >= 1:
        p *= p_cam_area
        for i in range(n - 1, s - 1, -1):
            p *= edge_pdf(xs[i + 1], xs[i], sigma_t)
    return p


def brute_force_weights(xs, sigma_t, p_light_area, p_cam_area):
    n = len(xs) - 1
    ps = [path_pdf(xs, s, sigma_t, p_light_area, p_cam_area) for s in range(n + 2)]
    tot = sum(ps)
    return [p / tot for p in ps], ps


# ---------------------------------------------------------------- recursion


def light_carries(xs, sigma_t, p_light_area, upto):
    """dVCM/dVC carried along the light subpath, up to vertex index `upto`.

    Mirrors bdpt.mojo: on arrival  dVCM *= d^2 ; then (surface only) all
    carries /= cos_fix.  On scatter  dVC = (cosOut/pdfDir)*(dVC*pdfRev + dVCM)
    with cosOut omitted at a volume vertex, and dVCM = 1/pdfDir.
    Free-flight factors enter exactly where the ground truth puts them.
    """
    dVCM = 1.0 / p_light_area
    dVC = 0.0
    for i in range(0, upto):
        a, b = xs[i], xs[i + 1]
        dvec = sub(b.pos, a.pos)
        d = norm(dvec)
        w = unit(dvec)
        # --- scatter at a (produces direction w) ---
        pdf_dir = p_dir(a, w)
        if i == 0:
            pdf_rev = 0.0            # light origin: no incoming direction
        else:
            wprev = unit(sub(xs[i - 1].pos, a.pos))
            pdf_rev = p_dir(a, wprev)
        cos_out = abs(dot(a.n, w)) if a.is_surface else 1.0
        dVC = (cos_out / pdf_dir) * (dVC * pdf_rev + dVCM)
        dVCM = 1.0 / pdf_dir
        # --- arrival at b ---
        ff_b = free_flight(b, d, sigma_t)
        ff_a = free_flight(a, d, sigma_t)
        # dVCM must become 1/P(a->b), so it takes 1/FF_b.  dVC's second-order
        # term carries G and FF of the DEPARTING vertex over those of the
        # arriving one, so it takes FF_a/FF_b -- for a homogeneous medium the
        # exponentials cancel and only sigma_t survives, and only when the two
        # endpoints differ in kind (volume vs surface).
        dVCM *= d * d / ff_b
        dVC *= ff_a / ff_b
        if b.is_surface:
            cos_fix = abs(dot(b.n, w))
            dVCM /= cos_fix
            dVC /= cos_fix
    return dVCM, dVC


def camera_carries(xs, sigma_t, p_cam_area, upto):
    """Same recursion walked from the camera end (index n downwards)."""
    n = len(xs) - 1
    dVCM = 1.0 / p_cam_area
    dVC = 0.0
    for i in range(n, upto, -1):
        a, b = xs[i], xs[i - 1]
        dvec = sub(b.pos, a.pos)
        d = norm(dvec)
        w = unit(dvec)
        pdf_dir = p_dir(a, w)
        if i == n:
            pdf_rev = 0.0
        else:
            wprev = unit(sub(xs[i + 1].pos, a.pos))
            pdf_rev = p_dir(a, wprev)
        cos_out = abs(dot(a.n, w)) if a.is_surface else 1.0
        dVC = (cos_out / pdf_dir) * (dVC * pdf_rev + dVCM)
        dVCM = 1.0 / pdf_dir
        ff_b = free_flight(b, d, sigma_t)
        ff_a = free_flight(a, d, sigma_t)
        dVCM *= d * d / ff_b
        dVC *= ff_a / ff_b
        if b.is_surface:
            cos_fix = abs(dot(b.n, w))
            dVCM /= cos_fix
            dVC /= cos_fix
    return dVCM, dVC


def recursion_weight(xs, s, sigma_t, p_light_area, p_cam_area):
    """MIS weight for strategy s from the local carries, as _connect does."""
    n = len(xs) - 1
    if s == 0 or s == n + 1:
        return None                      # endpoint strategies: not tested here
    lv, cv = xs[s - 1], xs[s]
    dvec = sub(cv.pos, lv.pos)
    d = norm(dvec)
    w_l2c = unit(dvec)
    w_c2l = (-w_l2c[0], -w_l2c[1], -w_l2c[2])

    lv_dVCM, lv_dVC = light_carries(xs, sigma_t, p_light_area, s - 1)
    cv_dVCM, cv_dVC = camera_carries(xs, sigma_t, p_cam_area, s)

    cam_dir_pdf_w = p_dir(cv, w_c2l)
    lig_dir_pdf_w = p_dir(lv, w_l2c)
    if s - 1 == 0:
        lig_rev_pdf_w = 0.0
    else:
        lig_rev_pdf_w = p_dir(lv, unit(sub(xs[s - 2].pos, lv.pos)))
    if s == n:
        cam_rev_pdf_w = 0.0
    else:
        cam_rev_pdf_w = p_dir(cv, unit(sub(xs[s + 1].pos, cv.pos)))

    # convert each side's directional pdf into the OTHER endpoint's measure
    cam_dir_pdf_a = cam_dir_pdf_w * geom_to_measure(lv, d, w_c2l) * free_flight(lv, d, sigma_t)
    lig_dir_pdf_a = lig_dir_pdf_w * geom_to_measure(cv, d, w_l2c) * free_flight(cv, d, sigma_t)

    w_light = cam_dir_pdf_a * (lv_dVCM + lv_dVC * lig_rev_pdf_w)
    w_camera = lig_dir_pdf_a * (cv_dVCM + cv_dVC * cam_rev_pdf_w)
    return 1.0 / (w_light + 1.0 + w_camera)


def make_path(kinds, rng):
    xs = []
    for i, k in enumerate(kinds):
        pos = (rng.uniform(-2, 2), rng.uniform(-2, 2), float(i) * 1.7 + rng.uniform(0.1, 0.4))
        if k == "S":
            nrm = unit((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.3, 1)))
        else:
            nrm = None
        xs.append(V(pos, nrm))
    return xs


def run(kinds, sigma_t, rng, verbose=True):
    xs = make_path(kinds, rng)
    p_light_area, p_cam_area = 0.37, 0.53
    w_bf, ps = brute_force_weights(xs, sigma_t, p_light_area, p_cam_area)
    n = len(xs) - 1
    worst = 0.0
    rows = []
    for s in range(1, n + 1):
        w_rec = recursion_weight(xs, s, sigma_t, p_light_area, p_cam_area)
        err = abs(w_rec - w_bf[s]) / max(w_bf[s], 1e-30)
        worst = max(worst, err)
        rows.append((s, w_bf[s], w_rec, err))
    if verbose:
        print(f"  path {''.join(kinds):12s} sigma_t={sigma_t:<5g} sum(w)={sum(w_bf):.12f}")
        for s, a, b, e in rows:
            print(f"    s={s}  brute={a:.10f}  recursion={b:.10f}  relerr={e:.2e}")
    return worst


def main():
    rng = random.Random(20260910)
    cases = [
        ("SSSS", 0.0), ("SSSS", 0.6),
        ("SVSS", 0.6), ("SSVS", 0.6),
        ("SVVS", 0.6), ("SVVS", 2.5),
        ("SVSVS", 0.9), ("SVVVS", 1.4),
        ("SSVVSS", 0.8),
    ]
    print("=== dVCM/dVC recursion vs brute-force path pdfs (S=surface, V=volume) ===")
    worst_all = 0.0
    for kinds, st in cases:
        w = run(list(kinds), st, rng)
        worst_all = max(worst_all, w)
    print(f"\nworst relative error across all cases: {worst_all:.3e}")

    print("\n=== anti-vacuity: corrupt the volume Jacobian (add a bogus cosine) ===")
    global geom_to_measure
    good = geom_to_measure

    def bad(b, d, wab):
        if b.is_surface:
            return abs(dot(b.n, wab)) / (d * d)
        return 0.5 / (d * d)      # wrong constant for volume vertices

    geom_to_measure = bad
    rng2 = random.Random(20260910)
    worst_bad = 0.0
    for kinds, st in cases:
        if "V" not in kinds:
            rng2.random()
            continue
        worst_bad = max(worst_bad, run(list(kinds), st, rng2, verbose=False))
    geom_to_measure = good
    print(f"worst relative error with corrupted volume Jacobian: {worst_bad:.3e}")
    print("(must be LARGE for volume paths -- otherwise the test proves nothing)")


if __name__ == "__main__":
    main()
