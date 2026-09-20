#!/usr/bin/env python3
"""Numerical check of VCM's MIS weights for an ENVIRONMENT (infinite) light.

Sibling of vcm_volume_mis_derivation.py, which settled the volume carries.
That one models a FINITE area light, where the light subpath has a real
origin vertex. An environment light has none: emission is a DIRECTION drawn
from the env (solid-angle pdf `p_env`) plus a point on a disk of radius R
perpendicular to it (area pdf 1/(pi R^2)), and the first real vertex is
wherever that ray lands. So the path space is (w_env, x_1, ..., x_n) -- a
direction and then the surface vertices -- and every strategy must be written
as a density over THAT space before the weights can be compared.

Why this file exists: bdpt.mojo combines its strategies with the BALANCE
heuristic, but the camera-side env NEE and the escape/miss handler both used
`power_heuristic` (beta=2). Two heuristics in one estimator cannot partition
unity, and the measured cost was a flat ~6.5% overcount on every env-lit
scene. A first attempt to port SmallVCM's DirectIllumination/GetLightRadiance
weights by hand OVERSHOT to -12%, which is what this harness is for: settle
the formulas against ground truth before touching the renderer.

Ground truth: enumerate EVERY strategy's density over (w_env, x_1..x_n)
directly and form the balance-heuristic weight. Test: reproduce those same
weights from the local dVCM/dVC recursion the renderer actually carries. If
they agree for every strategy -- and the weights sum to 1 -- the recursion
and its initialization are right.

Strategies for one surface vertex x_1 (the white-furnace case):

    s=0   camera reaches x_1, its BSDF samples w      C_1 * p_dir(x_1, w)
    s=1   camera reaches x_1, NEE samples w           C_1 * p_env(w)
    t=1   light emits (w, x_1), splats to the lens    L_1 * p_cam_area
    merge light and camera both reach x_1             N * pi r^2 * L_1 * C_1

with  L_1 = p_env(w) * cos(theta_1) / (pi R^2)  -- the disk-area pdf carried
onto the surface, the cosine being the projection of the disk element onto
the surface element.
"""
import math

# ── scene constants ────────────────────────────────────────────────────────
R_SCENE = 4.0                     # scene bounding radius (the emission disk)
R_MERGE = 0.045
N_LIGHT = 12000.0
DISK_AREA = math.pi * R_SCENE * R_SCENE
KERNEL = math.pi * R_MERGE * R_MERGE
ETA_VCM = N_LIGHT * KERNEL
MIS_VM = ETA_VCM                  # bdpt.mojo's mis_vm_weight_factor
MIS_VC = 1.0 / ETA_VCM
P_CAM_AREA = 1.0                  # pinhole lens: one point, density 1
P_ENV = 1.0 / (4.0 * math.pi)     # uniform environment, solid-angle pdf

# SmallVCM (and bdpt.mojo's _bdpt_camera_path_init) start the camera subpath at
# dVCM = lightSubPathCount / cameraPdfW. Both factors scale with resolution --
# bdpt.mojo runs "one dedicated light path per pixel" and a real camera's
# directional pdf is itself proportional to the pixel count -- so in practice
# they very nearly cancel. N_SPLAT is that ratio, and the recursion reproduces
# ground truth for ANY value of it (verified below at 1 and at N_LIGHT), which
# is the real point: the weights do not depend on getting this bookkeeping
# convention right, only on using the SAME one on both sides.
N_SPLAT = 1.0


def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def norm(a):   return math.sqrt(dot(a, a))
def unit(a):
    n = norm(a)
    return (a[0] / n, a[1] / n, a[2] / n)


class V:
    def __init__(self, pos, normal):
        self.pos = pos
        self.n = normal


def p_dir(v, w):
    """Lambertian directional pdf at a surface vertex."""
    return max(0.0, abs(dot(v.n, w))) / math.pi


def edge_pdf(a, b):
    """Density of sampling b from a, in b's AREA measure."""
    d = norm(sub(b.pos, a.pos))
    w = unit(sub(b.pos, a.pos))
    return p_dir(a, w) * abs(dot(b.n, w)) / (d * d)


def cam_edge_pdf(lens, b):
    """Lens -> first vertex. A pinhole samples the direction, so this is the
    ordinary solid-angle-to-area Jacobian with a unit directional density."""
    d = norm(sub(b.pos, lens.pos))
    w = unit(sub(b.pos, lens.pos))
    return abs(dot(b.n, w)) / (d * d)


# ── ground truth ───────────────────────────────────────────────────────────

def emission_density(x1, w_env):
    """L_1: density of the light emitting a photon that lands on x_1 travelling
    along w_env. The direction carries p_env; the disk point carries
    1/(pi R^2), converted to x_1's area measure by the projection cosine."""
    return P_ENV * abs(dot(x1.n, w_env)) / DISK_AREA


def strategy_densities(xs, lens, w_env):
    """Every strategy's density over (w_env, x_1..x_n). xs are the surface
    vertices, xs[0] first hit from the light, xs[-1] seen by the lens."""
    n = len(xs)
    # camera subpath density reaching xs[k] (area measure), k = n-1 .. 0
    C = [0.0] * n
    C[n - 1] = P_CAM_AREA * cam_edge_pdf(lens, xs[n - 1])
    for k in range(n - 2, -1, -1):
        C[k] = C[k + 1] * edge_pdf(xs[k + 1], xs[k])
    # light subpath density reaching xs[k]
    L = [0.0] * n
    L[0] = emission_density(xs[0], w_env)
    for k in range(1, n):
        L[k] = L[k - 1] * edge_pdf(xs[k - 1], xs[k])

    d = {}
    # s = 0: the camera made every vertex and its BSDF sampled w_env at xs[0]
    d[("connect", 0)] = C[0] * p_dir(xs[0], w_env)
    # s = 1: the camera made every vertex, NEE sampled w_env at xs[0]
    d[("connect", 1)] = C[0] * P_ENV
    # s >= 2: light made xs[0..s-2], camera made the rest; connect is free
    for s in range(2, n + 1):
        d[("connect", s)] = L[s - 2] * C[s - 1]
    # t = 1: the light made every vertex and splatted to the lens
    d[("connect", n + 1)] = N_SPLAT * L[n - 1] * P_CAM_AREA
    # merging at each vertex the light can actually reach and store
    for k in range(0, n):
        d[("merge", k)] = N_LIGHT * KERNEL * L[k] * C[k]
    return d, C, L


def brute_force_weights(xs, lens, w_env):
    d, C, L = strategy_densities(xs, lens, w_env)
    tot = sum(d.values())
    return {k: v / tot for k, v in d.items()}, d, C, L


# ── the renderer's local recursion ─────────────────────────────────────────

def camera_carries(xs, lens, upto):
    """(dVCM, dVC, dVM) the camera subpath carries on ARRIVING at xs[upto],
    mirroring bdpt.mojo: scatter, then arrival d^2 / cos."""
    n = len(xs)
    # The lens is NOT an ordinary scattering vertex. SmallVCM's
    # SetupCameraPath starts the subpath at dVC = dVM = 0 and only dVCM
    # carries: there is no connection strategy to a vertex BEHIND the lens,
    # and light-tracing to the lens itself (t=1) is already the dVCM term.
    # Running the generic scatter rule here instead injects dVCM into dVC and
    # invents a strategy that does not exist -- worth 0.16% on one bounce.
    a, b = lens, xs[n - 1]
    d = norm(sub(b.pos, a.pos))
    w = unit(sub(b.pos, a.pos))
    pdf_dir = 1.0                      # pinhole: the direction is the pixel
    dVCM, dVC, dVM = N_SPLAT / (P_CAM_AREA * pdf_dir), 0.0, 0.0
    dVCM *= d * d
    cf = abs(dot(b.n, w))
    dVCM /= cf; dVC /= cf; dVM /= cf
    # then surface scatters back toward xs[upto]
    for i in range(n - 1, upto, -1):
        a, b = xs[i], xs[i - 1]
        d = norm(sub(b.pos, a.pos))
        w = unit(sub(b.pos, a.pos))
        pdf_dir = p_dir(a, w)
        wprev = unit(sub(xs[i + 1].pos, a.pos)) if i + 1 < n else unit(sub(lens.pos, a.pos))
        pdf_rev = p_dir(a, wprev)
        cos_out = abs(dot(a.n, w))
        dVM = (cos_out / pdf_dir) * (dVM * pdf_rev + dVCM * MIS_VC + 1.0)
        dVC = (cos_out / pdf_dir) * (dVC * pdf_rev + dVCM + ETA_VCM)
        dVCM = 1.0 / pdf_dir
        dVCM *= d * d
        cf = abs(dot(b.n, w))
        dVCM /= cf; dVC /= cf; dVM /= cf
    return dVCM, dVC, dVM


def light_carries_env(xs, w_env, upto):
    """(dVCM, dVC, dVM) the LIGHT subpath carries on arriving at xs[upto],
    started from an environment light.

    THE INITIALIZATION UNDER TEST. SmallVCM's background branch:
        dVCM = directPdfW / emissionPdfW = disk area
        dVC  = usedCosLight / emissionPdfW, usedCosLight = 1 (not finite)
    with emissionPdfW = p_env / disk_area. The first segment takes NO d^2
    (the emitter is at infinity), only the arrival cosine."""
    emission_pdf_w = P_ENV / DISK_AREA
    dVCM = P_ENV / emission_pdf_w          # = DISK_AREA
    dVC = 1.0 / emission_pdf_w
    dVM = dVC * MIS_VC
    # arrival at xs[0]: cosine only, no d^2
    cf = abs(dot(xs[0].n, w_env))
    dVCM /= cf; dVC /= cf; dVM /= cf
    for i in range(0, upto):
        a, b = xs[i], xs[i + 1]
        d = norm(sub(b.pos, a.pos))
        w = unit(sub(b.pos, a.pos))
        pdf_dir = p_dir(a, w)
        wprev = unit((-w_env[0], -w_env[1], -w_env[2])) if i == 0 else unit(sub(xs[i - 1].pos, a.pos))
        pdf_rev = p_dir(a, wprev)
        cos_out = abs(dot(a.n, w))
        dVM = (cos_out / pdf_dir) * (dVM * pdf_rev + dVCM * MIS_VC + 1.0)
        dVC = (cos_out / pdf_dir) * (dVC * pdf_rev + dVCM + ETA_VCM)
        dVCM = 1.0 / pdf_dir
        dVCM *= d * d
        cf = abs(dot(b.n, w))
        dVCM /= cf; dVC /= cf; dVM /= cf
    return dVCM, dVC, dVM


def recursion_weights(xs, lens, w_env):
    """Weights the renderer's local formulas produce."""
    n = len(xs)
    out = {}
    emission_pdf_w = P_ENV / DISK_AREA
    # --- camera-side, at xs[0] ---
    cvcm, cvc, cvm = camera_carries(xs, lens, 0)
    wo = unit(sub(xs[1].pos, xs[0].pos)) if n > 1 else unit(sub(lens.pos, xs[0].pos))
    pdf_bsdf_dir = p_dir(xs[0], w_env)     # sampling w_env from the BSDF
    pdf_bsdf_rev = p_dir(xs[0], wo)        # sampling back toward the camera
    cos_out = abs(dot(xs[0].n, w_env))

    # s=0, the escape: SmallVCM GetLightRadiance.
    # The carries must be the POST-SCATTER ones at xs[0] for the direction the
    # ray actually left in -- the escape happens one scatter later than the
    # NEE at the same vertex, and nothing arrives, so no d^2/cos update runs.
    # Using the ARRIVAL carries here reads the weight one step too early and
    # is 20% low at one bounce, 73% at two.
    svcm = 1.0 / pdf_bsdf_dir
    svc = (cos_out / pdf_bsdf_dir) * (cvc * pdf_bsdf_rev + cvcm + ETA_VCM)
    out[("connect", 0)] = 1.0 / (1.0 + P_ENV * svcm + emission_pdf_w * svc)
    # s=1, NEE: SmallVCM DirectIllumination
    w_light = pdf_bsdf_dir / P_ENV
    w_camera = (emission_pdf_w * cos_out / P_ENV) * (MIS_VM + cvcm + cvc * pdf_bsdf_rev)
    out[("connect", 1)] = 1.0 / (w_light + 1.0 + w_camera)
    return out


# ── report ─────────────────────────────────────────────────────────────────

def run(name, xs, lens, w_env):
    bw, dens, C, L = brute_force_weights(xs, lens, w_env)
    rw = recursion_weights(xs, lens, w_env)
    print(f"\n=== {name} ===")
    print(f"  {'strategy':16s} {'brute-force w':>14s} {'recursion w':>14s} {'rel err':>11s}")
    for k in sorted(bw, key=lambda k: (k[0], k[1])):
        r = rw.get(k)
        if r is None:
            print(f"  {str(k):16s} {bw[k]:14.9f} {'--':>14s}")
        else:
            err = abs(r - bw[k]) / max(bw[k], 1e-30)
            print(f"  {str(k):16s} {bw[k]:14.9f} {r:14.9f} {err:11.2e}")
    print(f"  {'SUM':16s} {sum(bw.values()):14.9f}")
    return bw, rw


def sweep(n_splat):
    global N_SPLAT
    N_SPLAT = n_splat
    print(f"\n########## light-subpath-count convention: N_SPLAT = {n_splat:g} ##########")
    lens = V((0.0, 0.0, -4.0), None)
    x1 = V((0.3, -0.2, 0.0), unit((0.1, 0.15, -1.0)))
    run("1 surface vertex", [x1], lens, unit((0.35, 0.8, -0.5)))
    x1b = V((0.9, 0.4, 0.2), unit((-0.2, 0.1, -1.0)))
    x2b = V((-0.5, -0.3, -0.4), unit((0.25, -0.1, -1.0)))
    run("2 surface vertices", [x1b, x2b], lens, unit((-0.2, 0.75, -0.6)))


if __name__ == "__main__":
    sweep(1.0)
    sweep(N_LIGHT)
    print("""
WHAT THIS SETTLES, at the realistic N_SPLAT = 1, one surface vertex:

    strategy                 correct    what bdpt.mojo used to give
    escape (s=0)              0.5400    0.7995  (power heuristic)
    NEE    (s=1)              0.2107    0.2005  (power heuristic)
    t=1 light tracing         0.0442    -- its own balance weight
    merging                   0.2050    -- its own balance weight

The ESCAPE is the whole problem: the power heuristic hands it 0.80 where the
balance heuristic over ALL strategies gives 0.54, and the 0.26 it takes too
much is almost exactly merging + t=1's combined share (0.249). NEE is nearly
right already, which is why fixing NEE alone barely moves a render and fixing
the escape alone moves it a lot.

Two initialization rules the exactness depends on, both easy to get wrong:
  * the camera subpath starts at dVC = dVM = 0. The lens is not an ordinary
    scattering vertex -- running the generic scatter rule there invents a
    connection strategy behind the lens.
  * the ESCAPE weight reads the POST-SCATTER carries at the last vertex, one
    step later than the NEE weight at that same vertex, and takes no d^2/cos
    (nothing was arrived at). Reading the arrival carries instead is 20% low
    at one bounce and 73% low at two.
""")
