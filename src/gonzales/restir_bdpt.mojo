# Phase 8.1 groundwork: recovering BDPT's strategy MIS weight in O(1) at
# connection time, and the property that makes it survive a ReSTIR shift.
#
# Nothing in the renderer calls this yet. It exists so the Phase 8.1
# derivation is *executable* -- `Tests/unit/test_restir_bdpt_mis.mojo`
# checks the recursion against a brute-force sum over every strategy, and
# then checks the shift-stability claim Phase 8 actually depends on. See
# docs/A2_restir_migration_plan.md's Phase 8 section for the writeup.
#
# Deliberately dependency-free (plain scalars and Lists, no scene, no BVH,
# no Vertex) -- the same scope restir_vol.mojo keeps, and for the same
# reason: the math can then be verified in isolation, before any of it is
# wired into bdpt.mojo's much larger machinery.
#
# Float64 throughout: this is a reference/verification module, and the
# brute-force side multiplies O(path length) pdfs together, which is
# exactly the cancellation the recursive form exists to avoid. The eventual
# renderer-side implementation would be Float32, matching bdpt.mojo's
# existing dVCM/dVC carries.

from std.math import abs


# ── Path model ──────────────────────────────────────────────────────────────
#
# A full path has k+1 vertices x_0 .. x_k, x_0 on a light and x_k on the
# camera, and k edges. Strategy `s` means "the light subpath supplied s
# vertices (x_0 .. x_{s-1}) and the camera subpath supplied the rest",
# so s ranges over 0 ..= k+1 and the connection happens across edge s-1.
#
# Per EDGE e (between x_e and x_{e+1}) the caller supplies four numbers,
# all of which bdpt.mojo already computes at each bounce:
#
#   pf[e]  forward solid-angle pdf, sampling x_e -> x_{e+1}   (light-ward)
#   pr[e]  reverse solid-angle pdf, sampling x_{e+1} -> x_e   (camera-ward)
#   gl[e]  solid-angle -> area conversion at x_{e+1}, i.e. cos_{e+1}/dist^2
#   gc[e]  solid-angle -> area conversion at x_e,     i.e. cos_e/dist^2
#
# plus the two endpoint area pdfs pa_light (for x_0) and pa_cam (for x_k).
#
# Splitting the conversion into gl/gc rather than one geometry term keeps
# each factor attached to the vertex it is measured at, which is what makes
# the locality property below inspectable.


def bdpt_strategy_pdf(
    pa_light: Float64,
    pa_cam: Float64,
    pf: List[Float64],
    pr: List[Float64],
    gl: List[Float64],
    gc: List[Float64],
    s: Int,
) -> Float64:
    """Area-measure pdf of generating this exact path with strategy `s`.

    p_s = P_L(s) * P_C(s), where the light side contributes x_0..x_{s-1}
    and the camera side contributes x_s..x_k. Written out multiplicatively
    with no telescoping -- this is the honest, expensive definition the
    recursive form has to reproduce."""
    var k = len(pf)
    var p_l = Float64(1.0)
    if s >= 1:
        p_l = pa_light
        for e in range(0, s - 1):
            p_l *= pf[e] * gl[e]
    var p_c = Float64(1.0)
    if s <= k:
        p_c = pa_cam
        for e in range(s, k):
            p_c *= pr[e] * gc[e]
    return p_l * p_c


def bdpt_mis_weight_bruteforce(
    pa_light: Float64,
    pa_cam: Float64,
    pf: List[Float64],
    pr: List[Float64],
    gl: List[Float64],
    gc: List[Float64],
    s: Int,
) -> Float64:
    """Balance-heuristic MIS weight for strategy `s`, by explicit summation
    over every strategy that could have produced this path. O(k^2) and
    numerically the worse of the two forms -- a reference, not a design."""
    var k = len(pf)
    var total = Float64(0.0)
    for sp in range(0, k + 2):
        total += bdpt_strategy_pdf(pa_light, pa_cam, pf, pr, gl, gc, sp)
    if total <= Float64(0.0):
        return Float64(0.0)
    return bdpt_strategy_pdf(pa_light, pa_cam, pf, pr, gl, gc, s) / total


# ── The recursive form ──────────────────────────────────────────────────────
#
# Writing r(j) = p_{j+1}/p_j, the two tails of the strategy sum telescope,
# and each one factors into (a quantity belonging to ONE subpath) times (a
# quantity belonging to the connecting edge):
#
#   sum over s' < s  of p_{s'}/p_s  =  (pr[s-1] * gc[s-1]) * B_L(s)
#   sum over s' > s  of p_{s'}/p_s  =  (pf[s-1] * gl[s-1]) * B_C(s)
#
#   omega_s = 1 / ( 1 + (pf[s-1]*gl[s-1])*B_C(s) + (pr[s-1]*gc[s-1])*B_L(s) )
#
# with
#
#   B_L(0)   = 0                B_C(k+1) = 0
#   B_L(1)   = 1 / pa_light     B_C(k)   = 1 / pa_cam
#   B_L(s+1) = (1 + (pr[s-1]*gc[s-1]) * B_L(s)) / (pf[s-1]*gl[s-1])
#   B_C(s)   = (1 + (pf[s]  *gl[s]  ) * B_C(s+1)) / (pr[s]*gc[s])
#
# B_L(s) reads only pa_light and edges <= s-2 -- strictly inside the light
# subpath. B_C(s) reads only pa_cam and edges >= s -- strictly inside the
# camera subpath. NEITHER references the other side. That is the whole
# point: it is what lets a shift reuse one side untouched.
#
# This is the same shape bdpt.mojo already ships at its connect sites
#   w_light  = camera_bsdf_dir_pdf_a * (eta_vm + lv.dVCM + lv.dVC * ...)
#   w_camera = light_bsdf_dir_pdf_a  * (eta_vm + cv.dVCM + cv.dVC * ...)
#   mis      = 1 / (w_light + 1 + w_camera)
# with eta_vm = 0 (merging off, which is ReSTIR BDPT's setting) and the
# vertex's own reverse pdf folded into B_L/B_C rather than carried beside
# it -- bdpt.mojo splits the same tail across two channels (dVCM being the
# "connect right here" term, dVC the deeper tail). The two factorings agree
# algebraically, but note what is and isn't checked here: the tests verify
# THIS module against brute force. They do not re-verify bdpt.mojo's
# two-channel split, which has its own reference validation against
# SmallVCM (see project_vcm_stage2_mis_derivation).


def bdpt_light_accumulator(
    pa_light: Float64,
    pf: List[Float64],
    pr: List[Float64],
    gl: List[Float64],
    gc: List[Float64],
    s: Int,
) -> Float64:
    """B_L(s): the light subpath's own contribution to the MIS denominator.

    Depends ONLY on pa_light and edges 0..s-2, i.e. on vertices the light
    subpath actually generated. Shift-invariant under any change to the
    camera prefix -- see this module's header and the Phase 8 writeup."""
    if s <= 0:
        return Float64(0.0)
    var acc = Float64(1.0) / pa_light
    for j in range(1, s):
        acc = (Float64(1.0) + (pr[j - 1] * gc[j - 1]) * acc) / (pf[j - 1] * gl[j - 1])
    return acc


def bdpt_camera_accumulator(
    pa_cam: Float64,
    pf: List[Float64],
    pr: List[Float64],
    gl: List[Float64],
    gc: List[Float64],
    s: Int,
) -> Float64:
    """B_C(s): the camera subpath's own contribution to the MIS denominator.

    Depends ONLY on pa_cam and edges s..k-1. This is the half a reconnection
    shift invalidates, and the half the receiving pixel already carries for
    its own base path."""
    var k = len(pf)
    if s > k:
        return Float64(0.0)
    var acc = Float64(1.0) / pa_cam
    var j = k - 1
    while j >= s:
        acc = (Float64(1.0) + (pf[j] * gl[j]) * acc) / (pr[j] * gc[j])
        j -= 1
    return acc


def bdpt_mis_weight_recursive(
    b_light: Float64,
    b_camera: Float64,
    pf_conn: Float64,
    pr_conn: Float64,
    gl_conn: Float64,
    gc_conn: Float64,
) -> Float64:
    """Combine the two subpath accumulators with the connecting edge's own
    four quantities. O(1): no retracing, no sum over strategies.

    `*_conn` are edge s-1's values -- the edge the connection is made
    across, whose pdfs must be evaluated anyway to compute the path's
    contribution, so this adds no geometric work beyond the connection."""
    var denom = (
        Float64(1.0) + (pf_conn * gl_conn) * b_camera + (pr_conn * gc_conn) * b_light
    )
    if denom <= Float64(0.0):
        return Float64(0.0)
    return Float64(1.0) / denom
