# Phase 8.1: does the O(1) recursive MIS weight actually equal the honest
# sum over every BDPT strategy -- and does it survive a ReSTIR shift?
#
# The second question is the one Phase 8 rests on, and it is the reason
# this file exists before any renderer integration: if the light subpath's
# accumulator were NOT invariant under replacing the camera prefix, the
# whole "recover omega cheaply during reconnection" plan would collapse and
# every shifted candidate would need its subpath retraced.
#
# The two implementations under test are genuinely independent: the
# reference multiplies out every strategy's pdf from scratch, the recursion
# telescopes. Neither is written in terms of the other.

from std.math import abs
from std.testing import assert_true, TestSuite
from gonzales.restir_bdpt import (
    bdpt_strategy_pdf,
    bdpt_mis_weight_bruteforce,
    bdpt_light_accumulator,
    bdpt_camera_accumulator,
    bdpt_mis_weight_recursive,
)


def _rel_close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) <= Float64(1e-9) * max(abs(a), abs(b)) + Float64(1e-12)


# ── Fixtures ────────────────────────────────────────────────────────────────
#
# Deliberately irregular numbers. Uniform pdfs make several wrong groupings
# of the recursion accidentally correct, which would defeat the point.

def _pf() -> List[Float64]:
    return [0.31, 0.72, 0.19, 0.88, 0.44]

def _pr() -> List[Float64]:
    return [0.63, 0.27, 0.81, 0.36, 0.59]

def _gl() -> List[Float64]:
    return [0.47, 1.30, 0.22, 0.95, 0.71]

def _gc() -> List[Float64]:
    return [0.84, 0.39, 1.10, 0.28, 0.66]

comptime PA_LIGHT = Float64(0.23)
comptime PA_CAM = Float64(1.7)


def _omega_recursive(
    pa_light: Float64,
    pa_cam: Float64,
    pf: List[Float64],
    pr: List[Float64],
    gl: List[Float64],
    gc: List[Float64],
    s: Int,
) -> Float64:
    """Assemble the O(1) form the way a connection site would: one
    accumulator from each subpath, plus the connecting edge."""
    var b_l = bdpt_light_accumulator(pa_light, pf, pr, gl, gc, s)
    var b_c = bdpt_camera_accumulator(pa_cam, pf, pr, gl, gc, s)
    return bdpt_mis_weight_recursive(
        b_l, b_c, pf[s - 1], pr[s - 1], gl[s - 1], gc[s - 1]
    )


# ── 1. The recursion reproduces the explicit strategy sum ───────────────────

def test_recursive_matches_bruteforce_every_connection() raises:
    """For every interior strategy s (every edge a connection could be made
    across), the O(1) recursive weight must equal the explicit
    balance-heuristic sum over all k+2 strategies."""
    var pf = _pf(); var pr = _pr(); var gl = _gl(); var gc = _gc()
    var k = len(pf)
    for s in range(1, k + 1):
        var want = bdpt_mis_weight_bruteforce(PA_LIGHT, PA_CAM, pf, pr, gl, gc, s)
        var got = _omega_recursive(PA_LIGHT, PA_CAM, pf, pr, gl, gc, s)
        assert_true(_rel_close(got, want))


def test_strategy_weights_partition_unity() raises:
    """Balance-heuristic weights must sum to 1 over all strategies -- the
    property that lets omega compose with GRIS's own resampling MIS weight
    without double-counting."""
    var pf = _pf(); var pr = _pr(); var gl = _gl(); var gc = _gc()
    var k = len(pf)
    var total = Float64(0.0)
    for s in range(0, k + 2):
        total += bdpt_mis_weight_bruteforce(PA_LIGHT, PA_CAM, pf, pr, gl, gc, s)
    assert_true(_rel_close(total, Float64(1.0)))


def test_shorter_path_still_matches() raises:
    """A 2-edge path exercises the recursion's base cases, where B_L(1) and
    B_C(k) are hit directly with no iteration."""
    var pf = List[Float64](); pf.append(0.31); pf.append(0.72)
    var pr = List[Float64](); pr.append(0.63); pr.append(0.27)
    var gl = List[Float64](); gl.append(0.47); gl.append(1.30)
    var gc = List[Float64](); gc.append(0.84); gc.append(0.39)
    for s in range(1, 3):
        var want = bdpt_mis_weight_bruteforce(PA_LIGHT, PA_CAM, pf, pr, gl, gc, s)
        var got = _omega_recursive(PA_LIGHT, PA_CAM, pf, pr, gl, gc, s)
        assert_true(_rel_close(got, want))


# ── 2. Locality: each accumulator ignores the other subpath ─────────────────

def test_light_accumulator_ignores_camera_side() raises:
    """B_L(s) must not change when edges on the camera side of the
    connection change. This is the property, stated directly."""
    var pf = _pf(); var pr = _pr(); var gl = _gl(); var gc = _gc()
    var s = 3
    var before = bdpt_light_accumulator(PA_LIGHT, pf, pr, gl, gc, s)

    # Perturb every edge at or beyond the connection edge, plus pa_cam --
    # everything the camera subpath owns.
    var pf2 = _pf(); var pr2 = _pr(); var gl2 = _gl(); var gc2 = _gc()
    for e in range(s - 1, len(pf2)):
        pf2[e] *= 3.7; pr2[e] *= 0.41; gl2[e] *= 2.2; gc2[e] *= 0.13
    var after = bdpt_light_accumulator(PA_LIGHT, pf2, pr2, gl2, gc2, s)
    assert_true(_rel_close(before, after))


def test_camera_accumulator_ignores_light_side() raises:
    """The mirror property: B_C(s) is untouched by the light subpath."""
    var pf = _pf(); var pr = _pr(); var gl = _gl(); var gc = _gc()
    var s = 3
    var before = bdpt_camera_accumulator(PA_CAM, pf, pr, gl, gc, s)

    var pf2 = _pf(); var pr2 = _pr(); var gl2 = _gl(); var gc2 = _gc()
    for e in range(0, s - 1):
        pf2[e] *= 3.7; pr2[e] *= 0.41; gl2[e] *= 2.2; gc2[e] *= 0.13
    var after = bdpt_camera_accumulator(PA_CAM, pf2, pr2, gl2, gc2, s)
    assert_true(_rel_close(before, after))


# ── 3. Shift stability -- the actual Phase 8 claim ──────────────────────────

def test_shifted_path_mis_weight_reuses_light_accumulator() raises:
    """THE Phase 8.1 claim.

    A reconnection shift keeps a candidate's light subpath and swaps in the
    receiving pixel's camera prefix. Simulate that: build a donor path and a
    receiver path that share their light-side edges but differ everywhere on
    the camera side, then compute the shifted path's MIS weight from
      - the DONOR's stored light accumulator (reused verbatim, never
        recomputed), and
      - the RECEIVER's own camera accumulator (which it already has),
    and require it to equal a from-scratch brute-force evaluation of the
    shifted path. If this holds, omega survives a shift in O(1)."""
    var s = 3

    # Donor: the candidate whose light subpath we keep.
    var pf_d = _pf(); var pr_d = _pr(); var gl_d = _gl(); var gc_d = _gc()

    # Receiver: same light-side edges (0 .. s-2), different camera side and
    # a different connecting edge, as a different pixel's prefix would give.
    var pf_r = _pf(); var pr_r = _pr(); var gl_r = _gl(); var gc_r = _gc()
    for e in range(s - 1, len(pf_r)):
        pf_r[e] = pf_r[e] * 1.9 + 0.05
        pr_r[e] = pr_r[e] * 0.53 + 0.02
        gl_r[e] = gl_r[e] * 0.61 + 0.07
        gc_r[e] = gc_r[e] * 1.4 + 0.03
    var pa_cam_r = Float64(0.83)

    # The shifted path IS the receiver's geometry with the donor's light
    # prefix -- which, by construction, is exactly `pf_r/pr_r/gl_r/gc_r`
    # with pa_light unchanged, since the two agree on edges 0..s-2.
    var want = bdpt_mis_weight_bruteforce(PA_LIGHT, pa_cam_r, pf_r, pr_r, gl_r, gc_r, s)

    # Now the O(1) route: reuse the donor's light accumulator untouched.
    var b_l_donor = bdpt_light_accumulator(PA_LIGHT, pf_d, pr_d, gl_d, gc_d, s)
    var b_c_recv = bdpt_camera_accumulator(pa_cam_r, pf_r, pr_r, gl_r, gc_r, s)
    var got = bdpt_mis_weight_recursive(
        b_l_donor, b_c_recv, pf_r[s - 1], pr_r[s - 1], gl_r[s - 1], gc_r[s - 1]
    )
    assert_true(_rel_close(got, want))


def test_reusing_donor_camera_accumulator_is_wrong() raises:
    """The failure mode worth locking down.

    Reusing the donor's CAMERA accumulator too -- the tempting shortcut,
    since the candidate already carries it -- evaluates omega at the
    unshifted path. It must NOT agree with the shifted path's true weight,
    or this test is not actually detecting the mistake it exists to catch.
    Such a bug would be silent: a plausible image with quietly wrong
    weights."""
    var s = 3
    var pf_d = _pf(); var pr_d = _pr(); var gl_d = _gl(); var gc_d = _gc()
    var pf_r = _pf(); var pr_r = _pr(); var gl_r = _gl(); var gc_r = _gc()
    for e in range(s - 1, len(pf_r)):
        pf_r[e] = pf_r[e] * 1.9 + 0.05
        pr_r[e] = pr_r[e] * 0.53 + 0.02
        gl_r[e] = gl_r[e] * 0.61 + 0.07
        gc_r[e] = gc_r[e] * 1.4 + 0.03
    var pa_cam_r = Float64(0.83)

    var want = bdpt_mis_weight_bruteforce(PA_LIGHT, pa_cam_r, pf_r, pr_r, gl_r, gc_r, s)
    var b_l_donor = bdpt_light_accumulator(PA_LIGHT, pf_d, pr_d, gl_d, gc_d, s)
    var b_c_donor = bdpt_camera_accumulator(PA_CAM, pf_d, pr_d, gl_d, gc_d, s)
    var wrong = bdpt_mis_weight_recursive(
        b_l_donor, b_c_donor, pf_r[s - 1], pr_r[s - 1], gl_r[s - 1], gc_r[s - 1]
    )
    assert_true(not _rel_close(wrong, want))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
