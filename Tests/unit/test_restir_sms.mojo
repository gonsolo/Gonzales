from std.math import abs, sqrt
from std.testing import assert_true, TestSuite
from gonzales.geometry import RGB, dot, cross
from gonzales.sms import sms_walk, SMSVertex, MAX_SMS_VERTICES
from gonzales.restir_sms import (
    SMSReservoir, sms_reservoir_init, sms_target_pdf, sms_shift,
)

comptime EPS: Float32 = 1e-4

def _flat_vert(pos: SIMD[DType.float32, 3], eta: Float32) -> SMSVertex:
    return SMSVertex(
        pos,
        SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(1.0)),
        SIMD[DType.float32, 3](Float32(1.0), Float32(0.0), Float32(0.0)),
        SIMD[DType.float32, 3](Float32(0.0), Float32(1.0), Float32(0.0)),
        eta,
    )

def _empty_verts() -> InlineArray[SMSVertex, MAX_SMS_VERTICES]:
    return InlineArray[SMSVertex, MAX_SMS_VERTICES](fill=SMSVertex(
        SIMD[DType.float32, 3](Float32(0.0)), SIMD[DType.float32, 3](Float32(0.0)),
        SIMD[DType.float32, 3](Float32(0.0)), SIMD[DType.float32, 3](Float32(0.0)), Float32(1.0)))

def _snell_residual(
    x0: SIMD[DType.float32, 3], xL: SIMD[DType.float32, 3],
    verts: InlineArray[SMSVertex, MAX_SMS_VERTICES], n: Int,
) -> Float32:
    var worst = Float32(0.0)
    for i in range(n):
        var prev_pos = x0 if i == 0 else verts[i-1].pos
        var next_pos = xL if i == n-1 else verts[i+1].pos
        var wiv = prev_pos - verts[i].pos; var wil = sqrt(dot(wiv,wiv))
        var wov = next_pos - verts[i].pos; var wol = sqrt(dot(wov,wov))
        var wi = wiv*(Float32(1)/wil); var wo = wov*(Float32(1)/wol)
        var Hv = -(wi + wo*verts[i].eta); var Hl = sqrt(dot(Hv,Hv))
        var H = Hv*(Float32(1)/Hl)
        var s3 = verts[i].dp_du - verts[i].normal*dot(verts[i].dp_du, verts[i].normal)
        var s = s3*(Float32(1)/sqrt(dot(s3,s3)))
        var t = cross(verts[i].normal, s)
        var res = sqrt(dot(H,s)*dot(H,s) + dot(H,t)*dot(H,t))
        if res > worst:
            worst = res
    return worst

# ── sms_reservoir_init ───────────────────────────────────────────────────────

def test_sms_reservoir_init_has_no_winner() raises:
    var res = sms_reservoir_init()
    assert_true(res.n_vertices == Int32(0))
    assert_true(_close(res.state.w_sum, Float32(0.0)))

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# ── sms_target_pdf ───────────────────────────────────────────────────────────

def test_sms_target_pdf_positive_for_valid_forward_facing_geometry() raises:
    # first_vertex sits at +Z from hit_point; normal must point TOWARD it
    # (cos_s_x0 uses -wi_fn, the outgoing hit_point->first_vertex direction).
    var hit_point = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(0.0))
    var normal = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(1.0))
    var alb = RGB(Float32(0.8))
    var first_vertex = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(1.0))
    var first_normal = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(1.0))
    var le = RGB(Float32(10.0))
    var p_hat = sms_target_pdf(hit_point, normal, alb, first_vertex, first_normal, le, Float32(0.9), Float32(0.5))
    assert_true(p_hat > Float32(0.0))

def test_sms_target_pdf_zero_when_backfacing() raises:
    # Same geometry, but normal points AWAY from first_vertex -- cos_s_x0 <= 0.
    var hit_point = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(0.0))
    var normal = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(-1.0))
    var alb = RGB(Float32(0.8))
    var first_vertex = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(1.0))
    var first_normal = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(1.0))
    var le = RGB(Float32(10.0))
    var p_hat = sms_target_pdf(hit_point, normal, alb, first_vertex, first_normal, le, Float32(0.9), Float32(0.5))
    assert_true(_close(p_hat, Float32(0.0)))

def test_sms_target_pdf_zero_when_degenerate_distance() raises:
    var hit_point = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(0.0))
    var normal = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(-1.0))
    var alb = RGB(Float32(0.8))
    var le = RGB(Float32(10.0))
    var p_hat = sms_target_pdf(hit_point, normal, alb, hit_point, normal, le, Float32(0.9), Float32(0.5))
    assert_true(_close(p_hat, Float32(0.0)))

# ── sms_shift ────────────────────────────────────────────────────────────────
# Two-glass-pane synthetic setup shared with test_sms.mojo: a "neighbor"
# pixel's already-solved chain gets shifted to a nearby "current" pixel
# with the SAME (static) light point, exactly the case gonzales's own
# no-animation-support constraint restricts this to.

def test_sms_shift_succeeds_and_lands_on_a_valid_solution_for_a_nearby_pixel() raises:
    var xL = SIMD[DType.float32, 3](Float32(0.2), Float32(-0.1), Float32(3.0))
    var du = SIMD[DType.float32, 3](Float32(1.0), Float32(0.0), Float32(0.0))
    var dv = SIMD[DType.float32, 3](Float32(0.0), Float32(1.0), Float32(0.0))

    var src_x0 = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(0.0))
    var verts = _empty_verts()
    verts[0] = _flat_vert(SIMD[DType.float32, 3](Float32(0.1), Float32(0.1), Float32(1.0)), Float32(1.5))
    verts[1] = _flat_vert(SIMD[DType.float32, 3](Float32(0.15), Float32(0.05), Float32(2.0)), Float32(1.0)/Float32(1.5))
    var (src_ok, src_pos, _b, _j) = sms_walk(src_x0, xL, verts, 2, du, dv)
    assert_true(src_ok)
    var src_verts = _empty_verts()
    src_verts[0] = verts[0]; src_verts[0].pos = src_pos[0]
    src_verts[1] = verts[1]; src_verts[1].pos = src_pos[1]

    # A nearby current pixel, small perturbation from the neighbor -- the
    # regime manifold shift is meant for (spatial reuse between adjacent
    # pixels, or temporal reuse under a near-static scene).
    var dst_x0 = SIMD[DType.float32, 3](Float32(0.02), Float32(-0.01), Float32(0.0))

    var (ok, shifted, bsdf, jac) = sms_shift(
        dst_x0, xL, du, dv,
        src_x0, xL, du, dv,
        src_verts, 2)
    assert_true(ok)
    assert_true(bsdf > Float32(0.0))
    assert_true(jac >= Float32(0.0))

    var check_verts = _empty_verts()
    check_verts[0] = src_verts[0]; check_verts[0].pos = shifted[0]
    check_verts[1] = src_verts[1]; check_verts[1].pos = shifted[1]
    assert_true(_snell_residual(dst_x0, xL, check_verts, 2) < Float32(1e-3))

def test_sms_shift_rejects_degenerate_source_chain() raises:
    """Src's own first vertex coincides with src_x0 -- original_dir is
    undefined (zero length); must reject cleanly, not NaN."""
    var xL = SIMD[DType.float32, 3](Float32(0.2), Float32(-0.1), Float32(3.0))
    var du = SIMD[DType.float32, 3](Float32(1.0), Float32(0.0), Float32(0.0))
    var dv = SIMD[DType.float32, 3](Float32(0.0), Float32(1.0), Float32(0.0))
    var src_x0 = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(0.0))
    var src_verts = _empty_verts()
    src_verts[0] = _flat_vert(src_x0, Float32(1.5))
    src_verts[1] = _flat_vert(SIMD[DType.float32, 3](Float32(0.15), Float32(0.05), Float32(2.0)), Float32(1.0)/Float32(1.5))
    var (ok, _shifted, _bsdf, _jac) = sms_shift(
        src_x0, xL, du, dv, src_x0, xL, du, dv, src_verts, 2)
    assert_true(not ok)

def test_sms_shift_rejects_when_forward_walk_fails() raises:
    """Shifting into a domain with no reachable solution (dst_x0 placed
    such that the seeded Newton walk cannot converge, e.g. behind both
    glass planes with an incompatible transmissive check) must report
    failure rather than returning a bogus chain."""
    var xL = SIMD[DType.float32, 3](Float32(0.2), Float32(-0.1), Float32(3.0))
    var du = SIMD[DType.float32, 3](Float32(1.0), Float32(0.0), Float32(0.0))
    var dv = SIMD[DType.float32, 3](Float32(0.0), Float32(1.0), Float32(0.0))
    var src_x0 = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(0.0))
    var verts = _empty_verts()
    verts[0] = _flat_vert(SIMD[DType.float32, 3](Float32(0.1), Float32(0.1), Float32(1.0)), Float32(1.5))
    verts[1] = _flat_vert(SIMD[DType.float32, 3](Float32(0.15), Float32(0.05), Float32(2.0)), Float32(1.0)/Float32(1.5))
    var (src_ok, src_pos, _b, _j) = sms_walk(src_x0, xL, verts, 2, du, dv)
    assert_true(src_ok)
    var src_verts = _empty_verts()
    src_verts[0] = verts[0]; src_verts[0].pos = src_pos[0]
    src_verts[1] = verts[1]; src_verts[1].pos = src_pos[1]

    # Same position as one of the glass vertices -- wi/wo length collapses
    # to zero at vertex 0, forcing an immediate Newton-walk failure.
    var dst_x0 = src_pos[0]
    var (ok, _shifted, _bsdf, _jac) = sms_shift(
        dst_x0, xL, du, dv, src_x0, xL, du, dv, src_verts, 2)
    assert_true(not ok)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
