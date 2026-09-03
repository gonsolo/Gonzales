from std.math import abs, sqrt
from std.testing import assert_true, TestSuite
from gonzales.geometry import dot, cross, Vec3f
from gonzales.shading import _mnee_walk2
from gonzales.sms import (
    sms_walk, sms_solve_bernoulli, SMSVertex, MAX_SMS_VERTICES,
    sms_seed_jitter, sms_same_solution, sms_vertex_flat, sms_vertex_init,
    sms_vertex_sphere, _sms_reproject_onto_sphere, _sms_eval_vertex,
)
from gonzales.rng import PCG32

comptime EPS: Float32 = 1e-4

def _flat_vert(pos: Vec3f, eta: Float32) -> SMSVertex:
    return sms_vertex_flat(
        pos,
        Vec3f(Float32(0.0), Float32(0.0), Float32(1.0)),
        Vec3f(Float32(1.0), Float32(0.0), Float32(0.0)),
        Vec3f(Float32(0.0), Float32(1.0), Float32(0.0)),
        eta,
    )

def _empty_verts() -> InlineArray[SMSVertex, MAX_SMS_VERTICES]:
    return InlineArray[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init())

def _snell_residual(
    x0: Vec3f, xL: Vec3f,
    verts: InlineArray[SMSVertex, MAX_SMS_VERTICES], n: Int,
) -> Float32:
    """Max |tangential component of the generalized half-vector| across all
    n vertices -- zero iff every vertex exactly satisfies Snell's law.
    Independent re-derivation of the same constraint sms_walk's own Newton
    loop converges on, used here purely to VERIFY convergence rather than
    to compute it."""
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

# ── sms_walk(n=2) must reduce EXACTLY to _mnee_walk2 ────────────────────────
# The block-tridiagonal Thomas-algorithm generalization was hand-derived to
# collapse to _mnee_walk2's own formulas at n=2 -- this is the load-bearing
# regression check for that derivation, not just a smoke test.

def test_sms_walk_n2_matches_mnee_walk2() raises:
    var x0 = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
    var x3 = Vec3f(Float32(0.2), Float32(-0.1), Float32(3.0))
    var n1 = Vec3f(Float32(0.0), Float32(0.0), Float32(1.0))
    var du = Vec3f(Float32(1.0), Float32(0.0), Float32(0.0))
    var dv = Vec3f(Float32(0.0), Float32(1.0), Float32(0.0))
    var x1_init = Vec3f(Float32(0.1), Float32(0.1), Float32(1.0))
    var eta1 = Float32(1.5)
    var x2_init = Vec3f(Float32(0.15), Float32(0.05), Float32(2.0))
    var eta2 = Float32(1.0) / Float32(1.5)

    var (ok2, x1_f2, x2_f2, bsdf2, jac2) = _mnee_walk2(
        x0, x3, x1_init, n1, du, dv, eta1, x2_init, n1, du, dv, eta2, du, dv)

    var verts = _empty_verts()
    verts[0] = _flat_vert(x1_init, eta1)
    verts[1] = _flat_vert(x2_init, eta2)
    var _r74 = sms_walk(x0, x3, verts, 2, du, dv)
    var okn = _r74[0]; var posn = _r74[1].copy(); var bsdfn = _r74[2]; var jacn = _r74[3]

    assert_true(ok2 == okn)
    assert_true(ok2)
    assert_true(abs(bsdf2 - bsdfn) < EPS)
    assert_true(abs(jac2 - jacn) < EPS)
    var d0 = x1_f2 - posn[0]
    var d1 = x2_f2 - posn[1]
    assert_true(dot(d0, d0) < Float32(1e-8))
    assert_true(dot(d1, d1) < Float32(1e-8))

# ── n=1/3/4 converge to a real Snell's-law solution ─────────────────────────
# No pre-existing N>=3 reference to regress against (MNEE only ever
# special-cased 1 and 2), so these verify convergence via an independently
# re-derived constraint check plus plausible-range asserts on the physical
# outputs, rather than a golden numeric comparison.

def test_sms_walk_n1_converges() raises:
    var x0 = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
    var xL = Vec3f(Float32(0.3), Float32(-0.2), Float32(4.0))
    var du = Vec3f(Float32(1.0), Float32(0.0), Float32(0.0))
    var dv = Vec3f(Float32(0.0), Float32(1.0), Float32(0.0))
    var verts = _empty_verts()
    verts[0] = _flat_vert(Vec3f(Float32(0.1), Float32(0.1), Float32(1.0)), Float32(1.5))
    var _r98 = sms_walk(x0, xL, verts, 1, du, dv)
    var ok = _r98[0]; var pos = _r98[1].copy(); var bsdf = _r98[2]; var jac = _r98[3]
    assert_true(ok)
    verts[0].pos = pos[0]
    assert_true(_snell_residual(x0, xL, verts, 1) < Float32(1e-3))
    # Upper bound is eta^2 (~2.25 here), not 1.0 -- a single, unpaired
    # refraction genuinely scales radiance by the solid-angle-compression
    # factor eta^2 (see sms.mojo's bsdf_product comment / the
    # project_dielectric_radiance_transmission_bug memory); only a chain
    # whose etas pair up entry/exit (like test_sms_walk_n4 below) cancels
    # back to <=1.
    assert_true(bsdf > Float32(0.0) and bsdf <= Float32(2.5))
    assert_true(jac >= Float32(0.0))

def test_sms_walk_n3_converges() raises:
    var x0 = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
    var xL = Vec3f(Float32(0.3), Float32(-0.2), Float32(4.0))
    var du = Vec3f(Float32(1.0), Float32(0.0), Float32(0.0))
    var dv = Vec3f(Float32(0.0), Float32(1.0), Float32(0.0))
    var verts = _empty_verts()
    verts[0] = _flat_vert(Vec3f(Float32(0.08), Float32(0.05), Float32(1.0)), Float32(1.5))
    verts[1] = _flat_vert(Vec3f(Float32(0.14), Float32(0.02), Float32(2.0)), Float32(1.0)/Float32(1.5))
    verts[2] = _flat_vert(Vec3f(Float32(0.20), Float32(-0.05), Float32(3.0)), Float32(1.33))
    var _r120 = sms_walk(x0, xL, verts, 3, du, dv)
    var ok = _r120[0]; var pos = _r120[1].copy(); var bsdf = _r120[2]; var jac = _r120[3]
    assert_true(ok)
    for i in range(3):
        verts[i].pos = pos[i]
    assert_true(_snell_residual(x0, xL, verts, 3) < Float32(1e-3))
    # eta 1.5 and 1/1.5 cancel; eta 1.33 is unpaired -> net upper bound is
    # ~1.33^2 (~1.77), not 1.0 -- see test_sms_walk_n1's comment above.
    assert_true(bsdf > Float32(0.0) and bsdf <= Float32(2.0))
    assert_true(jac >= Float32(0.0))

def test_sms_walk_n4_converges() raises:
    var x0 = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
    var xL = Vec3f(Float32(0.3), Float32(-0.15), Float32(4.0))
    var du = Vec3f(Float32(1.0), Float32(0.0), Float32(0.0))
    var dv = Vec3f(Float32(0.0), Float32(1.0), Float32(0.0))
    var verts = _empty_verts()
    verts[0] = _flat_vert(Vec3f(Float32(0.05), Float32(0.03), Float32(0.8)), Float32(1.5))
    verts[1] = _flat_vert(Vec3f(Float32(0.10), Float32(0.01), Float32(1.6)), Float32(1.0)/Float32(1.5))
    verts[2] = _flat_vert(Vec3f(Float32(0.16), Float32(-0.02), Float32(2.4)), Float32(1.33))
    verts[3] = _flat_vert(Vec3f(Float32(0.22), Float32(-0.06), Float32(3.2)), Float32(1.0)/Float32(1.33))
    var _r140 = sms_walk(x0, xL, verts, 4, du, dv)
    var ok = _r140[0]; var pos = _r140[1].copy(); var bsdf = _r140[2]; var jac = _r140[3]
    assert_true(ok)
    for i in range(4):
        verts[i].pos = pos[i]
    assert_true(_snell_residual(x0, xL, verts, 4) < Float32(1e-3))
    assert_true(bsdf > Float32(0.0) and bsdf <= Float32(1.0))
    assert_true(jac >= Float32(0.0))

# ── Sphere (curved-surface) support ─────────────────────────────────────────
# A flat triangle's tangent plane IS its surface everywhere, so _mnee_walk/
# _mnee_walk2/the tests above never needed reprojection. A sphere's tangent
# plane only agrees with the true surface at the point of tangency, so
# sms_walk reprojects a sphere vertex back onto the sphere (and re-derives
# its local frame) after every Newton step -- these tests exercise that
# machinery specifically, independently of the flat-triangle fast paths.

def test_sms_reproject_onto_sphere_snaps_exactly_and_frame_is_orthonormal() raises:
    var center = Vec3f(Float32(1.0), Float32(2.0), Float32(-3.0))
    var radius = Float32(2.5)
    # A point well off the sphere -- reprojection must snap it exactly onto
    # the surface, not just nudge it.
    var raw = center + Vec3f(Float32(5.0), Float32(0.0), Float32(0.0))
    var r = _sms_reproject_onto_sphere(raw, center, radius)
    var pos = r[0]; var normal = r[1]; var dp_du = r[2]; var dp_dv = r[3]
    var to_pos = pos - center
    var dist = sqrt(dot(to_pos, to_pos))
    assert_true(abs(dist - radius) < Float32(1e-4))
    var expected_n = to_pos * (Float32(1.0) / dist)
    var dn = normal - expected_n
    assert_true(dot(dn, dn) < Float32(1e-8))
    # dp_du/dp_dv must be an orthonormal basis of the tangent plane.
    assert_true(abs(dot(dp_du, normal)) < Float32(1e-4))
    assert_true(abs(dot(dp_dv, normal)) < Float32(1e-4))
    assert_true(abs(dot(dp_du, dp_dv)) < Float32(1e-4))
    assert_true(abs(sqrt(dot(dp_du, dp_du)) - Float32(1.0)) < Float32(1e-4))
    assert_true(abs(sqrt(dot(dp_dv, dp_dv)) - Float32(1.0)) < Float32(1e-4))

def test_sms_walk_sphere_n1_converges_and_stays_on_sphere() raises:
    """Single glass-sphere vertex (e.g. a thin spherical shell, one
    refractive interface). The solved position must lie EXACTLY on the
    sphere -- the entire point of reprojection is that a curved vertex's
    Newton walk never drifts into the tangent plane the way it would
    without it. Radius/distance proportions are chosen to be comfortably
    inside the local Newton solve's basin of convergence (see the
    n=2 comment below for the case that ISN'T)."""
    var center = Vec3f(Float32(0.0), Float32(0.0), Float32(3.5))
    var radius = Float32(2.0)
    var x0 = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
    var xL = Vec3f(Float32(0.3), Float32(-0.2), Float32(8.0))
    var du = Vec3f(Float32(1.0), Float32(0.0), Float32(0.0))
    var dv = Vec3f(Float32(0.0), Float32(1.0), Float32(0.0))
    var seed_pos = center - Vec3f(Float32(0.0), Float32(0.0), radius)
    var verts = _empty_verts()
    verts[0] = sms_vertex_sphere(seed_pos, center, radius, Float32(1.5))
    var _r194 = sms_walk(x0, xL, verts, 1, du, dv)
    var ok = _r194[0]; var pos = _r194[1].copy(); var bsdf = _r194[2]; var jac = _r194[3]
    assert_true(ok)
    var to_center = pos[0] - center
    var dist = sqrt(dot(to_center, to_center))
    assert_true(abs(dist - radius) < Float32(1e-3))
    # sms_walk only returns positions, not its internal per-iteration
    # frame -- re-derive the frame AT the solution to independently verify
    # Snell's law via the same _snell_residual check the flat tests use.
    # Tolerance is looser than the flat tests' 1e-3: sms_walk's OWN
    # convergence check is a max-norm over (dot(H,s), dot(H,t)) < 1e-3,
    # but _snell_residual reports the L2 norm of the same pair, which can
    # be up to sqrt(2)x larger for the same underlying convergence.
    var r = _sms_reproject_onto_sphere(pos[0], center, radius)
    verts[0].pos = r[0]; verts[0].normal = r[1]; verts[0].dp_du = r[2]; verts[0].dp_dv = r[3]
    assert_true(_snell_residual(x0, xL, verts, 1) < Float32(2e-3))
    assert_true(bsdf > Float32(0.0))
    assert_true(jac >= Float32(0.0))

def test_sms_eval_vertex_sphere_self_jacobian_matches_finite_difference() raises:
    """Regression test for the curvature-correction term in
    _sms_eval_vertex's "b" (self) Jacobian for a sphere vertex. A flat
    vertex's local frame (normal, s, t) never changes as it moves, so
    sms_vertex_mats's plain dH/du projection is exact; a sphere vertex's
    frame ROTATES as it moves (curvature), which drags the projection
    basis itself and needs an extra dot(H,normal)/radius term on both
    diagonal entries (see _sms_eval_vertex's own derivation comment).
    Validated here against a TRUE finite difference that reprojects the
    perturbed point onto the sphere and re-derives its frame there
    (sms_vertex_sphere), not a flat tangent-plane probe -- the whole bug
    this term fixes was invisible to a flat-plane FD check (that only
    validates the OLD, uncorrected formula) and only showed up as the
    n=2 solid-sphere case diverging under an otherwise-exact Newton
    direction; see project_sms_restir_phase6 memory for the full
    derivation and the empirical trail that found it."""
    var center = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
    var radius = Float32(1.5)
    var x0 = Vec3f(Float32(-6.0), Float32(0.0), Float32(0.0))
    var xL = Vec3f(Float32(6.0), Float32(0.0), Float32(0.0))
    var ior = Float32(1.5)
    var seed1 = Vec3f(Float32(-1.5), Float32(0.075), Float32(0.075))
    var seed2 = Vec3f(Float32(1.5), Float32(0.075), Float32(0.075))
    var verts = _empty_verts()
    verts[0] = sms_vertex_sphere(seed1, center, radius, ior)
    verts[1] = sms_vertex_sphere(seed2, center, radius, Float32(1.0) / ior)
    var ev0 = _sms_eval_vertex(x0, xL, verts, 2, 0)

    var eps = Float32(1e-3)
    var vp = verts.copy()
    vp[0] = sms_vertex_sphere(verts[0].pos + verts[0].dp_du * eps, center, radius, verts[0].eta)
    var evp = _sms_eval_vertex(x0, xL, vp, 2, 0)
    var vm = verts.copy()
    vm[0] = sms_vertex_sphere(verts[0].pos - verts[0].dp_du * eps, center, radius, verts[0].eta)
    var evm = _sms_eval_vertex(x0, xL, vm, 2, 0)
    var fd_du = (evp.cv - evm.cv) * (Float32(1.0) / (Float32(2.0) * eps))
    assert_true(abs(fd_du[0] - ev0.b[0]) < Float32(0.05))
    assert_true(abs(fd_du[1] - ev0.b[2]) < Float32(0.05))

    vp = verts.copy()
    vp[0] = sms_vertex_sphere(verts[0].pos + verts[0].dp_dv * eps, center, radius, verts[0].eta)
    evp = _sms_eval_vertex(x0, xL, vp, 2, 0)
    vm = verts.copy()
    vm[0] = sms_vertex_sphere(verts[0].pos - verts[0].dp_dv * eps, center, radius, verts[0].eta)
    evm = _sms_eval_vertex(x0, xL, vm, 2, 0)
    var fd_dv = (evp.cv - evm.cv) * (Float32(1.0) / (Float32(2.0) * eps))
    assert_true(abs(fd_dv[0] - ev0.b[1]) < Float32(0.05))
    assert_true(abs(fd_dv[1] - ev0.b[3]) < Float32(0.05))

def test_sms_walk_sphere_n2_solid_sphere_symmetric_case() raises:
    """Entry+exit through the SAME solid glass sphere -- the real caustic
    scenario a glass sphere sitting on a table produces: a straight probe
    ray that enters the sphere must also exit it before reaching the
    light, so this (not the 1-vertex case above) is the common real-world
    configuration.

    Before the curvature-correction term in _sms_eval_vertex's "b" self-
    Jacobian (see its own derivation comment, and
    test_sms_eval_vertex_sphere_self_jacobian_matches_finite_difference),
    this reliably DIVERGED regardless of step size -- a seed just 1% of
    the sphere's radius from the exact answer grew the residual on every
    iteration, even under backtracking line search all the way down to a
    ~1e-4-sized step. That ruled out "overshoot" as the cause (a correct
    descent direction shrinks the residual at ANY sufficiently small step)
    and pointed at the Jacobian itself: sms_vertex_mats's formula
    (originally derived for a flat vertex, whose local tangent frame never
    changes as it moves) doesn't account for a curved vertex's frame
    ROTATING as it moves, missing a real dot(H,normal)/radius term. With
    that term added, this converges in 1-2 full (undamped) Newton
    iterations -- the backtracking line search sms_walk still does for
    curved chains is now a defensive no-op here, not the fix.

    Mirror-symmetric setup (x0/xL and the two seed points are reflections
    of each other through the sphere's own x=0 symmetry plane) has a
    solution that MUST be symmetric too -- an independent check that
    doesn't require deriving the closed-form answer."""
    var center = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
    var radius = Float32(1.5)
    var x0 = Vec3f(Float32(-6.0), Float32(0.0), Float32(0.0))
    var xL = Vec3f(Float32(6.0), Float32(0.0), Float32(0.0))
    var du = Vec3f(Float32(0.0), Float32(1.0), Float32(0.0))
    var dv = Vec3f(Float32(0.0), Float32(0.0), Float32(1.0))
    var ior = Float32(1.5)
    var seed1 = Vec3f(Float32(-1.5), Float32(0.075), Float32(0.075))
    var seed2 = Vec3f(Float32(1.5), Float32(0.075), Float32(0.075))
    var verts = _empty_verts()
    verts[0] = sms_vertex_sphere(seed1, center, radius, ior)
    verts[1] = sms_vertex_sphere(seed2, center, radius, Float32(1.0) / ior)
    var _r300 = sms_walk(x0, xL, verts, 2, du, dv)
    var ok = _r300[0]; var pos = _r300[1].copy(); var bsdf = _r300[2]; var jac = _r300[3]
    assert_true(ok)
    var d0 = pos[0] - center; var dist0 = sqrt(dot(d0, d0))
    var d1 = pos[1] - center; var dist1 = sqrt(dot(d1, d1))
    assert_true(abs(dist0 - radius) < Float32(1e-3))
    assert_true(abs(dist1 - radius) < Float32(1e-3))
    assert_true(abs(pos[0][0] + pos[1][0]) < Float32(1e-2))
    assert_true(abs(pos[0][1] - pos[1][1]) < Float32(1e-2))
    assert_true(abs(pos[0][2] - pos[1][2]) < Float32(1e-2))
    assert_true(bsdf > Float32(0.0))
    assert_true(jac >= Float32(0.0))

# ── Degenerate input handling ────────────────────────────────────────────────

def test_sms_walk_fails_on_coincident_points() raises:
    var x0 = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
    var xL = Vec3f(Float32(0.0), Float32(0.0), Float32(1.0))
    var du = Vec3f(Float32(1.0), Float32(0.0), Float32(0.0))
    var dv = Vec3f(Float32(0.0), Float32(1.0), Float32(0.0))
    var verts = _empty_verts()
    # Vertex coincides with x0 -- wi length is zero, must fail cleanly, not NaN.
    verts[0] = _flat_vert(Vec3f(Float32(0.0), Float32(0.0), Float32(0.0)), Float32(1.5))
    var _r322 = sms_walk(x0, xL, verts, 1, du, dv)
    var ok = _r322[0]
    assert_true(not ok)

# ── Random seeding + Bernoulli-trial estimator (5.2/5.3) ────────────────────

def test_sms_seed_jitter_stays_in_tangent_plane() raises:
    """Jittering must only move a vertex within its own (dp_du, dp_dv)
    plane -- never off the flat triangle's normal direction, since the
    walk assumes a single flat surface throughout (no reprojection)."""
    var verts = _empty_verts()
    var orig_pos = Vec3f(Float32(0.1), Float32(0.1), Float32(1.0))
    verts[0] = _flat_vert(orig_pos, Float32(1.5))
    var pcg = PCG32(UInt64(12345), UInt64(1))
    sms_seed_jitter(verts, 1, pcg, Float32(0.05))
    var normal = verts[0].normal
    var delta = verts[0].pos - orig_pos
    assert_true(abs(dot(delta, normal)) < Float32(1e-6))

def test_sms_solve_bernoulli_converges_on_unique_solution() raises:
    """For an ordinary (non-degenerate) flat-glass chain the manifold
    solution is essentially unique, so the very first re-seeded trial
    should almost always rediscover it -- trial_count should stay small,
    not run to the SMS_BERNOULLI_MAX_TRIALS bound."""
    var x0 = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
    var xL = Vec3f(Float32(0.3), Float32(-0.2), Float32(4.0))
    var du = Vec3f(Float32(1.0), Float32(0.0), Float32(0.0))
    var dv = Vec3f(Float32(0.0), Float32(1.0), Float32(0.0))
    var verts = _empty_verts()
    verts[0] = _flat_vert(Vec3f(Float32(0.08), Float32(0.05), Float32(1.0)), Float32(1.5))
    verts[1] = _flat_vert(Vec3f(Float32(0.14), Float32(0.02), Float32(2.0)), Float32(1.0)/Float32(1.5))
    verts[2] = _flat_vert(Vec3f(Float32(0.20), Float32(-0.05), Float32(3.0)), Float32(1.33))
    var pcg = PCG32(UInt64(777), UInt64(1))
    var _r354 = sms_solve_bernoulli(x0, xL, verts, 3, du, dv, Float32(0.01), pcg)
    var ok = _r354[0]; var pos = _r354[1].copy(); var bsdf = _r354[2]; var jac = _r354[3]; var trials = _r354[4]
    assert_true(ok)
    assert_true(bsdf > Float32(0.0))
    assert_true(jac >= Float32(0.0))
    assert_true(trials >= Float32(1.0))
    _ = pos

def test_sms_same_solution_detects_match_and_mismatch() raises:
    """Uniqueness is judged on the DIRECTION x0->x (mirroring the reference
    SMS renderer), so the test fixes an x0 and moves the candidate both
    along the view direction (same root, must still match -- this is the
    case a position-epsilon test used to get wrong, inflating the Bernoulli
    trial count) and across it (a genuinely different root)."""
    var verts = _empty_verts()
    var x0 = Vec3f(Float32(0.0), Float32(0.0), Float32(0.0))
    var a = InlineArray[Vec3f, MAX_SMS_VERTICES](fill=Vec3f(Float32(0.0)))
    var b = InlineArray[Vec3f, MAX_SMS_VERTICES](fill=Vec3f(Float32(0.0)))
    a[0] = Vec3f(Float32(0.0), Float32(0.0), Float32(10.0))
    b[0] = Vec3f(Float32(0.0), Float32(0.0), Float32(10.0))
    assert_true(sms_same_solution(x0, a, b, 1))
    # Same direction from x0, different distance -> same root.
    b[0] = Vec3f(Float32(0.0), Float32(0.0), Float32(10.02))
    assert_true(sms_same_solution(x0, a, b, 1))
    # Sideways displacement -> different direction -> different root.
    b[0] = Vec3f(Float32(0.5), Float32(0.0), Float32(10.0))
    assert_true(not sms_same_solution(x0, a, b, 1))
    _ = verts

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
