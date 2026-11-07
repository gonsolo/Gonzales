from std.collections import Array
from std.math import abs, sqrt, cos, sin
from std.testing import assert_true, TestSuite
from gonzales.core.geometry import dot, cross, Vec3f, PI
from gonzales.core.rng import PCG32
from gonzales.scene.appearance.materials import fr_dielectric
from gonzales.transport.manifold import _mnee_walk, _mnee_walk2, _mnee_chain_g, _mnee_chain_t_eta
from gonzales.transport.sms import (
    sms_walk, SMSVertex, MAX_SMS_VERTICES, sms_vertex_flat, sms_vertex_init, sms_vertex_sphere,
    sms_refresh_solved_frames, sms_chain_eval,
)

# Scene of the brute-force check (y up): a dielectric ball reflects toward a glass slab, which refracts twice into a
# square lamp facing down. x0 sees the lamp only through that chain.
comptime C_R: Float32 = 0.5        # ball radius, centre at the origin
comptime BALL_IOR: Float32 = 1.8
comptime SLAB_Y0: Float32 = 1.0
comptime SLAB_Y1: Float32 = 1.15
comptime SLAB_IOR: Float32 = 1.5
comptime LAMP_Y: Float32 = 2.0
comptime LAMP_H: Float32 = 0.5     # half size

def _v(x: Float64, y: Float64, z: Float64) -> Vec3f:
    return Vec3f(Float32(x), Float32(y), Float32(z))

def _norm(v: Vec3f) -> Vec3f:
    return v * (Float32(1.0) / sqrt(dot(v, v)))

def _empty() -> Array[SMSVertex, MAX_SMS_VERTICES]:
    return Array[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init())

def _x0() -> Vec3f:
    return _v(1.2, -0.8, 0.1)

def _chain(x0: Vec3f, y: Vec3f, tir_x: Float32 = Float32(0)) -> Tuple[Array[SMSVertex, MAX_SMS_VERTICES], Int]:
    """Seed chain: reflection off the ball, into the slab, (a total internal reflection off the wall x = tir_x), out."""
    var verts = _empty()
    # seeds on the unfolded straight line toward the lamp point mirrored in the wall
    var yv = y
    if tir_x != Float32(0):
        yv = Vec3f(Float32(2) * tir_x - y[0], y[1], y[2])
    var s = _norm(_norm(x0) + _norm(yv)) * C_R
    var ball = sms_vertex_sphere(s, Vec3f(Float32(0)), C_R, BALL_IOR, eta_raw=BALL_IOR)
    ball.refl = Int8(1)
    verts[0] = ball
    var down = _v(0, -1, 0)
    var ex = _v(1, 0, 0); var ez = _v(0, 0, 1)
    var p1 = s + (yv - s) * ((SLAB_Y0 - s[1]) / (yv[1] - s[1]))
    var p3 = s + (yv - s) * ((SLAB_Y1 - s[1]) / (yv[1] - s[1]))
    verts[1] = sms_vertex_flat(p1, down, ex, ez, SLAB_IOR)
    if tir_x == Float32(0):
        verts[2] = sms_vertex_flat(p3, down, ex, ez, Float32(1) / SLAB_IOR)
        return (verts^, 3)
    # the wall faces the path's side (+x) and reflects inside the glass, where F = 1 past the critical angle
    var pw = p1 + (p3 - p1) * ((tir_x - p1[0]) / (p3[0] - p1[0]))
    var wall = sms_vertex_flat(pw, _v(1, 0, 0), _v(0, 1, 0), ez, Float32(1) / SLAB_IOR)
    wall.refl = Int8(1)
    verts[2] = wall
    verts[3] = sms_vertex_flat(Vec3f(Float32(2) * tir_x - p3[0], SLAB_Y1, p3[2]), down, ex, ez, Float32(1) / SLAB_IOR)
    return (verts^, 4)

def _solve(x0: Vec3f, y: Vec3f, var verts: Array[SMSVertex, MAX_SMS_VERTICES], n: Int) -> Tuple[Bool, Array[SMSVertex, MAX_SMS_VERTICES], Float32, Float32]:
    """(ok, solved chain, radiance factor product, g = dw0/dA_y)."""
    var r = sms_walk(x0, y, verts, n, _v(1, 0, 0), _v(0, 0, 1))
    if not r[0]:
        return (False, verts^, Float32(0), Float32(0))
    for i in range(n):
        verts[i].pos = r[1][i]
    sms_refresh_solved_frames(verts, n)
    var g = _mnee_chain_g(x0, verts, r[3])
    return (True, verts^, r[2], g)

# ── A flat mirror: the root is the virtual image's, g is the unfolded inverse square ──────────────────────────────────

def test_flat_reflection_root_and_g() raises:
    var x0 = _v(-1, 0, 1); var xL = _v(1, 0, 2)
    var n = _v(0, 0, 1); var du = _v(1, 0, 0); var dv = _v(0, 1, 0)
    var w1 = _mnee_walk(x0, xL, _v(0.2, 0.1, 0), n, du, dv, Float32(1.5), refl=True)
    assert_true(w1[0])
    assert_true(abs(w1[1][0] + Float32(1.0 / 3.0)) < Float32(1e-4) and abs(w1[1][1]) < Float32(1e-4))
    var verts = _empty()
    var v = sms_vertex_flat(_v(0.2, 0.1, 0), n, du, dv, Float32(1.5))
    v.refl = Int8(1)
    verts[0] = v
    # a lamp tilted against the ray, so its cosine is not trivial
    var ldu = _norm(_v(1, 0, 0.3)); var ldv = _v(0, 1, 0)
    var r = sms_walk(x0, xL, verts, 1, ldu, ldv)
    assert_true(r[0])
    var d = r[1][0] - w1[1]
    assert_true(dot(d, d) < Float32(1e-8))
    verts[0].pos = r[1][0]
    var g = _mnee_chain_g(x0, verts, r[3])
    var ln = _norm(cross(ldu, ldv))
    var to_x1 = _norm(r[1][0] - xL)
    var a = r[1][0] - x0; var b = xL - r[1][0]
    var dist = sqrt(dot(a, a)) + sqrt(dot(b, b))
    var g_ref = abs(dot(ln, to_x1)) / (dist * dist)
    assert_true(abs(g / g_ref - Float32(1)) < Float32(1e-3))
    # weight: F at the incidence angle, no compression
    var cos_i = dot(n, _norm(x0 - r[1][0]))
    assert_true(abs(r[2] - fr_dielectric(cos_i, Float32(1.5))) < Float32(1e-4))
    assert_true(abs(_mnee_chain_t_eta(x0, verts, 1)[0] - fr_dielectric(cos_i, Float32(1.5))) < Float32(1e-4))

def test_walk2_reflect_matches_sms_walk() raises:
    """A mirror then a refraction: the 2-vertex fast path and the general solve agree."""
    var x0 = _v(-1, 0, 1); var xL = _v(1.5, 0.2, 3.0)
    var n1 = _v(0, 0, 1); var du = _v(1, 0, 0); var dv = _v(0, 1, 0)
    var n2 = _v(0, 0, -1)   # faces the path's side: the ray arrives from below the plane z = 1.5
    var x1i = _v(-0.2, 0, 0); var x2i = _v(0.3, 0.05, 1.5)
    var w2 = _mnee_walk2(x0, xL, x1i, n1, du, dv, Float32(1.5), x2i, n2, du, dv, Float32(1.33), du, dv, refl1=True)
    assert_true(w2[0])
    var verts = _empty()
    var a = sms_vertex_flat(x1i, n1, du, dv, Float32(1.5)); a.refl = Int8(1)
    verts[0] = a
    verts[1] = sms_vertex_flat(x2i, n2, du, dv, Float32(1.33))
    var r = sms_walk(x0, xL, verts, 2, du, dv)
    assert_true(r[0])
    var d0 = r[1][0] - w2[1]; var d1 = r[1][1] - w2[2]
    assert_true(dot(d0, d0) < Float32(1e-8) and dot(d1, d1) < Float32(1e-8))
    assert_true(abs(r[2] / w2[3] - Float32(1)) < Float32(1e-3))
    assert_true(abs(r[3] / w2[4] - Float32(1)) < Float32(1e-3))

# ── g of curved chains with reflections against finite differences of the solved first direction ──────────────────

def _fd_g(x0: Vec3f, y: Vec3f, tir_x: Float32) raises -> Tuple[Float32, Float32]:
    var c = _chain(x0, y, tir_x)
    var s = _solve(x0, y, c[0].copy(), c[1])
    assert_true(s[0])
    var w = _norm(s[1][0].pos - x0)
    var h = Float32(2e-3)
    var su = _solve(x0, y + _v(1, 0, 0) * h, s[1].copy(), c[1])
    var sv = _solve(x0, y + _v(0, 0, 1) * h, s[1].copy(), c[1])
    assert_true(su[0] and sv[0])
    var dwu = _norm(su[1][0].pos - x0) - w
    var dwv = _norm(sv[1][0].pos - x0) - w
    var area = sqrt(dot(cross(dwu, dwv), cross(dwu, dwv))) / (h * h)
    return (s[3], area)

def test_reflect_then_slab_g_matches_finite_difference() raises:
    var r = _fd_g(_x0(), Vec3f(Float32(0.1), LAMP_Y, Float32(-0.05)), Float32(0))
    assert_true(abs(r[0] / r[1] - Float32(1)) < Float32(0.02))

def test_reflect_slab_tir_g_matches_finite_difference() raises:
    # the path runs left inside the slab and folds back off the wall x = -0.15 (point found by forward tracing)
    var x0 = _x0()
    var y = Vec3f(Float32(0.42), LAMP_Y, Float32(0.78))
    var c = _chain(x0, y, Float32(-0.15))
    var s = _solve(x0, y, c[0].copy(), c[1])
    assert_true(s[0])
    # past the critical angle inside the glass: the wall reflects everything
    var wi = _norm(s[1][1].pos - s[1][2].pos)
    assert_true(abs(wi[0]) < Float32(0.6))   # incidence from the wall normal (x) beyond asin(1/1.5) = 41.8 deg
    var t = _mnee_chain_t_eta(x0, s[1], 4)
    assert_true(t[0] > Float32(0) and abs(t[1] - Float32(1)) < Float32(1e-4))
    var r = _fd_g(x0, y, Float32(-0.15))
    print("TIR chain: wall hit at y", s[1][2].pos[1], "g", r[0], "finite difference", r[1])
    assert_true(abs(r[0] / r[1] - Float32(1)) < Float32(0.02))

# ── The estimator against brute force: irradiance at x0 through ball reflection + slab ─────────────────────────────

def _refract_up(d: Vec3f, eta_i_over_t: Float32) -> Tuple[Bool, Vec3f]:
    """Refraction through a horizontal face crossed upward (the face normal toward the ray is -y)."""
    var nf = _v(0, -1, 0)
    var ci = -dot(d, nf)
    var s2 = eta_i_over_t * eta_i_over_t * (Float32(1) - ci * ci)
    if s2 >= Float32(1):
        return (False, d)
    return (True, _norm(d * eta_i_over_t + nf * (eta_i_over_t * ci - sqrt(Float32(1) - s2))))

def _brute_force(n_samples: Int) -> Float64:
    var x0 = _x0()
    var axis = _norm(-x0)
    var d2 = Float64(dot(x0, x0))
    var cos_max = Float32(sqrt(1.0 - Float64(C_R * C_R) / d2))
    var pdf = Float64(1.0) / (2.0 * Float64(PI) * Float64(1 - cos_max))
    var up = _v(0, 1, 0)
    var e1 = _norm(cross(axis, up)); var e2 = cross(axis, e1)
    var pcg = PCG32(UInt64(4242), UInt64(7))
    var acc = Float64(0)
    for _ in range(n_samples):
        var ct = Float32(1) - pcg.next_float() * (Float32(1) - cos_max)
        var st = sqrt(max(Float32(0), Float32(1) - ct * ct))
        var ph = Float32(2) * PI * pcg.next_float()
        var d = axis * ct + e1 * (st * cos(ph)) + e2 * (st * sin(ph))
        var b = dot(x0, d)
        var disc = b * b - (dot(x0, x0) - C_R * C_R)
        if disc <= Float32(0):
            continue
        var p1 = x0 + d * (-b - sqrt(disc))
        var n1 = p1 * (Float32(1) / C_R)
        var ci = -dot(d, n1)
        var r = d + n1 * (Float32(2) * ci)
        if r[1] <= Float32(0):
            continue
        var f = fr_dielectric(ci, BALL_IOR)
        var p2 = p1 + r * ((SLAB_Y0 - p1[1]) / r[1])
        var a = _refract_up(r, Float32(1) / SLAB_IOR)
        f *= Float32(1) - fr_dielectric(r[1], SLAB_IOR)
        var p3 = p2 + a[1] * ((SLAB_Y1 - p2[1]) / a[1][1])
        var o = _refract_up(a[1], SLAB_IOR)
        if not o[0]:
            continue
        f *= Float32(1) - fr_dielectric(a[1][1], Float32(1) / SLAB_IOR)
        var p4 = p3 + o[1] * ((LAMP_Y - p3[1]) / o[1][1])
        if abs(p4[0]) > LAMP_H or abs(p4[2]) > LAMP_H:
            continue
        acc += Float64(f * dot(axis, d)) / pdf
    return acc / Float64(n_samples)

def _mnee_estimate(n_samples: Int) -> Float64:
    var x0 = _x0()
    var axis = _norm(-x0)
    var area = Float64(4 * LAMP_H * LAMP_H)
    var pcg = PCG32(UInt64(99), UInt64(3))
    var acc = Float64(0)
    for _ in range(n_samples):
        var y = Vec3f((pcg.next_float() * Float32(2) - Float32(1)) * LAMP_H, LAMP_Y,
                      (pcg.next_float() * Float32(2) - Float32(1)) * LAMP_H)
        var c = _chain(x0, y)
        var s = _solve(x0, y, c[0].copy(), 3)
        if not s[0]:
            continue
        var w = _norm(s[1][0].pos - x0)
        acc += Float64(s[2] * s[3] * dot(axis, w)) * area
    return acc / Float64(n_samples)

def test_reflect_slab_mnee_matches_brute_force() raises:
    var bf = _brute_force(300000)
    var mn = _mnee_estimate(4000)
    print("irradiance through ball reflection + slab: brute force", bf, "MNEE", mn, "ratio", mn / bf)
    assert_true(bf > 0.0)
    assert_true(abs(mn / bf - 1.0) < 0.03)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
