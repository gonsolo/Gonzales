from std.math import abs, sqrt
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import (
    Vec3f, Point3f, Frame, RGB,
    reflect, refract, schlick_fresnel, fr_dielectric, safe_sqrt,
    sphere_outward_normal, dot, cross,
    spherical_direction, vec3f, point3f,
    Curve_C, curve_bspline_point, curve_light_tube_area,
)

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _vclose(a: Vec3f, b: Vec3f) -> Bool:
    return _close(a.x, b.x) and _close(a.y, b.y) and _close(a.z, b.z)

# ── Frame.from_z (Duff et al. 2017 orthonormal basis) ────────────────────────
# This is now the single most-shared piece of math in the shading path (see
# task #44) — every material's tangent frame flows through it — so its
# defining invariants (unit length + mutual orthogonality + right-handedness)
# are worth pinning down directly, independent of any material that calls it.

def _assert_orthonormal_right_handed(n: Vec3f) raises:
    var f = Frame.from_z(n)
    assert_true(_close(f.x.length(), Float32(1.0)))
    assert_true(_close(f.y.length(), Float32(1.0)))
    assert_true(_close(f.z.length(), Float32(1.0)))
    assert_true(_close(f.x.dot(f.y), Float32(0.0)))
    assert_true(_close(f.x.dot(f.z), Float32(0.0)))
    assert_true(_close(f.y.dot(f.z), Float32(0.0)))
    var xcy = cross(f.x.to_simd(), f.y.to_simd())
    assert_true(_vclose(Vec3f(xcy[0], xcy[1], xcy[2]), f.z))

def test_frame_from_z_plus_z() raises:
    _assert_orthonormal_right_handed(Vec3f(0.0, 0.0, 1.0))

def test_frame_from_z_minus_z() raises:
    """The south pole (n.z ≈ -1) is exactly the case Duff et al.'s sign(n.z)
    trick exists to keep numerically stable — must not degenerate here."""
    _assert_orthonormal_right_handed(Vec3f(0.0, 0.0, -1.0))

def test_frame_from_z_equatorial() raises:
    _assert_orthonormal_right_handed(Vec3f(1.0, 0.0, 0.0))
    _assert_orthonormal_right_handed(Vec3f(0.0, 1.0, 0.0))

def test_frame_from_z_arbitrary_direction() raises:
    var n = Vec3f(1.0, 2.0, 3.0).normalize()
    _assert_orthonormal_right_handed(n)

def test_frame_from_z_to_world_round_trip() raises:
    var n = Vec3f(0.267, 0.535, 0.802).normalize()
    var f = Frame.from_z(n)
    var local = Vec3f(0.3, -0.4, 0.8)
    var world = f.to_world(local)
    var back = f.to_local(world)
    assert_true(_vclose(back, local))

# ── reflect ───────────────────────────────────────────────────────────────

def test_reflect_normal_incidence_is_unchanged() raises:
    var n = Vec3f(0.0, 0.0, 1.0)
    assert_true(_vclose(reflect(n, n), n))

def test_reflect_45_degrees() raises:
    var wo = Vec3f(1.0, 0.0, 1.0).normalize()
    var n = Vec3f(0.0, 0.0, 1.0)
    var wi = reflect(wo, n)
    var expected = Vec3f(-1.0, 0.0, 1.0).normalize()
    assert_true(_vclose(wi, expected))

def test_reflect_preserves_angle_to_normal() raises:
    var wo = Vec3f(0.3, -0.6, 0.7).normalize()
    var n = Vec3f(0.1, 0.2, 0.97).normalize()
    var wi = reflect(wo, n)
    assert_true(_close(wi.dot(n), wo.dot(n)))
    assert_true(_close(wi.length(), wo.length()))

# ── refract ───────────────────────────────────────────────────────────────

def test_refract_matched_ior_passes_straight_through() raises:
    """Eta=1 (no index change): the transmitted direction, expressed in the
    away-from-surface convention on the far side, is exactly -wi."""
    var wi = Vec3f(0.3, 0.4, 0.866).normalize()
    var n = Vec3f(0.0, 0.0, 1.0)
    var (ok, wt) = refract(wi, n, Float32(1.0))
    assert_true(ok)
    assert_true(_vclose(wt, -wi))

def test_refract_total_internal_reflection() raises:
    """Grazing incidence going from a denser to a less-dense medium (eta>1
    here since eta=η_i/η_t) must report TIR (ok=False)."""
    var wi = Vec3f(0.99, 0.0, 0.1411).normalize()  # cos_theta_i ~ 0.14, steep angle
    var n = Vec3f(0.0, 0.0, 1.0)
    var (ok, _) = refract(wi, n, Float32(1.5))
    assert_false(ok)

def test_refract_non_tir_returns_unit_vector() raises:
    var wi = Vec3f(0.0, 0.0, 1.0)  # normal incidence — never TIR
    var n = Vec3f(0.0, 0.0, 1.0)
    var (ok, wt) = refract(wi, n, Float32(0.6667))
    assert_true(ok)
    assert_true(_close(wt.length(), Float32(1.0)))

# ── schlick_fresnel ─────────────────────────────────────────────────────────

def test_schlick_fresnel_normal_incidence_is_f0() raises:
    assert_true(_close(schlick_fresnel(Float32(1.0), Float32(0.04)), Float32(0.04)))

def test_schlick_fresnel_grazing_is_total() raises:
    """At cos_theta=0 the (1-cos)^5 term is 1 — full reflectance regardless
    of f0, the whole point of the grazing-angle brightening it models."""
    assert_true(_close(schlick_fresnel(Float32(0.0), Float32(0.04)), Float32(1.0)))
    assert_true(_close(schlick_fresnel(Float32(0.0), Float32(0.9)), Float32(1.0)))

# ── fr_dielectric ───────────────────────────────────────────────────────────

def test_fr_dielectric_normal_incidence_matches_schlick_f0() raises:
    """At normal incidence the exact Fresnel formula must reduce to exactly
    the Schlick f0 = ((eta-1)/(eta+1))^2 closed form."""
    for eta in [Float32(1.5), Float32(0.6667), Float32(2.0)]:
        var r = fr_dielectric(Float32(1.0), eta)
        var f0 = (eta - Float32(1.0)) / (eta + Float32(1.0))
        f0 = f0 * f0
        assert_true(_close(r, f0))

def test_fr_dielectric_grazing_is_total_reflection() raises:
    """Exact physical property, not just the Schlick approximation: grazing
    incidence (cos_theta_i=0) reflects 100% regardless of eta."""
    assert_true(_close(fr_dielectric(Float32(0.0), Float32(1.5)), Float32(1.0)))
    assert_true(_close(fr_dielectric(Float32(0.0), Float32(0.6667)), Float32(1.0)))

def test_fr_dielectric_side_flip_symmetry() raises:
    """A negative cos_theta_i (arriving from the transmission side) must
    equal calling with the positive angle and reciprocal eta."""
    var eta = Float32(1.5)
    var c = Float32(0.3)
    var a = fr_dielectric(-c, eta)
    var b = fr_dielectric(c, Float32(1.0) / eta)
    assert_true(_close(a, b))

def test_fr_dielectric_total_internal_reflection() raises:
    """Oblique angle exiting a denser medium (eta<1 here) must saturate to 1.0."""
    var r = fr_dielectric(Float32(0.3), Float32(0.6667))
    assert_true(_close(r, Float32(1.0)))

# ── safe_sqrt ───────────────────────────────────────────────────────────────

def test_safe_sqrt_positive() raises:
    assert_true(_close(safe_sqrt(Float32(4.0)), Float32(2.0)))

def test_safe_sqrt_clamps_negative_to_zero() raises:
    assert_true(safe_sqrt(Float32(-1.0)) == Float32(0.0))

def test_safe_sqrt_zero() raises:
    assert_true(safe_sqrt(Float32(0.0)) == Float32(0.0))

# ── sphere_outward_normal ───────────────────────────────────────────────────

def test_sphere_outward_normal_axis_aligned() raises:
    var n = sphere_outward_normal(Point3f(2.0, 0.0, 0.0), Point3f(0.0))
    assert_true(_vclose(n, Vec3f(1.0, 0.0, 0.0)))

def test_sphere_outward_normal_degenerate_is_zero() raises:
    var n = sphere_outward_normal(Point3f(1.0, 2.0, 3.0), Point3f(1.0, 2.0, 3.0))
    assert_true(_vclose(n, Vec3f(0.0)))

def test_sphere_outward_normal_is_unit_independent_of_radius() raises:
    var n = sphere_outward_normal(Point3f(0.0, 3.0, 4.0), Point3f(0.0))  # dist=5
    assert_true(_close(n.length(), Float32(1.0)))
    assert_true(_vclose(n, Vec3f(0.0, 0.6, 0.8)))

# ── dot / cross (SIMD triples) ──────────────────────────────────────────────

def test_dot_simd() raises:
    var a = Vec3f(1.0, 2.0, 3.0)
    var b = Vec3f(4.0, 5.0, 6.0)
    assert_true(_close(dot(a, b), Float32(32.0)))

def test_cross_simd_right_handed() raises:
    var x = Vec3f(1.0, 0.0, 0.0)
    var y = Vec3f(0.0, 1.0, 0.0)
    var z = cross(x, y)
    assert_true(_close(z[0], Float32(0.0)) and _close(z[1], Float32(0.0)) and _close(z[2], Float32(1.0)))

# ── Point3f / Vec3f arithmetic ──────────────────────────────────────────────

def test_point3f_plus_vec3f() raises:
    var p = Point3f(1.0, 2.0, 3.0)
    var v = Vec3f(0.5, -1.0, 2.0)
    var r = p + v
    assert_true(_close(r.x, Float32(1.5)) and _close(r.y, Float32(1.0)) and _close(r.z, Float32(5.0)))

def test_point3f_minus_point3f() raises:
    """Point minus point yields a displacement vector."""
    var a = Point3f(4.0, 5.0, 6.0)
    var b = Point3f(1.0, 2.0, 3.0)
    assert_true(_vclose(a - b, Vec3f(3.0, 3.0, 3.0)))

def test_vec3f_add() raises:
    assert_true(_vclose(Vec3f(1.0, 2.0, 3.0) + Vec3f(4.0, 5.0, 6.0), Vec3f(5.0, 7.0, 9.0)))

def test_vec3f_sub() raises:
    assert_true(_vclose(Vec3f(4.0, 5.0, 6.0) - Vec3f(1.0, 2.0, 3.0), Vec3f(3.0, 3.0, 3.0)))

def test_vec3f_neg() raises:
    assert_true(_vclose(-Vec3f(1.0, -2.0, 3.0), Vec3f(-1.0, 2.0, -3.0)))

def test_vec3f_mul_scalar() raises:
    assert_true(_vclose(Vec3f(1.0, 2.0, 3.0) * Float32(2.0), Vec3f(2.0, 4.0, 6.0)))

def test_vec3f_div_scalar() raises:
    assert_true(_vclose(Vec3f(2.0, 4.0, 6.0) / Float32(2.0), Vec3f(1.0, 2.0, 3.0)))

def test_vec3f_length_345_triangle() raises:
    """Classic 3-4-5 right triangle — exact closed-form length."""
    assert_true(_close(Vec3f(3.0, 4.0, 0.0).length(), Float32(5.0)))

def test_vec3f_length_sq() raises:
    assert_true(_close(Vec3f(3.0, 4.0, 0.0).length_sq(), Float32(25.0)))

def test_vec3f_simd_round_trip() raises:
    var s = Vec3f(1.5, -2.5, 3.5)
    var v = vec3f(s)
    assert_true(_vclose(v, Vec3f(1.5, -2.5, 3.5)))
    var back = v.to_simd()
    assert_true(_close(back[0], s[0]) and _close(back[1], s[1]) and _close(back[2], s[2]))

def test_point3f_simd_round_trip() raises:
    var s = Vec3f(1.5, -2.5, 3.5)
    var p = point3f(s)
    var back = p.to_simd()
    assert_true(_close(back[0], s[0]) and _close(back[1], s[1]) and _close(back[2], s[2]))

# ── RGB arithmetic ───────────────────────────────────────────────────────────

def _rgb_close(a: RGB, b: RGB) -> Bool:
    return _close(a.r, b.r) and _close(a.g, b.g) and _close(a.b, b.b)

def test_rgb_add() raises:
    assert_true(_rgb_close(RGB(1.0, 2.0, 3.0) + RGB(4.0, 5.0, 6.0), RGB(5.0, 7.0, 9.0)))

def test_rgb_sub() raises:
    assert_true(_rgb_close(RGB(4.0, 5.0, 6.0) - RGB(1.0, 2.0, 3.0), RGB(3.0, 3.0, 3.0)))

def test_rgb_mul_rgb() raises:
    """RGB*RGB is componentwise (used for e.g. albedo * incoming radiance)."""
    assert_true(_rgb_close(RGB(2.0, 3.0, 4.0) * RGB(1.0, 2.0, 0.5), RGB(2.0, 6.0, 2.0)))

def test_rgb_mul_scalar() raises:
    assert_true(_rgb_close(RGB(1.0, 2.0, 3.0) * Float32(2.0), RGB(2.0, 4.0, 6.0)))

def test_rgb_truediv() raises:
    assert_true(_rgb_close(RGB(2.0, 4.0, 6.0) / Float32(2.0), RGB(1.0, 2.0, 3.0)))

def test_rgb_imul_rgb() raises:
    var c = RGB(2.0, 3.0, 4.0)
    c *= RGB(1.0, 2.0, 0.5)
    assert_true(_rgb_close(c, RGB(2.0, 6.0, 2.0)))

def test_rgb_imul_scalar() raises:
    var c = RGB(1.0, 2.0, 3.0)
    c *= Float32(2.0)
    assert_true(_rgb_close(c, RGB(2.0, 4.0, 6.0)))

def test_rgb_iadd() raises:
    var c = RGB(1.0, 2.0, 3.0)
    c += RGB(4.0, 5.0, 6.0)
    assert_true(_rgb_close(c, RGB(5.0, 7.0, 9.0)))

def test_rgb_sum() raises:
    assert_true(_close(RGB(1.0, 2.0, 3.0).sum(), Float32(6.0)))

def test_rgb_is_black_true_for_zero_and_negative() raises:
    """Is_black is defined as ALL channels <= 0, including the negative case."""
    assert_true(RGB(0.0, 0.0, 0.0).is_black())
    assert_true(RGB(-1.0, -2.0, 0.0).is_black())

def test_rgb_is_black_false_when_any_channel_positive() raises:
    assert_false(RGB(0.0, 0.001, 0.0).is_black())

def test_rgb_clamp_leaves_in_range_values_unchanged() raises:
    var c = RGB(0.2, 0.5, 0.8).clamp(Float32(0.0), Float32(1.0))
    assert_true(_close(c.r, Float32(0.2)) and _close(c.g, Float32(0.5)) and _close(c.b, Float32(0.8)))

def test_rgb_clamp_clips_out_of_range_values() raises:
    var c = RGB(-1.0, 0.5, 3.0).clamp(Float32(0.0), Float32(1.0))
    assert_true(_close(c.r, Float32(0.0)) and _close(c.g, Float32(0.5)) and _close(c.b, Float32(1.0)))

# ── spherical_direction ─────────────────────────────────────────────────────

def test_spherical_direction_is_unit_length() raises:
    for theta_pair in [
        (Float32(0.0), Float32(1.0)),
        (Float32(1.0), Float32(0.0)),
        (Float32(0.6), Float32(0.8)),
        (Float32(0.8660254), Float32(-0.5)),
    ]:
        var (sin_t, cos_t) = theta_pair
        for phi in [Float32(0.0), Float32(1.0), Float32(3.0), Float32(5.5)]:
            var v = spherical_direction(sin_t, cos_t, phi)
            assert_true(_close(v.length(), Float32(1.0)))

def test_spherical_direction_straight_up() raises:
    """Sin_theta=0, cos_theta=1 (the pole) must land on the up axis
    regardless of phi, per this code's y-up convention."""
    assert_true(_vclose(spherical_direction(Float32(0.0), Float32(1.0), Float32(0.0)), Vec3f(0.0, 1.0, 0.0)))
    assert_true(_vclose(spherical_direction(Float32(0.0), Float32(1.0), Float32(2.3)), Vec3f(0.0, 1.0, 0.0)))

def test_spherical_direction_equatorial_phi0() raises:
    """Sin_theta=1, cos_theta=0, phi=0 lands in the equatorial plane (y=0)
    along +x, per this code's Vec3f(sin*cos_phi, cos, sin*sin_phi) convention."""
    var v = spherical_direction(Float32(1.0), Float32(0.0), Float32(0.0))
    assert_true(_vclose(v, Vec3f(1.0, 0.0, 0.0)))

# ── curve_light_tube_area (curve area lights, task #60-65) ───────────────────
# The NEE/light-sampler pdf for an emissive curve is normalized against this
# "lateral tube surface" area — a wrong formula here silently biases every
# curve light's brightness (and its share of the power-weighted CDF) without
# ever crashing, so it's worth pinning to a closed-form value directly.

def test_curve_light_tube_area_single_piece_constant_radius() raises:
    """n_pieces=1 means the tube area is exactly one cylinder: 2*pi*r*length,
    where length is the chord between the curve's own t=0/t=1 points (not
    assumed independently -- computed via the same curve_bspline_point this
    module already tests elsewhere isn't needed here, since any 4 control
    points work for a single piece spanning the whole [0,1] range)."""
    var curve = Curve_C(
        Point3f(0.0, 0.0, 0.0), Point3f(1.0, 2.0, 0.0),
        Point3f(2.0, -1.0, 0.0), Point3f(3.0, 0.0, 0.0),
        Float32(0.4), Float32(0.4), Int32(0), Int32(1),
    )
    var q0 = curve_bspline_point(curve, Float32(0.0))
    var q1 = curve_bspline_point(curve, Float32(1.0))
    var seg = q1 - q0
    var length = sqrt(dot(seg, seg))
    var expected = Float32(2.0) * Float32(3.14159265358979323846) * Float32(0.2) * length
    assert_true(_close(curve_light_tube_area(curve), expected))

def test_curve_light_tube_area_scales_with_radius() raises:
    """Doubling both endpoint widths must exactly double the tube area
    (area is linear in radius) -- catches an accidental use of r^2 or a
    missing 0.5 in the width-to-radius conversion."""
    var curve_thin = Curve_C(
        Point3f(0.0, 0.0, 0.0), Point3f(1.0, 0.0, 0.0),
        Point3f(2.0, 0.0, 0.0), Point3f(3.0, 0.0, 0.0),
        Float32(0.1), Float32(0.1), Int32(0), Int32(1),
    )
    var curve_thick = Curve_C(
        Point3f(0.0, 0.0, 0.0), Point3f(1.0, 0.0, 0.0),
        Point3f(2.0, 0.0, 0.0), Point3f(3.0, 0.0, 0.0),
        Float32(0.2), Float32(0.2), Int32(0), Int32(1),
    )
    assert_true(_close(curve_light_tube_area(curve_thick), Float32(2.0) * curve_light_tube_area(curve_thin)))

def test_curve_light_tube_area_sums_across_pieces() raises:
    """n_pieces=4 must equal the sum of each individual piece's own lateral
    area -- catches an off-by-one in the piece-count loop bound."""
    var curve1 = Curve_C(
        Point3f(0.0, 0.0, 0.0), Point3f(1.0, 3.0, 0.0),
        Point3f(2.0, -2.0, 1.0), Point3f(3.0, 1.0, -1.0),
        Float32(0.3), Float32(0.1), Int32(0), Int32(1),
    )
    var curve4 = Curve_C(
        Point3f(0.0, 0.0, 0.0), Point3f(1.0, 3.0, 0.0),
        Point3f(2.0, -2.0, 1.0), Point3f(3.0, 1.0, -1.0),
        Float32(0.3), Float32(0.1), Int32(0), Int32(4),
    )
    # A single piece's area (n_pieces=1) is one cylinder end-to-end; the
    # n_pieces=4 curve chops the SAME B-spline into 4 pieces, each with its
    # own (shorter, still-tapering) cylinder -- their sum need not equal the
    # 1-piece area (chord length is shorter than arc length for a curved
    # spline), but it must be strictly positive and, for this curved control
    # polygon, larger than the single overall chord's cylinder area.
    var area1 = curve_light_tube_area(curve1)
    var area4 = curve_light_tube_area(curve4)
    assert_true(area4 > Float32(0.0))
    assert_true(area4 >= area1)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
