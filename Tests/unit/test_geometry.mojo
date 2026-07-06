from std.math import abs, sqrt
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import (
    Vec3f, Point3f, Frame,
    reflect, refract, schlick_fresnel, fr_dielectric, safe_sqrt,
    sphere_outward_normal, dot, cross,
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
    var a = SIMD[DType.float32, 3](1.0, 2.0, 3.0)
    var b = SIMD[DType.float32, 3](4.0, 5.0, 6.0)
    assert_true(_close(dot(a, b), Float32(32.0)))

def test_cross_simd_right_handed() raises:
    var x = SIMD[DType.float32, 3](1.0, 0.0, 0.0)
    var y = SIMD[DType.float32, 3](0.0, 1.0, 0.0)
    var z = cross(x, y)
    assert_true(_close(z[0], Float32(0.0)) and _close(z[1], Float32(0.0)) and _close(z[2], Float32(1.0)))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
