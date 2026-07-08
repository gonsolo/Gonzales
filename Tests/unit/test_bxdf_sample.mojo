from std.math import abs, sqrt
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import RGB, Vec3f, reflect, fr_dielectric, PI, INV_PI
from gonzales.bxdf import (
    GeomContext, Material_C, BxDFFlags,
    bxdf_sample_conductor, bxdf_sample_dielectric, bxdf_sample_thin_dielectric,
    bxdf_sample_diffuse, bxdf_pdf_diffuse, bxdf_sample_diffuse_transmit,
)

comptime EPS: Float32 = 1e-3

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _simd_close(a: SIMD[DType.float32, 3], b: SIMD[DType.float32, 3]) -> Bool:
    return _close(a[0], b[0]) and _close(a[1], b[1]) and _close(a[2], b[2])

def _simd_len(v: SIMD[DType.float32, 3]) -> Float32:
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

def _make_material(albedo: RGB, roughU: Float32, roughV: Float32) -> Material_C:
    return Material_C(Int8(0), Int8(0), Int8(0), Int8(0), albedo, RGB(Float32(0.0)),
        Int32(-1), roughU, roughV, Int32(-1), Int32(-1), Int32(-1),
        RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(1.0), Float32(1.0))

def _make_gc(
    normal: SIMD[DType.float32, 3], wo: SIMD[DType.float32, 3],
    tangent: SIMD[DType.float32, 3], bitangent: SIMD[DType.float32, 3],
) -> GeomContext:
    return GeomContext(normal, normal, SIMD[DType.float32, 3](0.0, 0.0, 0.0), wo,
        tangent, bitangent, RGB(Float32(0.0)), Float32(0.0))

def NORMAL_Z() -> SIMD[DType.float32, 3]:
    return SIMD[DType.float32, 3](0.0, 0.0, 1.0)

def TANGENT_X() -> SIMD[DType.float32, 3]:
    return SIMD[DType.float32, 3](1.0, 0.0, 0.0)

def BITANGENT_Y() -> SIMD[DType.float32, 3]:
    return SIMD[DType.float32, 3](0.0, 1.0, 0.0)

# ── bxdf_sample_conductor ────────────────────────────────────────────────────

def test_conductor_mirror_obeys_reflection_law() raises:
    """RoughU=roughV=0 takes the perfect-mirror branch — wi must exactly
    match the geometric reflection law, independent of the BxDF machinery."""
    var wo = Vec3f(1.0, 0.0, 1.0).normalize().to_simd()
    var gc = _make_gc(NORMAL_Z(), wo, TANGENT_X(), BITANGENT_Y())
    var mat = _make_material(RGB(Float32(0.9)), Float32(0.0), Float32(0.0))
    var bs = bxdf_sample_conductor(gc, mat, Float32(0.5), Float32(0.5))

    var expected = reflect(Vec3f(wo[0], wo[1], wo[2]), Vec3f(0.0, 0.0, 1.0)).to_simd()
    assert_true(_simd_close(bs.wi, expected))
    assert_true(bs.pdf == Float32(1.0))
    assert_true(Int(bs.is_valid) == 1)
    assert_true((Int(bs.flags) & Int(BxDFFlags.delta)) != 0)
    assert_true((Int(bs.flags) & Int(BxDFFlags.reflect)) != 0)

def test_conductor_mirror_fresnel_matches_schlick() raises:
    var wo = Vec3f(1.0, 0.0, 1.0).normalize().to_simd()
    var gc = _make_gc(NORMAL_Z(), wo, TANGENT_X(), BITANGENT_Y())
    var albedo = RGB(Float32(0.5), Float32(0.7), Float32(0.9))
    var mat = _make_material(albedo, Float32(0.0), Float32(0.0))
    var bs = bxdf_sample_conductor(gc, mat, Float32(0.5), Float32(0.5))

    var cos_i = wo[2]  # dot(wo, normal) with normal=(0,0,1)
    var one_m = Float32(1.0) - cos_i
    var schlick = one_m * one_m * one_m * one_m * one_m
    var expected_r = albedo.r + (Float32(1.0) - albedo.r) * schlick
    assert_true(_close(bs.f.r, expected_r))

def test_conductor_rough_sample_is_unit_and_valid() raises:
    """Rough (GGX) branch: regardless of the exact sampled direction, the
    invariants that must always hold are unit length and (when valid)
    staying on the same side of the normal as a reflection."""
    var gc = _make_gc(NORMAL_Z(), NORMAL_Z(), TANGENT_X(), BITANGENT_Y())
    var mat = _make_material(RGB(Float32(0.8)), Float32(0.3), Float32(0.3))
    var bs = bxdf_sample_conductor(gc, mat, Float32(0.37), Float32(0.61))
    assert_true(_close(_simd_len(bs.wi), Float32(1.0)))
    if Int(bs.is_valid) == 1:
        assert_true(bs.wi[2] > Float32(0.0))
        assert_true((Int(bs.flags) & Int(BxDFFlags.glossy)) != 0)

# ── bxdf_sample_dielectric ──────────────────────────────────────────────────

def test_dielectric_total_internal_reflection_always_reflects() raises:
    """Grazing ray exiting a denser medium: TIR must force the reflect
    branch even when u_reflect is chosen to almost never reflect."""
    var geom_normal = NORMAL_Z()
    var ray_dir = Vec3f(0.99, 0.0, 0.1411).normalize().to_simd()  # dot(ray,n) > 0: exiting
    var (bs, _) = bxdf_sample_dielectric(geom_normal, ray_dir, Float32(1.5), False, Float32(0.99))
    assert_true((Int(bs.flags) & Int(BxDFFlags.reflect)) != 0)
    assert_true(Int(bs.is_valid) == 1)

def test_dielectric_normal_incidence_transmits_straight_through() raises:
    """At normal incidence with u_reflect forced past the (small) Fresnel
    reflectance, the ray must transmit essentially undeviated."""
    var geom_normal = NORMAL_Z()
    var ray_dir = SIMD[DType.float32, 3](0.0, 0.0, -1.0)  # straight in, entering
    var (bs, _) = bxdf_sample_dielectric(geom_normal, ray_dir, Float32(1.5), True, Float32(0.999))
    assert_true((Int(bs.flags) & Int(BxDFFlags.transmit)) != 0)
    assert_true(_simd_close(bs.wi, ray_dir))

# ── bxdf_sample_thin_dielectric ─────────────────────────────────────────────

def test_thin_dielectric_transmit_keeps_original_direction() raises:
    """A thin slab's transmitted ray is defined to keep the incoming
    direction unchanged (entry/exit refractions cancel) — this must hold
    exactly whenever the transmit branch is taken."""
    var geom_normal = NORMAL_Z()
    var ray_dir = Vec3f(0.2, 0.0, -0.9798).normalize().to_simd()
    var (bs, _) = bxdf_sample_thin_dielectric(geom_normal, ray_dir, Float32(1.5), Float32(0.999))
    assert_true((Int(bs.flags) & Int(BxDFFlags.transmit)) != 0)
    assert_true(_simd_close(bs.wi, ray_dir))

def test_thin_dielectric_reflect_obeys_reflection_law() raises:
    var geom_normal = NORMAL_Z()
    var ray_dir = Vec3f(0.2, 0.0, -0.9798).normalize().to_simd()
    var (bs, normal) = bxdf_sample_thin_dielectric(geom_normal, ray_dir, Float32(1.5), Float32(0.0))
    assert_true((Int(bs.flags) & Int(BxDFFlags.reflect)) != 0)
    var expected = reflect(-Vec3f(ray_dir[0], ray_dir[1], ray_dir[2]), Vec3f(normal[0], normal[1], normal[2])).to_simd()
    assert_true(_simd_close(bs.wi, expected))

# ── bxdf_sample_diffuse ──────────────────────────────────────────────────────

def test_diffuse_pdf_matches_cosine_law() raises:
    """The sampled pdf must equal the closed-form cosine-hemisphere pdf
    evaluated at the actual sampled direction's angle to the normal —
    they're coupled by construction and must never drift apart."""
    var gc = _make_gc(NORMAL_Z(), NORMAL_Z(), TANGENT_X(), BITANGENT_Y())
    var bs = bxdf_sample_diffuse(gc, RGB(Float32(0.5)), Float32(0.3), Float32(0.6))
    var cos_wi = bs.wi[2]  # dot(wi, normal) with normal=(0,0,1)
    assert_true(_close(bs.pdf, bxdf_pdf_diffuse(cos_wi)))

def test_diffuse_f_is_albedo_over_pi_independent_of_direction() raises:
    var gc = _make_gc(NORMAL_Z(), NORMAL_Z(), TANGENT_X(), BITANGENT_Y())
    var alb = RGB(Float32(0.2), Float32(0.4), Float32(0.6))
    var bs = bxdf_sample_diffuse(gc, alb, Float32(0.1), Float32(0.9))
    assert_true(_close(bs.f.r, alb.r * INV_PI))
    assert_true(_close(bs.f.g, alb.g * INV_PI))
    assert_true(_close(bs.f.b, alb.b * INV_PI))

# ── bxdf_sample_diffuse_transmit ─────────────────────────────────────────────

def test_diffuse_transmit_degenerate_when_both_lobes_black() raises:
    var (bs, _, _, _, _) = bxdf_sample_diffuse_transmit(
        NORMAL_Z(), RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(0.5), Float32(0.3), Float32(0.6))
    assert_true(bs.pdf == Float32(0.0))
    assert_true(Int(bs.is_valid) == 0)

def test_diffuse_transmit_picks_the_only_nonzero_lobe() raises:
    """Pure-transmission material (refl=0): must always choose transmit,
    regardless of u_lobe, with lobe_w == 1 (no reweighting needed since
    there is only one possible lobe)."""
    var (_, bounce_normal, _, lobe_w, chosen) = bxdf_sample_diffuse_transmit(
        NORMAL_Z(), RGB(Float32(0.0)), RGB(Float32(0.6)), Float32(0.999), Float32(0.3), Float32(0.6))
    assert_false(chosen)
    assert_true(_close(lobe_w, Float32(1.0)))
    assert_true(_simd_close(bounce_normal, -NORMAL_Z()))

def test_diffuse_transmit_equal_lobes_give_weight_two() raises:
    """Equal reflect/transmit luma: pr/total == 0.5, so whichever lobe is
    picked, lobe_w = total/luma(chosen) == 2.0 exactly — independent of
    which lobe u_lobe happens to select."""
    var refl = RGB(Float32(0.4))
    var trans = RGB(Float32(0.4))
    var (_, _, _, w_a, chosen_a) = bxdf_sample_diffuse_transmit(
        NORMAL_Z(), refl, trans, Float32(0.1), Float32(0.3), Float32(0.6))  # picks reflect
    var (_, _, _, w_b, chosen_b) = bxdf_sample_diffuse_transmit(
        NORMAL_Z(), refl, trans, Float32(0.9), Float32(0.3), Float32(0.6))  # picks transmit
    assert_true(chosen_a)
    assert_false(chosen_b)
    assert_true(_close(w_a, Float32(2.0)))
    assert_true(_close(w_b, Float32(2.0)))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
