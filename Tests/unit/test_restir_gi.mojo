from std.math import abs
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import RGB, INV_PI
from gonzales.reservoir import ReservoirState
from gonzales.restir_gi import GIReservoir, gi_reservoir_init, gi_target_pdf

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# ── gi_reservoir_init ────────────────────────────────────────────────────────

def test_gi_reservoir_init_has_no_winner() raises:
    var res = gi_reservoir_init()
    assert_true(res.valid == Int8(0))
    assert_true(res.recon_is_delta == Int8(0))
    assert_true(_close(res.state.w_sum, Float32(0.0)))
    assert_true(_close(res.state.m, Float32(0.0)))

# ── gi_target_pdf ────────────────────────────────────────────────────────────
# Same physical setup as test_restir_di.mojo's di_target_pdf tests, since
# the two target functions share the exact same G-term shape -- only the
# "light point/normal/Le" role is renamed to "reconnection point/normal/Lo".

def test_gi_target_pdf_recon_directly_above_matches_hand_computation() raises:
    """Flat surface at origin (normal +Y), reconnection vertex 2 units
    straight up with its normal facing straight down at x1 -- cos_x1=1,
    cos_x2=1, G=1/dist^2=0.25. p_hat = luminance(alb * lo * (1/pi) * G)."""
    var hit_point = SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8), Float32(0.8), Float32(0.8))
    var recon_point = SIMD[DType.float32, 3](Float32(0), Float32(2), Float32(0))
    var recon_normal = SIMD[DType.float32, 3](Float32(0), Float32(-1), Float32(0))
    var lo = RGB(Float32(10.0), Float32(10.0), Float32(10.0))
    var p_hat = gi_target_pdf(hit_point, normal, alb, recon_point, recon_normal, lo)
    var g = Float32(1.0) / Float32(4.0)  # dist=2 -> dist^2=4
    var expected_channel = Float32(0.8) * Float32(10.0) * INV_PI * g
    assert_true(_close(p_hat, expected_channel))

def test_gi_target_pdf_zero_when_recon_below_surface_horizon() raises:
    """Surface normal +Y, reconnection point BELOW the surface -- cos_x1 <=
    0, must return exactly 0."""
    var hit_point = SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var recon_point = SIMD[DType.float32, 3](Float32(0), Float32(-2), Float32(0))
    var recon_normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))
    var lo = RGB(Float32(10.0))
    var p_hat = gi_target_pdf(hit_point, normal, alb, recon_point, recon_normal, lo)
    assert_true(_close(p_hat, Float32(0.0)))

def test_gi_target_pdf_zero_when_recon_faces_away() raises:
    """Reconnection point geometrically above x1, but its OWN normal points
    further away (toward x1's side) -- cos_x2 <= 0, its back face is turned
    toward x1."""
    var hit_point = SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var recon_point = SIMD[DType.float32, 3](Float32(0), Float32(2), Float32(0))
    var recon_normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))  # facing up, away from x1
    var lo = RGB(Float32(10.0))
    var p_hat = gi_target_pdf(hit_point, normal, alb, recon_point, recon_normal, lo)
    assert_true(_close(p_hat, Float32(0.0)))

def test_gi_target_pdf_zero_at_degenerate_distance() raises:
    var hit_point = SIMD[DType.float32, 3](Float32(1), Float32(1), Float32(1))
    var normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var recon_point = SIMD[DType.float32, 3](Float32(1), Float32(1), Float32(1))
    var recon_normal = SIMD[DType.float32, 3](Float32(0), Float32(-1), Float32(0))
    var lo = RGB(Float32(10.0))
    var p_hat = gi_target_pdf(hit_point, normal, alb, recon_point, recon_normal, lo)
    assert_true(_close(p_hat, Float32(0.0)))

def test_gi_target_pdf_closer_recon_point_gives_higher_value() raises:
    """Inverse-square falloff: halving the distance should roughly
    quadruple p_hat (same angles, only G's 1/dist^2 term changes)."""
    var hit_point = SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var recon_normal = SIMD[DType.float32, 3](Float32(0), Float32(-1), Float32(0))
    var lo = RGB(Float32(10.0))
    var far = gi_target_pdf(hit_point, normal, alb, SIMD[DType.float32, 3](Float32(0), Float32(4), Float32(0)), recon_normal, lo)
    var near = gi_target_pdf(hit_point, normal, alb, SIMD[DType.float32, 3](Float32(0), Float32(2), Float32(0)), recon_normal, lo)
    assert_true(near > far * Float32(3.9) and near < far * Float32(4.1))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
