from std.math import abs
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import RGB, INV_PI
from gonzales.reservoir import ReservoirState
from gonzales.restir_di import DIReservoir, di_reservoir_init, di_target_pdf

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# ── di_reservoir_init ────────────────────────────────────────────────────────

def test_di_reservoir_init_has_no_winner() raises:
    var res = di_reservoir_init()
    assert_true(res.light_idx == Int32(-1))
    assert_true(_close(res.state.w_sum, Float32(0.0)))
    assert_true(_close(res.state.m, Float32(0.0)))

# ── di_target_pdf ────────────────────────────────────────────────────────────

def test_di_target_pdf_light_directly_above_matches_hand_computation() raises:
    """Flat surface at origin (normal +Y), light 2 units straight up with
    its normal facing straight down at the surface -- cos_s=1, cos_l=1,
    G=1/dist^2=0.25. p_hat = luminance(alb * le * (1/pi) * G)."""
    var hit_point = SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8), Float32(0.8), Float32(0.8))
    var sample_point = SIMD[DType.float32, 3](Float32(0), Float32(2), Float32(0))
    var light_normal = SIMD[DType.float32, 3](Float32(0), Float32(-1), Float32(0))
    var le = RGB(Float32(10.0), Float32(10.0), Float32(10.0))
    var p_hat = di_target_pdf(hit_point, normal, alb, sample_point, light_normal, le)
    var g = Float32(1.0) / Float32(4.0)  # dist=2 -> dist^2=4
    var expected_channel = Float32(0.8) * Float32(10.0) * INV_PI * g
    # luminance of an equal-channel RGB is just that channel's value
    # (0.2126+0.7152+0.0722 == 1.0).
    assert_true(_close(p_hat, expected_channel))

def test_di_target_pdf_zero_when_light_below_surface_horizon() raises:
    """Surface normal +Y, light BELOW the surface (sample_point.y < 0) --
    cos_s <= 0, must return exactly 0, not a negative or NaN value."""
    var hit_point = SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var sample_point = SIMD[DType.float32, 3](Float32(0), Float32(-2), Float32(0))
    var light_normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))
    var le = RGB(Float32(10.0))
    var p_hat = di_target_pdf(hit_point, normal, alb, sample_point, light_normal, le)
    assert_true(_close(p_hat, Float32(0.0)))

def test_di_target_pdf_zero_when_light_faces_away() raises:
    """Light geometrically above the surface but its OWN normal points
    further away (toward the surface's side, not down at it) -- cos_l <= 0,
    the light's back face is turned toward the shading point."""
    var hit_point = SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var sample_point = SIMD[DType.float32, 3](Float32(0), Float32(2), Float32(0))
    var light_normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))  # facing up, away from hit_point
    var le = RGB(Float32(10.0))
    var p_hat = di_target_pdf(hit_point, normal, alb, sample_point, light_normal, le)
    assert_true(_close(p_hat, Float32(0.0)))

def test_di_target_pdf_zero_at_degenerate_distance() raises:
    var hit_point = SIMD[DType.float32, 3](Float32(1), Float32(1), Float32(1))
    var normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var sample_point = SIMD[DType.float32, 3](Float32(1), Float32(1), Float32(1))
    var light_normal = SIMD[DType.float32, 3](Float32(0), Float32(-1), Float32(0))
    var le = RGB(Float32(10.0))
    var p_hat = di_target_pdf(hit_point, normal, alb, sample_point, light_normal, le)
    assert_true(_close(p_hat, Float32(0.0)))

def test_di_target_pdf_closer_light_gives_higher_value() raises:
    """Inverse-square falloff: halving the distance should roughly
    quadruple p_hat (same angles, only G's 1/dist^2 term changes)."""
    var hit_point = SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var normal = SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))
    var alb = RGB(Float32(0.8))
    var light_normal = SIMD[DType.float32, 3](Float32(0), Float32(-1), Float32(0))
    var le = RGB(Float32(10.0))
    var far = di_target_pdf(hit_point, normal, alb, SIMD[DType.float32, 3](Float32(0), Float32(4), Float32(0)), light_normal, le)
    var near = di_target_pdf(hit_point, normal, alb, SIMD[DType.float32, 3](Float32(0), Float32(2), Float32(0)), light_normal, le)
    assert_true(near > far * Float32(3.9) and near < far * Float32(4.1))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
