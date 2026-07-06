# Unit tests for pure formatting/normalization helpers in rendering.mojo:
# _fmt_f1/fmt_time/progress_str (progress-bar string formatting) and
# normalize_film (TileResult_C accumulator -> per-pixel beauty/albedo
# arrays). render_tile/render_all_tiles/render_aux_buffers all need a real
# built BVH and SceneDescriptor2_C (and, for render_tile's medium sampling
# branch, a full heterogeneous-media scene) and are out of scope here.

from std.math import abs
from std.memory import alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import RGB, TileResult_C
from gonzales.rendering import _fmt_f1, fmt_time, progress_str, normalize_film

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# ── _fmt_f1 ───────────────────────────────────────────────────────────────────

def test_fmt_f1_no_rounding_needed() raises:
    assert_true(_fmt_f1(Float64(2.0)) == "2.0")

def test_fmt_f1_rounds_to_nearest_tenth() raises:
    assert_true(_fmt_f1(Float64(2.94)) == "2.9")

def test_fmt_f1_carries_into_the_integer_part_on_round_up() raises:
    """Frac is computed as Int(0.96*10 + 0.5) == 10, which the function must
    detect and carry into the integer part (i += 1; frac = 0) rather than
    printing an invalid "2.10"."""
    assert_true(_fmt_f1(Float64(2.96)) == "3.0")

# ── fmt_time ──────────────────────────────────────────────────────────────────

def test_fmt_time_under_a_minute_uses_plain_seconds() raises:
    assert_true(fmt_time(Float64(45.0)) == "45.0s")

def test_fmt_time_over_a_minute_zero_pads_seconds() raises:
    assert_true(fmt_time(Float64(125.0)) == "2m 05s")

def test_fmt_time_minutes_ge_ten_no_extra_padding() raises:
    assert_true(fmt_time(Float64(660.0)) == "11m 00s")

# ── progress_str ──────────────────────────────────────────────────────────────

def test_progress_str_matches_expected_layout() raises:
    var s = progress_str(50, 100, Float64(10.0), "spp")
    # pct = 50/100*100 = 50.0%; est = elapsed*total/done = 10*100/50 = 20.0s
    assert_true(s == "Rendering: 50 / 100 spp (50.0%) | Elapsed: 10.0s | Total Est.: 20.0s                ")

def test_progress_str_zero_done_leaves_estimate_at_zero() raises:
    var s = progress_str(0, 100, Float64(5.0), "tiles")
    assert_true(s == "Rendering: 0 / 100 tiles (0.0%) | Elapsed: 5.0s | Total Est.: 0.0s                ")

# ── normalize_film ────────────────────────────────────────────────────────────

def _make_result(r: Float32, g: Float32, b: Float32, ar: Float32, ag: Float32, ab: Float32, w: Float32) -> TileResult_C:
    return TileResult_C(estimate=RGB(r, g, b), albedo=RGB(ar, ag, ab), filterWeight=w, pixelX=Int32(0), pixelY=Int32(0))

def test_normalize_film_zero_filter_weight_gives_zero_output() raises:
    """The w==0 early-out must zero BOTH beauty and albedo, even though the
    stored estimate/albedo are non-zero -- avoids a 0/0 division."""
    var results = alloc[TileResult_C](1)
    results[0] = _make_result(Float32(5.0), Float32(5.0), Float32(5.0), Float32(1.0), Float32(1.0), Float32(1.0), Float32(0.0))
    var beauty = alloc[Float32](3)
    var albedo = alloc[Float32](3)
    normalize_film(results, Int32(1), Float32(100.0), Float32(0.0), beauty, albedo)
    for i in range(3):
        assert_true(_close(beauty[i], Float32(0.0)))
        assert_true(_close(albedo[i], Float32(0.0)))
    results.free(); beauty.free(); albedo.free()

def test_normalize_film_scales_beauty_by_iso_but_leaves_albedo_unscaled() raises:
    """Beauty = estimate/weight * (iso/100); albedo = albedo_sum/weight with
    NO iso scaling at all -- these are genuinely different formulas, worth
    pinning down separately since they're easy to accidentally conflate."""
    var results = alloc[TileResult_C](1)
    results[0] = _make_result(Float32(2.0), Float32(4.0), Float32(6.0), Float32(0.5), Float32(0.25), Float32(0.75), Float32(2.0))
    var beauty = alloc[Float32](3)
    var albedo = alloc[Float32](3)
    normalize_film(results, Int32(1), Float32(200.0), Float32(0.0), beauty, albedo)
    # scale = 200/100 = 2; beauty = (2/2, 4/2, 6/2) * 2 = (2, 4, 6)
    assert_true(_close(beauty[0], Float32(2.0)))
    assert_true(_close(beauty[1], Float32(4.0)))
    assert_true(_close(beauty[2], Float32(6.0)))
    # albedo = (0.5/2, 0.25/2, 0.75/2) -- no iso factor at all
    assert_true(_close(albedo[0], Float32(0.25)))
    assert_true(_close(albedo[1], Float32(0.125)))
    assert_true(_close(albedo[2], Float32(0.375)))
    results.free(); beauty.free(); albedo.free()

def test_normalize_film_clamps_negative_beauty_to_zero() raises:
    var results = alloc[TileResult_C](1)
    results[0] = _make_result(Float32(-1.0), Float32(3.0), Float32(-5.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(1.0))
    var beauty = alloc[Float32](3)
    var albedo = alloc[Float32](3)
    normalize_film(results, Int32(1), Float32(100.0), Float32(0.0), beauty, albedo)
    assert_true(_close(beauty[0], Float32(0.0)))
    assert_true(_close(beauty[1], Float32(3.0)))
    assert_true(_close(beauty[2], Float32(0.0)))
    results.free(); beauty.free(); albedo.free()

def test_normalize_film_clamps_nan_beauty_to_zero() raises:
    """A NaN component (e.g. propagated from an earlier 0/0) fails self-
    equality -- the function relies on exactly that (b.r != b.r) to detect
    and zero it, since a plain `< 0` check would let NaN through."""
    var results = alloc[TileResult_C](1)
    var zero = Float32(0.0)
    var nan_val = zero / zero
    results[0] = _make_result(nan_val, Float32(1.0), Float32(1.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(1.0))
    var beauty = alloc[Float32](3)
    var albedo = alloc[Float32](3)
    normalize_film(results, Int32(1), Float32(100.0), Float32(0.0), beauty, albedo)
    assert_true(_close(beauty[0], Float32(0.0)))
    assert_true(_close(beauty[1], Float32(1.0)))
    assert_true(_close(beauty[2], Float32(1.0)))
    results.free(); beauty.free(); albedo.free()

def test_normalize_film_max_component_clamp_preserves_color_ratio() raises:
    """When the brightest channel exceeds max_component_value, ALL channels
    are scaled down by the same factor (max_component_value/mx) -- a hue-
    preserving clamp, not an independent per-channel clamp."""
    var results = alloc[TileResult_C](1)
    results[0] = _make_result(Float32(4.0), Float32(8.0), Float32(2.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(1.0))
    var beauty = alloc[Float32](3)
    var albedo = alloc[Float32](3)
    normalize_film(results, Int32(1), Float32(100.0), Float32(4.0), beauty, albedo)
    # mx=8 > 4 -> factor = 4/8 = 0.5
    assert_true(_close(beauty[0], Float32(2.0)))
    assert_true(_close(beauty[1], Float32(4.0)))
    assert_true(_close(beauty[2], Float32(1.0)))
    results.free(); beauty.free(); albedo.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
