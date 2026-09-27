# Unit tests for pure formatting/normalization helpers in rendering.mojo:
# _fmt_f1/fmt_time/progress_str (progress-bar string formatting) and
# normalize_film (TileResult accumulator -> per-pixel beauty/albedo
# arrays). render_tile/render_all_tiles/render_aux_buffers all need a real
# built BVH and SceneView (and, for render_tile's medium sampling
# branch, a full heterogeneous-media scene) and are out of scope here.

from std.math import abs
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import RGB
from gonzales.render_state import TileResult
from gonzales.rendering import normalize_film, apply_film_sensor
from gonzales.progress import _fmt_f1, fmt_time, progress_str

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

def _make_result(r: Float32, g: Float32, b: Float32, ar: Float32, ag: Float32, ab: Float32, w: Float32) -> TileResult:
    return TileResult(estimate=RGB(r, g, b), albedo=RGB(ar, ag, ab), filterWeight=w, pixelX=Int32(0), pixelY=Int32(0))

def test_normalize_film_zero_filter_weight_gives_zero_output() raises:
    """The w==0 early-out must zero BOTH beauty and albedo, even though the
    stored estimate/albedo are non-zero -- avoids a 0/0 division."""
    var results = unsafe_alloc[TileResult](1)
    results[unsafe_offset=0] = _make_result(Float32(5.0), Float32(5.0), Float32(5.0), Float32(1.0), Float32(1.0), Float32(1.0), Float32(0.0))
    var beauty = unsafe_alloc[Float32](3)
    var albedo = unsafe_alloc[Float32](3)
    normalize_film(results, Int32(1), Float32(100.0), Float32(0.0), beauty, albedo)
    for i in range(3):
        assert_true(_close(beauty[unsafe_offset=i], Float32(0.0)))
        assert_true(_close(albedo[unsafe_offset=i], Float32(0.0)))
    results.unsafe_free(); beauty.unsafe_free(); albedo.unsafe_free()

def test_normalize_film_scales_beauty_by_iso_but_leaves_albedo_unscaled() raises:
    """Beauty = estimate/weight * (iso/100); albedo = albedo_sum/weight with
    NO iso scaling at all -- these are genuinely different formulas, worth
    pinning down separately since they're easy to accidentally conflate."""
    var results = unsafe_alloc[TileResult](1)
    results[unsafe_offset=0] = _make_result(Float32(2.0), Float32(4.0), Float32(6.0), Float32(0.5), Float32(0.25), Float32(0.75), Float32(2.0))
    var beauty = unsafe_alloc[Float32](3)
    var albedo = unsafe_alloc[Float32](3)
    normalize_film(results, Int32(1), Float32(200.0), Float32(0.0), beauty, albedo)
    # scale = 200/100 = 2; beauty = (2/2, 4/2, 6/2) * 2 = (2, 4, 6)
    assert_true(_close(beauty[unsafe_offset=0], Float32(2.0)))
    assert_true(_close(beauty[unsafe_offset=1], Float32(4.0)))
    assert_true(_close(beauty[unsafe_offset=2], Float32(6.0)))
    # albedo = (0.5/2, 0.25/2, 0.75/2) -- no iso factor at all
    assert_true(_close(albedo[unsafe_offset=0], Float32(0.25)))
    assert_true(_close(albedo[unsafe_offset=1], Float32(0.125)))
    assert_true(_close(albedo[unsafe_offset=2], Float32(0.375)))
    results.unsafe_free(); beauty.unsafe_free(); albedo.unsafe_free()

def test_normalize_film_passes_negative_beauty_through_unclamped() raises:
    """`normalize_film` does NOT clamp negative components -- only apply_film_sensor
    does, downstream, AFTER the white-balance/sensor-colour matrix multiply.
    A negative component here is real (out-of-gamut colour) information that
    matrix multiply needs; clamping it away first and mixing channels through
    a non-identity matrix afterward silently drops a colour-correcting term.
    Measured on explosion.pbrt's nikon_d850 sensor: clamping here made a flat
    mid-grey test light render R:G:B = 0.80:0.56:0.52 instead of pbrt's
    near-neutral 0.47:0.50:0.53 -- see apply_film_sensor's docstring."""
    var results = unsafe_alloc[TileResult](1)
    results[unsafe_offset=0] = _make_result(Float32(-1.0), Float32(3.0), Float32(-5.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(1.0))
    var beauty = unsafe_alloc[Float32](3)
    var albedo = unsafe_alloc[Float32](3)
    normalize_film(results, Int32(1), Float32(100.0), Float32(0.0), beauty, albedo)
    assert_true(_close(beauty[unsafe_offset=0], Float32(-1.0)))
    assert_true(_close(beauty[unsafe_offset=1], Float32(3.0)))
    assert_true(_close(beauty[unsafe_offset=2], Float32(-5.0)))
    results.unsafe_free(); beauty.unsafe_free(); albedo.unsafe_free()

def test_normalize_film_clamps_nan_beauty_to_zero() raises:
    """A NaN component (e.g. propagated from an earlier 0/0) fails self-
    equality -- the function relies on exactly that (b.r != b.r) to detect
    and zero it, since a plain `< 0` check would let NaN through."""
    var results = unsafe_alloc[TileResult](1)
    var zero = Float32(0.0)
    var nan_val = zero / zero
    results[unsafe_offset=0] = _make_result(nan_val, Float32(1.0), Float32(1.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(1.0))
    var beauty = unsafe_alloc[Float32](3)
    var albedo = unsafe_alloc[Float32](3)
    normalize_film(results, Int32(1), Float32(100.0), Float32(0.0), beauty, albedo)
    assert_true(_close(beauty[unsafe_offset=0], Float32(0.0)))
    assert_true(_close(beauty[unsafe_offset=1], Float32(1.0)))
    assert_true(_close(beauty[unsafe_offset=2], Float32(1.0)))
    results.unsafe_free(); beauty.unsafe_free(); albedo.unsafe_free()

def test_normalize_film_max_component_clamp_preserves_color_ratio() raises:
    """When the value exceeds max_component_value, ALL channels are scaled
    down by the same factor -- a hue-preserving clamp, not an independent
    per-channel clamp. The factor is pbrt's: limit / max(X, Y, Z), the
    sensor space it clamps in (RGB.sensor_clamped)."""
    var results = unsafe_alloc[TileResult](1)
    results[unsafe_offset=0] = _make_result(Float32(4.0), Float32(8.0), Float32(2.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(1.0))
    var beauty = unsafe_alloc[Float32](3)
    var albedo = unsafe_alloc[Float32](3)
    normalize_film(results, Int32(1), Float32(100.0), Float32(4.0), beauty, albedo)
    # Y = 0.212671*4 + 0.715160*8 + 0.072169*2 = 6.7163 is the max of X, Y, Z
    # (X = 4.8712, Z = 2.9313) -> factor = 4 / 6.7163 = 0.59557
    assert_true(_close(beauty[unsafe_offset=0], Float32(2.38228)))
    assert_true(_close(beauty[unsafe_offset=1], Float32(4.76456)))
    assert_true(_close(beauty[unsafe_offset=2], Float32(1.19114)))
    results.unsafe_free(); beauty.unsafe_free(); albedo.unsafe_free()

# ── apply_film_sensor ────────────────────────────────────────────────────────

comptime _IDENTITY_WB = SIMD[DType.float32, 16](
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
    0, 0, 0, 0, 0, 0, 0,
)

def test_apply_film_sensor_identity_wb_still_clamps_negative_to_zero() raises:
    """`apply_film_sensor` is the ONLY place negative components get clamped
    now (normalize_film passes them through -- see that test above), even
    on the trivial (identity wb, exposure 1) path, which used to skip its
    per-pixel loop entirely and rely on normalize_film's now-removed clamp."""
    var buf = unsafe_alloc[Float32](3)
    buf[unsafe_offset=0] = Float32(-1.0); buf[unsafe_offset=1] = Float32(3.0); buf[unsafe_offset=2] = Float32(-5.0)
    apply_film_sensor(buf, 1, Float32(1.0), _IDENTITY_WB)
    assert_true(_close(buf[unsafe_offset=0], Float32(0.0)))
    assert_true(_close(buf[unsafe_offset=1], Float32(3.0)))
    assert_true(_close(buf[unsafe_offset=2], Float32(0.0)))
    buf.unsafe_free()

def test_apply_film_sensor_nonidentity_wb_uses_negative_input() raises:
    """The regression this whole pair of tests exists for: a non-identity wb
    matrix (a named sensor's colour-correction matrix, or an explicit
    whitebalance's Bradford adaptation) must see the REAL, possibly-negative
    pre-multiply value -- a swap-the-channels matrix applied to a negative
    component must show up as a NEGATIVE contribution reaching the output
    channel it's mixed into, not zero."""
    var wb = SIMD[DType.float32, 16](
        0, 1, 0,
        1, 0, 0,
        0, 0, 1,
        0, 0, 0, 0, 0, 0, 0,
    )
    var buf = unsafe_alloc[Float32](3)
    buf[unsafe_offset=0] = Float32(-2.0); buf[unsafe_offset=1] = Float32(3.0); buf[unsafe_offset=2] = Float32(1.0)
    apply_film_sensor(buf, 1, Float32(1.0), wb)
    # nr = 0*r + 1*g + 0*b = g = 3          -> clamps to 3
    # ng = 1*r + 0*g + 0*b = r = -2         -> clamps to 0 (NOT skipped/left as 3)
    # nb = 0*r + 0*g + 1*b = b = 1          -> unchanged
    assert_true(_close(buf[unsafe_offset=0], Float32(3.0)))
    assert_true(_close(buf[unsafe_offset=1], Float32(0.0)))
    assert_true(_close(buf[unsafe_offset=2], Float32(1.0)))
    buf.unsafe_free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
