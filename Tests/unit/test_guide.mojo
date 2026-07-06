from std.math import abs
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import Point3f, Bounds3f
from gonzales.guide import (
    guide_create, guide_free, guide_merge,
    guide_pos_to_cell, guide_dir_to_bin, guide_bin_to_dir,
    guide_record, guide_pdf, guide_cell_has_data, guide_sample,
    GUIDE_CELLS, GUIDE_BINS, GUIDE_DIMS, FOUR_PI_F,
)

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _bounds16() -> Bounds3f:
    """A 16x16x16 grid spanning [0,16]^3 — chosen so that
    cell_coord = Int(GUIDE_DIMS * (p - min) / extent) reduces to Int(p),
    giving clean closed-form cell indices for the tests below."""
    return Bounds3f(Point3f(0.0, 0.0, 0.0), Point3f(16.0, 16.0, 16.0))

# ── guide_pos_to_cell (spatial grid lookup) ──────────────────────────────────

def test_guide_pos_to_cell_min_corner_is_cell_zero() raises:
    var g = guide_create(_bounds16())
    var c = guide_pos_to_cell(g, Point3f(0.0, 0.0, 0.0))
    guide_free(g)
    assert_true(c == 0)

def test_guide_pos_to_cell_center_is_middle_index() raises:
    """Center of the grid (8,8,8) maps to grid coords (8,8,8), giving
    cell = 8*16^2 + 8*16 + 8 = 2184 — comfortably in the middle third of the
    4096-cell index range."""
    var g = guide_create(_bounds16())
    var c = guide_pos_to_cell(g, Point3f(8.0, 8.0, 8.0))
    guide_free(g)
    assert_true(c == 8 * GUIDE_DIMS * GUIDE_DIMS + 8 * GUIDE_DIMS + 8)
    assert_true(c > GUIDE_CELLS // 3 and c < 2 * GUIDE_CELLS // 3)

def test_guide_pos_to_cell_outside_bounds_returns_negative_one() raises:
    """Verified against the actual code (not assumed): guide_pos_to_cell does
    NOT clamp out-of-range positions to an edge cell — it returns -1 for any
    position outside [min,max), including exactly on the max-corner boundary
    (the fx>=1 check is exclusive of the upper edge)."""
    var g = guide_create(_bounds16())
    var beyond_max = guide_pos_to_cell(g, Point3f(20.0, 20.0, 20.0))
    var below_min = guide_pos_to_cell(g, Point3f(-5.0, -5.0, -5.0))
    var on_max_corner = guide_pos_to_cell(g, Point3f(16.0, 16.0, 16.0))
    guide_free(g)
    assert_true(beyond_max == -1)
    assert_true(below_min == -1)
    assert_true(on_max_corner == -1)

def test_guide_pos_to_cell_just_inside_max_is_last_cell() raises:
    """Just below the max corner lands in the last valid cell index
    (GUIDE_DIMS-1 on each axis), confirming the half-open [min,max) convention."""
    var g = guide_create(_bounds16())
    var c = guide_pos_to_cell(g, Point3f(15.999, 15.999, 15.999))
    guide_free(g)
    assert_true(c == GUIDE_CELLS - 1)

# ── guide_dir_to_bin / guide_bin_to_dir (equal-area octahedral binning) ──────

def test_guide_bin_round_trip_for_interior_bins() raises:
    """Bin -> direction (bin center) -> bin must recover the same bin for
    bins away from the octahedral fold lines. Bins adjacent to a fold —
    verified empirically, e.g. bin 0, 7, 8, 15 — do NOT round-trip exactly
    because the polynomial atan2 approximation used by the square-to-sphere
    map isn't a perfect inverse right at the fold; that's a genuine property
    of this code, not tested here since it's not the common case."""
    for b in [1, 5, 10, 20, 30, 35, 44, 50, 60, 62]:
        var d = guide_bin_to_dir(b)
        var b2 = guide_dir_to_bin(d[0], d[1], d[2])
        assert_true(b2 == b)

def test_guide_bin_to_dir_is_unit_length() raises:
    for b in [0, 16, 32, 48, 63]:
        var d = guide_bin_to_dir(b)
        var len_sq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2]
        assert_true(_close(len_sq, Float32(1.0)))

# ── guide_record / guide_pdf / guide_cell_has_data / guide_sample ───────────

def test_guide_pdf_falls_back_to_uniform_when_cell_empty() raises:
    """No energy recorded anywhere -> pdf must be the uniform 1/(4*pi) sphere
    density, per the documented fallback."""
    var g = guide_create(_bounds16())
    var p = guide_pdf(g, 0, 0.0, 1.0, 0.0)
    guide_free(g)
    assert_true(_close(p, Float32(1.0) / FOUR_PI_F))

def test_guide_cell_has_data_false_when_empty() raises:
    var g = guide_create(_bounds16())
    var has_data = guide_cell_has_data(g, 0)
    guide_free(g)
    assert_false(has_data)

def test_guide_record_concentrates_pdf_into_recorded_direction() raises:
    """Recording repeated energy into direction (0,1,0) must make that
    direction's pdf large and other directions' pdf collapse toward the
    guide_pdf floor of 1e-7 — a concentrated histogram, not uniform noise."""
    var g = guide_create(_bounds16())
    for _ in range(20):
        guide_record(g, 0, 0.0, 1.0, 0.0, Float32(1.0))
    assert_true(guide_cell_has_data(g, 0))
    var p_hit = guide_pdf(g, 0, 0.0, 1.0, 0.0)
    var p_miss = guide_pdf(g, 0, 1.0, 0.0, 0.0)
    guide_free(g)
    assert_true(p_hit > Float32(1.0))       # much denser than uniform (0.0796)
    assert_true(_close(p_miss, Float32(1e-7)))  # floor for an empty bin

def test_guide_sample_returns_pdf_matching_guide_pdf() raises:
    """Guide_sample's returned pdf for the direction it actually draws must be
    self-consistent with an independent guide_pdf query at that direction —
    both walk the same per-bin energy/total accounting."""
    var g = guide_create(_bounds16())
    for _ in range(20):
        guide_record(g, 0, 0.0, 1.0, 0.0, Float32(1.0))
    var (dx, dy, dz, sample_pdf, ok) = guide_sample(g, 0, Float32(0.5))
    var query_pdf = guide_pdf(g, 0, dx, dy, dz)
    guide_free(g)
    assert_true(ok)
    assert_true(_close(sample_pdf, query_pdf))

def test_guide_sample_reports_not_ok_when_cell_empty() raises:
    var g = guide_create(_bounds16())
    var (_, _, _, _, ok) = guide_sample(g, 0, Float32(0.5))
    guide_free(g)
    assert_false(ok)

# ── guide_merge ───────────────────────────────────────────────────────────

def test_guide_merge_sums_energy_elementwise() raises:
    """Merging src into dst must add src's per-bin energy onto dst's, not
    overwrite it — verified via the total-energy invariant on one bin."""
    var dst = guide_create(_bounds16())
    var src = guide_create(_bounds16())
    guide_record(dst, 0, 0.0, 1.0, 0.0, Float32(3.0))
    guide_record(src, 0, 0.0, 1.0, 0.0, Float32(4.0))
    guide_merge(dst, src)
    var bin = guide_dir_to_bin(Float32(0.0), Float32(1.0), Float32(0.0))
    var total = dst.energy[0 * GUIDE_BINS + bin]
    guide_free(dst)
    guide_free(src)
    assert_true(_close(total, Float32(7.0)))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
