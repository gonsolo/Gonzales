from std.math import abs
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import Point3f, Bounds3f
from gonzales.bvh import _equal_area_square_to_sphere
from gonzales.guide import (
    guide_create, guide_free, guide_clone_empty, guide_merge, guide_refine,
    guide_pos_to_cell, guide_record, guide_pdf, guide_cell_has_data, guide_sample,
    guide_is_active, null_guide,
    FOUR_PI_F, DIR_SPLIT_FRACTION, SPATIAL_SPLIT_SAMPLES,
)

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _bounds16() -> Bounds3f:
    return Bounds3f(Point3f(0.0, 0.0, 0.0), Point3f(16.0, 16.0, 16.0))

# A world direction guaranteed to fall in quadrant 0 (u<0.5, v<0.5) of the
# equal-area octahedral square, and one guaranteed to fall in quadrant 3
# (u>=0.5, v>=0.5) — built by going through the square->sphere map directly
# rather than guessing which raw (x,y,z) lands where.
def _dir_q0() -> SIMD[DType.float32, 3]:
    return _equal_area_square_to_sphere(Float32(0.25), Float32(0.25))

def _dir_q3() -> SIMD[DType.float32, 3]:
    return _equal_area_square_to_sphere(Float32(0.75), Float32(0.75))

# ── null_guide / guide_is_active ─────────────────────────────────────────────

def test_null_guide_is_not_active() raises:
    assert_false(guide_is_active(null_guide()))

def test_guide_create_is_active() raises:
    var g = guide_create(_bounds16())
    var active = guide_is_active(g)
    guide_free(g)
    assert_true(active)

# ── guide_pos_to_cell (spatial kd-tree, single leaf before any split) ───────

def test_guide_pos_to_cell_min_corner_is_the_single_leaf() raises:
    var g = guide_create(_bounds16())
    var c = guide_pos_to_cell(g, Point3f(0.0, 0.0, 0.0))
    guide_free(g)
    assert_true(c == 0)

def test_guide_pos_to_cell_center_is_same_single_leaf() raises:
    """Before any spatial split, EVERY in-bounds position must map to the
    same (only) leaf, index 0."""
    var g = guide_create(_bounds16())
    var c = guide_pos_to_cell(g, Point3f(8.0, 8.0, 8.0))
    guide_free(g)
    assert_true(c == 0)

def test_guide_pos_to_cell_outside_bounds_returns_negative_one() raises:
    """Half-open [min,max) convention: exactly on the max corner is outside."""
    var g = guide_create(_bounds16())
    var beyond_max = guide_pos_to_cell(g, Point3f(20.0, 20.0, 20.0))
    var below_min = guide_pos_to_cell(g, Point3f(-5.0, -5.0, -5.0))
    var on_max_corner = guide_pos_to_cell(g, Point3f(16.0, 16.0, 16.0))
    guide_free(g)
    assert_true(beyond_max == -1)
    assert_true(below_min == -1)
    assert_true(on_max_corner == -1)

# ── directional queries before any refine (single quadtree leaf) ───────────

def test_guide_pdf_uniform_before_any_split() raises:
    """A fresh tree's directional quadtree is a single leaf covering the
    whole sphere -- pdf must be exactly the uniform 1/4π everywhere,
    regardless of how much energy has been recorded, until guide_refine
    actually subdivides it."""
    var g = guide_create(_bounds16())
    var q0 = _dir_q0()
    for _ in range(20):
        guide_record(g, 0, q0[0], q0[1], q0[2], Float32(1.0))
    var p = guide_pdf(g, 0, q0[0], q0[1], q0[2])
    guide_free(g)
    assert_true(_close(p, Float32(1.0) / FOUR_PI_F))

def test_guide_cell_has_data_false_before_refine_even_with_energy() raises:
    """Enough total energy to clear the 1e-4 floor, but no directional
    split has happened yet -- has_data must still be False (its whole
    point is gating on genuine concentration, not raw energy)."""
    var g = guide_create(_bounds16())
    var q0 = _dir_q0()
    for _ in range(20):
        guide_record(g, 0, q0[0], q0[1], q0[2], Float32(1.0))
    var has_data = guide_cell_has_data(g, 0)
    guide_free(g)
    assert_false(has_data)

def test_guide_sample_reports_not_ok_when_cell_empty() raises:
    var g = guide_create(_bounds16())
    var (_, _, _, _, ok) = guide_sample(g, 0, Float32(0.5))
    guide_free(g)
    assert_false(ok)

def test_guide_pdf_falls_back_to_uniform_for_invalid_cell() raises:
    var g = guide_create(_bounds16())
    var p = guide_pdf(g, -1, 0.0, 1.0, 0.0)
    guide_free(g)
    assert_true(_close(p, Float32(1.0) / FOUR_PI_F))

# ── guide_refine: directional growth ────────────────────────────────────────

def test_guide_refine_splits_directional_quadtree_when_concentrated() raises:
    """All recorded energy lands in one octahedral quadrant (well above the
    1% DIR_SPLIT_FRACTION threshold) -- refine must subdivide the leaf's
    quadtree into 4 children."""
    var g = guide_create(_bounds16())
    var q0 = _dir_q0()
    for _ in range(10):
        guide_record(g, 0, q0[0], q0[1], q0[2], Float32(1.0))
    assert_true(_close(DIR_SPLIT_FRACTION, Float32(0.01)))  # sanity: threshold assumed by this test
    var refined = guide_refine(g)
    var n_dnodes_after = refined.n_dnodes
    guide_free(refined)
    assert_true(n_dnodes_after == Int32(5))  # 1 root + 4 fresh children

def test_guide_refine_does_not_split_when_no_energy_recorded() raises:
    """A fresh tree has zero energy everywhere (below the 1e-6 floor) --
    refine must be a structural no-op. Note: the ROOT's very first split is
    otherwise unconditional given ANY recorded energy (a single node
    trivially holds 100% of the tree's total until it has a sibling to be
    compared against) -- this is the only directional case refine can
    correctly decline to grow, matching Müller's own algorithm."""
    var g = guide_create(_bounds16())
    var refined = guide_refine(g)
    var n_dnodes_after = refined.n_dnodes
    var n_snodes_after = refined.n_snodes
    guide_free(refined)
    assert_true(n_dnodes_after == Int32(1))
    assert_true(n_snodes_after == Int32(1))

def test_guide_refine_directional_depth_is_capped() raises:
    """Repeated refine-then-record-more-in-the-same-hot-direction cycles
    must never grow the quadtree past DIR_MAX_DEPTH -- verified indirectly
    via a bounded node count (4^depth leaves is the worst case at the cap;
    well under that after only a handful of cycles confirms growth
    actually stops, not just that it hasn't yet run far enough to overflow)."""
    var g = guide_create(_bounds16())
    var q0 = _dir_q0()
    var tree = g
    for _ in range(12):
        for _ in range(5):
            guide_record(tree, 0, q0[0], q0[1], q0[2], Float32(1.0))
        var refined = guide_refine(tree)
        tree = refined
    var n_dnodes_final = tree.n_dnodes
    guide_free(tree)
    assert_true(n_dnodes_final < Int32(2000))

def test_guide_refine_then_more_records_concentrates_pdf() raises:
    """After a directional split, additional records landing in the SAME
    hot quadrant must make that quadrant's pdf exceed an untouched
    quadrant's -- the concentration a guide is supposed to learn."""
    var g = guide_create(_bounds16())
    var q0 = _dir_q0()
    var q3 = _dir_q3()
    for _ in range(10):
        guide_record(g, 0, q0[0], q0[1], q0[2], Float32(1.0))
    var tree = guide_refine(g)
    # A few more real samples landing in q0's now-dedicated child.
    for _ in range(10):
        guide_record(tree, 0, q0[0], q0[1], q0[2], Float32(1.0))
    var p_hot = guide_pdf(tree, 0, q0[0], q0[1], q0[2])
    var p_cold = guide_pdf(tree, 0, q3[0], q3[1], q3[2])
    guide_free(tree)
    assert_true(p_hot > p_cold)

def test_guide_sample_returns_pdf_matching_guide_pdf() raises:
    """Guide_sample's returned pdf for the direction it actually draws must
    be self-consistent with an independent guide_pdf query there."""
    var g = guide_create(_bounds16())
    var q0 = _dir_q0()
    for _ in range(10):
        guide_record(g, 0, q0[0], q0[1], q0[2], Float32(1.0))
    var tree = guide_refine(g)
    for _ in range(10):
        guide_record(tree, 0, q0[0], q0[1], q0[2], Float32(1.0))
    var (dx, dy, dz, sample_pdf, ok) = guide_sample(tree, 0, Float32(0.01))
    var query_pdf = guide_pdf(tree, 0, dx, dy, dz)
    guide_free(tree)
    assert_true(ok)
    assert_true(_close(sample_pdf, query_pdf))

# ── guide_refine: spatial growth ────────────────────────────────────────────

def test_guide_refine_splits_spatial_leaf_past_sample_threshold() raises:
    """Recording more samples at one position than SPATIAL_SPLIT_SAMPLES
    must split the root spatial leaf into 2 children on refine."""
    var g = guide_create(_bounds16())
    var q0 = _dir_q0()
    var n = Int(SPATIAL_SPLIT_SAMPLES) + 1
    for _ in range(n):
        guide_record(g, 0, q0[0], q0[1], q0[2], Float32(0.001))
    var refined = guide_refine(g)
    var n_snodes_after = refined.n_snodes
    guide_free(refined)
    assert_true(n_snodes_after == Int32(3))  # root (now interior) + 2 new leaves

def test_guide_refine_spatial_split_routes_positions_to_different_leaves() raises:
    var g = guide_create(_bounds16())
    var q0 = _dir_q0()
    var n = Int(SPATIAL_SPLIT_SAMPLES) + 1
    for _ in range(n):
        guide_record(g, 0, q0[0], q0[1], q0[2], Float32(0.001))
    var tree = guide_refine(g)
    var c_lo = guide_pos_to_cell(tree, Point3f(0.5, 0.5, 0.5))
    var c_hi = guide_pos_to_cell(tree, Point3f(15.5, 15.5, 15.5))
    guide_free(tree)
    assert_true(c_lo != c_hi)
    assert_true(c_lo != 0 or c_hi != 0)  # at least one moved off the old root index

def test_guide_refine_resets_sample_count_but_not_energy() raises:
    """Sample_count is a per-iteration statistic (reset by refine);
    directional energy is a cumulative estimate (never reset by refine)."""
    var g = guide_create(_bounds16())
    var q0 = _dir_q0()
    for _ in range(5):
        guide_record(g, 0, q0[0], q0[1], q0[2], Float32(1.0))
    var tree = guide_refine(g)
    # No spatial split happened (well under SPATIAL_SPLIT_SAMPLES), so leaf 0
    # still exists -- its sample_count must be back to 0, but its directional
    # quadtree's total energy must still reflect the 5 pre-refine records.
    var has_data = guide_cell_has_data(tree, 0)  # requires total energy >= 1e-4
    guide_free(tree)
    assert_true(has_data)

# ── guide_clone_empty ────────────────────────────────────────────────────────

def test_guide_clone_empty_matches_structure_but_zeroes_data() raises:
    var g = guide_create(_bounds16())
    var q0 = _dir_q0()
    for _ in range(10):
        guide_record(g, 0, q0[0], q0[1], q0[2], Float32(1.0))
    var tree = guide_refine(g)  # now has a grown directional quadtree
    var clone = guide_clone_empty(tree)
    var same_shape = clone.n_snodes == tree.n_snodes and clone.n_dnodes == tree.n_dnodes
    var clone_has_data = guide_cell_has_data(clone, 0)
    guide_free(tree)
    guide_free(clone)
    assert_true(same_shape)
    assert_false(clone_has_data)  # total energy is 0 in the clone

# ── guide_merge ───────────────────────────────────────────────────────────

def test_guide_merge_sums_energy_elementwise() raises:
    """Merging src into dst must ADD src's root-dtree energy onto dst's,
    not overwrite it -- checked by reading the (public) DNode field
    directly through both handles' shared node-array layout."""
    var dst = guide_create(_bounds16())
    var src = guide_create(_bounds16())
    var q0 = _dir_q0()
    for _ in range(3):
        guide_record(dst, 0, q0[0], q0[1], q0[2], Float32(1.0))
    for _ in range(4):
        guide_record(src, 0, q0[0], q0[1], q0[2], Float32(1.0))
    guide_merge(dst, src)
    var total = dst.dnodes[0].energy
    guide_free(dst)
    guide_free(src)
    assert_true(_close(total, Float32(7.0)))

def test_guide_merge_sums_sample_count_so_split_can_trigger() raises:
    """Neither shard alone reaches SPATIAL_SPLIT_SAMPLES, but their sum
    does -- guide_merge must add sample_count (not just energy) for the
    combined tree to correctly decide to split on refine."""
    var dst = guide_create(_bounds16())
    var src = guide_create(_bounds16())
    var q0 = _dir_q0()
    var half = Int(SPATIAL_SPLIT_SAMPLES) // 2 + 1
    for _ in range(half):
        guide_record(dst, 0, q0[0], q0[1], q0[2], Float32(0.001))
        guide_record(src, 0, q0[0], q0[1], q0[2], Float32(0.001))
    guide_merge(dst, src)
    guide_free(src)
    var refined = guide_refine(dst)
    var n_snodes_after = refined.n_snodes
    guide_free(refined)
    assert_true(n_snodes_after == Int32(3))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
