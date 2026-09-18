from std.math import abs, max
from std.memory import alloc
from std.testing import assert_true, assert_false, assert_equal, TestSuite
from gonzales.geometry import Point3f, Vec3f
from gonzales.bvh import intersect_aabb, build_bvh2, BVH2Node

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# ── intersect_aabb (hottest function in the renderer's BVH traversal) ────────
# intersect_aabb takes precomputed `rdir` (1/direction) and `org` (the ray
# origin) rather than a raw Ray_C — this mirrors the exact call-site
# convention used in traverse_bvh2_core, e.g.:
#   var rdir = Vec3f(1/ray.direction.x, 1/ray.direction.y, 1/ray.direction.z)
#   var org  = Vec3f(ray.origin.x, ray.origin.y, ray.origin.z)
#   var nearXIsMin = rdir.x >= 0.0   (similarly for Y, Z)
# It returns Tuple[Bool, Float32] = (hit, tNear); tFar is used internally for
# the hit test but is NOT returned.
#
# Formerly this took a precomputed `orgRdir = origin*rdir` instead of `org`,
# computing tNear/tFar as `near*rdir - orgRdir`. For a genuinely axis-aligned
# ray (rdir.x = +/-inf) the two separately-computed infinite products were
# routinely like-signed, so subtracting them gave inf-inf=NaN and a false
# miss even when the ray truly passed through the box. Fixed by subtracting
# the (finite) coordinates first and only then multiplying by the (possibly
# infinite) rdir — same op count per node, so no perf cost. See the fix's
# commit for the full derivation.

def _aabb_hit(origin: Point3f, direction: Vec3f, tMax: Float32 = Float32(100.0)) -> Tuple[Bool, Float32]:
    """Reproduces the exact rdir/org/nearIsMin derivation used at every
    intersect_aabb call site in bvh.mojo (traverse_bvh2_core etc.), against a
    fixed axis-aligned box min=(-1,-1,-1), max=(1,1,1)."""
    var bmin = Point3f(-1.0, -1.0, -1.0)
    var bmax = Point3f(1.0, 1.0, 1.0)
    var rdir = Vec3f(Float32(1.0) / direction.x, Float32(1.0) / direction.y, Float32(1.0) / direction.z)
    var org = Vec3f(origin.x, origin.y, origin.z)
    var nearXIsMin = rdir.x >= Float32(0.0)
    var nearYIsMin = rdir.y >= Float32(0.0)
    var nearZIsMin = rdir.z >= Float32(0.0)
    return intersect_aabb(bmin, bmax, rdir, org, nearXIsMin, nearYIsMin, nearZIsMin, tMax)

def test_intersect_aabb_hit_through_center_from_outside() raises:
    """Ray from outside, mostly +Z (small xy drift keeps it clear of the
    axis-aligned NaN edge case), enters the z=-1 face. Closed form:
    tNear = (bmin.z - origin.z) / direction.z = (-1 - (-5)) / 1 = 4."""
    var origin = Point3f(0.2, -0.1, -5.0)
    var direction = Vec3f(0.02, -0.01, 1.0)
    var (hit, tNear) = _aabb_hit(origin, direction)
    assert_true(hit)
    assert_true(_close(tNear, Float32(4.0)))

def test_intersect_aabb_misses_when_offset_well_outside() raises:
    """Same direction as the hit case above, but shifted 10 units on X — the
    ray's X coordinate never comes close to the [-1,1] slab, so it must miss."""
    var origin = Point3f(10.0, 0.0, -5.0)
    var direction = Vec3f(0.02, -0.01, 1.0)
    var (hit, _) = _aabb_hit(origin, direction)
    assert_false(hit)

def test_intersect_aabb_origin_inside_reports_non_positive_tnear() raises:
    """Origin at the box's exact center: every per-axis tNear candidate is
    negative (the ray already passed each slab's near face), so the explicit
    `max(..., 0)` clamp in intersect_aabb makes tNear exactly 0 — the
    convention this code uses for 'ray started inside the box'."""
    var origin = Point3f(0.0, 0.0, 0.0)
    var direction = Vec3f(0.3, 0.4, 0.866)
    var (hit, tNear) = _aabb_hit(origin, direction)
    assert_true(hit)
    assert_true(tNear <= Float32(0.0))
    assert_true(_close(tNear, Float32(0.0)))

def test_intersect_aabb_tmax_cutoff_turns_a_hit_into_a_miss() raises:
    """Same ray/box as the first hit test (true entry at t=4), but tMax=2
    cuts off before the box is reached — tMax must be honored as a hard
    upper bound on the search interval."""
    var origin = Point3f(0.2, -0.1, -5.0)
    var direction = Vec3f(0.02, -0.01, 1.0)
    var (hit, _) = _aabb_hit(origin, direction, Float32(2.0))
    assert_false(hit)

def test_intersect_aabb_axis_aligned_ray_through_center_hits() raises:
    """The exact case that used to false-miss via inf-inf=NaN before the
    (near-org)*rdir fix: a perfectly axis-aligned ray (direction with two
    exactly-zero components) straight through the box's center."""
    var origin = Point3f(0.0, 0.0, -5.0)
    var direction = Vec3f(0.0, 0.0, 1.0)
    var (hit, tNear) = _aabb_hit(origin, direction)
    assert_true(hit)
    assert_true(_close(tNear, Float32(4.0)))

def test_intersect_aabb_axis_aligned_ray_offset_misses() raises:
    """Same axis-aligned direction, but shifted off the box on X — must
    still correctly miss (not a NaN-driven false hit either)."""
    var origin = Point3f(5.0, 0.0, -5.0)
    var direction = Vec3f(0.0, 0.0, 1.0)
    var (hit, _) = _aabb_hit(origin, direction)
    assert_false(hit)

# ── build_bvh2: the parallel build must reproduce the serial tree exactly ────

def _fill_boxes(bounds: UnsafePointer[Float32, MutExternalOrigin], n: Int, seed: UInt64,
                clusters: Int, dup_every: Int):
    """Deterministic pseudo-random AABBs, 6 floats each (min xyz, max xyz).
    clusters > 0 gathers centres around that many points; dup_every > 0 makes
    every dup_every-th box an exact copy of the one before (dup_every == 1:
    all boxes identical), which exercises degenerate splits."""
    var s = seed
    var cc = alloc[Float32](max(clusters, 1) * 3)
    for k in range(max(clusters, 1) * 3):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        cc[unsafe_offset=k] = Float32(s >> 40) / Float32(1 << 24) * Float32(100.0)
    for i in range(n):
        if dup_every > 0 and i > 0 and i % dup_every == 0:
            for a in range(6):
                bounds[unsafe_offset=i * 6 + a] = bounds[unsafe_offset=(i - 1) * 6 + a]
            continue
        var r = InlineArray[Float32, 4](fill=Float32(0))
        for a in range(4):
            s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            r[a] = Float32(s >> 40) / Float32(1 << 24)
        var half = Float32(0.01) + r[3]
        for a in range(3):
            var c = r[a] * Float32(100.0)
            if clusters > 0:
                c = cc[unsafe_offset=(i % clusters) * 3 + a] + (r[a] - Float32(0.5)) * Float32(4.0)
            bounds[unsafe_offset=i * 6 + a] = c - half
            bounds[unsafe_offset=i * 6 + 3 + a] = c + half
    cc.unsafe_free()

# A small subtree size makes the parallel build split many levels near the
# root and build hundreds of subtrees, on inputs small enough to test quickly.
comptime _TEST_SUBTREE_PRIMS = 64

def _check_parallel_matches_serial(n: Int, seed: UInt64, clusters: Int, dup_every: Int) raises:
    var bounds = alloc[Float32](n * 6)
    _fill_boxes(bounds, n, seed, clusters, dup_every)
    var serial_nodes = alloc[BVH2Node](2 * n + 4)
    var parallel_nodes = alloc[BVH2Node](2 * n + 4)
    var serial_order = alloc[Int32](n)
    var parallel_order = alloc[Int32](n)
    var serial_count = build_bvh2(bounds, Int32(n), serial_nodes, serial_order, parallel=False)
    var parallel_count = build_bvh2(bounds, Int32(n), parallel_nodes, parallel_order,
                                    parallel=True, subtree_prims=_TEST_SUBTREE_PRIMS)
    assert_equal(serial_count, parallel_count)
    var mismatches = 0
    for i in range(Int(serial_count)):
        var a = serial_nodes[unsafe_offset=i]; var b = parallel_nodes[unsafe_offset=i]
        if a.offset != b.offset or a.count != b.count or \
           a.min.x != b.min.x or a.min.y != b.min.y or a.min.z != b.min.z or \
           a.max.x != b.max.x or a.max.y != b.max.y or a.max.z != b.max.z:
            mismatches += 1
    for i in range(n):
        if serial_order[unsafe_offset=i] != parallel_order[unsafe_offset=i]:
            mismatches += 1
    assert_equal(mismatches, 0)
    bounds.unsafe_free(); serial_nodes.unsafe_free(); parallel_nodes.unsafe_free()
    serial_order.unsafe_free(); parallel_order.unsafe_free()

def test_build_bvh2_parallel_matches_serial_uniform() raises:
    _check_parallel_matches_serial(5000, UInt64(1), 0, 0)

def test_build_bvh2_parallel_matches_serial_clustered_with_duplicates() raises:
    _check_parallel_matches_serial(5000, UInt64(7), 16, 5)

def test_build_bvh2_parallel_matches_serial_all_identical() raises:
    """All boxes identical: the root is one degenerate leaf in both builds."""
    _check_parallel_matches_serial(2000, UInt64(3), 0, 1)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
