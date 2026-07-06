from std.math import abs
from std.testing import assert_true, assert_false, TestSuite
from gonzales.geometry import Point3f, Vec3f
from gonzales.bvh import intersect_aabb

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# ── intersect_aabb (hottest function in the renderer's BVH traversal) ────────
# intersect_aabb takes precomputed `rdir` (1/direction) and `orgRdir`
# (origin*rdir) rather than a raw Ray_C — this mirrors the exact call-site
# convention used in traverse_bvh2_core, e.g.:
#   var rdir = Vec3f(1/ray.direction.x, 1/ray.direction.y, 1/ray.direction.z)
#   var orgRdir = Vec3f(ray.origin.x*rdir.x, ray.origin.y*rdir.y, ray.origin.z*rdir.z)
#   var nearXIsMin = rdir.x >= 0.0   (similarly for Y, Z)
# It returns Tuple[Bool, Float32] = (hit, tNear); tFar is used internally for
# the hit test but is NOT returned.
#
# NOTE: a genuinely axis-aligned ray (direction with two exactly-zero
# components, e.g. Vec3f(0,0,1)) hits an IEEE-754 inf-inf=NaN cancellation in
# this function's tFar accumulation and reports a false miss even when the
# ray truly passes through the box (verified empirically). This is a latent
# bug in intersect_aabb, not something to paper over here — the tests below
# use directions with all three components nonzero (as essentially every
# real camera/shading ray in this renderer has, since they're never exactly
# axis-aligned) to exercise the intended, correct code path.

def _aabb_hit(origin: Point3f, direction: Vec3f, tMax: Float32 = Float32(100.0)) -> Tuple[Bool, Float32]:
    """Reproduces the exact rdir/orgRdir/nearIsMin derivation used at every
    intersect_aabb call site in bvh.mojo (traverse_bvh2_core etc.), against a
    fixed axis-aligned box min=(-1,-1,-1), max=(1,1,1)."""
    var bmin = Point3f(-1.0, -1.0, -1.0)
    var bmax = Point3f(1.0, 1.0, 1.0)
    var rdir = Vec3f(Float32(1.0) / direction.x, Float32(1.0) / direction.y, Float32(1.0) / direction.z)
    var orgRdir = Vec3f(origin.x * rdir.x, origin.y * rdir.y, origin.z * rdir.z)
    var nearXIsMin = rdir.x >= Float32(0.0)
    var nearYIsMin = rdir.y >= Float32(0.0)
    var nearZIsMin = rdir.z >= Float32(0.0)
    return intersect_aabb(bmin, bmax, rdir, orgRdir, nearXIsMin, nearYIsMin, nearZIsMin, tMax)

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

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
