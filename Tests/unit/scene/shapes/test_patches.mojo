from std.ffi import external_call
from std.math import sqrt, cos, sin, abs, max
from std.memory import Layout, alloc, dealloc
from std.testing import assert_true, TestSuite
from gonzales.core.geometry import Vec3f, dot
from gonzales.scene.shapes.patches import BezierPatch, intersect_patch, patch_eval


def _cube_patches(isolation: Int) -> List[BezierPatch]:
    """The Catmull-Clark limit surface of the cube [-1, 1]^3 as Bezier patches (src/oiio/osd.cc)."""
    var pts: List[Float32] = [-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1]
    var counts: List[Int32] = [4, 4, 4, 4, 4, 4]
    var idx: List[Int32] = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
    var n = List[Int32](length=1, fill=0)
    var h = external_call["osd_build", Int](Int32(8), pts.unsafe_ptr(), Int32(6), counts.unsafe_ptr(), idx.unsafe_ptr(),
        Int32(0), Int(0), Int(0), Int32(0), Int(0), Int(0), Int32(isolation), n.unsafe_ptr())
    var np = Int(n[0])
    var cp = List[Float32](length=np * 48, fill=0)
    var uv = List[Float32](length=np * 3, fill=0)
    var face = List[Int32](length=np, fill=0)
    external_call["osd_take", NoneType](h, cp.unsafe_ptr(), uv.unsafe_ptr(), face.unsafe_ptr())
    var out = List[BezierPatch]()
    for p in range(np):
        var x = SIMD[DType.float32, 16](0); var y = SIMD[DType.float32, 16](0); var z = SIMD[DType.float32, 16](0)
        for k in range(16):
            x[k] = cp[p * 48 + k * 3]; y[k] = cp[p * 48 + k * 3 + 1]; z[k] = cp[p * 48 + k * 3 + 2]
        out.append(BezierPatch(x, y, z))
    return out^


def _nearest(patches: List[BezierPatch], org: Vec3f, dir: Vec3f) -> Tuple[Int, Float32, Float32, Float32, Int]:
    """(patch, t, u, v, number of patches hit) over all patches."""
    var best = Float32(1e30); var bi = -1; var bu = Float32(0); var bv = Float32(0); var n_hit = 0
    for i in range(len(patches)):
        var r = intersect_patch(org, dir, patches[i], Float32(1e30))
        if r[0]:
            n_hit += 1
            if r[1] < best:
                best = r[1]; bi = i; bu = r[2]; bv = r[3]
    return (bi, best, bu, bv, n_hit)


def _dir(k: Int, n: Int) -> Vec3f:
    """Fibonacci sphere directions."""
    var z = Float32(1) - (Float32(2) * Float32(k) + Float32(1)) / Float32(n)
    var r = sqrt(max(Float32(0), Float32(1) - z * z))
    var phi = Float32(k) * Float32(2.399963229728653)
    return Vec3f(r * cos(phi), r * sin(phi), z)


def test_patches_watertight_from_inside() raises:
    var patches = _cube_patches(2)
    var misses = 0; var worst = Float32(0)
    for k in range(2000):
        var d = _dir(k, 2000)
        var r = _nearest(patches, Vec3f(0, 0, 0), d)
        if r[0] < 0:
            misses += 1
            continue
        var on = patch_eval(patches[r[0]], r[2], r[3])[0]
        var p = d * r[1]
        var e = p - on
        worst = max(worst, sqrt(dot(e, e)))
    assert_true(misses == 0, String("rays through the closed surface missed: ") + String(misses))
    assert_true(worst < Float32(2e-3), String("hit point off the surface by ") + String(worst))


def test_patches_from_outside() raises:
    var patches = _cube_patches(2)
    for k in range(500):
        var d = _dir(k, 500)
        var r = _nearest(patches, d * Float32(5), -d)
        assert_true(r[0] >= 0, "a ray at the cube missed it")
        var radius = Float32(5) - r[1]
        assert_true(radius > Float32(0.835) and radius < Float32(0.870), String("hit at radius ") + String(radius))


def test_patches_from_far() raises:
    var patches = _cube_patches(2)
    for k in range(200):   # float spacing at 3e5 is coarser than the patches' flatness: the descent must still end
        var d = _dir(k, 200)
        var r = _nearest(patches, d * Float32(3e5), -d)
        assert_true(r[0] >= 0, "a ray at the cube from far missed it")
        var radius = Float32(3e5) - r[1]
        assert_true(radius > Float32(0.78) and radius < Float32(0.93), String("hit at radius ") + String(radius))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
