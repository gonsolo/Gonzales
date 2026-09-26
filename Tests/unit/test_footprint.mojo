# Unit tests for footprint.mojo, pbrt-v4's texture/bump footprint: the (u, v)
# least squares, Camera::Approximate_dp_dxy, the specular-chain cone, the
# minimum-differential search and the instance transform of the triangle.

from std.math import abs, sqrt, cos, sin
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import Vec3f, dot
from gonzales.primitives import TriangleMesh, Instance
from gonzales.footprint import (
    CameraFootprint, UVFootprint, uv_footprint, dp_dxy_approx_camera, dp_dxy_cone,
    camera_footprint, tri_world, hit_uv_footprint, TriWorld,
)


def _close(a: Float32, b: Float32, rel: Float32 = 1e-3) -> Bool:
    return abs(a - b) <= rel * max(abs(b), Float32(1e-6))


def _len(v: Vec3f) -> Float32:
    return sqrt(dot(v, v))


def _cam(m: Float32, spp_scale: Float32) -> CameraFootprint:
    """Camera at the origin looking down +z, identity axes, a minimum
    direction differential of `m` along x and y."""
    return CameraFootprint(
        Vec3f(0.0, 0.0, 0.0), Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0), Vec3f(0.0, 0.0, 1.0),
        Vec3f(m, 0.0, 0.0), Vec3f(0.0, m, 0.0), spp_scale, m * spp_scale)


def test_uv_least_squares_on_a_unit_parameterisation() raises:
    var fp = uv_footprint(Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0),
                          Vec3f(0.2, 0.0, 0.0), Vec3f(0.0, 0.1, 0.0))
    assert_true(_close(fp.du, 0.1))     # .5 * (|dudx| + |dudy|) = .5 * 0.2
    assert_true(_close(fp.dv, 0.05))
    assert_true(_close(fp.width, 0.4))  # 2 * max |d(u,v)/d(x,y)|


def test_uv_least_squares_follows_the_parameterisation_scale() raises:
    """A surface whose u runs twice as fast in space moves half as far in u."""
    var fp = uv_footprint(Vec3f(2.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0),
                          Vec3f(0.2, 0.0, 0.0), Vec3f(0.0, 0.2, 0.0))
    assert_true(_close(fp.du, 0.05))
    assert_true(_close(fp.dv, 0.1))


def test_degenerate_parameterisation_gives_no_footprint() raises:
    var fp = uv_footprint(Vec3f(1.0, 0.0, 0.0), Vec3f(2.0, 0.0, 0.0),
                          Vec3f(0.2, 0.0, 0.0), Vec3f(0.0, 0.2, 0.0))
    assert_true(fp.du == 0.0 and fp.dv == 0.0 and fp.width == 0.0)


def test_approx_camera_on_a_facing_plane() raises:
    """p on the optical axis at distance 5, plane facing the camera: the
    footprint is distance x minimum differential x spp scale."""
    var m = Float32(1e-3)
    var d = dp_dxy_approx_camera(_cam(m, 0.5), Vec3f(0.0, 0.0, 5.0), Vec3f(0.0, 0.0, -1.0))
    assert_true(_close(d[0][0], 5.0 * m * 0.5))
    assert_true(abs(d[0][1]) < 1e-9 and abs(d[0][2]) < 1e-9)
    assert_true(_close(_len(d[1]), 5.0 * m * 0.5))


def test_approx_camera_stretches_by_one_over_cos_on_a_tilted_plane() raises:
    """The tangent-plane intersection pbrt uses: a plane tilted 60 degrees
    about y stretches the x footprint by 1/cos(60) = 2 and leaves y alone."""
    var m = Float32(1e-4)
    var th = Float32(3.14159265 / 3.0)
    var n = Vec3f(sin(th), 0.0, -cos(th))
    var d = dp_dxy_approx_camera(_cam(m, 1.0), Vec3f(0.0, 0.0, 5.0), n)
    assert_true(_close(_len(d[0]), 10.0 * m, 1e-2))
    assert_true(_close(_len(d[1]), 5.0 * m, 1e-2))
    assert_true(abs(dot(d[0], n)) < Float32(1e-2) * _len(d[0]))   # stays in the tangent plane


def test_approx_camera_is_invariant_under_a_camera_rotation() raises:
    """Same view geometry, camera turned 90 degrees about y: same magnitudes."""
    var m = Float32(1e-3)
    var cam = CameraFootprint(
        Vec3f(1.0, 2.0, 3.0), Vec3f(0.0, 0.0, -1.0), Vec3f(0.0, 1.0, 0.0), Vec3f(1.0, 0.0, 0.0),
        Vec3f(m, 0.0, 0.0), Vec3f(0.0, m, 0.0), Float32(1.0), m)
    var d = dp_dxy_approx_camera(cam, Vec3f(6.0, 2.0, 3.0), Vec3f(-1.0, 0.0, 0.0))
    assert_true(_close(_len(d[0]), 5.0 * m))
    assert_true(_close(_len(d[1]), 5.0 * m))


def test_cone_matches_the_approximation_on_axis() raises:
    var m = Float32(1e-4)
    var th = Float32(3.14159265 / 3.0)
    var n = Vec3f(sin(th), 0.0, -cos(th))
    var c = dp_dxy_cone(_cam(m, 1.0), n, Vec3f(0.0, 0.0, 1.0), Float32(5.0) * m)
    var a = dp_dxy_approx_camera(_cam(m, 1.0), Vec3f(0.0, 0.0, 5.0), n)
    assert_true(_close(_len(c[0]), _len(a[0]), 1e-2))
    assert_true(_close(_len(c[1]), _len(a[1]), 1e-2))


def test_camera_footprint_finds_the_corner_differential() raises:
    """An affine raster->camera map (x' = a x + b, y' = a y + b, z = 1): the
    normalized direction changes least per pixel at the film corners, by
    a sqrt(1 + y'^2) / (1 + x'^2 + y'^2) -- pbrt's minimum differential."""
    var w = 200; var h = 100
    var a = Float32(0.01)
    var r2c = unsafe_alloc[Float32](16)
    for i in range(16): r2c[unsafe_offset=i] = Float32(0)
    r2c[unsafe_offset=0] = a; r2c[unsafe_offset=5] = a
    r2c[unsafe_offset=12] = -a * Float32(w) * 0.5; r2c[unsafe_offset=13] = -a * Float32(h) * 0.5
    r2c[unsafe_offset=14] = Float32(1); r2c[unsafe_offset=15] = Float32(1)
    var c2w = unsafe_alloc[Float32](16)
    for i in range(16): c2w[unsafe_offset=i] = Float32(0)
    c2w[unsafe_offset=0] = 1; c2w[unsafe_offset=5] = 1; c2w[unsafe_offset=10] = 1; c2w[unsafe_offset=15] = 1
    var cf = camera_footprint(r2c, c2w, w, h, 16)
    var xc = a * Float32(w) * 0.5; var yc = a * Float32(h) * 0.5
    var r2 = Float32(1) + xc * xc + yc * yc
    assert_true(_close(_len(cf.min_dx), a * sqrt(Float32(1) + yc * yc) / r2, 1e-2))
    assert_true(_close(_len(cf.min_dy), a * sqrt(Float32(1) + xc * xc) / r2, 1e-2))
    assert_true(_close(cf.spp_scale, 0.25))
    assert_true(_close(cf.cone_spread, a * 0.25, 1e-2))
    assert_true(_close(camera_footprint(r2c, c2w, w, h, 4096).spp_scale, 0.125))
    r2c.unsafe_free(); c2w.unsafe_free()


def test_tri_world_applies_the_instance_transform() raises:
    """Bump tangents are built from these corners; object space would mix a
    rotated instance's tangents with its world-space normal."""
    var pts = unsafe_alloc[Float32](12)
    for i in range(12): pts[unsafe_offset=i] = Float32(0)
    pts[unsafe_offset=4] = 1.0          # p1 = (1, 0, 0)
    pts[unsafe_offset=9] = 1.0          # p2 = (0, 1, 0)
    var mesh = TriangleMesh(pts, Pointer[Int64, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Int64, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling())
    # objToWorld: rotate 90 degrees about z (x -> y), then translate (1, 2, 3); column-major.
    var o2w = SIMD[DType.float32, 16](0)
    o2w[1] = 1.0; o2w[4] = -1.0; o2w[10] = 1.0; o2w[15] = 1.0
    o2w[12] = 1.0; o2w[13] = 2.0; o2w[14] = 3.0
    var inst = unsafe_alloc[Instance](1)
    inst[unsafe_offset=0] = Instance(o2w, SIMD[DType.float32, 16](0), Int32(0))
    var tw = tri_world(mesh, 0, 1, 2, Int32(0), inst)
    assert_true(_close(tw.p0[0], 1.0) and _close(tw.p0[1], 2.0) and _close(tw.p0[2], 3.0))
    assert_true(_close(tw.p1[0], 1.0) and _close(tw.p1[1], 3.0))   # (1,0,0) -> (0,1,0) + t
    assert_true(_close(tw.p2[0], 0.0) and _close(tw.p2[1], 2.0))   # (0,1,0) -> (-1,0,0) + t
    var tn = tri_world(mesh, 0, 1, 2, Int32(-1), inst)
    assert_true(_close(tn.p1[0], 1.0) and abs(tn.p1[1]) < 1e-9)
    inst.unsafe_free(); pts.unsafe_free()


def test_no_camera_means_no_footprint() raises:
    var fp = hit_uv_footprint(CameraFootprint.none(), _dummy_tri(), _dummy_mesh(), 0, 1, 2,
        Vec3f(0.0, 0.0, 1.0), Vec3f(0.0, 0.0, 1.0), Float32(1.0))
    assert_true(fp.du == 0.0 and fp.width == 0.0)


def _dummy_mesh() -> TriangleMesh:
    return TriangleMesh(Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Int64, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Int64, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling())


def _dummy_tri() -> TriWorld:
    var z = Vec3f(0.0, 0.0, 0.0)
    return TriWorld(z, Vec3f(1.0, 0.0, 0.0), Vec3f(0.0, 1.0, 0.0), z, z, z, False)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
