# Unit test for bdpt.mojo's `_bdpt_world_to_raster` -- the world->film
# projection the t=1 light-tracing strategy needs.
#
# Why this deserves its own test: it is the exact inverse of the camera
# transform in sampling.mojo's `gen_primary_ray_state`, and nothing else in
# the renderer inverts that transform, so a sign or transpose error here
# would be invisible until it showed up as misplaced light-traced splats --
# a failure mode that looks like noise rather than like a bug. Round-tripping
# through the forward transform pins it exactly.
from std.math import sqrt, abs, max
from std.memory import alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import Vec3f
from gonzales.bdpt import _bdpt_world_to_raster
from gonzales.transform import matrix_invert

comptime FW: Int32 = 64
comptime FH: Int32 = 48

def _make_camera() -> Tuple[
    UnsafePointer[Float32, MutExternalOrigin],
    UnsafePointer[Float32, MutExternalOrigin],
    UnsafePointer[Float32, MutExternalOrigin],
]:
    """(rasterToCamera, worldToCamera, cameraToRaster3x3) for a simple
    pinhole: film spans [-1,1] x [-0.75,0.75] at z=1, camera at the origin
    looking down +z, so cameraToWorld is the identity."""
    var r2c = alloc[Float32](16)
    for k in range(16): r2c[k] = Float32(0)
    # column-major, matching gen_primary_ray_state's indexing:
    # cx = r2c[0]*fx + r2c[4]*fy + r2c[12]
    r2c[0]  = Float32(2.0) / Float32(FW)        # fx -> x
    r2c[12] = Float32(-1.0)
    r2c[5]  = Float32(-1.5) / Float32(FH)       # fy -> y (film y is flipped)
    r2c[13] = Float32(0.75)
    r2c[14] = Float32(1.0)                      # z = 1 plane
    r2c[15] = Float32(1.0)

    var c2w = alloc[Float32](16)
    for k in range(16): c2w[k] = Float32(0)
    c2w[0] = Float32(1); c2w[5] = Float32(1); c2w[10] = Float32(1); c2w[15] = Float32(1)
    var w2c = alloc[Float32](16)
    _ = matrix_invert(c2w, w2c)

    var c2r = alloc[Float32](9)
    var a0 = r2c[0]; var a1 = r2c[4]; var a2 = r2c[12]
    var b0 = r2c[1]; var b1 = r2c[5]; var b2 = r2c[13]
    var g0 = r2c[2]; var g1 = r2c[6]; var g2 = r2c[14]
    var d0 = b1*g2 - b2*g1
    var d1 = b0*g2 - b2*g0
    var d2 = b0*g1 - b1*g0
    var idet = Float32(1) / (a0*d0 - a1*d1 + a2*d2)
    c2r[0] =  d0*idet; c2r[1] = -(a1*g2 - a2*g1)*idet; c2r[2] =  (a1*b2 - a2*b1)*idet
    c2r[3] = -d1*idet; c2r[4] =  (a0*g2 - a2*g0)*idet; c2r[5] = -(a0*b2 - a2*b0)*idet
    c2r[6] =  d2*idet; c2r[7] = -(a0*g1 - a1*g0)*idet; c2r[8] =  (a0*b1 - a1*b0)*idet
    return (r2c, w2c, c2r)

def test_world_to_raster_round_trips_the_camera_transform() raises:
    var (r2c, w2c, c2r) = _make_camera()
    var worst = Float32(0)
    for ty in range(0, Int(FH), 5):
        for tx in range(0, Int(FW), 5):
            var fx0 = Float32(tx) + Float32(0.5)
            var fy0 = Float32(ty) + Float32(0.5)
            # forward: exactly gen_primary_ray_state's camera transform
            var cx = r2c[0]*fx0 + r2c[4]*fy0 + r2c[12]
            var cy = r2c[1]*fx0 + r2c[5]*fy0 + r2c[13]
            var cz = r2c[2]*fx0 + r2c[6]*fy0 + r2c[14]
            # a world point partway along that ray (c2w is the identity here)
            var t = Float32(7.0) / sqrt(cx*cx + cy*cy + cz*cz)
            var pr = _bdpt_world_to_raster(Vec3f(cx*t, cy*t, cz*t), w2c, c2r, FW, FH)
            assert_true(pr[0])
            var e = max(abs(pr[1] - fx0), abs(pr[2] - fy0))
            if e > worst: worst = e
    assert_true(worst < Float32(1e-3))
    r2c.free(); w2c.free(); c2r.free()

def test_world_to_raster_rejects_points_behind_the_camera() raises:
    var (r2c, w2c, c2r) = _make_camera()
    var behind = _bdpt_world_to_raster(Vec3f(Float32(0), Float32(0), Float32(-5)), w2c, c2r, FW, FH)
    assert_true(not behind[0])
    var off = _bdpt_world_to_raster(Vec3f(Float32(50), Float32(0), Float32(1)), w2c, c2r, FW, FH)
    assert_true(not off[0])
    r2c.free(); w2c.free(); c2r.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
