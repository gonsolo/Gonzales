"""pbrt's TextureMapping2D on an image lookup, then its V-flip t -> 1 - t."""
from std.math import abs, pi
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, TestSuite
from gonzales.core.geometry import Vec3f
from gonzales.scene.appearance.materials import GpuTexture
from gonzales.scene.appearance.image_sample import sample_image

comptime M = SIMD[DType.float32, 16]


def _lookup(m: M, u: Float32, v: Float32, p: Vec3f = Vec3f(0, 0, 0)) raises -> SIMD[DType.float32, 2]:
    # 2x2 float RGB, row 0 at the top: texel (x, y) = (x, y, 0)
    var data = unsafe_alloc[Float32](12)
    for y in range(2):
        for x in range(2):
            data[unsafe_offset=3 * (2 * y + x)] = Float32(x)
            data[unsafe_offset=3 * (2 * y + x) + 1] = Float32(y)
            data[unsafe_offset=3 * (2 * y + x) + 2] = Float32(0)
    var tex = GpuTexture(data.unsafe_bitcast[UInt8](), Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
                         Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(), 2, 2, 1, 3, Int32(GpuTexture.FORMAT_F32), m)
    var c = sample_image(tex, u, v, Float32(0), p)
    data.unsafe_free()
    return SIMD[DType.float32, 2](c.r, c.g)


def _uv(su: Float32, sv: Float32, du: Float32, dv: Float32) -> M:
    var m = GpuTexture.UV_IDENTITY
    m[0] = su; m[1] = sv; m[2] = du; m[3] = dv
    return m


def _close(a: SIMD[DType.float32, 2], r: Float32, g: Float32) -> Bool:
    return abs(a[0] - r) < Float32(1e-4) and abs(a[1] - g) < Float32(1e-4)


def test_identity() raises:
    # s = 0.1: x = 0.1 * 2 - 0.5 wraps between texels 1 and 0
    assert_true(_close(_lookup(GpuTexture.UV_IDENTITY, 0.1, 0.5), 0.3, 0.5))


def test_uscale_udelta() raises:
    assert_true(_close(_lookup(_uv(2, 1, 0.5, 0), 0.1, 0.75), 0.9, 0))


def test_vdelta_before_flip() raises:
    assert_true(_close(_lookup(_uv(1, 1, 0, 0.25), 0.25, 0.5), 0, 0))


def test_planar_reads_the_point() raises:
    # s = p.x = 0.25, t = p.z + 0.25 = 0.75; the uv is ignored
    var m = M(0)
    m[0] = 1; m[6] = 1; m[7] = 0.25; m[15] = Float32(GpuTexture.MAP_PLANAR)
    assert_true(_close(_lookup(m, 0.9, 0.9, Vec3f(0.25, 7, 0.5)), 0, 0))


def test_cylindrical() raises:
    # identity texture space: s = (pi + atan2(y, x)) / 2pi, t = z
    var m = M(0)
    m[0] = 1; m[5] = 1; m[10] = 1; m[15] = Float32(GpuTexture.MAP_CYLINDRICAL)
    assert_true(_close(_lookup(m, 0, 0, Vec3f(0, 1, 0.75)), 1, 0))   # s = 0.75, t = 0.75


def test_spherical() raises:
    # the +y axis: theta = pi/2 -> s = 0.5, phi = pi/2 -> t = 0.25
    var m = M(0)
    m[0] = 1; m[5] = 1; m[10] = 1; m[15] = Float32(GpuTexture.MAP_SPHERICAL)
    assert_true(_close(_lookup(m, 0, 0, Vec3f(0, 3, 0)), 0.5, 1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
