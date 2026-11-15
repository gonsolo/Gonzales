"""pbrt's UVMapping on an image lookup: s = su*u + du, t = 1 - (sv*v + dv)."""
from std.math import abs
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, TestSuite
from gonzales.scene.appearance.materials import GpuTexture
from gonzales.scene.appearance.image_sample import sample_image


def _lookup(m: SIMD[DType.float32, 4], u: Float32, v: Float32) raises -> SIMD[DType.float32, 2]:
    # 2x2 float RGB, row 0 at the top: texel (x, y) = (x, y, 0)
    var data = unsafe_alloc[Float32](12)
    for y in range(2):
        for x in range(2):
            data[unsafe_offset=3 * (2 * y + x)] = Float32(x)
            data[unsafe_offset=3 * (2 * y + x) + 1] = Float32(y)
            data[unsafe_offset=3 * (2 * y + x) + 2] = Float32(0)
    var tex = GpuTexture(data.unsafe_bitcast[UInt8](), Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
                         Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(), 2, 2, 1, 3, Int32(GpuTexture.FORMAT_F32), m)
    var c = sample_image(tex, u, v, Float32(0))
    data.unsafe_free()
    return SIMD[DType.float32, 2](c.r, c.g)


def _close(a: SIMD[DType.float32, 2], r: Float32, g: Float32) -> Bool:
    return abs(a[0] - r) < Float32(1e-5) and abs(a[1] - g) < Float32(1e-5)


def test_identity() raises:
    assert_true(_close(_lookup(GpuTexture.UV_IDENTITY, 0.1, 0.5), 0.3, 0.5))


def test_uscale_udelta() raises:
    assert_true(_close(_lookup(SIMD[DType.float32, 4](2, 1, 0.5, 0), 0.1, 0.75), 0.9, 0))


def test_vdelta_before_flip() raises:
    assert_true(_close(_lookup(SIMD[DType.float32, 4](1, 1, 0, 0.25), 0.25, 0.5), 0, 0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
