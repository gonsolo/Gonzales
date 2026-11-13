from std.math import abs, sqrt
from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_equal, TestSuite
from gonzales.scene.appearance.materials import GpuTexture
from gonzales.scene.appearance.shader import (
    F4, ShaderOp, MathOp, VectorOp, Blend, MapRange, Mapping, ImageExt, eval_shader, shader_texture_name, is_shader_name,
    shader_words, shader_images,
)


struct Prog(Movable):
    var w: List[Int32]

    def __init__(out self):
        self.w = [Int32(0)]

    def op(mut self, op: Int32, dst: Int, a: Int = 0, b: Int = 0, c: Int = 0, d: Int = 0, i0: Int = 0, i1: Int = 0):
        self.w.append(op); self.w.append(Int32(dst)); self.w.append(Int32(a)); self.w.append(Int32(b))
        self.w.append(Int32(c)); self.w.append(Int32(d)); self.w.append(Int32(i0)); self.w.append(Int32(i1))

    def const(mut self, dst: Int, x: Float32, y: Float32, z: Float32, w: Float32 = 1):
        self.w.append(ShaderOp.const); self.w.append(Int32(dst))
        for f in [x, y, z, w]:
            self.w.append(bitcast[DType.int32](f))
        self.w.append(0); self.w.append(0)

    def value(mut self, dst: Int, x: Float32):
        self.const(dst, x, x, x, x)

    def table(mut self, entries: List[Float32]):
        for f in entries:
            self.w.append(bitcast[DType.int32](f))

    def run(mut self, out_reg: Int, u: Float32 = 0, v: Float32 = 0,
            textures: Pointer[GpuTexture, MutUntrackedOrigin] = Pointer[GpuTexture, MutUntrackedOrigin].unsafe_dangling(),
            n_textures: Int = 0) -> F4:
        self.w[0] = Int32(out_reg)
        return eval_shader(self.w.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), len(self.w), u, v, Float32(0),
                           textures, n_textures)


def _close(a: F4, b: F4, eps: Float32 = 1e-4) -> Bool:
    for k in range(4):
        if abs(a[k] - b[k]) > eps:
            return False
    return True


def _close3(a: F4, b: F4, eps: Float32 = 1e-4) -> Bool:
    return _close(F4(a[0], a[1], a[2], 0), F4(b[0], b[1], b[2], 0), eps)


def test_const_and_uv() raises:
    var p = Prog()
    p.const(3, 0.25, 0.5, 0.75)
    assert_true(_close(p.run(3), F4(0.25, 0.5, 0.75, 1)))
    p.op(ShaderOp.uv, 1)
    assert_true(_close(p.run(1, 0.3, 0.6), F4(0.3, 0.6, 0, 1)))


def test_math() raises:
    var p = Prog()
    p.value(0, 2.0); p.value(1, 3.0); p.value(2, -5.5)
    p.op(ShaderOp.math, 3, 0, 1, i0=MathOp.power)
    p.op(ShaderOp.math, 4, 2, 1, i0=MathOp.modulo)
    p.op(ShaderOp.math, 5, 2, 1, i0=MathOp.floored_modulo)
    p.op(ShaderOp.math, 6, 0, 1, i0=MathOp.sub, i1=1)
    p.op(ShaderOp.math, 7, 2, 1, 0, i0=MathOp.wrap)
    p.op(ShaderOp.math, 8, 0, 1, 1, i0=MathOp.mul_add)
    p.op(ShaderOp.math, 9, 1, 0, i0=MathOp.div)
    assert_true(_close(p.run(3), F4(8.0)))
    assert_true(_close(p.run(4), F4(-2.5)))
    assert_true(_close(p.run(5), F4(0.5)))
    assert_true(_close(p.run(6), F4(0.0)))   # clamped
    assert_true(_close(p.run(7), F4(2.5)))   # wrap(-5.5, 3, 2): range 1 -> 2.5
    assert_true(_close(p.run(8), F4(9.0)))
    assert_true(_close(p.run(9), F4(1.5)))


def test_vector_math() raises:
    var p = Prog()
    p.const(0, 1, 0, 0); p.const(1, 0, 1, 0)
    p.op(ShaderOp.vmath, 2, 0, 1, i0=VectorOp.cross)
    p.op(ShaderOp.vmath, 3, 0, 1, i0=VectorOp.dot)
    p.op(ShaderOp.math, 4, 0, 1, i0=MathOp.add)
    p.op(ShaderOp.vmath, 5, 4, i0=VectorOp.length)
    assert_true(_close3(p.run(2), F4(0, 0, 1, 0)))
    assert_true(_close3(p.run(3), F4(0)))
    assert_true(_close3(p.run(5), F4(sqrt(Float32(2)))))


def test_blend_modes() raises:
    var p = Prog()
    p.value(0, 0.5)
    p.const(1, 0.2, 0.4, 0.8); p.const(2, 0.6, 0.6, 0.6)
    p.op(ShaderOp.blend, 3, 0, 1, 2, i0=Blend.mix)
    p.op(ShaderOp.blend, 4, 0, 1, 2, i0=Blend.mul)
    p.op(ShaderOp.blend, 5, 0, 1, 2, i0=Blend.add)
    p.op(ShaderOp.blend, 6, 0, 1, 2, i0=Blend.screen)
    p.op(ShaderOp.blend, 7, 0, 1, 2, i0=Blend.overlay)
    assert_true(_close3(p.run(3), F4(0.4, 0.5, 0.7, 0)))
    assert_true(_close3(p.run(4), F4(0.16, 0.32, 0.64, 0)))   # a * (0.5 + 0.5 b)
    assert_true(_close3(p.run(5), F4(0.5, 0.7, 1.1, 0)))
    # screen: 1 - (0.5 + 0.5 (1 - b)) (1 - a)
    assert_true(_close3(p.run(6), F4(1 - 0.7 * 0.8, 1 - 0.7 * 0.6, 1 - 0.7 * 0.2, 0)))
    # overlay: a < .5: a (0.5 + b); else 1 - (0.5 + (1 - b)) (1 - a)
    assert_true(_close3(p.run(7), F4(0.2 * 1.1, 0.4 * 1.1, 1 - 0.9 * 0.2, 0)))


def test_hsv_and_colour_ops() raises:
    var p = Prog()
    p.const(0, 0.8, 0.4, 0.2)
    p.value(1, 0.5); p.value(2, 1.0); p.value(3, 0.0)
    p.op(ShaderOp.hsv, 4, 0, 1, 2, 2, i0=2)   # hue 0.5, saturation 1, value 1: unchanged
    p.op(ShaderOp.invert, 5, 0, 2)
    p.op(ShaderOp.bright, 6, 0, 3, 3)
    p.op(ShaderOp.hsv_rgb, 7, 0, i0=0)
    p.op(ShaderOp.hsv_rgb, 8, 7, i0=1)
    p.op(ShaderOp.luma, 9, 0)
    p.op(ShaderOp.split, 10, 0, i0=1)
    p.op(ShaderOp.combine, 11, 10, 1, 2)
    assert_true(_close3(p.run(4), F4(0.8, 0.4, 0.2, 0)))
    assert_true(_close3(p.run(5), F4(0.2, 0.6, 0.8, 0)))
    assert_true(_close3(p.run(6), F4(0.8, 0.4, 0.2, 0)))
    assert_true(_close3(p.run(7), F4(1.0 / 18.0, 0.75, 0.8, 0)))
    assert_true(_close3(p.run(8), F4(0.8, 0.4, 0.2, 0)))
    assert_true(_close3(p.run(9), F4(0.8 * 0.2126 + 0.4 * 0.7152 + 0.2 * 0.0722)))
    assert_true(_close3(p.run(11), F4(0.4, 0.5, 1.0, 0)))


def test_ramp_and_map_range() raises:
    var p = Prog()
    p.value(0, 0.25)
    p.op(ShaderOp.ramp, 1, 0, i0=3, i1=1)
    p.table([Float32(0), 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1])
    p.op(ShaderOp.ramp, 2, 0, i0=3, i1=0)
    p.table([Float32(0), 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1])
    p.value(3, 0.0); p.value(4, 0.5); p.value(5, 10.0); p.value(6, 20.0); p.value(7, 4.0)
    p.op(ShaderOp.map_range, 8, 0, 3, 4, 5, i0=MapRange.linear, i1=6 | (7 << 8))
    p.value(9, 2.0)
    p.op(ShaderOp.map_range, 10, 9, 3, 4, 5, i0=MapRange.linear, i1=6 | (7 << 8) | (1 << 16))
    assert_true(_close(p.run(1), F4(0.5, 0, 0, 1)))
    assert_true(_close(p.run(2), F4(0, 0, 0, 1)))
    assert_true(_close3(p.run(8), F4(15.0)))
    assert_true(_close3(p.run(10), F4(20.0)))


def test_mapping() raises:
    var p = Prog()
    p.const(0, 1, 0, 0); p.const(1, 0.5, 0, 0); p.const(2, 0, 0, Float32(1.5707963)); p.const(3, 2, 2, 2)
    p.op(ShaderOp.mapping, 4, 0, 1, 2, 3, i0=Mapping.point)
    p.op(ShaderOp.mapping, 5, 4, 1, 2, 3, i0=Mapping.texture)
    assert_true(_close3(p.run(4), F4(0.5, 2, 0, 0)))
    assert_true(_close3(p.run(5), F4(1, 0, 0, 0)))   # texture undoes point


def test_normal_map_strength() raises:
    var p = Prog()
    p.const(0, 0.8, 0.5, 0.9); p.value(1, 1.0); p.value(2, 0.0)
    p.op(ShaderOp.normal_map, 3, 0, 1)
    p.op(ShaderOp.normal_map, 4, 0, 2)
    assert_true(_close3(p.run(3), F4(0.8, 0.5, 0.9, 0)))
    assert_true(_close3(p.run(4), F4(0.5, 0.5, 1.0, 0)))


def test_image() raises:
    # 2x2 float RGB, row 0 at the top: texel (x, y) = (x, y, 0.5)
    var data = unsafe_alloc[Float32](12)
    for y in range(2):
        for x in range(2):
            data[unsafe_offset=3 * (2 * y + x)] = Float32(x)
            data[unsafe_offset=3 * (2 * y + x) + 1] = Float32(y)
            data[unsafe_offset=3 * (2 * y + x) + 2] = Float32(0.5)
    var tex = unsafe_alloc[GpuTexture](1)
    tex[unsafe_offset=0] = GpuTexture(data.unsafe_bitcast[UInt8](), Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
                                      Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(), 2, 2, 1, 3, Int32(GpuTexture.FORMAT_F32))
    var p = Prog()
    p.op(ShaderOp.uv, 0)
    p.op(ShaderOp.image, 1, 0, i0=0)
    p.const(2, 1.25, 0.25, 0)
    p.op(ShaderOp.image, 3, 2, i0=0, i1=Int(ImageExt.clip) << 1)
    p.op(ShaderOp.image, 4, 2, i0=0)
    # Blender's v points up: v = 0.25 is the bottom row, the image's row 1
    assert_true(_close(p.run(1, 0.75, 0.25, tex, 1), F4(1, 1, 0.5, 1)))
    assert_true(_close(p.run(1, 0.25, 0.75, tex, 1), F4(0, 0, 0.5, 1)))
    assert_true(_close(p.run(3, 0, 0, tex, 1), F4(0)))
    assert_true(_close(p.run(4, 0, 0, tex, 1), F4(0, 1, 0.5, 1)))   # repeat
    tex.unsafe_free(); data.unsafe_free()


def test_texture_name_round_trip() raises:
    var p = Prog()
    p.op(ShaderOp.uv, 0)
    p.op(ShaderOp.image, 1, 0, i0=7, i1=1)
    p.value(2, -0.5)
    p.op(ShaderOp.ramp, 3, 2, i0=2)
    p.table([Float32(0), 0, 0, 1, 1, 1, 1, 1])
    p.op(ShaderOp.image, 4, 0, i0=9)
    var name = shader_texture_name(p.w)
    var c = name + "\0"
    var ptr = c.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]().unsafe_mut_cast[True]()
    assert_true(is_shader_name(ptr))
    var back = shader_words(ptr)
    assert_equal(len(back), len(p.w))
    for i in range(len(back)):
        assert_equal(back[i], p.w[i])
    var images = shader_images(back)
    assert_equal(len(images), 2)
    assert_equal(images[0][0], 7)
    assert_true(images[0][1])
    assert_equal(images[1][0], 9)
    assert_true(not images[1][1])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
