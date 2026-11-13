"""Material node graphs as programs (Cycles' SVM): one program per texture slot, kept in the texture table, run at each lookup.

A program is Int32 words: the output register, then instructions of SHADER_STRIDE words
[op, dst, a, b, c, d, i0, i1] (a..d registers, i0/i1 immediates; CONST keeps its four floats in a..d).
RAMP and CURVES are followed by i0 RGBA table entries. Registers hold four floats; a float value fills all lanes.
The Blender add-on compiles node trees to it (src/hdgonzales/blender/shader.py)."""
from std.math import sqrt, floor, ceil, sin, cos, exp, log, abs, min, max
from std.memory import bitcast
from gonzales.core.geometry import _atan2f
from gonzales.scene.appearance.materials import GpuTexture
from gonzales.scene.appearance.image_sample import _sample_tex, _footprint_lod

comptime F4 = SIMD[DType.float32, 4]
comptime SHADER_STRIDE = 8
comptime SHADER_REGS = 16
comptime SHADER_PREFIX = "#gzs:"   # a texture table name holding a program, its words in hex


struct ShaderOp:
    comptime const = Int32(1)
    comptime uv = Int32(2)
    comptime image = Int32(3)      # i0 texture, i1 flags: 1 data (no sRGB decode), 2..4 extension, see ImageExt
    comptime math = Int32(4)       # lane-wise MathOp i0 of a, b, c; i1 = 1 clamps to [0, 1]
    comptime vmath = Int32(5)      # VectorOp i0 of a, b, c, scale d
    comptime mix = Int32(6)        # a + (b - a) * c
    comptime blend = Int32(7)      # Blend i0 of colours b, c by factor a; i1 = 1 clamps the result
    comptime clamp = Int32(8)      # min(max(a, b), c)
    comptime map_range = Int32(9)  # MapRange i0 of a from [b, c] to [d, reg i1 & 255]; steps reg (i1 >> 8) & 255; clamp bit 16
    comptime ramp = Int32(10)      # table of i0 entries at a; i1 = 1 interpolates
    comptime hsv = Int32(11)       # colour a, hue b, saturation c, value d, factor reg i0
    comptime invert = Int32(12)    # colour a, factor b
    comptime bright = Int32(13)    # colour a, brightness b, contrast c
    comptime gamma = Int32(14)     # colour a, gamma b
    comptime split = Int32(15)     # lane i0 of a
    comptime combine = Int32(16)   # (a, b, c)
    comptime hsv_rgb = Int32(17)   # i0 = 0: RGB to HSV, 1: HSV to RGB
    comptime mapping = Int32(18)   # Mapping i0 of vector a: location b, rotation c, scale d
    comptime normal_map = Int32(19)  # tangent-space colour a at strength b
    comptime curves = Int32(20)    # per channel table of i0 entries over [0, 1] at colour a, factor b
    comptime luma = Int32(21)      # colour to float
    comptime average = Int32(22)   # vector to float


struct ImageExt:
    comptime repeat = Int32(0)
    comptime extend = Int32(1)
    comptime clip = Int32(2)
    comptime mirror = Int32(3)


struct MathOp:
    comptime add = 0; comptime sub = 1; comptime mul = 2; comptime div = 3; comptime mul_add = 4; comptime power = 5
    comptime log = 6; comptime sqrt = 7; comptime inv_sqrt = 8; comptime abs = 9; comptime exp = 10; comptime min = 11
    comptime max = 12; comptime less = 13; comptime greater = 14; comptime sign = 15; comptime compare = 16
    comptime smooth_min = 17; comptime smooth_max = 18; comptime round = 19; comptime floor = 20; comptime ceil = 21
    comptime trunc = 22; comptime fract = 23; comptime modulo = 24; comptime floored_modulo = 25; comptime wrap = 26
    comptime snap = 27; comptime pingpong = 28; comptime sin = 29; comptime cos = 30; comptime tan = 31
    comptime asin = 32; comptime acos = 33; comptime atan = 34; comptime atan2 = 35; comptime sinh = 36
    comptime cosh = 37; comptime tanh = 38; comptime radians = 39; comptime degrees = 40


struct VectorOp:
    comptime cross = 0; comptime project = 1; comptime reflect = 2; comptime refract = 3; comptime faceforward = 4
    comptime dot = 5; comptime distance = 6; comptime length = 7; comptime scale = 8; comptime normalize = 9


struct Blend:
    comptime mix = 0; comptime add = 1; comptime mul = 2; comptime sub = 3; comptime screen = 4; comptime div = 5
    comptime diff = 6; comptime dark = 7; comptime light = 8; comptime overlay = 9; comptime dodge = 10
    comptime burn = 11; comptime hue = 12; comptime sat = 13; comptime val = 14; comptime color = 15
    comptime soft = 16; comptime linear = 17; comptime exclusion = 18


struct MapRange:
    comptime linear = 0; comptime stepped = 1; comptime smoothstep = 2; comptime smootherstep = 3


struct Mapping:
    comptime point = 0; comptime texture = 1; comptime vector = 2; comptime normal = 3


# ── The program as a texture table name ──────────────────────────────────────

def shader_texture_name(code: List[Int32]) -> String:
    comptime digits = "0123456789abcdef"
    var s = String(SHADER_PREFIX)
    for w in code:
        var u = UInt32(w)
        for k in range(8):
            s += String(digits[byte=Int((u >> UInt32(28 - 4 * k)) & 15)])
    return s^


def is_shader_name(name: Pointer[UInt8, MutUntrackedOrigin]) -> Bool:
    var p = SHADER_PREFIX.unsafe_ptr()
    for k in range(SHADER_PREFIX.byte_length()):
        if name[unsafe_offset=k] != p[unsafe_offset=k]:
            return False
    return True


def shader_words(name: Pointer[UInt8, MutUntrackedOrigin]) -> List[Int32]:
    """The program a shader texture name holds; empty when it is malformed."""
    var out = List[Int32]()
    var i = SHADER_PREFIX.byte_length()
    while name[unsafe_offset=i] != UInt8(0):
        var u = UInt32(0)
        for k in range(8):
            var c = Int(name[unsafe_offset=i + k])
            var d = c - 48 if c >= 48 and c <= 57 else (c - 87 if c >= 97 and c <= 102 else -1)
            if d < 0:
                return List[Int32]()
            u = (u << 4) | UInt32(d)
        out.append(Int32(u))
        i += 8
    return out^


@always_inline
def instruction_words(op: Int32, i0: Int32) -> Int:
    if op == ShaderOp.ramp or op == ShaderOp.curves:
        return SHADER_STRIDE + 4 * Int(i0)
    return SHADER_STRIDE


def shader_images(code: List[Int32]) -> List[Tuple[Int, Bool]]:
    """The textures a program reads, each with whether it reads them as data."""
    var out = List[Tuple[Int, Bool]]()
    var pc = 1
    while pc + SHADER_STRIDE <= len(code):
        if code[pc] == ShaderOp.image:
            out.append((Int(code[pc + 6]), (code[pc + 7] & 1) != 0))
        pc += instruction_words(code[pc], code[pc + 6])
    return out^


# ── The interpreter ──────────────────────────────────────────────────────────

@always_inline
def _f(w: Int32) -> Float32:
    return bitcast[DType.float32](w)


@always_inline
def _div(a: Float32, b: Float32) -> Float32:
    return a / b if b != Float32(0) else Float32(0)


@always_inline
def _fract(a: Float32) -> Float32:
    return a - floor(a)


@always_inline
def _asin(a: Float32) -> Float32:
    var x = min(max(a, Float32(-1)), Float32(1))
    return _atan2f(x, sqrt(max(Float32(1) - x * x, Float32(0))))


def _math(op: Int, a: Float32, b: Float32, c: Float32) -> Float32:
    """Cycles' svm_math."""
    if op == MathOp.add: return a + b
    if op == MathOp.sub: return a - b
    if op == MathOp.mul: return a * b
    if op == MathOp.div: return _div(a, b)
    if op == MathOp.mul_add: return a * b + c
    if op == MathOp.power:
        if a < Float32(0) and b != floor(b):
            return Float32(0)
        if a == Float32(0):
            return Float32(1) if b == Float32(0) else Float32(0)
        var p = exp(b * log(abs(a)))
        return -p if a < Float32(0) and Int(b) % 2 != 0 else p
    if op == MathOp.log: return _div(log(a), log(b)) if a > Float32(0) and b > Float32(0) else Float32(0)
    if op == MathOp.sqrt: return sqrt(max(a, Float32(0)))
    if op == MathOp.inv_sqrt: return Float32(1) / sqrt(a) if a > Float32(0) else Float32(0)
    if op == MathOp.abs: return abs(a)
    if op == MathOp.exp: return exp(a)
    if op == MathOp.min: return min(a, b)
    if op == MathOp.max: return max(a, b)
    if op == MathOp.less: return Float32(1) if a < b else Float32(0)
    if op == MathOp.greater: return Float32(1) if a > b else Float32(0)
    if op == MathOp.sign: return Float32(1) if a > Float32(0) else (Float32(-1) if a < Float32(0) else Float32(0))
    if op == MathOp.compare: return Float32(1) if a == b or abs(a - b) <= max(c, Float32(1.1920929e-7)) else Float32(0)
    if op == MathOp.smooth_min or op == MathOp.smooth_max:
        var s = Float32(1) if op == MathOp.smooth_min else Float32(-1)
        var x = a * s; var y = b * s
        if c == Float32(0):
            return min(x, y) * s
        var h = max(c - abs(x - y), Float32(0)) / c
        return (min(x, y) - h * h * h * c * Float32(1.0 / 6.0)) * s
    if op == MathOp.round: return floor(a + Float32(0.5))
    if op == MathOp.floor: return floor(a)
    if op == MathOp.ceil: return ceil(a)
    if op == MathOp.trunc: return floor(a) if a >= Float32(0) else ceil(a)
    if op == MathOp.fract: return _fract(a)
    if op == MathOp.modulo:
        if b == Float32(0): return Float32(0)
        var q = a / b
        return a - (floor(q) if q >= Float32(0) else ceil(q)) * b
    if op == MathOp.floored_modulo: return a - floor(a / b) * b if b != Float32(0) else Float32(0)
    if op == MathOp.wrap:
        var r = b - c
        return a - r * floor((a - c) / r) if r != Float32(0) else c
    if op == MathOp.snap: return floor(_div(a, b)) * b
    if op == MathOp.pingpong:
        return abs(_fract((a - b) / (b * Float32(2))) * b * Float32(2) - b) if b != Float32(0) else Float32(0)
    if op == MathOp.sin: return sin(a)
    if op == MathOp.cos: return cos(a)
    if op == MathOp.tan: return sin(a) / cos(a)
    if op == MathOp.asin: return _asin(a)
    if op == MathOp.acos: return Float32(1.5707963) - _asin(a)
    if op == MathOp.atan: return _atan2f(a, Float32(1))
    if op == MathOp.atan2: return _atan2f(a, b)
    if op == MathOp.sinh: return (exp(a) - exp(-a)) * Float32(0.5)
    if op == MathOp.cosh: return (exp(a) + exp(-a)) * Float32(0.5)
    if op == MathOp.tanh:
        var e = exp(Float32(-2) * abs(a))
        var t = (Float32(1) - e) / (Float32(1) + e)
        return t if a >= Float32(0) else -t
    if op == MathOp.radians: return a * Float32(0.017453292)
    if op == MathOp.degrees: return a * Float32(57.29578)
    return Float32(0)


@always_inline
def _dot3(a: F4, b: F4) -> Float32:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@always_inline
def _normalize3(a: F4) -> F4:
    var l = sqrt(_dot3(a, a))
    return a / l if l > Float32(0) else F4(0)


def _vmath(op: Int, a: F4, b: F4, c: F4, s: Float32) -> F4:
    """Cycles' svm_vector_math for the ops that are not lane-wise."""
    if op == VectorOp.cross:
        return F4(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0], 0)
    if op == VectorOp.project:
        var bb = _dot3(b, b)
        return b * (_dot3(a, b) / bb) if bb != Float32(0) else F4(0)
    if op == VectorOp.reflect:
        var n = _normalize3(b)
        return a - n * (Float32(2) * _dot3(n, a))
    if op == VectorOp.refract:
        var n = _normalize3(b)
        var d = _dot3(n, a)
        var k = Float32(1) - s * s * (Float32(1) - d * d)
        return a * s - n * (s * d + sqrt(k)) if k >= Float32(0) else F4(0)
    if op == VectorOp.faceforward:
        return a if _dot3(c, b) < Float32(0) else -a
    if op == VectorOp.dot: return F4(_dot3(a, b))
    if op == VectorOp.distance: return F4(sqrt(_dot3(a - b, a - b)))
    if op == VectorOp.length: return F4(sqrt(_dot3(a, a)))
    if op == VectorOp.scale: return a * s
    if op == VectorOp.normalize: return _normalize3(a)
    return F4(0)


def _rgb_to_hsv(c: F4) -> F4:
    var cmax = max(c[0], max(c[1], c[2]))
    var cmin = min(c[0], min(c[1], c[2]))
    var delta = cmax - cmin
    var s = delta / cmax if cmax != Float32(0) else Float32(0)
    var h = Float32(0)
    if s != Float32(0):
        var cr = (cmax - c[0]) / delta; var cg = (cmax - c[1]) / delta; var cb = (cmax - c[2]) / delta
        if c[0] == cmax: h = cb - cg
        elif c[1] == cmax: h = Float32(2) + cr - cb
        else: h = Float32(4) + cg - cr
        h /= Float32(6)
        if h < Float32(0): h += Float32(1)
    return F4(h, s, cmax, c[3])


def _hsv_to_rgb(hsv: F4) -> F4:
    var h = hsv[0]; var s = hsv[1]; var v = hsv[2]
    if s == Float32(0):
        return F4(v, v, v, hsv[3])
    if h == Float32(1): h = Float32(0)
    h *= Float32(6)
    var i = floor(h)
    var f = h - i
    var p = v * (Float32(1) - s); var q = v * (Float32(1) - s * f); var t = v * (Float32(1) - s * (Float32(1) - f))
    if i == Float32(0): return F4(v, t, p, hsv[3])
    if i == Float32(1): return F4(q, v, p, hsv[3])
    if i == Float32(2): return F4(p, v, t, hsv[3])
    if i == Float32(3): return F4(p, q, v, hsv[3])
    if i == Float32(4): return F4(t, p, v, hsv[3])
    return F4(v, p, q, hsv[3])


def _blend(mode: Int, t: Float32, a: F4, b: F4) -> F4:
    """Cycles' svm_mix (colour blend modes); alpha is a's."""
    var tm = Float32(1) - t
    var r = a
    if mode == Blend.mix: r = a * tm + b * t
    elif mode == Blend.add: r = a + b * t
    elif mode == Blend.mul: r = a * (F4(tm) + b * t)
    elif mode == Blend.sub: r = a - b * t
    elif mode == Blend.screen: r = F4(1) - (F4(tm) + (F4(1) - b) * t) * (F4(1) - a)
    elif mode == Blend.diff: r = a * tm + abs(a - b) * t
    elif mode == Blend.dark: r = a * tm + min(a, b) * t
    elif mode == Blend.light: r = a * tm + max(a, b) * t
    elif mode == Blend.exclusion: r = max(a * tm + (a + b - a * b * Float32(2)) * t, F4(0))
    elif mode == Blend.soft:
        var scr = F4(1) - (F4(1) - b) * (F4(1) - a)
        r = a * tm + ((F4(1) - a) * b * a + a * scr) * t
    elif mode == Blend.linear: r = a + (b * Float32(2) - F4(1)) * t
    elif mode == Blend.div or mode == Blend.overlay or mode == Blend.dodge or mode == Blend.burn:
        comptime for k in range(3):
            var x = a[k]; var y = b[k]
            if mode == Blend.div:
                if y != Float32(0): x = tm * x + t * x / y
            elif mode == Blend.overlay:
                x = x * (tm + Float32(2) * t * y) if x < Float32(0.5) else Float32(1) - (tm + Float32(2) * t * (Float32(1) - y)) * (Float32(1) - x)
            elif mode == Blend.dodge:
                if x != Float32(0):
                    var d = Float32(1) - t * y
                    x = Float32(1) if d <= Float32(0) else min(x / d, Float32(1))
            else:
                var d = tm + t * y
                x = Float32(0) if d <= Float32(0) else min(max(Float32(1) - (Float32(1) - x) / d, Float32(0)), Float32(1))
            r[k] = x
    else:
        var h2 = _rgb_to_hsv(b)
        var h1 = _rgb_to_hsv(a)
        if mode == Blend.hue:
            if h2[1] != Float32(0):
                h1[0] = h2[0]
                r = a * tm + _hsv_to_rgb(h1) * t
        elif mode == Blend.sat:
            if h1[1] != Float32(0):
                h1[1] = tm * h1[1] + t * h2[1]
                r = _hsv_to_rgb(h1)
        elif mode == Blend.val:
            h1[2] = tm * h1[2] + t * h2[2]
            r = _hsv_to_rgb(h1)
        elif mode == Blend.color:
            if h2[1] != Float32(0):
                h1[0] = h2[0]; h1[1] = h2[1]
                r = a * tm + _hsv_to_rgb(h1) * t
    r[3] = a[3]
    return r


@always_inline
def _smooth(x: Float32, order: Int) -> Float32:
    var f = min(max(x, Float32(0)), Float32(1))
    if order == MapRange.smoothstep:
        return (Float32(3) - Float32(2) * f) * f * f
    return f * f * f * (f * (f * Float32(6) - Float32(15)) + Float32(10))


def _map_range(mode: Int, v: Float32, f0: Float32, f1: Float32, t0: Float32, t1: Float32, steps: Float32, clamp: Bool) -> Float32:
    """Cycles' vector Map Range, one lane."""
    var f = _div(v - f0, f1 - f0)
    if mode == MapRange.stepped:
        f = floor(f * (steps + Float32(1))) / steps if steps > Float32(0) else Float32(0)
    elif mode == MapRange.smoothstep or mode == MapRange.smootherstep:
        f = _smooth(f, mode)
    var r = t0 + f * (t1 - t0)
    if clamp and (mode == MapRange.linear or mode == MapRange.stepped):
        r = min(max(r, min(t0, t1)), max(t0, t1))
    return r


def _mapping(mode: Int, v: F4, loc: F4, rot: F4, scale: F4) -> F4:
    """Cycles' svm_mapping with its euler_to_transform."""
    var cx = cos(rot[0]); var cy = cos(rot[1]); var cz = cos(rot[2])
    var sx = sin(rot[0]); var sy = sin(rot[1]); var sz = sin(rot[2])
    var r0 = F4(cy * cz, sy * sx * cz - cx * sz, sy * cx * cz + sx * sz, 0)
    var r1 = F4(cy * sz, sy * sx * sz + cx * cz, sy * cx * sz - sx * cz, 0)
    var r2 = F4(-sy, cy * sx, cy * cx, 0)
    if mode == Mapping.texture:
        var d = v - loc
        var t = r0 * d[0] + r1 * d[1] + r2 * d[2]
        return F4(_div(t[0], scale[0]), _div(t[1], scale[1]), _div(t[2], scale[2]), 0)
    var w = v * scale
    if mode == Mapping.normal:
        w = F4(_div(v[0], scale[0]), _div(v[1], scale[1]), _div(v[2], scale[2]), 0)
    var o = F4(_dot3(r0, w), _dot3(r1, w), _dot3(r2, w), 0)
    if mode == Mapping.point:
        return o + loc
    if mode == Mapping.normal:
        return _normalize3(o)
    return o


@always_inline
def _table(data: Pointer[Int32, MutUntrackedOrigin], n: Int, x: Float32, lane: Int, interpolate: Bool) -> Float32:
    var f = min(max(x, Float32(0)), Float32(1)) * Float32(n - 1)
    var i = min(Int(f), n - 1)
    var a = _f(data[unsafe_offset=4 * i + lane])
    var t = f - Float32(i)
    if interpolate and t > Float32(0) and i + 1 < n:
        a = a * (Float32(1) - t) + _f(data[unsafe_offset=4 * i + 4 + lane]) * t
    return a


def _image(textures: Pointer[GpuTexture, MutUntrackedOrigin], n_textures: Int, ti: Int, flags: Int32, uv: F4,
           fp_width: Float32) -> F4:
    if ti < 0 or ti >= n_textures:
        return F4(0)
    var tex = textures[unsafe_offset=ti]
    if Int(tex.width) <= 0 or Int(tex.format) == GpuTexture.FORMAT_PTEX or Int(tex.format) == GpuTexture.FORMAT_SHADER:
        return F4(0)
    var u = uv[0]; var v = uv[1]
    var ext = (flags >> 1) & 3
    if ext == ImageExt.clip and (u < Float32(0) or u > Float32(1) or v < Float32(0) or v > Float32(1)):
        return F4(0)
    if ext == ImageExt.extend:
        u = min(max(u, Float32(0)), Float32(0.99999)); v = min(max(v, Float32(0)), Float32(0.99999))
    elif ext == ImageExt.mirror:
        u = Float32(1) - abs(Float32(1) - u + Float32(2) * floor(u * Float32(0.5)))
        v = Float32(1) - abs(Float32(1) - v + Float32(2) * floor(v * Float32(0.5)))
    var c = _sample_tex(tex, u, Float32(1) - v, _footprint_lod(tex, fp_width))
    return F4(c.r, c.g, c.b, 1)


# The register file is one vector, read and written by unrolled selects, so it stays out of private memory.
comptime RegFile = SIMD[DType.float32, 4 * SHADER_REGS]


@always_inline
def _reg(regs: RegFile, i: Int32) -> F4:
    var out = F4(0)
    comptime for k in range(SHADER_REGS):
        if Int(i) == k:
            out = regs.slice[4, offset=4 * k]()
    return out


@always_inline
def _set_reg(mut regs: RegFile, i: Int, v: F4):
    comptime for k in range(SHADER_REGS):
        if i == k:
            regs = regs.insert[offset=4 * k](v)


@no_inline   # one copy per kernel; inlined at every texture lookup it would multiply the shading kernels' size
def eval_shader(code: Pointer[Int32, MutUntrackedOrigin], n_words: Int, u: Float32, v: Float32, fp_width: Float32,
                textures: Pointer[GpuTexture, MutUntrackedOrigin], n_textures: Int) -> F4:
    """Runs a program at texture coordinates (u, v); returns its output register."""
    var regs = RegFile(0)
    var pc = 1
    while pc + SHADER_STRIDE <= n_words:
        var op = code[unsafe_offset=pc]
        var dst = Int(code[unsafe_offset=pc + 1]) & (SHADER_REGS - 1)
        var a = _reg(regs, code[unsafe_offset=pc + 2])
        var b = _reg(regs, code[unsafe_offset=pc + 3])
        var c = _reg(regs, code[unsafe_offset=pc + 4])
        var d = _reg(regs, code[unsafe_offset=pc + 5])
        var i0 = code[unsafe_offset=pc + 6]
        var i1 = code[unsafe_offset=pc + 7]
        var r = F4(0)
        if op == ShaderOp.const:
            r = F4(_f(code[unsafe_offset=pc + 2]), _f(code[unsafe_offset=pc + 3]), _f(code[unsafe_offset=pc + 4]),
                   _f(code[unsafe_offset=pc + 5]))
        elif op == ShaderOp.uv:
            r = F4(u, v, 0, 1)
        elif op == ShaderOp.image:
            r = _image(textures, n_textures, Int(i0), i1, a, fp_width)
        elif op == ShaderOp.math:
            comptime for k in range(4):
                var x = _math(Int(i0), a[k], b[k], c[k])
                r[k] = min(max(x, Float32(0)), Float32(1)) if i1 != Int32(0) else x
        elif op == ShaderOp.vmath:
            r = _vmath(Int(i0), a, b, c, d[0])
        elif op == ShaderOp.mix:
            r = a + (b - a) * c
        elif op == ShaderOp.blend:
            r = _blend(Int(i0), a[0], b, c)
            if i1 != Int32(0):
                r = min(max(r, F4(0)), F4(1))
        elif op == ShaderOp.clamp:
            r = min(max(a, b), c)
        elif op == ShaderOp.map_range:
            var t1 = _reg(regs, i1 & 255)
            var st = _reg(regs, (i1 >> 8) & 255)
            comptime for k in range(4):
                r[k] = _map_range(Int(i0), a[k], b[k], c[k], d[k], t1[k], st[k], (i1 >> 16) & 1 != 0)
        elif op == ShaderOp.ramp:
            var data = code.unsafe_offset(pc + SHADER_STRIDE)
            comptime for k in range(4):
                r[k] = _table(data, Int(i0), a[0], k, i1 != Int32(0))
        elif op == ShaderOp.hsv:
            var fac = _reg(regs, i0)[0]
            var h = _rgb_to_hsv(a)
            h[0] = _fract(h[0] + b[0] + Float32(0.5))
            h[1] = min(max(h[1] * c[0], Float32(0)), Float32(1))
            h[2] *= d[0]
            r = max(_hsv_to_rgb(h) * fac + a * (Float32(1) - fac), F4(0))
            r[3] = a[3]
        elif op == ShaderOp.invert:
            r = (F4(1) - a) * b[0] + a * (Float32(1) - b[0])
            r[3] = a[3]
        elif op == ShaderOp.bright:
            r = max(a * (Float32(1) + c[0]) + F4(b[0] - c[0] * Float32(0.5)), F4(0))
            r[3] = a[3]
        elif op == ShaderOp.gamma:
            r = a
            if b[0] == Float32(0):
                r = F4(1, 1, 1, a[3])
            else:
                comptime for k in range(3):
                    if a[k] > Float32(0):
                        r[k] = exp(b[0] * log(a[k]))
        elif op == ShaderOp.split:
            r = F4(a[0] if i0 == 0 else (a[1] if i0 == 1 else (a[2] if i0 == 2 else a[3])))
        elif op == ShaderOp.combine:
            r = F4(a[0], b[0], c[0], 1)
        elif op == ShaderOp.hsv_rgb:
            r = _hsv_to_rgb(a) if i0 != Int32(0) else _rgb_to_hsv(a)
        elif op == ShaderOp.mapping:
            r = _mapping(Int(i0), a, b, c, d)
        elif op == ShaderOp.normal_map:
            var n = a * Float32(2) - F4(1)
            if b[0] != Float32(1):
                var z = F4(0, 0, 1, 0)
                n = _normalize3(z + (_normalize3(n) - z) * max(b[0], Float32(0)))
            r = n * Float32(0.5) + F4(0.5)
        elif op == ShaderOp.curves:
            var data = code.unsafe_offset(pc + SHADER_STRIDE)
            r = a
            comptime for k in range(3):
                r[k] = _table(data, Int(i0), a[k], k, True) * b[0] + a[k] * (Float32(1) - b[0])
        elif op == ShaderOp.luma:
            r = F4(a[0] * Float32(0.2126) + a[1] * Float32(0.7152) + a[2] * Float32(0.0722))
        elif op == ShaderOp.average:
            r = F4((a[0] + a[1] + a[2]) * Float32(1.0 / 3.0))
        _set_reg(regs, dst, r)
        pc += instruction_words(op, i0)
    return _reg(regs, code[unsafe_offset=0])


@always_inline
def sample_shader(tex: GpuTexture, u: Float32, v: Float32, fp_width: Float32,
                  textures: Pointer[GpuTexture, MutUntrackedOrigin], n_textures: Int) -> F4:
    """A shader texture at (u, v): Blender's convention, v up."""
    return eval_shader(tex.data.unsafe_bitcast[Int32](), Int(tex.width), u, v, fp_width, textures, n_textures)
