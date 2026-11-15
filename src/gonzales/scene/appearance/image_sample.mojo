"""Bilinear and trilinear lookups in one image of the texture table."""
from std.math import abs, floor, log2, acos, sqrt, pi
from gonzales.core.geometry import RGB, Vec3f, _atan2f
from gonzales.scene.appearance.materials import GpuTexture


@always_inline
def _srgb_to_linear(c: Float32) -> Float32:
    if c <= Float32(0.04045):
        return c / Float32(12.92)
    else:
        return Float32(((c + Float32(0.055)) / Float32(1.055)) ** Float32(2.4))


# Linear RGB of texel `i`, counted in texels from the start of tex.data. uint8
# textures decode each channel through tex.lut, so every blend happens in
# linear space; single-channel textures replicate into all three channels.
@always_inline
def _texel(tex: GpuTexture, i: Int) -> RGB:
    if Int(tex.format) == GpuTexture.FORMAT_U8:
        if Int(tex.channels) == 1:
            var l = tex.lut[unsafe_offset=Int(tex.data[unsafe_offset=i])]
            return RGB(l, l, l)
        var j = i * 3
        return RGB(tex.lut[unsafe_offset=Int(tex.data[unsafe_offset=j])], tex.lut[unsafe_offset=Int(tex.data[unsafe_offset=j + 1])], tex.lut[unsafe_offset=Int(tex.data[unsafe_offset=j + 2])])
    var f = tex.data.unsafe_bitcast[Float32]()
    var k = i * 3
    return RGB(f[unsafe_offset=k], f[unsafe_offset=k + 1], f[unsafe_offset=k + 2])


# Bilinear sample of ONE mip level: `off` = texel offset of the level in
# tex.data, (lw, lh) = that level's dimensions. Pixel centres at +0.5, wrap.
@always_inline
def _sample_level(tex: GpuTexture, off: Int, lw: Int, lh: Int, u: Float32, v: Float32) -> RGB:
    var s = u - Float32(Int(u))
    if s < Float32(0.0): s += Float32(1.0)
    var t = v - Float32(Int(v))
    if t < Float32(0.0): t += Float32(1.0)
    var fx = s * Float32(lw) - Float32(0.5)
    var fy = t * Float32(lh) - Float32(0.5)
    var x0 = Int(floor(fx)); var y0 = Int(floor(fy))
    var wx = fx - Float32(x0); var wy = fy - Float32(y0)
    # Wrap without integer division: x0 is in [-1, lw-1] here; % only for out-of-range inputs.
    var x0w = x0
    if x0w < 0 or x0w >= lw: x0w = ((x0 % lw) + lw) % lw
    var y0w = y0
    if y0w < 0 or y0w >= lh: y0w = ((y0 % lh) + lh) % lh
    var x1w = x0w + 1
    if x1w == lw: x1w = 0
    var y1w = y0w + 1
    if y1w == lh: y1w = 0
    var i00 = off + y0w * lw + x0w
    var i10 = off + y0w * lw + x1w
    var i01 = off + y1w * lw + x0w
    var i11 = off + y1w * lw + x1w
    var w00 = (Float32(1.0) - wx) * (Float32(1.0) - wy)
    var w10 = wx * (Float32(1.0) - wy)
    var w01 = (Float32(1.0) - wx) * wy
    var w11 = wx * wy
    return _texel(tex, i00) * w00 + _texel(tex, i10) * w10 + _texel(tex, i01) * w01 + _texel(tex, i11) * w11


# Trilinear mip sample. lod 0 = base level (full res); higher = coarser.
# With a 1-level texture (no pyramid) this is plain bilinear on the base.
@always_inline
def _sample_tex(tex: GpuTexture, u: Float32, v: Float32, lod: Float32 = Float32(0.0)) -> RGB:
    var nl = Int(tex.n_levels)
    if nl <= 1:
        return _sample_level(tex, 0, Int(tex.width), Int(tex.height), u, v)
    var clamped = lod
    if clamped < Float32(0.0): clamped = Float32(0.0)
    var maxl = Float32(nl - 1)
    if clamped > maxl: clamped = maxl
    var l0 = Int(floor(clamped))
    var f = clamped - Float32(l0)
    # Walk to level l0, tracking its texel offset and dims.
    var off = 0; var w = Int(tex.width); var h = Int(tex.height)
    for _k in range(l0):
        off += w * h
        w = max(1, w // 2); h = max(1, h // 2)
    var c0 = _sample_level(tex, off, w, h, u, v)
    if f <= Float32(0.0) or l0 >= nl - 1:
        return c0
    var off1 = off + w * h
    var w1 = max(1, w // 2); var h1 = max(1, h // 2)
    var c1 = _sample_level(tex, off1, w1, h1, u, v)
    return c0 + (c1 - c0) * f


# pbrt's MIPMap level for its default "bilinear" filter: floor(nLevels - 1 +
# log2(width)), one level, no blend. `width` is UVFootprint.width (uv units).
@always_inline
def _footprint_lod(tex: GpuTexture, width: Float32) -> Float32:
    var texels = width * Float32(max(Int(tex.width), Int(tex.height)))
    if texels > Float32(1.0):
        return floor(log2(texels))
    return Float32(0.0)


@always_inline
def _map_row(m: SIMD[DType.float32, 16], i: Int, p: Vec3f) -> Float32:
    return m[4 * i] * p.x + m[4 * i + 1] * p.y + m[4 * i + 2] * p.z + m[4 * i + 3]


# pbrt's TextureMapping2D (GpuTexture.map) at mesh uv (u, v) or world point p, then its V-flip.
# The uv footprint filters only the uv mapping; the others sample level 0.
@always_inline
def sample_image(tex: GpuTexture, u: Float32, v: Float32, fp_width: Float32, p: Vec3f = Vec3f(0, 0, 0)) -> RGB:
    var m = tex.map
    var kind = Int(m[15])
    var s: Float32
    var t: Float32
    var w = Float32(0)
    if kind == GpuTexture.MAP_UV:
        s = m[0] * u + m[2]; t = m[1] * v + m[3]
        w = fp_width * max(abs(m[0]), abs(m[1]))
    elif kind == GpuTexture.MAP_PLANAR:
        s = _map_row(m, 0, p); t = _map_row(m, 1, p)
    else:
        var x = _map_row(m, 0, p); var y = _map_row(m, 1, p); var z = _map_row(m, 2, p)
        var phi = _atan2f(y, x)
        if kind == GpuTexture.MAP_CYLINDRICAL:
            s = (Float32(pi) + phi) * Float32(0.5 / pi); t = z
        else:
            var r = sqrt(x * x + y * y + z * z)
            s = acos(min(max(z / max(r, Float32(1e-20)), Float32(-1)), Float32(1))) * Float32(1 / pi)
            t = (phi + Float32(2 * pi) if phi < Float32(0) else phi) * Float32(0.5 / pi)
    return _sample_tex(tex, s, Float32(1) - t, _footprint_lod(tex, w))
