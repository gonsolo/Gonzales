# A repro for an IGC miscompile: RMIP's prism step (two triangles, three bilinear sides, the first and last crossing)
# on synthetic prisms, GPU against CPU.

from std.math import max, min, sqrt, sin, cos, ceildiv
from std.memory.alloc import unsafe_alloc
from std.os import getenv
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from gonzales.core.geometry import Vec3f, cross, dot
from gonzales.scene.shapes.displaced_patches import _tri_hit
from gonzales.scene.shapes.rmip import _bilinear_hits


@always_inline
def _h(i: Int, k: Int) -> Float32:
    var x = UInt32(i) * UInt32(2654435761) + UInt32(k) * UInt32(40503) + UInt32(12345)
    x ^= x >> 13; x *= UInt32(1274126177); x ^= x >> 16
    return Float32(Int(x & UInt32(0xFFFF))) / Float32(65535)


def _bilinear_hits_c(q00: Vec3f, q10: Vec3f, q11: Vec3f, q01: Vec3f, dir: Vec3f) -> SIMD[DType.float32, 2]:
    """_bilinear_hits with a compile-time root loop."""
    var res = SIMD[DType.float32, 2](-1)
    var e10 = q10 - q00; var e11 = q11 - q10; var e00 = q01 - q00
    var qn = cross(q10 - q00, q01 - q11)
    var a = dot(cross(q00, dir), e00); var c = dot(qn, dir); var b = dot(cross(q10, dir), e11) - a - c
    var det = b * b - Float32(4) * a * c
    if det < Float32(0):
        return res
    det = sqrt(det)
    var u1: Float32; var u2: Float32
    if c == Float32(0):
        u1 = -a / b if b != Float32(0) else Float32(-1); u2 = Float32(-1)
    else:
        u1 = (-b - (det if b >= Float32(0) else -det)) * Float32(0.5)
        u2 = a / u1 if u1 != Float32(0) else Float32(-1)
        u1 = u1 / c
    comptime e = Float32(1e-4)
    comptime for r in range(2):
        var ur = u1 if r == 0 else u2
        if ur >= -e and ur <= Float32(1) + e:
            var pa = q00 + e10 * ur; var pb = e00 + (e11 - e00) * ur
            var n = cross(dir, pb); var dd = dot(n, n)
            if dd > Float32(0):
                n = cross(n, pa)
                var vr = dot(n, dir)
                if vr >= -e * dd and vr <= (Float32(1) + e) * dd:
                    res[r] = dot(n, pb) / dd
    return res


def prism[V: Int](i: Int) -> SIMD[DType.float32, 4]:
    """Prism i: a jittered triangle near z = 5 offset by [-0.1, 0.1] along jittered normals; ray from the origin."""
    var a0 = Vec3f(_h(i, 0) - 0.5, _h(i, 1) - 0.5, 5); var a1 = Vec3f(_h(i, 2) + 0.5, _h(i, 3) - 0.5, 5)
    var a2 = Vec3f(_h(i, 4) - 0.5, _h(i, 5) + 0.5, 5)
    var n0 = Vec3f(_h(i, 6) * 0.2 - 0.1, _h(i, 7) * 0.2 - 0.1, -1); var n1 = Vec3f(_h(i, 8) * 0.2 - 0.1, _h(i, 9) * 0.2 - 0.1, -1)
    var n2 = Vec3f(_h(i, 10) * 0.2 - 0.1, _h(i, 11) * 0.2 - 0.1, -1)
    var dir = Vec3f(_h(i, 12) * 0.4 - 0.1, _h(i, 13) * 0.4 - 0.1, 1); dir = dir * (Float32(1) / sqrt(dot(dir, dir)))
    var lo_h = Float32(-0.1); var hi_h = Float32(0.1)
    var b0 = a0 + n0 * hi_h; var b1 = a1 + n1 * hi_h; var b2 = a2 + n2 * hi_h
    a0 = a0 + n0 * lo_h; a1 = a1 + n1 * lo_h; a2 = a2 + n2 * lo_h
    comptime if V == 5:   # a SIMD element written at a runtime index
        var o = SIMD[DType.float32, 4](-1)
        o[i & 3] = Float32(i & 1023)
        return o
    comptime if V == 7:   # a SIMD element read at a runtime index
        var o = SIMD[DType.float32, 4](Float32(i & 1023), Float32(i & 1023) + 0.25, Float32(i & 1023) + 0.5, Float32(i & 1023) + 0.75)
        return SIMD[DType.float32, 4](o[i & 3], 0, 0, 0)
    comptime if V == 8:   # a sixteen-wide SIMD element read at a runtime index
        var o = SIMD[DType.float32, 16](0)
        comptime for k in range(16):
            o[k] = _h(i, k)
        return SIMD[DType.float32, 4](o[(i >> 2) & 15], o[i & 15], 0, 0)
    comptime if V == 6:   # one side, with the root loop unrolled
        var y0 = _bilinear_hits_c(a0, a1, b1, b0, dir)
        return SIMD[DType.float32, 4](y0[0], y0[1], 0, 0)
    comptime if V == 1:   # the triangles only
        var x0 = _tri_hit(a0, a1, a2, dir, Float32(1e30)); var x1 = _tri_hit(b0, b1, b2, dir, Float32(1e30))
        return SIMD[DType.float32, 4](x0[1], x1[1], Float32(Int(x0[0]) + Int(x1[0])), 0)
    comptime if V == 2:   # the sides only
        var y0 = _bilinear_hits(a0, a1, b1, b0, dir); var y1 = _bilinear_hits(a1, a2, b2, b1, dir); var y2 = _bilinear_hits(a2, a0, b0, b2, dir)
        return SIMD[DType.float32, 4](y0[0], y1[0], y2[0], y0[1] + y1[1] + y2[1])
    comptime if V == 3:   # one side
        var y0 = _bilinear_hits(a0, a1, b1, b0, dir)
        return SIMD[DType.float32, 4](y0[0], y0[1], 0, 0)
    var q0 = _tri_hit(a0, a1, a2, dir, Float32(1e30)); var q1 = _tri_hit(b0, b1, b2, dir, Float32(1e30))
    var e0 = _bilinear_hits(a0, a1, b1, b0, dir); var e1 = _bilinear_hits(a1, a2, b2, b1, dir); var e2 = _bilinear_hits(a2, a0, b0, b2, dir)
    comptime if V == 4:   # all, no min/max chain
        return SIMD[DType.float32, 4](q0[1] + q1[1], e0[0] + e1[0] + e2[0], e0[1] + e1[1] + e2[1], Float32(Int(q0[0]) + Int(q1[0])))
    var lo = Float32(1e30); var hi = Float32(-1e30); var nt = 0
    if q0[0]:
        lo = min(lo, q0[1]); hi = max(hi, q0[1]); nt += 1
    if q1[0]:
        lo = min(lo, q1[1]); hi = max(hi, q1[1]); nt += 1
    comptime for g in range(2):
        if e0[g] > Float32(0):
            lo = min(lo, e0[g]); hi = max(hi, e0[g]); nt += 1
        if e1[g] > Float32(0):
            lo = min(lo, e1[g]); hi = max(hi, e1[g]); nt += 1
        if e2[g] > Float32(0):
            lo = min(lo, e2[g]); hi = max(hi, e2[g]); nt += 1
    return SIMD[DType.float32, 4](lo, hi, Float32(nt), 0)


def kernel[V: Int](dst: Pointer[Float32, MutUntrackedOrigin], n: Int32):
    var i = global_idx.x
    if i >= Int(n):
        return
    var r = prism[V](i)
    dst[unsafe_offset=4 * i] = r[0]; dst[unsafe_offset=4 * i + 1] = r[1]; dst[unsafe_offset=4 * i + 2] = r[2]
    dst[unsafe_offset=4 * i + 3] = r[3]


def main() raises:
    var n = 1 << 20
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](4 * n)
    var p = buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var v = atol(getenv("V", "0"))
    if v == 1:
        ctx.enqueue_function[kernel[1]](p, Int32(n), grid_dim=ceildiv(n, 64), block_dim=64)
    elif v == 2:
        ctx.enqueue_function[kernel[2]](p, Int32(n), grid_dim=ceildiv(n, 64), block_dim=64)
    elif v == 3:
        ctx.enqueue_function[kernel[3]](p, Int32(n), grid_dim=ceildiv(n, 64), block_dim=64)
    elif v == 5:
        ctx.enqueue_function[kernel[5]](p, Int32(n), grid_dim=ceildiv(n, 64), block_dim=64)
    elif v == 6:
        ctx.enqueue_function[kernel[6]](p, Int32(n), grid_dim=ceildiv(n, 64), block_dim=64)
    elif v == 7:
        ctx.enqueue_function[kernel[7]](p, Int32(n), grid_dim=ceildiv(n, 64), block_dim=64)
    elif v == 8:
        ctx.enqueue_function[kernel[8]](p, Int32(n), grid_dim=ceildiv(n, 64), block_dim=64)
    elif v == 4:
        ctx.enqueue_function[kernel[4]](p, Int32(n), grid_dim=ceildiv(n, 64), block_dim=64)
    else:
        ctx.enqueue_function[kernel[0]](p, Int32(n), grid_dim=ceildiv(n, 64), block_dim=64)
    ctx.synchronize()
    var bad = 0; var hits = 0
    with buf.map_to_host() as h:
        for i in range(n):
            var c: SIMD[DType.float32, 4]
            if v == 1:
                c = prism[1](i)
            elif v == 2:
                c = prism[2](i)
            elif v == 3:
                c = prism[3](i)
            elif v == 5:
                c = prism[5](i)
            elif v == 6:
                c = prism[6](i)
            elif v == 7:
                c = prism[7](i)
            elif v == 8:
                c = prism[8](i)
            elif v == 4:
                c = prism[4](i)
            else:
                c = prism[0](i)
            if c[2] > Float32(0):
                hits += 1
            var diff = Float32(0)
            for k in range(4):
                diff = max(diff, abs(h[4 * i + k] - c[k]) / max(abs(c[k]), Float32(1)))
            if not (diff <= Float32(1e-3)):
                if bad < 5:
                    print("mismatch", i, ": gpu", h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3], "cpu", c[0], c[1], c[2], c[3])
                bad += 1
    print("prisms", n, "with crossings", hits, "mismatches", bad)
