# RMIP (Thonat, Georgiev, Beaune and Boubekeur, SIGGRAPH Asia 2023) on our sub-squares, for comparison: each sub-square two
# triangles with linearly interpolated limit corners and normals, displaced by the same Ptex heights.

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import max, min, sqrt, abs, floor, ceil
from std.memory.alloc import unsafe_alloc
from std.memory import stack_allocation
from std.sys.info import is_gpu
from max.gpu import thread_idx
from max.gpu.memory import AddressSpace
from gonzales.core.geometry import Vec3f, cross, dot
from gonzales.scene.shapes.displaced_patches import DispSurface, BLFrame, prim_patch, sub_square, _bl_frame, _tri_hit

comptime _STACK = 17
comptime _TPB = 64   # work-items per group: the traversal stacks live in shared local memory


@fieldwise_init
struct Rmip(TrivialRegisterPassable, DevicePassable):
    """Per face an RMIP over its bordered finest heights, downscaled so that R (1 + log2 R) <= N per axis (the paper's
    linear-memory variant): layer (p, q) holds the (min, max) bytes of every 2^p x 2^q block rectangle."""
    comptime device_type: AnyType = Self

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "Rmip"

    var off: Pointer[UInt32, MutUntrackedOrigin]   # per face: its first (min, max) pair, and log2 Ru + 256 log2 Rv
    var mm: Pointer[UInt8, MutUntrackedOrigin]


def _log_res(n: Int) -> Int:
    """log2 R for N = 2^n texels: the largest R with R (1 + log2 R) <= N."""
    var r = n
    while r > 0 and (1 << r) * (1 + r) > (1 << n):
        r -= 1
    return r


def rmip_build(d: DispSurface, nf: Int) -> Tuple[Rmip, Int]:
    """The RMIP of every face from its bordered height chain (bilinear support: each block widened by a texel); and its bytes."""
    var off = unsafe_alloc[UInt32](2 * nf)
    var total = 0
    for f in range(nf):
        var lu = _log_res(Int(d.hdr[unsafe_offset=3 * f + 1])); var lv = _log_res(Int(d.hdr[unsafe_offset=3 * f + 2]))
        off[unsafe_offset=2 * f] = UInt32(total); off[unsafe_offset=2 * f + 1] = UInt32(lu + 256 * lv)
        total += (1 << lu) * (1 << lv) * (1 + lu) * (1 + lv)
    var mm = unsafe_alloc[UInt8](2 * total)
    for f in range(nf):
        var w = 1 << Int(d.hdr[unsafe_offset=3 * f + 1]); var h = 1 << Int(d.hdr[unsafe_offset=3 * f + 2])
        var lu = Int(off[unsafe_offset=2 * f + 1] & 255); var lv = Int(off[unsafe_offset=2 * f + 1] >> 8); var ru = 1 << lu; var rv = 1 << lv
        var bu = w // ru; var bv = h // rv
        var co = Int(d.chain_off[unsafe_offset=f]); var base = 2 * Int(off[unsafe_offset=2 * f])
        for by in range(rv):
            for bx in range(ru):
                var lo = 255; var hi = 0
                for y in range(max(by * bv - 1, -1), min((by + 1) * bv, h) + 1):
                    for x in range(max(bx * bu - 1, -1), min((bx + 1) * bu, w) + 1):
                        var t = Int(d.chain[unsafe_offset=co + (y + 1) * (w + 2) + x + 1])
                        lo = min(lo, t); hi = max(hi, t)
                mm[unsafe_offset=base + 2 * (by * ru + bx)] = UInt8(lo); mm[unsafe_offset=base + 2 * (by * ru + bx) + 1] = UInt8(hi)
        for q in range(lv + 1):
            for p in range(lu):   # merge along x
                var src = base + 2 * ru * rv * (p * (lv + 1) + q); var dst = base + 2 * ru * rv * ((p + 1) * (lv + 1) + q)
                for y in range(rv - (1 << q) + 1):
                    for x in range(ru - (1 << (p + 1)) + 1):
                        var a = src + 2 * (y * ru + x); var b = src + 2 * (y * ru + x + (1 << p)); var o = dst + 2 * (y * ru + x)
                        mm[unsafe_offset=o] = min(mm[unsafe_offset=a], mm[unsafe_offset=b]); mm[unsafe_offset=o + 1] = max(mm[unsafe_offset=a + 1], mm[unsafe_offset=b + 1])
            if q < lv:   # merge along y
                var src = base + 2 * ru * rv * q; var dst = base + 2 * ru * rv * (q + 1)
                for y in range(rv - (1 << (q + 1)) + 1):
                    for x in range(ru):
                        var a = src + 2 * (y * ru + x); var b = src + 2 * ((y + (1 << q)) * ru + x); var o = dst + 2 * (y * ru + x)
                        mm[unsafe_offset=o] = min(mm[unsafe_offset=a], mm[unsafe_offset=b]); mm[unsafe_offset=o + 1] = max(mm[unsafe_offset=a + 1], mm[unsafe_offset=b + 1])
    return (Rmip(off, mm), 2 * total + 8 * nf)


@always_inline
def _ilog2(n: Int) -> Int:
    var l = 0
    while (2 << l) <= n:
        l += 1
    return l


@always_inline
def rmip_query(r: Rmip, f: Int, u0: Float32, u1: Float32, v0: Float32, v1: Float32) -> Tuple[Float32, Float32]:
    """(min, max) height bytes over face f's [u0, u1] x [v0, v1]: four 2^p x 2^q lookups (Eq. 5)."""
    var lw = Int(r.off[unsafe_offset=2 * f + 1]); var lu = lw & 255; var lv = lw >> 8; var ru = 1 << lu; var rv = 1 << lv
    var x0 = min(max(Int(floor(u0 * Float32(ru))), 0), ru - 1); var x1 = min(max(Int(ceil(u1 * Float32(ru))), x0 + 1), ru)
    var y0 = min(max(Int(floor(v0 * Float32(rv))), 0), rv - 1); var y1 = min(max(Int(ceil(v1 * Float32(rv))), y0 + 1), rv)
    var p = _ilog2(x1 - x0); var q = _ilog2(y1 - y0)
    var mx = x1 - (1 << p); var my = y1 - (1 << q)
    var base = 2 * (Int(r.off[unsafe_offset=2 * f]) + ru * rv * (p * (lv + 1) + q))
    var lo = 255; var hi = 0
    for c in range(4):
        var x = x0 if c == 0 or c == 3 else mx; var y = y0 if c < 2 else my
        var o = base + 2 * (y * ru + x)
        lo = min(lo, Int(r.mm[unsafe_offset=o])); hi = max(hi, Int(r.mm[unsafe_offset=o + 1]))
    return (Float32(lo), Float32(hi))


@fieldwise_init
struct _Tri(TrivialRegisterPassable):
    """A base triangle over local (a, b) in [0, 1]^2 of its sub-square, anchored at the ray origin: P = A + Pa a + Pb b and
    N = Nc + Na a + Nb b; the face square (u0, v0, side) and the height scale."""
    var A: Vec3f
    var Pa: Vec3f
    var Pb: Vec3f
    var Nc: Vec3f
    var Na: Vec3f
    var Nb: Vec3f
    var u0: Float32
    var v0: Float32
    var side: Float32
    var upper: Bool   # b >= a (corners 00 11 01), else b <= a (00 10 11)


@always_inline
def _corner(fr: BLFrame, c: Int) -> Tuple[Vec3f, Vec3f]:
    return (Vec3f(fr.px[c], fr.py[c], fr.pz[c]), Vec3f(fr.nx[c], fr.ny[c], fr.nz[c]))


@always_inline
def _tri(d: DispSurface, id: Int, org: Vec3f) -> Tuple[_Tri, Int]:
    """Triangle id (sub-square id >> 1, half id & 1) and its face."""
    var pk = prim_patch(d, id >> 1); var pi = pk[0]; var ss = sub_square(pk[2], pk[1])
    var fr = _bl_frame(d, pi, pk[1], org)
    var c0 = _corner(fr, 0); var c1 = _corner(fr, 1); var c2 = _corner(fr, 2); var c3 = _corner(fr, 3)
    var psz = d.sq[unsafe_offset=3 * pi + 2]
    var u0 = d.sq[unsafe_offset=3 * pi] + ss[0] * psz; var v0 = d.sq[unsafe_offset=3 * pi + 1] + ss[1] * psz
    var up = (id & 1) == 1
    var t: _Tri
    if up:   # 00 11 01: P = P00 + (P11 - P01) a + (P01 - P00) b
        t = _Tri(c0[0], c2[0] - c3[0], c3[0] - c0[0], c0[1], c2[1] - c3[1], c3[1] - c0[1], u0, v0, ss[2] * psz, True)
    else:    # 00 10 11: P = P00 + (P10 - P00) a + (P11 - P10) b
        t = _Tri(c0[0], c1[0] - c0[0], c2[0] - c1[0], c0[1], c1[1] - c0[1], c2[1] - c1[1], u0, v0, ss[2] * psz, False)
    return (t, Int(d.face[unsafe_offset=pi]))


@always_inline
def _P(t: _Tri, a: Float32, b: Float32) -> Vec3f:
    return t.A + t.Pa * a + t.Pb * b


@always_inline
def _N(t: _Tri, a: Float32, b: Float32) -> Vec3f:
    return t.Nc + t.Na * a + t.Nb * b


@always_inline
def _inside(t: _Tri, a: Float32, b: Float32, e: Float32) -> Bool:
    if a < -e or b < -e or a > Float32(1) + e or b > Float32(1) + e:
        return False
    return b >= a - e if t.upper else b <= a + e


@always_inline
def _height_range(d: DispSurface, r: Rmip, t: _Tri, f: Int, a0: Float32, a1: Float32, b0: Float32, b1: Float32) -> Tuple[Float32, Float32]:
    var q = rmip_query(r, f, t.u0 + a0 * t.side, t.u0 + a1 * t.side, t.v0 + b0 * t.side, t.v0 + b1 * t.side)
    return (q[0] * d.scale, q[1] * d.scale)


@always_inline
def _prism_range(d: DispSurface, r: Rmip, t: _Tri, f: Int) -> Tuple[Float32, Float32]:
    """The prism's height range over the sub-square, padded: a flat face's prism keeps a thickness."""
    var hr = _height_range(d, r, t, f, Float32(0), Float32(1), Float32(0), Float32(1))
    var pad = Float32(1e-3) * (sqrt(dot(t.Pa, t.Pa)) + sqrt(dot(t.Pb, t.Pb)))
    return (hr[0] - pad, hr[1] + pad)


def rmip_prism_box(d: DispSurface, r: Rmip, id: Int) -> Tuple[Vec3f, Vec3f]:
    """Triangle id's bounding prism (its corners offset by the height range over the sub-square), boxed."""
    var tf = _tri(d, id, Vec3f(0, 0, 0)); var t = tf[0]
    var hr = _prism_range(d, r, t, tf[1])
    var lo = Vec3f(1e30, 1e30, 1e30); var hi = Vec3f(-1e30, -1e30, -1e30)
    for c in range(3):
        var a = Float32(0) if c == 0 or (c == 2 and t.upper) else Float32(1)
        var b = Float32(0) if c == 0 or (c == 1 and not t.upper) else Float32(1)
        for s in range(2):
            var x = _P(t, a, b) + _N(t, a, b) * (hr[0] if s == 0 else hr[1])
            lo = Vec3f(min(lo.x, x.x), min(lo.y, x.y), min(lo.z, x.z)); hi = Vec3f(max(hi.x, x.x), max(hi.y, x.y), max(hi.z, x.z))
    return (lo, hi)


@always_inline
def _det(x: Vec3f, y: Vec3f, z: Vec3f) -> Float32:
    return dot(x, cross(y, z))


@always_inline
def _psi(k: SIMD[DType.float32, 8], a: Float32, b: Float32) -> Float32:
    """The projected ray's implicit form det(P - O, N, D) (Eq. 3), a quadric in (a, b)."""
    return k[0] + k[1] * a + k[2] * b + k[3] * a * a + k[4] * a * b + k[5] * b * b


@always_inline
def _ray_t(t: _Tri, dir: Vec3f, a: Float32, b: Float32) -> Float32:
    """The ray parameter where it passes the displacement line through (a, b): least squares of t D - s N = P."""
    var p = _P(t, a, b); var n = _N(t, a, b)
    var dn = dot(dir, n); var nn = dot(n, n); var det = dn * dn - nn
    if det == Float32(0):
        return dot(dir, p)
    return (dot(dir, p) * (-nn) + dn * dot(n, p)) / det


@always_inline
def _invert(t: _Tri, x: Vec3f, a0: Float32, b0: Float32) -> SIMD[DType.float32, 2]:
    """(a, b) whose displacement line passes x: Gauss-Newton on (P - x) x N = 0 (supplement Eq. 3, 4)."""
    var a = a0; var b = b0
    for _ in range(10):
        var p = _P(t, a, b) - x; var n = _N(t, a, b)
        var fv = cross(p, n)
        var ja = cross(t.Pa, n) + cross(p, t.Na); var jb = cross(t.Pb, n) + cross(p, t.Nb)
        var m00 = dot(ja, ja); var m01 = dot(ja, jb); var m11 = dot(jb, jb)
        var det = m00 * m11 - m01 * m01
        if not (abs(det) > Float32(0)):
            break
        var ga = dot(ja, fv); var gb = dot(jb, fv)
        var da = -(m11 * ga - m01 * gb) / det; var db = -(m00 * gb - m01 * ga) / det
        a += da; b += db
        if da * da + db * db < Float32(1e-10):
            break
    return SIMD[DType.float32, 2](a, b)


@always_inline
def _line_roots(k: SIMD[DType.float32, 8], al: Float32, be: Float32, ga: Float32) -> SIMD[DType.float32, 4]:
    """Points (a, b) (two, NaN where none) where the line al a + be b + ga = 0 meets psi = 0."""
    var nan = Float32(0) / Float32(0)
    var ll = al * al + be * be
    if not (ll > Float32(0)):
        return SIMD[DType.float32, 4](nan)
    var pa = -ga * al / ll; var pb = -ga * be / ll; var da = -be; var db = al
    var f0 = _psi(k, pa, pb); var fp = _psi(k, pa + da, pb + db); var fm = _psi(k, pa - da, pb - db)
    var c2 = Float32(0.5) * (fp + fm) - f0; var c1 = Float32(0.5) * (fp - fm)
    var r = SIMD[DType.float32, 4](nan)
    if abs(c2) < Float32(1e-12) * (abs(c1) + abs(f0)):
        if c1 != Float32(0):
            var s = -f0 / c1; r[0] = pa + da * s; r[1] = pb + db * s
        return r
    var disc = c1 * c1 - Float32(4) * c2 * f0
    if disc < Float32(0):
        return r
    var sq = sqrt(disc); var q = Float32(-0.5) * (c1 + (sq if c1 >= Float32(0) else -sq))
    var s1 = q / c2; var s2 = f0 / q if q != Float32(0) else s1
    r[0] = pa + da * s1; r[1] = pb + db * s1; r[2] = pa + da * s2; r[3] = pb + db * s2
    return r


@always_inline
def _bilinear_hits(q00: Vec3f, q10: Vec3f, q11: Vec3f, q01: Vec3f, dir: Vec3f) -> SIMD[DType.float32, 2]:
    """Both ray parameters (from the origin, -1 where none) where the ray crosses bilinear patch q00 q10 q11 q01 (Reshetov)."""
    var out = SIMD[DType.float32, 2](-1)
    var e10 = q10 - q00; var e11 = q11 - q10; var e00 = q01 - q00
    var qn = cross(q10 - q00, q01 - q11)
    var a = dot(cross(q00, dir), e00); var c = dot(qn, dir); var b = dot(cross(q10, dir), e11) - a - c
    var det = b * b - Float32(4) * a * c
    if det < Float32(0):
        return out
    det = sqrt(det)
    var u1: Float32; var u2: Float32
    if c == Float32(0):
        u1 = -a / b if b != Float32(0) else Float32(-1); u2 = Float32(-1)
    else:
        u1 = (-b - (det if b >= Float32(0) else -det)) * Float32(0.5)
        u2 = a / u1 if u1 != Float32(0) else Float32(-1)
        u1 = u1 / c
    comptime e = Float32(1e-3)   # the prism's faces overlap a hair, as the leaves'
    comptime for r in range(2):
        var ur = u1 if r == 0 else u2
        if ur < -e or ur > Float32(1) + e:
            continue
        var pa = q00 + e10 * ur; var pb = e00 + (e11 - e00) * ur
        var n = cross(dir, pb); var dd = dot(n, n)
        if dd <= Float32(0):
            continue
        n = cross(n, pa)
        var vr = dot(n, dir)
        if vr >= -e * dd and vr <= (Float32(1) + e) * dd:
            out[r] = dot(n, pb) / dd
    return out


@always_inline
def _box_t(lo: Vec3f, hi: Vec3f, inv: Vec3f) -> SIMD[DType.float32, 2]:
    var ax = lo.x * inv.x; var bx = hi.x * inv.x; var ay = lo.y * inv.y; var by = hi.y * inv.y; var az = lo.z * inv.z; var bz = hi.z * inv.z
    return SIMD[DType.float32, 2](max(max(min(ax, bx), min(ay, by)), min(az, bz)), min(min(max(ax, bx), max(ay, by)), max(az, bz)))


@always_inline
def _march(d: DispSurface, t: _Tri, f: Int, k: SIMD[DType.float32, 8], dir: Vec3f, a0: Float32, b0: Float32, a1: Float32, b1: Float32,
           t_best: Float32, mut steps: Int) -> SIMD[DType.float32, 4]:
    """Texel marching from (a0, b0) to (a1, b1) along the monotone projected ray: each cell between texel centres two
    triangles through its displaced corners; the exit edge from psi's signs (Sec. 4.4). (t, a, b, hit)."""
    var w = 1 << Int(d.hdr[unsafe_offset=3 * f + 1]); var h = 1 << Int(d.hdr[unsafe_offset=3 * f + 2])
    var co = Int(d.chain_off[unsafe_offset=f])
    var sw = t.side * Float32(w); var sh = t.side * Float32(h)
    var ox = t.u0 * Float32(w) - Float32(0.5); var oy = t.v0 * Float32(h) - Float32(0.5)   # texel-centre coordinates of a = b = 0
    var ca0 = min(max(a0, Float32(0)), Float32(1)); var cb0 = min(max(b0, Float32(0)), Float32(1))   # inside the sub-square:
    var ca1 = min(max(a1, Float32(0)), Float32(1)); var cb1 = min(max(b1, Float32(0)), Float32(1))   # a bounded march
    var x0 = ox + ca0 * sw; var y0 = oy + cb0 * sh; var x1 = ox + ca1 * sw; var y1 = oy + cb1 * sh
    var si = 1 if x1 >= x0 else -1; var sj = 1 if y1 >= y0 else -1
    # Ends on a cell edge: start and stop a thousandth of a texel inside, so rounding cannot start on the far side.
    var i = Int(floor(x0 + Float32(si) * Float32(1e-3))); var j = Int(floor(y0 + Float32(sj) * Float32(1e-3)))
    var i1 = Int(floor(x1 - Float32(si) * Float32(1e-3))); var j1 = Int(floor(y1 - Float32(sj) * Float32(1e-3)))
    if (i1 - i) * si < 0:
        i1 = i
    if (j1 - j) * sj < 0:
        j1 = j
    var res = SIMD[DType.float32, 4](t_best, 0, 0, 0)
    for _ in range(abs(i1 - i) + abs(j1 - j) + 1):
        steps += 1
        if i >= -1 and i < w and j >= -1 and j < h:
            var ca = SIMD[DType.float32, 4](0); var cb = SIMD[DType.float32, 4](0); var hv = SIMD[DType.float32, 4](0)
            comptime for c in range(4):
                var x = i + (1 if c == 1 or c == 2 else 0); var y = j + (1 if c >= 2 else 0)
                ca[c] = (Float32(x) - ox) / sw; cb[c] = (Float32(y) - oy) / sh
                hv[c] = Float32(Int(d.chain[unsafe_offset=co + (min(y, h) + 1) * (w + 2) + min(x, w) + 1])) * d.scale
            var c0 = _P(t, ca[0], cb[0]) + _N(t, ca[0], cb[0]) * hv[0]; var c1 = _P(t, ca[1], cb[1]) + _N(t, ca[1], cb[1]) * hv[1]
            var c2 = _P(t, ca[2], cb[2]) + _N(t, ca[2], cb[2]) * hv[2]; var c3 = _P(t, ca[3], cb[3]) + _N(t, ca[3], cb[3]) * hv[3]
            comptime for g in range(2):
                var q = _tri_hit(c0, c1 if g == 0 else c2, c2 if g == 0 else c3, dir, res[0])
                if q[0]:
                    var a = ca[0] + (ca[1 + g] - ca[0]) * q[2] + (ca[2 + g] - ca[0]) * q[3]
                    var b = cb[0] + (cb[1 + g] - cb[0]) * q[2] + (cb[2 + g] - cb[0]) * q[3]
                    if _inside(t, a, b, Float32(1e-3)):
                        res = SIMD[DType.float32, 4](q[1], a, b, 1)
            if res[3] > Float32(0):
                return res
        if i == i1 and j == j1:
            break
        if i == i1:
            j += sj
        elif j == j1:
            i += si
        else:   # leave through the i edge if psi changes sign along it
            var xe = i + (1 if si > 0 else 0)
            var ae = (Float32(xe) - ox) / sw
            var p0 = _psi(k, ae, (Float32(j) - oy) / sh); var p1 = _psi(k, ae, (Float32(j + 1) - oy) / sh)
            if (p0 <= Float32(0)) != (p1 <= Float32(0)):
                i += si
            else:
                j += sj
    return res


def intersect_rmip(d: DispSurface, r: Rmip, id: Int, org: Vec3f, dir: Vec3f, t_max: Float32, march: Float32) -> Tuple[Bool, Float32, Float32, Float32, Int]:
    """Algorithm 1 of the paper for triangle id: (hit, t, s, t in the patch, steps). march: the rectangle size in texels
    below which the traversal marches texels."""
    var tf = _tri(d, id, org); var t = tf[0]; var f = tf[1]
    var steps = 0
    var hr = _prism_range(d, r, t, f)
    # Work from a point on the ray just short of the prism: small coordinates keep float32 precise far from the camera.
    var rad = Float32(2) * (sqrt(dot(t.Pa, t.Pa)) + sqrt(dot(t.Pb, t.Pb))) + Float32(2) * max(abs(hr[0]), abs(hr[1])) * (
        sqrt(dot(t.Nc, t.Nc)) + sqrt(dot(t.Na, t.Na)) + sqrt(dot(t.Nb, t.Nb)))
    var tp = max(dot(t.A, dir) - rad, Float32(0))
    t.A = t.A - dir * tp
    var t_lim = t_max - tp
    # The prism: offset triangles and three bilinear sides.
    var bu = Float32(0) if t.upper else Float32(1)   # the second corner: (1, 1) upper, (1, 0) lower; the third (0, 1) or (1, 1)
    var cu = Float32(1) if t.upper else Float32(0)
    var p0 = _P(t, 0, 0); var n0 = _N(t, 0, 0); var p1 = _P(t, 1, 1 - bu); var n1 = _N(t, 1, 1 - bu)
    var p2 = _P(t, 1 - cu, 1); var n2 = _N(t, 1 - cu, 1)
    var a0 = p0 + n0 * hr[0]; var a1 = p1 + n1 * hr[0]; var a2 = p2 + n2 * hr[0]
    var b0 = p0 + n0 * hr[1]; var b1 = p1 + n1 * hr[1]; var b2 = p2 + n2 * hr[1]
    # One interval from the first to the last prism crossing: conservative where the paper pairs them; the box tests
    # drop the gaps.
    var q0 = _tri_hit(a0, a1, a2, dir, Float32(1e30)); var q1 = _tri_hit(b0, b1, b2, dir, Float32(1e30))
    var e0 = _bilinear_hits(a0, a1, b1, b0, dir); var e1 = _bilinear_hits(a1, a2, b2, b1, dir); var e2 = _bilinear_hits(a2, a0, b0, b2, dir)
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
    if nt == 0:
        return (False, t_max, Float32(0), Float32(0), 1)
    if nt % 2 == 1:   # a crossing lost on an edge, or the origin inside: the prism's box instead
        var plo = Vec3f(min(min(min(a0.x, a1.x), min(a2.x, b0.x)), min(b1.x, b2.x)), min(min(min(a0.y, a1.y), min(a2.y, b0.y)), min(b1.y, b2.y)),
                        min(min(min(a0.z, a1.z), min(a2.z, b0.z)), min(b1.z, b2.z)))
        var phi = Vec3f(max(max(max(a0.x, a1.x), max(a2.x, b0.x)), max(b1.x, b2.x)), max(max(max(a0.y, a1.y), max(a2.y, b0.y)), max(b1.y, b2.y)),
                        max(max(max(a0.z, a1.z), max(a2.z, b0.z)), max(b1.z, b2.z)))
        var bt = _box_t(plo, phi, Vec3f(Float32(1) / dir.x, Float32(1) / dir.y, Float32(1) / dir.z))
        lo = max(bt[0], Float32(0)); hi = bt[1]
    var cd = cross(t.Nc, dir); var ca = cross(t.Na, dir); var cb = cross(t.Nb, dir)
    var k = SIMD[DType.float32, 8](dot(t.A, cd), dot(t.A, ca) + dot(t.Pa, cd), dot(t.A, cb) + dot(t.Pb, cd),
                                   dot(t.Pa, ca), dot(t.Pa, cb) + dot(t.Pb, ca), dot(t.Pb, cb), 0, 0)
    var cpu_stack = Array[Float32, 1 if is_gpu() else 6 * _STACK](uninitialized=True)
    var stack: Pointer[Float32, MutUntrackedOrigin]
    comptime if is_gpu():   # the 17-entry stacks in shared local memory, not private memory
        stack = stack_allocation[_TPB * 6 * _STACK, Scalar[DType.float32], address_space=AddressSpace.SHARED]().unsafe_offset(
            Int(thread_idx.x) * 6 * _STACK).unsafe_address_space_cast[AddressSpace.GENERIC]().unsafe_origin_cast[MutUntrackedOrigin]()
    else:
        stack = cpu_stack.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var sp = 0
    var ac = Float32(1) / 3 if t.upper else Float32(2) / 3; var bc = Float32(2) / 3 if t.upper else Float32(1) / 3   # the centroid
    # The interval, split where the projected ray turns (psi_a = 0 or psi_b = 0), pushed back to front.
    var ti0 = lo; var ti1 = min(hi, t_lim)
    if ti1 >= ti0:
        var u0 = _invert(t, dir * ti0, ac, bc); var u1 = _invert(t, dir * ti1, ac, bc)
        var cut = SIMD[DType.float32, 8](Float32(1e30)); var nc = 0
        for line in range(2):
            var rr = _line_roots(k, Float32(2) * k[3], k[4], k[1]) if line == 0 else _line_roots(k, k[4], Float32(2) * k[5], k[2])
            for g in range(2):
                var a = rr[2 * g]; var b = rr[2 * g + 1]
                if a == a and _inside(t, a, b, Float32(1e-3)):
                    var tc = _ray_t(t, dir, a, b)
                    if tc > ti0 and tc < ti1 and nc < 4:
                        cut[nc] = tc; nc += 1
        for x in range(1, nc):
            var y = x
            while y > 0 and cut[y - 1] < cut[y]:   # descending: pushed back first
                var s = cut[y]; cut[y] = cut[y - 1]; cut[y - 1] = s; y -= 1
        var hi_t = ti1; var hi_uv = u1
        for x in range(nc + 1):
            var lo_t = cut[x] if x < nc else ti0
            var lo_uv = _invert(t, dir * lo_t, ac, bc) if x < nc else u0
            if sp < _STACK:
                stack[unsafe_offset=6 * sp] = lo_t; stack[unsafe_offset=6 * sp + 1] = hi_t; stack[unsafe_offset=6 * sp + 2] = lo_uv[0]; stack[unsafe_offset=6 * sp + 3] = lo_uv[1]
                stack[unsafe_offset=6 * sp + 4] = hi_uv[0]; stack[unsafe_offset=6 * sp + 5] = hi_uv[1]; sp += 1
            hi_t = lo_t; hi_uv = lo_uv
    var inv = Vec3f(Float32(1) / dir.x, Float32(1) / dir.y, Float32(1) / dir.z)
    var sw = t.side * Float32(1 << Int(d.hdr[unsafe_offset=3 * f + 1])); var sh = t.side * Float32(1 << Int(d.hdr[unsafe_offset=3 * f + 2]))
    var best = t_lim
    while sp > 0 and steps < 4096:
        sp -= 1
        var t0 = stack[unsafe_offset=6 * sp]; var t1 = min(stack[unsafe_offset=6 * sp + 1], best)
        if t1 < t0:
            continue
        var a0 = stack[unsafe_offset=6 * sp + 2]; var b0 = stack[unsafe_offset=6 * sp + 3]; var a1 = stack[unsafe_offset=6 * sp + 4]; var b1 = stack[unsafe_offset=6 * sp + 5]
        steps += 1
        var amin = max(min(a0, a1), Float32(0)); var amax = min(max(a0, a1), Float32(1))
        var bmin = max(min(b0, b1), Float32(0)); var bmax = min(max(b0, b1), Float32(1))
        if amax < amin or bmax < bmin:
            continue
        if (amax - amin) * sw <= march and (bmax - bmin) * sh <= march:
            var m = _march(d, t, f, k, dir, a0, b0, a1, b1, best, steps)
            if m[3] > Float32(0):   # front to back: the first hit is the nearest
                var pk = prim_patch(d, id >> 1); var ss = sub_square(pk[2], pk[1])
                return (True, m[0] + tp, ss[0] + m[1] * ss[2], ss[1] + m[2] * ss[2], steps)
            continue
        # The rectangle's surface box by affine arithmetic over (a, b, s).
        var hb = _height_range(d, r, t, f, amin, amax, bmin, bmax)
        var am = Float32(0.5) * (amin + amax); var bm = Float32(0.5) * (bmin + bmax); var sm = Float32(0.5) * (hb[0] + hb[1])
        var ra = Float32(0.5) * (amax - amin); var rb = Float32(0.5) * (bmax - bmin); var rs = Float32(0.5) * (hb[1] - hb[0])
        var nm = _N(t, am, bm)
        var cen = _P(t, am, bm) + nm * sm
        var ea = t.Pa + t.Na * sm; var eb = t.Pb + t.Nb * sm
        var rad = Vec3f(abs(ea.x) * ra + abs(eb.x) * rb + abs(nm.x) * rs + (abs(t.Na.x) * ra + abs(t.Nb.x) * rb) * rs,
                        abs(ea.y) * ra + abs(eb.y) * rb + abs(nm.y) * rs + (abs(t.Na.y) * ra + abs(t.Nb.y) * rb) * rs,
                        abs(ea.z) * ra + abs(eb.z) * rb + abs(nm.z) * rs + (abs(t.Na.z) * ra + abs(t.Nb.z) * rb) * rs)
        var bt = _box_t(cen - rad, cen + rad, inv)
        var n0 = max(t0, bt[0]); var n1 = min(t1, bt[1])
        if n1 < n0:
            continue
        if n0 > t0:   # reduce: project the box interval back (Sec. 4.3)
            var e = _invert(t, dir * n0, am, bm); a0 = min(max(e[0], amin), amax); b0 = min(max(e[1], bmin), bmax); t0 = n0
        if n1 < t1:
            var e = _invert(t, dir * n1, am, bm); a1 = min(max(e[0], amin), amax); b1 = min(max(e[1], bmin), bmax); t1 = n1
        # Split the longer side in the middle where the projected ray crosses it.
        var along_a = abs(a1 - a0) * sw >= abs(b1 - b0) * sh
        var mid = Float32(0.5) * ((a0 + a1) if along_a else (b0 + b1))
        var rr = _line_roots(k, Float32(1), Float32(0), -mid) if along_a else _line_roots(k, Float32(0), Float32(1), -mid)
        var ma = Float32(0); var mb = Float32(0); var found = False
        var lo2 = min(b0, b1) if along_a else min(a0, a1); var hi2 = max(b0, b1) if along_a else max(a0, a1)
        for g in range(2):
            var a = rr[2 * g]; var b = rr[2 * g + 1]
            var o = b if along_a else a
            if a == a and o >= lo2 - Float32(1e-4) and o <= hi2 + Float32(1e-4):
                ma = a; mb = b; found = True
        var tm: Float32
        if found:
            tm = min(max(_ray_t(t, dir, ma, mb), t0), t1)
        else:
            tm = Float32(0.5) * (t0 + t1)
            var e = _invert(t, dir * tm, Float32(0.5) * (a0 + a1), Float32(0.5) * (b0 + b1)); ma = e[0]; mb = e[1]
        if along_a:   # the split point on the split line, inside the rectangle: each side halves
            ma = mid; mb = min(max(mb, bmin), bmax)
        else:
            mb = mid; ma = min(max(ma, amin), amax)
        if sp + 2 <= _STACK:
            stack[unsafe_offset=6 * sp] = tm; stack[unsafe_offset=6 * sp + 1] = t1; stack[unsafe_offset=6 * sp + 2] = ma; stack[unsafe_offset=6 * sp + 3] = mb
            stack[unsafe_offset=6 * sp + 4] = a1; stack[unsafe_offset=6 * sp + 5] = b1; sp += 1
            stack[unsafe_offset=6 * sp] = t0; stack[unsafe_offset=6 * sp + 1] = tm; stack[unsafe_offset=6 * sp + 2] = a0; stack[unsafe_offset=6 * sp + 3] = b0
            stack[unsafe_offset=6 * sp + 4] = ma; stack[unsafe_offset=6 * sp + 5] = mb; sp += 1
    return (False, t_max, Float32(0), Float32(0), steps)
