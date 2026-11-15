"""Tessellation-free displacement: RMIP (Thonat, Georgiev, Beaune, Boubekeur, SIGGRAPH Asia 2023).

A base triangle with vertex normals is displaced by a scalar height map, S(u, v) = P(u, v) + h(u, v) N / |N|,
as pbrt-v4 displaces a plymesh. Nothing is tessellated: a ray that enters the triangle's bounding prism is
projected into texture space, where a rectangular min-max pyramid bounds h over any axis-aligned rectangle;
the projected ray is split until it spans a few texels, whose micro-triangles are then intersected.
Texture coordinates are texel units, sample (i, j) at integer (i, j), the map repeating."""
from std.math import sqrt, min, max, abs, floor, ceil
from std.memory.alloc import unsafe_alloc
from std.ffi import external_call
from gonzales.core.geometry import Vec3f, cross, dot

comptime MARCH = 2   # a bound this many texels wide is intersected texel by texel
comptime STACK = 32


# ── RMIP: min-max over power-of-two rectangles of blocks ─────────────────────

@fieldwise_init
struct Rmip(TrivialRegisterPassable):
    var h: Pointer[Float32, MutUntrackedOrigin]     # w x hh height samples, row-major
    var tab: Pointer[Float32, MutUntrackedOrigin]   # (min, max) per (q, p, y, x) block window
    var w: Int32
    var hh: Int32
    var b: Int32    # samples per block side
    var rx: Int32   # blocks
    var ry: Int32
    var lx: Int32   # window sizes 2^0 .. 2^(lx-1) blocks
    var ly: Int32

    @always_inline
    def sample(self, i: Int, j: Int) -> Float32:
        var w = Int(self.w); var hh = Int(self.hh)
        return self.h[unsafe_offset=((j % hh + hh) % hh) * w + (i % w + w) % w]

    @always_inline
    def _entry(self, p: Int, q: Int, x: Int, y: Int) -> Tuple[Float32, Float32]:
        var k = 2 * (((q * Int(self.lx) + p) * Int(self.ry) + y) * Int(self.rx) + x)
        return (self.tab[unsafe_offset=k], self.tab[unsafe_offset=k + 1])

    @always_inline
    def _blocks(self, i0: Int, i1: Int, j0: Int, j1: Int) -> Tuple[Float32, Float32]:
        """Min-max over samples [i0, i1] x [j0, j1], inside the map."""
        var b = Int(self.b)
        var bx0 = i0 // b; var bx1 = i1 // b; var by0 = j0 // b; var by1 = j1 // b
        var p = _log2(bx1 - bx0 + 1); var q = _log2(by1 - by0 + 1)
        var xb = bx1 - (1 << p) + 1; var yb = by1 - (1 << q) + 1
        var e0 = self._entry(p, q, bx0, by0); var e1 = self._entry(p, q, xb, by0)
        var e2 = self._entry(p, q, bx0, yb); var e3 = self._entry(p, q, xb, yb)
        return (min(min(e0[0], e1[0]), min(e2[0], e3[0])), max(max(e0[1], e1[1]), max(e2[1], e3[1])))

    @always_inline
    def query(self, i0_in: Int, i1_in: Int, j0_in: Int, j1_in: Int) -> Tuple[Float32, Float32]:
        """Min-max of h over samples [i0, i1] x [j0, j1], any integers: the map repeats."""
        var w = Int(self.w); var hh = Int(self.hh)
        var i0 = i0_in; var i1 = i1_in; var j0 = j0_in; var j1 = j1_in
        if i1 < i0 or i1 - i0 + 1 >= w: i0 = 0; i1 = w - 1
        if j1 < j0: j0 = 0; j1 = hh - 1
        if j1 - j0 + 1 >= hh: j0 = 0; j1 = hh - 1
        var si = (i0 % w + w) % w - i0; i0 += si; i1 += si
        var sj = (j0 % hh + hh) % hh - j0; j0 += sj; j1 += sj
        var lo = Float32(3.0e38); var hi = Float32(-3.0e38)
        for xs in range(2):
            var a0 = i0 if xs == 0 else 0
            var a1 = min(i1, w - 1) if xs == 0 else i1 - w
            if a1 < a0: continue
            for ys in range(2):
                var c0 = j0 if ys == 0 else 0
                var c1 = min(j1, hh - 1) if ys == 0 else j1 - hh
                if c1 < c0: continue
                var e = self._blocks(a0, a1, c0, c1)
                lo = min(lo, e[0]); hi = max(hi, e[1])
        return (lo, hi)


@always_inline
def _log2(n: Int) -> Int:
    var p = 0
    while (2 << p) <= n:
        p += 1
    return p


def rmip_build(h: Pointer[Float32, MutUntrackedOrigin], w: Int, hh: Int) -> Rmip:
    """The RMIP of a w x hh height map, its blocks grown until it holds no more entries than the map has samples."""
    var b = 1
    while True:
        var rx = (w + b - 1) // b; var ry = (hh + b - 1) // b
        if rx * ry * (_log2(rx) + 1) * (_log2(ry) + 1) <= max(w * hh, 1) or (rx == 1 and ry == 1):
            break
        b *= 2
    var rx = (w + b - 1) // b; var ry = (hh + b - 1) // b
    var lx = _log2(rx) + 1; var ly = _log2(ry) + 1
    var tab = unsafe_alloc[Float32](2 * rx * ry * lx * ly)
    @always_inline
    def at(p: Int, q: Int, x: Int, y: Int) {imm} -> Int:
        return 2 * (((q * lx + p) * ry + y) * rx + x)
    for by in range(ry):
        for bx in range(rx):
            var lo = Float32(3.0e38); var hi = Float32(-3.0e38)
            for l in range(b):
                for k in range(b):
                    var v = h[unsafe_offset=((by * b + l) % hh) * w + (bx * b + k) % w]
                    lo = min(lo, v); hi = max(hi, v)
            tab[unsafe_offset=at(0, 0, bx, by)] = lo; tab[unsafe_offset=at(0, 0, bx, by) + 1] = hi
    for p in range(1, lx):
        for y in range(ry):
            for x in range(rx):
                var a = at(p - 1, 0, x, y); var c = at(p - 1, 0, (x + (1 << (p - 1))) % rx, y); var d = at(p, 0, x, y)
                tab[unsafe_offset=d] = min(tab[unsafe_offset=a], tab[unsafe_offset=c])
                tab[unsafe_offset=d + 1] = max(tab[unsafe_offset=a + 1], tab[unsafe_offset=c + 1])
    for q in range(1, ly):
        for p in range(lx):
            for y in range(ry):
                for x in range(rx):
                    var a = at(p, q - 1, x, y); var c = at(p, q - 1, x, (y + (1 << (q - 1))) % ry); var d = at(p, q, x, y)
                    tab[unsafe_offset=d] = min(tab[unsafe_offset=a], tab[unsafe_offset=c])
                    tab[unsafe_offset=d + 1] = max(tab[unsafe_offset=a + 1], tab[unsafe_offset=c + 1])
    return Rmip(h, tab, Int32(w), Int32(hh), Int32(b), Int32(rx), Int32(ry), Int32(lx), Int32(ly))


def load_height_map(filename: String, scale: Float32, bias: Float32) -> Rmip:
    """pbrt's float imagemap as heights: bias + scale * its red channel (an 8-bit png decoded as sRGB). w = 0 if unreadable."""
    var n = filename.byte_length()
    var cs = unsafe_alloc[UInt8](n + 1)
    for k in range(n): cs[unsafe_offset=k] = filename.unsafe_ptr()[unsafe_offset=k]
    cs[unsafe_offset=n] = UInt8(0)
    var data = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](1)
    var wh = unsafe_alloc[Int32](2)
    var ok = external_call["load_texture_rgb", Int32, Pointer[UInt8, MutUntrackedOrigin],
        Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin],
        Pointer[Int32, MutUntrackedOrigin], Int32](cs, data, wh, wh.unsafe_offset(1), Int32(0))
    cs.unsafe_free()
    var w = Int(wh[unsafe_offset=0]); var hh = Int(wh[unsafe_offset=1])
    var rgb = data[unsafe_offset=0]
    data.unsafe_free(); wh.unsafe_free()
    if ok == Int32(0) or w <= 0 or hh <= 0:
        return Rmip(Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
                    0, 0, 1, 0, 0, 0, 0)
    var h = unsafe_alloc[Float32](w * hh)
    for k in range(w * hh):
        h[unsafe_offset=k] = bias + scale * rgb[unsafe_offset=3 * k]
    _ = external_call["free_texture_rgb", Int32, Pointer[Float32, MutUntrackedOrigin]](rgb)
    return rmip_build(h, w, hh)


# ── A displaced base triangle ────────────────────────────────────────────────

@fieldwise_init
struct DisplacedMesh(TrivialRegisterPassable):
    """A mesh's displaced triangles, one per triangle in order, and its height map."""
    var tris: Pointer[DisplacedTri, MutUntrackedOrigin]
    var rm: Rmip


@fieldwise_init
struct DisplacedTri(TrivialRegisterPassable):
    """P and N are linear in texel coordinates (u, v) taken relative to the texture-space centroid tc."""
    var pc: Vec3f
    var pu: Vec3f
    var pv: Vec3f
    var nc: Vec3f
    var nu: Vec3f
    var nv: Vec3f
    var tc: SIMD[DType.float32, 2]
    var t: SIMD[DType.float32, 8]       # the vertices' local texel coordinates u0 v0 u1 v1 u2 v2
    var binv: SIMD[DType.float32, 4]    # local (u, v) -> barycentrics (b1, b2): b1 = binv0 du + binv1 dv from vertex 0
    var smin: Float32                   # the prism: P + s N, s in [smin, smax]
    var smax: Float32
    var nlo: Float32                    # |N| over the triangle
    var nhi: Float32

    @always_inline
    def p(self, u: Float32, v: Float32) -> Vec3f:
        return self.pc + self.pu * u + self.pv * v

    @always_inline
    def ok(self) -> Bool:
        """False for a triangle without area in texture space: it stays a plain triangle."""
        var d = self.binv[0] * self.binv[3] - self.binv[1] * self.binv[2]   # 1 / (twice the texel area)
        return d != Float32(0) and abs(d) < Float32(1e10)

    @always_inline
    def bounds(self) -> Tuple[Vec3f, Vec3f]:
        """The prism's AABB."""
        var lo = Vec3f(3.0e38); var hi = Vec3f(-3.0e38)
        for k in range(3):
            var p = self.p(self.t[2 * k], self.t[2 * k + 1]); var n = self.n(self.t[2 * k], self.t[2 * k + 1])
            for s in range(2):
                var q = p + n * (self.smin if s == 0 else self.smax)
                lo = Vec3f(min(lo.x, q.x), min(lo.y, q.y), min(lo.z, q.z)); hi = Vec3f(max(hi.x, q.x), max(hi.y, q.y), max(hi.z, q.z))
        return (lo, hi)

    @always_inline
    def n(self, u: Float32, v: Float32) -> Vec3f:
        return self.nc + self.nu * u + self.nv * v

    @always_inline
    def clamp_uv(self, uv: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
        """Into the triangle's texel box, padded a texel: the ray's projection inside the prism never leaves it."""
        var lo = SIMD[DType.float32, 2](min(min(self.t[0], self.t[2]), self.t[4]), min(min(self.t[1], self.t[3]), self.t[5])) - Float32(1)
        var hi = SIMD[DType.float32, 2](max(max(self.t[0], self.t[2]), self.t[4]), max(max(self.t[1], self.t[3]), self.t[5])) + Float32(1)
        var r = uv
        comptime for k in range(2):
            if not (r[k] >= lo[k]): r[k] = lo[k]   # NaN too
            if not (r[k] <= hi[k]): r[k] = hi[k]
        return r

    @always_inline
    def bary(self, u: Float32, v: Float32) -> SIMD[DType.float32, 2]:
        var du = u - self.t[0]; var dv = v - self.t[1]
        return SIMD[DType.float32, 2](self.binv[0] * du + self.binv[1] * dv, self.binv[2] * du + self.binv[3] * dv)

    @always_inline
    def surface(self, rm: Rmip, i: Int, j: Int) -> Vec3f:
        """The displaced surface at sample (i, j), in absolute texel coordinates."""
        var u = Float32(i) - self.tc[0]; var v = Float32(j) - self.tc[1]
        var n = self.n(u, v)
        return self.p(u, v) + n * (rm.sample(i, j) / max(n.length(), Float32(1e-12)))


def displaced_tri(p0: Vec3f, p1: Vec3f, p2: Vec3f, n0: Vec3f, n1: Vec3f, n2: Vec3f,
                  t0: SIMD[DType.float32, 2], t1: SIMD[DType.float32, 2], t2: SIMD[DType.float32, 2],
                  rm: Rmip) -> DisplacedTri:
    """A base triangle (unit vertex normals, texel coordinates t_i) and its bounding prism over the map."""
    var tc = (t0 + t1 + t2) * Float32(1.0 / 3.0)
    var a = t0 - tc; var b = t1 - tc; var c = t2 - tc
    var du1 = b[0] - a[0]; var dv1 = b[1] - a[1]; var du2 = c[0] - a[0]; var dv2 = c[1] - a[1]
    var det = du1 * dv2 - du2 * dv1
    var inv = Float32(1) / det if det != Float32(0) else Float32(0)
    var binv = SIMD[DType.float32, 4](dv2 * inv, -du2 * inv, -dv1 * inv, du1 * inv)
    var pu = (p1 - p0) * binv[0] + (p2 - p0) * binv[2]
    var pv = (p1 - p0) * binv[1] + (p2 - p0) * binv[3]
    var nu = (n1 - n0) * binv[0] + (n2 - n0) * binv[2]
    var nv = (n1 - n0) * binv[1] + (n2 - n0) * binv[3]
    var pc = (p0 + p1 + p2) * Float32(1.0 / 3.0)
    var nc = (n0 + n1 + n2) * Float32(1.0 / 3.0)
    var nlo = _min_normal_length(n0, n1, n2)
    var hb = rm.query(Int(floor(min(min(t0[0], t1[0]), t2[0]))), Int(ceil(max(max(t0[0], t1[0]), t2[0]))),
                      Int(floor(min(min(t0[1], t1[1]), t2[1]))), Int(ceil(max(max(t0[1], t1[1]), t2[1]))))
    var pad = (hb[1] - hb[0]) * Float32(0.01) + Float32(1e-6)   # micro-triangles bend off the field a little
    var smin = min(hb[0] / nlo, hb[0]) - pad
    var smax = max(hb[1] / nlo, hb[1]) + pad
    return DisplacedTri(pc, pu, pv, nc, nu, nv, tc, SIMD[DType.float32, 8](a[0], a[1], b[0], b[1], c[0], c[1], 0, 0),
                        binv, smin, smax, nlo, Float32(1))


def _min_normal_length(n0: Vec3f, n1: Vec3f, n2: Vec3f) -> Float32:
    """min |b0 n0 + b1 n1 + b2 n2| over the triangle, unit n_i (RMIP supplement, eq. 1-2)."""
    var a = n0 - n2; var b = n1 - n2
    var g00 = dot(a, a); var g01 = dot(a, b); var g11 = dot(b, b)
    var r0 = -dot(n2, a); var r1 = -dot(n2, b)
    var det = g00 * g11 - g01 * g01
    var best = min(min(Float32(1), (n0 + n1).length() * Float32(0.5)), min((n1 + n2).length(), (n2 + n0).length()) * Float32(0.5))
    if det > Float32(1e-12):
        var x = (r0 * g11 - r1 * g01) / det; var y = (g00 * r1 - g01 * r0) / det
        if x >= Float32(0) and y >= Float32(0) and x + y <= Float32(1):
            best = min(best, (n0 * x + n1 * y + n2 * (Float32(1) - x - y)).length())
    return max(best, Float32(1e-3))


# ── Ray against the prism ────────────────────────────────────────────────────

@always_inline
def _tri_t(o: Vec3f, d: Vec3f, a: Vec3f, b: Vec3f, c: Vec3f) -> Float32:
    """Two-sided ray-triangle distance, -1 for a miss."""
    var e1 = b - a; var e2 = c - a
    var pv = cross(d, e2); var det = dot(e1, pv)
    if det == Float32(0): return Float32(-1)
    var inv = Float32(1) / det
    var tv = o - a; var u = dot(tv, pv) * inv
    if u < Float32(0) or u > Float32(1): return Float32(-1)
    var qv = cross(tv, e1); var v = dot(d, qv) * inv
    if v < Float32(0) or u + v > Float32(1): return Float32(-1)
    return dot(e2, qv) * inv


@always_inline
def _patch_t(o: Vec3f, d: Vec3f, q00: Vec3f, q10: Vec3f, q11: Vec3f, q01: Vec3f) -> SIMD[DType.float32, 2]:
    """Both ray-bilinear patch distances (Reshetov 2019, Cool Patches), -1 for none."""
    var e11 = q11 - q10; var e00 = q01 - q00
    var qn = cross(q10 - q00, q01 - q11)
    var a0 = q00 - o; var a1 = q10 - o
    var a = dot(cross(a0, d), e00); var c = dot(qn, d)
    var b = dot(cross(a1, d), e11) - a - c
    var out = SIMD[DType.float32, 2](-1, -1)
    var disc = b * b - Float32(4) * a * c
    if disc < Float32(0): return out
    disc = sqrt(disc)
    var u1: Float32; var u2: Float32
    if c == Float32(0):
        if b == Float32(0): return out
        u1 = -a / b; u2 = Float32(-1)
    else:
        u1 = (-b - (disc if b >= Float32(0) else -disc)) * Float32(0.5)
        u2 = a / u1 if u1 != Float32(0) else Float32(-1)
        u1 /= c
    comptime for k in range(2):
        var u = u1 if k == 0 else u2
        if u >= Float32(0) and u <= Float32(1):
            var pa = a0 + (a1 - a0) * u; var pb = e00 + (e11 - e00) * u
            var n = cross(d, pb); var dd = dot(n, n)
            if dd != Float32(0):
                n = cross(n, pa)
                var v = dot(n, d)
                if v >= Float32(0) and v <= dd: out[k] = dot(n, pb) / dd
    return out


# ── Texture space: inversion and the projected ray ───────────────────────────

@always_inline
def invert(tri: DisplacedTri, x: Vec3f) -> SIMD[DType.float32, 2]:
    """The local (u, v) whose displacement line passes through x: (P - x) x N = 0, Gauss-Newton."""
    # start from x projected onto the base plane
    var n0 = cross(tri.pu, tri.pv)
    var r = x - tri.pc
    var r2 = r - n0 * (dot(r, n0) / max(dot(n0, n0), Float32(1e-30)))
    var g00 = dot(tri.pu, tri.pu); var g01 = dot(tri.pu, tri.pv); var g11 = dot(tri.pv, tri.pv)
    var det0 = g00 * g11 - g01 * g01
    var u = (dot(r2, tri.pu) * g11 - dot(r2, tri.pv) * g01) / det0
    var v = (dot(r2, tri.pv) * g00 - dot(r2, tri.pu) * g01) / det0
    return invert_from(tri, x, u, v)


@always_inline
def invert_from(tri: DisplacedTri, x: Vec3f, u_in: Float32, v_in: Float32) -> SIMD[DType.float32, 2]:
    """invert, starting from a nearby (u, v)."""
    var u = u_in; var v = v_in
    for _ in range(10):
        var p = tri.p(u, v); var n = tri.n(u, v)
        var f = cross(p - x, n)
        var ju = cross(tri.pu, n) + cross(p - x, tri.nu)
        var jv = cross(tri.pv, n) + cross(p - x, tri.nv)
        var a00 = dot(ju, ju); var a01 = dot(ju, jv); var a11 = dot(jv, jv)
        var b0 = -dot(ju, f); var b1 = -dot(jv, f)
        var det = a00 * a11 - a01 * a01
        if det == Float32(0): break
        var du = (b0 * a11 - b1 * a01) / det; var dv = (a00 * b1 - a01 * b0) / det
        u += du; v += dv
        if du * du + dv * dv < Float32(1e-10): break
    return SIMD[DType.float32, 2](u, v)


@fieldwise_init
struct _Psi(TrivialRegisterPassable):
    """psi(u, v) = det(P - o, N, d) = c0 + c1 u + c2 v + c3 u^2 + c4 u v + c5 v^2: the ray projected into texture space."""
    var c: SIMD[DType.float32, 8]

    @always_inline
    def at(self, u: Float32, v: Float32) -> Float32:
        return self.c[0] + self.c[1] * u + self.c[2] * v + self.c[3] * u * u + self.c[4] * u * v + self.c[5] * v * v


@always_inline
def _det(a: Vec3f, b: Vec3f, c: Vec3f) -> Float32:
    return dot(a, cross(b, c))


@always_inline
def _psi(tri: DisplacedTri, o: Vec3f, d: Vec3f) -> _Psi:
    var a0 = tri.pc - o
    return _Psi(SIMD[DType.float32, 8](
        _det(a0, tri.nc, d),
        _det(tri.pu, tri.nc, d) + _det(a0, tri.nu, d),
        _det(tri.pv, tri.nc, d) + _det(a0, tri.nv, d),
        _det(tri.pu, tri.nu, d),
        _det(tri.pu, tri.nv, d) + _det(tri.pv, tri.nu, d),
        _det(tri.pv, tri.nv, d), 0, 0))


@always_inline
def _ray_t(tri: DisplacedTri, o: Vec3f, d: Vec3f, u: Float32, v: Float32) -> Float32:
    """The ray distance whose point lies on the displacement line at (u, v): o + t d = P + s N, least squares."""
    var p = tri.p(u, v) - o; var n = tri.n(u, v)
    var dd = dot(d, d); var dn = dot(d, n); var nn = dot(n, n)
    var det = dd * nn - dn * dn
    if abs(det) < Float32(1e-20): return dot(p, d) / dd
    return (dot(p, d) * nn - dot(p, n) * dn) / det


@always_inline
def _quad_roots(a: Float32, b: Float32, c: Float32) -> SIMD[DType.float32, 2]:
    """Real roots of a x^2 + b x + c, NaN-free: 3e38 for a missing one."""
    var none = Float32(3.0e38)
    if abs(a) < Float32(1e-12) * (abs(b) + abs(c) + Float32(1e-30)):
        return SIMD[DType.float32, 2](-c / b if b != Float32(0) else none, none)
    var disc = b * b - Float32(4) * a * c
    if disc < Float32(0): return SIMD[DType.float32, 2](none, none)
    var q = Float32(-0.5) * (b + (sqrt(disc) if b >= Float32(0) else -sqrt(disc)))
    return SIMD[DType.float32, 2](q / a, c / q if q != Float32(0) else none)


@always_inline
def _cross_line(ps: _Psi, along_u: Bool, x: Float32, lo: Float32, hi: Float32) -> Float32:
    """Where psi = 0 crosses u = x (along_u) or v = x, the other coordinate in [lo, hi]; 3e38 if nowhere."""
    var a: Float32; var b: Float32; var c: Float32
    if along_u:   # u = x: psi(v)
        a = ps.c[5]; b = ps.c[2] + ps.c[4] * x; c = ps.c[0] + ps.c[1] * x + ps.c[3] * x * x
    else:
        a = ps.c[3]; b = ps.c[1] + ps.c[4] * x; c = ps.c[0] + ps.c[2] * x + ps.c[5] * x * x
    var r = _quad_roots(a, b, c)
    var eps = (hi - lo) * Float32(1e-3) + Float32(1e-6)
    var best = Float32(3.0e38)
    comptime for k in range(2):
        if r[k] >= lo - eps and r[k] <= hi + eps:
            best = r[k] if best == Float32(3.0e38) else best
    return best


# ── The intersection ─────────────────────────────────────────────────────────

@fieldwise_init
struct DisplacedHit(TrivialRegisterPassable):
    var hit: Bool
    var t: Float32
    var b1: Float32    # base barycentrics
    var b2: Float32
    var ng: Vec3f      # the micro-triangle's normal (unnormalized)
    var u: Float32     # local texel coordinates of the hit
    var v: Float32


@always_inline
def _box_hit(o: Vec3f, inv_d: Vec3f, lo: Vec3f, hi: Vec3f, t0: Float32, t1: Float32) -> Bool:
    var tx0 = (lo.x - o.x) * inv_d.x; var tx1 = (hi.x - o.x) * inv_d.x
    var ty0 = (lo.y - o.y) * inv_d.y; var ty1 = (hi.y - o.y) * inv_d.y
    var tz0 = (lo.z - o.z) * inv_d.z; var tz1 = (hi.z - o.z) * inv_d.z
    var tn = max(max(min(tx0, tx1), min(ty0, ty1)), max(min(tz0, tz1), t0))
    var tf = min(min(max(tx0, tx1), max(ty0, ty1)), min(max(tz0, tz1), t1))
    return tn <= tf * Float32(1.00001)


@always_inline
def _interval_mul(a0: Float32, a1: Float32, b0: Float32, b1: Float32) -> SIMD[DType.float32, 2]:
    var p = SIMD[DType.float32, 4](a0 * b0, a0 * b1, a1 * b0, a1 * b1)
    return SIMD[DType.float32, 2](p.reduce_min(), p.reduce_max())


@always_inline
def _surface_box(tri: DisplacedTri, rm: Rmip, u0: Float32, u1: Float32, v0: Float32, v1: Float32) -> Tuple[Vec3f, Vec3f]:
    """An AABB of the displaced surface over the local rectangle (affine arithmetic, h from the RMIP)."""
    var hb = rm.query(Int(floor(u0 + tri.tc[0])), Int(ceil(u1 + tri.tc[0])), Int(floor(v0 + tri.tc[1])), Int(ceil(v1 + tri.tc[1])))
    var s0 = min(hb[0] / tri.nlo, hb[0] / tri.nhi); var s1 = max(hb[1] / tri.nlo, hb[1] / tri.nhi)
    var lo = SIMD[DType.float32, 4](0); var hi = SIMD[DType.float32, 4](0)
    var pc = tri.pc; var pu = tri.pu; var pv = tri.pv; var nc = tri.nc; var nu = tri.nu; var nv = tri.nv
    comptime for k in range(3):
        var pck = pc.x if k == 0 else (pc.y if k == 1 else pc.z)
        var puk = pu.x if k == 0 else (pu.y if k == 1 else pu.z)
        var pvk = pv.x if k == 0 else (pv.y if k == 1 else pv.z)
        var nck = nc.x if k == 0 else (nc.y if k == 1 else nc.z)
        var nuk = nu.x if k == 0 else (nu.y if k == 1 else nu.z)
        var nvk = nv.x if k == 0 else (nv.y if k == 1 else nv.z)
        var a = _interval_mul(puk, puk, u0, u1); var b = _interval_mul(pvk, pvk, v0, v1)
        var c = _interval_mul(nuk, nuk, u0, u1); var e = _interval_mul(nvk, nvk, v0, v1)
        var n0 = nck + c[0] + e[0]; var n1 = nck + c[1] + e[1]
        var sn = _interval_mul(s0, s1, n0, n1)
        lo[k] = pck + a[0] + b[0] + sn[0]; hi[k] = pck + a[1] + b[1] + sn[1]
    var pad = (hi - lo).reduce_max() * Float32(1e-4) + Float32(1e-6)
    return (Vec3f(lo[0] - pad, lo[1] - pad, lo[2] - pad), Vec3f(hi[0] + pad, hi[1] + pad, hi[2] + pad))


@always_inline
def _march(tri: DisplacedTri, rm: Rmip, o: Vec3f, d: Vec3f, u0: Float32, u1: Float32, v0: Float32, v1: Float32,
           mut best: DisplacedHit):
    """Intersect the micro-triangles of the texel cells under a local rectangle; keep hits inside the base triangle."""
    var lo = tri.clamp_uv(SIMD[DType.float32, 2](u0, v0)); var hi = tri.clamp_uv(SIMD[DType.float32, 2](u1, v1))
    var i0 = Int(floor(lo[0] + tri.tc[0])); var i1 = Int(floor(hi[0] + tri.tc[0]))
    var j0 = Int(floor(lo[1] + tri.tc[1])); var j1 = Int(floor(hi[1] + tri.tc[1]))
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            var s00 = tri.surface(rm, i, j); var s10 = tri.surface(rm, i + 1, j)
            var s01 = tri.surface(rm, i, j + 1); var s11 = tri.surface(rm, i + 1, j + 1)
            for k in range(2):   # (00, 10, 11) and (00, 11, 01)
                var b = s10 if k == 0 else s11
                var c = s11 if k == 0 else s01
                var e1 = b - s00; var e2 = c - s00
                var pv = cross(d, e2); var det = dot(e1, pv)
                if det == Float32(0): continue
                var inv = Float32(1) / det
                var tv = o - s00; var a = dot(tv, pv) * inv
                if a < Float32(0) or a > Float32(1): continue
                var qv = cross(tv, e1); var bb = dot(d, qv) * inv
                if bb < Float32(0) or a + bb > Float32(1): continue
                var t = dot(e2, qv) * inv
                if t <= Float32(0) or t >= best.t: continue
                # the triangle owns the points whose displacement line starts in it, as its prism does
                var hu = Float32(i) + (a + bb if k == 0 else a) - tri.tc[0]
                var hv = Float32(j) + (a if k == 0 else a + bb) - tri.tc[1]
                var uv = invert_from(tri, o + d * t, hu, hv)
                var bc = tri.bary(uv[0], uv[1])
                comptime EPS = Float32(1e-4)
                if bc[0] < -EPS or bc[1] < -EPS or bc[0] + bc[1] > Float32(1) + EPS: continue
                best = DisplacedHit(True, t, bc[0], bc[1], cross(e1, e2), uv[0], uv[1])


@always_inline
def intersect_displaced(tri: DisplacedTri, rm: Rmip, o: Vec3f, d: Vec3f, tmax: Float32) -> DisplacedHit:
    """The nearest hit of the displaced triangle in (0, tmax): RMIP's traversal (its Algorithm 1).
    Its lists live in scalar arrays read through pointers, as the BVH stacks do: a runtime index into a SIMD value
    or an array of structs miscompiles on the Intel GPU (wrong prism distances, then faults)."""
    var best = DisplacedHit(False, tmax, 0, 0, Vec3f(0), 0, 0)
    # the prism: two triangles and three bilinear patches, every crossing in front of the origin
    var lo0 = tri.p(tri.t[0], tri.t[1]) + tri.n(tri.t[0], tri.t[1]) * tri.smin
    var lo1 = tri.p(tri.t[2], tri.t[3]) + tri.n(tri.t[2], tri.t[3]) * tri.smin
    var lo2 = tri.p(tri.t[4], tri.t[5]) + tri.n(tri.t[4], tri.t[5]) * tri.smin
    var hi0 = tri.p(tri.t[0], tri.t[1]) + tri.n(tri.t[0], tri.t[1]) * tri.smax
    var hi1 = tri.p(tri.t[2], tri.t[3]) + tri.n(tri.t[2], tri.t[3]) * tri.smax
    var hi2 = tri.p(tri.t[4], tri.t[5]) + tri.n(tri.t[4], tri.t[5]) * tri.smax
    var c01 = _patch_t(o, d, lo0, lo1, hi1, hi0); var c12 = _patch_t(o, d, lo1, lo2, hi2, hi1); var c20 = _patch_t(o, d, lo2, lo0, hi0, hi2)
    var cand = SIMD[DType.float32, 8](_tri_t(o, d, lo0, lo1, lo2), _tri_t(o, d, hi0, hi1, hi2), c01[0], c01[1], c12[0], c12[1], c20[0], c20[1])
    var tsa = Array[Float32, 8](fill=Float32(3.0e38)); var ts = tsa.unsafe_ptr()
    var cnt = 0
    comptime for k in range(8):
        if cand[k] > Float32(0):
            ts[unsafe_offset=cnt] = cand[k]; cnt += 1
    if cnt == 0: return best
    for i in range(1, cnt):   # insertion sort
        var x = ts[unsafe_offset=i]; var j = i - 1
        while j >= 0 and ts[unsafe_offset=j] > x:
            ts[unsafe_offset=j + 1] = ts[unsafe_offset=j]; j -= 1
        ts[unsafe_offset=j + 1] = x
    var ps = _psi(tri, o, d)
    var inv_d = Vec3f(Float32(1) / d.x, Float32(1) / d.y, Float32(1) / d.z)
    # the stack: a bound is (u, v, t) at both ends of a monotone piece of the projected ray
    var sta = Array[Float32, 6 * STACK](uninitialized=True); var st = sta.unsafe_ptr()
    var sp = 0
    var pta = Array[Float32, 18](uninitialized=True); var pt = pta.unsafe_ptr()   # a piece's split points
    var cta = Array[Float32, 4](uninitialized=True); var cut = cta.unsafe_ptr()
    # intervals inside the prism, front to back; an odd count means the ray starts inside
    var k0 = cnt % 2
    var iv = (cnt + k0) // 2 - 1
    while iv >= 0:   # push back to front so the front pops first
        var ta = Float32(0) if (k0 == 1 and iv == 0) else ts[unsafe_offset=2 * iv - k0]
        var tb = ts[unsafe_offset=2 * iv + 1 - k0]
        iv -= 1
        if ta >= best.t: continue
        tb = min(tb, best.t)
        # split at the turning points (psi_u = 0 or psi_v = 0 on the curve), keeping each piece monotone
        var nc = 0
        comptime for axis in range(2):
            # grad psi component: psi_u = c1 + 2 c3 u + c4 v, psi_v = c2 + c4 u + 2 c5 v; a line in (u, v)
            var g0 = ps.c[1] if axis == 0 else ps.c[2]
            var gu = Float32(2) * ps.c[3] if axis == 0 else ps.c[4]
            var gv = ps.c[4] if axis == 0 else Float32(2) * ps.c[5]
            # parametrize the line g0 + gu u + gv v = 0 as q + r * (-gv, gu)
            var gg = gu * gu + gv * gv
            if gg >= Float32(1e-30):
                var qu = -g0 * gu / gg; var qv = -g0 * gv / gg
                var ru = -gv; var rv = gu
                var a = ps.c[3] * ru * ru + ps.c[4] * ru * rv + ps.c[5] * rv * rv
                var b = ps.c[1] * ru + ps.c[2] * rv + Float32(2) * ps.c[3] * qu * ru + ps.c[4] * (qu * rv + qv * ru) + Float32(2) * ps.c[5] * qv * rv
                var r = _quad_roots(a, b, ps.at(qu, qv))
                comptime for m in range(2):
                    if r[m] != Float32(3.0e38):
                        var t = _ray_t(tri, o, d, qu + r[m] * ru, qv + r[m] * rv)
                        if t > ta and t < tb and nc < 4:
                            cut[unsafe_offset=nc] = t; nc += 1
        for i in range(1, nc):
            var x = cut[unsafe_offset=i]; var j = i - 1
            while j >= 0 and cut[unsafe_offset=j] > x:
                cut[unsafe_offset=j + 1] = cut[unsafe_offset=j]; j -= 1
            cut[unsafe_offset=j + 1] = x
        for m in range(nc + 2):
            var t = ta if m == 0 else (tb if m == nc + 1 else cut[unsafe_offset=m - 1])
            var uv = tri.clamp_uv(invert(tri, o + d * t))
            pt[unsafe_offset=3 * m] = uv[0]; pt[unsafe_offset=3 * m + 1] = uv[1]; pt[unsafe_offset=3 * m + 2] = t
        var m = nc
        while m >= 0 and sp < STACK:
            for c in range(6): st[unsafe_offset=6 * sp + c] = pt[unsafe_offset=3 * m + c]
            sp += 1; m -= 1
    while sp > 0:
        sp -= 1
        var au = st[unsafe_offset=6 * sp]; var av = st[unsafe_offset=6 * sp + 1]; var at = st[unsafe_offset=6 * sp + 2]
        var bu = st[unsafe_offset=6 * sp + 3]; var bv = st[unsafe_offset=6 * sp + 4]; var bt = st[unsafe_offset=6 * sp + 5]
        if at >= best.t: continue
        var u0 = min(au, bu); var u1 = max(au, bu)
        var v0 = min(av, bv); var v1 = max(av, bv)
        if u1 - u0 <= Float32(MARCH) and v1 - v0 <= Float32(MARCH):
            _march(tri, rm, o, d, u0, u1, v0, v1, best)
            if best.hit and best.t <= bt:   # front to back: nothing later is nearer
                return best
            continue
        var box = _surface_box(tri, rm, u0, u1, v0, v1)
        if not _box_hit(o, inv_d, box[0], box[1], at, min(bt, best.t)):
            continue
        # split the longer side in the middle where the projected ray crosses it
        var along_u = (u1 - u0) >= (v1 - v0)
        var x = (u0 + u1) * Float32(0.5) if along_u else (v0 + v1) * Float32(0.5)
        var y = _cross_line(ps, along_u, x, v0 if along_u else u0, v1 if along_u else u1)
        var mu: Float32; var mv: Float32; var tm: Float32
        if y == Float32(3.0e38):   # numerically missed: split the distance instead
            tm = (at + bt) * Float32(0.5)
            var uv = tri.clamp_uv(invert(tri, o + d * tm))
            mu = uv[0]; mv = uv[1]
        else:
            mu = x if along_u else y; mv = y if along_u else x
            tm = _ray_t(tri, o, d, mu, mv)
            if not (tm >= at and tm <= bt): tm = (at + bt) * Float32(0.5)   # NaN or off the piece
        if sp + 2 <= STACK:   # the back half below, the front half on top
            st[unsafe_offset=6 * sp] = mu; st[unsafe_offset=6 * sp + 1] = mv; st[unsafe_offset=6 * sp + 2] = tm
            st[unsafe_offset=6 * sp + 3] = bu; st[unsafe_offset=6 * sp + 4] = bv; st[unsafe_offset=6 * sp + 5] = bt
            st[unsafe_offset=6 * sp + 6] = au; st[unsafe_offset=6 * sp + 7] = av; st[unsafe_offset=6 * sp + 8] = at
            st[unsafe_offset=6 * sp + 9] = mu; st[unsafe_offset=6 * sp + 10] = mv; st[unsafe_offset=6 * sp + 11] = tm
            sp += 2
        else:
            _march(tri, rm, o, d, u0, u1, v0, v1, best)
    return best


@always_inline
def intersect_displaced_brute(tri: DisplacedTri, rm: Rmip, o: Vec3f, d: Vec3f, tmax: Float32) -> DisplacedHit:
    """The same micro-triangles, all of them: the reference the traversal must reproduce."""
    var best = DisplacedHit(False, tmax, 0, 0, Vec3f(0), 0, 0)
    var u0 = min(min(tri.t[0], tri.t[2]), tri.t[4]); var u1 = max(max(tri.t[0], tri.t[2]), tri.t[4])
    var v0 = min(min(tri.t[1], tri.t[3]), tri.t[5]); var v1 = max(max(tri.t[1], tri.t[3]), tri.t[5])
    _march(tri, rm, o, d, u0 - Float32(1), u1, v0 - Float32(1), v1, best)
    return best


@always_inline
def _corner_normal(tri: DisplacedTri, rm: Rmip, i: Int, j: Int) -> Vec3f:
    var su = tri.surface(rm, i + 1, j) - tri.surface(rm, i - 1, j)
    var sv = tri.surface(rm, i, j + 1) - tri.surface(rm, i, j - 1)
    var n = cross(su, sv)
    return n if dot(n, tri.n(Float32(i) - tri.tc[0], Float32(j) - tri.tc[1])) >= Float32(0) else -n


@always_inline
def displaced_normal(tri: DisplacedTri, rm: Rmip, u: Float32, v: Float32) -> Vec3f:
    """The smooth shading normal at local (u, v): central-difference normals at the texel's corners, interpolated,
    as pbrt recomputes vertex normals after displacing; unit, on the side of N."""
    var x = u + tri.tc[0]; var y = v + tri.tc[1]
    var i = Int(floor(x)); var j = Int(floor(y))
    var fx = x - Float32(i); var fy = y - Float32(j)
    var n = _corner_normal(tri, rm, i, j).normalize() * ((Float32(1) - fx) * (Float32(1) - fy)) \
          + _corner_normal(tri, rm, i + 1, j).normalize() * (fx * (Float32(1) - fy)) \
          + _corner_normal(tri, rm, i, j + 1).normalize() * ((Float32(1) - fx) * fy) \
          + _corner_normal(tri, rm, i + 1, j + 1).normalize() * (fx * fy)
    return n.normalize()


@always_inline
def displaced_hit_normal(disp: Pointer[DisplacedMesh, MutUntrackedOrigin], base_vidx: Int, b1: Float32, b2: Float32) -> Vec3f:
    """The smooth normal of a displaced triangle's hit at base barycentrics (b1, b2), on its winding's side;
    zero for a triangle that stayed plain. It serves as geometric and shading normal alike."""
    var dm = disp[unsafe_offset=0]
    var tri = dm.tris[unsafe_offset=base_vidx // 3]
    if not tri.ok():
        return Vec3f(0)
    var u = tri.t[0] + b1 * (tri.t[2] - tri.t[0]) + b2 * (tri.t[4] - tri.t[0])
    var v = tri.t[1] + b1 * (tri.t[3] - tri.t[1]) + b2 * (tri.t[5] - tri.t[1])
    var ns = displaced_normal(tri, dm.rm, u, v)
    var p0 = tri.p(tri.t[0], tri.t[1])
    var w = cross(tri.p(tri.t[2], tri.t[3]) - p0, tri.p(tri.t[4], tri.t[5]) - p0)
    return ns if dot(ns, w) >= Float32(0) else -ns
