# Displaced Catmull-Clark surfaces traced without tessellation: Binder and Keller's stackless patch subdivision, with each
# sub-patch's box widened by its height interval (per-face planes + residuals, Xu et al. 2026; src/oiio/disp.cc) times the range of its
# unit normal (interval arithmetic over the derivative nets, after Munkberg et al. 2010). Gregory patches are exact: their
# rational interior points become boxes over the sub-domain's blend range (Binder and Keller 3.2). Leaves are boxes no
# larger than the camera's footprint at them, so every ray sees the same surface and neighbouring depths cannot open cracks.
from std.bit import count_leading_zeros
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import max, min, sqrt, abs, floor, ceil, log2, pow
from gonzales.core.geometry import Vec3f, cross, dot
from gonzales.scene.shapes.patches import BezierPatch, patch_eval, _crop, _blossom, _domain, PATCH_MAX_DEPTH

comptime _U8 = Pointer[UInt8, MutUntrackedOrigin]
comptime _I32 = Pointer[Int32, MutUntrackedOrigin]
comptime _F32 = Pointer[Float32, MutUntrackedOrigin]
comptime NODE_B = 6   # bytes a height node (src/oiio/disp.cc)


comptime LEVELS = 7   # sub-square levels 0..6: 1 x 1 .. 64 x 64 a patch


@fieldwise_init
struct Levels(TrivialRegisterPassable):
    """Patches sorted by level l (2^l x 2^l sub-squares); per level its first patch, primitive and corner grid float.
    Entries past the last level hold the totals."""
    var patch: SIMD[DType.int32, 8]
    var prim: SIMD[DType.int32, 8]
    var corner: SIMD[DType.int32, 8]


def make_levels(lev: List[Int]) -> Levels:
    """Levels of patches already sorted by their level lev[p]."""
    var c = List[Int](length=8, fill=0)
    for l in lev:
        c[l] += 1
    var pa = SIMD[DType.int32, 8](0); var pr = SIMD[DType.int32, 8](0); var co = SIMD[DType.int32, 8](0)
    for l in range(1, 8):
        var n = 1 << (l - 1)
        pa[l] = pa[l - 1] + Int32(c[l - 1]); pr[l] = pr[l - 1] + Int32(c[l - 1] * n * n); co[l] = co[l - 1] + Int32(c[l - 1] * corner_words(n))
    return Levels(pa, pr, co)


@always_inline
def _class_of(v: SIMD[DType.int32, 8], x: Int) -> Int:
    var c = 0
    comptime for l in range(1, LEVELS):
        if Int(v[l]) <= x:
            c = l
    return c


@always_inline
def prim_patch(d: DispSurface, prim: Int) -> Tuple[Int, Int, Int]:
    """Primitive prim: its patch, sub-square k (row * n + column) and level."""
    var l = _class_of(d.lv[].prim, prim); var loc = prim - Int(d.lv[].prim[l])
    return (Int(d.lv[].patch[l]) + (loc >> (2 * l)), loc & ((1 << (2 * l)) - 1), l)


@always_inline
def patch_level(d: DispSurface, pi: Int) -> Int:
    return _class_of(d.lv[].patch, pi)


@always_inline
def corner_off(lv: Levels, pi: Int) -> Int:
    var l = _class_of(lv.patch, pi)
    return Int(lv.corner[l]) + (pi - Int(lv.patch[l])) * corner_words(1 << l)


@always_inline
def prim_total(d: DispSurface) -> Int:
    return Int(d.lv[].prim[LEVELS])


@fieldwise_init
struct DispSurface(TrivialRegisterPassable, DevicePassable):
    """Patches with their Ptex face and square (u0, v0, size), the faces' height nodes, the height scale and the camera."""
    comptime device_type: AnyType = Self

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "DispSurface"

    var patches: Pointer[BezierPatch, MutUntrackedOrigin]
    var sq: _F32
    var face: _I32
    var hdr: _I32        # per face: node offset, log2 width, log2 height
    var pyr: _U8         # per face from 4 x 4-texel cells up: (centre, half u slope / 2, half v slope / 2, residual, min, max) bytes
    var scale: Float32   # world units per height byte
    var cam: Vec3f
    var pixel: Float32   # footprint per unit distance (pixel angle)
    var chain_off: _I32  # per face: its bordered height chain in `chain` (src/oiio/disp.cc)
    var chain: _U8
    var fm: _F32         # per patch: the Fm interior points (1,1), (2,1), (2,2), (1,2), 12 floats; patches hold Fp there
    var greg: _U8        # per patch: 1 for a Gregory patch, 2 for an end cap fitted as Bezier
    var lv: Pointer[Levels, MutUntrackedOrigin]   # the BVH's primitives: each patch's 2^l x 2^l sub-squares (behind a pointer: fewer live registers in the walk)
    var leaf_mode: Int32   # 0: box leaves no larger than the footprint; 1: exact bilinear leaves at a texel or a pixel; 2: at a texel only;
                           # 3: as 1, cells walked in the sub-square's affine (s, t, h) frame; 4: as 1, every box cropped from the patch
                           # 5 (6: at a texel only): the sub-square's base bilinear through its limit corners
    var corners: _F32    # per patch the sub-square corners on its (n + 1)^2 grid (corner_grid): limit point, unit normal


@always_inline
def _texel(d: DispSurface, off: Int, w: Int, x: Int, y: Int) -> Float32:
    """Height byte (x, y) of a face's finest level, read from its bordered chain."""
    return Float32(Int(d.chain[unsafe_offset=off + (y + 1) * (w + 2) + x + 1]))


def face_height(d: DispSurface, f: Int, u: Float32, v: Float32) -> Float32:
    """The bilinear height (in bytes) at (u, v) of face f's finest level, its border texels included."""
    var off = Int(d.chain_off[unsafe_offset=f]); var w = 1 << Int(d.hdr[unsafe_offset=3 * f + 1]); var h = 1 << Int(d.hdr[unsafe_offset=3 * f + 2])
    var x = min(max(u * Float32(w) - Float32(0.5), Float32(-0.5)), Float32(w) - Float32(0.5))   # into the border: faces agree on edges
    var y = min(max(v * Float32(h) - Float32(0.5), Float32(-0.5)), Float32(h) - Float32(0.5))
    var x0 = min(Int(floor(x)), w - 1); var y0 = min(Int(floor(y)), h - 1)
    var x1 = x0 + 1; var y1 = y0 + 1
    var fx = x - Float32(x0); var fy = y - Float32(y0)
    var a = _texel(d, off, w, x0, y0) * (Float32(1) - fx) + _texel(d, off, w, x1, y0) * fx
    var b = _texel(d, off, w, x0, y1) * (Float32(1) - fx) + _texel(d, off, w, x1, y1) * fx
    return a * (Float32(1) - fy) + b * fy


@always_inline
def _plane_range(d: DispSurface, node: Int, X0: Float32, X1: Float32, Y0: Float32, Y1: Float32,
                 xc: Float32, yc: Float32, hw: Float32, hh: Float32) -> Tuple[Float32, Float32]:
    """A node's plane +- residual over the texel-position rectangle [X0, X1] x [Y0, Y1] (inside its cell)."""
    var a = Float32(Int(d.pyr[unsafe_offset=node]))
    var bx = Float32(2) * d.pyr[unsafe_offset=node + 1].cast[DType.int8]().cast[DType.float32]()
    var by = Float32(2) * d.pyr[unsafe_offset=node + 2].cast[DType.int8]().cast[DType.float32]()
    var r = Float32(Int(d.pyr[unsafe_offset=node + 3]))
    var ex0 = bx * (X0 - xc) / hw; var ex1 = bx * (X1 - xc) / hw
    var ey0 = by * (Y0 - yc) / hh; var ey1 = by * (Y1 - yc) / hh
    var mn = Float32(Int(d.pyr[unsafe_offset=node + 4])); var mx = Float32(Int(d.pyr[unsafe_offset=node + 5]))   # the cell's exact range
    return (max(a + min(ex0, ex1) + min(ey0, ey1) - r, mn), min(a + max(ex0, ex1) + max(ey0, ey1) + r, mx))


def height_range(d: DispSurface, f: Int, u0: Float32, u1: Float32, v0: Float32, v1: Float32) -> Tuple[Float32, Float32]:
    """Bounds (in bytes) of the bilinear heights over [u0, u1] x [v0, v1] of face f: exact up to 4 texels, else the planes and
    residuals of the covering nodes (src/oiio/disp.cc)."""
    var ul = Int(d.hdr[unsafe_offset=3 * f + 1]); var vl = Int(d.hdr[unsafe_offset=3 * f + 2])
    var w = 1 << ul; var h = 1 << vl
    # Texels whose cells touch the rectangle, border texels included.
    var x0 = min(max(Int(floor(u0 * Float32(w) - Float32(0.5))), -1), w)
    var x1 = min(max(Int(floor(u1 * Float32(w) - Float32(0.5))) + 1, -1), w)
    var y0 = min(max(Int(floor(v0 * Float32(h) - Float32(0.5))), -1), h)
    var y1 = min(max(Int(floor(v1 * Float32(h) - Float32(0.5))) + 1, -1), h)
    var off = Int(d.chain_off[unsafe_offset=f])
    if x1 - x0 <= 4 and y1 - y0 <= 4:   # the covering texels: bilinear heights stay inside their four texels' range
        var lo = Float32(1e30); var hi = Float32(-1e30)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                var t = _texel(d, off, w, min(max(x, -1), w), min(max(y, -1), h))
                lo = min(lo, t); hi = max(hi, t)
        return (lo, hi)
    var edge = _border_range(d, off, w, h, x0, x1, y0, y1)
    var X0 = u0 * Float32(w); var X1 = u1 * Float32(w); var Y0 = v0 * Float32(h); var Y1 = v1 * Float32(h)
    var node = Int(d.hdr[unsafe_offset=3 * f])
    var k = 2
    var cw = 1; var ch = 1; var tw = Float32(1); var th = Float32(1)
    var cx0 = 0; var cx1 = 0; var cy0 = 0; var cy1 = 0
    while True:
        cw = max(w >> k, 1); ch = max(h >> k, 1)
        tw = Float32(w // cw); th = Float32(h // ch)
        cx0 = min(Int(X0 / tw), cw - 1); cx1 = max(min(Int(ceil(X1 / tw)) - 1, cw - 1), cx0)
        cy0 = min(Int(Y0 / th), ch - 1); cy1 = max(min(Int(ceil(Y1 / th)) - 1, ch - 1), cy0)
        if (cx1 - cx0 <= 1 and cy1 - cy0 <= 1) or (cw == 1 and ch == 1):
            break
        node += NODE_B * cw * ch
        k += 1
    var lo = edge[0]; var hi = edge[1]
    for cy in range(cy0, cy1 + 1):
        for cx in range(cx0, cx1 + 1):
            var bx0 = Float32(cx) * tw; var by0 = Float32(cy) * th
            var r = _plane_range(d, node + NODE_B * (cy * cw + cx), max(X0, bx0), min(X1, bx0 + tw), max(Y0, by0), min(Y1, by0 + th),
                                 bx0 + Float32(0.5) * tw, by0 + Float32(0.5) * th, Float32(0.5) * tw, Float32(0.5) * th)
            lo = min(lo, r[0]); hi = max(hi, r[1])
    return (lo, hi)


def _border_range(d: DispSurface, off: Int, w: Int, h: Int, x0: Int, x1: Int, y0: Int, y1: Int) -> Tuple[Float32, Float32]:
    """The border texels (from the neighbouring faces) next to a texel range that reaches the face's edge; empty if none."""
    var lo = Float32(1e30); var hi = Float32(-1e30)
    if x0 < 0 or x1 >= w:
        for y in range(y0, y1 + 1):
            if x0 < 0:
                var t = _texel(d, off, w, -1, y); lo = min(lo, t); hi = max(hi, t)
            if x1 >= w:
                var t = _texel(d, off, w, w, y); lo = min(lo, t); hi = max(hi, t)
    if y0 < 0 or y1 >= h:
        for x in range(x0, x1 + 1):
            if y0 < 0:
                var t = _texel(d, off, w, x, -1); lo = min(lo, t); hi = max(hi, t)
            if y1 >= h:
                var t = _texel(d, off, w, x, h); lo = min(lo, t); hi = max(hi, t)
    return (lo, hi)


@always_inline
def _imul(a0: Float32, a1: Float32, b0: Float32, b1: Float32) -> Tuple[Float32, Float32]:
    var p = a0 * b0; var q = a0 * b1; var r = a1 * b0; var s = a1 * b1
    return (min(min(p, q), min(r, s)), max(max(p, q), max(r, s)))


@always_inline
def _unit_comp(a: Float32, b: Float32, m: Float32, big: Float32) -> Tuple[Float32, Float32]:
    """Bounds of N_k / |N| for N_k in [a, b] and |N| in [m, big]."""
    return (max(min(a / m, a / big), Float32(-1)), min(max(b / m, b / big), Float32(1)))


def _unit_from_boxes(ulo: Vec3f, uhi: Vec3f, vlo: Vec3f, vhi: Vec3f) -> Tuple[Vec3f, Vec3f]:
    """Component bounds of N / |N|, N = Pu x Pv, for Pu in [ulo, uhi] and Pv in [vlo, vhi]."""
    var a1 = _imul(ulo.y, uhi.y, vlo.z, vhi.z); var a2 = _imul(ulo.z, uhi.z, vlo.y, vhi.y)
    var b1 = _imul(ulo.z, uhi.z, vlo.x, vhi.x); var b2 = _imul(ulo.x, uhi.x, vlo.z, vhi.z)
    var c1 = _imul(ulo.x, uhi.x, vlo.y, vhi.y); var c2 = _imul(ulo.y, uhi.y, vlo.x, vhi.x)
    var lo = Vec3f(a1[0] - a2[1], b1[0] - b2[1], c1[0] - c2[1])
    var hi = Vec3f(a1[1] - a2[0], b1[1] - b2[0], c1[1] - c2[0])
    var nx = max(max(lo.x, -hi.x), Float32(0)); var ny = max(max(lo.y, -hi.y), Float32(0)); var nz = max(max(lo.z, -hi.z), Float32(0))
    var m = sqrt(nx * nx + ny * ny + nz * nz)    # |N| is at least the box's distance from the origin
    if m <= Float32(0):
        return (Vec3f(-1, -1, -1), Vec3f(1, 1, 1))
    var fx = max(abs(lo.x), abs(hi.x)); var fy = max(abs(lo.y), abs(hi.y)); var fz = max(abs(lo.z), abs(hi.z))
    var big = sqrt(fx * fx + fy * fy + fz * fz)
    var x = _unit_comp(lo.x, hi.x, m, big); var y = _unit_comp(lo.y, hi.y, m, big); var z = _unit_comp(lo.z, hi.z, m, big)
    return (Vec3f(x[0], y[0], z[0]), Vec3f(x[1], y[1], z[1]))


comptime _SLOTS = SIMD[DType.int32, 4](5, 6, 10, 9)   # Bezier slots of the Gregory interior pairs (OSD's order)


@always_inline
def _ratio_range(alo: Float32, ahi: Float32, blo: Float32, bhi: Float32) -> Tuple[Float32, Float32]:
    """Range of a / (a + b) for a in [alo, ahi], b in [blo, bhi], a, b >= 0 (OSD's blend weights)."""
    var lo = alo / (alo + bhi) if alo + bhi > Float32(0) else Float32(0)
    var hi = ahi / (ahi + blo) if ahi + blo > Float32(0) else Float32(1)
    return (lo, hi)


@always_inline
def _lam_range(k: Int, s0: Float32, s1: Float32, t0: Float32, t1: Float32) -> Tuple[Float32, Float32]:
    """Range over [s0, s1] x [t0, t1] of the weight of Fp in interior slot k: s/(s+t), t/(1-s+t), (1-s)/(2-s-t), (1-t)/(1+s-t)."""
    if k == 0:
        return _ratio_range(s0, s1, t0, t1)
    if k == 1:
        return _ratio_range(t0, t1, Float32(1) - s1, Float32(1) - s0)
    if k == 2:
        return _ratio_range(Float32(1) - s1, Float32(1) - s0, Float32(1) - t1, Float32(1) - t0)
    return _ratio_range(Float32(1) - t1, Float32(1) - t0, s0, s1)


@always_inline
def _lam(k: Int, s: Float32, t: Float32) -> Float32:
    var r = _lam_range(k, s, s, t, t)
    return r[0]


@fieldwise_init
struct INet(TrivialRegisterPassable):
    """A control net with interval points: lower and upper corners per point."""
    var lo: BezierPatch
    var hi: BezierPatch


def _inet(d: DispSurface, pi: Int, o: BezierPatch, org: Vec3f, s0: Float32, s1: Float32, t0: Float32, t1: Float32) -> INet:
    """Patch pi (anchored at org, p = o) over [s0, s1] x [t0, t1] as an uncropped interval net: a Gregory patch's interior
    points range over Fm + w (Fp - Fm) for the sub-domain's weights w."""
    if d.greg[unsafe_offset=pi] != 1:
        return INet(o, o)
    var lo = o; var hi = o
    for k in range(4):
        var sl = Int(_SLOTS[k])
        var w = _lam_range(k, s0, s1, t0, t1)
        var mx = d.fm[unsafe_offset=12 * pi + 3 * k] - org.x; var my = d.fm[unsafe_offset=12 * pi + 3 * k + 1] - org.y; var mz = d.fm[unsafe_offset=12 * pi + 3 * k + 2] - org.z
        var ax = mx + (o.x[sl] - mx) * w[0]; var bx = mx + (o.x[sl] - mx) * w[1]
        var ay = my + (o.y[sl] - my) * w[0]; var by = my + (o.y[sl] - my) * w[1]
        var az = mz + (o.z[sl] - mz) * w[0]; var bz = mz + (o.z[sl] - mz) * w[1]
        lo.x[sl] = min(ax, bx); hi.x[sl] = max(ax, bx)
        lo.y[sl] = min(ay, by); hi.y[sl] = max(ay, by)
        lo.z[sl] = min(az, bz); hi.z[sl] = max(az, bz)
    return INet(lo, hi)


@always_inline
def _blossom2(q0: Float32, q1: Float32, q2: Float32, a: Float32, b: Float32) -> Float32:
    var c0 = q0 + (q1 - q0) * a; var c1 = q1 + (q2 - q1) * a
    return c0 + (c1 - c0) * b


@always_inline
def _range_crop_du(p: SIMD[DType.float32, 16], q: SIMD[DType.float32, 16], s0: Float32, s1: Float32, t0: Float32, t1: Float32) -> Tuple[Float32, Float32]:
    """Range over the crop of the u-difference net d_ij = p_(i+1)j - q_ij (3 x 4, quadratic in u, cubic in v)."""
    var r = SIMD[DType.float32, 16](0)   # rows j, columns 0..2 cropped along u
    for j in range(4):
        var d0 = p[j * 4 + 1] - q[j * 4]; var d1 = p[j * 4 + 2] - q[j * 4 + 1]; var d2 = p[j * 4 + 3] - q[j * 4 + 2]
        r[j * 4] = _blossom2(d0, d1, d2, s0, s0); r[j * 4 + 1] = _blossom2(d0, d1, d2, s0, s1); r[j * 4 + 2] = _blossom2(d0, d1, d2, s1, s1)
    var lo = Float32(1e30); var hi = Float32(-1e30)
    for i in range(3):
        var c0 = r[i]; var c1 = r[4 + i]; var c2 = r[8 + i]; var c3 = r[12 + i]
        var e0 = _blossom(c0, c1, c2, c3, t0, t0, t0); var e1 = _blossom(c0, c1, c2, c3, t0, t0, t1)
        var e2 = _blossom(c0, c1, c2, c3, t0, t1, t1); var e3 = _blossom(c0, c1, c2, c3, t1, t1, t1)
        lo = min(lo, min(min(e0, e1), min(e2, e3))); hi = max(hi, max(max(e0, e1), max(e2, e3)))
    return (lo, hi)


@always_inline
def _range_crop_dv(p: SIMD[DType.float32, 16], q: SIMD[DType.float32, 16], s0: Float32, s1: Float32, t0: Float32, t1: Float32) -> Tuple[Float32, Float32]:
    """Range over the crop of the v-difference net d_ij = p_i(j+1) - q_ij (4 x 3, cubic in u, quadratic in v)."""
    var r = SIMD[DType.float32, 16](0)   # columns i, rows 0..2 cropped along v
    for i in range(4):
        var d0 = p[4 + i] - q[i]; var d1 = p[8 + i] - q[4 + i]; var d2 = p[12 + i] - q[8 + i]
        r[i] = _blossom2(d0, d1, d2, t0, t0); r[4 + i] = _blossom2(d0, d1, d2, t0, t1); r[8 + i] = _blossom2(d0, d1, d2, t1, t1)
    var lo = Float32(1e30); var hi = Float32(-1e30)
    for j in range(3):
        var c0 = r[j * 4]; var c1 = r[j * 4 + 1]; var c2 = r[j * 4 + 2]; var c3 = r[j * 4 + 3]
        var e0 = _blossom(c0, c1, c2, c3, s0, s0, s0); var e1 = _blossom(c0, c1, c2, c3, s0, s0, s1)
        var e2 = _blossom(c0, c1, c2, c3, s0, s1, s1); var e3 = _blossom(c0, c1, c2, c3, s1, s1, s1)
        lo = min(lo, min(min(e0, e1), min(e2, e3))); hi = max(hi, max(max(e0, e1), max(e2, e3)))
    return (lo, hi)


def _net_normal_range(n: INet, exact: Bool, s0: Float32, s1: Float32, t0: Float32, t1: Float32) -> Tuple[Vec3f, Vec3f]:
    """Unit-normal bounds over the sub-domain from the difference nets of the uncropped interval net, cropped by blossoming:
    no cancellation however small the crop. Upper differences pair upper with lower points and vice versa."""
    if exact:   # a Bezier patch: one net
        var ex = _range_crop_du(n.lo.x, n.lo.x, s0, s1, t0, t1); var ey = _range_crop_du(n.lo.y, n.lo.y, s0, s1, t0, t1)
        var ez = _range_crop_du(n.lo.z, n.lo.z, s0, s1, t0, t1)
        var fx = _range_crop_dv(n.lo.x, n.lo.x, s0, s1, t0, t1); var fy = _range_crop_dv(n.lo.y, n.lo.y, s0, s1, t0, t1)
        var fz = _range_crop_dv(n.lo.z, n.lo.z, s0, s1, t0, t1)
        return _unit_from_boxes(Vec3f(ex[0], ey[0], ez[0]), Vec3f(ex[1], ey[1], ez[1]), Vec3f(fx[0], fy[0], fz[0]), Vec3f(fx[1], fy[1], fz[1]))
    var ux = (_range_crop_du(n.lo.x, n.hi.x, s0, s1, t0, t1)[0], _range_crop_du(n.hi.x, n.lo.x, s0, s1, t0, t1)[1])
    var uy = (_range_crop_du(n.lo.y, n.hi.y, s0, s1, t0, t1)[0], _range_crop_du(n.hi.y, n.lo.y, s0, s1, t0, t1)[1])
    var uz = (_range_crop_du(n.lo.z, n.hi.z, s0, s1, t0, t1)[0], _range_crop_du(n.hi.z, n.lo.z, s0, s1, t0, t1)[1])
    var vx = (_range_crop_dv(n.lo.x, n.hi.x, s0, s1, t0, t1)[0], _range_crop_dv(n.hi.x, n.lo.x, s0, s1, t0, t1)[1])
    var vy = (_range_crop_dv(n.lo.y, n.hi.y, s0, s1, t0, t1)[0], _range_crop_dv(n.hi.y, n.lo.y, s0, s1, t0, t1)[1])
    var vz = (_range_crop_dv(n.lo.z, n.hi.z, s0, s1, t0, t1)[0], _range_crop_dv(n.hi.z, n.lo.z, s0, s1, t0, t1)[1])
    return _unit_from_boxes(Vec3f(ux[0], uy[0], uz[0]), Vec3f(ux[1], uy[1], uz[1]), Vec3f(vx[0], vy[0], vz[0]), Vec3f(vx[1], vy[1], vz[1]))


def patch_at(d: DispSurface, pi: Int, s: Float32, t: Float32) -> BezierPatch:
    """The Bezier patch that evaluates like patch pi at (s, t): a Gregory patch's interior points blended for (s, t)."""
    var p = d.patches[unsafe_offset=pi]
    if d.greg[unsafe_offset=pi] != 1:
        return p
    for k in range(4):
        var sl = Int(_SLOTS[k])
        var w = _lam(k, s, t)
        var mx = d.fm[unsafe_offset=12 * pi + 3 * k]; var my = d.fm[unsafe_offset=12 * pi + 3 * k + 1]; var mz = d.fm[unsafe_offset=12 * pi + 3 * k + 2]
        p.x[sl] = mx + (p.x[sl] - mx) * w; p.y[sl] = my + (p.y[sl] - my) * w; p.z[sl] = mz + (p.z[sl] - mz) * w
    return p


def displaced_box(d: DispSurface, pi: Int, o: BezierPatch, org: Vec3f, s0: Float32, s1: Float32, t0: Float32, t1: Float32,
                  parent_lo: Vec3f = Vec3f(-1, -1, -1), parent_hi: Vec3f = Vec3f(1, 1, 1)) -> Tuple[Vec3f, Vec3f, Vec3f, Vec3f, Float32]:
    """The box of patch pi's displaced surface over [s0, s1] x [t0, t1] (o: the patch anchored at org), and its unit-normal
    range (within the parent's: both are conservative), and the undisplaced sub-patch's extent. Everything is cropped from the
    original nets."""
    var n = _inet(d, pi, o, org, s0, s1, t0, t1)
    var cl = _crop(n.lo, s0, s1, t0, t1)
    var ch = cl if d.greg[unsafe_offset=pi] != 1 else _crop(n.hi, s0, s1, t0, t1)
    var lo = Vec3f(cl.x.reduce_min(), cl.y.reduce_min(), cl.z.reduce_min())
    var hi = Vec3f(ch.x.reduce_max(), ch.y.reduce_max(), ch.z.reduce_max())
    var base = max(max(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z)
    if d.scale <= Float32(0):
        return (lo, hi, parent_lo, parent_hi, base)
    var u0 = d.sq[unsafe_offset=3 * pi]; var v0 = d.sq[unsafe_offset=3 * pi + 1]; var sz = d.sq[unsafe_offset=3 * pi + 2]
    var hr = height_range(d, Int(d.face[unsafe_offset=pi]), u0 + s0 * sz, u0 + s1 * sz, v0 + t0 * sz, v0 + t1 * sz)
    var h0 = hr[0] * d.scale; var h1 = hr[1] * d.scale
    var nc = _net_normal_range(n, d.greg[unsafe_offset=pi] != 1, s0, s1, t0, t1)
    var nlo = Vec3f(max(nc[0].x, parent_lo.x), max(nc[0].y, parent_lo.y), max(nc[0].z, parent_lo.z))
    var nhi = Vec3f(min(nc[1].x, parent_hi.x), min(nc[1].y, parent_hi.y), min(nc[1].z, parent_hi.z))
    var ox = _imul(h0, h1, nlo.x, nhi.x); var oy = _imul(h0, h1, nlo.y, nhi.y); var oz = _imul(h0, h1, nlo.z, nhi.z)
    return (Vec3f(lo.x + ox[0], lo.y + oy[0], lo.z + oz[0]), Vec3f(hi.x + ox[1], hi.y + oy[1], hi.z + oz[1]), nlo, nhi, base)


def patch_box(d: DispSurface, pi: Int) -> Tuple[Vec3f, Vec3f]:
    """The whole displaced patch's box in world space (for the BVH)."""
    var b = displaced_box(d, pi, d.patches[unsafe_offset=pi], Vec3f(0, 0, 0), Float32(0), Float32(1), Float32(0), Float32(1))
    return (b[0], b[1])


@always_inline
def _slab(lo: Vec3f, hi: Vec3f, inv: Vec3f, t_max: Float32) -> Tuple[Bool, Float32]:
    var size = max(max(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z)
    var pad = size * Float32(1e-5) + Float32(1e-6)
    var ax = (lo.x - pad) * inv.x; var bx = (hi.x + pad) * inv.x
    var ay = (lo.y - pad) * inv.y; var by = (hi.y + pad) * inv.y
    var az = (lo.z - pad) * inv.z; var bz = (hi.z + pad) * inv.z
    var t0 = max(max(min(ax, bx), min(ay, by)), max(min(az, bz), Float32(0)))
    var t1 = min(min(max(ax, bx), max(ay, by)), min(max(az, bz), t_max))
    return (t0 <= t1, t0)


@always_inline
def _leaf_size(lo: Vec3f, hi: Vec3f, cam: Vec3f, pixel: Float32) -> Tuple[Float32, Float32]:
    """(the box's size, the camera footprint at it): the same for every ray."""
    var c = (lo + hi) * Float32(0.5) - cam
    var e = hi - lo
    var r = Float32(0.5) * sqrt(dot(e, e))
    var dist = max(sqrt(dot(c, c)) - r, r * Float32(0.1) + Float32(1e-4))
    return (max(max(e.x, e.y), e.z), dist * pixel)


@always_inline
def sub_square(la: Int, k: Int) -> Tuple[Float32, Float32, Float32, UInt64]:
    """Sub-square k of a level-la patch (k = row * n + column): its corner (u0, v0), its size, and the halving path that reaches it."""
    var n = 1 << la
    var ku = k % n; var kv = k // n
    var path = UInt64(0)
    for i in range(la):   # even depths split u, odd ones v, the coarsest first
        path |= UInt64((ku >> (la - 1 - i)) & 1) << UInt64(2 * i)
        path |= UInt64((kv >> (la - 1 - i)) & 1) << UInt64(2 * i + 1)
    var sz = Float32(1) / Float32(n)
    return (Float32(ku) * sz, Float32(kv) * sz, sz, path)


def sub_box(d: DispSurface, prim: Int) -> Tuple[Vec3f, Vec3f]:
    """The world box of primitive prim: the bilinear base's box in leaf modes 5 and 6 (what the walk tests), else the bicubic's."""
    var pk = prim_patch(d, prim); var pi = pk[0]; var sq = sub_square(pk[2], pk[1])
    if d.leaf_mode >= 5:
        var bb = _bl_box(d, pi, Int(d.face[unsafe_offset=pi]), _bl_frame(d, pi, pk[1], Vec3f(0, 0, 0)), sq[0], sq[1], Float32(1) / sq[2],
                         (sq[0], sq[1], sq[2], sq[2]))
        return (bb[0], bb[1])
    var b = displaced_box(d, pi, d.patches[unsafe_offset=pi], Vec3f(0, 0, 0), sq[0], sq[0] + sq[2], sq[1], sq[1] + sq[2])
    return (b[0], b[1])


@always_inline
def _tri_hit(a: Vec3f, b: Vec3f, c: Vec3f, dir: Vec3f, t_max: Float32) -> Tuple[Bool, Float32, Float32, Float32]:
    """Ray (from the origin) against triangle a b c (Moller-Trumbore, a hair of overlap): (hit, t, b1, b2)."""
    var e1 = b - a; var e2 = c - a
    var pv = cross(dir, e2)
    var det = dot(e1, pv)
    if det == Float32(0):
        return (False, t_max, Float32(0), Float32(0))
    var inv = Float32(1) / det
    var tv = -a
    var u = dot(tv, pv) * inv
    var qv = cross(tv, e1)
    var v = dot(dir, qv) * inv
    var t = dot(e2, qv) * inv
    comptime e = Float32(1e-3)
    if u < -e or v < -e or u + v > Float32(1) + e or t <= Float32(0) or t >= t_max:
        return (False, t_max, Float32(0), Float32(0))
    return (True, t, u, v)


@always_inline
def _bilinear_hit(q00: Vec3f, q10: Vec3f, q11: Vec3f, q01: Vec3f, dir: Vec3f, t_max: Float32) -> Tuple[Bool, Float32, Float32, Float32]:
    """Ray (from the origin) against the bilinear patch q00 q10 q11 q01 (Reshetov, Cool Patches, Ray Tracing Gems 2019): (hit, t, u, v)."""
    var e10 = q10 - q00; var e11 = q11 - q10; var e00 = q01 - q00
    var qn = cross(q10 - q00, q01 - q11)
    var a = dot(cross(q00, dir), e00)
    var c = dot(qn, dir)
    var b = dot(cross(q10, dir), e11) - a - c
    var det = b * b - Float32(4) * a * c
    var t = t_max; var u = Float32(0); var v = Float32(0); var hit = False
    if det < Float32(0):
        return (False, t, u, v)
    det = sqrt(det)
    var u1: Float32; var u2: Float32
    if c == Float32(0):
        u1 = -a / b; u2 = Float32(-1)
    else:
        u1 = (-b - (det if b >= Float32(0) else -det)) * Float32(0.5)
        u2 = a / u1 if u1 != Float32(0) else Float32(-1)
        u1 = u1 / c
    comptime e = Float32(1e-3)   # neighbouring leaves overlap a hair: rounding must not let a ray slip between them
    for r in range(2):
        var ur = u1 if r == 0 else u2
        if ur < -e or ur > Float32(1) + e:
            continue
        var pa = q00 + e10 * ur
        var pb = e00 + (e11 - e00) * ur
        var n = cross(dir, pb)
        var dd = dot(n, n)
        if dd <= Float32(0):
            continue
        n = cross(n, pa)
        var tr = dot(n, pb) / dd
        var vr = dot(n, dir)
        if vr >= -e * dd and vr <= (Float32(1) + e) * dd and tr > Float32(0) and tr < t:
            t = tr; u = min(max(ur, Float32(0)), Float32(1)); v = min(max(vr / dd, Float32(0)), Float32(1)); hit = True
    return (hit, t, u, v)


def _leaf_hit(cp: Array[Vec3f, 8], s0: Float32, s1: Float32, t0: Float32, t1: Float32, dir: Vec3f,
              t_max: Float32) -> Tuple[Bool, Float32, Float32, Float32]:
    """The bilinear patch through the leaf's corners, plus edge fans through the midpoints, which close the T gap to a
    neighbour one level finer: (hit, t, s, t)."""
    var sm = s0 + Float32(0.5) * (s1 - s0); var tm = t0 + Float32(0.5) * (t1 - t0)
    var cs = SIMD[DType.float32, 8](s0, s1, s1, s0, sm, s1, sm, s0)
    var ct = SIMD[DType.float32, 8](t0, t0, t1, t1, t0, tm, t1, tm)
    var best = t_max; var hit = False; var hs = Float32(0); var ht = Float32(0)
    var bh = _bilinear_hit(cp[0], cp[1], cp[2], cp[3], dir, best)
    if bh[0]:
        best = bh[1]; hit = True; hs = s0 + bh[2] * (s1 - s0); ht = t0 + bh[3] * (t1 - t0)
    for k in range(4):
        var ka = k; var kb = (k + 1) % 4; var km = 4 + k
        var th = _tri_hit(cp[ka], cp[km], cp[kb], dir, best)
        if th[0]:
            best = th[1]; hit = True
            var w0 = Float32(1) - th[2] - th[3]
            hs = w0 * cs[ka] + th[2] * cs[km] + th[3] * cs[kb]; ht = w0 * ct[ka] + th[2] * ct[km] + th[3] * ct[kb]
    return (hit, best, hs, ht)


def _exact_leaf(d: DispSurface, pi: Int, dom: Tuple[Float32, Float32, Float32, Float32], org: Vec3f, dir: Vec3f,
                t_max: Float32) -> Tuple[Bool, Float32, Float32, Float32]:
    """The exact leaf: displaced corners and edge midpoints from global (s, t), so neighbouring leaves share them."""
    var s0 = dom[0]; var s1 = dom[0] + dom[2]; var t0 = dom[1]; var t1 = dom[1] + dom[3]
    var sm = s0 + Float32(0.5) * dom[2]; var tm = t0 + Float32(0.5) * dom[3]
    var cs = SIMD[DType.float32, 8](s0, s1, s1, s0, sm, s1, sm, s0)   # corners 00 10 11 01, then the edge midpoints
    var ct = SIMD[DType.float32, 8](t0, t0, t1, t1, t0, tm, t1, tm)
    var cp = Array[Vec3f, 8](fill=Vec3f(0, 0, 0))
    for k in range(8):
        cp[k] = displaced_point(d, pi, cs[k], ct[k]) - org
    return _leaf_hit(cp, s0, s1, t0, t1, dir, t_max)


comptime _A16 = SIMD[DType.float32, 16](0, 1.0 / 3, 2.0 / 3, 1, 0, 1.0 / 3, 2.0 / 3, 1, 0, 1.0 / 3, 2.0 / 3, 1, 0, 1.0 / 3, 2.0 / 3, 1)
comptime _B16 = SIMD[DType.float32, 16](0, 0, 0, 0, 1.0 / 3, 1.0 / 3, 1.0 / 3, 1.0 / 3, 2.0 / 3, 2.0 / 3, 2.0 / 3, 2.0 / 3, 1, 1, 1, 1)


@always_inline
def _bilerp[w: Int](q: SIMD[DType.float32, 4], a: SIMD[DType.float32, w], b: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """The bilinear map with corners q (00 10 11 01) at (a, b)."""
    return (q[0] + (q[1] - q[0]) * a) * (Float32(1) - b) + (q[3] + (q[2] - q[3]) * a) * b


@fieldwise_init
struct FineFrame(TrivialRegisterPassable):
    """A sub-square's base patch as the bilinear map through its corners (anchored at the ray origin) with the bicubic's
    largest deviation from it per axis, and the sub-square's unit-normal range: boxes inside it need no crop."""
    var qx: SIMD[DType.float32, 4]
    var qy: SIMD[DType.float32, 4]
    var qz: SIMD[DType.float32, 4]
    var eps: Vec3f
    var nlo: Vec3f
    var nhi: Vec3f


def _fine_frame(o: BezierPatch, s0: Float32, t0: Float32, size: Float32, nlo: Vec3f, nhi: Vec3f) -> FineFrame:
    var c = _crop(o, s0, s0 + size, t0, t0 + size)
    var qx = SIMD[DType.float32, 4](c.x[0], c.x[3], c.x[15], c.x[12])
    var qy = SIMD[DType.float32, 4](c.y[0], c.y[3], c.y[15], c.y[12])
    var qz = SIMD[DType.float32, 4](c.z[0], c.z[3], c.z[15], c.z[12])
    # The difference to the degree-elevated bilinear map is a Bezier patch: its control points bound it.
    var ex = abs(c.x - _bilerp[16](qx, _A16, _B16)).reduce_max(); var ey = abs(c.y - _bilerp[16](qy, _A16, _B16)).reduce_max()
    var ez = abs(c.z - _bilerp[16](qz, _A16, _B16)).reduce_max()
    var pad = Float32(1e-6) * (abs(qx).reduce_max() + abs(qy).reduce_max() + abs(qz).reduce_max()) + Float32(1e-7)
    return FineFrame(qx, qy, qz, Vec3f(ex + pad, ey + pad, ez + pad), nlo, nhi)


def _fine_box(d: DispSurface, pi: Int, fr: FineFrame, s0: Float32, t0: Float32, size: Float32,
              ds0: Float32, ds1: Float32, dt0: Float32, dt1: Float32) -> Tuple[Vec3f, Vec3f, Float32]:
    """The box over [ds0, ds1] x [dt0, dt1] (patch units) inside the frame's sub-square: the bilinear corners' box widened
    by the deviation, plus the height range times the frame's normal range; and the undisplaced extent."""
    var inv = Float32(1) / size
    var a = SIMD[DType.float32, 4](ds0, ds1, ds1, ds0) * inv - s0 * inv
    var b = SIMD[DType.float32, 4](dt0, dt0, dt1, dt1) * inv - t0 * inv
    var x = _bilerp[4](fr.qx, a, b); var y = _bilerp[4](fr.qy, a, b); var z = _bilerp[4](fr.qz, a, b)
    var lo = Vec3f(x.reduce_min(), y.reduce_min(), z.reduce_min()); var hi = Vec3f(x.reduce_max(), y.reduce_max(), z.reduce_max())
    var base = max(max(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z)
    lo = lo - fr.eps; hi = hi + fr.eps
    if d.scale <= Float32(0):
        return (lo, hi, base)
    var u0 = d.sq[unsafe_offset=3 * pi]; var v0 = d.sq[unsafe_offset=3 * pi + 1]; var sz = d.sq[unsafe_offset=3 * pi + 2]
    var hr = height_range(d, Int(d.face[unsafe_offset=pi]), u0 + ds0 * sz, u0 + ds1 * sz, v0 + dt0 * sz, v0 + dt1 * sz)
    var h0 = hr[0] * d.scale; var h1 = hr[1] * d.scale
    var ox = _imul(h0, h1, fr.nlo.x, fr.nhi.x); var oy = _imul(h0, h1, fr.nlo.y, fr.nhi.y); var oz = _imul(h0, h1, fr.nlo.z, fr.nhi.z)
    return (Vec3f(lo.x + ox[0], lo.y + oy[0], lo.z + oz[0]), Vec3f(hi.x + ox[1], hi.y + oy[1], hi.z + oz[1]), base)


def _intersect_fine(d: DispSurface, prim: Int, org: Vec3f, dir: Vec3f, t_max: Float32) -> Tuple[Bool, Float32, Float32, Float32, Int, Float32]:
    """intersect_displaced with exact leaves for a Bezier patch, its boxes from the sub-square's bilinear frame: no crops
    and no normal ranges per step, and no per-depth state beyond the halving path."""
    var pk = prim_patch(d, prim); var pi = pk[0]; var sq = sub_square(pk[2], pk[1])
    var src = d.patches[unsafe_offset=pi]
    var o = BezierPatch(src.x - org.x, src.y - org.y, src.z - org.z)
    var cam = d.cam - org
    var inv = Vec3f(Float32(1) / dir.x, Float32(1) / dir.y, Float32(1) / dir.z)
    var best = t_max; var hit = False; var hs = Float32(0); var ht = Float32(0); var hsz = Float32(1)
    var depth = 2 * pk[2]; var steps = 0
    var path = sq[3]; var pend = UInt64(0)
    var dom = (sq[0], sq[1], sq[2], sq[2])
    var f = Int(d.face[unsafe_offset=pi])
    var texels = d.sq[unsafe_offset=3 * pi + 2] * Float32(1 << max(Int(d.hdr[unsafe_offset=3 * f + 1]), Int(d.hdr[unsafe_offset=3 * f + 2])))
    var root = displaced_box(d, pi, o, org, sq[0], sq[0] + sq[2], sq[1], sq[1] + sq[2])
    var fr = _fine_frame(o, sq[0], sq[1], sq[2], root[2], root[3])
    var cur = (root[0], root[1], root[4])
    var cur_hit = _slab(cur[0], cur[1], inv, best)
    while True:
        steps += 1
        var descend = False
        if cur_hit[0]:
            var sz = _leaf_size(cur[0], cur[1], cam, d.pixel)
            if depth >= PATCH_MAX_DEPTH or max(dom[2], dom[3]) * texels <= Float32(1) or (d.leaf_mode & 1 == 1 and cur[2] <= sz[1]):
                var lh = _exact_leaf(d, pi, dom, org, dir, best)
                if lh[0]:
                    best = lh[1]; hit = True; hs = lh[2]; ht = lh[3]; hsz = max(dom[2], dom[3])
            else:
                var along_u = depth % 2 == 0
                var l_dom = (dom[0], dom[1], dom[2] * Float32(0.5) if along_u else dom[2], dom[3] if along_u else dom[3] * Float32(0.5))
                var u_dom = (dom[0] + dom[2] * Float32(0.5) if along_u else dom[0], dom[1] if along_u else dom[1] + dom[3] * Float32(0.5), l_dom[2], l_dom[3])
                var lb = _fine_box(d, pi, fr, sq[0], sq[1], sq[2], l_dom[0], l_dom[0] + l_dom[2], l_dom[1], l_dom[1] + l_dom[3])
                var ub = _fine_box(d, pi, fr, sq[0], sq[1], sq[2], u_dom[0], u_dom[0] + u_dom[2], u_dom[1], u_dom[1] + u_dom[3])
                var h0 = _slab(lb[0], lb[1], inv, best); var h1 = _slab(ub[0], ub[1], inv, best)
                if h0[0] or h1[0]:
                    var upper_first = h1[0] and (not h0[0] or h1[1] < h0[1])
                    if h0[0] and h1[0]:
                        pend |= UInt64(1) << UInt64(depth)
                    if upper_first:
                        path |= UInt64(1) << UInt64(depth)
                        dom = u_dom; cur = ub; cur_hit = h1
                    else:
                        dom = l_dom; cur = lb; cur_hit = h0
                    depth += 1
                    descend = True
        if descend:
            continue
        if pend == 0:
            break
        var level = 63 - Int(count_leading_zeros(pend))
        pend &= ~(UInt64(1) << UInt64(level))
        path = (path & ((UInt64(1) << UInt64(level)) - 1)) | ((~path) & (UInt64(1) << UInt64(level)))
        depth = level + 1
        dom = _domain(path, depth)
        cur = _fine_box(d, pi, fr, sq[0], sq[1], sq[2], dom[0], dom[0] + dom[2], dom[1], dom[1] + dom[3])
        cur_hit = _slab(cur[0], cur[1], inv, best)
    return (hit, best, hs, ht, steps, hsz)


@always_inline
def _len(v: Vec3f) -> Float32:
    return sqrt(dot(v, v))


@always_inline
def _corner_normal(c: BezierPatch, k0: Int, ka: Int, kb: Int) -> Vec3f:
    """The unit normal at net point k0 from the differences to ka and kb (ordered to give Pu x Pv); 0 if degenerate."""
    var a = Vec3f(c.x[ka] - c.x[k0], c.y[ka] - c.y[k0], c.z[ka] - c.z[k0]); var b = Vec3f(c.x[kb] - c.x[k0], c.y[kb] - c.y[k0], c.z[kb] - c.z[k0])
    var n = cross(a, b)
    var l = _len(n)
    return n * (Float32(1) / l) if l > Float32(0) else Vec3f(0, 0, 0)


def _intersect_hf(d: DispSurface, prim: Int, org: Vec3f, dir: Vec3f, t_max: Float32) -> Tuple[Bool, Float32, Float32, Float32, Int, Float32]:
    """A Bezier patch's sub-square as a height field: the ray goes into the affine frame (a, b, h) -> C + Ba (a - 1/2) +
    Bb (b - 1/2) + h Nc of the bilinear base through its corners and their interpolated normals; the surface is within E of
    that frame (the bicubic's deviation from the bilinear, the bilinear's twist, |h| times the normals' spread), which pads
    every cell. Cells are the halvings of the sub-square, each a box [a] x [b] x [height range] in the frame; leaves are exact.
    Nc is the corner normals' mean."""
    var pk = prim_patch(d, prim); var pi = pk[0]; var sq = sub_square(pk[2], pk[1])
    var S = sq[2]
    var src = d.patches[unsafe_offset=pi]
    var o = BezierPatch(src.x - org.x, src.y - org.y, src.z - org.z)
    var c = _crop(o, sq[0], sq[0] + S, sq[1], sq[1] + S)
    var qx = SIMD[DType.float32, 4](c.x[0], c.x[3], c.x[15], c.x[12])
    var qy = SIMD[DType.float32, 4](c.y[0], c.y[3], c.y[15], c.y[12])
    var qz = SIMD[DType.float32, 4](c.z[0], c.z[3], c.z[15], c.z[12])
    var e_b = _len(Vec3f(abs(c.x - _bilerp[16](qx, _A16, _B16)).reduce_max(), abs(c.y - _bilerp[16](qy, _A16, _B16)).reduce_max(),
                         abs(c.z - _bilerp[16](qz, _A16, _B16)).reduce_max()))
    var q00 = Vec3f(qx[0], qy[0], qz[0]); var q10 = Vec3f(qx[1], qy[1], qz[1]); var q11 = Vec3f(qx[2], qy[2], qz[2]); var q01 = Vec3f(qx[3], qy[3], qz[3])
    var nc = (_corner_normal(c, 0, 1, 4) + _corner_normal(c, 3, 7, 2) + _corner_normal(c, 15, 14, 11) + _corner_normal(c, 12, 8, 13)) * Float32(0.25)
    var ba = ((q10 - q00) + (q11 - q01)) * Float32(0.5); var bb = ((q01 - q00) + (q11 - q10)) * Float32(0.5)
    var cc = (q00 + q10 + q11 + q01) * Float32(0.25)
    var r0 = cross(bb, nc); var r1 = cross(nc, ba); var r2 = cross(ba, bb)
    var det = dot(ba, r0)
    var nr = _net_normal_range(INet(o, o), True, sq[0], sq[0] + S, sq[1], sq[1] + S)
    var f = Int(d.face[unsafe_offset=pi]); var u0 = d.sq[unsafe_offset=3 * pi]; var v0 = d.sq[unsafe_offset=3 * pi + 1]; var psz = d.sq[unsafe_offset=3 * pi + 2]
    var texels = psz * Float32(1 << max(Int(d.hdr[unsafe_offset=3 * f + 1]), Int(d.hdr[unsafe_offset=3 * f + 2])))
    var hr = height_range(d, f, u0 + sq[0] * psz, u0 + (sq[0] + S) * psz, v0 + sq[1] * psz, v0 + (sq[1] + S) * psz)
    var hmax = max(abs(hr[0]), abs(hr[1])) * max(d.scale, Float32(0))
    var spread = _len(nr[1] - nr[0]) * Float32(0.5) + _len((nr[0] + nr[1]) * Float32(0.5) - nc)   # |N - Nc| inside the normal range
    var e = e_b + _len(q00 - q10 + q11 - q01) * Float32(0.25) + hmax * spread
    e = e * Float32(1.0001) + Float32(1e-6) * (_len(cc) + _len(ba) + _len(bb))
    if abs(det) < Float32(1e-20) * _len(ba) * _len(bb) or not (e == e):
        return (False, t_max, Float32(0), Float32(0), 1, S)   # degenerate frame: nothing to walk
    var idet = Float32(1) / det
    r0 = r0 * idet; r1 = r1 * idet; r2 = r2 * idet   # the rows of the frame's inverse
    var lo = Vec3f(Float32(0.5) - dot(r0, cc), Float32(0.5) - dot(r1, cc), -dot(r2, cc))   # the ray origin (0) in the frame
    var ld = Vec3f(dot(r0, dir), dot(r1, dir), dot(r2, dir))
    var pad = Vec3f(_len(r0) * e / S, _len(r1) * e / S, _len(r2) * e)   # a and b in sub-square units, then per patch unit below
    var linv = Vec3f(Float32(1) / ld.x, Float32(1) / ld.y, Float32(1) / ld.z)
    var cam = d.cam - org
    var ext = max(_len(ba), _len(bb))   # the base's extent per sub-square unit
    var best = t_max; var hit = False; var hs = Float32(0); var ht = Float32(0); var hsz = Float32(1)
    var depth = 2 * pk[2]; var steps = 0
    var path = sq[3]; var pend = UInt64(0)
    var dom = (sq[0], sq[1], S, S)
    var cur_hit = _hf_cell(d, f, u0, v0, psz, sq[0], sq[1], S, dom, lo, linv, pad, best)
    while True:
        steps += 1
        var descend = False
        if cur_hit[0]:
            var am = (dom[0] + Float32(0.5) * dom[2] - sq[0]) / S - Float32(0.5); var bm = (dom[1] + Float32(0.5) * dom[3] - sq[1]) / S - Float32(0.5)
            var wc = cc + ba * am + bb * bm - cam
            var cext = ext * max(dom[2], dom[3]) / S
            var dist = max(_len(wc) - cext, cext * Float32(0.1) + Float32(1e-4))
            if depth >= PATCH_MAX_DEPTH or max(dom[2], dom[3]) * texels <= Float32(1) or cext <= dist * d.pixel:
                var lh = _exact_leaf(d, pi, dom, org, dir, best)
                if lh[0]:
                    best = lh[1]; hit = True; hs = lh[2]; ht = lh[3]; hsz = max(dom[2], dom[3])
            else:
                var along_u = depth % 2 == 0
                var l_dom = (dom[0], dom[1], dom[2] * Float32(0.5) if along_u else dom[2], dom[3] if along_u else dom[3] * Float32(0.5))
                var u_dom = (dom[0] + dom[2] * Float32(0.5) if along_u else dom[0], dom[1] if along_u else dom[1] + dom[3] * Float32(0.5), l_dom[2], l_dom[3])
                var h0 = _hf_cell(d, f, u0, v0, psz, sq[0], sq[1], S, l_dom, lo, linv, pad, best)
                var h1 = _hf_cell(d, f, u0, v0, psz, sq[0], sq[1], S, u_dom, lo, linv, pad, best)
                if h0[0] or h1[0]:
                    var upper_first = h1[0] and (not h0[0] or h1[1] < h0[1])
                    if h0[0] and h1[0]:
                        pend |= UInt64(1) << UInt64(depth)
                    if upper_first:
                        path |= UInt64(1) << UInt64(depth)
                        dom = u_dom; cur_hit = h1
                    else:
                        dom = l_dom; cur_hit = h0
                    depth += 1
                    descend = True
        if descend:
            continue
        if pend == 0:
            break
        var level = 63 - Int(count_leading_zeros(pend))
        pend &= ~(UInt64(1) << UInt64(level))
        path = (path & ((UInt64(1) << UInt64(level)) - 1)) | ((~path) & (UInt64(1) << UInt64(level)))
        depth = level + 1
        dom = _domain(path, depth)
        cur_hit = _hf_cell(d, f, u0, v0, psz, sq[0], sq[1], S, dom, lo, linv, pad, best)
    return (hit, best, hs, ht, steps, hsz)


@always_inline
def _hf_cell(d: DispSurface, f: Int, u0: Float32, v0: Float32, psz: Float32, s0: Float32, t0: Float32, S: Float32,
             dom: Tuple[Float32, Float32, Float32, Float32], lo: Vec3f, linv: Vec3f, pad: Vec3f, t_max: Float32) -> Tuple[Bool, Float32]:
    """The ray (frame origin lo, inverse direction linv) against a cell's box in the frame: [a] x [b] x [height range],
    padded by the frame's error."""
    var a0 = (dom[0] - s0) / S - pad.x; var a1 = (dom[0] + dom[2] - s0) / S + pad.x
    var b0 = (dom[1] - t0) / S - pad.y; var b1 = (dom[1] + dom[3] - t0) / S + pad.y
    var h = Tuple[Float32, Float32](0, 0)
    if d.scale > Float32(0):
        var hr = height_range(d, f, u0 + dom[0] * psz, u0 + (dom[0] + dom[2]) * psz, v0 + dom[1] * psz, v0 + (dom[1] + dom[3]) * psz)
        h = (hr[0] * d.scale, hr[1] * d.scale)
    var h0 = h[0] - pad.z; var h1 = h[1] + pad.z
    var ax = (a0 - lo.x) * linv.x; var bx = (a1 - lo.x) * linv.x
    var ay = (b0 - lo.y) * linv.y; var by = (b1 - lo.y) * linv.y
    var az = (h0 - lo.z) * linv.z; var bz = (h1 - lo.z) * linv.z
    var tn = max(max(min(ax, bx), min(ay, by)), max(min(az, bz), Float32(0)))
    var tf = min(min(max(ax, bx), max(ay, by)), min(max(az, bz), t_max))
    return (tn <= tf, tn)


@fieldwise_init
struct BLFrame(TrivialRegisterPassable):
    """A sub-square's limit corners (anchored at the ray origin) and unit normals, 00 10 11 01, per axis."""
    var px: SIMD[DType.float32, 4]
    var py: SIMD[DType.float32, 4]
    var pz: SIMD[DType.float32, 4]
    var nx: SIMD[DType.float32, 4]
    var ny: SIMD[DType.float32, 4]
    var nz: SIMD[DType.float32, 4]


def grid_vertex(d: DispSurface, pi: Int, i: Int, j: Int, n: Int) -> Tuple[Vec3f, Vec3f]:
    """Grid point (i, j) of patch pi's n x n sub-squares: the limit point and its unit normal (0 where degenerate)."""
    var s = Float32(i) / Float32(n); var t = Float32(j) / Float32(n)
    var e = patch_eval(patch_at(d, pi, s, t), s, t)
    var nv = cross(e[1], e[2]); var ln = sqrt(dot(nv, nv))
    return (e[0], nv * (Float32(1) / ln) if ln > Float32(0) else Vec3f(0, 0, 0))


def corner_words(n: Int) -> Int:
    """Floats per patch in DispSurface.corners: origin, step, the edges' coarser neighbours (4 bits each: log2 ratio),
    then 5 x u16 per grid point."""
    return 5 + (5 * (n + 1) * (n + 1) + 1) // 2


def corner_step(extent: Float32) -> Float32:
    """The power of two that spans the largest patch extent in 16 bits."""
    var s = Float32(1)
    while extent > s * Float32(65000):
        s *= Float32(2)
    while extent <= s * Float32(32500):
        s *= Float32(0.5)
    return s


def _oct16(v: Float32) -> UInt16:
    return UInt16(Int(round(min(max(v, Float32(-1)), Float32(1)) * Float32(32767))) + 32768)


def _oct_enc(nv: Vec3f) -> Tuple[UInt16, UInt16]:
    """A unit normal as two 16-bit octahedral codes; (0, 0) for none."""
    var l1 = abs(nv.x) + abs(nv.y) + abs(nv.z)
    if l1 == Float32(0):
        return (UInt16(0), UInt16(0))
    var u = nv.x / l1; var v = nv.y / l1
    if nv.z < Float32(0):
        var uu = (Float32(1) - abs(v)) * (Float32(1) if u >= Float32(0) else Float32(-1))
        v = (Float32(1) - abs(u)) * (Float32(1) if v >= Float32(0) else Float32(-1)); u = uu
    return (_oct16(u), _oct16(v))


@always_inline
def _oct_dec(e0: UInt16, e1: UInt16) -> Vec3f:
    var u = (Float32(e0) - Float32(32768)) * Float32(1.0 / 32767); var v = (Float32(e1) - Float32(32768)) * Float32(1.0 / 32767)
    var z = Float32(1) - abs(u) - abs(v)
    if z < Float32(0):
        var uu = (Float32(1) - abs(v)) * (Float32(1) if u >= Float32(0) else Float32(-1))
        v = (Float32(1) - abs(u)) * (Float32(1) if v >= Float32(0) else Float32(-1)); u = uu
    var r = Float32(0) if e0 == UInt16(0) and e1 == UInt16(0) else Float32(1) / sqrt(u * u + v * v + z * z)
    return Vec3f(u * r, v * r, z * r)


def corner_grid(d: DispSurface, pi: Int, lo: Vec3f, step: Float32, dst: _F32):
    """Patch pi's grid points into dst, rows along t: positions as global multiples of step (so patches sharing
    a point store the same one), unit normals octahedral; (0, 0) is a degenerate normal."""
    var n = 1 << patch_level(d, pi)
    var k = Vec3f(floor(lo.x / step), floor(lo.y / step), floor(lo.z / step))
    dst[unsafe_offset=0] = k.x * step; dst[unsafe_offset=1] = k.y * step; dst[unsafe_offset=2] = k.z * step; dst[unsafe_offset=3] = step
    dst.unsafe_offset(4).bitcast[UInt32]()[unsafe_offset=0] = UInt32(0)
    var q = dst.unsafe_offset(5).bitcast[UInt16]()
    for j in range(n + 1):
        for i in range(n + 1):
            var g = grid_vertex(d, pi, i, j, n); var o = 5 * (j * (n + 1) + i)
            q[unsafe_offset=o] = UInt16(min(max(Int(round(g[0].x / step) - k.x), 0), 65535))
            q[unsafe_offset=o + 1] = UInt16(min(max(Int(round(g[0].y / step) - k.y), 0), 65535))
            q[unsafe_offset=o + 2] = UInt16(min(max(Int(round(g[0].z / step) - k.z), 0), 65535))
            var e = _oct_enc(g[1]); q[unsafe_offset=o + 3] = e[0]; q[unsafe_offset=o + 4] = e[1]


@always_inline
def _edge_idx(n: Int, e: Int, k: Int) -> Int:
    """Grid index of point k along edge e of an n x n grid (counter-clockwise from corner e)."""
    var ij = (k, 0) if e == 0 else ((n, k) if e == 1 else ((n - k, n) if e == 2 else (0, n - k)))
    return ij[1] * (n + 1) + ij[0]


def weld_corner_grid(np: Int, lv: Levels, grid: _F32) raises:
    """Points on shared patch edges (edges matched by their end points) get one position and their mean normal:
    the two patches' bicubics agree there only up to rounding, which the 16-bit grid can turn into a step. Where the
    neighbour is coarser, only its points are welded and the edge word tells _bl_frame to interpolate the others."""
    var pb = List[Int](length=np + 1, fill=0)   # first point id of each patch
    for p in range(np):
        var n = 1 << _class_of(lv.patch, p); pb[p + 1] = pb[p] + (n + 1) * (n + 1)
    var parent = List[Int32](length=pb[np], fill=Int32(-1)); var joined = List[Bool](length=pb[np], fill=False)
    def find(par: List[Int32], x: Int) -> Int:
        var r = x
        while Int(par[r]) >= 0:
            r = Int(par[r])
        return r
    def gpos(p: Int, o: Int) {imm} -> Tuple[Int, Int, Int]:
        var g = grid.unsafe_offset(corner_off(lv, p)); var st = g[unsafe_offset=3]; var q = g.unsafe_offset(5).bitcast[UInt16]()
        return (Int(g[unsafe_offset=0] / st) + Int(q[unsafe_offset=5 * o]), Int(g[unsafe_offset=1] / st) + Int(q[unsafe_offset=5 * o + 1]),
                Int(g[unsafe_offset=2] / st) + Int(q[unsafe_offset=5 * o + 2]))
    def key(p: Tuple[Int, Int, Int]) -> String:
        return String(p[0] // 64) + "," + String(p[1] // 64) + "," + String(p[2] // 64)
    var ends = Dict[String, Int]()
    var mixed = 0
    for p in range(np):
        var n = 1 << _class_of(lv.patch, p)
        for e in range(4):
            var ka = key(gpos(p, _edge_idx(n, e, 0))); var kb = key(gpos(p, _edge_idx(n, e, n)))
            var ek = ka + "|" + kb if ka < kb else kb + "|" + ka
            if ek not in ends:
                ends[ek] = p * 4 + e
                continue
            var p2 = ends[ek] // 4; var e2 = ends[ek] % 4; var n2 = 1 << _class_of(lv.patch, p2)
            var same = key(gpos(p2, _edge_idx(n2, e2, 0))) == ka
            var nc = min(n, n2)
            for kc in range(nc + 1):
                var x1 = pb[p] + _edge_idx(n, e, kc * (n // nc)); var x2 = pb[p2] + _edge_idx(n2, e2, (kc if same else nc - kc) * (n2 // nc))
                joined[x1] = True; joined[x2] = True
                var r1 = find(parent, x1); var r2 = find(parent, x2)
                if r1 != r2:
                    parent[r1] = Int32(r2)
            if n != n2:   # the finer side interpolates its points between the coarser one's
                var pf = p if n > n2 else p2; var ef = e if n > n2 else e2; var r = max(n, n2) // nc; var b = 0
                while (1 << b) < r:
                    b += 1
                var w = grid.unsafe_offset(corner_off(lv, pf) + 4).bitcast[UInt32]()
                w[unsafe_offset=0] = w[unsafe_offset=0] | UInt32(b << (4 * ef)); mixed += 1
    var slot = Dict[Int, Int](); var sp = List[Int](); var sn = List[Vec3f]()   # per group: x, y, z sums and count; normal sum
    var owner = List[Int32](length=pb[np], fill=Int32(0))
    for p in range(np):
        for x in range(pb[p], pb[p + 1]):
            owner[x] = Int32(p)
    for x in range(pb[np]):
        if not joined[x]:
            continue
        var r = find(parent, x); var p = Int(owner[x]); var o = x - pb[p]
        if r not in slot:
            slot[r] = len(sn); sp.append(0); sp.append(0); sp.append(0); sp.append(0); sn.append(Vec3f(0, 0, 0))
        var s = slot[r]; var gp = gpos(p, o); var q = grid.unsafe_offset(corner_off(lv, p) + 5).bitcast[UInt16]()
        sp[4 * s] += gp[0]; sp[4 * s + 1] += gp[1]; sp[4 * s + 2] += gp[2]; sp[4 * s + 3] += 1
        sn[s] = sn[s] + _oct_dec(q[unsafe_offset=5 * o + 3], q[unsafe_offset=5 * o + 4])
    var moved = 0
    for x in range(pb[np]):
        if not joined[x]:
            continue
        var s = slot[find(parent, x)]; var p = Int(owner[x]); var o = x - pb[p]
        var g = grid.unsafe_offset(corner_off(lv, p)); var st = g[unsafe_offset=3]; var q = g.unsafe_offset(5).bitcast[UInt16]()
        var c = Float64(sp[4 * s + 3])
        for a in range(3):
            var v = UInt16(min(max(Int(round(Float64(sp[4 * s + a]) / c)) - Int(g[unsafe_offset=a] / st), 0), 65535))
            if v != q[unsafe_offset=5 * o + a]:
                moved += 1
            q[unsafe_offset=5 * o + a] = v
        var nl = sqrt(dot(sn[s], sn[s]))
        var e = _oct_enc(sn[s] * (Float32(1) / nl) if nl > Float32(0) else Vec3f(0, 0, 0))
        q[unsafe_offset=5 * o + 3] = e[0]; q[unsafe_offset=5 * o + 4] = e[1]
    print("welded edge grid points:", len(sn), "groups,", len(ends), "edge keys,", moved, "coordinates moved,", mixed, "edges to a coarser patch")


def _bl_frame(d: DispSurface, pi: Int, k: Int, org: Vec3f) -> BLFrame:
    """Sub-square k's corners from the patch's grid: neighbouring sub-squares read the same points and normals. On an edge
    to a coarser patch the points between the coarse ones are interpolated from them, as the coarse side sees its edge."""
    var l = patch_level(d, pi); var n = 1 << l
    var ku = k & (n - 1); var kv = k >> l
    var f = BLFrame(SIMD[DType.float32, 4](0), SIMD[DType.float32, 4](0), SIMD[DType.float32, 4](0), SIMD[DType.float32, 4](0),
                    SIMD[DType.float32, 4](0), SIMD[DType.float32, 4](0))
    var g = d.corners.unsafe_offset(corner_off(d.lv[], pi))
    var st = g[unsafe_offset=3]
    var b = Vec3f(g[unsafe_offset=0], g[unsafe_offset=1], g[unsafe_offset=2])
    var ew = g.unsafe_offset(4).bitcast[UInt32]()[unsafe_offset=0]
    var q = g.unsafe_offset(5).bitcast[UInt16]()
    comptime for c in range(4):   # b + q st is exact, the same float in every patch that stores the point
        var o = 5 * ((kv + (1 if c >= 2 else 0)) * (n + 1) + ku + (1 if c == 1 or c == 2 else 0))
        f.px[c] = (b.x + Float32(q[unsafe_offset=o]) * st) - org.x; f.py[c] = (b.y + Float32(q[unsafe_offset=o + 1]) * st) - org.y
        f.pz[c] = (b.z + Float32(q[unsafe_offset=o + 2]) * st) - org.z
        var nv = _oct_dec(q[unsafe_offset=o + 3], q[unsafe_offset=o + 4])
        f.nx[c] = nv.x; f.ny[c] = nv.y; f.nz[c] = nv.z
    if ew != UInt32(0):
        comptime for c in range(4):
            var i = ku + (1 if c == 1 or c == 2 else 0); var j = kv + (1 if c >= 2 else 0)
            var e = 0 if j == 0 else (1 if i == n else (2 if j == n else (3 if i == 0 else -1)))
            if e >= 0:
                var kk = i if e == 0 else (j if e == 1 else (n - i if e == 2 else n - j))
                var r = 1 << Int((ew >> UInt32(4 * e)) & UInt32(15)); var m = kk & (r - 1)
                if m != 0:
                    var o0 = 5 * _edge_idx(n, e, kk - m); var o1 = 5 * _edge_idx(n, e, kk - m + r); var a = Float32(m) / Float32(r)
                    var p0 = Vec3f((b.x + Float32(q[unsafe_offset=o0]) * st) - org.x, (b.y + Float32(q[unsafe_offset=o0 + 1]) * st) - org.y,
                                   (b.z + Float32(q[unsafe_offset=o0 + 2]) * st) - org.z)
                    var p1 = Vec3f((b.x + Float32(q[unsafe_offset=o1]) * st) - org.x, (b.y + Float32(q[unsafe_offset=o1 + 1]) * st) - org.y,
                                   (b.z + Float32(q[unsafe_offset=o1 + 2]) * st) - org.z)
                    var n0 = _oct_dec(q[unsafe_offset=o0 + 3], q[unsafe_offset=o0 + 4]); var n1 = _oct_dec(q[unsafe_offset=o1 + 3], q[unsafe_offset=o1 + 4])
                    f.px[c] = p0.x + (p1.x - p0.x) * a; f.py[c] = p0.y + (p1.y - p0.y) * a; f.pz[c] = p0.z + (p1.z - p0.z) * a
                    f.nx[c] = n0.x + (n1.x - n0.x) * a; f.ny[c] = n0.y + (n1.y - n0.y) * a; f.nz[c] = n0.z + (n1.z - n0.z) * a
    return f


def _heights8(d: DispSurface, f: Int, u0: Float32, v0: Float32, sz: Float32, s: SIMD[DType.float32, 8],
              t: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    """face_height at 8 points of patch square (u0, v0, sz): the face's header is read once."""
    var off = Int(d.chain_off[unsafe_offset=f]); var w = 1 << Int(d.hdr[unsafe_offset=3 * f + 1]); var h = 1 << Int(d.hdr[unsafe_offset=3 * f + 2])
    var x = min(max((s * sz + u0) * Float32(w) - Float32(0.5), Float32(-0.5)), Float32(w) - Float32(0.5))
    var y = min(max((t * sz + v0) * Float32(h) - Float32(0.5), Float32(-0.5)), Float32(h) - Float32(0.5))
    var r = SIMD[DType.float32, 8](0)
    comptime for k in range(8):
        var x0 = min(Int(floor(x[k])), w - 1); var y0 = min(Int(floor(y[k])), h - 1)
        var fx = x[k] - Float32(x0); var fy = y[k] - Float32(y0)
        var a = _texel(d, off, w, x0, y0) * (Float32(1) - fx) + _texel(d, off, w, x0 + 1, y0) * fx
        var b = _texel(d, off, w, x0, y0 + 1) * (Float32(1) - fx) + _texel(d, off, w, x0 + 1, y0 + 1) * fx
        r[k] = a * (Float32(1) - fy) + b * fy
    return r


@always_inline
def _bl_point(d: DispSurface, fr: BLFrame, s0: Float32, t0: Float32, inv: Float32, s: Float32, t: Float32, hgt: Float32) -> Vec3f:
    """X = B + h N at global (s, t): B and N bilinear in the sub-square's corners, h (bytes) the face's height there."""
    var a = SIMD[DType.float32, 1]((s - s0) * inv); var b = SIMD[DType.float32, 1]((t - t0) * inv)
    var p = Vec3f(_bilerp[1](fr.px, a, b)[0], _bilerp[1](fr.py, a, b)[0], _bilerp[1](fr.pz, a, b)[0])
    if d.scale <= Float32(0):
        return p
    var n = Vec3f(_bilerp[1](fr.nx, a, b)[0], _bilerp[1](fr.ny, a, b)[0], _bilerp[1](fr.nz, a, b)[0])
    return p + n * (hgt * d.scale)


def _bl_box(d: DispSurface, pi: Int, f: Int, fr: BLFrame, s0: Float32, t0: Float32, inv: Float32,
            dom: Tuple[Float32, Float32, Float32, Float32]) -> Tuple[Vec3f, Vec3f, Float32]:
    """The cell's box: B's and N's ranges are their values at the cell's corners (bilinear), times the height range."""
    var a = SIMD[DType.float32, 4](dom[0], dom[0] + dom[2], dom[0] + dom[2], dom[0]) * inv - s0 * inv
    var b = SIMD[DType.float32, 4](dom[1], dom[1], dom[1] + dom[3], dom[1] + dom[3]) * inv - t0 * inv
    var x = _bilerp[4](fr.px, a, b); var y = _bilerp[4](fr.py, a, b); var z = _bilerp[4](fr.pz, a, b)
    var lo = Vec3f(x.reduce_min(), y.reduce_min(), z.reduce_min()); var hi = Vec3f(x.reduce_max(), y.reduce_max(), z.reduce_max())
    var base = max(max(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z)
    if d.scale <= Float32(0):
        return (lo, hi, base)
    var nx = _bilerp[4](fr.nx, a, b); var ny = _bilerp[4](fr.ny, a, b); var nz = _bilerp[4](fr.nz, a, b)
    var sz = d.sq[unsafe_offset=3 * pi + 2]; var u0 = d.sq[unsafe_offset=3 * pi]; var v0 = d.sq[unsafe_offset=3 * pi + 1]
    var hr = height_range(d, f, u0 + dom[0] * sz, u0 + (dom[0] + dom[2]) * sz, v0 + dom[1] * sz, v0 + (dom[1] + dom[3]) * sz)
    var h0 = hr[0] * d.scale; var h1 = hr[1] * d.scale
    var ox = _imul(h0, h1, nx.reduce_min(), nx.reduce_max()); var oy = _imul(h0, h1, ny.reduce_min(), ny.reduce_max())
    var oz = _imul(h0, h1, nz.reduce_min(), nz.reduce_max())
    var pad = Float32(1e-6) * (abs(x).reduce_max() + abs(y).reduce_max() + abs(z).reduce_max())
    return (Vec3f(lo.x + ox[0] - pad, lo.y + oy[0] - pad, lo.z + oz[0] - pad), Vec3f(hi.x + ox[1] + pad, hi.y + oy[1] + pad, hi.z + oz[1] + pad), base)


def _bl_leaf(d: DispSurface, pi: Int, f: Int, fr: BLFrame, s0q: Float32, t0q: Float32, inv_s: Float32,
            dom: Tuple[Float32, Float32, Float32, Float32], dir: Vec3f, t_max: Float32) -> Tuple[Bool, Float32, Float32, Float32]:
    var s0 = dom[0]; var s1 = dom[0] + dom[2]; var t0 = dom[1]; var t1 = dom[1] + dom[3]
    var sm = s0 + Float32(0.5) * dom[2]; var tm = t0 + Float32(0.5) * dom[3]
    var cs = SIMD[DType.float32, 8](s0, s1, s1, s0, sm, s1, sm, s0)
    var ct = SIMD[DType.float32, 8](t0, t0, t1, t1, t0, tm, t1, tm)
    var hs = _heights8(d, f, d.sq[unsafe_offset=3 * pi], d.sq[unsafe_offset=3 * pi + 1], d.sq[unsafe_offset=3 * pi + 2], cs, ct)
    var cp = Array[Vec3f, 8](fill=Vec3f(0, 0, 0))
    for k in range(8):
        cp[k] = _bl_point(d, fr, s0q, t0q, inv_s, cs[k], ct[k], hs[k])
    return _leaf_hit(cp, s0, s1, t0, t1, dir, t_max)


def _in_box(p: Vec3f, c: Tuple[Vec3f, Vec3f, Float32]) -> Bool:
    """Whether point p lies in box c, widened by the leaf tests' overlap: they can return far roots off the patch."""
    var lo = c[0]; var hi = c[1]
    var e = max(max(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z) * Float32(5e-3) + Float32(1e-5) * (abs(p.x) + abs(p.y) + abs(p.z)) + Float32(1e-6)
    return p.x >= lo.x - e and p.x <= hi.x + e and p.y >= lo.y - e and p.y <= hi.y + e and p.z >= lo.z - e and p.z <= hi.z + e


def _intersect_bl(d: DispSurface, prim: Int, org: Vec3f, dir: Vec3f, t_max: Float32) -> Tuple[Bool, Float32, Float32, Float32, Int, Float32]:
    """The sub-square's base as the bilinear patch through its limit corners, displaced along the bilinear blend of their
    unit normals: X = B + h N. Neighbours share the corners, so their edges agree; no bicubic is touched after entry."""
    var pk = prim_patch(d, prim); var pi = pk[0]; var sq = sub_square(pk[2], pk[1])
    var S = sq[2]; var inv_s = Float32(1) / S
    var f = Int(d.face[unsafe_offset=pi])
    var texels = d.sq[unsafe_offset=3 * pi + 2] * Float32(1 << max(Int(d.hdr[unsafe_offset=3 * f + 1]), Int(d.hdr[unsafe_offset=3 * f + 2])))
    var kq = pk[1]
    var fr = _bl_frame(d, pi, kq, org)
    var cam = d.cam - org
    var inv = Vec3f(Float32(1) / dir.x, Float32(1) / dir.y, Float32(1) / dir.z)
    var best = t_max; var hit = False; var hs = Float32(0); var ht = Float32(0); var hsz = Float32(1)
    var depth = 2 * pk[2]; var steps = 0
    var path = sq[3]; var pend = UInt64(0)
    var dom = (sq[0], sq[1], S, S)
    var lo0 = Vec3f(fr.px.reduce_min(), fr.py.reduce_min(), fr.pz.reduce_min()); var hi0 = Vec3f(fr.px.reduce_max(), fr.py.reduce_max(), fr.pz.reduce_max())
    if S * texels <= Float32(1) or (d.leaf_mode == 5 and max(max(hi0.x - lo0.x, hi0.y - lo0.y), hi0.z - lo0.z) <= _leaf_size(lo0, hi0, cam, d.pixel)[1]):
        var lh = _bl_leaf(d, pi, f, fr, sq[0], sq[1], inv_s, dom, dir, best)   # a leaf already; the caller tested its box
        return (lh[0] and _in_box(dir * lh[1], _bl_box(d, pi, f, fr, sq[0], sq[1], inv_s, dom)), lh[1], lh[2], lh[3], 1, S)
    var cur = _bl_box(d, pi, f, fr, sq[0], sq[1], inv_s, dom)
    var cur_hit = _slab(cur[0], cur[1], inv, best)
    while True:
        steps += 1
        var descend = False
        if cur_hit[0]:
            var sz = _leaf_size(cur[0], cur[1], cam, d.pixel)
            if depth >= PATCH_MAX_DEPTH or max(dom[2], dom[3]) * texels <= Float32(1) or (d.leaf_mode == 5 and cur[2] <= sz[1]):
                var lh = _bl_leaf(d, pi, f, _bl_frame(d, pi, kq, org), sq[0], sq[1], inv_s, dom, dir, best)
                if lh[0] and _in_box(dir * lh[1], cur):
                    best = lh[1]; hit = True; hs = lh[2]; ht = lh[3]; hsz = max(dom[2], dom[3])
            else:
                var along_u = depth % 2 == 0
                var l_dom = (dom[0], dom[1], dom[2] * Float32(0.5) if along_u else dom[2], dom[3] if along_u else dom[3] * Float32(0.5))
                var u_dom = (dom[0] + dom[2] * Float32(0.5) if along_u else dom[0], dom[1] if along_u else dom[1] + dom[3] * Float32(0.5), l_dom[2], l_dom[3])
                var lb = _bl_box(d, pi, f, _bl_frame(d, pi, kq, org), sq[0], sq[1], inv_s, l_dom); var ub = _bl_box(d, pi, f, _bl_frame(d, pi, kq, org), sq[0], sq[1], inv_s, u_dom)
                var h0 = _slab(lb[0], lb[1], inv, best); var h1 = _slab(ub[0], ub[1], inv, best)
                if h0[0] or h1[0]:
                    var upper_first = h1[0] and (not h0[0] or h1[1] < h0[1])
                    if h0[0] and h1[0]:
                        pend |= UInt64(1) << UInt64(depth)
                    if upper_first:
                        path |= UInt64(1) << UInt64(depth)
                        dom = u_dom; cur = ub; cur_hit = h1
                    else:
                        dom = l_dom; cur = lb; cur_hit = h0
                    depth += 1
                    descend = True
        if descend:
            continue
        if pend == 0:
            break
        var level = 63 - Int(count_leading_zeros(pend))
        pend &= ~(UInt64(1) << UInt64(level))
        path = (path & ((UInt64(1) << UInt64(level)) - 1)) | ((~path) & (UInt64(1) << UInt64(level)))
        depth = level + 1
        dom = _domain(path, depth)
        cur = _bl_box(d, pi, f, _bl_frame(d, pi, kq, org), sq[0], sq[1], inv_s, dom)
        cur_hit = _slab(cur[0], cur[1], inv, best)
    return (hit, best, hs, ht, steps, hsz)


def intersect_displaced[M: Int = -1](d: DispSurface, prim: Int, org: Vec3f, dir: Vec3f, t_max: Float32) -> Tuple[Bool, Float32, Float32, Float32, Int, Float32]:
    """The nearest hit of primitive prim (a patch's sub-square, prim_patch) before t_max: (hit, t, s, t, steps, the leaf square's size).
    M = 5 compiles only the bilinear-base walk (leaf modes 5 and 6).
    Box leaves end where the displaced box is no larger than the footprint; exact leaves where the undisplaced square reaches a
    texel or the footprint, and intersect the bilinear patch through its four displaced corners, so the depth does not grow with
    the displacement's amplitude."""
    comptime if M == 5:   # a kernel with only the bilinear-base walk keeps few registers live
        return _intersect_bl(d, prim, org, dir, t_max)
    if d.leaf_mode >= 5:
        return _intersect_bl(d, prim, org, dir, t_max)
    if d.greg[unsafe_offset=prim_patch(d, prim)[0]] != 1:   # Gregory patches keep the cropping loop
        if d.leaf_mode == 1 or d.leaf_mode == 2:
            return _intersect_fine(d, prim, org, dir, t_max)
        if d.leaf_mode == 3:
            return _intersect_hf(d, prim, org, dir, t_max)
    var pk = prim_patch(d, prim); var pi = pk[0]; var sq = sub_square(pk[2], pk[1])
    var src = d.patches[unsafe_offset=pi]
    var o = BezierPatch(src.x - org.x, src.y - org.y, src.z - org.z)   # anchored at the ray origin
    var cam = d.cam - org
    var inv = Vec3f(Float32(1) / dir.x, Float32(1) / dir.y, Float32(1) / dir.z)
    var best = t_max; var hit = False; var hs = Float32(0); var ht = Float32(0); var hsz = Float32(1)
    var nlo = Array[Vec3f, PATCH_MAX_DEPTH + 1](fill=Vec3f(-1, -1, -1)); var nhi = Array[Vec3f, PATCH_MAX_DEPTH + 1](fill=Vec3f(1, 1, 1))
    var depth = 2 * pk[2]; var steps = 0
    var path = sq[3]; var pend = UInt64(0)
    var dom = (sq[0], sq[1], sq[2], sq[2])
    var f = Int(d.face[unsafe_offset=pi])
    var texels = d.sq[unsafe_offset=3 * pi + 2] * Float32(1 << max(Int(d.hdr[unsafe_offset=3 * f + 1]), Int(d.hdr[unsafe_offset=3 * f + 2])))   # a patch side in finest texels
    var cur = displaced_box(d, pi, o, org, dom[0], dom[0] + dom[2], dom[1], dom[1] + dom[3])
    nlo[depth] = cur[2]; nhi[depth] = cur[3]
    var cur_hit = _slab(cur[0], cur[1], inv, best)
    while True:
        steps += 1
        var descend = False
        if cur_hit[0]:
            var sz = _leaf_size(cur[0], cur[1], cam, d.pixel)
            var leaf: Bool
            if d.leaf_mode == 0:
                leaf = sz[0] <= sz[1]
            else:
                leaf = max(dom[2], dom[3]) * texels <= Float32(1) or (d.leaf_mode != 2 and cur[4] <= sz[1])
            if depth >= PATCH_MAX_DEPTH or leaf:
                if d.leaf_mode == 0:
                    if cur_hit[1] < best:
                        best = cur_hit[1]; hit = True
                        hs = dom[0] + Float32(0.5) * dom[2]; ht = dom[1] + Float32(0.5) * dom[3]; hsz = max(dom[2], dom[3])
                else:
                    var lh = _exact_leaf(d, pi, dom, org, dir, best)
                    if lh[0]:
                        best = lh[1]; hit = True; hs = lh[2]; ht = lh[3]; hsz = max(dom[2], dom[3])
            else:
                var along_u = depth % 2 == 0
                var hu0 = dom[0]; var hv0 = dom[1]; var hdu = dom[2]; var hdv = dom[3]
                var l_dom = (hu0, hv0, hdu * Float32(0.5) if along_u else hdu, hdv if along_u else hdv * Float32(0.5))
                var u_dom = (hu0 + hdu * Float32(0.5) if along_u else hu0, hv0 if along_u else hv0 + hdv * Float32(0.5), l_dom[2], l_dom[3])
                var lb = displaced_box(d, pi, o, org, l_dom[0], l_dom[0] + l_dom[2], l_dom[1], l_dom[1] + l_dom[3], nlo[depth], nhi[depth])
                var ub = displaced_box(d, pi, o, org, u_dom[0], u_dom[0] + u_dom[2], u_dom[1], u_dom[1] + u_dom[3], nlo[depth], nhi[depth])
                var h0 = _slab(lb[0], lb[1], inv, best); var h1 = _slab(ub[0], ub[1], inv, best)
                if h0[0] or h1[0]:
                    var upper_first = h1[0] and (not h0[0] or h1[1] < h0[1])
                    if h0[0] and h1[0]:
                        pend |= UInt64(1) << UInt64(depth)
                    if upper_first:
                        path |= UInt64(1) << UInt64(depth)
                        dom = u_dom; cur = ub; cur_hit = h1
                    else:
                        dom = l_dom; cur = lb; cur_hit = h0
                    depth += 1
                    nlo[min(depth, PATCH_MAX_DEPTH)] = cur[2]; nhi[min(depth, PATCH_MAX_DEPTH)] = cur[3]
                    descend = True
        if descend:
            continue
        if pend == 0:
            break
        var level = 63 - Int(count_leading_zeros(pend))   # the deepest pending sibling
        pend &= ~(UInt64(1) << UInt64(level))
        path = (path & ((UInt64(1) << UInt64(level)) - 1)) | ((~path) & (UInt64(1) << UInt64(level)))
        depth = level + 1
        var dd = _domain(path, depth)
        dom = dd
        cur = displaced_box(d, pi, o, org, dd[0], dd[0] + dd[2], dd[1], dd[1] + dd[3], nlo[level], nhi[level])
        nlo[min(depth, PATCH_MAX_DEPTH)] = cur[2]; nhi[min(depth, PATCH_MAX_DEPTH)] = cur[3]
        cur_hit = _slab(cur[0], cur[1], inv, best)
    return (hit, best, hs, ht, steps, hsz)


@always_inline
def _chain_h(d: DispSurface, off: Int, w: Int, x: Int, y: Int) -> Float32:
    """Height byte (x, y) of a bordered level, x in [-1, w], y in [-1, h]."""
    return Float32(Int(d.chain[unsafe_offset=off + (y + 1) * (w + 2) + x + 1]))


def _level_slope(d: DispSurface, f: Int, lvl: Int, u: Float32, v: Float32) -> Tuple[Float32, Float32]:
    """(dh/du, dh/dv) in bytes per face unit at level lvl: central differences at the texel centres, interpolated bilinearly.
    Texels next to an edge difference against the neighbour's border texels, so the slopes continue across face edges."""
    var ul = Int(d.hdr[unsafe_offset=3 * f + 1]); var vl = Int(d.hdr[unsafe_offset=3 * f + 2])
    var off = Int(d.chain_off[unsafe_offset=f])
    var l = 0; var cul = ul; var cvl = vl
    while l < lvl and (cul > 0 or cvl > 0):
        off += ((1 << cul) + 2) * ((1 << cvl) + 2)
        cul = max(cul - 1, 0); cvl = max(cvl - 1, 0); l += 1
    var w = 1 << cul; var h = 1 << cvl
    var fx = min(max(u * Float32(w) - Float32(0.5), Float32(-0.5)), Float32(w) - Float32(0.5))
    var fy = min(max(v * Float32(h) - Float32(0.5), Float32(-0.5)), Float32(h) - Float32(0.5))
    var x0 = Int(floor(fx)); var y0 = Int(floor(fy))
    var tx = fx - Float32(x0); var ty = fy - Float32(y0)
    var su = Float32(0); var sv = Float32(0)
    for k in range(4):
        var x = min(max(x0 + (k % 2), -1), w); var y = min(max(y0 + (k // 2), -1), h)
        var wk = (tx if k % 2 == 1 else Float32(1) - tx) * (ty if k >= 2 else Float32(1) - ty)
        # Central differences inside the bordered level; one-sided on the border texels themselves.
        var xa = max(x - 1, -1); var xb = min(x + 1, w); var ya = max(y - 1, -1); var yb = min(y + 1, h)
        su += wk * (_chain_h(d, off, w, xb, y) - _chain_h(d, off, w, xa, y)) / Float32(xb - xa)
        sv += wk * (_chain_h(d, off, w, x, yb) - _chain_h(d, off, w, x, ya)) / Float32(yb - ya)
    return (su * Float32(w), sv * Float32(h))   # per texel to per face unit


def displaced_normal(d: DispSurface, pi: Int, s: Float32, t: Float32, step: Float32) -> Vec3f:
    """The displaced surface's normal at (s, t) with slopes from the prefiltered level whose texels match `step` (the leaf
    square's size in patch units), blended between the two nearest levels: Su = Pu + hu N, Sv = Pv + hv N."""
    var e = patch_eval(patch_at(d, pi, s, t), s, t)
    var n = cross(e[1], e[2])
    var ln = sqrt(dot(n, n))
    if ln <= Float32(0):
        return Vec3f(0, 0, 1)
    n = n * (Float32(1) / ln)
    if d.scale <= Float32(0):
        return n
    var f = Int(d.face[unsafe_offset=pi]); var u0 = d.sq[unsafe_offset=3 * pi]; var v0 = d.sq[unsafe_offset=3 * pi + 1]; var sz = d.sq[unsafe_offset=3 * pi + 2]
    var ul = Int(d.hdr[unsafe_offset=3 * f + 1]); var vl = Int(d.hdr[unsafe_offset=3 * f + 2])
    var res = Float32(1 << max(ul, vl))
    var lod = min(max(log2(max(step * sz * res, Float32(1e-6))), Float32(0)), Float32(max(ul, vl)))
    var l0 = Int(floor(lod)); var fr = lod - Float32(l0)
    var u = u0 + s * sz; var v = v0 + t * sz
    var g = _level_slope(d, f, l0, u, v)
    if fr > Float32(0) and l0 < max(ul, vl):
        var g1 = _level_slope(d, f, l0 + 1, u, v)
        g = (g[0] + (g1[0] - g[0]) * fr, g[1] + (g1[1] - g[1]) * fr)
    var hs = g[0] * sz * d.scale; var ht = g[1] * sz * d.scale   # dh/ds = size * dh/du
    var su = e[1] + n * hs; var sv = e[2] + n * ht
    var dn = cross(su, sv)
    var l2 = sqrt(dot(dn, dn))
    return dn * (Float32(1) / l2) if l2 > Float32(0) else n


def displaced_point(d: DispSurface, pi: Int, s: Float32, t: Float32) -> Vec3f:
    """The displaced surface at (s, t) (the tessellated reference)."""
    var e = patch_eval(patch_at(d, pi, s, t), s, t)
    var n = cross(e[1], e[2])
    var ln = sqrt(dot(n, n))
    if ln <= Float32(0) or d.scale <= Float32(0):
        return e[0]
    var f = Int(d.face[unsafe_offset=pi]); var u0 = d.sq[unsafe_offset=3 * pi]; var v0 = d.sq[unsafe_offset=3 * pi + 1]; var sz = d.sq[unsafe_offset=3 * pi + 2]
    return e[0] + n * (face_height(d, f, u0 + s * sz, v0 + t * sz) * d.scale / ln)


def level_height(d: DispSurface, f: Int, lvl: Int, u: Float32, v: Float32) -> Float32:
    """The bilinear height (bytes) at (u, v) of face f's bordered level lvl (0: finest)."""
    var ul = Int(d.hdr[unsafe_offset=3 * f + 1]); var vl = Int(d.hdr[unsafe_offset=3 * f + 2])
    var off = Int(d.chain_off[unsafe_offset=f])
    var l = 0; var cul = ul; var cvl = vl
    while l < lvl and (cul > 0 or cvl > 0):
        off += ((1 << cul) + 2) * ((1 << cvl) + 2)
        cul = max(cul - 1, 0); cvl = max(cvl - 1, 0); l += 1
    var w = 1 << cul; var h = 1 << cvl
    var x = min(max(u * Float32(w) - Float32(0.5), Float32(-0.5)), Float32(w) - Float32(0.5))
    var y = min(max(v * Float32(h) - Float32(0.5), Float32(-0.5)), Float32(h) - Float32(0.5))
    var x0 = min(Int(floor(x)), w - 1); var y0 = min(Int(floor(y)), h - 1)
    var fx = x - Float32(x0); var fy = y - Float32(y0)
    var a = _chain_h(d, off, w, x0, y0) * (Float32(1) - fx) + _chain_h(d, off, w, x0 + 1, y0) * fx
    var b = _chain_h(d, off, w, x0, y0 + 1) * (Float32(1) - fx) + _chain_h(d, off, w, x0 + 1, y0 + 1) * fx
    return a * (Float32(1) - fy) + b * fy


def displaced_point_n(d: DispSurface, pi: Int, s: Float32, t: Float32, n: Int) -> Vec3f:
    """displaced_point with heights from the level whose texels match an n x n grid on the patch (a tessellation's)."""
    var e = patch_eval(patch_at(d, pi, s, t), s, t)
    var nn = cross(e[1], e[2])
    var ln = sqrt(dot(nn, nn))
    if ln <= Float32(0) or d.scale <= Float32(0):
        return e[0]
    var f = Int(d.face[unsafe_offset=pi]); var sz = d.sq[unsafe_offset=3 * pi + 2]
    var ul = Int(d.hdr[unsafe_offset=3 * f + 1]); var vl = Int(d.hdr[unsafe_offset=3 * f + 2])
    var lvl = Int(min(max(floor(log2(max(sz * Float32(1 << max(ul, vl)) / Float32(n), Float32(1e-6)))), Float32(0)), Float32(max(ul, vl))))
    var h = level_height(d, f, lvl, d.sq[unsafe_offset=3 * pi] + s * sz, d.sq[unsafe_offset=3 * pi + 1] + t * sz)
    return e[0] + nn * (h * d.scale / ln)


@always_inline
def _rgb_texel(rgb: _U8, off: Int, w: Int, x: Int, y: Int) -> Vec3f:
    var i = 3 * (off + (y + 1) * (w + 2) + x + 1)
    return Vec3f(Float32(Int(rgb[unsafe_offset=i])), Float32(Int(rgb[unsafe_offset=i + 1])), Float32(Int(rgb[unsafe_offset=i + 2])))


def chain_color(d: DispSurface, rgb: _U8, rgb_off: _I32, pi: Int, s: Float32, t: Float32, step: Float32) -> Vec3f:
    """Linear colour at (s, t) of patch pi from bordered RGB8 chains with the displacement's faces (src/oiio/disp.cc), bilinear
    at the level whose texels match `step` (the leaf square's size in patch units); the borders hide the face edges."""
    var f = Int(d.face[unsafe_offset=pi]); var u0 = d.sq[unsafe_offset=3 * pi]; var v0 = d.sq[unsafe_offset=3 * pi + 1]; var sz = d.sq[unsafe_offset=3 * pi + 2]
    var ul = Int(d.hdr[unsafe_offset=3 * f + 1]); var vl = Int(d.hdr[unsafe_offset=3 * f + 2])
    var lvl = Int(min(max(log2(max(step * sz * Float32(1 << max(ul, vl)), Float32(1e-6))), Float32(0)), Float32(max(ul, vl))))
    var off = Int(rgb_off[unsafe_offset=f])
    var l = 0; var cul = ul; var cvl = vl
    while l < lvl and (cul > 0 or cvl > 0):
        off += ((1 << cul) + 2) * ((1 << cvl) + 2)
        cul = max(cul - 1, 0); cvl = max(cvl - 1, 0); l += 1
    var w = 1 << cul; var h = 1 << cvl
    var fx = min(max((u0 + s * sz) * Float32(w) - Float32(0.5), Float32(-0.5)), Float32(w) - Float32(0.5))
    var fy = min(max((v0 + t * sz) * Float32(h) - Float32(0.5), Float32(-0.5)), Float32(h) - Float32(0.5))
    var x0 = Int(floor(fx)); var y0 = Int(floor(fy))
    var tx = fx - Float32(x0); var ty = fy - Float32(y0)
    var c = (_rgb_texel(rgb, off, w, x0, y0) * (Float32(1) - tx) + _rgb_texel(rgb, off, w, x0 + 1, y0) * tx) * (Float32(1) - ty) \
          + (_rgb_texel(rgb, off, w, x0, y0 + 1) * (Float32(1) - tx) + _rgb_texel(rgb, off, w, x0 + 1, y0 + 1) * tx) * ty
    var k = Float32(1) / Float32(255)
    return Vec3f(pow(c.x * k, Float32(2.2)), pow(c.y * k, Float32(2.2)), pow(c.z * k, Float32(2.2)))   # 8-bit colour is gamma 2.2


@fieldwise_init
struct SubdivSurface(TrivialRegisterPassable):
    """A Ptex-displaced mesh on its TriangleMesh (the cage, two triangles a Ptex face): the surface and, per Ptex face f,
    its patches face_patch[face_off[f] .. face_off[f + 1]], to find a hit's patch from its face and (u, v)."""
    var d: DispSurface
    var face_off: _I32
    var face_patch: _I32
    var nf: Int64       # array sizes, for the upload (gpu/scene.mojo)
    var npyr: Int64
    var nchain: Int64


def subdiv_patch_at(sub: SubdivSurface, f: Int, u: Float32, v: Float32) -> Tuple[Int, Float32, Float32]:
    """The patch of Ptex face f that holds face (u, v), and (s, t) on it."""
    var k0 = Int(sub.face_off[unsafe_offset=f]); var k1 = Int(sub.face_off[unsafe_offset=f + 1])
    var p = Int(sub.face_patch[unsafe_offset=k0])
    for k in range(k0, k1):
        var q = Int(sub.face_patch[unsafe_offset=k])
        var u0 = sub.d.sq[unsafe_offset=3 * q]; var v0 = sub.d.sq[unsafe_offset=3 * q + 1]; var sz = sub.d.sq[unsafe_offset=3 * q + 2]
        if u >= u0 and u <= u0 + sz and v >= v0 and v <= v0 + sz:
            p = q
            break
    var u0 = sub.d.sq[unsafe_offset=3 * p]; var v0 = sub.d.sq[unsafe_offset=3 * p + 1]; var sz = sub.d.sq[unsafe_offset=3 * p + 2]
    return (p, min(max((u - u0) / sz, Float32(0)), Float32(1)), min(max((v - v0) / sz, Float32(0)), Float32(1)))


def subdiv_hit_normal(sub: SubdivSurface, f: Int, u: Float32, v: Float32, t_hit: Float32) -> Vec3f:
    """The displaced surface's normal at Ptex face f's (u, v), its heights filtered to a pixel at distance t_hit."""
    var r = subdiv_patch_at(sub, f, u, v)
    var b = sub.d.patches[unsafe_offset=r[0]]
    var e = Vec3f(b.x.reduce_max() - b.x.reduce_min(), b.y.reduce_max() - b.y.reduce_min(), b.z.reduce_max() - b.z.reduce_min())
    var step = max(sub.d.pixel * t_hit / max(sqrt(dot(e, e)), Float32(1e-12)), Float32(1.0 / 512.0))
    return displaced_normal(sub.d, r[0], r[1], r[2], step)


def subdiv_tri_normal(sub: Pointer[SubdivSurface, MutUntrackedOrigin], bv: Int, b1: Float32, b2: Float32, t_hit: Float32) -> Vec3f:
    """The normal at a hit reported as cage triangle bv // 3 (scene/accel/xert.mojo): Ptex face tri // 2, face (u, v) from the
    barycentrics, (0,0) (1,0) (1,1) for the face's first triangle and (0,0) (1,1) (0,1) for its second."""
    var tri = bv // 3
    var upper = tri % 2 == 1
    return subdiv_hit_normal(sub[unsafe_offset=0], tri // 2, b1 if upper else b1 + b2, b1 + b2 if upper else b2, t_hit)
