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
    var sub_levels: Int32  # the BVH's primitives: a patch's sub-squares at this halving depth (4: 4 x 4, 6: 8 x 8, 8: 16 x 16)
    var leaf_mode: Int32   # 0: box leaves no larger than the footprint; 1: exact bilinear leaves at a texel or a pixel; 2: at a texel only


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
    if x1 - x0 <= 4 and y1 - y0 <= 4:   # exact: piecewise bilinear extremes lie at the corners, where texel-centre lines cross the edges, and at centres inside
        var lo = Float32(1e30); var hi = Float32(-1e30)
        for c in range(4):
            var hv = face_height(d, f, u0 if c % 2 == 0 else u1, v0 if c < 2 else v1)
            lo = min(lo, hv); hi = max(hi, hv)
        var fy0 = min(max(v0 * Float32(h) - Float32(0.5), Float32(-0.5)), Float32(h) - Float32(0.5))
        var fy1 = min(max(v1 * Float32(h) - Float32(0.5), Float32(-0.5)), Float32(h) - Float32(0.5))
        var fx0 = min(max(u0 * Float32(w) - Float32(0.5), Float32(-0.5)), Float32(w) - Float32(0.5))
        var fx1 = min(max(u1 * Float32(w) - Float32(0.5), Float32(-0.5)), Float32(w) - Float32(0.5))
        for x in range(max(x0, -1), min(x1, w) + 1):   # centre columns between the edges: their crossings with v0 and v1
            if Float32(x) <= fx0 or Float32(x) >= fx1:
                continue
            for e in range(2):
                var fy = fy0 if e == 0 else fy1
                var yb = min(Int(floor(fy)), h - 1); var ty = fy - Float32(yb)
                var hv = _texel(d, off, w, x, yb) * (Float32(1) - ty) + _texel(d, off, w, x, yb + 1) * ty
                lo = min(lo, hv); hi = max(hi, hv)
        for y in range(max(y0, -1), min(y1, h) + 1):   # centre rows: crossings with u0 and u1, and the centres inside
            if Float32(y) <= fy0 or Float32(y) >= fy1:
                continue
            for e in range(2):
                var fx = fx0 if e == 0 else fx1
                var xb = min(Int(floor(fx)), w - 1); var tx = fx - Float32(xb)
                var hv = _texel(d, off, w, xb, y) * (Float32(1) - tx) + _texel(d, off, w, xb + 1, y) * tx
                lo = min(lo, hv); hi = max(hi, hv)
            for x in range(max(x0, -1), min(x1, w) + 1):
                if Float32(x) <= fx0 or Float32(x) >= fx1:
                    continue
                var t = _texel(d, off, w, x, y)
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
def sub_count(d: DispSurface) -> Int:
    """Primitives (sub-squares) a patch."""
    return 1 << Int(d.sub_levels)


@always_inline
def sub_square(d: DispSurface, k: Int) -> Tuple[Float32, Float32, Float32, UInt64]:
    """Sub-square k of a patch (k = row * n + column): its corner (u0, v0), its size, and the halving path that reaches it."""
    var la = Int(d.sub_levels) // 2; var n = 1 << la
    var ku = k % n; var kv = k // n
    var path = UInt64(0)
    for i in range(la):   # even depths split u, odd ones v, the coarsest first
        path |= UInt64((ku >> (la - 1 - i)) & 1) << UInt64(2 * i)
        path |= UInt64((kv >> (la - 1 - i)) & 1) << UInt64(2 * i + 1)
    var sz = Float32(1) / Float32(n)
    return (Float32(ku) * sz, Float32(kv) * sz, sz, path)


def sub_box(d: DispSurface, prim: Int) -> Tuple[Vec3f, Vec3f]:
    """The world box of primitive prim = patch * sub_count + sub-square."""
    var pi = prim // sub_count(d); var sq = sub_square(d, prim % sub_count(d))
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


def intersect_displaced(d: DispSurface, prim: Int, org: Vec3f, dir: Vec3f, t_max: Float32) -> Tuple[Bool, Float32, Float32, Float32, Int, Float32]:
    """The nearest hit of primitive prim (patch * sub_count + sub-square) before t_max: (hit, t, s, t, steps, the leaf square's size).
    Box leaves end where the displaced box is no larger than the footprint; exact leaves where the undisplaced square reaches a
    texel or the footprint, and intersect the bilinear patch through its four displaced corners, so the depth does not grow with
    the displacement's amplitude."""
    var pi = prim // sub_count(d); var sq = sub_square(d, prim % sub_count(d))
    var src = d.patches[unsafe_offset=pi]
    var o = BezierPatch(src.x - org.x, src.y - org.y, src.z - org.z)   # anchored at the ray origin
    var cam = d.cam - org
    var inv = Vec3f(Float32(1) / dir.x, Float32(1) / dir.y, Float32(1) / dir.z)
    var best = t_max; var hit = False; var hs = Float32(0); var ht = Float32(0); var hsz = Float32(1)
    var nlo = Array[Vec3f, PATCH_MAX_DEPTH + 1](fill=Vec3f(-1, -1, -1)); var nhi = Array[Vec3f, PATCH_MAX_DEPTH + 1](fill=Vec3f(1, 1, 1))
    var depth = Int(d.sub_levels); var steps = 0
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
                leaf = max(dom[2], dom[3]) * texels <= Float32(1) or (d.leaf_mode == 1 and cur[4] <= sz[1])
            if depth >= PATCH_MAX_DEPTH or leaf:
                if d.leaf_mode == 0:
                    if cur_hit[1] < best:
                        best = cur_hit[1]; hit = True
                        hs = dom[0] + Float32(0.5) * dom[2]; ht = dom[1] + Float32(0.5) * dom[3]; hsz = max(dom[2], dom[3])
                else:   # corners from global (s, t): neighbouring leaves of one depth share them
                    var s0 = dom[0]; var s1 = dom[0] + dom[2]; var t0 = dom[1]; var t1 = dom[1] + dom[3]
                    var sm = s0 + Float32(0.5) * dom[2]; var tm = t0 + Float32(0.5) * dom[3]
                    var cs = SIMD[DType.float32, 8](s0, s1, s1, s0, sm, s1, sm, s0)   # corners 00 10 11 01, then the edge midpoints
                    var ct = SIMD[DType.float32, 8](t0, t0, t1, t1, t0, tm, t1, tm)
                    var cp = Array[Vec3f, 8](fill=Vec3f(0, 0, 0))
                    for k in range(8):
                        cp[k] = displaced_point(d, pi, cs[k], ct[k]) - org
                    var bh = _bilinear_hit(cp[0], cp[1], cp[2], cp[3], dir, best)
                    if bh[0]:
                        best = bh[1]; hit = True
                        hs = s0 + bh[2] * dom[2]; ht = t0 + bh[3] * dom[3]; hsz = max(dom[2], dom[3])
                    # Edge fans: a neighbour one level finer bends its edge through the midpoint; these triangles close the T gap.
                    for k in range(4):
                        var ka = k; var kb = (k + 1) % 4; var km = 4 + k
                        var th = _tri_hit(cp[ka], cp[km], cp[kb], dir, best)
                        if th[0]:
                            best = th[1]; hit = True
                            var w0 = Float32(1) - th[2] - th[3]
                            hs = w0 * cs[ka] + th[2] * cs[km] + th[3] * cs[kb]; ht = w0 * ct[ka] + th[2] * ct[km] + th[3] * ct[kb]; hsz = max(dom[2], dom[3])
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
