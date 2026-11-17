# Displaced Catmull-Clark surfaces traced without tessellation: Binder and Keller's stackless patch subdivision, with each
# sub-patch's box widened by its height interval (per-face Ptex min/max pyramids, src/oiio/disp.cc) times the range of its
# unit normal (interval arithmetic over the derivative nets, after Munkberg et al. 2010). Leaves are boxes no larger than
# the camera's footprint at them, so every ray sees the same surface and neighbouring depths cannot open cracks.
from std.bit import count_leading_zeros
from std.math import max, min, sqrt, abs, floor
from gonzales.core.geometry import Vec3f, cross, dot
from gonzales.scene.shapes.patches import BezierPatch, patch_eval, _crop, _half, _domain, PATCH_MAX_DEPTH

comptime _U8 = Pointer[UInt8, MutUntrackedOrigin]
comptime _I32 = Pointer[Int32, MutUntrackedOrigin]
comptime _F32 = Pointer[Float32, MutUntrackedOrigin]


@fieldwise_init
struct DispSurface(TrivialRegisterPassable):
    """Patches with their Ptex face and square (u0, v0, size), the faces' pyramids, the height scale and the camera."""
    var patches: Pointer[BezierPatch, MutUntrackedOrigin]
    var sq: _F32
    var face: _I32
    var hdr: _I32        # per face: pyramid offset, log2 width, log2 height
    var pyr: _U8
    var scale: Float32   # world units per height byte
    var cam: Vec3f
    var pixel: Float32   # footprint per unit distance (pixel angle)


@always_inline
def _texel(d: DispSurface, off: Int, w: Int, x: Int, y: Int) -> Float32:
    return Float32(Int(d.pyr[unsafe_offset=off + y * w + x]))


def face_height(d: DispSurface, f: Int, u: Float32, v: Float32) -> Float32:
    """The bilinear height (in bytes) at (u, v) of face f's finest level."""
    var off = Int(d.hdr[unsafe_offset=3 * f]); var w = 1 << Int(d.hdr[unsafe_offset=3 * f + 1]); var h = 1 << Int(d.hdr[unsafe_offset=3 * f + 2])
    var x = min(max(u * Float32(w) - Float32(0.5), Float32(0)), Float32(w - 1))
    var y = min(max(v * Float32(h) - Float32(0.5), Float32(0)), Float32(h - 1))
    var x0 = min(Int(x), w - 1); var y0 = min(Int(y), h - 1)
    var x1 = min(x0 + 1, w - 1); var y1 = min(y0 + 1, h - 1)
    var fx = x - Float32(x0); var fy = y - Float32(y0)
    var a = _texel(d, off, w, x0, y0) * (Float32(1) - fx) + _texel(d, off, w, x1, y0) * fx
    var b = _texel(d, off, w, x0, y1) * (Float32(1) - fx) + _texel(d, off, w, x1, y1) * fx
    return a * (Float32(1) - fy) + b * fy


def height_range(d: DispSurface, f: Int, u0: Float32, u1: Float32, v0: Float32, v1: Float32) -> Tuple[Float32, Float32]:
    """Bounds (in bytes) of the bilinear heights over [u0, u1] x [v0, v1] of face f: exact below two texels, else the pyramid."""
    var off = Int(d.hdr[unsafe_offset=3 * f]); var ul = Int(d.hdr[unsafe_offset=3 * f + 1]); var vl = Int(d.hdr[unsafe_offset=3 * f + 2])
    var w = 1 << ul; var h = 1 << vl
    var x0 = min(max(Int(floor(u0 * Float32(w) - Float32(0.5))), 0), w - 1)
    var x1 = min(max(Int(floor(u1 * Float32(w) - Float32(0.5))) + 1, 0), w - 1)
    var y0 = min(max(Int(floor(v0 * Float32(h) - Float32(0.5))), 0), h - 1)
    var y1 = min(max(Int(floor(v1 * Float32(h) - Float32(0.5))) + 1, 0), h - 1)
    if x1 - x0 <= 2 and y1 - y0 <= 2:   # a bilinear patch's extremes lie on its corners and the texel centres inside
        var lo = Float32(1e30); var hi = Float32(-1e30)
        for c in range(4):
            var hv = face_height(d, f, u0 if c % 2 == 0 else u1, v0 if c < 2 else v1)
            lo = min(lo, hv); hi = max(hi, hv)
        for y in range(y0, y1 + 1):
            var cy = (Float32(y) + Float32(0.5)) / Float32(h)
            if cy < v0 or cy > v1:
                continue
            for x in range(x0, x1 + 1):
                var cx = (Float32(x) + Float32(0.5)) / Float32(w)
                if cx < u0 or cx > u1:
                    continue
                var t = _texel(d, off, w, x, y)
                lo = min(lo, t); hi = max(hi, t)
        return (lo, hi)
    var lvl = 0
    while (x1 >> lvl) - (x0 >> lvl) > 1 or (y1 >> lvl) - (y0 >> lvl) > 1:
        lvl += 1
    if lvl == 0:
        lvl = 1
    var o = off + w * h   # level 1's min cells follow the heights
    var cw = max(w // 2, 1); var ch = max(h // 2, 1)
    for _ in range(1, lvl):
        o += 2 * cw * ch
        cw = max(cw // 2, 1); ch = max(ch // 2, 1)
    var lo = Float32(1e30); var hi = Float32(-1e30)
    for y in range(min(y0 >> lvl, ch - 1), min(y1 >> lvl, ch - 1) + 1):
        for x in range(min(x0 >> lvl, cw - 1), min(x1 >> lvl, cw - 1) + 1):
            lo = min(lo, Float32(Int(d.pyr[unsafe_offset=o + y * cw + x])))
            hi = max(hi, Float32(Int(d.pyr[unsafe_offset=o + cw * ch + y * cw + x])))
    return (lo, hi)


@always_inline
def _imul(a0: Float32, a1: Float32, b0: Float32, b1: Float32) -> Tuple[Float32, Float32]:
    var p = a0 * b0; var q = a0 * b1; var r = a1 * b0; var s = a1 * b1
    return (min(min(p, q), min(r, s)), max(max(p, q), max(r, s)))


@always_inline
def _unit_comp(a: Float32, b: Float32, m: Float32, big: Float32) -> Tuple[Float32, Float32]:
    """Bounds of N_k / |N| for N_k in [a, b] and |N| in [m, big]."""
    return (max(min(a / m, a / big), Float32(-1)), min(max(b / m, b / big), Float32(1)))


def _unit_normal_range(p: BezierPatch) -> Tuple[Vec3f, Vec3f]:
    """Component bounds of N / |N| over the patch, N = Pu x Pv, from the boxes of the derivative nets."""
    var ux0 = Float32(1e30); var ux1 = Float32(-1e30); var uy0 = Float32(1e30); var uy1 = Float32(-1e30); var uz0 = Float32(1e30); var uz1 = Float32(-1e30)
    var vx0 = Float32(1e30); var vx1 = Float32(-1e30); var vy0 = Float32(1e30); var vy1 = Float32(-1e30); var vz0 = Float32(1e30); var vz1 = Float32(-1e30)
    for j in range(4):
        for i in range(3):
            var a = j * 4 + i; var b = i * 4 + j   # u differences along rows, v differences along columns
            ux0 = min(ux0, p.x[a + 1] - p.x[a]); ux1 = max(ux1, p.x[a + 1] - p.x[a])
            uy0 = min(uy0, p.y[a + 1] - p.y[a]); uy1 = max(uy1, p.y[a + 1] - p.y[a])
            uz0 = min(uz0, p.z[a + 1] - p.z[a]); uz1 = max(uz1, p.z[a + 1] - p.z[a])
            vx0 = min(vx0, p.x[b + 4] - p.x[b]); vx1 = max(vx1, p.x[b + 4] - p.x[b])
            vy0 = min(vy0, p.y[b + 4] - p.y[b]); vy1 = max(vy1, p.y[b + 4] - p.y[b])
            vz0 = min(vz0, p.z[b + 4] - p.z[b]); vz1 = max(vz1, p.z[b + 4] - p.z[b])
    var a1 = _imul(uy0, uy1, vz0, vz1); var a2 = _imul(uz0, uz1, vy0, vy1)
    var b1 = _imul(uz0, uz1, vx0, vx1); var b2 = _imul(ux0, ux1, vz0, vz1)
    var c1 = _imul(ux0, ux1, vy0, vy1); var c2 = _imul(uy0, uy1, vx0, vx1)
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


def _sampled_normal_range(p: BezierPatch) -> Tuple[Vec3f, Vec3f]:
    """Unit normals at a 3 x 3 grid just inside the patch, widened by a margin."""
    var lo = Vec3f(1, 1, 1); var hi = Vec3f(-1, -1, -1)
    for k in range(9):
        var e = patch_eval(p, Float32(0.02) + Float32(0.48) * Float32(k % 3), Float32(0.02) + Float32(0.48) * Float32(k // 3))
        var n = cross(e[1], e[2])
        var l = sqrt(dot(n, n))
        if l <= Float32(0):
            return (Vec3f(-1, -1, -1), Vec3f(1, 1, 1))
        n = n * (Float32(1) / l)
        lo = Vec3f(min(lo.x, n.x), min(lo.y, n.y), min(lo.z, n.z)); hi = Vec3f(max(hi.x, n.x), max(hi.y, n.y), max(hi.z, n.z))
    var m = Float32(0.1)
    return (Vec3f(max(lo.x - m, Float32(-1)), max(lo.y - m, Float32(-1)), max(lo.z - m, Float32(-1))),
            Vec3f(min(hi.x + m, Float32(1)), min(hi.y + m, Float32(1)), min(hi.z + m, Float32(1))))


def displaced_box(d: DispSurface, pi: Int, p: BezierPatch, s0: Float32, s1: Float32, t0: Float32, t1: Float32,
                  parent_lo: Vec3f = Vec3f(-1, -1, -1), parent_hi: Vec3f = Vec3f(1, 1, 1)) -> Tuple[Vec3f, Vec3f, Vec3f, Vec3f]:
    """The box of patch pi's (cropped to p = [s0, s1] x [t0, t1]) displaced surface, and its unit-normal range (within the
    parent's: tiny crops' derivative nets are rounding noise)."""
    var lo = Vec3f(p.x.reduce_min(), p.y.reduce_min(), p.z.reduce_min())
    var hi = Vec3f(p.x.reduce_max(), p.y.reduce_max(), p.z.reduce_max())
    if d.scale <= Float32(0):
        return (lo, hi, parent_lo, parent_hi)
    var u0 = d.sq[unsafe_offset=3 * pi]; var v0 = d.sq[unsafe_offset=3 * pi + 1]; var sz = d.sq[unsafe_offset=3 * pi + 2]
    var hr = height_range(d, Int(d.face[unsafe_offset=pi]), u0 + s0 * sz, u0 + s1 * sz, v0 + t0 * sz, v0 + t1 * sz)
    var h0 = hr[0] * d.scale; var h1 = hr[1] * d.scale
    var nc = _unit_normal_range(p)
    if (s1 - s0) * (t1 - t0) < Float32(1.0 / 4096.0) and nc[1].x - nc[0].x + nc[1].y - nc[0].y + nc[1].z - nc[0].z > Float32(3):
        nc = _sampled_normal_range(p)   # Pu x Pv vanishes at extraordinary corners: not rigorous, see the prototype's notes
    var nlo = Vec3f(max(nc[0].x, parent_lo.x), max(nc[0].y, parent_lo.y), max(nc[0].z, parent_lo.z))
    var nhi = Vec3f(min(nc[1].x, parent_hi.x), min(nc[1].y, parent_hi.y), min(nc[1].z, parent_hi.z))
    var nr = (nlo, nhi)
    var ox = _imul(h0, h1, nr[0].x, nr[1].x); var oy = _imul(h0, h1, nr[0].y, nr[1].y); var oz = _imul(h0, h1, nr[0].z, nr[1].z)
    return (Vec3f(lo.x + ox[0], lo.y + oy[0], lo.z + oz[0]), Vec3f(hi.x + ox[1], hi.y + oy[1], hi.z + oz[1]), nlo, nhi)


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


def intersect_displaced(d: DispSurface, pi: Int, org: Vec3f, dir: Vec3f, t_max: Float32) -> Tuple[Bool, Float32, Float32, Float32, Int, Float32]:
    """The nearest hit of patch pi before t_max: (hit, t, s, t, steps, the leaf square's size)."""
    var src = d.patches[unsafe_offset=pi]
    var o = BezierPatch(src.x - org.x, src.y - org.y, src.z - org.z)   # anchored at the ray origin
    var cam = d.cam - org
    var inv = Vec3f(Float32(1) / dir.x, Float32(1) / dir.y, Float32(1) / dir.z)
    var best = t_max; var hit = False; var hs = Float32(0); var ht = Float32(0); var hsz = Float32(1)
    var nlo = Array[Vec3f, PATCH_MAX_DEPTH + 1](fill=Vec3f(-1, -1, -1)); var nhi = Array[Vec3f, PATCH_MAX_DEPTH + 1](fill=Vec3f(1, 1, 1))
    var p = o
    var depth = 0; var steps = 0
    var path = UInt64(0); var pend = UInt64(0)
    var dom = (Float32(0), Float32(0), Float32(1), Float32(1))
    var cur = displaced_box(d, pi, p, Float32(0), Float32(1), Float32(0), Float32(1))
    nlo[0] = cur[2]; nhi[0] = cur[3]
    var cur_hit = _slab(cur[0], cur[1], inv, best)
    while True:
        steps += 1
        var descend = False
        if cur_hit[0]:
            var sz = _leaf_size(cur[0], cur[1], cam, d.pixel)
            if depth >= PATCH_MAX_DEPTH or sz[0] <= sz[1]:
                if cur_hit[1] < best:
                    best = cur_hit[1]; hit = True
                    hs = dom[0] + Float32(0.5) * dom[2]; ht = dom[1] + Float32(0.5) * dom[3]; hsz = max(dom[2], dom[3])
            else:
                var along_u = depth % 2 == 0
                var lo_p = _half(p, along_u, False); var up_p = _half(p, along_u, True)
                var hu0 = dom[0]; var hv0 = dom[1]; var hdu = dom[2]; var hdv = dom[3]
                var l_dom = (hu0, hv0, hdu * Float32(0.5) if along_u else hdu, hdv if along_u else hdv * Float32(0.5))
                var u_dom = (hu0 + hdu * Float32(0.5) if along_u else hu0, hv0 if along_u else hv0 + hdv * Float32(0.5), l_dom[2], l_dom[3])
                var lb = displaced_box(d, pi, lo_p, l_dom[0], l_dom[0] + l_dom[2], l_dom[1], l_dom[1] + l_dom[3], nlo[depth], nhi[depth])
                var ub = displaced_box(d, pi, up_p, u_dom[0], u_dom[0] + u_dom[2], u_dom[1], u_dom[1] + u_dom[3], nlo[depth], nhi[depth])
                var h0 = _slab(lb[0], lb[1], inv, best); var h1 = _slab(ub[0], ub[1], inv, best)
                if h0[0] or h1[0]:
                    var upper_first = h1[0] and (not h0[0] or h1[1] < h0[1])
                    if h0[0] and h1[0]:
                        pend |= UInt64(1) << UInt64(depth)
                    if upper_first:
                        path |= UInt64(1) << UInt64(depth)
                        p = up_p; dom = u_dom; cur = ub; cur_hit = h1
                    else:
                        p = lo_p; dom = l_dom; cur = lb; cur_hit = h0
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
        p = _crop(o, dd[0], dd[0] + dd[2], dd[1], dd[1] + dd[3])
        cur = displaced_box(d, pi, p, dd[0], dd[0] + dd[2], dd[1], dd[1] + dd[3], nlo[level], nhi[level])
        nlo[min(depth, PATCH_MAX_DEPTH)] = cur[2]; nhi[min(depth, PATCH_MAX_DEPTH)] = cur[3]
        cur_hit = _slab(cur[0], cur[1], inv, best)
    return (hit, best, hs, ht, steps, hsz)


def displaced_normal(d: DispSurface, pi: Int, s: Float32, t: Float32, step: Float32) -> Vec3f:
    """The displaced surface's normal at (s, t): Su = Pu + hu N, Sv = Pv + hv N, heights differenced over `step`."""
    var e = patch_eval(d.patches[unsafe_offset=pi], s, t)
    var n = cross(e[1], e[2])
    var ln = sqrt(dot(n, n))
    if ln <= Float32(0):
        return Vec3f(0, 0, 1)
    n = n * (Float32(1) / ln)
    if d.scale <= Float32(0):
        return n
    var f = Int(d.face[unsafe_offset=pi]); var u0 = d.sq[unsafe_offset=3 * pi]; var v0 = d.sq[unsafe_offset=3 * pi + 1]; var sz = d.sq[unsafe_offset=3 * pi + 2]
    var hs = (face_height(d, f, u0 + (s + step) * sz, v0 + t * sz) - face_height(d, f, u0 + (s - step) * sz, v0 + t * sz)) / (Float32(2) * step)
    var ht = (face_height(d, f, u0 + s * sz, v0 + (t + step) * sz) - face_height(d, f, u0 + s * sz, v0 + (t - step) * sz)) / (Float32(2) * step)
    var su = e[1] + n * (hs * d.scale); var sv = e[2] + n * (ht * d.scale)
    var dn = cross(su, sv)
    var l2 = sqrt(dot(dn, dn))
    return dn * (Float32(1) / l2) if l2 > Float32(0) else n


def displaced_point(d: DispSurface, pi: Int, s: Float32, t: Float32) -> Vec3f:
    """The displaced surface at (s, t) (the tessellated reference)."""
    var e = patch_eval(d.patches[unsafe_offset=pi], s, t)
    var n = cross(e[1], e[2])
    var ln = sqrt(dot(n, n))
    if ln <= Float32(0) or d.scale <= Float32(0):
        return e[0]
    var f = Int(d.face[unsafe_offset=pi]); var u0 = d.sq[unsafe_offset=3 * pi]; var v0 = d.sq[unsafe_offset=3 * pi + 1]; var sz = d.sq[unsafe_offset=3 * pi + 2]
    return e[0] + n * (face_height(d, f, u0 + s * sz, v0 + t * sz) * d.scale / ln)
