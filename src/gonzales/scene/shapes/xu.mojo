# Xu, Zhou and Budge 2026 (Geometric queries on tessellation-free displaced surfaces, Algorithm 1) on the triangles and
# prisms of RMIP, for comparison: the ray's curved texture-space image cut into linear pieces (0.06 texel), each traced
# by a hierarchical DDA over our plane + residual nodes from 4 x 4-texel blocks up, the blocks' texel cells as two
# triangles each.

from std.math import max, min, sqrt, abs, floor, ceil
from gonzales.core.geometry import Vec3f, dot
from gonzales.scene.shapes.displaced_patches import DispSurface, NODE_B, prim_patch, sub_square, _tri_hit
from gonzales.scene.shapes.rmip import Rmip, _Tri, _prism_span, _P, _N, _inside, _det

comptime _THR = Float32(0.06)   # texels: the pieces' midpoint deviation


@always_inline
def _lvl_off(d: DispSurface, f: Int, ul: Int, vl: Int, k: Int) -> Int:
    """Face f's first node of level k, from the 4 x 4 blocks (k = 2) up."""
    var o = Int(d.hdr[unsafe_offset=3 * f])
    for l in range(2, k):
        o += NODE_B * max((1 << ul) >> l, 1) * max((1 << vl) >> l, 1)
    return o


@always_inline
def _udvt(t: _Tri, dir: Vec3f, H: Float32) -> SIMD[DType.float32, 4]:
    """Where the ray meets the plane of the triangle offset by H: a = U / D, b = V / D, t = T / D; (U, V, T, D)."""
    var ea = t.Pa + t.Na * H; var eb = t.Pb + t.Nb * H; var c = -(t.A + t.Nc * H); var m = -dir
    return SIMD[DType.float32, 4](_det(c, eb, m), _det(ea, c, m), _det(ea, eb, c), _det(ea, eb, m))


@always_inline
def _quad(f0: Float32, fm: Float32, f1: Float32) -> SIMD[DType.float32, 4]:
    """The quadratic through f(0), f(1/2), f(1): (c0, c1, c2)."""
    var c2 = Float32(2) * (f0 + f1) - Float32(4) * fm
    return SIMD[DType.float32, 4](f0, f1 - f0 - c2, c2, 0)


@always_inline
def _roots01(c: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 2]:
    """Roots of c0 + c1 s + c2 s^2 in (0, 1); -1 for none."""
    var r = SIMD[DType.float32, 2](-1)
    var sc = abs(c[0]) + abs(c[1]) + abs(c[2])
    if abs(c[2]) <= Float32(1e-7) * sc:
        if abs(c[1]) > Float32(1e-7) * sc:
            r[0] = -c[0] / c[1]
    else:
        var disc = c[1] * c[1] - Float32(4) * c[2] * c[0]
        if disc >= Float32(0):
            var q = Float32(-0.5) * (c[1] + (sqrt(disc) if c[1] >= Float32(0) else -sqrt(disc)))
            r[0] = q / c[2]; r[1] = c[0] / q if q != Float32(0) else Float32(-1)
    if not (r[0] > Float32(0) and r[0] < Float32(1)):
        r[0] = -1
    if not (r[1] > Float32(0) and r[1] < Float32(1)):
        r[1] = -1
    return r


@always_inline
def _xyht(t: _Tri, dir: Vec3f, h0: Float32, dh: Float32, s: Float32, w: Float32, h: Float32, scale: Float32) -> SIMD[DType.float32, 4]:
    """The mapped ray at s of the height interval: texel position (X, Y), height in bytes, ray t."""
    var H = h0 + dh * s
    var q = _udvt(t, dir, H); var id = Float32(1) / q[3]
    return SIMD[DType.float32, 4]((t.u0 + q[0] * id * t.side) * w, (t.v0 + q[1] * id * t.side) * h, H / scale, q[2] * id)


def _leaf(d: DispSurface, t: _Tri, f: Int, dir: Vec3f, p0: SIMD[DType.float32, 4], p1: SIMD[DType.float32, 4], la: Float32, lb: Float32,
          bx0: Float32, by0: Float32, tw: Float32, th: Float32, pad: Float32, best: Float32, mut steps: Int, st: Int) -> SIMD[DType.float32, 4]:
    """A 4 x 4 block's texel cells near the piece over [la, lb], each two triangles through its displaced corners: the
    nearest hit (t, a, b, hit)."""
    var w = 1 << Int(d.hdr[unsafe_offset=3 * f + 1]); var h = 1 << Int(d.hdr[unsafe_offset=3 * f + 2])
    var co = Int(d.chain_off[unsafe_offset=f])
    var sw = t.side * Float32(w); var sh = t.side * Float32(h)
    var ox = t.u0 * Float32(w) - Float32(0.5); var oy = t.v0 * Float32(h) - Float32(0.5)
    var dX = p1[0] - p0[0]; var dY = p1[1] - p0[1]
    var xa = p0[0] + dX * la; var xb = p0[0] + dX * lb; var ya = p0[1] + dY * la; var yb = p0[1] + dY * lb
    var xlo = max(min(xa, xb), bx0) - pad; var xhi = min(max(xa, xb), bx0 + tw) + pad
    var ylo = max(min(ya, yb), by0) - pad; var yhi = min(max(ya, yb), by0 + th) + pad
    var i0 = max(Int(floor(xlo - Float32(0.5))), -1); var i1 = min(Int(floor(xhi - Float32(0.5))), w - 1)
    var j0 = max(Int(floor(ylo - Float32(0.5))), -1); var j1 = min(Int(floor(yhi - Float32(0.5))), h - 1)
    var ln = sqrt(dX * dX + dY * dY); var nx = Float32(0); var ny = Float32(0)
    if ln > Float32(1e-6):
        nx = -dY / ln; ny = dX / ln
    var res = SIMD[DType.float32, 4](best, 0, 0, 0)
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            # Beside the piece by more than the pad: skip.
            var e0 = (Float32(i) + Float32(0.5) - xa) * nx + (Float32(j) + Float32(0.5) - ya) * ny
            var ex = abs(nx); var ey = abs(ny)
            if e0 - pad > ex + ey or e0 + ex + ey < -pad:
                continue
            if st == 0 or st == 2:
                steps += 1
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
    return res


def _piece(d: DispSurface, t: _Tri, f: Int, dir: Vec3f, p0: SIMD[DType.float32, 4], p1: SIMD[DType.float32, 4], pad: Float32,
           best: Float32, mut steps: Int, st: Int) -> SIMD[DType.float32, 4]:
    """Hierarchical DDA along one linear piece (Sec. 5.2): reject nodes whose plane +- residual (widened for the piece's
    deviation) the piece stays above or below, narrow, descend; blocks test their cells. (t, a, b, hit)."""
    var ul = Int(d.hdr[unsafe_offset=3 * f + 1]); var vl = Int(d.hdr[unsafe_offset=3 * f + 2]); var w = 1 << ul; var h = 1 << vl
    var top = max(max(ul, vl), 2)
    var dX = p1[0] - p0[0]; var dY = p1[1] - p0[1]; var dH = p1[2] - p0[2]
    var ln = sqrt(dX * dX + dY * dY)
    var ext = pad / max(ln, Float32(1e-6))   # the tested interval's extension, in the piece's parameter
    var ks = 2
    while ks < top and Float32(1 << ks) < max(abs(dX), abs(dY)):   # the start level: cells as large as the piece
        ks += 1
    var gx = Float32(1e-4) if dX > Float32(0) else (Float32(-1e-4) if dX < Float32(0) else Float32(0))
    var gy = Float32(1e-4) if dY > Float32(0) else (Float32(-1e-4) if dY < Float32(0) else Float32(0))
    var s = Float32(0); var k = ks; var lo = _lvl_off(d, f, ul, vl, k)
    for _ in range(1024):
        var cw = max(w >> k, 1); var ch = max(h >> k, 1)
        var tw = Float32(w // cw); var th = Float32(h // ch)
        var px = p0[0] + dX * s + gx; var py = p0[1] + dY * s + gy
        var cx = min(max(Int(floor(px / tw)), 0), cw - 1); var cy = min(max(Int(floor(py / th)), 0), ch - 1)
        var bx0 = Float32(cx) * tw; var by0 = Float32(cy) * th
        var sx = Float32(2); var sy = Float32(2)
        if dX > Float32(0):
            sx = (bx0 + tw - p0[0]) / dX
        elif dX < Float32(0):
            sx = (bx0 - p0[0]) / dX
        if dY > Float32(0):
            sy = (by0 + th - p0[1]) / dY
        elif dY < Float32(0):
            sy = (by0 - p0[1]) / dY
        var se = max(min(min(sx, sy), Float32(1)), s)
        if st == 0 or st == 3:
            steps += 1
        var node = lo + NODE_B * (cy * cw + cx)
        var pa = Float32(Int(d.pyr[unsafe_offset=node]))
        var gu = Float32(2) * d.pyr[unsafe_offset=node + 1].cast[DType.int8]().cast[DType.float32]() / (Float32(0.5) * tw)
        var gv = Float32(2) * d.pyr[unsafe_offset=node + 2].cast[DType.int8]().cast[DType.float32]() / (Float32(0.5) * th)
        var band = Float32(Int(d.pyr[unsafe_offset=node + 3])) + pad * (abs(gu) + abs(gv))   # residual + deviation padding
        var xc = bx0 + Float32(0.5) * tw; var yc = by0 + Float32(0.5) * th
        var sa = s - ext; var sb = se + ext
        var qa = p0[2] + dH * sa - (pa + gu * (p0[0] + dX * sa - xc) + gv * (p0[1] + dY * sa - yc))   # Eq. 17
        var qb = p0[2] + dH * sb - (pa + gu * (p0[0] + dX * sb - xc) + gv * (p0[1] + dY * sb - yc))
        if not (max(qa, qb) < -band or min(qa, qb) > band):   # Eq. 18
            var n0 = sa; var n1 = sb   # narrowed to where the piece is inside the band: q is affine
            if qb != qa:
                var u0 = sa + (-band - qa) / (qb - qa) * (sb - sa); var u1 = sa + (band - qa) / (qb - qa) * (sb - sa)
                n0 = max(sa, min(u0, u1)); n1 = min(sb, max(u0, u1))
            if k > 2:
                k -= 1; lo = _lvl_off(d, f, ul, vl, k)
                s = max(s, min(n0, se))
                continue
            var lf = _leaf(d, t, f, dir, p0, p1, max(n0, sa), min(n1, sb), bx0, by0, tw, th, pad, best, steps, st)
            if lf[3] > Float32(0):
                return lf
        var mx = p0[0] + dX * Float32(0.5) * (s + se); var my = p0[1] + dY * Float32(0.5) * (s + se)
        s = se
        if s >= Float32(1):
            break
        var nx = p0[0] + dX * s + gx; var ny = p0[1] + dY * s + gy
        while k < ks:   # ascend while the step left the parent too
            var w2 = Float32(w // max(w >> (k + 1), 1)); var h2 = Float32(h // max(h >> (k + 1), 1))
            if Int(floor(mx / w2)) == Int(floor(nx / w2)) and Int(floor(my / h2)) == Int(floor(ny / h2)):
                break
            k += 1; lo = _lvl_off(d, f, ul, vl, k)
    return SIMD[DType.float32, 4](best, 0, 0, 0)


def intersect_xu(d: DispSurface, r: Rmip, id: Int, org: Vec3f, dir: Vec3f, t_max: Float32, st: Int = 0) -> Tuple[Bool, Float32, Float32, Float32, Int]:
    """Algorithm 1 for triangle id: (hit, t, s, t in the patch, steps). st counts other steps: 1 pieces, 2 leaf cells,
    3 nodes."""
    var sp = _prism_span(d, r, id, org, dir, t_max)
    if sp.nt == 0:
        return (False, t_max, Float32(0), Float32(0), 1)
    var t = sp.t; var f = sp.f; var h0 = sp.h0; var dh = sp.h1 - sp.h0
    var ul = Int(d.hdr[unsafe_offset=3 * f + 1]); var vl = Int(d.hdr[unsafe_offset=3 * f + 2])
    var w = Float32(1 << ul); var h = Float32(1 << vl)
    # Events over the height interval (Sec. 5.1): singularities (D), turning points of a and b, the triangle's edges.
    var e0 = _udvt(t, dir, h0); var em = _udvt(t, dir, h0 + Float32(0.5) * dh); var e1 = _udvt(t, dir, h0 + dh)
    var U = _quad(e0[0], em[0], e1[0]); var V = _quad(e0[1], em[1], e1[1]); var D = _quad(e0[3], em[3], e1[3])
    var cut = Array[Float32, 18](fill=Float32(2)); var nc = 0
    comptime for q in range(8):
        var c: SIMD[DType.float32, 4]
        comptime if q == 0:
            c = D
        elif q == 1:
            c = U
        elif q == 2:
            c = V
        elif q == 3:
            c = U - D
        elif q == 4:
            c = V - D
        elif q == 5:
            c = V - U
        elif q == 6:   # a' = 0: U' D - U D'
            c = SIMD[DType.float32, 4](U[1] * D[0] - U[0] * D[1], Float32(2) * (U[2] * D[0] - U[0] * D[2]), U[2] * D[1] - U[1] * D[2], 0)
        else:
            c = SIMD[DType.float32, 4](V[1] * D[0] - V[0] * D[1], Float32(2) * (V[2] * D[0] - V[0] * D[2]), V[2] * D[1] - V[1] * D[2], 0)
        var rr = _roots01(c)
        comptime for g in range(2):
            if rr[g] > Float32(0):
                cut[nc] = rr[g]; nc += 1
    cut[nc] = Float32(0); cut[nc + 1] = Float32(1); nc += 2
    for x in range(1, nc):
        var y = x
        while y > 0 and cut[y - 1] > cut[y]:
            var z = cut[y]; cut[y] = cut[y - 1]; cut[y - 1] = z; y -= 1
    # The intervals inside the triangle along the ray (t is monotone in the height on a branch); each keeps the nearest hit.
    var up = e1[2] / e1[3] >= e0[2] / e0[3]
    var best = sp.t_lim; var steps = 0; var res = SIMD[DType.float32, 4](0)
    for xx in range(nc - 1):
        var x = xx if up else nc - 2 - xx
        var sa = cut[x]; var sb = cut[x + 1]
        if sb - sa <= Float32(1e-6):
            continue
        var q = _udvt(t, dir, h0 + dh * Float32(0.5) * (sa + sb))
        if not (abs(q[3]) > Float32(0) and _inside(t, q[0] / q[3], q[1] / q[3], Float32(1e-4))):
            continue
        var pa = _xyht(t, dir, h0, dh, sa, w, h, d.scale); var pb = _xyht(t, dir, h0, dh, sb, w, h, d.scale)
        if min(pa[3], pb[3]) > best:
            continue
        var pm = _xyht(t, dir, h0, dh, Float32(0.5) * (sa + sb), w, h, d.scale)
        var dx = pm[0] - Float32(0.5) * (pa[0] + pb[0]); var dy = pm[1] - Float32(0.5) * (pa[1] + pb[1])
        var dev = sqrt(dx * dx + dy * dy)
        var n = min(max(Int(ceil(sqrt(dev / _THR))), 1), 32)
        var pad = Float32(2) * dev / Float32(n * n) + Float32(0.01)   # the pieces' deviation, with a margin
        var fwd = pb[3] >= pa[3]
        var prev = pa if fwd else pb
        for m in range(n):
            var sn = sa + (sb - sa) * Float32(m + 1) / Float32(n) if fwd else sb - (sb - sa) * Float32(m + 1) / Float32(n)
            var nxt = _xyht(t, dir, h0, dh, sn, w, h, d.scale)
            if prev[3] > best:
                break
            if nxt[3] >= Float32(0):
                if st == 1:
                    steps += 1
                var hit = _piece(d, t, f, dir, prev, nxt, pad, best, steps, st)
                if hit[3] > Float32(0):
                    best = hit[0]; res = hit
                    break
            prev = nxt
    if res[3] > Float32(0):
        var pk = prim_patch(d, id >> 1); var ss = sub_square(pk[2], pk[1])
        return (True, best + sp.tp, ss[0] + res[1] * ss[2], ss[1] + res[2] * ss[2], steps)
    return (False, t_max, Float32(0), Float32(0), max(steps, 1))
