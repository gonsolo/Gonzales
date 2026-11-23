# A marching traversal of a sub-square, for tall relief: the ray's projection onto the bilinear base, traced cell by cell
# through an n x n grid of leaf-sized cells in ray order; each cell intersects the walk's exact leaf. Few live values
# where the bisecting walk keeps a box, a slab and a path per level.

from std.bit import count_leading_zeros
from std.math import max, min, sqrt, abs, floor
from gonzales.core.geometry import Vec3f, cross, dot
from gonzales.scene.shapes.displaced_patches import DispSurface, BLFrame, prim_patch, sub_square, _bl_frame, _bl_cell, _mean_normal, \
    _bl_leaf, _leaf_size


@always_inline
def _corner3(v: SIMD[DType.float32, 4], a: Float32, b: Float32) -> Float32:
    return (v[0] + (v[1] - v[0]) * a) * (Float32(1) - b) + (v[3] + (v[2] - v[3]) * a) * b


@always_inline
def _pn(fr: BLFrame, a: Float32, b: Float32) -> Tuple[Vec3f, Vec3f]:
    """The base point and blended normal at local (a, b): bilinear in the corners 00 10 11 01."""
    return (Vec3f(_corner3(fr.px, a, b), _corner3(fr.py, a, b), _corner3(fr.pz, a, b)),
            Vec3f(_corner3(fr.nx, a, b), _corner3(fr.ny, a, b), _corner3(fr.nz, a, b)))


@always_inline
def _psi(fr: BLFrame, dir: Vec3f, a: Float32, b: Float32) -> Float32:
    """det(B - O, N, D): zero where the displacement line through (a, b) meets the ray (O at the frame's origin)."""
    var q = _pn(fr, a, b)
    return dot(q[0], cross(q[1], dir))


def _invert(fr: BLFrame, x: Vec3f, a0: Float32, b0: Float32) -> SIMD[DType.float32, 2]:
    """(a, b) whose displacement line passes x: Gauss-Newton on (B - x) x N = 0."""
    var a = a0; var b = b0
    for _ in range(8):
        var q = _pn(fr, a, b)
        var p = q[0] - x; var n = q[1]
        var pa = Vec3f(_corner3(fr.px, Float32(1), b) - _corner3(fr.px, Float32(0), b), _corner3(fr.py, Float32(1), b) - _corner3(fr.py, Float32(0), b),
                       _corner3(fr.pz, Float32(1), b) - _corner3(fr.pz, Float32(0), b))
        var pb = Vec3f(_corner3(fr.px, a, Float32(1)) - _corner3(fr.px, a, Float32(0)), _corner3(fr.py, a, Float32(1)) - _corner3(fr.py, a, Float32(0)),
                       _corner3(fr.pz, a, Float32(1)) - _corner3(fr.pz, a, Float32(0)))
        var na = Vec3f(_corner3(fr.nx, Float32(1), b) - _corner3(fr.nx, Float32(0), b), _corner3(fr.ny, Float32(1), b) - _corner3(fr.ny, Float32(0), b),
                       _corner3(fr.nz, Float32(1), b) - _corner3(fr.nz, Float32(0), b))
        var nb = Vec3f(_corner3(fr.nx, a, Float32(1)) - _corner3(fr.nx, a, Float32(0)), _corner3(fr.ny, a, Float32(1)) - _corner3(fr.ny, a, Float32(0)),
                       _corner3(fr.nz, a, Float32(1)) - _corner3(fr.nz, a, Float32(0)))
        var fv = cross(p, n); var ja = cross(pa, n) + cross(p, na); var jb = cross(pb, n) + cross(p, nb)
        var m00 = dot(ja, ja); var m01 = dot(ja, jb); var m11 = dot(jb, jb); var det = m00 * m11 - m01 * m01
        if not (abs(det) > Float32(0)):
            break
        var ga = dot(ja, fv); var gb = dot(jb, fv)
        var da = -(m11 * ga - m01 * gb) / det; var db = -(m00 * gb - m01 * ga) / det
        a += da; b += db
        if da * da + db * db < Float32(1e-10):
            break
    return SIMD[DType.float32, 2](a, b)


@always_inline
def _ray_t(fr: BLFrame, dir: Vec3f, a: Float32, b: Float32) -> Float32:
    """The ray parameter where the ray passes the displacement line through (a, b): least squares of t D - s N = B."""
    var q = _pn(fr, a, b); var p = q[0]; var n = q[1]
    var dn = dot(dir, n); var nn = dot(n, n); var det = dn * dn - nn
    if det == Float32(0):
        return dot(dir, p)
    return (dot(dir, p) * (-nn) + dn * dot(n, p)) / det


@always_inline
def _edge_pt(e: Int, x: Float32) -> SIMD[DType.float32, 2]:
    """Point x of the square's edge e: 0 b = 0, 1 a = 1, 2 b = 1, 3 a = 0."""
    if e == 0:
        return SIMD[DType.float32, 2](x, 0)
    if e == 1:
        return SIMD[DType.float32, 2](1, x)
    if e == 2:
        return SIMD[DType.float32, 2](x, 1)
    return SIMD[DType.float32, 2](0, x)


def _boundary(fr: BLFrame, dir: Vec3f, t0: Float32, t1: Float32) -> SIMD[DType.float32, 8]:
    """Where the projection crosses the square's boundary within [t0, t1]: the first crossing's (a, b, t) and the last's,
    and their count (psi is quadratic along an edge)."""
    var first = SIMD[DType.float32, 4](0, 0, Float32(1e30), 0); var last = SIMD[DType.float32, 4](0, 0, Float32(-1e30), 0); var cnt = 0
    for e in range(4):
        var f0 = _psi(fr, dir, _edge_pt(e, 0)[0], _edge_pt(e, 0)[1]); var fm = _psi(fr, dir, _edge_pt(e, 0.5)[0], _edge_pt(e, 0.5)[1])
        var f1 = _psi(fr, dir, _edge_pt(e, 1)[0], _edge_pt(e, 1)[1])
        var c2 = Float32(2) * (f0 + f1) - Float32(4) * fm; var c1 = f1 - f0 - c2   # psi(x) = f0 + c1 x + c2 x^2
        var r = SIMD[DType.float32, 2](-1)
        if abs(c2) < Float32(1e-12) * (abs(c1) + abs(f0)):
            if c1 != Float32(0):
                r[0] = -f0 / c1
        else:
            var disc = c1 * c1 - Float32(4) * c2 * f0
            if disc >= Float32(0):
                var sq = sqrt(disc); var q = Float32(-0.5) * (c1 + (sq if c1 >= Float32(0) else -sq))
                r[0] = q / c2; r[1] = f0 / q if q != Float32(0) else Float32(-1)
        comptime for g in range(2):
            if r[g] >= Float32(0) and r[g] <= Float32(1):
                var pt = _edge_pt(e, r[g]); var tt = _ray_t(fr, dir, pt[0], pt[1])
                if tt >= t0 and tt <= t1:
                    cnt += 1
                    if tt < first[2]:
                        first = SIMD[DType.float32, 4](pt[0], pt[1], tt, Float32(e))
                    if tt > last[2]:
                        last = SIMD[DType.float32, 4](pt[0], pt[1], tt, Float32(e))
    return SIMD[DType.float32, 8](first[0], first[1], first[2], first[3], last[0], last[1], last[2], Float32(cnt))


@always_inline
def _span(c: Tuple[Vec3f, Vec3f, Float32, Float32, Float32], n: Vec3f, dir: Vec3f, inv: Vec3f, t_max: Float32, grow: Float32) -> SIMD[DType.float32, 2]:
    """The ray's interval in cell c's box cut by its slab along n, both widened by grow times the box."""
    var lo = c[0]; var hi = c[1]
    var size = max(max(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z); var bp = size * grow + Float32(1e-6)
    var ax = (lo.x - bp) * inv.x; var bx = (hi.x + bp) * inv.x; var ay = (lo.y - bp) * inv.y; var by = (hi.y + bp) * inv.y
    var az = (lo.z - bp) * inv.z; var bz = (hi.z + bp) * inv.z
    var t0 = max(max(min(ax, bx), min(ay, by)), max(min(az, bz), Float32(0)))
    var t1 = min(min(max(ax, bx), max(ay, by)), min(max(az, bz), t_max))
    var dd = dot(n, dir)
    if abs(dd) > Float32(1e-20):
        var pad = bp + Float32(1e-5) * (abs(c[3]) + abs(c[4]))
        var ta = (c[3] - pad) / dd; var tb = (c[4] + pad) / dd
        t0 = max(t0, min(ta, tb)); t1 = min(t1, max(ta, tb))
    return SIMD[DType.float32, 2](t0, t1)


def intersect_march(d: DispSurface, prim: Int, org: Vec3f, dir: Vec3f, t_max: Float32) -> Tuple[Bool, Float32, Float32, Float32, Int, Float32]:
    """The nearest hit of sub-square prim, marching its leaf-sized cells along the ray's projection: (hit, t, s, t, steps,
    the cell's size)."""
    var pk = prim_patch(d, prim); var pi = pk[0]; var sq = sub_square(pk[2], pk[1])
    var S = sq[2]; var inv_s = Float32(1) / S
    var f = Int(d.face[unsafe_offset=pi])
    var texels = S * d.sq[unsafe_offset=3 * pi + 2] * Float32(1 << max(Int(d.hdr[unsafe_offset=3 * f + 1]), Int(d.hdr[unsafe_offset=3 * f + 2])))
    var fr = _bl_frame(d, pi, pk[1], org)
    var nrm = _mean_normal(fr)
    var inv = Vec3f(Float32(1) / dir.x, Float32(1) / dir.y, Float32(1) / dir.z)
    var c = _bl_cell(d, pi, f, fr, sq[0], sq[1], inv_s, (sq[0], sq[1], S, S), nrm)
    var sp = _span(c, nrm, dir, inv, t_max, Float32(5e-3))
    if sp[1] < sp[0]:
        return (False, t_max, Float32(0), Float32(0), 0, S)
    # Leaf-sized cells: the footprint at the sub-square's nearest point, no finer than a texel.
    var lo0 = Vec3f(fr.px.reduce_min(), fr.py.reduce_min(), fr.pz.reduce_min()); var hi0 = Vec3f(fr.px.reduce_max(), fr.py.reduce_max(), fr.pz.reduce_max())
    var ls = _leaf_size(lo0, hi0, d.cam - org, d.pixel)
    var nc = 1
    while Float32(2 * nc) <= texels and ls[0] / Float32(nc) > ls[1]:
        nc *= 2
    var nf = Float32(nc); var i = 0; var j = 0; var i1 = 0; var j1 = 0; var side = -1   # side: the start cell's entry edge, -1 inside
    if nc > 1:   # one cell needs no projection
        # Ends of the projection inside the square: an end's inversion when it lies inside, else a boundary crossing.
        var a0b0 = _invert(fr, dir * sp[0], Float32(0.5), Float32(0.5)); var a1b1 = _invert(fr, dir * sp[1], a0b0[0], a0b0[1])
        comptime e = Float32(1e-4)
        var in0 = a0b0[0] >= -e and a0b0[0] <= 1 + e and a0b0[1] >= -e and a0b0[1] <= 1 + e
        var in1 = a1b1[0] >= -e and a1b1[0] <= 1 + e and a1b1[1] >= -e and a1b1[1] <= 1 + e
        if not in0 or not in1:
            var bd = _boundary(fr, dir, sp[0], sp[1])
            if not in0:
                if bd[7] == Float32(0):
                    return (False, t_max, Float32(0), Float32(0), 1, S)
                a0b0 = SIMD[DType.float32, 2](bd[0], bd[1]); side = Int(bd[3])
            if not in1 and bd[7] > Float32(0):
                a1b1 = SIMD[DType.float32, 2](bd[4], bd[5])
        i = min(max(Int(floor(min(max(a0b0[0], Float32(0)), Float32(1)) * nf)), 0), nc - 1)
        j = min(max(Int(floor(min(max(a0b0[1], Float32(0)), Float32(1)) * nf)), 0), nc - 1)
        i1 = min(max(Int(floor(min(max(a1b1[0], Float32(0)), Float32(1)) * nf)), 0), nc - 1)
        j1 = min(max(Int(floor(min(max(a1b1[1], Float32(0)), Float32(1)) * nf)), 0), nc - 1)
    var cw = S / nf; var steps = 0
    # Trace the projection: leave each cell through an edge psi changes sign along, not the one it came in by.
    for _ in range(4 * nc):
        steps += 1
        var lh = _bl_leaf(d, pi, f, fr, sq[0], sq[1], inv_s, (sq[0] + Float32(i) * cw, sq[1] + Float32(j) * cw, cw, cw), dir, t_max)
        if lh[0]:   # in ray order: the first hit is the nearest
            return (True, lh[1], lh[2], lh[3], steps, cw)
        if i == i1 and j == j1:
            break
        var a_lo = Float32(i) / nf; var a_hi = Float32(i + 1) / nf; var b_lo = Float32(j) / nf; var b_hi = Float32(j + 1) / nf
        var s00 = _psi(fr, dir, a_lo, b_lo) <= Float32(0); var s10 = _psi(fr, dir, a_hi, b_lo) <= Float32(0)
        var s11 = _psi(fr, dir, a_hi, b_hi) <= Float32(0); var s01 = _psi(fr, dir, a_lo, b_hi) <= Float32(0)
        var best_e = -1; var best_d = 1 << 30
        comptime for ed in range(4):
            var crosses = (s00 != s10) if ed == 0 else ((s10 != s11) if ed == 1 else ((s11 != s01) if ed == 2 else (s01 != s00)))
            if crosses and ed != side:
                var ni = i + (1 if ed == 1 else (-1 if ed == 3 else 0)); var nj = j + (1 if ed == 2 else (-1 if ed == 0 else 0))
                var dd = abs(i1 - ni) + abs(j1 - nj)
                if dd < best_d:
                    best_d = dd; best_e = ed
        if best_e < 0:
            break
        var ni = i + (1 if best_e == 1 else (-1 if best_e == 3 else 0)); var nj = j + (1 if best_e == 2 else (-1 if best_e == 0 else 0))
        if ni < 0 or ni >= nc or nj < 0 or nj >= nc:
            break
        i = ni; j = nj; side = (best_e + 2) % 4
    return (False, t_max, Float32(0), Float32(0), steps, cw)
