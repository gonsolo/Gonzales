# Bicubic Bezier patches traced directly (Binder and Keller 2018): the parameter square is halved alternately in u and v,
# both halves' boxes are tested and the nearer one is entered first. A pending bit per level replaces the stack;
# backtracking crops the original patch to the sibling's square. A piece flat enough is hit as its two corner triangles.
# Patches come from Catmull-Clark meshes (src/oiio/osd.cc).
from std.bit import count_leading_zeros
from std.math import max, min
from gonzales.core.geometry import Vec3f, cross, dot

comptime PATCH_MAX_DEPTH = 40           # 20 halvings a direction
comptime PATCH_REL_EPS = Float32(1.0 / 1024.0)   # leaves: pieces this flat against the patch's own box
comptime _ULPS = Float32(1.0 / 2097152.0)   # 4 float ulps, relative
comptime _U = SIMD[DType.float32, 16](0, 1.0 / 3.0, 2.0 / 3.0, 1, 0, 1.0 / 3.0, 2.0 / 3.0, 1, 0, 1.0 / 3.0, 2.0 / 3.0, 1,
                                      0, 1.0 / 3.0, 2.0 / 3.0, 1)
comptime _V = SIMD[DType.float32, 16](0, 0, 0, 0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0,
                                      2.0 / 3.0, 1, 1, 1, 1)


@fieldwise_init
struct BezierPatch(TrivialRegisterPassable):
    """Control points p[j * 4 + i] (i along u, j along v) as x, y and z planes: 192 bytes."""
    var x: SIMD[DType.float32, 16]
    var y: SIMD[DType.float32, 16]
    var z: SIMD[DType.float32, 16]


@always_inline
def _bern(t: Float32) -> SIMD[DType.float32, 4]:
    var s = Float32(1) - t
    return SIMD[DType.float32, 4](s * s * s, Float32(3) * t * s * s, Float32(3) * t * t * s, t * t * t)


@always_inline
def _dbern(t: Float32) -> SIMD[DType.float32, 4]:
    var s = Float32(1) - t
    return SIMD[DType.float32, 4](-Float32(3) * s * s, Float32(3) * s * (s - Float32(2) * t), Float32(3) * t * (Float32(2) * s - t),
                                  Float32(3) * t * t)


@always_inline
def _tensor(p: SIMD[DType.float32, 16], bu: SIMD[DType.float32, 4], bv: SIMD[DType.float32, 4]) -> Float32:
    var acc = Float32(0)
    comptime for j in range(4):
        comptime for i in range(4):
            acc += bu[i] * bv[j] * p[j * 4 + i]
    return acc


def patch_eval(p: BezierPatch, u: Float32, v: Float32) -> Tuple[Vec3f, Vec3f, Vec3f]:
    """Position and the u and v derivatives at (u, v)."""
    var bu = _bern(u); var bv = _bern(v); var du = _dbern(u); var dv = _dbern(v)
    return (Vec3f(_tensor(p.x, bu, bv), _tensor(p.y, bu, bv), _tensor(p.z, bu, bv)),
            Vec3f(_tensor(p.x, du, bv), _tensor(p.y, du, bv), _tensor(p.z, du, bv)),
            Vec3f(_tensor(p.x, bu, dv), _tensor(p.y, bu, dv), _tensor(p.z, bu, dv)))


@always_inline
def _blossom(b0: Float32, b1: Float32, b2: Float32, b3: Float32, t1: Float32, t2: Float32, t3: Float32) -> Float32:
    var c0 = b0 + (b1 - b0) * t1; var c1 = b1 + (b2 - b1) * t1; var c2 = b2 + (b3 - b2) * t1
    var d0 = c0 + (c1 - c0) * t2; var d1 = c1 + (c2 - c1) * t2
    return d0 + (d1 - d0) * t3


@always_inline
def _crop_axis[along_u: Bool](p: SIMD[DType.float32, 16], a: Float32, b: Float32) -> SIMD[DType.float32, 16]:
    """The cubics along one axis restricted to [a, b]: control points are the blossoms f(a,a,a), f(a,a,b), f(a,b,b), f(b,b,b)."""
    var r = SIMD[DType.float32, 16](0)
    comptime for k in range(4):
        var i0 = k * 4 if along_u else k
        comptime step = 1 if along_u else 4
        var q0 = p[i0]; var q1 = p[i0 + step]; var q2 = p[i0 + 2 * step]; var q3 = p[i0 + 3 * step]
        r[i0] = _blossom(q0, q1, q2, q3, a, a, a)
        r[i0 + step] = _blossom(q0, q1, q2, q3, a, a, b)
        r[i0 + 2 * step] = _blossom(q0, q1, q2, q3, a, b, b)
        r[i0 + 3 * step] = _blossom(q0, q1, q2, q3, b, b, b)
    return r


@always_inline
def _crop(p: BezierPatch, u0: Float32, u1: Float32, v0: Float32, v1: Float32) -> BezierPatch:
    return BezierPatch(_crop_axis[False](_crop_axis[True](p.x, u0, u1), v0, v1),
                       _crop_axis[False](_crop_axis[True](p.y, u0, u1), v0, v1),
                       _crop_axis[False](_crop_axis[True](p.z, u0, u1), v0, v1))


@always_inline
def _half(p: BezierPatch, along_u: Bool, upper: Bool) -> BezierPatch:
    var a = Float32(0.5) if upper else Float32(0)
    var b = Float32(1) if upper else Float32(0.5)
    if along_u:
        return BezierPatch(_crop_axis[True](p.x, a, b), _crop_axis[True](p.y, a, b), _crop_axis[True](p.z, a, b))
    return BezierPatch(_crop_axis[False](p.x, a, b), _crop_axis[False](p.y, a, b), _crop_axis[False](p.z, a, b))


@always_inline
def _mag(p: BezierPatch) -> Float32:
    return max(max(abs(p.x).reduce_max(), abs(p.y).reduce_max()), abs(p.z).reduce_max())


@always_inline
def _box_hit(p: BezierPatch, inv: Vec3f, t_max: Float32) -> Tuple[Bool, Float32, Float32]:
    """The ray (from the origin: patches are anchored at it) against the patch's control-point box: (hit, entry, size)."""
    var lo = Vec3f(p.x.reduce_min(), p.y.reduce_min(), p.z.reduce_min())
    var hi = Vec3f(p.x.reduce_max(), p.y.reduce_max(), p.z.reduce_max())
    var size = max(max(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z)
    var mag = max(max(max(abs(lo.x), abs(hi.x)), max(abs(lo.y), abs(hi.y))), max(abs(lo.z), abs(hi.z)))
    var pad = size * Float32(1e-5) + mag * _ULPS + Float32(1e-7)    # rounding must not open cracks (Binder and Keller 3.4)
    lo = lo - Vec3f(pad, pad, pad); hi = hi + Vec3f(pad, pad, pad)
    var ax = lo.x * inv.x; var bx = hi.x * inv.x
    var ay = lo.y * inv.y; var by = hi.y * inv.y
    var az = lo.z * inv.z; var bz = hi.z * inv.z
    var t0 = max(max(min(ax, bx), min(ay, by)), max(min(az, bz), Float32(0)))
    var t1 = min(min(max(ax, bx), max(ay, by)), min(max(az, bz), t_max))
    return (t0 <= t1, t0, size)


@always_inline
def _flat_axis(p: SIMD[DType.float32, 16]) -> Float32:
    """How far the control points stray from the corners' bilinear patch, and that from its two triangles."""
    var bil = (1 - _U) * (1 - _V) * p[0] + _U * (1 - _V) * p[3] + (1 - _U) * _V * p[12] + _U * _V * p[15]
    return max(abs(p - bil).reduce_max(), abs(p[0] - p[3] - p[12] + p[15]) * Float32(0.25))


@always_inline
def _flatness(p: BezierPatch) -> Float32:
    return max(max(_flat_axis(p.x), _flat_axis(p.y)), _flat_axis(p.z))


@always_inline
def _tri_hit(a: Vec3f, b: Vec3f, c: Vec3f, dir: Vec3f, slack: Float32) -> Tuple[Bool, Float32, Float32, Float32]:
    """The ray from the origin against triangle abc, its edges widened by `slack`: (hit, t, b1, b2)."""
    var e1 = b - a; var e2 = c - a
    var pv = cross(dir, e2)
    var det = dot(e1, pv)
    if det == Float32(0):
        return (False, Float32(0), Float32(0), Float32(0))
    var inv = Float32(1) / det
    var tv = Vec3f(0, 0, 0) - a
    var b1 = dot(tv, pv) * inv
    var qv = cross(tv, e1)
    var b2 = dot(dir, qv) * inv
    var t = dot(e2, qv) * inv
    return (b1 >= -slack and b2 >= -slack and b1 + b2 <= Float32(1) + slack, t, b1, b2)


@always_inline
def _domain(path: UInt64, depth: Int) -> Tuple[Float32, Float32, Float32, Float32]:
    """The sub-square (u0, v0, du, dv) at `depth` along `path` (bit d: upper half at level d; even levels split u)."""
    var u0 = Float32(0); var v0 = Float32(0); var du = Float32(1); var dv = Float32(1)
    for d in range(depth):
        var up = ((path >> UInt64(d)) & 1) != 0
        if d % 2 == 0:
            du *= Float32(0.5)
            if up: u0 += du
        else:
            dv *= Float32(0.5)
            if up: v0 += dv
    return (u0, v0, du, dv)


def intersect_patch(org: Vec3f, dir: Vec3f, patch: BezierPatch, t_max: Float32) -> Tuple[Bool, Float32, Float32, Float32]:
    """The nearest hit before t_max: (hit, t, u, v), on a flat piece's corner triangles."""
    var inv = Vec3f(Float32(1) / dir.x, Float32(1) / dir.y, Float32(1) / dir.z)
    var far = BezierPatch(patch.x - org.x, patch.y - org.y, patch.z - org.z)
    var root = _box_hit(far, inv, t_max)
    if not root[0]:
        return (False, t_max, Float32(0), Float32(0))
    # Anchored just before the box: from a far origin, rounding is coarser than flatness and the descent never ends.
    var t0 = max(root[1] - root[2] - _mag(far) * Float32(8) * _ULPS, Float32(0))
    var o0 = org + dir * t0
    var o = BezierPatch(patch.x - o0.x, patch.y - o0.y, patch.z - o0.z)
    var eps = max(root[2] * PATCH_REL_EPS, _mag(o) * Float32(8) * _ULPS)
    var best = t_max - t0; var hit = False; var hu = Float32(0); var hv = Float32(0)
    var p = o
    var depth = 0
    var path = UInt64(0); var pend = UInt64(0)
    while True:
        var descend = False
        var leaf = _box_hit(p, inv, best)
        if leaf[0]:
            if depth >= PATCH_MAX_DEPTH or _flatness(p) <= eps:
                var c00 = Vec3f(p.x[0], p.y[0], p.z[0]); var c30 = Vec3f(p.x[3], p.y[3], p.z[3])
                var c03 = Vec3f(p.x[12], p.y[12], p.z[12]); var c33 = Vec3f(p.x[15], p.y[15], p.z[15])
                var slack = min(Float32(0.25), Float32(2) * eps / max(leaf[2], Float32(1e-30)))   # neighbours at other depths leave T-junction gaps
                var ta = _tri_hit(c00, c30, c33, dir, slack); var tb = _tri_hit(c00, c33, c03, dir, slack)
                var use_a = ta[0] and (not tb[0] or ta[1] < tb[1])
                var th = ta if use_a else tb
                if th[0] and th[1] < best and th[1] + t0 > Float32(2) * eps:   # a hit at the origin is the surface the ray leaves
                    best = th[1]; hit = True
                    var s = th[2] + th[3] if use_a else th[2]
                    var t = th[3] if use_a else th[2] + th[3]
                    var d = _domain(path, depth)
                    hu = d[0] + min(max(s, Float32(0)), Float32(1)) * d[2]; hv = d[1] + min(max(t, Float32(0)), Float32(1)) * d[3]
            else:
                var along_u = depth % 2 == 0
                var h0 = _box_hit(_half(p, along_u, False), inv, best)   # one half live at a time: registers
                var up = _half(p, along_u, True)
                var h1 = _box_hit(up, inv, best)
                if h0[0] or h1[0]:
                    var upper_first = h1[0] and (not h0[0] or h1[1] < h0[1])
                    if h0[0] and h1[0]:
                        pend |= UInt64(1) << UInt64(depth)
                    if upper_first:
                        path |= UInt64(1) << UInt64(depth)
                        p = up
                    else:
                        p = _half(p, along_u, False)
                    depth += 1
                    descend = True
        if descend:
            continue
        if pend == 0:
            break
        var level = 63 - Int(count_leading_zeros(pend))   # the deepest pending sibling
        pend &= ~(UInt64(1) << UInt64(level))
        path = (path & ((UInt64(1) << UInt64(level)) - 1)) | ((~path) & (UInt64(1) << UInt64(level)))
        depth = level + 1
        var d = _domain(path, depth)
        p = _crop(o, d[0], d[0] + d[2], d[1], d[1] + d[3])
    return (hit, best + t0 if hit else t_max, hu, hv)


def patch_bounds(p: BezierPatch) -> Tuple[Vec3f, Vec3f]:
    """The control points' box, which holds the patch."""
    return (Vec3f(p.x.reduce_min(), p.y.reduce_min(), p.z.reduce_min()), Vec3f(p.x.reduce_max(), p.y.reduce_max(), p.z.reduce_max()))


def patch_normal(patch: BezierPatch, u: Float32, v: Float32) -> Vec3f:
    """The unnormalized surface normal dP/du x dP/dv."""
    var e = patch_eval(patch, u, v)
    return cross(e[1], e[2])


def patch_hit_normal(patches: Pointer[BezierPatch, MutUntrackedOrigin], base_vidx: Int, u: Float32, v: Float32) -> Vec3f:
    """The patch normal at a stand-in triangle's hit; geometric and shading normal alike."""
    return patch_normal(patches[unsafe_offset=base_vidx // 3], u, v)
