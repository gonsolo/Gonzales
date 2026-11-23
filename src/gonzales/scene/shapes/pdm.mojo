# Projective displacement mapping (Hoetzlein, Pacific Graphics 2025; after the reference RaytraceDisplacementMesh,
# IntersectSurface) on the same triangles and prisms as RMIP, for comparison: a fixed-step march through the prism, each
# point projected into the scanning triangle parallel to the base at its plane distance.

from std.math import max, min, sqrt, abs
from gonzales.core.geometry import Vec3f, cross, dot
from gonzales.scene.shapes.displaced_patches import DispSurface, prim_patch, sub_square, face_height
from gonzales.scene.shapes.rmip import Rmip, _prism_span, _P, _N, _inside


@always_inline
def _bary(x: Vec3f, q0: Vec3f, q1: Vec3f, q2: Vec3f) -> Tuple[Float32, Float32]:
    """Barycentric weights of q1 and q2 for x in the plane of q0 q1 q2."""
    var e1 = q1 - q0; var e2 = q2 - q0; var p = x - q0
    var d11 = dot(e1, e1); var d12 = dot(e1, e2); var d22 = dot(e2, e2); var p1 = dot(p, e1); var p2 = dot(p, e2)
    var den = d11 * d22 - d12 * d12
    if not (abs(den) > Float32(0)):
        return (Float32(-1), Float32(-1))
    return ((d22 * p1 - d12 * p2) / den, (d11 * p2 - d12 * p1) / den)


def intersect_pdm[DBG: Bool = False](d: DispSurface, r: Rmip, id: Int, org: Vec3f, dir: Vec3f, t_max: Float32, step: Float32) -> Tuple[Bool, Float32, Float32, Float32, Int]:
    """Triangle id's first crossing below the displaced surface, marching `step` texels at a time and interpolating
    between the last two samples: (hit, t, s, t in the patch, steps)."""
    var sp = _prism_span(d, r, id, org, dir, t_max)
    if sp.nt == 0:
        return (False, t_max, Float32(0), Float32(0), 1)
    var t = sp.t; var f = sp.f
    var bu = Float32(0) if t.upper else Float32(1); var cu = Float32(1) if t.upper else Float32(0)
    var a1 = Float32(1); var b1 = 1 - bu; var a2 = 1 - cu; var b2 = Float32(1)   # corner parameters
    var v0 = _P(t, 0, 0); var v1 = _P(t, a1, b1); var v2 = _P(t, a2, b2)
    var n0 = _N(t, 0, 0); var n1 = _N(t, a1, b1); var n2 = _N(t, a2, b2)
    var nrm = cross(v1 - v0, v2 - v0); nrm = nrm * (Float32(1) / sqrt(dot(nrm, nrm)))
    if dot(nrm, n0 + n1 + n2) < Float32(0):
        nrm = -nrm
    var k0 = Float32(1) / dot(nrm, n0); var k1 = Float32(1) / dot(nrm, n1); var k2 = Float32(1) / dot(nrm, n2)
    var w = Float32(1 << Int(d.hdr[unsafe_offset=3 * f + 1])); var h = Float32(1 << Int(d.hdr[unsafe_offset=3 * f + 2]))
    var dt = step * min(sqrt(dot(t.Pa, t.Pa)) / (t.side * w), sqrt(dot(t.Pb, t.Pb)) / (t.side * h))
    var t1 = min(sp.hi, sp.t_lim)
    var tt = sp.lo; var steps = 0
    comptime if DBG:
        print("    span", sp.lo, sp.hi, "nt", sp.nt, "dt", dt, "heights", sp.h0, sp.h1, "tp", sp.tp)
    var pd = Float32(0); var pt = Float32(0); var pa = Float32(0); var pb = Float32(0); var have = False
    while tt <= t1 and steps < (1 << 20):   # a safety stop only: the march is the method's cost
        steps += 1
        var x = dir * tt
        var de = dot(nrm, x - v0)
        var q = _bary(x, v0 + n0 * (de * k0), v1 + n1 * (de * k1), v2 + n2 * (de * k2))   # the scanning triangle
        var a = q[0] * a1 + q[1] * a2; var b = q[0] * b1 + q[1] * b2
        comptime if DBG:
            var p_ = _P(t, a, b); var n_ = _N(t, a, b)
            print("    t", tt, "ab", a, b, "inside", _inside(t, a, b, Float32(1e-3)), "ray h", dot(x - p_, n_) / dot(n_, n_), "surface h",
                  face_height(d, f, t.u0 + a * t.side, t.v0 + b * t.side) * d.scale)
        if _inside(t, a, b, Float32(1e-3)):
            var p = _P(t, a, b); var n = _N(t, a, b)
            var dd = dot(x - p, n) / dot(n, n) - face_height(d, f, t.u0 + a * t.side, t.v0 + b * t.side) * d.scale
            if dd <= Float32(0):
                var th = tt; var ah = a; var bh = b
                if have:   # between the last sample above and this one below
                    var u = pd / (pd - dd)
                    th = pt + u * (tt - pt); ah = pa + u * (a - pa); bh = pb + u * (b - pb)
                var pk = prim_patch(d, id >> 1); var ss = sub_square(pk[2], pk[1])
                return (True, th + sp.tp, ss[0] + ah * ss[2], ss[1] + bh * ss[2], steps)
            pd = dd; pt = tt; pa = a; pb = b; have = True
        else:
            have = False
        if tt < t1 and tt + dt > t1:   # a last sample at the exit: a surface on the prism's floor is crossed there
            tt = t1
        else:
            tt += dt
    return (False, t_max, Float32(0), Float32(0), steps)
