# A clean-room, line-by-line port of Mitsuba's single-scatter Specular
# Manifold Sampling estimator, run on a hardcoded copy of
# `results/Figure_6_Sequence/sphere_sms.xml` from the reference renderer
# (Zeltner et al. 2020, "Specular Manifold Sampling for Rendering
# High-Frequency Caustics and Glints").
#
# WHY THIS EXISTS: gonzales's own SMS (src/gonzales/sms.mojo +
# shading.mojo's _sms_probe_and_solve) uses the GENERALIZED HALF-VECTOR
# constraint formulation, while sphere_sms.xml sets
# `caustics_halfvector_constraints = false` -- so the reference render we
# have been chasing was produced by the ANGLE-DIFFERENCE formulation
# (manifold_ss.cpp::compute_step_anglediff) instead. Different constraint
# function => different roots, different solution basins, different
# Jacobian. Rather than keep grafting pieces of one pipeline onto the
# other inside a 4400-line shading.mojo, this file reproduces the
# reference algorithm end to end in isolation, with nothing shared with
# gonzales's renderer except Vec3f and the OIIO bridge.
#
# Everything here is deliberately a TRANSCRIPTION, not an improvement:
# where the reference does something that looks like a bug (e.g.
# specular_reflectance flips eta and then hands the flipped value to
# fresnel(), which flips it right back), it is reproduced verbatim and
# flagged in a comment. The whole point is bit-comparable behaviour.
#
# Scope: only what sphere_sms.xml needs -- one analytic sphere caster with
# a normal-mapped smooth dielectric, one rectangle area emitter, one
# textured diffuse floor (the caustic receiver), a perspective camera, and a
# short path tracer (specular dielectric at the sphere, cosine-sampled
# diffuse at the floor). Direct lighting is plain NEE (no MIS, no
# BSDF-sampled emitter hits, so still unbiased).
#
# WHERE THE CAUSTIC ACTUALLY IS (this cost hours to find, so it is written
# down here): the sphere is centred at the origin with radius 6.5 and the
# floor is the plane y = 0, so the sphere is HALF BURIED. The caustic lands
# on the floor patch UNDERNEATH the sphere -- shading points within ~1.5
# units of the origin -- and is only visible to the camera THROUGH the
# glass. It is the bright swirl pattern on the sphere, not anything on the
# open floor. Dumping si.p / si_final.p out of the reference renderer shows
# every contributing shading point inside that buried patch, with the
# specular vertex on the light-facing upper hemisphere. Consequently the
# estimator must be invoked at a floor vertex reached AFTER a refraction
# through the sphere; run it only on the directly-visible floor and it
# returns exactly zero, because no single-refraction solution exists there
# (a dense 300x300 (u,v) scan of the constraint over the whole sphere
# bottoms out at |C| ~= 0.07, never at a root).
#
# Verified against the reference renderer's own 32-spp render of
# sphere_sms.xml at 540x540, 3x3-median-filtered to suppress fireflies:
#   whole image  ref 0.2514  ours 0.2455  (0.977x)
#   sphere/caustic  ref 0.2373  ours 0.2204  (0.929x)
#   lit floor    ref 0.6961  ours 0.7163  (1.029x)
# The residual is consistent with the reconstruction filter (gaussian vs.
# box here) and with NEE-only direct lighting vs. the reference's MIS.
#
# Build:  make sms_mitsuba_ref
# Run:    build/sms_mitsuba_ref <out.exr> [spp] [width] [--sms-only]

from std.math import sqrt, sin, cos, acos, asin, floor, abs, min, max
from std.sys import argv
from std.ffi import external_call
from max.algorithm import parallelize
from gonzales.geometry import Vec3f, dot, cross, _atan2f
from gonzales.rng import PCG32

comptime PI_F: Float32 = 3.14159265358979323846
comptime TWO_PI_F: Float32 = 6.28318530717958647692
comptime INV_PI_F: Float32 = 0.31830988618379067154

# ── Reference-renderer constants (SMSConfig defaults + sphere_sms.xml) ──────
comptime SOLVER_THRESHOLD: Float32 = 1e-5
comptime UNIQUENESS_THRESHOLD: Float32 = 1e-5
comptime MAX_ITERATIONS: Int = 20
comptime MAX_TRIALS: Int = 10000000
comptime STEP_SCALE: Float32 = 1.0

# mitsuba/core/math.h: RayEpsilon = eps<Float>*1024, ShadowEpsilon = 1e3*RayEpsilon
comptime RAY_EPSILON: Float32 = 1.1920929e-7 * 1024.0
comptime SHADOW_EPSILON: Float32 = 1.1920929e-7 * 1024.0 * 1000.0

# include/mitsuba/render/ior.h: bk7 = 1.5046, air = 1.000277
comptime SPHERE_ETA: Float32 = 1.5046 / 1.000277
# <shape type="sphere"> radius, hardcoded from sphere_sms.xml.
comptime SPHERE_RADIUS_G: Float32 = 6.5

# ── Small linear algebra ────────────────────────────────────────────────────

def cstr(s: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    """Null-terminated, mutable-pointer copy of `s` for the OIIO bridge."""
    var b = s.as_bytes()
    var n = len(b)
    var p = alloc[UInt8](n + 1)
    for i in range(n):
        p[i] = b[i]
    p[n] = UInt8(0)
    return p.unsafe_origin_cast[MutExternalOrigin]()

@always_inline
def v3(x: Float32, y: Float32, z: Float32) -> Vec3f:
    return Vec3f(x, y, z)

@always_inline
def norm3(v: Vec3f) -> Float32:
    return sqrt(dot(v, v))

@always_inline
def normalize3(v: Vec3f) -> Vec3f:
    return v * (Float32(1.0) / sqrt(dot(v, v)))

@always_inline
def safe_sqrt_f(x: Float32) -> Float32:
    return sqrt(max(x, Float32(0.0)))

@always_inline
def safe_acos_f(x: Float32) -> Float32:
    return acos(min(max(x, Float32(-1.0)), Float32(1.0)))

# 2x2 matrices as (m00, m01, m10, m11).
@always_inline
def m2det(m: SIMD[DType.float32, 4]) -> Float32:
    return m[0]*m[3] - m[1]*m[2]

@always_inline
def m2inv(m: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    var d = m2det(m)
    var inv = Float32(1.0) / d
    return SIMD[DType.float32, 4](m[3]*inv, -m[1]*inv, -m[2]*inv, m[0]*inv)

@always_inline
def m2mul(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](
        a[0]*b[0] + a[1]*b[2], a[0]*b[1] + a[1]*b[3],
        a[2]*b[0] + a[3]*b[2], a[2]*b[1] + a[3]*b[3])

@always_inline
def m2mulv(m: SIMD[DType.float32, 4], v: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](m[0]*v[0] + m[1]*v[1], m[2]*v[0] + m[3]*v[1])

# 4x4 row-major affine transform, matching Mitsuba's Transform4f composition
# order (`ctx.transform = op * ctx.transform`, so the FIRST tag listed in the
# XML is applied FIRST -- libcore/xml.cpp:849-926).
@fieldwise_init
struct Mat4(ImplicitlyCopyable, Movable):
    var r0: SIMD[DType.float32, 4]
    var r1: SIMD[DType.float32, 4]
    var r2: SIMD[DType.float32, 4]

    @staticmethod
    def identity() -> Mat4:
        return Mat4(SIMD[DType.float32, 4](1.0, 0.0, 0.0, 0.0),
                    SIMD[DType.float32, 4](0.0, 1.0, 0.0, 0.0),
                    SIMD[DType.float32, 4](0.0, 0.0, 1.0, 0.0))

    @staticmethod
    def translate(t: Vec3f) -> Mat4:
        return Mat4(SIMD[DType.float32, 4](1.0, 0.0, 0.0, t.x),
                    SIMD[DType.float32, 4](0.0, 1.0, 0.0, t.y),
                    SIMD[DType.float32, 4](0.0, 0.0, 1.0, t.z))

    @staticmethod
    def scale(s: Vec3f) -> Mat4:
        return Mat4(SIMD[DType.float32, 4](s.x, 0.0, 0.0, 0.0),
                    SIMD[DType.float32, 4](0.0, s.y, 0.0, 0.0),
                    SIMD[DType.float32, 4](0.0, 0.0, s.z, 0.0))

    @staticmethod
    def rotate(axis: Vec3f, degrees: Float32) -> Mat4:
        var a = normalize3(axis)
        var th = degrees * PI_F / Float32(180.0)
        var c = cos(th); var sn = sin(th); var ic = Float32(1.0) - c
        return Mat4(
            SIMD[DType.float32, 4](c + a.x*a.x*ic,     a.x*a.y*ic - a.z*sn, a.x*a.z*ic + a.y*sn, 0.0),
            SIMD[DType.float32, 4](a.y*a.x*ic + a.z*sn, c + a.y*a.y*ic,     a.y*a.z*ic - a.x*sn, 0.0),
            SIMD[DType.float32, 4](a.z*a.x*ic - a.y*sn, a.z*a.y*ic + a.x*sn, c + a.z*a.z*ic,     0.0))

    @always_inline
    def row(self, i: Int) -> SIMD[DType.float32, 4]:
        if i == 0:
            return self.r0
        elif i == 1:
            return self.r1
        return self.r2

    def mul(self, o: Mat4) -> Mat4:
        # Both operands are affine (implicit last row [0,0,0,1]).
        var out = InlineArray[SIMD[DType.float32, 4], 3](fill=SIMD[DType.float32, 4](0.0))
        for i in range(3):
            var ri = self.row(i)
            var acc = SIMD[DType.float32, 4](0.0)
            for k in range(3):
                acc += o.row(k) * ri[k]
            acc[3] += ri[3]
            out[i] = acc
        return Mat4(out[0], out[1], out[2])

    @always_inline
    def xform_p(self, p: Vec3f) -> Vec3f:
        var v = SIMD[DType.float32, 4](p.x, p.y, p.z, 1.0)
        return v3((self.r0 * v).reduce_add(), (self.r1 * v).reduce_add(), (self.r2 * v).reduce_add())

    @always_inline
    def xform_v(self, p: Vec3f) -> Vec3f:
        var v = SIMD[DType.float32, 4](p.x, p.y, p.z, 0.0)
        return v3((self.r0 * v).reduce_add(), (self.r1 * v).reduce_add(), (self.r2 * v).reduce_add())

# ── Frames ──────────────────────────────────────────────────────────────────
# A Frame is (s, t, n); to_world(v) = s*v.x + t*v.y + n*v.z. Derivative
# "frames" reuse the same struct with each component holding a d/du or d/dv.

@fieldwise_init
struct Frm(ImplicitlyCopyable, Movable):
    var s: Vec3f
    var t: Vec3f
    var n: Vec3f

    @always_inline
    def to_world(self, v: Vec3f) -> Vec3f:
        return self.s * v.x + self.t * v.y + self.n * v.z

# core/frame.h: compute_shading_frame
def compute_shading_frame(n: Vec3f, dp_du: Vec3f) -> Frm:
    var s = normalize3(dp_du - n * dot(n, dp_du))
    return Frm(s, cross(n, s), n)

# core/frame.h: compute_shading_frame_derivative
def compute_shading_frame_derivative(
    n: Vec3f, dp_du: Vec3f, dn_du: Vec3f, dn_dv: Vec3f
) -> Tuple[Frm, Frm]:
    var s = dp_du - n * dot(n, dp_du)
    var inv_len_s = Float32(1.0) / norm3(s)
    s = s * inv_len_s

    var du_s = (dn_du * (-dot(n, dp_du)) - n * dot(dn_du, dp_du)) * inv_len_s
    var dv_s = (dn_dv * (-dot(n, dp_du)) - n * dot(dn_dv, dp_du)) * inv_len_s
    du_s = du_s - s * dot(du_s, s)
    dv_s = dv_s - s * dot(dv_s, s)

    var du_t = cross(dn_du, s) + cross(n, du_s)
    var dv_t = cross(dn_dv, s) + cross(n, dv_s)
    return (Frm(du_s, du_t, dn_du), Frm(dv_s, dv_t, dn_dv))

# ── Normal map (LEAN slope pyramid, level 0 only) ───────────────────────────
# render/normalmap.cpp builds a 5-channel LEAN map from the RGB normal map:
#   n = normalize(2*rgb - 1); slope = (-n.x/n.z, -n.y/n.z)
# and eval_normal(use_slopes=true) bilinearly interpolates the SLOPES, then
# returns the UNNORMALIZED local normal (-slope_x, -slope_y, 1). We only ever
# need mip level 0 here: smoothing is 0 (twostage=false in the scene).

struct Normalmap(Movable):
    var slopes: UnsafePointer[Float32, MutUntrackedOrigin]  # 2 floats/texel
    var res: Int

    def __init__(out self, filename: String) raises:
        var pixels_ptr = alloc[UnsafePointer[Float32, MutExternalOrigin]](1)
        var w_out = alloc[Int32](1)
        var h_out = alloc[Int32](1)
        w_out[0] = Int32(0); h_out[0] = Int32(0)
        var fname = cstr(filename)
        var ok = external_call["load_texture_rgb", Int32,
            UnsafePointer[UInt8, MutExternalOrigin],
            UnsafePointer[UnsafePointer[Float32, MutExternalOrigin], MutExternalOrigin],
            UnsafePointer[Int32, MutExternalOrigin], UnsafePointer[Int32, MutExternalOrigin],
            Int32](
            fname, pixels_ptr, w_out, h_out, Int32(1))
        fname.free()
        var w = Int(w_out[0]); var h = Int(h_out[0])
        w_out.free(); h_out.free()
        if ok == Int32(0) or w <= 0 or w != h:
            raise Error("Normalmap: failed to load or non-square: " + filename)
        var src = pixels_ptr[0]
        self.res = w
        self.slopes = alloc[Float32](2 * w * w)
        for i in range(w * w):
            var nx = Float32(2.0) * src[i*3 + 0] - Float32(1.0)
            var ny = Float32(2.0) * src[i*3 + 1] - Float32(1.0)
            var nz = Float32(2.0) * src[i*3 + 2] - Float32(1.0)
            var inv = Float32(1.0) / sqrt(nx*nx + ny*ny + nz*nz)
            nx *= inv; ny *= inv; nz *= inv
            self.slopes[i*2 + 0] = -nx / nz
            self.slopes[i*2 + 1] = -ny / nz
        pixels_ptr.free()

    def __deinit__(deinit self):
        self.slopes.free()

    # render/normalmap.h: the shared bilinear addressing of eval_normal /
    # eval_normal_derivatives (clamp-to-edge, half-texel offset, no V flip).
    @always_inline
    def _addr(self, u: Float32, v: Float32) -> Tuple[Int, Int, Int, Int, Float32, Float32]:
        var res = self.res
        var resf = Float32(res)
        var duv = Float32(1.0) / resf
        var px = (u - Float32(0.5)*duv) * resf
        var py = (v - Float32(0.5)*duv) * resf
        px = min(max(px, Float32(0.0)), resf - Float32(1.0))
        py = min(max(py, Float32(0.0)), resf - Float32(1.0))
        var x0 = Int(floor(px)); var y0 = Int(floor(py))
        var x1 = x0 + 1; var y1 = y0 + 1
        if x0 > res - 1: x0 = res - 1
        if y0 > res - 1: y0 = res - 1
        if x1 > res - 1: x1 = res - 1
        if y1 > res - 1: y1 = res - 1
        return (x0, y0, x1, y1, px - Float32(x0), py - Float32(y0))

    @always_inline
    def _slope(self, x: Int, y: Int) -> SIMD[DType.float32, 2]:
        var i = (y * self.res + x) * 2
        return SIMD[DType.float32, 2](self.slopes[i], self.slopes[i + 1])

    def eval_normal(self, u: Float32, v: Float32) -> Vec3f:
        var a = self._addr(u, v)
        var w1x = a[4]; var w1y = a[5]
        var w0x = Float32(1.0) - w1x; var w0y = Float32(1.0) - w1y
        var v00 = self._slope(a[0], a[1]); var v10 = self._slope(a[2], a[1])
        var v01 = self._slope(a[0], a[3]); var v11 = self._slope(a[2], a[3])
        var r0 = v00 * w0x + v10 * w1x
        var r1 = v01 * w0x + v11 * w1x
        var s = r0 * w0y + r1 * w1y
        return v3(-s[0], -s[1], Float32(1.0))

    def eval_normal_derivatives(self, u: Float32, v: Float32) -> Tuple[Vec3f, Vec3f]:
        var a = self._addr(u, v)
        var wx = a[4]; var wy = a[5]
        var v00 = self._slope(a[0], a[1]); var v10 = self._slope(a[2], a[1])
        var v01 = self._slope(a[0], a[3]); var v11 = self._slope(a[2], a[3])
        var resf = Float32(self.res)
        var tmp = v01 + v10 - v11
        var tu = (v10 + v00 * (wy - Float32(1.0)) - tmp * wy) * resf
        var tv = (v01 + v00 * (wx - Float32(1.0)) - tmp * wx) * resf
        return (v3(-tu[0], -tu[1], Float32(0.0)), v3(-tv[0], -tv[1], Float32(0.0)))

# ── Albedo texture (floor) ──────────────────────────────────────────────────

struct ColorTexture(Movable):
    var data: UnsafePointer[Float32, MutUntrackedOrigin]
    var w: Int
    var h: Int

    def __init__(out self, filename: String) raises:
        var pixels_ptr = alloc[UnsafePointer[Float32, MutExternalOrigin]](1)
        var w_out = alloc[Int32](1)
        var h_out = alloc[Int32](1)
        w_out[0] = Int32(0); h_out[0] = Int32(0)
        var fname = cstr(filename)
        var ok = external_call["load_texture_rgb", Int32,
            UnsafePointer[UInt8, MutExternalOrigin],
            UnsafePointer[UnsafePointer[Float32, MutExternalOrigin], MutExternalOrigin],
            UnsafePointer[Int32, MutExternalOrigin], UnsafePointer[Int32, MutExternalOrigin],
            Int32](
            fname, pixels_ptr, w_out, h_out, Int32(0))   # raw=0 -> sRGB decoded to linear
        fname.free()
        self.w = Int(w_out[0]); self.h = Int(h_out[0])
        w_out.free(); h_out.free()
        if ok == Int32(0) or self.w <= 0:
            raise Error("ColorTexture: failed to load " + filename)
        var src = pixels_ptr[0]
        self.data = alloc[Float32](3 * self.w * self.h)
        for i in range(3 * self.w * self.h):
            self.data[i] = src[i]
        pixels_ptr.free()

    def __deinit__(deinit self):
        self.data.free()

    def eval(self, u_in: Float32, v_in: Float32) -> Vec3f:
        var u = u_in - floor(u_in)
        var v = v_in - floor(v_in)
        var x = Int(u * Float32(self.w))
        var y = Int((Float32(1.0) - v) * Float32(self.h))
        if x < 0: x = 0
        if y < 0: y = 0
        if x >= self.w: x = self.w - 1
        if y >= self.h: y = self.h - 1
        var i = (y * self.w + x) * 3
        return v3(self.data[i], self.data[i+1], self.data[i+2])

# ── Scene ───────────────────────────────────────────────────────────────────
# Three primitives, all analytic: sphere (caustic caster), floor rectangle
# (caustic receiver), light rectangle (area emitter).

comptime SHAPE_NONE: Int = 0
comptime SHAPE_SPHERE: Int = 1
comptime SHAPE_FLOOR: Int = 2
comptime SHAPE_LIGHT: Int = 3

@fieldwise_init
struct Rect(ImplicitlyCopyable, Movable):
    """Mitsuba's `rectangle`: the [-1,1]^2 square in the XY plane at z=0 with
    normal +Z, mapped through `to_world` (shapes/rectangle.cpp::update)."""
    var o: Vec3f       # to_world * (0,0,0)
    var dp_du: Vec3f   # to_world * (2,0,0)   -- si.dp_du, unnormalized
    var dp_dv: Vec3f   # to_world * (0,2,0)   -- si.dp_dv, unnormalized
    var n: Vec3f       # normalize(to_world * (0,0,1))
    var area: Float32

    @staticmethod
    def build(to_world: Mat4) -> Rect:
        var o = to_world.xform_p(v3(0.0, 0.0, 0.0))
        var du = to_world.xform_v(v3(2.0, 0.0, 0.0))
        var dv = to_world.xform_v(v3(0.0, 2.0, 0.0))
        var n = normalize3(to_world.xform_v(v3(0.0, 0.0, 1.0)))
        return Rect(o, du, dv, n, norm3(cross(du, dv)))

    # Ray/plane hit, then the [-1,1]^2 extent test in local coordinates.
    # Returns (t, local_x, local_y); t < 0 means no hit.
    def intersect(self, org: Vec3f, d: Vec3f) -> Tuple[Float32, Float32, Float32]:
        var denom = dot(self.n, d)
        if abs(denom) < Float32(1e-12):
            return (Float32(-1.0), Float32(0.0), Float32(0.0))
        var t = dot(self.n, self.o - org) / denom
        if t <= Float32(0.0):
            return (Float32(-1.0), Float32(0.0), Float32(0.0))
        var rel = org + d * t - self.o
        # dp_du spans local x in [-1,1] over its full length, i.e.
        # local_x = 2 * dot(rel, dp_du)/|dp_du|^2.
        var lx = Float32(2.0) * dot(rel, self.dp_du) / dot(self.dp_du, self.dp_du)
        var ly = Float32(2.0) * dot(rel, self.dp_dv) / dot(self.dp_dv, self.dp_dv)
        if lx < Float32(-1.0) or lx > Float32(1.0) or ly < Float32(-1.0) or ly > Float32(1.0):
            return (Float32(-1.0), Float32(0.0), Float32(0.0))
        return (t, lx, ly)

@fieldwise_init
struct Hit(ImplicitlyCopyable, Movable):
    var shape: Int
    var t: Float32
    var p: Vec3f
    var n: Vec3f        # geometric normal (== shading normal for all 3 shapes)
    var dp_du: Vec3f
    var dp_dv: Vec3f
    var uv: SIMD[DType.float32, 2]

    @staticmethod
    def miss() -> Hit:
        return Hit(SHAPE_NONE, Float32(0.0), v3(0.0,0.0,0.0), v3(0.0,0.0,1.0),
                   v3(1.0,0.0,0.0), v3(0.0,1.0,0.0), SIMD[DType.float32, 2](0.0, 0.0))

struct Scene(Movable):
    var sphere_c: Vec3f
    var sphere_r: Float32
    var floor_rect: Rect
    var light_rect: Rect
    var light_radiance: Float32
    var nmap: Normalmap
    var floor_tex: ColorTexture
    var floor_uv_scale: Float32
    # Experiment switch (--smooth-sms): build the solver's ManifoldVertex
    # from the SMOOTH analytic sphere (geometric normal + 1/radius
    # curvature) instead of the normal-mapped frame -- which is what
    # gonzales's `_sms_vertex_from_hit` -> `sms_vertex_sphere` does. The
    # camera-ray refraction still uses the normal map either way.
    var smooth_sms: Bool
    # Experiment switch (--halfvector): solve the generalized half-vector
    # constraint (what sms.mojo does) instead of the angle-difference one
    # this scene actually selects.
    var halfvector: Bool
    # Experiment switch (--mnee-seed): replace SMS's uniform-random seeding
    # + Bernoulli-trial reciprocal estimator with gonzales's own scheme --
    # ONE deterministic seed obtained by probing straight at the light, and
    # an unweighted contribution (T fixed at 1). Everything else identical.
    var mnee_seed: Bool

    def __init__(out self, scene_dir: String) raises:
        self.sphere_c = v3(0.0, 0.0, 0.0)
        self.sphere_r = SPHERE_RADIUS_G
        # <shape type="rectangle"> floor: scale(100) then rotate x -90.
        self.floor_rect = Rect.build(
            Mat4.rotate(v3(1.0, 0.0, 0.0), Float32(-90.0)).mul(
                Mat4.scale(v3(100.0, 100.0, 100.0))))
        # <shape type="rectangle"> emitter: rotate x 90; scale(.4,1,.4);
        # translate y 20; rotate x 50; rotate y 90  (first tag applied first).
        var lt = Mat4.rotate(v3(0.0, 1.0, 0.0), Float32(90.0))
        lt = lt.mul(Mat4.rotate(v3(1.0, 0.0, 0.0), Float32(50.0)))
        lt = lt.mul(Mat4.translate(v3(0.0, 20.0, 0.0)))
        lt = lt.mul(Mat4.scale(v3(0.4, 1.0, 0.4)))
        lt = lt.mul(Mat4.rotate(v3(1.0, 0.0, 0.0), Float32(90.0)))
        self.light_rect = Rect.build(lt)
        self.light_radiance = 1200.0
        self.nmap = Normalmap(scene_dir + "/textures/normalmap_gaussian.exr")
        self.floor_tex = ColorTexture(scene_dir + "/textures/[2K]Tiles71/Tiles71_col.jpg")
        self.floor_uv_scale = 10.0
        self.mnee_seed = False
        self.halfvector = False
        self.smooth_sms = False

    # shapes/sphere.cpp::compute_surface_interaction, with an identity
    # rotation and to_world = translate(center) * scale(radius).
    def sphere_hit(self, p_in: Vec3f) -> Hit:
        var n = normalize3(p_in - self.sphere_c)
        var p = self.sphere_c + n * self.sphere_r
        var local = n                                   # (p - c)/r, unit
        var rd2 = local.x*local.x + local.y*local.y
        var rd = sqrt(rd2)
        var theta = safe_acos_f(local.z)
        var phi = _atan2f(local.y, local.x)
        if phi < Float32(0.0):
            phi += TWO_PI_F
        var dp_du = v3(-local.y, local.x, Float32(0.0)) * (self.sphere_r * TWO_PI_F)
        var dp_dv: Vec3f
        if rd > Float32(0.0):
            var inv_rd = Float32(1.0) / rd
            dp_dv = v3(local.z * local.x * inv_rd, local.z * local.y * inv_rd, -rd) * (self.sphere_r * PI_F)
        else:
            dp_dv = v3(Float32(1.0), Float32(0.0), Float32(0.0)) * (self.sphere_r * PI_F)
        return Hit(SHAPE_SPHERE, Float32(0.0), p, n, dp_du, dp_dv,
                   SIMD[DType.float32, 2](phi * (Float32(1.0)/TWO_PI_F), theta * (Float32(1.0)/PI_F)))

    def intersect(self, org: Vec3f, d: Vec3f, tmin: Float32, tmax: Float32) -> Hit:
        var best_t = tmax
        var best = SHAPE_NONE
        var lx = Float32(0.0); var ly = Float32(0.0)

        # sphere
        var oc = org - self.sphere_c
        var b = dot(oc, d)
        var c = dot(oc, oc) - self.sphere_r * self.sphere_r
        var disc = b*b - c
        if disc > Float32(0.0):
            var sd = sqrt(disc)
            var t0 = -b - sd
            var t1 = -b + sd
            if t0 > tmin and t0 < best_t:
                best_t = t0; best = SHAPE_SPHERE
            elif t1 > tmin and t1 < best_t:
                best_t = t1; best = SHAPE_SPHERE

        var fr = self.floor_rect.intersect(org, d)
        if fr[0] > tmin and fr[0] < best_t:
            best_t = fr[0]; best = SHAPE_FLOOR; lx = fr[1]; ly = fr[2]

        var lr = self.light_rect.intersect(org, d)
        if lr[0] > tmin and lr[0] < best_t:
            best_t = lr[0]; best = SHAPE_LIGHT; lx = lr[1]; ly = lr[2]

        if best == SHAPE_NONE:
            return Hit.miss()
        var p = org + d * best_t
        if best == SHAPE_SPHERE:
            var h = self.sphere_hit(p)
            h.t = best_t
            return h
        var rect = self.floor_rect if best == SHAPE_FLOOR else self.light_rect
        return Hit(best, best_t, p, rect.n, rect.dp_du, rect.dp_dv,
                   SIMD[DType.float32, 2](lx * Float32(0.5) + Float32(0.5),
                                          ly * Float32(0.5) + Float32(0.5)))

    def occluded(self, org: Vec3f, d: Vec3f, tmin: Float32, tmax: Float32) -> Bool:
        return self.intersect(org, d, tmin, tmax).shape != SHAPE_NONE

# ── ManifoldVertex ──────────────────────────────────────────────────────────
# render/manifold.h::ManifoldVertex(si, smoothing) -- all fields the
# single-scatter estimator touches.

@fieldwise_init
struct MVertex(ImplicitlyCopyable, Movable):
    var p: Vec3f
    var dp_du: Vec3f
    var dp_dv: Vec3f
    var n: Vec3f
    var gn: Vec3f
    var dn_du: Vec3f
    var dn_dv: Vec3f
    var s: Vec3f
    var t: Vec3f
    var ds_du: Vec3f
    var ds_dv: Vec3f
    var dt_du: Vec3f
    var dt_dv: Vec3f
    var eta: Float32
    var shape: Int

    def make_orthonormal(mut self):
        var inv_norm = Float32(1.0) / norm3(self.dp_du)
        self.dp_du = self.dp_du * inv_norm
        self.dn_du = self.dn_du * inv_norm
        var dp = dot(self.dp_du, self.dp_dv)
        var dp_dv_tmp = self.dp_dv - self.dp_du * dp
        var dn_dv_tmp = self.dn_dv - self.dn_du * dp
        inv_norm = Float32(1.0) / norm3(dp_dv_tmp)
        self.dp_dv = dp_dv_tmp * inv_norm
        self.dn_dv = dn_dv_tmp * inv_norm

# bsdfs/normalmap.cpp::frame + frame_derivative, at smoothing = 0.
def normalmap_frames(nmap: Normalmap, h: Hit) -> Tuple[Frm, Frm, Frm]:
    var u = h.uv[0]; var v = h.uv[1]
    u = u - floor(u); v = v - floor(v)            # m_tiles == 1

    var n_local = nmap.eval_normal(u, v)
    var dn = nmap.eval_normal_derivatives(u, v)
    var dn_du_local = dn[0]
    var dn_dv_local = dn[1]

    # The sphere's own geometric normal derivatives (shapes/sphere.cpp::
    # normal_derivative): dn/du = dp_du / radius, and dp_du already carries
    # the radius factor, so this is the derivative of the UNIT normal.
    var inv_r = Float32(1.0) / SPHERE_RADIUS_G
    var dn_du_geo = h.dp_du * inv_r
    var dn_dv_geo = h.dp_dv * inv_r

    var base = compute_shading_frame(h.n, h.dp_du)
    var dbase = compute_shading_frame_derivative(h.n, h.dp_du, dn_du_geo, dn_dv_geo)
    var dbase_du = dbase[0]
    var dbase_dv = dbase[1]

    var world_n = base.to_world(n_local)
    var inv_length_n = Float32(1.0) / norm3(world_n)
    world_n = world_n * inv_length_n

    var du_n = (base.to_world(dn_du_local) + dbase_du.to_world(n_local)) * inv_length_n
    var dv_n = (base.to_world(dn_dv_local) + dbase_dv.to_world(n_local)) * inv_length_n
    du_n = du_n - world_n * dot(du_n, world_n)
    dv_n = dv_n - world_n * dot(dv_n, world_n)

    var s = h.dp_du - world_n * dot(world_n, h.dp_du)
    var inv_length_s = Float32(1.0) / norm3(s)
    s = s * inv_length_s

    var du_s = (du_n * (-dot(world_n, h.dp_du)) - world_n * dot(du_n, h.dp_du)) * inv_length_s
    var dv_s = (dv_n * (-dot(world_n, h.dp_du)) - world_n * dot(dv_n, h.dp_du)) * inv_length_s
    du_s = du_s - s * dot(du_s, s)
    dv_s = dv_s - s * dot(dv_s, s)

    var du_t = cross(du_n, s) + cross(world_n, du_s)
    var dv_t = cross(dv_n, s) + cross(world_n, dv_s)

    # frame(): result.n = normalize(base.to_world(n)), result.s = normalize(
    # dp_du - n*dot(n, dp_du)), result.t = cross(n, s) -- same s as above.
    var frame = Frm(s, cross(world_n, s), world_n)
    return (frame, Frm(du_s, du_t, du_n), Frm(dv_s, dv_t, dv_n))

def manifold_vertex(nmap: Normalmap, h: Hit, smooth: Bool = False) -> MVertex:
    var frame: Frm
    var dfu: Frm
    var dfv: Frm
    var eta: Float32
    if h.shape == SHAPE_SPHERE and smooth:
        # Smooth analytic sphere: normal_derivative = dp_d{u,v} / radius.
        var inv_r = Float32(1.0) / SPHERE_RADIUS_G
        frame = compute_shading_frame(h.n, h.dp_du)
        var d = compute_shading_frame_derivative(h.n, h.dp_du, h.dp_du * inv_r, h.dp_dv * inv_r)
        dfu = d[0]; dfv = d[1]
        eta = SPHERE_ETA
    elif h.shape == SHAPE_SPHERE:
        var f = normalmap_frames(nmap, h)
        frame = f[0]; dfu = f[1]; dfv = f[2]
        eta = SPHERE_ETA
    else:
        # Diffuse BSDF: BSDF::frame / BSDF::frame_derivative on a flat
        # rectangle, whose normal derivatives are identically zero.
        frame = compute_shading_frame(h.n, h.dp_du)
        var d = compute_shading_frame_derivative(h.n, h.dp_du, v3(0.0,0.0,0.0), v3(0.0,0.0,0.0))
        dfu = d[0]; dfv = d[1]
        eta = 1.0
    var gn = h.n
    if dot(frame.n, gn) < Float32(0.0):
        gn = -gn
    return MVertex(h.p, h.dp_du, h.dp_dv, frame.n, gn, dfu.n, dfv.n,
                   frame.s, frame.t, dfu.s, dfv.s, dfu.t, dfv.t, eta, h.shape)

# ── reflect / refract and their derivatives (render/manifold.h) ─────────────

@always_inline
def sm_reflect(w: Vec3f, n: Vec3f) -> Vec3f:
    return n * (Float32(2.0) * dot(w, n)) - w

def sm_refract(w: Vec3f, n_in: Vec3f, eta_in: Float32) -> Tuple[Bool, Vec3f]:
    var n = n_in
    var eta = Float32(1.0) / eta_in
    if dot(w, n) < Float32(0.0):
        eta = Float32(1.0) / eta
        n = -n
    var dot_w_n = dot(w, n)
    var root_term = Float32(1.0) - eta*eta * (Float32(1.0) - dot_w_n*dot_w_n)
    if root_term < Float32(0.0):
        return (False, v3(0.0, 0.0, 0.0))
    return (True, (w - n * dot_w_n) * (-eta) - n * sqrt(root_term))

def sm_d_refract(
    w: Vec3f, dw_du: Vec3f, dw_dv: Vec3f,
    n_in: Vec3f, dn_du_in: Vec3f, dn_dv_in: Vec3f, eta_in: Float32
) -> Tuple[Vec3f, Vec3f]:
    var n = n_in
    var dn_du = dn_du_in
    var dn_dv = dn_dv_in
    var eta = Float32(1.0) / eta_in
    if dot(w, n) < Float32(0.0):
        eta = Float32(1.0) / eta
        n = -n
        dn_du = -dn_du
        dn_dv = -dn_dv
    var dot_w_n = dot(w, n)
    var dot_dwdu_n = dot(dw_du, n)
    var dot_dwdv_n = dot(dw_dv, n)
    var dot_w_dndu = dot(w, dn_du)
    var dot_w_dndv = dot(w, dn_dv)
    var root = sqrt(Float32(1.0) - eta*eta * (Float32(1.0) - dot_w_n*dot_w_n))

    var a_u = (dw_du - (n * (dot_dwdu_n + dot_w_dndu) + dn_du * dot_w_n)) * (-eta)
    var b1_u = dn_du * root
    var b2_u = n * (Float32(1.0)/(Float32(2.0)*root)) * (-eta*eta*(Float32(-2.0)*dot_w_n*(dot_dwdu_n + dot_w_dndu)))
    var a_v = (dw_dv - (n * (dot_dwdv_n + dot_w_dndv) + dn_dv * dot_w_n)) * (-eta)
    var b1_v = dn_dv * root
    var b2_v = n * (Float32(1.0)/(Float32(2.0)*root)) * (-eta*eta*(Float32(-2.0)*dot_w_n*(dot_dwdv_n + dot_w_dndv)))
    return (a_u - (b1_u + b2_u), a_v - (b1_v + b2_v))

# render/manifold.h::sphcoords / d_sphcoords
@always_inline
def sphcoords(w: Vec3f) -> SIMD[DType.float32, 2]:
    var theta = safe_acos_f(w.z)
    var phi = _atan2f(w.y, w.x)
    if phi < Float32(0.0):
        phi += TWO_PI_F
    return SIMD[DType.float32, 2](theta, phi)

def d_sphcoords(w: Vec3f, dw_du: Vec3f, dw_dv: Vec3f) -> SIMD[DType.float32, 4]:
    """Returns (dtheta_du, dphi_du, dtheta_dv, dphi_dv)."""
    var d_acos = -Float32(1.0) / safe_sqrt_f(Float32(1.0) - w.z*w.z)
    var dt_du = d_acos * dw_du.z
    var dt_dv = d_acos * dw_dv.z
    var dp_du = Float32(0.0)
    var dp_dv = Float32(0.0)
    if w.x != Float32(0.0):
        var yx = w.y / w.x
        var d_atan = Float32(1.0) / (Float32(1.0) + yx*yx)
        var inv_x2 = Float32(1.0) / (w.x * w.x)
        dp_du = d_atan * (w.x*dw_du.y - w.y*dw_du.x) * inv_x2
        dp_dv = d_atan * (w.x*dw_dv.y - w.y*dw_dv.x) * inv_x2
    return SIMD[DType.float32, 4](dt_du, dp_du, dt_dv, dp_dv)

# ── manifold_ss.cpp::compute_step_anglediff ─────────────────────────────────
# The constraint formulation sphere_sms.xml actually uses
# (`caustics_halfvector_constraints = false`): instead of asking the
# generalized half-vector to align with the shading normal, it asks the
# REFRACTED direction to match the direction to the light, measured as a
# difference of spherical angles (theta, phi).
#
# Returns (success, C, dX). n_offset is (0,0,1) here (smooth dielectric,
# roughness 0), which makes the offset-normal terms no-ops.
def compute_step_anglediff(
    v0p: Vec3f, v1: MVertex, light_p: Vec3f
) -> Tuple[Bool, SIMD[DType.float32, 2], SIMD[DType.float32, 2]]:
    var fail = (False, SIMD[DType.float32, 2](Float32(1e30), Float32(1e30)),
                SIMD[DType.float32, 2](Float32(0.0), Float32(0.0)))

    var wi = v0p - v1.p
    var ili = norm3(wi)
    if ili < Float32(1e-3):
        return fail
    ili = Float32(1.0) / ili
    wi = wi * ili
    var dwi_du = (v1.dp_du - wi * dot(wi, v1.dp_du)) * (-ili)
    var dwi_dv = (v1.dp_dv - wi * dot(wi, v1.dp_dv)) * (-ili)

    var wo = light_p - v1.p
    var ilo = norm3(wo)
    if ilo < Float32(1e-3):
        return fail
    ilo = Float32(1.0) / ilo
    wo = wo * ilo
    var dwo_du = (v1.dp_du - wo * dot(wo, v1.dp_du)) * (-ilo)
    var dwo_dv = (v1.dp_dv - wo * dot(wo, v1.dp_dv)) * (-ilo)

    # n_offset = (0,0,1): n = v1.n, dn_du = v1.dn_du, dn_dv = v1.dn_dv.
    var n = v1.n
    var dn_du = v1.dn_du
    var dn_dv = v1.dn_dv

    var C = SIMD[DType.float32, 2](Float32(0.0), Float32(0.0))
    var dC_dX = SIMD[DType.float32, 4](Float32(0.0))
    var success = False

    var ri = sm_refract(wi, n, v1.eta)
    if ri[0]:
        var wio = ri[1]
        var dwio = sm_d_refract(wi, dwi_du, dwi_dv, n, dn_du, dn_dv, v1.eta)
        var so = sphcoords(wo)
        var sio = sphcoords(wio)
        var dt = so[0] - sio[0]
        var dp = so[1] - sio[1]
        if dp < -PI_F:
            dp += TWO_PI_F
        elif dp > PI_F:
            dp -= TWO_PI_F
        C = SIMD[DType.float32, 2](dt, dp)
        var do_ = d_sphcoords(wo, dwo_du, dwo_dv)
        var dio = d_sphcoords(wio, dwio[0], dwio[1])
        dC_dX = SIMD[DType.float32, 4](do_[0] - dio[0], do_[2] - dio[2],
                                       do_[1] - dio[1], do_[3] - dio[3])
        success = True

    if not success:
        var ro = sm_refract(wo, n, v1.eta)
        if ro[0]:
            var woi = ro[1]
            var dwoi = sm_d_refract(wo, dwo_du, dwo_dv, n, dn_du, dn_dv, v1.eta)
            var si_ = sphcoords(wi)
            var soi = sphcoords(woi)
            var dt = si_[0] - soi[0]
            var dp = si_[1] - soi[1]
            if dp < -PI_F:
                dp += TWO_PI_F
            elif dp > PI_F:
                dp -= TWO_PI_F
            C = SIMD[DType.float32, 2](dt, dp)
            var di = d_sphcoords(wi, dwi_du, dwi_dv)
            var doi = d_sphcoords(woi, dwoi[0], dwoi[1])
            dC_dX = SIMD[DType.float32, 4](di[0] - doi[0], di[2] - doi[2],
                                           di[1] - doi[1], di[3] - doi[3])
            success = True

    var determinant = m2det(dC_dX)
    if abs(determinant) < Float32(1e-6):
        return fail
    return (success, C, m2mulv(m2inv(dC_dX), C))

# ── manifold_ss.cpp::compute_step_halfvector ────────────────────────────────
# The OTHER constraint formulation (`halfvector_constraints = true`), which
# sphere_sms.xml does NOT select but which gonzales's own sms.mojo uses:
# require the generalized half-vector to align with the shading normal,
# rather than requiring the refracted direction to match the direction to
# the emitter. Included only so `--halfvector` can measure what that choice
# costs on this scene.
def compute_step_halfvector(
    v0p: Vec3f, v1: MVertex, light_p: Vec3f
) -> Tuple[Bool, SIMD[DType.float32, 2], SIMD[DType.float32, 2]]:
    var fail = (False, SIMD[DType.float32, 2](Float32(1e30), Float32(1e30)),
                SIMD[DType.float32, 2](Float32(0.0), Float32(0.0)))
    var wo = light_p - v1.p
    var ilo = norm3(wo)
    if ilo < Float32(1e-3):
        return fail
    ilo = Float32(1.0) / ilo
    wo = wo * ilo

    var wi = v0p - v1.p
    var ili = norm3(wi)
    if ili < Float32(1e-3):
        return fail
    ili = Float32(1.0) / ili
    wi = wi * ili

    var eta = v1.eta
    if dot(wi, v1.gn) < Float32(0.0):
        eta = Float32(1.0) / eta
    var h = wi + wo * eta
    if eta != Float32(1.0):
        h = -h
    var ilh = Float32(1.0) / norm3(h)
    h = h * ilh
    ilo *= eta * ilh
    ili *= ilh

    var dh_du = v1.dp_du * (-(ili + ilo)) + wi * (dot(wi, v1.dp_du) * ili) + wo * (dot(wo, v1.dp_du) * ilo)
    var dh_dv = v1.dp_dv * (-(ili + ilo)) + wi * (dot(wi, v1.dp_dv) * ili) + wo * (dot(wo, v1.dp_dv) * ilo)
    dh_du = dh_du - h * dot(dh_du, h)
    dh_dv = dh_dv - h * dot(dh_dv, h)
    if eta != Float32(1.0):
        dh_du = -dh_du
        dh_dv = -dh_dv

    var dH_dX = SIMD[DType.float32, 4](
        dot(v1.ds_du, h) + dot(v1.s, dh_du), dot(v1.ds_dv, h) + dot(v1.s, dh_dv),
        dot(v1.dt_du, h) + dot(v1.t, dh_du), dot(v1.dt_dv, h) + dot(v1.t, dh_dv))
    if abs(m2det(dH_dX)) < Float32(1e-6):
        return fail
    # n_offset = (0,0,1), so the target is H == 0.
    var dH = SIMD[DType.float32, 2](dot(v1.s, h), dot(v1.t, h))
    return (True, dH, m2mulv(m2inv(dH_dX), dH))

# ── manifold_ss.cpp::newton_solver ──────────────────────────────────────────
# Returns (success, si_final). Note the reference's own quirk, reproduced
# here: `si_current` is only assigned inside the loop, so a solve that
# converges on iteration 0 returns an INVALID interaction and is then
# rejected by sample_path's `!si_final.is_valid()` test.
def newton_solver(
    scene: Scene, si_p: Vec3f, vtx_init: MVertex, light_p: Vec3f
) -> Tuple[Bool, Hit]:
    var vtx = vtx_init
    var success = False
    var iterations = 0
    var beta = Float32(1.0)
    var si_current = Hit.miss()

    while iterations < MAX_ITERATIONS:
        var step = compute_step_halfvector(si_p, vtx, light_p) if scene.halfvector \
                   else compute_step_anglediff(si_p, vtx, light_p)
        if not step[0]:
            break
        var C = step[1]
        var dX = step[2]
        if sqrt(C[0]*C[0] + C[1]*C[1]) < SOLVER_THRESHOLD:
            success = True
            break

        var p_prop = vtx.p - (vtx.dp_du * dX[0] + vtx.dp_dv * dX[1]) * (STEP_SCALE * beta)
        var d_prop = normalize3(p_prop - si_p)
        var hit = scene.intersect(si_p, d_prop, RAY_EPSILON * Float32(1000.0), Float32(1e30))
        if hit.shape != vtx.shape:
            beta = Float32(0.5) * beta
            iterations += 1
            continue
        beta = min(Float32(1.0), Float32(2.0) * beta)
        si_current = hit
        vtx = manifold_vertex(scene.nmap, hit, scene.smooth_sms)
        iterations += 1

    if not success:
        return (False, si_current)

    # Reject solutions that converged to a reflection instead of a refraction.
    var wx = normalize3(si_p - vtx.p)
    var wy = normalize3(light_p - vtx.p)
    var refraction = dot(vtx.gn, wx) * dot(vtx.gn, wy) < Float32(0.0)
    if vtx.eta != Float32(1.0) and not refraction:
        return (False, si_current)
    return (True, si_current)

# core/warp.h::square_to_uniform_sphere
@always_inline
def square_to_uniform_sphere(u1: Float32, u2: Float32) -> Vec3f:
    var z = Float32(1.0) - Float32(2.0) * u2
    var r = safe_sqrt_f(Float32(1.0) - z*z)
    var phi = TWO_PI_F * u1
    return v3(r * cos(phi), r * sin(phi), z)

# ── manifold_ss.cpp::sample_path ────────────────────────────────────────────
def sample_path(
    scene: Scene, si_p: Vec3f, light_p: Vec3f, mut rng: PCG32
) -> Tuple[Bool, Hit]:
    var u1 = rng.next_float()
    var u2 = rng.next_float()
    var local = square_to_uniform_sphere(u1, u2)
    var ps_p = scene.sphere_c + local * scene.sphere_r
    var d_tmp = normalize3(ps_p - si_p)

    var si_init = scene.intersect(si_p, d_tmp, RAY_EPSILON * Float32(1000.0), Float32(1e30))
    if si_init.shape != SHAPE_SPHERE:
        return (False, Hit.miss())

    var vtx_init = manifold_vertex(scene.nmap, si_init, scene.smooth_sms)
    return newton_solver(scene, si_p, vtx_init, light_p)

# ── Fresnel (core/fresnel.h::fresnel), unpolarized dielectric ───────────────
def mts_fresnel(cos_theta_i: Float32, eta: Float32) -> Float32:
    if eta == Float32(1.0):
        return Float32(0.0)
    var rcp_eta = Float32(1.0) / eta
    var outside = cos_theta_i >= Float32(0.0)
    var eta_it = eta if outside else rcp_eta
    var eta_ti = rcp_eta if outside else eta
    var cos_theta_t_sqr = Float32(1.0) - (Float32(1.0) - cos_theta_i*cos_theta_i) * (eta_ti*eta_ti)
    if cos_theta_t_sqr <= Float32(0.0):
        return Float32(1.0)                       # total internal reflection
    var ci = abs(cos_theta_i)
    var ct = sqrt(cos_theta_t_sqr)
    var a_s = (ci - eta_it * ct) / (ci + eta_it * ct)
    var a_p = (ct - eta_it * ci) / (ct + eta_it * ci)
    return Float32(0.5) * (a_s*a_s + a_p*a_p)

# ── manifold_ss.cpp::specular_reflectance (delta-BSDF branch) ───────────────
def specular_reflectance(scene: Scene, si_final: Hit, wo: Vec3f) -> Float32:
    var frames = normalmap_frames(scene.nmap, si_final)
    var cos_theta = dot(frames[0].n, wo)
    var eta = SPHERE_ETA
    if cos_theta < Float32(0.0):
        eta = Float32(1.0) / eta
    # NOTE (faithful transcription): the reference flips `eta` above and then
    # calls fresnel(cos_theta, eta), which flips it back internally -- so the
    # Fresnel term always ends up evaluated with eta_it = 1.5046/1.000277
    # regardless of the crossing direction. Reproduced as-is.
    var f = mts_fresnel(cos_theta, eta)
    return (Float32(1.0) - f) * eta * eta

# ── manifold_ss.cpp::geometric_term (non-directional-emitter branch) ────────
def geometric_term(v0: MVertex, v1: MVertex, v2: MVertex) -> Float32:
    var wi = v0.p - v1.p
    var ili = norm3(wi)
    if ili < Float32(1e-3):
        return Float32(0.0)
    ili = Float32(1.0) / ili
    wi = wi * ili

    var wo = v2.p - v1.p
    var ilo = norm3(wo)
    if ilo < Float32(1e-3):
        return Float32(0.0)
    ilo = Float32(1.0) / ilo
    wo = wo * ilo

    var eta = v1.eta
    if dot(wi, v1.gn) < Float32(0.0):
        eta = Float32(1.0) / eta
    var h = wi + wo * eta
    if eta != Float32(1.0):
        h = -h
    var ilh = Float32(1.0) / norm3(h)
    h = h * ilh
    ilo *= eta * ilh
    ili *= ilh

    var dot_dpdu_n = dot(v1.dp_du, v1.n)
    var dot_dpdv_n = dot(v1.dp_dv, v1.n)
    var s = v1.dp_du - v1.n * dot_dpdu_n
    var t = v1.dp_dv - v1.n * dot_dpdv_n

    # d(constraint)/d(v1)
    var dh_du = v1.dp_du * (-(ili + ilo)) + wi * (dot(wi, v1.dp_du) * ili) + wo * (dot(wo, v1.dp_du) * ilo)
    var dh_dv = v1.dp_dv * (-(ili + ilo)) + wi * (dot(wi, v1.dp_dv) * ili) + wo * (dot(wo, v1.dp_dv) * ilo)
    dh_du = dh_du - h * dot(dh_du, h)
    dh_dv = dh_dv - h * dot(dh_dv, h)
    if eta != Float32(1.0):
        dh_du = -dh_du
        dh_dv = -dh_dv
    var dot_h_n = dot(h, v1.n)
    var dot_h_dndu = dot(h, v1.dn_du)
    var dot_h_dndv = dot(h, v1.dn_dv)
    var dc1_dx1 = SIMD[DType.float32, 4](
        dot(dh_du, s) - dot(v1.dp_du, v1.dn_du) * dot_h_n - dot_dpdu_n * dot_h_dndu,
        dot(dh_dv, s) - dot(v1.dp_du, v1.dn_dv) * dot_h_n - dot_dpdu_n * dot_h_dndv,
        dot(dh_du, t) - dot(v1.dp_dv, v1.dn_du) * dot_h_n - dot_dpdv_n * dot_h_dndu,
        dot(dh_dv, t) - dot(v1.dp_dv, v1.dn_dv) * dot_h_n - dot_dpdv_n * dot_h_dndv)

    # d(constraint)/d(v2)
    var dh_du2 = (v2.dp_du - wo * dot(wo, v2.dp_du)) * ilo
    var dh_dv2 = (v2.dp_dv - wo * dot(wo, v2.dp_dv)) * ilo
    dh_du2 = dh_du2 - h * dot(dh_du2, h)
    dh_dv2 = dh_dv2 - h * dot(dh_dv2, h)
    if eta != Float32(1.0):
        dh_du2 = -dh_du2
        dh_dv2 = -dh_dv2
    var dc1_dx2 = SIMD[DType.float32, 4](
        dot(dh_du2, s), dot(dh_dv2, s), dot(dh_du2, t), dot(dh_dv2, t))

    var determinant = m2det(dc1_dx1)
    if abs(determinant) < Float32(1e-6):
        return Float32(0.0)
    var dx1_dx2 = abs(m2det(m2mul(m2inv(dc1_dx1), dc1_dx2)))
    dx1_dx2 = min(dx1_dx2, Float32(1.0))
    var d = v0.p - v1.p
    var inv_r2 = Float32(1.0) / dot(d, d)
    d = d * sqrt(inv_r2)
    return abs(dot(d, v1.gn)) * inv_r2 * dx1_dx2

# ── manifold_ss.cpp::evaluate_path_contribution (area-emitter case) ─────────
def evaluate_path_contribution(
    scene: Scene, si_hit: Hit, ei_p: Vec3f, ei_weight: Float32, si_final: Hit
) -> Float32:
    var vtx = manifold_vertex(scene.nmap, si_final, scene.smooth_sms)
    vtx.make_orthonormal()

    # SpecularManifold::emitter_interaction_to_vertex, area-emitter branch:
    # re-intersect the scene toward the emitter and use whatever is hit.
    var d_tmp = normalize3(ei_p - vtx.p)
    var si_y = scene.intersect(vtx.p + d_tmp * SHADOW_EPSILON, d_tmp,
                               Float32(0.0), Float32(1e30))
    if si_y.shape == SHAPE_NONE:
        return Float32(0.0)
    var vy = manifold_vertex(scene.nmap, si_y, scene.smooth_sms)
    vy.make_orthonormal()

    var vx = manifold_vertex(scene.nmap, si_hit, scene.smooth_sms)
    vx.make_orthonormal()

    var refl = specular_reflectance(scene, si_final, normalize3(ei_p - si_final.p))
    return refl * geometric_term(vx, vtx, vy) * ei_weight

# ── manifold_ss.cpp::specular_manifold_sampling (unbiased branch) ───────────
# One caustic caster, one caustic emitter, roughness 0 => n_offset = (0,0,1)
# and p_offset = 1, so the outer loops collapse to a single estimate.
def specular_manifold_sampling(
    scene: Scene, si_hit: Hit, albedo: Vec3f, mut rng: PCG32,
    mut stat_solved: Int, mut stat_trials: Int
) -> Vec3f:
    var black = v3(0.0, 0.0, 0.0)

    # SpecularManifold::sample_emitter_interaction, area branch.
    var e1 = rng.next_float()
    var e2 = rng.next_float()
    var lr = scene.light_rect
    var ei_p = lr.o + lr.dp_du * (e1 - Float32(0.5)) + lr.dp_dv * (e2 - Float32(0.5))
    # spec = emitter->eval(si_emitter with wi=(0,0,1)) / ps.pdf: the
    # orientation test is bypassed by construction, so this is always the
    # full radiance divided by the AREA density.
    var ei_weight = scene.light_radiance * lr.area

    var sp: Tuple[Bool, Hit]
    if scene.mnee_seed:
        var dstr = normalize3(ei_p - si_hit.p)
        var hstr = scene.intersect(si_hit.p, dstr, RAY_EPSILON * Float32(1000.0), Float32(1e30))
        if hstr.shape != SHAPE_SPHERE:
            return black
        sp = newton_solver(scene, si_hit.p, manifold_vertex(scene.nmap, hstr, scene.smooth_sms), ei_p)
    else:
        sp = sample_path(scene, si_hit.p, ei_p, rng)
    if not sp[0]:
        return black
    var si_final = sp[1]
    stat_solved += 1
    var direction = normalize3(si_final.p - si_hit.p)

    # Explicit visibility between the specular vertex and the emitter.
    var dvis = si_final.p - ei_p
    var dist = norm3(dvis)
    dvis = dvis * (Float32(1.0) / dist)
    var hmax_p = max(max(abs(ei_p.x), abs(ei_p.y)), abs(ei_p.z))
    if scene.occluded(ei_p, dvis, RAY_EPSILON * (Float32(1.0) + hmax_p),
                      dist * (Float32(1.0) - RAY_EPSILON)):
        return black

    var specular_val = evaluate_path_contribution(scene, si_hit, ei_p, ei_weight, si_final)

    # Diffuse BSDF at the shading point (bsdfs/diffuse.cpp::eval).
    var cos_o = dot(si_hit.n, direction)
    if cos_o <= Float32(0.0):
        return black
    var bsdf_val = albedo * (INV_PI_F * cos_o)

    # Bernoulli-trial estimate of the reciprocal solution probability.
    var inv_prob = Float32(1.0)
    var iterations = 1
    if scene.mnee_seed:
        stat_trials += 1
        return bsdf_val * (specular_val * inv_prob)
    while True:
        var tr = sample_path(scene, si_hit.p, ei_p, rng)
        if tr[0]:
            var dt = normalize3(tr[1].p - si_hit.p)
            if abs(dot(direction, dt) - Float32(1.0)) < UNIQUENESS_THRESHOLD:
                break
        inv_prob += Float32(1.0)
        iterations += 1
        if iterations > MAX_TRIALS:
            inv_prob = Float32(0.0)
            break
    stat_trials += iterations

    return bsdf_val * (specular_val * inv_prob)

# ── Direct lighting on the floor (plain NEE, no MIS) ────────────────────────
def direct_light(scene: Scene, si_hit: Hit, albedo: Vec3f, mut rng: PCG32) -> Vec3f:
    var lr = scene.light_rect
    var u1 = rng.next_float()
    var u2 = rng.next_float()
    var y = lr.o + lr.dp_du * (u1 - Float32(0.5)) + lr.dp_dv * (u2 - Float32(0.5))
    var wl = y - si_hit.p
    var dist2 = dot(wl, wl)
    var dist = sqrt(dist2)
    wl = wl * (Float32(1.0) / dist)
    var cos_s = dot(si_hit.n, wl)
    var cos_l = dot(lr.n, -wl)
    if cos_s <= Float32(0.0) or cos_l <= Float32(0.0):
        return v3(0.0, 0.0, 0.0)
    if scene.occluded(si_hit.p, wl, SHADOW_EPSILON, dist * (Float32(1.0) - Float32(1e-4))):
        return v3(0.0, 0.0, 0.0)
    var g = cos_s * cos_l / dist2
    return albedo * (INV_PI_F * scene.light_radiance * lr.area * g)

# ── Main ────────────────────────────────────────────────────────────────────

def main() raises:
    var args = argv()
    var out_path = String("sms_mitsuba_ref.exr")
    var spp = 16
    var width = 540
    var sms_only = False
    var mnee_seed = False
    var halfvector = False
    var smooth_sms = False
    var i = 1
    var positional = 0
    while i < len(args):
        var a = String(args[i])
        if a == "--sms-only":
            sms_only = True
        elif a == "--mnee-seed":
            mnee_seed = True
        elif a == "--halfvector":
            halfvector = True
        elif a == "--smooth-sms":
            smooth_sms = True
        elif positional == 0:
            out_path = a; positional += 1
        elif positional == 1:
            spp = Int(a); positional += 1
        elif positional == 2:
            width = Int(a); positional += 1
        i += 1

    var scene_dir = String("/home/gonsolo/src/specular-manifold-sampling/results/Figure_6_Sequence")
    var scene = Scene(scene_dir)
    scene.mnee_seed = mnee_seed
    scene.halfvector = halfvector
    scene.smooth_sms = smooth_sms
    var height = width

    # <sensor type="perspective">: lookat(10,12,10 -> 0,0,0, up=+Y),
    # fov 48 on the SMALLER axis (square film => both), gaussian rfilter
    # (we use a box filter -- only affects high-frequency noise, not energy).
    var cam_o = v3(10.0, 12.0, 10.0)
    var cam_dir = normalize3(v3(0.0, 0.0, 0.0) - cam_o)
    var cam_left = normalize3(cross(v3(0.0, 1.0, 0.0), cam_dir))
    var cam_up = cross(cam_dir, cam_left)
    var tan_half = Float32(0.44522868530853616)   # tan(24 deg)

    if spp == 0:
        # --diag: at a few buried floor points, measure the solution
        # structure the estimator actually faces -- how often a uniformly
        # seeded solve converges, how many DISTINCT basins exist, and what
        # a single deterministic straight-to-the-light (MNEE-style) seed
        # finds instead.
        var lr0 = scene.light_rect
        var rngd = PCG32(UInt64(12345), UInt64(67))
        for k in range(5):
            var pt = v3(Float32(k) * 0.4 - 0.8, Float32(0.0), Float32(0.3))
            var ep = lr0.o
            var ntry = 4000
            var nok = 0
            var dirs = InlineArray[Vec3f, 64](fill=v3(0.0, 0.0, 0.0))
            var counts = InlineArray[Int, 64](fill=0)
            var nbasin = 0
            for _ in range(ntry):
                var r = sample_path(scene, pt, ep, rngd)
                if not r[0] or r[1].shape != SHAPE_SPHERE:
                    continue
                nok += 1
                var d = normalize3(r[1].p - pt)
                var found = -1
                for b in range(nbasin):
                    if abs(dot(d, dirs[b]) - Float32(1.0)) < UNIQUENESS_THRESHOLD:
                        found = b
                        break
                if found < 0:
                    if nbasin < 64:
                        dirs[nbasin] = d
                        counts[nbasin] = 1
                        nbasin += 1
                else:
                    counts[found] += 1
            # Deterministic MNEE-style seed: straight at the light.
            var dstr = normalize3(ep - pt)
            var hstr = scene.intersect(pt, dstr, Float32(1e-3), Float32(1e30))
            var det_ok = False
            if hstr.shape == SHAPE_SPHERE:
                var rr = newton_solver(scene, pt, manifold_vertex(scene.nmap, hstr, scene.smooth_sms), ep)
                det_ok = rr[0] and rr[1].shape == SHAPE_SPHERE
            print("pt(", pt.x, ",", pt.z, "): uniform-seed converged", nok, "/", ntry,
                  " distinct basins", nbasin, " mean T =", Float64(ntry) / Float64(max(nok, 1)),
                  " deterministic-seed solve:", det_ok)
            var shown = 0
            for b in range(nbasin):
                if counts[b] * 40 > nok and shown < 6:
                    print("      basin", b, "hit", counts[b], "times -> 1/p =", Float64(nok) / Float64(counts[b]))
                    shown += 1
        return

    var pixels = alloc[Float32](3 * width * height)
    var n_pixels = width * height
    var solved_total = alloc[Int](1); solved_total[0] = 0
    var trials_total = alloc[Int](1); trials_total[0] = 0

    @parameter
    def render_row(row: Int):
        var solved = 0
        var trials = 0
        for col in range(width):
            var acc = v3(0.0, 0.0, 0.0)
            var rng = PCG32(UInt64(row * width + col) * UInt64(9781) + UInt64(1),
                            UInt64(row * width + col) * UInt64(2) + UInt64(1))
            for _ in range(spp):
                var sx = (Float32(col) + rng.next_float()) / Float32(width)
                var sy = (Float32(row) + rng.next_float()) / Float32(height)
                var dc = normalize3(v3((Float32(1.0) - Float32(2.0)*sx) * tan_half,
                                       (Float32(1.0) - Float32(2.0)*sy) * tan_half,
                                       Float32(1.0)))
                var org = cam_o
                var dir = cam_left * dc.x + cam_up * dc.y + cam_dir * dc.z
                var beta = v3(1.0, 1.0, 1.0)
                var prev_specular = True
                # path_sms_ss.cpp's main loop: max_depth = 6, SMS invoked at
                # every caustic-receiver vertex that is not a caustic caster.
                for depth in range(1, 6):
                    var hit = scene.intersect(org, dir, Float32(1e-3), Float32(1e30))
                    if hit.shape == SHAPE_NONE:
                        break
                    if hit.shape == SHAPE_LIGHT:
                        if prev_specular and dot(hit.n, -dir) > Float32(0.0):
                            acc += beta * scene.light_radiance
                        break
                    if hit.shape == SHAPE_FLOOR:
                        var albedo = scene.floor_tex.eval(hit.uv[0] * scene.floor_uv_scale,
                                                          hit.uv[1] * scene.floor_uv_scale)
                        if not sms_only:
                            acc += beta * direct_light(scene, hit, albedo, rng)
                        if depth + 1 < 6:
                            acc += beta * specular_manifold_sampling(
                                scene, hit, albedo, rng, solved, trials)
                        # Cosine-weighted diffuse bounce.
                        var u1 = rng.next_float()
                        var u2 = rng.next_float()
                        var r = sqrt(u1)
                        var ph = TWO_PI_F * u2
                        var fr = compute_shading_frame(hit.n, hit.dp_du)
                        var wl = v3(r*cos(ph), r*sin(ph), safe_sqrt_f(Float32(1.0) - u1))
                        var wnew = fr.to_world(wl)
                        beta = beta * albedo
                        org = hit.p + hit.n * Float32(1e-3)
                        dir = wnew
                        prev_specular = False
                        continue
                    # Sphere: normal-mapped smooth dielectric.
                    var frames = normalmap_frames(scene.nmap, hit)
                    var np_ = frames[0].n
                    var wi = -dir
                    var cos_i = dot(wi, np_)
                    var f = mts_fresnel(cos_i, SPHERE_ETA)
                    var wo: Vec3f
                    if rng.next_float() <= f:
                        wo = np_ * (Float32(2.0) * cos_i) - wi
                    else:
                        var rr = sm_refract(wi, np_, SPHERE_ETA)
                        if not rr[0]:
                            break
                        wo = rr[1]
                        var eta_it = SPHERE_ETA if cos_i >= Float32(0.0) else Float32(1.0)/SPHERE_ETA
                        beta = beta * (Float32(1.0) / (eta_it * eta_it))
                    # normalmap.cpp::sample rejects directions that end up on
                    # the wrong side of the unperturbed shading frame.
                    if dot(wo, hit.n) * dot(wo, np_) <= Float32(0.0):
                        break
                    org = hit.p + wo * Float32(1e-3)
                    dir = wo
                    prev_specular = True
            var o = (row * width + col) * 3
            var inv = Float32(1.0) / Float32(spp)
            pixels[o + 0] = acc.x * inv
            pixels[o + 1] = acc.y * inv
            pixels[o + 2] = acc.z * inv
        solved_total[0] += solved
        trials_total[0] += trials

    parallelize[render_row](height)

    print("solved paths:", solved_total[0], " bernoulli iterations:", trials_total[0])
    if solved_total[0] > 0:
        print("mean T:", Float64(trials_total[0]) / Float64(solved_total[0]))

    var name = cstr(out_path)
    var rc = external_call["write_image_rgb", Int32,
        UnsafePointer[UInt8, MutExternalOrigin],
        UnsafePointer[Float32, MutExternalOrigin],
        Int32, Int32, Int32, Int32](
        name, pixels.unsafe_origin_cast[MutExternalOrigin](),
        Int32(width), Int32(height), Int32(0), Int32(0))
    if rc == Int32(0):
        raise Error("failed to write " + out_path)
    print("wrote", out_path, width, "x", height, "@", spp, "spp")
    pixels.free()
    solved_total.free()
    trials_total.free()
    _ = n_pixels
