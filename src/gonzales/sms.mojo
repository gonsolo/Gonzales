# Specular Manifold Sampling, Phase 5 of docs/A2_restir_migration_plan.md.
#
# Generalizes shading.mojo's 1-/2-vertex MNEE manifold walk (_mnee_walk,
# _mnee_walk2) to an arbitrary-length chain of specular vertices between a
# shading point and a light. MNEE's own per-vertex machinery -- the
# tangential half-vector constraint and its (a, b, c) coupling-matrix
# derivatives (Cycles mnee.h's formulas) -- already generalizes to N
# vertices via a block-tridiagonal solve; MNEE itself just special-cased
# N=1/N=2 to skip the general solve's indexing overhead. Those two fast
# paths stay in shading.mojo untouched (5.4); this file only carries N>=3.
#
# Also adds what a fixed-length 1-/2-vertex solve never needed: random
# seeding within each probed triangle (5.2) and a Bernoulli-trial
# reciprocal estimator (5.3), following Zeltner et al. 2020 ("Specular
# Manifold Sampling for Rendering High-Frequency Caustics and Glints").
# For a flat single-triangle refraction (MNEE's own scope) the manifold
# solution given a starting probe is essentially unique, so a deterministic
# seed is enough. For a chain of 3+ specular vertices that uniqueness
# argument no longer holds in general -- a single deterministic Newton
# solve seeded at the straight-line probe hit is not guaranteed to be the
# ONLY solution contributing along that light direction, so the estimator
# needs the Bernoulli-trial correction to stay unbiased.
#
# The Newton solve itself is a direct generalization of _mnee_walk2's block
# tridiagonal system
#   [b0 c0      ] [dx0]   [cv0]
#   [a1 b1 c1   ] [dx1] = [cv1]
#   [   a2 b2 c2] [dx2]   [cv2]
#   [      .. ..] [.. ]   [.. ]
# via the standard block Thomas algorithm (forward elimination + back
# substitution on 2x2 blocks) -- verified by hand to reduce EXACTLY to
# _mnee_walk2's own formulas at n=2 (see test_sms.mojo's regression tests
# against _mnee_walk/_mnee_walk2 output).

from std.math import sqrt, abs, max, min, cos, sin, acos
from .geometry import RGB, dot, cross, fr_dielectric, Frame, Vec3f, Point3f, Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Curve_C, Instance_C, Sphere_C, NormalSlopeMap_C, normal_slope_map_none, _atan2f, PI, TWO_PI
from .rng import PCG32
from .bvh import ray_sphere_hit, traverse_bvh2_core, BVH2Node

comptime MAX_SMS_VERTICES: Int = 6
# Cap on the Bernoulli estimator's trial count. For a chain whose manifold
# solution is essentially unique the very first trial matches and this is
# never approached. It has to be generous for a NORMAL-MAPPED caster, where
# the whole point is that there are many solutions and each individual one is
# found by only a small fraction of seeds: the estimator's T is a sample from
# Geometric(q) with q = P(a random seed finds THIS root), so a 20-root
# surface whose seeds converge at all a third of the time needs a cap well
# past 1/q ~ 60 before truncation starts eating energy. Truncating biases
# DOWNWARD (the tail is never counted), so the cap is a cost/darkness
# trade-off, not a correctness switch; the reference renderer's own scenes
# set it to 10^7 and simply rely on the loop breaking early.
comptime SMS_BERNOULLI_MAX_TRIALS: Int = 512
# Two Newton solves count as the SAME root when the directions x0->x_i agree
# to within this much of cos=1 -- ported verbatim from the reference SMS
# renderer's own uniqueness test (manifold_ss.cpp:
# `abs(dot(direction, direction_trial) - 1.f) < m_config.uniqueness_threshold`,
# whose scenes default to 1e-5). Comparing DIRECTIONS rather than world
# POSITIONS is what makes this consistent with the solver that produced them:
# sms_walk stops at a RESIDUAL below 1e-3, so two seeds landing in the same
# basin routinely differ by far more than the 1e-4 position epsilon this
# replaced -- scoring
# those as distinct roots inflates the Bernoulli trial count T, and since the
# estimator multiplies the contribution by T, that inflation shows up
# directly as too much caustic energy (measured 1.57x too bright against the
# reference on sphere_sms.xml before this was aligned).
comptime SMS_UNIQUENESS_COS_EPS: Float32 = Float32(1e-5)

# Newton convergence: the walk stops once the L2 norm of the tangential
# constraint residual falls below this, matching the reference renderer's
# own `caustics_solver_threshold` (1e-5 in every scene that ships with it).
#
# This has to be tight for a reason beyond just solution accuracy: the
# Bernoulli estimator asks whether two independent solves found the SAME
# root, with a tolerance of SMS_UNIQUENESS_COS_EPS on the DIRECTION to the
# vertex. A loose stopping criterion lets two seeds in the same basin halt
# far enough apart to fail that test, which makes the estimator hunt for a
# root it has effectively already found -- running the trial count up to its
# cap and, through the cap's downward truncation, DARKENING the caustic. The
# 1e-3 used here before (a leftover from the flat 1-/2-vertex walks, where
# essentially every seed lands in one basin and the question never arises)
# cost roughly a third of the sphere caustic's energy that way.
comptime SMS_SOLVER_THRESHOLD: Float32 = Float32(1e-5)

@always_inline
def mnee_orthonormal_basis(du: Vec3f, dv: Vec3f) -> Tuple[Vec3f, Vec3f]:
    """Gram-Schmidt a surface's tangent pair into an orthonormal one.

    A manifold walk's |dx1/dxL| is only an AREA-to-area Jacobian when both
    the specular vertex's basis and the light's are orthonormal -- otherwise
    it comes out per-parametric-unit and no longer pairs with the
    area-measure light density the estimator divides by. The reference
    renderer makes the same call (ManifoldVertex::make_orthonormal, applied
    to its specular AND emitter vertices before building the geometric
    term).

    The bases actually handed in are neither unit nor perpendicular -- a
    triangle's raw edges (lp1-lp0, lp2-lp0) or a sphere's radius-scaled
    dp/dphi, dp/dtheta -- so this is not optional bookkeeping: getting it
    wrong scaled gonzales's sphere caustic by 0.61x until it was fixed.
    It lives here, rather than in each caller, precisely because it is an
    invariant of the walk and every caller was getting it wrong the same
    way.

    Degenerate input (zero-length or parallel) is returned unchanged, which
    leaves the caller in exactly the state it was in before -- the walk's
    own degeneracy checks reject it downstream."""
    var e1 = du
    var e1_len = sqrt(dot(e1, e1))
    if e1_len <= Float32(1e-12):
        return (du, dv)
    e1 = e1 * (Float32(1.0) / e1_len)
    var e2 = dv - e1 * dot(e1, dv)
    var e2_len = sqrt(dot(e2, e2))
    if e2_len <= Float32(1e-12):
        return (du, dv)
    return (e1, e2 * (Float32(1.0) / e2_len))

# ── 2x2 matrix helpers ───────────────────────────────────────────────────────
# Shared with shading.mojo's _mnee_walk2, which imports these from here
# rather than keeping its own private copies (Phase 5.1: one source of
# truth for the block-tridiagonal linear algebra).

@always_inline
def mat22_mul(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3], a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3])

@always_inline
def mat22_mul_v(m: SIMD[DType.float32, 4], v: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    return SIMD[DType.float32, 2](m[0]*v[0]+m[1]*v[1], m[2]*v[0]+m[3]*v[1])

@always_inline
def mat22_inv(m: SIMD[DType.float32, 4]) -> Tuple[SIMD[DType.float32, 4], Float32]:
    var det = m[0]*m[3] - m[1]*m[2]
    if abs(det) < Float32(1e-5):
        return (SIMD[DType.float32, 4](Float32(0)), Float32(0))
    return (SIMD[DType.float32, 4](m[3], -m[1], -m[2], m[0]) * (Float32(1.0)/det), det)

@always_inline
def sms_vertex_mats(
    wi: Vec3f, wo: Vec3f,
    H: Vec3f, s: Vec3f, t: Vec3f,
    dp_du: Vec3f, dp_dv: Vec3f,
    ili: Float32, ilo: Float32,
    dp_du_prev: Vec3f, dp_dv_prev: Vec3f,
    dp_du_next: Vec3f, dp_dv_next: Vec3f,
    has_prev: Bool, has_next: Bool,
) -> Tuple[SIMD[DType.float32, 4], SIMD[DType.float32, 4], SIMD[DType.float32, 4], SIMD[DType.float32, 2]]:
    """Computes (a, b, c, constraint) at one specular vertex -- a=coupling to
    prev vertex, b=self, c=coupling to next vertex. Identical formula to
    shading.mojo's former _mnee_vertex_mats (moved here so both the N=1/2
    fast paths and this file's general solve share one implementation)."""
    var b_du = -(dp_du*(ili+ilo)) + wi*(dot(wi,dp_du)*ili) + wo*(dot(wo,dp_du)*ilo)
    var b_dv = -(dp_dv*(ili+ilo)) + wi*(dot(wi,dp_dv)*ili) + wo*(dot(wo,dp_dv)*ilo)
    b_du -= H*dot(b_du,H); b_du = -b_du
    b_dv -= H*dot(b_dv,H); b_dv = -b_dv
    var b = SIMD[DType.float32, 4](dot(b_du,s), dot(b_dv,s), dot(b_du,t), dot(b_dv,t))
    var a = SIMD[DType.float32, 4](Float32(0))
    if has_prev:
        var a_du = (dp_du_prev - wi*dot(wi,dp_du_prev)) * ili
        var a_dv = (dp_dv_prev - wi*dot(wi,dp_dv_prev)) * ili
        a_du -= H*dot(a_du,H); a_du = -a_du
        a_dv -= H*dot(a_dv,H); a_dv = -a_dv
        a = SIMD[DType.float32, 4](dot(a_du,s), dot(a_dv,s), dot(a_du,t), dot(a_dv,t))
    var c = SIMD[DType.float32, 4](Float32(0))
    if has_next:
        var c_du = (dp_du_next - wo*dot(wo,dp_du_next)) * ilo
        var c_dv = (dp_dv_next - wo*dot(wo,dp_dv_next)) * ilo
        c_du -= H*dot(c_du,H); c_du = -c_du
        c_dv -= H*dot(c_dv,H); c_dv = -c_dv
        c = SIMD[DType.float32, 4](dot(c_du,s), dot(c_dv,s), dot(c_du,t), dot(c_dv,t))
    var constraint = SIMD[DType.float32, 2](dot(H,s), dot(H,t))
    return (a, b, c, constraint)

# ── N-vertex chain state ─────────────────────────────────────────────────────

@fieldwise_init
struct SMSVertex(TrivialRegisterPassable):
    """One specular vertex of the chain. `pos` is the Newton-walk variable;
    normal/dp_du/dp_dv are the vertex's LOCAL tangent-plane linearization,
    re-derived every iteration for a curved vertex (see `is_sphere` below)
    but otherwise a fixed property of a flat triangle (valid as long as the
    true manifold solution stays within that triangle -- unchanged from
    the original flat-only design). `eta` is the relative IOR crossing
    this vertex (ior if entering, 1/ior if exiting), precomputed by the
    caller from probe geometry exactly like _mnee_walk2's eta1/eta2.

    `is_sphere`/`sphere_center`/`sphere_radius`: when set, this vertex
    lies on an analytic Sphere_C rather than a flat triangle. A flat
    triangle's tangent plane IS its surface everywhere, so the ordinary
    Newton step (move within the fixed tangent plane) is exact; a
    sphere's tangent plane only agrees with the true surface at the point
    of tangency, so `sms_walk` reprojects this vertex's position back onto
    the sphere and re-derives normal/dp_du/dp_dv after every step (see
    `_sms_reproject_onto_sphere`) -- the rest of the Newton machinery
    (`sms_vertex_mats`, the block-tridiagonal solve) is completely generic
    in dp_du/dp_dv/normal and needs no other change to support this."""
    var pos:    Vec3f
    var normal: Vec3f
    var dp_du:  Vec3f
    var dp_dv:  Vec3f
    var eta:    Float32
    var is_sphere:     Int8
    var sphere_center: Vec3f
    var sphere_radius: Float32
    # d(normal)/d(arc length) along dp_du / dp_dv -- how the SHADING normal
    # turns as the vertex slides across the surface. This is the quantity
    # the frame-derivative term of the Newton Jacobian needs (see
    # _sms_eval_vertex); keeping it as explicit per-vertex state rather
    # than re-deriving it from `sphere_radius` is what lets a vertex whose
    # normal comes from a NORMAL MAP contribute its texture gradient here,
    # exactly as the reference renderer's ManifoldVertex carries dn_du/
    # dn_dv straight from `BSDF::frame_derivative`. Zero for a flat
    # triangle (fixed frame); dp_du/radius, dp_dv/radius for a smooth
    # sphere, which reproduces the analytic 1/radius curvature term this
    # generalizes.
    var dn_du:  Vec3f
    var dn_dv:  Vec3f
    # This vertex's normal map, in slope space, or `res == 0` for none.
    # Carried per-vertex rather than passed down through sms_walk because
    # the walk RE-EVALUATES it: every Newton iteration moves the vertex to
    # a new point on the sphere, which is a new texel, which is a new
    # normal and a new pair of normal derivatives. A seed-time-only lookup
    # would solve the smooth surface's constraint while reporting the
    # perturbed surface's Jacobian.
    var nmap:   NormalSlopeMap_C
    # The material's OWN relative IOR, unoriented. `eta` above is already
    # oriented for the half-vector constraint (see _sms_vertex_from_hit),
    # which is what that formulation and the BSDF product want, but the
    # angle-difference formulation orients eta for itself inside its
    # refract() -- so it needs the raw value, and reconstructing it from
    # `eta` would mean re-deriving the very sign test whose two conflicting
    # readings caused the eta^4 bug in the first place. Cheaper and far
    # less error-prone to carry both. Equal to `eta` for a flat vertex,
    # which never reaches the angle-difference path.
    var eta_raw: Float32

@always_inline
def sms_vertex_init() -> SMSVertex:
    var z = Vec3f(Float32(0.0))
    return SMSVertex(z, z, z, z, Float32(1.0), Int8(0), z, Float32(0.0), z, z, normal_slope_map_none(), Float32(1.0))

@always_inline
def sms_vertex_flat(
    pos: Vec3f,
    normal: Vec3f,
    dp_du: Vec3f,
    dp_dv: Vec3f,
    eta: Float32,
) -> SMSVertex:
    """Constructs a flat-triangle SMSVertex (is_sphere=False) -- the
    original SMSVertex shape before sphere support was added, kept as a
    named constructor so call sites don't each spell out the dummy
    sphere_center/sphere_radius padding a flat vertex never uses."""
    var z = Vec3f(Float32(0.0))
    return SMSVertex(pos, normal, dp_du, dp_dv, eta, Int8(0), z, Float32(0.0), z, z, normal_slope_map_none(), eta)

@always_inline
def sms_vertex_sphere(
    pos: Vec3f,
    center: Vec3f,
    radius: Float32,
    eta: Float32,
    nmap: NormalSlopeMap_C = normal_slope_map_none(),
    eta_raw: Float32 = Float32(1.0),
) -> SMSVertex:
    """Constructs a sphere-caster SMSVertex from a probe hit's world-space
    position (assumed already ON the sphere, e.g. from a real intersection)
    -- snaps it exactly onto the sphere and derives an initial tangent
    frame via `_sms_reproject_onto_sphere`, matching what `sms_walk` itself
    will re-derive on every subsequent iteration. `nmap`, when present,
    perturbs the shading normal and its derivatives on top of that (see
    `_sms_apply_sphere_frame`)."""
    var z = Vec3f(Float32(0.0))
    var vtx = SMSVertex(pos, z, z, z, eta, Int8(1), center, radius, z, z, nmap, eta_raw)
    _sms_apply_sphere_frame(vtx, _sms_reproject_onto_sphere(pos, center, radius))
    return vtx

@always_inline
def _sms_reproject_onto_sphere(
    x_raw: Vec3f,
    center: Vec3f,
    radius: Float32,
) -> Tuple[Vec3f, Vec3f, Vec3f, Vec3f]:
    """Projects a tangent-plane Newton step back onto the true (curved)
    sphere surface, and re-derives the local normal/tangent frame there.
    A flat triangle's tangent plane IS its surface everywhere, so
    _mnee_walk/_mnee_walk2 never needed this; a sphere's tangent plane
    only agrees with the true surface at the point of tangency, so every
    Newton step must be corrected back onto the curved surface before the
    next iteration's linearization -- this is the standard manifold-Newton
    technique (project, don't just linearize), not an approximation: it
    keeps the walk exactly ON the sphere at every iteration instead of
    silently drifting into the tangent plane and away from the actual
    specular constraint.

    Returns (x_on_sphere, normal, dp_du, dp_dv). dp_du/dp_dv are an
    arbitrary (Frisvad/Duff) orthonormal tangent basis, not a true (u,v)
    sphere parameterization -- sms_vertex_mats's own math re-derives its
    working (s,t) frame from dp_du/dp_dv+normal via Gram-Schmidt every
    iteration regardless (see sms_vertex_mats/_sms_eval_vertex), so any
    non-degenerate tangent basis is equally valid; no phi/theta
    parameterization (and its pole singularity) is needed."""
    var to_x = x_raw - center
    var to_x_len = sqrt(dot(to_x, to_x))
    var normal = to_x
    var x_on_sphere = x_raw
    if to_x_len > Float32(1e-8):
        normal = to_x * (Float32(1.0) / to_x_len)
        x_on_sphere = center + normal * radius
    return _sms_sphere_frame_at(x_on_sphere, normal)

@always_inline
def _sms_sphere_frame_at(
    x_on_sphere: Vec3f,
    normal: Vec3f,
) -> Tuple[Vec3f, Vec3f, Vec3f, Vec3f]:
    var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
    var dp_du = Vec3f(frame.x.x, frame.x.y, frame.x.z)
    var dp_dv = Vec3f(frame.y.x, frame.y.y, frame.y.z)
    return (x_on_sphere, normal, dp_du, dp_dv)

# ── Normal-mapped specular vertices ──────────────────────────────────────────
# A manifold walk over a normal-mapped surface needs three things the smooth
# analytic surface does not provide: the PERTURBED shading normal (the
# specular constraint is stated in terms of it), and its two first
# derivatives (the constraint Jacobian's frame-derivative term -- see
# _sms_eval_vertex -- is built from them). Ordinary shading only ever needs
# the first, which is why `_apply_normal_map_sphere` in shading.mojo, the
# camera-ray path's own normal-map lookup, stops there.
#
# All three come out of the same four texels of the LEAN slope map
# (NormalSlopeMap_C), interpolated analytically, exactly as the reference
# does it (render/normalmap.h, `use_slopes=true`).

@always_inline
def _nmap_addr(res: Int, u: Float32, v: Float32) -> Tuple[Int, Int, Int, Int, Float32, Float32]:
    """The shared bilinear addressing of the reference's eval_normal and
    eval_normal_derivatives: half-texel offset, CLAMP (not wrap), and no
    V flip. Returns (x0, y0, x1, y1, wx, wy)."""
    var resf = Float32(res)
    var duv = Float32(1.0) / resf
    var px = (u - Float32(0.5)*duv) * resf
    var py = (v - Float32(0.5)*duv) * resf
    px = min(max(px, Float32(0.0)), resf - Float32(1.0))
    py = min(max(py, Float32(0.0)), resf - Float32(1.0))
    var x0 = Int(px); var y0 = Int(py)
    if x0 > res - 1: x0 = res - 1
    if y0 > res - 1: y0 = res - 1
    var x1 = x0 + 1; var y1 = y0 + 1
    if x1 > res - 1: x1 = res - 1
    if y1 > res - 1: y1 = res - 1
    return (x0, y0, x1, y1, px - Float32(x0), py - Float32(y0))

@always_inline
def _nmap_slope(m: NormalSlopeMap_C, x: Int, y: Int) -> SIMD[DType.float32, 2]:
    var i = (y * Int(m.res) + x) * 2
    return SIMD[DType.float32, 2](m.slopes[i], m.slopes[i + 1])

@always_inline
def nmap_eval(m: NormalSlopeMap_C, u: Float32, v: Float32) -> Vec3f:
    """Bilinearly interpolated slope, returned as the UNNORMALIZED local
    (tangent-space) normal `(-sx, -sy, 1)`."""
    var a = _nmap_addr(Int(m.res), u, v)
    var w1x = a[4]; var w1y = a[5]
    var w0x = Float32(1.0) - w1x; var w0y = Float32(1.0) - w1y
    var v00 = _nmap_slope(m, a[0], a[1]); var v10 = _nmap_slope(m, a[2], a[1])
    var v01 = _nmap_slope(m, a[0], a[3]); var v11 = _nmap_slope(m, a[2], a[3])
    var r0 = v00 * w0x + v10 * w1x
    var r1 = v01 * w0x + v11 * w1x
    var sl = r0 * w0y + r1 * w1y
    return Vec3f(-sl[0], -sl[1], Float32(1.0))

@always_inline
def nmap_eval_derivs(m: NormalSlopeMap_C, u: Float32, v: Float32) -> Tuple[Vec3f, Vec3f]:
    """d(local normal)/du and /dv -- the exact analytic derivative of
    `nmap_eval`'s bilinear interpolant, from the same four texels."""
    var a = _nmap_addr(Int(m.res), u, v)
    var wx = a[4]; var wy = a[5]
    var v00 = _nmap_slope(m, a[0], a[1]); var v10 = _nmap_slope(m, a[2], a[1])
    var v01 = _nmap_slope(m, a[0], a[3]); var v11 = _nmap_slope(m, a[2], a[3])
    var resf = Float32(Int(m.res))
    var tmp = v01 + v10 - v11
    var tu = (v10 + v00 * (wy - Float32(1.0)) - tmp * wy) * resf
    var tv = (v01 + v00 * (wx - Float32(1.0)) - tmp * wx) * resf
    return (Vec3f(-tu[0], -tu[1], Float32(0.0)), Vec3f(-tv[0], -tv[1], Float32(0.0)))

@always_inline
def _sms_sphere_nmap_frame(
    m: NormalSlopeMap_C,
    x_on_sphere: Vec3f,
    center: Vec3f,
    radius: Float32,
    n_geo: Vec3f,
    dp_du: Vec3f,
    dp_dv: Vec3f,
) -> Tuple[Vec3f, Vec3f, Vec3f, Bool]:
    """The normal-mapped counterpart of `_sms_sphere_frame_at`'s smooth
    normal: returns (shading_normal, dn_du, dn_dv, ok).

    (u, v) come from the same spherical parameterization the camera-ray
    path uses (`_apply_normal_map_sphere` -- Mitsuba's own convention:
    polar angle from +Z, phi = atan2(y, x)), so the walk and the shading
    that follows it see the SAME perturbed surface.

    `dn_du`/`dn_dv` are returned differentiated with respect to
    `dp_du`/`dp_dv` -- the caller's own (arbitrary, Frisvad) tangent basis,
    NOT the sphere's (u, v). The normal map's derivatives are naturally
    per-texture-coordinate, so they are first assembled into the linear map
    "world tangential displacement -> change in normal" and then applied to
    the caller's basis vectors; that map is basis-independent, which is
    what lets the walk keep using the pole-free Frisvad frame it already
    reprojects onto. As a check on the construction: with a flat normal map
    this collapses to dn_du = dp_du/radius, dn_dv = dp_dv/radius -- exactly
    the smooth-sphere values `_sms_reproject_onto_sphere`'s caller sets.

    ok=False at the parameterization's poles (where the u tangent vanishes
    and no meaningful texture frame exists); the caller falls back to the
    smooth sphere there, as the camera-ray path already does."""
    var local = x_on_sphere - center
    var rd2 = local[0]*local[0] + local[1]*local[1]
    var rd = sqrt(rd2)
    if rd <= Float32(1e-6) or radius <= Float32(1e-8):
        return (n_geo, Vec3f(Float32(0.0)), Vec3f(Float32(0.0)), False)
    var theta = _atan2f(rd, local[2])
    var phi = _atan2f(local[1], local[0])
    if phi < Float32(0.0):
        phi += Float32(2.0) * PI
    var u = phi / (Float32(2.0) * PI)
    var v = theta / PI
    # Sphere::compute_surface_interaction's parametric tangents.
    var e_u = Vec3f(-local[1], local[0], Float32(0.0)) * (Float32(2.0) * PI)
    var inv_rd = Float32(1.0) / rd
    var e_v = Vec3f(local[2] * local[0] * inv_rd, local[2] * local[1] * inv_rd, -rd) * PI
    var eu2 = dot(e_u, e_u)
    var ev2 = dot(e_v, e_v)
    if eu2 <= Float32(1e-12) or ev2 <= Float32(1e-12):
        return (n_geo, Vec3f(Float32(0.0)), Vec3f(Float32(0.0)), False)
    # Base (geometric) shading frame and its derivative, core/frame.h's
    # compute_shading_frame / compute_shading_frame_derivative. The smooth
    # sphere's own normal derivatives are dp_d{u,v}/radius.
    var inv_r = Float32(1.0) / radius
    var dnu_geo = e_u * inv_r
    var dnv_geo = e_v * inv_r
    var s_un = e_u - n_geo * dot(n_geo, e_u)
    var s_len2 = dot(s_un, s_un)
    if s_len2 <= Float32(1e-12):
        return (n_geo, Vec3f(Float32(0.0)), Vec3f(Float32(0.0)), False)
    var inv_len_s = Float32(1.0) / sqrt(s_len2)
    var bs = s_un * inv_len_s
    var bt = cross(n_geo, bs)
    var dot_n_dpdu = dot(n_geo, e_u)
    var du_s = (dnu_geo * (-dot_n_dpdu) - n_geo * dot(dnu_geo, e_u)) * inv_len_s
    var dv_s = (dnv_geo * (-dot_n_dpdu) - n_geo * dot(dnv_geo, e_u)) * inv_len_s
    du_s = du_s - bs * dot(du_s, bs)
    dv_s = dv_s - bs * dot(dv_s, bs)
    var du_t = cross(dnu_geo, bs) + cross(n_geo, du_s)
    var dv_t = cross(dnv_geo, bs) + cross(n_geo, dv_s)
    # Perturbed normal and its (u, v) derivatives, bsdfs/normalmap.cpp's
    # frame() / frame_derivative().
    var nl = nmap_eval(m, u, v)
    var dnl = nmap_eval_derivs(m, u, v)
    var world_n = bs * nl[0] + bt * nl[1] + n_geo * nl[2]
    var wn_len2 = dot(world_n, world_n)
    if wn_len2 <= Float32(1e-12):
        return (n_geo, Vec3f(Float32(0.0)), Vec3f(Float32(0.0)), False)
    var inv_len_n = Float32(1.0) / sqrt(wn_len2)
    world_n = world_n * inv_len_n
    var du_n = (bs * dnl[0][0] + bt * dnl[0][1] + n_geo * dnl[0][2]
                + du_s * nl[0] + du_t * nl[1] + dnu_geo * nl[2]) * inv_len_n
    var dv_n = (bs * dnl[1][0] + bt * dnl[1][1] + n_geo * dnl[1][2]
                + dv_s * nl[0] + dv_t * nl[1] + dnv_geo * nl[2]) * inv_len_n
    du_n = du_n - world_n * dot(du_n, world_n)
    dv_n = dv_n - world_n * dot(dv_n, world_n)
    # Re-express in the caller's tangent basis. e_u and e_v are orthogonal
    # on a sphere, so the displacement -> (du, dv) inverse is just two
    # projections.
    var iu = Float32(1.0) / eu2
    var iv = Float32(1.0) / ev2
    var dn_du = du_n * (dot(dp_du, e_u) * iu) + dv_n * (dot(dp_du, e_v) * iv)
    var dn_dv = du_n * (dot(dp_dv, e_u) * iu) + dv_n * (dot(dp_dv, e_v) * iv)
    return (world_n, dn_du, dn_dv, True)

@always_inline
def _sms_apply_sphere_frame(mut vtx: SMSVertex, r: Tuple[Vec3f, Vec3f, Vec3f, Vec3f]):
    """Writes a freshly reprojected sphere frame into `vtx`, applying the
    vertex's normal map (if it has one) on top of the smooth frame. The one
    place that decides what a curved vertex's shading normal and normal
    derivatives are, shared by the seed constructor and every Newton
    iteration -- they MUST agree, or the walk converges to the root of a
    different surface than the one whose Jacobian it reports."""
    vtx.pos = r[0]; vtx.dp_du = r[2]; vtx.dp_dv = r[3]
    var inv_r = Float32(1.0) / vtx.sphere_radius if vtx.sphere_radius > Float32(1e-8) else Float32(0.0)
    if vtx.nmap.res > Int32(0):
        var nm = _sms_sphere_nmap_frame(vtx.nmap, r[0], vtx.sphere_center, vtx.sphere_radius,
                                        r[1], r[2], r[3])
        if nm[3]:
            vtx.normal = nm[0]; vtx.dn_du = nm[1]; vtx.dn_dv = nm[2]
            return
    vtx.normal = r[1]
    vtx.dn_du = r[2] * inv_r
    vtx.dn_dv = r[3] * inv_r

@always_inline
def _sms_reproject_onto_sphere_anchored(
    anchor: Vec3f,
    x_raw: Vec3f,
    center: Vec3f,
    radius: Float32,
    anchor_on_surface: Bool,
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin] = UnsafePointer[BVH2Node, MutExternalOrigin].unsafe_dangling(),
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin] = UnsafePointer[PrimId_C, MutExternalOrigin].unsafe_dangling(),
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin] = UnsafePointer[TriangleMesh_C, MutExternalOrigin].unsafe_dangling(),
    curves: UnsafePointer[Curve_C, MutExternalOrigin] = UnsafePointer[Curve_C, MutExternalOrigin].unsafe_dangling(),
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin] = UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin] = UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
    instances: UnsafePointer[Instance_C, MutExternalOrigin] = UnsafePointer[Instance_C, MutExternalOrigin].unsafe_dangling(),
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin] = UnsafePointer[Sphere_C, MutExternalOrigin].unsafe_dangling(),
    n_spheres: Int = 0,
) -> Tuple[Vec3f, Vec3f, Vec3f, Vec3f, Bool]:
    """Reprojects a raw Newton-step proposal onto the sphere by actually
    RAY-CASTING from a fixed anchor through the proposal, instead of
    `_sms_reproject_onto_sphere`'s closed-form `normalize(x_raw-center)*
    radius` snap -- ports the real SMS reference renderer's own reprojection
    strategy (`SpecularManifoldSingleScatter::newton_solver`,
    manifold_ss.cpp: re-intersects a ray from the ORIGINAL shading point
    through the proposed point with the actual scene every iteration,
    rejecting/shrinking the step if it misses or lands on a different
    shape).

    This matters because the closed-form snap has no notion of WHICH side
    of the sphere is reachable from the rest of the chain: it accepts
    whatever point is nearest to `x_raw` in ANY direction from the sphere's
    center, including the far/hidden hemisphere. Root-caused via a
    real-scene diagnostic on `sphere_sms.xml`: the coupled 2-vertex
    (entry+exit) Newton solve was converging "successfully" (small
    residual, plausible-looking BSDF/Jacobian) to entry points on the
    UNDERSIDE of the sphere -- physically unreachable, embedded below the
    scene's own floor -- 100% of the time, silently discarded by the
    caller's downstream visibility check. A ray cast from the anchor can
    only ever hit the surface actually visible/reachable from that anchor,
    which structurally forecloses that failure mode.

    `anchor` is the fixed reference point to cast from: `x0` (the shading
    point) for the first vertex in a chain, or the PREVIOUS vertex's
    (already reprojected) position for any later vertex -- mirroring how
    `_sms_probe_and_solve`'s own initial straight-line probe sequentially
    marches from one hit to the next. `anchor_on_surface` distinguishes the
    two cases: False (anchor is off-sphere, e.g. x0) takes the ray's FIRST
    crossing (t_min close to 0) -- the near/visible hemisphere from that
    anchor; True (anchor is itself already ON this same sphere, e.g. the
    previous vertex) skips a small epsilon past the anchor's own surface so
    the ray finds the FAR crossing -- the point reached by continuing
    through the sphere's interior, exactly the entry-to-exit relationship a
    solid glass sphere needs.

    Returns (pos, normal, dp_du, dp_dv, ok) -- `ok=False` when the proposal
    direction is degenerate, the ray misses the sphere entirely (should
    only happen for a wildly oversized step), or (when `n_spheres > 0`
    opts into it) the anchor-to-sphere ray is blocked by OTHER scene
    geometry first. That last check matters just as much as the ray-cast
    itself: `ray_sphere_hit` alone only knows about this one analytic
    sphere, so on its own it can still "successfully" reproject onto a
    sphere point that a real ray from the anchor could never actually
    reach because something else (a floor, in the scene that exposed this)
    sits in the way first -- exactly mirroring the reference
    implementation's own `vtx.shape != si_current.shape` rejection
    (manifold_ss.cpp), which re-intersects the FULL scene each iteration,
    not just the specular shape in isolation. The caller falls back to the
    closed-form snap when `ok=False`, matching the reference's own
    "missed/blocked, shrink the step" recovery."""
    var dir_raw = x_raw - anchor
    var dir_len = sqrt(dot(dir_raw, dir_raw))
    if dir_len <= Float32(1e-9):
        return (x_raw, Vec3f(Float32(0.0)), Vec3f(Float32(0.0)), Vec3f(Float32(0.0)), False)
    var dir = dir_raw * (Float32(1.0) / dir_len)
    var t_min = radius * Float32(1e-4) if anchor_on_surface else Float32(1e-5)
    var t_max = radius * Float32(8.0) + dir_len
    var ray = Ray_C(Point3f(anchor[0], anchor[1], anchor[2]), Vec3f(dir[0], dir[1], dir[2]))
    var t = ray_sphere_hit(Point3f(center[0], center[1], center[2]), radius, ray, t_min, t_max)
    if t <= Float32(0.0):
        return (x_raw, Vec3f(Float32(0.0)), Vec3f(Float32(0.0)), Vec3f(Float32(0.0)), False)
    if n_spheres > 0:
        # Offset the occlusion ray's ORIGIN forward along `dir` by a small
        # epsilon before re-tracing the full scene -- `anchor` can sit
        # essentially ON another surface (x0 is a shading point epsilon
        # above its own floor/mesh; a previous chain vertex sits exactly on
        # its sphere), so tracing from `anchor` itself would immediately
        # self-intersect that surface at (near-)zero t and report a false
        # "occluded" on every single call. Same shadow-acne fix
        # `_sms_probe_and_solve`'s own probes already use
        # (`hit_point + shadow_dir * 0.0002`), just applied here too.
        var occl_eps = Float32(0.001)
        var occl_org = anchor + dir * occl_eps
        var occl_tmax = (t - occl_eps) + radius * Float32(1e-3)
        if occl_tmax > Float32(0.0):
            var occl_ray = Ray_C(Point3f(occl_org[0], occl_org[1], occl_org[2]), Vec3f(dir[0], dir[1], dir[2]))
            var dummy_prim = PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
            var dummy_inter = Intersection_C(dummy_prim, occl_tmax, Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
            var store = InlineArray[Intersection_C, 1](fill=dummy_inter)
            traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, occl_ray, occl_tmax, store.unsafe_ptr(),
                                blasNodesArr, blasPrimIdsArr, instances, spheres, n_spheres)
            if store[0].hit != Int8(0) and store[0].tHit < (t - occl_eps) - radius * Float32(1e-4):
                return (x_raw, Vec3f(Float32(0.0)), Vec3f(Float32(0.0)), Vec3f(Float32(0.0)), False)
    var pos = anchor + dir * t
    var normal_raw = pos - center
    var normal_len = sqrt(dot(normal_raw, normal_raw))
    if normal_len <= Float32(1e-8):
        return (x_raw, Vec3f(Float32(0.0)), Vec3f(Float32(0.0)), Vec3f(Float32(0.0)), False)
    var normal = normal_raw * (Float32(1.0) / normal_len)
    var r = _sms_sphere_frame_at(pos, normal)
    return (r[0], r[1], r[2], r[3], True)

# ── Angle-difference constraint (manifold_ss.cpp) ────────────────────────────
# The OTHER of the reference's two constraint formulations, and the one its
# own scenes select by default (`caustics_halfvector_constraints = false`).
#
# The half-vector formulation below asks the generalized half-vector to line
# up with the shading normal. This one instead REFRACTS the direction to the
# shading point through the vertex and asks the result to match the direction
# to the light, measured as a difference of spherical angles (theta, phi).
# The two have the same roots, but not the same Newton behaviour: the
# half-vector constraint degenerates as the half-vector approaches the
# surface's tangent plane, and it happily converges to REFLECTIONS that then
# have to be rejected after the fact. The paper introduces the angle-
# difference form precisely because it is better conditioned on strongly
# curved and normal-mapped surfaces -- which is exactly the case a caustic
# from a normal-mapped sphere is made of.
#
# Used for a single curved vertex only (see sms_walk): the reference defines
# it for one specular vertex, and it does not generalize to the coupled
# block-tridiagonal system the N-vertex chain solves.

@always_inline
def _sm_refract(w: Vec3f, n_in: Vec3f, eta_in: Float32) -> Tuple[Bool, Vec3f]:
    """render/manifold.h's `refract`. Self-orienting: `eta_in` is the RAW
    material eta and the side is decided here from dot(w, n). Returns
    ok=False on total internal reflection."""
    var n = n_in
    var eta = Float32(1.0) / eta_in
    if dot(w, n) < Float32(0.0):
        eta = Float32(1.0) / eta
        n = -n
    var dot_w_n = dot(w, n)
    var root_term = Float32(1.0) - eta*eta * (Float32(1.0) - dot_w_n*dot_w_n)
    if root_term < Float32(0.0):
        return (False, Vec3f(Float32(0.0)))
    return (True, (w - n * dot_w_n) * (-eta) - n * sqrt(root_term))

@always_inline
def _sm_d_refract(
    w: Vec3f, dw_du: Vec3f, dw_dv: Vec3f,
    n_in: Vec3f, dn_du_in: Vec3f, dn_dv_in: Vec3f, eta_in: Float32,
) -> Tuple[Vec3f, Vec3f]:
    """render/manifold.h's `d_refract` -- the (u, v) derivatives of
    `_sm_refract`, with the same self-orienting convention."""
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
    var rt = Float32(1.0) - eta*eta * (Float32(1.0) - dot_w_n*dot_w_n)
    if rt < Float32(1e-12):
        return (Vec3f(Float32(0.0)), Vec3f(Float32(0.0)))
    var root = sqrt(rt)
    var inv_2root = Float32(1.0) / (Float32(2.0) * root)
    var a_u = (dw_du - (n * (dot_dwdu_n + dot_w_dndu) + dn_du * dot_w_n)) * (-eta)
    var b1_u = dn_du * root
    var b2_u = n * (inv_2root * (-eta*eta*(Float32(-2.0)*dot_w_n*(dot_dwdu_n + dot_w_dndu))))
    var a_v = (dw_dv - (n * (dot_dwdv_n + dot_w_dndv) + dn_dv * dot_w_n)) * (-eta)
    var b1_v = dn_dv * root
    var b2_v = n * (inv_2root * (-eta*eta*(Float32(-2.0)*dot_w_n*(dot_dwdv_n + dot_w_dndv))))
    return (a_u - (b1_u + b2_u), a_v - (b1_v + b2_v))

@always_inline
def _sm_sphcoords(w: Vec3f) -> SIMD[DType.float32, 2]:
    """(theta, phi) of a unit direction, phi wrapped to [0, 2pi)."""
    var z = min(max(w[2], Float32(-1.0)), Float32(1.0))
    var theta = acos(z)
    var phi = _atan2f(w[1], w[0])
    if phi < Float32(0.0):
        phi += TWO_PI
    return SIMD[DType.float32, 2](theta, phi)

@always_inline
def _sm_d_sphcoords(w: Vec3f, dw_du: Vec3f, dw_dv: Vec3f) -> SIMD[DType.float32, 4]:
    """(dtheta_du, dphi_du, dtheta_dv, dphi_dv)."""
    var s2 = Float32(1.0) - w[2]*w[2]
    var d_acos = Float32(0.0)
    if s2 > Float32(1e-12):
        d_acos = -Float32(1.0) / sqrt(s2)
    var dt_du = d_acos * dw_du[2]
    var dt_dv = d_acos * dw_dv[2]
    var dp_du = Float32(0.0)
    var dp_dv = Float32(0.0)
    if abs(w[0]) > Float32(1e-12):
        var yx = w[1] / w[0]
        var d_atan = Float32(1.0) / (Float32(1.0) + yx*yx)
        var inv_x2 = Float32(1.0) / (w[0] * w[0])
        dp_du = d_atan * (w[0]*dw_du[1] - w[1]*dw_du[0]) * inv_x2
        dp_dv = d_atan * (w[0]*dw_dv[1] - w[1]*dw_dv[0]) * inv_x2
    return SIMD[DType.float32, 4](dt_du, dp_du, dt_dv, dp_dv)

@always_inline
def _sms_step_anglediff(
    x0: Vec3f, xL: Vec3f, v: SMSVertex,
) -> Tuple[Bool, SIMD[DType.float32, 2], SIMD[DType.float32, 2]]:
    """manifold_ss.cpp's `compute_step_anglediff`, for one specular vertex.
    Returns (ok, C, dX): the constraint residual and the Newton step in the
    vertex's own (dp_du, dp_dv) parameterization, to be applied as
    `pos -= dp_du*dX[0] + dp_dv*dX[1]` exactly like the half-vector step.

    The offset-normal terms of the reference are omitted: they are no-ops
    for a smooth (zero-roughness) dielectric, which is the only kind of
    caster this reaches -- a rough one would need the microfacet normal
    sampled in the vertex's shading frame, and gonzales does not sample one
    for SMS at all."""
    var fail = (False, SIMD[DType.float32, 2](Float32(1e30), Float32(1e30)),
                SIMD[DType.float32, 2](Float32(0.0), Float32(0.0)))
    var wiv = x0 - v.pos
    var ili = sqrt(dot(wiv, wiv))
    if ili < Float32(1e-3):
        return fail
    ili = Float32(1.0) / ili
    var wi = wiv * ili
    var dwi_du = (v.dp_du - wi * dot(wi, v.dp_du)) * (-ili)
    var dwi_dv = (v.dp_dv - wi * dot(wi, v.dp_dv)) * (-ili)

    var wov = xL - v.pos
    var ilo = sqrt(dot(wov, wov))
    if ilo < Float32(1e-3):
        return fail
    ilo = Float32(1.0) / ilo
    var wo = wov * ilo
    var dwo_du = (v.dp_du - wo * dot(wo, v.dp_du)) * (-ilo)
    var dwo_dv = (v.dp_dv - wo * dot(wo, v.dp_dv)) * (-ilo)

    var n = v.normal
    var C = SIMD[DType.float32, 2](Float32(0.0), Float32(0.0))
    var dC = SIMD[DType.float32, 4](Float32(0.0))
    var ok = False

    # Refract the direction to the shading point and compare against the
    # direction to the light; if that side is in total internal reflection,
    # do it the other way round instead (both express the same constraint).
    var ri = _sm_refract(wi, n, v.eta_raw)
    if ri[0]:
        var d = _sm_d_refract(wi, dwi_du, dwi_dv, n, v.dn_du, v.dn_dv, v.eta_raw)
        var so = _sm_sphcoords(wo)
        var sio = _sm_sphcoords(ri[1])
        var dp = so[1] - sio[1]
        if dp < -PI: dp += TWO_PI
        elif dp > PI: dp -= TWO_PI
        C = SIMD[DType.float32, 2](so[0] - sio[0], dp)
        var a = _sm_d_sphcoords(wo, dwo_du, dwo_dv)
        var b = _sm_d_sphcoords(ri[1], d[0], d[1])
        dC = SIMD[DType.float32, 4](a[0]-b[0], a[2]-b[2], a[1]-b[1], a[3]-b[3])
        ok = True
    else:
        var ro = _sm_refract(wo, n, v.eta_raw)
        if ro[0]:
            var d = _sm_d_refract(wo, dwo_du, dwo_dv, n, v.dn_du, v.dn_dv, v.eta_raw)
            var si = _sm_sphcoords(wi)
            var soi = _sm_sphcoords(ro[1])
            var dp = si[1] - soi[1]
            if dp < -PI: dp += TWO_PI
            elif dp > PI: dp -= TWO_PI
            C = SIMD[DType.float32, 2](si[0] - soi[0], dp)
            var a = _sm_d_sphcoords(wi, dwi_du, dwi_dv)
            var b = _sm_d_sphcoords(ro[1], d[0], d[1])
            dC = SIMD[DType.float32, 4](a[0]-b[0], a[2]-b[2], a[1]-b[1], a[3]-b[3])
            ok = True
    if not ok:
        return fail
    var (Li, det) = mat22_inv(dC)
    if abs(det) < Float32(1e-6):
        return fail
    return (True, C, mat22_mul_v(Li, C))

@fieldwise_init
struct SMSVertexEval(TrivialRegisterPassable):
    """Per-vertex quantities recomputed every Newton iteration (and once
    more after convergence for the BSDF product / Jacobian chain).
    `ok`=0 means this vertex's geometry degenerated (grazing, coincident
    points, non-transmissive) -- caller must abort the walk."""
    var a:   SIMD[DType.float32, 4]
    var b:   SIMD[DType.float32, 4]
    var c:   SIMD[DType.float32, 4]
    var cv:  SIMD[DType.float32, 2]
    var wi:  Vec3f
    var wo:  Vec3f
    var H:   Vec3f
    var wol: Float32
    var ili: Float32
    var ilo: Float32
    var ok:  Int8

@always_inline
def _sms_eval_bad() -> SMSVertexEval:
    var z4 = SIMD[DType.float32, 4](Float32(0.0))
    var z3 = Vec3f(Float32(0.0))
    var z2 = SIMD[DType.float32, 2](Float32(0.0))
    return SMSVertexEval(z4, z4, z4, z2, z3, z3, z3, Float32(0.0), Float32(0.0), Float32(0.0), Int8(0))

@always_inline
def _sms_eval_vertex(
    x0: Vec3f, xL: Vec3f,
    verts: InlineArray[SMSVertex, MAX_SMS_VERTICES], n: Int, i: Int,
) -> SMSVertexEval:
    var prev_pos = x0 if i == 0 else verts[i-1].pos
    var next_pos = xL if i == n-1 else verts[i+1].pos
    var wiv = prev_pos - verts[i].pos; var wil = sqrt(dot(wiv,wiv))
    var wov = next_pos - verts[i].pos; var wol = sqrt(dot(wov,wov))
    if wil < Float32(1e-6) or wol < Float32(1e-6):
        return _sms_eval_bad()
    var wi = wiv*(Float32(1)/wil); var wo = wov*(Float32(1)/wol)
    if dot(verts[i].normal, wi) * dot(verts[i].normal, wo) >= Float32(0):
        return _sms_eval_bad()
    var Hv = -(wi + wo*verts[i].eta); var Hl = sqrt(dot(Hv,Hv))
    if Hl < Float32(1e-10):
        return _sms_eval_bad()
    var H = Hv*(Float32(1)/Hl)
    var s3 = verts[i].dp_du - verts[i].normal*dot(verts[i].dp_du, verts[i].normal)
    var sl2 = dot(s3,s3)
    if sl2 < Float32(1e-10):
        return _sms_eval_bad()
    var s = s3*(Float32(1)/sqrt(sl2)); var t = cross(verts[i].normal, s)
    var ili = Float32(1)/(Hl*wil); var ilo = verts[i].eta/(Hl*wol)
    var has_prev = i > 0
    var has_next = i < n-1
    var z3 = Vec3f(Float32(0.0))
    var dpu_prev = verts[i-1].dp_du if has_prev else z3
    var dpv_prev = verts[i-1].dp_dv if has_prev else z3
    var dpu_next = verts[i+1].dp_du if has_next else z3
    var dpv_next = verts[i+1].dp_dv if has_next else z3
    var (ai, bi, ci, cvi) = sms_vertex_mats(wi, wo, H, s, t, verts[i].dp_du, verts[i].dp_dv, ili, ilo,
        dpu_prev, dpv_prev, dpu_next, dpv_next, has_prev, has_next)
    if verts[i].is_sphere != Int8(0):
        # Curvature correction to the SELF ("b") Jacobian, missing from
        # sms_vertex_mats because that formula was derived assuming a FLAT
        # vertex, where the local frame (normal, s, t) never changes as the
        # vertex moves. For a sphere it does: moving distance du along s
        # rotates the true local normal by du/radius TOWARD s (curvature),
        # which drags s itself by -normal/radius and t by 0 (and
        # symmetrically for v/t) -- a real first-order effect a flat
        # vertex's fixed frame never has. Differentiating cv_s=dot(H,s)
        # therefore needs a `dot(H, ds/du)` term beyond sms_vertex_mats's
        # `dot(dH/du, s)`; working through ds/du=-normal/radius, dt/du=0,
        # ds/dv=0, dt/dv=-normal/radius (derived by hand, verified against
        # a full finite-difference reprojected-perturbation probe -- see
        # project_sms_restir_phase6 memory) gives a simple diagonal
        # correction: subtract dot(H, normal)/radius from BOTH diagonal
        # entries of b. Without this, the "b" Jacobian is measurably wrong
        # (by up to ~1/radius, comparable to its own diagonal magnitude for
        # a modestly-curved sphere) and the Newton direction it produces is
        # not even a local descent direction -- confirmed empirically: a
        # coupled 2-curved-vertex chain diverged even under backtracking
        # line search until this term was added.
        # Generalized from the sphere-only `dot(H,n)/radius` diagonal term
        # to an arbitrary varying normal, so a normal-mapped vertex can
        # contribute its texture gradient through dn_du/dn_dv. Since s and
        # t stay unit and perpendicular to n, differentiating them gives
        # ds/du = -n*dot(dn_du, s), dt/du = -n*dot(dn_du, t) (and likewise
        # for v), so the `dot(H, ds/du)` term this adds to each entry of b
        # is -dot(H,n) * dot(dn_d{u,v}, {s,t}). For a smooth sphere
        # dn_du = dp_du/radius (~s/radius) and dn_dv = dp_dv/radius (~t/
        # radius), which collapses to exactly the old diagonal-only
        # -dot(H,n)/radius; a normal map breaks that alignment and lights
        # up the off-diagonals too.
        var hn = dot(H, verts[i].normal)
        var dnu = verts[i].dn_du
        var dnv = verts[i].dn_dv
        bi = SIMD[DType.float32, 4](
            bi[0] - hn * dot(dnu, s),
            bi[1] - hn * dot(dnv, s),
            bi[2] - hn * dot(dnu, t),
            bi[3] - hn * dot(dnv, t))
    return SMSVertexEval(ai, bi, ci, cvi, wi, wo, H, wol, ili, ilo, Int8(1))

def sms_walk(
    x0: Vec3f, xL: Vec3f,
    verts_init: InlineArray[SMSVertex, MAX_SMS_VERTICES], n: Int,
    ldp_du: Vec3f, ldp_dv: Vec3f,
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin] = UnsafePointer[BVH2Node, MutExternalOrigin].unsafe_dangling(),
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin] = UnsafePointer[PrimId_C, MutExternalOrigin].unsafe_dangling(),
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin] = UnsafePointer[TriangleMesh_C, MutExternalOrigin].unsafe_dangling(),
    curves: UnsafePointer[Curve_C, MutExternalOrigin] = UnsafePointer[Curve_C, MutExternalOrigin].unsafe_dangling(),
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin] = UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin] = UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
    instances: UnsafePointer[Instance_C, MutExternalOrigin] = UnsafePointer[Instance_C, MutExternalOrigin].unsafe_dangling(),
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin] = UnsafePointer[Sphere_C, MutExternalOrigin].unsafe_dangling(),
    n_spheres: Int = 0,
) -> Tuple[Bool, InlineArray[Vec3f, MAX_SMS_VERTICES], Float32, Float32]:
    """N-vertex generalization of _mnee_walk/_mnee_walk2 (kept in
    shading.mojo as fast paths for n=1/2 -- Phase 5.4). Newton iteration on
    a chain of `n` specular vertices via a block-tridiagonal solve of the
    per-vertex tangential half-vector constraint. `verts_init` supplies the
    Newton seed (a local copy is mutated, the caller's array is untouched)
    plus each vertex's fixed normal/dp_du/dp_dv/eta.

    Returns (converged, solved positions, bsdf_product, dx1_dxlight):
    `bsdf_product` is the product of each vertex's rough-dielectric
    transmission term (Walter et al.'s microfacet BTDF specialized to a
    perfect mirror lobe, same formula _mnee_walk/_mnee_walk2 already use);
    `dx1_dxlight` is the Jacobian |d(vertex_0 position)/d(light uv)| --
    exactly _mnee_walk's/_mnee_walk2's own quantity of the same name."""
    var verts = verts_init.copy()
    # A randomized seed (sms_seed_randomize moves .pos over the surface)
    # can land slightly off a curved vertex's true surface -- snap it back
    # on before the first iteration reads normal/dp_du/dp_dv from it.
    var has_curved = False
    for i0 in range(n):
        if verts[i0].is_sphere != Int8(0):
            has_curved = True
            _sms_apply_sphere_frame(verts[i0],
                _sms_reproject_onto_sphere(verts[i0].pos, verts[i0].sphere_center, verts[i0].sphere_radius))
    # The reference solves a single specular vertex with the angle-difference
    # constraint by default and only offers the half-vector one as an option;
    # a longer chain has no angle-difference form at all. Reflective (eta==1)
    # vertices are excluded because _sms_step_anglediff only implements the
    # refractive branch -- SMS in gonzales is reached from dielectrics only.
    var use_anglediff = (n == 1 and verts[0].is_sphere != Int8(0)
                         and verts[0].eta_raw != Float32(1.0))
    var converged = False
    for _iter in range(20):
        # Constraint formulation. A single CURVED vertex uses the reference's
        # angle-difference form (better conditioned exactly where a caustic
        # caster lives -- see _sms_step_anglediff); everything else keeps the
        # generalized half-vector constraint, which is what the coupled
        # N-vertex block-tridiagonal system is built on and the only form
        # that generalizes to it.
        var dx = InlineArray[SIMD[DType.float32, 2], MAX_SMS_VERTICES](fill=SIMD[DType.float32, 2](Float32(0.0)))
        var wol = InlineArray[Float32, MAX_SMS_VERTICES](fill=Float32(0.0))
        var err = Float32(0.0)
        if use_anglediff:
            var st = _sms_step_anglediff(x0, xL, verts[0])
            if not st[0]:
                break
            err = sqrt(st[1][0]*st[1][0] + st[1][1]*st[1][1])
            if err < SMS_SOLVER_THRESHOLD:
                converged = True; break
            dx[0] = st[2]
            var wv = xL - verts[0].pos
            wol[0] = sqrt(dot(wv, wv))
        else:
            var ev = InlineArray[SMSVertexEval, MAX_SMS_VERTICES](fill=_sms_eval_bad())
            var bad = False
            for i in range(n):
                ev[i] = _sms_eval_vertex(x0, xL, verts, n, i)
                if ev[i].ok == Int8(0):
                    bad = True; break
            if bad:
                break
            for i in range(n):
                wol[i] = ev[i].wol
            # L2 norm of each vertex's tangential-constraint residual, not
            # max-norm: max-norm is basis-dependent, and a curved vertex's
            # (s,t) basis is a FRESH, arbitrarily-ROTATED Frisvad frame every
            # iteration (see _sms_reproject_onto_sphere) -- the same physical
            # residual can land disproportionately on one axis before a
            # reprojection and the other axis after, making max-norm compare
            # apples to oranges across iterations for a curved chain (a flat
            # vertex's frame never rotates, so this never mattered before).
            # L2 norm is the tangential-H-vector's actual magnitude and is
            # invariant to which orthonormal basis happens to span the plane.
            err = Float32(0.0)
            for i in range(n):
                err = max(err, sqrt(ev[i].cv[0]*ev[i].cv[0] + ev[i].cv[1]*ev[i].cv[1]))
            if err < SMS_SOLVER_THRESHOLD:
                converged = True; break
            # ── Block tridiagonal forward sweep ────────────────────────────────
            var cprime = InlineArray[SIMD[DType.float32, 4], MAX_SMS_VERTICES](fill=SIMD[DType.float32, 4](Float32(0.0)))
            var dprime = InlineArray[SIMD[DType.float32, 2], MAX_SMS_VERTICES](fill=SIMD[DType.float32, 2](Float32(0.0)))
            var (Li0, det0) = mat22_inv(ev[0].b)
            if det0 == Float32(0.0):
                break
            cprime[0] = mat22_mul(Li0, ev[0].c)
            dprime[0] = mat22_mul_v(Li0, ev[0].cv)
            var solve_failed = False
            for i in range(1, n):
                var pivot = ev[i].b - mat22_mul(ev[i].a, cprime[i-1])
                var (Lii, deti) = mat22_inv(pivot)
                if deti == Float32(0.0):
                    solve_failed = True; break
                cprime[i] = mat22_mul(Lii, ev[i].c)
                dprime[i] = mat22_mul_v(Lii, ev[i].cv - mat22_mul_v(ev[i].a, dprime[i-1]))
            if solve_failed:
                break
            # ── Back substitution ───────────────────────────────────────────────
            dx = InlineArray[SIMD[DType.float32, 2], MAX_SMS_VERTICES](fill=SIMD[DType.float32, 2](Float32(0.0)))
            dx[n-1] = dprime[n-1]
            for ridx in range(n-1):
                var i = n-2-ridx
                dx[i] = dprime[i] - mat22_mul_v(cprime[i], dx[i+1])
        # ── Step clamp + position update ────────────────────────────────────
        if not has_curved:
            # Flat-only chain: unchanged from before sphere support existed --
            # one full (clamped) Newton step, no line search needed (a flat
            # vertex's tangent plane IS its surface, so the linearization
            # used to compute `dx` is exact everywhere, not just locally).
            for i in range(n):
                var step_len = sqrt(dx[i][0]*dx[i][0] + dx[i][1]*dx[i][1])
                var max_step = wol[i] * Float32(0.5)
                var dxi = dx[i]
                if step_len > max_step and step_len > Float32(1e-12):
                    dxi = dxi * (max_step/step_len)
                verts[i].pos = verts[i].pos - (verts[i].dp_du*dxi[0] + verts[i].dp_dv*dxi[1])
        else:
            # A curved vertex's tangent plane only agrees with the true
            # surface AT the current point -- moving even the exact Newton
            # direction too far still overshoots once reprojected onto real
            # curvature, and with TWO simultaneously-curved coupled vertices
            # (e.g. the entry+exit points of one solid glass sphere) this
            # overshoot can compound and diverge outright rather than merely
            # converge slowly (confirmed empirically: a full step from a
            # seed just 1% of the sphere's radius from the exact answer grew
            # the residual every iteration). The Newton DIRECTION `dx` above
            # is still correct -- it comes from the same Jacobian formula
            # already proven exact for a flat chain (test_sms_walk_n2_
            # matches_mnee_walk2) -- what's missing for a curved chain is a
            # STEP-LENGTH safeguard. Backtracking line search (Numerical
            # Recipes §9.7's standard remedy for Newton overshoot): try the
            # full clamped step; if it doesn't actually reduce the residual,
            # halve it and retry, holding the search direction fixed.
            var scale = Float32(1.0)
            var applied = False
            for _bt in range(8):
                var trial = verts.copy()
                var reproj_failed = False
                for i in range(n):
                    var step_len = sqrt(dx[i][0]*dx[i][0] + dx[i][1]*dx[i][1]) * scale
                    var max_step = wol[i] * Float32(0.5)
                    if trial[i].is_sphere != Int8(0):
                        max_step = min(max_step, trial[i].sphere_radius * Float32(0.3))
                    var dxi = dx[i] * scale
                    if step_len > max_step and step_len > Float32(1e-12):
                        dxi = dxi * (max_step/step_len)
                    trial[i].pos = trial[i].pos - (trial[i].dp_du*dxi[0] + trial[i].dp_dv*dxi[1])
                    if trial[i].is_sphere != Int8(0):
                        # Anchor: x0 for the first vertex, else the PREVIOUS
                        # vertex's own (already reprojected, this same trial)
                        # position -- see _sms_reproject_onto_sphere_anchored's
                        # docstring for why this is the real fix for the
                        # wrong-hemisphere convergence bug (ray-cast from a
                        # fixed anchor can't land on a hemisphere invisible
                        # from that anchor, unlike the closed-form snap).
                        var anchor = x0
                        var anchor_on_surface = False
                        if i > 0 and trial[i-1].is_sphere != Int8(0):
                            var dc = trial[i-1].sphere_center - trial[i].sphere_center
                            if dot(dc, dc) < Float32(1e-6) and abs(trial[i-1].sphere_radius - trial[i].sphere_radius) < Float32(1e-4):
                                anchor = trial[i-1].pos
                                anchor_on_surface = True
                        elif i > 0:
                            anchor = trial[i-1].pos
                        var ra = _sms_reproject_onto_sphere_anchored(anchor, trial[i].pos, trial[i].sphere_center, trial[i].sphere_radius, anchor_on_surface,
                            bvh2Nodes, primIds, meshes, curves, blasNodesArr, blasPrimIdsArr, instances, spheres, n_spheres)
                        if ra[4]:
                            _sms_apply_sphere_frame(trial[i], (ra[0], ra[1], ra[2], ra[3]))
                        else:
                            # Anchor ray missed the sphere or was blocked by
                            # other scene geometry first -- this step is
                            # invalid (NOT "fall back to the unconstrained
                            # closed-form snap", which is exactly how the
                            # wrong-hemisphere bug happened in the first
                            # place). Mirrors the reference implementation's
                            # own recovery: treat like a failed step and
                            # shrink `scale` on the next backtracking attempt.
                            reproj_failed = True
                # The line search has to score the trial with the SAME
                # constraint the step came from -- comparing an
                # angle-difference step against a half-vector residual would
                # backtrack on a quantity the step was never reducing.
                var bad2 = reproj_failed
                var err2 = Float32(0.0)
                if use_anglediff:
                    if not bad2:
                        var st2 = _sms_step_anglediff(x0, xL, trial[0])
                        if st2[0]:
                            err2 = sqrt(st2[1][0]*st2[1][0] + st2[1][1]*st2[1][1])
                        else:
                            bad2 = True
                else:
                    var ev2 = InlineArray[SMSVertexEval, MAX_SMS_VERTICES](fill=_sms_eval_bad())
                    for i in range(n):
                        if bad2:
                            break
                        ev2[i] = _sms_eval_vertex(x0, xL, trial, n, i)
                        if ev2[i].ok == Int8(0):
                            bad2 = True; break
                    if not bad2:
                        for i in range(n):
                            err2 = max(err2, sqrt(ev2[i].cv[0]*ev2[i].cv[0] + ev2[i].cv[1]*ev2[i].cv[1]))
                if not bad2:
                    if err2 < err or _bt == 7:
                        verts = trial.copy()
                        applied = True
                        break
                scale *= Float32(0.5)
            if not applied:
                break
    var zero_positions = InlineArray[Vec3f, MAX_SMS_VERTICES](fill=Vec3f(Float32(0.0)))
    if not converged:
        return (False, zero_positions.copy(), Float32(0.0), Float32(0.0))
    # Post-solve validity check, straight from the reference
    # (manifold_ss.cpp's newton_solver: "the half-vector formulation of
    # manifold walks will often converge to invalid solutions that are
    # actually reflections -- here we need to reject those"). A refractive
    # vertex must have its two neighbours on OPPOSITE sides of the surface.
    #
    # It has to be the GEOMETRIC normal: on a normal-mapped vertex the
    # shading normal -- the one _sms_eval_vertex's own sign test uses,
    # correctly, since the constraint is stated in it -- can be tilted far
    # enough that a path physically entering and leaving on the same side
    # still passes. Cheap, and it is the check that keeps a randomly-seeded
    # walk from reporting the sphere's far side as a solution.
    for i in range(n):
        if verts[i].eta == Float32(1.0):
            continue
        var gn = verts[i].normal
        if verts[i].is_sphere != Int8(0):
            var gnv = verts[i].pos - verts[i].sphere_center
            var gnl = sqrt(dot(gnv, gnv))
            if gnl <= Float32(1e-8):
                return (False, zero_positions.copy(), Float32(0.0), Float32(0.0))
            gn = gnv * (Float32(1.0) / gnl)
        var wx = (x0 if i == 0 else verts[i-1].pos) - verts[i].pos
        var wy = (xL if i == n-1 else verts[i+1].pos) - verts[i].pos
        if dot(gn, wx) * dot(gn, wy) >= Float32(0.0):
            return (False, zero_positions.copy(), Float32(0.0), Float32(0.0))
    # ── Final recompute: BSDF product + light-Jacobian chain ────────────────
    var evf = InlineArray[SMSVertexEval, MAX_SMS_VERTICES](fill=_sms_eval_bad())
    for i in range(n):
        evf[i] = _sms_eval_vertex(x0, xL, verts, n, i)
        if evf[i].ok == Int8(0):
            return (False, zero_positions.copy(), Float32(0.0), Float32(0.0))
    var cprimef = InlineArray[SIMD[DType.float32, 4], MAX_SMS_VERTICES](fill=SIMD[DType.float32, 4](Float32(0.0)))
    var (Li0f, det0f) = mat22_inv(evf[0].b)
    if det0f == Float32(0.0):
        return (False, zero_positions.copy(), Float32(0.0), Float32(0.0))
    cprimef[0] = mat22_mul(Li0f, evf[0].c)
    var last_Li = Li0f  # overwritten below when n > 1; stays Li0f when n == 1
    for i in range(1, n):
        var pivot = evf[i].b - mat22_mul(evf[i].a, cprimef[i-1])
        var (Lii, deti) = mat22_inv(pivot)
        if deti == Float32(0.0):
            return (False, zero_positions.copy(), Float32(0.0), Float32(0.0))
        cprimef[i] = mat22_mul(Lii, evf[i].c)
        last_Li = Lii
    # dc_dlight: coupling of the LAST vertex's constraint to the light's own
    # tangent basis (exactly _mnee_walk's dHdu_l/dHdv_l or _mnee_walk2's
    # dc_du/dc_dv at x2, generalized to vertex n-1).
    var last = n - 1
    var s3_last = verts[last].dp_du - verts[last].normal*dot(verts[last].dp_du, verts[last].normal)
    var s_last = s3_last * (Float32(1)/sqrt(dot(s3_last,s3_last)))
    var t_last = cross(verts[last].normal, s_last)
    var dc_du = (ldp_du - evf[last].wo*dot(evf[last].wo,ldp_du)) * evf[last].ilo
    var dc_dv = (ldp_dv - evf[last].wo*dot(evf[last].wo,ldp_dv)) * evf[last].ilo
    dc_du -= evf[last].H*dot(dc_du, evf[last].H); dc_du = -dc_du
    dc_dv -= evf[last].H*dot(dc_dv, evf[last].H); dc_dv = -dc_dv
    var dc_dlight = SIMD[DType.float32, 4](dot(dc_du,s_last), dot(dc_dv,s_last), dot(dc_du,t_last), dot(dc_dv,t_last))
    var Tp = mat22_mul(last_Li, dc_dlight) * Float32(-1.0)
    for ridx in range(n-1):
        var i = n-2-ridx
        Tp = mat22_mul(cprimef[i], Tp) * Float32(-1.0)
    var dx1_dxlight = abs(Tp[0]*Tp[3] - Tp[1]*Tp[2])
    # BSDF product across the whole chain.
    var bsdf_product = Float32(1.0)
    for i in range(n):
        var cosNI = abs(dot(verts[i].normal, evf[i].wi))
        var cosHI = abs(dot(evf[i].H, evf[i].wi))
        var cosTM = abs(dot(verts[i].normal, evf[i].H))
        var F = fr_dielectric(cosNI, verts[i].eta)
        # Solid-angle compression across a delta refraction (reference:
        # manifold_ss.cpp `specular_reflectance`, delta branch --
        # `bsdf_val = 1 - F; bsdf_val *= sqr(eta)`).
        #
        # The reference stores the RAW material eta on its manifold vertex
        # and re-orients it separately at each of the two places it is
        # used, and those two places look at the interface from OPPOSITE
        # sides: the generalized half-vector constraint tests
        # `dot(wi, gn)` (the SHADING-POINT side, wi pointing back toward
        # x0), while this compression factor tests `dot(n, wo)` (the
        # EMITTER side). `verts[i].eta` here carries the constraint
        # orientation -- that is how _sms_vertex_from_hit derives it, and
        # it is what _sms_eval_vertex above consumes -- so this use needs
        # the other one. _sms_eval_vertex has already rejected every
        # vertex whose wi and wo do NOT straddle the surface, so for any
        # vertex that reaches here the emitter-side orientation is exactly
        # the reciprocal.
        #
        # Squaring the constraint eta directly (what this line did before)
        # is therefore off by eta^4 -- ~5x for glass. It cancels silently
        # in the common 2-vertex enter+exit chain, where eta_2 = 1/eta_1
        # makes the product 1 either way, which is why it survived: only a
        # SINGLE-refraction chain (an analytic sphere caster, modeled as
        # one idealized bend -- see _sms_probe_and_solve) exposes it.
        var eta_o = Float32(1.0) / verts[i].eta
        bsdf_product *= (Float32(1.0)-F)*cosHI/max(cosNI*cosTM*cosTM, Float32(1e-6)) * eta_o*eta_o
    var out_positions = InlineArray[Vec3f, MAX_SMS_VERTICES](fill=Vec3f(Float32(0.0)))
    for i in range(n):
        out_positions[i] = verts[i].pos
    return (True, out_positions.copy(), bsdf_product, dx1_dxlight)

# ── Random seeding + Bernoulli-trial reciprocal estimator (5.2/5.3) ─────────

@always_inline
def sms_refresh_solved_frames(mut verts: InlineArray[SMSVertex, MAX_SMS_VERTICES], n: Int):
    """Re-derive each curved vertex's frame at its SOLVED position.

    `sms_walk` returns only the solved POSITIONS -- its own updated vertex
    frames are local to it -- so a caller that writes those positions back
    into its seed array is left holding the SEED's normal, tangents and
    normal derivatives. Anything the caller then computes from
    `verts[i].normal` is evaluated at the wrong point on the surface.

    That was invisible while the seed was MNEE's deterministic probe hit,
    because the solution lands essentially on top of it. It stops being
    invisible the moment seeds are drawn at random across the caster (see
    sms_seed_randomize): the solved point is then unrelated to the seed, and
    `_mnee_area_light_contribute`'s geometric term -- which needs
    |dot(wi, n)| at the solved vertex, the reference's
    `dw0_dx1 = abs_dot(d, v1.gn) / r^2` -- was reading a normal from a
    random other place on the sphere. Wrong on average, and occasionally
    wrong by a lot, which is exactly the sample the Bernoulli estimator then
    multiplies by a large trial count."""
    for i in range(n):
        if verts[i].is_sphere != Int8(0):
            _sms_apply_sphere_frame(verts[i], _sms_reproject_onto_sphere(
                verts[i].pos, verts[i].sphere_center, verts[i].sphere_radius))

@always_inline
def sms_seed_randomize(x0: Vec3f, mut verts: InlineArray[SMSVertex, MAX_SMS_VERTICES], n: Int, mut pcg: PCG32, jitter_scale: Float32):
    """5.2: draw a fresh random Newton seed for each vertex of the chain.

    A FLAT vertex is perturbed within its own triangle's tangent plane by a
    uniform random offset of at most `jitter_scale`, which the caller
    derives from inter-vertex probe spacing -- the straight-line probe
    already established that the solution is near that triangle, and the
    flat-tangent-plane assumption is only valid nearby anyway.

    A CURVED (sphere) vertex is instead seeded UNIFORMLY OVER THE WHOLE
    SPHERE, which is what the reference renderer does (its `sample_path`
    draws the seed with `square_to_uniform_sphere`). Local jitter is the
    wrong proposal there for two separate reasons. The estimator's own
    correctness is one: `sms_solve_bernoulli` weights its result by the
    reciprocal probability of REDISCOVERING the solution it found, which is
    only an unbiased estimate of the sum over ALL solutions if every
    solution is reachable from the seed distribution. The other is that on a
    normal-mapped sphere there genuinely are many solutions, spread right
    across the surface -- jittering around one probe hit would find the same
    root every time and silently drop the rest of the caustic.

    The uniform sphere point is not used as the seed directly: as in the
    reference, a ray is cast from the shading point `x0` TOWARDS it and the
    first surface hit becomes the seed. That is what makes every seed a
    point actually reachable from x0 along a straight line, instead of
    (half the time) a point on the far side that the Newton walk has to
    escape from first. Both are valid proposal distributions -- the
    estimator's reciprocal weighting makes it unbiased either way -- but
    this one wastes far fewer solves, which for a fixed trial budget is the
    difference between resolving a root and never seeing it."""
    for i in range(n):
        if verts[i].is_sphere != Int8(0):
            var z = Float32(1.0) - Float32(2.0)*pcg.next_float()
            var r = sqrt(max(Float32(0.0), Float32(1.0) - z*z))
            var phi = Float32(2.0) * PI * pcg.next_float()
            var dir = Vec3f(r*cos(phi), r*sin(phi), z)
            var target = verts[i].sphere_center + dir * verts[i].sphere_radius
            verts[i].pos = target
            var toward = target - x0
            var tl2 = dot(toward, toward)
            if tl2 > Float32(1e-12):
                var d = toward * (Float32(1.0) / sqrt(tl2))
                var ctr = verts[i].sphere_center
                var t_hit = ray_sphere_hit(
                    Point3f(ctr[0], ctr[1], ctr[2]), verts[i].sphere_radius,
                    Ray_C(Point3f(x0[0], x0[1], x0[2]), Vec3f(d[0], d[1], d[2])),
                    Float32(1e-4), Float32(1e30))
                if t_hit > Float32(0.0):
                    verts[i].pos = x0 + d * t_hit
        else:
            var ju = (pcg.next_float()*Float32(2.0) - Float32(1.0)) * jitter_scale
            var jv = (pcg.next_float()*Float32(2.0) - Float32(1.0)) * jitter_scale
            verts[i].pos = verts[i].pos + verts[i].dp_du*ju + verts[i].dp_dv*jv

@always_inline
def sms_same_solution(
    x0: Vec3f,
    a: InlineArray[Vec3f, MAX_SMS_VERTICES],
    b: InlineArray[Vec3f, MAX_SMS_VERTICES], n: Int,
) -> Bool:
    """Do two solved chains represent the SAME specular root? Compares the
    directions x0->x_i (see SMS_UNIQUENESS_COS_EPS for why direction rather
    than position), mirroring the reference implementation's own test."""
    for i in range(n):
        var da = a[i] - x0
        var db = b[i] - x0
        var la2 = dot(da, da)
        var lb2 = dot(db, db)
        if la2 <= Float32(1e-16) or lb2 <= Float32(1e-16):
            return False
        var c = dot(da, db) / sqrt(la2 * lb2)
        if abs(c - Float32(1.0)) > SMS_UNIQUENESS_COS_EPS:
            return False
    return True

def sms_solve_bernoulli(
    x0: Vec3f, xL: Vec3f,
    verts_seed: InlineArray[SMSVertex, MAX_SMS_VERTICES], n: Int,
    ldp_du: Vec3f, ldp_dv: Vec3f,
    jitter_scale: Float32, mut pcg: PCG32,
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin] = UnsafePointer[BVH2Node, MutExternalOrigin].unsafe_dangling(),
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin] = UnsafePointer[PrimId_C, MutExternalOrigin].unsafe_dangling(),
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin] = UnsafePointer[TriangleMesh_C, MutExternalOrigin].unsafe_dangling(),
    curves: UnsafePointer[Curve_C, MutExternalOrigin] = UnsafePointer[Curve_C, MutExternalOrigin].unsafe_dangling(),
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin] = UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin] = UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
    instances: UnsafePointer[Instance_C, MutExternalOrigin] = UnsafePointer[Instance_C, MutExternalOrigin].unsafe_dangling(),
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin] = UnsafePointer[Sphere_C, MutExternalOrigin].unsafe_dangling(),
    n_spheres: Int = 0,
) -> Tuple[Bool, InlineArray[Vec3f, MAX_SMS_VERTICES], Float32, Float32, Float32]:
    """5.3: Zeltner et al. 2020's Bernoulli-trial reciprocal estimator.
    Solves once from a randomly-jittered seed to fix the primary candidate
    solution X* (the one whose contribution this call reports), then draws
    FRESH independent random seeds and re-solves, counting trials until one
    converges back to X*. That count T is a sample from Geometric(q) where
    q = P(a random seed converges to X* specifically), so E[T] = 1/q --
    T itself is the unbiased estimator (no separate division needed).

    Returns (converged, positions, bsdf_product, dx1_dxlight, trial_count).
    Multiply trial_count into the final contribution alongside the other
    reciprocal-density factors (inv_pdf_area, lobe_w).

    Bounded at SMS_BERNOULLI_MAX_TRIALS: past that, trial_count is reported
    as that bound rather than run unboundedly -- a small, documented
    tail bias in the (rare) case where X* has very low probability of
    being rediscovered, traded for guaranteed bounded cost. For the common
    case of an essentially-unique manifold solution, q is close to 1 and
    the very first trial matches, so this degenerates to trial_count~1 --
    i.e. plain single-solve behavior -- with negligible overhead."""
    var seed0 = verts_seed.copy()
    sms_seed_randomize(x0, seed0, n, pcg, jitter_scale)
    var _walk0 = sms_walk(x0, xL, seed0, n, ldp_du, ldp_dv,
        bvh2Nodes, primIds, meshes, curves, blasNodesArr, blasPrimIdsArr, instances, spheres, n_spheres)
    var ok0 = _walk0[0]
    var pos0 = _walk0[1].copy()
    var bsdf0 = _walk0[2]
    var jac0 = _walk0[3]
    if not ok0:
        var zero_positions = InlineArray[Vec3f, MAX_SMS_VERTICES](fill=Vec3f(Float32(0.0)))
        return (False, zero_positions.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
    var trials = Float32(0.0)
    var matched = False
    for _t in range(SMS_BERNOULLI_MAX_TRIALS):
        trials += Float32(1.0)
        var seed_k = verts_seed.copy()
        sms_seed_randomize(x0, seed_k, n, pcg, jitter_scale)
        var _walkk = sms_walk(x0, xL, seed_k, n, ldp_du, ldp_dv,
            bvh2Nodes, primIds, meshes, curves, blasNodesArr, blasPrimIdsArr, instances, spheres, n_spheres)
        var okk = _walkk[0]
        var posk = _walkk[1].copy()
        if okk and sms_same_solution(x0, posk, pos0, n):
            matched = True; break
    _ = matched
    return (True, pos0.copy(), bsdf0, jac0, trials)
