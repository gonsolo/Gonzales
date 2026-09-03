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

from std.math import sqrt, abs, max, min
from .geometry import RGB, dot, cross, fr_dielectric, Frame, Vec3f, Point3f, Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Curve_C, Instance_C, Sphere_C
from .rng import PCG32
from .bvh import ray_sphere_hit, traverse_bvh2_core, BVH2Node

comptime MAX_SMS_VERTICES: Int = 6
comptime SMS_BERNOULLI_MAX_TRIALS: Int = 16
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

@always_inline
def sms_vertex_init() -> SMSVertex:
    var z = Vec3f(Float32(0.0))
    return SMSVertex(z, z, z, z, Float32(1.0), Int8(0), z, Float32(0.0))

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
    return SMSVertex(pos, normal, dp_du, dp_dv, eta, Int8(0), z, Float32(0.0))

@always_inline
def sms_vertex_sphere(
    pos: Vec3f,
    center: Vec3f,
    radius: Float32,
    eta: Float32,
) -> SMSVertex:
    """Constructs a sphere-caster SMSVertex from a probe hit's world-space
    position (assumed already ON the sphere, e.g. from a real intersection)
    -- snaps it exactly onto the sphere and derives an initial tangent
    frame via `_sms_reproject_onto_sphere`, matching what `sms_walk` itself
    will re-derive on every subsequent iteration."""
    var r = _sms_reproject_onto_sphere(pos, center, radius)
    return SMSVertex(r[0], r[1], r[2], r[3], eta, Int8(1), center, radius)

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
        var curv = dot(H, verts[i].normal) / verts[i].sphere_radius
        bi = SIMD[DType.float32, 4](bi[0] - curv, bi[1], bi[2], bi[3] - curv)
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
    # A jittered seed (sms_seed_jitter moves .pos within the tangent plane)
    # can land slightly off a curved vertex's true surface -- snap it back
    # on before the first iteration reads normal/dp_du/dp_dv from it.
    var has_curved = False
    for i0 in range(n):
        if verts[i0].is_sphere != Int8(0):
            has_curved = True
            var r0 = _sms_reproject_onto_sphere(verts[i0].pos, verts[i0].sphere_center, verts[i0].sphere_radius)
            verts[i0].pos = r0[0]; verts[i0].normal = r0[1]; verts[i0].dp_du = r0[2]; verts[i0].dp_dv = r0[3]
    var converged = False
    for _iter in range(20):
        var ev = InlineArray[SMSVertexEval, MAX_SMS_VERTICES](fill=_sms_eval_bad())
        var bad = False
        for i in range(n):
            ev[i] = _sms_eval_vertex(x0, xL, verts, n, i)
            if ev[i].ok == Int8(0):
                bad = True; break
        if bad:
            break
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
        var err = Float32(0.0)
        for i in range(n):
            err = max(err, sqrt(ev[i].cv[0]*ev[i].cv[0] + ev[i].cv[1]*ev[i].cv[1]))
        if err < Float32(1e-3):
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
        var dx = InlineArray[SIMD[DType.float32, 2], MAX_SMS_VERTICES](fill=SIMD[DType.float32, 2](Float32(0.0)))
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
                var max_step = ev[i].wol * Float32(0.5)
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
                    var max_step = ev[i].wol * Float32(0.5)
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
                            trial[i].pos = ra[0]; trial[i].normal = ra[1]; trial[i].dp_du = ra[2]; trial[i].dp_dv = ra[3]
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
                var ev2 = InlineArray[SMSVertexEval, MAX_SMS_VERTICES](fill=_sms_eval_bad())
                var bad2 = reproj_failed
                for i in range(n):
                    if bad2:
                        break
                    ev2[i] = _sms_eval_vertex(x0, xL, trial, n, i)
                    if ev2[i].ok == Int8(0):
                        bad2 = True; break
                if not bad2:
                    var err2 = Float32(0.0)
                    for i in range(n):
                        err2 = max(err2, sqrt(ev2[i].cv[0]*ev2[i].cv[0] + ev2[i].cv[1]*ev2[i].cv[1]))
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
        # eta*eta: see shading.mojo's _mnee_walk2 for the full derivation/
        # reference citation (same formula family, same missing factor).
        bsdf_product *= (Float32(1.0)-F)*cosHI/max(cosNI*cosTM*cosTM, Float32(1e-6)) * verts[i].eta*verts[i].eta
    var out_positions = InlineArray[Vec3f, MAX_SMS_VERTICES](fill=Vec3f(Float32(0.0)))
    for i in range(n):
        out_positions[i] = verts[i].pos
    return (True, out_positions.copy(), bsdf_product, dx1_dxlight)

# ── Random seeding + Bernoulli-trial reciprocal estimator (5.2/5.3) ─────────

@always_inline
def sms_seed_jitter(mut verts: InlineArray[SMSVertex, MAX_SMS_VERTICES], n: Int, mut pcg: PCG32, jitter_scale: Float32):
    """5.2: perturb each vertex's initial position within its own flat
    triangle's tangent plane by a uniform random offset. `jitter_scale`
    should be small relative to the flat-triangle assumption's own
    validity radius -- the caller derives it from inter-vertex probe
    spacing (see _mnee_area_light_contribute)."""
    for i in range(n):
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
    sms_seed_jitter(seed0, n, pcg, jitter_scale)
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
        sms_seed_jitter(seed_k, n, pcg, jitter_scale)
        var _walkk = sms_walk(x0, xL, seed_k, n, ldp_du, ldp_dv,
            bvh2Nodes, primIds, meshes, curves, blasNodesArr, blasPrimIdsArr, instances, spheres, n_spheres)
        var okk = _walkk[0]
        var posk = _walkk[1].copy()
        if okk and sms_same_solution(x0, posk, pos0, n):
            matched = True; break
    _ = matched
    return (True, pos0.copy(), bsdf0, jac0, trials)
