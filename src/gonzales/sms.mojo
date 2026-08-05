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

from std.math import sqrt, abs, max
from .geometry import RGB, dot, cross, fr_dielectric
from .rng import PCG32

comptime MAX_SMS_VERTICES: Int = 6
comptime SMS_BERNOULLI_MAX_TRIALS: Int = 16
comptime SMS_SOLUTION_EPS: Float32 = Float32(1e-4)

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
    wi: SIMD[DType.float32, 3], wo: SIMD[DType.float32, 3],
    H: SIMD[DType.float32, 3], s: SIMD[DType.float32, 3], t: SIMD[DType.float32, 3],
    dp_du: SIMD[DType.float32, 3], dp_dv: SIMD[DType.float32, 3],
    ili: Float32, ilo: Float32,
    dp_du_prev: SIMD[DType.float32, 3], dp_dv_prev: SIMD[DType.float32, 3],
    dp_du_next: SIMD[DType.float32, 3], dp_dv_next: SIMD[DType.float32, 3],
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
    normal/dp_du/dp_dv/eta are fixed properties of the vertex's flat
    triangle (same no-reprojection limitation as _mnee_walk/_mnee_walk2 --
    valid as long as the true manifold solution stays within that
    triangle). `eta` is the relative IOR crossing this vertex (ior if
    entering, 1/ior if exiting), precomputed by the caller from probe
    geometry exactly like _mnee_walk2's eta1/eta2."""
    var pos:    SIMD[DType.float32, 3]
    var normal: SIMD[DType.float32, 3]
    var dp_du:  SIMD[DType.float32, 3]
    var dp_dv:  SIMD[DType.float32, 3]
    var eta:    Float32

@always_inline
def sms_vertex_init() -> SMSVertex:
    var z = SIMD[DType.float32, 3](Float32(0.0))
    return SMSVertex(z, z, z, z, Float32(1.0))

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
    var wi:  SIMD[DType.float32, 3]
    var wo:  SIMD[DType.float32, 3]
    var H:   SIMD[DType.float32, 3]
    var wol: Float32
    var ili: Float32
    var ilo: Float32
    var ok:  Int8

@always_inline
def _sms_eval_bad() -> SMSVertexEval:
    var z4 = SIMD[DType.float32, 4](Float32(0.0))
    var z3 = SIMD[DType.float32, 3](Float32(0.0))
    var z2 = SIMD[DType.float32, 2](Float32(0.0))
    return SMSVertexEval(z4, z4, z4, z2, z3, z3, z3, Float32(0.0), Float32(0.0), Float32(0.0), Int8(0))

@always_inline
def _sms_eval_vertex(
    x0: SIMD[DType.float32, 3], xL: SIMD[DType.float32, 3],
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
    var z3 = SIMD[DType.float32, 3](Float32(0.0))
    var dpu_prev = verts[i-1].dp_du if has_prev else z3
    var dpv_prev = verts[i-1].dp_dv if has_prev else z3
    var dpu_next = verts[i+1].dp_du if has_next else z3
    var dpv_next = verts[i+1].dp_dv if has_next else z3
    var (ai, bi, ci, cvi) = sms_vertex_mats(wi, wo, H, s, t, verts[i].dp_du, verts[i].dp_dv, ili, ilo,
        dpu_prev, dpv_prev, dpu_next, dpv_next, has_prev, has_next)
    return SMSVertexEval(ai, bi, ci, cvi, wi, wo, H, wol, ili, ilo, Int8(1))

def sms_walk(
    x0: SIMD[DType.float32, 3], xL: SIMD[DType.float32, 3],
    verts_init: InlineArray[SMSVertex, MAX_SMS_VERTICES], n: Int,
    ldp_du: SIMD[DType.float32, 3], ldp_dv: SIMD[DType.float32, 3],
) -> Tuple[Bool, InlineArray[SIMD[DType.float32, 3], MAX_SMS_VERTICES], Float32, Float32]:
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
    var verts = verts_init
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
        var err = Float32(0.0)
        for i in range(n):
            err = max(err, max(abs(ev[i].cv[0]), abs(ev[i].cv[1])))
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
        for i in range(n):
            var step_len = sqrt(dx[i][0]*dx[i][0] + dx[i][1]*dx[i][1])
            var max_step = ev[i].wol * Float32(0.5)
            var dxi = dx[i]
            if step_len > max_step and step_len > Float32(1e-12):
                dxi = dxi * (max_step/step_len)
            verts[i].pos = verts[i].pos - (verts[i].dp_du*dxi[0] + verts[i].dp_dv*dxi[1])
    var zero_positions = InlineArray[SIMD[DType.float32, 3], MAX_SMS_VERTICES](fill=SIMD[DType.float32, 3](Float32(0.0)))
    if not converged:
        return (False, zero_positions, Float32(0.0), Float32(0.0))
    # ── Final recompute: BSDF product + light-Jacobian chain ────────────────
    var evf = InlineArray[SMSVertexEval, MAX_SMS_VERTICES](fill=_sms_eval_bad())
    for i in range(n):
        evf[i] = _sms_eval_vertex(x0, xL, verts, n, i)
        if evf[i].ok == Int8(0):
            return (False, zero_positions, Float32(0.0), Float32(0.0))
    var cprimef = InlineArray[SIMD[DType.float32, 4], MAX_SMS_VERTICES](fill=SIMD[DType.float32, 4](Float32(0.0)))
    var (Li0f, det0f) = mat22_inv(evf[0].b)
    if det0f == Float32(0.0):
        return (False, zero_positions, Float32(0.0), Float32(0.0))
    cprimef[0] = mat22_mul(Li0f, evf[0].c)
    var last_Li = Li0f  # overwritten below when n > 1; stays Li0f when n == 1
    for i in range(1, n):
        var pivot = evf[i].b - mat22_mul(evf[i].a, cprimef[i-1])
        var (Lii, deti) = mat22_inv(pivot)
        if deti == Float32(0.0):
            return (False, zero_positions, Float32(0.0), Float32(0.0))
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
        bsdf_product *= (Float32(1.0)-F)*cosHI/max(cosNI*cosTM*cosTM, Float32(1e-6))
    var out_positions = InlineArray[SIMD[DType.float32, 3], MAX_SMS_VERTICES](fill=SIMD[DType.float32, 3](Float32(0.0)))
    for i in range(n):
        out_positions[i] = verts[i].pos
    return (True, out_positions, bsdf_product, dx1_dxlight)

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
    a: InlineArray[SIMD[DType.float32, 3], MAX_SMS_VERTICES],
    b: InlineArray[SIMD[DType.float32, 3], MAX_SMS_VERTICES], n: Int,
) -> Bool:
    for i in range(n):
        var d = a[i] - b[i]
        if dot(d,d) > SMS_SOLUTION_EPS*SMS_SOLUTION_EPS:
            return False
    return True

def sms_solve_bernoulli(
    x0: SIMD[DType.float32, 3], xL: SIMD[DType.float32, 3],
    verts_seed: InlineArray[SMSVertex, MAX_SMS_VERTICES], n: Int,
    ldp_du: SIMD[DType.float32, 3], ldp_dv: SIMD[DType.float32, 3],
    jitter_scale: Float32, mut pcg: PCG32,
) -> Tuple[Bool, InlineArray[SIMD[DType.float32, 3], MAX_SMS_VERTICES], Float32, Float32, Float32]:
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
    var seed0 = verts_seed
    sms_seed_jitter(seed0, n, pcg, jitter_scale)
    var (ok0, pos0, bsdf0, jac0) = sms_walk(x0, xL, seed0, n, ldp_du, ldp_dv)
    if not ok0:
        var zero_positions = InlineArray[SIMD[DType.float32, 3], MAX_SMS_VERTICES](fill=SIMD[DType.float32, 3](Float32(0.0)))
        return (False, zero_positions, Float32(0.0), Float32(0.0), Float32(0.0))
    var trials = Float32(0.0)
    var matched = False
    for _t in range(SMS_BERNOULLI_MAX_TRIALS):
        trials += Float32(1.0)
        var seed_k = verts_seed
        sms_seed_jitter(seed_k, n, pcg, jitter_scale)
        var (okk, posk, _bsdfk, _jack) = sms_walk(x0, xL, seed_k, n, ldp_du, ldp_dv)
        if okk and sms_same_solution(posk, pos0, n):
            matched = True; break
    _ = matched
    return (True, pos0, bsdf0, jac0, trials)
