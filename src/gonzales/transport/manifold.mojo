"""Solving a specular chain to a light: the probe, the manifold walks, and their geometry term."""
from std.collections import Array
from std.math import sqrt, abs
from gonzales.core.geometry import Point3f, Vec3f, dot, cross, offset_eps, _is_real_ptr
from gonzales.scene.appearance.materials import fr_dielectric
from gonzales.scene.appearance.materials import normal_slope_map_none
from gonzales.scene.shapes.primitives import Intersection, PrimId, TriangleMesh, Instance, mesh_normal
from gonzales.core.geometry import Ray
from gonzales.scene.shapes.primitives import mesh_pos
from gonzales.core.rng import PCG32
from gonzales.scene.accel.bvh import any_hit_dispatch, traverse_dispatch
from gonzales.core.transform import transform_normal, Mat4
from gonzales.transport.sms import MAX_SMS_VERTICES, SMSVertex, sms_vertex_init, sms_vertex_flat, sms_vertex_sphere, sms_vertex_mats, mat22_mul, mat22_mul_v, mat22_inv, sms_walk, SMS_SOLVER_THRESHOLD, mnee_orthonormal_basis, sms_refresh_solved_frames, sms_refresh_mesh_frames, sms_vertex_mesh, sms_curved, tri_barycentrics
from gonzales.transport.texturing import _get_tri_verts
from gonzales.transport.shade_context import ShadeContext


# Orient an emitter triangle's WINDING normal to agree with the mesh's own
# supplied per-vertex normals, which is what decides whether a one-sided
# pbrt area light faces a given direction.
#
# pbrt does this in Triangle::InteractionFromIntersection: when the mesh
# carries `normal N`, the geometric normal is flipped to the shading
# normal's hemisphere (`isect.n = FaceForward(isect.n, ns)`), and
# DiffuseAreaLight::L's one-sided `Dot(n, w) < 0` test then uses THAT
# normal. Winding alone is not authoritative: an exporter is free to emit a
# quad whose index order winds opposite to the `N` it also supplies, and
# staircase2 has exactly that -- its big window emitter supplies
# N = (1,0,0) while its indices wind to (-1,0,0). Testing the raw winding
# normal there made the light face away from the room: it rendered pure
# black when looked at (4906 px, 2% of frame, against a reference showing
# its full 4.575/3.591/1.550 radiance) AND lit nothing via NEE, since the
# same sign test gates both.
#
# Returns the (unnormalized) winding normal, flipped if needed -- callers
# that need a unit vector normalize it themselves, exactly as before.
@always_inline
def _emitter_face_normal(
    mesh: TriangleMesh,
    v0: Int, v1: Int, v2: Int,
    bu: Float32, bv: Float32,
    p0: Vec3f, p1: Vec3f, p2: Vec3f,
    instance_idx: Int32 = Int32(-1),
    instances: Pointer[Instance, MutUntrackedOrigin] = Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(),
) -> Vec3f:
    var gn = cross(p1 - p0, p2 - p0)
    if not _is_real_ptr(mesh.normals):
        return gn
    var w0 = Float32(1.0) - bu - bv
    var n0 = mesh_normal(mesh, v0)
    var n1 = mesh_normal(mesh, v1)
    var n2 = mesh_normal(mesh, v2)
    var sn = n0 * w0 + n1 * bu + n2 * bv
    if instance_idx >= Int32(0):
        sn = transform_normal(Mat4(instances[unsafe_offset=Int(instance_idx)].world_to_obj()), sn)
    if dot(sn, sn) <= Float32(1e-12):
        return gn
    if dot(gn, sn) < Float32(0.0):
        return -gn
    return gn

# ── Emitter hits, resolved ONCE for all three integrators ─────────────────────
# Both helpers below were inline blocks in the path tracer's emitter handler.
# SPPM and VCM wrote their own and got both wrong, silently:
#
#  * Facing. They tested the RAW WINDING normal, while the light sampler
#    (sample_area_light_uniform) orients emission by the mesh's DECLARED
#    normals, as pbrt does. An emitter whose winding disagrees with its "N" --
#    the ceiling-light case that sampler's docstring cites -- then emitted into
#    the room for NEE and light paths, yet a camera ray looking straight at it
#    saw a back face and got nothing. sss-backlit-slab's emitter fills most of
#    the frame: the path tracer's median is 57 (L = 60), SPPM's was 0, and
#    VCM's lit it only through the light-source splat that was double-counting
#    every visible emitter.
#  * Curves. For a curve hit primId.id1 is the CURVE index, not an AreaLight
#    index, but both indexed areaLights with it, and _geom_normal returns a
#    +Y placeholder for a curve.
#
# One definition, so the three integrators cannot disagree about which side of
# a light is lit.


@always_inline
def _interp_vertex_normal(
    mesh: TriangleMesh,
    v0: Int, v1: Int, v2: Int,
    bu: Float32, bv: Float32,
    instance_idx: Int32 = Int32(-1),
    instances: Pointer[Instance, MutUntrackedOrigin] = Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(),
) -> Vec3f:
    """The mesh's own interpolated per-vertex normal N, unit length, in world
    space and NOT flipped toward anything; the zero vector when the mesh has
    no normals. pbrt treats this N as the surface's outward direction and
    flips the geometric normal to agree with it -- see shade_dielectric."""
    if not _is_real_ptr(mesh.normals):
        return Vec3f(Float32(0.0))
    var w0 = Float32(1.0) - bu - bv
    var n0 = mesh_normal(mesh, v0)
    var n1 = mesh_normal(mesh, v1)
    var n2 = mesh_normal(mesh, v2)
    var sn = n0 * w0 + n1 * bu + n2 * bv
    if instance_idx >= Int32(0):
        sn = transform_normal(Mat4(instances[unsafe_offset=Int(instance_idx)].world_to_obj()), sn)
    var slen = dot(sn, sn)
    if slen <= Float32(1e-12):
        return Vec3f(Float32(0.0))
    return sn * (Float32(1.0) / sqrt(slen))


# 2-vertex MNEE: Newton walk for x0→x1(glass1)→x2(glass2)→x3.
# etas must be pre-computed: eta = ior if entering glass, 1/ior if exiting.
# Returns (converged, x1_f, x2_f, bsdf_product, dx1_dxlight).
@always_inline
def _mnee_walk2(
    x0: Vec3f, x3: Vec3f,
    x1_init: Vec3f, n1: Vec3f,
    dp_du1: Vec3f, dp_dv1: Vec3f, eta1: Float32,
    x2_init: Vec3f, n2: Vec3f,
    dp_du2: Vec3f, dp_dv2: Vec3f, eta2: Float32,
    ldp_du: Vec3f, ldp_dv: Vec3f,
) -> Tuple[Bool, Vec3f, Vec3f, Float32, Float32]:
    var x1 = x1_init
    var x2 = x2_init
    var converged = False
    for _iter in range(20):
        # ── Vertex 0 (x1): wi from x0, wo to x2 ────────────────────────────
        var wi1v = x0 - x1; var wi1l = sqrt(dot(wi1v,wi1v))
        var wo1v = x2 - x1; var wo1l = sqrt(dot(wo1v,wo1v))
        if wi1l < Float32(1e-6) or wo1l < Float32(1e-6): break
        var wi1 = wi1v*(Float32(1)/wi1l); var wo1 = wo1v*(Float32(1)/wo1l)
        if dot(n1,wi1)*dot(n1,wo1) >= Float32(0): break
        var H1v = -(wi1 + wo1*eta1); var H1l = sqrt(dot(H1v,H1v))
        if H1l < Float32(1e-10): break
        var H1 = H1v*(Float32(1)/H1l)
        var s1 = dp_du1 - n1*dot(dp_du1,n1); var s1l = sqrt(dot(s1,s1))
        if s1l < Float32(1e-10): break
        s1 = s1*(Float32(1)/s1l); var t1 = cross(n1,s1)
        var ili1 = Float32(1)/(H1l*wi1l); var ilo1 = eta1/(H1l*wo1l)
        var (a0,b0,c0,cv0) = sms_vertex_mats(wi1,wo1,H1,s1,t1,dp_du1,dp_dv1,ili1,ilo1,
            Vec3f(0),Vec3f(0),dp_du2,dp_dv2,False,True)
        # ── Vertex 1 (x2): wi from x1, wo to x3 ────────────────────────────
        var wi2v = x1 - x2; var wi2l = sqrt(dot(wi2v,wi2v))
        var wo2v = x3 - x2; var wo2l = sqrt(dot(wo2v,wo2v))
        if wi2l < Float32(1e-6) or wo2l < Float32(1e-6): break
        var wi2 = wi2v*(Float32(1)/wi2l); var wo2 = wo2v*(Float32(1)/wo2l)
        if dot(n2,wi2)*dot(n2,wo2) >= Float32(0): break
        var H2v = -(wi2 + wo2*eta2); var H2l = sqrt(dot(H2v,H2v))
        if H2l < Float32(1e-10): break
        var H2 = H2v*(Float32(1)/H2l)
        var s2 = dp_du2 - n2*dot(dp_du2,n2); var s2l = sqrt(dot(s2,s2))
        if s2l < Float32(1e-10): break
        s2 = s2*(Float32(1)/s2l); var t2 = cross(n2,s2)
        var ili2 = Float32(1)/(H2l*wi2l); var ilo2 = eta2/(H2l*wo2l)
        var (a1,b1,c1,cv1) = sms_vertex_mats(wi2,wo2,H2,s2,t2,dp_du2,dp_dv2,ili2,ilo2,
            dp_du1,dp_dv1,Vec3f(0),Vec3f(0),True,False)
        _ = a0; _ = c1
        # ── Convergence ──────────────────────────────────────────────────────
        var err = max(max(abs(cv0[0]),abs(cv0[1])), max(abs(cv1[0]),abs(cv1[1])))
        if err < SMS_SOLVER_THRESHOLD:
            converged = True; break
        # ── Block tridiagonal solve: [b0 c0; a1 b1]*[dx0;dx1]=[cv0;cv1] ────
        var (Li0, det0) = mat22_inv(b0)
        if det0 == Float32(0): break
        var A1 = mat22_mul(a1, Li0)
        var Lk1 = b1 - mat22_mul(A1, c0)
        var (Li1, det1) = mat22_inv(Lk1)
        if det1 == Float32(0): break
        var C1_red = cv1 - mat22_mul_v(A1, cv0)
        var dx1 = mat22_mul_v(Li1, C1_red)
        var dx0 = mat22_mul_v(Li0, cv0 - mat22_mul_v(c0, dx1))
        # ── Step clamp ──────────────────────────────────────────────────────
        var step0 = sqrt(dx0[0]*dx0[0]+dx0[1]*dx0[1]); var ms0 = wo1l*Float32(0.5)
        if step0 > ms0: dx0 = dx0*(ms0/step0)
        var step1 = sqrt(dx1[0]*dx1[0]+dx1[1]*dx1[1]); var ms1 = wo2l*Float32(0.5)
        if step1 > ms1: dx1 = dx1*(ms1/step1)
        x1 = x1 - (dp_du1*dx0[0] + dp_dv1*dx0[1])
        x2 = x2 - (dp_du2*dx1[0] + dp_dv2*dx1[1])
    if not converged:
        return (False, x1_init, x2_init, Float32(0), Float32(0))
    # ── Recompute final vertex quantities for transfer matrix ────────────────
    var wi1v = x0 - x1; var wi1l = sqrt(dot(wi1v,wi1v))
    var wo1v = x2 - x1; var wo1l = sqrt(dot(wo1v,wo1v))
    var wi2v = x1 - x2; var wi2l = sqrt(dot(wi2v,wi2v))
    var wo2v = x3 - x2; var wo2l = sqrt(dot(wo2v,wo2v))
    if wi1l < Float32(1e-8) or wo1l < Float32(1e-8) or wi2l < Float32(1e-8) or wo2l < Float32(1e-8):
        return (False, x1_init, x2_init, Float32(0), Float32(0))
    var wi1 = wi1v*(Float32(1)/wi1l); var wo1 = wo1v*(Float32(1)/wo1l)
    var wi2 = wi2v*(Float32(1)/wi2l); var wo2 = wo2v*(Float32(1)/wo2l)
    var H1v = -(wi1+wo1*eta1); var H1l = sqrt(dot(H1v,H1v))
    var H2v = -(wi2+wo2*eta2); var H2l = sqrt(dot(H2v,H2v))
    if H1l < Float32(1e-10) or H2l < Float32(1e-10):
        return (False, x1_init, x2_init, Float32(0), Float32(0))
    var H1 = H1v*(Float32(1)/H1l); var H2 = H2v*(Float32(1)/H2l)
    var s1 = dp_du1-n1*dot(dp_du1,n1); s1 = s1*(Float32(1)/sqrt(dot(s1,s1))); var t1 = cross(n1,s1)
    var s2 = dp_du2-n2*dot(dp_du2,n2); s2 = s2*(Float32(1)/sqrt(dot(s2,s2))); var t2 = cross(n2,s2)
    var ili1 = Float32(1)/(H1l*wi1l); var ilo1 = eta1/(H1l*wo1l)
    var ili2 = Float32(1)/(H2l*wi2l); var ilo2 = eta2/(H2l*wo2l)
    var (a0f,b0f,c0f,cv0f) = sms_vertex_mats(wi1,wo1,H1,s1,t1,dp_du1,dp_dv1,ili1,ilo1,
        Vec3f(0),Vec3f(0),dp_du2,dp_dv2,False,True)
    var (a1f,b1f,c1f,cv1f) = sms_vertex_mats(wi2,wo2,H2,s2,t2,dp_du2,dp_dv2,ili2,ilo2,
        dp_du1,dp_dv1,Vec3f(0),Vec3f(0),True,False)
    _ = a0f; _ = c1f; _ = cv0f; _ = cv1f
    # ── Block tridiagonal LU factorization for transfer matrix ───────────────
    var (Li0f, det0f) = mat22_inv(b0f)
    if det0f == Float32(0): return (False, x1_init, x2_init, Float32(0), Float32(0))
    var U0 = mat22_mul(Li0f, c0f)
    var Lk1f = b1f - mat22_mul(a1f, U0)
    var (Li1f, det1f) = mat22_inv(Lk1f)
    if det1f == Float32(0): return (False, x1_init, x2_init, Float32(0), Float32(0))
    # ── dc_dlight at x2 (wrt x3 position) ───────────────────────────────────
    var dc_du = (ldp_du - wo2*dot(wo2,ldp_du)) * ilo2
    var dc_dv = (ldp_dv - wo2*dot(wo2,ldp_dv)) * ilo2
    dc_du -= H2*dot(dc_du,H2); dc_du = -dc_du
    dc_dv -= H2*dot(dc_dv,H2); dc_dv = -dc_dv
    var dc_dlight = SIMD[DType.float32, 4](dot(dc_du,s2), dot(dc_dv,s2), dot(dc_du,t2), dot(dc_dv,t2))
    # ── Transfer matrix chain ────────────────────────────────────────────────
    var Tp = SIMD[DType.float32, 4](Float32(0)-Li1f[0]*dc_dlight[0]-Li1f[1]*dc_dlight[2],
        Float32(0)-Li1f[0]*dc_dlight[1]-Li1f[1]*dc_dlight[3],
        Float32(0)-Li1f[2]*dc_dlight[0]-Li1f[3]*dc_dlight[2],
        Float32(0)-Li1f[2]*dc_dlight[1]-Li1f[3]*dc_dlight[3])
    Tp = SIMD[DType.float32, 4](Float32(0)-U0[0]*Tp[0]-U0[1]*Tp[2], Float32(0)-U0[0]*Tp[1]-U0[1]*Tp[3],
        Float32(0)-U0[2]*Tp[0]-U0[3]*Tp[2], Float32(0)-U0[2]*Tp[1]-U0[3]*Tp[3])
    var dx1_dxlight = abs(Tp[0]*Tp[3]-Tp[1]*Tp[2])
    # ── BSDF product at x1 and x2 ────────────────────────────────────────────
    # Solid-angle-compression factor for refraction through a smooth
    # (delta) dielectric interface -- confirmed against the original SMS
    # paper's reference implementation (specular-manifold-sampling's
    # manifold_ss.cpp specular_reflectance, delta-BSDF branch: `f = 1-F;
    # f *= sqr(eta);`). Verified by hand at normal incidence, where
    # cosHI/cosTM/cosNI all collapse to 1 and the two formulas must agree
    # up to exactly this factor. See
    # project_dielectric_radiance_transmission_bug memory. The eta used is
    # the EMITTER-side orientation (see sms.mojo's sms_walk for the full
    # argument); for this symmetric enter+exit pair eta2 = 1/eta1, so the
    # product is 1 either way and this spelling changes no pixel here --
    # it is the single-refraction chains that actually depend on it.
    var eta1_o = Float32(1)/eta1; var eta2_o = Float32(1)/eta2
    var cosNI1 = abs(dot(n1,wi1)); var cosHI1 = abs(dot(H1,wi1)); var cosTM1 = abs(dot(n1,H1))
    var F1 = fr_dielectric(cosNI1, eta1); var bsdf1 = (Float32(1)-F1)*cosHI1/max(cosNI1*cosTM1*cosTM1,Float32(1e-6)) * eta1_o*eta1_o
    var cosNI2 = abs(dot(n2,wi2)); var cosHI2 = abs(dot(H2,wi2)); var cosTM2 = abs(dot(n2,H2))
    var F2 = fr_dielectric(cosNI2, eta2); var bsdf2 = (Float32(1)-F2)*cosHI2/max(cosNI2*cosTM2*cosTM2,Float32(1e-6)) * eta2_o*eta2_o
    return (True, x1, x2, bsdf1*bsdf2, dx1_dxlight)


@always_inline
# MNEE manifold walk: Newton iteration to find x1 on a glass surface such that
# Snell's law holds for the path x0 → x1 → x2. Works for flat glass surfaces
# (no reprojection needed). Returns (converged, x1_final, det_b, eta_final).
# det_b is the 2x2 constraint Jacobian determinant needed for the transfer matrix.
@always_inline
def _mnee_walk(
    x0: Vec3f,
    x2: Vec3f,
    x1_init: Vec3f,
    n_in: Vec3f,
    dp_du: Vec3f,
    dp_dv: Vec3f,
    eta_in: Float32,
) -> Tuple[Bool, Vec3f, Float32, Float32]:
    var x1 = x1_init
    var n = n_in
    for _iter in range(20):
        var wi3 = x0 - x1
        var wi_len2 = dot(wi3, wi3)
        if wi_len2 < Float32(1e-8):
            return (False, x1_init, Float32(0), eta_in)
        var wi_len = sqrt(wi_len2)
        var wi = wi3 * (Float32(1.0) / wi_len)
        var wo3 = x2 - x1
        var wo_len2 = dot(wo3, wo3)
        if wo_len2 < Float32(1e-8):
            return (False, x1_init, Float32(0), eta_in)
        var wo_len = sqrt(wo_len2)
        var wo = wo3 * (Float32(1.0) / wo_len)
        # eta: pre-computed by caller from dot(pgeo_n_raw, shadow_dir)
        var eta = eta_in
        # Generalized half-vector H = -(wi + eta*wo), Cycles mnee.h convention
        var H3 = -(wi + wo * eta)
        var H_len2 = dot(H3, H3)
        if H_len2 < Float32(1e-10):
            return (False, x1_init, Float32(0), eta)
        var H_len = sqrt(H_len2)
        var H = H3 * (Float32(1.0) / H_len)
        # Tangent frame: s from dp_du projected off n, t = n × s
        var dp_du_dot_n = dot(dp_du, n)
        var s3 = dp_du - n * dp_du_dot_n
        var s_len2 = dot(s3, s3)
        if s_len2 < Float32(1e-10):
            return (False, x1_init, Float32(0), eta)
        var s = s3 * (Float32(1.0) / sqrt(s_len2))
        var tv = cross(n, s)
        # Transmissive check: wi and wo must be on opposite sides of surface
        if dot(n, wi) * dot(n, wo) >= Float32(0.0):
            return (False, x1_init, Float32(0), eta)
        # Constraint: tangential component of H (should be zero at valid refraction)
        var cs = dot(H, s)
        var ct = dot(H, tv)
        # Jacobian b from Cycles mnee_compute_constraint_derivatives (single vertex, flat surface)
        var ilo = eta / (H_len * wo_len)
        var ili = Float32(1.0) / (H_len * wi_len)
        var dH_du = -(dp_du * (ili + ilo)) + wi * (dot(wi, dp_du) * ili) + wo * (dot(wo, dp_du) * ilo)
        var dH_dv = -(dp_dv * (ili + ilo)) + wi * (dot(wi, dp_dv) * ili) + wo * (dot(wo, dp_dv) * ilo)
        dH_du -= H * dot(dH_du, H)
        dH_dv -= H * dot(dH_dv, H)
        dH_du = -dH_du
        dH_dv = -dH_dv
        var b00 = dot(dH_du, s);  var b01 = dot(dH_dv, s)
        var b10 = dot(dH_du, tv); var b11 = dot(dH_dv, tv)
        var det_b = b00 * b11 - b01 * b10
        # Convergence check
        if max(abs(cs), abs(ct)) < SMS_SOLVER_THRESHOLD:
            return (True, x1, det_b, eta)
        # Newton step: solve b * [du, dv] = [cs, ct]
        if abs(det_b) < Float32(1e-5):
            return (False, x1_init, Float32(0), eta)
        var delta_u = (cs * b11 - ct * b01) / det_b
        var delta_v = (-cs * b10 + ct * b00) / det_b
        # Clamp step to prevent divergence
        var step2 = delta_u * delta_u + delta_v * delta_v
        var max_step = wo_len * Float32(0.5)
        if step2 > max_step * max_step:
            var scale = max_step / sqrt(step2)
            delta_u *= scale
            delta_v *= scale
        x1 = x1 - (dp_du * delta_u + dp_dv * delta_v)
    return (False, x1_init, Float32(0), eta_in)


@always_inline
@always_inline
def _sms_vertex_from_hit(
    ctx: ShadeContext,
    inter: Intersection,
    ray_org: Vec3f,
    shadow_dir: Vec3f,
) -> Tuple[SMSVertex, Bool]:
    """Builds an SMSVertex (flat triangle or curved analytic sphere) from a
    probe intersection along a shadow ray -- the ONE place that decides how
    to turn a glass-surface hit into manifold-walk vertex data, shared by
    all three probe call sites in _sms_probe_and_solve (x1, x2, and the
    N-vertex extra_hits loop) instead of each hand-rolling its own
    triangle-only extraction (that used to be genuinely triangle-only,
    which is why a sphere caustic caster was invisible to SMS/MNEE at all
    -- see project_sms_restir_phase6 memory). Entering-vs-exiting eta
    determination (and, for a triangle, the normal-orientation flip) is
    identical between the two primitive kinds; only how the hit POSITION
    and RAW normal are derived differs. Returns (vertex, ok) -- ok=False
    for a degenerate glass triangle (zero-area) or a degenerate sphere hit
    (exactly at its own center, geometrically impossible but checked for
    safety)."""
    var mat = ctx.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var ior = mat.albedo.r
    if inter.primId.type == Int8(4):
        var sph = ctx.lights.spheres[unsafe_offset=Int(inter.primId.id1)]
        var hit_pt = ray_org + shadow_dir * inter.tHit
        var center = Vec3f(sph.center.x, sph.center.y, sph.center.z)
        var to_hit = hit_pt - center
        var to_hit_len = sqrt(dot(to_hit, to_hit))
        if to_hit_len <= Float32(1e-10):
            return (sms_vertex_init(), False)
        var n_raw = to_hit * (Float32(1.0) / to_hit_len)
        var eta = ior if dot(n_raw, shadow_dir) <= Float32(0.0) else (Float32(1.0) / ior)
        # A normal-mapped caster is a genuinely different manifold: the
        # perturbed normal is what the specular constraint is stated in,
        # and the walk has to re-read the map at every position it visits,
        # so the map travels ON the vertex (see sms.mojo's SMSVertex.nmap).
        # Without this the walk solves the SMOOTH sphere -- one broad root
        # per shading point instead of the many sharp ones a normal map
        # creates, which is the difference between a smooth wash and actual
        # caustic filaments.
        var nmap = normal_slope_map_none()
        if mat.normal_tex_idx >= Int32(0) and _is_real_ptr(ctx.nmaps):
            nmap = ctx.nmaps[unsafe_offset=Int(mat.normal_tex_idx)]
        return (sms_vertex_sphere(hit_pt, center, sph.radius, eta, nmap, ior), True)
    elif inter.primId.type == Int8(0):
        var (mesh, v0, v1, v2, _) = _get_tri_verts(inter, ctx.meshes)
        if _is_real_ptr(mesh.normals) and inter.primId.instanceIdx < Int32(0):
            # A smooth mesh: curved, walked on its interpolated normal (sms.mojo's sms_mesh_frame).
            var ns = _interp_vertex_normal(mesh, v0, v1, v2, inter.u, inter.v)
            var eta_m = ior if dot(ns, shadow_dir) <= Float32(0.0) else (Float32(1.0) / ior)
            return sms_vertex_mesh(ctx.meshes, Int(inter.primId.id1), Int(inter.primId.id2), inter.u, inter.v, eta_m, ior)
        var p0 = mesh_pos(mesh, v0)
        var p1 = mesh_pos(mesh, v1)
        var p2 = mesh_pos(mesh, v2)
        var dp_du = p1 - p0
        var dp_dv = p2 - p0
        var n3 = cross(dp_du, dp_dv)
        var n_len = sqrt(dot(n3, n3))
        if n_len <= Float32(1e-10):
            return (sms_vertex_init(), False)
        var n_raw = n3 * (Float32(1.0) / n_len)
        var eta = ior if dot(n_raw, shadow_dir) <= Float32(0.0) else (Float32(1.0) / ior)
        var n = n_raw
        if dot(n, shadow_dir) > Float32(0.0):
            n = -n
        var bu = inter.u; var bv = inter.v
        var point = p0 * (Float32(1.0) - bu - bv) + p1 * bu + p2 * bv
        # Orthonormalize the tangent basis (the reference's
        # ManifoldVertex::make_orthonormal). The triangle's raw edge vectors
        # span the right plane but are neither unit nor perpendicular, so a
        # Jacobian expressed in them is per-PARAMETRIC-unit -- while the
        # estimator pairs it with an AREA-measure light density. Both the
        # specular vertex's basis and the light's have to be orthonormal for
        # |dx1/dxL| to actually be an area-to-area ratio; a sphere vertex
        # already is (its Frisvad frame), which is why only the triangle
        # needed this.
        var (e1, e2) = mnee_orthonormal_basis(dp_du, dp_dv)
        return (sms_vertex_flat(point, n, e1, e2, eta), True)
    else:
        return (sms_vertex_init(), False)


@always_inline
def _sms_on_probed_triangle(ctx: ShadeContext, inter: Intersection, pos: Vec3f) -> Bool:
    """A flat solve runs on the probed triangle's plane; a root outside the triangle belongs to another facet's chain."""
    var (mesh, v0, v1, v2, _) = _get_tri_verts(inter, ctx.meshes)
    var p0 = mesh_pos(mesh, v0)
    return tri_barycentrics(p0, mesh_pos(mesh, v1) - p0, mesh_pos(mesh, v2) - p0, pos)[2]


def _sms_probe_glass_chain(
    ctx: ShadeContext,
    start_point: Vec3f,
    shadow_dir: Vec3f,
    start_remaining: Float32,
    max_count: Int,
) -> Tuple[Int, Array[Intersection, MAX_SMS_VERTICES], Array[Vec3f, MAX_SMS_VERTICES]]:
    """Probes forward from `start_point` along `shadow_dir` for up to
    `max_count` MORE consecutive dielectric surfaces (Phase 5.1's
    generalization of _mnee_area_light_contribute's own hardcoded
    probe/probe2 pair), stopping at the first non-glass hit or miss.
    Returns (count, hits, origins) -- `hits[0..count)` are valid
    Intersection values from ctx's own BVH, same approximate remaining-
    distance bookkeeping the existing probe2 step already uses (each hit's
    tHit is measured from its own segment origin, not the original
    hit_point). `origins[k]` is the ray origin `hits[k].tHit` is measured
    from -- a triangle hit's position can be recovered from barycentrics
    alone, but an analytic sphere hit's cannot (it carries no
    barycentrics), so the caller needs this to reconstruct it via
    origins[k] + shadow_dir*hits[k].tHit."""
    var dummy_prim = PrimId(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var dummy_inter = Intersection(dummy_prim, Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
    var hits = Array[Intersection, MAX_SMS_VERTICES](fill=dummy_inter)
    var origins = Array[Vec3f, MAX_SMS_VERTICES](fill=Vec3f(Float32(0.0)))
    var count = 0
    var seg_org = start_point
    var seg_remaining = start_remaining
    for _k in range(max_count):
        var pk_tmax = seg_remaining * Float32(0.9995)
        if pk_tmax <= Float32(0.001):
            break
        var pk_org = seg_org + shadow_dir * Float32(0.0005)
        var pk_ray = Ray(Point3f(pk_org[0], pk_org[1], pk_org[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
        var pk_store = Array[Intersection, 1](fill=dummy_inter)
        traverse_dispatch(ctx.bvh4, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, pk_ray, pk_tmax, pk_store.unsafe_ptr(),
                           ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                           ctx.lights.spheres, ctx.lights.sphere_count)
        var pk_inter = pk_store[0]
        if pk_inter.hit == Int8(0) or (pk_inter.primId.type != Int8(0) and pk_inter.primId.type != Int8(4)):
            break
        var pk_mat = ctx.materials[unsafe_offset=Int(pk_inter.primId.materialIndex)]
        if not pk_mat.is_specular_glass():
            break
        hits[count] = pk_inter
        origins[count] = pk_org
        count += 1
        var gk_point: Vec3f
        if pk_inter.primId.type == Int8(4):
            gk_point = pk_org + shadow_dir * pk_inter.tHit
        else:
            var (gk_mesh, gk_v0, gk_v1, gk_v2, _) = _get_tri_verts(pk_inter, ctx.meshes)
            var gk_p0 = mesh_pos(gk_mesh, gk_v0)
            var gk_p1 = mesh_pos(gk_mesh, gk_v1)
            var gk_p2 = mesh_pos(gk_mesh, gk_v2)
            var gk_u = pk_inter.u; var gk_v = pk_inter.v
            gk_point = gk_p0*(Float32(1.0)-gk_u-gk_v) + gk_p1*gk_u + gk_p2*gk_v
        seg_remaining = seg_remaining - pk_inter.tHit
        seg_org = gk_point
    return (count, hits^, origins^)


@always_inline
def _mnee_first_probe(
    ctx: ShadeContext, hit_point: Vec3f, shadow_dir: Vec3f, dist: Float32,
) -> Tuple[Bool, Intersection]:
    """The MNEE probe toward the light: (the nearest hit is a triangle of a specular glass material, that hit)."""
    var probe_org = hit_point + shadow_dir * Float32(0.0002)
    var probe_ray = Ray(
        Point3f(probe_org[0], probe_org[1], probe_org[2]),
        Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
    var probe_tmax = dist * Float32(0.9995)
    var dummy_prim = PrimId(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var dummy_inter = Intersection(dummy_prim, probe_tmax, Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
    var probe_store = Array[Intersection, 1](fill=dummy_inter)
    traverse_dispatch(ctx.bvh4, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, probe_ray, probe_tmax, probe_store.unsafe_ptr(),
                       ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                       ctx.lights.spheres, ctx.lights.sphere_count)
    var probe_inter = probe_store[0]
    # A solid sphere is not MNEE's: one refraction misses the exit surface and the exact BSDF path is no
    # longer suppressed at the emitter hit, so plain sampling covers it (PT read 1.4-1.9% under pbrt otherwise).
    if probe_inter.hit == Int8(0) or probe_inter.primId.type != Int8(0):
        return (False, probe_inter)
    var probe_mat = ctx.materials[unsafe_offset=Int(probe_inter.primId.materialIndex)]
    return (probe_mat.is_specular_glass(), probe_inter)


@no_inline   # one copy of the solver per kernel; inlined at every caller it made VCM's compile take tens of minutes
def _sms_solve_from_probe(
    ctx: ShadeContext,
    hit_point: Vec3f,
    shadow_dir: Vec3f,
    dist: Float32,
    light_point: Vec3f,
    ldp_du_v: Vec3f,
    ldp_dv_v: Vec3f,
    mut pcg: PCG32,
    probe_inter: Intersection,
) -> Tuple[Bool, Bool, Int, Array[SMSVertex, MAX_SMS_VERTICES], Float32, Float32, Float32]:
    """Probe for up to MAX_SMS_VERTICES glass surfaces between `hit_point`
    and `light_point` and, if found, solve the resulting specular chain --
    pure geometry/Newton-solve logic shared by `_mnee_area_light_contribute`
    (direct-contribution plain-NEE/ReSTIR-DI path) and ReSTIR SMS's
    candidate generation, factored out so both draw from ONE implementation
    of the probing+dispatch logic instead of two copies drifting apart.
    No access to path_ptr/alb/emission/lobe_w -- those are caller-specific.

    Returns (dielectric_found, solve_ok, n_vertices, verts, bsdf_product,
    dx1_dxlight, trials). `dielectric_found` alone governs whether a
    caller should skip its own straight shadow ray (the light IS occluded
    by glass either way); `solve_ok` governs whether there's an actual
    refracted contribution to use -- mirrors the original function
    returning True (skip shadow ray) even when the manifold walk itself
    failed to converge. `verts[i].pos` holds the SOLVED positions on
    success; `verts[i].normal/dp_du/dp_dv/eta` are always the probed
    triangle's own fixed properties, valid even when `solve_ok=False`
    (a caller doing manifold-shift reuse needs the probed vertex 0's
    geometry regardless of whether the direct-probe seed itself
    converged). `trials` is always 1.0: every chain is one deterministic solve from the probe seed."""
    var zero_verts = Array[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init())
    # MNEE: probe for up to 2 glass surfaces between hit_point and light.
    # For each probe hit we detect entering/exiting from dot(n_raw, probe_dir).
    var probe_org = hit_point + shadow_dir * Float32(0.0002)
    var dummy_prim = PrimId(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var dummy_inter = Intersection(dummy_prim, dist * Float32(0.9995), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))

    # --- Extract x1 geometry (triangle or analytic sphere) ---
    var (v1, v1_ok) = _sms_vertex_from_hit(ctx, probe_inter, probe_org, shadow_dir)
    if not v1_ok:
        # Degenerate glass surface: a dielectric IS in the way (so the
        # straight shadow ray is blocked regardless) but no usable manifold
        # geometry.
        return (True, False, 0, zero_verts.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
    var is_sphere1 = sms_curved(v1)
    var pgeo_n = v1.normal
    var pdp_du = v1.dp_du
    var pdp_dv = v1.dp_dv
    var eta1 = v1.eta
    var x1_init = v1.pos
    # --- Probe for second glass surface beyond x1 ---
    # A solid curved caster is two refractions, entry and exit, solved together below.
    # Modelling it as one idealized refraction (the SMS reference's single scatter) leaves pbrt's answer.
    var probe2_t0 = probe_inter.tHit
    var probe2_rem = (dist - probe2_t0) * Float32(0.9995)
    var probe2_org = x1_init + shadow_dir * Float32(0.0005)
    var probe2_inter = dummy_inter
    if probe2_rem > Float32(0.001):
        var probe2_ray = Ray(
            Point3f(probe2_org[0], probe2_org[1], probe2_org[2]),
            Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
        var probe2_store = Array[Intersection, 1](fill=dummy_inter)
        traverse_dispatch(ctx.bvh4, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, probe2_ray, probe2_rem, probe2_store.unsafe_ptr(),
                   ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                   ctx.lights.spheres, ctx.lights.sphere_count)
        probe2_inter = probe2_store[0]
    var has_second_glass = False
    if probe2_inter.hit != Int8(0) and (probe2_inter.primId.type == Int8(0) or probe2_inter.primId.type == Int8(4)):
        var probe2_mat_c = ctx.materials[unsafe_offset=Int(probe2_inter.primId.materialIndex)]
        has_second_glass = probe2_mat_c.is_specular_glass()

    if not has_second_glass:
        var verts1 = Array[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init())
        verts1[0] = v1
        if is_sphere1:
            # A curved caster needs sms_walk's curvature-aware Newton solve. One deterministic seed finds one root;
            # the BSDF keeps the paths of the others (see _mnee_bsdf_weight), so a caster with many roots stays unbiased.
            var w1 = sms_walk(hit_point, light_point, verts1, 1, ldp_du_v, ldp_dv_v,
                ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                ctx.lights.spheres, ctx.lights.sphere_count)
            if not w1[0]:
                return (True, False, 1, verts1.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
            verts1[0].pos = w1[1][0]
            sms_refresh_solved_frames(verts1, 1)
            if not sms_refresh_mesh_frames(hit_point, verts1, 1, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves,
                                           ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances):
                return (True, False, 1, verts1.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
            return (True, True, 1, verts1.copy(), w1[2], w1[3], Float32(1.0))
        # --- 1-vertex MNEE fast path (flat triangle only, unchanged) ---
        var (mnee_ok, x1_f, det_b, eta_f) = _mnee_walk(
            hit_point, light_point, x1_init, pgeo_n, pdp_du, pdp_dv, eta1)
        if not mnee_ok or not _sms_on_probed_triangle(ctx, probe_inter, x1_f):
            return (True, False, 1, verts1.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
        var wi_f = hit_point - x1_f
        var wi_len2_f = dot(wi_f, wi_f)
        var wo_f = light_point - x1_f
        var wo_len2_f = dot(wo_f, wo_f)
        if wi_len2_f <= Float32(1e-8) or wo_len2_f <= Float32(1e-8):
            return (True, False, 1, verts1.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
        var wi_len_f = sqrt(wi_len2_f)
        var wo_len_f = sqrt(wo_len2_f)
        var wi_fn = wi_f * (Float32(1.0) / wi_len_f)
        var wo_fn = wo_f * (Float32(1.0) / wo_len_f)
        var H3_f = -(wi_fn + wo_fn * eta_f)
        var H_len2_f = dot(H3_f, H3_f)
        if H_len2_f <= Float32(1e-10):
            return (True, False, 1, verts1.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
        var H_len_f = sqrt(H_len2_f)
        var H_f = H3_f * (Float32(1.0) / H_len_f)
        var dp_du_dot_n = dot(pdp_du, pgeo_n)
        var s3_f = pdp_du - pgeo_n * dp_du_dot_n
        var s_len2_f = dot(s3_f, s3_f)
        if s_len2_f <= Float32(1e-10):
            return (True, False, 1, verts1.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
        var s_f = s3_f * (Float32(1.0) / sqrt(s_len2_f))
        var t_f = cross(pgeo_n, s_f)
        var ilo_l = eta_f / (H_len_f * wo_len_f)
        var dHdu_l = (ldp_du_v - wo_fn * dot(wo_fn, ldp_du_v)) * ilo_l
        var dHdv_l = (ldp_dv_v - wo_fn * dot(wo_fn, ldp_dv_v)) * ilo_l
        dHdu_l -= H_f * dot(dHdu_l, H_f); dHdu_l = -dHdu_l
        dHdv_l -= H_f * dot(dHdv_l, H_f); dHdv_l = -dHdv_l
        var dc00 = dot(dHdu_l, s_f); var dc01 = dot(dHdv_l, s_f)
        var dc10 = dot(dHdu_l, t_f); var dc11 = dot(dHdv_l, t_f)
        var det_dc = dc00*dc11 - dc01*dc10
        var dx1_dxl = abs(det_dc) / max(abs(det_b), Float32(1e-8))
        var cosNI = abs(dot(pgeo_n, wi_fn))
        var cosHI = abs(dot(H_f, wi_fn))
        var cosTM = abs(dot(pgeo_n, H_f))
        var F_r = fr_dielectric(cosNI, eta_f)
        var T_f = Float32(1.0) - F_r
        # 1/(eta*eta): see sms.mojo's sms_walk for why the compression
        # factor takes the EMITTER-side orientation of eta while `eta_f`
        # carries the constraint-side one. Unlike the 2-vertex slab above,
        # a single refraction has nothing to cancel against, so getting
        # this backwards here really is an eta^4 error.
        var eta_f_o = Float32(1.0) / eta_f
        var bsdf_s = T_f * cosHI / max(cosNI * cosTM * cosTM, Float32(1e-6)) * eta_f_o*eta_f_o
        verts1[0].pos = x1_f
        return (True, True, 1, verts1.copy(), bsdf_s, dx1_dxl, Float32(1.0))

    var (v2, v2_ok) = _sms_vertex_from_hit(ctx, probe2_inter, probe2_org, shadow_dir)
    if not v2_ok:
        return (True, False, 0, zero_verts.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
    var is_sphere2 = sms_curved(v2)
    var pgeo_n2 = v2.normal
    var pdp_du2 = v2.dp_du
    var pdp_dv2 = v2.dp_dv
    var eta2 = v2.eta
    var x2_init = v2.pos

    # ── Phase 5.1: probe past x2 for a 3rd+ glass surface. Only entered
    # when the chain is genuinely longer than MNEE's own 1-/2-vertex scope
    # (5.4) -- an ordinary 2-surface pane (the overwhelming majority of
    # real glass) never reaches this, so this branch adds no cost there.
    var _chain_res = _sms_probe_glass_chain(
        ctx, x2_init, shadow_dir, dist - probe2_inter.tHit - probe2_t0,
        MAX_SMS_VERTICES - 2)
    var extra_count = _chain_res[0]
    var extra_hits = _chain_res[1].copy()
    var extra_origins = _chain_res[2].copy()

    if extra_count == 0 and not is_sphere1 and not is_sphere2:
        # --- 2-vertex MNEE (unchanged fast path, Phase 5.4) ---
        var verts2 = Array[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init())
        verts2[0] = sms_vertex_flat(x1_init, pgeo_n, pdp_du, pdp_dv, eta1)
        verts2[1] = sms_vertex_flat(x2_init, pgeo_n2, pdp_du2, pdp_dv2, eta2)
        var (ok2, x1_f2, x2_f2, bsdf_prod, dx1_dxl2) = _mnee_walk2(
            hit_point, light_point,
            x1_init, pgeo_n, pdp_du, pdp_dv, eta1,
            x2_init, pgeo_n2, pdp_du2, pdp_dv2, eta2,
            ldp_du_v, ldp_dv_v)
        if not ok2 or not _sms_on_probed_triangle(ctx, probe_inter, x1_f2) or not _sms_on_probed_triangle(ctx, probe2_inter, x2_f2):
            return (True, False, 2, verts2.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
        verts2[0].pos = x1_f2
        verts2[1].pos = x2_f2
        return (True, True, 2, verts2.copy(), bsdf_prod, dx1_dxl2, Float32(1.0))

    var n_total = 2 + extra_count
    var verts = Array[SMSVertex, MAX_SMS_VERTICES](fill=sms_vertex_init())
    verts[0] = v1
    verts[1] = v2
    var chain_ok = True
    for k in range(extra_count):
        var (ek_vert, ek_ok) = _sms_vertex_from_hit(ctx, extra_hits[k], extra_origins[k], shadow_dir)
        if not ek_ok:
            chain_ok = False; break
        verts[2+k] = ek_vert
    if not chain_ok:
        return (True, False, n_total, verts.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
    # Jitter scale (5.2): a small fraction of the average spacing between
    # adjacent probed vertices -- keeps the re-seed within the flat-
    # triangle validity radius each vertex's own straight-line probe
    # already established.
    var _walk_res = sms_walk(hit_point, light_point, verts, n_total, ldp_du_v, ldp_dv_v,
        ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
        ctx.lights.spheres, ctx.lights.sphere_count)
    var sms_ok = _walk_res[0]
    var sms_pos = _walk_res[1].copy()
    var sms_bsdf = _walk_res[2]
    var sms_jac = _walk_res[3]
    var sms_trials = Float32(1.0)
    if not sms_ok:
        return (True, False, n_total, verts.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
    for i in range(n_total):
        verts[i].pos = sms_pos[i]
    sms_refresh_solved_frames(verts, n_total)
    if not sms_refresh_mesh_frames(hit_point, verts, n_total, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves,
                                   ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances):
        return (True, False, n_total, verts.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
    return (True, True, n_total, verts.copy(), sms_bsdf, sms_jac, sms_trials)


@always_inline
def _mnee_chain_g(x0: Vec3f, verts: Array[SMSVertex, MAX_SMS_VERTICES], dx1_dxlight: Float32) -> Float32:
    """dw0/dA_light of a solved chain: solid angle at x0 per light area."""
    # GEOMETRIC normal, as in the reference's geometric term: a normal map changes the BSDF, not the surface's area.
    var first_normal = verts[0].normal
    if verts[0].is_sphere != Int8(0):
        var gnv = verts[0].pos - verts[0].sphere_center
        var gnl = sqrt(dot(gnv, gnv))
        if gnl > Float32(1e-8):
            first_normal = gnv * (Float32(1.0) / gnl)
    elif verts[0].mesh_idx >= Int32(0):
        # dx1_dxlight measures area in the triangle's own tangents
        var tn = cross(verts[0].dp_du, verts[0].dp_dv)
        var tnl = sqrt(dot(tn, tn))
        if tnl > Float32(1e-8):
            first_normal = tn * (Float32(1.0) / tnl)
    var wi = x0 - verts[0].pos
    var wi_len2 = dot(wi, wi)
    if wi_len2 <= Float32(1e-16):
        return Float32(0.0)
    return abs(dot(wi, first_normal)) / (wi_len2 * sqrt(wi_len2)) * dx1_dxlight


@always_inline
def _mnee_chain_t_eta(x0: Vec3f, verts: Array[SMSVertex, MAX_SMS_VERTICES], n: Int) -> Tuple[Float32, Float32]:
    """(probability of refracting at every vertex, the product of 1 - F; the product of the vertices' eta, which is
    n_light / n_x0 for the media at the chain's two ends)."""
    var t = Float32(1.0)
    var eta = Float32(1.0)
    for i in range(n):
        var prev = x0 if i == 0 else verts[i-1].pos
        var wi = prev - verts[i].pos
        var cos_i = abs(dot(verts[i].normal, wi)) / sqrt(max(dot(wi, wi), Float32(1e-30)))
        t *= Float32(1.0) - fr_dielectric(cos_i, verts[i].eta)
        eta *= verts[i].eta
    return (t, eta)


@always_inline
def _mnee_bsdf_over_light(p_x0: Float32, t_chain: Float32, g: Float32, p_light: Float32) -> Float32:
    """BSDF sampling's density of a solved chain over MNEE's, both per solid angle at x0: the deterministic solve's is
    the light's area density over g, BSDF sampling's the lobe at x0 times refracting at every vertex."""
    return p_x0 * t_chain * g / p_light


def _mnee_reproduce(
    ctx: ShadeContext, x0: Vec3f, n_spec: Int, last_spec: Vec3f, light_hit: Vec3f, l_du: Vec3f, l_dv: Vec3f,
    mut pcg: PCG32,
) -> Tuple[Bool, Int, Array[SMSVertex, MAX_SMS_VERTICES], Float32]:
    """Whether MNEE's deterministic solve from x0 to light_hit is the path that got there through n_spec glass
    vertices, the last at last_spec, with the chain and its g. No glass first on the straight segment: MNEE never tries."""
    var verts = Array[SMSVertex, MAX_SMS_VERTICES](uninitialized=True)
    var seg = light_hit - x0
    var seg_len = sqrt(dot(seg, seg))
    if seg_len <= Float32(1e-4):
        return (False, 0, verts^, Float32(0.0))
    var seg_dir = seg * (Float32(1.0) / seg_len)
    var (glass, probe_inter) = _mnee_first_probe(ctx, x0, seg_dir, seg_len)
    if not glass:
        return (False, 0, verts^, Float32(0.0))
    var (o_du, o_dv) = mnee_orthonormal_basis(l_du, l_dv)
    var res = _sms_solve_from_probe(ctx, x0, seg_dir, seg_len, light_hit, o_du, o_dv, pcg, probe_inter)
    var n = res[2]
    verts = res[3].copy()
    if n_spec != n or not res[1]:
        return (False, n, verts^, Float32(0.0))
    var d = verts[n-1].pos - last_spec
    var r = last_spec - x0
    # 0.03% of the distance: another root of the same chain is further away. Never below the ray's spawn offset from
    # the vertex, or a receiver close to the glass finds its own path foreign and it counts twice.
    var off = Float32(4) * offset_eps(last_spec.x, last_spec.y, last_spec.z)
    if dot(d, d) > max(Float32(1e-7) * dot(r, r), off * off):
        return (False, n, verts^, Float32(0.0))
    var g = _mnee_chain_g(x0, verts, res[5])
    return (True, n, verts^, g)


def _mnee_chain(
    ctx: ShadeContext, normal: Vec3f, hit_point: Vec3f, shadow_dir: Vec3f, dist: Float32,
    light_point: Vec3f, ldp_du_v: Vec3f, ldp_dv_v: Vec3f, mut pcg: PCG32, probe_inter: Intersection,
) -> Tuple[Bool, Bool, Int, Array[SMSVertex, MAX_SMS_VERTICES], Float32, Float32, Float32]:
    """MNEE's chain from hit_point through the glass the probe found to light_point, shared by the path tracer and VCM:
    (glass found, solved and visible, vertex count, solved vertices, BSDF product, g = dw0/dA_light, cos at hit_point)."""
    # ORTHONORMALIZE the light's tangent basis before handing it to the
    # manifold walk. `_sample_light_point_and_normal` returns raw triangle
    # EDGES (lp1-lp0, lp2-lp0), which are neither unit-length nor
    # perpendicular, so d(constraint)/d(light) comes out in that triangle's
    # own parametric units -- while `inv_pdf_area` is an AREA-measure
    # density. The two only pair up when the light basis is orthonormal, and
    # the reference makes exactly this call (`vy.make_orthonormal()` on the
    # emitter vertex inside geometric_term) for the same reason.
    var (l_du, l_dv) = mnee_orthonormal_basis(ldp_du_v, ldp_dv_v)
    var _probe_res = _sms_solve_from_probe(
        ctx, hit_point, shadow_dir, dist, light_point, l_du, l_dv, pcg, probe_inter)
    var n = _probe_res[2]
    var verts = _probe_res[3].copy()
    var fail = (_probe_res[0], False, n, verts.copy(), Float32(0.0), Float32(0.0), Float32(0.0))
    if not _probe_res[0] or not _probe_res[1]:
        return fail^

    var first_vertex = verts[0].pos
    var last_vertex = verts[n-1].pos
    var wi_f = hit_point - first_vertex
    var wi_len = sqrt(dot(wi_f, wi_f))
    if wi_len <= Float32(1e-8):
        return fail^
    var wi_fn = wi_f * (Float32(1.0) / wi_len)
    var cos_s_x0 = dot(normal, -wi_fn)
    if cos_s_x0 <= Float32(0.0):
        return fail^
    var g = _mnee_chain_g(hit_point, verts, _probe_res[5])
    var wo_f = light_point - last_vertex
    var wo_len = sqrt(dot(wo_f, wo_f))
    if wo_len <= Float32(1e-8):
        return fail^
    var wo_fn = wo_f * (Float32(1.0) / wo_len)
    var vis_org = last_vertex + wo_fn * Float32(0.001)
    var vis_ray = Ray(Point3f(vis_org[0], vis_org[1], vis_org[2]), Vec3f(wo_fn[0], wo_fn[1], wo_fn[2]))
    # A sphere caustic caster's outgoing leg necessarily re-crosses its OWN
    # far side (the unmodeled exit refraction of the single-vertex model,
    # see any_hit_bvh2_core's own docstring) -- exclude just that one
    # sphere from this occlusion test, not a real distinct occluder.
    var ign_center = Vec3f(Float32(0.0))
    var ign_radius = Float32(-1.0)
    if verts[n-1].is_sphere != Int8(0):
        ign_center = verts[n-1].sphere_center
        ign_radius = verts[n-1].sphere_radius
    if any_hit_dispatch(ctx.bvh4, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves, vis_ray, wo_len * Float32(0.999),
                          ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances,
                          ctx.lights.spheres, ctx.lights.sphere_count,
                          ign_center, ign_radius, materials=ctx.materials):
        return fail^
    return (True, True, n, verts.copy(), _probe_res[4], g, cos_s_x0)
