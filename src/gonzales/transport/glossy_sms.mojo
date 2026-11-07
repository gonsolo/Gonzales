"""Glossy SMS: manifold next-event estimation from a glossy vertex through a chain of smooth dielectrics that may
reflect as well as refract, seeded by the glossy lobe (Zeltner et al. 2020, with their reciprocal root estimate)."""
from std.collections import Array
from std.math import sqrt, abs
from std.sys import get_defined_int
from gonzales.core.geometry import Point3f, Vec3f, Ray, dot
from gonzales.core.rng import PCG32
from gonzales.core.spectrum import SampledWavelengths
from gonzales.scene.appearance.materials import Material, MatKind, LobeKind
from gonzales.scene.shapes.primitives import Intersection, PrimId, mesh_pos
from gonzales.scene.accel.bvh import traverse_dispatch
from gonzales.transport.shade_context import ShadeContext
from gonzales.transport.bxdf import BSDF, LobeTables
from gonzales.transport.texturing import _get_tri_verts
from gonzales.transport.sms import (MAX_SMS_VERTICES, SMSVertex, sms_vertex_init, sms_walk, sms_chain_eval,
    sms_refresh_solved_frames, sms_refresh_mesh_frames, mnee_orthonormal_basis, SMS_UNIQUENESS_COS_EPS)
from gonzales.transport.manifold import _sms_vertex_from_hit, _mnee_chain_g, _mnee_chain_t_eta, _mnee_bsdf_over_light

# Glossy lobes up to this GGX alpha start SMS; MIS keeps any threshold unbiased, a wider lobe just gains little.
comptime GLOSSY_SMS_MAX_ALPHA: Float32 = Float32(0.3)
# Cap on the reciprocal estimate's re-solves. Truncation darkens a root a seed finds with chance q by (1 - q)^cap.
comptime GLOSSY_SMS_MAX_TRIALS: Int = 64
comptime _OFF: Float32 = Float32(2e-4)
# DEBUG (-D GSMS_DEBUG=n): only glossy SMS (1), only BSDF chain paths from glossy vertices with their MIS share (2) or
# unweighted (3); 1 + 2 must match 3. Everything else contributes nothing.
comptime GSMS_DEBUG: Int = get_defined_int["GSMS_DEBUG", 0]()

comptime SMSChain = Array[SMSVertex, MAX_SMS_VERTICES]


@always_inline
def glossy_sms_lobe(bsdf: BSDF) -> Bool:
    """A conductor lobe narrow enough that its specular chains are worth solving."""
    return bsdf.kind == LobeKind.ggx and not bsdf.is_delta and bsdf.param <= GLOSSY_SMS_MAX_ALPHA


@always_inline
def glossy_sms_live(ctx: ShadeContext) -> Bool:
    return ctx.glossy_sms and ctx.cycles[3] == 0 and ctx.lights.area_light_count > 0


@always_inline
def _chain_kind(mat: Material) -> Int:
    """0: not a chain surface; 1: smooth glass (refracts, or reflects totally); 2: a smooth skin boundary (reflects,
    as what it transmits starts a walk)."""
    if mat.type != MatKind.dielectric or mat.is_rough_dielectric():
        return 0
    return 2 if mat.sss_walk_interior() else 1


@always_inline
def _first_hit(ctx: ShadeContext, org: Vec3f, dir: Vec3f, tmax: Float32) -> Intersection:
    var o = org + dir * _OFF
    var dummy_prim = PrimId(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var store = Array[Intersection, 1](fill=Intersection(dummy_prim, tmax, Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0)))
    traverse_dispatch(ctx.bvh4, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves,
                      Ray(Point3f(o[0], o[1], o[2]), Vec3f(dir[0], dir[1], dir[2])), tmax, store.unsafe_ptr(),
                      ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances, ctx.lights.spheres, ctx.lights.sphere_count)
    return store[0]


def sms_trace_chain(ctx: ShadeContext, org: Vec3f, dir_in: Vec3f) -> Tuple[Int, SMSChain, Intersection, Vec3f, Vec3f]:
    """Follows a ray through the chain surfaces by the chain rule: smooth glass refracts unless it reflects totally,
    a smooth skin boundary reflects. Returns (vertex count, -1 when the ray escapes, enters skin or exceeds
    MAX_SMS_VERTICES; the vertices; the first other surface hit; the last segment's origin and direction).

    The rule makes the chain a function of (org, dir): seeds follow it, and a path found by BSDF sampling belongs to
    glossy SMS exactly when re-tracing its first direction by the rule reproduces it."""
    var verts = SMSChain(fill=sms_vertex_init())
    var o = org
    var d = dir_in
    var n = 0
    for _k in range(MAX_SMS_VERTICES + 1):
        var hit = _first_hit(ctx, o, d, Float32(1e30))
        if hit.hit == Int8(0):
            return (-1, verts^, hit, o, d)
        var kind = 0
        if hit.primId.type == Int8(0) or hit.primId.type == Int8(4):
            kind = _chain_kind(ctx.materials[unsafe_offset=Int(hit.primId.materialIndex)])
        if kind == 0:
            return (n, verts^, hit, o, d)
        if n == MAX_SMS_VERTICES:
            return (-1, verts^, hit, o, d)
        var ro = o + d * _OFF
        var (v, ok) = _sms_vertex_from_hit(ctx, hit, ro, d)
        if not ok:
            return (-1, verts^, hit, o, d)
        var nf = v.normal if dot(v.normal, d) < Float32(0) else -v.normal   # toward the side the ray comes from
        var ci = -dot(d, nf)
        var e = Float32(1) / v.eta   # n_incident / n_transmitted; v.eta > 1 when the ray enters
        var s2 = e * e * (Float32(1) - ci * ci)
        if kind == 2 and v.eta < Float32(1):
            return (-1, verts^, hit, o, d)   # inside skin: the walk's, not a chain's
        if kind == 2 or s2 >= Float32(1):
            v.refl = Int8(1)
            d = d + nf * (Float32(2) * ci)
        else:
            d = d * e + nf * (e * ci - sqrt(Float32(1) - s2))
        var dl = sqrt(dot(d, d))
        if dl <= Float32(1e-12):
            return (-1, verts^, hit, o, d)
        d = d * (Float32(1) / dl)
        verts[n] = v
        n += 1
        o = v.pos
    return (-1, verts^, _first_hit(ctx, o, d, Float32(1e30)), o, d)


def sms_chain_lands(ctx: ShadeContext, x0: Vec3f, dir: Vec3f, n_spec: Int, y: Vec3f) -> Tuple[Bool, Intersection, SMSChain, Int]:
    """Whether re-tracing dir from x0 by the chain rule crosses n_spec chain vertices and lands on a light at y: what
    makes a chain glossy SMS's. A solved root must pass it too, not only a BSDF path: a flat vertex is solved on its
    whole plane, a curved one re-cast from its anchor, and a seed's total internal reflection may end up a partial one
    at the root, which the rule (and so the BSDF side) would refract instead."""
    var tr = sms_trace_chain(ctx, x0, dir)
    var n = tr[0]
    var ok = n >= 1 and n == n_spec and tr[2].primId.type == Int8(3)
    if ok:
        var d = tr[3] + tr[4] * (tr[2].tHit + _OFF) - y
        var r = y - x0
        ok = dot(d, d) <= max(Float32(1e-7) * dot(r, r), Float32(1e-8))
    return (ok, tr[2], tr[1].copy(), n)


@always_inline
def _same_root(x0: Vec3f, a: SMSChain, b: SMSChain, n: Int) -> Bool:
    for i in range(n):
        var da = a[i].pos - x0
        var db = b[i].pos - x0
        var c = dot(da, db) / sqrt(max(dot(da, da) * dot(db, db), Float32(1e-30)))
        if abs(c - Float32(1.0)) > SMS_UNIQUENESS_COS_EPS:
            return False
    return True


def _glossy_sms_try(
    ctx: ShadeContext, x0: Vec3f, bsdf: BSDF, tab: LobeTables, wl: SampledWavelengths,
    y: Vec3f, ldu: Vec3f, ldv: Vec3f, mut pcg: PCG32,
) -> Tuple[Bool, Int, SMSChain, Float32, Float32]:
    """One seed: a lobe direction traced by the chain rule and solved to end at y. A root whose re-trace crosses other
    surfaces (a reflection solved straight to the light through the glass in between) seeds a solve of the chain that
    re-trace found, up to twice: the seed rarely meets the glass a light point beyond it needs, a nearby root does.
    (ok, n, chain, weight, |dx1/dy|)."""
    var u0 = pcg.next_float(); var u1 = pcg.next_float(); var u2 = pcg.next_float()
    var s = bsdf.sample[want_rev=False](u0, u1, u2, Float32(0), tab, wl)
    var tr = sms_trace_chain(ctx, x0, s.wi)
    var n = tr[0]
    var verts = tr[1].copy()
    if not s.valid or n < 1:
        return (False, 0, verts^, Float32(0), Float32(0))
    for _round in range(3):
        var w = sms_walk(x0, y, verts, n, ldu, ldv, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves,
                         ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances, ctx.lights.spheres, ctx.lights.sphere_count)
        if not w[0]:
            return (False, 0, verts^, Float32(0), Float32(0))
        for i in range(n):
            verts[i].pos = w[1][i]
        sms_refresh_solved_frames(verts, n)
        if not sms_refresh_mesh_frames(x0, verts, n, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves,
                                       ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances):
            return (False, 0, verts^, Float32(0), Float32(0))
        var d0 = verts[0].pos - x0
        var la = sms_chain_lands(ctx, x0, d0 * (Float32(1) / sqrt(dot(d0, d0))), n, y)
        if la[0]:
            return (True, n, verts^, w[2], w[3])
        if la[3] < 1 or la[3] == n:
            return (False, 0, verts^, Float32(0), Float32(0))
        n = la[3]
        verts = la[2].copy()
    return (False, 0, verts^, Float32(0), Float32(0))


def glossy_sms_solve(
    ctx: ShadeContext, x0: Vec3f, bsdf: BSDF, tab: LobeTables, wl: SampledWavelengths,
    y: Vec3f, ldu: Vec3f, ldv: Vec3f, mut pcg: PCG32,
) -> Tuple[Bool, Int, SMSChain, Float32, Float32, Float32]:
    """A root of the chains from x0 to the light point y, found from a random lobe seed, and the reciprocal of the
    chance a seed finds it, estimated by re-seeding until one does (Zeltner et al. 2020, Sec. 4.3): (ok, n, chain,
    weight, g = dw0/dA_y, trial count). ldu, ldv: the light's orthonormal tangents."""
    var first = _glossy_sms_try(ctx, x0, bsdf, tab, wl, y, ldu, ldv, pcg)
    var n = first[1]
    var verts = first[2].copy()
    if not first[0]:
        return (False, 0, verts^, Float32(0), Float32(0), Float32(0))
    var trials = Float32(0)
    for _t in range(GLOSSY_SMS_MAX_TRIALS):
        trials += Float32(1)
        var k = _glossy_sms_try(ctx, x0, bsdf, tab, wl, y, ldu, ldv, pcg)
        if k[0] and k[1] == n and _same_root(x0, k[2], verts, n):
            break
    var g = _mnee_chain_g(x0, verts, first[4])
    return (True, n, verts^, first[3], g, trials)


@always_inline
def glossy_sms_mis(p_dir: Float32, t_chain: Float32, g: Float32, p_area: Float32) -> Float32:
    """Glossy SMS's power-heuristic share of a chain path. Both sides weigh it with the deterministic solve's density
    p_area / g, not the seeded one (which only the reciprocal estimate knows): any function of the path that both
    evaluate alike partitions unity, and every root it gives a share has a seed basin. BSDF sampling: the lobe's
    p_dir times the chance of each vertex's event, t_chain."""
    if g <= Float32(0) or p_area <= Float32(0):
        return Float32(1)
    var q = _mnee_bsdf_over_light(p_dir, t_chain, g, p_area)
    return Float32(1) / (Float32(1) + q * q)


def glossy_sms_reproduce(
    ctx: ShadeContext, x0: Vec3f, dir: Vec3f, n_spec: Int, light_hit: Vec3f, light: Intersection,
) -> Tuple[Bool, Int, SMSChain, Float32]:
    """Whether the BSDF path that left x0 along dir and reached light_hit on the triangle `light` through n_spec
    specular vertices is a chain glossy SMS samples (re-traced by the chain rule, it lands there), with its chain and g."""
    var la = sms_chain_lands(ctx, x0, dir, n_spec, light_hit)
    var n = n_spec
    var verts = la[2].copy()
    if not la[0] or la[1].primId.id1 != light.primId.id1:
        return (False, n, verts^, Float32(0))
    var (lmesh, lv0, lv1, lv2, _) = _get_tri_verts(light, ctx.meshes)
    var lp0 = mesh_pos(lmesh, lv0)
    var (ldu, ldv) = mnee_orthonormal_basis(mesh_pos(lmesh, lv1) - lp0, mesh_pos(lmesh, lv2) - lp0)
    var ev = sms_chain_eval(x0, light_hit, verts, n, ldu, ldv)
    if not ev[0]:
        return (False, n, verts^, Float32(0))
    var g = _mnee_chain_g(x0, verts, ev[2])
    return (True, n, verts^, g)


@always_inline
def glossy_sms_t_chain(x0: Vec3f, verts: SMSChain, n: Int) -> Float32:
    return _mnee_chain_t_eta(x0, verts, n)[0]
