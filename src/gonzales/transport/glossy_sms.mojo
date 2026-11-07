"""Glossy SMS: manifold next-event estimation from a glossy vertex through a chain of smooth dielectrics that may
reflect as well as refract, seeded by the glossy lobe (Zeltner et al. 2020, with their reciprocal root estimate)."""
from std.collections import Array
from std.math import sqrt, abs
from std.sys import get_defined_int, is_gpu
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
comptime GLOSSY_SMS_MAX_ALPHA: Float32 = Float32(0.2)
# Glossy SMS takes no share of a chain whose first direction the lobe samples with less density than this: seeds come
# from the lobe, so such a root's chance q is tiny, and its reciprocal estimate would run into the cap below. Both
# sides of MIS test the same density, so the paths it gives up stay with BSDF sampling, unbiased.
comptime GLOSSY_SMS_MIN_PDF: Float32 = Float32(1.0)
# Cap on the reciprocal estimate's re-solves. Truncation darkens a root a seed finds with chance q by (1 - q)^cap.
comptime GLOSSY_SMS_MAX_TRIALS: Int = get_defined_int["GSMS_CAP", 256]()
comptime GLOSSY_SMS_STRAIGHT: Float32 = Float32(0.25)   # share of seeds aimed straight at the light point
comptime _OFF: Float32 = Float32(2e-4)
# A re-traced chain lands on its light point when it misses by at most sqrt(_LAND2) of the distance from x0.
comptime _LAND2: Float32 = Float32(1e-7)
# DEBUG (-D GSMS_DEBUG=n): only glossy SMS (1), only BSDF chain paths from glossy vertices with their MIS share (2) or
# unweighted (3); 1 + 2 must match 3. Everything else contributes nothing. 6: glossy_sms_stage's success rate.
comptime GSMS_DEBUG: Int = get_defined_int["GSMS_DEBUG", 0]()

comptime SMSChain = Array[SMSVertex, MAX_SMS_VERTICES]


@always_inline
def glossy_sms_lobe(bsdf: BSDF) -> Bool:
    """A conductor lobe narrow enough that its specular chains are worth solving."""
    return bsdf.kind == LobeKind.ggx and not bsdf.is_delta and bsdf.param <= GLOSSY_SMS_MAX_ALPHA


# Off on the GPU unless -D GSMS_GPU=1: there the solver finds the reflection chains' roots (skin ball + glass block in
# po/polish.pbrt) far less often than on the CPU, while BSDF sampling, alike on both, still hands them to it, so the
# highlight came out dark. Walk convergence and the mesh re-projection match the CPU; the loss is later (cause open).
comptime _GSMS_GPU = get_defined_int["GSMS_GPU", 0]() != 0


@always_inline
def glossy_sms_live(ctx: ShadeContext) -> Bool:
    comptime if is_gpu() and not _GSMS_GPU:
        return False
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


def sms_trace_chain(ctx: ShadeContext, org: Vec3f, dir_in: Vec3f, mut verts: SMSChain) -> Tuple[Int, Intersection, Vec3f, Vec3f]:
    """Follows a ray through the chain surfaces by the chain rule: smooth glass refracts unless it reflects totally,
    a smooth skin boundary reflects. Writes the vertices to `verts` and returns (their count, -1 when the ray escapes,
    enters skin or exceeds MAX_SMS_VERTICES; the first other surface hit; the last segment's origin and direction).
    The chain arrays travel as out-parameters, not in returned tuples: returned that way, IGC lost them on the GPU.

    The rule makes the chain a function of (org, dir): seeds follow it, and a path found by BSDF sampling belongs to
    glossy SMS exactly when re-tracing its first direction by the rule reproduces it."""
    var o = org
    var d = dir_in
    var n = 0
    for _k in range(MAX_SMS_VERTICES + 1):
        var hit = _first_hit(ctx, o, d, Float32(1e30))
        if hit.hit == Int8(0):
            return (-1, hit, o, d)
        var kind = 0
        if hit.primId.type == Int8(0) or hit.primId.type == Int8(4):
            kind = _chain_kind(ctx.materials[unsafe_offset=Int(hit.primId.materialIndex)])
        if kind == 0:
            return (n, hit, o, d)
        if n == MAX_SMS_VERTICES:
            return (-1, hit, o, d)
        var ro = o + d * _OFF
        var (v, ok) = _sms_vertex_from_hit(ctx, hit, ro, d)
        if not ok:
            return (-1, hit, o, d)
        var nf = v.normal if dot(v.normal, d) < Float32(0) else -v.normal   # toward the side the ray comes from
        var ci = -dot(d, nf)
        var e = Float32(1) / v.eta   # n_incident / n_transmitted; v.eta > 1 when the ray enters
        var s2 = e * e * (Float32(1) - ci * ci)
        if kind == 2 and v.eta < Float32(1):
            return (-1, hit, o, d)   # inside skin: the walk's, not a chain's
        if kind == 2 or s2 >= Float32(1):
            v.refl = Int8(1)
            d = d + nf * (Float32(2) * ci)
        else:
            d = d * e + nf * (e * ci - sqrt(Float32(1) - s2))
        var dl = sqrt(dot(d, d))
        if dl <= Float32(1e-12):
            return (-1, hit, o, d)
        d = d * (Float32(1) / dl)
        verts[n] = v
        n += 1
        o = v.pos
    return (-1, _first_hit(ctx, o, d, Float32(1e30)), o, d)


def sms_chain_lands(
    ctx: ShadeContext, x0: Vec3f, dir: Vec3f, n_spec: Int, y: Vec3f, mut verts: SMSChain,
) -> Tuple[Bool, Intersection, Int]:
    """Whether re-tracing dir from x0 by the chain rule (into `verts`) crosses n_spec chain vertices and lands on a
    light at y: what makes a chain glossy SMS's. (lands, the surface it ends on, the vertex count traced). A solved
    root must pass it too, not only a BSDF path: a flat vertex is solved on its whole plane, a curved one re-cast from
    its anchor, and a seed's total internal reflection may end up a partial one at the root, which the rule (and so
    the BSDF side) would refract instead."""
    var tr = sms_trace_chain(ctx, x0, dir, verts)
    var n = tr[0]
    var ok = n >= 1 and n == n_spec and tr[1].primId.type == Int8(3)
    if ok:
        var d = tr[2] + tr[3] * (tr[1].tHit + _OFF) - y
        var r = y - x0
        ok = dot(d, d) <= max(_LAND2 * dot(r, r), Float32(1e-8))
    return (ok, tr[1], n)


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
    y: Vec3f, ldu: Vec3f, ldv: Vec3f, mut pcg: PCG32, mut verts: SMSChain,
) -> Tuple[Bool, Int, Float32, Float32]:
    """One seed: a direction traced by the chain rule and solved to end at y. The direction is the straight one to y
    (MNEE's seed, for chains in front of the light) with chance GLOSSY_SMS_STRAIGHT, else a lobe sample. A root whose
    re-trace crosses other surfaces (a reflection solved straight to the light through the glass in between) seeds a
    solve of the chain that re-trace found, up to twice: the seed rarely meets the glass a light point beyond it needs,
    a nearby root does. (ok, n, weight, |dx1/dy|), the chain in `verts`."""
    var uc = pcg.next_float(); var u0 = pcg.next_float(); var u1 = pcg.next_float(); var u2 = pcg.next_float()
    var dir = y - x0
    dir = dir * (Float32(1) / sqrt(max(dot(dir, dir), Float32(1e-30))))
    var valid = True
    if uc >= GLOSSY_SMS_STRAIGHT:
        var s = bsdf.sample[want_rev=False](u0, u1, u2, Float32(0), tab, wl)
        dir = s.wi
        valid = s.valid
    var n = sms_trace_chain(ctx, x0, dir, verts)[0]
    if not valid or n < 1:
        return (False, 0, Float32(0), Float32(0))
    for _round in range(3):
        var w = sms_walk(x0, y, verts, n, ldu, ldv, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves,
                         ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances, ctx.lights.spheres, ctx.lights.sphere_count)
        if not w[0]:
            return (False, 0, Float32(0), Float32(0))
        for i in range(n):
            verts[i].pos = w[1][i]
        sms_refresh_solved_frames(verts, n)
        if not sms_refresh_mesh_frames(x0, verts, n, ctx.bvh2Nodes, ctx.primIds, ctx.meshes, ctx.curves,
                                       ctx.blasNodesArr, ctx.blasPrimIdsArr, ctx.instances):
            return (False, 0, Float32(0), Float32(0))
        var d0 = verts[0].pos - x0
        var traced = SMSChain(fill=sms_vertex_init())
        var la = sms_chain_lands(ctx, x0, d0 * (Float32(1) / sqrt(dot(d0, d0))), n, y, traced)
        if la[0]:
            return (True, n, w[2], w[3])
        if la[2] < 1 or la[2] == n:
            return (False, 0, Float32(0), Float32(0))
        n = la[2]
        for i in range(n):
            verts[i] = traced[i]
    return (False, 0, Float32(0), Float32(0))


def glossy_sms_stage(
    ctx: ShadeContext, x0: Vec3f, bsdf: BSDF, tab: LobeTables, wl: SampledWavelengths,
    y: Vec3f, ldu: Vec3f, ldv: Vec3f, mut pcg: PCG32,
) -> Float32:
    """DEBUG (GSMS_DEBUG=6): 1 when a solve finds a root with a share (GSMS_STAGE=2) or one try does (else)."""
    var verts = SMSChain(fill=sms_vertex_init())
    comptime if get_defined_int["GSMS_STAGE", 0]() == 2:
        return Float32(1) if glossy_sms_solve(ctx, x0, bsdf, tab, wl, y, ldu, ldv, pcg, verts)[0] else Float32(0)
    return Float32(1) if _glossy_sms_try(ctx, x0, bsdf, tab, wl, y, ldu, ldv, pcg, verts)[0] else Float32(0)


def glossy_sms_solve(
    ctx: ShadeContext, x0: Vec3f, bsdf: BSDF, tab: LobeTables, wl: SampledWavelengths,
    y: Vec3f, ldu: Vec3f, ldv: Vec3f, mut pcg: PCG32, mut verts: SMSChain,
) -> Tuple[Bool, Int, Float32, Float32, Float32]:
    """A root of the chains from x0 to the light point y, found from a random seed (_glossy_sms_try), and the
    reciprocal of the chance a seed finds it, estimated by re-seeding until one does (Zeltner et al. 2020, Sec. 4.3);
    none when the lobe's density toward it is below GLOSSY_SMS_MIN_PDF. (ok, n, weight, g = dw0/dA_y, trial count),
    the chain in `verts`. ldu, ldv: the light's orthonormal tangents."""
    var first = _glossy_sms_try(ctx, x0, bsdf, tab, wl, y, ldu, ldv, pcg, verts)
    var n = first[1]
    if not first[0]:
        return (False, 0, Float32(0), Float32(0), Float32(0))
    var d0 = verts[0].pos - x0
    if bsdf.f(d0 * (Float32(1) / sqrt(dot(d0, d0))), tab, wl).pdf_fwd < GLOSSY_SMS_MIN_PDF:
        return (False, 0, Float32(0), Float32(0), Float32(0))   # no share (glossy_sms_mis): skip the re-solves
    var trials = Float32(0)
    for _t in range(GLOSSY_SMS_MAX_TRIALS):
        trials += Float32(1)
        var other = SMSChain(fill=sms_vertex_init())
        var k = _glossy_sms_try(ctx, x0, bsdf, tab, wl, y, ldu, ldv, pcg, other)
        if k[0] and k[1] == n and _same_root(x0, other, verts, n):
            break
    var g = _mnee_chain_g(x0, verts, first[3])
    return (True, n, first[2], g, trials)


@always_inline
def glossy_sms_mis(p_dir: Float32, t_chain: Float32, g: Float32, p_area: Float32) -> Float32:
    """Glossy SMS's power-heuristic share of a chain path. Both sides weigh it with the deterministic solve's density
    p_area / g, not the seeded one (which only the reciprocal estimate knows): any function of the path that both
    evaluate alike partitions unity, and every root it gives a share has a seed basin. BSDF sampling: the lobe's
    p_dir times the chance of each vertex's event, t_chain."""
    if p_dir < GLOSSY_SMS_MIN_PDF:
        return Float32(0)
    if g <= Float32(0) or p_area <= Float32(0):
        return Float32(1)
    var q = _mnee_bsdf_over_light(p_dir, t_chain, g, p_area)
    return Float32(1) / (Float32(1) + q * q)


def glossy_sms_reproduce(
    ctx: ShadeContext, x0: Vec3f, dir: Vec3f, n_spec: Int, light_hit: Vec3f, light: Intersection, mut verts: SMSChain,
) -> Tuple[Bool, Float32]:
    """Whether the BSDF path that left x0 along dir and reached light_hit on the triangle `light` through n_spec
    specular vertices is a chain glossy SMS samples (re-traced by the chain rule, it lands there), with g; the chain
    in `verts`."""
    var la = sms_chain_lands(ctx, x0, dir, n_spec, light_hit, verts)
    if not la[0] or la[1].primId.id1 != light.primId.id1:
        return (False, Float32(0))
    var (lmesh, lv0, lv1, lv2, _) = _get_tri_verts(light, ctx.meshes)
    var lp0 = mesh_pos(lmesh, lv0)
    var (ldu, ldv) = mnee_orthonormal_basis(mesh_pos(lmesh, lv1) - lp0, mesh_pos(lmesh, lv2) - lp0)
    var ev = sms_chain_eval(x0, light_hit, verts, n_spec, ldu, ldv)
    if not ev[0]:
        return (False, Float32(0))
    return (True, _mnee_chain_g(x0, verts, ev[2]))


@always_inline
def glossy_sms_t_chain(x0: Vec3f, verts: SMSChain, n: Int) -> Float32:
    return _mnee_chain_t_eta(x0, verts, n)[0]
