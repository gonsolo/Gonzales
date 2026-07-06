# Bidirectional Path Tracing (CPU, single-threaded)
# Supports homogeneous participating media and specular chains (glass).
# Strategies: t >= 1, s >= 1 only (no lens sampling for s=0).
# MIS: balance heuristic over all valid connection strategies.

from std.math import sqrt, cos, sin, floor, log, exp, max, abs
from std.memory import alloc
from .geometry import (
    RGB, SampledSpectrum, Point3f, Vec3f, vec3f, point3f, Ray_C, Intersection_C, Frame,
    TriangleMesh_C, Material_C, MatKind, AreaLight_C, Medium_C, MediumInterface_C,
    dot, cross, fr_dielectric, sphere_outward_normal, PI, INV_FOUR_PI, INV_PI,
)
from .bvh import BVH2Node, SceneDescriptor2_C, traverse_bvh2_core, any_hit_bvh2_core, test_spheres
from .rng import PCG32
from .pbrt_parser import ParsedScene_Mojo
from .postprocess import write_image
from .sppm import _geom_normal, _dielectric_bounce, _sppm_update_medium, _cosine_hemisphere_sample, sample_homogeneous_free_flight, sample_area_light_uniform
from .bxdf import GeomContext, bxdf_sample_conductor, bxdf_is_delta, ggx_D, ggx_G2

comptime _BDPT_MAX_DEPTH = 40  # max surface/medium interactions per subpath (incl.
                                # non-stored delta/dielectric bounces — glass-of-water's
                                # nested water/ice/glass interfaces need ~30 crossings
                                # just to reach a real (diffuse) vertex)
comptime _BDPT_MAX_VERTS = 10  # storage per subpath (only non-delta vertices are stored)

# ── Vertex types ──────────────────────────────────────────────────────────────

@fieldwise_init
struct BDPTVertex(TrivialRegisterPassable):
    """A vertex on a camera or light subpath."""
    var pos:    Point3f  # world position
    var normal: Vec3f    # geometric normal (0 for volume)
    var beta: RGB  # throughput to here
    var alb:  RGB  # BSDF albedo (F0 for conductor)
    var pdf_fwd: Float32  # area PDF forward (from previous vertex)
    var pdf_bwd: Float32  # unused by the equal-weight MIS scheme — repurposed to hold
                           # the isotropic GGX alpha for mat_kind=1 (conductor) vertices
    var is_surface: Int32  # 1 = surface hit, 0 = volume scatter
    var is_delta:   Int32  # 1 = specular (mirror conductor / dielectric) — cannot be connected
    var is_light:   Int32  # 1 = this is a light-source vertex (s=0 in BDPT notation)
    var med_idx:    Int32  # medium index AFTER this vertex (-1 = vacuum)
    var mat_kind:   Int32  # 0 = Lambertian (diffuse/volume), 1 = rough conductor (GGX)
    # Direction back toward this vertex's own predecessor on its subpath
    # (-incoming ray direction). Only populated for mat_kind=1 (GGX needs both
    # directions around the half-vector); Lambertian/volume don't need it.
    var wo: Vec3f

@always_inline
def _null_vertex() -> BDPTVertex:
    return BDPTVertex(
        pos=Point3f(Float32(0)),
        normal=Vec3f(Float32(0), Float32(1), Float32(0)),
        beta=RGB(Float32(0)),
        alb=RGB(Float32(0)),
        pdf_fwd=Float32(0), pdf_bwd=Float32(0),
        is_surface=Int32(0), is_delta=Int32(0), is_light=Int32(0),
        med_idx=Int32(-1), mat_kind=Int32(0),
        wo=Vec3f(Float32(0)),
    )

# ── Geometry helpers ──────────────────────────────────────────────────────────

@always_inline
def _bdpt_medium_update(
    ray_dir: SIMD[DType.float32, 3],
    inter:   Intersection_C,
    mat:     Material_C,
    sd:      SceneDescriptor2_C,
    hit: Point3f,
) -> Int32:
    """Determine new current_medium_idx after crossing a surface."""
    if mat.medium_interface_idx < Int32(0) or sd.mediumIfaceCount == Int64(0):
        return Int32(-1)
    var iface = sd.mediumInterfaces[Int(mat.medium_interface_idx)]
    var n: Vec3f
    if inter.primId.type == Int8(4):
        # Analytic sphere: outward normal = normalize(hit - center)
        var si = Int(inter.primId.id1)
        var sph = sd.spheres[si]
        n = sphere_outward_normal(hit, sph.center)
    else:
        var mi: Int; var bv: Int
        if inter.primId.type == 0:
            mi = Int(inter.primId.id1); bv = Int(inter.primId.id2)
        else:
            mi = Int(inter.primId.id2 >> 32); bv = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
        var m  = sd.meshes[mi]
        var v0 = Int(m.vertexIndices[bv]); var v1 = Int(m.vertexIndices[bv+1]); var v2 = Int(m.vertexIndices[bv+2])
        var p0 = Point3f(m.points[v0*4], m.points[v0*4+1], m.points[v0*4+2])
        var p1 = Point3f(m.points[v1*4], m.points[v1*4+1], m.points[v1*4+2])
        var p2 = Point3f(m.points[v2*4], m.points[v2*4+1], m.points[v2*4+2])
        var e1 = p1 - p0; var e2 = p2 - p0
        n = Vec3f(e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x)
    var md = ray_dir[0]*n.x + ray_dir[1]*n.y + ray_dir[2]*n.z
    return iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx

# ── Visibility with transmittance ─────────────────────────────────────────────

def _visible_transmittance(
    a: Point3f, b: Point3f,
    med_idx: Int32,
    sd:      SceneDescriptor2_C,
) -> SIMD[DType.float32, 3]:
    """Returns transmittance along segment AB, or (0,0,0) if occluded.
    Glass (dielectric) surfaces are passed through with Fresnel transmittance."""
    var d = b - a
    var dist_total = d.length()
    if dist_total < Float32(1e-5):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var inv = Float32(1) / dist_total
    var dir = SIMD[DType.float32, 3](d.x*inv, d.y*inv, d.z*inv)

    var Tr = RGB(Float32(1))
    var org = a + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
    var remaining = dist_total - Float32(0.0002)
    var cur_med = med_idx

    var inter_mem = alloc[Intersection_C](1)
    for _ in range(8):
        if remaining < Float32(1e-4): break
        var ray = Ray_C(org, Vec3f(dir[0], dir[1], dir[2]))
        inter_mem[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, remaining * Float32(0.9995), inter_mem,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        # test_spheres (analytic spheres, e.g. the caustic sphere) aren't part of
        # the BVH — traverse_bvh2_core only tests triangles/curves — so they need
        # a separate pass. Seed a sentinel tHit=remaining*0.9995 when the BVH found
        # nothing, so test_spheres's own internal tMax (it only bounds itself by
        # result[0].tHit when result[0].hit is already set) respects the shadow
        # ray's segment length instead of defaulting to unbounded (1e38).
        var had_bvh_hit = inter_mem[0].hit != Int8(0)
        if not had_bvh_hit:
            inter_mem[0].hit = Int8(1)
            inter_mem[0].tHit = remaining * Float32(0.9995)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, inter_mem)
        if not had_bvh_hit and inter_mem[0].primId.type != Int8(4):
            inter_mem[0].hit = Int8(0)
        if inter_mem[0].hit == Int8(0):
            # Nothing between here and destination: apply remaining Beer-Lambert
            if Int(cur_med) >= 0:
                var med = sd.mediums[Int(cur_med)]
                var sigma_t = med.sigma_a + med.sigma_s
                Tr *= RGB(exp(-sigma_t.r*remaining), exp(-sigma_t.g*remaining), exp(-sigma_t.b*remaining))
            break

        var inter = inter_mem[0]
        var t_hit = inter.tHit
        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hit = org + Vec3f(dir[0], dir[1], dir[2]) * t_hit

        # Beer-Lambert through medium segment up to hit
        if Int(cur_med) >= 0:
            var med = sd.mediums[Int(cur_med)]
            var sigma_t = med.sigma_a + med.sigma_s
            Tr *= RGB(exp(-sigma_t.r*t_hit), exp(-sigma_t.g*t_hit), exp(-sigma_t.b*t_hit))

        if mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            # Pass through glass with Fresnel transmittance
            var gn: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si = Int(inter.primId.id1)
                var sph = sd.spheres[si]
                gn = sphere_outward_normal(hit, sph.center).to_simd()
            else:
                gn = _geom_normal(inter, sd.meshes, sd.instances)
            var facing = dot(dir, gn) < Float32(0)
            var n_for_cos = gn if facing else gn*Float32(-1)
            var cos_i = -dot(dir, n_for_cos)
            if cos_i < Float32(0): cos_i = -cos_i
            var ior = mat.albedo.r
            var fr = fr_dielectric(cos_i, Float32(1)/ior if facing else ior)
            var T = Float32(1) - fr
            Tr *= T
            if Tr.r < Float32(1e-7) and Tr.g < Float32(1e-7) and Tr.b < Float32(1e-7):
                inter_mem.free()
                return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
            # Update medium after crossing glass surface
            if mat.medium_interface_idx >= Int32(0) and sd.mediumIfaceCount > Int64(0):
                var iface = sd.mediumInterfaces[Int(mat.medium_interface_idx)]
                var md = dir[0]*gn[0]+dir[1]*gn[1]+dir[2]*gn[2]
                cur_med = iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx
            org = hit + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
            remaining = remaining - t_hit - Float32(0.0002)

        elif mat.type == MatKind.interface:
            # Pure medium boundary: update medium, continue
            if mat.medium_interface_idx >= Int32(0) and sd.mediumIfaceCount > Int64(0):
                var iface = sd.mediumInterfaces[Int(mat.medium_interface_idx)]
                var igna = _geom_normal(inter, sd.meshes, sd.instances)
                var md = dir[0]*igna[0]+dir[1]*igna[1]+dir[2]*igna[2]
                cur_med = iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx
            org = hit + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
            remaining = remaining - t_hit - Float32(0.0002)

        else:
            # Opaque surface blocks the segment
            inter_mem.free()
            return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))

    inter_mem.free()
    return SIMD[DType.float32, 3](Tr.r, Tr.g, Tr.b)

# ── Cosine-area PDF conversion ────────────────────────────────────────────────

@always_inline
def _pdf_solid_to_area(pdf_solid: Float32, cos_theta: Float32, dist2: Float32) -> Float32:
    """Convert solid-angle PDF to area PDF: p_A = p_ω * |cosθ| / r²."""
    if dist2 < Float32(1e-8): return Float32(0)
    return pdf_solid * (cos_theta if cos_theta > Float32(0) else -cos_theta) / dist2

# ── Build camera subpath ──────────────────────────────────────────────────────

def _build_camera_path(
    verts:   UnsafePointer[BDPTVertex, MutAnyOrigin],
    r2c:     UnsafePointer[Float32, MutAnyOrigin],
    c2w:     UnsafePointer[Float32, MutAnyOrigin],
    px:      Int, py:      Int,
    sd:      SceneDescriptor2_C,
    mut pcg: PCG32,
    has_med: Bool,
    start_med_idx: Int32 = Int32(-1),
) -> Int:
    """Trace camera subpath from pixel (px,py). Returns (n_verts, updated_pcg)."""
    var inter_mem = alloc[Intersection_C](1)

    # Generate primary ray
    var fX = Float32(px) + Float32(0.5)
    var fY = Float32(py) + Float32(0.5)
    var cx = r2c[0]*fX + r2c[4]*fY + r2c[12]
    var cy = r2c[1]*fX + r2c[5]*fY + r2c[13]
    var cz = r2c[2]*fX + r2c[6]*fY + r2c[14]
    var cw = r2c[3]*fX + r2c[7]*fY + r2c[15]
    if cw != Float32(0.0) and cw != Float32(1.0):
        cx /= cw; cy /= cw; cz /= cw
    var cl = sqrt(cx*cx + cy*cy + cz*cz)
    if cl > Float32(0.0): cx /= cl; cy /= cl; cz /= cl
    var rd = Vec3f(
        c2w[0]*cx + c2w[4]*cy + c2w[8]*cz,
        c2w[1]*cx + c2w[5]*cy + c2w[9]*cz,
        c2w[2]*cx + c2w[6]*cy + c2w[10]*cz,
    )
    var dl = rd.length()
    if dl > Float32(0.0): rd = rd / dl
    var ro = Point3f(c2w[12], c2w[13], c2w[14])

    var n_verts = 0
    var n_bounces = 0  # total surface hits including glass (for _dielectric_bounce entering logic)
    var beta = RGB(Float32(1))
    var cur_med_idx = start_med_idx

    for _ in range(_BDPT_MAX_DEPTH):
        if n_verts >= _BDPT_MAX_VERTS: break
        var ray = Ray_C(ro, rd)
        inter_mem[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), inter_mem,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, inter_mem)
        if inter_mem[0].hit == Int8(0): break

        var inter = inter_mem[0]
        var t_hit = inter.tHit
        var ray_dir = rd.to_simd()

        # Volume free-flight
        if has_med and Int(cur_med_idx) >= 0:
            var med = sd.mediums[Int(cur_med_idx)]
            var ff = sample_homogeneous_free_flight(med, t_hit, pcg)
            if ff.collided:
                # Volume scatter vertex
                var sp = ro + rd*ff.t_free
                # BDPT vertex beta: Tr/pdf_free × phase/pdf_phase = 1/sig_t × sig_s = alb_s
                # (exp(-sig_t×t) cancels between Tr numerator and pdf denominator)
                var v = _null_vertex()
                v.pos = sp
                v.beta = beta * ff.albedo
                v.alb = ff.albedo
                v.is_surface = Int32(0); v.is_delta = Int32(0)
                v.pdf_fwd = ff.sig_t * exp(-ff.sig_t * ff.t_free)
                v.med_idx = cur_med_idx
                verts[n_verts] = v; n_verts += 1
                # Continuation beta = prev × alb_s (same as stored vertex beta)
                beta *= ff.albedo
                var u1 = pcg.next_float(); var u2 = pcg.next_float()
                var cosT = Float32(2)*u1 - Float32(1)
                var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                var phi  = Float32(2)*PI*u2
                rd = Vec3f(sinT*cos(phi), sinT*sin(phi), cosT)
                ro = sp + rd*Float32(0.0002)
                continue
            else:
                # Beer-Lambert through full segment to surface
                beta *= ff.transmittance

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hit = ro + rd*t_hit

        if mat.type == MatKind.area_light:
            break  # camera hits light directly → handled by unidirectional contribution

        elif mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
            var gn = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn, ray_dir) > Float32(0): gn = gn * Float32(-1)
            var v = _null_vertex()
            v.pos = hit
            v.normal = vec3f(gn)
            v.beta = beta
            v.alb = mat.albedo
            v.is_surface = Int32(1); v.is_delta = Int32(0)
            v.pdf_fwd = Float32(1)  # placeholder, set by MIS
            v.med_idx = cur_med_idx
            verts[n_verts] = v; n_verts += 1
            # Cosine-weighted scatter direction
            var u1 = pcg.next_float(); var u2 = pcg.next_float()
            rd = vec3f(_cosine_hemisphere_sample(gn, u1, u2))
            ro = hit + rd*Float32(0.0002)
            # Update beta: f/pdf for Lambertian = (alb/π) / (cosθ/π) = alb
            beta *= mat.albedo

        elif mat.type == MatKind.conductor:
            var gn_c: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si_c = Int(inter.primId.id1)
                var sph_c = sd.spheres[si_c]
                gn_c = sphere_outward_normal(hit, sph_c.center).to_simd()
            else:
                gn_c = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn_c, ray_dir) > Float32(0): gn_c = gn_c * Float32(-1)
            var wo_c = (-rd).to_simd()
            var frm_c = Frame.from_z(Vec3f(gn_c[0], gn_c[1], gn_c[2]))
            var gc_c = GeomContext(
                normal=gn_c, geo_normal=gn_c, hit_point=hit.to_simd(), wo=wo_c,
                tangent=SIMD[DType.float32, 3](frm_c.x.x, frm_c.x.y, frm_c.x.z),
                bitangent=SIMD[DType.float32, 3](frm_c.y.x, frm_c.y.y, frm_c.y.z),
                alb=mat.albedo, pixel_uv=Float32(0),
            )
            var uc1 = pcg.next_float(); var uc2 = pcg.next_float()
            var bs_c = bxdf_sample_conductor(gc_c, mat, uc1, uc2)
            if bs_c.is_valid == Int8(0):
                break
            if not bxdf_is_delta(bs_c.flags):
                var v = _null_vertex()
                v.pos = hit
                v.normal = vec3f(gn_c)
                v.beta = beta
                v.alb = mat.albedo
                v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = Int32(1)
                v.pdf_bwd = max(mat.roughU, mat.roughV)
                v.wo = vec3f(wo_c)
                v.pdf_fwd = Float32(1)
                v.med_idx = cur_med_idx
                verts[n_verts] = v; n_verts += 1
            beta *= bs_c.f
            rd = vec3f(bs_c.wi)
            ro = hit + rd*Float32(0.0002)

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            var gn: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si = Int(inter.primId.id1)
                var sph = sd.spheres[si]
                gn = sphere_outward_normal(hit, sph.center).to_simd()
            else:
                gn = _geom_normal(inter, sd.meshes, sd.instances)
            var (new_dir, new_org) = _dielectric_bounce(ray_dir, hit.to_simd(), gn, mat.albedo.r, n_bounces, pcg)
            n_bounces += 1
            # Specular vertex: no BSDF record needed, just track throughput
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            rd = vec3f(new_dir)
            ro = point3f(new_org)

        elif mat.type == MatKind.interface:
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            ro = hit + rd*Float32(0.0002)

        else:
            break

    inter_mem.free()
    return n_verts

# ── Build light subpath ───────────────────────────────────────────────────────

def _build_light_path(
    verts:   UnsafePointer[BDPTVertex, MutAnyOrigin],
    sd:      SceneDescriptor2_C,
    mut pcg: PCG32,
    has_med: Bool,
    default_emit_med: Int32,
) -> Int:
    """Emit a photon from a random light and trace a light subpath.
    Returns (n_verts, pcg, flux, total_light_pdf)."""
    var n_lights = Int(sd.areaLightCount)
    if n_lights == 0:
        return 0

    # Pick a light uniformly + a random triangle + barycentric point on it.
    var light_sample = sample_area_light_uniform(sd.areaLights, sd.meshes, n_lights, pcg)
    var al = light_sample.light
    var lp = light_sample.point
    var ln = light_sample.normal

    # Cosine-weighted emission direction
    var du1 = pcg.next_float(); var du2 = pcg.next_float()
    var pdir = _cosine_hemisphere_sample(ln, du1, du2)
    var ly = sqrt(max(Float32(0), Float32(1)-du1))  # cos_theta_emitted, reused below

    # Light vertex (the emission point, s=1 strategy connects here directly).
    # beta = 1/p_A = area × n_lights (area sampling PDF correction only).
    # alb  = Le (emission) — used as f_lgt in _connect for the light vertex.
    # G already handles the cos_l factor, so f_lgt = Le (no extra cos multiply).
    var area_weight = al.total_area * Float32(n_lights)
    var lv0_vert = _null_vertex()
    lv0_vert.pos = point3f(lp)
    lv0_vert.normal = vec3f(ln)
    lv0_vert.beta = RGB(area_weight, area_weight, area_weight)
    lv0_vert.alb = al.emission
    lv0_vert.is_surface = Int32(1); lv0_vert.is_light = Int32(1)
    lv0_vert.pdf_fwd = Float32(1) / area_weight
    lv0_vert.med_idx = default_emit_med
    verts[0] = lv0_vert

    # For traced vertices: beta = Le / (p_A × p_ω) where p_ω = cos_θ/π for cosine-weighted emission.
    # cos_θ_emitted = ly (the cosine of the sampled direction against the light normal).
    # β = Le × area × n_lights × π / cos_θ_emitted
    var cos_theta_emit = max(ly, Float32(0.01))
    var flux = al.emission * (area_weight * PI / cos_theta_emit)
    var ro = point3f(lp) + vec3f(ln)*Float32(0.0001)
    var rd = vec3f(pdir)
    var cur_med_idx = default_emit_med
    var n_verts = 1  # vertex 0 is the light point itself
    var n_lbounces = 1  # counts all surface hits (1 = after initial light vertex)

    var inter_mem = alloc[Intersection_C](1)
    for _ in range(_BDPT_MAX_DEPTH):
        if n_verts >= _BDPT_MAX_VERTS: break
        var ray = Ray_C(ro, rd)
        inter_mem[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), inter_mem,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, inter_mem)
        if inter_mem[0].hit == Int8(0): break

        var inter = inter_mem[0]
        var t_hit = inter.tHit
        var ray_dir = rd.to_simd()

        # Volume free-flight
        if has_med and Int(cur_med_idx) >= 0:
            var med = sd.mediums[Int(cur_med_idx)]
            var ff = sample_homogeneous_free_flight(med, t_hit, pcg)
            if ff.collided:
                var sp = ro + rd*ff.t_free
                # BDPT vertex beta: exp(-sig_t×t)/pdf_free × alb_s = 1/sig_t × alb_s = alb_s (for sig_t=1)
                var v = _null_vertex()
                v.pos = sp
                v.beta = flux * ff.albedo
                v.alb = ff.albedo
                v.is_surface = Int32(0); v.is_delta = Int32(0)
                v.pdf_fwd = ff.sig_t * exp(-ff.sig_t * ff.t_free)
                v.med_idx = cur_med_idx
                verts[n_verts] = v; n_verts += 1
                # Continuation: flux = prev × alb_s (same as stored vertex beta)
                flux *= ff.albedo
                var u1 = pcg.next_float(); var u2 = pcg.next_float()
                var cosT = Float32(2)*u1 - Float32(1)
                var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                var phi  = Float32(2)*PI*u2
                rd = Vec3f(sinT*cos(phi), sinT*sin(phi), cosT)
                ro = sp + rd*Float32(0.0002)
                continue
            else:
                flux *= ff.transmittance

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hit = ro + rd*t_hit

        if mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
            var gn = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn, ray_dir) > Float32(0): gn = gn * Float32(-1)
            var v = _null_vertex()
            v.pos = hit
            v.normal = vec3f(gn)
            v.beta = flux
            v.alb = mat.albedo
            v.is_surface = Int32(1); v.is_delta = Int32(0)
            v.pdf_fwd = Float32(1); v.med_idx = cur_med_idx
            verts[n_verts] = v; n_verts += 1
            # Scatter
            var u1 = pcg.next_float(); var u2 = pcg.next_float()
            rd = vec3f(_cosine_hemisphere_sample(gn, u1, u2))
            ro = hit + rd*Float32(0.0002)
            flux *= mat.albedo

        elif mat.type == MatKind.conductor:
            var gn_c: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si_c = Int(inter.primId.id1)
                var sph_c = sd.spheres[si_c]
                gn_c = sphere_outward_normal(hit, sph_c.center).to_simd()
            else:
                gn_c = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn_c, ray_dir) > Float32(0): gn_c = gn_c * Float32(-1)
            var wo_c = (-rd).to_simd()
            var frm_c = Frame.from_z(Vec3f(gn_c[0], gn_c[1], gn_c[2]))
            var gc_c = GeomContext(
                normal=gn_c, geo_normal=gn_c, hit_point=hit.to_simd(), wo=wo_c,
                tangent=SIMD[DType.float32, 3](frm_c.x.x, frm_c.x.y, frm_c.x.z),
                bitangent=SIMD[DType.float32, 3](frm_c.y.x, frm_c.y.y, frm_c.y.z),
                alb=mat.albedo, pixel_uv=Float32(0),
            )
            var uc1 = pcg.next_float(); var uc2 = pcg.next_float()
            var bs_c = bxdf_sample_conductor(gc_c, mat, uc1, uc2)
            if bs_c.is_valid == Int8(0):
                break
            if not bxdf_is_delta(bs_c.flags):
                var v = _null_vertex()
                v.pos = hit
                v.normal = vec3f(gn_c)
                v.beta = flux
                v.alb = mat.albedo
                v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = Int32(1)
                v.pdf_bwd = max(mat.roughU, mat.roughV)
                v.wo = vec3f(wo_c)
                v.pdf_fwd = Float32(1)
                v.med_idx = cur_med_idx
                verts[n_verts] = v; n_verts += 1
            flux *= bs_c.f
            rd = vec3f(bs_c.wi)
            ro = hit + rd*Float32(0.0002)

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            var gn: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si = Int(inter.primId.id1)
                var sph = sd.spheres[si]
                gn = sphere_outward_normal(hit, sph.center).to_simd()
            else:
                gn = _geom_normal(inter, sd.meshes, sd.instances)
            var (new_dir, new_org) = _dielectric_bounce(ray_dir, hit.to_simd(), gn, mat.albedo.r, n_lbounces, pcg)
            n_lbounces += 1
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            rd = vec3f(new_dir)
            ro = point3f(new_org)

        elif mat.type == MatKind.interface:
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hit)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            ro = hit + rd*Float32(0.0002)

        else:
            break

    inter_mem.free()
    return n_verts

# ── BSDF/phase evaluation at a vertex ────────────────────────────────────────

@always_inline
def _eval_conductor_ggx(
    n:     SIMD[DType.float32, 3],
    wo:    SIMD[DType.float32, 3],   # toward this vertex's own predecessor
    wi:    SIMD[DType.float32, 3],   # toward the other connected vertex
    alpha: Float32,
    f0: RGB,
) -> SIMD[DType.float32, 3]:
    """Isotropic GGX (Trowbridge-Reitz) conductor f(wo,wi) × |cos(wi,n)|, for an
    arbitrary (not self-sampled) direction pair — the BDPT connection case that
    bxdf_sample_conductor (a self-sampled-direction-only throughput multiplier)
    can't serve. Schlick Fresnel at the half-vector, height-correlated Smith G2."""
    var cos_o = dot(wo, n)
    var cos_i = dot(wi, n)
    if cos_o <= Float32(0) or cos_i <= Float32(0):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var wh = wo + wi
    var whl = dot(wh, wh)
    if whl <= Float32(0):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    wh = wh * (Float32(1) / sqrt(whl))
    var cos_h = dot(wh, n)
    var cos_wo_h = dot(wo, wh)
    if cos_wo_h < Float32(0): cos_wo_h = -cos_wo_h
    var d = ggx_D(cos_h, alpha)
    var g = ggx_G2(cos_o, cos_i, alpha)
    var one_m = Float32(1) - cos_wo_h
    var one_m2 = one_m * one_m
    var schlick = one_m2 * one_m2 * one_m
    var fr = f0 + (RGB(Float32(1)) - f0) * schlick
    var k = d * g / (Float32(4) * cos_o * cos_i) * cos_i
    return SIMD[DType.float32, 3](k * fr.r, k * fr.g, k * fr.b)

@always_inline
def _eval_vertex(
    v:   BDPTVertex,
    dir_to_other:  SIMD[DType.float32, 3],   # direction from v toward the other connected vertex
) -> SIMD[DType.float32, 3]:
    """Evaluate BSDF (or phase) × cos at vertex v toward dir_to_other."""
    if v.is_delta != Int32(0):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    if v.is_surface == Int32(0):
        # Volume scatter: isotropic phase function 1/(4π), no cosine term
        return SIMD[DType.float32, 3](v.alb.r*INV_FOUR_PI, v.alb.g*INV_FOUR_PI, v.alb.b*INV_FOUR_PI)
    var vn = v.normal.to_simd()
    if v.mat_kind == Int32(1):
        var vwo = v.wo.to_simd()
        return _eval_conductor_ggx(vn, vwo, dir_to_other, v.pdf_bwd, v.alb)
    # Surface: Lambertian f = alb/π × |cos(wo,n)|
    var cos_o = dot(dir_to_other, vn)
    if cos_o < Float32(0): cos_o = -cos_o
    return SIMD[DType.float32, 3](v.alb.r*INV_PI*cos_o, v.alb.g*INV_PI*cos_o, v.alb.b*INV_PI*cos_o)

# ── Geometry term ─────────────────────────────────────────────────────────────

@always_inline
def _geom_term(
    a: BDPTVertex, b: BDPTVertex,
) -> Float32:
    """Geometry factor G(a,b) = |cos_a| × |cos_b| / dist²."""
    var d3 = b.pos - a.pos
    var dist2 = d3.length_sq()
    if dist2 < Float32(1e-8): return Float32(0)
    var d = sqrt(dist2)
    var dir = d3.to_simd() / d
    var cos_a: Float32
    if a.is_surface == Int32(1):
        cos_a = dot(dir, a.normal.to_simd())
        if cos_a < Float32(0): cos_a = -cos_a
    else:
        cos_a = Float32(1)   # volume: no cosine
    var cos_b: Float32
    if b.is_surface == Int32(1):
        cos_b = dot(dir, b.normal.to_simd())
        if cos_b < Float32(0): cos_b = -cos_b
    else:
        cos_b = Float32(1)
    return cos_a * cos_b / dist2

# ── Connect one camera vertex to one light vertex ─────────────────────────────

def _connect(
    cv: BDPTVertex,  # camera-subpath vertex
    lv: BDPTVertex,  # light-subpath vertex (including light point itself)
    sd: SceneDescriptor2_C,
    has_med: Bool,
) -> SIMD[DType.float32, 3]:
    """Evaluate the contribution of connecting cv to lv via a shadow ray.
    Returns colour contribution (not yet MIS-weighted; caller divides by n_strategies)."""
    if cv.is_delta != Int32(0) or lv.is_delta != Int32(0):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))

    var d3 = lv.pos - cv.pos
    var dist2 = d3.length_sq()
    if dist2 < Float32(1e-8):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var dist = sqrt(dist2)

    # Determine medium for the shadow segment.
    # Use camera vertex's medium (both should agree in a well-defined scene).
    var seg_med = cv.med_idx
    var Tr = _visible_transmittance(cv.pos, lv.pos, seg_med, sd)
    if Tr[0] < Float32(1e-7) and Tr[1] < Float32(1e-7) and Tr[2] < Float32(1e-7):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))

    var dir = d3.to_simd() / dist
    var neg_dir = -dir

    # BSDF at camera vertex (toward light)
    var f_cam = _eval_vertex(cv, dir)
    # BSDF at light vertex (toward camera)
    var f_lgt: SIMD[DType.float32, 3]
    if lv.is_light == Int32(1):
        # Light emission: f_lgt = Le (emission radiance, no cosine here).
        # _geom_term already computes cos_l at the light surface, so don't multiply again.
        var ln = lv.normal.to_simd()
        var cos_l = dot(neg_dir, ln)  # emission normal vs direction toward cv
        if cos_l <= Float32(0):
            return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
        f_lgt = SIMD[DType.float32, 3](lv.alb.r, lv.alb.g, lv.alb.b)
    else:
        f_lgt = _eval_vertex(lv, neg_dir)

    # Geometry term G = |cos_cv| × |cos_lv| / dist²
    var G = _geom_term(cv, lv)

    var beta = cv.beta * lv.beta
    var contrib = f_cam * f_lgt * SIMD[DType.float32, 3](G, G, G) * Tr
    contrib[0] *= beta.r
    contrib[1] *= beta.g
    contrib[2] *= beta.b
    return contrib

# ── Main BDPT render ──────────────────────────────────────────────────────────

def bdpt_render(
    psc:      UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
    sd:       SceneDescriptor2_C,
    n_spp:    Int,
    no_denoise: Bool,
    verbose:  Bool,
) -> Int32:
    """Bidirectional Path Tracing main loop — one output EXR per pixel."""
    var fw = Int(psc[0].film_w)
    var fh = Int(psc[0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[0].film_iso / Float32(100)
    var max_comp  = psc[0].film_max_comp

    print("BDPT: " + String(fw) + "x" + String(fh) + "  " + String(n_spp) + " spp")

    var has_med = Int(sd.mediumCount) > 0

    # Determine starting medium for light subpaths (same logic as SPPM)
    var default_emit_med = Int32(-1)
    if has_med and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    # Output buffer: one RGB per pixel
    var buf = alloc[RGB](n_pix)
    for i in range(n_pix):
        buf[i] = RGB(Float32(0))

    var cam_verts   = alloc[BDPTVertex](_BDPT_MAX_VERTS)
    var light_verts = alloc[BDPTVertex](_BDPT_MAX_VERTS)

    var r2c = psc[0].raster_to_camera
    var c2w = psc[0].camera_to_world
    var base_seed = psc[0].rng_seed

    for pix in range(n_pix):
        var px = pix % fw; var py = pix // fw
        var acc = RGB(Float32(0))

        for si in range(n_spp):
            var pcg = PCG32(base_seed ^ UInt64(pix * 6364136223846793005 + 1442695040888963407),
                            UInt64(si * 2654435761 + 1))

            # ── Camera subpath ──────────────────────────────────────────────
            var n_cam = _build_camera_path(
                cam_verts, r2c, c2w, px, py, sd, pcg, has_med)

            # ── Light subpath ───────────────────────────────────────────────
            var n_light = _build_light_path(
                light_verts, sd, pcg, has_med, default_emit_med)

            if n_cam == 0 or n_light == 0:
                continue

            # ── Connect all camera × light vertex pairs with equal MIS weights ──
            # A connection (ci,li) samples a path of total length n = ci+li+2.
            # Multiple (ci',li') pairs with ci'+li'=ci+li estimate the SAME path
            # integral I_n, so divide each by the count of valid strategies for
            # that n (equal-weight MIS: unbiased and prevents overcounting).
            var strat_count: SIMD[DType.int32, 20] = SIMD[DType.int32, 20](0)
            for ci in range(n_cam):
                if cam_verts[ci].is_delta != Int32(0): continue
                for li in range(n_light):
                    if light_verts[li].is_delta != Int32(0): continue
                    var k = ci + li
                    if k < 20: strat_count[k] = strat_count[k] + Int32(1)

            var sum = RGB(Float32(0))
            for ci in range(n_cam):
                if cam_verts[ci].is_delta != Int32(0): continue
                for li in range(n_light):
                    if light_verts[li].is_delta != Int32(0): continue
                    var k = ci + li
                    var ns = Float32(strat_count[k]) if k < 20 else Float32(1)
                    if ns < Float32(1): ns = Float32(1)
                    var c = _connect(cam_verts[ci], light_verts[li], sd, has_med)
                    sum += RGB(c[0], c[1], c[2]) / ns

            acc += sum

        var inv_spp = iso_scale / Float32(n_spp)
        buf[pix] = acc * inv_spp

        if verbose and pix % (n_pix // 10 + 1) == 0:
            print("BDPT: " + String(pix * 100 // n_pix) + "%")

    cam_verts.free(); light_verts.free()

    # Clamp and write image
    var pixels = alloc[Float32](n_pix * 3)
    for i in range(n_pix):
        var c = buf[i]
        if max_comp > Float32(0):
            c.r = c.r if c.r < max_comp else max_comp
            c.g = c.g if c.g < max_comp else max_comp
            c.b = c.b if c.b < max_comp else max_comp
        pixels[i*3]   = c.r
        pixels[i*3+1] = c.g
        pixels[i*3+2] = c.b
    buf.free()

    _ = write_image(pixels, psc[0].film_w, psc[0].film_h, psc[0].film_filename, Int32(32), Int32(32))
    pixels.free()
    return Int32(0)
