# Bidirectional Path Tracing (CPU, single-threaded)
# Supports homogeneous participating media and specular chains (glass).
# Strategies: t >= 1, s >= 1 only (no lens sampling for s=0).
# MIS: balance heuristic over all valid connection strategies.

from std.math import sqrt, cos, sin, floor, log, exp, max, abs
from std.memory import alloc
from .geometry import (
    RGB, SampledSpectrum, Point3f, Vec3f, Ray_C, Intersection_C, Frame,
    TriangleMesh_C, Material_C, MatKind, AreaLight_C, Medium_C, MediumInterface_C,
    dot, cross, fr_dielectric, PI, INV_FOUR_PI, INV_PI,
)
from .bvh import BVH2Node, SceneDescriptor2_C, traverse_bvh2_core, any_hit_bvh2_core
from .rng import PCG32
from .pbrt_parser import ParsedScene_Mojo
from .postprocess import write_image
from .sppm import _geom_normal, _dielectric_bounce, _sppm_update_medium
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
    var px: Float32; var py: Float32; var pz: Float32   # world position
    var nx: Float32; var ny: Float32; var nz: Float32   # geometric normal (0 for volume)
    var beta_r: Float32; var beta_g: Float32; var beta_b: Float32  # throughput to here
    var alb_r:  Float32; var alb_g:  Float32; var alb_b: Float32   # BSDF albedo (F0 for conductor)
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
    var wx: Float32; var wy: Float32; var wz: Float32

@always_inline
def _null_vertex() -> BDPTVertex:
    return BDPTVertex(
        px=Float32(0), py=Float32(0), pz=Float32(0),
        nx=Float32(0), ny=Float32(1), nz=Float32(0),
        beta_r=Float32(0), beta_g=Float32(0), beta_b=Float32(0),
        alb_r=Float32(0), alb_g=Float32(0), alb_b=Float32(0),
        pdf_fwd=Float32(0), pdf_bwd=Float32(0),
        is_surface=Int32(0), is_delta=Int32(0), is_light=Int32(0),
        med_idx=Int32(-1), mat_kind=Int32(0),
        wx=Float32(0), wy=Float32(0), wz=Float32(0),
    )

# ── Geometry helpers ──────────────────────────────────────────────────────────

@always_inline
def _bdpt_medium_update(
    ray_dir: SIMD[DType.float32, 3],
    inter:   Intersection_C,
    mat:     Material_C,
    sd:      SceneDescriptor2_C,
    hx: Float32, hy: Float32, hz: Float32,
) -> Int32:
    """Determine new current_medium_idx after crossing a surface."""
    if mat.medium_interface_idx < Int32(0) or sd.mediumIfaceCount == Int64(0):
        return Int32(-1)
    var iface = sd.mediumInterfaces[Int(mat.medium_interface_idx)]
    var nx: Float32; var ny: Float32; var nz: Float32
    if inter.primId.type == Int8(4):
        # Analytic sphere: outward normal = normalize(hit - center)
        var si = Int(inter.primId.id1)
        var sph = sd.spheres[si]
        nx = hx - sph.center.x; ny = hy - sph.center.y; nz = hz - sph.center.z
        var nl = sqrt(nx*nx + ny*ny + nz*nz)
        if nl > Float32(0): nx /= nl; ny /= nl; nz /= nl
    else:
        var mi: Int; var bv: Int
        if inter.primId.type == 0:
            mi = Int(inter.primId.id1); bv = Int(inter.primId.id2)
        else:
            mi = Int(inter.primId.id2 >> 32); bv = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
        var m  = sd.meshes[mi]
        var v0 = Int(m.vertexIndices[bv]); var v1 = Int(m.vertexIndices[bv+1]); var v2 = Int(m.vertexIndices[bv+2])
        var e1x = m.points[v1*4]-m.points[v0*4]; var e1y = m.points[v1*4+1]-m.points[v0*4+1]; var e1z = m.points[v1*4+2]-m.points[v0*4+2]
        var e2x = m.points[v2*4]-m.points[v0*4]; var e2y = m.points[v2*4+1]-m.points[v0*4+1]; var e2z = m.points[v2*4+2]-m.points[v0*4+2]
        nx = e1y*e2z - e1z*e2y; ny = e1z*e2x - e1x*e2z; nz = e1x*e2y - e1y*e2x
    var md = ray_dir[0]*nx + ray_dir[1]*ny + ray_dir[2]*nz
    return iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx

@always_inline
def _transmittance(
    sd:      SceneDescriptor2_C,
    med_idx: Int32,
    dist:    Float32,
) -> SIMD[DType.float32, 3]:
    """Beer-Lambert transmittance through a homogeneous medium over distance."""
    if Int(med_idx) < 0 or sd.mediumCount == Int64(0):
        return SIMD[DType.float32, 3](Float32(1), Float32(1), Float32(1))
    var med = sd.mediums[Int(med_idx)]
    var tr = exp(-(med.sigma_a.r + med.sigma_s.r) * dist)
    var tg = exp(-(med.sigma_a.g + med.sigma_s.g) * dist)
    var tb = exp(-(med.sigma_a.b + med.sigma_s.b) * dist)
    return SIMD[DType.float32, 3](tr, tg, tb)

# ── Visibility with transmittance ─────────────────────────────────────────────

def _visible_transmittance(
    ax: Float32, ay: Float32, az: Float32,
    bx: Float32, by: Float32, bz: Float32,
    med_idx: Int32,
    sd:      SceneDescriptor2_C,
) -> SIMD[DType.float32, 3]:
    """Returns transmittance along segment AB, or (0,0,0) if occluded.
    Glass (dielectric) surfaces are passed through with Fresnel transmittance."""
    var dx = bx - ax; var dy = by - ay; var dz = bz - az
    var dist_total = sqrt(dx*dx + dy*dy + dz*dz)
    if dist_total < Float32(1e-5):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var inv = Float32(1) / dist_total
    var dir = SIMD[DType.float32, 3](dx*inv, dy*inv, dz*inv)

    var Tr_r = Float32(1); var Tr_g = Float32(1); var Tr_b = Float32(1)
    var ox = ax + dir[0]*Float32(0.0002)
    var oy = ay + dir[1]*Float32(0.0002)
    var oz = az + dir[2]*Float32(0.0002)
    var remaining = dist_total - Float32(0.0002)
    var cur_med = med_idx

    var inter_mem = alloc[Intersection_C](1)
    for _ in range(8):
        if remaining < Float32(1e-4): break
        var ray = Ray_C(Point3f(ox, oy, oz), Vec3f(dir[0], dir[1], dir[2]))
        inter_mem[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, remaining * Float32(0.9995), inter_mem)
        if inter_mem[0].hit == Int8(0):
            # Nothing between here and destination: apply remaining Beer-Lambert
            if Int(cur_med) >= 0:
                var med = sd.mediums[Int(cur_med)]
                Tr_r *= exp(-(med.sigma_s.r+med.sigma_a.r)*remaining)
                Tr_g *= exp(-(med.sigma_s.g+med.sigma_a.g)*remaining)
                Tr_b *= exp(-(med.sigma_s.b+med.sigma_a.b)*remaining)
            break

        var inter = inter_mem[0]
        var t_hit = inter.tHit
        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hx = ox + dir[0]*t_hit; var hy = oy + dir[1]*t_hit; var hz = oz + dir[2]*t_hit

        # Beer-Lambert through medium segment up to hit
        if Int(cur_med) >= 0:
            var med = sd.mediums[Int(cur_med)]
            Tr_r *= exp(-(med.sigma_s.r+med.sigma_a.r)*t_hit)
            Tr_g *= exp(-(med.sigma_s.g+med.sigma_a.g)*t_hit)
            Tr_b *= exp(-(med.sigma_s.b+med.sigma_a.b)*t_hit)

        if mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            # Pass through glass with Fresnel transmittance
            var gn: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si = Int(inter.primId.id1)
                var sph = sd.spheres[si]
                var snx = hx-sph.center.x; var sny = hy-sph.center.y; var snz = hz-sph.center.z
                var snl = sqrt(snx*snx+sny*sny+snz*snz)
                if snl > Float32(0): snx/=snl; sny/=snl; snz/=snl
                gn = SIMD[DType.float32, 3](snx, sny, snz)
            else:
                gn = _geom_normal(inter, sd.meshes)
            var facing = dot(dir, gn) < Float32(0)
            var n_for_cos = gn if facing else gn*Float32(-1)
            var cos_i = -dot(dir, n_for_cos)
            if cos_i < Float32(0): cos_i = -cos_i
            var ior = mat.albedo.r
            var fr = fr_dielectric(cos_i, Float32(1)/ior if facing else ior)
            var T = Float32(1) - fr
            Tr_r *= T; Tr_g *= T; Tr_b *= T
            if Tr_r < Float32(1e-7) and Tr_g < Float32(1e-7) and Tr_b < Float32(1e-7):
                inter_mem.free()
                return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
            # Update medium after crossing glass surface
            if mat.medium_interface_idx >= Int32(0) and sd.mediumIfaceCount > Int64(0):
                var iface = sd.mediumInterfaces[Int(mat.medium_interface_idx)]
                var md = dir[0]*gn[0]+dir[1]*gn[1]+dir[2]*gn[2]
                cur_med = iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx
            ox = hx+dir[0]*Float32(0.0002); oy = hy+dir[1]*Float32(0.0002); oz = hz+dir[2]*Float32(0.0002)
            remaining = remaining - t_hit - Float32(0.0002)

        elif mat.type == MatKind.interface:
            # Pure medium boundary: update medium, continue
            if mat.medium_interface_idx >= Int32(0) and sd.mediumIfaceCount > Int64(0):
                var iface = sd.mediumInterfaces[Int(mat.medium_interface_idx)]
                var igna = _geom_normal(inter, sd.meshes)
                var md = dir[0]*igna[0]+dir[1]*igna[1]+dir[2]*igna[2]
                cur_med = iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx
            ox = hx+dir[0]*Float32(0.0002); oy = hy+dir[1]*Float32(0.0002); oz = hz+dir[2]*Float32(0.0002)
            remaining = remaining - t_hit - Float32(0.0002)

        else:
            # Opaque surface blocks the segment
            inter_mem.free()
            return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))

    inter_mem.free()
    return SIMD[DType.float32, 3](Tr_r, Tr_g, Tr_b)

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
    var rdx = c2w[0]*cx + c2w[4]*cy + c2w[8]*cz
    var rdy = c2w[1]*cx + c2w[5]*cy + c2w[9]*cz
    var rdz = c2w[2]*cx + c2w[6]*cy + c2w[10]*cz
    var dl = sqrt(rdx*rdx + rdy*rdy + rdz*rdz)
    if dl > Float32(0.0): rdx /= dl; rdy /= dl; rdz /= dl
    var rox = c2w[12]; var roy = c2w[13]; var roz = c2w[14]

    var n_verts = 0
    var n_bounces = 0  # total surface hits including glass (for _dielectric_bounce entering logic)
    var beta_r = Float32(1); var beta_g = Float32(1); var beta_b = Float32(1)
    var cur_med_idx = start_med_idx

    for _ in range(_BDPT_MAX_DEPTH):
        if n_verts >= _BDPT_MAX_VERTS: break
        var ray = Ray_C(Point3f(rox, roy, roz), Vec3f(rdx, rdy, rdz))
        inter_mem[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), inter_mem)
        if inter_mem[0].hit == Int8(0): break

        var inter = inter_mem[0]
        var t_hit = inter.tHit
        var ray_dir = SIMD[DType.float32, 3](rdx, rdy, rdz)

        # Volume free-flight
        if has_med and Int(cur_med_idx) >= 0:
            var med = sd.mediums[Int(cur_med_idx)]
            var sig_t = med.sigma_s.r + med.sigma_a.r
            if sig_t > Float32(0):
                var u = pcg.next_float()
                var t_free = -log(max(u, Float32(1e-7))) / sig_t
                if t_free < t_hit:
                    # Volume scatter vertex
                    var sx = rox + rdx*t_free
                    var sy = roy + rdy*t_free
                    var sz = roz + rdz*t_free
                    var alb_s = med.sigma_s.r / sig_t
                    var alb_g_s = med.sigma_s.g / (med.sigma_a.g + med.sigma_s.g) if (med.sigma_a.g + med.sigma_s.g) > Float32(0) else alb_s
                    var alb_b_s = med.sigma_s.b / (med.sigma_a.b + med.sigma_s.b) if (med.sigma_a.b + med.sigma_s.b) > Float32(0) else alb_s
                    # BDPT vertex beta: Tr/pdf_free × phase/pdf_phase = 1/sig_t × sig_s = alb_s
                    # (exp(-sig_t×t) cancels between Tr numerator and pdf denominator)
                    var v = _null_vertex()
                    v.px = sx; v.py = sy; v.pz = sz
                    v.beta_r = beta_r * alb_s; v.beta_g = beta_g * alb_g_s; v.beta_b = beta_b * alb_b_s
                    v.alb_r = alb_s; v.alb_g = alb_g_s; v.alb_b = alb_b_s
                    v.is_surface = Int32(0); v.is_delta = Int32(0)
                    v.pdf_fwd = sig_t * exp(-sig_t * t_free)
                    v.med_idx = cur_med_idx
                    verts[n_verts] = v; n_verts += 1
                    # Continuation beta = prev × alb_s (same as stored vertex beta)
                    beta_r *= alb_s; beta_g *= alb_g_s; beta_b *= alb_b_s
                    var u1 = pcg.next_float(); var u2 = pcg.next_float()
                    var cosT = Float32(2)*u1 - Float32(1)
                    var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                    var phi  = Float32(2)*PI*u2
                    rdx = sinT*cos(phi); rdy = sinT*sin(phi); rdz = cosT
                    rox = sx + rdx*Float32(0.0002)
                    roy = sy + rdy*Float32(0.0002)
                    roz = sz + rdz*Float32(0.0002)
                    continue
                else:
                    # Beer-Lambert through full segment to surface
                    beta_r *= exp(-(med.sigma_a.r + med.sigma_s.r) * t_hit)
                    beta_g *= exp(-(med.sigma_a.g + med.sigma_s.g) * t_hit)
                    beta_b *= exp(-(med.sigma_a.b + med.sigma_s.b) * t_hit)

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hx = rox + rdx*t_hit; var hy = roy + rdy*t_hit; var hz = roz + rdz*t_hit

        if mat.type == MatKind.area_light:
            break  # camera hits light directly → handled by unidirectional contribution

        elif mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
            var gn = _geom_normal(inter, sd.meshes)
            if dot(gn, ray_dir) > Float32(0): gn = gn * Float32(-1)
            var v = _null_vertex()
            v.px = hx; v.py = hy; v.pz = hz
            v.nx = gn[0]; v.ny = gn[1]; v.nz = gn[2]
            v.beta_r = beta_r; v.beta_g = beta_g; v.beta_b = beta_b
            v.alb_r = mat.albedo.r; v.alb_g = mat.albedo.g; v.alb_b = mat.albedo.b
            v.is_surface = Int32(1); v.is_delta = Int32(0)
            v.pdf_fwd = Float32(1)  # placeholder, set by MIS
            v.med_idx = cur_med_idx
            verts[n_verts] = v; n_verts += 1
            # Cosine-weighted scatter direction
            var u1 = pcg.next_float(); var u2 = pcg.next_float()
            var r_samp = sqrt(u1); var phi = Float32(2)*PI*u2
            var lx = r_samp*cos(phi); var lz_loc = r_samp*sin(phi)
            var ly = sqrt(max(Float32(0), Float32(1)-u1))
            var gn_n = gn
            var sgn = Float32(1) if gn_n[2] >= Float32(0) else Float32(-1)
            var a_tf = Float32(-1) / (sgn + gn_n[2])
            var b_tf = gn_n[0]*gn_n[1]*a_tf
            var tx = Float32(1)+sgn*gn_n[0]*gn_n[0]*a_tf; var ty = sgn*b_tf; var tz = -sgn*gn_n[0]
            var bx_v = b_tf; var by_v = sgn+gn_n[1]*gn_n[1]*a_tf; var bz_v = -gn_n[1]
            rdx = tx*lx + bx_v*lz_loc + gn_n[0]*ly
            rdy = ty*lx + by_v*lz_loc + gn_n[1]*ly
            rdz = tz*lx + bz_v*lz_loc + gn_n[2]*ly
            var nd = sqrt(rdx*rdx+rdy*rdy+rdz*rdz)
            if nd > Float32(0): rdx /= nd; rdy /= nd; rdz /= nd
            rox = hx + rdx*Float32(0.0002); roy = hy + rdy*Float32(0.0002); roz = hz + rdz*Float32(0.0002)
            # Update beta: f/pdf for Lambertian = (alb/π) / (cosθ/π) = alb
            beta_r *= mat.albedo.r; beta_g *= mat.albedo.g; beta_b *= mat.albedo.b

        elif mat.type == MatKind.conductor:
            var gn_c: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si_c = Int(inter.primId.id1)
                var sph_c = sd.spheres[si_c]
                var cnx = hx-sph_c.center.x; var cny = hy-sph_c.center.y; var cnz = hz-sph_c.center.z
                var cnl = sqrt(cnx*cnx+cny*cny+cnz*cnz)
                if cnl > Float32(0): cnx /= cnl; cny /= cnl; cnz /= cnl
                gn_c = SIMD[DType.float32, 3](cnx, cny, cnz)
            else:
                gn_c = _geom_normal(inter, sd.meshes)
            if dot(gn_c, ray_dir) > Float32(0): gn_c = gn_c * Float32(-1)
            var wo_c = SIMD[DType.float32, 3](-rdx, -rdy, -rdz)
            var frm_c = Frame.from_z(Vec3f(gn_c[0], gn_c[1], gn_c[2]))
            var gc_c = GeomContext(
                normal=gn_c, geo_normal=gn_c, hit_point=SIMD[DType.float32, 3](hx, hy, hz), wo=wo_c,
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
                v.px = hx; v.py = hy; v.pz = hz
                v.nx = gn_c[0]; v.ny = gn_c[1]; v.nz = gn_c[2]
                v.beta_r = beta_r; v.beta_g = beta_g; v.beta_b = beta_b
                v.alb_r = mat.albedo.r; v.alb_g = mat.albedo.g; v.alb_b = mat.albedo.b
                v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = Int32(1)
                v.pdf_bwd = max(mat.roughU, mat.roughV)
                v.wx = wo_c[0]; v.wy = wo_c[1]; v.wz = wo_c[2]
                v.pdf_fwd = Float32(1)
                v.med_idx = cur_med_idx
                verts[n_verts] = v; n_verts += 1
            beta_r *= bs_c.f.r; beta_g *= bs_c.f.g; beta_b *= bs_c.f.b
            rdx = bs_c.wi[0]; rdy = bs_c.wi[1]; rdz = bs_c.wi[2]
            rox = hx + rdx*Float32(0.0002); roy = hy + rdy*Float32(0.0002); roz = hz + rdz*Float32(0.0002)

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            var gn: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si = Int(inter.primId.id1)
                var sph = sd.spheres[si]
                var snx = hx-sph.center.x; var sny = hy-sph.center.y; var snz = hz-sph.center.z
                var snl = sqrt(snx*snx+sny*sny+snz*snz)
                if snl > Float32(0): snx /= snl; sny /= snl; snz /= snl
                gn = SIMD[DType.float32, 3](snx, sny, snz)
            else:
                gn = _geom_normal(inter, sd.meshes)
            var hit = SIMD[DType.float32, 3](hx, hy, hz)
            var (new_dir, new_org) = _dielectric_bounce(ray_dir, hit, gn, mat.albedo.r, n_bounces, pcg)
            n_bounces += 1
            # Specular vertex: no BSDF record needed, just track throughput
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hx, hy, hz)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            rdx = new_dir[0]; rdy = new_dir[1]; rdz = new_dir[2]
            rox = new_org[0]; roy = new_org[1]; roz = new_org[2]

        elif mat.type == MatKind.interface:
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hx, hy, hz)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            rox = hx + rdx*Float32(0.0002); roy = hy + rdy*Float32(0.0002); roz = hz + rdz*Float32(0.0002)

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
    Returns (n_verts, pcg, flux_r, flux_g, flux_b, total_light_pdf)."""
    var n_lights = Int(sd.areaLightCount)
    if n_lights == 0:
        return 0

    # Pick a light uniformly
    var li = Int(pcg.next_uint() % UInt32(n_lights))
    var al = sd.areaLights[li]
    var lmesh = sd.meshes[Int(al.meshIdx)]
    var n_tris = Int(max(Int(al.n_tris), 1))
    var ti = Int(pcg.next_uint() % UInt32(n_tris))
    var lb = ti * 3
    var lv0 = Int(lmesh.vertexIndices[lb]); var lv1 = Int(lmesh.vertexIndices[lb+1]); var lv2 = Int(lmesh.vertexIndices[lb+2])
    var lp0 = SIMD[DType.float32, 3](lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
    var lp1 = SIMD[DType.float32, 3](lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
    var lp2 = SIMD[DType.float32, 3](lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])
    var ru1 = pcg.next_float(); var ru2 = pcg.next_float(); var sr1 = sqrt(ru1)
    var lp  = lp0*(Float32(1)-sr1) + lp1*(sr1*(Float32(1)-ru2)) + lp2*(sr1*ru2)
    var ln  = cross(lp1-lp0, lp2-lp0)
    var lnl = sqrt(dot(ln,ln))
    if lnl > Float32(0): ln = ln*(Float32(1)/lnl)
    # Use shading normals when provided — they give the correct emission hemisphere
    # (e.g. ceiling lights with winding that produces an upward geometric normal but
    # scene-specified normals pointing downward into the room).
    if Int(lmesh.normals) > 4:
        var sn0x = lmesh.normals[lv0*3]; var sn0y = lmesh.normals[lv0*3+1]; var sn0z = lmesh.normals[lv0*3+2]
        var sn1x = lmesh.normals[lv1*3]; var sn1y = lmesh.normals[lv1*3+1]; var sn1z = lmesh.normals[lv1*3+2]
        var sn2x = lmesh.normals[lv2*3]; var sn2y = lmesh.normals[lv2*3+1]; var sn2z = lmesh.normals[lv2*3+2]
        var snx = (sn0x+sn1x+sn2x)/Float32(3); var sny = (sn0y+sn1y+sn2y)/Float32(3); var snz = (sn0z+sn1z+sn2z)/Float32(3)
        var snl = sqrt(snx*snx+sny*sny+snz*snz)
        if snl > Float32(0): ln = SIMD[DType.float32, 3](snx/snl, sny/snl, snz/snl)

    # Cosine-weighted emission direction
    var du1 = pcg.next_float(); var du2 = pcg.next_float()
    var r_s = sqrt(du1); var theta = Float32(2)*PI*du2
    var lx = r_s*cos(theta); var lz_l = r_s*sin(theta)
    var ly = sqrt(max(Float32(0), Float32(1)-du1))
    var sgn = Float32(1) if ln[2] >= Float32(0) else Float32(-1)
    var a_tf = Float32(-1)/(sgn+ln[2]); var b_tf = ln[0]*ln[1]*a_tf
    var tx = Float32(1)+sgn*ln[0]*ln[0]*a_tf; var ty = sgn*b_tf; var tz = -sgn*ln[0]
    var bx_v = b_tf; var by_v = sgn+ln[1]*ln[1]*a_tf; var bz_v = -ln[1]
    var pdir = SIMD[DType.float32, 3](tx*lx+bx_v*lz_l+ln[0]*ly, ty*lx+by_v*lz_l+ln[1]*ly, tz*lx+bz_v*lz_l+ln[2]*ly)
    var pdl = sqrt(dot(pdir,pdir))
    if pdl > Float32(0): pdir = pdir*(Float32(1)/pdl)

    # Light vertex (the emission point, s=1 strategy connects here directly).
    # beta = 1/p_A = area × n_lights (area sampling PDF correction only).
    # alb  = Le (emission) — used as f_lgt in _connect for the light vertex.
    # G already handles the cos_l factor, so f_lgt = Le (no extra cos multiply).
    var area_weight = al.total_area * Float32(n_lights)
    var lv0_vert = _null_vertex()
    lv0_vert.px = lp[0]; lv0_vert.py = lp[1]; lv0_vert.pz = lp[2]
    lv0_vert.nx = ln[0]; lv0_vert.ny = ln[1]; lv0_vert.nz = ln[2]
    lv0_vert.beta_r = area_weight; lv0_vert.beta_g = area_weight; lv0_vert.beta_b = area_weight
    lv0_vert.alb_r = al.emission.r; lv0_vert.alb_g = al.emission.g; lv0_vert.alb_b = al.emission.b
    lv0_vert.is_surface = Int32(1); lv0_vert.is_light = Int32(1)
    lv0_vert.pdf_fwd = Float32(1) / area_weight
    lv0_vert.med_idx = default_emit_med
    verts[0] = lv0_vert

    # For traced vertices: beta = Le / (p_A × p_ω) where p_ω = cos_θ/π for cosine-weighted emission.
    # cos_θ_emitted = ly (the cosine of the sampled direction against the light normal).
    # β = Le × area × n_lights × π / cos_θ_emitted
    var cos_theta_emit = max(ly, Float32(0.01))
    var flux_r = al.emission.r * area_weight * PI / cos_theta_emit
    var flux_g = al.emission.g * area_weight * PI / cos_theta_emit
    var flux_b = al.emission.b * area_weight * PI / cos_theta_emit
    var rox = lp[0]+ln[0]*Float32(0.0001); var roy = lp[1]+ln[1]*Float32(0.0001); var roz = lp[2]+ln[2]*Float32(0.0001)
    var rdx = pdir[0]; var rdy = pdir[1]; var rdz = pdir[2]
    var cur_med_idx = default_emit_med
    var n_verts = 1  # vertex 0 is the light point itself
    var n_lbounces = 1  # counts all surface hits (1 = after initial light vertex)

    var inter_mem = alloc[Intersection_C](1)
    for _ in range(_BDPT_MAX_DEPTH):
        if n_verts >= _BDPT_MAX_VERTS: break
        var ray = Ray_C(Point3f(rox, roy, roz), Vec3f(rdx, rdy, rdz))
        inter_mem[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), inter_mem)
        if inter_mem[0].hit == Int8(0): break

        var inter = inter_mem[0]
        var t_hit = inter.tHit
        var ray_dir = SIMD[DType.float32, 3](rdx, rdy, rdz)

        # Volume free-flight
        if has_med and Int(cur_med_idx) >= 0:
            var med = sd.mediums[Int(cur_med_idx)]
            var sig_t = med.sigma_s.r + med.sigma_a.r
            if sig_t > Float32(0):
                var u = pcg.next_float()
                var t_free = -log(max(u, Float32(1e-7))) / sig_t
                if t_free < t_hit:
                    var sx = rox+rdx*t_free; var sy = roy+rdy*t_free; var sz = roz+rdz*t_free
                    var alb_s = med.sigma_s.r / sig_t
                    var alb_g_s = med.sigma_s.g / (med.sigma_a.g + med.sigma_s.g) if (med.sigma_a.g + med.sigma_s.g) > Float32(0) else alb_s
                    var alb_b_s = med.sigma_s.b / (med.sigma_a.b + med.sigma_s.b) if (med.sigma_a.b + med.sigma_s.b) > Float32(0) else alb_s
                    # BDPT vertex beta: exp(-sig_t×t)/pdf_free × alb_s = 1/sig_t × alb_s = alb_s (for sig_t=1)
                    var v = _null_vertex()
                    v.px = sx; v.py = sy; v.pz = sz
                    v.beta_r = flux_r * alb_s; v.beta_g = flux_g * alb_g_s; v.beta_b = flux_b * alb_b_s
                    v.alb_r = alb_s; v.alb_g = alb_g_s; v.alb_b = alb_b_s
                    v.is_surface = Int32(0); v.is_delta = Int32(0)
                    v.pdf_fwd = sig_t * exp(-sig_t * t_free)
                    v.med_idx = cur_med_idx
                    verts[n_verts] = v; n_verts += 1
                    # Continuation: flux = prev × alb_s (same as stored vertex beta)
                    flux_r *= alb_s; flux_g *= alb_g_s; flux_b *= alb_b_s
                    var u1 = pcg.next_float(); var u2 = pcg.next_float()
                    var cosT = Float32(2)*u1 - Float32(1)
                    var sinT = sqrt(max(Float32(0), Float32(1)-cosT*cosT))
                    var phi  = Float32(2)*PI*u2
                    rdx = sinT*cos(phi); rdy = sinT*sin(phi); rdz = cosT
                    rox = sx+rdx*Float32(0.0002); roy = sy+rdy*Float32(0.0002); roz = sz+rdz*Float32(0.0002)
                    continue
                else:
                    flux_r *= exp(-(med.sigma_a.r+med.sigma_s.r)*t_hit)
                    flux_g *= exp(-(med.sigma_a.g+med.sigma_s.g)*t_hit)
                    flux_b *= exp(-(med.sigma_a.b+med.sigma_s.b)*t_hit)

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hx = rox+rdx*t_hit; var hy = roy+rdy*t_hit; var hz = roz+rdz*t_hit

        if mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
            var gn = _geom_normal(inter, sd.meshes)
            if dot(gn, ray_dir) > Float32(0): gn = gn * Float32(-1)
            var v = _null_vertex()
            v.px = hx; v.py = hy; v.pz = hz
            v.nx = gn[0]; v.ny = gn[1]; v.nz = gn[2]
            v.beta_r = flux_r; v.beta_g = flux_g; v.beta_b = flux_b
            v.alb_r = mat.albedo.r; v.alb_g = mat.albedo.g; v.alb_b = mat.albedo.b
            v.is_surface = Int32(1); v.is_delta = Int32(0)
            v.pdf_fwd = Float32(1); v.med_idx = cur_med_idx
            verts[n_verts] = v; n_verts += 1
            # Scatter
            var u1 = pcg.next_float(); var u2 = pcg.next_float()
            var r_s2 = sqrt(u1); var phi = Float32(2)*PI*u2
            var lx2 = r_s2*cos(phi); var lz_l2 = r_s2*sin(phi); var ly2 = sqrt(max(Float32(0),Float32(1)-u1))
            var sgn2 = Float32(1) if gn[2] >= Float32(0) else Float32(-1)
            var a_tf2 = Float32(-1)/(sgn2+gn[2]); var b_tf2 = gn[0]*gn[1]*a_tf2
            var tx2 = Float32(1)+sgn2*gn[0]*gn[0]*a_tf2; var ty2 = sgn2*b_tf2; var tz2 = -sgn2*gn[0]
            var bx_v2 = b_tf2; var by_v2 = sgn2+gn[1]*gn[1]*a_tf2; var bz_v2 = -gn[1]
            rdx = tx2*lx2+bx_v2*lz_l2+gn[0]*ly2
            rdy = ty2*lx2+by_v2*lz_l2+gn[1]*ly2
            rdz = tz2*lx2+bz_v2*lz_l2+gn[2]*ly2
            var nd = sqrt(rdx*rdx+rdy*rdy+rdz*rdz)
            if nd > Float32(0):
                rdx /= nd; rdy /= nd; rdz /= nd
            rox = hx+rdx*Float32(0.0002); roy = hy+rdy*Float32(0.0002); roz = hz+rdz*Float32(0.0002)
            flux_r *= mat.albedo.r; flux_g *= mat.albedo.g; flux_b *= mat.albedo.b

        elif mat.type == MatKind.conductor:
            var gn_c: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si_c = Int(inter.primId.id1)
                var sph_c = sd.spheres[si_c]
                var cnx = hx-sph_c.center.x; var cny = hy-sph_c.center.y; var cnz = hz-sph_c.center.z
                var cnl = sqrt(cnx*cnx+cny*cny+cnz*cnz)
                if cnl > Float32(0): cnx /= cnl; cny /= cnl; cnz /= cnl
                gn_c = SIMD[DType.float32, 3](cnx, cny, cnz)
            else:
                gn_c = _geom_normal(inter, sd.meshes)
            if dot(gn_c, ray_dir) > Float32(0): gn_c = gn_c * Float32(-1)
            var wo_c = SIMD[DType.float32, 3](-rdx, -rdy, -rdz)
            var frm_c = Frame.from_z(Vec3f(gn_c[0], gn_c[1], gn_c[2]))
            var gc_c = GeomContext(
                normal=gn_c, geo_normal=gn_c, hit_point=SIMD[DType.float32, 3](hx, hy, hz), wo=wo_c,
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
                v.px = hx; v.py = hy; v.pz = hz
                v.nx = gn_c[0]; v.ny = gn_c[1]; v.nz = gn_c[2]
                v.beta_r = flux_r; v.beta_g = flux_g; v.beta_b = flux_b
                v.alb_r = mat.albedo.r; v.alb_g = mat.albedo.g; v.alb_b = mat.albedo.b
                v.is_surface = Int32(1); v.is_delta = Int32(0); v.mat_kind = Int32(1)
                v.pdf_bwd = max(mat.roughU, mat.roughV)
                v.wx = wo_c[0]; v.wy = wo_c[1]; v.wz = wo_c[2]
                v.pdf_fwd = Float32(1)
                v.med_idx = cur_med_idx
                verts[n_verts] = v; n_verts += 1
            flux_r *= bs_c.f.r; flux_g *= bs_c.f.g; flux_b *= bs_c.f.b
            rdx = bs_c.wi[0]; rdy = bs_c.wi[1]; rdz = bs_c.wi[2]
            rox = hx + rdx*Float32(0.0002); roy = hy + rdy*Float32(0.0002); roz = hz + rdz*Float32(0.0002)

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            var gn: SIMD[DType.float32, 3]
            if inter.primId.type == Int8(4):
                var si = Int(inter.primId.id1)
                var sph = sd.spheres[si]
                var snx = hx-sph.center.x; var sny = hy-sph.center.y; var snz = hz-sph.center.z
                var snl = sqrt(snx*snx+sny*sny+snz*snz)
                if snl > Float32(0): snx /= snl; sny /= snl; snz /= snl
                gn = SIMD[DType.float32, 3](snx, sny, snz)
            else:
                gn = _geom_normal(inter, sd.meshes)
            var hit = SIMD[DType.float32, 3](hx, hy, hz)
            var (new_dir, new_org) = _dielectric_bounce(ray_dir, hit, gn, mat.albedo.r, n_lbounces, pcg)
            n_lbounces += 1
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hx, hy, hz)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            rdx = new_dir[0]; rdy = new_dir[1]; rdz = new_dir[2]
            rox = new_org[0]; roy = new_org[1]; roz = new_org[2]

        elif mat.type == MatKind.interface:
            if has_med:
                var new_idx = _bdpt_medium_update(ray_dir, inter, mat, sd, hx, hy, hz)
                if mat.medium_interface_idx >= Int32(0): cur_med_idx = new_idx
            rox = hx+rdx*Float32(0.0002); roy = hy+rdy*Float32(0.0002); roz = hz+rdz*Float32(0.0002)

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
    f0_r: Float32, f0_g: Float32, f0_b: Float32,
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
    var fr_r = f0_r + (Float32(1) - f0_r) * schlick
    var fr_g = f0_g + (Float32(1) - f0_g) * schlick
    var fr_b = f0_b + (Float32(1) - f0_b) * schlick
    var k = d * g / (Float32(4) * cos_o * cos_i) * cos_i
    return SIMD[DType.float32, 3](k * fr_r, k * fr_g, k * fr_b)

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
        return SIMD[DType.float32, 3](v.alb_r*INV_FOUR_PI, v.alb_g*INV_FOUR_PI, v.alb_b*INV_FOUR_PI)
    var vn = SIMD[DType.float32, 3](v.nx, v.ny, v.nz)
    if v.mat_kind == Int32(1):
        var vwo = SIMD[DType.float32, 3](v.wx, v.wy, v.wz)
        return _eval_conductor_ggx(vn, vwo, dir_to_other, v.pdf_bwd, v.alb_r, v.alb_g, v.alb_b)
    # Surface: Lambertian f = alb/π × |cos(wo,n)|
    var cos_o = dot(dir_to_other, vn)
    if cos_o < Float32(0): cos_o = -cos_o
    return SIMD[DType.float32, 3](v.alb_r*INV_PI*cos_o, v.alb_g*INV_PI*cos_o, v.alb_b*INV_PI*cos_o)

# ── Geometry term ─────────────────────────────────────────────────────────────

@always_inline
def _geom_term(
    a: BDPTVertex, b: BDPTVertex,
) -> Float32:
    """Geometry factor G(a,b) = |cos_a| × |cos_b| / dist²."""
    var dx = b.px-a.px; var dy = b.py-a.py; var dz = b.pz-a.pz
    var dist2 = dx*dx+dy*dy+dz*dz
    if dist2 < Float32(1e-8): return Float32(0)
    var d = sqrt(dist2)
    var dir = SIMD[DType.float32, 3](dx/d, dy/d, dz/d)
    var cos_a: Float32
    if a.is_surface == Int32(1):
        var an = SIMD[DType.float32, 3](a.nx, a.ny, a.nz)
        cos_a = dot(dir, an)
        if cos_a < Float32(0): cos_a = -cos_a
    else:
        cos_a = Float32(1)   # volume: no cosine
    var cos_b: Float32
    if b.is_surface == Int32(1):
        var bn = SIMD[DType.float32, 3](b.nx, b.ny, b.nz)
        cos_b = dot(dir, bn)
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

    var dx = lv.px-cv.px; var dy = lv.py-cv.py; var dz = lv.pz-cv.pz
    var dist2 = dx*dx+dy*dy+dz*dz
    if dist2 < Float32(1e-8):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
    var dist = sqrt(dist2)

    # Determine medium for the shadow segment.
    # Use camera vertex's medium (both should agree in a well-defined scene).
    var seg_med = cv.med_idx
    var Tr = _visible_transmittance(cv.px, cv.py, cv.pz, lv.px, lv.py, lv.pz, seg_med, sd)
    if Tr[0] < Float32(1e-7) and Tr[1] < Float32(1e-7) and Tr[2] < Float32(1e-7):
        return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))

    var dir = SIMD[DType.float32, 3](dx/dist, dy/dist, dz/dist)
    var neg_dir = SIMD[DType.float32, 3](-dir[0], -dir[1], -dir[2])

    # BSDF at camera vertex (toward light)
    var f_cam = _eval_vertex(cv, dir)
    # BSDF at light vertex (toward camera)
    var f_lgt: SIMD[DType.float32, 3]
    if lv.is_light == Int32(1):
        # Light emission: f_lgt = Le (emission radiance, no cosine here).
        # _geom_term already computes cos_l at the light surface, so don't multiply again.
        var ln = SIMD[DType.float32, 3](lv.nx, lv.ny, lv.nz)
        var cos_l = dot(neg_dir, ln)  # emission normal vs direction toward cv
        if cos_l <= Float32(0):
            return SIMD[DType.float32, 3](Float32(0), Float32(0), Float32(0))
        f_lgt = SIMD[DType.float32, 3](lv.alb_r, lv.alb_g, lv.alb_b)
    else:
        f_lgt = _eval_vertex(lv, neg_dir)

    # Geometry term G = |cos_cv| × |cos_lv| / dist²
    var G = _geom_term(cv, lv)

    var contrib = f_cam * f_lgt * SIMD[DType.float32, 3](G, G, G) * Tr
    contrib[0] *= cv.beta_r * lv.beta_r
    contrib[1] *= cv.beta_g * lv.beta_g
    contrib[2] *= cv.beta_b * lv.beta_b
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

    # Output buffer: R, G, B per pixel
    var buf_r = alloc[Float32](n_pix)
    var buf_g = alloc[Float32](n_pix)
    var buf_b = alloc[Float32](n_pix)
    for i in range(n_pix):
        buf_r[i] = Float32(0); buf_g[i] = Float32(0); buf_b[i] = Float32(0)

    var cam_verts   = alloc[BDPTVertex](_BDPT_MAX_VERTS)
    var light_verts = alloc[BDPTVertex](_BDPT_MAX_VERTS)

    var r2c = psc[0].raster_to_camera
    var c2w = psc[0].camera_to_world
    var base_seed = psc[0].rng_seed

    for pix in range(n_pix):
        var px = pix % fw; var py = pix // fw
        var acc_r = Float32(0); var acc_g = Float32(0); var acc_b = Float32(0)

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

            var sum_r = Float32(0); var sum_g = Float32(0); var sum_b = Float32(0)
            for ci in range(n_cam):
                if cam_verts[ci].is_delta != Int32(0): continue
                for li in range(n_light):
                    if light_verts[li].is_delta != Int32(0): continue
                    var k = ci + li
                    var ns = Float32(strat_count[k]) if k < 20 else Float32(1)
                    if ns < Float32(1): ns = Float32(1)
                    var c = _connect(cam_verts[ci], light_verts[li], sd, has_med)
                    sum_r += c[0] / ns; sum_g += c[1] / ns; sum_b += c[2] / ns

            acc_r += sum_r
            acc_g += sum_g
            acc_b += sum_b

        var inv_spp = iso_scale / Float32(n_spp)
        buf_r[pix] = acc_r * inv_spp
        buf_g[pix] = acc_g * inv_spp
        buf_b[pix] = acc_b * inv_spp

        if verbose and pix % (n_pix // 10 + 1) == 0:
            print("BDPT: " + String(pix * 100 // n_pix) + "%")

    cam_verts.free(); light_verts.free()

    # Clamp and write image
    var pixels = alloc[Float32](n_pix * 3)
    for i in range(n_pix):
        var r = buf_r[i]; var g = buf_g[i]; var b = buf_b[i]
        if max_comp > Float32(0):
            r = r if r < max_comp else max_comp
            g = g if g < max_comp else max_comp
            b = b if b < max_comp else max_comp
        pixels[i*3]   = r
        pixels[i*3+1] = g
        pixels[i*3+2] = b
    buf_r.free(); buf_g.free(); buf_b.free()

    _ = write_image(pixels, psc[0].film_w, psc[0].film_h, psc[0].film_filename, Int32(32), Int32(32))
    pixels.free()
    return Int32(0)
