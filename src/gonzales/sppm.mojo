# Stochastic Progressive Photon Mapping (CPU, single-threaded)
# Reference: Hachisuka et al. 2008 "Progressive Photon Mapping"

from std.math import sqrt, cos, sin, floor, log, exp, max
from std.memory import alloc
from .geometry import (
    RGB, SampledSpectrum, Point3f, Vec3f, Ray_C, Intersection_C,
    TriangleMesh_C, Material_C, MatKind, AreaLight_C, Medium_C, MediumInterface_C,
    Instance_C, dot, cross, fr_dielectric, PI, INV_FOUR_PI,
)
from .bvh import BVH2Node, SceneDescriptor2_C, traverse_bvh2_core
from .transform import transform_normal_by_instance
from .rng import PCG32
from .pbrt_parser import ParsedScene_Mojo
from .postprocess import write_image

comptime _ALPHA  = Float32(0.7)
comptime _MAX_B  = 10
comptime _HSIZE  = 1048576   # 2^20 hash buckets


# ── Data structures ───────────────────────────────────────────────────────────

@fieldwise_init
struct SPPMPixel(TrivialRegisterPassable):
    """Visible point from one camera ray + SPPM accumulators."""
    var pos_x: Float32; var pos_y: Float32; var pos_z: Float32
    var nx: Float32;    var ny: Float32;    var nz: Float32
    var beta_r: Float32; var beta_g: Float32; var beta_b: Float32  # camera throughput
    var alb_r:  Float32; var alb_g:  Float32; var alb_b:  Float32  # surface albedo
    var tau_r:  Float32; var tau_g:  Float32; var tau_b:  Float32  # accumulated flux
    var N_acc:  Float32   # photon count (alpha-weighted sum)
    var r2:     Float32   # current search radius²
    var valid:  Int32     # 1 = has VP
    var pidx:   Int32     # flat pixel index
    var is_volume: Int32  # 1 = volume scatter VP (isotropic phase fn); 0 = surface

@fieldwise_init
struct SPPMPhoton(TrivialRegisterPassable):
    """Photon stored at a scatter event (surface diffuse or volume)."""
    var px: Float32; var py: Float32; var pz: Float32
    var fr: Float32; var fg: Float32; var fb: Float32  # flux at stored position
    var nxt:       Int32  # chained-list link in hash grid (-1 = end)
    var is_volume: Int32  # 1 = volume scatter photon; 0 = surface diffuse


# ── Geometry helpers ──────────────────────────────────────────────────────────

@always_inline
def _geom_normal(
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin] = UnsafePointer[Instance_C, MutAnyOrigin].unsafe_dangling(),
) -> SIMD[DType.float32, 3]:
    """Normalized geometric normal from triangle cross product. If this hit
    came from inside an instanced BLAS (primId.instanceIdx >= 0 — see
    bvh.mojo's traverse_bvh2_core type==6 branch), the mesh data is in that
    instance's object space, so the normal is transformed to world space
    before returning (the hit *point*, elsewhere computed as
    ray_org + ray_dir*tHit, needs no such fixup — see transform.mojo's
    transform_normal_by_instance for why)."""
    var mi: Int; var bv: Int
    if inter.primId.type == 0:
        mi = Int(inter.primId.id1); bv = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mi = Int(inter.primId.id2 >> 32); bv = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    else:
        return SIMD[DType.float32, 3](Float32(0), Float32(1), Float32(0))
    var m = meshes[mi]
    var v0 = Int(m.vertexIndices[bv])
    var v1 = Int(m.vertexIndices[bv + 1])
    var v2 = Int(m.vertexIndices[bv + 2])
    var p0 = SIMD[DType.float32, 3](m.points[v0*4], m.points[v0*4+1], m.points[v0*4+2])
    var p1 = SIMD[DType.float32, 3](m.points[v1*4], m.points[v1*4+1], m.points[v1*4+2])
    var p2 = SIMD[DType.float32, 3](m.points[v2*4], m.points[v2*4+1], m.points[v2*4+2])
    var n = cross(p1 - p0, p2 - p0)
    if inter.primId.instanceIdx >= Int32(0):
        n = transform_normal_by_instance(instances[Int(inter.primId.instanceIdx)].worldToObj, n)
    var l = dot(n, n)
    if l > Float32(0.0):
        n = n * (Float32(1.0) / sqrt(l))
    return n

@always_inline
def _hash_cell(ix: Int, iy: Int, iz: Int) -> Int:
    var h = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791
    return (h % _HSIZE + _HSIZE) % _HSIZE


# ── Dielectric bounce helper ──────────────────────────────────────────────────
# Returns (new_dir, new_org) after reflection or refraction. Mutates pcg.

@always_inline
def _dielectric_bounce(
    ray_dir: SIMD[DType.float32, 3],
    hit_point: SIMD[DType.float32, 3],
    geom_normal: SIMD[DType.float32, 3],
    ior: Float32,
    bounce: Int,
    mut pcg: PCG32,
) -> Tuple[SIMD[DType.float32, 3], SIMD[DType.float32, 3]]:
    var facing = dot(ray_dir, geom_normal) < Float32(0.0)
    var entering = facing
    if bounce == 0:
        entering = True  # primary ray always enters (fixes inward-normal meshes)
    var normal = geom_normal if entering else (geom_normal * Float32(-1.0))
    var eta = (Float32(1.0) / ior) if entering else ior   # n_i / n_t
    var cos_i = -dot(ray_dir, normal)
    var sin2_t = eta * eta * (Float32(1.0) - cos_i * cos_i)
    var tir = sin2_t > Float32(1.0)
    var fresnel = fr_dielectric(cos_i, Float32(1.0) / eta)

    if tir or pcg.next_float() < fresnel:
        # Reflect: r = d + 2*cos_i*n
        var refl = ray_dir + normal * (Float32(2.0) * cos_i)
        var rl = dot(refl, refl)
        if rl > Float32(0.0): refl = refl * (Float32(1.0) / sqrt(rl))
        return (refl, hit_point + normal * Float32(0.0001))
    else:
        # Refract: t = eta*d + (eta*cos_i - sqrt(1 - sin2_t))*n
        var cos_t = sqrt(max(Float32(0.0), Float32(1.0) - sin2_t))
        var refr = ray_dir * eta + normal * (eta * cos_i - cos_t)
        var rl = dot(refr, refr)
        if rl > Float32(0.0): refr = refr * (Float32(1.0) / sqrt(rl))
        return (refr, hit_point - normal * Float32(0.0001))


# ── Camera pass ───────────────────────────────────────────────────────────────

@always_inline
def _sppm_update_medium(
    ray_dir: SIMD[DType.float32, 3],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
    sd: SceneDescriptor2_C,
    hx: Float32 = Float32(0), hy: Float32 = Float32(0), hz: Float32 = Float32(0),
) -> Int32:
    """Return new current_medium_idx after crossing a surface with MediumInterface."""
    if mat.medium_interface_idx < Int32(0) or sd.mediumIfaceCount == Int64(0):
        return Int32(-1)  # stays vacuum; caller keeps existing idx if needed
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
        var m = meshes[mi]
        var v0 = Int(m.vertexIndices[bv]); var v1 = Int(m.vertexIndices[bv+1]); var v2 = Int(m.vertexIndices[bv+2])
        var e1x = m.points[v1*4] - m.points[v0*4]; var e1y = m.points[v1*4+1] - m.points[v0*4+1]; var e1z = m.points[v1*4+2] - m.points[v0*4+2]
        var e2x = m.points[v2*4] - m.points[v0*4]; var e2y = m.points[v2*4+1] - m.points[v0*4+1]; var e2z = m.points[v2*4+2] - m.points[v0*4+2]
        nx = e1y*e2z - e1z*e2y; ny = e1z*e2x - e1x*e2z; nz = e1x*e2y - e1y*e2x
    var md = ray_dir[0]*nx + ray_dir[1]*ny + ray_dir[2]*nz
    return iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx

def _sppm_camera_pass(
    vps:      UnsafePointer[SPPMPixel, MutAnyOrigin],
    n_pix:    Int,
    fw:       Int32,
    r2c:      UnsafePointer[Float32, MutAnyOrigin],
    c2w:      UnsafePointer[Float32, MutAnyOrigin],
    sd:       SceneDescriptor2_C,
    init_r2:  Float32,
    seed:     UInt64,
):
    """Trace one primary ray per pixel, record visible points at diffuse/volume hits."""
    var ox = c2w[12]; var oy = c2w[13]; var oz = c2w[14]
    var inter_mem = alloc[Intersection_C](1)
    var has_media = Int(sd.mediumCount) > 0

    for pix in range(n_pix):
        var px = pix % Int(fw)
        var py = pix // Int(fw)

        var vp = SPPMPixel(
            pos_x=Float32(0), pos_y=Float32(0), pos_z=Float32(0),
            nx=Float32(0), ny=Float32(1), nz=Float32(0),
            beta_r=Float32(1), beta_g=Float32(1), beta_b=Float32(1),
            alb_r=Float32(0), alb_g=Float32(0), alb_b=Float32(0),
            tau_r=Float32(0), tau_g=Float32(0), tau_b=Float32(0),
            N_acc=Float32(0), r2=init_r2, valid=Int32(0), pidx=Int32(pix),
            is_volume=Int32(0),
        )

        var pcg = PCG32(seed ^ UInt64(pix * 6364136223846793005 + 1), UInt64(1))

        # Sub-pixel jitter, redrawn every pass (this function is called once per
        # SPPM pass — see sppm_render). Without it, every pass's primary ray is
        # bit-identical, so a dielectric surface's *deterministic* refracted
        # direction (only the reflect-vs-refract choice is stochastic, not the
        # refracted angle itself — see _dielectric_bounce) lands on the exact
        # same floor point every single pass. Jitter is the only thing that
        # actually diversifies which floor point gets sampled pass to pass;
        # re-tracing alone (this function moving inside the pass loop) does not.
        var fX = Float32(px) + pcg.next_float()
        var fY = Float32(py) + pcg.next_float()
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
        var rox = ox; var roy = oy; var roz = oz

        var cur_med_idx = Int32(-1)  # camera starts in vacuum

        for bounce in range(_MAX_B):
            var ray = Ray_C(Point3f(rox, roy, roz), Vec3f(rdx, rdy, rdz))
            inter_mem[0].hit = Int8(0)
            traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1.0e38), inter_mem,
                               sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
            if inter_mem[0].hit == Int8(0):
                break

            var inter = inter_mem[0]
            var ray_dir = SIMD[DType.float32, 3](rdx, rdy, rdz)
            var t_hit = inter.tHit

            # ── Volume free-flight ────────────────────────────────────────────
            if has_media and Int(cur_med_idx) >= 0:
                var med = sd.mediums[Int(cur_med_idx)]
                var sig_t = med.sigma_a.r + med.sigma_s.r
                if sig_t > Float32(0):
                    var t_free = -log(max(pcg.next_float(), Float32(1e-7))) / sig_t
                    if t_free < t_hit:
                        # Volume scatter — store VP here
                        var alb_s = med.sigma_s.r / sig_t  # single-scatter albedo
                        vp.pos_x = rox + rdx * t_free
                        vp.pos_y = roy + rdy * t_free
                        vp.pos_z = roz + rdz * t_free
                        vp.nx = Float32(0); vp.ny = Float32(1); vp.nz = Float32(0)
                        vp.alb_r = alb_s; vp.alb_g = alb_s; vp.alb_b = alb_s
                        vp.is_volume = Int32(1)
                        vp.valid = Int32(1)
                        break
                    else:
                        # Transmittance through full segment to surface
                        vp.beta_r *= exp(-(med.sigma_a.r + med.sigma_s.r) * t_hit)
                        vp.beta_g *= exp(-(med.sigma_a.g + med.sigma_s.g) * t_hit)
                        vp.beta_b *= exp(-(med.sigma_a.b + med.sigma_s.b) * t_hit)

            var mat_idx = Int(inter.primId.materialIndex)
            var mat = sd.materials[mat_idx]
            var hx = rox + rdx * t_hit
            var hy = roy + rdy * t_hit
            var hz = roz + rdz * t_hit

            if mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
                var gn = _geom_normal(inter, sd.meshes, sd.instances)
                if dot(gn, ray_dir) > Float32(0.0):
                    gn = gn * Float32(-1.0)
                vp.pos_x = hx; vp.pos_y = hy; vp.pos_z = hz
                vp.nx = gn[0]; vp.ny = gn[1]; vp.nz = gn[2]
                vp.alb_r = mat.albedo.r; vp.alb_g = mat.albedo.g; vp.alb_b = mat.albedo.b
                vp.is_volume = Int32(0)
                vp.valid = Int32(1)
                break

            elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
                var ior = mat.albedo.r
                var gn = _geom_normal(inter, sd.meshes, sd.instances)
                var hit = SIMD[DType.float32, 3](hx, hy, hz)
                var (new_dir, new_org) = _dielectric_bounce(ray_dir, hit, gn, ior, bounce, pcg)
                rdx = new_dir[0]; rdy = new_dir[1]; rdz = new_dir[2]
                rox = new_org[0]; roy = new_org[1]; roz = new_org[2]
                if has_media:
                    var new_idx = _sppm_update_medium(ray_dir, inter, sd.meshes, mat, sd)
                    if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                        cur_med_idx = new_idx

            elif mat.type == MatKind.interface:
                # Transparent boundary — update medium, continue ray
                if has_media:
                    var new_idx = _sppm_update_medium(ray_dir, inter, sd.meshes, mat, sd)
                    if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                        cur_med_idx = new_idx
                rox = hx + rdx * Float32(0.0002)
                roy = hy + rdy * Float32(0.0002)
                roz = hz + rdz * Float32(0.0002)

            else:
                break  # area_light, conductor, etc.

        # Preserve the persistent SPPM accumulators (r2/tau/N_acc) — only the
        # visible-point sample itself (position/normal/albedo/beta/valid) is
        # refreshed by this call. See the docstring above: re-tracing the
        # camera path per pass (instead of once for the whole render) is the
        # "stochastic" part of SPPM, and matters a lot here specifically
        # because the camera ray passes through the scene's dielectric water
        # surface — each call makes an independent random reflect-vs-refract
        # choice, landing the visible point at a different spot on the floor
        # each pass instead of freezing on whichever single spot pass 0 hit.
        vp.r2    = vps[pix].r2
        vp.tau_r = vps[pix].tau_r; vp.tau_g = vps[pix].tau_g; vp.tau_b = vps[pix].tau_b
        vp.N_acc = vps[pix].N_acc
        vps[pix] = vp

    inter_mem.free()


# ── Photon pass ───────────────────────────────────────────────────────────────

def _sppm_photon_pass(
    photons:      UnsafePointer[SPPMPhoton, MutAnyOrigin],
    n_emit:       Int,
    sd:           SceneDescriptor2_C,
    seed:         UInt64,
    pass_idx:     Int,
) -> Int:
    """Emit photons from area lights, trace through glass/media, store at diffuse/volume hits.
    Returns actual number of stored photons."""
    var n_lights = Int(sd.areaLightCount)
    if n_lights == 0:
        return 0

    var n_stored = 0
    var inter_mem = alloc[Intersection_C](1)
    var has_media = Int(sd.mediumCount) > 0

    # Determine the "default" starting medium for photons emitted into a medium.
    # Convention: photon is cosine-sampled from ln, so dot(pdir,ln)>0 always.
    # We use the first MediumInterface whose outside_medium_idx is valid.
    var default_emit_med = Int32(-1)
    if has_media and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    for k in range(n_emit):
        var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))

        # Pick a random area light (uniform)
        var li = Int(pcg.next_uint() % UInt32(n_lights))
        var al = sd.areaLights[li]
        var lmesh = sd.meshes[Int(al.meshIdx)]
        var n_tris = Int(max(Int(al.n_tris), 1))

        # Pick a random triangle on the light
        var ti = Int(pcg.next_uint() % UInt32(n_tris))
        var lb = ti * 3
        var lv0 = Int(lmesh.vertexIndices[lb])
        var lv1 = Int(lmesh.vertexIndices[lb + 1])
        var lv2 = Int(lmesh.vertexIndices[lb + 2])
        var lp0 = SIMD[DType.float32, 3](lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
        var lp1 = SIMD[DType.float32, 3](lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
        var lp2 = SIMD[DType.float32, 3](lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])

        # Sample point on triangle
        var ru1 = pcg.next_float()
        var ru2 = pcg.next_float()
        var sr1 = sqrt(ru1)
        var lp = lp0 * (Float32(1.0) - sr1) + lp1 * (sr1 * (Float32(1.0) - ru2)) + lp2 * (sr1 * ru2)

        # Compute light normal (geometric)
        var ln = cross(lp1 - lp0, lp2 - lp0)
        var lnl = dot(ln, ln)
        if lnl > Float32(0.0): ln = ln * (Float32(1.0) / sqrt(lnl))

        # Sample cosine-weighted direction from light hemisphere
        var du1 = pcg.next_float()
        var du2 = pcg.next_float()
        var r_samp = sqrt(du1)
        var theta = Float32(2.0) * PI * du2
        var lx = r_samp * cos(theta)
        var lz_loc = r_samp * sin(theta)
        var ly = sqrt(max(Float32(0.0), Float32(1.0) - du1))
        # Build tangent frame around light normal (Frisvad method)
        var sgn = Float32(1.0) if ln[2] >= Float32(0.0) else Float32(-1.0)
        var a_tf = Float32(-1.0) / (sgn + ln[2])
        var b_tf = ln[0] * ln[1] * a_tf
        var tangent  = SIMD[DType.float32, 3](Float32(1.0) + sgn*ln[0]*ln[0]*a_tf, sgn*b_tf, -sgn*ln[0])
        var bitangent = SIMD[DType.float32, 3](b_tf, sgn + ln[1]*ln[1]*a_tf, -ln[1])
        var pdir = tangent * lx + bitangent * lz_loc + ln * ly
        var pdl = dot(pdir, pdir)
        if pdl > Float32(0.0): pdir = pdir * (Float32(1.0) / sqrt(pdl))

        # Photon flux: total_light_power / n_emit
        # total_power = emission * pi * total_area * n_lights (uniform light selection)
        var scale = PI * al.total_area * Float32(n_lights) / Float32(n_emit)
        var flux_r = al.emission.r * scale
        var flux_g = al.emission.g * scale
        var flux_b = al.emission.b * scale

        var rox = lp[0] + ln[0] * Float32(0.0001)
        var roy = lp[1] + ln[1] * Float32(0.0001)
        var roz = lp[2] + ln[2] * Float32(0.0001)
        var rdx = pdir[0]; var rdy = pdir[1]; var rdz = pdir[2]
        var cur_med_idx = default_emit_med  # start in medium if light is above one

        for bounce in range(_MAX_B):
            var ray = Ray_C(Point3f(rox, roy, roz), Vec3f(rdx, rdy, rdz))
            inter_mem[0].hit = Int8(0)
            traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1.0e38), inter_mem,
                               sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
            if inter_mem[0].hit == Int8(0):
                break  # miss

            var inter = inter_mem[0]
            var ray_dir = SIMD[DType.float32, 3](rdx, rdy, rdz)
            var t_hit = inter.tHit

            # ── Volume free-flight ────────────────────────────────────────────
            if has_media and Int(cur_med_idx) >= 0:
                var med = sd.mediums[Int(cur_med_idx)]
                var sig_t = med.sigma_s.r + med.sigma_a.r
                if sig_t > Float32(0):
                    var t_free = -log(max(pcg.next_float(), Float32(1e-7))) / sig_t
                    if t_free < t_hit:
                        # Volume scatter — store photon and sample new direction
                        var sx = rox + rdx * t_free
                        var sy = roy + rdy * t_free
                        var sz = roz + rdz * t_free
                        if n_stored < n_emit:
                            photons[n_stored] = SPPMPhoton(
                                px=sx, py=sy, pz=sz,
                                fr=flux_r, fg=flux_g, fb=flux_b,
                                nxt=Int32(-1), is_volume=Int32(1),
                            )
                            n_stored += 1
                        # Scatter: isotropic phase function, modulate by albedo
                        var alb_s = med.sigma_s.r / sig_t
                        flux_r *= alb_s; flux_g *= alb_s; flux_b *= alb_s
                        # Sample new isotropic direction (uniform sphere)
                        var usp1 = pcg.next_float()
                        var usp2 = pcg.next_float()
                        var cosT = Float32(2.0) * usp1 - Float32(1.0)
                        var sinT = sqrt(max(Float32(0), Float32(1) - cosT*cosT))
                        var phiS = Float32(2.0) * PI * usp2
                        rdx = sinT * cos(phiS); rdy = sinT * sin(phiS); rdz = cosT
                        rox = sx + rdx * Float32(0.0001)
                        roy = sy + rdy * Float32(0.0001)
                        roz = sz + rdz * Float32(0.0001)
                        continue
                    else:
                        # Apply Beer-Lambert transmittance through segment
                        flux_r *= exp(-(med.sigma_a.r + med.sigma_s.r) * t_hit)
                        flux_g *= exp(-(med.sigma_a.g + med.sigma_s.g) * t_hit)
                        flux_b *= exp(-(med.sigma_a.b + med.sigma_s.b) * t_hit)

            var mat_idx = Int(inter.primId.materialIndex)
            var mat = sd.materials[mat_idx]
            var hx = rox + rdx * t_hit
            var hy = roy + rdy * t_hit
            var hz = roz + rdz * t_hit

            if mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
                if n_stored < n_emit:
                    photons[n_stored] = SPPMPhoton(
                        px=hx, py=hy, pz=hz,
                        fr=flux_r, fg=flux_g, fb=flux_b,
                        nxt=Int32(-1), is_volume=Int32(0),
                    )
                    n_stored += 1
                break

            elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
                var ior = mat.albedo.r
                var gn = _geom_normal(inter, sd.meshes, sd.instances)
                var hit = SIMD[DType.float32, 3](hx, hy, hz)
                var (new_dir, new_org) = _dielectric_bounce(ray_dir, hit, gn, ior, bounce, pcg)
                rdx = new_dir[0]; rdy = new_dir[1]; rdz = new_dir[2]
                rox = new_org[0]; roy = new_org[1]; roz = new_org[2]
                if has_media:
                    var new_idx = _sppm_update_medium(ray_dir, inter, sd.meshes, mat, sd)
                    if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                        cur_med_idx = new_idx

            elif mat.type == MatKind.interface:
                if has_media:
                    var new_idx = _sppm_update_medium(ray_dir, inter, sd.meshes, mat, sd)
                    if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                        cur_med_idx = new_idx
                rox = hx + rdx * Float32(0.0002)
                roy = hy + rdy * Float32(0.0002)
                roz = hz + rdz * Float32(0.0002)

            else:
                # area_light self-hit, conductor, etc.: absorb
                break

    inter_mem.free()
    return n_stored


# ── Hash grid ─────────────────────────────────────────────────────────────────

def _build_grid(
    photons:  UnsafePointer[SPPMPhoton, MutAnyOrigin],
    n_phot:   Int,
    heads:    UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
):
    # Reset all buckets
    for i in range(_HSIZE):
        heads[i] = Int32(-1)
    # Insert photons into chained hash table
    for k in range(n_phot):
        var ix = Int(floor(photons[k].px * inv_cell))
        var iy = Int(floor(photons[k].py * inv_cell))
        var iz = Int(floor(photons[k].pz * inv_cell))
        var h = _hash_cell(ix, iy, iz)
        photons[k].nxt = heads[h]
        heads[h] = Int32(k)


# ── Gather + SPPM update ──────────────────────────────────────────────────────

def _gather_update(
    vps:      UnsafePointer[SPPMPixel, MutAnyOrigin],
    n_pix:    Int,
    photons:  UnsafePointer[SPPMPhoton, MutAnyOrigin],
    heads:    UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
):
    for i in range(n_pix):
        if vps[i].valid == Int32(0):
            continue
        var vp = vps[i]
        var vx = vp.pos_x; var vy = vp.pos_y; var vz = vp.pos_z
        var vnx = vp.nx;    var vny = vp.ny;    var vnz = vp.nz
        var r2 = vp.r2

        # Accumulate contributions from photons in 3x3x3 neighborhood
        var phi_r = Float32(0); var phi_g = Float32(0); var phi_b = Float32(0)
        var M = Float32(0)

        var cix = Int(floor(vx * inv_cell))
        var ciy = Int(floor(vy * inv_cell))
        var ciz = Int(floor(vz * inv_cell))
        for ddx in range(-1, 2):
            for ddy in range(-1, 2):
                for ddz in range(-1, 2):
                    var h = _hash_cell(cix + ddx, ciy + ddy, ciz + ddz)
                    var k = Int(heads[h])
                    while k != -1:
                        var ph = photons[k]
                        var ex = ph.px - vx; var ey = ph.py - vy; var ez = ph.pz - vz
                        var dist2 = ex*ex + ey*ey + ez*ez
                        if dist2 <= r2 and ph.is_volume == vp.is_volume:
                            # Surface VP: Lambertian f=alb/π; Volume VP: isotropic phase f=alb/(4π)
                            var f: Float32
                            if vp.is_volume == Int32(1):
                                f = INV_FOUR_PI
                            else:
                                f = Float32(1.0) / PI
                            phi_r += vp.alb_r * f * ph.fr
                            phi_g += vp.alb_g * f * ph.fg
                            phi_b += vp.alb_b * f * ph.fb
                            M += Float32(1.0)
                        k = Int(ph.nxt)

        # SPPM update (only if new photons found)
        if M > Float32(0.0):
            var N = vp.N_acc
            var ratio = (N + _ALPHA * M) / (N + M)
            vps[i].r2    = r2 * ratio
            vps[i].tau_r = (vp.tau_r + phi_r) * ratio
            vps[i].tau_g = (vp.tau_g + phi_g) * ratio
            vps[i].tau_b = (vp.tau_b + phi_b) * ratio
            vps[i].N_acc = N + _ALPHA * M


# ── Public entry point ────────────────────────────────────────────────────────

def sppm_render(
    psc:      UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
    sd:       SceneDescriptor2_C,
    n_passes: Int,
    n_photons_per_pass: Int,
    initial_radius: Float32,
    no_denoise: Bool,
    verbose:  Bool,
) -> Int32:
    """Stochastic Progressive Photon Mapping main loop."""
    var fw = Int(psc[0].film_w)
    var fh = Int(psc[0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[0].film_iso / Float32(100)
    var max_comp = psc[0].film_max_comp

    if Int(sd.areaLightCount) == 0:
        print("SPPM: no area lights in scene, cannot emit photons")
        return Int32(-1)

    print("SPPM: " + String(fw) + "x" + String(fh)
          + " " + String(n_passes) + " passes x "
          + String(n_photons_per_pass) + " photons  r=" + String(initial_radius))

    # Allocate visible points and photon buffer
    var vps     = alloc[SPPMPixel](n_pix)
    var photons = alloc[SPPMPhoton](n_photons_per_pass)
    var heads   = alloc[Int32](_HSIZE)
    var init_r2 = initial_radius * initial_radius
    var inv_cell = Float32(1.0) / initial_radius  # cell size == initial radius

    # Seed the persistent per-pixel accumulators. The actual visible-point
    # sample (position/normal/albedo/valid) is (re-)traced fresh every pass
    # below — see _sppm_camera_pass's docstring for why re-tracing per pass
    # (not once for the whole render) matters.
    for i in range(n_pix):
        vps[i] = SPPMPixel(
            pos_x=Float32(0), pos_y=Float32(0), pos_z=Float32(0),
            nx=Float32(0), ny=Float32(1), nz=Float32(0),
            beta_r=Float32(1), beta_g=Float32(1), beta_b=Float32(1),
            alb_r=Float32(0), alb_g=Float32(0), alb_b=Float32(0),
            tau_r=Float32(0), tau_g=Float32(0), tau_b=Float32(0),
            N_acc=Float32(0), r2=init_r2, valid=Int32(0), pidx=Int32(i),
            is_volume=Int32(0),
        )

    # Photon passes
    for pass_idx in range(n_passes):
        var cam_seed = psc[0].rng_seed ^ UInt64(pass_idx * 0x9E3779B97F4A7C15 + 7)
        _sppm_camera_pass(
            vps, n_pix, psc[0].film_w,
            psc[0].raster_to_camera, psc[0].camera_to_world,
            sd, init_r2, cam_seed,
        )
        if verbose and pass_idx == 0:
            var n_valid = 0
            for i in range(n_pix):
                if vps[i].valid != Int32(0): n_valid += 1
            print("SPPM: " + String(n_valid) + "/" + String(n_pix) + " visible points found")

        var pass_seed = psc[0].rng_seed ^ UInt64(pass_idx * 2654435761 + 1)
        var n_stored = _sppm_photon_pass(photons, n_photons_per_pass, sd, pass_seed, pass_idx)
        if n_stored > 0:
            _build_grid(photons, n_stored, heads, inv_cell)
            _gather_update(vps, n_pix, photons, heads, inv_cell)
        if verbose or (pass_idx + 1) % 10 == 0:
            print("SPPM: pass " + String(pass_idx + 1) + "/" + String(n_passes)
                  + " stored=" + String(n_stored), end="\r")

    print("")  # newline after progress

    # Assemble output image
    var total_photons = Float32(n_passes * n_photons_per_pass)
    var out_pixels = alloc[Float32](n_pix * 3)

    for i in range(n_pix):
        var r = Float32(0); var g = Float32(0); var b = Float32(0)
        var vp = vps[i]
        # N_acc > 0 (not vp.valid) is the right gate: vp.valid only reflects
        # whether the *last* pass's re-traced camera ray happened to land on a
        # diffuse surface (see _sppm_camera_pass — re-traced every pass, and a
        # dielectric surface in the path makes that a real coin flip). A pixel
        # can have perfectly good accumulated tau/N_acc from earlier passes yet
        # have vp.valid=0 simply because pass 200 happened to reflect instead
        # of refract — gating on vp.valid here would wrongly zero it out.
        if vp.N_acc > Float32(0.0) and vp.r2 > Float32(0.0):
            # L = beta * tau / (pi * r² * n_passes)
            # tau already has albedo/pi folded in at gather time
            # So final: L = beta * tau / (pi * r² * n_passes)
            var denom = PI * vp.r2 * Float32(n_passes)
            r = vp.beta_r * vp.tau_r / denom
            g = vp.beta_g * vp.tau_g / denom
            b = vp.beta_b * vp.tau_b / denom

        # ISO exposure compensation (matches normalize_film)
        r *= iso_scale; g *= iso_scale; b *= iso_scale

        # NaN guard and optional max-component clamp
        if r != r or r < Float32(0): r = Float32(0)
        if g != g or g < Float32(0): g = Float32(0)
        if b != b or b < Float32(0): b = Float32(0)
        if max_comp > Float32(0):
            var mx = max(r, max(g, b))
            if mx > max_comp:
                var s = max_comp / mx
                r *= s; g *= s; b *= s

        out_pixels[i * 3 + 0] = r
        out_pixels[i * 3 + 1] = g
        out_pixels[i * 3 + 2] = b

    _ = write_image(out_pixels, psc[0].film_w, psc[0].film_h,
                    psc[0].film_filename, Int32(32), Int32(32))

    out_pixels.free()
    heads.free()
    photons.free()
    vps.free()
    return Int32(0)
