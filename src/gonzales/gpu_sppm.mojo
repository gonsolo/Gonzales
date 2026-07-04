# Stochastic Progressive Photon Mapping — GPU port.
# Mirrors sppm.mojo's CPU algorithm exactly (same per-pixel/per-photon bounce
# logic, same RNG seeding, same SPPM update math); reuses sppm.mojo's shared
# device-portable helpers (_geom_normal, _dielectric_bounce,
# _sppm_update_medium, _hash_cell) rather than duplicating them, the same way
# bdpt.mojo already does on CPU. One difference from the CPU version: photon
# storage uses an atomic counter (matching CPU's "if n_stored < n_emit: store"
# cap-and-drop behavior) instead of a plain running index, since GPU threads
# race to append to the shared photon buffer.

from std.sys import has_accelerator
from std.sys.info import size_of
from std.gpu import block_idx, thread_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.atomic import Atomic
from std.math import ceildiv, sqrt, cos, sin, floor, log, exp, max
from std.memory import alloc
from .geometry import (
    RGB, SampledSpectrum, Point3f, Vec3f, Ray_C, Intersection_C, PrimId_C,
    TriangleMesh_C, Material_C, MatKind, AreaLight_C, Sphere_C, Curve_C,
    DistantLight_C, PointLight_C, InfiniteLight_C, Medium_C, MediumInterface_C,
    Grid_C, LightSampler_C, Instance_C, dot, cross, PI, INV_FOUR_PI,
)
from .bvh import BVH2Node, SceneDescriptor2_C, traverse_bvh2_core, any_hit_bvh2_core
from .rng import PCG32
from .sppm import SPPMPixel, SPPMPhoton, _geom_normal, _shading_normal_at, _dielectric_bounce, _sppm_update_medium, _hash_cell, _cosine_hemisphere_sample, _ALPHA, _MAX_B, _HSIZE, _VP_SAMPLES
from .pbrt_parser import ParsedScene_Mojo
from .postprocess import write_image
from .gpu import GpuSceneHandle


@always_inline
def _mk_sd_medium(
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    n_mediums: Int64,
    mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    n_medium_ifaces: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int64,
) -> SceneDescriptor2_C:
    """Minimal SceneDescriptor2_C used only to satisfy _sppm_update_medium's
    signature (it only reads .mediums/.mediumInterfaces/.mediumIfaceCount/
    .spheres) — every other field is a never-dereferenced dangling default,
    same convention as traverse_bvh2_core's own optional instancing args."""
    return SceneDescriptor2_C(
        bvh2Nodes=UnsafePointer[BVH2Node, MutAnyOrigin].unsafe_dangling(),
        primIds=UnsafePointer[PrimId_C, MutAnyOrigin].unsafe_dangling(),
        meshes=meshes,
        meshCount=Int64(0),
        materials=UnsafePointer[Material_C, MutAnyOrigin].unsafe_dangling(),
        materialCount=Int64(0),
        areaLights=UnsafePointer[AreaLight_C, MutAnyOrigin].unsafe_dangling(),
        areaLightCount=Int64(0),
        textures=UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        textureCount=Int64(0),
        distantLights=UnsafePointer[DistantLight_C, MutAnyOrigin].unsafe_dangling(),
        distantLightCount=Int64(0),
        pointLights=UnsafePointer[PointLight_C, MutAnyOrigin].unsafe_dangling(),
        pointLightCount=Int64(0),
        infiniteLights=UnsafePointer[InfiniteLight_C, MutAnyOrigin].unsafe_dangling(),
        infiniteLightCount=Int64(0),
        spheres=spheres,
        sphereCount=n_spheres,
        curves=UnsafePointer[Curve_C, MutAnyOrigin].unsafe_dangling(),
        curveCount=Int64(0),
        mediums=mediums,
        mediumCount=n_mediums,
        mediumInterfaces=mediumInterfaces,
        mediumIfaceCount=n_medium_ifaces,
        grids=UnsafePointer[Grid_C, MutAnyOrigin].unsafe_dangling(),
        gridCount=Int64(0),
        lightSampler=LightSampler_C(cdf=UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(), n=Int32(0), _pad=Int32(0)),
        blasNodesArr=UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        blasPrimIdsArr=UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        blasCount=Int64(0),
        instances=UnsafePointer[Instance_C, MutAnyOrigin].unsafe_dangling(),
        instanceCount=Int64(0),
    )


# ── Kernels ────────────────────────────────────────────────────────────────

def sppm_reset_i32_gpu(counter: UnsafePointer[Int32, MutAnyOrigin]):
    if block_idx.x == 0 and thread_idx.x == 0:
        counter[0] = Int32(0)


def sppm_gen_vp_gpu(
    vps: UnsafePointer[SPPMPixel, MutAnyOrigin],
    inter_scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    n_pix: Int,
    vp_samples: Int,
    fw: Int32,
    r2c: UnsafePointer[Float32, MutAnyOrigin],
    c2w: UnsafePointer[Float32, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    n_mediums: Int64,
    mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    n_medium_ifaces: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int64,
    init_r2: Float32,
    seed: UInt64,
):
    """Traces vp_samples independent primary rays per pixel, called ONCE for
    the whole render (not once per SPPM pass — a per-pass re-trace breaks
    SPPM's convergence guarantee, since the accumulated r2/tau/N_acc would
    then track a wandering position/surface instead of a fixed one whenever
    the path crosses a dielectric surface with a stochastic reflect/refract
    choice). Each of the vp_samples per pixel gets its own persistent
    accumulator; sppm_finalize_gpu averages them at the end. Multiple
    independent samples (rather than one, traced once) is what avoids the
    old black-speckle bug: P(all vp_samples reflect away from the diffuse
    surface) = fresnel^vp_samples, negligible for reasonable vp_samples."""
    var combined = Int(block_idx.x * block_dim.x + thread_idx.x)
    if combined >= n_pix * vp_samples:
        return
    var pix = combined // vp_samples

    var sd = _mk_sd_medium(meshes, mediums, n_mediums, mediumInterfaces, n_medium_ifaces, spheres, n_spheres)
    var has_media = n_mediums > Int64(0)

    var ox = c2w[12]; var oy = c2w[13]; var oz = c2w[14]
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
        ld_r=Float32(0), ld_g=Float32(0), ld_b=Float32(0),
    )

    var pcg = PCG32(seed ^ UInt64(combined * 6364136223846793005 + 1), UInt64(1))

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

    var cur_med_idx = Int32(-1)
    var inter_ptr = inter_scratch + combined

    for bounce in range(_MAX_B):
        var ray = Ray_C(Point3f(rox, roy, roz), Vec3f(rdx, rdy, rdz))
        inter_ptr[0].hit = Int8(0)
        traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, ray, Float32(1.0e38), inter_ptr,
                           blasNodesArr, blasPrimIdsArr, instances)
        if inter_ptr[0].hit == Int8(0):
            break

        var inter = inter_ptr[0]
        var ray_dir = SIMD[DType.float32, 3](rdx, rdy, rdz)
        var t_hit = inter.tHit

        if has_media and Int(cur_med_idx) >= 0:
            var med = mediums[Int(cur_med_idx)]
            var sig_t = med.sigma_a.r + med.sigma_s.r
            if sig_t > Float32(0):
                var t_free = -log(max(pcg.next_float(), Float32(1e-7))) / sig_t
                if t_free < t_hit:
                    var alb_s = med.sigma_s.r / sig_t
                    vp.pos_x = rox + rdx * t_free
                    vp.pos_y = roy + rdy * t_free
                    vp.pos_z = roz + rdz * t_free
                    vp.nx = Float32(0); vp.ny = Float32(1); vp.nz = Float32(0)
                    vp.alb_r = alb_s; vp.alb_g = alb_s; vp.alb_b = alb_s
                    vp.is_volume = Int32(1)
                    vp.valid = Int32(1)
                    break
                else:
                    vp.beta_r *= exp(-(med.sigma_a.r + med.sigma_s.r) * t_hit)
                    vp.beta_g *= exp(-(med.sigma_a.g + med.sigma_s.g) * t_hit)
                    vp.beta_b *= exp(-(med.sigma_a.b + med.sigma_s.b) * t_hit)

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = materials[mat_idx]
        var hx = rox + rdx * t_hit
        var hy = roy + rdy * t_hit
        var hz = roz + rdz * t_hit

        if mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
            var gn = _geom_normal(inter, meshes, instances)
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
            var gn = _shading_normal_at(inter, meshes, instances)
            var hit = SIMD[DType.float32, 3](hx, hy, hz)
            var (new_dir, new_org) = _dielectric_bounce(ray_dir, hit, gn, ior, bounce, pcg)
            rdx = new_dir[0]; rdy = new_dir[1]; rdz = new_dir[2]
            rox = new_org[0]; roy = new_org[1]; roz = new_org[2]
            if has_media:
                var new_idx = _sppm_update_medium(ray_dir, inter, meshes, mat, sd)
                if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                    cur_med_idx = new_idx

        elif mat.type == MatKind.interface:
            if has_media:
                var new_idx = _sppm_update_medium(ray_dir, inter, meshes, mat, sd)
                if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                    cur_med_idx = new_idx
            rox = hx + rdx * Float32(0.0002)
            roy = hy + rdy * Float32(0.0002)
            roz = hz + rdz * Float32(0.0002)

        else:
            break

    vps[combined] = vp


def sppm_emit_photons_gpu(
    photons: UnsafePointer[SPPMPhoton, MutAnyOrigin],
    n_emit: Int,
    max_photons: Int,
    inter_scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    stored_counter: UnsafePointer[Int32, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    n_lights: Int32,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    n_mediums: Int64,
    mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    n_medium_ifaces: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int64,
    default_emit_med: Int32,
    seed: UInt64,
    pass_idx: Int,
):
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= n_emit or n_lights == 0:
        return

    var sd = _mk_sd_medium(meshes, mediums, n_mediums, mediumInterfaces, n_medium_ifaces, spheres, n_spheres)
    var has_media = n_mediums > Int64(0)

    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))

    var li = Int(pcg.next_uint() % UInt32(n_lights))
    var al = areaLights[li]
    var lmesh = meshes[Int(al.meshIdx)]
    var n_tris = Int(max(Int(al.n_tris), 1))

    var ti = Int(pcg.next_uint() % UInt32(n_tris))
    var lb = ti * 3
    var lv0 = Int(lmesh.vertexIndices[lb])
    var lv1 = Int(lmesh.vertexIndices[lb + 1])
    var lv2 = Int(lmesh.vertexIndices[lb + 2])
    var lp0 = SIMD[DType.float32, 3](lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
    var lp1 = SIMD[DType.float32, 3](lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
    var lp2 = SIMD[DType.float32, 3](lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])

    var ru1 = pcg.next_float()
    var ru2 = pcg.next_float()
    var sr1 = sqrt(ru1)
    var lp = lp0 * (Float32(1.0) - sr1) + lp1 * (sr1 * (Float32(1.0) - ru2)) + lp2 * (sr1 * ru2)

    var ln = cross(lp1 - lp0, lp2 - lp0)
    var lnl = dot(ln, ln)
    if lnl > Float32(0.0): ln = ln * (Float32(1.0) / sqrt(lnl))

    var du1 = pcg.next_float()
    var du2 = pcg.next_float()
    var r_samp = sqrt(du1)
    var theta = Float32(2.0) * PI * du2
    var lx = r_samp * cos(theta)
    var lz_loc = r_samp * sin(theta)
    var ly = sqrt(max(Float32(0.0), Float32(1.0) - du1))
    var sgn = Float32(1.0) if ln[2] >= Float32(0.0) else Float32(-1.0)
    var a_tf = Float32(-1.0) / (sgn + ln[2])
    var b_tf = ln[0] * ln[1] * a_tf
    var tangent  = SIMD[DType.float32, 3](Float32(1.0) + sgn*ln[0]*ln[0]*a_tf, sgn*b_tf, -sgn*ln[0])
    var bitangent = SIMD[DType.float32, 3](b_tf, sgn + ln[1]*ln[1]*a_tf, -ln[1])
    var pdir = tangent * lx + bitangent * lz_loc + ln * ly
    var pdl = dot(pdir, pdir)
    if pdl > Float32(0.0): pdir = pdir * (Float32(1.0) / sqrt(pdl))

    var scale = PI * al.total_area * Float32(n_lights) / Float32(n_emit)
    var flux_r = al.emission.r * scale
    var flux_g = al.emission.g * scale
    var flux_b = al.emission.b * scale

    var rox = lp[0] + ln[0] * Float32(0.0001)
    var roy = lp[1] + ln[1] * Float32(0.0001)
    var roz = lp[2] + ln[2] * Float32(0.0001)
    var rdx = pdir[0]; var rdy = pdir[1]; var rdz = pdir[2]
    var cur_med_idx = default_emit_med
    var inter_ptr = inter_scratch + k

    for bounce in range(_MAX_B):
        var ray = Ray_C(Point3f(rox, roy, roz), Vec3f(rdx, rdy, rdz))
        inter_ptr[0].hit = Int8(0)
        traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, ray, Float32(1.0e38), inter_ptr,
                           blasNodesArr, blasPrimIdsArr, instances)
        if inter_ptr[0].hit == Int8(0):
            break

        var inter = inter_ptr[0]
        var ray_dir = SIMD[DType.float32, 3](rdx, rdy, rdz)
        var t_hit = inter.tHit

        if has_media and Int(cur_med_idx) >= 0:
            var med = mediums[Int(cur_med_idx)]
            var sig_t = med.sigma_s.r + med.sigma_a.r
            if sig_t > Float32(0):
                var t_free = -log(max(pcg.next_float(), Float32(1e-7))) / sig_t
                if t_free < t_hit:
                    var sx = rox + rdx * t_free
                    var sy = roy + rdy * t_free
                    var sz = roz + rdz * t_free
                    # bounce > 0: skip storing at the light's own first
                    # segment — covered by sppm_nee_gpu instead (matches
                    # pbrt's own SPPM depth>0 gather skip).
                    if bounce > 0:
                        var slot = Int(Atomic.fetch_add(stored_counter, Int32(1)))
                        if slot < max_photons:
                            photons[slot] = SPPMPhoton(
                                px=sx, py=sy, pz=sz,
                                fr=flux_r, fg=flux_g, fb=flux_b,
                                nxt=Int32(-1), is_volume=Int32(1),
                            )
                    var alb_s = med.sigma_s.r / sig_t
                    flux_r *= alb_s; flux_g *= alb_s; flux_b *= alb_s
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
                    flux_r *= exp(-(med.sigma_a.r + med.sigma_s.r) * t_hit)
                    flux_g *= exp(-(med.sigma_a.g + med.sigma_s.g) * t_hit)
                    flux_b *= exp(-(med.sigma_a.b + med.sigma_s.b) * t_hit)

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = materials[mat_idx]
        var hx = rox + rdx * t_hit
        var hy = roy + rdy * t_hit
        var hz = roz + rdz * t_hit

        if mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
            # bounce > 0: skip storing a photon hit directly by the light
            # with no intermediate bounce — covered by sppm_nee_gpu instead
            # (matches pbrt's own SPPM depth>0 gather skip, avoiding
            # double-counting direct light once via NEE and again here).
            if bounce > 0:
                var slot = Int(Atomic.fetch_add(stored_counter, Int32(1)))
                if slot < max_photons:
                    photons[slot] = SPPMPhoton(
                        px=hx, py=hy, pz=hz,
                        fr=flux_r, fg=flux_g, fb=flux_b,
                        nxt=Int32(-1), is_volume=Int32(0),
                    )
            # Russian-roulette continuation for indirect diffuse-diffuse
            # bounces (color bleeding) — see sppm.mojo's _sppm_photon_pass
            # for why this is needed (photons used to always terminate here).
            var rr_prob = max(mat.albedo.r, max(mat.albedo.g, mat.albedo.b))
            if rr_prob <= Float32(0.0) or pcg.next_float() >= rr_prob:
                break
            var gn = _geom_normal(inter, meshes, instances)
            if dot(gn, ray_dir) > Float32(0.0):
                gn = gn * Float32(-1.0)
            var new_dir = _cosine_hemisphere_sample(gn, pcg.next_float(), pcg.next_float())
            flux_r *= mat.albedo.r / rr_prob
            flux_g *= mat.albedo.g / rr_prob
            flux_b *= mat.albedo.b / rr_prob
            rdx = new_dir[0]; rdy = new_dir[1]; rdz = new_dir[2]
            rox = hx + gn[0] * Float32(0.0001)
            roy = hy + gn[1] * Float32(0.0001)
            roz = hz + gn[2] * Float32(0.0001)
            continue

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            var ior = mat.albedo.r
            var gn = _shading_normal_at(inter, meshes, instances)
            var hit = SIMD[DType.float32, 3](hx, hy, hz)
            var (new_dir, new_org) = _dielectric_bounce(ray_dir, hit, gn, ior, bounce, pcg)
            rdx = new_dir[0]; rdy = new_dir[1]; rdz = new_dir[2]
            rox = new_org[0]; roy = new_org[1]; roz = new_org[2]
            if has_media:
                var new_idx = _sppm_update_medium(ray_dir, inter, meshes, mat, sd)
                if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                    cur_med_idx = new_idx

        elif mat.type == MatKind.interface:
            if has_media:
                var new_idx = _sppm_update_medium(ray_dir, inter, meshes, mat, sd)
                if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                    cur_med_idx = new_idx
            rox = hx + rdx * Float32(0.0002)
            roy = hy + rdy * Float32(0.0002)
            roz = hz + rdz * Float32(0.0002)

        else:
            break


def sppm_grid_reset_gpu(heads: UnsafePointer[Int32, MutAnyOrigin], hsize: Int):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= hsize:
        return
    heads[tid] = Int32(-1)


def sppm_grid_insert_gpu(
    photons: UnsafePointer[SPPMPhoton, MutAnyOrigin],
    n_stored: Int,
    heads: UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
):
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= n_stored:
        return
    var ix = Int(floor(photons[k].px * inv_cell))
    var iy = Int(floor(photons[k].py * inv_cell))
    var iz = Int(floor(photons[k].pz * inv_cell))
    var h = _hash_cell(ix, iy, iz)
    var old = Atomic._xchg(heads + h, Int32(k))
    photons[k].nxt = old


def sppm_gather_gpu(
    vps: UnsafePointer[SPPMPixel, MutAnyOrigin],
    n_pix: Int,
    photons: UnsafePointer[SPPMPhoton, MutAnyOrigin],
    heads: UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n_pix:
        return
    if vps[i].valid == Int32(0):
        return
    var vp = vps[i]
    var vx = vp.pos_x; var vy = vp.pos_y; var vz = vp.pos_z
    var r2 = vp.r2

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

    if M > Float32(0.0):
        var N = vp.N_acc
        var ratio = (N + _ALPHA * M) / (N + M)
        vps[i].r2    = r2 * ratio
        vps[i].tau_r = (vp.tau_r + phi_r) * ratio
        vps[i].tau_g = (vp.tau_g + phi_g) * ratio
        vps[i].tau_b = (vp.tau_b + phi_b) * ratio
        vps[i].N_acc = N + _ALPHA * M


def sppm_nee_gpu(
    vps: UnsafePointer[SPPMPixel, MutAnyOrigin],
    n_vps: Int,
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    n_lights: Int32,
    seed: UInt64,
    pass_idx: Int,
):
    """Direct (NEE) lighting update — see sppm.mojo's _sppm_nee_update for
    why this separate term is needed (matches pbrt's "pixel.Ld")."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n_vps or n_lights == 0:
        return
    var vp = vps[i]
    if vp.valid == Int32(0) or vp.is_volume == Int32(1):
        return

    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + i), UInt64(11))

    var li = Int(pcg.next_uint() % UInt32(n_lights))
    var al = areaLights[li]
    var lmesh = meshes[Int(al.meshIdx)]
    var n_tris = Int(max(Int(al.n_tris), 1))
    var ti = Int(pcg.next_uint() % UInt32(n_tris))
    var lb = ti * 3
    var lv0 = Int(lmesh.vertexIndices[lb])
    var lv1 = Int(lmesh.vertexIndices[lb + 1])
    var lv2 = Int(lmesh.vertexIndices[lb + 2])
    var lp0 = SIMD[DType.float32, 3](lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
    var lp1 = SIMD[DType.float32, 3](lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
    var lp2 = SIMD[DType.float32, 3](lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])
    var ru1 = pcg.next_float()
    var ru2 = pcg.next_float()
    var sr1 = sqrt(ru1)
    var lp = lp0 * (Float32(1.0) - sr1) + lp1 * (sr1 * (Float32(1.0) - ru2)) + lp2 * (sr1 * ru2)
    var ln = cross(lp1 - lp0, lp2 - lp0)
    var lnl = dot(ln, ln)
    if lnl > Float32(0.0): ln = ln * (Float32(1.0) / sqrt(lnl))

    var vpos = SIMD[DType.float32, 3](vp.pos_x, vp.pos_y, vp.pos_z)
    var vn   = SIMD[DType.float32, 3](vp.nx, vp.ny, vp.nz)
    var to_light = lp - vpos
    var dist2 = dot(to_light, to_light)
    var dist = sqrt(dist2)
    if dist <= Float32(0.0):
        return
    var wi = to_light * (Float32(1.0) / dist)

    var cos_surface = dot(vn, wi)
    var cos_light = -dot(ln, wi)
    if cos_surface <= Float32(0.0) or cos_light <= Float32(0.0):
        return

    var shadow_org = vpos + vn * Float32(0.0001)
    var shadow_ray = Ray_C(Point3f(shadow_org[0], shadow_org[1], shadow_org[2]),
                            Vec3f(wi[0], wi[1], wi[2]))
    var t_max = dist * Float32(0.999)
    if any_hit_bvh2_core(bvh2Nodes, primIds, meshes, curves, shadow_ray, t_max,
                          blasNodesArr, blasPrimIdsArr, instances):
        return

    var inv_pdf_area = Float32(n_lights) * al.total_area
    var geom = cos_surface * cos_light / dist2 * inv_pdf_area
    var brdf_r = vp.alb_r / PI
    var brdf_g = vp.alb_g / PI
    var brdf_b = vp.alb_b / PI
    vps[i].ld_r += brdf_r * al.emission.r * geom
    vps[i].ld_g += brdf_g * al.emission.g * geom
    vps[i].ld_b += brdf_b * al.emission.b * geom


def sppm_finalize_gpu(
    vps: UnsafePointer[SPPMPixel, MutAnyOrigin],
    n_pix: Int,
    vp_samples: Int,
    n_passes: Int32,
    iso_scale: Float32,
    max_comp: Float32,
    out_pixels: UnsafePointer[Float32, MutAnyOrigin],
):
    """Averages the vp_samples independently-converged samples per pixel —
    see sppm_gen_vp_gpu's docstring for why each sample has its own fixed
    visible point/accumulator rather than sharing one per pixel."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n_pix:
        return
    var r = Float32(0); var g = Float32(0); var b = Float32(0)
    for vs in range(vp_samples):
        var vp = vps[i * vp_samples + vs]
        if vp.valid == Int32(0):
            continue
        var vp_r = Float32(0); var vp_g = Float32(0); var vp_b = Float32(0)
        if vp.N_acc > Float32(0.0) and vp.r2 > Float32(0.0):
            var denom = PI * vp.r2 * Float32(n_passes)
            vp_r += vp.tau_r / denom
            vp_g += vp.tau_g / denom
            vp_b += vp.tau_b / denom
        if vp.is_volume == Int32(0):
            vp_r += vp.ld_r / Float32(n_passes)
            vp_g += vp.ld_g / Float32(n_passes)
            vp_b += vp.ld_b / Float32(n_passes)
        r += vp.beta_r * vp_r
        g += vp.beta_g * vp_g
        b += vp.beta_b * vp_b
    r /= Float32(vp_samples); g /= Float32(vp_samples); b /= Float32(vp_samples)

    r *= iso_scale; g *= iso_scale; b *= iso_scale

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


# ── Host orchestration ───────────────────────────────────────────────────────

def sppm_render_gpu(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
    sd: SceneDescriptor2_C,
    n_passes: Int,
    n_photons_per_pass: Int,
    initial_radius: Float32,
    no_denoise: Bool,
    verbose: Bool,
) -> Int32:
    """GPU-accelerated Stochastic Progressive Photon Mapping — same algorithm
    as sppm_render (sppm.mojo), parallelized: one thread per pixel for the
    camera pass/gather/finalize, one thread per emitted photon for the photon
    pass, atomic-exchange hash-grid build (classic parallel linked-list
    insertion)."""
    if Int(sd.areaLightCount) == 0:
        print("SPPM: no area lights in scene, cannot emit photons")
        return Int32(-1)

    var fw = Int(psc[0].film_w)
    var fh = Int(psc[0].film_h)
    var n_pix = fw * fh
    var iso_scale = psc[0].film_iso / Float32(100)
    var max_comp = psc[0].film_max_comp

    print("SPPM (GPU): " + String(fw) + "x" + String(fh)
          + " " + String(n_passes) + " passes x "
          + String(n_photons_per_pass) + " photons  r=" + String(initial_radius))

    var default_emit_med = Int32(-1)
    if Int(sd.mediumCount) > 0 and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    var init_r2 = initial_radius * initial_radius
    var inv_cell = Float32(1.0) / initial_radius

    var ret = Int32(0)
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            comptime block_size = 256

            var n_vps = n_pix * _VP_SAMPLES
            # max_photons intentionally equals n_photons_per_pass — see
            # sppm.mojo's sppm_render for why a larger buffer (letting every
            # Russian-roulette diffuse-continuation event store
            # unconditionally) was tried and measurably over-brightened
            # the render instead of helping.
            var max_photons = n_photons_per_pass
            var vps_buf     = handle[].ctx.enqueue_create_buffer[DType.uint8](n_vps * size_of[SPPMPixel]())
            var photons_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](max(max_photons, 1) * size_of[SPPMPhoton]())
            var heads_buf   = handle[].ctx.enqueue_create_buffer[DType.uint8](_HSIZE * size_of[Int32]())
            var inter_cam_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_vps * size_of[Intersection_C]())
            var inter_ph_buf  = handle[].ctx.enqueue_create_buffer[DType.uint8](max(n_photons_per_pass, 1) * size_of[Intersection_C]())
            var counter_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](size_of[Int32]())
            var out_buf     = handle[].ctx.enqueue_create_buffer[DType.uint8](n_pix * 3 * size_of[Float32]())

            var r2c_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with r2c_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[0].raster_to_camera.bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[i] = src[i]
            var c2w_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
            with c2w_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = psc[0].camera_to_world.bitcast[UInt8]()
                for i in range(16 * size_of[Float32]()):
                    dst[i] = src[i]

            var vps_ptr    = vps_buf.unsafe_ptr().bitcast[SPPMPixel]()
            var photons_ptr = photons_buf.unsafe_ptr().bitcast[SPPMPhoton]()
            var heads_ptr  = heads_buf.unsafe_ptr().bitcast[Int32]()
            var inter_cam_ptr = inter_cam_buf.unsafe_ptr().bitcast[Intersection_C]()
            var inter_ph_ptr  = inter_ph_buf.unsafe_ptr().bitcast[Intersection_C]()
            var counter_ptr = counter_buf.unsafe_ptr().bitcast[Int32]()
            var out_ptr     = out_buf.unsafe_ptr().bitcast[Float32]()
            var r2c_ptr = r2c_buf.unsafe_ptr().bitcast[Float32]()
            var c2w_ptr = c2w_buf.unsafe_ptr().bitcast[Float32]()

            var bvh2Nodes = handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node]()
            var primIds = handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C]()
            var meshes = handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C]()
            var curves = handle[].curves_buf.unsafe_ptr().bitcast[Curve_C]()
            var blasNodesArr = handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]]()
            var blasPrimIdsArr = handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]]()
            var instances = handle[].instances_buf.unsafe_ptr().bitcast[Instance_C]()
            var materials = handle[].materials_buf.unsafe_ptr().bitcast[Material_C]()
            var mediums = handle[].mediums_buf.unsafe_ptr().bitcast[Medium_C]()
            var mediumInterfaces = handle[].medium_ifaces_buf.unsafe_ptr().bitcast[MediumInterface_C]()
            var spheres = handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C]()
            var areaLights = handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C]()
            var n_mediums = Int64(handle[].n_mediums)
            var n_medium_ifaces = Int64(handle[].n_medium_ifaces)
            var n_spheres = Int64(handle[].n_spheres)
            var n_lights = Int32(handle[].n_area_lights)

            var grid_pix = ceildiv(n_pix, block_size)
            var grid_vps = ceildiv(n_vps, block_size)
            var grid_hsize = ceildiv(_HSIZE, block_size)

            # Camera/visible-point samples are traced ONCE for the whole
            # render, not per SPPM pass — see sppm_gen_vp_gpu's docstring for
            # why a per-pass re-trace breaks SPPM's convergence guarantee.
            var cam_seed = psc[0].rng_seed ^ UInt64(0x9E3779B97F4A7C15 + 7)
            handle[].ctx.enqueue_function[sppm_gen_vp_gpu](
                vps_ptr, inter_cam_ptr, n_pix, _VP_SAMPLES, psc[0].film_w, r2c_ptr, c2w_ptr,
                bvh2Nodes, primIds, meshes, curves, blasNodesArr, blasPrimIdsArr, instances,
                materials, mediums, n_mediums, mediumInterfaces, n_medium_ifaces, spheres, n_spheres,
                init_r2, cam_seed,
                grid_dim=grid_vps, block_dim=block_size)

            for pass_idx in range(n_passes):
                handle[].ctx.enqueue_function[sppm_reset_i32_gpu](
                    counter_ptr, grid_dim=1, block_dim=1)

                var pass_seed = psc[0].rng_seed ^ UInt64(pass_idx * 2654435761 + 1)
                var grid_emit = ceildiv(max(n_photons_per_pass, 1), block_size)
                handle[].ctx.enqueue_function[sppm_emit_photons_gpu](
                    photons_ptr, n_photons_per_pass, max_photons, inter_ph_ptr, counter_ptr,
                    areaLights, n_lights, meshes,
                    bvh2Nodes, primIds, curves, blasNodesArr, blasPrimIdsArr, instances,
                    materials, mediums, n_mediums, mediumInterfaces, n_medium_ifaces, spheres, n_spheres,
                    default_emit_med, pass_seed, pass_idx,
                    grid_dim=grid_emit, block_dim=block_size)

                handle[].ctx.synchronize()
                var n_stored_raw: Int32
                with counter_buf.map_to_host() as host_buf:
                    var src = host_buf.unsafe_ptr().bitcast[Int32]()
                    n_stored_raw = src[0]
                var n_stored = min(Int(n_stored_raw), max_photons)

                if n_stored > 0:
                    handle[].ctx.enqueue_function[sppm_grid_reset_gpu](
                        heads_ptr, _HSIZE, grid_dim=grid_hsize, block_dim=block_size)
                    var grid_ins = ceildiv(n_stored, block_size)
                    handle[].ctx.enqueue_function[sppm_grid_insert_gpu](
                        photons_ptr, n_stored, heads_ptr, inv_cell,
                        grid_dim=grid_ins, block_dim=block_size)
                    handle[].ctx.enqueue_function[sppm_gather_gpu](
                        vps_ptr, n_vps, photons_ptr, heads_ptr, inv_cell,
                        grid_dim=grid_vps, block_dim=block_size)

                var nee_seed = psc[0].rng_seed ^ UInt64(pass_idx * 0xBF58476D1CE4E5B9 + 3)
                handle[].ctx.enqueue_function[sppm_nee_gpu](
                    vps_ptr, n_vps,
                    bvh2Nodes, primIds, meshes, curves, blasNodesArr, blasPrimIdsArr, instances,
                    areaLights, n_lights, nee_seed, pass_idx,
                    grid_dim=grid_vps, block_dim=block_size)

                if verbose or (pass_idx + 1) % 10 == 0:
                    print("SPPM (GPU): pass " + String(pass_idx + 1) + "/" + String(n_passes)
                          + " stored=" + String(n_stored), end="\r")

            print("")

            handle[].ctx.enqueue_function[sppm_finalize_gpu](
                vps_ptr, n_pix, _VP_SAMPLES, Int32(n_passes), iso_scale, max_comp, out_ptr,
                grid_dim=grid_pix, block_dim=block_size)
            handle[].ctx.synchronize()

            var out_pixels = alloc[Float32](n_pix * 3)
            with out_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = out_pixels.bitcast[UInt8]()
                for i in range(n_pix * 3 * size_of[Float32]()):
                    dst[i] = src[i]

            _ = write_image(out_pixels, psc[0].film_w, psc[0].film_h,
                            psc[0].film_filename, Int32(32), Int32(32))
            out_pixels.free()
        except e:
            print("SPPM GPU render failed: " + String(e))
            ret = Int32(-1)
    else:
        print("SPPM GPU: no accelerator")
        ret = Int32(-1)
    return ret
