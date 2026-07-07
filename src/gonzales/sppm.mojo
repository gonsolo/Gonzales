# Stochastic Progressive Photon Mapping — CPU driver + GPU kernels, sharing
# one comptime[use_gpu]-parameterized core (same pattern as bdpt.mojo's
# LVC-BPT port; see project_unified_renderer_roadmap in memory).
# Reference: Hachisuka et al. 2008 "Progressive Photon Mapping"

from std.sys import has_accelerator
from std.sys.info import size_of
from std.gpu import block_idx, thread_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.algorithm import parallelize
from std.math import sqrt, cos, sin, floor, log, exp, max, min, ceildiv
from std.memory import alloc
from std.atomic import Atomic
from .geometry import (
    RGB, SampledSpectrum, Point3f, Point2f, Vec3f, vec3f, point3f, Ray_C, Intersection_C, PrimId_C,
    TriangleMesh_C, Material_C, MatKind, AreaLight_C, Sphere_C, Medium_C, MediumInterface_C,
    Instance_C, dot, cross, fr_dielectric, sphere_outward_normal, PI, INV_FOUR_PI,
    Curve_C, curve_piece_endpoints, _curve_perp_axis, DistantLight_C, InfiniteLight_C,
)
from .bvh import (
    BVH2Node, SceneDescriptor2_C, traverse_bvh2_core, any_hit_bvh2_core, _mk_sd_full,
    _scene_bounding_sphere, _sample_disk_perpendicular, _sample_infinite_light_dir, _eval_infinite_light_and_pdf,
)
from .sampling import power_heuristic
from .transform import transform_normal_by_instance
from .rng import PCG32
from .pbrt_parser import ParsedScene_Mojo
from .postprocess import write_image
from .gpu import GpuSceneHandle

comptime _ALPHA  = Float32(0.7)
comptime _MAX_B  = 10
comptime _HSIZE  = 1048576   # 2^20 hash buckets
# Independent visible-point samples per pixel, traced ONCE for the whole
# render (not re-traced every SPPM pass — see _sppm_camera_pass's docstring).
# Needed because the camera ray can cross a dielectric surface (stochastic
# reflect-vs-refract choice): with only 1 sample, a pixel unlucky enough to
# draw "reflect" would never see the diffuse surface behind the glass/water
# at all (the original black-speckle bug). With _VP_SAMPLES independent
# draws, P(all reflect) is (fresnel)^_VP_SAMPLES — negligible in practice —
# and each sample gets its own persistent (r2, tau, N_acc) accumulator that
# converges correctly since its surface/position never changes pass to pass.
comptime _VP_SAMPLES = 16


# ── Data structures ───────────────────────────────────────────────────────────

@fieldwise_init
struct SPPMPixel(TrivialRegisterPassable):
    """Visible point from one camera ray + SPPM accumulators."""
    var pos:    Point3f
    var normal: Vec3f
    var beta: RGB  # camera throughput
    var alb:  RGB  # surface albedo
    var tau:  RGB  # accumulated flux
    var N_acc:  Float32   # photon count (alpha-weighted sum)
    var r2:     Float32   # current search radius²
    var valid:  Int32     # 1 = has VP
    var pidx:   Int32     # flat pixel index
    var is_volume: Int32  # 1 = volume scatter VP (isotropic phase fn); 0 = surface
    # Direct (NEE) lighting accumulator — resampled fresh each SPPM pass (one
    # shadow ray per pass, same cadence as the photon pass), summed here and
    # divided by n_passes at finalize time. This is what pbrt's own SPPM
    # calls "pixel.Ld": a completely separate term from tau/photon-density,
    # capturing direct illumination at the visible point (which the
    # photon-density term alone can't reconstruct without heavy noise, since
    # it's estimating both direct AND indirect/caustic lighting through a
    # single, indirect-only channel otherwise). Surface VPs only — volume VPs
    # leave this at 0 (this scene has no participating media; NEE from a
    # volume scatter point would need a phase-function-weighted variant,
    # not implemented).
    var ld: RGB
    # Infinite-light radiance for a VP sample whose traced ray escaped the
    # scene entirely (valid stays 0 — there's no surface to gather photons
    # at or run NEE from) instead of hitting a diffuse/volume scatterer.
    # Already beta-weighted at trace time (see _sppm_trace_visible_point's
    # miss branch), so _sppm_finalize_one_pixel adds it directly rather than
    # multiplying by vp.beta again like the tau/ld terms.
    var env: RGB

@fieldwise_init
struct SPPMPhoton(TrivialRegisterPassable):
    """Photon stored at a scatter event (surface diffuse or volume)."""
    var pos: Point3f
    var flux: RGB  # flux at stored position
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
def _shading_normal_at(
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin] = UnsafePointer[Instance_C, MutAnyOrigin].unsafe_dangling(),
) -> SIMD[DType.float32, 3]:
    """Barycentrically-interpolated SMOOTH shading normal at a triangle hit,
    falling back to the flat geometric normal when the mesh has no per-vertex
    normals. Refraction through a finely-tessellated curved surface (e.g. the
    wavy water sheet in water-caustic, whose PLY carries per-vertex normals)
    MUST use the smooth normal — the flat per-triangle normal refracts each
    triangle's whole patch of light in one direction, producing a blocky/
    blotchy caustic and the wrong energy distribution. This is what pbrt (and
    gonzales's own main path tracer via shading.mojo::_shading_normal) does;
    SPPM previously used only _geom_normal here, which was the discrepancy."""
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
    var gn = cross(p1 - p0, p2 - p0)
    if inter.primId.instanceIdx >= Int32(0):
        gn = transform_normal_by_instance(instances[Int(inter.primId.instanceIdx)].worldToObj, gn)
    var gl = dot(gn, gn)
    if gl > Float32(0.0): gn = gn * (Float32(1.0) / sqrt(gl))
    # No per-vertex normals → flat normal (sentinel addr <= 4, see GPU-nullable convention).
    if Int(m.normals) <= 4:
        return gn
    var w0 = Float32(1.0) - inter.u - inter.v
    var n0 = SIMD[DType.float32, 3](m.normals[v0*3], m.normals[v0*3+1], m.normals[v0*3+2])
    var n1 = SIMD[DType.float32, 3](m.normals[v1*3], m.normals[v1*3+1], m.normals[v1*3+2])
    var n2 = SIMD[DType.float32, 3](m.normals[v2*3], m.normals[v2*3+1], m.normals[v2*3+2])
    var sn = n0 * w0 + n1 * inter.u + n2 * inter.v
    if inter.primId.instanceIdx >= Int32(0):
        sn = transform_normal_by_instance(instances[Int(inter.primId.instanceIdx)].worldToObj, sn)
    var sl = dot(sn, sn)
    if sl <= Float32(1e-12):
        return gn
    sn = sn * (Float32(1.0) / sqrt(sl))
    if dot(sn, gn) < Float32(0.0):
        sn = -sn
    return sn

@always_inline
def _hash_cell(ix: Int, iy: Int, iz: Int) -> Int:
    var h = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791
    return (h % _HSIZE + _HSIZE) % _HSIZE


@always_inline
def _cosine_hemisphere_sample(n: SIMD[DType.float32, 3], u1: Float32, u2: Float32) -> SIMD[DType.float32, 3]:
    """Cosine-weighted random direction in the hemisphere around normal n
    (Frisvad tangent frame, same construction used for area-light emission
    sampling in _sppm_photon_pass)."""
    var r_samp = sqrt(u1)
    var theta = Float32(2.0) * PI * u2
    var lx = r_samp * cos(theta)
    var lz_loc = r_samp * sin(theta)
    var ly = sqrt(max(Float32(0.0), Float32(1.0) - u1))
    var sgn = Float32(1.0) if n[2] >= Float32(0.0) else Float32(-1.0)
    var a_tf = Float32(-1.0) / (sgn + n[2])
    var b_tf = n[0] * n[1] * a_tf
    var tangent   = SIMD[DType.float32, 3](Float32(1.0) + sgn*n[0]*n[0]*a_tf, sgn*b_tf, -sgn*n[0])
    var bitangent = SIMD[DType.float32, 3](b_tf, sgn + n[1]*n[1]*a_tf, -n[1])
    var d = tangent * lx + bitangent * lz_loc + n * ly
    var dl = dot(d, d)
    if dl > Float32(0.0): d = d * (Float32(1.0) / sqrt(dl))
    return d


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


# ── Homogeneous-medium free-flight sampling ───────────────────────────────────
# Shared by BDPT and SPPM (CPU + GPU), which only ever deal with homogeneous
# media (glass-of-water / volumetric-caustic scenes) — no delta-tracking
# needed. The main wavefront path tracer (rendering.mojo / gpu.mojo) has its
# own richer version that also handles heterogeneous "uniformgrid" media.

@fieldwise_init
struct HomogeneousFreeFlight(TrivialRegisterPassable):
    """Result of sampling a free-flight distance through a homogeneous medium
    by its red/hero-wavelength extinction coefficient (the same "sample by
    one channel, let the rest cancel analytically" convention used
    throughout gonzales's spectral MIS)."""
    var collided: Bool
    var t_free: Float32     # sampled distance (meaningful either way)
    var sig_t:   Float32    # red-channel extinction sigma_a.r + sigma_s.r
    var albedo:  RGB        # single-scattering albedo at the collision point (only if collided)
    var transmittance: RGB  # Beer-Lambert factor over the full segment (only if NOT collided)

@always_inline
def sample_homogeneous_free_flight(med: Medium_C, t_surf: Float32, mut pcg: PCG32) -> HomogeneousFreeFlight:
    var sigma_t = med.sigma_a + med.sigma_s
    var sig_t = sigma_t.r
    if sig_t <= Float32(0.0):
        return HomogeneousFreeFlight(False, t_surf, sig_t, RGB(Float32(0)), RGB(Float32(1)))
    var t_free = -log(max(pcg.next_float(), Float32(1e-7))) / sig_t
    if t_free < t_surf:
        var alb_s = med.sigma_s.r / sig_t
        var alb_g_s = med.sigma_s.g / sigma_t.g if sigma_t.g > Float32(0.0) else alb_s
        var alb_b_s = med.sigma_s.b / sigma_t.b if sigma_t.b > Float32(0.0) else alb_s
        return HomogeneousFreeFlight(True, t_free, sig_t, RGB(alb_s, alb_g_s, alb_b_s), RGB(Float32(1)))
    var Tr = RGB(exp(-sigma_t.r * t_surf), exp(-sigma_t.g * t_surf), exp(-sigma_t.b * t_surf))
    return HomogeneousFreeFlight(False, t_free, sig_t, RGB(Float32(0)), Tr)


# ── Uniform area-light sampling ───────────────────────────────────────────────
# Shared by BDPT's light-subpath emission and SPPM's photon emission + NEE
# (CPU + GPU) — all three use the same "uniform over all lights, uniform over
# triangles, uniform barycentric point" scheme (as opposed to shading.mojo's
# power-weighted light_sampler_sample used by the main path tracer's NEE).

@fieldwise_init
struct AreaLightSample(TrivialRegisterPassable):
    var light:  AreaLight_C
    var point:  SIMD[DType.float32, 3]
    var normal: SIMD[DType.float32, 3]

@always_inline
def sample_area_light_uniform(
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    meshes:     UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    n_lights:   Int,
    mut pcg:    PCG32,
    curves:     UnsafePointer[Curve_C, MutAnyOrigin] = UnsafePointer[Curve_C, MutAnyOrigin].unsafe_dangling(),
) -> AreaLightSample:
    """Uniformly picks one area light, then a point + geometric normal on
    it: a random triangle + barycentric point on a mesh light (kind==0,
    using the mesh's per-vertex shading normals when present — needed for
    e.g. ceiling lights whose winding gives an upward geometric normal but
    whose scene-specified normals point down into the room), or a random
    piece + point on a curve's swept tube (kind==1, `curves` must be a real
    pointer whenever any curve lights exist)."""
    var li = Int(pcg.next_uint() % UInt32(n_lights))
    var al = areaLights[li]
    if al.kind == Int8(1):
        var curve = curves[Int(al.meshIdx)]
        var piece = Int(pcg.next_uint() % UInt32(max(Int(curve.n_pieces), 1)))
        var (q0, q1, r0, r1) = curve_piece_endpoints(curve, piece)
        var axis = q1 - q0
        var axis_len = sqrt(dot(axis, axis))
        var axis_dir = SIMD[DType.float32, 3](Float32(0.0), Float32(0.0), Float32(1.0))
        if axis_len > Float32(1e-8):
            axis_dir = axis * (Float32(1.0) / axis_len)
        var ru1 = pcg.next_float(); var ru2 = pcg.next_float()
        var r = r0 + (r1 - r0) * ru1
        var u_perp = _curve_perp_axis(axis_dir)
        var v_perp = cross(axis_dir, u_perp)
        var theta = ru2 * (Float32(2.0) * PI)
        var radial = u_perp * cos(theta) + v_perp * sin(theta)
        var point = q0 + axis_dir * (axis_len * ru1) + radial * r
        return AreaLightSample(al, point, radial)
    var lmesh = meshes[Int(al.meshIdx)]
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
    var lnl = dot(ln, ln)
    if lnl > Float32(0): ln = ln*(Float32(1)/sqrt(lnl))
    # Use shading normals when provided — they give the correct emission hemisphere.
    if Int(lmesh.normals) > 4:
        var sn0 = Vec3f(lmesh.normals[lv0*3], lmesh.normals[lv0*3+1], lmesh.normals[lv0*3+2])
        var sn1 = Vec3f(lmesh.normals[lv1*3], lmesh.normals[lv1*3+1], lmesh.normals[lv1*3+2])
        var sn2 = Vec3f(lmesh.normals[lv2*3], lmesh.normals[lv2*3+1], lmesh.normals[lv2*3+2])
        var sn_avg = (sn0 + sn1 + sn2) / Float32(3)
        var snl = sn_avg.length()
        if snl > Float32(0): ln = (sn_avg / snl).to_simd()
    return AreaLightSample(al, lp, ln)


# ── Camera pass ───────────────────────────────────────────────────────────────

@always_inline
def _sppm_update_medium(
    ray_dir: SIMD[DType.float32, 3],
    inter: Intersection_C,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    mat: Material_C,
    sd: SceneDescriptor2_C,
    hit: Point3f = Point3f(Float32(0)),
) -> Int32:
    """Return new current_medium_idx after crossing a surface with MediumInterface."""
    if mat.medium_interface_idx < Int32(0) or sd.mediumIfaceCount == Int64(0):
        return Int32(-1)  # stays vacuum; caller keeps existing idx if needed
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
        var m = meshes[mi]
        var v0 = Int(m.vertexIndices[bv]); var v1 = Int(m.vertexIndices[bv+1]); var v2 = Int(m.vertexIndices[bv+2])
        var p0 = Point3f(m.points[v0*4], m.points[v0*4+1], m.points[v0*4+2])
        var p1 = Point3f(m.points[v1*4], m.points[v1*4+1], m.points[v1*4+2])
        var p2 = Point3f(m.points[v2*4], m.points[v2*4+1], m.points[v2*4+2])
        var e1 = p1 - p0; var e2 = p2 - p0
        n = Vec3f(e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x)
    var md = ray_dir[0]*n.x + ray_dir[1]*n.y + ray_dir[2]*n.z
    return iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx

def _sppm_trace_visible_point(
    sd:       SceneDescriptor2_C,
    mut pcg:  PCG32,
    r2c:      UnsafePointer[Float32, MutAnyOrigin],
    c2w:      UnsafePointer[Float32, MutAnyOrigin],
    px: Int, py: Int,
    pidx:     Int32,
    init_r2:  Float32,
    scratch:  UnsafePointer[Intersection_C, MutAnyOrigin],
) -> SPPMPixel:
    """Trace one primary ray for pixel (px,py), returning its visible point.
    Shared verbatim between the CPU driver (_sppm_camera_pass, one reused
    `scratch` slot) and the GPU kernel (sppm_gen_vp_gpu, one slot per
    thread) — no comptime[use_gpu] split needed here, the bounce loop
    itself has zero CPU/GPU divergence. `scratch` is caller-owned (no
    internal alloc/free) so this is safe to call from a GPU kernel thread,
    same convention as bdpt.mojo's shared subpath tracers."""
    var org = Point3f(c2w[12], c2w[13], c2w[14])
    var has_media = Int(sd.mediumCount) > 0

    var vp = SPPMPixel(
        pos=Point3f(Float32(0)),
        normal=Vec3f(Float32(0), Float32(1), Float32(0)),
        beta=RGB(Float32(1)),
        alb=RGB(Float32(0)),
        tau=RGB(Float32(0)),
        N_acc=Float32(0), r2=init_r2, valid=Int32(0), pidx=pidx,
        is_volume=Int32(0),
        ld=RGB(Float32(0)),
        env=RGB(Float32(0)),
    )

    # Sub-pixel jitter — diversifies which point on a dielectric-obscured
    # surface (e.g. the caustic floor seen through the water) each of the
    # vp_samples independent samples lands on.
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
    var rd = Vec3f(
        c2w[0]*cx + c2w[4]*cy + c2w[8]*cz,
        c2w[1]*cx + c2w[5]*cy + c2w[9]*cz,
        c2w[2]*cx + c2w[6]*cy + c2w[10]*cz,
    )
    var dl = rd.length()
    if dl > Float32(0.0): rd = rd / dl
    var ro = org

    var cur_med_idx = Int32(-1)  # camera starts in vacuum

    for bounce in range(_MAX_B):
        var ray = Ray_C(ro, rd)
        scratch[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1.0e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        if scratch[0].hit == Int8(0):
            for inf_i in range(Int(sd.infiniteLightCount)):
                var ilight = sd.infiniteLights[inf_i]
                var (Le, _pdf_unused) = _eval_infinite_light_and_pdf(ilight, rd)
                vp.env += vp.beta * Le
            break

        var inter = scratch[0]
        var ray_dir = rd.to_simd()
        var t_hit = inter.tHit

        # ── Volume free-flight ────────────────────────────────────────────
        if has_media and Int(cur_med_idx) >= 0:
            var med = sd.mediums[Int(cur_med_idx)]
            var ff = sample_homogeneous_free_flight(med, t_hit, pcg)
            if ff.collided:
                # Volume scatter — store VP here
                vp.pos = ro + rd * ff.t_free
                vp.normal = Vec3f(Float32(0), Float32(1), Float32(0))
                vp.alb = ff.albedo
                vp.is_volume = Int32(1)
                vp.valid = Int32(1)
                break
            else:
                # Transmittance through full segment to surface
                vp.beta *= ff.transmittance

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hit = ro + rd * t_hit

        # Mix material: stochastically resolve to one of two sub-materials
        # (mirrors shading.mojo's shade_mix / bdpt.mojo's own resolution).
        if mat.type == MatKind.mix:
            var mix_idx1 = Int(mat.tex_idx & Int32(0xFFFF))
            var mix_idx2 = Int((mat.tex_idx >> 16) & Int32(0xFFFF))
            var mix_amount = mat.roughU
            var mix_chosen = mix_idx2 if pcg.next_float() < mix_amount else mix_idx1
            mat = sd.materials[mix_chosen]
            if mat.type == MatKind.mix:
                mat.type = MatKind.diffuse

        if mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
            var gn = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn, ray_dir) > Float32(0.0):
                gn = gn * Float32(-1.0)
            vp.pos = hit
            vp.normal = vec3f(gn)
            vp.alb = mat.albedo
            vp.is_volume = Int32(0)
            vp.valid = Int32(1)
            break

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            var ior = mat.albedo.r
            var gn = _shading_normal_at(inter, sd.meshes, sd.instances)
            var (new_dir, new_org) = _dielectric_bounce(ray_dir, hit.to_simd(), gn, ior, bounce, pcg)
            rd = vec3f(new_dir)
            ro = point3f(new_org)
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
            ro = hit + rd * Float32(0.0002)

        else:
            break  # area_light, conductor, etc.

    return vp

def _sppm_camera_pass(
    vps:        UnsafePointer[SPPMPixel, MutAnyOrigin],
    n_pix:      Int,
    vp_samples: Int,
    fw:       Int32,
    r2c:      UnsafePointer[Float32, MutAnyOrigin],
    c2w:      UnsafePointer[Float32, MutAnyOrigin],
    sd:       SceneDescriptor2_C,
    init_r2:  Float32,
    seed:     UInt64,
):
    """Trace `vp_samples` independent primary rays per pixel, ONCE for the
    whole render (not once per SPPM pass — see sppm_render's docstring for
    why a per-pass re-trace breaks SPPM's convergence guarantee). Each sample
    gets its own persistent (r2, tau, N_acc) accumulator, fixed at this one
    surface/position for every subsequent photon pass; sppm_render averages
    the `vp_samples` independently-converged estimates per pixel at the end.
    Having multiple independent samples (rather than one, traced once) is
    what avoids the old black-speckle bug: a pixel where the camera ray
    crosses a dielectric surface makes a genuine random reflect-vs-refract
    choice each sample, so as long as vp_samples is large enough, at least
    some samples land on the diffuse floor even if others reflect away."""
    # One scratch Intersection_C per worker (indexed by `combined`) instead
    # of one shared slot — same convention the GPU kernel already uses
    # (inter_scratch + combined, one per thread) — needed now that this loop
    # runs across CPU threads too, not just GPU ones.
    var scratch = alloc[Intersection_C](max(n_pix * vp_samples, 1))

    @parameter
    def trace_one(combined: Int):
        var pix = combined // vp_samples
        var px = pix % Int(fw)
        var py = pix // Int(fw)
        var pcg = PCG32(seed ^ UInt64(combined * 6364136223846793005 + 1), UInt64(1))
        vps[combined] = _sppm_trace_visible_point(sd, pcg, r2c, c2w, px, py, Int32(pix), init_r2, scratch + combined)

    parallelize[trace_one](n_pix * vp_samples)

    scratch.free()


# ── Photon pass ───────────────────────────────────────────────────────────────

def _sppm_store_photon[use_gpu: Bool](
    ph:          SPPMPhoton,
    photons:     UnsafePointer[SPPMPhoton, MutAnyOrigin],
    max_photons: Int,
    counter:     UnsafePointer[Int32, MutAnyOrigin],
):
    """Reserve the next photon slot and store `ph` there, dropping it if the
    buffer is already full. Comptime-branches only on the slot-reservation
    primitive (atomic fetch-add for racing GPU threads vs. a plain
    increment for the serial CPU loop) — mirrors bdpt.mojo's
    _bdpt_store_lvc_vertex[use_gpu] exactly."""
    comptime if use_gpu:
        var slot = Int(Atomic.fetch_add(counter, Int32(1)))
        if slot < max_photons:
            photons[slot] = ph
    else:
        var slot = Int(counter[0])
        counter[0] = Int32(slot + 1)
        if slot < max_photons:
            photons[slot] = ph


def _sppm_trace_photon[use_gpu: Bool](
    sd:               SceneDescriptor2_C,
    mut pcg:          PCG32,
    scratch:          UnsafePointer[Intersection_C, MutAnyOrigin],
    n_emit:           Int,
    photons:          UnsafePointer[SPPMPhoton, MutAnyOrigin],
    max_photons:      Int,
    counter:          UnsafePointer[Int32, MutAnyOrigin],
    default_emit_med: Int32,
):
    """Emit one photon path from a random light (area, distant, or
    infinite), trace through glass/media, storing at diffuse/volume hits via
    _sppm_store_photon. n_emit is the number of emitted photon PATHS (drives
    the flux-per-photon scale factor); max_photons is the storage buffer's
    capacity, which can be smaller than the number of storage attempts
    since the Russian-roulette diffuse-diffuse continuation (see the
    diffuse-hit branch below) means one emitted path can generate several
    storage attempts, not just one. Shared verbatim between the CPU driver
    (_sppm_photon_pass, one reused `scratch`/plain counter) and the GPU
    kernel (sppm_emit_photons_gpu, one scratch slot + atomic counter per
    thread) — only _sppm_store_photon's comptime branch differs.

    Lights are chosen uniformly across ALL light types (area + distant +
    infinite) — `n_lights` below is this combined total. Distant/infinite
    lights have no finite position, so unlike an area light's surface point
    there's nothing meaningful to store as a "direct" photon at the light
    itself (SPPM never did that anyway — only diffuse/volume hits ever get
    stored); a disk-sampled point on the scene's bounding sphere (see
    bvh.mojo's _scene_bounding_sphere/_sample_disk_perpendicular) just seeds
    where this photon's path starts. Direct illumination from these lights
    is provided separately by _sppm_nee_one's own distant/infinite sampling."""
    var has_media = Int(sd.mediumCount) > 0
    var n_area = Int(sd.areaLightCount)
    var n_distant = Int(sd.distantLightCount)
    var n_infinite = Int(sd.infiniteLightCount)
    var n_lights = n_area + n_distant + n_infinite
    if n_lights == 0:
        return

    var ro: Point3f
    var rd: Vec3f
    var flux: RGB

    var light_pick = Int(pcg.next_uint() % UInt32(n_lights))
    if light_pick < n_area:
        # Pick a random area light + triangle + barycentric point on it.
        var light_sample = sample_area_light_uniform(sd.areaLights, sd.meshes, n_area, pcg, sd.curves)
        var al = light_sample.light
        var lp = light_sample.point
        var ln = light_sample.normal

        # Sample cosine-weighted emission direction from the light hemisphere
        # (same Frisvad-frame construction as _cosine_hemisphere_sample).
        var du1 = pcg.next_float()
        var du2 = pcg.next_float()
        var pdir = _cosine_hemisphere_sample(ln, du1, du2)

        # Photon flux: total_light_power / n_emit
        # total_power = emission * pi * total_area * n_lights (uniform light selection)
        var scale = PI * al.total_area * Float32(n_lights) / Float32(n_emit)
        flux = al.emission * scale
        ro = point3f(lp) + vec3f(ln) * Float32(0.0001)
        rd = vec3f(pdir)
    elif light_pick < n_area + n_distant:
        var dl = sd.distantLights[light_pick - n_area]
        var (center, radius) = _scene_bounding_sphere(sd)
        var dir = Vec3f(dl.direction.x, dl.direction.y, dl.direction.z)
        var disk_pt = _sample_disk_perpendicular(dir, center, radius, Point2f(pcg.next_float(), pcg.next_float()))
        flux = dl.emission * (Float32(n_lights) * PI * radius * radius) / Float32(n_emit)
        ro = disk_pt
        rd = dir
    else:
        var il = sd.infiniteLights[light_pick - n_area - n_distant]
        var (center, radius) = _scene_bounding_sphere(sd)
        # _sample_infinite_light_dir returns env_dir in the NEE convention
        # ("direction FROM a shading point TOWARD the light"). A photon
        # leaving the light travels the opposite way — negate for emission.
        var (env_dir, env_rgb, pdf_dir) = _sample_infinite_light_dir(il, Point2f(pcg.next_float(), pcg.next_float()))
        if pdf_dir <= Float32(0):
            return
        var emit_dir = -env_dir
        var disk_pt = _sample_disk_perpendicular(emit_dir, center, radius, Point2f(pcg.next_float(), pcg.next_float()))
        flux = env_rgb * (Float32(n_lights) * PI * radius * radius / pdf_dir) / Float32(n_emit)
        ro = disk_pt
        rd = emit_dir
    var cur_med_idx = default_emit_med  # start in medium if light is above one

    for bounce in range(_MAX_B):
        var ray = Ray_C(ro, rd)
        scratch[0].hit = Int8(0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1.0e38), scratch,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        if scratch[0].hit == Int8(0):
            break  # miss

        var inter = scratch[0]
        var ray_dir = rd.to_simd()
        var t_hit = inter.tHit

        # ── Volume free-flight ────────────────────────────────────────────
        if has_media and Int(cur_med_idx) >= 0:
            var med = sd.mediums[Int(cur_med_idx)]
            var ff = sample_homogeneous_free_flight(med, t_hit, pcg)
            if ff.collided:
                # Volume scatter — store photon and sample new direction
                var sp = ro + rd * ff.t_free
                # bounce > 0: skip storing at the light's own first
                # segment — that direct contribution is now covered
                # by _sppm_nee_update instead (matches pbrt's own
                # SPPM, which skips photon-grid gathering at depth 0
                # specifically to avoid double-counting with its NEE
                # term).
                if bounce > 0:
                    _sppm_store_photon[use_gpu](
                        SPPMPhoton(pos=sp, flux=flux, nxt=Int32(-1), is_volume=Int32(1)),
                        photons, max_photons, counter)
                # Scatter: isotropic phase function, modulate by albedo
                flux *= ff.albedo
                # Sample new isotropic direction (uniform sphere)
                var usp1 = pcg.next_float()
                var usp2 = pcg.next_float()
                var cosT = Float32(2.0) * usp1 - Float32(1.0)
                var sinT = sqrt(max(Float32(0), Float32(1) - cosT*cosT))
                var phiS = Float32(2.0) * PI * usp2
                rd = Vec3f(sinT * cos(phiS), sinT * sin(phiS), cosT)
                ro = sp + rd * Float32(0.0001)
                continue
            else:
                # Apply Beer-Lambert transmittance through segment
                flux *= ff.transmittance

        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[mat_idx]
        var hit = ro + rd * t_hit

        if mat.type == MatKind.mix:
            var mix_idx1 = Int(mat.tex_idx & Int32(0xFFFF))
            var mix_idx2 = Int((mat.tex_idx >> 16) & Int32(0xFFFF))
            var mix_amount = mat.roughU
            var mix_chosen = mix_idx2 if pcg.next_float() < mix_amount else mix_idx1
            mat = sd.materials[mix_chosen]
            if mat.type == MatKind.mix:
                mat.type = MatKind.diffuse

        if mat.type == MatKind.diffuse or mat.type == MatKind.coated_diffuse or mat.type == MatKind.diffuse_transmit:
            # bounce > 0: skip storing a photon at a surface directly hit
            # by the light with no intermediate bounce — that direct
            # contribution is now covered by _sppm_nee_update instead
            # (matches pbrt's own SPPM, which skips gathering at depth 0
            # for the same reason: avoid double-counting direct light
            # once via NEE and again via an unfiltered photon density).
            if bounce > 0:
                _sppm_store_photon[use_gpu](
                    SPPMPhoton(pos=hit, flux=flux, nxt=Int32(-1), is_volume=Int32(0)),
                    photons, max_photons, counter)
            # Russian-roulette continuation for indirect diffuse-diffuse
            # bounces (color bleeding) — without this, photons always
            # terminated at the first diffuse hit, so light could never
            # bounce off one diffuse surface onto another (e.g. a red
            # wall tinting a nearby box's facing side).
            var rr_prob = max(mat.albedo.r, max(mat.albedo.g, mat.albedo.b))
            if rr_prob <= Float32(0.0) or pcg.next_float() >= rr_prob:
                break
            var gn = _geom_normal(inter, sd.meshes, sd.instances)
            if dot(gn, ray_dir) > Float32(0.0):
                gn = gn * Float32(-1.0)
            var new_dir = _cosine_hemisphere_sample(gn, pcg.next_float(), pcg.next_float())
            flux *= mat.albedo / rr_prob
            rd = vec3f(new_dir)
            ro = hit + vec3f(gn) * Float32(0.0001)
            continue

        elif mat.type == MatKind.dielectric or mat.type == MatKind.thin_dielectric:
            var ior = mat.albedo.r
            var gn = _shading_normal_at(inter, sd.meshes, sd.instances)
            var (new_dir, new_org) = _dielectric_bounce(ray_dir, hit.to_simd(), gn, ior, bounce, pcg)
            rd = vec3f(new_dir)
            ro = point3f(new_org)
            if has_media:
                var new_idx = _sppm_update_medium(ray_dir, inter, sd.meshes, mat, sd)
                if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                    cur_med_idx = new_idx

        elif mat.type == MatKind.interface:
            if has_media:
                var new_idx = _sppm_update_medium(ray_dir, inter, sd.meshes, mat, sd)
                if new_idx != Int32(-1) or mat.medium_interface_idx >= Int32(0):
                    cur_med_idx = new_idx
            ro = hit + rd * Float32(0.0002)

        else:
            # area_light self-hit, conductor, etc.: absorb
            break


def _sppm_photon_pass(
    photons:      UnsafePointer[SPPMPhoton, MutAnyOrigin],
    n_emit:       Int,
    max_photons:  Int,
    sd:           SceneDescriptor2_C,
    seed:         UInt64,
    pass_idx:     Int,
) -> Int:
    """CPU driver: emit n_emit photon paths, returning the number actually
    stored (clamped to max_photons)."""
    var n_lights = Int(sd.areaLightCount) + Int(sd.distantLightCount) + Int(sd.infiniteLightCount)
    if n_lights == 0:
        return 0

    # One scratch Intersection_C per worker (indexed by k), same convention
    # as the GPU kernel's inter_scratch + k — needed now that this loop runs
    # across CPU threads too.
    var scratch = alloc[Intersection_C](max(n_emit, 1))
    var counter = alloc[Int32](1)
    counter[0] = Int32(0)

    # Determine the "default" starting medium for photons emitted into a medium.
    # Convention: photon is cosine-sampled from ln, so dot(pdir,ln)>0 always.
    # We use the first MediumInterface whose outside_medium_idx is valid.
    var default_emit_med = Int32(-1)
    if Int(sd.mediumCount) > 0 and Int(sd.mediumIfaceCount) > 0:
        for mi in range(Int(sd.mediumIfaceCount)):
            var iface = sd.mediumInterfaces[mi]
            if Int(iface.outside_medium_idx) >= 0:
                default_emit_med = iface.outside_medium_idx
                break

    # [True]: parallel CPU workers now need the SAME atomic slot-reservation
    # _sppm_store_photon uses for GPU threads (concurrent racing writers to
    # the shared `photons` buffer/`counter`), regardless of which backend
    # is actually running.
    @parameter
    def emit_one(k: Int):
        var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))
        _sppm_trace_photon[True](sd, pcg, scratch + k, n_emit, photons, max_photons, counter, default_emit_med)

    parallelize[emit_one](n_emit)

    var n_stored = min(Int(counter[0]), max_photons)
    counter.free()
    scratch.free()
    return n_stored


# ── Hash grid ─────────────────────────────────────────────────────────────────

def _sppm_reset_grid_cell(heads: UnsafePointer[Int32, MutAnyOrigin], h: Int):
    heads[h] = Int32(-1)


def _sppm_insert_photon[use_gpu: Bool](
    k:        Int,
    photons:  UnsafePointer[SPPMPhoton, MutAnyOrigin],
    heads:    UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
):
    """Insert stored photon `k` into the hash grid. Comptime-branches only on
    the bucket-head update primitive (atomic exchange for racing GPU threads
    vs. a plain read-modify-write for the serial CPU loop)."""
    var ix = Int(floor(photons[k].pos.x * inv_cell))
    var iy = Int(floor(photons[k].pos.y * inv_cell))
    var iz = Int(floor(photons[k].pos.z * inv_cell))
    var h = _hash_cell(ix, iy, iz)
    comptime if use_gpu:
        var old = Atomic._xchg(heads + h, Int32(k))
        photons[k].nxt = old
    else:
        photons[k].nxt = heads[h]
        heads[h] = Int32(k)


def _build_grid(
    photons:  UnsafePointer[SPPMPhoton, MutAnyOrigin],
    n_phot:   Int,
    heads:    UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
):
    @parameter
    def reset_one(i: Int):
        _sppm_reset_grid_cell(heads, i)

    parallelize[reset_one](_HSIZE)

    # [True]: parallel CPU workers race on the same bucket heads a GPU
    # kernel's threads would, so need the same atomic-exchange insert.
    @parameter
    def insert_one(k: Int):
        _sppm_insert_photon[True](k, photons, heads, inv_cell)

    parallelize[insert_one](n_phot)


# ── Gather + SPPM update ──────────────────────────────────────────────────────

def _sppm_gather_one(
    vps:      UnsafePointer[SPPMPixel, MutAnyOrigin],
    i:        Int,
    photons:  UnsafePointer[SPPMPhoton, MutAnyOrigin],
    heads:    UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
):
    """Gather nearby photons into visible point `i` and apply the SPPM
    radius/flux update. Shared verbatim between the CPU driver
    (_gather_update, one call per pixel in a loop) and the GPU kernel
    (sppm_gather_gpu, one call per thread) — no divergence, each call only
    ever touches its own vps[i]."""
    if vps[i].valid == Int32(0):
        return
    var vp = vps[i]
    var r2 = vp.r2

    # Accumulate contributions from photons in 3x3x3 neighborhood
    var phi = RGB(Float32(0))
    var M = Float32(0)

    var cix = Int(floor(vp.pos.x * inv_cell))
    var ciy = Int(floor(vp.pos.y * inv_cell))
    var ciz = Int(floor(vp.pos.z * inv_cell))
    for ddx in range(-1, 2):
        for ddy in range(-1, 2):
            for ddz in range(-1, 2):
                var h = _hash_cell(cix + ddx, ciy + ddy, ciz + ddz)
                var k = Int(heads[h])
                while k != -1:
                    var ph = photons[k]
                    var e = ph.pos - vp.pos
                    var dist2 = e.length_sq()
                    if dist2 <= r2 and ph.is_volume == vp.is_volume:
                        # Surface VP: Lambertian f=alb/π; Volume VP: isotropic phase f=alb/(4π)
                        var f: Float32
                        if vp.is_volume == Int32(1):
                            f = INV_FOUR_PI
                        else:
                            f = Float32(1.0) / PI
                        phi += (vp.alb * f) * ph.flux
                        M += Float32(1.0)
                    k = Int(ph.nxt)

    # SPPM update (only if new photons found)
    if M > Float32(0.0):
        var N = vp.N_acc
        var ratio = (N + _ALPHA * M) / (N + M)
        vps[i].r2  = r2 * ratio
        vps[i].tau = (vp.tau + phi) * ratio
        vps[i].N_acc = N + _ALPHA * M


def _gather_update(
    vps:      UnsafePointer[SPPMPixel, MutAnyOrigin],
    n_pix:    Int,
    photons:  UnsafePointer[SPPMPhoton, MutAnyOrigin],
    heads:    UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
):
    @parameter
    def gather_one(i: Int):
        _sppm_gather_one(vps, i, photons, heads, inv_cell)

    parallelize[gather_one](n_pix)


# ── Direct (NEE) lighting update ──────────────────────────────────────────────
# Mirrors pbrt-v4's SPPM "pixel.Ld" term: a shadow-ray light sample taken at
# each visible point EVERY SPPM pass (same cadence as the photon pass),
# summed here and divided by n_passes at finalize time. Needed because the
# photon-density (tau) term alone has to represent BOTH direct and indirect
# lighting through one noisy channel — pbrt keeps them separate, which is
# why its SPPM output is much smoother/less "blotchy" for the same pass
# count. Surface VPs only (is_volume == 0); this scene has no participating
# media, so a phase-function-weighted variant for volume VPs isn't needed.

def _sppm_nee_one(
    vps:     UnsafePointer[SPPMPixel, MutAnyOrigin],
    i:       Int,
    sd:      SceneDescriptor2_C,
    mut pcg: PCG32,
):
    """Direct (NEE) lighting update for visible point `i`. Shared verbatim
    between the CPU driver (_sppm_nee_update, one call per VP in a loop) and
    the GPU kernel (sppm_nee_gpu, one call per thread) — no divergence, each
    call only ever touches its own vps[i]. Samples area lights (one random
    CDF-uniform pick), distant lights, and infinite lights (all of the
    latter two — few and typically dominant, matching shading.mojo's own
    NEE asymmetry rationale) — no MIS weighting needed for infinite lights
    here, unlike bdpt.mojo's camera-path NEE: a VP is only ever `valid` when
    its ONE traced ray hit a real surface, and _sppm_trace_visible_point's
    own miss-escape env contribution (vp.env) only fires when it didn't —
    mutually exclusive per sample, so there's no competing strategy to
    double-count against."""
    var vp = vps[i]
    if vp.valid == Int32(0) or vp.is_volume == Int32(1):
        return
    var vpos = vp.pos.to_simd()
    var vn   = vp.normal.to_simd()

    var n_area = Int(sd.areaLightCount)
    if n_area > 0:
        # Pick a random area light + triangle + point on it (same scheme as
        # _sppm_trace_photon's emission sampling).
        var light_sample = sample_area_light_uniform(sd.areaLights, sd.meshes, n_area, pcg, sd.curves)
        var al = light_sample.light
        var lp = light_sample.point
        var ln = light_sample.normal

        var to_light = lp - vpos
        var dist2 = dot(to_light, to_light)
        var dist = sqrt(dist2)
        if dist > Float32(0.0):
            var wi = to_light * (Float32(1.0) / dist)
            var cos_surface = dot(vn, wi)
            var cos_light = -dot(ln, wi)
            if cos_surface > Float32(0.0) and cos_light > Float32(0.0):
                # Shadow ray, offset from both ends to avoid self-intersection.
                var shadow_org = vp.pos + vp.normal * Float32(0.0001)
                var shadow_ray = Ray_C(shadow_org, vec3f(wi))
                var t_max = dist * Float32(0.999)
                if not any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, shadow_ray, t_max,
                                      sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances):
                    # pdf_area = 1/(n_area * total_area) — uniform-over-all-lights
                    # assumption, same as the emission-sampling flux scale factor.
                    var inv_pdf_area = Float32(n_area) * al.total_area
                    var geom = cos_surface * cos_light / dist2 * inv_pdf_area
                    var brdf = vp.alb / PI
                    vps[i].ld += (brdf * al.emission) * geom

    for dl_i in range(Int(sd.distantLightCount)):
        var dl = sd.distantLights[dl_i]
        var to_light_d = Vec3f(-dl.direction.x, -dl.direction.y, -dl.direction.z)
        var cos_s = dot(vn, to_light_d.to_simd())
        if cos_s > Float32(0.0):
            var shadow_org_d = vp.pos + vp.normal * Float32(0.0001)
            var shadow_ray_d = Ray_C(shadow_org_d, to_light_d)
            if not any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, shadow_ray_d, Float32(2000),
                                  sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances):
                vps[i].ld += (vp.alb / PI) * dl.emission * cos_s

    for inf_i in range(Int(sd.infiniteLightCount)):
        var ilight = sd.infiniteLights[inf_i]
        var (env_dir, env_rgb, pdf_env) = _sample_infinite_light_dir(ilight, Point2f(pcg.next_float(), pcg.next_float()))
        var cos_env = dot(vn, env_dir.to_simd())
        if cos_env > Float32(0.0) and pdf_env > Float32(0.0):
            var shadow_org_e = vp.pos + vp.normal * Float32(0.0001)
            var shadow_ray_e = Ray_C(shadow_org_e, env_dir)
            if not any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, shadow_ray_e, Float32(2000),
                                  sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances):
                vps[i].ld += (vp.alb / PI) * env_rgb * (cos_env / pdf_env)


def _sppm_nee_update(
    vps:     UnsafePointer[SPPMPixel, MutAnyOrigin],
    n_vps:   Int,
    sd:      SceneDescriptor2_C,
    seed:    UInt64,
    pass_idx: Int,
):
    var n_lights = Int(sd.areaLightCount) + Int(sd.distantLightCount) + Int(sd.infiniteLightCount)
    if n_lights == 0:
        return

    @parameter
    def nee_one(i: Int):
        var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + i), UInt64(11))
        _sppm_nee_one(vps, i, sd, pcg)

    parallelize[nee_one](n_vps)


# ── Finalize ──────────────────────────────────────────────────────────────────

def _sppm_finalize_one_pixel(
    vps:        UnsafePointer[SPPMPixel, MutAnyOrigin],
    i:          Int,
    vp_samples: Int,
    n_passes:   Int32,
    iso_scale:  Float32,
    max_comp:   Float32,
) -> RGB:
    """Averages the vp_samples independently-converged samples for pixel i —
    see _sppm_trace_visible_point's docstring for why each sample has its
    own fixed visible point/accumulator rather than sharing one per pixel.
    Shared verbatim between the CPU driver (sppm_render's tail loop) and the
    GPU kernel (sppm_finalize_gpu)."""
    var acc = RGB(Float32(0))
    for vs in range(vp_samples):
        var vp = vps[i * vp_samples + vs]
        if vp.valid == Int32(0):
            # No surface hit — either a dead sample, or the traced ray
            # escaped the scene into an infinite (environment) light, whose
            # already beta-weighted radiance _sppm_trace_visible_point
            # stored directly in vp.env (0 if neither happened).
            acc += vp.env
            continue
        var vp_l = RGB(Float32(0))
        if vp.N_acc > Float32(0.0) and vp.r2 > Float32(0.0):
            # L = tau / (pi * r² * n_passes) — tau already has
            # albedo/pi folded in at gather time
            var denom = PI * vp.r2 * Float32(n_passes)
            vp_l += vp.tau / denom
        if vp.is_volume == Int32(0):
            # Direct (NEE) term — pbrt's "pixel.Ld", resampled once per
            # pass, averaged over n_passes.
            vp_l += vp.ld / Float32(n_passes)
        acc += vp.beta * vp_l
    acc = acc / Float32(vp_samples)

    # ISO exposure compensation (matches normalize_film)
    acc *= iso_scale

    # NaN guard and optional max-component clamp
    if acc.r != acc.r or acc.r < Float32(0): acc.r = Float32(0)
    if acc.g != acc.g or acc.g < Float32(0): acc.g = Float32(0)
    if acc.b != acc.b or acc.b < Float32(0): acc.b = Float32(0)
    if max_comp > Float32(0):
        var mx = max(acc.r, max(acc.g, acc.b))
        if mx > max_comp:
            acc *= max_comp / mx
    return acc


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

    if Int(sd.areaLightCount) + Int(sd.distantLightCount) + Int(sd.infiniteLightCount) == 0:
        print("SPPM: no lights in scene, cannot emit photons")
        return Int32(-1)

    print("SPPM: " + String(fw) + "x" + String(fh)
          + " " + String(n_passes) + " passes x "
          + String(n_photons_per_pass) + " photons  r=" + String(initial_radius))

    # Allocate visible points (n_pix * _VP_SAMPLES independent samples) and photon buffer.
    # max_photons intentionally equals n_photons_per_pass, NOT a multiple of
    # it: letting every Russian-roulette diffuse-diffuse continuation event
    # (see _sppm_photon_pass) store unconditionally was tried and measurably
    # over-brightened the render (each stored event's flux isn't re-weighted
    # for "there are now more data points per emitted photon than the
    # density-estimation formula assumes") — the cap at n_photons_per_pass,
    # dropping excess bounce events once the buffer fills, is not a bug.
    var n_vps    = n_pix * _VP_SAMPLES
    var vps     = alloc[SPPMPixel](n_vps)
    var max_photons = n_photons_per_pass
    var photons = alloc[SPPMPhoton](max_photons)
    var heads   = alloc[Int32](_HSIZE)
    var init_r2 = initial_radius * initial_radius
    var inv_cell = Float32(1.0) / initial_radius  # cell size == initial radius

    # Trace the camera/visible-point samples ONCE for the whole render — see
    # _sppm_camera_pass's docstring for why a per-pass re-trace (the old
    # design) breaks SPPM's convergence guarantee.
    var cam_seed = psc[0].rng_seed ^ UInt64(0x9E3779B97F4A7C15 + 7)
    _sppm_camera_pass(
        vps, n_pix, _VP_SAMPLES, psc[0].film_w,
        psc[0].raster_to_camera, psc[0].camera_to_world,
        sd, init_r2, cam_seed,
    )
    if verbose:
        var n_valid = 0
        for i in range(n_vps):
            if vps[i].valid != Int32(0): n_valid += 1
        print("SPPM: " + String(n_valid) + "/" + String(n_vps) + " visible points found")

    # Photon passes
    for pass_idx in range(n_passes):
        var pass_seed = psc[0].rng_seed ^ UInt64(pass_idx * 2654435761 + 1)
        var n_stored = _sppm_photon_pass(photons, n_photons_per_pass, max_photons, sd, pass_seed, pass_idx)
        if n_stored > 0:
            _build_grid(photons, n_stored, heads, inv_cell)
            _gather_update(vps, n_vps, photons, heads, inv_cell)
        var nee_seed = psc[0].rng_seed ^ UInt64(pass_idx * 0xBF58476D1CE4E5B9 + 3)
        _sppm_nee_update(vps, n_vps, sd, nee_seed, pass_idx)
        if verbose or (pass_idx + 1) % 10 == 0:
            print("SPPM: pass " + String(pass_idx + 1) + "/" + String(n_passes)
                  + " stored=" + String(n_stored), end="\r")

    print("")  # newline after progress

    # Assemble output image: average the _VP_SAMPLES independently-converged
    # samples per pixel (each sample's own r2/tau/N_acc converges correctly
    # since its position/surface is fixed for the whole render — see
    # _sppm_camera_pass's docstring). A sample that never found a diffuse/
    # volume hit (N_acc == 0, e.g. it reflected off the water into the void)
    # contributes 0 for that sample, same as a path tracer sample that misses
    # everything — the average over all samples is what correctly reproduces
    # the fresnel-weighted reflect/refract blend a real specular interface
    # would show.
    var out_pixels = alloc[Float32](n_pix * 3)

    @parameter
    def finalize_one(i: Int):
        var acc = _sppm_finalize_one_pixel(vps, i, _VP_SAMPLES, Int32(n_passes), iso_scale, max_comp)
        out_pixels[i * 3 + 0] = acc.r
        out_pixels[i * 3 + 1] = acc.g
        out_pixels[i * 3 + 2] = acc.b

    parallelize[finalize_one](n_pix)

    _ = write_image(out_pixels, psc[0].film_w, psc[0].film_h,
                    psc[0].film_filename, Int32(32), Int32(32))

    out_pixels.free()
    heads.free()
    photons.free()
    vps.free()
    return Int32(0)


# ── GPU kernels ───────────────────────────────────────────────────────────────
# Each kernel is a thin wrapper: compute this thread's index, build a complete
# SceneDescriptor2_C via _mk_sd_full (bvh.mojo), then call the EXACT SAME
# shared function the CPU driver above calls (comptime[use_gpu]-branching
# only at the two genuine concurrency-primitive divergence points: photon-
# slot reservation and hash-grid bucket insertion) — mirrors bdpt.mojo's
# GPU kernels, deliberately unlike the old gpu_sppm.mojo (a full duplicate
# reimplementation).

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
    init_r2: Float32,
    seed: UInt64,
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int64,
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    curveCount: Int64,
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    mediumCount: Int64,
    mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    blasCount: Int64,
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    instanceCount: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    distantLightCount: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    infiniteLightCount: Int64,
):
    """One thread per (pixel, vp_sample). Calls the SAME
    _sppm_trace_visible_point the CPU driver (_sppm_camera_pass) calls."""
    var combined = Int(block_idx.x * block_dim.x + thread_idx.x)
    if combined >= n_pix * vp_samples:
        return
    var pix = combined // vp_samples
    var px = pix % Int(fw)
    var py = pix // Int(fw)
    var sd = _mk_sd_full(
        bvh2Nodes, primIds, meshes, Int64(0), materials, Int64(0),
        areaLights, areaLightCount, spheres, sphereCount, curves, curveCount,
        mediums, mediumCount, mediumInterfaces, mediumIfaceCount,
        blasNodesArr, blasPrimIdsArr, blasCount, instances, instanceCount,
        distantLights, distantLightCount, infiniteLights, infiniteLightCount,
    )
    var pcg = PCG32(seed ^ UInt64(combined * 6364136223846793005 + 1), UInt64(1))
    vps[combined] = _sppm_trace_visible_point(sd, pcg, r2c, c2w, px, py, Int32(pix), init_r2, inter_scratch + combined)


def sppm_emit_photons_gpu(
    photons: UnsafePointer[SPPMPhoton, MutAnyOrigin],
    n_emit: Int,
    max_photons: Int,
    inter_scratch: UnsafePointer[Intersection_C, MutAnyOrigin],
    stored_counter: UnsafePointer[Int32, MutAnyOrigin],
    default_emit_med: Int32,
    seed: UInt64,
    pass_idx: Int,
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int64,
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    curveCount: Int64,
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    mediumCount: Int64,
    mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    blasCount: Int64,
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    instanceCount: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    distantLightCount: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    infiniteLightCount: Int64,
):
    """One thread per emitted photon path. Calls the SAME _sppm_trace_photon
    the CPU driver (_sppm_photon_pass) calls, with use_gpu=True so
    _sppm_store_photon reserves its slot via an atomic fetch-add (CPU uses a
    plain counter increment instead — no other difference)."""
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= n_emit or (areaLightCount == Int64(0) and distantLightCount == Int64(0) and infiniteLightCount == Int64(0)):
        return
    var sd = _mk_sd_full(
        bvh2Nodes, primIds, meshes, Int64(0), materials, Int64(0),
        areaLights, areaLightCount, spheres, sphereCount, curves, curveCount,
        mediums, mediumCount, mediumInterfaces, mediumIfaceCount,
        blasNodesArr, blasPrimIdsArr, blasCount, instances, instanceCount,
        distantLights, distantLightCount, infiniteLights, infiniteLightCount,
    )
    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + k), UInt64(7))
    _sppm_trace_photon[True](sd, pcg, inter_scratch + k, n_emit, photons, max_photons, stored_counter, default_emit_med)


def sppm_grid_reset_gpu(heads: UnsafePointer[Int32, MutAnyOrigin], hsize: Int):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= hsize:
        return
    _sppm_reset_grid_cell(heads, tid)


def sppm_grid_insert_gpu(
    photons:  UnsafePointer[SPPMPhoton, MutAnyOrigin],
    n_stored: Int,
    heads:    UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
):
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= n_stored:
        return
    _sppm_insert_photon[True](k, photons, heads, inv_cell)


def sppm_gather_gpu(
    vps:      UnsafePointer[SPPMPixel, MutAnyOrigin],
    n_pix:    Int,
    photons:  UnsafePointer[SPPMPhoton, MutAnyOrigin],
    heads:    UnsafePointer[Int32, MutAnyOrigin],
    inv_cell: Float32,
):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n_pix:
        return
    _sppm_gather_one(vps, i, photons, heads, inv_cell)


def sppm_nee_gpu(
    vps:    UnsafePointer[SPPMPixel, MutAnyOrigin],
    n_vps:  Int,
    seed:   UInt64,
    pass_idx: Int,
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int64,
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    curveCount: Int64,
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    mediumCount: Int64,
    mediumInterfaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    mediumIfaceCount: Int64,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    blasCount: Int64,
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    instanceCount: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    distantLightCount: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    infiniteLightCount: Int64,
):
    """One thread per visible point. Calls the SAME _sppm_nee_one the CPU
    driver (_sppm_nee_update) calls."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n_vps:
        return
    var sd = _mk_sd_full(
        bvh2Nodes, primIds, meshes, Int64(0), materials, Int64(0),
        areaLights, areaLightCount, spheres, sphereCount, curves, curveCount,
        mediums, mediumCount, mediumInterfaces, mediumIfaceCount,
        blasNodesArr, blasPrimIdsArr, blasCount, instances, instanceCount,
        distantLights, distantLightCount, infiniteLights, infiniteLightCount,
    )
    var pcg = PCG32(seed ^ UInt64(pass_idx * 1000003 + i), UInt64(11))
    _sppm_nee_one(vps, i, sd, pcg)


def sppm_finalize_gpu(
    vps:        UnsafePointer[SPPMPixel, MutAnyOrigin],
    n_pix:      Int,
    vp_samples: Int,
    n_passes:   Int32,
    iso_scale:  Float32,
    max_comp:   Float32,
    out_pixels: UnsafePointer[Float32, MutAnyOrigin],
):
    """One thread per pixel. Calls the SAME _sppm_finalize_one_pixel the CPU
    driver (sppm_render's tail loop) calls."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n_pix:
        return
    var acc = _sppm_finalize_one_pixel(vps, i, vp_samples, n_passes, iso_scale, max_comp)
    out_pixels[i * 3 + 0] = acc.r
    out_pixels[i * 3 + 1] = acc.g
    out_pixels[i * 3 + 2] = acc.b


# ── GPU host driver ───────────────────────────────────────────────────────────

def sppm_render_gpu(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    psc:      UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
    sd:       SceneDescriptor2_C,
    n_passes: Int,
    n_photons_per_pass: Int,
    initial_radius: Float32,
    no_denoise: Bool,
    verbose:  Bool,
) -> Int32:
    """GPU-accelerated Stochastic Progressive Photon Mapping — same algorithm
    as sppm_render, parallelized: one thread per visible-point sample for the
    camera pass/gather/NEE/finalize, one thread per emitted photon for the
    photon pass, atomic-exchange hash-grid build (classic parallel linked-list
    insertion). Mirrors bdpt_render_gpu's per-pass reset-counter -> emit ->
    sync+readback+clamp -> consume shape."""
    if Int(sd.areaLightCount) + Int(sd.distantLightCount) + Int(sd.infiniteLightCount) == 0:
        print("SPPM: no lights in scene, cannot emit photons")
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
            # sppm_render's docstring for why a larger buffer (letting every
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
            var distantLights = handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C]()
            var infiniteLights = handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C]()
            var n_mediums = Int64(handle[].n_mediums)
            var n_medium_ifaces = Int64(handle[].n_medium_ifaces)
            var n_spheres = Int64(handle[].n_spheres)
            var n_curves = Int64(handle[].n_curves)
            var n_area_lights = Int64(handle[].n_area_lights)
            var n_distant_lights = Int64(handle[].n_distant_lights)
            var n_infinite_lights = Int64(handle[].n_infinite_lights)
            var n_blas = Int64(handle[].n_blas)
            var n_instances = Int64(handle[].n_instances)

            var grid_pix = ceildiv(n_pix, block_size)
            var grid_vps = ceildiv(n_vps, block_size)
            var grid_hsize = ceildiv(_HSIZE, block_size)

            # Camera/visible-point samples are traced ONCE for the whole
            # render, not per SPPM pass — see _sppm_trace_visible_point's
            # docstring for why a per-pass re-trace breaks SPPM's
            # convergence guarantee.
            var cam_seed = psc[0].rng_seed ^ UInt64(0x9E3779B97F4A7C15 + 7)
            handle[].ctx.enqueue_function[sppm_gen_vp_gpu](
                vps_ptr, inter_cam_ptr, n_pix, _VP_SAMPLES, psc[0].film_w, r2c_ptr, c2w_ptr,
                init_r2, cam_seed,
                bvh2Nodes, primIds, meshes, materials,
                areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
                grid_dim=grid_vps, block_dim=block_size)

            for pass_idx in range(n_passes):
                handle[].ctx.enqueue_function[sppm_reset_i32_gpu](
                    counter_ptr, grid_dim=1, block_dim=1)

                var pass_seed = psc[0].rng_seed ^ UInt64(pass_idx * 2654435761 + 1)
                var grid_emit = ceildiv(max(n_photons_per_pass, 1), block_size)
                handle[].ctx.enqueue_function[sppm_emit_photons_gpu](
                    photons_ptr, n_photons_per_pass, max_photons, inter_ph_ptr, counter_ptr,
                    default_emit_med, pass_seed, pass_idx,
                    bvh2Nodes, primIds, meshes, materials,
                    areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                    mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                    blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                    distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
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
                    vps_ptr, n_vps, nee_seed, pass_idx,
                    bvh2Nodes, primIds, meshes, materials,
                    areaLights, n_area_lights, spheres, n_spheres, curves, n_curves,
                    mediums, n_mediums, mediumInterfaces, n_medium_ifaces,
                    blasNodesArr, blasPrimIdsArr, n_blas, instances, n_instances,
                    distantLights, n_distant_lights, infiniteLights, n_infinite_lights,
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
