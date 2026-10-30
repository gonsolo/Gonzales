# BDPT/VCM next-event estimation: shadow-ray transmittance and simple lights (MNEE: bdpt_eval._vcm_mnee).
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.collections import Array
from std.math import abs
from .gpu_media import medium_after
from .geometry import Point3f, Vec3f, vec3f, dot
from .materials import MatKind, fr_dielectric
from .primitives import Ray, Intersection, PrimId
from .shadow_media import segment_transmittance
from .bvh import traverse_dispatch
from .bvh import (
    SceneView, test_spheres, LightSample, _sample_distant_light_nee,
    _sample_point_light_nee, _sample_sphere_light_nee,
)
from .rng import PCG32
from .sppm import _geom_normal
from .spectrum import SampledWavelengths, SpectralSample

@always_inline
# ── Visibility with transmittance ─────────────────────────────────────────────

def _visible_transmittance(
    a: Point3f, b: Point3f,
    med_idx: Int32,
    ref sd:      SceneView,
    scratch: Pointer[Intersection, MutUntrackedOrigin],
    wl:      SampledWavelengths,
) -> SpectralSample:
    """Returns transmittance along segment AB, spectrally, or black if
    occluded. Glass (dielectric) surfaces are passed through with Fresnel"""
    var d = b - a
    var dist_total = d.length()
    if dist_total < Float32(1e-5):
        return SpectralSample(Float32(0))
    var inv = Float32(1) / dist_total
    var dir = Vec3f(d.x*inv, d.y*inv, d.z*inv)

    var Tr = SpectralSample(Float32(1.0))
    var org = a + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
    var remaining = dist_total - Float32(0.0002)
    var cur_med = med_idx
    # Ratio tracking through grid media needs draws; seed from the endpoints so the call keeps its signature.
    var pcg = PCG32(UInt64(Int(abs(a.x) * Float32(4096.0))) * UInt64(73856093) ^ UInt64(Int(abs(a.y) * Float32(4096.0))) * UInt64(19349663) ^ UInt64(Int(abs(b.z) * Float32(4096.0))) * UInt64(83492791), UInt64(Int(abs(b.x) * Float32(4096.0))) + UInt64(Int(abs(b.y) * Float32(4096.0))))

    # Private local slot instead of the caller's `scratch`. The caller's slot
    # is simultaneously live in the enclosing traversal that called us, and
    # writing through both aliases is what the GPU build faults on.
    var _local_inter = Array[Intersection, 1](fill=Intersection(
        PrimId(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0)),
        Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0)))
    var inter_mem = _local_inter.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    for _ in range(8):
        if remaining < Float32(1e-4): break
        var ray = Ray(org, Vec3f(dir[0], dir[1], dir[2]))
        inter_mem[unsafe_offset=0].hit = Int8(0)
        traverse_dispatch(sd.bvh4, sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, remaining * Float32(0.9995), inter_mem,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        # test_spheres (analytic spheres, e.g. the caustic sphere) aren't part of
        # the BVH — traverse_bvh2_core only tests triangles/curves — so they need
        var had_bvh_hit = inter_mem[unsafe_offset=0].hit != Int8(0)
        if not had_bvh_hit:
            inter_mem[unsafe_offset=0].hit = Int8(1)
            inter_mem[unsafe_offset=0].tHit = remaining * Float32(0.9995)
            # Clear the primId along with the sentinel. `scratch` is a
            # caller-owned slot reused across bounces, samples and (on GPU)
            inter_mem[unsafe_offset=0].primId.type = Int8(0)
        test_spheres(sd.spheres, Int(sd.sphereCount), ray, inter_mem)
        if not had_bvh_hit and inter_mem[unsafe_offset=0].primId.type != Int8(4):
            inter_mem[unsafe_offset=0].hit = Int8(0)
        if inter_mem[unsafe_offset=0].hit == Int8(0):
            # Nothing between here and destination: apply remaining Beer-Lambert
            if Int(cur_med) >= 0:
                Tr *= segment_transmittance(sd.mediums[unsafe_offset=Int(cur_med)], org.to_simd(), dir, remaining, sd.grids, sd.nvdbGrids, sd.spectral, wl, pcg)
            break

        var inter = inter_mem[unsafe_offset=0]
        var t_hit = inter.tHit
        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[unsafe_offset=mat_idx]
        var hit = org + Vec3f(dir[0], dir[1], dir[2]) * t_hit

        # Beer-Lambert through medium segment up to hit
        if Int(cur_med) >= 0:
            Tr *= segment_transmittance(sd.mediums[unsafe_offset=Int(cur_med)], org.to_simd(), dir, t_hit, sd.grids, sd.nvdbGrids, sd.spectral, wl, pcg)

        if mat.type == MatKind.thin_dielectric:
            var gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
            var facing = dot(dir, gn) < Float32(0)
            var n_for_cos = gn if facing else gn*Float32(-1)
            var cos_i = -dot(dir, n_for_cos)
            if cos_i < Float32(0): cos_i = -cos_i
            var ior = mat.albedo.r
            var fr = fr_dielectric(cos_i, Float32(1)/ior if facing else ior)
            var T = Float32(1) - fr
            Tr *= T
            if Tr.is_black():
                return SpectralSample(Float32(0))
            # Update medium after crossing glass surface
            cur_med = medium_after(dir, cur_med, inter, mat, sd, hit.to_simd())
            org = hit + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
            remaining = remaining - t_hit - Float32(0.0002)

        elif mat.type == MatKind.interface:
            # Pure medium boundary: update medium, continue
            cur_med = medium_after(dir, cur_med, inter, mat, sd, hit.to_simd())
            org = hit + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
            remaining = remaining - t_hit - Float32(0.0002)

        else:
            # Opaque surface blocks the segment
            return SpectralSample(Float32(0))

    return Tr

@always_inline
def _bdpt_simple_light_count(ref sd: SceneView) -> Int:
    """Number of lights reachable through _bdpt_sample_simple_light: distant +
    point + sphere, the three every BDPT material-loop samples the SAME way."""
    return Int(sd.distantLightCount) + Int(sd.pointLightCount) + Int(sd.sphereCount)


@always_inline
def _bdpt_sample_simple_light(
    ref sd: SceneView, i: Int, hit_point: Vec3f, mut pcg: PCG32,
) -> LightSample:
    """The i-th distant/point/sphere light. Unlike shading.mojo's twin
    (_nee_sample_simple_light), this returns ONLY the LightSample -- BDPT's"""
    var nd = Int(sd.distantLightCount)
    var np_ = Int(sd.pointLightCount)
    if i < nd:
        var ls_d = _sample_distant_light_nee(sd.distantLights[unsafe_offset=i])
        return ls_d^
    if i < nd + np_:
        var ls_p = _sample_point_light_nee(sd.pointLights[unsafe_offset=i - nd], hit_point)
        return ls_p^
    var si = i - nd - np_
    var ls_s = _sample_sphere_light_nee(sd.spheres[unsafe_offset=si], Int(sd.sphereCount), hit_point, pcg)
    return ls_s^


@always_inline
def _bdpt_nee_contribute(
    beta: SpectralSample,
    w: SpectralSample,
    ls: LightSample,
    hit: Point3f,
    gn: Vec3f,
    cur_med_idx: Int32,
    ref sd: SceneView,
    scratch: Pointer[Intersection, MutUntrackedOrigin],
    wl: SampledWavelengths,
    eps: Float32 = Float32(0.0001),
    two_sided: Bool = False,
) -> SpectralSample:
    """BDPT-side NEE glue shared by every per-material light loop below:
    given a LightSample + material weight (from the shared Light interface"""
    if w.is_black():
        return SpectralSample(Float32(0))
    # `gn` is the GEOMETRIC normal (pbrt's OffsetRayOrigin), never the shading
    # one: that is turned toward wo (face_toward) and can point into the
    var side_eps = eps
    if two_sided and dot(gn, Vec3f(ls.wi[0], ls.wi[1], ls.wi[2])) < Float32(0):
        side_eps = -eps
    var shadow_org = hit + vec3f(gn) * side_eps
    # Stop 3 offsets short: shadow_org sits `eps` off the surface, so a light closer than
    # eps / 0.0005 (0.2 units at eps=1e-4) would otherwise end past its own surface and occlude itself.
    var shadow_end = shadow_org + Vec3f(ls.wi[0], ls.wi[1], ls.wi[2]) * (ls.dist - Float32(3) * abs(side_eps))
    var Tr = _visible_transmittance(shadow_org, shadow_end, cur_med_idx, sd, scratch, wl)
    if not Tr.is_black():
        return beta * w * Tr
    return SpectralSample(Float32(0))
