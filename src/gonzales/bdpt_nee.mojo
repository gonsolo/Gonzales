# BDPT/VCM next-event estimation: shadow-ray transmittance, simple lights, MNEE through glass.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.collections import Array
from std.math import sqrt, cos, sin, exp, max, min, abs
from .geometry import RGB, Point3f, Vec3f, vec3f, dot, cross, PI
from .materials import MatKind, fr_dielectric, is_specular_glass
from .primitives import Ray, Intersection, PrimId
from .media import medium_sigma_t_spectral
from .lights import area_light_pick_triangle
from .bvh import (
    SceneView, traverse_bvh2_core, any_hit_bvh2_core, test_spheres, LightSample, _sample_distant_light_nee,
    _sample_point_light_nee, _sample_sphere_light_nee,
)
from .rng import PCG32
from .sppm import _geom_normal
from .shading import _get_tri_verts, _mnee_walk, _mnee_walk2
from .bxdf import bxdf_eval_diffuse
from .spectrum import SampledWavelengths, SpectralSample, spec_refl, spec_illum

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
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, remaining * Float32(0.9995), inter_mem,
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
                var med = sd.mediums[unsafe_offset=Int(cur_med)]
                var st_spec = medium_sigma_t_spectral(med, wl, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                Tr *= SpectralSample(exp(-st_spec.v0*remaining), exp(-st_spec.v1*remaining), exp(-st_spec.v2*remaining), exp(-st_spec.v3*remaining))
            break

        var inter = inter_mem[unsafe_offset=0]
        var t_hit = inter.tHit
        var mat_idx = Int(inter.primId.materialIndex)
        var mat = sd.materials[unsafe_offset=mat_idx]
        var hit = org + Vec3f(dir[0], dir[1], dir[2]) * t_hit

        # Beer-Lambert through medium segment up to hit
        if Int(cur_med) >= 0:
            var med = sd.mediums[unsafe_offset=Int(cur_med)]
            var st_spec = medium_sigma_t_spectral(med, wl, sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
            Tr *= SpectralSample(exp(-st_spec.v0*t_hit), exp(-st_spec.v1*t_hit), exp(-st_spec.v2*t_hit), exp(-st_spec.v3*t_hit))

        if mat.type == MatKind.thin_dielectric or (
                mat.type == MatKind.dielectric and mat.sss_boundary != Int8(0)):
            # ... and a SUBSURFACE boundary, which is a dielectric too but
            # whose transport the BSSRDF models separately -- blocking it
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
            if mat.medium_interface_idx >= Int32(0) and sd.mediumIfaceCount > Int64(0):
                var iface = sd.mediumInterfaces[unsafe_offset=Int(mat.medium_interface_idx)]
                var md = dir[0]*gn[0]+dir[1]*gn[1]+dir[2]*gn[2]
                cur_med = iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx
            org = hit + Vec3f(dir[0], dir[1], dir[2]) * Float32(0.0002)
            remaining = remaining - t_hit - Float32(0.0002)

        elif mat.type == MatKind.interface:
            # Pure medium boundary: update medium, continue
            if mat.medium_interface_idx >= Int32(0) and sd.mediumIfaceCount > Int64(0):
                var iface = sd.mediumInterfaces[unsafe_offset=Int(mat.medium_interface_idx)]
                # An ANALYTIC SPHERE boundary has no mesh to read a normal
                # from -- _geom_normal would index sd.meshes with a sphere's
                var igna = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit.to_simd())
                var md = dir[0]*igna[0]+dir[1]*igna[1]+dir[2]*igna[2]
                cur_med = iface.outside_medium_idx if md > Float32(0) else iface.inside_medium_idx
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

def _bdpt_mnee_diffuse_area_light(
    ref sd: SceneView, hit: Point3f, gn: Vec3f, eff_alb: RGB,
    beta: SpectralSample, mut pcg: PCG32, wl: SampledWavelengths,
    ior: Float32 = Float32(1.0),
) -> SpectralSample:
    """Real MNEE (manifold next-event estimation, task #161): for a diffuse
    (or coateddiffuse base-layer, see `ior` below) camera vertex, probe"""
    var n_area = Int(sd.areaLightCount)
    if n_area <= 0:
        return SpectralSample(Float32(0))
    var li = Int(pcg.next_uint() % UInt32(n_area))
    var al = sd.areaLights[unsafe_offset=li]
    if al.kind == Int8(1):
        return SpectralSample(Float32(0))
    var lmesh = sd.meshes[unsafe_offset=Int(al.meshIdx)]
    var ti = area_light_pick_triangle(al, pcg.next_float())
    var lb = ti * 3
    var lv0 = Int(lmesh.vertexIndices[unsafe_offset=lb]); var lv1 = Int(lmesh.vertexIndices[unsafe_offset=lb+1]); var lv2 = Int(lmesh.vertexIndices[unsafe_offset=lb+2])
    var lp0 = Vec3f(lmesh.points[unsafe_offset=lv0*4], lmesh.points[unsafe_offset=lv0*4+1], lmesh.points[unsafe_offset=lv0*4+2])
    var lp1 = Vec3f(lmesh.points[unsafe_offset=lv1*4], lmesh.points[unsafe_offset=lv1*4+1], lmesh.points[unsafe_offset=lv1*4+2])
    var lp2 = Vec3f(lmesh.points[unsafe_offset=lv2*4], lmesh.points[unsafe_offset=lv2*4+1], lmesh.points[unsafe_offset=lv2*4+2])
    var ru1 = pcg.next_float(); var ru2 = pcg.next_float(); var sr1 = sqrt(ru1)
    var light_point = lp0*(Float32(1)-sr1) + lp1*(sr1*(Float32(1)-ru2)) + lp2*(sr1*ru2)
    var ldp_du = lp1 - lp0
    var ldp_dv = lp2 - lp0

    var hit_v = hit.to_simd()
    var to_light = light_point - hit_v
    var dist_sq = dot(to_light, to_light)
    if dist_sq < Float32(1e-8) or al.total_area <= Float32(0):
        return SpectralSample(Float32(0))
    var dist = sqrt(dist_sq)
    var shadow_dir = to_light * (Float32(1) / dist)

    var probe_org = hit_v + shadow_dir * Float32(0.0002)
    var probe_ray = Ray(Point3f(probe_org[0], probe_org[1], probe_org[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
    var probe_tmax = dist * Float32(0.9995)
    var dummy_prim = PrimId(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var dummy_inter = Intersection(dummy_prim, probe_tmax, Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
    var probe_store = Array[Intersection, 1](fill=dummy_inter)
    traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, probe_ray, probe_tmax, probe_store.unsafe_ptr(),
                       sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
    var probe_inter = probe_store[0]
    if probe_inter.hit == Int8(0) or probe_inter.primId.type != Int8(0):
        return SpectralSample(Float32(0))
    var probe_mat = sd.materials[unsafe_offset=Int(probe_inter.primId.materialIndex)]
    if not is_specular_glass(probe_mat):
        return SpectralSample(Float32(0))

    var (pmesh, pv0, pv1, pv2, ptok) = _get_tri_verts(probe_inter, sd.meshes)
    if not ptok:
        return SpectralSample(Float32(0))
    var pp0 = Vec3f(pmesh.points[unsafe_offset=pv0*4], pmesh.points[unsafe_offset=pv0*4+1], pmesh.points[unsafe_offset=pv0*4+2])
    var pp1 = Vec3f(pmesh.points[unsafe_offset=pv1*4], pmesh.points[unsafe_offset=pv1*4+1], pmesh.points[unsafe_offset=pv1*4+2])
    var pp2 = Vec3f(pmesh.points[unsafe_offset=pv2*4], pmesh.points[unsafe_offset=pv2*4+1], pmesh.points[unsafe_offset=pv2*4+2])
    var pdp_du = pp1 - pp0
    var pdp_dv = pp2 - pp0
    var pgeo_n3 = cross(pdp_du, pdp_dv)
    var pgeo_n_len = sqrt(dot(pgeo_n3, pgeo_n3))
    if pgeo_n_len <= Float32(1e-10):
        return SpectralSample(Float32(0))
    var pgeo_n_raw = pgeo_n3 * (Float32(1) / pgeo_n_len)
    var ior1 = probe_mat.albedo.r
    var eta1 = ior1 if dot(pgeo_n_raw, shadow_dir) <= Float32(0) else (Float32(1) / ior1)
    var pgeo_n = pgeo_n_raw
    if dot(pgeo_n, shadow_dir) > Float32(0):
        pgeo_n = -pgeo_n
    var pu = probe_inter.u; var pvb = probe_inter.v
    var x1_init = pp0*(Float32(1)-pu-pvb) + pp1*pu + pp2*pvb

    var pdf_sel = Float32(1) / Float32(n_area)  # uniform light+point pick, matches this file's own convention

    var probe2_t0 = probe_inter.tHit
    var probe2_rem = (dist - probe2_t0) * Float32(0.9995)
    var probe2_org = x1_init + shadow_dir * Float32(0.0005)
    var probe2_inter = dummy_inter
    if probe2_rem > Float32(0.001):
        var probe2_ray = Ray(Point3f(probe2_org[0], probe2_org[1], probe2_org[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
        var probe2_store = Array[Intersection, 1](fill=dummy_inter)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, probe2_ray, probe2_rem, probe2_store.unsafe_ptr(),
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        probe2_inter = probe2_store[0]

    if probe2_inter.hit != Int8(0) and probe2_inter.primId.type == Int8(0):
        var probe2_mat = sd.materials[unsafe_offset=Int(probe2_inter.primId.materialIndex)]
        if is_specular_glass(probe2_mat):
            # --- 2-vertex MNEE ---
            var (p2mesh, p2v0, p2v1, p2v2, p2ok) = _get_tri_verts(probe2_inter, sd.meshes)
            if not p2ok:
                return SpectralSample(Float32(0))
            var p2p0 = Vec3f(p2mesh.points[unsafe_offset=p2v0*4], p2mesh.points[unsafe_offset=p2v0*4+1], p2mesh.points[unsafe_offset=p2v0*4+2])
            var p2p1 = Vec3f(p2mesh.points[unsafe_offset=p2v1*4], p2mesh.points[unsafe_offset=p2v1*4+1], p2mesh.points[unsafe_offset=p2v1*4+2])
            var p2p2 = Vec3f(p2mesh.points[unsafe_offset=p2v2*4], p2mesh.points[unsafe_offset=p2v2*4+1], p2mesh.points[unsafe_offset=p2v2*4+2])
            var pdp_du2 = p2p1 - p2p0; var pdp_dv2 = p2p2 - p2p0
            var pgeo_n3_2 = cross(pdp_du2, pdp_dv2)
            var pgeo_n_len2 = sqrt(dot(pgeo_n3_2, pgeo_n3_2))
            if pgeo_n_len2 <= Float32(1e-10):
                return SpectralSample(Float32(0))
            var pgeo_n2_raw = pgeo_n3_2 * (Float32(1) / pgeo_n_len2)
            var ior2 = probe2_mat.albedo.r
            var eta2 = ior2 if dot(pgeo_n2_raw, shadow_dir) <= Float32(0) else (Float32(1) / ior2)
            var pgeo_n2 = pgeo_n2_raw
            if dot(pgeo_n2, shadow_dir) > Float32(0):
                pgeo_n2 = -pgeo_n2
            var pu2 = probe2_inter.u; var pvb2 = probe2_inter.v
            var x2_init = p2p0*(Float32(1)-pu2-pvb2) + p2p1*pu2 + p2p2*pvb2
            var (ok2, x1_f2, x2_f2, bsdf_prod, dx1_dxl2) = _mnee_walk2(
                hit_v, light_point,
                x1_init, pgeo_n, pdp_du, pdp_dv, eta1,
                x2_init, pgeo_n2, pdp_du2, pdp_dv2, eta2,
                ldp_du, ldp_dv)
            if not ok2:
                return SpectralSample(Float32(0))
            var wi2f = hit_v - x1_f2
            var wi2fl = sqrt(dot(wi2f, wi2f))
            if wi2fl <= Float32(1e-8):
                return SpectralSample(Float32(0))
            var wi2fn = wi2f * (Float32(1) / wi2fl)
            var cos_s_x0 = dot(gn, -wi2fn)
            if cos_s_x0 <= Float32(0):
                return SpectralSample(Float32(0))
            var G2 = min(abs(dot(wi2fn, pgeo_n)) / (wi2fl*wi2fl) * dx1_dxl2, Float32(2))
            var pdf_area2 = pdf_sel / al.total_area
            var wo2f = light_point - x2_f2
            var wo2fl = sqrt(dot(wo2f, wo2f))
            if wo2fl <= Float32(1e-8):
                return SpectralSample(Float32(0))
            var wo2fn = wo2f * (Float32(1) / wo2fl)
            var vis2_org = x2_f2 + wo2fn * Float32(0.001)
            var vis2_ray = Ray(Point3f(vis2_org[0], vis2_org[1], vis2_org[2]), Vec3f(wo2fn[0], wo2fn[1], wo2fn[2]))
            if any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, vis2_ray, wo2fl * Float32(0.999),
                                  sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                                  sd.spheres, Int(sd.sphereCount)):
                return SpectralSample(Float32(0))
            var coat_t2 = Float32(1.0) - fr_dielectric(cos_s_x0, ior)
            var f_r = bxdf_eval_diffuse(eff_alb) * coat_t2
            return (beta
                * spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, f_r.r, f_r.g, f_r.b, wl)
                * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al.emission.r, al.emission.g, al.emission.b, wl)
                * (cos_s_x0 * G2 * bsdf_prod / pdf_area2))
        return SpectralSample(Float32(0))
    else:
        # --- 1-vertex MNEE ---
        var (mnee_ok, x1_f, det_b, eta_f) = _mnee_walk(hit_v, light_point, x1_init, pgeo_n, pdp_du, pdp_dv, eta1)
        if not mnee_ok:
            return SpectralSample(Float32(0))
        var wi_f = hit_v - x1_f
        var wi_len2_f = dot(wi_f, wi_f)
        var wo_f = light_point - x1_f
        var wo_len2_f = dot(wo_f, wo_f)
        if wi_len2_f <= Float32(1e-8) or wo_len2_f <= Float32(1e-8):
            return SpectralSample(Float32(0))
        var wi_len_f = sqrt(wi_len2_f)
        var wo_len_f = sqrt(wo_len2_f)
        var wi_fn = wi_f * (Float32(1) / wi_len_f)
        var wo_fn = wo_f * (Float32(1) / wo_len_f)
        var cos_s_x0 = dot(gn, -wi_fn)
        if cos_s_x0 <= Float32(0):
            return SpectralSample(Float32(0))
        var H3_f = -(wi_fn + wo_fn * eta_f)
        var H_len2_f = dot(H3_f, H3_f)
        if H_len2_f <= Float32(1e-10):
            return SpectralSample(Float32(0))
        var H_len_f = sqrt(H_len2_f)
        var H_f = H3_f * (Float32(1) / H_len_f)
        var dp_du_dot_n = dot(pdp_du, pgeo_n)
        var s3_f = pdp_du - pgeo_n * dp_du_dot_n
        var s_len2_f = dot(s3_f, s3_f)
        if s_len2_f <= Float32(1e-10):
            return SpectralSample(Float32(0))
        var s_f = s3_f * (Float32(1) / sqrt(s_len2_f))
        var t_f = cross(pgeo_n, s_f)
        var ilo_l = eta_f / (H_len_f * wo_len_f)
        var dHdu_l = (ldp_du - wo_fn * dot(wo_fn, ldp_du)) * ilo_l
        var dHdv_l = (ldp_dv - wo_fn * dot(wo_fn, ldp_dv)) * ilo_l
        dHdu_l -= H_f * dot(dHdu_l, H_f); dHdu_l = -dHdu_l
        dHdv_l -= H_f * dot(dHdv_l, H_f); dHdv_l = -dHdv_l
        var dc00 = dot(dHdu_l, s_f); var dc01 = dot(dHdv_l, s_f)
        var dc10 = dot(dHdu_l, t_f); var dc11 = dot(dHdv_l, t_f)
        var det_dc = dc00*dc11 - dc01*dc10
        var dx1_dxl = abs(det_dc) / max(abs(det_b), Float32(1e-8))
        var dw0_dx1 = abs(dot(wi_fn, pgeo_n)) / wi_len2_f
        var G = min(dw0_dx1 * dx1_dxl, Float32(2))
        var cosNI = abs(dot(pgeo_n, wi_fn))
        var cosHI = abs(dot(H_f, wi_fn))
        var cosTM = abs(dot(pgeo_n, H_f))
        var F_r = fr_dielectric(cosNI, eta_f)
        var T_f = Float32(1) - F_r
        var bsdf_s = T_f * cosHI / max(cosNI * cosTM * cosTM, Float32(1e-6))
        var pdf_area_x2 = pdf_sel / al.total_area
        var vis_org = x1_f + wo_fn * Float32(0.001)
        var vis_ray = Ray(Point3f(vis_org[0], vis_org[1], vis_org[2]), Vec3f(wo_fn[0], wo_fn[1], wo_fn[2]))
        if any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, vis_ray, wo_len_f * Float32(0.999),
                              sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                              sd.spheres, Int(sd.sphereCount)):
            return SpectralSample(Float32(0))
        var coat_t1 = Float32(1.0) - fr_dielectric(cos_s_x0, ior)
        var f_r = bxdf_eval_diffuse(eff_alb) * coat_t1
        return (beta
                * spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, f_r.r, f_r.g, f_r.b, wl)
                * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, al.emission.r, al.emission.g, al.emission.b, wl)
                * (cos_s_x0 * G * bsdf_s / pdf_area_x2))


def _bdpt_mnee_sphere_light(
    ref sd: SceneView, hit: Point3f, gn: Vec3f, eff_alb: RGB,
    beta: SpectralSample, mut pcg: PCG32, sph_idx: Int, n_spheres: Int,
    wl: SampledWavelengths, ior: Float32 = Float32(1.0),
) -> SpectralSample:
    """Real MNEE (task #161 follow-up, 2026-07-13) against an ANALYTIC
    SPHERE area light behind glass -- sibling to"""
    var sph = sd.spheres[unsafe_offset=sph_idx]
    if sph.isAreaLight == Int8(0):
        return SpectralSample(Float32(0))
    var u1 = pcg.next_float(); var u2 = pcg.next_float()
    var cosT = Float32(1) - Float32(2) * u1
    var sinT = sqrt(max(Float32(0), Float32(1) - cosT*cosT))
    var phi = Float32(2) * PI * u2
    var cosPhi = cos(phi); var sinPhi = sin(phi)
    var dir = Vec3f(sinT*cosPhi, sinT*sinPhi, cosT)
    var light_point = sph.center.to_simd() + dir * sph.radius
    # Analytic tangents of p(theta,phi) = center + r*dir(theta,phi) w.r.t.
    # (phi,theta) -- the same two parameters this point was just sampled
    # from (matches pbrt's own dpdu/dpdv convention for spheres).
    var ldp_du = Vec3f(-sinT*sinPhi, sinT*cosPhi, Float32(0)) * sph.radius   # dp/dphi
    var ldp_dv = Vec3f(cosT*cosPhi, cosT*sinPhi, -sinT) * sph.radius          # dp/dtheta
    var total_area = Float32(4) * PI * sph.radius * sph.radius
    if total_area <= Float32(0):
        return SpectralSample(Float32(0))
    var hit_v = hit.to_simd()
    var to_light = light_point - hit_v
    var dist_sq = dot(to_light, to_light)
    if dist_sq < Float32(1e-8) or total_area <= Float32(0):
        return SpectralSample(Float32(0))
    var dist = sqrt(dist_sq)
    var shadow_dir = to_light * (Float32(1) / dist)

    var probe_org = hit_v + shadow_dir * Float32(0.0002)
    var probe_ray = Ray(Point3f(probe_org[0], probe_org[1], probe_org[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
    var probe_tmax = dist * Float32(0.9995)
    var dummy_prim = PrimId(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var dummy_inter = Intersection(dummy_prim, probe_tmax, Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
    var probe_store = Array[Intersection, 1](fill=dummy_inter)
    traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, probe_ray, probe_tmax, probe_store.unsafe_ptr(),
                       sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
    var probe_inter = probe_store[0]
    if probe_inter.hit == Int8(0) or probe_inter.primId.type != Int8(0):
        return SpectralSample(Float32(0))
    var probe_mat = sd.materials[unsafe_offset=Int(probe_inter.primId.materialIndex)]
    if not is_specular_glass(probe_mat):
        return SpectralSample(Float32(0))

    var (pmesh, pv0, pv1, pv2, ptok) = _get_tri_verts(probe_inter, sd.meshes)
    if not ptok:
        return SpectralSample(Float32(0))
    var pp0 = Vec3f(pmesh.points[unsafe_offset=pv0*4], pmesh.points[unsafe_offset=pv0*4+1], pmesh.points[unsafe_offset=pv0*4+2])
    var pp1 = Vec3f(pmesh.points[unsafe_offset=pv1*4], pmesh.points[unsafe_offset=pv1*4+1], pmesh.points[unsafe_offset=pv1*4+2])
    var pp2 = Vec3f(pmesh.points[unsafe_offset=pv2*4], pmesh.points[unsafe_offset=pv2*4+1], pmesh.points[unsafe_offset=pv2*4+2])
    var pdp_du = pp1 - pp0
    var pdp_dv = pp2 - pp0
    var pgeo_n3 = cross(pdp_du, pdp_dv)
    var pgeo_n_len = sqrt(dot(pgeo_n3, pgeo_n3))
    if pgeo_n_len <= Float32(1e-10):
        return SpectralSample(Float32(0))
    var pgeo_n_raw = pgeo_n3 * (Float32(1) / pgeo_n_len)
    var ior1 = probe_mat.albedo.r
    var eta1 = ior1 if dot(pgeo_n_raw, shadow_dir) <= Float32(0) else (Float32(1) / ior1)
    var pgeo_n = pgeo_n_raw
    if dot(pgeo_n, shadow_dir) > Float32(0):
        pgeo_n = -pgeo_n
    var pu = probe_inter.u; var pvb = probe_inter.v
    var x1_init = pp0*(Float32(1)-pu-pvb) + pp1*pu + pp2*pvb

    var pdf_sel = Float32(1) / Float32(max(n_spheres, 1))  # uniform light+point pick, matches this file's own convention

    var probe2_t0 = probe_inter.tHit
    var probe2_rem = (dist - probe2_t0) * Float32(0.9995)
    var probe2_org = x1_init + shadow_dir * Float32(0.0005)
    var probe2_inter = dummy_inter
    if probe2_rem > Float32(0.001):
        var probe2_ray = Ray(Point3f(probe2_org[0], probe2_org[1], probe2_org[2]), Vec3f(shadow_dir[0], shadow_dir[1], shadow_dir[2]))
        var probe2_store = Array[Intersection, 1](fill=dummy_inter)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, probe2_ray, probe2_rem, probe2_store.unsafe_ptr(),
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        probe2_inter = probe2_store[0]

    if probe2_inter.hit != Int8(0) and probe2_inter.primId.type == Int8(0):
        var probe2_mat = sd.materials[unsafe_offset=Int(probe2_inter.primId.materialIndex)]
        if is_specular_glass(probe2_mat):
            # --- 2-vertex MNEE ---
            var (p2mesh, p2v0, p2v1, p2v2, p2ok) = _get_tri_verts(probe2_inter, sd.meshes)
            if not p2ok:
                return SpectralSample(Float32(0))
            var p2p0 = Vec3f(p2mesh.points[unsafe_offset=p2v0*4], p2mesh.points[unsafe_offset=p2v0*4+1], p2mesh.points[unsafe_offset=p2v0*4+2])
            var p2p1 = Vec3f(p2mesh.points[unsafe_offset=p2v1*4], p2mesh.points[unsafe_offset=p2v1*4+1], p2mesh.points[unsafe_offset=p2v1*4+2])
            var p2p2 = Vec3f(p2mesh.points[unsafe_offset=p2v2*4], p2mesh.points[unsafe_offset=p2v2*4+1], p2mesh.points[unsafe_offset=p2v2*4+2])
            var pdp_du2 = p2p1 - p2p0; var pdp_dv2 = p2p2 - p2p0
            var pgeo_n3_2 = cross(pdp_du2, pdp_dv2)
            var pgeo_n_len2 = sqrt(dot(pgeo_n3_2, pgeo_n3_2))
            if pgeo_n_len2 <= Float32(1e-10):
                return SpectralSample(Float32(0))
            var pgeo_n2_raw = pgeo_n3_2 * (Float32(1) / pgeo_n_len2)
            var ior2 = probe2_mat.albedo.r
            var eta2 = ior2 if dot(pgeo_n2_raw, shadow_dir) <= Float32(0) else (Float32(1) / ior2)
            var pgeo_n2 = pgeo_n2_raw
            if dot(pgeo_n2, shadow_dir) > Float32(0):
                pgeo_n2 = -pgeo_n2
            var pu2 = probe2_inter.u; var pvb2 = probe2_inter.v
            var x2_init = p2p0*(Float32(1)-pu2-pvb2) + p2p1*pu2 + p2p2*pvb2
            var (ok2, x1_f2, x2_f2, bsdf_prod, dx1_dxl2) = _mnee_walk2(
                hit_v, light_point,
                x1_init, pgeo_n, pdp_du, pdp_dv, eta1,
                x2_init, pgeo_n2, pdp_du2, pdp_dv2, eta2,
                ldp_du, ldp_dv)
            if not ok2:
                return SpectralSample(Float32(0))
            var wi2f = hit_v - x1_f2
            var wi2fl = sqrt(dot(wi2f, wi2f))
            if wi2fl <= Float32(1e-8):
                return SpectralSample(Float32(0))
            var wi2fn = wi2f * (Float32(1) / wi2fl)
            var cos_s_x0 = dot(gn, -wi2fn)
            if cos_s_x0 <= Float32(0):
                return SpectralSample(Float32(0))
            var G2 = min(abs(dot(wi2fn, pgeo_n)) / (wi2fl*wi2fl) * dx1_dxl2, Float32(2))
            var pdf_area2 = pdf_sel / total_area
            var wo2f = light_point - x2_f2
            var wo2fl = sqrt(dot(wo2f, wo2f))
            if wo2fl <= Float32(1e-8):
                return SpectralSample(Float32(0))
            var wo2fn = wo2f * (Float32(1) / wo2fl)
            var vis2_org = x2_f2 + wo2fn * Float32(0.001)
            var vis2_ray = Ray(Point3f(vis2_org[0], vis2_org[1], vis2_org[2]), Vec3f(wo2fn[0], wo2fn[1], wo2fn[2]))
            if any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, vis2_ray, wo2fl * Float32(0.999),
                                  sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                                  sd.spheres, Int(sd.sphereCount)):
                return SpectralSample(Float32(0))
            # coat_t (coateddiffuse coat-transmittance, mirrors
            # _bdpt_mnee_diffuse_area_light's `ior` handling) is DELIBERATELY
            var f_r = bxdf_eval_diffuse(eff_alb)
            return (beta
                * spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, f_r.r, f_r.g, f_r.b, wl)
                * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, sph.emission.r, sph.emission.g, sph.emission.b, wl)
                * (cos_s_x0 * G2 * bsdf_prod / pdf_area2))
        return SpectralSample(Float32(0))
    else:
        # --- 1-vertex MNEE ---
        var (mnee_ok, x1_f, det_b, eta_f) = _mnee_walk(hit_v, light_point, x1_init, pgeo_n, pdp_du, pdp_dv, eta1)
        if not mnee_ok:
            return SpectralSample(Float32(0))
        var wi_f = hit_v - x1_f
        var wi_len2_f = dot(wi_f, wi_f)
        var wo_f = light_point - x1_f
        var wo_len2_f = dot(wo_f, wo_f)
        if wi_len2_f <= Float32(1e-8) or wo_len2_f <= Float32(1e-8):
            return SpectralSample(Float32(0))
        var wi_len_f = sqrt(wi_len2_f)
        var wo_len_f = sqrt(wo_len2_f)
        var wi_fn = wi_f * (Float32(1) / wi_len_f)
        var wo_fn = wo_f * (Float32(1) / wo_len_f)
        var cos_s_x0 = dot(gn, -wi_fn)
        if cos_s_x0 <= Float32(0):
            return SpectralSample(Float32(0))
        var H3_f = -(wi_fn + wo_fn * eta_f)
        var H_len2_f = dot(H3_f, H3_f)
        if H_len2_f <= Float32(1e-10):
            return SpectralSample(Float32(0))
        var H_len_f = sqrt(H_len2_f)
        var H_f = H3_f * (Float32(1) / H_len_f)
        var dp_du_dot_n = dot(pdp_du, pgeo_n)
        var s3_f = pdp_du - pgeo_n * dp_du_dot_n
        var s_len2_f = dot(s3_f, s3_f)
        if s_len2_f <= Float32(1e-10):
            return SpectralSample(Float32(0))
        var s_f = s3_f * (Float32(1) / sqrt(s_len2_f))
        var t_f = cross(pgeo_n, s_f)
        var ilo_l = eta_f / (H_len_f * wo_len_f)
        var dHdu_l = (ldp_du - wo_fn * dot(wo_fn, ldp_du)) * ilo_l
        var dHdv_l = (ldp_dv - wo_fn * dot(wo_fn, ldp_dv)) * ilo_l
        dHdu_l -= H_f * dot(dHdu_l, H_f); dHdu_l = -dHdu_l
        dHdv_l -= H_f * dot(dHdv_l, H_f); dHdv_l = -dHdv_l
        var dc00 = dot(dHdu_l, s_f); var dc01 = dot(dHdv_l, s_f)
        var dc10 = dot(dHdu_l, t_f); var dc11 = dot(dHdv_l, t_f)
        var det_dc = dc00*dc11 - dc01*dc10
        var dx1_dxl = abs(det_dc) / max(abs(det_b), Float32(1e-8))
        var dw0_dx1 = abs(dot(wi_fn, pgeo_n)) / wi_len2_f
        var G = min(dw0_dx1 * dx1_dxl, Float32(2))
        var cosNI = abs(dot(pgeo_n, wi_fn))
        var cosHI = abs(dot(H_f, wi_fn))
        var cosTM = abs(dot(pgeo_n, H_f))
        var F_r = fr_dielectric(cosNI, eta_f)
        var T_f = Float32(1) - F_r
        var bsdf_s = T_f * cosHI / max(cosNI * cosTM * cosTM, Float32(1e-6))
        var pdf_area_x2 = pdf_sel / total_area
        var vis_org = x1_f + wo_fn * Float32(0.001)
        var vis_ray = Ray(Point3f(vis_org[0], vis_org[1], vis_org[2]), Vec3f(wo_fn[0], wo_fn[1], wo_fn[2]))
        if any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, vis_ray, wo_len_f * Float32(0.999),
                              sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances,
                              sd.spheres, Int(sd.sphereCount)):
            return SpectralSample(Float32(0))
        # coat_t deliberately not applied -- see the 2-vertex branch's
        # identical comment above (this function's docstring has the full
        var f_r = bxdf_eval_diffuse(eff_alb)
        return (beta
            * spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, f_r.r, f_r.g, f_r.b, wl)
            * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, sph.emission.r, sph.emission.g, sph.emission.b, wl)
            * (cos_s_x0 * G * bsdf_s / pdf_area_x2))

# ── Cosine-area PDF conversion ────────────────────────────────────────────────
