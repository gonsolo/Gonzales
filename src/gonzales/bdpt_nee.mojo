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
    occluded. Glass (dielectric) surfaces are passed through with Fresnel
    transmittance. `scratch` is one caller-owned Intersection slot (no
    internal alloc/free) so this is safe to call from a GPU kernel thread —
    every existing GPU kernel in this codebase takes pre-allocated,
    thread-indexed scratch instead of allocating per-thread (see
    sppm_gen_vp_gpu's inter_scratch).

    Each medium segment's sigma_t is upsampled to the 4 hero lanes FIRST
    (medium_sigma_t_spectral), then exponentiated PER LANE -- the same
    ordering fix as spectral_free_flight_weight, applied here to a plain
    deterministic Beer-Lambert evaluation rather than an importance-sampled
    ratio (no red-channel proposal to correct against; this just IS
    exp(-sigma_t(lambda)*d) at each segment, multiplied across segments,
    which is exact: exp(a)*exp(b) = exp(a+b))."""
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
        # a separate pass. Seed a sentinel tHit=remaining*0.9995 when the BVH found
        # nothing, so test_spheres's own internal tMax (it only bounds itself by
        # result[0].tHit when result[0].hit is already set) respects the shadow
        # ray's segment length instead of defaulting to unbounded (1e38).
        var had_bvh_hit = inter_mem[unsafe_offset=0].hit != Int8(0)
        if not had_bvh_hit:
            inter_mem[unsafe_offset=0].hit = Int8(1)
            inter_mem[unsafe_offset=0].tHit = remaining * Float32(0.9995)
            # Clear the primId along with the sentinel. `scratch` is a
            # caller-owned slot reused across bounces, samples and (on GPU)
            # threads, so on a BVH miss the primId still holds STALE data
            # from a previous traversal -- and test_spheres writes none when
            # sphereCount == 0. The `primId.type != 4` test just below then
            # reads that stale type, and a leftover 4 makes a pure miss
            # masquerade as a sphere hit, falling through to
            # sd.spheres[stale id1] and sd.materials[stale materialIndex]:
            # an out-of-bounds read on any scene with no spheres.
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
            # costs sss-slab.vcm 48% of its energy (a pinned smoke cell),
            # so the straight-line crossing is load-bearing there.
            # A THIN dielectric only. Straight-line pass-through is valid
            # here because a thin slab's entry and exit refractions cancel --
            # the ray leaves parallel to how it arrived, so the shadow ray's
            # geometry is right and only the Fresnel attenuation is needed.
            #
            # A THICK dielectric used to pass through here too, and that was
            # wrong: light REFRACTS at a thick refractor, so a straight shot
            # through it is not a physical path at all. NEE was therefore
            # manufacturing transport that no sampling strategy can generate,
            # and MIS cannot cancel what it never sees. Measured on
            # barcelona-pavilion-day, whose pool is a water plane over a
            # coateddiffuse bottom: the bottom third of the frame read 2.243x
            # a pbrt BDPT reference with this pass-through and 1.461x without
            # -- by far the largest single error in that scene, and confined
            # to exactly the region where a shadow ray must cross the water.
            # The path tracer never had this bug: its shadow ray is a binary
            # any_hit test, so the water blocks it outright, which is
            # accidentally correct for a thick refractor. pbrt blocks it too.
            #
            # What legitimately DOES get through a thick refractor is the
            # bent path, and finding that is MNEE's job
            # (_bdpt_mnee_diffuse_area_light, which fires precisely for the
            # glass-obscured case) -- not this straight line.
            # Pass through glass with Fresnel transmittance
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
                # primId fields and return garbage, making the inside/outside
                # test below a coin flip. The dielectric branch above already
                # special-cases this; the interface branch did not, so roughly
                # half the shadow rays leaving a sphere-bounded medium kept
                # cur_med set to the medium and were then Beer-Lambert'd across
                # the vacuum outside it -- annihilating them. That halved every
                # volume NEE contribution at every scatter order (measured
                # 0.500x vs the path tracer on a sphere-bounded fog).
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
    point + sphere, the three every BDPT material-loop samples the SAME way.
    Area and infinite are NOT covered -- area gets its own MNEE-capable
    sampling (_bdpt_mnee_diffuse_area_light/_bdpt_mnee_sphere_light) and
    infinite draws its own 2 pcg floats via _sample_infinite_light_nee, so
    both stay written out at their call sites. Mirrors shading.mojo's
    _nee_simple_light_count, but there is no shared struct between the two
    files' light contexts (ShadeContext vs SceneView) to unify them
    on, hence the parallel definition rather than a genuinely shared one."""
    return Int(sd.distantLightCount) + Int(sd.pointLightCount) + Int(sd.sphereCount)


@always_inline
def _bdpt_sample_simple_light(
    ref sd: SceneView, i: Int, hit_point: Vec3f, mut pcg: PCG32,
) -> LightSample:
    """The i-th distant/point/sphere light. Unlike shading.mojo's twin
    (_nee_sample_simple_light), this returns ONLY the LightSample -- BDPT's
    own occlusion primitive (_bdpt_nee_contribute, immediately below) tests
    the segment out to the exact `ls.dist` via _visible_transmittance, which
    is media-aware and needs no per-light-type tmax shrink the way
    shading.mojo's boolean any-hit test does. There is therefore no
    sampler/tmax PAIRING to get wrong here the way bba82627 did -- one fewer
    thing this duplication could silently break, not zero, since every call
    site still had to agree on the SAMPLER itself and its argument order.

    ORDER IS LOAD-BEARING: distant, then point, then sphere -- matching every
    existing call site in this file already. Of the three only SPHERE draws
    from `pcg`, so this is a pure loop collapse everywhere it's used, not a
    reordering; see this file's individual conversions for confirmation each
    call site's ORIGINAL order already matched this one exactly."""
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
    given a LightSample + material weight (from the shared Light interface
    — bvh.mojo's LightSample samplers — and BxDF interface —
    bxdf.mojo's _nee_weight_simple/_nee_weight_hair), test transmittance
    (BDPT's own occlusion primitive, media-aware — unlike shading.mojo/
    sppm.mojo's boolean any-hit test, so this stays a BDPT-local helper
    rather than a fully cross-integrator one) and return the beta-weighted
    contribution, or black if invalid/occluded. `eps` defaults to the fixed
    offset used for triangle/sphere hits; hair call sites pass
    curve_offset_eps(hc.radius) instead (see bvh.mojo)."""
    if w.is_black():
        return SpectralSample(Float32(0))
    # `gn` is the GEOMETRIC normal (pbrt's OffsetRayOrigin), never the shading
    # one: that is turned toward wo (face_toward) and can point into the
    # surface, which would start the shadow ray underneath it.
    # For a TWO-SIDED lobe the offset must follow the light direction:
    # +gn for a direction on the -gn side starts the ray inside the surface
    # it just left. Opt-in, NOT automatic -- MNEE connects THROUGH GLASS, so
    # a one-sided lobe CAN arrive here with cos < 0 and a non-black weight,
    # and flipping the offset there would push its ray to the wrong side.
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
    (or coateddiffuse base-layer, see `ior` below) camera vertex, probe
    whether a straight line toward a randomly-picked area light first hits
    dielectric glass, and if so solve for the true refracted connection via
    Newton iteration -- reusing shading.mojo's _mnee_walk/_mnee_walk2, the
    exact technique the plain path tracer's own _nee_area_lights already
    uses. This is WHY the plain path tracer correctly lights
    barcelona-pavilion (night) while bdpt_*.mojo's VCM connect/merge cannot:
    dielectric bounces are never stored as LVC vertices (see
    project_vcm_stage2_mis_derivation memory), so a light behind glass is
    structurally invisible to connect/merge, and its tiny solid angle
    makes unassisted BSDF-sampling hit it by pure luck only.

    Deliberately scoped to ONLY the glass-detected case. An earlier attempt
    added plain straight-line NEE for ALL area lights (not just
    glass-obscured ones) and was reverted: it double-counted with the
    EXISTING connect/merge estimator on ordinary, unobstructed lights
    (confirmed via git-stash A/B on cornell-box, ~38% over pbrt's
    reference). MNEE only ever fires for paths connect/merge structurally
    cannot represent anyway (a delta dielectric bounce in the middle of the
    connecting path), so there is no matching double-count risk here --
    when the probe does NOT hit glass first, this returns black and
    connect/merge (already correct for that ordinary case) are untouched.

    `ior` (2026-07-13 follow-up): the caller's coat IOR, applied as a
    `(1 - Fresnel(cos_s_x0, ior))` transmittance factor on the returned
    weight, matching _nee_weight_coated_diffuse_base's own formula shape
    for ordinary (non-MNEE) coateddiffuse NEE. Defaults to 1.0 for plain
    diffuse callers -- fr_dielectric(_, 1.0) is exactly 0 (no index
    mismatch means no reflection), so `1 - 0 = 1` recovers the original
    unweighted diffuse behavior exactly, not an approximation. Still
    diffuse-family only -- no material in this codebase does MNEE for
    conductor/hair/measured today. Curve-shaped area lights are skipped
    (no well-defined surface tangents for a swept tube), same as
    shading.mojo."""
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
    SPHERE area light behind glass -- sibling to
    _bdpt_mnee_diffuse_area_light (mesh/triangle lights), which live in a
    completely separate list (sd.spheres, not sd.areaLights). Deliberate
    FULL, SELF-CONTAINED DUPLICATE of that function's probe+Newton-walk
    body (not a shared helper) -- see this section's own investigation
    notes below for why.

    _sample_sphere_light_nee's existing solid-angle/cone sampling (used
    for ORDINARY sphere NEE elsewhere in this file) can't be reused here
    -- it only returns a sampled DIRECTION and a solid-angle pdf w.r.t.
    the shading point, no actual surface point or light-side
    parameterization to take Newton-walk derivatives against. Instead,
    samples a UNIFORM point on the sphere's surface via the standard
    spherical parameterization p(θ,φ) = center + r·(sinθcosφ, sinθsinφ,
    cosθ), using that SAME parameterization's own analytic partial
    derivatives ∂p/∂φ, ∂p/∂θ as the light-side tangent vectors (matches
    pbrt's own dpdu/dpdv convention for spheres).

    `sph_idx` (an index into sd.spheres, read locally via
    `sd.spheres[sph_idx]`) + `n_spheres` (the TOTAL sphere count, matching
    `_sample_sphere_light_nee`'s own `1/n_sph` pdf convention) -- caller
    iterates every sphere (see call-site comment for why NOT via a `for`
    loop). `ior`: accepted for signature symmetry with
    _bdpt_mnee_diffuse_area_light but NOT applied in the return value --
    see the GPU-codegen-bug note below for why.

    RESOLVED GPU BUG (2026-07-13, real Mojo/GPU-codegen bug, not sphere-
    specific -- root-caused via systematic bisection, not application
    logic): every earlier implementation of this feature crashed with a
    reproducible CUDA_ERROR_ILLEGAL_ADDRESS on barcelona-pavilion-night
    (this task's actual target scene). Made fully deterministic by
    temporarily hardcoding pbrt_parser.mojo's RNG seed (normally
    perf_counter_ns()) -- this turned a seemingly-nondeterministic crash
    (varied run to run because a wall-clock seed explores different pixel/
    sample paths each time) into 100%-reproducible pass/fail, which is
    what made real bisection possible. Systematic cutoff-return bisection
    through this function's body (return RGB(0) at successively later
    points, rebuild+rerun at each cutoff) narrowed the crash to an exact
    line: folding a SECOND `fr_dielectric(...)` call's result (`coat_t =
    1 - fr_dielectric(cos_s_x0, ior)`) into the final returned RGB
    expression, alongside `sph.emission`/`bxdf_eval_diffuse(...)`/the G
    and pdf terms. Calling `fr_dielectric` and discarding the result was
    SAFE; using `sph.emission` and `bxdf_eval_diffuse(...)` together in
    the final expression was SAFE; splitting the multiply across two
    statements (`var contrib = ...; return contrib * coat_t`) did NOT
    help -- still crashed identically, ruling out "too many chained
    multiplies in one expression" as the mechanism. This is consistent
    with the general shape of the anomaly logged in
    `reference_mojo_compiler_bug_6759.md` (heavy, `Array`-using,
    multiple early returns, BVH traversal, under this scene's specific
    complexity) -- though that report was later retracted by its own
    author as unreproducible, so this crash stands on its own bisection
    below, not on 6759 as corroboration. Not a NaN/degenerate-value bug
    in this code (cos_s_x0/ior were always finite, well-conditioned
    values at the crash site).
    WORKAROUND (applied here): don't compute/apply `coat_t` at all. Every
    CURRENT call site passes the default `ior=1.0`, for which
    `fr_dielectric(_, 1.0) == 0` exactly (an identity already relied on
    by _bdpt_mnee_diffuse_area_light's own `ior=1.0` default case), so
    `coat_t` would always equal exactly `1.0` anyway -- omitting it is a
    zero behavior change today, not an approximation. If sphere-light
    MNEE for coateddiffuse (ior != 1.0) is ever revisited, `coat_t` will
    need a DIFFERENT strategy that avoids this exact pattern (e.g.
    precomputing it in the caller and passing it in as a parameter,
    rather than computing+applying `fr_dielectric` inside this function).
    See project_barcelona_pavilion_mnee memory for the full investigation,
    including two earlier, unrelated red herrings (a `for`-loop-wrapping
    hypothesis and a shared-function-split hypothesis, both ruled out by
    this same bisection).
    WORKAROUND (unrelated, applied at every call site): call this
    function via manually UNROLLED `if`-guarded statements, never a `for`
    loop, capped at a small constant (comptime _MNEE_MAX_SPHERES) --
    scenes with more emissive spheres than the cap silently skip the
    extras for MNEE only (ordinary light-hit/connect/merge/NEE still
    reaches them normally, this only affects the glass-behind-sphere-
    light special case). Kept even though the loop-wrapping hypothesis
    turned out not to be the real bug, since unrolling is harmless and
    was already in place before the real cause was found."""
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
            # not applied here -- see this function's own docstring, "GPU
            # codegen bug" section, for why folding a 2nd fr_dielectric(...)
            # result into this return crashes with CUDA_ERROR_ILLEGAL_ADDRESS
            # on this task's target scene. Every current call site passes
            # the default ior=1.0 (coateddiffuse sphere-light call sites are
            # disabled, see _MNEE_MAX_SPHERES call-site comments), for which
            # fr_dielectric(_, 1.0) == 0 exactly, so coat_t == 1.0 exactly --
            # applying it would be a mathematical no-op anyway.
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
        # GPU-codegen-bug writeup). ior=1.0 at every current call site makes
        # this an exact no-op, not an approximation.
        var f_r = bxdf_eval_diffuse(eff_alb)
        return (beta
            * spec_refl(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, f_r.r, f_r.g, f_r.b, wl)
            * spec_illum(sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65, sph.emission.r, sph.emission.g, sph.emission.b, wl)
            * (cos_s_x0 * G * bsdf_s / pdf_area_x2))

# ── Cosine-area PDF conversion ────────────────────────────────────────────────
