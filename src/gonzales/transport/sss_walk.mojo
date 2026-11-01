"""A subsurface walk: the hop from a boundary's entry to its exit, shared by every integrator."""
from gonzales.scene.accel.bvh import traverse_bvh2_core_defer_curves, SceneView
from gonzales.core.geometry import Frame, Point3f, Vec3f, _is_real_ptr, cross, dot, face_toward, offset_point
from gonzales.scene.shapes.geom_normal import _geom_normal
from gonzales.scene.appearance.materials import Material, fr_dielectric, sss_boundary_lobe
from gonzales.core.sampling import sample_cosine_hemisphere_world
from gonzales.scene.appearance.media import hg_phase, hg_sample
from gonzales.scene.shapes.primitives import Intersection, mesh_vi
from gonzales.core.geometry import Ray
from gonzales.scene.shapes.primitives import mesh_pos
from gonzales.transport.render_state import PathState, SssWalk
from gonzales.core.rng import PCG32
from gonzales.transport.surface import sss_entry_tint
from gonzales.core.spectrum import SampledWavelengths, SpectralSample, rgb_bands_to_spectral_sample
from std.collections import Array
from std.math import abs, cos, exp, log, pow, sin, sqrt


@always_inline
def sss_walk_start(mat: Material, inter: Intersection, hit: Vec3f, geo_normal: Vec3f, wi: Vec3f, wl: SampledWavelengths) -> Tuple[Ray, SssWalk]:
    """The ray and state of a walk entering a pbrt subsurface boundary at `hit` along the refracted direction wi."""
    var st = mat.checker_tex1
    var al = mat.checker_tex2
    var out_n = face_toward(geo_normal, -wi)
    var org = offset_point(hit, -out_n)
    return (Ray(Point3f(org[0], org[1], org[2]), Vec3f(wi[0], wi[1], wi[2])),
            SssWalk(rgb_bands_to_spectral_sample(st.r, st.g, st.b, wl), rgb_bands_to_spectral_sample(al.r, al.g, al.b, wl),
                    mat.checker_uscale, Int32(inter.primId.id1), inter.primId.instanceIdx, Int32(0), Int32(0),
                    hit, out_n, Float32(-1), Float32(1), Float32(1), Float32(1), Float32(0), Float32(0), Float32(0)))


@always_inline
def sss_entry_lobe(in_n: Vec3f, eta: Float32, mut pcg: PCG32) -> Vec3f:
    """A walk's first direction: a cosine lobe weighted by interior transmission, the mirror of its exit."""
    var wi = sample_cosine_hemisphere_world(pcg.next_float(), pcg.next_float(), in_n)[0]
    for _ in range(8):
        if pcg.next_float() >= fr_dielectric(abs(dot(wi, in_n)), Float32(1) / max(eta, Float32(1.0001))):
            break
        wi = sample_cosine_hemisphere_world(pcg.next_float(), pcg.next_float(), in_n)[0]
    return wi


@fieldwise_init
struct SssLeave(TrivialRegisterPassable):
    var wi: Vec3f
    var cos_out: Float32
    var pdf: Float32
    var weight: Float32             # the boundary lobe over its cosine sampling; mean 1


@always_inline
def sss_leave(n_o: Vec3f, eta: Float32, u1: Float32, u2: Float32) -> SssLeave:
    """Leaves a walk's exit along its boundary lobe, for every integrator."""
    var (wi, pdf) = sample_cosine_hemisphere_world(u1, u2, n_o)
    var cos_out = abs(dot(wi, n_o))
    return SssLeave(wi, cos_out, pdf, sss_boundary_lobe(cos_out, eta))


@fieldwise_init
struct SssExit(TrivialRegisterPassable):
    var ok: Bool
    var x_o: Point3f
    var n_o: Vec3f                  # outward geometric normal
    var weight: SpectralSample


@always_inline
def sss_walk_to_exit(ref sd: SceneView, mat: Material, inter: Intersection, hit: Vec3f, geo_normal: Vec3f, wi: Vec3f,
                     wl: SampledWavelengths, mut pcg: PCG32) -> SssExit:
    """The path tracer's walk, run to its end in one go for the integrators without a walking wavefront."""
    var (ray, walk) = sss_walk_start(mat, inter, hit, geo_normal, wi, wl)
    var state = Array[PathState, 1](uninitialized=True)
    var store = Array[Intersection, 1](fill=inter)
    var ps = state.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var hits = store.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    ps[].ray = ray
    ps[].sss = walk
    ps[].throughput = SpectralSample(Float32(1))
    ps[].wavelengths = wl
    ps[].active = Int8(1)
    ps[].volume_scattered = Int8(0)
    ps[].pcgState = pcg.state
    ps[].pcgInc = pcg.inc
    _sss_walk_trace(sd, walk, ray, hits)
    _sss_walk_step(ps, hits, 0, sd)
    pcg.state = ps[].pcgState
    if ps[].active == 0 or ps[].sss.exiting == Int32(0):
        return SssExit(False, Point3f(hit[0], hit[1], hit[2]), geo_normal, SpectralSample(Float32(0)))
    var e = hits[unsafe_offset=0]
    var r = ps[].ray
    var x = Point3f(r.origin.x + r.direction.x * e.tHit, r.origin.y + r.direction.y * e.tHit, r.origin.z + r.direction.z * e.tHit)
    var n = _geom_normal(e, sd.meshes, sd.instances, sd.spheres, x.to_simd())
    if dot(n, Vec3f(r.direction.x, r.direction.y, r.direction.z)) < Float32(0):
        n = -n
    return SssExit(True, x, n, ps[].throughput)


@always_inline
def _sss_walk_trace(ref sd: SceneView, walk: SssWalk, ray: Ray, store: Pointer[Intersection, MutUntrackedOrigin]):
    """One segment of a walk: in the mesh's own BVH when it has one."""
    var local_bvh = -1
    if _is_real_ptr(sd.walkBvhOfMesh) and walk.instance < Int32(0):
        local_bvh = Int(sd.walkBvhOfMesh[unsafe_offset=Int(walk.mesh)])
    if local_bvh >= 0:
        traverse_bvh2_core_defer_curves[False, False, False](
            sd.walkNodesArr[unsafe_offset=local_bvh], sd.walkPrimsArr[unsafe_offset=local_bvh], sd.meshes, sd.curves,
            ray, Float32(1.0e38), store,
            Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(), Pointer[Int32, MutUntrackedOrigin].unsafe_dangling())
    else:
        traverse_bvh2_core_defer_curves[False, True, False](
            sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1.0e38),
            store,
            Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(), Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
            sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)


# Inlined so `sd` stays in kernel param space: a non-inlined call needs its
# address, which copies the whole descriptor to local memory per thread.
@always_inline
def _sss_walk_step[steps: Int = 256](
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    i: Int,
    ref sd: SceneView,
):
    """A random walk under a surface (PathState.sss), as Cycles' subsurface_random_walk: each distance comes from one
    hero wavelength picked by albedo times throughput, directions are Henyey-Greenstein or, for a guided fraction,
    Dwivedi-guided toward the entry surface (or away, toward a nearby opposite one), and every step is weighted by the
    balance heuristic over wavelengths and strategies. Up to `steps` steps a call: the first uses the segment the caller
    traced (`intersections`), later ones trace their own in the mesh's own BVH; the next hit on the mesh is the exit, left
    in `intersections` for the shading kernels. A walk out of steps leaves its next segment for the caller to trace."""
    # Cycles' SUBSURFACE_RANDOM_WALK_SIMILARITY_LEVEL: after these steps the walk continues isotropic with the reduced
    # scattering sigma_s (1 - g), the similarity relation, which makes strongly forward-scattering walks much shorter.
    comptime _SSS_SIMILARITY_STEPS = 9
    comptime _SSS_MAX_STEPS = 256
    var path_ptr = paths.unsafe_offset(i)
    var walk = path_ptr[].sss
    var st0 = SIMD[DType.float32, 4](walk.sigma_t.v0, walk.sigma_t.v1, walk.sigma_t.v2, walk.sigma_t.v3)
    var al0 = SIMD[DType.float32, 4](walk.alpha.v0, walk.alpha.v1, walk.alpha.v2, walk.alpha.v3)
    var st_iso = st0 * (Float32(1) - al0 * walk.g)
    var al_iso = al0 * (Float32(1) - walk.g) / max(Float32(1) - al0 * walk.g, Float32(1e-7))
    # Dwivedi's diffusion length for the highest albedo (Cycles: a lower one gives fireflies).
    var amax = min(max(al0.reduce_max(), Float32(1e-4)), Float32(0.9999))   # at exactly 1 the diffusion length diverges
    var dl = Float32(1) / sqrt(max(Float32(1) - pow(amax, Float32(2.44294) - Float32(0.0215813) * amax + Float32(0.578637) / amax), Float32(1e-12)))
    var phase_log = log((dl + Float32(1)) / max(dl - Float32(1), Float32(1e-7)))
    var guided_g = Float32(1) - max(Float32(0.5), pow(abs(walk.g), Float32(0.125)))
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var store = Array[Intersection, 1](fill=intersections[unsafe_offset=i])
    var ray = path_ptr[].ray
    for it in range(steps):
        if it > 0:
            _sss_walk_trace(sd, walk, ray, store.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
        var step = path_ptr[].sss.steps
        if step >= Int32(_SSS_MAX_STEPS):
            path_ptr[].active = 0   # Cycles' BSSRDF_MAX_BOUNCES: a walk this long is given up
            path_ptr[].pcgState = pcg.state
            return
        var iso = step > Int32(_SSS_SIMILARITY_STEPS)
        var st = st_iso if iso else st0
        var al = al_iso if iso else al0
        var guided_fraction = Float32(0.75) if iso else guided_g
        path_ptr[].sss.steps += 1
        var inter = store[0]
        if inter.hit == 0:
            path_ptr[].active = 0   # left through a hole in the mesh
            path_ptr[].pcgState = pcg.state
            return
        var org = Vec3f(ray.origin.x, ray.origin.y, ray.origin.z)
        var dir = Vec3f(ray.direction.x, ray.direction.y, ray.direction.z)
        if step == Int32(0):
            # The first segment looks for an opposite side within 10 mean free paths, for backward guiding.
            var reach = Float32(10) / max(st0.reduce_min(), Float32(1e-12))
            if inter.tHit <= reach:
                path_ptr[].sss.opposite = dot(org + dir * inter.tHit - walk.p0, -walk.n0)
        var tps = path_ptr[].throughput
        var tp = max(SIMD[DType.float32, 4](tps.v0, tps.v1, tps.v2, tps.v3), SIMD[DType.float32, 4](0)) * al
        var tsum = tp.reduce_add()
        if tsum <= Float32(0):
            path_ptr[].active = 0
            path_ptr[].pcgState = pcg.state
            return
        var p = tp / tsum
        var u = pcg.next_float()
        var j = 3
        var acc = Float32(0)
        for k in range(3):
            acc += p[k]
            if u < acc:
                j = k
                break
        var w_now = path_ptr[].sss
        var t = -log(max(Float32(1) - pcg.next_float(), Float32(1e-7))) / max(st[j] * w_now.stretch, Float32(1e-12))
        var hit = t >= inter.tHit
        if hit:
            t = inter.tHit
        var tr = exp(-st * t)
        var pdf = tr if hit else st * tr
        if step > Int32(0):
            var sf = st * w_now.fwd_stretch
            var guided = (exp(-sf * t) if hit else sf * exp(-sf * t)) * w_now.fwd_pdf
            if w_now.opposite >= Float32(0):
                var sb = st * w_now.bwd_stretch
                var back = (exp(-sb * t) if hit else sb * exp(-sb * t)) * w_now.bwd_pdf
                guided = guided * (Float32(1) - w_now.bwd_fraction) + back * w_now.bwd_fraction
            pdf = pdf * (Float32(1) - guided_fraction) + guided * guided_fraction
        var w = (tr if hit else al * st * tr) / max((p * pdf).reduce_add(), Float32(1e-30))
        path_ptr[].throughput *= SpectralSample(w[0], w[1], w[2], w[3])
        if hit:
            var at_exit = inter.primId.type == Int8(0) and Int32(inter.primId.id1) == walk.mesh and inter.primId.instanceIdx == walk.instance
            var emat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
            if at_exit and emat.sss_walk_interior() and walk.instance < Int32(0):   # object-space positions otherwise
                # pbrt's subsurface boundary is a dielectric: from inside, a share F of the walk is reflected back in.
                var m = sd.meshes[unsafe_offset=Int(inter.primId.id1)]
                var bv = Int(inter.primId.id2)
                var p0 = mesh_pos(m, Int(mesh_vi(m, bv)))
                var gn = cross(mesh_pos(m, Int(mesh_vi(m, bv + 1))) - p0, mesh_pos(m, Int(mesh_vi(m, bv + 2))) - p0)
                gn = gn * (Float32(1) / max(sqrt(dot(gn, gn)), Float32(1e-30)))
                var cos_i = abs(dot(dir, gn))
                if pcg.next_float() < fr_dielectric(cos_i, Float32(1) / max(emat.albedo.r, Float32(1.0001))):
                    var on = gn if dot(dir, gn) > Float32(0) else -gn
                    var rd = dir - on * (Float32(2) * dot(dir, on))
                    var back = org + dir * inter.tHit - on * Float32(1e-4)
                    ray = Ray(Point3f(back[0], back[1], back[2]), Vec3f(rd[0], rd[1], rd[2]))
                    # The new direction is not the scatter's: its distance is sampled unguided.
                    path_ptr[].sss.stretch = Float32(1); path_ptr[].sss.fwd_stretch = Float32(1); path_ptr[].sss.bwd_stretch = Float32(1)
                    path_ptr[].sss.fwd_pdf = Float32(1); path_ptr[].sss.bwd_pdf = Float32(1)
                    continue
            if at_exit:
                path_ptr[].sss.mesh = Int32(-1)
                path_ptr[].sss.exiting = Int32(1)
                path_ptr[].volume_scattered = Int8(0)       # no longer a pending segment: a later miss is a real one
                path_ptr[].ray = ray
                intersections[unsafe_offset=i] = inter
                path_ptr[].pcgState = pcg.state
                return
            var past = org + dir * (inter.tHit * Float32(1.0001) + Float32(1e-5))
            ray = Ray(Point3f(past[0], past[1], past[2]), ray.direction)
            path_ptr[].sss.stretch = Float32(1)
        else:
            var pt = org + dir * t
            # The next direction: guided toward the surface (or away, near an opposite one), or Henyey-Greenstein.
            var next_iso = path_ptr[].sss.steps > Int32(_SSS_SIMILARITY_STEPS)
            var g_next = Float32(0) if next_iso else walk.g
            var gf_next = Float32(0.75) if next_iso else guided_g
            var bwd_fraction = Float32(0)
            var opp = path_ptr[].sss.opposite
            if opp >= Float32(0):
                var x = min(max(dot(pt - walk.p0, -walk.n0), Float32(0)), opp)
                bwd_fraction = Float32(1) / (Float32(1) + exp((opp - Float32(2) * x) / dl))
            var use_guided = pcg.next_float() < gf_next
            var backward = opp >= Float32(0) and pcg.next_float() < bwd_fraction
            var r1 = pcg.next_float()
            var r2 = pcg.next_float()
            var nd: Vec3f
            var cos_n: Float32
            var hg_pdf: Float32
            if use_guided:
                cos_n = dl - (dl + Float32(1)) * exp(-r1 * phase_log)
                if backward:
                    cos_n = -cos_n
                var fr = Frame.from_z(walk.n0)
                var sn = sqrt(max(Float32(1) - cos_n * cos_n, Float32(0)))
                var phi = Float32(6.283185307) * r2
                nd = Vec3f(fr.x.x, fr.x.y, fr.x.z) * (sn * cos(phi)) + Vec3f(fr.y.x, fr.y.y, fr.y.z) * (sn * sin(phi)) + walk.n0 * cos_n
                hg_pdf = hg_phase(dot(-dir, nd), g_next)
            else:
                var hs = hg_sample(-dir, g_next, r1, r2)
                nd = Vec3f(hs[0], hs[1], hs[2])
                hg_pdf = hs[3]
                cos_n = dot(nd, walk.n0)
            var inv_hg = Float32(1) / max(hg_pdf, Float32(1e-12))
            path_ptr[].sss.fwd_pdf = Float32(0.159154943) / ((dl - cos_n) * phase_log) * inv_hg
            path_ptr[].sss.bwd_pdf = Float32(0.159154943) / ((dl + cos_n) * phase_log) * inv_hg
            path_ptr[].sss.fwd_stretch = Float32(1) - cos_n / dl
            path_ptr[].sss.bwd_stretch = Float32(1) + cos_n / dl
            path_ptr[].sss.bwd_fraction = bwd_fraction
            path_ptr[].sss.stretch = (path_ptr[].sss.bwd_stretch if backward else path_ptr[].sss.fwd_stretch) if use_guided else Float32(1)
            ray = Ray(Point3f(pt[0], pt[1], pt[2]), nd)
        # Russian roulette once a walk has lost most of its weight.
        var m = path_ptr[].throughput.max_component()
        if m < Float32(0.1):
            var q = max(m * Float32(10), Float32(0.05))
            if pcg.next_float() >= q:
                path_ptr[].active = 0
                path_ptr[].pcgState = pcg.state
                return
            path_ptr[].throughput *= Float32(1) / q
    # Out of steps this call: the caller traces the pending segment and calls again.
    path_ptr[].ray = ray
    path_ptr[].volume_scattered = Int8(1)
    intersections[unsafe_offset=i].hit = Int8(0)
    path_ptr[].pcgState = pcg.state


@always_inline
def sss_walks_below(mat: Material, inter: Intersection, med_before: Int32) -> Bool:
    """Would a refraction at this hit start a walk? It then carries no eta^2: the walk leaves into the medium it came from."""
    return mat.sss_walk_interior() and inter.primId.type == Int8(0) and Int(med_before) < 0


@always_inline
def sss_entered(mat: Material, inter: Intersection, ray_dir: Vec3f, new_dir: Vec3f, med_before: Int32, ref sd: SceneView, hit: Vec3f) -> Bool:
    """Did this dielectric crossing refract into a subsurface interior?"""
    if not sss_walks_below(mat, inter, med_before):
        return False
    var gn = _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit)
    return dot(gn, ray_dir) * dot(gn, new_dir) > Float32(0)


@always_inline
def sss_hop[use_gpu: Bool](ref sd: SceneView, mat: Material, inter: Intersection, hit: Vec3f, gn: Vec3f, refracted: Vec3f,
                           wl: SampledWavelengths, mut pcg: PCG32) -> SssExit:
    """The whole hop for an integrator that does not walk across wavefront rounds: entry lobe, walk, surface tint."""
    var ex = sss_walk_to_exit(sd, mat, inter, hit, gn, sss_entry_lobe(face_toward(gn, refracted), mat.albedo.r, pcg), wl, pcg)
    if ex.ok:
        ex.weight = ex.weight * sss_entry_tint[use_gpu](sd, mat, inter, wl)
    return ex
