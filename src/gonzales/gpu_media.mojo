from .bvh import BVH2Node, LightSample, _sample_distant_light_nee, _sample_infinite_light_nee, _sample_point_light_nee, _sample_sphere_light_nee, any_hit_bvh2_core, test_spheres, traverse_bvh2_core, SceneDescriptor2_C
from .curves import Curve_C
from .geometry import Point2f, Point3f, RGB, Vec3f, _is_real_ptr, cross, dot, point3f, vec3f
from .lights import AreaLight_C, DistantLight_C, InfiniteLight_C, LightSampler_C, PointLight_C, area_light_pick_triangle, light_sampler_sample
from .materials import MatKind, Material_C
from .media import Grid_C, MEDIUM_TRACK_MAX_ITERS, MediumInterface_C, Medium_C, NvdbGrid_C, grid_ray_range, grid_sample_density, hg_phase, hg_sample, medium_emission_spectral, medium_grid_for, medium_nvdb_for, medium_sigma_s_spectral, medium_sigma_t_spectral, medium_transmittance_ratio_spectral, nvdb_index_ray, nvdb_majorant_at_world, nvdb_node_exit_t, nvdb_ray_range, nvdb_sample_density, sample_free_flight
from .primitives import Instance_C, Intersection_C, PrimId_C, Ray_C, Sphere_C, TriangleMesh_C, sphere_outward_normal
from .render_state import PathState_C
from .reservoir import reservoir_finalize, reservoir_update
from .restir_vol import VolReservoir, VOL_RIS_CANDIDATES, VOL_RIS_DISTANCE, VOL_TR_UNIT, VolShiftMode, vol_reservoir_init, vol_reservoir_io_null, vol_target_pdf, vol_temporal_spatial_combine
from .rng import PCG32
from .sampling import power_heuristic
from .spectrum import SpectralSample
from max.gpu import block_dim, block_idx, thread_idx
from std.collections import Array
from std.math import exp, log, sqrt
from .gpu_scene import GpuSceneHandle


def update_medium_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    count_dp: Int64,
):
    """Update current_medium_idx for any surface hit with a MediumInterface bound.
    Runs after all material shaders; uses the post-scatter ray direction (same
    convention as CPU rendering.mojo) to determine inside vs outside."""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].active == 0:
        return
    var inter = intersections[unsafe_offset=tid]
    if inter.hit == 0:
        return
    var mat = sd.materials[unsafe_offset=Int(inter.primId.materialIndex)]
    if mat.medium_interface_idx < Int32(0):
        return
    var iface = sd.mediumInterfaces[unsafe_offset=Int(mat.medium_interface_idx)]
    var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var geom_n: Vec3f
    if inter.primId.type == 4:
        # Sphere: outward normal = hit point - center. Medium-bounding
        # volumes (e.g. smoke-plume's "MediumInterface .. Shape sphere")
        # are commonly a big invisible sphere, so this case matters even
        # though sd.spheres otherwise rarely carry sd.materials with real shading.
        var sph = sd.spheres[unsafe_offset=Int(inter.primId.id1)]
        # ray.origin is ALREADY the hit point -- this kernel runs after all
        # material shaders (see the docstring above), and each shader rewrites
        # path.ray to the outgoing ray whose origin sits on the surface.
        # Advancing by tHit again walked a second full hit distance past the
        # sphere and inverted the inside/outside test below. Same bug and same
        # fix as rendering.mojo's CPU medium-interface loop -- see the longer
        # writeup there.
        var ray_org = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
        var hit_pt = ray_org
        geom_n = sphere_outward_normal(point3f(hit_pt), sph.center).to_simd()
    else:
        var mi: Int
        var bv: Int
        if inter.primId.type == 0:
            mi = Int(inter.primId.id1)
            bv = Int(inter.primId.id2)
        elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
            mi = Int(inter.primId.id2 >> 32)
            bv = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
        else:
            return
        var m = sd.meshes[unsafe_offset=mi]
        var v0 = Int(m.vertexIndices[unsafe_offset=bv])
        var v1 = Int(m.vertexIndices[unsafe_offset=bv + 1])
        var v2 = Int(m.vertexIndices[unsafe_offset=bv + 2])
        var p0 = Vec3f(m.points[unsafe_offset=v0*4], m.points[unsafe_offset=v0*4+1], m.points[unsafe_offset=v0*4+2])
        var p1 = Vec3f(m.points[unsafe_offset=v1*4], m.points[unsafe_offset=v1*4+1], m.points[unsafe_offset=v1*4+2])
        var p2 = Vec3f(m.points[unsafe_offset=v2*4], m.points[unsafe_offset=v2*4+1], m.points[unsafe_offset=v2*4+2])
        geom_n = cross(p1 - p0, p2 - p0)
    if dot(ray_dir, geom_n) > Float32(0.0):
        path_ptr[].current_medium_idx = iface.outside_medium_idx
    else:
        path_ptr[].current_medium_idx = iface.inside_medium_idx



@always_inline
def _volume_nee_light(
    path_ptr: Pointer[PathState_C, MutUntrackedOrigin],
    ls: LightSample,
    scatter_pt_w: Vec3f,
    wo: Vec3f,
    g: Float32,
    mut pcg: PCG32,
    use_nvdb: Bool,
    use_dense: Bool,
    grid: Grid_C,
    nvdb_grid: NvdbGrid_C,
    sigma_maj: Float32,
    sigma_t_r: Float32,
    ref sd: SceneDescriptor2_C,
):
    """One NEE sample from ONE non-area light toward a volume scatter point.

    A phase function has no cosine factor, so the BSDF-shaped part of the
    estimator is just the Henyey-Greenstein value for the angle between wo and
    the light direction -- which is why this can consume the same LightSample
    interface the surface shaders use (bvh.mojo's _sample_*_light_nee) without
    a phase-specific weight function. Delta lights (distant/point) carry pdf=1 with any falloff
    already folded into Li, and take MIS weight 1 because no competing
    phase-sampling strategy can hit them; the others are MIS-weighted against
    phase sampling, which shade_core's miss handler weights from the other
    side."""
    var n_spheres = Int(sd.sphereCount)
    if not ls.valid or ls.pdf <= Float32(0.0):
        return
    var edir = Vec3f(ls.wi[0], ls.wi[1], ls.wi[2])
    var e_org = point3f(scatter_pt_w + edir * Float32(0.0002))
    var e_ray = Ray_C(e_org, vec3f(edir))
    # `ls.dist` is measured from scatter_pt_w but the ray starts 0.0002 FURTHER
    # ALONG it, so an untrimmed tmax of ls.dist reaches 0.0002 PAST the light
    # sample -- every time, at any distance. Harmless for a point/distant/
    # infinite light (no geometry sits at that end to be hit) but fatal for a
    # SPHERE light, which is real geometry: the ray hit the sphere and every
    # volume scatter vertex reported it occluded, so a sphere light lit a
    # participating medium only through phase-sampled escapes. Measured on a
    # sphere light over a homogeneous box: 0.39x pbrt. Same defect as the area
    # -light volume NEE one fixed in f79999f4.
    var e_tmax = max(ls.dist - Float32(0.0002), Float32(0.0)) * Float32(0.9995)
    if any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, e_ray, e_tmax,
                         sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances, sd.spheres, n_spheres,
                         materials=sd.materials):
        return
    # Ratio-track transmittance, but only across the span the ray actually
    # spends inside the density grid -- see nvdb_ray_range's docstring for why
    # an unbounded march is not an option for a light with no finite distance.
    var is_het = use_dense or use_nvdb
    if not is_het:
        # Homogeneous: transmittance is closed-form, but only over the span
        # the ray actually spends INSIDE the medium. That span ends at the
        # medium's bounding interface, which this function does not otherwise
        # know, so find it with a closest-hit query. Interface surfaces are
        # invisible to any_hit (they must not occlude), so this deliberately
        # uses the ordinary traversal, whose first hit IS that shell.
        var exit_i = Intersection_C(
            PrimId_C(Int64(0), Int64(0), Int64(-1), Int32(-1), Int8(0), 0, 0, 0),
            Float32(0), Float32(0), Float32(0), Int8(0), 0, 0, 0)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, e_ray, ls.dist,
                           Pointer(to=exit_i), sd.blasNodesArr, sd.blasPrimIdsArr,
                           sd.instances, sd.spheres, n_spheres)
        var span = ls.dist if exit_i.hit == Int8(0) else exit_i.tHit
        var Th = exp(-sigma_t_r * span)
        var ph_h = hg_phase(dot(wo, edir), g)
        var mis_h = Float32(1.0) if ls.is_delta else power_heuristic(ls.pdf, ph_h)
        path_ptr[].estimate += path_ptr[].throughput * medium_emission_spectral(
        ls.Li, path_ptr[].wavelengths, sd.spectral.coeffs, sd.spectral.res,
        sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65) * (Th * ph_h * mis_h / ls.pdf)
        return
    var rng = nvdb_ray_range(nvdb_grid, scatter_pt_w, edir) if use_nvdb else grid_ray_range(grid, scatter_pt_w, edir)
    var t_lo = max(rng[0], Float32(0.0))
    var t_hi = min(rng[1], ls.dist)
    var Te = Float32(1.0)
    if t_hi > t_lo and sigma_maj > Float32(0.0):
        # Same local-majorant segment walk as the free-flight loop -- see the
        # comment there. Ratio tracking stays unbiased under a piecewise
        # majorant for the same memorylessness reason.
        var eray = nvdb_index_ray(nvdb_grid, scatter_pt_w, edir) if use_nvdb else SIMD[DType.float32, 8](0)
        var te = t_lo
        var eseg_end = t_lo - Float32(1.0)
        var esig = Float32(0.0)
        var eiters = 0
        while eiters < MEDIUM_TRACK_MAX_ITERS:
            eiters += 1
            if te >= eseg_end:
                if te >= t_hi:
                    break
                if use_nvdb:
                    var emr = nvdb_majorant_at_world(nvdb_grid, scatter_pt_w + edir * te)
                    esig = emr[0] * sigma_t_r
                    eseg_end = min(nvdb_node_exit_t(eray, te, emr[1]), t_hi)
                else:
                    esig = sigma_maj
                    eseg_end = t_hi
                if esig <= Float32(0.0):
                    te = eseg_end
                    continue
            var ue = pcg.next_float()
            var te_next = te + (-log(max(ue, Float32(1e-7))) / esig)
            if te_next >= eseg_end:
                te = eseg_end
                continue
            te = te_next
            var pe = scatter_pt_w + edir * te
            var de = nvdb_sample_density(nvdb_grid, pe) if use_nvdb else grid_sample_density(grid, pe)
            Te *= Float32(1.0) - (de * sigma_t_r) / esig
            if Te < Float32(1e-4):
                Te = Float32(0.0)
                break
    if Te <= Float32(0.0):
        return
    var ph = hg_phase(dot(wo, edir), g)
    var mis = Float32(1.0) if ls.is_delta else power_heuristic(ls.pdf, ph)
    path_ptr[].estimate += path_ptr[].throughput * medium_emission_spectral(
        ls.Li, path_ptr[].wavelengths, sd.spectral.coeffs, sd.spectral.res,
        sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65) * (Te * ph * mis / ls.pdf)


@always_inline
def _volume_area_light_nee(
    path_ptr: Pointer[PathState_C, MutUntrackedOrigin],
    ref sd: SceneDescriptor2_C,
    i: Int,
    med: Medium_C,
    med_idx: Int,
    sigma_t: RGB,
    use_nvdb: Bool,
    use_dense: Bool,
    ray_org: Vec3f,
    ray_dir: Vec3f,
    t_surf: Float32,
    scatter_pt: Point3f,
    mut pcg: PCG32,
    vol_read: Pointer[VolReservoir, MutUntrackedOrigin],
    vol_write: Pointer[VolReservoir, MutUntrackedOrigin],
    pixel_idx: Int,
    vol_used: Pointer[Int8, MutUntrackedOrigin],
    vol_gbuf_depth: Pointer[Float32, MutUntrackedOrigin],
    vol_gbuf_world_pos: Pointer[Float32, MutUntrackedOrigin],
    vol_frame_w: Int32,
    vol_frame_h: Int32,
):
    """Area-light NEE at a volume scatter vertex: RIS over VOL_RIS_CANDIDATES
    light samples, optional volume-ReSTIR reuse, then one resolved shadow ray."""
    var n_light_sampler = Int(sd.lightSampler.n)
    var n_spheres = Int(sd.sphereCount)
    # ── Volume scatter NEE — area light direct lighting ──────────────
    # Phase 7.2 (docs/A2_restir_migration_plan.md): resampled importance
    # sampling over VOL_RIS_CANDIDATES light samples instead of one.
    #
    # The whole point is the asymmetry between the two halves. Generating
    # a candidate is cheap -- pick a light, a triangle, a barycentric
    # point, evaluate the UNSHADOWED target -- while resolving one costs a
    # visibility ray plus, in a heterogeneous medium, a whole
    # ratio-tracking march for transmittance. So M candidates are
    # generated and exactly ONE is resolved, which is why this can afford
    # to look at many lights for barely more than the price of the single
    # sample it replaces.
    #
    # `tr` is VOL_TR_UNIT at every target evaluation here on purpose: this
    # is the resampling stage, and the target must not contain
    # intermediate transmittance (restir_vol.mojo, seam 1). The real
    # transmittance appears once below, in the resolve, along a ray that
    # is actually traced.
    #
    # With VOL_RIS_CANDIDATES == 1 this reduces EXACTLY to the single-
    # sample estimator it replaced: W = w_sum/(m*p_hat) = (p_hat/q)/p_hat
    # = 1/q, and 1/q is precisely the `al.total_area / light_sel_pdf`
    # factor the old `geom` term carried. That equivalence is the cheapest
    # correctness check available here and is worth preserving.
    if Int(sd.areaLightCount) > 0 and n_light_sampler > 0:
        var ls = LightSampler_C(sd.lightSampler.cdf, Int32(n_light_sampler), Int32(0))
        var scatter_pt_s = scatter_pt.to_simd()
        var scatter_v = Vec3f(scatter_pt_s[0], scatter_pt_s[1], scatter_pt_s[2])
        var res = vol_reservoir_init()
        res.scatter_point = scatter_v
        # sigma_s is a CONSTANT across every candidate at this fixed
        # vertex, so it cancels between w_sum and p_hat(winner) and cannot
        # affect the estimate. It is passed (rather than 1.0) so the
        # payload field means what it says, for the distance-resampling
        # half where vertices genuinely differ in density.
        res.sigma_s = max(med.sigma_s.r, Float32(1e-30))
        res.phase_g = med.g
        res.medium_idx = Int32(med_idx)
        var p_hat_win = Float32(0.0)

        # ── Distance resampling (restir_vol.mojo's VOL_RIS_DISTANCE) ──
        # Each candidate draws its own scatter distance as well as its own
        # light, from the EXACT conditional collision density
        # q(t) = sigma_t e^{-sigma_t t} / (1 - e^{-sigma_t t_surf}). That
        # choice is what makes this cheap: q(t) cancels out of both the
        # RIS weight and the resolve (see the derivation on
        # VOL_RIS_DISTANCE), so nothing below changes except WHERE the
        # target is evaluated and which point gets shadowed.
        #
        # Restricted to homogeneous, achromatic media. Homogeneous because
        # only there is the conditional analytic (heterogeneous needs a
        # rejection-conditioned delta-tracking walk per candidate --
        # unbiased, no marches, but ~M walks per segment). Achromatic
        # because the per-channel transmittance ratio applied to
        # throughput above was computed at t_free, and a resampled vertex
        # sits at a different optical depth; for a grey medium that factor
        # is exactly 1, so the question does not arise. Both guards fail
        # CLOSED -- a medium that does not qualify silently keeps 7.2's
        # fixed-vertex behavior, which is always correct.
        # VOL_RIS_DISTANCE is a compile-time kill switch, so it gates the
        # whole test at compile time rather than sitting in the runtime
        # `and` chain: with the flag off the qualification test is not
        # emitted at all, instead of being evaluated and ANDed with False.
        var dist_ris: Bool
        comptime if VOL_RIS_DISTANCE:
            dist_ris = ((not use_dense) and (not use_nvdb)
                and sigma_t.r > Float32(0.0) and t_surf > Float32(0.0)
                and sigma_t.g == sigma_t.r and sigma_t.b == sigma_t.r
                and med.sigma_s.g == med.sigma_s.r and med.sigma_s.b == med.sigma_s.r)
        else:
            dist_ris = False
        var pc_norm = Float32(0.0)
        if dist_ris:
            pc_norm = Float32(1.0) - exp(-sigma_t.r * t_surf)
            # A segment with essentially no collision probability cannot
            # produce a usable conditional draw; fall back rather than
            # divide by a vanishing normalizer.
            if pc_norm < Float32(1e-6):
                dist_ris = False

        for _cand in range(VOL_RIS_CANDIDATES):
            # Candidate vertex. When distance resampling is off this is
            # exactly the delta-tracking vertex, for every candidate --
            # i.e. bit-identical to 7.2, no extra RNG draw taken.
            var cand_v = scatter_v
            if dist_ris:
                var u_t = pcg.next_float()
                # Inverse CDF of the truncated exponential: exact, cheap.
                var t_c = -log(max(Float32(1.0) - u_t * pc_norm, Float32(1e-7))) / sigma_t.r
                cand_v = ray_org + ray_dir * t_c
            var u_nee = pcg.next_float()
            var ls_result = light_sampler_sample(ls, u_nee)
            var light_idx = ls_result[0]
            var light_sel_pdf = ls_result[1]
            var al = sd.areaLights[unsafe_offset=light_idx]
            var lmesh = sd.meshes[unsafe_offset=Int(al.meshIdx)]
            var lti = area_light_pick_triangle(al, pcg.next_float())
            var r1 = pcg.next_float()
            var r2 = pcg.next_float()
            var lb = lti * 3
            var lv0 = Int(lmesh.vertexIndices[unsafe_offset=lb])
            var lv1 = Int(lmesh.vertexIndices[unsafe_offset=lb + 1])
            var lv2 = Int(lmesh.vertexIndices[unsafe_offset=lb + 2])
            var lp0 = Vec3f(lmesh.points[unsafe_offset=lv0*4], lmesh.points[unsafe_offset=lv0*4+1], lmesh.points[unsafe_offset=lv0*4+2])
            var lp1 = Vec3f(lmesh.points[unsafe_offset=lv1*4], lmesh.points[unsafe_offset=lv1*4+1], lmesh.points[unsafe_offset=lv1*4+2])
            var lp2 = Vec3f(lmesh.points[unsafe_offset=lv2*4], lmesh.points[unsafe_offset=lv2*4+1], lmesh.points[unsafe_offset=lv2*4+2])
            var sqrt_r1 = sqrt(r1)
            var light_point = lp0 * (Float32(1) - sqrt_r1) + lp1 * (sqrt_r1 * (Float32(1) - r2)) + lp2 * (sqrt_r1 * r2)
            var lcross = cross(lp1 - lp0, lp2 - lp0)
            var lcross_len = sqrt(max(Float32(1e-14), dot(lcross, lcross)))
            var light_normal = lcross * (Float32(1) / lcross_len)

            # Every candidate must be streamed, including a rejected one:
            # reservoir_update increments m unconditionally, and RIS's 1/M
            # normalization is only right if m counts candidates CONSIDERED
            # rather than candidates that happened to be usable.
            var w_cand = Float32(0.0)
            var p_hat_cand = Float32(0.0)
            var to_light_c = light_point - cand_v
            var dist_c = sqrt(dot(to_light_c, to_light_c))
            if dist_c > Float32(0.0001) and al.total_area > Float32(0) and light_sel_pdf > Float32(0):
                var lp_v = Vec3f(light_point[0], light_point[1], light_point[2])
                var ln_v = Vec3f(light_normal[0], light_normal[1], light_normal[2])
                p_hat_cand = vol_target_pdf(
                    ray_dir, cand_v, res.sigma_s, med.g,
                    lp_v, ln_v, al.emission, VOL_TR_UNIT)
                if p_hat_cand > Float32(0.0):
                    # q is the AREA-measure pdf of this sample: probability
                    # of picking this light, times a uniform 1/total_area.
                    var q_cand = light_sel_pdf / al.total_area
                    w_cand = p_hat_cand / q_cand
            if reservoir_update(res.state, w_cand, pcg.next_float()):
                res.light_point = Vec3f(light_point[0], light_point[1], light_point[2])
                res.light_normal = Vec3f(light_normal[0], light_normal[1], light_normal[2])
                res.le = al.emission
                res.light_idx = Int32(light_idx)
                res.valid = Int8(1)
                # The winning VERTEX travels with the winning light: the
                # resolve below shadow-rays from here, and the payload is
                # what a reusing pixel would read. Identical to scatter_v
                # when distance resampling is off.
                res.scatter_point = cand_v
                p_hat_win = p_hat_cand

        # Phase 7.3: temporal reuse when this call has a real per-pixel
        # slot (gpu_render_sample only); otherwise the single-frame path
        # 7.2 already shipped, unchanged. vol_temporal_spatial_combine
        # finalizes res.state AND writes it back to vol_write[pixel_idx]
        # internally -- no separate persistence step needed here. Spatial
        # reuse is NOT enabled: gbuf_depth/gbuf_world_pos are left at
        # their null-sentinel default inside vol_reservoir_io_null(), and
        # vol_temporal_spatial_combine's spatial pass self-disables on
        # that (see restir_vol.mojo's own null-safety contract).
        # `vol_used[i]` (one entry per PATH SLOT for this dispatch/frame,
        # NOT per pixel across frames -- that's vol_read/vol_write's job)
        # guards a real bug found verifying the CPU wiring: a single
        # path can have MULTIPLE real scatter events inside a dense
        # medium within one frame (this scene's optical depth is ~8
        # through the sphere, so 10+ scatters per sample is common) --
        # _sample_medium_core runs once per bounce ROUND, so each of
        # those events independently called vol_temporal_spatial_combine
        # and overwrote vol_write[pixel_idx], leaving only the LAST
        # in-frame scatter's result actually persisted. Traced live: the
        # reservoir's state.m plateaued around 40 (never reaching
        # VOL_TEMPORAL_M_CAP=64) and MSE-vs-a-16384spp-reference got
        # WORSE from 16 to 256 accumulated frames instead of better --
        # exactly the "stalled convergence = bias" signature documented
        # in project_restir_migration's DI Bug 2 section. This affected
        # the GPU-only commit (1685154c) too, silently, since that
        # verification pass's methodology (fixed-budget MSE across 5
        # seeds) didn't happen to expose it the way this session's CPU
        # convergence-rate check did. Fix: only the path's FIRST real
        # scatter this frame gets the temporal combine (mirrors DI's own
        # "one NEE per pixel per frame" scoping, applied per-PATH since
        # media have no fixed bounce-0 the way surfaces do); every later
        # in-frame scatter falls back to the plain single-frame RIS
        # estimator (7.2's original, always-correct behavior) instead of
        # corrupting the persisted reservoir.
        # Distance resampling and temporal reuse COMPOSE (they were once
        # mutually exclusive here, on a premise that turned out to be
        # backwards -- see the shift-mode choice below).
        var vol_reuse_ok = (pixel_idx >= 0 and _is_real_ptr(vol_read)
            and _is_real_ptr(vol_used) and vol_used[unsafe_offset=i] == Int8(0))
        if vol_reuse_ok:
            vol_used[unsafe_offset=i] = Int8(1)
            var vol_io = vol_reservoir_io_null()
            vol_io.read = vol_read
            vol_io.write = vol_write
            vol_io.gbuf_depth = vol_gbuf_depth
            vol_io.gbuf_world_pos = vol_gbuf_world_pos
            vol_io.frame_w = vol_frame_w
            vol_io.frame_h = vol_frame_h
            var ray_o_s = path_ptr[].ray.origin.to_simd()
            var ray_d_s = path_ptr[].ray.direction.to_simd()
            var ray_o = Vec3f(ray_o_s[0], ray_o_s[1], ray_o_s[2])
            var ray_d = Vec3f(ray_d_s[0], ray_d_s[1], ray_d_s[2])
            # Which shift is valid depends on how this frame's vertex was
            # produced, and the two cases are opposites:
            #
            # dist_ris ON -> `identity`. The vertex came from q(t), which
            # depends only on sigma_t and t_surf -- the same for every
            # frame at this pixel -- so a donor's vertex is a draw from
            # exactly our own proposal. Domains match, Jacobian 1.
            #
            # dist_ris OFF -> `retarget`. The vertex is delta-tracking's
            # single t_free, a point mass that differs every frame; our
            # proposal could never have produced the donor's. Import only
            # the light sample and keep our own vertex, which is plain
            # ReSTIR DI reuse.
            #
            # The guard that used to sit here had this backwards: it
            # claimed `identity` re-targets onto this pixel's vertex and
            # so could not survive distance resampling. `identity` does
            # the opposite -- it keeps the DONOR's vertex verbatim -- so
            # the configuration it permitted (dist_ris off + reuse) was
            # the inconsistent one, and the configuration it forbade was
            # the well-founded one. See the resolve below for the bug
            # that inconsistency caused.
            var vol_shift = VolShiftMode.identity if dist_ris else VolShiftMode.retarget
            vol_temporal_spatial_combine(
                res, ray_o, ray_d, Int32(med_idx), pcg,
                vol_io, pixel_idx, vol_shift)
        else:
            reservoir_finalize(res.state, p_hat_win)

        # ── Resolve: one visibility ray + one transmittance march, for
        # the winner only.
        if res.valid != Int8(0) and res.state.w > Float32(0.0):
            var light_point = res.light_point.to_simd()
            var light_normal = res.light_normal.to_simd()
            # Shadow-ray from the WINNER's vertex, ALWAYS -- this is the
            # one point that must agree with the target evaluation, since
            # reservoir_finalize set W = w_sum/(m * p_hat(winner)) using
            # p_hat at res.scatter_point. Tracing F from anywhere else
            # multiplies an F from one vertex by a W from another and the
            # RIS identity is gone.
            #
            # This used to read `res.scatter_point if dist_ris else
            # scatter_pt_s`, which was a live bias whenever temporal reuse
            # won with a donor sample: the donor's vertex went into p_hat
            # (and into W) while the shadow ray still left from THIS
            # frame's vertex. It stayed hidden because both points lie on
            # the same camera ray in a homogeneous fog, so the two targets
            # are close and the error is a quiet scale factor rather than
            # anything visible.
            #
            # Equal to scatter_pt_s whenever nothing moved the vertex --
            # every candidate writes cand_v, which is scatter_v itself
            # unless distance resampling drew a new one -- so the default
            # (no reuse, no distance resampling) path is unchanged.
            var resolve_pt = res.scatter_point.to_simd()
            var to_light = light_point - resolve_pt
            var dist_sq = dot(to_light, to_light)
            var dist = sqrt(dist_sq)
            var shadow_dir = to_light * (Float32(1) / dist)
            var cos_l = -dot(light_normal, shadow_dir)
            if cos_l > Float32(0):
                var shad_org = point3f(resolve_pt + shadow_dir * Float32(0.0002))
                var shad_ray = Ray_C(shad_org, vec3f(shadow_dir))
                var shad_tmax = max(dist - Float32(0.0002), Float32(0.0)) * Float32(0.9995)
                if not any_hit_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, shad_ray, shad_tmax, sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances, sd.spheres, n_spheres, materials=sd.materials):
                    var T: RGB
                    if med.grid_idx >= Int32(0) or med.nvdb_idx >= Int32(0):
                        # Ratio-tracking transmittance through the grid (see
                        # sample_medium_gpu's docstring). The density
                        # lookup returns 0 past the grid's bounds for
                        # EITHER source (grid_sample_density's [p0,p1] box,
                        # nvdb_sample_density's index bbox), so this
                        # naturally stops attenuating once the shadow ray
                        # exits the medium -- same dual-source dispatch as
                        # the free-flight sampling above.
                        var use_nvdb_s = med.nvdb_idx >= Int32(0)
                        var grid_s = sd.grids[unsafe_offset=Int(med.grid_idx)] if not use_nvdb_s else Grid_C(
                            Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Int32(0), Int32(0), Int32(0),
                            Point3f(Float32(0), Float32(0), Float32(0)), Point3f(Float32(0), Float32(0), Float32(0)),
                            SIMD[DType.float32, 16](0), Float32(0))
                        var nvdb_grid_s = sd.nvdbGrids[unsafe_offset=Int(med.nvdb_idx)] if use_nvdb_s else NvdbGrid_C(
                            Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(), Int64(0), SIMD[DType.float32, 16](0),
                            SIMD[DType.float32, 16](0), Vec3f(Float32(0), Float32(0), Float32(0)),
                            Point3f(Float32(0), Float32(0), Float32(0)), Point3f(Float32(0), Float32(0), Float32(0)), Float32(0))
                        var majorant_s = nvdb_grid_s.max_density if use_nvdb_s else grid_s.max_density
                        var sigma_maj_s = majorant_s * sigma_t.r
                        var Tval = Float32(1.0)
                        if sigma_maj_s > Float32(0.0):
                            var ts = Float32(0.0)
                            var siters = 0
                            while siters < MEDIUM_TRACK_MAX_ITERS:
                                siters += 1
                                var us = pcg.next_float()
                                ts += -log(max(us, Float32(1e-7))) / sigma_maj_s
                                if ts >= dist:
                                    break
                                var ps = resolve_pt + shadow_dir * ts
                                var density_s = nvdb_sample_density(nvdb_grid_s, ps) if use_nvdb_s else grid_sample_density(grid_s, ps)
                                Tval *= Float32(1.0) - (density_s * sigma_t.r) / sigma_maj_s
                                if Tval < Float32(1e-4):
                                    Tval = Float32(0.0)
                                    break
                        T = RGB(Tval, Tval, Tval)
                    else:
                        # Beer-Lambert over the part of the segment that is
                        # actually INSIDE the medium, not the whole way to
                        # the light.
                        #
                        # This used to attenuate over `dist` unconditionally,
                        # which charges the vacuum between the medium's
                        # boundary and the light for extinction it never
                        # applies. The grid branch above is accidentally
                        # immune -- its density lookup returns 0 outside the
                        # grid, so ratio tracking simply stops attenuating --
                        # which is why only homogeneous media showed it.
                        # Measured on an area-lit slab: homogeneous read
                        # 0.117x pbrt where uniformgrid read 0.954x at the
                        # same geometry, against a predicted e^2 = 7.4x for
                        # the 2 units of vacuum involved.
                        #
                        # The exit distance is the first interface surface
                        # along the segment. Occlusion has already been
                        # ruled out above, so any hit here is a non-opaque
                        # boundary. Scope: this finds ONE exit, which is
                        # exact for a ray leaving a single convex medium --
                        # the case every medium scene in the corpus has --
                        # and does not model re-entry or nested media. A
                        # general version needs the medium-transition walk
                        # bdpt.mojo's _visible_transmittance already does.
                        #
                        # test_spheres is REQUIRED here, not optional:
                        # traverse_bvh2_core walks the mesh/curve BVH only,
                        # and analytic sd.spheres live in their own flat array.
                        # A `MediumInterface .. Shape "sphere"` boundary --
                        # the single most common way to bound a medium, and
                        # what every fog/cloud test scene here uses -- was
                        # therefore never found, so t_med stayed at the FULL
                        # distance to the light and Beer-Lambert charged the
                        # vacuum outside the medium for extinction it never
                        # applies. Measured on a tau=8 fog sphere lit by a
                        # mesh quad: exp(-2.02*6) instead of exp(-2.02*2),
                        # i.e. PT read 0.0000567 where the same scene with a
                        # mesh-box boundary reads 0.0355 (574x too dark).
                        # Same root cause and same shape as the sphere case
                        # bdpt.mojo's _visible_transmittance needed, and as
                        # the vacuum-attenuation bug this very branch was
                        # written to fix -- that fix just never covered the
                        # sphere-bounded case.
                        var t_med = dist
                        var _exit_inter = Array[Intersection_C, 1](fill=Intersection_C(
                            PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0)),
                            Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0)))
                        var exit_ptr = _exit_inter.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
                        exit_ptr[unsafe_offset=0].hit = Int8(0)
                        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, shad_ray,
                                           shad_tmax, exit_ptr, sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
                        test_spheres(sd.spheres, n_spheres, shad_ray, exit_ptr)
                        # test_spheres ignores shad_tmax (it bounds only by
                        # an already-recorded closer hit), so a sphere past
                        # the light would otherwise set t_med > dist and
                        # over-attenuate instead of under-.
                        if exit_ptr[unsafe_offset=0].hit != Int8(0) and exit_ptr[unsafe_offset=0].tHit <= shad_tmax:
                            var exit_mat = sd.materials[unsafe_offset=Int(exit_ptr[unsafe_offset=0].primId.materialIndex)]
                            if exit_mat.type == MatKind.interface:
                                t_med = exit_ptr[unsafe_offset=0].tHit
                        T = RGB(exp(-sigma_t.r * t_med), exp(-sigma_t.g * t_med), exp(-sigma_t.b * t_med))
                    # The old `geom` also carried al.total_area/light_sel_pdf,
                    # i.e. 1/q -- that now lives inside res.state.w, so the
                    # geometry factor here is the bare cos_l/dist^2.
                    var geom = cos_l / dist_sq
                    var ph_a = hg_phase(dot(-ray_dir, shadow_dir), med.g)
                    # MIS against phase sampling. A volume scatter sets
                    # lastBsdfPdf to the phase pdf and specularBounce to 0,
                    # so a phase-sampled ray that lands on this same emitter
                    # is ALREADY weighted by power_heuristic(pdf_bsdf,
                    # pdf_light) in shading.mojo's emitter-hit handler --
                    # but this side carried no weight at all, so the two
                    # strategies summed to more than one. Invisible for
                    # small/distant lights, where phase sampling almost
                    # never finds the emitter and this weight is ~1; it grew
                    # to 1.70x too bright once the lights subtended a large
                    # solid angle. pdf_light is deliberately spelled exactly
                    # as the emitter-hit side spells it -- MIS is only
                    # correct if both halves agree on the pdf.
                    var al_win = sd.areaLights[unsafe_offset=Int(res.light_idx)]
                    var sel_lo = sd.lightSampler.cdf[unsafe_offset=Int(res.light_idx)]
                    var sel_hi = sd.lightSampler.cdf[unsafe_offset=Int(res.light_idx) + 1]
                    var sel_pdf_win = max(sel_hi - sel_lo, Float32(1e-6))
                    var mis_w = Float32(1.0)
                    if al_win.total_area > Float32(0.0):
                        var pdf_light = dist_sq * sel_pdf_win / (cos_l * al_win.total_area)
                        mis_w = power_heuristic(pdf_light, ph_a)
                    path_ptr[].estimate += path_ptr[].throughput * medium_emission_spectral(
                        res.le * T, path_ptr[].wavelengths, sd.spectral.coeffs, sd.spectral.res,
                        sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65) * (geom * ph_a * mis_w * res.state.w)


# Inlined so `sd` stays in kernel param space: a non-inlined call needs its
# address, which copies the whole descriptor to local memory per thread.
@always_inline
def _sample_medium_core(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    i: Int,
    ref sd: SceneDescriptor2_C,
    # Phase 7.3 (docs/A2_restir_migration_plan.md, project_restir_migration
    # memory): volume-scatter TEMPORAL reuse. Decomposed pointers, not one
    # `vol_io: VolReservoirIO` argument -- same defensive convention this
    # file already applies to SpectralHandle at this same kind of boundary
    # (see spectrum.mojo's comment on rgb_to_spectral_sample). `pixel_idx`
    # only means anything when this call came from gpu_render_sample (one
    # path per pixel); the wavefront batch path always leaves it at -1,
    # which the code below treats identically to "no reuse".
    vol_read: Pointer[VolReservoir, MutUntrackedOrigin] = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
    vol_write: Pointer[VolReservoir, MutUntrackedOrigin] = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
    pixel_idx: Int = -1,
    # One Int8 per PATH SLOT (indexed by `i`, this call's own index -- NOT
    # by pixel_idx), reset to 0 once at the start of this dispatch/frame by
    # the caller: guards against a single path scattering more than once
    # inside a dense medium within one frame (common -- see the long
    # comment at this buffer's read site for the real bug this fixes).
    vol_used: Pointer[Int8, MutUntrackedOrigin] = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling(),
    # Phase 7.3 spatial reuse (2026-09-08): SAME G-buffers DI's own spatial
    # reuse already reads (handle[].atrous_depth_buf/gbuf_worldpos_buf on
    # GPU, depth_int/world_pos_int on CPU) -- harmless to pass unconditionally
    # (mirrors DI's own convention), vol_temporal_spatial_combine's own
    # `_is_real_ptr`/frame_w>0/frame_h>0 checks gate the spatial pass off
    # when they're not real or the caller (batch wavefront) has no G-buffer.
    vol_gbuf_depth: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    vol_gbuf_world_pos: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    vol_frame_w: Int32 = Int32(0),
    vol_frame_h: Int32 = Int32(0),
):
    """Apply medium transmittance along the ray segment and possibly scatter
    or absorb inside the medium. On scatter, performs direct area-light NEE
    with isotropic phase function. Shared verbatim between the GPU kernel
    (sample_medium_gpu, one call per thread) and the CPU driver
    (render_all_tiles's per-sample loop). See docs/09_volumetric_media.md
    for the delta/ratio-tracking theory, the local-majorant optimization,
    and the volumetric NEE/connect bug history behind the choices below.

    Homogeneous media (grid_idx < 0): closed-form analytic transmittance.
    Heterogeneous media (grid_idx >= 0 or nvdb_idx >= 0): delta tracking
    against a local majorant; NEE shadow rays use the matching ratio-tracking
    transmittance estimator, which naturally stops attenuating once the ray
    exits the density source's bounds.

    Both heterogeneous sources use the RED channel exclusively for
    majorant/accept-reject decisions: exact for the achromatic density
    fields supported today, would need per-wavelength free-flight sampling
    with spectral MIS to extend to a colored medium. The homogeneous branch
    carries only the RATIO of each channel's transmittance to the
    red-channel one actually sampled, lifted into the 4 hero lanes by
    band-picking (see spectrum.mojo's rgb_bands_to_spectral_sample) — real
    chromatic extinction is the same unimplemented, separate piece of work.
    """
    var n_spheres = Int(sd.sphereCount)
    var path_ptr = paths.unsafe_offset(i)
    if path_ptr[].active == 0:
        return
    var med_idx = Int(path_ptr[].current_medium_idx)
    if med_idx < 0 or med_idx >= Int(sd.mediumCount):
        return
    var inter = intersections[unsafe_offset=i]
    if inter.hit == 0:
        return
    var med = sd.mediums[unsafe_offset=med_idx]
    var sigma_t = med.sigma_a + med.sigma_s
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var t_surf = inter.tHit
    var ray_org = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)

    # ── Free flight ────────────────────────────────────────────────────────
    # ONE call for both medium kinds: sample_free_flight (geometry.mojo) picks
    # the homogeneous closed form or heterogeneous delta tracking against a
    # local majorant, and hands both back in the same shape. The delta-tracking
    # loop used to be written out inline right here, which is precisely why it
    # was the path tracer's alone -- SPPM and BDPT/VCM called the homogeneous
    # sampler unconditionally and rendered every density field as uniform fog.
    # See sample_free_flight's own comment for that bug.
    #
    # `use_dense`/`use_nvdb`/`grid`/`nvdb_grid`/`sigma_maj` stay resolved HERE
    # too, not because the free flight needs them (it resolves its own), but
    # because the volume-scatter NEE further down ratio-tracks its shadow ray
    # against the same grid and majorant.
    var use_nvdb = med.nvdb_idx >= Int32(0)
    var use_dense = med.grid_idx >= Int32(0)
    var grid = medium_grid_for(med, sd.grids)
    var nvdb_grid = medium_nvdb_for(med, sd.nvdbGrids)
    var majorant_density = nvdb_grid.max_density if use_nvdb else grid.max_density
    var sigma_maj = majorant_density * sigma_t.r

    var ff = sample_free_flight(
        med, sd.grids, sd.nvdbGrids, ray_org, ray_dir, t_surf, pcg,
        path_ptr[].wavelengths, sd.spectral.coeffs, sd.spectral.res,
        sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
    # Volumetric emission (pbrt NanoVDBMedium's temperature grid) accumulated
    # over every majorant candidate along the tracked segment, already spectral
    # and already weighted by each candidate's absorption fraction. Zero for a
    # non-emissive or homogeneous medium, so this costs nothing there.
    path_ptr[].estimate += path_ptr[].throughput * ff.emission

    # Set by the homogeneous branch when a spectral table is available: the
    # LANE-AVERAGED single-scattering albedo, which the scatter/absorb coin
    # below is played on instead of red's. Negative means "not set".
    var p_scatter_spec = Float32(-1.0)
    if not (use_dense or use_nvdb):
        # ── Homogeneous-only throughput bookkeeping ────────────────────────
        # Delta tracking carries transmittance implicitly in its accept/reject
        # decisions; the closed form does not, so the analytic branch -- and
        # ONLY it -- multiplies the chromatic transmittance ratio in, on the
        # pass-through path as well as the collision one.
        var t_seg = min(ff.t_free, t_surf)
        # The weight MUST be the ratio of each channel's transmittance to the
        # one the distance was sampled from, using the SAME sigma_t. It used
        # to use a spectral Beer-Lambert whose sigma_t came from an
        # illuminant-style RGB->spectral conversion the function itself
        # documents as "not a physically rigorous spectral-extinction fit".
        # That made the numerator's effective extinction differ from the
        # sampling one, so the weight was exp((sigma_t.r - sigma_t_spec) * t)
        # -- growing exponentially with distance and compounding once per
        # scattering event. Measured against an analytic answer of exactly
        # 1.0, a conservative homogeneous medium rendered 0.470 at tau=2,
        # 6.0 at tau=4 and 1979 at tau=8 (single pixels at 6.1e6), and the
        # divergence survived normalising that round trip. Spectral colouring
        # of EXTINCTION is therefore not applied here; doing it properly
        # means sampling the free flight from a hero wavelength and combining
        # wavelengths with MIS, which is real chromatic-media work, not a
        # colour conversion. Upsample sigma_t to the 4 hero lanes FIRST via
        # spec_refl_unbounded (the coefficient-safe smooth upsampler; grey
        # media pass through it exactly -- verified in
        # Tests/unit/test_coefficient_upsampling.mojo), THEN exponentiate PER
        # LANE -- the same shared helper bdpt.mojo/sppm.mojo use, so CPU PT /
        # GPU PT / VCM / SPPM all treat a medium's colour identically.
        # See docs/02_spectra_and_color.md, "Chromatic extinction".
        path_ptr[].throughput *= medium_transmittance_ratio_spectral(
            med, t_seg, ff.pdf, path_ptr[].wavelengths,
            sd.spectral.coeffs, sd.spectral.res, sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
        if ff.collided:
            # Chromatic scattering ratio; 1 for a grey medium.
            #
            # The extra sigma_t.r closes the estimator against the scatter/
            # absorb coin below, which is played at RED's albedo
            # (sigma_s.r/sigma_t.r) whatever lane the distance came from.
            # With the segment factor above now exp(-sigma_i*t)/p_bar rather
            # than a ratio to red's own exponential, the full product is
            #   p_bar * (sigma_s.r/sigma_t.r)            <- actually sampled
            #     * exp(-sigma_i*t)/p_bar * sigma_t.r * sigma_s_i/sigma_s.r
            #   = sigma_s_i * exp(-sigma_i*t)            <- what is wanted
            # for every lane i. p_bar cancels, which is the point: correctness
            # does not depend on WHICH lane the free flight was drawn from.
            # sigma_s(lambda) comes from the SAME smooth upsampler as the
            # sigma_t(lambda) in the exponential above. It used to be
            # BAND-PICKED off the RGB triple, which made the per-lane albedo
            # sigma_s(lambda)/sigma_t(lambda) a ratio of two inconsistent
            # conversions -- invisible at 2-3 scatters, hue-inverting over a
            # subsurface walk's hundreds (see medium_sigma_s_spectral).
            var ss_r = max(med.sigma_s.r, Float32(1e-30))
            if sd.spectral.res > 0:
                # The scatter/absorb coin is ONE coin for all four lanes, so
                # whichever albedo it is played on becomes the reference every
                # lane is corrected against. Playing it on RED made that
                # correction sigma_s(lambda)/sigma_s.r, which for skin runs up
                # to 1.48 and is systematically >1 in blue -- and a subsurface
                # walk multiplies hundreds of them, so the product is
                # log-normal and explodes. Measured on head.pbrt the moment
                # the albedo became chromatic: max 1.3 -> 1.75e9, with 1.4% of
                # pixels above 100x the median against pbrt's 0.000%.
                #
                # Play it on the LANE MEAN instead. The correction is then
                # alpha_i/alpha_bar, centred on 1 and bounded either side, and
                # it pairs with the free-flight MIS weight (also centred on 1)
                # so neither factor drifts. Unbiased either way -- E[.] is
                # alpha_i per scatter for both -- this is purely variance.
                var sig_t_spec = medium_sigma_t_spectral(
                    med, path_ptr[].wavelengths, sd.spectral.coeffs, sd.spectral.res,
                    sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                var sig_s_spec = medium_sigma_s_spectral(
                    med, path_ptr[].wavelengths, sd.spectral.coeffs, sd.spectral.res,
                    sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                var a0 = sig_s_spec.v0 / max(sig_t_spec.v0, Float32(1e-30))
                var a1 = sig_s_spec.v1 / max(sig_t_spec.v1, Float32(1e-30))
                var a2 = sig_s_spec.v2 / max(sig_t_spec.v2, Float32(1e-30))
                var a3 = sig_s_spec.v3 / max(sig_t_spec.v3, Float32(1e-30))
                var abar = (a0 + a1 + a2 + a3) * Float32(0.25)
                if abar < Float32(1e-6): abar = Float32(1e-6)
                p_scatter_spec = abar
                var inv_ab = Float32(1.0) / abar
                path_ptr[].throughput *= SpectralSample(
                    sig_t_spec.v0 * a0 * inv_ab, sig_t_spec.v1 * a1 * inv_ab,
                    sig_t_spec.v2 * a2 * inv_ab, sig_t_spec.v3 * a3 * inv_ab)
            else:
                # No spectral table: lanes carry RGB, so red IS the reference
                # and this is the original expression unchanged.
                var sig_s_spec = medium_sigma_s_spectral(
                    med, path_ptr[].wavelengths, sd.spectral.coeffs, sd.spectral.res,
                    sd.spectral.cie_x, sd.spectral.cie_y, sd.spectral.cie_z, sd.spectral.d65)
                path_ptr[].throughput *= (sig_s_spec * (Float32(1.0) / ss_r)) * sigma_t.r

    if not ff.collided:
        path_ptr[].pcgState = pcg.state
        return
    var t_free = ff.t_free
    # Density cancels between sigma_s and sigma_t, so this is the same
    # expression for both medium kinds.
    var albedo_r = med.sigma_s.r / max(sigma_t.r, Float32(1e-7))

    # Both branches above already `return` early for the "no real collision"
    # case (homogeneous: t_free >= t_surf; heterogeneous: not collided) — so
    # reaching here always means a real scatter/absorb event at t_free.
    var p_scatter = p_scatter_spec if p_scatter_spec >= Float32(0.0) else albedo_r
    var u_mode = pcg.next_float()
    if u_mode < p_scatter:
        # A capped path dies at this real scatter, BEFORE this vertex's NEE
        # and before a new direction is sampled -- pbrt's volpath does exactly
        # this (`if (depth++ >= maxDepth) { terminated = true; return false; }`
        # ahead of its own SampleLd). The segment that BROUGHT the path here
        # was already traced and any emitter on it already collected, which is
        # the whole point of carrying `at_cap` instead of killing a round
        # earlier. Absorption below needs no such guard: it terminates anyway.
        if path_ptr[].at_cap != Int8(0):
            path_ptr[].pcgState = pcg.state
            path_ptr[].active = Int8(0)
            return
        # Volume scatter: compute scatter point
        var scatter_pt = path_ptr[].ray.origin + path_ptr[].ray.direction * t_free
        _volume_area_light_nee(path_ptr, sd, i, med, med_idx, sigma_t, use_nvdb, use_dense,
            ray_org, ray_dir, t_surf, scatter_pt, pcg,
            vol_read, vol_write, pixel_idx, vol_used,
            vol_gbuf_depth, vol_gbuf_world_pos, vol_frame_w, vol_frame_h)

        # ── Volume scatter NEE — distant / point / sphere / infinite ─────
        # The block above samples triangle AREA lights only, and is gated on
        # Int(sd.areaLightCount) > 0. Every other light type contributed nothing at a
        # volume scatter point, so a medium lit by a sky dome and/or a sun --
        # which is every nanovdb cloud scene in the pbrt-v4 corpus
        # (bunny-cloud, explosion, disney-cloud), none of which has an area
        # light -- received NO direct lighting at all. It was lit purely by
        # phase-sampled paths that random-walk back out and happen to escape,
        # which is both far too dark and extremely high variance: that is
        # where the sparse bright "firefly" dots on those renders came from.
        # Mirrors the same four-light-type sweep the surface shaders already
        # do via the shared LightSample interface.
        #
        # Heterogeneous only: the ratio-track inside needs real grid bounds to
        # terminate (see nvdb_ray_range). A HOMOGENEOUS medium has no density
        # grid to bound the march and its extent is the bounding shape, which
        # this function does not know -- so it keeps its previous behavior
        # rather than getting a subtly wrong transmittance. That remains a
        # real, pre-existing gap for homogeneous media.
        var scatter_w = scatter_pt.to_simd()
        var wo_v = -ray_dir
        for dl_i in range(Int(sd.distantLightCount)):
            _volume_nee_light(path_ptr, _sample_distant_light_nee(sd.distantLights[unsafe_offset=dl_i]),
                scatter_w, wo_v, med.g, pcg, use_nvdb, use_dense, grid, nvdb_grid, sigma_maj, sigma_t.r,
                sd)
        for pl_i in range(Int(sd.pointLightCount)):
            _volume_nee_light(path_ptr, _sample_point_light_nee(sd.pointLights[unsafe_offset=pl_i], scatter_w),
                scatter_w, wo_v, med.g, pcg, use_nvdb, use_dense, grid, nvdb_grid, sigma_maj, sigma_t.r,
                sd)
        for sph_i in range(n_spheres):
            if sd.spheres[unsafe_offset=sph_i].isAreaLight == Int8(1):
                _volume_nee_light(path_ptr, _sample_sphere_light_nee(sd.spheres[unsafe_offset=sph_i], n_spheres, scatter_w, pcg),
                    scatter_w, wo_v, med.g, pcg, use_nvdb, use_dense, grid, nvdb_grid, sigma_maj, sigma_t.r,
                    sd)
        for inf_i in range(Int(sd.infiniteLightCount)):
            _volume_nee_light(path_ptr,
                _sample_infinite_light_nee(sd.infiniteLights[unsafe_offset=inf_i], Point2f(pcg.next_float(), pcg.next_float())),
                scatter_w, wo_v, med.g, pcg, use_nvdb, use_dense, grid, nvdb_grid, sigma_maj, sigma_t.r,
                sd)
        # Sample the scatter direction from the medium's Henyey-Greenstein
        # phase function. `g` was parsed into Medium_C all along but never
        # used: scattering was hardcoded isotropic (uniform sphere), so a
        # strongly forward-scattering medium -- disney-cloud sets g=0.877 --
        # diffused light instead of forwarding it and rendered far too dark.
        # hg_sample falls back to the uniform sphere for |g| < 1e-3, so
        # isotropic media (bunny-cloud, explosion) are bit-for-bit unchanged.
        var u1 = pcg.next_float()
        var u2 = pcg.next_float()
        path_ptr[].pcgState = pcg.state
        var hs = hg_sample(-ray_dir, med.g, u1, u2)
        path_ptr[].ray = Ray_C(scatter_pt, Vec3f(hs[0], hs[1], hs[2]))
        path_ptr[].specularBounce = Int8(0)
        path_ptr[].lastBsdfPdf = hs[3]
        path_ptr[].volume_scattered = Int8(1)
        # A volume scatter IS the real scattering event the emitter-hit MIS
        # measures from, and it puts the ray origin exactly there.
        path_ptr[].mis_null_dist = Float32(0.0)
        # Interior random-walk steps of a `Material "subsurface"` object are
        # NOT path bounces and are not charged to maxdepth. Skin1 at
        # sssdragon's scale has a red-channel single-scattering albedo of
        # 0.996 and ~37 extinction events per scene unit, so a walk routinely
        # runs tens to hundreds of steps before it escapes or is absorbed --
        # against pbrt's default maxdepth of 5 the object would render nearly
        # black. pbrt never spends path depth on the interior either (its
        # BSSRDF resolves the whole thing analytically); the walk here is
        # bounded instead by absorption, by Russian roulette, and finally by
        # the render loop's own round budget, which is extended to cover it
        # (see _SSS_WALK_ROUNDS in rendering.mojo / gpu.mojo).
        if med.is_sss == Int32(0):
            path_ptr[].bounce += 1
        intersections[unsafe_offset=i].hit = Int8(0)  # no surface hit this bounce
    else:
        # Absorbed
        path_ptr[].pcgState = pcg.state
        path_ptr[].active = Int8(0)

def sample_medium_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    sd: SceneDescriptor2_C,
    count_dp: Int64,
    # Phase 7.3: only gpu_render_wavefront_kernels(...) callers that pass
    # use_vol_restir=1 AND real buffers get reuse -- see _sample_medium_core's
    # own comment for why these stay decomposed rather than one VolReservoirIO.
    use_vol_restir: Int32 = Int32(0),
    vol_read: Pointer[VolReservoir, MutUntrackedOrigin] = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
    vol_write: Pointer[VolReservoir, MutUntrackedOrigin] = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
    vol_used: Pointer[Int8, MutUntrackedOrigin] = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling(),
    vol_gbuf_depth: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    vol_gbuf_world_pos: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    vol_frame_w: Int32 = Int32(0),
    vol_frame_h: Int32 = Int32(0),
):
    """GPU kernel wrapper: bounds-check, then call the SAME
    _sample_medium_core the CPU driver (render_all_tiles) calls."""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    # tid IS the pixel index here only when use_vol_restir=1 -- that only
    # ever comes from gpu_render_sample (one path per pixel per dispatch),
    # mirroring shade_diffuse_gpu's identical restir_has_state contract.
    var vol_has_state = use_vol_restir != Int32(0) and _is_real_ptr(vol_read)
    _sample_medium_core(
        paths, intersections, tid, sd,
        vol_read=vol_read,
        vol_write=vol_write,
        pixel_idx=tid if vol_has_state else -1,
        vol_used=vol_used,
        vol_gbuf_depth=vol_gbuf_depth,
        vol_gbuf_world_pos=vol_gbuf_world_pos,
        vol_frame_w=vol_frame_w,
        vol_frame_h=vol_frame_h,
    )
