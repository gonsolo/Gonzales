from std.math import ceildiv, sqrt, log, exp, cos, sin, max
from std.memory import alloc
from std.algorithm import parallelize
from std.time import perf_counter_ns
from .geometry import RGB, Point3f, Vec3f, point3f, vec3f, sphere_outward_normal, Ray_C, Intersection_C, PrimId_C, PathState_C, TileResult_C, Sphere_C, AreaLight_C, LightSampler_C, light_sampler_sample, dot, cross, Medium_C, MediumInterface_C, Grid_C, grid_sample_density, INV_FOUR_PI, curve_piece_endpoints, _curve_perp_axis
from .bvh import SceneDescriptor2_C, traverse_bvh2_core, test_spheres, any_hit_bvh2_core
from .shading import shade_core_cpu_nee
from .rng import PCG32
from .sampling import TileSamplerParams_C, encode_morton2, sobol_get_sample_index, sobol_sample, gaussian_sample_1d, derive_pcg_seeds, gaussian_norm, mix_bits_u64, gen_primary_ray_state
from .guide import GuideGrid, guide_merge, null_guide
from .spectrum import SampledWavelengths


def render_tile(
    rasterToCamera: UnsafePointer[Float32, MutAnyOrigin],
    cameraToWorld: UnsafePointer[Float32, MutAnyOrigin],
    tileMinX: Int32, tileMinY: Int32, tileMaxX: Int32, tileMaxY: Int32,
    samplerParamsPtr: UnsafePointer[TileSamplerParams_C, MutAnyOrigin],
    scenePtr: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin],
    resultsPtr: UnsafePointer[TileResult_C, MutAnyOrigin],
    maxDepth: Int32,
    guide_read: GuideGrid,
    guide_write: GuideGrid,
):
    var sp = samplerParamsPtr[0]
    var scene = scenePtr[0]
    var maxD = Int(maxDepth)
    var tileW = Int(tileMaxX - tileMinX)
    var tileH = Int(tileMaxY - tileMinY)
    var spp = Int(sp.samplesPerPixel)
    var n = tileW * tileH * spp
    var log2spp = Int(sp.log2SamplesPerPixel)
    var n_base4 = Int(sp.nBase4Digits)
    var matrices = sp.sobolMatrices

    var hash_bits = UInt64(mix_bits_u64(UInt64(0) ^ UInt64(sp.sobolSeed)))
    var seed_dim0 = UInt32(hash_bits & UInt64(0xFFFFFFFF))
    var seed_dim1 = UInt32(0)

    var paths = alloc[PathState_C](n)
    var intersections = alloc[Intersection_C](n)

    # Generate primary rays from Sobol film samples
    var idx = 0
    var rng_seed = sp.rngSeed
    for iy in range(tileH):
        for ix in range(tileW):
            var px = Int32(tileMinX) + Int32(ix)
            var py = Int32(tileMinY) + Int32(iy)
            for si in range(spp):
                var (ray, pcg_state, pcg_inc, sobol_idx, wavelengths) = gen_primary_ray_state(
                    px, py, Int32(si) + sp.sampleIndexOffset,
                    log2spp, n_base4,
                    seed_dim0, seed_dim1, rng_seed, matrices,
                    rasterToCamera, cameraToWorld,
                    sp.filterNormX, sp.filterSigma, sp.filterSupportX,
                    sp.filterNormY, sp.filterSupportY,
                    sp.filterType,
                )
                paths[idx] = PathState_C(
                    ray,
                    RGB(Float32(1.0)),
                    RGB(Float32(0.0)),
                    RGB(Float32(0.0)),
                    Int32(0), pcg_state, pcg_inc,
                    Int8(1), Int8(0), Int8(0), Int8(0),
                    Float32(0.0),
                    Int32(-1),
                    Int32(3), sobol_idx,
                    wavelengths,
                )
                idx += 1

    # Multi-bounce path trace
    for _ in range(maxD):
        var anyActive = False
        for i in range(n):
            if paths[i].active != 0:
                anyActive = True
                break
        if not anyActive:
            break
        for i in range(n):
            if paths[i].active == 0:
                continue
            traverse_bvh2_core(scene.bvh2Nodes, scene.primIds, scene.meshes, scene.curves,
                               paths[i].ray, Float32(1.0e38), intersections + i,
                               scene.blasNodesArr, scene.blasPrimIdsArr, scene.instances)
            if scene.sphereCount > 0:
                test_spheres(scene.spheres, Int(scene.sphereCount), paths[i].ray, intersections + i)
        for i in range(n):
            if paths[i].active == 0:
                continue
            # ── Volume transmittance sampling ──────────────────────────
            # Homogeneous (grid_idx < 0): closed-form analytic transmittance.
            # Heterogeneous (grid_idx >= 0, "uniformgrid"): delta tracking
            # against the grid's majorant — see sample_medium_gpu's docstring
            # in gpu.mojo for the full rationale (same algorithm, mirrored
            # here for the CPU path).
            var med_idx = Int(paths[i].current_medium_idx)
            if med_idx >= 0 and Int(scene.mediumCount) > 0 and intersections[i].hit != Int8(0):
                var med = scene.mediums[med_idx]
                var sigma_t = med.sigma_a + med.sigma_s
                var t_surf = intersections[i].tHit
                var pcg_vol = PCG32(paths[i].pcgState, paths[i].pcgInc)

                var have_collision = False
                var t_free = Float32(0.0)
                var p_absorb = Float32(0.0)
                var p_scatter = Float32(0.0)

                if med.grid_idx >= Int32(0):
                    var grid = scene.grids[Int(med.grid_idx)]
                    var sigma_maj = grid.max_density * sigma_t.r
                    if sigma_maj > Float32(0.0):
                        var ray_org = SIMD[DType.float32, 3](paths[i].ray.origin.x, paths[i].ray.origin.y, paths[i].ray.origin.z)
                        var ray_dir = SIMD[DType.float32, 3](paths[i].ray.direction.x, paths[i].ray.direction.y, paths[i].ray.direction.z)
                        var t = Float32(0.0)
                        var iters = 0
                        while iters < 10000:
                            iters += 1
                            var u = pcg_vol.next_float()
                            t += -log(max(u, Float32(1e-7))) / sigma_maj
                            if t >= t_surf:
                                break
                            var density = grid_sample_density(grid, ray_org + t * ray_dir)
                            var sigma_t_real = density * sigma_t.r
                            var u2 = pcg_vol.next_float()
                            if u2 < sigma_t_real / sigma_maj:
                                have_collision = True
                                t_free = t
                                break
                        # Density cancels out of the scatter/absorb ratio given a
                        # real collision already happened — see gpu.mojo.
                        p_absorb  = med.sigma_a.r / max(sigma_t.r, Float32(1e-7))
                        p_scatter = med.sigma_s.r / max(sigma_t.r, Float32(1e-7))
                else:
                    var sigma_maj = sigma_t.r
                    if sigma_maj > Float32(0.0):
                        var u_free = pcg_vol.next_float()
                        t_free = -log(max(u_free, Float32(1e-7))) / sigma_maj
                        var t_seg = min(t_free, t_surf)
                        paths[i].throughput *= RGB(exp(-sigma_t.r * t_seg), exp(-sigma_t.g * t_seg), exp(-sigma_t.b * t_seg))
                        have_collision = t_free < t_surf
                        p_absorb  = med.sigma_a.r / sigma_maj
                        p_scatter = med.sigma_s.r / sigma_maj
                paths[i].pcgState = pcg_vol.state

                if have_collision:
                        var pcg2 = PCG32(paths[i].pcgState, paths[i].pcgInc)
                        var u_mode = pcg2.next_float()
                        paths[i].pcgState = pcg2.state
                        if u_mode < p_absorb:
                            paths[i].throughput = RGB(Float32(0))
                            paths[i].active = Int8(0)
                        elif u_mode < p_absorb + p_scatter:
                            var pcg3 = PCG32(paths[i].pcgState, paths[i].pcgInc)
                            var u1 = pcg3.next_float()
                            var u2 = pcg3.next_float()
                            paths[i].pcgState = pcg3.state
                            var cos_theta = Float32(2) * u1 - Float32(1)
                            var sin_theta = sqrt(max(Float32(0), Float32(1) - cos_theta * cos_theta))
                            var phi = Float32(6.28318530718) * u2
                            var scatter_pt = paths[i].ray.origin + paths[i].ray.direction * t_free
                            # ── Volume scatter NEE — area light direct lighting ──────────
                            if Int(scene.areaLightCount) > 0 and Int(scene.lightSampler.n) > 0:
                                var pcg_nee = PCG32(paths[i].pcgState, paths[i].pcgInc)
                                var u_nee = pcg_nee.next_float()
                                var ls_result = light_sampler_sample(scene.lightSampler, u_nee)
                                var light_idx = ls_result[0]
                                var light_sel_pdf = ls_result[1]
                                var al = scene.areaLights[light_idx]
                                var lmesh = scene.meshes[Int(al.meshIdx)]
                                var lti = Int(pcg_nee.next_uint() % UInt32(max(Int(al.n_tris), 1)))
                                var r1 = pcg_nee.next_float()
                                var r2 = pcg_nee.next_float()
                                paths[i].pcgState = pcg_nee.state
                                var lb = lti * 3
                                var lv0 = Int(lmesh.vertexIndices[lb])
                                var lv1 = Int(lmesh.vertexIndices[lb + 1])
                                var lv2 = Int(lmesh.vertexIndices[lb + 2])
                                var lp0 = SIMD[DType.float32, 3](lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
                                var lp1 = SIMD[DType.float32, 3](lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
                                var lp2 = SIMD[DType.float32, 3](lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])
                                var sqrt_r1 = sqrt(r1)
                                var light_point = lp0 * (Float32(1) - sqrt_r1) + lp1 * (sqrt_r1 * (Float32(1) - r2)) + lp2 * (sqrt_r1 * r2)
                                var lcross = cross(lp1 - lp0, lp2 - lp0)
                                var lcross_len = sqrt(dot(lcross, lcross))
                                var light_normal = lcross * (Float32(1) / max(lcross_len, Float32(1e-7)))
                                var scatter_pt_s = scatter_pt.to_simd()
                                var to_light = light_point - scatter_pt_s
                                var dist_sq = dot(to_light, to_light)
                                var dist = sqrt(dist_sq)
                                if dist > Float32(0.0001) and al.total_area > Float32(0):
                                    var shadow_dir = to_light * (Float32(1) / dist)
                                    var cos_l = -dot(light_normal, shadow_dir)
                                    if cos_l > Float32(0):
                                        var shad_org = point3f(scatter_pt_s + shadow_dir * Float32(0.0002))
                                        var shad_ray = Ray_C(shad_org, vec3f(shadow_dir))
                                        var shad_tmax = dist * Float32(0.9995)
                                        if not any_hit_bvh2_core(scene.bvh2Nodes, scene.primIds, scene.meshes, scene.curves, shad_ray, shad_tmax,
                                                                  scene.blasNodesArr, scene.blasPrimIdsArr, scene.instances):
                                            var T: RGB
                                            if med.grid_idx >= Int32(0):
                                                var grid_s = scene.grids[Int(med.grid_idx)]
                                                var sigma_maj_s = grid_s.max_density * sigma_t.r
                                                var Tval = Float32(1.0)
                                                if sigma_maj_s > Float32(0.0):
                                                    var ts = Float32(0.0)
                                                    var siters = 0
                                                    while siters < 10000:
                                                        siters += 1
                                                        var us = pcg_nee.next_float()
                                                        ts += -log(max(us, Float32(1e-7))) / sigma_maj_s
                                                        if ts >= dist:
                                                            break
                                                        var ps = scatter_pt_s + shadow_dir * ts
                                                        var density_s = grid_sample_density(grid_s, ps)
                                                        Tval *= Float32(1.0) - (density_s * sigma_t.r) / sigma_maj_s
                                                        if Tval < Float32(1e-4):
                                                            Tval = Float32(0.0)
                                                            break
                                                    paths[i].pcgState = pcg_nee.state
                                                T = RGB(Tval, Tval, Tval)
                                            else:
                                                T = RGB(exp(-sigma_t.r * dist), exp(-sigma_t.g * dist), exp(-sigma_t.b * dist))
                                            var geom = al.total_area * cos_l / (dist_sq * light_sel_pdf)
                                            paths[i].estimate += paths[i].throughput * al.emission * T * (geom * INV_FOUR_PI)
                            paths[i].volume_scattered = Int8(1)
                            paths[i].ray = Ray_C(scatter_pt, Vec3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta))
                            paths[i].specularBounce = Int8(0)
                            intersections[i].hit = Int8(0)
        for i in range(n):
            if paths[i].active == 0:
                continue
            shade_core_cpu_nee(paths, intersections, scene.bvh2Nodes, scene.primIds,
                               scene.meshes, scene.curves, scene.materials,
                               scene.areaLights, Int(scene.areaLightCount),
                               scene.textures, i,
                               scene.distantLights, Int(scene.distantLightCount),
                               scene.pointLights, Int(scene.pointLightCount),
                               scene.infiniteLights, Int(scene.infiniteLightCount),
                               scene.spheres, Int(scene.sphereCount),
                               scene.lightSampler, sp.sobolMatrices, guide_read,
                               blasNodesArr=scene.blasNodesArr, blasPrimIdsArr=scene.blasPrimIdsArr,
                               instances=scene.instances, guide_write=guide_write, spectral=scene.spectral)
        # ── Medium interface transitions ──────────────────────────
        for i in range(n):
            if paths[i].active == 0:
                continue
            if intersections[i].hit == Int8(0):
                continue
            var mi_mat_idx = Int(intersections[i].primId.materialIndex)
            if mi_mat_idx < 0 or scene.mediumIfaceCount == Int64(0):
                continue
            var mi_mat = scene.materials[mi_mat_idx]
            if mi_mat.medium_interface_idx < Int32(0):
                continue
            var iface = scene.mediumInterfaces[Int(mi_mat.medium_interface_idx)]
            var mi_n: Vec3f
            if intersections[i].primId.type == 4:
                # Sphere: outward normal = hit point - center. Medium-bounding
                # volumes are commonly a big invisible sphere (e.g.
                # smoke-plume's MediumInterface .. Shape sphere), so this
                # case matters even though spheres rarely carry real shading.
                var mi_sph = scene.spheres[Int(intersections[i].primId.id1)]
                var mi_hit = paths[i].ray.origin + paths[i].ray.direction * intersections[i].tHit
                mi_n = sphere_outward_normal(mi_hit, mi_sph.center)
            else:
                var mi_mesh_idx: Int
                var mi_base_vidx: Int
                if intersections[i].primId.type == 0:
                    mi_mesh_idx = Int(intersections[i].primId.id1)
                    mi_base_vidx = Int(intersections[i].primId.id2)
                else:
                    mi_mesh_idx = Int(intersections[i].primId.id2 >> 32)
                    mi_base_vidx = Int(intersections[i].primId.id2 & 0xFFFFFFFF) * 3
                var mi_m = scene.meshes[mi_mesh_idx]
                var mi_vi0 = Int(mi_m.vertexIndices[mi_base_vidx])
                var mi_vi1 = Int(mi_m.vertexIndices[mi_base_vidx + 1])
                var mi_vi2 = Int(mi_m.vertexIndices[mi_base_vidx + 2])
                var mi_p0 = Point3f(mi_m.points[mi_vi0*4], mi_m.points[mi_vi0*4+1], mi_m.points[mi_vi0*4+2])
                var mi_p1 = Point3f(mi_m.points[mi_vi1*4], mi_m.points[mi_vi1*4+1], mi_m.points[mi_vi1*4+2])
                var mi_p2 = Point3f(mi_m.points[mi_vi2*4], mi_m.points[mi_vi2*4+1], mi_m.points[mi_vi2*4+2])
                var mi_e1 = mi_p1 - mi_p0; var mi_e2 = mi_p2 - mi_p0
                mi_n = Vec3f(mi_e1.y*mi_e2.z - mi_e1.z*mi_e2.y, mi_e1.z*mi_e2.x - mi_e1.x*mi_e2.z, mi_e1.x*mi_e2.y - mi_e1.y*mi_e2.x)
            var mi_dot = paths[i].ray.direction.x * mi_n.x + paths[i].ray.direction.y * mi_n.y + paths[i].ray.direction.z * mi_n.z
            if mi_dot > Float32(0):
                paths[i].current_medium_idx = iface.outside_medium_idx
            else:
                paths[i].current_medium_idx = iface.inside_medium_idx

    # Accumulate the spp samples per pixel and emit one result per pixel.
    idx = 0
    var out = 0
    for iy in range(tileH):
        for ix in range(tileW):
            var px = Int32(tileMinX) + Int32(ix)
            var py = Int32(tileMinY) + Int32(iy)
            var sumL = RGB(Float32(0.0))
            var sumA = RGB(Float32(0.0))
            var sumW = Float32(0.0)
            for _ in range(spp):
                sumL += paths[idx].estimate
                sumA += paths[idx].albedo
                sumW += sp.filterWeight
                idx += 1
            resultsPtr[out] = TileResult_C(sumL, sumA, sumW, px, py)
            out += 1

    intersections.free()
    paths.free()



def _fmt_f1(v: Float64) -> String:
    var i = Int(v)
    var frac = Int((v - Float64(i)) * 10.0 + 0.5)
    if frac >= 10:
        i += 1; frac = 0
    return String(i) + "." + String(frac)

def fmt_time(s: Float64) -> String:
    var sec = Int(s)
    var min = sec // 60
    var rem = sec % 60
    if min > 0:
        var rs = String(rem)
        if rem < 10: rs = "0" + rs
        return String(min) + "m " + rs + "s"
    return _fmt_f1(s) + "s"

def progress_str(done: Int, total: Int, elapsed: Float64, unit: String) -> String:
    var pct = _fmt_f1(Float64(done) * 100.0 / Float64(total))
    var est = Float64(0.0)
    if done > 0:
        est = elapsed * Float64(total) / Float64(done)
    return ("Rendering: " + String(done) + " / " + String(total)
        + " " + unit + " (" + pct + "%) | Elapsed: " + fmt_time(elapsed)
        + " | Total Est.: " + fmt_time(est) + "                ")


def render_all_tiles(
    raster_to_camera: UnsafePointer[Float32, MutAnyOrigin],
    camera_to_world: UnsafePointer[Float32, MutAnyOrigin],
    min_x: Int32, min_y: Int32, max_x: Int32, max_y: Int32,
    tile_w: Int32, tile_h: Int32,
    sampler_params: UnsafePointer[TileSamplerParams_C, MutAnyOrigin],
    scene: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin],
    results: UnsafePointer[TileResult_C, MutAnyOrigin],
    max_depth: Int32,
    quiet: Bool = False,
    guide_read: GuideGrid = null_guide(),
    write_guides: UnsafePointer[GuideGrid, MutAnyOrigin] = UnsafePointer[GuideGrid, MutAnyOrigin].unsafe_dangling(),
    n_write_guides: Int = 0,
):
    var res_x = Int(max_x - min_x)
    var tw = Int(tile_w)
    var th = Int(tile_h)
    var max_tile_pixels = tw * th

    var n_tiles_x = ceildiv(Int(max_x) - Int(min_x), tw)
    var n_tiles_y = ceildiv(Int(max_y) - Int(min_y), th)
    var n_tiles = n_tiles_x * n_tiles_y

    # One scratch buffer per tile so threads never alias each other's writes.
    var tile_bufs = alloc[TileResult_C](n_tiles * max_tile_pixels)

    # Progress counter — incremented after each tile (racy, display-only).
    var done_ptr = alloc[Int32](1)
    done_ptr[0] = Int32(0)
    var t0 = perf_counter_ns()
    # Print every ~5% of tiles (at least every 1 tile).
    var print_step = max(n_tiles // 20, 1)

    @parameter
    def render_one(tile_idx: Int):
        var ty_i = tile_idx // n_tiles_x
        var tx_i = tile_idx % n_tiles_x
        var tx = Int(min_x) + tx_i * tw
        var ty = Int(min_y) + ty_i * th
        var tx_max = Int32(min(tx + tw, Int(max_x)))
        var ty_max = Int32(min(ty + th, Int(max_y)))
        var tw_actual = Int(tx_max) - tx
        var th_actual = Int(ty_max) - ty
        var tile_buf = tile_bufs + tile_idx * max_tile_pixels
        var gw = null_guide()
        if n_write_guides > 0:
            # Range-based assignment: tiles [k*stride, (k+1)*stride) → write_guides[k].
            # This guarantees no two concurrently running tiles (on different cores)
            # ever write to the same shard, eliminating all cross-core cache ping-pong.
            var stride = max(n_tiles // n_write_guides, 1)
            var shard_idx = min(tile_idx // stride, n_write_guides - 1)
            gw = write_guides[shard_idx]
        render_tile(
            raster_to_camera, camera_to_world,
            Int32(tx), Int32(ty), tx_max, ty_max,
            sampler_params, scene, tile_buf, max_depth, guide_read, gw)
        for iy in range(th_actual):
            for ix in range(tw_actual):
                var src = iy * tw_actual + ix
                var dst = (ty + iy - Int(min_y)) * res_x + (tx + ix - Int(min_x))
                results[dst] = tile_buf[src]
        done_ptr[0] += Int32(1)
        var d = Int(done_ptr[0])
        if not quiet and (d % print_step == 0 or d == n_tiles):
            var elapsed = Float64(perf_counter_ns() - t0) / 1.0e9
            print(progress_str(d, n_tiles, elapsed, "tiles"), end="\r")

    parallelize[render_one](n_tiles)
    # Merge per-tile-group write guides into [0] so caller gets unified result.
    if n_write_guides > 1:
        for i in range(1, n_write_guides):
            guide_merge(write_guides[0], write_guides[i])
    if not quiet:
        var total_s = Float64(perf_counter_ns() - t0) / 1.0e9
        print("Rendering: " + String(n_tiles) + " / " + String(n_tiles)
            + " tiles (100.0%) | Done: " + fmt_time(total_s) + "                ")
    done_ptr.free()
    tile_bufs.free()


# Shoot one unjittered center ray per pixel; record geometric normal and depth.
# normals_out: n_pixels*3 floats (Nx,Ny,Nz unit vectors; background = (0,0,1)).
# depth_out:   n_pixels floats (first-hit tHit; background = 1e38).
def render_aux_buffers(
    rasterToCamera: UnsafePointer[Float32, MutAnyOrigin],
    cameraToWorld:  UnsafePointer[Float32, MutAnyOrigin],
    min_x: Int32, min_y: Int32, max_x: Int32, max_y: Int32,
    scene: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin],
    normals_out: UnsafePointer[Float32, MutAnyOrigin],
    depth_out:   UnsafePointer[Float32, MutAnyOrigin],
):
    var w  = Int(max_x - min_x)
    var h  = Int(max_y - min_y)
    var n_pixels = w * h
    var sd = scene[0]
    var org = Point3f(cameraToWorld[12], cameraToWorld[13], cameraToWorld[14])
    var isects = alloc[Intersection_C](n_pixels)

    @parameter
    def trace_pixel(i: Int):
        var py = i // w
        var px = i % w
        var filmX = Float32(Int(min_x) + px) + Float32(0.5)
        var filmY = Float32(Int(min_y) + py) + Float32(0.5)

        # rasterToCamera (column-major 4×4), no filter offset
        var cx = rasterToCamera[0]*filmX + rasterToCamera[4]*filmY + rasterToCamera[12]
        var cy = rasterToCamera[1]*filmX + rasterToCamera[5]*filmY + rasterToCamera[13]
        var cz = rasterToCamera[2]*filmX + rasterToCamera[6]*filmY + rasterToCamera[14]
        var cw = rasterToCamera[3]*filmX + rasterToCamera[7]*filmY + rasterToCamera[15]
        if cw != Float32(0.0) and cw != Float32(1.0):
            cx /= cw; cy /= cw; cz /= cw
        var cl = sqrt(cx*cx + cy*cy + cz*cz)
        if cl > Float32(0): cx /= cl; cy /= cl; cz /= cl

        # cameraToWorld rotation (upper-left 3×3)
        var dir = Vec3f(
            cameraToWorld[0]*cx + cameraToWorld[4]*cy + cameraToWorld[8]*cz,
            cameraToWorld[1]*cx + cameraToWorld[5]*cy + cameraToWorld[9]*cz,
            cameraToWorld[2]*cx + cameraToWorld[6]*cy + cameraToWorld[10]*cz,
        )
        var dl = dir.length()
        if dl > Float32(0): dir = dir / dl

        var ray = Ray_C(org, dir)
        traverse_bvh2_core(sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves, ray, Float32(1e38), isects + i,
                           sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances)
        if Int(sd.sphereCount) > 0:
            test_spheres(sd.spheres, Int(sd.sphereCount), ray, isects + i)

        var normal = Vec3f(Float32(0), Float32(0), Float32(1))   # background
        var d  = Float32(1e38)

        if isects[i].hit != Int8(0):
            d = isects[i].tHit
            var typ = Int(isects[i].primId.type)
            if typ == 4:
                # Sphere: normal = normalize(hit_point - center)
                var si  = Int(isects[i].primId.id1)
                normal = sphere_outward_normal(org + dir*d, sd.spheres[si].center)
            elif typ == 5:
                # Native curve: reconstruct the geometric normal the same way
                # shade_hair does (shading.mojo) — h/v come straight from
                # intersect_curve via Intersection_C.u/.v, no tessellated mesh.
                var curve = sd.curves[Int(isects[i].primId.id1)]
                var h = max(Float32(-0.99), min(Float32(0.99), isects[i].u))
                var piece = min(Int(curve.n_pieces) - 1, max(0, Int(isects[i].v * Float32(curve.n_pieces))))
                var (q0, q1, _, _) = curve_piece_endpoints(curve, piece)
                var seg_axis = q1 - q0
                var seg_len = sqrt(dot(seg_axis, seg_axis))
                var tangent: SIMD[DType.float32, 3]
                if seg_len > Float32(1e-8):
                    tangent = seg_axis * (Float32(1.0) / seg_len)
                else:
                    tangent = SIMD[DType.float32, 3](Float32(1), Float32(0), Float32(0))
                var n_perp = _curve_perp_axis(tangent)
                var b_perp0 = cross(tangent, n_perp)
                var geo = n_perp * h + b_perp0 * sqrt(max(Float32(0.0), Float32(1.0) - h*h))
                var gl = sqrt(dot(geo, geo))
                if gl > Float32(0):
                    normal = Vec3f(geo[0] / gl, geo[1] / gl, geo[2] / gl)
            elif typ == 0 or typ == 1 or typ == 2 or typ == 3:
                # Triangle: geometric normal from edge cross product
                var mesh_idx: Int
                var base_vidx: Int
                if typ == 0:
                    mesh_idx  = Int(isects[i].primId.id1)
                    base_vidx = Int(isects[i].primId.id2)
                else:
                    mesh_idx  = Int(isects[i].primId.id2 >> 32)
                    base_vidx = Int(isects[i].primId.id2 & 0xFFFFFFFF) * 3
                var mesh = sd.meshes[mesh_idx]
                var vi0 = Int(mesh.vertexIndices[base_vidx])
                var vi1 = Int(mesh.vertexIndices[base_vidx + 1])
                var vi2 = Int(mesh.vertexIndices[base_vidx + 2])
                var p0 = Point3f(mesh.points[vi0*4], mesh.points[vi0*4+1], mesh.points[vi0*4+2])
                var p1 = Point3f(mesh.points[vi1*4], mesh.points[vi1*4+1], mesh.points[vi1*4+2])
                var p2 = Point3f(mesh.points[vi2*4], mesh.points[vi2*4+1], mesh.points[vi2*4+2])
                var e1 = p1 - p0; var e2 = p2 - p0
                normal = Vec3f(e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x)
                var nl = normal.length()
                if nl > Float32(0): normal = normal / nl
            # else: unrecognized hit type (shouldn't occur for a real
            # Intersection_C — type==6 only ever exists as a transient BVH
            # leaf marker, resolved into type 0-3 with instanceIdx set before
            # traversal returns) — leave `normal` at its background default
            # rather than guessing.
            # Flip to face incoming ray
            if normal.dot(-dir) < Float32(0):
                normal = -normal

        normals_out[i*3 + 0] = normal.x
        normals_out[i*3 + 1] = normal.y
        normals_out[i*3 + 2] = normal.z
        depth_out[i] = d

    parallelize[trace_pixel](n_pixels)
    isects.free()


# Normalize TileResult_C[] → per-pixel float RGB arrays.
def normalize_film(
    results: UnsafePointer[TileResult_C, MutAnyOrigin],
    count: Int32,
    iso: Float32,
    max_component_value: Float32,
    beauty_out: UnsafePointer[Float32, MutAnyOrigin],
    albedo_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var scale = iso / Float32(100)
    for i in range(Int(count)):
        var r = results[i]
        var w = r.filterWeight
        if w == Float32(0):
            beauty_out[i * 3 + 0] = Float32(0)
            beauty_out[i * 3 + 1] = Float32(0)
            beauty_out[i * 3 + 2] = Float32(0)
            albedo_out[i * 3 + 0] = Float32(0)
            albedo_out[i * 3 + 1] = Float32(0)
            albedo_out[i * 3 + 2] = Float32(0)
            continue
        var b = RGB(r.estimate.r / w * scale, r.estimate.g / w * scale, r.estimate.b / w * scale)
        if b.r != b.r or b.r < Float32(0): b.r = Float32(0)
        if b.g != b.g or b.g < Float32(0): b.g = Float32(0)
        if b.b != b.b or b.b < Float32(0): b.b = Float32(0)
        if max_component_value > Float32(0):
            var mx = max(b.r, max(b.g, b.b))
            if mx > max_component_value:
                b *= max_component_value / mx
        beauty_out[i * 3 + 0] = b.r
        beauty_out[i * 3 + 1] = b.g
        beauty_out[i * 3 + 2] = b.b
        var a = r.albedo / w
        albedo_out[i * 3 + 0] = a.r
        albedo_out[i * 3 + 1] = a.g
        albedo_out[i * 3 + 2] = a.b

