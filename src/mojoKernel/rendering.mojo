from std.math import ceildiv, sqrt, log, exp, cos, sin
from std.memory import alloc
from std.algorithm import parallelize
from std.time import perf_counter_ns
from .geometry import RGB, Point3f, Vec3f, Ray_C, Intersection_C, PathState_C, TileResult_C, Sphere_C, dot, Medium_C, MediumInterface_C
from .bvh import SceneDescriptor2_C, traverse_bvh2_core, test_spheres
from .shading import shade_core_cpu_nee
from .rng import PCG32
from .sampling import TileSamplerParams_C, encode_morton2, sobol_get_sample_index, sobol_sample, gaussian_sample_1d, derive_pcg_seeds, mojo_gaussian_norm, mix_bits_u64


@export
def mojo_render_tile_v2(
    rasterToCamera: UnsafePointer[Float32, MutAnyOrigin],
    cameraToWorld: UnsafePointer[Float32, MutAnyOrigin],
    tileMinX: Int32, tileMinY: Int32, tileMaxX: Int32, tileMaxY: Int32,
    samplerParamsPtr: UnsafePointer[TileSamplerParams_C, MutAnyOrigin],
    scenePtr: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin],
    resultsPtr: UnsafePointer[TileResult_C, MutAnyOrigin],
    maxDepth: Int32,
):
    var sp = samplerParamsPtr[0]
    var scene = scenePtr[0]
    var maxD = Int(maxDepth)
    var orgX = cameraToWorld[12]; var orgY = cameraToWorld[13]; var orgZ = cameraToWorld[14]
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
    for iy in range(tileH):
        for ix in range(tileW):
            var px = Int32(tileMinX) + Int32(ix)
            var py = Int32(tileMinY) + Int32(iy)
            var morton_base = encode_morton2(UInt32(px), UInt32(py)) << UInt64(log2spp)
            for si in range(spp):
                var morton_idx = morton_base | UInt64(si)
                var sobol_idx = sobol_get_sample_index(morton_idx, 0, log2spp, n_base4)
                var u0 = sobol_sample(Int(sobol_idx), 0, seed_dim0, matrices)
                var u1 = sobol_sample(Int(sobol_idx), 1, seed_dim1, matrices)
                var deltaX = gaussian_sample_1d(u0, sp.filterNormX, sp.filterSigma, sp.filterSupportX)
                var deltaY = gaussian_sample_1d(u1, sp.filterNormY, sp.filterSigma, sp.filterSupportY)
                var filmX = Float32(px) + Float32(0.5) + deltaX
                var filmY = Float32(py) + Float32(0.5) + deltaY

                # rasterToCamera transform (column-major)
                var cx = rasterToCamera[0]*filmX + rasterToCamera[4]*filmY + rasterToCamera[12]
                var cy = rasterToCamera[1]*filmX + rasterToCamera[5]*filmY + rasterToCamera[13]
                var cz = rasterToCamera[2]*filmX + rasterToCamera[6]*filmY + rasterToCamera[14]
                var cw = rasterToCamera[3]*filmX + rasterToCamera[7]*filmY + rasterToCamera[15]
                if cw != Float32(0.0) and cw != Float32(1.0):
                    cx /= cw; cy /= cw; cz /= cw
                var camDir = SIMD[DType.float32, 3](cx, cy, cz)
                var camLen = dot(camDir, camDir)
                if camLen > Float32(0.0):
                    camDir = camDir * (Float32(1.0) / sqrt(camLen))

                # cameraToWorld rotation (3×3)
                var dx = cameraToWorld[0]*camDir[0] + cameraToWorld[4]*camDir[1] + cameraToWorld[8]*camDir[2]
                var dy = cameraToWorld[1]*camDir[0] + cameraToWorld[5]*camDir[1] + cameraToWorld[9]*camDir[2]
                var dz = cameraToWorld[2]*camDir[0] + cameraToWorld[6]*camDir[1] + cameraToWorld[10]*camDir[2]
                var worldDir = SIMD[DType.float32, 3](dx, dy, dz)
                var dirLen = dot(worldDir, worldDir)
                if dirLen > Float32(0.0):
                    worldDir = worldDir * (Float32(1.0) / sqrt(dirLen))

                var (pcg_state, pcg_inc) = derive_pcg_seeds(px, py, Int32(si), sp.rngSeed)
                paths[idx] = PathState_C(
                    Ray_C(Point3f(orgX, orgY, orgZ), Vec3f(worldDir[0], worldDir[1], worldDir[2])),
                    RGB(Float32(1.0), Float32(1.0), Float32(1.0)),
                    RGB(Float32(0.0), Float32(0.0), Float32(0.0)),
                    RGB(Float32(0.0), Float32(0.0), Float32(0.0)),
                    Int32(0), pcg_state, pcg_inc,
                    Int8(1), Int8(0), Int8(0), Int8(0),
                    Float32(0.0),
                    Int32(-1),
                )
                idx += 1

    # Multi-bounce path trace
    for _ in range(maxD + 1):
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
            traverse_bvh2_core(scene.bvh2Nodes, scene.primIds, scene.meshes,
                               paths[i].ray, Float32(1.0e38), intersections + i)
            if scene.sphereCount > 0:
                test_spheres(scene.spheres, Int(scene.sphereCount), paths[i].ray, intersections + i)
        for i in range(n):
            if paths[i].active == 0:
                continue
            # ── Volume transmittance sampling ──────────────────────────
            var med_idx = Int(paths[i].current_medium_idx)
            if med_idx >= 0 and Int(scene.mediumCount) > 0 and intersections[i].hit != Int8(0):
                var med = scene.mediums[med_idx]
                var sigma_t_r = med.sigma_a.r + med.sigma_s.r
                var sigma_t_g = med.sigma_a.g + med.sigma_s.g
                var sigma_t_b = med.sigma_a.b + med.sigma_s.b
                var sigma_maj = sigma_t_r
                if sigma_maj > Float32(0.0):
                    var pcg_vol = PCG32(paths[i].pcgState, paths[i].pcgInc)
                    var u_free  = pcg_vol.next_float()
                    paths[i].pcgState = pcg_vol.state
                    var t_free = -log(max(u_free, Float32(1e-7))) / sigma_maj
                    var t_surf = intersections[i].tHit
                    var t_seg  = min(t_free, t_surf)
                    paths[i].throughput.r *= exp(-sigma_t_r * t_seg)
                    paths[i].throughput.g *= exp(-sigma_t_g * t_seg)
                    paths[i].throughput.b *= exp(-sigma_t_b * t_seg)
                    if t_free < t_surf:
                        var p_absorb  = (med.sigma_a.r) / sigma_maj
                        var p_scatter = (med.sigma_s.r) / sigma_maj
                        var pcg2 = PCG32(paths[i].pcgState, paths[i].pcgInc)
                        var u_mode = pcg2.next_float()
                        paths[i].pcgState = pcg2.state
                        if u_mode < p_absorb:
                            paths[i].throughput = RGB(Float32(0), Float32(0), Float32(0))
                            paths[i].active = Int8(0)
                        elif u_mode < p_absorb + p_scatter:
                            var pcg3 = PCG32(paths[i].pcgState, paths[i].pcgInc)
                            var u1 = pcg3.next_float()
                            var u2 = pcg3.next_float()
                            paths[i].pcgState = pcg3.state
                            var cos_theta = Float32(2) * u1 - Float32(1)
                            var sin_theta = sqrt(max(Float32(0), Float32(1) - cos_theta * cos_theta))
                            var phi = Float32(6.28318530718) * u2
                            var ox = paths[i].ray.origin.x + t_free * paths[i].ray.direction.x
                            var oy = paths[i].ray.origin.y + t_free * paths[i].ray.direction.y
                            var oz = paths[i].ray.origin.z + t_free * paths[i].ray.direction.z
                            paths[i].ray = Ray_C(Point3f(ox, oy, oz), Vec3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta))
                            paths[i].specularBounce = Int8(0)
                            intersections[i].hit = Int8(0)
        for i in range(n):
            if paths[i].active == 0:
                continue
            shade_core_cpu_nee(paths, intersections, scene.bvh2Nodes, scene.primIds,
                               scene.meshes, scene.materials,
                               scene.areaLights, Int(scene.areaLightCount),
                               scene.textures, i,
                               scene.distantLights, Int(scene.distantLightCount),
                               scene.pointLights, Int(scene.pointLightCount),
                               scene.infiniteLights, Int(scene.infiniteLightCount),
                               scene.spheres, Int(scene.sphereCount))
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
            var mi_e1x = mi_m.points[mi_vi1*4]   - mi_m.points[mi_vi0*4]
            var mi_e1y = mi_m.points[mi_vi1*4+1] - mi_m.points[mi_vi0*4+1]
            var mi_e1z = mi_m.points[mi_vi1*4+2] - mi_m.points[mi_vi0*4+2]
            var mi_e2x = mi_m.points[mi_vi2*4]   - mi_m.points[mi_vi0*4]
            var mi_e2y = mi_m.points[mi_vi2*4+1] - mi_m.points[mi_vi0*4+1]
            var mi_e2z = mi_m.points[mi_vi2*4+2] - mi_m.points[mi_vi0*4+2]
            var mi_nx = mi_e1y * mi_e2z - mi_e1z * mi_e2y
            var mi_ny = mi_e1z * mi_e2x - mi_e1x * mi_e2z
            var mi_nz = mi_e1x * mi_e2y - mi_e1y * mi_e2x
            var mi_dot = paths[i].ray.direction.x * mi_nx + paths[i].ray.direction.y * mi_ny + paths[i].ray.direction.z * mi_nz
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
            var sumLR = Float32(0.0); var sumLG = Float32(0.0); var sumLB = Float32(0.0)
            var sumAR = Float32(0.0); var sumAG = Float32(0.0); var sumAB = Float32(0.0)
            var sumW = Float32(0.0)
            for _ in range(spp):
                sumLR += paths[idx].estimate.r
                sumLG += paths[idx].estimate.g
                sumLB += paths[idx].estimate.b
                sumAR += paths[idx].albedo.r
                sumAG += paths[idx].albedo.g
                sumAB += paths[idx].albedo.b
                sumW += sp.filterWeight
                idx += 1
            resultsPtr[out] = TileResult_C(
                RGB(sumLR, sumLG, sumLB), RGB(sumAR, sumAG, sumAB), sumW, px, py,
            )
            out += 1

    intersections.free()
    paths.free()



fn _fmt_f1(v: Float64) -> String:
    var i = Int(v)
    var frac = Int((v - Float64(i)) * 10.0 + 0.5)
    if frac >= 10:
        i += 1; frac = 0
    return String(i) + "." + String(frac)

fn _fmt_time(s: Float64) -> String:
    var sec = Int(s)
    var min = sec // 60
    var rem = sec % 60
    if min > 0:
        var rs = String(rem)
        if rem < 10: rs = "0" + rs
        return String(min) + "m " + rs + "s"
    return _fmt_f1(s) + "s"

fn _progress_str(done: Int, total: Int, elapsed: Float64, unit: String) -> String:
    var pct = _fmt_f1(Float64(done) * 100.0 / Float64(total))
    var est = Float64(0.0)
    if done > 0:
        est = elapsed * Float64(total) / Float64(done)
    return ("Rendering: " + String(done) + " / " + String(total)
        + " " + unit + " (" + pct + "%) | Elapsed: " + _fmt_time(elapsed)
        + " | Total Est.: " + _fmt_time(est) + "                ")


@export
def mojo_render_all_tiles(
    raster_to_camera: UnsafePointer[Float32, MutAnyOrigin],
    camera_to_world: UnsafePointer[Float32, MutAnyOrigin],
    min_x: Int32, min_y: Int32, max_x: Int32, max_y: Int32,
    tile_w: Int32, tile_h: Int32,
    sampler_params: UnsafePointer[TileSamplerParams_C, MutAnyOrigin],
    scene: UnsafePointer[SceneDescriptor2_C, MutAnyOrigin],
    results: UnsafePointer[TileResult_C, MutAnyOrigin],
    max_depth: Int32,
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
    fn render_one(tile_idx: Int):
        var ty_i = tile_idx // n_tiles_x
        var tx_i = tile_idx % n_tiles_x
        var tx = Int(min_x) + tx_i * tw
        var ty = Int(min_y) + ty_i * th
        var tx_max = Int32(min(tx + tw, Int(max_x)))
        var ty_max = Int32(min(ty + th, Int(max_y)))
        var tw_actual = Int(tx_max) - tx
        var th_actual = Int(ty_max) - ty
        var tile_buf = tile_bufs + tile_idx * max_tile_pixels
        mojo_render_tile_v2(
            raster_to_camera, camera_to_world,
            Int32(tx), Int32(ty), tx_max, ty_max,
            sampler_params, scene, tile_buf, max_depth)
        for iy in range(th_actual):
            for ix in range(tw_actual):
                var src = iy * tw_actual + ix
                var dst = (ty + iy - Int(min_y)) * res_x + (tx + ix - Int(min_x))
                results[dst] = tile_buf[src]
        done_ptr[0] += Int32(1)
        var d = Int(done_ptr[0])
        if d % print_step == 0 or d == n_tiles:
            var elapsed = Float64(perf_counter_ns() - t0) / 1.0e9
            print(_progress_str(d, n_tiles, elapsed, "tiles"), end="\r")

    parallelize[render_one](n_tiles)
    var total_s = Float64(perf_counter_ns() - t0) / 1.0e9
    print("Rendering: " + String(n_tiles) + " / " + String(n_tiles)
        + " tiles (100.0%) | Done: " + _fmt_time(total_s) + "                ")
    done_ptr.free()
    tile_bufs.free()


# Normalize TileResult_C[] → per-pixel float RGB arrays.
@export
fn mojo_normalize_film(
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
        var br = r.estimate.r / w * scale
        var bg = r.estimate.g / w * scale
        var bb = r.estimate.b / w * scale
        if max_component_value > Float32(0):
            var mx = max(br, max(bg, bb))
            if mx > max_component_value:
                var s = max_component_value / mx
                br *= s
                bg *= s
                bb *= s
        beauty_out[i * 3 + 0] = br
        beauty_out[i * 3 + 1] = bg
        beauty_out[i * 3 + 2] = bb
        albedo_out[i * 3 + 0] = r.albedo.r / w
        albedo_out[i * 3 + 1] = r.albedo.g / w
        albedo_out[i * 3 + 2] = r.albedo.b / w

