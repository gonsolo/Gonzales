from std.math import ceildiv, sqrt, log, exp, cos, sin, max
from std.memory import alloc
from max.algorithm import parallelize
from std.time import perf_counter_ns
from .geometry import RGB, Point3f, Vec3f, point3f, vec3f, sphere_outward_normal, Ray_C, Intersection_C, PrimId_C, PathState_C, TileResult_C, Sphere_C, AreaLight_C, LightSampler_C, light_sampler_sample, dot, cross, Medium_C, MediumInterface_C, Grid_C, grid_sample_density, INV_FOUR_PI, curve_piece_endpoints, _curve_perp_axis
from .bvh import SceneDescriptor2_C, traverse_bvh2_core, test_spheres, any_hit_bvh2_core
from .shading import shade_core_cpu_nee, GIPendingX1, gi_pending_x1_init
from .rng import PCG32
from .sampling import TileSamplerParams_C, encode_morton2, sobol_get_sample_index, sobol_sample, gaussian_sample_1d, derive_pcg_seeds, gaussian_norm, mix_bits_u64, gen_primary_ray_state
from .guide import GuideGrid, guide_merge, null_guide
from .spectrum import SampledWavelengths
from .gpu import _sample_medium_core
from .restir_di import ReservoirIO, reservoir_io_null
from .restir_gi import GIReservoirIO, gi_reservoir_io_null
from .restir_sms import SMSReservoirIO, sms_reservoir_io_null


def render_tile[Osp: Origin[mut=True], Oc2w: Origin[mut=True]](
    rasterToCamera: UnsafePointer[Float32, MutExternalOrigin],
    cameraToWorld: UnsafePointer[Float32, Oc2w],
    tileMinX: Int32, tileMinY: Int32, tileMaxX: Int32, tileMaxY: Int32,
    samplerParamsPtr: UnsafePointer[TileSamplerParams_C, Osp],
    scenePtr: UnsafePointer[SceneDescriptor2_C, MutExternalOrigin],
    resultsPtr: UnsafePointer[TileResult_C, MutExternalOrigin],
    maxDepth: Int32,
    guide_read: GuideGrid,
    guide_write: GuideGrid,
    use_restir: Bool = False,
    frame_w: Int32 = Int32(0),
    restir_io: ReservoirIO = reservoir_io_null(),
    # Phase 4 (docs/A2_restir_migration_plan.md), INDEPENDENT of use_restir
    # on purpose: `use_restir` alone must keep meaning exactly what it
    # already means to every existing --restir (DI-only) render. Below,
    # the per-path-slot GIPendingX1 scratch buffer is only allocated (a
    # real, non-dangling pointer) when `use_gi=True` -- it used to be
    # allocated unconditionally, which silently activated
    # _shade_diffuse_nee's GI generate/combine/resolve block (gated only
    # on that pointer being real, not on any flag) for every ordinary
    # --restir CPU render. Caught by A/B rendering cornell-box (mean error
    # 0.0054, far above the ~0.0005 noise floor) before it shipped any
    # further -- see project_restir_migration memory. `gi_io` is the
    # frame-wide, caller-owned read/write reservoir buffers + shared
    # G-buffer (same role as restir_io above); harmless if non-null with
    # use_gi=False, since the dangling gi_pending buffer already gates the
    # whole GI block off regardless.
    use_gi: Bool = False,
    gi_io: GIReservoirIO = gi_reservoir_io_null(),
    # Phase 6 (docs/A2_restir_migration_plan.md): ReSTIR SMS temporal reuse
    # for glass-caustic MNEE probing, INDEPENDENT of use_restir (unlike
    # use_gi, which requires it) -- SMS-ReSTIR only ever touches
    # _nee_area_lights' own glass-probing branch, which runs whenever
    # ReSTIR DI ISN'T handling bounce 0 (see _shade_diffuse_nee's own
    # docstring for the known use_restir-combination gap). sms_io is the
    # frame-wide, caller-owned read/write SMSReservoir buffer (no G-buffer
    # fields yet -- temporal-only, no spatial reuse).
    use_sms_restir: Bool = False,
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
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
    # Global (frame-wide) pixel index per path -- needed only for ReSTIR DI's
    # temporal reservoir buffer (Phase 2.3, docs/A2_restir_migration_plan.md),
    # which persists across frames and so must be indexed by a stable,
    # tile-independent key, not the tile-local `idx` below. -1 (never a
    # valid buffer index) whenever frame_w wasn't supplied -- matches
    # di_temporal_step's own "no temporal reuse" fallback.
    var pixel_idx_buf = alloc[Int](n)

    # Phase 4's per-path-slot scratch (GIPendingX1), tile-call-local like
    # `paths`/`intersections` above -- NOT frame-wide like gi_io.
    # alloc() doesn't zero memory, so every slot needs an explicit inactive
    # init; otherwise a path that never reaches bounce 1 (miss, RR kill, or
    # a non-diffuse x2) would leave garbage that a later stray read could
    # misinterpret as a real pending snapshot.
    var gi_pending_buf = UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling()
    if use_gi:
        gi_pending_buf = alloc[GIPendingX1](n)
        for gi_i in range(n):
            gi_pending_buf[gi_i] = gi_pending_x1_init()

    # Generate primary rays from Sobol film samples
    var idx = 0
    var rng_seed = sp.rngSeed
    for iy in range(tileH):
        for ix in range(tileW):
            var px = Int32(tileMinX) + Int32(ix)
            var py = Int32(tileMinY) + Int32(iy)
            var this_pixel_idx = Int(py * frame_w + px) if frame_w > Int32(0) else -1
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
                pixel_idx_buf[idx] = this_pixel_idx
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
            # Shared with the GPU wavefront path (gpu.mojo's
            # sample_medium_gpu kernel) via _sample_medium_core — staged
            # spectral rendering rollout, Stage 5a (unify before
            # spectralize), see project_spectral_rendering memory. See that
            # function's own docstring for the two real CPU/GPU behavioral
            # divergences this unification found and fixed (bounce/
            # lastBsdfPdf not set on CPU volume-scatter; reversed NEE-vs-
            # scatter-direction RNG draw order), not preserved as
            # intentional per-backend differences.
            _sample_medium_core(
                paths, intersections, i,
                scene.mediums, Int(scene.mediumCount), scene.grids,
                scene.bvh2Nodes, scene.primIds, scene.meshes, scene.curves,
                scene.blasNodesArr, scene.blasPrimIdsArr, scene.instances,
                scene.areaLights, Int(scene.areaLightCount),
                scene.lightSampler.cdf, Int(scene.lightSampler.n),
                scene.spheres, Int(scene.sphereCount),
                scene.spectral.coeffs, scene.spectral.res,
                scene.spectral.cie_x, scene.spectral.cie_y, scene.spectral.cie_z, scene.spectral.d65,
            )
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
                               instances=scene.instances, guide_write=guide_write, spectral=scene.spectral,
                               measured_brdfs=scene.measuredBrdfs, use_restir=use_restir,
                               restir_io=restir_io, pixel_idx=pixel_idx_buf[i],
                               gi_pending=gi_pending_buf, gi_io=gi_io, sms_io=sms_io,
                               nmaps=scene.normalSlopeMaps)
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
    pixel_idx_buf.free()
    if use_gi:
        gi_pending_buf.free()



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


def render_all_tiles[Osp: Origin[mut=True], Oc2w: Origin[mut=True], Ores: Origin[mut=True]](
    raster_to_camera: UnsafePointer[Float32, MutExternalOrigin],
    camera_to_world: UnsafePointer[Float32, Oc2w],
    min_x: Int32, min_y: Int32, max_x: Int32, max_y: Int32,
    tile_w: Int32, tile_h: Int32,
    sampler_params: UnsafePointer[TileSamplerParams_C, Osp],
    scene: UnsafePointer[SceneDescriptor2_C, MutExternalOrigin],
    results: UnsafePointer[TileResult_C, Ores],
    max_depth: Int32,
    quiet: Bool = False,
    guide_read: GuideGrid = null_guide(),
    write_guides: UnsafePointer[GuideGrid, MutExternalOrigin] = UnsafePointer[GuideGrid, MutExternalOrigin].unsafe_dangling(),
    n_write_guides: Int = 0,
    use_restir: Bool = False,
    frame_w: Int32 = Int32(0),
    restir_io: ReservoirIO = reservoir_io_null(),
    use_gi: Bool = False,
    gi_io: GIReservoirIO = gi_reservoir_io_null(),
    use_sms_restir: Bool = False,
    sms_io: SMSReservoirIO = sms_reservoir_io_null(),
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
            sampler_params, scene, tile_buf, max_depth, guide_read, gw, use_restir,
            frame_w, restir_io, use_gi, gi_io, use_sms_restir, sms_io)
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
# Normalize TileResult_C[] → per-pixel float RGB arrays.
def normalize_film[Ores: Origin[mut=True], Obo: Origin[mut=True], Oao: Origin[mut=True]](
    results: UnsafePointer[TileResult_C, Ores],
    count: Int32,
    iso: Float32,
    max_component_value: Float32,
    beauty_out: UnsafePointer[Float32, Obo],
    albedo_out: UnsafePointer[Float32, Oao],
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

