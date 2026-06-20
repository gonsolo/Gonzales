from std.memory import alloc, OwnedPointer
from std.collections import List
from std.math import sqrt, tan
from .pbrt_parser import ParsedScene_Mojo, mojo_parse_scene, mojo_parsed_free, mojo_parsed_scene_descriptor, resize_film, mojo_apply_overrides
from .rendering import render_all_tiles, render_aux_buffers, normalize_film, fmt_time, progress_str
from std.time import perf_counter_ns
from .geometry import RGB, Point3f, Vec3f, TileResult_C, PathState_C, Ray_C, dot
from .postprocess import denoise, write_image
from .sampling import TileSamplerParams_C, mix_bits_u64, encode_morton2, sobol_get_sample_index, sobol_sample, gaussian_sample_1d, derive_pcg_seeds
from .bvh import BVH2Node, SceneDescriptor2_C
from .gpu import GpuSceneHandle, WAVEFRONT_BATCH, gpu_available, gpu_upload_scene, gpu_render_sample, gpu_render_wavefront, gpu_download_film, gpu_download_albedo, gpu_clear_film, gpu_atrous_denoise, gpu_gen_aux_buffers, gpu_free_scene
from .viewer import CameraState, ViewerHandle, viewer_create, viewer_update_framebuffer, viewer_should_close, viewer_poll_events, viewer_get_camera_state, viewer_set_camera_state, viewer_destroy, build_camera_to_world

# Generate Sobol matrices from the Joe-Kuo data file.
# Returns a heap-allocated pointer to 21201*52 UInt32 values, or null on error.
def _generate_sobol_matrices(path: String) -> Optional[UnsafePointer[UInt32, MutAnyOrigin]]:
    var file_buf: UnsafePointer[UInt8, MutAnyOrigin]
    var file_size: Int
    try:
        var f = open(path, "r")
        var bytes = f.read_bytes()
        f.close()
        file_size = len(bytes)
        file_buf = alloc[UInt8](file_size + 1)
        var bytes_ptr = bytes.unsafe_ptr()
        for i in range(file_size):
            file_buf[i] = bytes_ptr[i]
        file_buf[file_size] = UInt8(0)
    except:
        print("Error: cannot open Sobol data file: " + path)
        return None

    # Allocate matrices: 21201 dimensions × 52 bits
    comptime N_DIMS = 21201
    comptime N_BITS = 52
    var matrices = alloc[UInt32](N_DIMS * N_BITS)
    # Zero-initialize
    for i in range(N_DIMS * N_BITS):
        matrices[i] = UInt32(0)

    # Dimension 0: all ones (standard Sobol)
    for j in range(N_BITS):
        matrices[j] = UInt32(1) << UInt32(31 - j)

    # Parse remaining dimensions from file
    var pos = 0
    var flen = Int(file_size)
    var dim = 1

    # Helper: skip whitespace (spaces and tabs, but not newlines)
    # Read one line at a time and parse
    while pos < flen and dim < N_DIMS:
        # Skip leading whitespace including newlines
        while pos < flen and (file_buf[pos] == UInt8(32) or file_buf[pos] == UInt8(9) or file_buf[pos] == UInt8(10) or file_buf[pos] == UInt8(13)):
            pos += 1
        if pos >= flen:
            break
        # Skip comment lines starting with '#'
        if file_buf[pos] == UInt8(35):  # '#'
            while pos < flen and file_buf[pos] != UInt8(10):
                pos += 1
            continue

        # Parse: s a m1 m2 ... ms
        # s = number of direction numbers
        var s = Int32(0)
        while pos < flen and file_buf[pos] >= UInt8(48) and file_buf[pos] <= UInt8(57):
            s = s * Int32(10) + Int32(file_buf[pos]) - Int32(48)
            pos += 1
        # skip whitespace
        while pos < flen and (file_buf[pos] == UInt8(32) or file_buf[pos] == UInt8(9)):
            pos += 1

        # a = polynomial
        var a = UInt32(0)
        while pos < flen and file_buf[pos] >= UInt8(48) and file_buf[pos] <= UInt8(57):
            a = a * UInt32(10) + UInt32(file_buf[pos]) - UInt32(48)
            pos += 1
        # skip whitespace
        while pos < flen and (file_buf[pos] == UInt8(32) or file_buf[pos] == UInt8(9)):
            pos += 1

        # m values
        var m = InlineArray[UInt32, 52](fill=UInt32(0))
        var num_m = Int(s)
        if num_m > N_BITS:
            num_m = N_BITS
        for i in range(num_m):
            var v = UInt32(0)
            while pos < flen and file_buf[pos] >= UInt8(48) and file_buf[pos] <= UInt8(57):
                v = v * UInt32(10) + UInt32(file_buf[pos]) - UInt32(48)
                pos += 1
            m[i] = v
            while pos < flen and (file_buf[pos] == UInt8(32) or file_buf[pos] == UInt8(9)):
                pos += 1

        # Skip to end of line
        while pos < flen and file_buf[pos] != UInt8(10):
            pos += 1

        # Compute direction numbers v[i] = m[i] << (32 - i - 1)
        var base = dim * N_BITS
        for i in range(Int(s)):
            if i >= N_BITS:
                break
            matrices[base + i] = m[i] << UInt32(31 - i)

        # Recurrence for i >= s
        for i in range(Int(s), N_BITS):
            var v_prev = matrices[base + i - Int(s)]
            var vi = v_prev ^ (v_prev >> UInt32(s))
            var j = 1
            var poly = a
            while j <= Int(s) - 1:
                if (poly & UInt32(1)) != UInt32(0):
                    vi ^= matrices[base + i - j]
                poly >>= 1
                j += 1
            matrices[base + i] = vi

        dim += 1

    file_buf.free()

    if dim < 2:
        print("Warning: Sobol file had fewer dimensions than expected")
    return Optional(matrices)


def _gpu_upload_scene(
    psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
    sobol: UnsafePointer[UInt32, MutAnyOrigin],
    n_pixels: Int,
) -> UnsafePointer[GpuSceneHandle, MutAnyOrigin]:
    var fw = psc[0].film_w
    var fh = psc[0].film_h
    var n_meshes = Int(psc[0].mesh_count)
    var pts_counts = List[Int64](capacity=max(n_meshes, 1))
    var fi_counts  = List[Int64](capacity=max(n_meshes, 1))
    var vi_counts  = List[Int64](capacity=max(n_meshes, 1))
    var uv_counts  = List[Int64](capacity=max(n_meshes, 1))
    var nrm_counts = List[Int64](capacity=max(n_meshes, 1))
    for _ in range(max(n_meshes, 1)):
        pts_counts.append(Int64(0)); fi_counts.append(Int64(0))
        vi_counts.append(Int64(0)); uv_counts.append(Int64(0))
        nrm_counts.append(Int64(0))
    for i in range(n_meshes):
        pts_counts[i] = Int64(psc[0].mesh_n_verts[i]) * 4
        fi_counts[i]  = Int64(psc[0].mesh_n_tris[i])
        vi_counts[i]  = Int64(psc[0].mesh_n_tris[i]) * 3
        uv_counts[i]  = Int64(psc[0].mesh_uv_n_verts[i])
        nrm_counts[i] = Int64(psc[0].mesh_nrm_n_verts[i])
    var handle = gpu_upload_scene(
        psc[0].bvh_nodes,      Int64(psc[0].bvh_node_count),
        psc[0].prim_ids,       Int64(psc[0].prim_count),
        psc[0].meshes,         Int64(n_meshes),
        pts_counts.unsafe_ptr(), fi_counts.unsafe_ptr(),
        vi_counts.unsafe_ptr(), uv_counts.unsafe_ptr(),
        nrm_counts.unsafe_ptr(),
        psc[0].tex_filenames,  psc[0].tex_count,
        psc[0].materials,      Int64(psc[0].material_count),
        psc[0].area_lights,    Int64(psc[0].area_light_count),
        psc[0].spheres,        Int64(psc[0].sphere_count),
        psc[0].distant_lights, Int64(psc[0].distant_count),
        psc[0].point_lights,   Int64(psc[0].point_count),
        psc[0].infinite_lights, Int64(psc[0].infinite_count),
        Int64(n_pixels),
        sobol,
        psc[0].raster_to_camera, psc[0].camera_to_world,
        psc[0].filter_sigma, psc[0].filter_support_x, psc[0].filter_support_y,
        psc[0].filter_norm_x, psc[0].filter_norm_y,
        psc[0].filter_type,
        fw, fh,
    )
    # pts_counts, fi_counts, vi_counts, uv_counts freed automatically
    return handle


def _dbg_vlen(x: Float32, y: Float32, z: Float32) -> Float32:
    return sqrt(x*x + y*y + z*z)

def debug_trace_pixel(
    path: UnsafePointer[UInt8, MutAnyOrigin],
    px: Int32, py: Int32,
):
    """Trace the centre ray of one pixel and print the path bounce-by-bounce
    (hit mesh/material/normal/t, dielectric entering/eta/Fresnel decision,
    envmap lookup). For comparing against `pbrt --pixelmaterial`."""
    from .bvh import traverse_bvh2_core, test_spheres
    from .geometry import Intersection_C, Material_C, cross, fr_dielectric
    from .shading import _equal_area_sphere_to_square

    var psc = mojo_parse_scene(path)
    if Int(psc) == 0:
        print("parse failed"); return

    # Centre ray (no jitter): raster_to_camera then camera_to_world rotation.
    var r2c = psc[0].raster_to_camera
    var c2w = psc[0].camera_to_world
    var fX = Float32(px) + Float32(0.5)
    var fY = Float32(py) + Float32(0.5)
    var cx = r2c[0]*fX + r2c[4]*fY + r2c[12]
    var cy = r2c[1]*fX + r2c[5]*fY + r2c[13]
    var cz = r2c[2]*fX + r2c[6]*fY + r2c[14]
    var cw = r2c[3]*fX + r2c[7]*fY + r2c[15]
    if cw != Float32(0.0) and cw != Float32(1.0):
        cx /= cw; cy /= cw; cz /= cw
    var cl = _dbg_vlen(cx, cy, cz)
    if cl > Float32(0.0): cx /= cl; cy /= cl; cz /= cl
    var dx = c2w[0]*cx + c2w[4]*cy + c2w[8]*cz
    var dy = c2w[1]*cx + c2w[5]*cy + c2w[9]*cz
    var dz = c2w[2]*cx + c2w[6]*cy + c2w[10]*cz
    var dl = _dbg_vlen(dx, dy, dz)
    if dl > Float32(0.0): dx /= dl; dy /= dl; dz /= dl
    var ox = c2w[12]; var oy = c2w[13]; var oz = c2w[14]
    print("PIXEL", px, py, "ray.o", ox, oy, oz, "ray.d", dx, dy, dz)

    var inter = alloc[Intersection_C](1)
    for bounce in range(8):
        var ray = Ray_C(Point3f(ox, oy, oz), Vec3f(dx, dy, dz))
        inter[0].hit = Int8(0)
        traverse_bvh2_core(psc[0].bvh_nodes, psc[0].prim_ids, psc[0].meshes, ray, Float32(1.0e38), inter)
        if psc[0].sphere_count > 0:
            test_spheres(psc[0].spheres, Int(psc[0].sphere_count), ray, inter)
        if inter[0].hit == Int8(0):
            # envmap miss
            if psc[0].infinite_count > 0:
                var il = psc[0].infinite_lights[0]
                var w2l = il.world_to_light
                var ldx = w2l[0]*dx + w2l[4]*dy + w2l[8]*dz
                var ldy = w2l[1]*dx + w2l[5]*dy + w2l[9]*dz
                var ldz = w2l[2]*dx + w2l[6]*dy + w2l[10]*dz
                var uv = _equal_area_sphere_to_square(ldx, ldy, ldz)
                var rgb_str = String("(no pixels)")
                if Int(il.pixels_ptr) > 1 and il.cdf_w > Int32(0):
                    var iw = Int(il.cdf_w); var ih = Int(il.cdf_h)
                    var pxe = Int(max(Float32(0), min(Float32(iw-1), uv[0]*Float32(iw))))
                    var pye = Int(max(Float32(0), min(Float32(ih-1), uv[1]*Float32(ih))))
                    var rr = il.pixels_ptr[(pye*iw+pxe)*3+0]
                    var gg = il.pixels_ptr[(pye*iw+pxe)*3+1]
                    var bb = il.pixels_ptr[(pye*iw+pxe)*3+2]
                    rgb_str = String(rr) + " " + String(gg) + " " + String(bb)
                print("  bounce", bounce, "MISS -> envmap localdir", ldx, ldy, ldz, "uv", uv[0], uv[1], "rgb", rgb_str)
            else:
                print("  bounce", bounce, "MISS (no envmap)")
            break

        # Identify mesh + material
        var mesh_idx: Int; var base_vidx: Int
        if inter[0].primId.type == 0:
            mesh_idx = Int(inter[0].primId.id1); base_vidx = Int(inter[0].primId.id2)
        else:
            mesh_idx = Int(inter[0].primId.id2 >> 32); base_vidx = Int(inter[0].primId.id2 & 0xFFFFFFFF) * 3
        var mat = psc[0].materials[Int(inter[0].primId.materialIndex)]
        var mesh = psc[0].meshes[mesh_idx]
        var v0 = Int(mesh.vertexIndices[base_vidx]); var v1 = Int(mesh.vertexIndices[base_vidx+1]); var v2 = Int(mesh.vertexIndices[base_vidx+2])
        var p0x = mesh.points[v0*4]; var p0y = mesh.points[v0*4+1]; var p0z = mesh.points[v0*4+2]
        var p1x = mesh.points[v1*4]; var p1y = mesh.points[v1*4+1]; var p1z = mesh.points[v1*4+2]
        var p2x = mesh.points[v2*4]; var p2y = mesh.points[v2*4+1]; var p2z = mesh.points[v2*4+2]
        var gnx = (p1y-p0y)*(p2z-p0z) - (p1z-p0z)*(p2y-p0y)
        var gny = (p1z-p0z)*(p2x-p0x) - (p1x-p0x)*(p2z-p0z)
        var gnz = (p1x-p0x)*(p2y-p0y) - (p1y-p0y)*(p2x-p0x)
        var gnl = _dbg_vlen(gnx, gny, gnz)
        if gnl > Float32(0.0): gnx /= gnl; gny /= gnl; gnz /= gnl
        var hx = ox + dx*inter[0].tHit; var hy = oy + dy*inter[0].tHit; var hz = oz + dz*inter[0].tHit
        print("  bounce", bounce, "HIT mesh", mesh_idx, "matType", Int(mat.type), "t", inter[0].tHit, "p", hx, hy, hz, "gN", gnx, gny, gnz)

        if Int(mat.type) == 4:
            # Dielectric — mirror shade_dielectric's decision (no RNG: report Fresnel, follow transmit)
            var ior = mat.albedo.r
            var facing = (dx*gnx + dy*gny + dz*gnz) < Float32(0.0)
            var entering = facing
            if bounce == 0: entering = True
            var nx = gnx if facing else -gnx
            var ny = gny if facing else -gny
            var nz = gnz if facing else -gnz
            var eta = (Float32(1.0)/ior) if entering else ior
            var cos_i = -(dx*nx + dy*ny + dz*nz)
            var sin2t = eta*eta*(Float32(1.0) - cos_i*cos_i)
            var tir = sin2t > Float32(1.0)
            var fres = fr_dielectric(cos_i, Float32(1.0)/eta)
            print("        DIELECTRIC entering", Int(entering), "eta", eta, "cos_i", cos_i, "fresnel", fres, "tir", Int(tir))
            # Probe the REFLECTED ray's envmap value (the bright contribution).
            var rcos = dx*nx + dy*ny + dz*nz
            var rfx = dx - nx*Float32(2.0)*rcos
            var rfy = dy - ny*Float32(2.0)*rcos
            var rfz = dz - nz*Float32(2.0)*rcos
            var rfl = _dbg_vlen(rfx, rfy, rfz)
            if rfl > Float32(0.0): rfx /= rfl; rfy /= rfl; rfz /= rfl
            var rray = Ray_C(Point3f(hx+nx*Float32(0.001), hy+ny*Float32(0.001), hz+nz*Float32(0.001)), Vec3f(rfx, rfy, rfz))
            var rint = alloc[Intersection_C](1); rint[0].hit = Int8(0)
            traverse_bvh2_core(psc[0].bvh_nodes, psc[0].prim_ids, psc[0].meshes, rray, Float32(1.0e38), rint)
            if rint[0].hit == Int8(0) and psc[0].infinite_count > 0:
                var il2 = psc[0].infinite_lights[0]
                var w2 = il2.world_to_light
                var l2x = w2[0]*rfx + w2[4]*rfy + w2[8]*rfz
                var l2y = w2[1]*rfx + w2[5]*rfy + w2[9]*rfz
                var l2z = w2[2]*rfx + w2[6]*rfy + w2[10]*rfz
                var uv2 = _equal_area_sphere_to_square(l2x, l2y, l2z)
                var rs = String("")
                if Int(il2.pixels_ptr) > 1 and il2.cdf_w > Int32(0):
                    var iw2 = Int(il2.cdf_w); var ih2 = Int(il2.cdf_h)
                    var ax = Int(max(Float32(0), min(Float32(iw2-1), uv2[0]*Float32(iw2))))
                    var ay = Int(max(Float32(0), min(Float32(ih2-1), uv2[1]*Float32(ih2))))
                    rs = String(il2.pixels_ptr[(ay*iw2+ax)*3+0]) + " " + String(il2.pixels_ptr[(ay*iw2+ax)*3+1]) + " " + String(il2.pixels_ptr[(ay*iw2+ax)*3+2])
                print("        REFLECT dir", rfx, rfy, rfz, "-> envmap uv", uv2[0], uv2[1], "rgb", rs)
            else:
                print("        REFLECT dir", rfx, rfy, rfz, "-> hits mesh (occluded), matType", Int(psc[0].materials[Int(rint[0].primId.materialIndex)].type) if rint[0].hit != Int8(0) else -1)
            rint.free()
            # Follow transmit branch (what pbrt did) if possible, else reflect
            if tir:
                var rl = dx*nx + dy*ny + dz*nz
                dx = dx - nx*Float32(2.0)*rl; dy = dy - ny*Float32(2.0)*rl; dz = dz - nz*Float32(2.0)*rl
                ox = hx + nx*Float32(0.0001); oy = hy + ny*Float32(0.0001); oz = hz + nz*Float32(0.0001)
                print("        -> REFLECT (TIR)")
            else:
                var cos_t = sqrt(Float32(1.0) - sin2t)
                dx = dx*eta + nx*(eta*cos_i - cos_t); dy = dy*eta + ny*(eta*cos_i - cos_t); dz = dz*eta + nz*(eta*cos_i - cos_t)
                var nl = _dbg_vlen(dx,dy,dz)
                if nl > Float32(0.0): dx /= nl; dy /= nl; dz /= nl
                ox = hx - nx*Float32(0.0001); oy = hy - ny*Float32(0.0001); oz = hz - nz*Float32(0.0001)
                print("        -> TRANSMIT dir", dx, dy, dz)
        else:
            print("        STOP (non-glass material)")
            break
    inter.free()
    mojo_parsed_free(psc)


def parse_and_render(
    path: UnsafePointer[UInt8, MutAnyOrigin],
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    use_gpu: Bool,
    override_w: Int32 = Int32(0), override_h: Int32 = Int32(0),
    no_denoise: Bool = False,
    spp_override: Int32 = Int32(0),
    verbose: Bool = False,
) -> Int32:
    if use_gpu and not gpu_available():
        print("No GPU available — compile with --target-accelerator sm_86 or similar")
        return Int32(-1)

    var psc = mojo_parse_scene(path, verbose)
    if Int(psc) == 0:
        return Int32(-1)
    if override_w > 0 and override_h > 0:
        resize_film(psc, override_w, override_h)

    mojo_apply_overrides(psc, spp_override, Int32(0), Int32(0))

    var fw = psc[0].film_w
    var fh = psc[0].film_h
    var n_pixels = Int(fw) * Int(fh)
    var results = List[TileResult_C](capacity=n_pixels)

    if use_gpu:
        var spp = Int(psc[0].samples_per_pixel)
        # World units spanned by one pixel per unit distance (for mip LOD):
        # 2*tan(fov/2)/height. fov is in degrees along the shorter axis.
        var px_scale = Float32(2.0) * tan(psc[0].camera_fov * Float32(3.14159265 / 360.0)) / Float32(Int(fh))
        var handle = _gpu_upload_scene(psc, sobol_matrices, n_pixels)
        if Int(handle) <= 8:
            mojo_parsed_free(psc)
            return Int32(-1)
        var hash_bits = UInt64(mix_bits_u64(UInt64(0)))
        var seed_dim0 = UInt32(hash_bits & UInt64(0xFFFFFFFF))
        var seed_dim1 = UInt32(0)
        gpu_clear_film(handle, Int64(n_pixels))
        var t0_gpu = perf_counter_ns()
        var si = 0
        while si < spp:
            var actual_batch = min(WAVEFRONT_BATCH, spp - si)
            gpu_render_wavefront(
                handle,
                psc[0].camera_to_world,
                Int32(si), Int32(actual_batch),
                psc[0].log2_spp, psc[0].n_base4_digits,
                seed_dim0, seed_dim1,
                UInt32(psc[0].rng_seed & UInt64(0xFFFFFFFF)),
                UInt32(psc[0].rng_seed >> UInt64(32)),
                Int64(n_pixels), psc[0].max_depth,
                px_scale,
            )
            si += actual_batch
            var elapsed = Float64(perf_counter_ns() - t0_gpu) / 1.0e9
            print(progress_str(si, spp, elapsed, "spp"), end="\r")
        var gpu_total_s = Float64(perf_counter_ns() - t0_gpu) / 1.0e9
        print("Rendering: " + String(spp) + " / " + String(spp)
            + " spp (100.0%) | Done: " + fmt_time(gpu_total_s) + "                ")
        var denoised_gpu = List[Float32](capacity=n_pixels * 3)
        var albedo_gpu   = List[Float32](capacity=n_pixels * 3)
        for _ in range(n_pixels * 3): denoised_gpu.append(Float32(0)); albedo_gpu.append(Float32(0))
        gpu_gen_aux_buffers(handle, psc[0].camera_to_world, Int64(n_pixels))
        gpu_atrous_denoise(handle, denoised_gpu.unsafe_ptr(), Int64(n_pixels),
                                Int32(spp), psc[0].film_iso, psc[0].film_max_comp,
                                apply_denoise=not no_denoise)
        gpu_download_albedo(handle, albedo_gpu.unsafe_ptr(), Int64(n_pixels))
        var inv_spp = Float32(1.0) / Float32(spp)
        for i in range(n_pixels * 3):
            albedo_gpu[i] *= inv_spp
        gpu_free_scene(handle)
        _ = write_image(denoised_gpu.unsafe_ptr(), fw, fh, psc[0].film_filename, Int32(32), Int32(32))
        var albedo_name_buf = List[UInt8](capacity=11)
        var albedo_name_str = "albedo.exr"
        var anp = albedo_name_str.unsafe_ptr()
        for i in range(10): albedo_name_buf.append(anp[i])
        albedo_name_buf.append(UInt8(0))
        _ = write_image(albedo_gpu.unsafe_ptr(), fw, fh, albedo_name_buf.unsafe_ptr(), Int32(32), Int32(32))
        # albedo_name_buf freed automatically
        # denoised_gpu, albedo_gpu, and results freed automatically
        mojo_parsed_free(psc)
        return Int32(0)
    elif psc[0].prim_count == 0:
        print("Warning: scene has no geometry, skipping render")
        mojo_parsed_free(psc)
        return Int32(0)
    else:
        var zero = TileResult_C(
            estimate=RGB(Float32(0), Float32(0), Float32(0)),
            albedo=RGB(Float32(0), Float32(0), Float32(0)),
            filterWeight=Float32(0), pixelX=Int32(0), pixelY=Int32(0))
        for _ in range(n_pixels):
            results.append(zero)
        var sd = mojo_parsed_scene_descriptor(psc)
        var sp = TileSamplerParams_C(
            sobolMatrices=sobol_matrices,
            rngSeed=psc[0].rng_seed,
            sobolSeed=Int32(0),
            log2SamplesPerPixel=psc[0].log2_spp,
            nBase4Digits=psc[0].n_base4_digits,
            samplesPerPixel=psc[0].samples_per_pixel,
            filterSigma=psc[0].filter_sigma,
            filterSupportX=psc[0].filter_support_x,
            filterSupportY=psc[0].filter_support_y,
            filterNormX=psc[0].filter_norm_x,
            filterNormY=psc[0].filter_norm_y,
            filterWeight=psc[0].filter_weight,
            filterType=psc[0].filter_type,
        )
        var sp_ptr = OwnedPointer[TileSamplerParams_C](sp)
        render_all_tiles(
            psc[0].raster_to_camera, psc[0].camera_to_world,
            Int32(0), Int32(0), fw, fh,
            Int32(32), Int32(32),
            sp_ptr.unsafe_ptr(), sd, results.unsafe_ptr(), psc[0].max_depth)
        # sp_ptr freed automatically
        # Unjittered normals and depth for edge-preserving denoising.
        var normals  = List[Float32](capacity=n_pixels * 3)
        var dept     = List[Float32](capacity=n_pixels)
        for _ in range(n_pixels * 3): normals.append(Float32(0))
        for _ in range(n_pixels):     dept.append(Float32(0))
        render_aux_buffers(
            psc[0].raster_to_camera, psc[0].camera_to_world,
            Int32(0), Int32(0), fw, fh, sd,
            normals.unsafe_ptr(), dept.unsafe_ptr())
        sd.free()

        # Normalize → beauty/albedo → denoise with normals+depth → write
        var beauty   = List[Float32](capacity=n_pixels * 3)
        var albedo   = List[Float32](capacity=n_pixels * 3)
        var denoised = List[Float32](capacity=n_pixels * 3)
        for _ in range(n_pixels * 3): beauty.append(Float32(0)); albedo.append(Float32(0)); denoised.append(Float32(0))
        normalize_film(results.unsafe_ptr(), Int32(n_pixels),
                            psc[0].film_iso, psc[0].film_max_comp,
                            beauty.unsafe_ptr(), albedo.unsafe_ptr())
        if no_denoise:
            # --no-denoise: write the normalized beauty directly (raw render).
            for i in range(n_pixels * 3): denoised[i] = beauty[i]
        else:
            denoise(beauty.unsafe_ptr(), albedo.unsafe_ptr(),
                    normals.unsafe_ptr(), dept.unsafe_ptr(),
                    fw, fh, denoised.unsafe_ptr(),
                    Int32(5), Float32(3.0), Float32(0.2), Float32(0.3), Float32(0.05))
        _ = write_image(denoised.unsafe_ptr(), fw, fh, psc[0].film_filename, Int32(32), Int32(32))
        var albedo_name_buf = List[UInt8](capacity=11)
        var albedo_name_str = "albedo.exr"
        var anp2 = albedo_name_str.unsafe_ptr()
        for i in range(10): albedo_name_buf.append(anp2[i])
        albedo_name_buf.append(UInt8(0))
        _ = write_image(albedo.unsafe_ptr(), fw, fh, albedo_name_buf.unsafe_ptr(), Int32(32), Int32(32))
        # albedo_name_buf, beauty, albedo, denoised, normals, dept freed automatically
    mojo_parsed_free(psc)
    return Int32(0)


def render_interactive(
    path: UnsafePointer[UInt8, MutAnyOrigin],
    sobol: UnsafePointer[UInt32, MutAnyOrigin],
    use_gpu: Bool,
    fullscreen: Bool = False,
    override_w: Int32 = Int32(0), override_h: Int32 = Int32(0),
    spp_override: Int32 = Int32(0),
    verbose: Bool = False,
):
    if use_gpu and not gpu_available():
        print("No GPU available — compile with --target-accelerator sm_86 or similar")
        return

    var psc = mojo_parse_scene(path, verbose)
    if Int(psc) == 0:
        print("Failed to parse scene")
        return
    if override_w > 0 and override_h > 0:
        resize_film(psc, override_w, override_h)

    mojo_apply_overrides(psc, spp_override, Int32(0), Int32(0))

    var fw = psc[0].film_w
    var fh = psc[0].film_h
    var n_pixels = Int(fw) * Int(fh)

    var handle = UnsafePointer[GpuSceneHandle, MutAnyOrigin].unsafe_dangling()
    if use_gpu:
        handle = _gpu_upload_scene(psc, sobol, n_pixels)
        if Int(handle) <= 8:
            mojo_parsed_free(psc)
            return

    var title_len: Int
    var title_str: String
    if use_gpu:
        title_str = "gonzales GPU"
        title_len = 12
    else:
        title_str = "gonzales"
        title_len = 8
    var title_buf = List[UInt8](capacity=title_len + 1)
    var ts = title_str.unsafe_ptr()
    for i in range(title_len):
        title_buf.append(ts[i])
    title_buf.append(UInt8(0))
    var v = viewer_create(fw, fh, title_buf.unsafe_ptr(), Int32(1) if fullscreen else Int32(0))
    # title_buf freed automatically
    if Int(v) == 0:
        print("Failed to create viewer window")
        if use_gpu:
            gpu_free_scene(handle)
        mojo_parsed_free(psc)
        return
    if not use_gpu and psc[0].prim_count == 0:
        print("Warning: scene has no geometry, skipping render")
        viewer_destroy(v)
        mojo_parsed_free(psc)
        return

    var c2w = psc[0].camera_to_world
    var cam_buf = OwnedPointer[CameraState](CameraState(
        position=Point3f(c2w[12], c2w[13], c2w[14]),
        direction=Vec3f(c2w[8],  c2w[9],  c2w[10]),
        up=Vec3f(c2w[4],  c2w[5],  c2w[6]),
        cameraChanged=Int32(0),
    ))
    viewer_set_camera_state(v, cam_buf.unsafe_ptr())

    var c2w_buf = List[Float32](capacity=16)
    for i in range(16): c2w_buf.append(c2w[i])

    var results  = List[TileResult_C](capacity=n_pixels)
    var beauty   = List[Float32](capacity=n_pixels * 3)
    var albedo   = List[Float32](capacity=n_pixels * 3)
    var denoised = List[Float32](capacity=n_pixels * 3)
    for _ in range(n_pixels):
        results.append(TileResult_C(
            estimate=RGB(Float32(0), Float32(0), Float32(0)),
            albedo=RGB(Float32(0), Float32(0), Float32(0)),
            filterWeight=Float32(0), pixelX=Int32(0), pixelY=Int32(0)))
    for _ in range(n_pixels * 3): beauty.append(Float32(0)); albedo.append(Float32(0)); denoised.append(Float32(0))
    var frame_count = 0

    # Mode-specific buffers — dangling until allocated below
    var sd           = UnsafePointer[SceneDescriptor2_C, MutAnyOrigin].unsafe_dangling()
    var accum        = List[Float32]()
    var albedo_acc   = List[Float32]()
    var normals_int  = List[Float32]()
    var depth_int    = List[Float32]()
    var sp_int     = OwnedPointer[TileSamplerParams_C](TileSamplerParams_C(
        sobolMatrices=sobol,
        rngSeed=UInt64(0), sobolSeed=Int32(0),
        log2SamplesPerPixel=Int32(0), nBase4Digits=Int32(1),
        samplesPerPixel=Int32(1),
        filterSigma=psc[0].filter_sigma,
        filterSupportX=psc[0].filter_support_x,
        filterSupportY=psc[0].filter_support_y,
        filterNormX=psc[0].filter_norm_x,
        filterNormY=psc[0].filter_norm_y,
        filterWeight=psc[0].filter_weight,
        filterType=psc[0].filter_type,
    ))

    if use_gpu:
        gpu_clear_film(handle, Int64(n_pixels))
        gpu_gen_aux_buffers(handle, psc[0].camera_to_world, Int64(n_pixels))
    else:
        sd = mojo_parsed_scene_descriptor(psc)
        for _ in range(n_pixels * 3):
            accum.append(Float32(0))
            albedo_acc.append(Float32(0))
            normals_int.append(Float32(0))
        for _ in range(n_pixels):
            depth_int.append(Float32(0))

    var zero = TileResult_C(
        estimate=RGB(Float32(0), Float32(0), Float32(0)),
        albedo=RGB(Float32(0), Float32(0), Float32(0)),
        filterWeight=Float32(0), pixelX=Int32(0), pixelY=Int32(0))

    while not viewer_should_close(v):
        viewer_poll_events(v)
        viewer_get_camera_state(v, result=cam_buf.unsafe_ptr())
        if cam_buf[].cameraChanged != Int32(0):
            frame_count = 0
            build_camera_to_world(cam_buf.unsafe_ptr(), c2w_buf.unsafe_ptr())
            if use_gpu:
                gpu_clear_film(handle, Int64(n_pixels))
                gpu_gen_aux_buffers(handle, c2w_buf.unsafe_ptr(), Int64(n_pixels))
            else:
                for i in range(n_pixels * 3):
                    accum[i]      = Float32(0)
                    albedo_acc[i] = Float32(0)

        if use_gpu:
            comptime log2spp_i = 16
            comptime n_base4_i = 8
            var si = Int32(frame_count % 65536)
            gpu_render_sample(
                handle, c2w_buf.unsafe_ptr(),
                si, Int32(log2spp_i), Int32(n_base4_i),
                UInt32(0), UInt32(0),
                UInt32(frame_count & 0xFFFFFFFF), UInt32(0),
                Int64(n_pixels), psc[0].max_depth,
            )
            frame_count += 1
            gpu_atrous_denoise(handle, denoised.unsafe_ptr(), Int64(n_pixels),
                                    Int32(frame_count),
                                    psc[0].film_iso, psc[0].film_max_comp)
        else:
            sp_int[] = TileSamplerParams_C(
                sobolMatrices=sobol,
                rngSeed=UInt64(frame_count),
                sobolSeed=Int32(frame_count % 65536),
                log2SamplesPerPixel=Int32(0),
                nBase4Digits=Int32(1),
                samplesPerPixel=Int32(1),
                filterSigma=psc[0].filter_sigma,
                filterSupportX=psc[0].filter_support_x,
                filterSupportY=psc[0].filter_support_y,
                filterNormX=psc[0].filter_norm_x,
                filterNormY=psc[0].filter_norm_y,
                filterWeight=psc[0].filter_weight,
                filterType=psc[0].filter_type,
            )
            for i in range(n_pixels):
                results[i] = zero
            render_all_tiles(
                psc[0].raster_to_camera, c2w_buf.unsafe_ptr(),
                Int32(0), Int32(0), fw, fh,
                Int32(32), Int32(32),
                sp_int.unsafe_ptr(), sd, results.unsafe_ptr(), psc[0].max_depth, True)
            if frame_count == 0:
                render_aux_buffers(
                    psc[0].raster_to_camera, c2w_buf.unsafe_ptr(),
                    Int32(0), Int32(0), fw, fh, sd,
                    normals_int.unsafe_ptr(), depth_int.unsafe_ptr())
            var beauty_frame = List[Float32](capacity=n_pixels * 3)
            var albedo_frame = List[Float32](capacity=n_pixels * 3)
            for _ in range(n_pixels * 3): beauty_frame.append(Float32(0)); albedo_frame.append(Float32(0))
            normalize_film(results.unsafe_ptr(), Int32(n_pixels),
                                psc[0].film_iso, psc[0].film_max_comp,
                                beauty_frame.unsafe_ptr(), albedo_frame.unsafe_ptr())
            frame_count += 1
            var w = Float32(1) / Float32(frame_count)
            if frame_count == 1:
                for i in range(n_pixels * 3):
                    accum[i]      = beauty_frame[i]
                    albedo_acc[i] = albedo_frame[i]
            else:
                for i in range(n_pixels * 3):
                    accum[i]      += (beauty_frame[i] - accum[i])      * w
                    albedo_acc[i] += (albedo_frame[i] - albedo_acc[i]) * w
            # beauty_frame and albedo_frame freed automatically
            for i in range(n_pixels * 3):
                beauty[i] = accum[i]
                albedo[i] = albedo_acc[i]
            denoise(beauty.unsafe_ptr(), albedo.unsafe_ptr(),
                    normals_int.unsafe_ptr(), depth_int.unsafe_ptr(),
                    fw, fh, denoised.unsafe_ptr(),
                    Int32(5), Float32(3.0), Float32(0.2), Float32(0.3), Float32(0.05))

        viewer_update_framebuffer(v, denoised.unsafe_ptr(), fw, fh)

    # results, beauty, albedo, denoised, c2w_buf, cam_buf, sp_int freed automatically
    if use_gpu:
        gpu_free_scene(handle)
    else:
        # accum, albedo_acc, sd freed automatically (accum/albedo_acc are List)
        sd.free()
    mojo_parsed_free(psc)
    viewer_destroy(v)
