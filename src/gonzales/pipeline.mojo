from std.memory import alloc, OwnedPointer
from std.collections import List
from std.math import sqrt, tan, ceil
from std.sys.info import size_of
from max.gpu.host import DeviceBuffer
from .pbrt_parser import ParsedScene_Mojo, mojo_parsed_free, mojo_parsed_scene_descriptor, resize_film, mojo_apply_overrides
from .scene_loader import mojo_parse_scene_any
from .rendering import render_all_tiles, normalize_film, fmt_time, progress_str
from std.time import perf_counter_ns
from .geometry import RGB, Point3f, Vec3f, Bounds3f, TileResult_C, PathState_C, Ray_C, dot, TriangleMesh_C, _is_real_ptr, Curve_C, curve_piece_bounds
from .postprocess import denoise, write_image, write_image_cropped
from .sampling import TileSamplerParams_C, mix_bits_u64, encode_morton2, sobol_get_sample_index, sobol_sample, gaussian_sample_1d, derive_pcg_seeds
from .bvh import BVH2Node, SceneDescriptor2_C, render_aux_buffers
from .sppm import sppm_render
from .bdpt import vcm_render, vcm_render_gpu, vcm_render_gpu_wavefront, _BDPT_MAX_VERTS, sppm_render_gpu
from .guide import GuideGrid, guide_create, guide_free, guide_clone_empty, guide_refine, null_guide, guide_merge, guide_cell_has_data
from .restir_di import DIReservoir, di_reservoir_init, ReservoirIO, reservoir_io_null
from .restir_gi import GIReservoir, gi_reservoir_init, GIReservoirIO, gi_reservoir_io_null
from .restir_sms import SMSReservoir, sms_reservoir_init, SMSReservoirIO, sms_reservoir_io_null
from .gpu import GpuSceneHandle, WAVEFRONT_BATCH, gpu_available, gpu_upload_scene, gpu_render_sample, gpu_render_wavefront, gpu_download_film, gpu_download_albedo, gpu_clear_film, gpu_clear_restir, gpu_atrous_denoise, gpu_gen_aux_buffers, gpu_free_scene
from .viewer import CameraState, ViewerHandle, viewer_create, viewer_update_framebuffer, viewer_should_close, viewer_poll_events, viewer_get_camera_state, viewer_set_camera_state, viewer_destroy, build_camera_to_world
from .spectrum import SpectralHandle, null_spectral_handle
from .vulkanrt import VulkanRtSceneHandle, vulkanrt_build_scene, vulkanrt_destroy_scene
from .vulkaninterop import (
    VulkanInteropRtSceneHandle, vulkaninterop_rt_create_scene,
    vulkaninterop_rt_get_rays_ptr, vulkaninterop_rt_get_results_ptr,
    vulkaninterop_rt_destroy_scene,
)

# Resolve effective SPPM radius/photons-per-pass: an explicit CLI flag
# (sentinel -1 = not passed) wins; otherwise fall back to what the scene's
# own `Integrator "sppm"` directive specifies; otherwise a last-resort
# default (0.05 for radius; film_w*film_h for photons, matching pbrt-v4's
# own SPPM integrator default of "one photon per pixel" when
# photonsperiteration is unspecified).
def _resolve_sppm_params(
    psc: UnsafePointer[ParsedScene_Mojo, MutExternalOrigin],
    sppm_photons_cli: Int32,
    sppm_radius_cli: Float32,
) -> Tuple[Int32, Float32]:
    var radius = sppm_radius_cli
    if radius <= Float32(0):
        radius = psc[0].sppm_radius if psc[0].sppm_radius > Float32(0) else Float32(0.05)
    var photons = sppm_photons_cli
    if photons <= Int32(0):
        if psc[0].sppm_photons_per_iter > Int32(0):
            photons = psc[0].sppm_photons_per_iter
        else:
            photons = psc[0].film_w * psc[0].film_h
    return (photons, radius)

# Resolve the VCM merge side's light-path budget for the current pass
# (task #152's fix): an explicit `--vcm-photons` CLI value (sentinel <=0 =
# not passed) wins if it's at least n_pix; otherwise falls back to n_pix,
# today's default (one light path per pixel, deterministically paired with
# that pixel's own camera subpath for CONNECTION -- see bdpt.mojo's VCM
# module comment). A value below n_pix would starve the per-pixel
# connect pairing of a light path to pair with, so it's clamped up, not
# silently accepted -- unlike SPPM's photon count, which has no such
# lower bound because SPPM has no per-pixel light-path pairing at all.
def _resolve_vcm_photons(vcm_photons_cli: Int32, n_pix: Int) -> Int:
    if vcm_photons_cli > Int32(0):
        return max(Int(vcm_photons_cli), n_pix)
    return n_pix

def _resolve_vcm_spp(vcm_spp_cli: Int32, scene_spp: Int32) -> Int:
    """--vcm-spp not passed on CLI (sentinel -1) falls back to the scene's
    own Sampler "pixelsamples" (psc[0].samples_per_pixel) -- same pattern
    _resolve_vcm_photons/_resolve_sppm_params already use for their own
    CLI-vs-scene-default fallbacks."""
    if vcm_spp_cli > Int32(0):
        return Int(vcm_spp_cli)
    return Int(scene_spp)

# Generate Sobol matrices from the Joe-Kuo data file.
# Returns a heap-allocated pointer to 21201*52 UInt32 values, or null on error.
def _generate_sobol_matrices(path: String) -> Optional[UnsafePointer[UInt32, MutExternalOrigin]]:
    var file_buf: UnsafePointer[UInt8, MutExternalOrigin]
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
    psc: UnsafePointer[ParsedScene_Mojo, MutExternalOrigin],
    sobol: UnsafePointer[UInt32, MutExternalOrigin],
    n_pixels: Int,
    # Decomposed, NOT a single by-value `spectral: SpectralHandle` param --
    # see spectrum.mojo's long comment on the confirmed by-value SpectralHandle
    # miscompilation; this GPU-upload path reproduced the same corruption
    # class (see project_priority_backlog memory item 3).
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
) -> UnsafePointer[GpuSceneHandle, MutExternalOrigin]:
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
        # CPU-inclusive TLAS (tris+curves+instances) — now that GPU has
        # BLAS/instance upload + traversal support, it uses the same TLAS
        # SceneDescriptor2_C does rather than the instance-free one.
        psc[0].bvh_nodes_cpu,      Int64(psc[0].bvh_node_count_cpu),
        psc[0].prim_ids_cpu,       Int64(psc[0].prim_count_cpu),
        psc[0].blas_nodes_arr, psc[0].blas_primids_arr,
        psc[0].blas_node_counts, psc[0].blas_primid_counts, Int64(psc[0].blas_count),
        psc[0].instances, Int64(psc[0].instance_count),
        psc[0].meshes,         Int64(n_meshes),
        pts_counts.unsafe_ptr(), fi_counts.unsafe_ptr(),
        vi_counts.unsafe_ptr(), uv_counts.unsafe_ptr(),
        nrm_counts.unsafe_ptr(),
        psc[0].tex_filenames,  psc[0].tex_count,
        psc[0].materials,      Int64(psc[0].material_count),
        psc[0].area_lights,    Int64(psc[0].area_light_count),
        psc[0].spheres,        Int64(psc[0].sphere_count),
        psc[0].curves,         Int64(psc[0].curve_count),
        psc[0].distant_lights, Int64(psc[0].distant_count),
        psc[0].point_lights,   Int64(psc[0].point_count),
        psc[0].light_sampler.cdf, Int64(psc[0].light_sampler.n),
        psc[0].infinite_lights, Int64(psc[0].infinite_count),
        psc[0].mediums,         Int64(psc[0].medium_count),
        psc[0].medium_ifaces,   Int64(psc[0].medium_iface_count),
        psc[0].grids,           Int64(psc[0].grid_count),
        psc[0].measured_brdfs, Int64(psc[0].measured_count),
        Int64(n_pixels),
        sobol,
        psc[0].raster_to_camera, psc[0].camera_to_world,
        psc[0].filter_sigma, psc[0].filter_support_x, psc[0].filter_support_y,
        psc[0].filter_norm_x, psc[0].filter_norm_y,
        psc[0].filter_type,
        fw, fh,
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
    )
    # pts_counts, fi_counts, vi_counts, uv_counts freed automatically
    return handle


def _dbg_vlen(x: Float32, y: Float32, z: Float32) -> Float32:
    return sqrt(x*x + y*y + z*z)

# Task #163: gonzales assigns exactly one material per mesh at parse time
# (see pbrt_parser.mojo's store_mesh/MeshAccum), but that mapping is only
# recorded per-triangle inside PrimId_C entries, not as a standalone
# per-mesh array -- vulkanrt_traverse_paths_gpu needs the latter (it gets
# a (mesh, triangle) hit back from Vulkan RT, not a PrimId_C).
#
# Area-light meshes need special handling: finalize_scene (pbrt_parser.mojo)
# encodes their triangles with PrimId_C.type == 3 (not the ordinary
# type == 0), where id1 is the AREA-LIGHT index (not the mesh index) and
# materialIndex points at a synthetic per-light material -- shade_nee_core's
# direct-emission-credit path (shading.mojo ~2744) keys off exactly this
# type==3/id1==al_idx encoding. Reconstructing type==0 for a light mesh's
# hits (the obvious-looking thing to do) silently loses all direct-hit
# light emission -- caught via a real cornell-box regression (Vulkan render
# ~40% dimmer, light strip rendering as unlit) before this fix.
#
# Recovers both mappings with one pass over the CPU-safe prim_ids array:
# for each mesh index not yet seen, records its first primitive's type,
# materialIndex, and (if type==3) area-light index -- every triangle of a
# given mesh shares the same encoding by construction, so any one is
# representative. Returns (mesh_material_idx, mesh_al_idx); the latter is
# -1 for ordinary (non-light) meshes.
def _build_mesh_light_info(
    psc: UnsafePointer[ParsedScene_Mojo, MutExternalOrigin],
) -> Tuple[UnsafePointer[Int64, MutExternalOrigin], UnsafePointer[Int32, MutExternalOrigin]]:
    var n_meshes = Int(psc[0].mesh_count)
    var mat_idx = alloc[Int64](max(n_meshes, 1))
    var al_idx = alloc[Int32](max(n_meshes, 1))
    var seen = alloc[UInt8](max(n_meshes, 1))
    for i in range(max(n_meshes, 1)):
        mat_idx[i] = Int64(0)
        al_idx[i] = Int32(-1)
        seen[i] = UInt8(0)
    for i in range(Int(psc[0].prim_count)):
        var p = psc[0].prim_ids[i]
        if p.type == Int8(0):
            var mi = Int(p.id1)
            if mi >= 0 and mi < n_meshes and seen[mi] == UInt8(0):
                mat_idx[mi] = p.materialIndex
                seen[mi] = UInt8(1)
        elif p.type == Int8(3):
            var mi = Int(p.id2 >> 32)
            if mi >= 0 and mi < n_meshes and seen[mi] == UInt8(0):
                mat_idx[mi] = p.materialIndex
                al_idx[mi] = Int32(p.id1)
                seen[mi] = UInt8(1)
    # Object-instancing templates: their mesh triangles live ONLY in each
    # template's own BLAS (psc[0].prim_ids above is the ordinary top-level
    # TLAS, which explicitly excludes template meshes -- see finalize_scene),
    # so without this second pass every template mesh index would be left at
    # the mat_idx=0/al_idx=-1 default, giving Vulkan RT instance hits the
    # wrong material. AreaLightSource is not supported inside ObjectBegin/
    # ObjectEnd (parser skips it there), so template triangles are always
    # type==0 -- no type==3 case needed here.
    for tmpl in range(Int(psc[0].blas_count)):
        var tprims = psc[0].blas_primids_arr[tmpl]
        for i in range(Int(psc[0].blas_primid_counts[tmpl])):
            var p = tprims[i]
            var mi = Int(p.id1)
            if mi >= 0 and mi < n_meshes and seen[mi] == UInt8(0):
                mat_idx[mi] = p.materialIndex
                seen[mi] = UInt8(1)
    seen.free()
    return (mat_idx, al_idx)

def debug_trace_pixel(
    path: UnsafePointer[UInt8, MutExternalOrigin],
    px: Int32, py: Int32,
):
    """Trace the centre ray of one pixel and print the path bounce-by-bounce
    (hit mesh/material/normal/t, dielectric entering/eta/Fresnel decision,
    envmap lookup). For comparing against `pbrt --pixelmaterial`."""
    from .bvh import traverse_bvh2_core, test_spheres, any_hit_bvh2_core, _equal_area_sphere_to_square
    from .geometry import Intersection_C, Material_C, cross, fr_dielectric

    var psc = mojo_parse_scene_any(path)
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
        traverse_bvh2_core(psc[0].bvh_nodes, psc[0].prim_ids, psc[0].meshes, psc[0].curves, ray, Float32(1.0e38), inter,
                            psc[0].blas_nodes_arr, psc[0].blas_primids_arr, psc[0].instances)
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
                if _is_real_ptr(il.pixels_ptr) and il.cdf_w > Int32(0):
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
            traverse_bvh2_core(psc[0].bvh_nodes, psc[0].prim_ids, psc[0].meshes, psc[0].curves, rray, Float32(1.0e38), rint,
                                psc[0].blas_nodes_arr, psc[0].blas_primids_arr, psc[0].instances)
            if rint[0].hit == Int8(0) and psc[0].infinite_count > 0:
                var il2 = psc[0].infinite_lights[0]
                var w2 = il2.world_to_light
                var l2x = w2[0]*rfx + w2[4]*rfy + w2[8]*rfz
                var l2y = w2[1]*rfx + w2[5]*rfy + w2[9]*rfz
                var l2z = w2[2]*rfx + w2[6]*rfy + w2[10]*rfz
                var uv2 = _equal_area_sphere_to_square(l2x, l2y, l2z)
                var rs = String("")
                if _is_real_ptr(il2.pixels_ptr) and il2.cdf_w > Int32(0):
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
        elif Int(mat.type) == 5:
            # CoatedDiffuse — mirror reflection off the coat only (ignore roughness/
            # transmit-into-base for this probe; just checking what the coat's
            # specular-ish lobe geometrically faces).
            var facing5 = (dx*gnx + dy*gny + dz*gnz) < Float32(0.0)
            var nx5 = gnx if facing5 else -gnx
            var ny5 = gny if facing5 else -gny
            var nz5 = gnz if facing5 else -gnz
            var rcos5 = dx*nx5 + dy*ny5 + dz*nz5
            var rfx5 = dx - nx5*Float32(2.0)*rcos5
            var rfy5 = dy - ny5*Float32(2.0)*rcos5
            var rfz5 = dz - nz5*Float32(2.0)*rcos5
            var rfl5 = _dbg_vlen(rfx5, rfy5, rfz5)
            if rfl5 > Float32(0.0): rfx5 /= rfl5; rfy5 /= rfl5; rfz5 /= rfl5
            var rray5 = Ray_C(Point3f(hx+nx5*Float32(0.001), hy+ny5*Float32(0.001), hz+nz5*Float32(0.001)), Vec3f(rfx5, rfy5, rfz5))
            var rint5 = alloc[Intersection_C](1); rint5[0].hit = Int8(0)
            traverse_bvh2_core(psc[0].bvh_nodes, psc[0].prim_ids, psc[0].meshes, psc[0].curves, rray5, Float32(1.0e38), rint5,
                                psc[0].blas_nodes_arr, psc[0].blas_primids_arr, psc[0].instances)
            if rint5[0].hit == Int8(0):
                print("        COAT REFLECT dir", rfx5, rfy5, rfz5, "-> MISS (no envmap in this scene)")
            else:
                var ptype5 = Int(rint5[0].primId.type)
                var pmatidx5 = Int(rint5[0].primId.materialIndex)
                print("        COAT REFLECT dir", rfx5, rfy5, rfz5, "-> hit primType", ptype5, "matType", Int(psc[0].materials[pmatidx5].type), "matIdx", pmatidx5, "t", rint5[0].tHit)
            rint5.free()
            print("        STOP (coateddiffuse probe only, not following further)")
            break
        elif Int(mat.type) == 3:
            # Conductor — mirror reflection only (ignore roughness for this probe).
            var facing3 = (dx*gnx + dy*gny + dz*gnz) < Float32(0.0)
            var nx3 = gnx if facing3 else -gnx
            var ny3 = gny if facing3 else -gny
            var nz3 = gnz if facing3 else -gnz
            var rcos3 = dx*nx3 + dy*ny3 + dz*nz3
            var rfx3 = dx - nx3*Float32(2.0)*rcos3
            var rfy3 = dy - ny3*Float32(2.0)*rcos3
            var rfz3 = dz - nz3*Float32(2.0)*rcos3
            var rfl3 = _dbg_vlen(rfx3, rfy3, rfz3)
            if rfl3 > Float32(0.0): rfx3 /= rfl3; rfy3 /= rfl3; rfz3 /= rfl3
            var rray3 = Ray_C(Point3f(hx+nx3*Float32(0.001), hy+ny3*Float32(0.001), hz+nz3*Float32(0.001)), Vec3f(rfx3, rfy3, rfz3))
            var rint3 = alloc[Intersection_C](1); rint3[0].hit = Int8(0)
            traverse_bvh2_core(psc[0].bvh_nodes, psc[0].prim_ids, psc[0].meshes, psc[0].curves, rray3, Float32(1.0e38), rint3,
                                psc[0].blas_nodes_arr, psc[0].blas_primids_arr, psc[0].instances)
            if rint3[0].hit == Int8(0):
                print("        REFLECT dir", rfx3, rfy3, rfz3, "-> MISS (no envmap in this scene)")
            else:
                var ptype3 = Int(rint3[0].primId.type)
                var pmatidx3 = Int(rint3[0].primId.materialIndex)
                print("        REFLECT dir", rfx3, rfy3, rfz3, "-> hit primType", ptype3, "matType", Int(psc[0].materials[pmatidx3].type), "matIdx", pmatidx3, "t", rint3[0].tHit)
            rint3.free()
            print("        STOP (conductor probe only, not following further)")
            break
        elif Int(mat.type) == 1:
            # Diffuse — occlusion probe toward every light in the scene, offset
            # along the geometric normal like a real shadow ray would be.
            var ox1 = hx + gnx*Float32(0.0001)
            var oy1 = hy + gny*Float32(0.0001)
            var oz1 = hz + gnz*Float32(0.0001)
            for dli in range(Int(psc[0].distant_count)):
                var dl = psc[0].distant_lights[dli]
                var ldx = -dl.direction.x; var ldy = -dl.direction.y; var ldz = -dl.direction.z
                var cos_s = gnx*ldx + gny*ldy + gnz*ldz
                var sray = Ray_C(Point3f(ox1, oy1, oz1), Vec3f(ldx, ldy, ldz))
                var occluded = any_hit_bvh2_core(psc[0].bvh_nodes, psc[0].prim_ids, psc[0].meshes, psc[0].curves, sray, Float32(2000.0),
                                                  psc[0].blas_nodes_arr, psc[0].blas_primids_arr, psc[0].instances,
                                                  psc[0].spheres, Int(psc[0].sphere_count))
                print("        DISTANT", dli, "dir", ldx, ldy, ldz, "cos_s", cos_s, "occluded", Int(occluded))
            for ali in range(Int(psc[0].area_light_count)):
                var al = psc[0].area_lights[ali]
                var almesh = psc[0].meshes[Int(al.meshIdx)]
                # Centroid of the light's first triangle — coarse but enough to
                # tell whether shadow rays toward this light are ever blocked.
                var lv0 = Int(almesh.vertexIndices[0]); var lv1 = Int(almesh.vertexIndices[1]); var lv2 = Int(almesh.vertexIndices[2])
                var lcx = (almesh.points[lv0*4]   + almesh.points[lv1*4]   + almesh.points[lv2*4])   / Float32(3.0)
                var lcy = (almesh.points[lv0*4+1] + almesh.points[lv1*4+1] + almesh.points[lv2*4+1]) / Float32(3.0)
                var lcz = (almesh.points[lv0*4+2] + almesh.points[lv1*4+2] + almesh.points[lv2*4+2]) / Float32(3.0)
                var tlx = lcx - ox1; var tly = lcy - oy1; var tlz = lcz - oz1
                var tdist = _dbg_vlen(tlx, tly, tlz)
                if tdist > Float32(0.0):
                    tlx /= tdist; tly /= tdist; tlz /= tdist
                var cos_sa = gnx*tlx + gny*tly + gnz*tlz
                var sray2 = Ray_C(Point3f(ox1, oy1, oz1), Vec3f(tlx, tly, tlz))
                var occluded2 = any_hit_bvh2_core(psc[0].bvh_nodes, psc[0].prim_ids, psc[0].meshes, psc[0].curves, sray2, tdist * Float32(0.999),
                                                   psc[0].blas_nodes_arr, psc[0].blas_primids_arr, psc[0].instances,
                                                   psc[0].spheres, Int(psc[0].sphere_count))
                print("        AREA", ali, "centroid", lcx, lcy, lcz, "dist", tdist, "cos_s", cos_sa, "occluded", Int(occluded2))
            print("        STOP (diffuse probe only, not following further)")
            break
        else:
            print("        STOP (non-glass material)")
            break
    inter.free()
    mojo_parsed_free(psc)


def debug_render_vulkanrt(
    path: UnsafePointer[UInt8, MutExternalOrigin],
    verbose: Bool = False,
):
    """Task #162 step 4: build a real Vulkan RT scene from the parsed
    scene's own triangle meshes, batch-trace one primary ray per pixel
    through hardware ray tracing (one vulkanrt_trace_rays dispatch for the
    whole image), cross-check every pixel's hit/miss and hit distance
    against the CPU software BVH tracing the identical ray, and write a
    simple N.V-shaded visibility image from the Vulkan RT result to
    vulkanrt_debug.exr.

    Scope: triangle geometry only (spheres/curves are never uploaded to
    the Vulkan scene, matching vulkanrt_build_scene's own scope) and no
    object instancing (Instance_C placements are not applied -- an
    ObjectInstance template's mesh would appear once at its raw local-space
    location instead of at each placed instance's transform). Scenes using
    either feature will show real disagreement in the printed CPU-vs-GPU
    stats below -- an honest, known limitation of this validation pass,
    not a bug to chase; see project_vulkan_rt_backend memory."""
    from .bvh import traverse_bvh2_core
    from .geometry import Intersection_C
    from .vulkanrt import vulkanrt_build_scene, vulkanrt_trace_rays, vulkanrt_destroy_scene
    from std.math import abs

    var psc = mojo_parse_scene_any(path, verbose)
    if Int(psc) == 0:
        print("parse failed"); return

    var w = Int(psc[0].film_w)
    var h = Int(psc[0].film_h)
    var n_meshes = Int(psc[0].mesh_count)
    if n_meshes == 0:
        print("Scene has no triangle meshes -- nothing for Vulkan RT to trace.")
        mojo_parsed_free(psc)
        return
    if psc[0].instance_count > Int32(0):
        print("WARNING: scene uses ObjectInstance -- the Vulkan RT scene will be missing instanced geometry placements (see project_vulkan_rt_backend memory)")

    # Build the Vulkan RT scene directly from the parsed meshes -- points/
    # vertexIndices pointers passed through as-is (vulkanrt.h's
    # VulkanRtMesh is a field-for-field mirror of TriangleMesh_C).
    var vmeshes = alloc[TriangleMesh_C](n_meshes)
    var point_counts = alloc[Int64](n_meshes)
    var vidx_counts = alloc[Int64](n_meshes)
    for i in range(n_meshes):
        vmeshes[i] = psc[0].meshes[i]
        point_counts[i] = Int64(psc[0].mesh_n_verts[i])
        vidx_counts[i] = Int64(psc[0].mesh_n_tris[i]) * 3

    var scene = vulkanrt_build_scene(vmeshes, Int64(n_meshes), point_counts, vidx_counts)
    if Int(scene) == 0:
        print("vulkanrt_build_scene FAILED -- see stderr for diagnostics")
        vmeshes.free(); point_counts.free(); vidx_counts.free()
        mojo_parsed_free(psc)
        return

    # One centre-ray (no AA jitter) per pixel, same raster_to_camera /
    # camera_to_world math as debug_trace_pixel above -- packed straight
    # into vulkanrt_trace_rays's flat 8-floats-per-ray layout.
    var n_pix = w * h
    var rays = alloc[Float32](n_pix * 8)
    var r2c = psc[0].raster_to_camera
    var c2w = psc[0].camera_to_world
    for py in range(h):
        for px in range(w):
            var fX = Float32(px) + Float32(0.5)
            var fY = Float32(py) + Float32(0.5)
            var cx = r2c[0]*fX + r2c[4]*fY + r2c[12]
            var cy = r2c[1]*fX + r2c[5]*fY + r2c[13]
            var cz = r2c[2]*fX + r2c[6]*fY + r2c[14]
            var cw = r2c[3]*fX + r2c[7]*fY + r2c[15]
            if cw != Float32(0.0) and cw != Float32(1.0):
                cx /= cw; cy /= cw; cz /= cw
            var cl = sqrt(cx*cx + cy*cy + cz*cz)
            if cl > Float32(0.0): cx /= cl; cy /= cl; cz /= cl
            var dx = c2w[0]*cx + c2w[4]*cy + c2w[8]*cz
            var dy = c2w[1]*cx + c2w[5]*cy + c2w[9]*cz
            var dz = c2w[2]*cx + c2w[6]*cy + c2w[10]*cz
            var dl = sqrt(dx*dx + dy*dy + dz*dz)
            if dl > Float32(0.0): dx /= dl; dy /= dl; dz /= dl
            var ox = c2w[12]; var oy = c2w[13]; var oz = c2w[14]
            var idx = (py * w + px) * 8
            rays[idx+0] = ox; rays[idx+1] = oy; rays[idx+2] = oz; rays[idx+3] = Float32(0.001)
            rays[idx+4] = dx; rays[idx+5] = dy; rays[idx+6] = dz; rays[idx+7] = Float32(1.0e8)

    var out_t = alloc[Float32](n_pix)
    var out_u = alloc[Float32](n_pix)
    var out_v = alloc[Float32](n_pix)
    var out_mesh = alloc[Int32](n_pix)
    var out_tri = alloc[Int32](n_pix)
    var out_hit = alloc[UInt8](n_pix)

    var rc = vulkanrt_trace_rays(scene, Int32(n_pix), rays, out_t, out_u, out_v,
                                  out_mesh, out_tri, out_hit)
    if Int(rc) == 0:
        print("vulkanrt_trace_rays FAILED -- see stderr for diagnostics")
        out_t.free(); out_u.free(); out_v.free(); out_mesh.free(); out_tri.free(); out_hit.free()
        rays.free()
        vulkanrt_destroy_scene(scene)
        vmeshes.free(); point_counts.free(); vidx_counts.free()
        mojo_parsed_free(psc)
        return

    # Cross-check every pixel against the CPU software BVH tracing the
    # identical ray (step 5's image-level validation, folded into step 4),
    # and build an N.V-shaded visibility image from the GPU hits.
    var img = alloc[Float32](n_pix * 3)
    var inter = alloc[Intersection_C](1)
    var cpu_hits = 0
    var gpu_hits = 0
    var both_hit = 0
    var agree = 0
    var depth_err_sum = Float64(0)
    var depth_err_max = Float32(0)
    for py in range(h):
        for px in range(w):
            var pi = py * w + px
            var idx = pi * 8
            var ray = Ray_C(Point3f(rays[idx+0], rays[idx+1], rays[idx+2]),
                             Vec3f(rays[idx+4], rays[idx+5], rays[idx+6]))
            inter[0].hit = Int8(0)
            traverse_bvh2_core(psc[0].bvh_nodes, psc[0].prim_ids, psc[0].meshes, psc[0].curves, ray, Float32(1.0e8), inter,
                                psc[0].blas_nodes_arr, psc[0].blas_primids_arr, psc[0].instances)
            var cpu_hit = inter[0].hit != Int8(0)
            var gpu_hit = out_hit[pi] == UInt8(1)
            if cpu_hit: cpu_hits += 1
            if gpu_hit: gpu_hits += 1
            if cpu_hit == gpu_hit: agree += 1
            if cpu_hit and gpu_hit:
                both_hit += 1
                var derr = abs(inter[0].tHit - out_t[pi])
                depth_err_sum += Float64(derr)
                if derr > depth_err_max: depth_err_max = derr

            var shade = Float32(0)
            if gpu_hit:
                var mi = Int(out_mesh[pi])
                var ti = Int(out_tri[pi]) * 3
                if mi >= 0 and mi < n_meshes:
                    var mesh = psc[0].meshes[mi]
                    var v0 = Int(mesh.vertexIndices[ti]); var v1 = Int(mesh.vertexIndices[ti+1]); var v2 = Int(mesh.vertexIndices[ti+2])
                    var p0x = mesh.points[v0*4]; var p0y = mesh.points[v0*4+1]; var p0z = mesh.points[v0*4+2]
                    var p1x = mesh.points[v1*4]; var p1y = mesh.points[v1*4+1]; var p1z = mesh.points[v1*4+2]
                    var p2x = mesh.points[v2*4]; var p2y = mesh.points[v2*4+1]; var p2z = mesh.points[v2*4+2]
                    var gnx = (p1y-p0y)*(p2z-p0z) - (p1z-p0z)*(p2y-p0y)
                    var gny = (p1z-p0z)*(p2x-p0x) - (p1x-p0x)*(p2z-p0z)
                    var gnz = (p1x-p0x)*(p2y-p0y) - (p1y-p0y)*(p2x-p0x)
                    var gnl = sqrt(gnx*gnx + gny*gny + gnz*gnz)
                    if gnl > Float32(0): gnx /= gnl; gny /= gnl; gnz /= gnl
                    shade = abs(rays[idx+4]*gnx + rays[idx+5]*gny + rays[idx+6]*gnz)
            img[pi*3+0] = shade; img[pi*3+1] = shade; img[pi*3+2] = shade

    var hit_agree_pct = Float64(agree) * 100.0 / Float64(max(n_pix, 1))
    var mean_depth_err = depth_err_sum / Float64(max(both_hit, 1))
    print("vulkanrt visibility check:", w, "x", h, "pixels")
    print("  CPU hits:", cpu_hits, " GPU hits:", gpu_hits, " both-hit:", both_hit)
    print("  hit/miss agreement:", hit_agree_pct, "%")
    print("  depth error (both-hit pixels): mean", mean_depth_err, " max", depth_err_max)

    var out_path = String("vulkanrt_debug.exr")
    var out_cstr = alloc[UInt8](out_path.byte_length() + 1)
    for k in range(out_path.byte_length()):
        out_cstr[k] = out_path.as_bytes()[k]
    out_cstr[out_path.byte_length()] = UInt8(0)
    _ = write_image(img, Int32(w), Int32(h), out_cstr, Int32(32), Int32(32))
    print("  wrote", out_path)
    out_cstr.free()

    inter.free()
    img.free()
    out_t.free(); out_u.free(); out_v.free(); out_mesh.free(); out_tri.free(); out_hit.free()
    rays.free()
    vulkanrt_destroy_scene(scene)
    vmeshes.free(); point_counts.free(); vidx_counts.free()
    mojo_parsed_free(psc)


def parse_and_render(
    path: UnsafePointer[UInt8, MutExternalOrigin],
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
    use_gpu: Bool,
    spectral: SpectralHandle = null_spectral_handle(),
    override_w: Int32 = Int32(0), override_h: Int32 = Int32(0),
    no_denoise: Bool = False,
    spp_override: Int32 = Int32(0),
    seed_override: Int64 = Int64(-1),
    verbose: Bool = False,
    use_sppm: Bool = False,
    sppm_passes: Int32 = Int32(64),
    sppm_photons: Int32 = Int32(-1),
    sppm_radius: Float32 = Float32(-1),
    use_guide: Bool = False,
    use_vcm: Bool = False,
    vcm_spp: Int32 = Int32(-1),  # -1 = not passed on CLI; fall back to scene pixelsamples
    vcm_photons: Int32 = Int32(-1),
    # Task #163: route the plain --gpu wavefront path tracer's per-bounce
    # primary intersection test through the Vulkan RT backend instead of
    # the CUDA software-BVH kernel. Only wired into the plain --gpu path
    # (not --sppm/--vcm) and only takes effect for scenes with no curves/
    # spheres/object instancing (see vulkanrt_traverse_paths_gpu's
    # docstring in gpu.mojo) -- falls back to CUDA traversal with a
    # warning otherwise.
    use_vulkan_rt_shade: Bool = False,
    # Task #163 stage 4 part 3: route --vcm --gpu through
    # vcm_render_gpu_wavefront (staged intersect+bounce kernels, one launch
    # per depth level) instead of vcm_render_gpu's single-mega-kernel-per-
    # subpath design. Same software-BVH intersect either way -- this only
    # exercises the staging itself, a prerequisite for a future Vulkan RT
    # swap, not yet a speed win. See vcm_render_gpu_wavefront's docstring.
    use_vcm_wavefront: Bool = False,
    # docs/A2_restir_migration_plan.md Phase 2/3: ReSTIR DI, wired into both
    # the CPU path and the plain --gpu path (batch mode renders 1 sample/
    # pixel per dispatch via gpu_render_sample for full reservoir reuse --
    # see the use_restir branch below). No effect combined with --sppm/
    # --vcm/--guide (see the warnings just below) -- Phase 2 only touches
    # the plain path tracer's diffuse-material NEE.
    use_restir: Bool = False,
    # Phase 4: ReSTIR GI, one-bounce reconnection scoped to diffuse x1 AND
    # diffuse x2 (shading.mojo's _gi_generate_recon_candidate). CPU-only,
    # batch mode only, no temporal/spatial reuse yet -- matches --restir's
    # own batch-mode scope (single-frame RIS, no persistence). No effect
    # without --restir (see the warning below); this flag exists so
    # --restir's existing, already-shipped DI-only meaning never changes
    # just because GI machinery happens to be linked in -- see
    # project_restir_migration memory for why that separation is load-
    # bearing (a real bug slipped through when it wasn't kept explicit).
    use_restir_gi: Bool = False,
    # Phase 6: ReSTIR SMS temporal reuse for glass-caustic MNEE probing
    # (shading.mojo's sms_temporal_step). Batch mode has no cross-frame
    # concept (frame_w is never passed to render_all_tiles here, so
    # pixel_idx stays -1 throughout, same as --restir's own batch scope) --
    # and unlike DI/GI, a single SMS candidate with no reservoir combine is
    # mathematically identical to plain per-frame MNEE (W collapses to
    # exactly inv_pdf_area*trials for a lone accepted candidate), so there
    # is no "generate-only, no reuse" batch-mode value to offer: this flag
    # genuinely has no effect at all without --interactive, unlike
    # --restir-gi's own batch-mode fallback.
    use_sms_restir: Bool = False,
) raises -> Int32:
    if use_gpu and not gpu_available():
        print("No GPU available — compile with --target-accelerator sm_86 or similar")
        return Int32(-1)

    var psc = mojo_parse_scene_any(path, verbose)
    if Int(psc) == 0:
        return Int32(-1)
    if override_w > 0 and override_h > 0:
        resize_film(psc, override_w, override_h)
    mojo_apply_overrides(psc, spp_override, Int32(0), Int32(0), seed_override)

    var fw = psc[0].film_w
    var fh = psc[0].film_h
    var n_pixels = Int(fw) * Int(fh)
    var results = List[TileResult_C](capacity=n_pixels)

    # Film "float cropwindow" [x0 x1 y0 y1] pixel bounds — ceil() on both
    # ends matches pbrt's own Film pixel-bounds computation exactly (verified
    # against a real cropwindow scene's reference render). Only applied at
    # image-write time (see write_image_cropped below) for the plain
    # CPU/GPU-wavefront path tracer — --sppm/--vcm don't honor cropwindow
    # yet (their own render+write call sites are in sppm.mojo/bdpt.mojo).
    var crop_x0px = Int32(ceil(psc[0].crop_x0 * Float32(fw)))
    var crop_y0px = Int32(ceil(psc[0].crop_y0 * Float32(fh)))
    var crop_x1px = Int32(ceil(psc[0].crop_x1 * Float32(fw)))
    var crop_y1px = Int32(ceil(psc[0].crop_y1 * Float32(fh)))
    var crop_w = crop_x1px - crop_x0px
    var crop_h = crop_y1px - crop_y0px

    if use_restir and use_sppm:
        print("--restir: no effect combined with --sppm (Phase 2 only touches the plain path tracer's diffuse-material NEE)")
    if use_restir and use_vcm:
        print("--restir: no effect combined with --vcm (Phase 2 only touches the plain path tracer's diffuse-material NEE)")
    if use_restir and use_guide:
        print("--restir: no effect combined with --guide (Phase 2 only wired into the plain, non-guided CPU render path so far)")
    if use_restir_gi and not use_restir:
        print("--restir-gi: no effect without --restir")
    if use_restir_gi and use_gpu:
        print("--restir-gi: CPU batch only so far, no effect combined with --gpu")
    if use_sms_restir:
        print("--sms-restir: no effect without --interactive (batch mode has no cross-frame reservoir persistence to reuse)")

    if use_gpu and use_sppm:
        var sd = mojo_parsed_scene_descriptor(psc, spectral)
        var handle = _gpu_upload_scene(psc, sobol_matrices, n_pixels, spectral.coeffs, spectral.res, spectral.cie_x, spectral.cie_y, spectral.cie_z, spectral.d65)
        if not _is_real_ptr(handle):
            sd.free()
            mojo_parsed_free(psc)
            return Int32(-1)
        var resolved = _resolve_sppm_params(psc, sppm_photons, sppm_radius)
        var ret = sppm_render_gpu(
            handle, psc, sd[0],
            Int(sppm_passes), Int(resolved[0]), resolved[1],
            no_denoise, verbose,
        )
        gpu_free_scene(handle)
        sd.free()
        mojo_parsed_free(psc)
        return ret
    elif use_gpu and use_vcm:
        var sd = mojo_parsed_scene_descriptor(psc, spectral)
        var handle = _gpu_upload_scene(psc, sobol_matrices, n_pixels, spectral.coeffs, spectral.res, spectral.cie_x, spectral.cie_y, spectral.cie_z, spectral.d65)
        if not _is_real_ptr(handle):
            sd.free()
            mojo_parsed_free(psc)
            return Int32(-1)
        var n_photons = _resolve_vcm_photons(vcm_photons, n_pixels)
        var resolved_vcm_spp = _resolve_vcm_spp(vcm_spp, psc[0].samples_per_pixel)
        var ret: Int32
        if use_vcm_wavefront:
            # Task #163 stage 4 part 4: build the same interop-AND-ray-
            # query-capable Vulkan RT scene the plain --gpu --vulkan-rt-
            # shade path builds (see that branch's own comment below) --
            # reused sequentially across BOTH the light pass and the
            # camera pass every sample (n_light_paths_merge is always
            # >= n_pix, so one scene sized for it covers both).
            var use_vk_vcm = use_vulkan_rt_shade
            var interop_scene_vcm = UnsafePointer[UInt8, MutExternalOrigin].unsafe_dangling()
            var interop_rays_buf_vcm: Optional[DeviceBuffer[DType.float32]] = None
            var interop_results_buf_vcm: Optional[DeviceBuffer[DType.float32]] = None
            var mesh_material_idx_buf_vcm: Optional[DeviceBuffer[DType.uint8]] = None
            var mesh_al_idx_buf_vcm: Optional[DeviceBuffer[DType.uint8]] = None
            var n_meshes_vk_vcm = 0
            if use_vk_vcm:
                if psc[0].curve_count > Int32(0) or psc[0].sphere_count > Int32(0) or psc[0].instance_count > Int32(0):
                    print("WARNING: --vulkan-rt-shade requested but scene uses curves/spheres/instancing (unsupported) -- falling back to CUDA intersection")
                    use_vk_vcm = False
                else:
                    var n_light_paths_merge_vk = max(n_photons, n_pixels)
                    # Task #163 stage 5 perf fix (2026-07-13): the scene's
                    # ray/results buffers must also cover ALL _BDPT_MAX_VERTS
                    # diffuse-branch connect shadow-ray slots for every
                    # pixel traced in ONE dispatch per bounce, not n_pix at
                    # a time -- see bdpt.mojo's shadow-ray batching loop.
                    # vulkaninterop_rt_create_scene's max_rays purely drives
                    # buffer sizing (raysBytes/resultsBytes = max_rays*8*4,
                    # confirmed in vulkaninterop.cpp), no other backend
                    # change needed to raise it.
                    var max_rays_vk_vcm = max(n_light_paths_merge_vk, n_pixels * _BDPT_MAX_VERTS)
                    n_meshes_vk_vcm = Int(psc[0].mesh_count)
                    var vmeshes_vcm = alloc[TriangleMesh_C](max(n_meshes_vk_vcm, 1))
                    var point_counts_vcm = alloc[Int64](max(n_meshes_vk_vcm, 1))
                    var vidx_counts_vcm = alloc[Int64](max(n_meshes_vk_vcm, 1))
                    for i in range(n_meshes_vk_vcm):
                        vmeshes_vcm[i] = psc[0].meshes[i]
                        point_counts_vcm[i] = Int64(psc[0].mesh_n_verts[i])
                        vidx_counts_vcm[i] = Int64(psc[0].mesh_n_tris[i]) * 3
                    # VCM's Vulkan RT path doesn't support object instancing
                    # yet (its own primary/bounce interop -- vulkaninterop_
                    # rt_traverse_light_paths_gpu/_camera_ in bdpt.mojo -- is
                    # a separate wiring from the plain wavefront path's
                    # _gpu_bounce_kernels, not extended this session; see
                    # project_vulkan_rt_backend memory). The guard above
                    # already keeps instanced/curve/sphere scenes off this
                    # path entirely, so template_count/instance_count/
                    # n_curve_leaves are always 0 here.
                    var no_templates_vcm = UnsafePointer[Int64, MutExternalOrigin].unsafe_dangling()
                    var no_instances_vcm = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling()
                    var no_instance_tmpl_vcm = UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling()
                    var no_curve_aabbs_vcm = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling()
                    var no_curve_i32_vcm = UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling()
                    var no_curve_data_vcm = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling()
                    interop_scene_vcm = vulkaninterop_rt_create_scene(
                        vmeshes_vcm, Int64(n_meshes_vk_vcm), point_counts_vcm, vidx_counts_vcm,
                        Int64(0), no_templates_vcm, no_templates_vcm,
                        Int64(0), no_instances_vcm, no_instance_tmpl_vcm,
                        Int64(0), no_curve_aabbs_vcm,
                        no_curve_i32_vcm, no_curve_i32_vcm, no_curve_i32_vcm,
                        Int64(0), no_curve_data_vcm, no_curve_i32_vcm,
                        Int64(max_rays_vk_vcm))
                    vmeshes_vcm.free(); point_counts_vcm.free(); vidx_counts_vcm.free()
                    if Int(interop_scene_vcm) == 0:
                        print("WARNING: vulkaninterop_rt_create_scene FAILED -- falling back to CUDA intersection")
                        use_vk_vcm = False
                    else:
                        var raysPtr_vcm = vulkaninterop_rt_get_rays_ptr(interop_scene_vcm)
                        var resultsPtr_vcm = vulkaninterop_rt_get_results_ptr(interop_scene_vcm)
                        interop_rays_buf_vcm = DeviceBuffer[DType.float32](handle[].ctx, raysPtr_vcm, max_rays_vk_vcm * 8, owning=False)
                        interop_results_buf_vcm = DeviceBuffer[DType.float32](handle[].ctx, resultsPtr_vcm, max_rays_vk_vcm * 8, owning=False)

                        var light_info_vcm = _build_mesh_light_info(psc)
                        var mesh_material_idx_vcm = light_info_vcm[0]
                        var mesh_al_idx_vcm = light_info_vcm[1]
                        var n_meshes_alloc_vcm = max(n_meshes_vk_vcm, 1)

                        var mmi_buf_vcm = handle[].ctx.enqueue_create_buffer[DType.uint8](n_meshes_alloc_vcm * size_of[Int64]())
                        with mmi_buf_vcm.map_to_host() as h:
                            var dst = h.unsafe_ptr().bitcast[Int64]()
                            for i in range(n_meshes_vk_vcm):
                                dst[i] = mesh_material_idx_vcm[i]
                        mesh_material_idx_buf_vcm = mmi_buf_vcm^

                        var mai_buf_vcm = handle[].ctx.enqueue_create_buffer[DType.uint8](n_meshes_alloc_vcm * size_of[Int32]())
                        with mai_buf_vcm.map_to_host() as h2:
                            var dst2 = h2.unsafe_ptr().bitcast[Int32]()
                            for i in range(n_meshes_vk_vcm):
                                dst2[i] = mesh_al_idx_vcm[i]
                        mesh_al_idx_buf_vcm = mai_buf_vcm^

                        mesh_material_idx_vcm.free()
                        mesh_al_idx_vcm.free()

            ret = vcm_render_gpu_wavefront(
                handle, psc, sd[0], resolved_vcm_spp, n_photons, no_denoise, verbose,
                use_vk_vcm, interop_scene_vcm, interop_rays_buf_vcm, interop_results_buf_vcm,
                mesh_material_idx_buf_vcm, mesh_al_idx_buf_vcm, n_meshes_vk_vcm,
            )
            if use_vk_vcm:
                vulkaninterop_rt_destroy_scene(interop_scene_vcm)
        else:
            ret = vcm_render_gpu(handle, psc, sd[0], resolved_vcm_spp, n_photons, no_denoise, verbose)
        gpu_free_scene(handle)
        sd.free()
        mojo_parsed_free(psc)
        return ret
    elif use_gpu:
        var spp = Int(psc[0].samples_per_pixel)
        # World units spanned by one pixel per unit distance (for mip LOD):
        # 2*tan(fov/2)/height. fov is in degrees along the shorter axis.
        var px_scale = Float32(2.0) * tan(psc[0].camera_fov * Float32(3.14159265 / 360.0)) / Float32(Int(fh))
        var handle = _gpu_upload_scene(psc, sobol_matrices, n_pixels, spectral.coeffs, spectral.res, spectral.cie_x, spectral.cie_y, spectral.cie_z, spectral.d65)
        if not _is_real_ptr(handle):
            mojo_parsed_free(psc)
            return Int32(-1)

        # Task #163 stage 3: build the interop-AND-ray-query-capable Vulkan
        # RT scene once (if requested and the scene is within scope) and
        # reuse it across every bounce of every sample -- mirrors
        # _gpu_upload_scene's one-time-per-render setup. Unlike the
        # retired vulkanrt_build_scene-based version, this also uploads
        # mesh_material_idx/mesh_al_idx as GPU device buffers (the new
        # unpack kernel runs on the GPU, not a Mojo host loop) and wraps
        # the interop rays/results CUDA pointers as Mojo DeviceBuffers
        # once, reused unchanged across every bounce.
        var use_vk = use_vulkan_rt_shade and not use_restir
        if use_vulkan_rt_shade and use_restir:
            print("Note: --restir batch mode does not support --vulkan-rt-shade yet "
                  "(gpu_render_sample has no Vulkan RT interop path) -- using CUDA intersection.")
        var interop_scene = UnsafePointer[UInt8, MutExternalOrigin].unsafe_dangling()
        var interop_rays_buf_opt: Optional[DeviceBuffer[DType.float32]] = None
        var interop_results_buf_opt: Optional[DeviceBuffer[DType.float32]] = None
        var mesh_material_idx_buf_opt: Optional[DeviceBuffer[DType.uint8]] = None
        var mesh_al_idx_buf_opt: Optional[DeviceBuffer[DType.uint8]] = None
        var instance_base_mesh_buf_opt: Optional[DeviceBuffer[DType.uint8]] = None
        var n_meshes_vk = 0
        var max_rays_vk = Int64(n_pixels) * Int64(WAVEFRONT_BATCH)
        if use_vk:
            n_meshes_vk = Int(psc[0].mesh_count)
            var vmeshes = alloc[TriangleMesh_C](max(n_meshes_vk, 1))
            var point_counts = alloc[Int64](max(n_meshes_vk, 1))
            var vidx_counts = alloc[Int64](max(n_meshes_vk, 1))
            for i in range(n_meshes_vk):
                vmeshes[i] = psc[0].meshes[i]
                point_counts[i] = Int64(psc[0].mesh_n_verts[i])
                vidx_counts[i] = Int64(psc[0].mesh_n_tris[i]) * 3

            # Object instancing (project_vulkan_rt_backend memory's
            # "close the Vulkan RT gap" plan, item 1): template_mesh_
            # start/end mark which mesh-index ranges are template-only
            # (excluded from the ordinary BLAS-per-mesh loop, merged
            # instead into their template's own multi-geometry BLAS
            # inside vulkaninterop_rt_create_scene); instance_obj_to_
            # world/instance_template_idx place one TLAS instance per
            # ObjectInstance. Empty (n_templates=0) is byte-identical to
            # the pre-instancing call.
            var n_templates_vk = Int(psc[0].blas_count)
            var template_mesh_start_vk = alloc[Int64](max(n_templates_vk, 1))
            var template_mesh_end_vk   = alloc[Int64](max(n_templates_vk, 1))
            for t in range(n_templates_vk):
                template_mesh_start_vk[t] = Int64(psc[0].template_mesh_start[t])
                template_mesh_end_vk[t]   = Int64(psc[0].template_mesh_end[t])

            var n_instances_vk = Int(psc[0].instance_count)
            var instance_o2w_vk = alloc[Float32](max(n_instances_vk, 1) * 16)
            var instance_tmpl_idx_vk = alloc[Int32](max(n_instances_vk, 1))
            # Precompute each instance's real BASE mesh index (its
            # template's own mstart) here on the host so the GPU decode
            # kernel only needs one array lookup + geometryIndex add --
            # see vulkaninterop_unpack_results_kernel (gpu.mojo).
            var instance_base_mesh_host = alloc[Int32](max(n_instances_vk, 1))
            for k in range(n_instances_vk):
                var inst = psc[0].instances[k]
                for ci in range(16):
                    instance_o2w_vk[k * 16 + ci] = inst.objToWorld[ci]
                instance_tmpl_idx_vk[k] = Int32(inst.blasIdx)
                instance_base_mesh_host[k] = psc[0].template_mesh_start[Int(inst.blasIdx)]

            # Curves: NOT tessellated. Scan the ordinary top-level prim_ids
            # for type==5 (curve) leaf entries -- each becomes one
            # procedural AABB (the union of its piece-group's bounds, same
            # computation finalize_scene already does for the CPU BVH's own
            # leaf bounds). Unlike earlier designs, intersect_batch.comp
            # resolves curve hits itself (real narrow-phase test + commit --
            # see vulkaninterop_rt_create_scene's docstring), so per leaf we
            # upload exactly what it needs to test a candidate and report a
            # hit with no further lookups: curve_idx (p.id1), piece_info
            # (p.id2, already packed first_piece*8+piece_count), and
            # mat_idx (p.materialIndex, which already resolves curve-area-
            # light emissive material selection -- see finalize_scene,
            # pbrt_parser.mojo). curve_data/curve_n_pieces hold each curve's
            # own control points/widths/piece count once (independent of
            # how many leaves reference it).
            var n_curve_leaves_vk = 0
            for i in range(Int(psc[0].prim_count)):
                if psc[0].prim_ids[i].type == Int8(5):
                    n_curve_leaves_vk += 1
            var curve_leaf_aabbs_vk = alloc[Float32](max(n_curve_leaves_vk, 1) * 6)
            var curve_leaf_curve_idx_vk = alloc[Int32](max(n_curve_leaves_vk, 1))
            var curve_leaf_piece_info_vk = alloc[Int32](max(n_curve_leaves_vk, 1))
            var curve_leaf_mat_idx_vk = alloc[Int32](max(n_curve_leaves_vk, 1))
            var curve_leaf_write = 0
            for i in range(Int(psc[0].prim_count)):
                var p = psc[0].prim_ids[i]
                if p.type != Int8(5):
                    continue
                var curve = psc[0].curves[Int(p.id1)]
                var first_piece = Int(p.id2) // 8
                var piece_count = Int(p.id2) % 8
                var (xmin, ymin, zmin, xmax, ymax, zmax) = curve_piece_bounds(curve, first_piece)
                for piece in range(first_piece + 1, first_piece + piece_count):
                    var (pxmin, pymin, pzmin, pxmax, pymax, pzmax) = curve_piece_bounds(curve, piece)
                    xmin = min(xmin, pxmin); ymin = min(ymin, pymin); zmin = min(zmin, pzmin)
                    xmax = max(xmax, pxmax); ymax = max(ymax, pymax); zmax = max(zmax, pzmax)
                var b = curve_leaf_write * 6
                curve_leaf_aabbs_vk[b+0] = xmin; curve_leaf_aabbs_vk[b+1] = ymin; curve_leaf_aabbs_vk[b+2] = zmin
                curve_leaf_aabbs_vk[b+3] = xmax; curve_leaf_aabbs_vk[b+4] = ymax; curve_leaf_aabbs_vk[b+5] = zmax
                curve_leaf_curve_idx_vk[curve_leaf_write] = Int32(p.id1)
                curve_leaf_piece_info_vk[curve_leaf_write] = Int32(p.id2)
                curve_leaf_mat_idx_vk[curve_leaf_write] = Int32(p.materialIndex)
                curve_leaf_write += 1

            var n_curves_vk = Int(psc[0].curve_count)
            var curve_data_vk = alloc[Float32](max(n_curves_vk, 1) * 14)
            var curve_n_pieces_vk = alloc[Int32](max(n_curves_vk, 1))
            for ci in range(n_curves_vk):
                var c = psc[0].curves[ci]
                var cb = ci * 14
                curve_data_vk[cb+0] = c.cp0.x; curve_data_vk[cb+1] = c.cp0.y; curve_data_vk[cb+2] = c.cp0.z
                curve_data_vk[cb+3] = c.cp1.x; curve_data_vk[cb+4] = c.cp1.y; curve_data_vk[cb+5] = c.cp1.z
                curve_data_vk[cb+6] = c.cp2.x; curve_data_vk[cb+7] = c.cp2.y; curve_data_vk[cb+8] = c.cp2.z
                curve_data_vk[cb+9] = c.cp3.x; curve_data_vk[cb+10] = c.cp3.y; curve_data_vk[cb+11] = c.cp3.z
                curve_data_vk[cb+12] = c.width0; curve_data_vk[cb+13] = c.width1
                curve_n_pieces_vk[ci] = c.n_pieces

            interop_scene = vulkaninterop_rt_create_scene(
                vmeshes, Int64(n_meshes_vk), point_counts, vidx_counts,
                Int64(n_templates_vk), template_mesh_start_vk, template_mesh_end_vk,
                Int64(n_instances_vk), instance_o2w_vk, instance_tmpl_idx_vk,
                Int64(n_curve_leaves_vk), curve_leaf_aabbs_vk,
                curve_leaf_curve_idx_vk, curve_leaf_piece_info_vk, curve_leaf_mat_idx_vk,
                Int64(n_curves_vk), curve_data_vk, curve_n_pieces_vk,
                max_rays_vk)
            vmeshes.free(); point_counts.free(); vidx_counts.free()
            template_mesh_start_vk.free(); template_mesh_end_vk.free()
            instance_o2w_vk.free(); instance_tmpl_idx_vk.free()
            curve_leaf_aabbs_vk.free()
            curve_leaf_curve_idx_vk.free(); curve_leaf_piece_info_vk.free(); curve_leaf_mat_idx_vk.free()
            curve_data_vk.free(); curve_n_pieces_vk.free()
            if Int(interop_scene) == 0:
                print("WARNING: vulkaninterop_rt_create_scene FAILED -- falling back to CUDA intersection")
                use_vk = False
                instance_base_mesh_host.free()
            else:
                var raysPtr = vulkaninterop_rt_get_rays_ptr(interop_scene)
                var resultsPtr = vulkaninterop_rt_get_results_ptr(interop_scene)
                interop_rays_buf_opt = DeviceBuffer[DType.float32](handle[].ctx, raysPtr, Int(max_rays_vk) * 8, owning=False)
                interop_results_buf_opt = DeviceBuffer[DType.float32](handle[].ctx, resultsPtr, Int(max_rays_vk) * 8, owning=False)

                var light_info = _build_mesh_light_info(psc)
                var mesh_material_idx = light_info[0]
                var mesh_al_idx = light_info[1]
                var n_meshes_alloc = max(n_meshes_vk, 1)

                var mmi_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_meshes_alloc * size_of[Int64]())
                with mmi_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Int64]()
                    for i in range(n_meshes_vk):
                        dst[i] = mesh_material_idx[i]
                mesh_material_idx_buf_opt = mmi_buf^

                var mai_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_meshes_alloc * size_of[Int32]())
                with mai_buf.map_to_host() as h2:
                    var dst2 = h2.unsafe_ptr().bitcast[Int32]()
                    for i in range(n_meshes_vk):
                        dst2[i] = mesh_al_idx[i]
                mesh_al_idx_buf_opt = mai_buf^

                mesh_material_idx.free()
                mesh_al_idx.free()

                if n_instances_vk > 0:
                    var ibm_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_instances_vk * size_of[Int32]())
                    with ibm_buf.map_to_host() as h3:
                        var dst3 = h3.unsafe_ptr().bitcast[Int32]()
                        for k in range(n_instances_vk):
                            dst3[k] = instance_base_mesh_host[k]
                    instance_base_mesh_buf_opt = ibm_buf^
                instance_base_mesh_host.free()

        var hash_bits = UInt64(mix_bits_u64(UInt64(0)))
        var seed_dim0 = UInt32(hash_bits & UInt64(0xFFFFFFFF))
        var seed_dim1 = UInt32(0)
        if use_restir:
            # Architecture change (docs/A2_restir_migration_plan.md, replaces
            # the Phase 3.1 wavefront-persistence attempt, reverted): rather
            # than adapt reservoir reuse to WAVEFRONT_BATCH concurrent
            # samples/pixel/dispatch -- which needs multiple threads per
            # pixel to touch shared reservoir state and, when tried, produced
            # a real, unexplained divergence with no literature precedent to
            # check the design against -- batch --restir now renders 1
            # sample/pixel/dispatch via gpu_render_sample, exactly like
            # --interactive-frames already does (proven correct: no bias,
            # real wins there). This inherits that path's full temporal+
            # spatial reuse for free, at the cost of WAVEFRONT_BATCH-wide
            # batching's throughput (more, smaller kernel launches).
            print("Note: --gpu --restir batch mode renders 1 sample/pixel per "
                  "dispatch (like --interactive-frames) for full reservoir "
                  "reuse, trading wavefront-batching throughput for it.")
            gpu_gen_aux_buffers(handle, psc[0].camera_to_world, Int64(n_pixels))
            gpu_clear_restir(handle, Int64(n_pixels))
        gpu_clear_film(handle, Int64(n_pixels))
        var t0_gpu = perf_counter_ns()
        if use_restir:
            for si in range(spp):
                gpu_render_sample(
                    handle,
                    psc[0].camera_to_world,
                    Int32(si), psc[0].log2_spp, psc[0].n_base4_digits,
                    seed_dim0, seed_dim1,
                    UInt32(psc[0].rng_seed & UInt64(0xFFFFFFFF)),
                    UInt32(psc[0].rng_seed >> UInt64(32)),
                    Int64(n_pixels), psc[0].max_depth,
                    px_scale,
                    use_restir=use_restir, frame_index=si,
                )
                var elapsed = Float64(perf_counter_ns() - t0_gpu) / 1.0e9
                print(progress_str(si + 1, spp, elapsed, "spp"), end="\r")
        else:
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
                    use_vk, interop_scene, interop_rays_buf_opt, interop_results_buf_opt,
                    mesh_material_idx_buf_opt, mesh_al_idx_buf_opt, n_meshes_vk,
                    instance_base_mesh_buf_opt,
                )
                si += actual_batch
                var elapsed = Float64(perf_counter_ns() - t0_gpu) / 1.0e9
                print(progress_str(si, spp, elapsed, "spp"), end="\r")
        var gpu_total_s = Float64(perf_counter_ns() - t0_gpu) / 1.0e9
        print("Rendering: " + String(spp) + " / " + String(spp)
            + " spp (100.0%) | Done: " + fmt_time(gpu_total_s) + "                ")
        if use_vk:
            vulkaninterop_rt_destroy_scene(interop_scene)
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
        _ = write_image_cropped(denoised_gpu.unsafe_ptr(), fw, fh, crop_x0px, crop_y0px, crop_w, crop_h,
                                 psc[0].film_filename, Int32(32), Int32(32))
        var albedo_name_buf = alloc[UInt8](11)
        var albedo_name_str = "albedo.exr"
        var anp = albedo_name_str.unsafe_ptr()
        for i in range(10): albedo_name_buf[i] = anp[i]
        albedo_name_buf[10] = UInt8(0)
        _ = write_image_cropped(albedo_gpu.unsafe_ptr(), fw, fh, crop_x0px, crop_y0px, crop_w, crop_h,
                                 albedo_name_buf.unsafe_origin_cast[MutExternalOrigin](), Int32(32), Int32(32))
        albedo_name_buf.free()
        # denoised_gpu, albedo_gpu, and results freed automatically
        mojo_parsed_free(psc)
        return Int32(0)
    elif psc[0].prim_count == 0:
        print("Warning: scene has no geometry, skipping render")
        mojo_parsed_free(psc)
        return Int32(0)
    elif use_vcm:
        var sd = mojo_parsed_scene_descriptor(psc, spectral)
        var n_photons = _resolve_vcm_photons(vcm_photons, n_pixels)
        var resolved_vcm_spp = _resolve_vcm_spp(vcm_spp, psc[0].samples_per_pixel)
        var ret = vcm_render(psc, sd[0], resolved_vcm_spp, n_photons, no_denoise, verbose)
        sd.free()
        mojo_parsed_free(psc)
        return ret
    elif use_sppm:
        var sd = mojo_parsed_scene_descriptor(psc, spectral)
        var resolved = _resolve_sppm_params(psc, sppm_photons, sppm_radius)
        var ret = sppm_render(
            psc, sd[0],
            Int(sppm_passes), Int(resolved[0]), resolved[1],
            no_denoise, verbose,
        )
        sd.free()
        mojo_parsed_free(psc)
        return ret
    else:
        var zero = TileResult_C(
            estimate=RGB(Float32(0)),
            albedo=RGB(Float32(0)),
            filterWeight=Float32(0), pixelX=Int32(0), pixelY=Int32(0))
        for _ in range(n_pixels):
            results.append(zero)
        var sd = mojo_parsed_scene_descriptor(psc, spectral)

        if use_guide and psc[0].bvh_node_count > Int32(0):
            # ── N-iteration guided rendering (adaptive SD-tree) ───────────────
            # Build an empty SD-tree from the BVH root AABB (guide.mojo). Each
            # iteration: clone the current tree into 16 empty per-tile-group
            # shards (avoids cross-core cache ping-pong on shared energy, same
            # reasoning as the old 2-batch design), render reading from the
            # PREVIOUS iteration's cumulative tree (iteration 0 reads null --
            # BSDF-only, tree is empty anyway), fold the shards' freshly
            # recorded energy into the tree, then -- except after the last
            # iteration -- refine (grow) the tree's structure for the next
            # iteration to read from. Equal spp per iteration is a
            # simplification vs. Müller's progressive-doubling schedule; both
            # are unbiased, doubling mainly reduces the final combined
            # estimator's variance, an optimization not attempted here.
            var root = psc[0].bvh_nodes[0]
            comptime N_GUIDE_THREADS: Int = 16
            comptime N_ITERATIONS: Int = 4
            var spp = psc[0].samples_per_pixel
            var n_iters = min(Int(spp), N_ITERATIONS)
            var base_spp = spp // Int32(n_iters)
            var tree = guide_create(Bounds3f(root.min, root.max))
            var write_guides = alloc[GuideGrid](N_GUIDE_THREADS)
            print("Path guiding: " + String(n_iters) + " iterations x ~" + String(base_spp)
                + " spp, adaptive SD-tree, 16 private shards")
            var t0_g = perf_counter_ns()
            var offset = Int32(0)
            for it in range(n_iters):
                var iter_spp = base_spp
                if it == n_iters - 1:
                    iter_spp = spp - offset  # absorb any remainder into the last iteration
                for gi in range(N_GUIDE_THREADS):
                    write_guides[gi] = guide_clone_empty(tree)
                var sp_iter = TileSamplerParams_C(
                    sobolMatrices=sobol_matrices,
                    rngSeed=psc[0].rng_seed,
                    sobolSeed=Int32(0),
                    log2SamplesPerPixel=psc[0].log2_spp,
                    nBase4Digits=psc[0].n_base4_digits,
                    samplesPerPixel=iter_spp,
                    filterSigma=psc[0].filter_sigma,
                    filterSupportX=psc[0].filter_support_x,
                    filterSupportY=psc[0].filter_support_y,
                    filterNormX=psc[0].filter_norm_x,
                    filterNormY=psc[0].filter_norm_y,
                    filterWeight=psc[0].filter_weight,
                    filterType=psc[0].filter_type,
                    sampleIndexOffset=offset,
                )
                var sp_iter_ptr = OwnedPointer[TileSamplerParams_C](sp_iter)
                var guide_read = null_guide() if it == 0 else tree
                if it == 0:
                    # First iteration writes straight into `results` (like the
                    # old pilot pass) -- no accumulation add needed.
                    render_all_tiles(
                        psc[0].raster_to_camera, psc[0].camera_to_world,
                        Int32(0), Int32(0), fw, fh,
                        Int32(32), Int32(32),
                        sp_iter_ptr.unsafe_ptr(), sd, results.unsafe_ptr(),
                        psc[0].max_depth, False,
                        guide_read, write_guides, N_GUIDE_THREADS)
                else:
                    var iter_buf = List[TileResult_C](capacity=n_pixels)
                    for _ in range(n_pixels): iter_buf.append(zero)
                    render_all_tiles(
                        psc[0].raster_to_camera, psc[0].camera_to_world,
                        Int32(0), Int32(0), fw, fh,
                        Int32(32), Int32(32),
                        sp_iter_ptr.unsafe_ptr(), sd, iter_buf.unsafe_ptr(),
                        psc[0].max_depth, False,
                        guide_read, write_guides, N_GUIDE_THREADS)
                    for i in range(n_pixels):
                        var p = results.unsafe_ptr()[i]
                        var m = iter_buf.unsafe_ptr()[i]
                        results.unsafe_ptr()[i] = TileResult_C(
                            p.estimate + m.estimate,
                            p.albedo   + m.albedo,
                            p.filterWeight + m.filterWeight,
                            m.pixelX, m.pixelY)
                offset += iter_spp
                # render_all_tiles already merged shards [1..N-1] into [0].
                guide_merge(tree, write_guides[0])
                for gi in range(N_GUIDE_THREADS):
                    guide_free(write_guides[gi])
                if it < n_iters - 1:
                    var refined = guide_refine(tree)
                    tree = refined
                var n_active = 0
                for ci in range(Int(tree.n_snodes)):
                    if guide_cell_has_data(tree, ci):
                        n_active += 1
                print("Path guiding: iter " + String(it) + " done, " + String(n_active) + "/"
                    + String(tree.n_snodes) + " active spatial leaves, " + String(tree.n_dnodes)
                    + " directional nodes")

            var total_g = Float64(perf_counter_ns() - t0_g) / 1.0e9
            print("Path guiding done in " + fmt_time(total_g) + "                ")
            guide_free(tree)
            write_guides.free()
        else:
            # ── Standard single-call rendering ───────────────────────────────
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
                sampleIndexOffset=Int32(0),
            )
            var sp_ptr = OwnedPointer[TileSamplerParams_C](sp)
            render_all_tiles(
                psc[0].raster_to_camera, psc[0].camera_to_world,
                Int32(0), Int32(0), fw, fh,
                Int32(32), Int32(32),
                sp_ptr.unsafe_ptr(), sd, results.unsafe_ptr(), psc[0].max_depth,
                quiet=False, guide_read=null_guide(),
                write_guides=UnsafePointer[GuideGrid, MutExternalOrigin].unsafe_dangling(), n_write_guides=0,
                use_restir=use_restir, use_gi=use_restir and use_restir_gi)
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
                    Int32(5), Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05))
        _ = write_image_cropped(denoised.unsafe_ptr(), fw, fh, crop_x0px, crop_y0px, crop_w, crop_h,
                                 psc[0].film_filename, Int32(32), Int32(32))
        var albedo_name_buf = alloc[UInt8](11)
        var albedo_name_str = "albedo.exr"
        var anp2 = albedo_name_str.unsafe_ptr()
        for i in range(10): albedo_name_buf[i] = anp2[i]
        albedo_name_buf[10] = UInt8(0)
        _ = write_image_cropped(albedo.unsafe_ptr(), fw, fh, crop_x0px, crop_y0px, crop_w, crop_h,
                                 albedo_name_buf.unsafe_origin_cast[MutExternalOrigin](), Int32(32), Int32(32))
        albedo_name_buf.free()
        # beauty, albedo, denoised, normals, dept freed automatically
    mojo_parsed_free(psc)
    return Int32(0)


def render_interactive(
    path: UnsafePointer[UInt8, MutExternalOrigin],
    sobol: UnsafePointer[UInt32, MutExternalOrigin],
    use_gpu: Bool,
    spectral: SpectralHandle = null_spectral_handle(),
    fullscreen: Bool = False,
    override_w: Int32 = Int32(0), override_h: Int32 = Int32(0),
    spp_override: Int32 = Int32(0),
    seed_override: Int64 = Int64(-1),
    verbose: Bool = False,
    use_restir: Bool = False,
    # Phase 4: ReSTIR GI, CPU only (matches use_restir's own use_gpu scope
    # here), requires use_restir (no effect alone -- see parse_and_render's
    # own warning for the batch path). No temporal/spatial reservoir reuse
    # here -- see the comment above restir_buf_a/b's declaration for why
    # (a real energy-bias bug found and reverted). use_gi still enables
    # per-frame generate+resolve with no reuse, same as batch mode.
    use_restir_gi: Bool = False,
    # Phase 6: ReSTIR SMS temporal reuse for glass-caustic MNEE probing
    # (shading.mojo's sms_temporal_step), CPU only, INDEPENDENT of
    # use_restir (unlike use_restir_gi, which requires it) -- see
    # _shade_diffuse_nee's own docstring for the known use_restir-
    # combination gap (bounce 0 goes through di_temporal_step instead of
    # _nee_area_lights when both are active, so SMS-ReSTIR's bounce-0
    # reuse doesn't run in that combination yet). Real, working
    # temporal-only reservoir reuse (no spatial yet) -- see
    # project_sms_restir_phase6 memory.
    use_sms_restir: Bool = False,
    headless_frames: Int32 = Int32(0),
):
    if use_gpu and not gpu_available():
        print("No GPU available — compile with --target-accelerator sm_86 or similar")
        return
    if use_restir_gi and not use_restir:
        print("--restir-gi: no effect without --restir")
    if use_restir_gi and use_gpu:
        print("--restir-gi: CPU only so far, no effect combined with --gpu")
    if use_sms_restir and use_gpu:
        print("--sms-restir: CPU only so far, no effect combined with --gpu")

    var psc = mojo_parse_scene_any(path, verbose)
    if Int(psc) == 0:
        print("Failed to parse scene")
        return
    if override_w > 0 and override_h > 0:
        resize_film(psc, override_w, override_h)

    mojo_apply_overrides(psc, spp_override, Int32(0), Int32(0), seed_override)

    var fw = psc[0].film_w
    var fh = psc[0].film_h
    var n_pixels = Int(fw) * Int(fh)

    var handle = UnsafePointer[GpuSceneHandle, MutExternalOrigin].unsafe_dangling()
    if use_gpu:
        handle = _gpu_upload_scene(psc, sobol, n_pixels, spectral.coeffs, spectral.res, spectral.cie_x, spectral.cie_y, spectral.cie_z, spectral.d65)
        if not _is_real_ptr(handle):
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
    var title_buf = alloc[UInt8](title_len + 1)
    var ts = title_str.unsafe_ptr()
    for i in range(title_len):
        title_buf[i] = ts[i]
    title_buf[title_len] = UInt8(0)
    var v = viewer_create(fw, fh, title_buf, Int32(1) if fullscreen else Int32(0))
    title_buf.free()
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
            estimate=RGB(Float32(0)),
            albedo=RGB(Float32(0)),
            filterWeight=Float32(0), pixelX=Int32(0), pixelY=Int32(0)))
    for _ in range(n_pixels * 3): beauty.append(Float32(0)); albedo.append(Float32(0)); denoised.append(Float32(0))
    var frame_count = 0

    # Mode-specific buffers — dangling until allocated below
    var sd           = UnsafePointer[SceneDescriptor2_C, MutExternalOrigin].unsafe_dangling()
    # Phase 2.3+2.5 (docs/A2_restir_migration_plan.md): two persistent
    # DIReservoir buffers per pixel, ping-ponged each frame -- CPU-only
    # (--restir has no GPU wiring yet, see restir_di.mojo's header).
    # Double-buffered (not read+written in place) because render_all_tiles
    # parallelizes per-tile: Phase 2.5's spatial reuse reads NEIGHBOR
    # pixels' reservoirs, which may live in a different, concurrently
    # running tile. Reading only from `restir_read` (strictly the PREVIOUS
    # frame's fully-resolved buffer, never touched again until it becomes
    # next frame's write target) and writing only to `restir_write` (each
    # pixel written by exactly one thread, at 1 spp/frame) makes both
    # race-free without any locking. Swapped after each frame completes,
    # below.
    var restir_buf_a = UnsafePointer[DIReservoir, MutExternalOrigin].unsafe_dangling()
    var restir_buf_b = UnsafePointer[DIReservoir, MutExternalOrigin].unsafe_dangling()
    var restir_read  = UnsafePointer[DIReservoir, MutExternalOrigin].unsafe_dangling()
    var restir_write = UnsafePointer[DIReservoir, MutExternalOrigin].unsafe_dangling()
    var gi_buf_a = UnsafePointer[GIReservoir, MutExternalOrigin].unsafe_dangling()
    var gi_buf_b = UnsafePointer[GIReservoir, MutExternalOrigin].unsafe_dangling()
    var gi_read  = UnsafePointer[GIReservoir, MutExternalOrigin].unsafe_dangling()
    var gi_write = UnsafePointer[GIReservoir, MutExternalOrigin].unsafe_dangling()
    # Phase 6: ping-ponged SMSReservoir buffers, same race-free scheme as
    # restir_buf_a/b and gi_buf_a/b above -- SMS_MAX_FINALIZED_WEIGHT
    # (shading.mojo) was applied proactively from the start (not
    # discovered via a live bug this time, unlike DI/GI's own history),
    # so no separate bug-fix narrative applies here.
    var sms_buf_a = UnsafePointer[SMSReservoir, MutExternalOrigin].unsafe_dangling()
    var sms_buf_b = UnsafePointer[SMSReservoir, MutExternalOrigin].unsafe_dangling()
    var sms_read  = UnsafePointer[SMSReservoir, MutExternalOrigin].unsafe_dangling()
    var sms_write = UnsafePointer[SMSReservoir, MutExternalOrigin].unsafe_dangling()
    # Phase 4: ping-ponged GIReservoir buffers, same scheme as
    # restir_buf_a/b above (race-free for the same reason: read only from
    # `gi_read`, write only to `gi_write`, swapped after each frame).
    # History: a first attempt at real temporal/spatial reuse here showed
    # energy growing without bound over successive frames on cornell-box.
    # Root-caused to two real bugs in restir_gi.mojo (both fixed, see that
    # file's own history/comments): (1) gi_target_pdf wrongly included a
    # cos_x2/dist_sq geometric falloff that gi_resolve's own contribution
    # never has (`lo` is already outgoing radiance toward x1, not an
    # area-measure quantity needing that conversion -- the mismatch is
    # invisible for a single unstreamed candidate but explodes once real
    # reuse combines candidates with different cos_x2/dist_sq); (2) even
    # after that fix, reservoir_combine's own weight formula (shared with
    # DI, reservoir.mojo) feeds a source reservoir's finalized state.w back
    # into future combines, so one rare near-degenerate-geometry outlier
    # compounds into unbounded growth over enough frames -- GI_MAX_FINALIZED_WEIGHT
    # (restir_gi.mojo) bounds this defensively. See project_restir_migration
    # memory for the full debugging story, including a separate,
    # pre-existing bug discovered along the way: plain --restir (no GI at
    # all) ALSO shows unbounded growth at high frame counts (~128+) on this
    # same scene -- a DI-only issue in reservoir.mojo/restir_di.mojo,
    # unrelated to and not fixed by this session's GI work.
    var accum        = List[Float32]()
    var albedo_acc   = List[Float32]()
    var normals_int  = List[Float32]()
    var depth_int    = List[Float32]()
    # Phase 0.3 G-buffer extension (normals_int/depth_int above cover the
    # denoiser's own two). Phase 2.5 needs both: material ID for the
    # neighbour-rejection test, world position for the Z normalization,
    # which re-evaluates the target function at a NEIGHBOUR's shading point
    # and so needs that point. Nothing else in render_interactive reads them.
    var material_id_int = List[Int32]()
    var world_pos_int   = List[Float32]()
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
        sampleIndexOffset=Int32(0),
    ))

    if use_gpu:
        gpu_clear_film(handle, Int64(n_pixels))
        if use_restir:
            gpu_clear_restir(handle, Int64(n_pixels))
        gpu_gen_aux_buffers(handle, psc[0].camera_to_world, Int64(n_pixels))
    else:
        sd = mojo_parsed_scene_descriptor(psc, spectral)
        for _ in range(n_pixels * 3):
            accum.append(Float32(0))
            albedo_acc.append(Float32(0))
            normals_int.append(Float32(0))
        for _ in range(n_pixels):
            depth_int.append(Float32(0))
        # World position + material id are the half of the G-buffer that only
        # a SPATIAL-reuse driver needs, and DI is no longer the only one --
        # GI and SMS both take neighbour taps now. Sizing these under
        # `use_restir` alone left the other two holding a dangling pointer,
        # which their own null-safety checks then read as "no G-buffer" and
        # skipped spatial reuse entirely, silently.
        if use_restir or use_restir_gi or use_sms_restir:
            for _ in range(n_pixels):
                material_id_int.append(Int32(-1))
            for _ in range(n_pixels * 3):
                world_pos_int.append(Float32(0))
        if use_restir:
            restir_buf_a = alloc[DIReservoir](n_pixels)
            restir_buf_b = alloc[DIReservoir](n_pixels)
            for i in range(n_pixels):
                restir_buf_a[i] = di_reservoir_init()
                restir_buf_b[i] = di_reservoir_init()
            restir_read = restir_buf_a
            restir_write = restir_buf_b
            if use_restir_gi:
                gi_buf_a = alloc[GIReservoir](n_pixels)
                gi_buf_b = alloc[GIReservoir](n_pixels)
                for i in range(n_pixels):
                    gi_buf_a[i] = gi_reservoir_init()
                    gi_buf_b[i] = gi_reservoir_init()
                gi_read = gi_buf_a
                gi_write = gi_buf_b
        if use_sms_restir:
            # Independent of use_restir (unlike use_restir_gi above) --
            # SMS-ReSTIR only ever touches _nee_area_lights' own glass-
            # probing branch, not ReSTIR DI's reservoir path.
            sms_buf_a = alloc[SMSReservoir](n_pixels)
            sms_buf_b = alloc[SMSReservoir](n_pixels)
            for i in range(n_pixels):
                sms_buf_a[i] = sms_reservoir_init()
                sms_buf_b[i] = sms_reservoir_init()
            sms_read = sms_buf_a
            sms_write = sms_buf_b

    var zero = TileResult_C(
        estimate=RGB(Float32(0)),
        albedo=RGB(Float32(0)),
        filterWeight=Float32(0), pixelX=Int32(0), pixelY=Int32(0))

    var headless = headless_frames > Int32(0)
    while (not headless and not viewer_should_close(v)) or (headless and frame_count < Int(headless_frames)):
        if not headless:
            viewer_poll_events(v)
            viewer_get_camera_state(v, result=cam_buf.unsafe_ptr())
            if cam_buf[].cameraChanged != Int32(0):
                frame_count = 0
                build_camera_to_world(cam_buf.unsafe_ptr(), c2w_buf.unsafe_ptr())
                if use_gpu:
                    gpu_clear_film(handle, Int64(n_pixels))
                    if use_restir:
                        # Same invalidation rule as the CPU path: a stale
                        # reservoir after a camera move describes a different
                        # shading point (identity reprojection).
                        gpu_clear_restir(handle, Int64(n_pixels))
                    gpu_gen_aux_buffers(handle, c2w_buf.unsafe_ptr(), Int64(n_pixels))
                else:
                    for i in range(n_pixels * 3):
                        accum[i]      = Float32(0)
                        albedo_acc[i] = Float32(0)
                    if use_restir:
                        # Reservoir invalidation on camera move (fact #2,
                        # docs/A2_restir_migration_plan.md) -- identity
                        # reprojection is only valid for a static camera; a
                        # stale reservoir after a move would reuse a light
                        # candidate resampled for a different view. Reset
                        # both buffers -- which one is "read" vs "write" is
                        # irrelevant right after both are identically empty.
                        for i in range(n_pixels):
                            restir_buf_a[i] = di_reservoir_init()
                            restir_buf_b[i] = di_reservoir_init()
                        if use_restir_gi:
                            for i in range(n_pixels):
                                gi_buf_a[i] = gi_reservoir_init()
                                gi_buf_b[i] = gi_reservoir_init()
                    if use_sms_restir:
                        # Same identity-reprojection invalidation rule,
                        # independent of use_restir.
                        for i in range(n_pixels):
                            sms_buf_a[i] = sms_reservoir_init()
                            sms_buf_b[i] = sms_reservoir_init()
        # headless: camera is never polled, so it never "changes" -- every
        # frame accumulates onto the same static view, exactly the
        # steady-state case temporal reuse (Phase 2.3) needs to be verified
        # against via ordinary render-diffing.

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
                use_restir=use_restir, frame_index=frame_count,
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
                sampleIndexOffset=Int32(0),
            )
            for i in range(n_pixels):
                results[i] = zero
            # Must run BEFORE render_all_tiles below, not after: Phase 2.5's
            # spatial reuse (shading.mojo's di_temporal_step) reads
            # normals_int/depth_int/material_id_int during THIS frame's
            # shading, so on frame 0 (or right after a camera move) they
            # need to already be populated, not still zero-initialized from
            # this function's own List() setup above.
            if frame_count == 0:
                # The FULL G-buffer (world position + material id, not just
                # normal/depth) is what any spatial-reuse driver needs, and
                # there are three of them now -- DI, GI and SMS. Gating it on
                # `use_restir` alone left --sms-restir with a zero depth
                # buffer, which its neighbour test reads as "degenerate" and
                # rejects, so spatial reuse silently did nothing.
                if use_restir or use_restir_gi or use_sms_restir:
                    render_aux_buffers(
                        psc[0].raster_to_camera, c2w_buf.unsafe_ptr(),
                        Int32(0), Int32(0), fw, fh, sd,
                        normals_int.unsafe_ptr(), depth_int.unsafe_ptr(),
                        world_pos_out=world_pos_int.unsafe_ptr(),
                        material_id_out=material_id_int.unsafe_ptr())
                else:
                    render_aux_buffers(
                        psc[0].raster_to_camera, c2w_buf.unsafe_ptr(),
                        Int32(0), Int32(0), fw, fh, sd,
                        normals_int.unsafe_ptr(), depth_int.unsafe_ptr())
            var restir_io = reservoir_io_null()
            if use_restir:
                restir_io = ReservoirIO(
                    read=restir_read, write=restir_write,
                    gbuf_normal=normals_int.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
                    gbuf_depth=depth_int.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
                    gbuf_material_id=material_id_int.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
                    gbuf_world_pos=world_pos_int.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
                    frame_w=fw, frame_h=fh,
                )
            var gi_io = gi_reservoir_io_null()
            if use_restir_gi:
                gi_io = GIReservoirIO(
                    read=gi_read, write=gi_write,
                    gbuf_normal=normals_int.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
                    gbuf_depth=depth_int.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
                    gbuf_material_id=material_id_int.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
                    gbuf_world_pos=world_pos_int.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
                    frame_w=fw, frame_h=fh,
                )
            var sms_io = sms_reservoir_io_null()
            if use_sms_restir:
                sms_io = SMSReservoirIO(
                    read=sms_read, write=sms_write,
                    gbuf_normal=normals_int.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
                    gbuf_depth=depth_int.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
                    gbuf_material_id=material_id_int.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
                    gbuf_world_pos=world_pos_int.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](),
                    frame_w=fw, frame_h=fh,
                )
            render_all_tiles(
                psc[0].raster_to_camera, c2w_buf.unsafe_ptr(),
                Int32(0), Int32(0), fw, fh,
                Int32(32), Int32(32),
                sp_int.unsafe_ptr(), sd, results.unsafe_ptr(), psc[0].max_depth, True,
                guide_read=null_guide(), write_guides=UnsafePointer[GuideGrid, MutExternalOrigin].unsafe_dangling(),
                n_write_guides=0, use_restir=use_restir, frame_w=fw, restir_io=restir_io,
                use_gi=use_restir_gi, gi_io=gi_io,
                use_sms_restir=use_sms_restir, sms_io=sms_io)
            if use_restir_gi:
                var gi_tmp = gi_read
                gi_read = gi_write
                gi_write = gi_tmp
            if use_sms_restir:
                # render_all_tiles above is synchronous, so sms_write is now
                # fully resolved for every pixel -- safe to become next
                # frame's read buffer, same reasoning as restir_read/write
                # below.
                var sms_tmp = sms_read
                sms_read = sms_write
                sms_write = sms_tmp
            if use_restir:
                # render_all_tiles above is synchronous (parallelize joins
                # before returning), so restir_write is now fully resolved
                # for every pixel -- safe to become next frame's read buffer.
                var tmp = restir_read
                restir_read = restir_write
                restir_write = tmp
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
                    Int32(5), Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05))

        viewer_update_framebuffer(v, denoised.unsafe_ptr(), fw, fh)

    if headless:
        _ = write_image_cropped(denoised.unsafe_ptr(), fw, fh, Int32(0), Int32(0), fw, fh,
                                 psc[0].film_filename, Int32(32), Int32(32))

    # results, beauty, albedo, denoised, c2w_buf, cam_buf, sp_int freed automatically
    if use_gpu:
        gpu_free_scene(handle)
    else:
        # accum, albedo_acc, sd freed automatically (accum/albedo_acc are List)
        sd.free()
        if use_restir:
            restir_buf_a.free()
            restir_buf_b.free()
            if use_restir_gi:
                gi_buf_a.free()
                gi_buf_b.free()
        if use_sms_restir:
            sms_buf_a.free()
            sms_buf_b.free()
    mojo_parsed_free(psc)
    viewer_destroy(v)
