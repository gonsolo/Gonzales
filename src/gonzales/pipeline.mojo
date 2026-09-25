from std.memory.alloc import unsafe_alloc
from std.memory import OwnedPointer
from std.collections import List, Array
from std.math import sqrt, tan, ceil
from std.sys.info import size_of
from max.gpu.host import DeviceBuffer
from .pbrt_parser import ParsedScene_Mojo, mojo_parsed_free, mojo_parsed_scene_descriptor, resize_film, mojo_apply_overrides
from .scene_loader import mojo_parse_scene_any
from .rendering import render_all_tiles, normalize_film, apply_film_sensor, fmt_time, progress_str
from std.time import perf_counter_ns
from .geometry import RGB, Point3f, Vec3f, Bounds3f, dot, _is_real_ptr
from .render_state import TileResult_C, PathState_C, FilmDims, FilterParams
from .primitives import Ray_C, TriangleMesh_C
from .curves import Curve_C, curve_piece_bounds
from .postprocess import denoise, write_image, write_image_cropped, write_image_cropwindow
from .transform import Mat4
from .sampling import TileSamplerParams_C, mix_bits_u64, encode_morton2, sobol_get_sample_index, sobol_sample, derive_pcg_seeds, camera_ray_from_film_xy
from .bvh import BVH2Node, SceneDescriptor2_C, render_aux_buffers, _scene_bounding_sphere
from .sppm import sppm_render
from .bdpt import vcm_render, vcm_render_gpu, vcm_render_gpu_wavefront, _BDPT_MAX_VERTS, sppm_render_gpu
from .guide import GuideGrid, guide_create, guide_free, guide_clone_empty, guide_refine, null_guide, guide_merge, guide_cell_has_data
from .restir_di import DIReservoir, di_reservoir_init, ReservoirIO, reservoir_io_null
from .restir_gi import GIReservoir, gi_reservoir_init, GIReservoirIO, gi_reservoir_io_null
from .restir_sms import SMSReservoir, sms_reservoir_init, SMSReservoirIO, sms_reservoir_io_null
from .restir_vol import VolReservoir, vol_reservoir_init, VolReservoirIO, vol_reservoir_io_null
from .gpu import GpuSceneHandle, WAVEFRONT_BATCH, gpu_available, gpu_upload_scene, gpu_render_sample, gpu_render_wavefront, gpu_download_film, gpu_download_albedo, gpu_clear_film, gpu_clear_restir, gpu_clear_restir_vol, gpu_atrous_denoise, gpu_gen_aux_buffers, gpu_free_scene
from .viewer import CameraState, ViewerHandle, viewer_create, viewer_update_framebuffer, viewer_should_close, viewer_poll_events, viewer_get_camera_state, viewer_set_camera_state, viewer_destroy, build_camera_to_world
from .spectrum import SpectralHandle, null_spectral_handle
from .vulkanrt import VulkanRtSceneHandle, vulkanrt_build_scene, vulkanrt_destroy_scene
from .vulkaninterop import (
    VulkanInteropRtSceneHandle, vulkaninterop_rt_create_scene,
    vulkaninterop_rt_get_rays_ptr, vulkaninterop_rt_get_results_ptr,
    vulkaninterop_rt_destroy_scene,
)

# Fraction of the scene's bounding-sphere radius used as SPPM's initial gather
# radius when nothing explicit is given. Calibrated on classroom, whose black
# fraction falls 79.3% -> 43.4% -> 2.7% -> 0.0% as the radius goes
# 0.05 -> 0.3 -> 1.0 -> 3.0, i.e. it wants ~1-3 units where the old fixed
# default handed it 0.05. Too small loses all indirect light (black pixels);
# too large over-blurs and slows the gather, so this deliberately sits at the
# low end of the working range and SPPM's own per-pass radius reduction takes
# it down from there.
comptime SPPM_DEFAULT_RADIUS_FRACTION = Float32(0.006)

# Resolve effective SPPM radius/photons-per-pass: an explicit CLI flag
# (sentinel -1 = not passed) wins; otherwise fall back to what the scene's
# own `Integrator "sppm"` directive specifies; otherwise a default derived
# from the SCENE'S OWN SIZE (photons: film_w*film_h, matching pbrt-v4's SPPM
# default of one photon per pixel when photonsperiteration is unspecified).
#
# The radius default used to be a hard-coded 0.05 WORLD UNITS, which is a
# quantity with a scale but no reference -- fine in a unit-sized test scene
# and hopeless in a room. A gather radius far below the photon spacing finds
# nothing, so the visible point contributes nothing and the pixel comes out
# EXACTLY BLACK; since direct light still lands, what is lost is precisely the
# dim, indirect-only regions. That is what it looked like across the corpus:
# 14 scenes blacker than the reference, worst in the big ones -- classroom
# 64% black, sanmiguel 51%, spaceship 44% -- and measured on classroom the
# black fraction runs 79.3% at r=0.05, 43.4% at 0.3, 2.7% at 1.0 and 0.0% at
# 3.0. (pbrt-v4's own default is a fixed 1.0, which has the same flaw and
# merely picks a luckier constant for metre-scale scenes.)
#
# Scaling by the scene's bounding sphere makes the default mean the same thing
# at every scale. An explicit CLI or scene-file radius still wins outright.

@always_inline
def _sample_clamp(psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin]) -> Float32:
    """`maxcomponentvalue` expressed in the units the accumulation kernels see.

    pbrt clamps each SAMPLE's sensor RGB (RGBFilm::AddSample); normalize_film
    clamps the finished pixel, which almost never trips and so removed
    essentially nothing. The kernels accumulate pre-iso radiance and
    normalize_film multiplies by iso/100 afterwards, so the per-sample limit
    has to be divided by that same scale to mean the same thing."""
    var mcv = psc[unsafe_offset=0].film_max_comp
    if mcv <= Float32(0):
        return Float32(0)
    var scale = psc[unsafe_offset=0].film_iso / Float32(100)
    if scale <= Float32(0):
        return Float32(0)
    return mcv / scale


def _resolve_sppm_params(
    psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd: SceneDescriptor2_C,
    sppm_photons_cli: Int32,
    sppm_radius_cli: Float32,
) -> Tuple[Int32, Float32]:
    var radius = sppm_radius_cli
    if radius <= Float32(0):
        if psc[unsafe_offset=0].sppm_radius > Float32(0):
            radius = psc[unsafe_offset=0].sppm_radius
        else:
            var (_, scene_radius) = _scene_bounding_sphere(sd)
            radius = scene_radius * SPPM_DEFAULT_RADIUS_FRACTION
    var photons = sppm_photons_cli
    if photons <= Int32(0):
        if psc[unsafe_offset=0].sppm_photons_per_iter > Int32(0):
            photons = psc[unsafe_offset=0].sppm_photons_per_iter
        else:
            photons = psc[unsafe_offset=0].film_w * psc[unsafe_offset=0].film_h
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
        if Int(vcm_photons_cli) < n_pix:
            # Said out loud: a silently raised count made a light-path sweep
            # below n_pix look like "the light pass costs nothing".
            print("Warning: --vcm-photons " + String(vcm_photons_cli) + " is below one light path per pixel; using "
                  + String(n_pix) + " (connections pair every pixel with its own light path)")
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
def _generate_sobol_matrices(path: String) -> Optional[Pointer[UInt32, MutUntrackedOrigin]]:
    var file_buf: Pointer[UInt8, MutUntrackedOrigin]
    var file_size: Int
    try:
        var f = open(path, "r")
        var bytes = f.read_bytes()
        f.close()
        file_size = len(bytes)
        file_buf = unsafe_alloc[UInt8](file_size + 1)
        var bytes_ptr = bytes.unsafe_ptr()
        for i in range(file_size):
            file_buf[unsafe_offset=i] = bytes_ptr[unsafe_offset=i]
        file_buf[unsafe_offset=file_size] = UInt8(0)
    except:
        print("Error: cannot open Sobol data file: " + path)
        return None

    # Allocate matrices: 21201 dimensions × 52 bits
    comptime N_DIMS = 21201
    comptime N_BITS = 52
    var matrices = unsafe_alloc[UInt32](N_DIMS * N_BITS)
    # Zero-initialize
    for i in range(N_DIMS * N_BITS):
        matrices[unsafe_offset=i] = UInt32(0)

    # Dimension 0: all ones (standard Sobol)
    for j in range(N_BITS):
        matrices[unsafe_offset=j] = UInt32(1) << UInt32(31 - j)

    # Parse remaining dimensions from file
    var pos = 0
    var flen = Int(file_size)
    var dim = 1

    # Helper: skip whitespace (spaces and tabs, but not newlines)
    # Read one line at a time and parse
    while pos < flen and dim < N_DIMS:
        # Skip leading whitespace including newlines
        while pos < flen and (file_buf[unsafe_offset=pos] == UInt8(32) or file_buf[unsafe_offset=pos] == UInt8(9) or file_buf[unsafe_offset=pos] == UInt8(10) or file_buf[unsafe_offset=pos] == UInt8(13)):
            pos += 1
        if pos >= flen:
            break
        # Skip comment lines starting with '#'
        if file_buf[unsafe_offset=pos] == UInt8(35):  # '#'
            while pos < flen and file_buf[unsafe_offset=pos] != UInt8(10):
                pos += 1
            continue

        # The new-joe-kuo file opens with a COLUMN-NAME HEADER line
        # ("d s a m_i"). It isn't a '#' comment, so it used to fall through
        # to the number parser below, which read s=0/a=0 from the letters,
        # wrote no direction numbers, ran the recurrence over zeros -- and
        # then still did `dim += 1`. That silently consumed dimension 1 and
        # left its whole 52-entry matrix zero, so sobol_sample(.., 1, ..)
        # returned exactly 0.0 for EVERY sample: the y pixel-filter offset
        # was pinned at -yradius, shifting every path-traced image down by
        # the filter's y radius (5px on cornell-box) with zero vertical
        # antialiasing, and shifting every higher dimension by one as well.
        # Skip any line that doesn't start with a digit, WITHOUT consuming
        # a dimension.
        if not (file_buf[unsafe_offset=pos] >= UInt8(48) and file_buf[unsafe_offset=pos] <= UInt8(57)):
            while pos < flen and file_buf[unsafe_offset=pos] != UInt8(10):
                pos += 1
            continue

        # Each data line is "d s a m1 m2 ... ms" -- FOUR leading columns, not
        # three. The `d` (dimension) column was previously not read at all,
        # so every field landed one column to the left: d was taken as s, s
        # as a, and a as the first m value. Read d explicitly and index by
        # it (file d=2 is 0-indexed dimension 1, matching dimension 0's
        # hardcoded identity matrix above).
        var d_col = Int32(0)
        while pos < flen and file_buf[unsafe_offset=pos] >= UInt8(48) and file_buf[unsafe_offset=pos] <= UInt8(57):
            d_col = d_col * Int32(10) + Int32(file_buf[unsafe_offset=pos]) - Int32(48)
            pos += 1
        while pos < flen and (file_buf[unsafe_offset=pos] == UInt8(32) or file_buf[unsafe_offset=pos] == UInt8(9)):
            pos += 1

        # s = number of direction numbers
        var s = Int32(0)
        while pos < flen and file_buf[unsafe_offset=pos] >= UInt8(48) and file_buf[unsafe_offset=pos] <= UInt8(57):
            s = s * Int32(10) + Int32(file_buf[unsafe_offset=pos]) - Int32(48)
            pos += 1
        # skip whitespace
        while pos < flen and (file_buf[unsafe_offset=pos] == UInt8(32) or file_buf[unsafe_offset=pos] == UInt8(9)):
            pos += 1

        # a = polynomial
        var a = UInt32(0)
        while pos < flen and file_buf[unsafe_offset=pos] >= UInt8(48) and file_buf[unsafe_offset=pos] <= UInt8(57):
            a = a * UInt32(10) + UInt32(file_buf[unsafe_offset=pos]) - UInt32(48)
            pos += 1
        # skip whitespace
        while pos < flen and (file_buf[unsafe_offset=pos] == UInt8(32) or file_buf[unsafe_offset=pos] == UInt8(9)):
            pos += 1

        # m values
        var m = Array[UInt32, 52](fill=UInt32(0))
        var num_m = Int(s)
        if num_m > N_BITS:
            num_m = N_BITS
        for i in range(num_m):
            var v = UInt32(0)
            while pos < flen and file_buf[unsafe_offset=pos] >= UInt8(48) and file_buf[unsafe_offset=pos] <= UInt8(57):
                v = v * UInt32(10) + UInt32(file_buf[unsafe_offset=pos]) - UInt32(48)
                pos += 1
            m[i] = v
            while pos < flen and (file_buf[unsafe_offset=pos] == UInt8(32) or file_buf[unsafe_offset=pos] == UInt8(9)):
                pos += 1

        # Skip to end of line
        while pos < flen and file_buf[unsafe_offset=pos] != UInt8(10):
            pos += 1

        # Compute direction numbers v[i] = m[i] << (32 - i - 1)
        dim = Int(d_col) - 1
        if dim < 1 or dim >= N_DIMS:
            continue
        var base = dim * N_BITS
        for i in range(Int(s)):
            if i >= N_BITS:
                break
            matrices[unsafe_offset=base + i] = m[i] << UInt32(31 - i)

        # Recurrence for i >= s
        for i in range(Int(s), N_BITS):
            var v_prev = matrices[unsafe_offset=base + i - Int(s)]
            var vi = v_prev ^ (v_prev >> UInt32(s))
            var j = 1
            var poly = a
            while j <= Int(s) - 1:
                if (poly & UInt32(1)) != UInt32(0):
                    vi ^= matrices[unsafe_offset=base + i - j]
                poly >>= 1
                j += 1
            matrices[unsafe_offset=base + i] = vi

        dim += 1

    file_buf.unsafe_free()

    if dim < 2:
        print("Warning: Sobol file had fewer dimensions than expected")
    return Optional(matrices)


def _gpu_upload_scene(
    psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    sobol: Pointer[UInt32, MutUntrackedOrigin],
    n_pixels: Int,
    # Decomposed, NOT a single by-value `spectral: SpectralHandle` param --
    # see spectrum.mojo's long comment on the confirmed by-value SpectralHandle
    # miscompilation; this GPU-upload path reproduced the same corruption
    # class (see project_priority_backlog memory item 3).
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
) -> Pointer[GpuSceneHandle, MutUntrackedOrigin]:
    var film = FilmDims(psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h)
    var n_meshes = Int(psc[unsafe_offset=0].mesh_count)
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
        pts_counts[i] = Int64(psc[unsafe_offset=0].mesh_n_verts[unsafe_offset=i]) * 4
        fi_counts[i]  = Int64(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=i])
        vi_counts[i]  = Int64(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=i]) * 3
        uv_counts[i]  = Int64(psc[unsafe_offset=0].mesh_uv_n_verts[unsafe_offset=i])
        nrm_counts[i] = Int64(psc[unsafe_offset=0].mesh_nrm_n_verts[unsafe_offset=i])
    var handle = gpu_upload_scene(
        # CPU-inclusive TLAS (tris+curves+instances) — now that GPU has
        # BLAS/instance upload + traversal support, it uses the same TLAS
        # SceneDescriptor2_C does rather than the instance-free one.
        psc[unsafe_offset=0].bvh_nodes_cpu,      Int64(psc[unsafe_offset=0].bvh_node_count_cpu),
        psc[unsafe_offset=0].prim_ids_cpu,       Int64(psc[unsafe_offset=0].prim_count_cpu),
        psc[unsafe_offset=0].blas_nodes_arr, psc[unsafe_offset=0].blas_primids_arr,
        psc[unsafe_offset=0].blas_node_counts, psc[unsafe_offset=0].blas_primid_counts, Int64(psc[unsafe_offset=0].blas_count),
        psc[unsafe_offset=0].instances, Int64(psc[unsafe_offset=0].instance_count),
        psc[unsafe_offset=0].meshes,         Int64(n_meshes),
        pts_counts.unsafe_ptr(), fi_counts.unsafe_ptr(),
        vi_counts.unsafe_ptr(), uv_counts.unsafe_ptr(),
        nrm_counts.unsafe_ptr(),
        psc[unsafe_offset=0].tex_filenames,  psc[unsafe_offset=0].tex_count,
        psc[unsafe_offset=0].materials,      Int64(psc[unsafe_offset=0].material_count),
        psc[unsafe_offset=0].area_lights,    Int64(psc[unsafe_offset=0].area_light_count),
        psc[unsafe_offset=0].spheres,        Int64(psc[unsafe_offset=0].sphere_count),
        psc[unsafe_offset=0].curves,         Int64(psc[unsafe_offset=0].curve_count),
        psc[unsafe_offset=0].distant_lights, Int64(psc[unsafe_offset=0].distant_count),
        psc[unsafe_offset=0].point_lights,   Int64(psc[unsafe_offset=0].point_count),
        psc[unsafe_offset=0].light_sampler.cdf, Int64(psc[unsafe_offset=0].light_sampler.n),
        psc[unsafe_offset=0].infinite_lights, Int64(psc[unsafe_offset=0].infinite_count),
        psc[unsafe_offset=0].mediums,         Int64(psc[unsafe_offset=0].medium_count),
        psc[unsafe_offset=0].medium_ifaces,   Int64(psc[unsafe_offset=0].medium_iface_count),
        psc[unsafe_offset=0].grids,           Int64(psc[unsafe_offset=0].grid_count),
        psc[unsafe_offset=0].nvdb_grids,      Int64(psc[unsafe_offset=0].nvdb_grid_count),
        psc[unsafe_offset=0].measured_brdfs, Int64(psc[unsafe_offset=0].measured_count),
        Int64(n_pixels),
        sobol,
        psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world,
        FilterParams(
            psc[unsafe_offset=0].filter_sigma, psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y,
            psc[unsafe_offset=0].filter_norm_x, psc[unsafe_offset=0].filter_norm_y, psc[unsafe_offset=0].filter_type,
        ),
        film,
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
    psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
) -> Tuple[Pointer[Int64, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin]]:
    var n_meshes = Int(psc[unsafe_offset=0].mesh_count)
    var mat_idx = unsafe_alloc[Int64](max(n_meshes, 1))
    var al_idx = unsafe_alloc[Int32](max(n_meshes, 1))
    var seen = unsafe_alloc[UInt8](max(n_meshes, 1))
    for i in range(max(n_meshes, 1)):
        mat_idx[unsafe_offset=i] = Int64(0)
        al_idx[unsafe_offset=i] = Int32(-1)
        seen[unsafe_offset=i] = UInt8(0)
    for i in range(Int(psc[unsafe_offset=0].prim_count)):
        var p = psc[unsafe_offset=0].prim_ids[unsafe_offset=i]
        if p.type == Int8(0):
            var mi = Int(p.id1)
            if mi >= 0 and mi < n_meshes and seen[unsafe_offset=mi] == UInt8(0):
                mat_idx[unsafe_offset=mi] = p.materialIndex
                seen[unsafe_offset=mi] = UInt8(1)
        elif p.type == Int8(3):
            var mi = Int(p.id2 >> 32)
            if mi >= 0 and mi < n_meshes and seen[unsafe_offset=mi] == UInt8(0):
                mat_idx[unsafe_offset=mi] = p.materialIndex
                al_idx[unsafe_offset=mi] = Int32(p.id1)
                seen[unsafe_offset=mi] = UInt8(1)
    # Object-instancing templates: their mesh triangles live ONLY in each
    # template's own BLAS (psc[0].prim_ids above is the ordinary top-level
    # TLAS, which explicitly excludes template meshes -- see finalize_scene),
    # so without this second pass every template mesh index would be left at
    # the mat_idx=0/al_idx=-1 default, giving Vulkan RT instance hits the
    # wrong material. AreaLightSource is not supported inside ObjectBegin/
    # ObjectEnd (parser skips it there), so template triangles are always
    # type==0 -- no type==3 case needed here.
    for tmpl in range(Int(psc[unsafe_offset=0].blas_count)):
        var tprims = psc[unsafe_offset=0].blas_primids_arr[unsafe_offset=tmpl]
        for i in range(Int(psc[unsafe_offset=0].blas_primid_counts[unsafe_offset=tmpl])):
            var p = tprims[unsafe_offset=i]
            var mi = Int(p.id1)
            if mi >= 0 and mi < n_meshes and seen[unsafe_offset=mi] == UInt8(0):
                mat_idx[unsafe_offset=mi] = p.materialIndex
                seen[unsafe_offset=mi] = UInt8(1)
    seen.unsafe_free()
    return (mat_idx, al_idx)

def debug_trace_pixel(
    path: Pointer[UInt8, MutUntrackedOrigin],
    px: Int32, py: Int32,
    override_w: Int32 = Int32(0), override_h: Int32 = Int32(0),
):
    """Trace the centre ray of one pixel and print the path bounce-by-bounce
    (hit mesh/material/normal/t, dielectric entering/eta/Fresnel decision,
    envmap lookup). For comparing against `pbrt --pixelmaterial`."""
    from .bvh import traverse_bvh2_core, test_spheres, any_hit_bvh2_core, _equal_area_sphere_to_square
    from .geometry import Material_C, cross, fr_dielectric
    from .primitives import Intersection_C, sphere_outward_normal
    from .bxdf import dielectric_interface
    from .sppm import _geom_normal

    var psc = mojo_parse_scene_any(path)
    if not _is_real_ptr[ParsedScene_Mojo](psc):
        print("parse failed"); return

    # --pixel used to ALWAYS trace against the scene's native resolution,
    # silently ignoring --width/--height/--resolution -- so a coordinate
    # meant for a downscaled render (e.g. "the pixel at 200,68 in a
    # --width 400 image") actually probed a completely different point in
    # the full-resolution frame, with no warning. Same missing-dimension
    # derivation as parse_and_render/render_interactive.
    if override_w > 0 or override_h > 0:
        var eff_w = override_w
        var eff_h = override_h
        if eff_w <= 0:
            eff_w = Int32(Int(eff_h) * Int(psc[unsafe_offset=0].film_w) / max(Int(psc[unsafe_offset=0].film_h), 1))
        if eff_h <= 0:
            eff_h = Int32(Int(eff_w) * Int(psc[unsafe_offset=0].film_h) / max(Int(psc[unsafe_offset=0].film_w), 1))
        resize_film(psc, eff_w, eff_h)
    if px >= psc[unsafe_offset=0].film_w or py >= psc[unsafe_offset=0].film_h:
        print("--pixel", px, py, "is outside the", psc[unsafe_offset=0].film_w, "x", psc[unsafe_offset=0].film_h,
              "frame being traced -- pass --width/--height/--resolution to match your render")
        return

    # Centre ray (no jitter): raster_to_camera then camera_to_world rotation.
    var r2c = psc[unsafe_offset=0].raster_to_camera
    var c2w = psc[unsafe_offset=0].camera_to_world
    var fX = Float32(px) + Float32(0.5)
    var fY = Float32(py) + Float32(0.5)
    var (dir1, org1, _cl1) = camera_ray_from_film_xy(fX, fY, r2c, c2w)
    # Decomposed to scalars for the bounce loop below, which pre-dates
    # camera_ray_from_film_xy and hand-rolls dot/reflect/refract in every
    # material branch rather than using Point3f/Vec3f's own operators
    # (__add__, __sub__, dot() all already exist -- see geometry.mojo). That
    # is real duplication too, just not this commit's: it is 250 lines with
    # several material branches, no automated coverage (a debug tool reached
    # only via --pixel), and a transcription slip in the middle of a
    # reflect/refract branch would have nothing to catch it. Left alone here
    # rather than risked inline; worth its own pass.
    var ox = org1.x; var oy = org1.y; var oz = org1.z
    var dx = dir1.x; var dy = dir1.y; var dz = dir1.z
    print("PIXEL", px, py, "ray.o", org1, "ray.d", dir1)

    var inter = unsafe_alloc[Intersection_C](1)
    var current_ior = Float32(1.0)   # mirrors PathState_C.current_dielectric_ior
    var previous_ior = Float32(1.0)  # mirrors PathState_C.previous_dielectric_ior
    for bounce in range(20):
        var ray = Ray_C(Point3f(ox, oy, oz), Vec3f(dx, dy, dz))
        inter[unsafe_offset=0].hit = Int8(0)
        traverse_bvh2_core(psc[unsafe_offset=0].bvh_nodes, psc[unsafe_offset=0].prim_ids, psc[unsafe_offset=0].meshes, psc[unsafe_offset=0].curves, ray, Float32(1.0e38), inter,
                            psc[unsafe_offset=0].blas_nodes_arr, psc[unsafe_offset=0].blas_primids_arr, psc[unsafe_offset=0].instances)
        if psc[unsafe_offset=0].sphere_count > 0:
            test_spheres(psc[unsafe_offset=0].spheres, Int(psc[unsafe_offset=0].sphere_count), ray, inter)
        if inter[unsafe_offset=0].hit == Int8(0):
            # envmap miss
            if psc[unsafe_offset=0].infinite_count > 0:
                var il = psc[unsafe_offset=0].infinite_lights[unsafe_offset=0]
                var ldir = Mat4.load(il.world_to_light) * Vec3f(dx, dy, dz)
                var uv = _equal_area_sphere_to_square(ldir.x, ldir.y, ldir.z)
                var rgb_str = String("(no pixels)")
                if _is_real_ptr(il.pixels_ptr) and il.cdf_w > Int32(0):
                    var iw = Int(il.cdf_w); var ih = Int(il.cdf_h)
                    var pxe = Int(max(Float32(0), min(Float32(iw-1), uv[0]*Float32(iw))))
                    var pye = Int(max(Float32(0), min(Float32(ih-1), uv[1]*Float32(ih))))
                    var rr = il.pixels_ptr[unsafe_offset=(pye*iw+pxe)*3+0]
                    var gg = il.pixels_ptr[unsafe_offset=(pye*iw+pxe)*3+1]
                    var bb = il.pixels_ptr[unsafe_offset=(pye*iw+pxe)*3+2]
                    rgb_str = String(rr) + " " + String(gg) + " " + String(bb)
                print("  bounce", bounce, "MISS -> envmap localdir", ldir, "uv", uv, "rgb", rgb_str)
            else:
                print("  bounce", bounce, "MISS (no envmap)")
            break

        # Identify primitive + material. `type == 4` is an ANALYTIC SPHERE and
        # has no mesh at all -- its id1 indexes psc[0].spheres, not
        # psc[0].meshes. Taking the triangle path for it indexed `meshes` with
        # a sphere's id2 and segfaulted, which is why --pixel died on any
        # scene containing a Shape "sphere" (test_spheres above makes such
        # hits reachable here). Same primId-type dispatch every other consumer
        # already does -- see gpu.mojo's medium-interface kernel,
        # rendering.mojo's CPU medium loop, and bdpt.mojo's
        # _visible_transmittance, which had this exact bug.
        var mat = psc[unsafe_offset=0].materials[unsafe_offset=Int(inter[unsafe_offset=0].primId.materialIndex)]
        var hx = ox + dx*inter[unsafe_offset=0].tHit; var hy = oy + dy*inter[unsafe_offset=0].tHit; var hz = oz + dz*inter[unsafe_offset=0].tHit
        # Geometry via the SAME resolver the renderer uses (spheres,
        # instance transforms and all) -- this used to re-derive the
        # triangle normal inline, which is how it went stale twice.
        var mesh_idx: Int = -1
        if inter[unsafe_offset=0].primId.type == Int8(0):
            mesh_idx = Int(inter[unsafe_offset=0].primId.id1)
        elif inter[unsafe_offset=0].primId.type != Int8(4):
            mesh_idx = Int(inter[unsafe_offset=0].primId.id2 >> 32)
        var gn_v = _geom_normal(inter[unsafe_offset=0], psc[unsafe_offset=0].meshes, psc[unsafe_offset=0].instances, psc[unsafe_offset=0].spheres,
                               Vec3f(hx, hy, hz))
        var gnx = gn_v.x; var gny = gn_v.y; var gnz = gn_v.z
        if mesh_idx >= 0:
            print("  bounce", bounce, "HIT mesh", mesh_idx, "matType", Int(mat.type), "t", inter[unsafe_offset=0].tHit, "p", hx, hy, hz, "gN", gnx, gny, gnz)
        else:
            print("  bounce", bounce, "HIT sphere", Int(inter[unsafe_offset=0].primId.id1), "matType", Int(mat.type), "t", inter[unsafe_offset=0].tHit, "p", hx, hy, hz, "gN", gnx, gny, gnz)

        if Int(mat.type) == 4:
            # Dielectric — mirror shade_dielectric's decision (no RNG: report Fresnel, follow transmit)
            var ior = mat.albedo.r
            var di = dielectric_interface(Vec3f(gnx, gny, gnz), Vec3f(dx, dy, dz), ior,
                                          bounce == 0, current_ior, previous_ior)
            var entering = di.entering
            var nx = di.normal.x; var ny = di.normal.y; var nz = di.normal.z
            var eta = di.eta
            var cos_i = di.cos_i
            var sin2t = di.sin2_t
            var tir = di.tir
            var fres = di.fresnel
            print("        DIELECTRIC entering", Int(entering), "current_ior", current_ior, "surface_ior", ior, "eta", eta, "cos_i", cos_i, "fresnel", fres, "tir", Int(tir))
            # Probe the REFLECTED ray's envmap value (the bright contribution).
            var rcos = dx*nx + dy*ny + dz*nz
            var rfx = dx - nx*Float32(2.0)*rcos
            var rfy = dy - ny*Float32(2.0)*rcos
            var rfz = dz - nz*Float32(2.0)*rcos
            var rfl = _dbg_vlen(rfx, rfy, rfz)
            if rfl > Float32(0.0): rfx /= rfl; rfy /= rfl; rfz /= rfl
            var rray = Ray_C(Point3f(hx+nx*Float32(0.001), hy+ny*Float32(0.001), hz+nz*Float32(0.001)), Vec3f(rfx, rfy, rfz))
            var rint = unsafe_alloc[Intersection_C](1); rint[unsafe_offset=0].hit = Int8(0)
            traverse_bvh2_core(psc[unsafe_offset=0].bvh_nodes, psc[unsafe_offset=0].prim_ids, psc[unsafe_offset=0].meshes, psc[unsafe_offset=0].curves, rray, Float32(1.0e38), rint,
                                psc[unsafe_offset=0].blas_nodes_arr, psc[unsafe_offset=0].blas_primids_arr, psc[unsafe_offset=0].instances)
            if rint[unsafe_offset=0].hit == Int8(0) and psc[unsafe_offset=0].infinite_count > 0:
                var il2 = psc[unsafe_offset=0].infinite_lights[unsafe_offset=0]
                var w2 = il2.world_to_light
                var l2x = w2[unsafe_offset=0]*rfx + w2[unsafe_offset=4]*rfy + w2[unsafe_offset=8]*rfz
                var l2y = w2[unsafe_offset=1]*rfx + w2[unsafe_offset=5]*rfy + w2[unsafe_offset=9]*rfz
                var l2z = w2[unsafe_offset=2]*rfx + w2[unsafe_offset=6]*rfy + w2[unsafe_offset=10]*rfz
                var uv2 = _equal_area_sphere_to_square(l2x, l2y, l2z)
                var rs = String("")
                if _is_real_ptr(il2.pixels_ptr) and il2.cdf_w > Int32(0):
                    var iw2 = Int(il2.cdf_w); var ih2 = Int(il2.cdf_h)
                    var ax = Int(max(Float32(0), min(Float32(iw2-1), uv2[0]*Float32(iw2))))
                    var ay = Int(max(Float32(0), min(Float32(ih2-1), uv2[1]*Float32(ih2))))
                    rs = String(il2.pixels_ptr[unsafe_offset=(ay*iw2+ax)*3+0]) + " " + String(il2.pixels_ptr[unsafe_offset=(ay*iw2+ax)*3+1]) + " " + String(il2.pixels_ptr[unsafe_offset=(ay*iw2+ax)*3+2])
                print("        REFLECT dir", rfx, rfy, rfz, "-> envmap uv", uv2[0], uv2[1], "rgb", rs)
            else:
                print("        REFLECT dir", rfx, rfy, rfz, "-> hits mesh (occluded), matType", Int(psc[unsafe_offset=0].materials[unsafe_offset=Int(rint[unsafe_offset=0].primId.materialIndex)].type) if rint[unsafe_offset=0].hit != Int8(0) else -1)
            rint.unsafe_free()
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
                var new_current_ior = ior if entering else previous_ior
                var new_previous_ior = current_ior if entering else Float32(1.0)
                current_ior = new_current_ior
                previous_ior = new_previous_ior
                print("        -> TRANSMIT dir", dx, dy, dz, "new current_ior", current_ior, "new previous_ior", previous_ior)
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
            var rint5 = unsafe_alloc[Intersection_C](1); rint5[unsafe_offset=0].hit = Int8(0)
            traverse_bvh2_core(psc[unsafe_offset=0].bvh_nodes, psc[unsafe_offset=0].prim_ids, psc[unsafe_offset=0].meshes, psc[unsafe_offset=0].curves, rray5, Float32(1.0e38), rint5,
                                psc[unsafe_offset=0].blas_nodes_arr, psc[unsafe_offset=0].blas_primids_arr, psc[unsafe_offset=0].instances)
            if rint5[unsafe_offset=0].hit == Int8(0):
                print("        COAT REFLECT dir", rfx5, rfy5, rfz5, "-> MISS (no envmap in this scene)")
            else:
                var ptype5 = Int(rint5[unsafe_offset=0].primId.type)
                var pmatidx5 = Int(rint5[unsafe_offset=0].primId.materialIndex)
                print("        COAT REFLECT dir", rfx5, rfy5, rfz5, "-> hit primType", ptype5, "matType", Int(psc[unsafe_offset=0].materials[unsafe_offset=pmatidx5].type), "matIdx", pmatidx5, "t", rint5[unsafe_offset=0].tHit)
            rint5.unsafe_free()
            ox = hx+nx5*Float32(0.0001); oy = hy+ny5*Float32(0.0001); oz = hz+nz5*Float32(0.0001)
            dx = rfx5; dy = rfy5; dz = rfz5
            print("        -> FOLLOW coat reflection (probe, ignores roughness)")
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
            var rint3 = unsafe_alloc[Intersection_C](1); rint3[unsafe_offset=0].hit = Int8(0)
            traverse_bvh2_core(psc[unsafe_offset=0].bvh_nodes, psc[unsafe_offset=0].prim_ids, psc[unsafe_offset=0].meshes, psc[unsafe_offset=0].curves, rray3, Float32(1.0e38), rint3,
                                psc[unsafe_offset=0].blas_nodes_arr, psc[unsafe_offset=0].blas_primids_arr, psc[unsafe_offset=0].instances)
            if rint3[unsafe_offset=0].hit == Int8(0):
                print("        REFLECT dir", rfx3, rfy3, rfz3, "-> MISS (no envmap in this scene)")
            else:
                var ptype3 = Int(rint3[unsafe_offset=0].primId.type)
                var pmatidx3 = Int(rint3[unsafe_offset=0].primId.materialIndex)
                print("        REFLECT dir", rfx3, rfy3, rfz3, "-> hit primType", ptype3, "matType", Int(psc[unsafe_offset=0].materials[unsafe_offset=pmatidx3].type), "matIdx", pmatidx3, "t", rint3[unsafe_offset=0].tHit)
            rint3.unsafe_free()
            ox = hx+nx3*Float32(0.0001); oy = hy+ny3*Float32(0.0001); oz = hz+nz3*Float32(0.0001)
            dx = rfx3; dy = rfy3; dz = rfz3
            print("        -> FOLLOW conductor reflection (probe, ignores roughness)")
        elif Int(mat.type) == 1:
            # Diffuse — occlusion probe toward every light in the scene, offset
            # along the geometric normal like a real shadow ray would be.
            var ox1 = hx + gnx*Float32(0.0001)
            var oy1 = hy + gny*Float32(0.0001)
            var oz1 = hz + gnz*Float32(0.0001)
            for dli in range(Int(psc[unsafe_offset=0].distant_count)):
                var dl = psc[unsafe_offset=0].distant_lights[unsafe_offset=dli]
                var ldx = -dl.direction.x; var ldy = -dl.direction.y; var ldz = -dl.direction.z
                var cos_s = gnx*ldx + gny*ldy + gnz*ldz
                var sray = Ray_C(Point3f(ox1, oy1, oz1), Vec3f(ldx, ldy, ldz))
                var occluded = any_hit_bvh2_core(psc[unsafe_offset=0].bvh_nodes, psc[unsafe_offset=0].prim_ids, psc[unsafe_offset=0].meshes, psc[unsafe_offset=0].curves, sray, Float32(2000.0),
                                                  psc[unsafe_offset=0].blas_nodes_arr, psc[unsafe_offset=0].blas_primids_arr, psc[unsafe_offset=0].instances,
                                                  psc[unsafe_offset=0].spheres, Int(psc[unsafe_offset=0].sphere_count))
                print("        DISTANT", dli, "dir", ldx, ldy, ldz, "cos_s", cos_s, "occluded", Int(occluded))
            for ali in range(Int(psc[unsafe_offset=0].area_light_count)):
                var al = psc[unsafe_offset=0].area_lights[unsafe_offset=ali]
                var almesh = psc[unsafe_offset=0].meshes[unsafe_offset=Int(al.meshIdx)]
                # Centroid of the light's first triangle — coarse but enough to
                # tell whether shadow rays toward this light are ever blocked.
                var lv0 = Int(almesh.vertexIndices[unsafe_offset=0]); var lv1 = Int(almesh.vertexIndices[unsafe_offset=1]); var lv2 = Int(almesh.vertexIndices[unsafe_offset=2])
                var lcx = (almesh.points[unsafe_offset=lv0*4]   + almesh.points[unsafe_offset=lv1*4]   + almesh.points[unsafe_offset=lv2*4])   / Float32(3.0)
                var lcy = (almesh.points[unsafe_offset=lv0*4+1] + almesh.points[unsafe_offset=lv1*4+1] + almesh.points[unsafe_offset=lv2*4+1]) / Float32(3.0)
                var lcz = (almesh.points[unsafe_offset=lv0*4+2] + almesh.points[unsafe_offset=lv1*4+2] + almesh.points[unsafe_offset=lv2*4+2]) / Float32(3.0)
                var tlx = lcx - ox1; var tly = lcy - oy1; var tlz = lcz - oz1
                var tdist = _dbg_vlen(tlx, tly, tlz)
                if tdist > Float32(0.0):
                    tlx /= tdist; tly /= tdist; tlz /= tdist
                var cos_sa = gnx*tlx + gny*tly + gnz*tlz
                var sray2 = Ray_C(Point3f(ox1, oy1, oz1), Vec3f(tlx, tly, tlz))
                var occluded2 = any_hit_bvh2_core(psc[unsafe_offset=0].bvh_nodes, psc[unsafe_offset=0].prim_ids, psc[unsafe_offset=0].meshes, psc[unsafe_offset=0].curves, sray2, tdist * Float32(0.999),
                                                   psc[unsafe_offset=0].blas_nodes_arr, psc[unsafe_offset=0].blas_primids_arr, psc[unsafe_offset=0].instances,
                                                   psc[unsafe_offset=0].spheres, Int(psc[unsafe_offset=0].sphere_count))
                print("        AREA", ali, "centroid", lcx, lcy, lcz, "dist", tdist, "cos_s", cos_sa, "occluded", Int(occluded2))
            print("        STOP (diffuse probe only, not following further)")
            break
        else:
            print("        STOP (non-glass material)")
            break
    inter.unsafe_free()
    mojo_parsed_free(psc)


def debug_render_vulkanrt(
    path: Pointer[UInt8, MutUntrackedOrigin],
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
    from .primitives import Intersection_C
    from .vulkanrt import vulkanrt_build_scene, vulkanrt_trace_rays, vulkanrt_destroy_scene
    from std.math import abs

    var psc = mojo_parse_scene_any(path, verbose)
    if not _is_real_ptr[ParsedScene_Mojo](psc):
        print("parse failed"); return

    var w = Int(psc[unsafe_offset=0].film_w)
    var h = Int(psc[unsafe_offset=0].film_h)
    var n_meshes = Int(psc[unsafe_offset=0].mesh_count)
    if n_meshes == 0:
        print("Scene has no triangle meshes -- nothing for Vulkan RT to trace.")
        mojo_parsed_free(psc)
        return
    if psc[unsafe_offset=0].instance_count > Int32(0):
        print("WARNING: scene uses ObjectInstance -- the Vulkan RT scene will be missing instanced geometry placements (see project_vulkan_rt_backend memory)")

    # Build the Vulkan RT scene directly from the parsed meshes -- points/
    # vertexIndices pointers passed through as-is (vulkanrt.h's
    # VulkanRtMesh is a field-for-field mirror of TriangleMesh_C).
    var vmeshes = unsafe_alloc[TriangleMesh_C](n_meshes)
    var point_counts = unsafe_alloc[Int64](n_meshes)
    var vidx_counts = unsafe_alloc[Int64](n_meshes)
    for i in range(n_meshes):
        vmeshes[unsafe_offset=i] = psc[unsafe_offset=0].meshes[unsafe_offset=i]
        point_counts[unsafe_offset=i] = Int64(psc[unsafe_offset=0].mesh_n_verts[unsafe_offset=i])
        vidx_counts[unsafe_offset=i] = Int64(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=i]) * 3

    var scene = vulkanrt_build_scene(vmeshes, Int64(n_meshes), point_counts, vidx_counts)
    if Int(scene) == 0:
        print("vulkanrt_build_scene FAILED -- see stderr for diagnostics")
        vmeshes.unsafe_free(); point_counts.unsafe_free(); vidx_counts.unsafe_free()
        mojo_parsed_free(psc)
        return

    # One centre-ray (no AA jitter) per pixel, same raster_to_camera /
    # camera_to_world math as debug_trace_pixel above -- packed straight
    # into vulkanrt_trace_rays's flat 8-floats-per-ray layout.
    var n_pix = w * h
    var rays = unsafe_alloc[Float32](n_pix * 8)
    var r2c = psc[unsafe_offset=0].raster_to_camera
    var c2w = psc[unsafe_offset=0].camera_to_world
    for py in range(h):
        for px in range(w):
            var fX = Float32(px) + Float32(0.5)
            var fY = Float32(py) + Float32(0.5)
            var (dir2, org2, _cl2) = camera_ray_from_film_xy(fX, fY, r2c, c2w)
            var idx = (py * w + px) * 8
            rays[unsafe_offset=idx+0] = org2.x; rays[unsafe_offset=idx+1] = org2.y; rays[unsafe_offset=idx+2] = org2.z; rays[unsafe_offset=idx+3] = Float32(0.001)
            rays[unsafe_offset=idx+4] = dir2.x; rays[unsafe_offset=idx+5] = dir2.y; rays[unsafe_offset=idx+6] = dir2.z; rays[unsafe_offset=idx+7] = Float32(1.0e8)

    var out_t = unsafe_alloc[Float32](n_pix)
    var out_u = unsafe_alloc[Float32](n_pix)
    var out_v = unsafe_alloc[Float32](n_pix)
    var out_mesh = unsafe_alloc[Int32](n_pix)
    var out_tri = unsafe_alloc[Int32](n_pix)
    var out_hit = unsafe_alloc[UInt8](n_pix)

    var rc = vulkanrt_trace_rays(scene, Int32(n_pix), rays, out_t, out_u, out_v,
                                  out_mesh, out_tri, out_hit)
    if Int(rc) == 0:
        print("vulkanrt_trace_rays FAILED -- see stderr for diagnostics")
        out_t.unsafe_free(); out_u.unsafe_free(); out_v.unsafe_free(); out_mesh.unsafe_free(); out_tri.unsafe_free(); out_hit.unsafe_free()
        rays.unsafe_free()
        vulkanrt_destroy_scene(scene)
        vmeshes.unsafe_free(); point_counts.unsafe_free(); vidx_counts.unsafe_free()
        mojo_parsed_free(psc)
        return

    # Cross-check every pixel against the CPU software BVH tracing the
    # identical ray (step 5's image-level validation, folded into step 4),
    # and build an N.V-shaded visibility image from the GPU hits.
    var img = unsafe_alloc[Float32](n_pix * 3)
    var inter = unsafe_alloc[Intersection_C](1)
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
            var ray = Ray_C(Point3f(rays[unsafe_offset=idx+0], rays[unsafe_offset=idx+1], rays[unsafe_offset=idx+2]),
                             Vec3f(rays[unsafe_offset=idx+4], rays[unsafe_offset=idx+5], rays[unsafe_offset=idx+6]))
            inter[unsafe_offset=0].hit = Int8(0)
            traverse_bvh2_core(psc[unsafe_offset=0].bvh_nodes, psc[unsafe_offset=0].prim_ids, psc[unsafe_offset=0].meshes, psc[unsafe_offset=0].curves, ray, Float32(1.0e8), inter,
                                psc[unsafe_offset=0].blas_nodes_arr, psc[unsafe_offset=0].blas_primids_arr, psc[unsafe_offset=0].instances)
            var cpu_hit = inter[unsafe_offset=0].hit != Int8(0)
            var gpu_hit = out_hit[unsafe_offset=pi] == UInt8(1)
            if cpu_hit: cpu_hits += 1
            if gpu_hit: gpu_hits += 1
            if cpu_hit == gpu_hit: agree += 1
            if cpu_hit and gpu_hit:
                both_hit += 1
                var derr = abs(inter[unsafe_offset=0].tHit - out_t[unsafe_offset=pi])
                depth_err_sum += Float64(derr)
                if derr > depth_err_max: depth_err_max = derr

            var shade = Float32(0)
            if gpu_hit:
                var mi = Int(out_mesh[unsafe_offset=pi])
                var ti = Int(out_tri[unsafe_offset=pi]) * 3
                if mi >= 0 and mi < n_meshes:
                    var mesh = psc[unsafe_offset=0].meshes[unsafe_offset=mi]
                    var v0 = Int(mesh.vertexIndices[unsafe_offset=ti]); var v1 = Int(mesh.vertexIndices[unsafe_offset=ti+1]); var v2 = Int(mesh.vertexIndices[unsafe_offset=ti+2])
                    var p0x = mesh.points[unsafe_offset=v0*4]; var p0y = mesh.points[unsafe_offset=v0*4+1]; var p0z = mesh.points[unsafe_offset=v0*4+2]
                    var p1x = mesh.points[unsafe_offset=v1*4]; var p1y = mesh.points[unsafe_offset=v1*4+1]; var p1z = mesh.points[unsafe_offset=v1*4+2]
                    var p2x = mesh.points[unsafe_offset=v2*4]; var p2y = mesh.points[unsafe_offset=v2*4+1]; var p2z = mesh.points[unsafe_offset=v2*4+2]
                    var gnx = (p1y-p0y)*(p2z-p0z) - (p1z-p0z)*(p2y-p0y)
                    var gny = (p1z-p0z)*(p2x-p0x) - (p1x-p0x)*(p2z-p0z)
                    var gnz = (p1x-p0x)*(p2y-p0y) - (p1y-p0y)*(p2x-p0x)
                    var gnl = sqrt(gnx*gnx + gny*gny + gnz*gnz)
                    if gnl > Float32(0): gnx /= gnl; gny /= gnl; gnz /= gnl
                    shade = abs(rays[unsafe_offset=idx+4]*gnx + rays[unsafe_offset=idx+5]*gny + rays[unsafe_offset=idx+6]*gnz)
            img[unsafe_offset=pi*3+0] = shade; img[unsafe_offset=pi*3+1] = shade; img[unsafe_offset=pi*3+2] = shade

    var hit_agree_pct = Float64(agree) * 100.0 / Float64(max(n_pix, 1))
    var mean_depth_err = depth_err_sum / Float64(max(both_hit, 1))
    print("vulkanrt visibility check:", w, "x", h, "pixels")
    print("  CPU hits:", cpu_hits, " GPU hits:", gpu_hits, " both-hit:", both_hit)
    print("  hit/miss agreement:", hit_agree_pct, "%")
    print("  depth error (both-hit pixels): mean", mean_depth_err, " max", depth_err_max)

    var out_path = String("vulkanrt_debug.exr")
    var out_cstr = unsafe_alloc[UInt8](out_path.byte_length() + 1)
    for k in range(out_path.byte_length()):
        out_cstr[unsafe_offset=k] = out_path.as_bytes()[k]
    out_cstr[unsafe_offset=out_path.byte_length()] = UInt8(0)
    _ = write_image(img, Int32(w), Int32(h), out_cstr, Int32(32), Int32(32))
    print("  wrote", out_path)
    out_cstr.unsafe_free()

    inter.unsafe_free()
    img.unsafe_free()
    out_t.unsafe_free(); out_u.unsafe_free(); out_v.unsafe_free(); out_mesh.unsafe_free(); out_tri.unsafe_free(); out_hit.unsafe_free()
    rays.unsafe_free()
    vulkanrt_destroy_scene(scene)
    vmeshes.unsafe_free(); point_counts.unsafe_free(); vidx_counts.unsafe_free()
    mojo_parsed_free(psc)


def parse_and_render(
    path: Pointer[UInt8, MutUntrackedOrigin],
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
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
    # Phase 7.3: volume-scatter TEMPORAL reuse (no spatial -- see
    # project_restir_migration memory's "7.3" section). INDEPENDENT of
    # use_restir (this is the medium sampler's own NEE, a different call
    # site from DI's diffuse-material NEE). GPU batch mode gets it by
    # joining use_restir's own dispatch-mode-switch condition below (1
    # sample/pixel per dispatch via gpu_render_sample). CPU BATCH mode
    # (this function's non-GPU branch) has no persistence, matching
    # use_restir's own CPU-batch scope exactly -- true cross-sample reuse,
    # CPU or GPU, only ever happens in render_interactive (below), the
    # only place a stable per-frame identity exists to key a reservoir on.
    use_vol_restir_reuse: Bool = False,
) raises -> Int32:
    if use_gpu and not gpu_available():
        print("No GPU available — compile with --target-accelerator sm_86 or similar")
        return Int32(-1)

    var psc = mojo_parse_scene_any(path, verbose)
    if not _is_real_ptr[ParsedScene_Mojo](psc):
        return Int32(-1)
    if override_w > 0 or override_h > 0:
        var eff_w = override_w
        var eff_h = override_h
        # --width (or --height) alone used to be silently DROPPED here --
        # the render then went out at the scene's own, possibly huge, native
        # resolution (bistro_cafe.pbrt is 1920x1080) with no indication the
        # flag did nothing. Derive the missing dimension from the scene's
        # native aspect ratio instead.
        if eff_w <= 0:
            eff_w = Int32(Int(eff_h) * Int(psc[unsafe_offset=0].film_w) / max(Int(psc[unsafe_offset=0].film_h), 1))
        if eff_h <= 0:
            eff_h = Int32(Int(eff_w) * Int(psc[unsafe_offset=0].film_h) / max(Int(psc[unsafe_offset=0].film_w), 1))
        resize_film(psc, eff_w, eff_h)
    mojo_apply_overrides(psc, spp_override, Int32(0), Int32(0), seed_override)

    var fw = psc[unsafe_offset=0].film_w
    var fh = psc[unsafe_offset=0].film_h
    var n_pixels = Int(fw) * Int(fh)
    var results = List[TileResult_C](capacity=n_pixels)

    # Film "float cropwindow" [x0 x1 y0 y1] pixel bounds — ceil() on both
    # ends matches pbrt's own Film pixel-bounds computation exactly (verified
    # against a real cropwindow scene's reference render). That conversion now
    # lives in postprocess.mojo's write_image_cropwindow, which every writer
    # goes through -- it used to be open-coded here only, so --sppm/--vcm
    # wrote the full frame and silently ignored the crop.

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
            sd.unsafe_free()
            mojo_parsed_free(psc)
            return Int32(-1)
        var resolved = _resolve_sppm_params(psc, sd[unsafe_offset=0], sppm_photons, sppm_radius)
        var ret = sppm_render_gpu(
            handle, psc, sd[unsafe_offset=0],
            Int(sppm_passes), Int(resolved[0]), resolved[1],
            no_denoise, verbose,
        )
        gpu_free_scene(handle)
        sd.unsafe_free()
        mojo_parsed_free(psc)
        return ret
    elif use_gpu and use_vcm:
        var sd = mojo_parsed_scene_descriptor(psc, spectral)
        var handle = _gpu_upload_scene(psc, sobol_matrices, n_pixels, spectral.coeffs, spectral.res, spectral.cie_x, spectral.cie_y, spectral.cie_z, spectral.d65)
        if not _is_real_ptr(handle):
            sd.unsafe_free()
            mojo_parsed_free(psc)
            return Int32(-1)
        var n_photons = _resolve_vcm_photons(vcm_photons, n_pixels)
        var resolved_vcm_spp = _resolve_vcm_spp(vcm_spp, psc[unsafe_offset=0].samples_per_pixel)
        var ret: Int32
        if use_vcm_wavefront:
            # Task #163 stage 4 part 4: build the same interop-AND-ray-
            # query-capable Vulkan RT scene the plain --gpu --vulkan-rt-
            # shade path builds (see that branch's own comment below) --
            # reused sequentially across BOTH the light pass and the
            # camera pass every sample (n_light_paths_merge is always
            # >= n_pix, so one scene sized for it covers both).
            var use_vk_vcm = use_vulkan_rt_shade
            var interop_scene_vcm = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling()
            var interop_rays_buf_vcm: Optional[DeviceBuffer[DType.float32]] = None
            var interop_results_buf_vcm: Optional[DeviceBuffer[DType.float32]] = None
            var mesh_material_idx_buf_vcm: Optional[DeviceBuffer[DType.uint8]] = None
            var mesh_al_idx_buf_vcm: Optional[DeviceBuffer[DType.uint8]] = None
            var n_meshes_vk_vcm = 0
            if use_vk_vcm:
                if psc[unsafe_offset=0].curve_count > Int32(0) or psc[unsafe_offset=0].sphere_count > Int32(0) or psc[unsafe_offset=0].instance_count > Int32(0):
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
                    n_meshes_vk_vcm = Int(psc[unsafe_offset=0].mesh_count)
                    var vmeshes_vcm = unsafe_alloc[TriangleMesh_C](max(n_meshes_vk_vcm, 1))
                    var point_counts_vcm = unsafe_alloc[Int64](max(n_meshes_vk_vcm, 1))
                    var vidx_counts_vcm = unsafe_alloc[Int64](max(n_meshes_vk_vcm, 1))
                    for i in range(n_meshes_vk_vcm):
                        vmeshes_vcm[unsafe_offset=i] = psc[unsafe_offset=0].meshes[unsafe_offset=i]
                        point_counts_vcm[unsafe_offset=i] = Int64(psc[unsafe_offset=0].mesh_n_verts[unsafe_offset=i])
                        vidx_counts_vcm[unsafe_offset=i] = Int64(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=i]) * 3
                    # VCM's Vulkan RT path doesn't support object instancing
                    # yet (its own primary/bounce interop -- vulkaninterop_
                    # rt_traverse_light_paths_gpu/_camera_ in bdpt.mojo -- is
                    # a separate wiring from the plain wavefront path's
                    # _gpu_bounce_kernels, not extended this session; see
                    # project_vulkan_rt_backend memory). The guard above
                    # already keeps instanced/curve/sphere scenes off this
                    # path entirely, so template_count/instance_count/
                    # n_curve_leaves are always 0 here.
                    var no_templates_vcm = Pointer[Int64, MutUntrackedOrigin].unsafe_dangling()
                    var no_instances_vcm = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
                    var no_instance_tmpl_vcm = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
                    var no_curve_aabbs_vcm = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
                    var no_curve_i32_vcm = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
                    var no_curve_data_vcm = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
                    interop_scene_vcm = vulkaninterop_rt_create_scene(
                        vmeshes_vcm, Int64(n_meshes_vk_vcm), point_counts_vcm, vidx_counts_vcm,
                        Int64(0), no_templates_vcm, no_templates_vcm,
                        Int64(0), no_instances_vcm, no_instance_tmpl_vcm,
                        Int64(0), no_curve_aabbs_vcm,
                        no_curve_i32_vcm, no_curve_i32_vcm, no_curve_i32_vcm,
                        Int64(0), no_curve_data_vcm, no_curve_i32_vcm,
                        Int64(max_rays_vk_vcm))
                    vmeshes_vcm.unsafe_free(); point_counts_vcm.unsafe_free(); vidx_counts_vcm.unsafe_free()
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
                            var dst = h.unsafe_ptr().unsafe_bitcast[Int64]()
                            for i in range(n_meshes_vk_vcm):
                                dst[unsafe_offset=i] = mesh_material_idx_vcm[unsafe_offset=i]
                        mesh_material_idx_buf_vcm = mmi_buf_vcm^

                        var mai_buf_vcm = handle[].ctx.enqueue_create_buffer[DType.uint8](n_meshes_alloc_vcm * size_of[Int32]())
                        with mai_buf_vcm.map_to_host() as h2:
                            var dst2 = h2.unsafe_ptr().unsafe_bitcast[Int32]()
                            for i in range(n_meshes_vk_vcm):
                                dst2[unsafe_offset=i] = mesh_al_idx_vcm[unsafe_offset=i]
                        mesh_al_idx_buf_vcm = mai_buf_vcm^

                        mesh_material_idx_vcm.unsafe_free()
                        mesh_al_idx_vcm.unsafe_free()

            ret = vcm_render_gpu_wavefront(
                handle, psc, sd[unsafe_offset=0], resolved_vcm_spp, n_photons, no_denoise, verbose,
                use_vk_vcm, interop_scene_vcm, interop_rays_buf_vcm, interop_results_buf_vcm,
                mesh_material_idx_buf_vcm, mesh_al_idx_buf_vcm, n_meshes_vk_vcm,
            )
            if use_vk_vcm:
                vulkaninterop_rt_destroy_scene(interop_scene_vcm)
        else:
            ret = vcm_render_gpu(handle, psc, sd[unsafe_offset=0], resolved_vcm_spp, n_photons, no_denoise, verbose)
        gpu_free_scene(handle)
        sd.unsafe_free()
        mojo_parsed_free(psc)
        return ret
    elif use_gpu:
        var spp = Int(psc[unsafe_offset=0].samples_per_pixel)
        # World units spanned by one pixel per unit distance (for mip LOD):
        # 2*tan(fov/2)/height. fov is in degrees along the shorter axis.
        var px_scale = Float32(2.0) * tan(psc[unsafe_offset=0].camera_fov * Float32(3.14159265 / 360.0)) / Float32(Int(fh))
        # Scale by the sampling rate, as pbrt does: integrators.cpp does
        # `rayDiffScale = max(0.125, 1/sqrt(spp))` before ScaleDifferentials.
        # The reason is that the FOOTPRINT a texture lookup should filter over
        # is not the whole pixel -- it is the spacing between samples, because
        # the spp samples themselves resolve everything finer than that.
        # Without this we filtered over the full pixel at every sample count,
        # i.e. 8x too wide at 64spp (three mip levels too coarse) on every
        # textured surface, which is the wrong direction to be wrong in: it
        # throws away texture detail that the samples had already paid for.
        px_scale *= max(Float32(0.125), Float32(1.0) / sqrt(Float32(max(spp, 1))))
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
        var interop_scene = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling()
        var interop_rays_buf_opt: Optional[DeviceBuffer[DType.float32]] = None
        var interop_results_buf_opt: Optional[DeviceBuffer[DType.float32]] = None
        var mesh_material_idx_buf_opt: Optional[DeviceBuffer[DType.uint8]] = None
        var mesh_al_idx_buf_opt: Optional[DeviceBuffer[DType.uint8]] = None
        var instance_base_mesh_buf_opt: Optional[DeviceBuffer[DType.uint8]] = None
        var n_meshes_vk = 0
        var max_rays_vk = Int64(n_pixels) * Int64(WAVEFRONT_BATCH)
        if use_vk:
            n_meshes_vk = Int(psc[unsafe_offset=0].mesh_count)
            var vmeshes = unsafe_alloc[TriangleMesh_C](max(n_meshes_vk, 1))
            var point_counts = unsafe_alloc[Int64](max(n_meshes_vk, 1))
            var vidx_counts = unsafe_alloc[Int64](max(n_meshes_vk, 1))
            for i in range(n_meshes_vk):
                vmeshes[unsafe_offset=i] = psc[unsafe_offset=0].meshes[unsafe_offset=i]
                point_counts[unsafe_offset=i] = Int64(psc[unsafe_offset=0].mesh_n_verts[unsafe_offset=i])
                vidx_counts[unsafe_offset=i] = Int64(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=i]) * 3

            # Object instancing (project_vulkan_rt_backend memory's
            # "close the Vulkan RT gap" plan, item 1): template_mesh_
            # start/end mark which mesh-index ranges are template-only
            # (excluded from the ordinary BLAS-per-mesh loop, merged
            # instead into their template's own multi-geometry BLAS
            # inside vulkaninterop_rt_create_scene); instance_obj_to_
            # world/instance_template_idx place one TLAS instance per
            # ObjectInstance. Empty (n_templates=0) is byte-identical to
            # the pre-instancing call.
            var n_templates_vk = Int(psc[unsafe_offset=0].blas_count)
            var template_mesh_start_vk = unsafe_alloc[Int64](max(n_templates_vk, 1))
            var template_mesh_end_vk   = unsafe_alloc[Int64](max(n_templates_vk, 1))
            for t in range(n_templates_vk):
                template_mesh_start_vk[unsafe_offset=t] = Int64(psc[unsafe_offset=0].template_mesh_start[unsafe_offset=t])
                template_mesh_end_vk[unsafe_offset=t]   = Int64(psc[unsafe_offset=0].template_mesh_end[unsafe_offset=t])

            var n_instances_vk = Int(psc[unsafe_offset=0].instance_count)
            var instance_o2w_vk = unsafe_alloc[Float32](max(n_instances_vk, 1) * 16)
            var instance_tmpl_idx_vk = unsafe_alloc[Int32](max(n_instances_vk, 1))
            # Precompute each instance's real BASE mesh index (its
            # template's own mstart) here on the host so the GPU decode
            # kernel only needs one array lookup + geometryIndex add --
            # see vulkaninterop_unpack_results_kernel (gpu.mojo).
            var instance_base_mesh_host = unsafe_alloc[Int32](max(n_instances_vk, 1))
            for k in range(n_instances_vk):
                var inst = psc[unsafe_offset=0].instances[unsafe_offset=k]
                for ci in range(16):
                    instance_o2w_vk[unsafe_offset=k * 16 + ci] = inst.objToWorld[ci]
                instance_tmpl_idx_vk[unsafe_offset=k] = Int32(inst.blasIdx)
                instance_base_mesh_host[unsafe_offset=k] = psc[unsafe_offset=0].template_mesh_start[unsafe_offset=Int(inst.blasIdx)]

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
            for i in range(Int(psc[unsafe_offset=0].prim_count)):
                if psc[unsafe_offset=0].prim_ids[unsafe_offset=i].type == Int8(5):
                    n_curve_leaves_vk += 1
            var curve_leaf_aabbs_vk = unsafe_alloc[Float32](max(n_curve_leaves_vk, 1) * 6)
            var curve_leaf_curve_idx_vk = unsafe_alloc[Int32](max(n_curve_leaves_vk, 1))
            var curve_leaf_piece_info_vk = unsafe_alloc[Int32](max(n_curve_leaves_vk, 1))
            var curve_leaf_mat_idx_vk = unsafe_alloc[Int32](max(n_curve_leaves_vk, 1))
            var curve_leaf_write = 0
            for i in range(Int(psc[unsafe_offset=0].prim_count)):
                var p = psc[unsafe_offset=0].prim_ids[unsafe_offset=i]
                if p.type != Int8(5):
                    continue
                var curve = psc[unsafe_offset=0].curves[unsafe_offset=Int(p.id1)]
                var first_piece = Int(p.id2) // 8
                var piece_count = Int(p.id2) % 8
                var (xmin, ymin, zmin, xmax, ymax, zmax) = curve_piece_bounds(curve, first_piece)
                for piece in range(first_piece + 1, first_piece + piece_count):
                    var (pxmin, pymin, pzmin, pxmax, pymax, pzmax) = curve_piece_bounds(curve, piece)
                    xmin = min(xmin, pxmin); ymin = min(ymin, pymin); zmin = min(zmin, pzmin)
                    xmax = max(xmax, pxmax); ymax = max(ymax, pymax); zmax = max(zmax, pzmax)
                var b = curve_leaf_write * 6
                curve_leaf_aabbs_vk[unsafe_offset=b+0] = xmin; curve_leaf_aabbs_vk[unsafe_offset=b+1] = ymin; curve_leaf_aabbs_vk[unsafe_offset=b+2] = zmin
                curve_leaf_aabbs_vk[unsafe_offset=b+3] = xmax; curve_leaf_aabbs_vk[unsafe_offset=b+4] = ymax; curve_leaf_aabbs_vk[unsafe_offset=b+5] = zmax
                curve_leaf_curve_idx_vk[unsafe_offset=curve_leaf_write] = Int32(p.id1)
                curve_leaf_piece_info_vk[unsafe_offset=curve_leaf_write] = Int32(p.id2)
                curve_leaf_mat_idx_vk[unsafe_offset=curve_leaf_write] = Int32(p.materialIndex)
                curve_leaf_write += 1

            var n_curves_vk = Int(psc[unsafe_offset=0].curve_count)
            var curve_data_vk = unsafe_alloc[Float32](max(n_curves_vk, 1) * 14)
            var curve_n_pieces_vk = unsafe_alloc[Int32](max(n_curves_vk, 1))
            for ci in range(n_curves_vk):
                var c = psc[unsafe_offset=0].curves[unsafe_offset=ci]
                var cb = ci * 14
                curve_data_vk[unsafe_offset=cb+0] = c.cp0.x; curve_data_vk[unsafe_offset=cb+1] = c.cp0.y; curve_data_vk[unsafe_offset=cb+2] = c.cp0.z
                curve_data_vk[unsafe_offset=cb+3] = c.cp1.x; curve_data_vk[unsafe_offset=cb+4] = c.cp1.y; curve_data_vk[unsafe_offset=cb+5] = c.cp1.z
                curve_data_vk[unsafe_offset=cb+6] = c.cp2.x; curve_data_vk[unsafe_offset=cb+7] = c.cp2.y; curve_data_vk[unsafe_offset=cb+8] = c.cp2.z
                curve_data_vk[unsafe_offset=cb+9] = c.cp3.x; curve_data_vk[unsafe_offset=cb+10] = c.cp3.y; curve_data_vk[unsafe_offset=cb+11] = c.cp3.z
                curve_data_vk[unsafe_offset=cb+12] = c.width0; curve_data_vk[unsafe_offset=cb+13] = c.width1
                curve_n_pieces_vk[unsafe_offset=ci] = c.n_pieces

            interop_scene = vulkaninterop_rt_create_scene(
                vmeshes, Int64(n_meshes_vk), point_counts, vidx_counts,
                Int64(n_templates_vk), template_mesh_start_vk, template_mesh_end_vk,
                Int64(n_instances_vk), instance_o2w_vk, instance_tmpl_idx_vk,
                Int64(n_curve_leaves_vk), curve_leaf_aabbs_vk,
                curve_leaf_curve_idx_vk, curve_leaf_piece_info_vk, curve_leaf_mat_idx_vk,
                Int64(n_curves_vk), curve_data_vk, curve_n_pieces_vk,
                max_rays_vk)
            vmeshes.unsafe_free(); point_counts.unsafe_free(); vidx_counts.unsafe_free()
            template_mesh_start_vk.unsafe_free(); template_mesh_end_vk.unsafe_free()
            instance_o2w_vk.unsafe_free(); instance_tmpl_idx_vk.unsafe_free()
            curve_leaf_aabbs_vk.unsafe_free()
            curve_leaf_curve_idx_vk.unsafe_free(); curve_leaf_piece_info_vk.unsafe_free(); curve_leaf_mat_idx_vk.unsafe_free()
            curve_data_vk.unsafe_free(); curve_n_pieces_vk.unsafe_free()
            if Int(interop_scene) == 0:
                print("WARNING: vulkaninterop_rt_create_scene FAILED -- falling back to CUDA intersection")
                use_vk = False
                instance_base_mesh_host.unsafe_free()
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
                    var dst = h.unsafe_ptr().unsafe_bitcast[Int64]()
                    for i in range(n_meshes_vk):
                        dst[unsafe_offset=i] = mesh_material_idx[unsafe_offset=i]
                mesh_material_idx_buf_opt = mmi_buf^

                var mai_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_meshes_alloc * size_of[Int32]())
                with mai_buf.map_to_host() as h2:
                    var dst2 = h2.unsafe_ptr().unsafe_bitcast[Int32]()
                    for i in range(n_meshes_vk):
                        dst2[unsafe_offset=i] = mesh_al_idx[unsafe_offset=i]
                mesh_al_idx_buf_opt = mai_buf^

                mesh_material_idx.unsafe_free()
                mesh_al_idx.unsafe_free()

                if n_instances_vk > 0:
                    var ibm_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](n_instances_vk * size_of[Int32]())
                    with ibm_buf.map_to_host() as h3:
                        var dst3 = h3.unsafe_ptr().unsafe_bitcast[Int32]()
                        for k in range(n_instances_vk):
                            dst3[unsafe_offset=k] = instance_base_mesh_host[unsafe_offset=k]
                    instance_base_mesh_buf_opt = ibm_buf^
                instance_base_mesh_host.unsafe_free()

        var hash_bits = UInt64(mix_bits_u64(UInt64(0)))
        var seed_dim0 = UInt32(hash_bits & UInt64(0xFFFFFFFF))
        # See rendering.mojo's matching comment: was a hardcoded 0.
        var seed_dim1 = UInt32(UInt64(mix_bits_u64(UInt64(1))) & UInt64(0xFFFFFFFF))
        # use_vol_restir_reuse joins use_restir's own condition here: both
        # need the "1 sample/pixel per dispatch" gpu_render_sample path
        # instead of gpu_render_wavefront to get persistent per-pixel
        # reservoir state (Phase 7.3, same reasoning as Phase 2/3 below --
        # WAVEFRONT_BATCH concurrent samples/pixel would race a shared
        # reservoir slot).
        var vol_reuse_needs_sample_dispatch = use_restir or use_vol_restir_reuse
        if vol_reuse_needs_sample_dispatch:
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
            print("Note: --gpu --restir/--vol-restir-reuse batch mode renders "
                  "1 sample/pixel per dispatch (like --interactive-frames) "
                  "for full reservoir reuse, trading wavefront-batching "
                  "throughput for it.")
            gpu_gen_aux_buffers(handle, psc[unsafe_offset=0].camera_to_world, Int64(n_pixels))
            if use_restir:
                gpu_clear_restir(handle, Int64(n_pixels))
            if use_vol_restir_reuse:
                gpu_clear_restir_vol(handle, Int64(n_pixels))
        gpu_clear_film(handle, Int64(n_pixels))
        var t0_gpu = perf_counter_ns()
        if vol_reuse_needs_sample_dispatch:
            for si in range(spp):
                gpu_render_sample(
                    handle,
                    psc[unsafe_offset=0].camera_to_world,
                    Int32(si), psc[unsafe_offset=0].log2_spp, psc[unsafe_offset=0].n_base4_digits,
                    seed_dim0, seed_dim1,
                    UInt32(psc[unsafe_offset=0].rng_seed & UInt64(0xFFFFFFFF)),
                    UInt32(psc[unsafe_offset=0].rng_seed >> UInt64(32)),
                    Int64(n_pixels), psc[unsafe_offset=0].max_depth,
                    px_scale,
                    sample_clamp=_sample_clamp(psc),
                    use_restir=use_restir, frame_index=si,
                    use_vol_restir_reuse=use_vol_restir_reuse,
                )
                var elapsed = Float64(perf_counter_ns() - t0_gpu) / 1.0e9
                print(progress_str(si + 1, spp, elapsed, "spp"), end="\r")
        else:
            var si = 0
            while si < spp:
                var actual_batch = min(WAVEFRONT_BATCH, spp - si)
                gpu_render_wavefront(
                    handle,
                    psc[unsafe_offset=0].camera_to_world,
                    Int32(si), Int32(actual_batch),
                    psc[unsafe_offset=0].log2_spp, psc[unsafe_offset=0].n_base4_digits,
                    seed_dim0, seed_dim1,
                    UInt32(psc[unsafe_offset=0].rng_seed & UInt64(0xFFFFFFFF)),
                    UInt32(psc[unsafe_offset=0].rng_seed >> UInt64(32)),
                    Int64(n_pixels), psc[unsafe_offset=0].max_depth,
                    px_scale, _sample_clamp(psc),
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
        gpu_gen_aux_buffers(handle, psc[unsafe_offset=0].camera_to_world, Int64(n_pixels))
        gpu_atrous_denoise(handle, denoised_gpu.unsafe_ptr(), Int64(n_pixels),
                                Int32(spp), psc[unsafe_offset=0].film_iso, psc[unsafe_offset=0].film_max_comp,
                                apply_denoise=not no_denoise)
        apply_film_sensor(denoised_gpu.unsafe_ptr(), n_pixels, psc[unsafe_offset=0].film_exposuretime, psc[unsafe_offset=0].film_wb)
        gpu_download_albedo(handle, albedo_gpu.unsafe_ptr(), Int64(n_pixels))
        var inv_spp = Float32(1.0) / Float32(spp)
        for i in range(n_pixels * 3):
            albedo_gpu[i] *= inv_spp
        gpu_free_scene(handle)
        _ = write_image_cropwindow(denoised_gpu.unsafe_ptr(), fw, fh,
                                 psc[unsafe_offset=0].crop_x0, psc[unsafe_offset=0].crop_y0, psc[unsafe_offset=0].crop_x1, psc[unsafe_offset=0].crop_y1,
                                 psc[unsafe_offset=0].film_filename, Int32(32), Int32(32))
        var albedo_name_buf = unsafe_alloc[UInt8](11)
        var albedo_name_str = "albedo.exr"
        var anp = albedo_name_str.unsafe_ptr()
        for i in range(10): albedo_name_buf[unsafe_offset=i] = anp[unsafe_offset=i]
        albedo_name_buf[unsafe_offset=10] = UInt8(0)
        _ = write_image_cropwindow(albedo_gpu.unsafe_ptr(), fw, fh,
                                 psc[unsafe_offset=0].crop_x0, psc[unsafe_offset=0].crop_y0, psc[unsafe_offset=0].crop_x1, psc[unsafe_offset=0].crop_y1,
                                 albedo_name_buf.unsafe_origin_cast[MutUntrackedOrigin](), Int32(32), Int32(32))
        albedo_name_buf.unsafe_free()
        # denoised_gpu, albedo_gpu, and results freed automatically
        mojo_parsed_free(psc)
        return Int32(0)
    elif psc[unsafe_offset=0].prim_count == 0 and psc[unsafe_offset=0].sphere_count == 0:
        # Analytic spheres are NOT in prim_count -- they live in their own flat
        # psc[0].spheres array (see bvh.mojo's test_spheres), so a scene whose
        # only geometry is Shape "sphere" has prim_count == 0 and was rejected
        # outright. Same mesh-only assumption as the primId.type==4 bug class.
        print("Warning: scene has no geometry, skipping render")
        mojo_parsed_free(psc)
        return Int32(0)
    elif use_vcm:
        var sd = mojo_parsed_scene_descriptor(psc, spectral)
        var n_photons = _resolve_vcm_photons(vcm_photons, n_pixels)
        var resolved_vcm_spp = _resolve_vcm_spp(vcm_spp, psc[unsafe_offset=0].samples_per_pixel)
        var ret = vcm_render(psc, sd[unsafe_offset=0], resolved_vcm_spp, n_photons, no_denoise, verbose)
        sd.unsafe_free()
        mojo_parsed_free(psc)
        return ret
    elif use_sppm:
        var sd = mojo_parsed_scene_descriptor(psc, spectral)
        var resolved = _resolve_sppm_params(psc, sd[unsafe_offset=0], sppm_photons, sppm_radius)
        var ret = sppm_render(
            psc, sd[unsafe_offset=0],
            Int(sppm_passes), Int(resolved[0]), resolved[1],
            no_denoise, verbose,
        )
        sd.unsafe_free()
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

        if use_guide and psc[unsafe_offset=0].bvh_node_count > Int32(0):
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
            var root = psc[unsafe_offset=0].bvh_nodes[unsafe_offset=0]
            comptime N_GUIDE_THREADS: Int = 16
            comptime N_ITERATIONS: Int = 4
            var spp = psc[unsafe_offset=0].samples_per_pixel
            var n_iters = min(Int(spp), N_ITERATIONS)
            var base_spp = spp // Int32(n_iters)
            var tree = guide_create(Bounds3f(root.min, root.max))
            var write_guides = unsafe_alloc[GuideGrid](N_GUIDE_THREADS)
            print("Path guiding: " + String(n_iters) + " iterations x ~" + String(base_spp)
                + " spp, adaptive SD-tree, 16 private shards")
            var t0_g = perf_counter_ns()
            var offset = Int32(0)
            for it in range(n_iters):
                var iter_spp = base_spp
                if it == n_iters - 1:
                    iter_spp = spp - offset  # absorb any remainder into the last iteration
                for gi in range(N_GUIDE_THREADS):
                    write_guides[unsafe_offset=gi] = guide_clone_empty(tree)
                var sp_iter = TileSamplerParams_C(
                    sobolMatrices=sobol_matrices,
                    rngSeed=psc[unsafe_offset=0].rng_seed,
                    sobolSeed=Int32(0),
                    log2SamplesPerPixel=psc[unsafe_offset=0].log2_spp,
                    nBase4Digits=psc[unsafe_offset=0].n_base4_digits,
                    samplesPerPixel=iter_spp,
                    filterSigma=psc[unsafe_offset=0].filter_sigma,
                    filterSupportX=psc[unsafe_offset=0].filter_support_x,
                    filterSupportY=psc[unsafe_offset=0].filter_support_y,
                    filterNormX=psc[unsafe_offset=0].filter_norm_x,
                    filterNormY=psc[unsafe_offset=0].filter_norm_y,
                    filterWeight=psc[unsafe_offset=0].filter_weight,
                    filterType=psc[unsafe_offset=0].filter_type,
                    sampleIndexOffset=offset,
                )
                var sp_iter_ptr = OwnedPointer[TileSamplerParams_C](sp_iter)
                var guide_read = null_guide() if it == 0 else tree
                if it == 0:
                    # First iteration writes straight into `results` (like the
                    # old pilot pass) -- no accumulation add needed.
                    render_all_tiles(
                        psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world,
                        Int32(0), Int32(0), fw, fh,
                        Int32(32), Int32(32),
                        sp_iter_ptr.unsafe_ptr(), sd, results.unsafe_ptr(),
                        psc[unsafe_offset=0].max_depth, False,
                        guide_read, write_guides, N_GUIDE_THREADS)
                else:
                    var iter_buf = List[TileResult_C](capacity=n_pixels)
                    for _ in range(n_pixels): iter_buf.append(zero)
                    render_all_tiles(
                        psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world,
                        Int32(0), Int32(0), fw, fh,
                        Int32(32), Int32(32),
                        sp_iter_ptr.unsafe_ptr(), sd, iter_buf.unsafe_ptr(),
                        psc[unsafe_offset=0].max_depth, False,
                        guide_read, write_guides, N_GUIDE_THREADS)
                    for i in range(n_pixels):
                        var p = results.unsafe_ptr()[unsafe_offset=i]
                        var m = iter_buf.unsafe_ptr()[unsafe_offset=i]
                        results.unsafe_ptr()[unsafe_offset=i] = TileResult_C(
                            p.estimate + m.estimate,
                            p.albedo   + m.albedo,
                            p.filterWeight + m.filterWeight,
                            m.pixelX, m.pixelY)
                offset += iter_spp
                # render_all_tiles already merged shards [1..N-1] into [0].
                guide_merge(tree, write_guides[unsafe_offset=0])
                for gi in range(N_GUIDE_THREADS):
                    guide_free(write_guides[unsafe_offset=gi])
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
            write_guides.unsafe_free()
        else:
            # ── Standard single-call rendering ───────────────────────────────
            var sp = TileSamplerParams_C(
                sobolMatrices=sobol_matrices,
                rngSeed=psc[unsafe_offset=0].rng_seed,
                sobolSeed=Int32(0),
                log2SamplesPerPixel=psc[unsafe_offset=0].log2_spp,
                nBase4Digits=psc[unsafe_offset=0].n_base4_digits,
                samplesPerPixel=psc[unsafe_offset=0].samples_per_pixel,
                filterSigma=psc[unsafe_offset=0].filter_sigma,
                filterSupportX=psc[unsafe_offset=0].filter_support_x,
                filterSupportY=psc[unsafe_offset=0].filter_support_y,
                filterNormX=psc[unsafe_offset=0].filter_norm_x,
                filterNormY=psc[unsafe_offset=0].filter_norm_y,
                filterWeight=psc[unsafe_offset=0].filter_weight,
                filterType=psc[unsafe_offset=0].filter_type,
                sampleIndexOffset=Int32(0),
            )
            var sp_ptr = OwnedPointer[TileSamplerParams_C](sp)
            render_all_tiles(
                psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world,
                Int32(0), Int32(0), fw, fh,
                Int32(32), Int32(32),
                sp_ptr.unsafe_ptr(), sd, results.unsafe_ptr(), psc[unsafe_offset=0].max_depth,
                quiet=False, guide_read=null_guide(),
                write_guides=Pointer[GuideGrid, MutUntrackedOrigin].unsafe_dangling(), n_write_guides=0,
                use_restir=use_restir, use_gi=use_restir and use_restir_gi)
            # sp_ptr freed automatically

        # Unjittered normals and depth for edge-preserving denoising.
        var normals  = List[Float32](capacity=n_pixels * 3)
        var dept     = List[Float32](capacity=n_pixels)
        for _ in range(n_pixels * 3): normals.append(Float32(0))
        for _ in range(n_pixels):     dept.append(Float32(0))
        render_aux_buffers(
            psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world,
            Int32(0), Int32(0), fw, fh, sd,
            normals.unsafe_ptr(), dept.unsafe_ptr())
        sd.unsafe_free()

        # Normalize → beauty/albedo → denoise with normals+depth → write
        var beauty   = List[Float32](capacity=n_pixels * 3)
        var albedo   = List[Float32](capacity=n_pixels * 3)
        var denoised = List[Float32](capacity=n_pixels * 3)
        for _ in range(n_pixels * 3): beauty.append(Float32(0)); albedo.append(Float32(0)); denoised.append(Float32(0))
        normalize_film(results.unsafe_ptr(), Int32(n_pixels),
                            psc[unsafe_offset=0].film_iso, psc[unsafe_offset=0].film_max_comp,
                            beauty.unsafe_ptr(), albedo.unsafe_ptr())
        apply_film_sensor(beauty.unsafe_ptr(), n_pixels, psc[unsafe_offset=0].film_exposuretime, psc[unsafe_offset=0].film_wb)
        if no_denoise:
            # --no-denoise: write the normalized beauty directly (raw render).
            for i in range(n_pixels * 3): denoised[i] = beauty[i]
        else:
            denoise(beauty.unsafe_ptr(), albedo.unsafe_ptr(),
                    normals.unsafe_ptr(), dept.unsafe_ptr(),
                    fw, fh, denoised.unsafe_ptr(),
                    Int32(5), Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05))
        _ = write_image_cropwindow(denoised.unsafe_ptr(), fw, fh,
                                 psc[unsafe_offset=0].crop_x0, psc[unsafe_offset=0].crop_y0, psc[unsafe_offset=0].crop_x1, psc[unsafe_offset=0].crop_y1,
                                 psc[unsafe_offset=0].film_filename, Int32(32), Int32(32))
        var albedo_name_buf = unsafe_alloc[UInt8](11)
        var albedo_name_str = "albedo.exr"
        var anp2 = albedo_name_str.unsafe_ptr()
        for i in range(10): albedo_name_buf[unsafe_offset=i] = anp2[unsafe_offset=i]
        albedo_name_buf[unsafe_offset=10] = UInt8(0)
        _ = write_image_cropwindow(albedo.unsafe_ptr(), fw, fh,
                                 psc[unsafe_offset=0].crop_x0, psc[unsafe_offset=0].crop_y0, psc[unsafe_offset=0].crop_x1, psc[unsafe_offset=0].crop_y1,
                                 albedo_name_buf.unsafe_origin_cast[MutUntrackedOrigin](), Int32(32), Int32(32))
        albedo_name_buf.unsafe_free()
        # beauty, albedo, denoised, normals, dept freed automatically
    mojo_parsed_free(psc)
    return Int32(0)


def render_interactive(
    path: Pointer[UInt8, MutUntrackedOrigin],
    sobol: Pointer[UInt32, MutUntrackedOrigin],
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
    # Phase 7.3: volume-scatter TEMPORAL reuse (no spatial), CPU and GPU
    # both -- see project_restir_migration memory's "7.3" section and
    # parse_and_render's matching flag. INDEPENDENT of use_restir.
    use_vol_restir_reuse: Bool = False,
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
    if not _is_real_ptr[ParsedScene_Mojo](psc):
        print("Failed to parse scene")
        return
    if override_w > 0 or override_h > 0:
        var eff_w = override_w
        var eff_h = override_h
        # See parse_and_render's identical comment: --width/--height alone
        # used to be silently dropped, rendering at the scene's native
        # resolution instead. Derive the missing side from native aspect.
        if eff_w <= 0:
            eff_w = Int32(Int(eff_h) * Int(psc[unsafe_offset=0].film_w) / max(Int(psc[unsafe_offset=0].film_h), 1))
        if eff_h <= 0:
            eff_h = Int32(Int(eff_w) * Int(psc[unsafe_offset=0].film_h) / max(Int(psc[unsafe_offset=0].film_w), 1))
        resize_film(psc, eff_w, eff_h)

    mojo_apply_overrides(psc, spp_override, Int32(0), Int32(0), seed_override)

    var fw = psc[unsafe_offset=0].film_w
    var fh = psc[unsafe_offset=0].film_h
    var n_pixels = Int(fw) * Int(fh)

    var handle = Pointer[GpuSceneHandle, MutUntrackedOrigin].unsafe_dangling()
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
    var title_buf = unsafe_alloc[UInt8](title_len + 1)
    var ts = title_str.unsafe_ptr()
    for i in range(title_len):
        title_buf[unsafe_offset=i] = ts[unsafe_offset=i]
    title_buf[unsafe_offset=title_len] = UInt8(0)
    var v = viewer_create(fw, fh, title_buf, Int32(1) if fullscreen else Int32(0))
    title_buf.unsafe_free()
    if Int(v) == 0:
        print("Failed to create viewer window")
        if use_gpu:
            gpu_free_scene(handle)
        mojo_parsed_free(psc)
        return
    if not use_gpu and psc[unsafe_offset=0].prim_count == 0 and psc[unsafe_offset=0].sphere_count == 0:
        # See the batch-path guard above: analytic spheres are not counted in
        # prim_count.
        print("Warning: scene has no geometry, skipping render")
        viewer_destroy(v)
        mojo_parsed_free(psc)
        return

    var c2w = psc[unsafe_offset=0].camera_to_world
    var cam_buf = OwnedPointer[CameraState](CameraState(
        position=Point3f(c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14]),
        direction=Vec3f(c2w[unsafe_offset=8],  c2w[unsafe_offset=9],  c2w[unsafe_offset=10]),
        up=Vec3f(c2w[unsafe_offset=4],  c2w[unsafe_offset=5],  c2w[unsafe_offset=6]),
        cameraChanged=Int32(0),
    ))
    viewer_set_camera_state(v, cam_buf.unsafe_ptr())

    var c2w_buf = List[Float32](capacity=16)
    for i in range(16): c2w_buf.append(c2w[unsafe_offset=i])

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
    var sd           = Pointer[SceneDescriptor2_C, MutUntrackedOrigin].unsafe_dangling()
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
    var restir_buf_a = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling()
    var restir_buf_b = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling()
    var restir_read  = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling()
    var restir_write = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling()
    var gi_buf_a = Pointer[GIReservoir, MutUntrackedOrigin].unsafe_dangling()
    var gi_buf_b = Pointer[GIReservoir, MutUntrackedOrigin].unsafe_dangling()
    var gi_read  = Pointer[GIReservoir, MutUntrackedOrigin].unsafe_dangling()
    var gi_write = Pointer[GIReservoir, MutUntrackedOrigin].unsafe_dangling()
    # Phase 6: ping-ponged SMSReservoir buffers, same race-free scheme as
    # restir_buf_a/b and gi_buf_a/b above -- SMS_MAX_FINALIZED_WEIGHT
    # (shading.mojo) was applied proactively from the start (not
    # discovered via a live bug this time, unlike DI/GI's own history),
    # so no separate bug-fix narrative applies here.
    var sms_buf_a = Pointer[SMSReservoir, MutUntrackedOrigin].unsafe_dangling()
    var sms_buf_b = Pointer[SMSReservoir, MutUntrackedOrigin].unsafe_dangling()
    var sms_read  = Pointer[SMSReservoir, MutUntrackedOrigin].unsafe_dangling()
    var sms_write = Pointer[SMSReservoir, MutUntrackedOrigin].unsafe_dangling()
    # Phase 7.3: ping-ponged VolReservoir buffers, same race-free scheme as
    # restir_buf_a/b above -- INDEPENDENT of use_restir, matching
    # use_sms_restir's own independence (this is the medium sampler's own
    # NEE, not DI's). TEMPORAL ONLY: no G-buffer pointers are threaded
    # through (see render_all_tiles's vol_io construction below), so
    # vol_temporal_spatial_combine's spatial pass self-disables, mirroring
    # the GPU wiring's own scope exactly (commit 1685154c).
    var vol_buf_a = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling()
    var vol_buf_b = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling()
    var vol_read  = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling()
    var vol_write = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling()
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
        filterSigma=psc[unsafe_offset=0].filter_sigma,
        filterSupportX=psc[unsafe_offset=0].filter_support_x,
        filterSupportY=psc[unsafe_offset=0].filter_support_y,
        filterNormX=psc[unsafe_offset=0].filter_norm_x,
        filterNormY=psc[unsafe_offset=0].filter_norm_y,
        filterWeight=psc[unsafe_offset=0].filter_weight,
        filterType=psc[unsafe_offset=0].filter_type,
        sampleIndexOffset=Int32(0),
    ))

    if use_gpu:
        gpu_clear_film(handle, Int64(n_pixels))
        if use_restir:
            gpu_clear_restir(handle, Int64(n_pixels))
        if use_vol_restir_reuse:
            gpu_clear_restir_vol(handle, Int64(n_pixels))
        gpu_gen_aux_buffers(handle, psc[unsafe_offset=0].camera_to_world, Int64(n_pixels))
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
        if use_restir or use_restir_gi or use_sms_restir or use_vol_restir_reuse:
            for _ in range(n_pixels):
                material_id_int.append(Int32(-1))
            for _ in range(n_pixels * 3):
                world_pos_int.append(Float32(0))
        if use_restir:
            restir_buf_a = unsafe_alloc[DIReservoir](n_pixels)
            restir_buf_b = unsafe_alloc[DIReservoir](n_pixels)
            for i in range(n_pixels):
                restir_buf_a[unsafe_offset=i] = di_reservoir_init()
                restir_buf_b[unsafe_offset=i] = di_reservoir_init()
            restir_read = restir_buf_a
            restir_write = restir_buf_b
            if use_restir_gi:
                gi_buf_a = unsafe_alloc[GIReservoir](n_pixels)
                gi_buf_b = unsafe_alloc[GIReservoir](n_pixels)
                for i in range(n_pixels):
                    gi_buf_a[unsafe_offset=i] = gi_reservoir_init()
                    gi_buf_b[unsafe_offset=i] = gi_reservoir_init()
                gi_read = gi_buf_a
                gi_write = gi_buf_b
        if use_sms_restir:
            # Independent of use_restir (unlike use_restir_gi above) --
            # SMS-ReSTIR only ever touches _nee_area_lights' own glass-
            # probing branch, not ReSTIR DI's reservoir path.
            sms_buf_a = unsafe_alloc[SMSReservoir](n_pixels)
            sms_buf_b = unsafe_alloc[SMSReservoir](n_pixels)
            for i in range(n_pixels):
                sms_buf_a[unsafe_offset=i] = sms_reservoir_init()
                sms_buf_b[unsafe_offset=i] = sms_reservoir_init()
            sms_read = sms_buf_a
            sms_write = sms_buf_b
        if use_vol_restir_reuse:
            # Independent of use_restir, same reasoning as use_sms_restir
            # above -- the medium sampler's own NEE.
            vol_buf_a = unsafe_alloc[VolReservoir](n_pixels)
            vol_buf_b = unsafe_alloc[VolReservoir](n_pixels)
            for i in range(n_pixels):
                vol_buf_a[unsafe_offset=i] = vol_reservoir_init()
                vol_buf_b[unsafe_offset=i] = vol_reservoir_init()
            vol_read = vol_buf_a
            vol_write = vol_buf_b

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
                    if use_vol_restir_reuse:
                        gpu_clear_restir_vol(handle, Int64(n_pixels))
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
                            restir_buf_a[unsafe_offset=i] = di_reservoir_init()
                            restir_buf_b[unsafe_offset=i] = di_reservoir_init()
                        if use_restir_gi:
                            for i in range(n_pixels):
                                gi_buf_a[unsafe_offset=i] = gi_reservoir_init()
                                gi_buf_b[unsafe_offset=i] = gi_reservoir_init()
                    if use_sms_restir:
                        # Same identity-reprojection invalidation rule,
                        # independent of use_restir.
                        for i in range(n_pixels):
                            sms_buf_a[unsafe_offset=i] = sms_reservoir_init()
                            sms_buf_b[unsafe_offset=i] = sms_reservoir_init()
                    if use_vol_restir_reuse:
                        for i in range(n_pixels):
                            vol_buf_a[unsafe_offset=i] = vol_reservoir_init()
                            vol_buf_b[unsafe_offset=i] = vol_reservoir_init()
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
                Int64(n_pixels), psc[unsafe_offset=0].max_depth,
                use_restir=use_restir, frame_index=frame_count,
                use_vol_restir_reuse=use_vol_restir_reuse,
            )
            frame_count += 1
            gpu_atrous_denoise(handle, denoised.unsafe_ptr(), Int64(n_pixels),
                                    Int32(frame_count),
                                    psc[unsafe_offset=0].film_iso, psc[unsafe_offset=0].film_max_comp)
            apply_film_sensor(denoised.unsafe_ptr(), n_pixels, psc[unsafe_offset=0].film_exposuretime, psc[unsafe_offset=0].film_wb)
        else:
            sp_int[] = TileSamplerParams_C(
                sobolMatrices=sobol,
                rngSeed=UInt64(frame_count),
                sobolSeed=Int32(frame_count % 65536),
                log2SamplesPerPixel=Int32(0),
                nBase4Digits=Int32(1),
                samplesPerPixel=Int32(1),
                filterSigma=psc[unsafe_offset=0].filter_sigma,
                filterSupportX=psc[unsafe_offset=0].filter_support_x,
                filterSupportY=psc[unsafe_offset=0].filter_support_y,
                filterNormX=psc[unsafe_offset=0].filter_norm_x,
                filterNormY=psc[unsafe_offset=0].filter_norm_y,
                filterWeight=psc[unsafe_offset=0].filter_weight,
                filterType=psc[unsafe_offset=0].filter_type,
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
                if use_restir or use_restir_gi or use_sms_restir or use_vol_restir_reuse:
                    render_aux_buffers(
                        psc[unsafe_offset=0].raster_to_camera, c2w_buf.unsafe_ptr(),
                        Int32(0), Int32(0), fw, fh, sd,
                        normals_int.unsafe_ptr(), depth_int.unsafe_ptr(),
                        world_pos_out=world_pos_int.unsafe_ptr(),
                        material_id_out=material_id_int.unsafe_ptr())
                else:
                    render_aux_buffers(
                        psc[unsafe_offset=0].raster_to_camera, c2w_buf.unsafe_ptr(),
                        Int32(0), Int32(0), fw, fh, sd,
                        normals_int.unsafe_ptr(), depth_int.unsafe_ptr())
            var restir_io = reservoir_io_null()
            if use_restir:
                restir_io = ReservoirIO(
                    read=restir_read, write=restir_write,
                    gbuf_normal=normals_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                    gbuf_depth=depth_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                    gbuf_material_id=material_id_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                    gbuf_world_pos=world_pos_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                    frame_w=fw, frame_h=fh,
                )
            var gi_io = gi_reservoir_io_null()
            if use_restir_gi:
                gi_io = GIReservoirIO(
                    read=gi_read, write=gi_write,
                    gbuf_normal=normals_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                    gbuf_depth=depth_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                    gbuf_material_id=material_id_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                    gbuf_world_pos=world_pos_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                    frame_w=fw, frame_h=fh,
                )
            var sms_io = sms_reservoir_io_null()
            if use_sms_restir:
                sms_io = SMSReservoirIO(
                    read=sms_read, write=sms_write,
                    gbuf_normal=normals_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                    gbuf_depth=depth_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                    gbuf_material_id=material_id_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                    gbuf_world_pos=world_pos_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                    frame_w=fw, frame_h=fh,
                )
            # TEMPORAL + SPATIAL (2026-09-08): gbuf_depth/gbuf_world_pos now
            # wired (same depth_int/world_pos_int buffers DI/GI/SMS already
            # populate above), so vol_temporal_spatial_combine's spatial pass
            # (VOL_SPATIAL_NEIGHBORS=4, restir_vol.mojo) is live, not just
            # unit-tested. See project_restir_migration memory's "Spatial
            # reuse for volumes" section for the verification methodology
            # and result -- DI's own spatial reuse was found not worth it,
            # but that was never actually tested for the volumetric case
            # until now.
            var vol_io = vol_reservoir_io_null()
            if use_vol_restir_reuse:
                vol_io.read = vol_read
                vol_io.write = vol_write
                vol_io.gbuf_depth = depth_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
                vol_io.gbuf_world_pos = world_pos_int.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
                vol_io.frame_w = fw
                vol_io.frame_h = fh
            render_all_tiles(
                psc[unsafe_offset=0].raster_to_camera, c2w_buf.unsafe_ptr(),
                Int32(0), Int32(0), fw, fh,
                Int32(32), Int32(32),
                sp_int.unsafe_ptr(), sd, results.unsafe_ptr(), psc[unsafe_offset=0].max_depth, True,
                guide_read=null_guide(), write_guides=Pointer[GuideGrid, MutUntrackedOrigin].unsafe_dangling(),
                n_write_guides=0, use_restir=use_restir, frame_w=fw, restir_io=restir_io,
                use_gi=use_restir_gi, gi_io=gi_io,
                use_sms_restir=use_sms_restir, sms_io=sms_io, vol_io=vol_io)
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
            if use_vol_restir_reuse:
                # Same synchronous-completion reasoning as restir above.
                var vol_tmp = vol_read
                vol_read = vol_write
                vol_write = vol_tmp
            var beauty_frame = List[Float32](capacity=n_pixels * 3)
            var albedo_frame = List[Float32](capacity=n_pixels * 3)
            for _ in range(n_pixels * 3): beauty_frame.append(Float32(0)); albedo_frame.append(Float32(0))
            normalize_film(results.unsafe_ptr(), Int32(n_pixels),
                                psc[unsafe_offset=0].film_iso, psc[unsafe_offset=0].film_max_comp,
                                beauty_frame.unsafe_ptr(), albedo_frame.unsafe_ptr())
            apply_film_sensor(beauty_frame.unsafe_ptr(), n_pixels, psc[unsafe_offset=0].film_exposuretime, psc[unsafe_offset=0].film_wb)
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
                                 psc[unsafe_offset=0].film_filename, Int32(32), Int32(32))

    # results, beauty, albedo, denoised, c2w_buf, cam_buf, sp_int freed automatically
    if use_gpu:
        gpu_free_scene(handle)
    else:
        # accum, albedo_acc, sd freed automatically (accum/albedo_acc are List)
        sd.unsafe_free()
        if use_restir:
            restir_buf_a.unsafe_free()
            restir_buf_b.unsafe_free()
            if use_restir_gi:
                gi_buf_a.unsafe_free()
                gi_buf_b.unsafe_free()
        if use_sms_restir:
            sms_buf_a.unsafe_free()
            sms_buf_b.unsafe_free()
        if use_vol_restir_reuse:
            vol_buf_a.unsafe_free()
            vol_buf_b.unsafe_free()
    mojo_parsed_free(psc)
    viewer_destroy(v)
