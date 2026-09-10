from std.sys import has_accelerator, has_nvidia_gpu_accelerator
from std.sys.info import size_of
from std.gpu import block_idx, thread_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from std.atomic import Atomic
from std.math import ceildiv, sqrt, cos, sin, log, exp
from std.memory import alloc, memcpy
from .geometry import RGB, Point3f, Point2f, Vec3f, vec3f, point3f, store_vec3, sphere_outward_normal, Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Material_C, AreaLight_C, Sphere_C, Curve_C, CURVE_N_PIECES, CURVE_DEFER_K, curve_piece_endpoints, _curve_perp_axis, intersect_curve, DistantLight_C, PointLight_C, InfiniteLight_C, PathState_C, GpuTexture_C, NormalSlopeMap_C, ShadowTask_C, LightSampler_C, light_sampler_sample, MatKind, Medium_C, MediumInterface_C, Grid_C, grid_sample_density, NvdbGrid_C, nvdb_sample_density, nvdb_ray_range, grid_ray_range, nvdb_index_ray, nvdb_node_exit_t, nvdb_majorant_at_world, hg_phase, hg_sample, blackbody_rgb, Instance_C, MeasuredBRDF_C, dot, cross, INV_PI, INV_FOUR_PI, _is_real_ptr
from std.ffi import external_call
from .bvh import BVH2Node, SceneDescriptor2_C, traverse_bvh2_core, traverse_bvh2_core_defer_curves, any_hit_bvh2_core, test_spheres, LightSample, _sample_infinite_light_nee, _sample_distant_light_nee, _sample_point_light_nee, _sample_sphere_light_nee
from .transform import transform_normal_by_instance
from std.atomic import Atomic
from .rng import PCG32
from .shading import shade_core, shade_nee_core, ShadeContext, LightContext, shade_diffuse, shade_coated_diffuse, shade_diffuse_transmission, shade_mix, shade_conductor, shade_dielectric, shade_thin_dielectric, shade_coated_conductor, shade_hair, shade_interface, shade_measured, GIPendingX1
from .guide import null_guide
from .restir_di import DIReservoir, di_reservoir_init, ReservoirIO, reservoir_io_null
from .restir_vol import (
    vol_reservoir_init, vol_target_pdf, VOL_TR_UNIT, VOL_RIS_CANDIDATES,
    VOL_RIS_DISTANCE,
    VolReservoir, VolReservoirIO, vol_reservoir_io_null,
    vol_temporal_spatial_combine, VolShiftMode,
)
from .reservoir import reservoir_update, reservoir_finalize
from .restir_gi import gi_reservoir_io_null
from .postprocess import _firefly_clamp_pixel, _atrous_tap_weight, _atrous_spatial_weight
from .sampling import power_heuristic, encode_morton2, sobol_get_sample_index, sobol_sample, gaussian_sample_1d, derive_pcg_seeds, gen_primary_ray_state
from .spectrum import SampledWavelengths, SpectralSample, SpectralHandle, null_spectral_handle, rgb_illuminant_to_spectral_sample, rgb_bands_to_spectral_sample, spectral_sample_to_rgb, spec_refl_unbounded
from .vulkaninterop import VulkanInteropRtSceneHandle, vulkaninterop_rt_trace
from max.gpu.host._nvidia_cuda import CUDA

# Number of samples per pixel processed together in one wavefront bounce loop.
# path_buf and inter_buf are pre-allocated at n_pixels × WAVEFRONT_BATCH.
comptime WAVEFRONT_BATCH: Int = 8

def _cstr_eq(a: UnsafePointer[UInt8, MutExternalOrigin], b: UnsafePointer[UInt8, MutExternalOrigin]) -> Bool:
    var i = 0
    while True:
        var ca = a[i]
        var cb = b[i]
        if ca != cb:
            return False
        if ca == UInt8(0):
            return True
        i += 1

# GPU scene handle — holds DeviceContext and device-resident scene buffers.
# Allocated on the heap, returned as an opaque pointer.
@fieldwise_init
struct GpuSceneHandle(Movable):
    var ctx: DeviceContext
    var bvh2Nodes_buf: DeviceBuffer[DType.uint8]
    var primIds_buf: DeviceBuffer[DType.uint8]
    # Object instancing (see [[project_object_instancing]]/geometry.mojo's
    # Instance_C docs): one device buffer per BLAS (kept alive here), plus two
    # small "array of device pointers" buffers so a kernel's
    # blasNodesArr[i]/blasPrimIdsArr[i] resolves to the right BLAS's buffer.
    var blas_nodes_bufs: List[DeviceBuffer[DType.uint8]]
    var blas_primids_bufs: List[DeviceBuffer[DType.uint8]]
    var blas_nodes_ptrs_buf: DeviceBuffer[DType.uint8]
    var blas_primids_ptrs_buf: DeviceBuffer[DType.uint8]
    var n_blas: Int
    var instances_buf: DeviceBuffer[DType.uint8]
    var n_instances: Int
    var meshes_buf: DeviceBuffer[DType.uint8]
    var mesh_count: Int
    var materials_buf: DeviceBuffer[DType.uint8]
    var material_count: Int
    # Keep all per-mesh device buffers alive
    var points_bufs: List[DeviceBuffer[DType.uint8]]
    var faceIndices_bufs: List[DeviceBuffer[DType.uint8]]
    var vertexIndices_bufs: List[DeviceBuffer[DType.uint8]]
    var uv_bufs: List[DeviceBuffer[DType.uint8]]
    var nrm_bufs: List[DeviceBuffer[DType.uint8]]
    var tex_data_bufs: List[DeviceBuffer[DType.uint8]]
    var textures_buf: DeviceBuffer[DType.uint8]  # array of GpuTexture_C
    var n_textures: Int
    var area_lights_buf: DeviceBuffer[DType.uint8]  # n_lights × sizeof(AreaLight_C) = 24
    var n_area_lights: Int
    var spheres_buf: DeviceBuffer[DType.uint8]   # n_spheres × sizeof(Sphere_C) = 36
    var n_spheres: Int
    var curves_buf: DeviceBuffer[DType.uint8]    # n_curves × sizeof(Curve_C)
    var n_curves: Int
    # Curve-divergence-mitigation scratch (see traverse_bvh2_core_defer_curves).
    # Only meaningfully sized when n_curves > 0; otherwise 1-byte dummies.
    var curve_cand_prim_buf: DeviceBuffer[DType.uint8]      # n_pixels×WAVEFRONT_BATCH×CURVE_DEFER_K × Int32
    var curve_cand_count_buf: DeviceBuffer[DType.uint8]     # n_pixels×WAVEFRONT_BATCH × Int32
    # Each ray's start offset into curve_cand_prim_buf. The CUDA-native path
    # (traverse_bvh2_core_defer_curves) always writes tid*CURVE_DEFER_K-
    # strided candidates, so this is initialized ONCE to that same formula
    # (init_curve_cand_offset_gpu) and never touched again for CUDA-only
    # rendering. The Vulkan RT path OVERWRITES it every bounce with real,
    # uncapped pool offsets from its own count-then-place scheme (see
    # intersect_batch.comp) -- resolve_curve_candidates_gpu always reads
    # through this indirection so one indexing scheme serves both backends.
    var curve_cand_offset_buf: DeviceBuffer[DType.uint8]    # n_pixels×WAVEFRONT_BATCH × Int32
    var curve_compact_path_buf: DeviceBuffer[DType.uint8]   # n_pixels×WAVEFRONT_BATCH × Int32
    var curve_compact_counter_buf: DeviceBuffer[DType.uint8] # 1 × Int32
    var distant_lights_buf: DeviceBuffer[DType.uint8]  # n_distant × sizeof(DistantLight_C) = 32
    var n_distant_lights: Int
    var point_lights_buf: DeviceBuffer[DType.uint8]    # n_point × sizeof(PointLight_C) = 16
    var n_point_lights: Int
    var light_sampler_buf: DeviceBuffer[DType.uint8]   # (n_area+1) × sizeof(Float32) CDF
    var n_light_sampler: Int                           # n_area lights (CDF has n+1 entries)
    var infinite_lights_buf: DeviceBuffer[DType.uint8]  # n_infinite × sizeof(InfiniteLight_C) = 48
    var il_pixels_bufs: List[DeviceBuffer[DType.uint8]] # per-light HDR pixel data on GPU
    var il_cdf_bufs: List[DeviceBuffer[DType.uint8]]    # per-light 2D CDF on GPU
    var il_w2l_bufs: List[DeviceBuffer[DType.uint8]]    # per-light world_to_light matrix on GPU
    var n_infinite_lights: Int
    var mediums_buf: DeviceBuffer[DType.uint8]        # n_mediums × sizeof(Medium_C)
    var n_mediums: Int
    # True if any medium is a `Material "subsurface"` interior. Read once on
    # the host to size the bounce-round budget -- an interior random walk
    # needs far more rounds than an ordinary path (see the round-count block
    # in gpu_render_sample and Medium_C.is_sss).
    var has_sss_medium: Bool
    var medium_ifaces_buf: DeviceBuffer[DType.uint8]  # n_medium_ifaces × sizeof(MediumInterface_C)
    var n_medium_ifaces: Int
    var grids_buf: DeviceBuffer[DType.uint8]          # n_grids × sizeof(Grid_C); Grid_C.density points into grid_density_bufs
    var n_grids: Int
    var grid_density_bufs: List[DeviceBuffer[DType.uint8]]  # kept alive; one per grid's density array
    var nvdb_grids_buf: DeviceBuffer[DType.uint8]     # n_nvdb_grids × sizeof(NvdbGrid_C); NvdbGrid_C.blob points into nvdb_blob_bufs
    var n_nvdb_grids: Int
    var nvdb_blob_bufs: List[DeviceBuffer[DType.uint8]]  # kept alive; one per grid's decompressed .nvdb blob
    var measured_brdfs_buf: DeviceBuffer[DType.uint8]  # n_measured_brdfs × sizeof(MeasuredBRDF_C); each entry's 12 pointer fields point into measured_field_bufs
    var n_measured_brdfs: Int
    var measured_field_bufs: List[DeviceBuffer[DType.uint8]]  # kept alive; 12 sub-array buffers per measured material
    # Persistent render buffers — sized for n_pixels × WAVEFRONT_BATCH (wavefront pass)
    # gpu_render_sample (interactive) only uses the first n_pixels slots.
    var path_buf: DeviceBuffer[DType.uint8]   # n_pixels × WAVEFRONT_BATCH × size_of[PathState_C]()
    var inter_buf: DeviceBuffer[DType.uint8]  # n_pixels × WAVEFRONT_BATCH × 48
    var film_buf: DeviceBuffer[DType.uint8]          # n_pixels × 3 × Float32 = 12 bytes
    var albedo_film_buf: DeviceBuffer[DType.uint8]   # n_pixels × 3 × Float32 = 12 bytes
    # À-trous wavelet denoiser buffers (interactive GPU path)
    var atrous_ping_buf: DeviceBuffer[DType.uint8]     # n_pixels × 12 — beauty ping, input to pass 0
    var atrous_pong_buf: DeviceBuffer[DType.uint8]     # n_pixels × 12 — ping-pong working buffer
    var atrous_albedo_buf: DeviceBuffer[DType.uint8]   # n_pixels × 12 — normalized albedo, constant across passes
    var atrous_variance_buf: DeviceBuffer[DType.uint8] # n_pixels × 4  — spatial luminance variance
    var atrous_normals_buf: DeviceBuffer[DType.uint8]  # n_pixels × 12 — unjittered geometric normals
    var atrous_depth_buf: DeviceBuffer[DType.uint8]    # n_pixels × 4  — unjittered first-hit depth
    var atrous_curve_mask_buf: DeviceBuffer[DType.uint8] # n_pixels × 4 — 1.0 if first hit was a curve (hair), else 0.0
    # G-buffer extension (Phase 0.3, docs/A2_restir_migration_plan.md): world
    # position + material ID at the primary hit. Not consumed anywhere yet --
    # normals+depth alone accept too many invalid shift-mapping candidates for
    # future ReSTIR spatial/temporal reuse, which is what these are for.
    var gbuf_worldpos_buf: DeviceBuffer[DType.uint8]     # n_pixels × 12 — world-space hit point; meaningless where depth==1e38 (miss)
    var gbuf_material_id_buf: DeviceBuffer[DType.uint8]  # n_pixels × 4  — Int32 material index of first hit, -1 on miss
    # ReSTIR DI reservoirs (Phase 2, --restir). Double-buffered and
    # ping-ponged per frame: spatial reuse reads NEIGHBOUR pixels, which other
    # threads are concurrently writing, so reads must come from the previous
    # frame's finished buffer. Same rule the CPU path follows, for the same
    # reason -- see pipeline.mojo's restir_buf_a/b.
    var restir_a_buf: DeviceBuffer[DType.uint8]     # n_pixels × sizeof(DIReservoir)
    var restir_b_buf: DeviceBuffer[DType.uint8]     # n_pixels × sizeof(DIReservoir)
    # Phase 7.3 (docs/A2_restir_migration_plan.md, project_restir_migration
    # memory): volume-scatter temporal reuse, TEMPORAL-ONLY (no spatial --
    # gbuf pointers are never wired to VolReservoirIO here, which self-
    # disables vol_temporal_spatial_combine's spatial pass). Same ping-pong
    # rule as restir_a_buf/b above.
    var restir_vol_a_buf: DeviceBuffer[DType.uint8] # n_pixels × sizeof(VolReservoir)
    var restir_vol_b_buf: DeviceBuffer[DType.uint8] # n_pixels × sizeof(VolReservoir)
    # Per-pixel "already combined this frame" guard (one Int8/pixel), reset
    # to 0 every gpu_render_sample call before its bounce-round loop starts
    # -- NOT ping-ponged, NOT persisted across frames, unlike the pair
    # above. See _sample_medium_core's vol_used comment for why this exists.
    var restir_vol_used_buf: DeviceBuffer[DType.uint8] # n_pixels × sizeof(Int8)
    var shadow_buf: DeviceBuffer[DType.uint8]       # n_pixels × WAVEFRONT_BATCH × sizeof(ShadowTask_C) = 48 -- must match path_buf/inter_buf sizing (gpu_render_sample only uses the first n_pixels slots; gpu_render_wavefront's _gpu_bounce_kernels call indexes up to n_pixels × WAVEFRONT_BATCH)
    var active_count_buf: DeviceBuffer[DType.uint8] # 1 × Int32
    var active_idx_buf: DeviceBuffer[DType.uint8]   # n_pixels × Int32
    var n_pixels: Int
    # Camera and sampling data for GPU-side ray generation
    var sobol_buf: DeviceBuffer[DType.uint8]  # 1024 dims × 52 UInt32 = 212992 bytes
    var r2c_buf: DeviceBuffer[DType.uint8]    # raster_to_camera: 16 Float32 = 64 bytes
    var c2w_buf: DeviceBuffer[DType.uint8]    # camera_to_world: 16 Float32 = 64 bytes (updated each frame)
    var filter_sigma: Float32
    var filter_support_x: Float32
    var filter_support_y: Float32
    var filter_norm_x: Float32
    var filter_norm_y: Float32
    var filter_type: Int32
    var fw: Int
    var fh: Int
    # Staged spectral rendering rollout (Stage 2c-1, see
    # project_spectral_rendering memory) — device-side twin of the host
    # SpectralHandle; spectral_res=0 means no real table was uploaded (dummy
    # 1-element buffers, BDPT/SPPM GPU dispatch, Stage 3/4 not wired yet).
    var spectral_coeffs_buf: DeviceBuffer[DType.uint8]
    var spectral_cie_x_buf:  DeviceBuffer[DType.uint8]
    var spectral_cie_y_buf:  DeviceBuffer[DType.uint8]
    var spectral_cie_z_buf:  DeviceBuffer[DType.uint8]
    var spectral_d65_buf:    DeviceBuffer[DType.uint8]
    var spectral_res: Int

def gpu_available() -> Bool:
    return has_accelerator()

# sRGB<->linear on a single byte, used ONLY to box-filter the u8 mip
# pyramid's tail levels correctly: averaging raw sRGB bytes directly (gamma
# space) is wrong and visibly shifts brightness/contrast at any real
# minification -- decode, average in linear, re-encode, matching what the
# float-texture path already does implicitly (it decodes once at load,
# before its own linear-space box filter).
@always_inline
def _srgb_byte_to_linear(c: UInt8) -> Float32:
    var x = Float32(c) * Float32(1.0 / 255.0)
    if x <= Float32(0.04045):
        return x / Float32(12.92)
    return Float32(((x + Float32(0.055)) / Float32(1.055)) ** Float32(2.4))

@always_inline
def _linear_to_srgb_byte(x: Float32) -> UInt8:
    if x <= Float32(0.0): return UInt8(0)
    if x >= Float32(1.0): return UInt8(255)
    var enc: Float32
    if x <= Float32(0.0031308):
        enc = Float32(12.92) * x
    else:
        enc = Float32(1.055) * (x ** Float32(1.0 / 2.4)) - Float32(0.055)
    var v = Int(enc * Float32(255.0) + Float32(0.5))
    if v < 0: v = 0
    if v > 255: v = 255
    return UInt8(v)

# Uploads count elements of T from a host array into a fresh device buffer
# (>= 1 elem so a zero-count scene never creates a 0-byte device buffer,
# which crashes on use/free). Shared by gpu_upload_scene's ~11 near-identical
# "size, create buffer, memcpy if non-empty" upload sites below.
def _gpu_upload_array[T: AnyType](
    ctx: DeviceContext,
    src: UnsafePointer[T, MutExternalOrigin],
    count: Int,
) raises -> DeviceBuffer[DType.uint8]:
    var n_bytes = max(count, 1) * size_of[T]()
    var buf = ctx.enqueue_create_buffer[DType.uint8](n_bytes)
    if count > 0:
        with buf.map_to_host() as host_buf:
            memcpy(dest=host_buf.unsafe_ptr(), src=src.bitcast[UInt8](), count=count * size_of[T]())
    return buf^

def gpu_upload_scene[Ompc: Origin[mut=True], Ofic: Origin[mut=True], Ovic: Origin[mut=True], Ouv: Origin[mut=True], Onv: Origin[mut=True]](
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    bvh2NodesCount: Int64,
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    primIdsCount: Int64,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    blasNodeCounts: UnsafePointer[Int32, MutExternalOrigin],
    blasPrimidCounts: UnsafePointer[Int32, MutExternalOrigin],
    blasCount: Int64,
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    instanceCount: Int64,
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    meshCount: Int64,
    meshPointsCounts: UnsafePointer[Int64, Ompc],
    meshFaceIndicesCounts: UnsafePointer[Int64, Ofic],
    meshVertexIndicesCounts: UnsafePointer[Int64, Ovic],
    meshUvNVerts: UnsafePointer[Int64, Ouv],
    meshNrmNVerts: UnsafePointer[Int64, Onv],
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin],
    n_tex: Int32,
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    materialCount: Int64,
    areaLights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    areaLightCount: Int64,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    sphereCount: Int64,
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    curveCount: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutExternalOrigin],
    distantLightCount: Int64,
    pointLights: UnsafePointer[PointLight_C, MutExternalOrigin],
    pointLightCount: Int64,
    lightSamplerCdf: UnsafePointer[Float32, MutExternalOrigin],
    lightSamplerN: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutExternalOrigin],
    infiniteLightCount: Int64,
    mediums: UnsafePointer[Medium_C, MutExternalOrigin],
    mediumCount: Int64,
    medium_ifaces: UnsafePointer[MediumInterface_C, MutExternalOrigin],
    medium_iface_count: Int64,
    grids: UnsafePointer[Grid_C, MutExternalOrigin],
    gridCount: Int64,
    nvdbGrids: UnsafePointer[NvdbGrid_C, MutExternalOrigin],
    nvdbGridCount: Int64,
    measured_brdfs: UnsafePointer[MeasuredBRDF_C, MutExternalOrigin],
    measuredBrdfCount: Int64,
    n_pixels: Int64,
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
    r2c: UnsafePointer[Float32, MutExternalOrigin],
    c2w_init: UnsafePointer[Float32, MutExternalOrigin],
    filter_sigma: Float32, filter_support_x: Float32, filter_support_y: Float32,
    filter_norm_x: Float32, filter_norm_y: Float32,
    filter_type: Int32,
    fw: Int32, fh: Int32,
    # Decomposed, NOT a single by-value `spectral: SpectralHandle` param --
    # see spectrum.mojo's long comment on the confirmed by-value SpectralHandle
    # miscompilation. This host function is part of the GPU-enabled
    # compilation unit (--target-accelerator build), and passing the handle
    # by value here reproduced the exact same corruption class (spectral_res
    # read back as 0, coeffs pointer read back as a tiny garbage address) --
    # see project_priority_backlog memory item 3 GPU-black-background bug.
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
) -> UnsafePointer[GpuSceneHandle, MutExternalOrigin]:
    comptime if has_accelerator():
        try:
            var ctx = DeviceContext()

            # Check GPU memory
            var mem_info = ctx.get_memory_info()
            var free_bytes = mem_info[0]
            var total_bytes = mem_info[1]

            # Guard against zero-size device buffers (scene with no geometry):
            # a 0-byte enqueue_create_buffer yields a misaligned/invalid device
            # pointer that crashes on use and on free. Allocate at least 1 elem.
            var bvh_bytes = max(Int(bvh2NodesCount), 1) * size_of[BVH2Node]()
            var prim_bytes = max(Int(primIdsCount), 1) * size_of[PrimId_C]()
            var mesh_struct_bytes = max(Int(meshCount), 1) * size_of[TriangleMesh_C]()
            var material_struct_bytes = max(Int(materialCount), 1) * size_of[Material_C]()

            # Estimate total mesh data
            var mesh_data_bytes = 0
            for i in range(Int(meshCount)):
                mesh_data_bytes += Int(meshPointsCounts[i]) * 4       # Float32
                mesh_data_bytes += Int(meshFaceIndicesCounts[i]) * 8  # Int64
                mesh_data_bytes += Int(meshVertexIndicesCounts[i]) * 8 # Int64

            var total_scene_bytes = bvh_bytes + prim_bytes + mesh_struct_bytes + mesh_data_bytes
            var free_mb = free_bytes // (1024 * 1024)
            var scene_mb = total_scene_bytes // (1024 * 1024)

            print("GPU: " + String(ctx.name()) + " — " + String(free_mb) + " MB free")

            if total_scene_bytes > Int(free_bytes):
                print("WARNING: Scene (" + String(scene_mb) + " MB) may exceed available GPU memory (" + String(free_mb) + " MB)!")

            # Upload BVH nodes and prim IDs (copy only the real bytes; the
            # buffer may be a 1-element placeholder when the scene has no
            # geometry).
            var bvh_buf = _gpu_upload_array[BVH2Node](ctx, bvh2Nodes, Int(bvh2NodesCount))
            var prim_buf = _gpu_upload_array[PrimId_C](ctx, primIds, Int(primIdsCount))

            # Upload object-instancing data: one device buffer per BLAS (its
            # nodes + primids), then two small "array of device pointers"
            # buffers so a kernel's blasNodesArr[i]/blasPrimIdsArr[i] resolves
            # to the right BLAS — same two-level indirection as the CPU side
            # (see bvh.mojo's _traverse_instance_leaf).
            var n_blas_int = Int(blasCount)
            var blas_nodes_bufs = List[DeviceBuffer[DType.uint8]]()
            var blas_primids_bufs = List[DeviceBuffer[DType.uint8]]()
            var blas_nodes_ptrs_host = alloc[UnsafePointer[UInt8, MutExternalOrigin]](max(n_blas_int, 1))
            var blas_primids_ptrs_host = alloc[UnsafePointer[UInt8, MutExternalOrigin]](max(n_blas_int, 1))
            for bi in range(n_blas_int):
                var bn_count = Int(blasNodeCounts[bi])
                var bn_bytes = max(bn_count, 1) * size_of[BVH2Node]()
                var bn_buf = ctx.enqueue_create_buffer[DType.uint8](bn_bytes)
                with bn_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = blasNodesArr[bi].bitcast[UInt8]()
                    memcpy(dest=dst, src=src, count=bn_count * size_of[BVH2Node]())
                blas_nodes_ptrs_host[bi] = bn_buf.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin]()
                blas_nodes_bufs.append(bn_buf^)

                var bp_count = Int(blasPrimidCounts[bi])
                var bp_bytes = max(bp_count, 1) * size_of[PrimId_C]()
                var bp_buf = ctx.enqueue_create_buffer[DType.uint8](bp_bytes)
                with bp_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = blasPrimIdsArr[bi].bitcast[UInt8]()
                    memcpy(dest=dst, src=src, count=bp_count * size_of[PrimId_C]())
                blas_primids_ptrs_host[bi] = bp_buf.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin]()
                blas_primids_bufs.append(bp_buf^)

            var blas_ptrs_bytes = max(n_blas_int, 1) * size_of[UnsafePointer[UInt8, MutExternalOrigin]]()
            var blas_nodes_ptrs_buf = ctx.enqueue_create_buffer[DType.uint8](blas_ptrs_bytes)
            with blas_nodes_ptrs_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().bitcast[UnsafePointer[UInt8, MutExternalOrigin]]()
                memcpy(dest=dst, src=blas_nodes_ptrs_host, count=n_blas_int)
            var blas_primids_ptrs_buf = ctx.enqueue_create_buffer[DType.uint8](blas_ptrs_bytes)
            with blas_primids_ptrs_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().bitcast[UnsafePointer[UInt8, MutExternalOrigin]]()
                memcpy(dest=dst, src=blas_primids_ptrs_host, count=n_blas_int)
            blas_nodes_ptrs_host.free(); blas_primids_ptrs_host.free()

            var n_instances_int = Int(instanceCount)
            var instances_gpu_buf = _gpu_upload_array[Instance_C](ctx, instances, n_instances_int)

            # Upload per-mesh vertex/index/uv data and build device-side mesh structs
            var points_bufs = List[DeviceBuffer[DType.uint8]]()
            var face_bufs = List[DeviceBuffer[DType.uint8]]()
            var vert_bufs = List[DeviceBuffer[DType.uint8]]()
            var uv_bufs   = List[DeviceBuffer[DType.uint8]]()
            var nrm_bufs  = List[DeviceBuffer[DType.uint8]]()

            var mesh_structs_host = alloc[TriangleMesh_C](max(Int(meshCount), 1))

            for i in range(Int(meshCount)):
                var host_mesh = meshes[i]

                # Upload points
                var pts_count = Int(meshPointsCounts[i])
                var pts_bytes = pts_count * 4
                var pts_buf = ctx.enqueue_create_buffer[DType.uint8](pts_bytes)
                with pts_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = host_mesh.points.bitcast[UInt8]()
                    memcpy(dest=dst, src=src, count=pts_bytes)

                # Upload face indices
                var fi_count = Int(meshFaceIndicesCounts[i])
                var fi_bytes = fi_count * 8
                var fi_buf = ctx.enqueue_create_buffer[DType.uint8](fi_bytes)
                with fi_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = host_mesh.faceIndices.bitcast[UInt8]()
                    memcpy(dest=dst, src=src, count=fi_bytes)

                # Upload vertex indices
                var vi_count = Int(meshVertexIndicesCounts[i])
                var vi_bytes = vi_count * 8
                var vi_buf = ctx.enqueue_create_buffer[DType.uint8](vi_bytes)
                with vi_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = host_mesh.vertexIndices.bitcast[UInt8]()
                    memcpy(dest=dst, src=src, count=vi_bytes)

                # Upload UVs (2 floats per vertex; zeros if mesh has no UVs)
                var uv_n = Int(meshUvNVerts[i])
                var uv_bytes = max(uv_n * 2 * 4, 4)
                var uv_buf = ctx.enqueue_create_buffer[DType.uint8](uv_bytes)
                with uv_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    if uv_n > 0:
                        var src = host_mesh.uvs.bitcast[UInt8]()
                        memcpy(dest=dst, src=src, count=uv_n * 2 * 4)
                    else:
                        for j in range(uv_bytes):
                            dst[j] = UInt8(0)

                # Upload shading normals (3 floats per vertex; zeros if mesh has none)
                var nrm_n = Int(meshNrmNVerts[i])
                var nrm_bytes = max(nrm_n * 3 * 4, 4)
                var nrm_buf = ctx.enqueue_create_buffer[DType.uint8](nrm_bytes)
                with nrm_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    if nrm_n > 0:
                        var src = host_mesh.normals.bitcast[UInt8]()
                        memcpy(dest=dst, src=src, count=nrm_n * 3 * 4)
                    else:
                        for j in range(nrm_bytes):
                            dst[j] = UInt8(0)

                mesh_structs_host[i] = TriangleMesh_C(
                    pts_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin](),
                    fi_buf.unsafe_ptr().bitcast[Int64]().unsafe_origin_cast[MutExternalOrigin](),
                    vi_buf.unsafe_ptr().bitcast[Int64]().unsafe_origin_cast[MutExternalOrigin](),
                    uv_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin](),
                    nrm_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]() if nrm_n > 0 else UnsafePointer[
                        Float32, MutExternalOrigin
                    ](unsafe_from_address=1),
                )

                points_bufs.append(pts_buf^)
                face_bufs.append(fi_buf^)
                vert_bufs.append(vi_buf^)
                uv_bufs.append(uv_buf^)
                nrm_bufs.append(nrm_buf^)

            # Upload mesh struct array
            var meshes_buf = ctx.enqueue_create_buffer[DType.uint8](mesh_struct_bytes)
            with meshes_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = mesh_structs_host.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=mesh_struct_bytes)

            mesh_structs_host.free()

            # Upload materials array (>= 1 elem to avoid a zero-size buffer)
            var mat_buf = _gpu_upload_array[Material_C](ctx, materials, Int(materialCount))

            ctx.synchronize()

            # Upload area lights
            var al_buf = _gpu_upload_array[AreaLight_C](ctx, areaLights, Int(areaLightCount))

            # Upload spheres (analytical sphere primitives + sphere area lights)
            var sphere_buf = _gpu_upload_array[Sphere_C](ctx, spheres, Int(sphereCount))

            # Upload curves (native hair/fur primitives — control points only, no tessellation)
            var curve_buf = _gpu_upload_array[Curve_C](ctx, curves, Int(curveCount))

            # Upload distant (directional) lights
            var dl_buf = _gpu_upload_array[DistantLight_C](ctx, distantLights, Int(distantLightCount))

            # Upload point lights
            var pl_buf = _gpu_upload_array[PointLight_C](ctx, pointLights, Int(pointLightCount))

            # Upload light sampler CDF (n+1 Float32 entries)
            var ls_entries = Int(lightSamplerN) + 1
            var ls_bytes = max(ls_entries, 2) * size_of[Float32]()
            var ls_buf = ctx.enqueue_create_buffer[DType.uint8](ls_bytes)
            if ls_entries > 0:
                with ls_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr().bitcast[Float32]()
                    memcpy(dest=dst, src=lightSamplerCdf, count=ls_entries)

            # Upload infinite/environment lights with GPU-resident pixel/CDF data
            var il_count = Int(infiniteLightCount)
            var il_pixels_bufs = List[DeviceBuffer[DType.uint8]]()
            var il_cdf_bufs    = List[DeviceBuffer[DType.uint8]]()
            var il_w2l_bufs    = List[DeviceBuffer[DType.uint8]]()
            var il_patched = alloc[InfiniteLight_C](max(il_count, 1))
            for ii in range(il_count):
                var il = infiniteLights[ii]
                # Upload world_to_light matrix (16 floats = 64 bytes)
                var w2l_buf = ctx.enqueue_create_buffer[DType.uint8](64)
                with w2l_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(16):
                        dst[k] = il.world_to_light[k]
                il.world_to_light = w2l_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                il_w2l_bufs.append(w2l_buf^)
                # Upload pixels + CDF when texture is present
                if il.cdf_w > Int32(0) and _is_real_ptr(il.pixels_ptr):
                    var iw = Int(il.cdf_w); var ih = Int(il.cdf_h)
                    # Pixel data: iw × ih × 3 floats
                    var pix_count = iw * ih * 3
                    var pix_buf = ctx.enqueue_create_buffer[DType.uint8](pix_count * 4)
                    with pix_buf.map_to_host() as h:
                        var dst = h.unsafe_ptr().bitcast[Float32]()
                        for k in range(pix_count):
                            dst[k] = il.pixels_ptr[k]
                    il.pixels_ptr = pix_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                    il_pixels_bufs.append(pix_buf^)
                    # CDF data: (ih+1) marginal rows + ih×(iw+1) conditional entries
                    var cdf_count = (ih + 1) + ih * (iw + 1)
                    var cdf_buf = ctx.enqueue_create_buffer[DType.uint8](cdf_count * 4)
                    with cdf_buf.map_to_host() as h:
                        var dst = h.unsafe_ptr().bitcast[Float32]()
                        for k in range(cdf_count):
                            dst[k] = il.cdf_ptr[k]
                    il.cdf_ptr = cdf_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                    il_cdf_bufs.append(cdf_buf^)
                il_patched[ii] = il
            var il_bytes = max(il_count, 1) * size_of[InfiniteLight_C]()
            var il_buf = ctx.enqueue_create_buffer[DType.uint8](il_bytes)
            with il_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = il_patched.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=il_count * size_of[InfiniteLight_C]())
            il_patched.free()
            print("GPU: " + String(il_count) + " infinite light(s) uploaded")

            # Upload participating media (small array; >= 1 elem to avoid zero-size buffer)
            var med_buf = _gpu_upload_array[Medium_C](ctx, mediums, Int(mediumCount))
            # Read the SSS flag off the host copy while it is still in reach --
            # the round-budget decision this feeds is made on the host, and
            # reading it back off the device later would need a sync.
            var has_sss_med = False
            for mi in range(Int(mediumCount)):
                if mediums[mi].is_sss != Int32(0):
                    has_sss_med = True
                    break

            # Upload medium interfaces
            var miface_buf = _gpu_upload_array[MediumInterface_C](ctx, medium_ifaces, Int(medium_iface_count))

            # Upload heterogeneous density grids ("uniformgrid" media). Each
            # grid's (potentially large) density array gets its own device
            # buffer, mirroring the per-mesh points_bufs pattern; the Grid_C
            # struct array embeds device-resident pointers into those buffers.
            var grid_density_bufs = List[DeviceBuffer[DType.uint8]]()
            var n_grids_int = Int(gridCount)
            var grid_structs_host = alloc[Grid_C](max(n_grids_int, 1))
            for gi in range(n_grids_int):
                var host_grid = grids[gi]
                var n_voxels = Int(host_grid.nx) * Int(host_grid.ny) * Int(host_grid.nz)
                var density_bytes = max(n_voxels, 1) * 4
                var density_buf = ctx.enqueue_create_buffer[DType.uint8](density_bytes)
                with density_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = host_grid.density.bitcast[UInt8]()
                    memcpy(dest=dst, src=src, count=n_voxels * 4)
                grid_structs_host[gi] = Grid_C(
                    density_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin](),
                    host_grid.nx, host_grid.ny, host_grid.nz,
                    host_grid.p0, host_grid.p1,
                    host_grid.world_to_medium, host_grid.max_density)
                grid_density_bufs.append(density_buf^)
            var grid_struct_bytes = max(n_grids_int, 1) * size_of[Grid_C]()
            var grids_buf = ctx.enqueue_create_buffer[DType.uint8](grid_struct_bytes)
            with grids_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = grid_structs_host.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=grid_struct_bytes)
            grid_structs_host.free()
            if n_grids_int > 0:
                print("GPU: " + String(n_grids_int) + " heterogeneous density grid(s) uploaded")

            # Upload sparse density grids ("nanovdb" media). Same shape as
            # the dense-grid upload just above: each grid's decompressed
            # blob gets its own device buffer, and the NvdbGrid_C struct
            # array embeds device-resident pointers into those buffers.
            var nvdb_blob_bufs = List[DeviceBuffer[DType.uint8]]()
            var n_nvdb_grids_int = Int(nvdbGridCount)
            var nvdb_structs_host = alloc[NvdbGrid_C](max(n_nvdb_grids_int, 1))
            for gi in range(n_nvdb_grids_int):
                var host_nvdb = nvdbGrids[gi]
                var blob_bytes = Int(host_nvdb.blob_size)
                var blob_buf = ctx.enqueue_create_buffer[DType.uint8](max(blob_bytes, 1))
                if blob_bytes > 0:
                    with blob_buf.map_to_host() as host_buf:
                        var dst = host_buf.unsafe_ptr()
                        var src = host_nvdb.blob
                        memcpy(dest=dst, src=src, count=blob_bytes)
                nvdb_structs_host[gi] = NvdbGrid_C(
                    blob_buf.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](), host_nvdb.blob_size,
                    host_nvdb.world_to_medium, host_nvdb.inv_map, host_nvdb.map_vec,
                    host_nvdb.index_min, host_nvdb.index_max, host_nvdb.max_density)
                nvdb_blob_bufs.append(blob_buf^)
            var nvdb_struct_bytes = max(n_nvdb_grids_int, 1) * size_of[NvdbGrid_C]()
            var nvdb_grids_buf = ctx.enqueue_create_buffer[DType.uint8](nvdb_struct_bytes)
            with nvdb_grids_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = nvdb_structs_host.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=nvdb_struct_bytes)
            nvdb_structs_host.free()
            if n_nvdb_grids_int > 0:
                print("GPU: " + String(n_nvdb_grids_int) + " sparse (nanovdb) density grid(s) uploaded")

            # Upload MeasuredBxDF tabulated-BRDF tensors ("measured" material,
            # see project_measured_bxdf memory / lovely-dazzling-meteor plan
            # Stage 3). Each measured material owns 12 flat Float32 arrays
            # (theta_i/phi_i/wavelengths + ndf/sigma/vndf/luminance/spectra
            # data+CDFs); each gets its own device buffer, mirroring the
            # per-mesh points_bufs / per-grid grid_density_bufs pattern. The
            # patched MeasuredBRDF_C struct array embeds device-resident
            # pointers into those buffers -- loaded from the array *inside*
            # the GPU kernel (a load, not a cross-call by-value pass) and
            # handed only to @always_inline helpers, per the by-value
            # pointer-struct hazard already documented on MeasuredBRDF_C.
            var measured_field_bufs = List[DeviceBuffer[DType.uint8]]()
            var n_measured_int = Int(measuredBrdfCount)
            var measured_structs_host = alloc[MeasuredBRDF_C](max(n_measured_int, 1))
            for mi in range(n_measured_int):
                var hm = measured_brdfs[mi]
                var n_theta_i = Int(hm.n_theta_i)
                var n_phi_i = Int(hm.n_phi_i)
                var n_wavelengths = Int(hm.n_wavelengths)
                var slices2 = n_phi_i * n_theta_i
                var slices3 = slices2 * n_wavelengths

                var theta_i_buf = ctx.enqueue_create_buffer[DType.uint8](max(n_theta_i, 1) * 4)
                with theta_i_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(n_theta_i):
                        dst[k] = hm.theta_i[k]
                var theta_i_dptr = theta_i_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                measured_field_bufs.append(theta_i_buf^)

                var phi_i_buf = ctx.enqueue_create_buffer[DType.uint8](max(n_phi_i, 1) * 4)
                with phi_i_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(n_phi_i):
                        dst[k] = hm.phi_i[k]
                var phi_i_dptr = phi_i_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                measured_field_bufs.append(phi_i_buf^)

                var wavelengths_buf = ctx.enqueue_create_buffer[DType.uint8](max(n_wavelengths, 1) * 4)
                with wavelengths_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(n_wavelengths):
                        dst[k] = hm.wavelengths[k]
                var wavelengths_dptr = wavelengths_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                measured_field_bufs.append(wavelengths_buf^)

                var ndf_n = Int(hm.ndf_xs) * Int(hm.ndf_ys)
                var ndf_buf = ctx.enqueue_create_buffer[DType.uint8](max(ndf_n, 1) * 4)
                with ndf_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(ndf_n):
                        dst[k] = hm.ndf_data[k]
                var ndf_dptr = ndf_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                measured_field_bufs.append(ndf_buf^)

                var sigma_n = Int(hm.sigma_xs) * Int(hm.sigma_ys)
                var sigma_buf = ctx.enqueue_create_buffer[DType.uint8](max(sigma_n, 1) * 4)
                with sigma_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(sigma_n):
                        dst[k] = hm.sigma_data[k]
                var sigma_dptr = sigma_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                measured_field_bufs.append(sigma_buf^)

                var vndf_n = slices2 * Int(hm.vndf_xs) * Int(hm.vndf_ys)
                var vndf_buf = ctx.enqueue_create_buffer[DType.uint8](max(vndf_n, 1) * 4)
                with vndf_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(vndf_n):
                        dst[k] = hm.vndf_data[k]
                var vndf_dptr = vndf_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                measured_field_bufs.append(vndf_buf^)

                var vndf_marg_n = slices2 * Int(hm.vndf_ys)
                var vndf_marg_buf = ctx.enqueue_create_buffer[DType.uint8](max(vndf_marg_n, 1) * 4)
                with vndf_marg_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(vndf_marg_n):
                        dst[k] = hm.vndf_marg[k]
                var vndf_marg_dptr = vndf_marg_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                measured_field_bufs.append(vndf_marg_buf^)

                var vndf_cond_buf = ctx.enqueue_create_buffer[DType.uint8](max(vndf_n, 1) * 4)
                with vndf_cond_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(vndf_n):
                        dst[k] = hm.vndf_cond[k]
                var vndf_cond_dptr = vndf_cond_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                measured_field_bufs.append(vndf_cond_buf^)

                var lum_n = slices2 * Int(hm.lum_xs) * Int(hm.lum_ys)
                var lum_buf = ctx.enqueue_create_buffer[DType.uint8](max(lum_n, 1) * 4)
                with lum_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(lum_n):
                        dst[k] = hm.lum_data[k]
                var lum_dptr = lum_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                measured_field_bufs.append(lum_buf^)

                var lum_marg_n = slices2 * Int(hm.lum_ys)
                var lum_marg_buf = ctx.enqueue_create_buffer[DType.uint8](max(lum_marg_n, 1) * 4)
                with lum_marg_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(lum_marg_n):
                        dst[k] = hm.lum_marg[k]
                var lum_marg_dptr = lum_marg_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                measured_field_bufs.append(lum_marg_buf^)

                var lum_cond_buf = ctx.enqueue_create_buffer[DType.uint8](max(lum_n, 1) * 4)
                with lum_cond_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(lum_n):
                        dst[k] = hm.lum_cond[k]
                var lum_cond_dptr = lum_cond_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                measured_field_bufs.append(lum_cond_buf^)

                var spectra_n = slices3 * Int(hm.spectra_xs) * Int(hm.spectra_ys)
                var spectra_buf = ctx.enqueue_create_buffer[DType.uint8](max(spectra_n, 1) * 4)
                with spectra_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(spectra_n):
                        dst[k] = hm.spectra_data[k]
                var spectra_dptr = spectra_buf.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin]()
                measured_field_bufs.append(spectra_buf^)

                measured_structs_host[mi] = MeasuredBRDF_C(
                    hm.isotropic, hm.n_theta_i, hm.n_phi_i, hm.n_wavelengths,
                    theta_i_dptr, phi_i_dptr, wavelengths_dptr,
                    ndf_dptr, hm.ndf_xs, hm.ndf_ys,
                    sigma_dptr, hm.sigma_xs, hm.sigma_ys,
                    vndf_dptr, vndf_marg_dptr, vndf_cond_dptr, hm.vndf_xs, hm.vndf_ys,
                    lum_dptr, lum_marg_dptr, lum_cond_dptr, hm.lum_xs, hm.lum_ys,
                    hm.stride2_phi, hm.stride2_theta,
                    spectra_dptr, hm.spectra_xs, hm.spectra_ys,
                    hm.stride3_phi, hm.stride3_theta, hm.stride3_lambda,
                )
            var measured_struct_bytes = max(n_measured_int, 1) * size_of[MeasuredBRDF_C]()
            var measured_brdfs_buf = ctx.enqueue_create_buffer[DType.uint8](measured_struct_bytes)
            with measured_brdfs_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = measured_structs_host.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=measured_struct_bytes)
            measured_structs_host.free()
            if n_measured_int > 0:
                print("GPU: " + String(n_measured_int) + " measured BRDF(s) uploaded")

            # Allocate persistent render buffers (zeroed film)
            var n_pix = max(Int(n_pixels), 1)
            var r_path_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[PathState_C]() * WAVEFRONT_BATCH)
            var r_inter_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[Intersection_C]() * WAVEFRONT_BATCH)
            var r_film_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * 12)
            var r_albedo_film_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * 12)
            var r_atrous_ping_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * 12)
            var r_atrous_pong_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * 12)
            var r_atrous_albedo_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * 12)
            var r_atrous_variance_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * 4)
            var r_atrous_normals_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * 12)
            var r_atrous_depth_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * 4)
            var r_atrous_curve_mask_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * 4)
            var r_gbuf_worldpos_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * 12)
            var r_gbuf_material_id_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * 4)
            var r_restir_a_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[DIReservoir]())
            var r_restir_b_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[DIReservoir]())
            var r_restir_vol_a_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[VolReservoir]())
            var r_restir_vol_b_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[VolReservoir]())
            var r_restir_vol_used_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[Int8]())
            var r_shadow_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[ShadowTask_C]() * WAVEFRONT_BATCH)
            var r_active_count_buf = ctx.enqueue_create_buffer[DType.uint8](4)
            var r_active_idx_buf   = ctx.enqueue_create_buffer[DType.uint8](n_pix * 4)

            # Curve-divergence-mitigation scratch. curve_cand_prim/curve_cand_count
            # are indexed by every path's tid unconditionally inside
            # traverse_paths_gpu (curve_cand_count[tid] = 0 runs for every active
            # path regardless of whether the scene has curves — it's the reset
            # before that ray's BVH walk), so they must always be sized for the
            # full n_pix×WAVEFRONT_BATCH path range. Sizing them to a 1-element
            # dummy for non-curve scenes (as a memory-saving measure) was an
            # out-of-bounds GPU write for every tid > 0 on every one of the ~57
            # non-curve scenes in the comparison suite — corrupting whatever
            # else the allocator happened to place nearby, hence the scene-
            # dependent dark/wrong-color regression this fixes.
            # curve_compact_path_buf/curve_compact_counter_buf are only ever
            # touched by compact_curve_paths_gpu/resolve_curve_candidates_gpu,
            # both gated behind `if handle[].n_curves > 0` at the dispatch site,
            # so those two are still safe to leave dummy-sized.
            var n_curve_paths = n_pix * WAVEFRONT_BATCH
            var r_curve_cand_prim_buf   = ctx.enqueue_create_buffer[DType.uint8](n_curve_paths * CURVE_DEFER_K * 4)
            var r_curve_cand_count_buf  = ctx.enqueue_create_buffer[DType.uint8](n_curve_paths * 4)
            var r_curve_cand_offset_buf = ctx.enqueue_create_buffer[DType.uint8](n_curve_paths * 4)
            ctx.enqueue_function[init_curve_cand_offset_gpu](
                r_curve_cand_offset_buf.unsafe_ptr().bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
                Int64(n_curve_paths),
                grid_dim=ceildiv(n_curve_paths, 256), block_dim=256,
            )
            var n_curve_compact_paths = n_curve_paths if Int(curveCount) > 0 else 1
            var r_curve_compact_path_buf = ctx.enqueue_create_buffer[DType.uint8](n_curve_compact_paths * 4)
            var r_curve_compact_counter_buf = ctx.enqueue_create_buffer[DType.uint8](4)
            with r_film_buf.map_to_host() as h:
                var p = h.unsafe_ptr()
                for i in range(n_pix * 12):
                    p[i] = UInt8(0)
            with r_albedo_film_buf.map_to_host() as h:
                var p = h.unsafe_ptr()
                for i in range(n_pix * 12):
                    p[i] = UInt8(0)

            # Load and upload textures
            var n_textures_int = Int(n_tex)
            # Textures referenced as normal maps hold linear data and must NOT be
            # sRGB-decoded on load. Mark those indices by scanning the materials.
            var tex_is_raw = alloc[Bool](max(n_textures_int, 1))
            for ti in range(n_textures_int):
                tex_is_raw[ti] = False
            for mi in range(Int(materialCount)):
                var nidx = Int(materials[mi].normal_tex_idx)
                if nidx >= 0 and nidx < n_textures_int:
                    tex_is_raw[nidx] = True
            # Many scenes (e.g. landscape) declare a separate named Texture per
            # instance even when several instances share the same underlying
            # image file (batch-exported "-renamed-N" duplicates). Dedup by
            # (filename, raw-ness) so each unique file is only loaded from disk
            # and uploaded to the GPU once, instead of once per declaration.
            var dup_of = alloc[Int32](max(n_textures_int, 1))
            for ti in range(n_textures_int):
                dup_of[ti] = Int32(-1)
                for tj in range(ti):
                    if dup_of[tj] == Int32(-1) and tex_is_raw[tj] == tex_is_raw[ti] and \
                       _cstr_eq(tex_filenames[ti], tex_filenames[tj]):
                        dup_of[ti] = Int32(tj)
                        break
            var tex_data_bufs = List[DeviceBuffer[DType.uint8]]()
            var gpu_textures_host = alloc[GpuTexture_C](max(n_textures_int, 1))
            for ti in range(n_textures_int):
                if dup_of[ti] != Int32(-1):
                    gpu_textures_host[ti] = gpu_textures_host[Int(dup_of[ti])]
                    continue
                var filename = tex_filenames[ti]
                var data_u8_out = alloc[UnsafePointer[UInt8, MutExternalOrigin]](1)
                var data_f32_out = alloc[UnsafePointer[Float32, MutExternalOrigin]](1)
                var w_out = alloc[Int32](1)
                var h_out = alloc[Int32](1)
                var is_u8_out = alloc[Int32](1)
                w_out[0] = Int32(0); h_out[0] = Int32(0); is_u8_out[0] = Int32(0)
                var raw_flag = Int32(1) if tex_is_raw[ti] else Int32(0)
                # Undecoded 8-bit sRGB bytes for a genuine 8-bit-per-channel,
                # non-HDR, non-raw source (4x less VRAM than the float path,
                # see project_gpu_texture_cache memory); everything else
                # (HDR, normal maps, non-8-bit sources) falls back to the
                # pre-linearised float32 path. sRGB decode for the u8 path
                # happens per bilinear tap at sample time (shading.mojo's
                # _sample_level), not here.
                var ok = external_call["load_texture_u8_or_float", Int32,
                    UnsafePointer[UInt8, MutExternalOrigin],
                    UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin],
                    UnsafePointer[UnsafePointer[Float32, MutExternalOrigin], MutExternalOrigin],
                    UnsafePointer[Int32, MutExternalOrigin],
                    UnsafePointer[Int32, MutExternalOrigin],
                    UnsafePointer[Int32, MutExternalOrigin],
                    Int32](filename, data_u8_out, data_f32_out, w_out, h_out, is_u8_out, raw_flag)
                if ok != 0 and Int(w_out[0]) > 0:
                    var tw = Int(w_out[0]); var th = Int(h_out[0])
                    var is_u8 = is_u8_out[0] != Int32(0)
                    # Mip pyramid: levels until 1x1, box-downsampled. Anti-aliases
                    # minified textures; trilinear-sampled on the GPU via the LOD.
                    var nlev = 1; var ww = tw; var hh = th
                    while ww > 1 or hh > 1:
                        ww = max(1, ww // 2); hh = max(1, hh // 2); nlev += 1
                    var total = 0; ww = tw; hh = th
                    for _k in range(nlev):
                        total += ww * hh * 3
                        ww = max(1, ww // 2); hh = max(1, hh // 2)
                    if is_u8:
                        var pyr8 = alloc[UInt8](total)
                        var src0_8 = data_u8_out[0]
                        memcpy(dest=pyr8, src=src0_8, count=tw * th * 3)
                        var off_prev8 = 0; var pw8 = tw; var ph8 = th
                        var off_cur8 = tw * th * 3
                        for _k in range(1, nlev):
                            var cw = max(1, pw8 // 2); var ch = max(1, ph8 // 2)
                            for y in range(ch):
                                for x in range(cw):
                                    var x0 = 2 * x; var x1 = min(2 * x + 1, pw8 - 1)
                                    var y0 = 2 * y; var y1 = min(2 * y + 1, ph8 - 1)
                                    for c in range(3):
                                        # Decode-average-reencode, NOT a raw
                                        # byte average -- box-filtering sRGB
                                        # bytes directly is gamma-space
                                        # filtering, visibly wrong at any
                                        # real minification (see
                                        # _srgb_byte_to_linear's docstring).
                                        var a = _srgb_byte_to_linear(pyr8[off_prev8 + (y0 * pw8 + x0) * 3 + c])
                                        var b = _srgb_byte_to_linear(pyr8[off_prev8 + (y0 * pw8 + x1) * 3 + c])
                                        var cc = _srgb_byte_to_linear(pyr8[off_prev8 + (y1 * pw8 + x0) * 3 + c])
                                        var d = _srgb_byte_to_linear(pyr8[off_prev8 + (y1 * pw8 + x1) * 3 + c])
                                        pyr8[off_cur8 + (y * cw + x) * 3 + c] = _linear_to_srgb_byte((a + b + cc + d) * Float32(0.25))
                            off_prev8 = off_cur8; off_cur8 += cw * ch * 3; pw8 = cw; ph8 = ch
                        var tex_buf8 = ctx.enqueue_create_buffer[DType.uint8](total)
                        with tex_buf8.map_to_host() as h8:
                            memcpy(dest=h8.unsafe_ptr(), src=pyr8, count=total)
                        pyr8.free()
                        gpu_textures_host[ti] = GpuTexture_C(tex_buf8.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](), Int32(tw), Int32(th), Int32(nlev), Int32(1))
                        _ = external_call["free_texture_u8", Int32, UnsafePointer[UInt8, MutExternalOrigin]](data_u8_out[0])
                        tex_data_bufs.append(tex_buf8^)
                    else:
                        var pyr = alloc[Float32](total)
                        var src0 = data_f32_out[0]
                        memcpy(dest=pyr, src=src0, count=tw * th * 3)
                        var off_prev = 0; var pw = tw; var ph = th
                        var off_cur = tw * th * 3
                        for _k in range(1, nlev):
                            var cw = max(1, pw // 2); var ch = max(1, ph // 2)
                            for y in range(ch):
                                for x in range(cw):
                                    var x0 = 2 * x; var x1 = min(2 * x + 1, pw - 1)
                                    var y0 = 2 * y; var y1 = min(2 * y + 1, ph - 1)
                                    for c in range(3):
                                        var a = pyr[off_prev + (y0 * pw + x0) * 3 + c]
                                        var b = pyr[off_prev + (y0 * pw + x1) * 3 + c]
                                        var cc = pyr[off_prev + (y1 * pw + x0) * 3 + c]
                                        var d = pyr[off_prev + (y1 * pw + x1) * 3 + c]
                                        pyr[off_cur + (y * cw + x) * 3 + c] = (a + b + cc + d) * Float32(0.25)
                            off_prev = off_cur; off_cur += cw * ch * 3; pw = cw; ph = ch
                        var tex_buf = ctx.enqueue_create_buffer[DType.uint8](total * 4)
                        with tex_buf.map_to_host() as h:
                            var dst = h.unsafe_ptr().bitcast[Float32]()
                            memcpy(dest=dst, src=pyr, count=total)
                        pyr.free()
                        gpu_textures_host[ti] = GpuTexture_C(tex_buf.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin](), Int32(tw), Int32(th), Int32(nlev), Int32(0))
                        _ = external_call["free_texture_rgb", Int32, UnsafePointer[Float32, MutExternalOrigin]](data_f32_out[0])
                        tex_data_bufs.append(tex_buf^)
                else:
                    gpu_textures_host[ti] = GpuTexture_C(UnsafePointer[UInt8, MutExternalOrigin].unsafe_dangling(), Int32(0), Int32(0), Int32(0), Int32(0))
                data_u8_out.free(); data_f32_out.free(); w_out.free(); h_out.free(); is_u8_out.free()
            var tex_struct_bytes = max(n_textures_int, 1) * size_of[GpuTexture_C]()
            var textures_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](tex_struct_bytes)
            with textures_gpu_buf.map_to_host() as h:
                var dst = h.unsafe_ptr()
                var src = gpu_textures_host.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=tex_struct_bytes)
            var n_unique_tex = 0
            for ti in range(n_textures_int):
                if dup_of[ti] == Int32(-1):
                    n_unique_tex += 1
            gpu_textures_host.free()
            tex_is_raw.free()
            dup_of.free()
            print("GPU: " + String(n_textures_int) + " texture(s) uploaded ("
                  + String(n_unique_tex) + " unique file(s) loaded)")

            # Upload Sobol matrices: first 1024 dimensions × 52 UInt32 = 212992 bytes
            comptime N_SOBOL_GPU_DIMS = 1024
            comptime N_SOBOL_GPU_WORDS = N_SOBOL_GPU_DIMS * 52
            comptime N_SOBOL_GPU_BYTES = N_SOBOL_GPU_WORDS * 4
            var sobol_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](N_SOBOL_GPU_BYTES)
            with sobol_gpu_buf.map_to_host() as h:
                var dst = h.unsafe_ptr().bitcast[UInt32]()
                memcpy(dest=dst, src=sobol_matrices, count=N_SOBOL_GPU_WORDS)

            # Upload raster_to_camera (16 floats = 64 bytes)
            var r2c_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](64)
            with r2c_gpu_buf.map_to_host() as h:
                var dst = h.unsafe_ptr().bitcast[Float32]()
                memcpy(dest=dst, src=r2c, count=16)

            # Upload camera_to_world (16 floats = 64 bytes)
            var c2w_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](64)
            with c2w_gpu_buf.map_to_host() as h:
                var dst = h.unsafe_ptr().bitcast[Float32]()
                memcpy(dest=dst, src=c2w_init, count=16)

            # Upload the spectral (Jakob-Hanika) coefficient table + CIE
            # X/Y/Z/D65 tables, if a real one was loaded (spectral.res > 0)
            # — Stage 2c-1, see project_spectral_rendering memory. Dummy
            # 1-element buffers otherwise (BDPT/SPPM GPU dispatch don't wire
            # spectral yet — Stage 3/4 — same "at least 1 elem" convention
            # already used above for zero-size scene data).
            comptime CIE_N = 95
            var spec_coeffs_count = (3 * spectral_res * spectral_res * spectral_res * 3) if spectral_res > 0 else 1
            var spec_coeffs_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](spec_coeffs_count * 4)
            if spectral_res > 0:
                with spec_coeffs_gpu_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    memcpy(dest=dst, src=spectral_coeffs, count=spec_coeffs_count)

            var spec_cie_count = CIE_N if spectral_res > 0 else 1
            var spec_cie_x_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](spec_cie_count * 4)
            var spec_cie_y_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](spec_cie_count * 4)
            var spec_cie_z_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](spec_cie_count * 4)
            var spec_d65_gpu_buf   = ctx.enqueue_create_buffer[DType.uint8](spec_cie_count * 4)
            if spectral_res > 0:
                with spec_cie_x_gpu_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    memcpy(dest=dst, src=spectral_cie_x, count=CIE_N)
                with spec_cie_y_gpu_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    memcpy(dest=dst, src=spectral_cie_y, count=CIE_N)
                with spec_cie_z_gpu_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    memcpy(dest=dst, src=spectral_cie_z, count=CIE_N)
                with spec_d65_gpu_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    memcpy(dest=dst, src=spectral_d65, count=CIE_N)

            # Allocate handle on heap
            var handle = alloc[GpuSceneHandle](1)
            handle.init_pointee_move(GpuSceneHandle(
                ctx=ctx^,
                bvh2Nodes_buf=bvh_buf^,
                primIds_buf=prim_buf^,
                blas_nodes_bufs=blas_nodes_bufs^,
                blas_primids_bufs=blas_primids_bufs^,
                blas_nodes_ptrs_buf=blas_nodes_ptrs_buf^,
                blas_primids_ptrs_buf=blas_primids_ptrs_buf^,
                n_blas=n_blas_int,
                instances_buf=instances_gpu_buf^,
                n_instances=n_instances_int,
                meshes_buf=meshes_buf^,
                mesh_count=Int(meshCount),
                materials_buf=mat_buf^,
                material_count=Int(materialCount),
                points_bufs=points_bufs^,
                faceIndices_bufs=face_bufs^,
                vertexIndices_bufs=vert_bufs^,
                uv_bufs=uv_bufs^,
                nrm_bufs=nrm_bufs^,
                tex_data_bufs=tex_data_bufs^,
                textures_buf=textures_gpu_buf^,
                n_textures=n_textures_int,
                area_lights_buf=al_buf^,
                n_area_lights=Int(areaLightCount),
                spheres_buf=sphere_buf^,
                n_spheres=Int(sphereCount),
                curves_buf=curve_buf^,
                n_curves=Int(curveCount),
                curve_cand_prim_buf=r_curve_cand_prim_buf^,
                curve_cand_count_buf=r_curve_cand_count_buf^,
                curve_cand_offset_buf=r_curve_cand_offset_buf^,
                curve_compact_path_buf=r_curve_compact_path_buf^,
                curve_compact_counter_buf=r_curve_compact_counter_buf^,
                distant_lights_buf=dl_buf^,
                n_distant_lights=Int(distantLightCount),
                point_lights_buf=pl_buf^,
                n_point_lights=Int(pointLightCount),
                light_sampler_buf=ls_buf^,
                n_light_sampler=Int(lightSamplerN),
                infinite_lights_buf=il_buf^,
                il_pixels_bufs=il_pixels_bufs^,
                il_cdf_bufs=il_cdf_bufs^,
                il_w2l_bufs=il_w2l_bufs^,
                n_infinite_lights=Int(infiniteLightCount),
                mediums_buf=med_buf^,
                n_mediums=Int(mediumCount),
                has_sss_medium=has_sss_med,
                medium_ifaces_buf=miface_buf^,
                n_medium_ifaces=Int(medium_iface_count),
                grids_buf=grids_buf^,
                n_grids=n_grids_int,
                grid_density_bufs=grid_density_bufs^,
                nvdb_grids_buf=nvdb_grids_buf^,
                n_nvdb_grids=n_nvdb_grids_int,
                nvdb_blob_bufs=nvdb_blob_bufs^,
                measured_brdfs_buf=measured_brdfs_buf^,
                n_measured_brdfs=n_measured_int,
                measured_field_bufs=measured_field_bufs^,
                path_buf=r_path_buf^,
                inter_buf=r_inter_buf^,
                film_buf=r_film_buf^,
                albedo_film_buf=r_albedo_film_buf^,
                atrous_ping_buf=r_atrous_ping_buf^,
                atrous_pong_buf=r_atrous_pong_buf^,
                atrous_albedo_buf=r_atrous_albedo_buf^,
                atrous_variance_buf=r_atrous_variance_buf^,
                atrous_normals_buf=r_atrous_normals_buf^,
                atrous_depth_buf=r_atrous_depth_buf^,
                atrous_curve_mask_buf=r_atrous_curve_mask_buf^,
                gbuf_worldpos_buf=r_gbuf_worldpos_buf^,
                gbuf_material_id_buf=r_gbuf_material_id_buf^,
                restir_a_buf=r_restir_a_buf^,
                restir_b_buf=r_restir_b_buf^,
                restir_vol_a_buf=r_restir_vol_a_buf^,
                restir_vol_b_buf=r_restir_vol_b_buf^,
                restir_vol_used_buf=r_restir_vol_used_buf^,
                shadow_buf=r_shadow_buf^,
                active_count_buf=r_active_count_buf^,
                active_idx_buf=r_active_idx_buf^,
                n_pixels=n_pix,
                sobol_buf=sobol_gpu_buf^,
                r2c_buf=r2c_gpu_buf^,
                c2w_buf=c2w_gpu_buf^,
                filter_sigma=filter_sigma,
                filter_support_x=filter_support_x,
                filter_support_y=filter_support_y,
                filter_norm_x=filter_norm_x,
                filter_norm_y=filter_norm_y,
                filter_type=filter_type,
                fw=Int(fw),
                fh=Int(fh),
                spectral_coeffs_buf=spec_coeffs_gpu_buf^,
                spectral_cie_x_buf=spec_cie_x_gpu_buf^,
                spectral_cie_y_buf=spec_cie_y_gpu_buf^,
                spectral_cie_z_buf=spec_cie_z_gpu_buf^,
                spectral_d65_buf=spec_d65_gpu_buf^,
                spectral_res=spectral_res,
            ))

            print("GPU: scene uploaded")
            return handle.bitcast[GpuSceneHandle]()
        except e:
            var msg = String(e)
            if "libnvidia" in msg or "nvidia-ml" in msg:
                print("GPU: no supported GPU driver found (requires NVIDIA or AMD)")
            else:
                print("GPU: Failed to upload scene: " + msg)
            return UnsafePointer[GpuSceneHandle, MutExternalOrigin].unsafe_dangling()
    else:
        return UnsafePointer[GpuSceneHandle, MutExternalOrigin].unsafe_dangling()


# GPU kernel function — one thread per ray
def traverse_bvh2_gpu(
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    rays: UnsafePointer[Ray_C, MutExternalOrigin],
    tMaxValues: UnsafePointer[Float32, MutExternalOrigin],
    results: UnsafePointer[Intersection_C, MutExternalOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var ray = rays[tid]
    var tMax = tMaxValues[tid]
    var result_ptr = results + tid
    # WARNING: no spheres/n_spheres are threaded into this kernel, so analytic
    # spheres (PrimId_C.type == 4, held in a separate flat array) are INVISIBLE
    # to it -- the same omission that made spheres invisible to SPPM and broke
    # volumetric-caustic. This kernel and its host wrapper gpu_traverse_batch
    # have no callers today. Before giving them one, add sphere params and a
    # test_spheres pass, exactly as sample_medium_gpu and the SPPM kernels do.
    traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, ray, tMax, result_ptr)

def gpu_traverse_batch(
    handlePtr: UnsafePointer[GpuSceneHandle, MutExternalOrigin],
    rays: UnsafePointer[Ray_C, MutExternalOrigin],
    tMaxValues: UnsafePointer[Float32, MutExternalOrigin],
    count: Int64,
    results: UnsafePointer[Intersection_C, MutExternalOrigin],
):
    if not handlePtr:
        return
    var handle = handlePtr

    var n = Int(count)
    if n == 0:
        return

    comptime if has_accelerator():
        try:
            # Upload rays to GPU
            var ray_bytes = n * 24  # sizeof(Ray_C) = 6 * 4 = 24
            var ray_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](ray_bytes)
            with ray_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = rays.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=ray_bytes)

            # Upload tMax values
            var tmax_bytes = n * 4
            var tmax_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](tmax_bytes)
            with tmax_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = tMaxValues.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=tmax_bytes)

            # Create output buffer
            var result_bytes = n * 48  # sizeof(Intersection_C)
            var result_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](result_bytes)

            # Launch kernel
            comptime block_size = 256
            var grid_dim = ceildiv(n, block_size)

            handle[].ctx.enqueue_function[traverse_bvh2_gpu](
                handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
                handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
                handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
                handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
                ray_buf.unsafe_ptr().bitcast[Ray_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
                tmax_buf.unsafe_ptr().bitcast[Float32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
                result_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
                Int64(n),
                grid_dim=grid_dim,
                block_dim=block_size,
            )

            handle[].ctx.synchronize()

            # Copy results back to host
            with result_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = results.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=result_bytes)
        except e:
            print("GPU: Batch traversal failed: " + String(e))


def shade_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    spectral: SpectralHandle,
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    shade_core(paths, intersections, meshes, materials, spectral, tid)



def shade_nee_preamble_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    areaLightCount_dp: Int64,
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures_dp: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutExternalOrigin],
    n_distant_lights_dp: Int64,
    pointLights: UnsafePointer[PointLight_C, MutExternalOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: UnsafePointer[Float32, MutExternalOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutExternalOrigin],
    n_infinite_lights_dp: Int64,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
    count_dp: Int64,
    px_scale: Float32,
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
):
    var spectral_res = Int(spectral_res_dp)
    var areaLightCount = Int(areaLightCount_dp)
    var n_textures = Int(n_textures_dp)
    var n_distant_lights = Int(n_distant_lights_dp)
    var n_point_lights = Int(n_point_lights_dp)
    var n_light_sampler = Int(n_light_sampler_dp)
    var n_infinite_lights = Int(n_infinite_lights_dp)
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].active == 0:
        return
    var inter = intersections[tid]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    # Do NOT early-exit on miss — shade_nee_core adds env-light contribution there.
    var ctx_no_shadow = ShadeContext(
        path_idx=0, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=UnsafePointer[NormalSlopeMap_C, MutExternalOrigin].unsafe_dangling(),
        shadow_tasks=UnsafePointer[ShadowTask_C, MutExternalOrigin].unsafe_dangling(),
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutExternalOrigin].unsafe_dangling(),
        gi_pending=UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_nee_core[True, False](path_ptr, inter, ctx_no_shadow)


# ── Per-material GPU kernels (G1) ─────────────────────────────────────────────
# shade_nee_preamble_gpu handles miss + emission, then sets pending_mat.
# Each kernel below checks pending_mat, clears it, and calls the shade function.

def shade_diffuse_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    areaLightCount_dp: Int64,
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures_dp: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutExternalOrigin],
    n_distant_lights_dp: Int64,
    pointLights: UnsafePointer[PointLight_C, MutExternalOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: UnsafePointer[Float32, MutExternalOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutExternalOrigin],
    n_infinite_lights_dp: Int64,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
    count_dp: Int64,
    px_scale: Float32,
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    # ReSTIR DI (Phase 2, --restir). Only gpu_render_sample ever passes
    # use_restir=True here -- gpu_render_wavefront has no ReSTIR concept at
    # all (see its own docstring: batch --restir renders via
    # gpu_render_sample instead, precisely to avoid the
    # WAVEFRONT_BATCH-concurrent-samples-per-pixel problem). All defaulted-
    # inert so gpu_render_wavefront's dispatch is unaffected.
    # Int32 rather than Bool: GPU kernel arguments must be DevicePassable and
    # Bool is not, which the compiler only reports at the enqueue site.
    use_restir: Int32 = Int32(0),
    restir_read: UnsafePointer[DIReservoir, MutExternalOrigin] = UnsafePointer[DIReservoir, MutExternalOrigin].unsafe_dangling(),
    restir_write: UnsafePointer[DIReservoir, MutExternalOrigin] = UnsafePointer[DIReservoir, MutExternalOrigin].unsafe_dangling(),
    gbuf_normal: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    gbuf_depth: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    gbuf_material_id: UnsafePointer[Int32, MutExternalOrigin] = UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
    gbuf_world_pos: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    frame_w: Int32 = Int32(0),
    frame_h: Int32 = Int32(0),
):
    var spectral_res = Int(spectral_res_dp)
    var areaLightCount = Int(areaLightCount_dp)
    var n_textures = Int(n_textures_dp)
    var n_distant_lights = Int(n_distant_lights_dp)
    var n_point_lights = Int(n_point_lights_dp)
    var n_light_sampler = Int(n_light_sampler_dp)
    var n_infinite_lights = Int(n_infinite_lights_dp)
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].pending_mat != MatKind.diffuse:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[tid]
    var mat = materials[Int(inter.primId.materialIndex)]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var restir_on = use_restir != Int32(0)
    var ctx = ShadeContext(
        path_idx=0, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=UnsafePointer[NormalSlopeMap_C, MutExternalOrigin].unsafe_dangling(),
        shadow_tasks=UnsafePointer[ShadowTask_C, MutExternalOrigin].unsafe_dangling(),
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=restir_on,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutExternalOrigin].unsafe_dangling(),
        gi_pending=UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    # tid IS the pixel index here: this kernel only ever sees use_restir=True
    # from gpu_render_sample, which runs exactly one path per pixel (see the
    # param block above -- gpu_render_wavefront never sets it). restir_on
    # without real buffers (non-restir renders) still needs pixel_idx=-1,
    # di_temporal_step's own "no reuse" sentinel.
    var restir_has_state = restir_on and _is_real_ptr(restir_read)
    var restir_io = reservoir_io_null()
    if restir_has_state:
        restir_io = ReservoirIO(
            read=restir_read, write=restir_write,
            gbuf_normal=gbuf_normal, gbuf_depth=gbuf_depth,
            gbuf_material_id=gbuf_material_id, gbuf_world_pos=gbuf_world_pos,
            frame_w=frame_w, frame_h=frame_h)
    shade_diffuse[True, False](path_ptr, inter, ctx, mat, null_guide(), restir_io, tid if restir_has_state else -1)


def shade_coated_diffuse_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    areaLightCount_dp: Int64,
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures_dp: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutExternalOrigin],
    n_distant_lights_dp: Int64,
    pointLights: UnsafePointer[PointLight_C, MutExternalOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: UnsafePointer[Float32, MutExternalOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutExternalOrigin],
    n_infinite_lights_dp: Int64,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: UnsafePointer[ShadowTask_C, MutExternalOrigin],
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
):
    var spectral_res = Int(spectral_res_dp)
    var areaLightCount = Int(areaLightCount_dp)
    var n_textures = Int(n_textures_dp)
    var n_distant_lights = Int(n_distant_lights_dp)
    var n_point_lights = Int(n_point_lights_dp)
    var n_light_sampler = Int(n_light_sampler_dp)
    var n_infinite_lights = Int(n_infinite_lights_dp)
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].pending_mat != MatKind.coated_diffuse:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[tid]
    var mat = materials[Int(inter.primId.materialIndex)]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=UnsafePointer[NormalSlopeMap_C, MutExternalOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutExternalOrigin].unsafe_dangling(),
        gi_pending=UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_coated_diffuse[True, False](path_ptr, inter, ctx, mat)


def shade_diffuse_transmit_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    areaLightCount_dp: Int64,
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures_dp: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutExternalOrigin],
    n_distant_lights_dp: Int64,
    pointLights: UnsafePointer[PointLight_C, MutExternalOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: UnsafePointer[Float32, MutExternalOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutExternalOrigin],
    n_infinite_lights_dp: Int64,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: UnsafePointer[ShadowTask_C, MutExternalOrigin],
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
):
    var spectral_res = Int(spectral_res_dp)
    var areaLightCount = Int(areaLightCount_dp)
    var n_textures = Int(n_textures_dp)
    var n_distant_lights = Int(n_distant_lights_dp)
    var n_point_lights = Int(n_point_lights_dp)
    var n_light_sampler = Int(n_light_sampler_dp)
    var n_infinite_lights = Int(n_infinite_lights_dp)
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].pending_mat != MatKind.diffuse_transmit:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[tid]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=UnsafePointer[NormalSlopeMap_C, MutExternalOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutExternalOrigin].unsafe_dangling(),
        gi_pending=UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_diffuse_transmission[True, False](path_ptr, inter, ctx)


# GPU-only: mix is a pure material SELECTOR, not a shader -- it has no BSDF,
# no NEE, no light/texture/spectral dependence of its own. The old
# implementation called shade_mix[True,False] with a full ShadeContext
# (33 parameters, mirroring every real per-material kernel's signature) which
# then called the @always_inline _shade_dispatch -- inlining the FULL shading
# code of every OTHER material (diffuse, conductor, dielectric,
# coated_diffuse, diffuse_transmission, coated_conductor, thin_dielectric,
# interface, measured) into this one function. That made shade_mix_gpu's
# compiled body easily the largest function in the codebase, and this
# machine's CUDA 13.3 driver / Modular 26.4.0 toolchain cannot produce valid
# PTX for it (confirmed via kernel-by-kernel bisection: shade_mix_gpu alone
# reproduces CUDA_ERROR_INVALID_PTX; every other kernel, including
# shade_measured_gpu, compiles and runs fine without it).
#
# Fix: don't shade anything here at all. Pick the sub-material (same RNG
# draw + mix-of-mix guard as shade_mix, shading.mojo) and redirect --
# overwrite this hit's materialIndex to the CHOSEN sub-material (still an
# index into the same `materials` array) and re-tag pending_mat with the
# sub-material's real type, exactly mirroring what shade_nee_core's own
# GPU branch does for every material ("mark material for its dedicated
# per-material kernel"). CUDA kernels launched on one stream execute in
# launch order, so as long as this kernel is enqueued BEFORE every other
# per-material kernel in the same bounce's dispatch sequence (see both
# gpu_render_sample/gpu_render_wavefront call sites), the sub-material's
# real kernel picks up the redirected pending_mat/materialIndex later in
# this SAME pass and shades it with its own full NEE/BSDF logic --
# identical end result to the CPU path, just via a two-step handoff instead
# of one big inlined function. This can't be deferred to the NEXT bounce:
# the ray is never advanced here, so a fresh intersection at the start of
# the next bounce would just re-hit the same mix material and re-roll the
# choice forever.
def shade_mix_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].pending_mat != MatKind.mix:
        return
    var inter = intersections[tid]
    var mat = materials[Int(inter.primId.materialIndex)]
    var packed = mat.tex_idx
    var idx1 = Int(packed & Int32(0xFFFF))
    var idx2 = Int((packed >> 16) & Int32(0xFFFF))
    var amount = mat.roughU  # blend factor: 0 = all mat1, 1 = all mat2
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var chosen_idx = idx2 if pcg.next_float() < amount else idx1
    path_ptr[].pcgState = pcg.state
    var sub_type = materials[chosen_idx].type
    if sub_type == MatKind.mix:
        sub_type = MatKind.diffuse  # guard against mix-of-mix cycle, matches shade_mix (shading.mojo)
    intersections[tid].primId.materialIndex = Int64(chosen_idx)
    path_ptr[].pending_mat = sub_type


def shade_conductor_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    areaLightCount_dp: Int64,
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures_dp: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutExternalOrigin],
    n_distant_lights_dp: Int64,
    pointLights: UnsafePointer[PointLight_C, MutExternalOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: UnsafePointer[Float32, MutExternalOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutExternalOrigin],
    n_infinite_lights_dp: Int64,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: UnsafePointer[ShadowTask_C, MutExternalOrigin],
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
):
    var spectral_res = Int(spectral_res_dp)
    var areaLightCount = Int(areaLightCount_dp)
    var n_textures = Int(n_textures_dp)
    var n_distant_lights = Int(n_distant_lights_dp)
    var n_point_lights = Int(n_point_lights_dp)
    var n_light_sampler = Int(n_light_sampler_dp)
    var n_infinite_lights = Int(n_infinite_lights_dp)
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].pending_mat != MatKind.conductor:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[tid]
    var mat = materials[Int(inter.primId.materialIndex)]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=UnsafePointer[NormalSlopeMap_C, MutExternalOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutExternalOrigin].unsafe_dangling(),
        gi_pending=UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_conductor[True, False](path_ptr, inter, ctx, mat)


def shade_measured_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    areaLightCount_dp: Int64,
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures_dp: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutExternalOrigin],
    n_distant_lights_dp: Int64,
    pointLights: UnsafePointer[PointLight_C, MutExternalOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: UnsafePointer[Float32, MutExternalOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutExternalOrigin],
    n_infinite_lights_dp: Int64,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: UnsafePointer[ShadowTask_C, MutExternalOrigin],
    measured_brdfs: UnsafePointer[MeasuredBRDF_C, MutExternalOrigin],
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
):
    var spectral_res = Int(spectral_res_dp)
    var areaLightCount = Int(areaLightCount_dp)
    var n_textures = Int(n_textures_dp)
    var n_distant_lights = Int(n_distant_lights_dp)
    var n_point_lights = Int(n_point_lights_dp)
    var n_light_sampler = Int(n_light_sampler_dp)
    var n_infinite_lights = Int(n_infinite_lights_dp)
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].pending_mat != MatKind.measured:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[tid]
    var mat = materials[Int(inter.primId.materialIndex)]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=UnsafePointer[NormalSlopeMap_C, MutExternalOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=measured_brdfs,
        gi_pending=UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_measured[True, False](path_ptr, inter, ctx, mat)


def shade_dielectric_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    count_dp: Int64,
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures_dp: Int64,
    px_scale: Float32,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].pending_mat != MatKind.dielectric:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[tid]
    var mat = materials[Int(inter.primId.materialIndex)]
    # textures/px_scale are here only so a dielectric carrying "texture
    # displacement"/"normalmap" gets it applied (barcelona-pavilion's water).
    # tex_filenames is CPU-only (GPU samples the uploaded texture table), so
    # the dangling default is correct on this path.
    shade_dielectric[True](path_ptr, inter, meshes, mat, spheres,
        UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        textures, Int(n_textures_dp), px_scale)


def shade_thin_dielectric_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].pending_mat != MatKind.thin_dielectric:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[tid]
    var mat = materials[Int(inter.primId.materialIndex)]
    shade_thin_dielectric(path_ptr, inter, meshes, mat, spheres)


def shade_coated_conductor_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    areaLightCount_dp: Int64,
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures_dp: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutExternalOrigin],
    n_distant_lights_dp: Int64,
    pointLights: UnsafePointer[PointLight_C, MutExternalOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: UnsafePointer[Float32, MutExternalOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutExternalOrigin],
    n_infinite_lights_dp: Int64,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: UnsafePointer[ShadowTask_C, MutExternalOrigin],
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
):
    var spectral_res = Int(spectral_res_dp)
    var areaLightCount = Int(areaLightCount_dp)
    var n_textures = Int(n_textures_dp)
    var n_distant_lights = Int(n_distant_lights_dp)
    var n_point_lights = Int(n_point_lights_dp)
    var n_light_sampler = Int(n_light_sampler_dp)
    var n_infinite_lights = Int(n_infinite_lights_dp)
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].pending_mat != MatKind.coated_conductor:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[tid]
    var mat = materials[Int(inter.primId.materialIndex)]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=UnsafePointer[NormalSlopeMap_C, MutExternalOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutExternalOrigin].unsafe_dangling(),
        gi_pending=UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_coated_conductor[True, False](path_ptr, inter, ctx, mat)


def shade_interface_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    medium_ifaces: UnsafePointer[MediumInterface_C, MutExternalOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    """Passthrough (interface) material: advance ray through the surface.
    Medium update is handled by update_medium_gpu which runs after all shaders."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].pending_mat != MatKind.interface:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[tid]
    shade_interface(path_ptr, inter)


def update_medium_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    medium_ifaces: UnsafePointer[MediumInterface_C, MutExternalOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    """Update current_medium_idx for any surface hit with a MediumInterface bound.
    Runs after all material shaders; uses the post-scatter ray direction (same
    convention as CPU rendering.mojo) to determine inside vs outside."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].active == 0:
        return
    var inter = intersections[tid]
    if inter.hit == 0:
        return
    var mat = materials[Int(inter.primId.materialIndex)]
    if mat.medium_interface_idx < Int32(0):
        return
    var iface = medium_ifaces[Int(mat.medium_interface_idx)]
    var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var geom_n: Vec3f
    if inter.primId.type == 4:
        # Sphere: outward normal = hit point - center. Medium-bounding
        # volumes (e.g. smoke-plume's "MediumInterface .. Shape sphere")
        # are commonly a big invisible sphere, so this case matters even
        # though spheres otherwise rarely carry materials with real shading.
        var sph = spheres[Int(inter.primId.id1)]
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
        var m = meshes[mi]
        var v0 = Int(m.vertexIndices[bv])
        var v1 = Int(m.vertexIndices[bv + 1])
        var v2 = Int(m.vertexIndices[bv + 2])
        var p0 = Vec3f(m.points[v0*4], m.points[v0*4+1], m.points[v0*4+2])
        var p1 = Vec3f(m.points[v1*4], m.points[v1*4+1], m.points[v1*4+2])
        var p2 = Vec3f(m.points[v2*4], m.points[v2*4+1], m.points[v2*4+2])
        geom_n = cross(p1 - p0, p2 - p0)
    if dot(ray_dir, geom_n) > Float32(0.0):
        path_ptr[].current_medium_idx = iface.outside_medium_idx
    else:
        path_ptr[].current_medium_idx = iface.inside_medium_idx


comptime MEDIUM_TRACK_MAX_ITERS: Int = 10000  # delta/ratio-tracking loop safety bound

@always_inline
def _med_spec_illum(
    c: RGB, wl: SampledWavelengths,
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin],
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin],
) -> SpectralSample:
    """RGB emission/radiance -> spectral, at the light boundary inside the
    medium kernel. Falls back to a flat spectrum when no spectral table is
    loaded, so a table-less build still transports the RGB magnitude."""
    if spectral_res <= 0:
        return SpectralSample(c.r, c.g, c.b, (c.r + c.g + c.b) * Float32(0.3333333))
    return rgb_illuminant_to_spectral_sample(spectral_coeffs, spectral_res,
        spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, c.r, c.g, c.b, wl)

@always_inline
def _volume_nee_light(
    path_ptr: UnsafePointer[PathState_C, MutExternalOrigin],
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
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    n_spheres: Int,
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin],
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin],
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
    if any_hit_bvh2_core(bvh2Nodes, primIds, meshes, curves, e_ray, e_tmax,
                         blasNodesArr, blasPrimIdsArr, instances, spheres, n_spheres,
                         materials=materials):
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
        traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, e_ray, ls.dist,
                           UnsafePointer(to=exit_i), blasNodesArr, blasPrimIdsArr,
                           instances, spheres, n_spheres)
        var span = ls.dist if exit_i.hit == Int8(0) else exit_i.tHit
        var Th = exp(-sigma_t_r * span)
        var ph_h = hg_phase(dot(wo, edir), g)
        var mis_h = Float32(1.0) if ls.is_delta else power_heuristic(ls.pdf, ph_h)
        path_ptr[].estimate += path_ptr[].throughput * _med_spec_illum(
        ls.Li, path_ptr[].wavelengths, spectral_coeffs, spectral_res,
        spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65) * (Th * ph_h * mis_h / ls.pdf)
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
    path_ptr[].estimate += path_ptr[].throughput * _med_spec_illum(
        ls.Li, path_ptr[].wavelengths, spectral_coeffs, spectral_res,
        spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65) * (Te * ph * mis / ls.pdf)


def _sample_medium_core(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    i: Int,
    mediums: UnsafePointer[Medium_C, MutExternalOrigin],
    n_mediums: Int,
    grids: UnsafePointer[Grid_C, MutExternalOrigin],
    nvdb_grids: UnsafePointer[NvdbGrid_C, MutExternalOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    n_area_lights: Int,
    lightSamplerCdf: UnsafePointer[Float32, MutExternalOrigin],
    n_light_sampler: Int,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin] = UnsafePointer[Sphere_C, MutExternalOrigin].unsafe_dangling(),
    n_spheres: Int = 0,
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    materials: UnsafePointer[Material_C, MutExternalOrigin] = UnsafePointer[Material_C, MutExternalOrigin].unsafe_dangling(),
    infiniteLights: UnsafePointer[InfiniteLight_C, MutExternalOrigin] = UnsafePointer[InfiniteLight_C, MutExternalOrigin].unsafe_dangling(),
    n_infinite_lights: Int = 0,
    distantLights: UnsafePointer[DistantLight_C, MutExternalOrigin] = UnsafePointer[DistantLight_C, MutExternalOrigin].unsafe_dangling(),
    n_distant_lights: Int = 0,
    pointLights: UnsafePointer[PointLight_C, MutExternalOrigin] = UnsafePointer[PointLight_C, MutExternalOrigin].unsafe_dangling(),
    n_point_lights: Int = 0,
    # Phase 7.3 (docs/A2_restir_migration_plan.md, project_restir_migration
    # memory): volume-scatter TEMPORAL reuse. Decomposed pointers, not one
    # `vol_io: VolReservoirIO` argument -- same defensive convention this
    # file already applies to SpectralHandle at this same kind of boundary
    # (see spectrum.mojo's comment on rgb_to_spectral_sample). `pixel_idx`
    # only means anything when this call came from gpu_render_sample (one
    # path per pixel); the wavefront batch path always leaves it at -1,
    # which the code below treats identically to "no reuse".
    vol_read: UnsafePointer[VolReservoir, MutExternalOrigin] = UnsafePointer[VolReservoir, MutExternalOrigin].unsafe_dangling(),
    vol_write: UnsafePointer[VolReservoir, MutExternalOrigin] = UnsafePointer[VolReservoir, MutExternalOrigin].unsafe_dangling(),
    pixel_idx: Int = -1,
    # One Int8 per PATH SLOT (indexed by `i`, this call's own index -- NOT
    # by pixel_idx), reset to 0 once at the start of this dispatch/frame by
    # the caller: guards against a single path scattering more than once
    # inside a dense medium within one frame (common -- see the long
    # comment at this buffer's read site for the real bug this fixes).
    vol_used: UnsafePointer[Int8, MutExternalOrigin] = UnsafePointer[Int8, MutExternalOrigin].unsafe_dangling(),
    # Phase 7.3 spatial reuse (2026-09-08): SAME G-buffers DI's own spatial
    # reuse already reads (handle[].atrous_depth_buf/gbuf_worldpos_buf on
    # GPU, depth_int/world_pos_int on CPU) -- harmless to pass unconditionally
    # (mirrors DI's own convention), vol_temporal_spatial_combine's own
    # `_is_real_ptr`/frame_w>0/frame_h>0 checks gate the spatial pass off
    # when they're not real or the caller (batch wavefront) has no G-buffer.
    vol_gbuf_depth: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    vol_gbuf_world_pos: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
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
    var path_ptr = paths + i
    if path_ptr[].active == 0:
        return
    var med_idx = Int(path_ptr[].current_medium_idx)
    if med_idx < 0 or med_idx >= n_mediums:
        return
    var inter = intersections[i]
    if inter.hit == 0:
        return
    var med = mediums[med_idx]
    var sigma_t = med.sigma_a + med.sigma_s
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var t_surf = inter.tHit
    var ray_org = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)

    var t_free: Float32
    var albedo_r: Float32
    # Two density sources share one Woodcock-tracking loop -- dense
    # "uniformgrid" (grid_idx) and sparse "nanovdb" (nvdb_idx), mutually
    # exclusive per medium (the parser never sets both). They differ only in
    # the per-candidate density lookup and majorant, so resolve which source
    # this medium has ONCE here, not per iteration. Resolved at function scope
    # (rather than inside the heterogeneous branch) because the volume-scatter
    # NEE further down needs the same grid + majorant to ratio-track its own
    # shadow ray. `use_dense` guards the grids[] index: a homogeneous medium
    # has grid_idx == -1 and must never index that array.
    var use_nvdb = med.nvdb_idx >= Int32(0)
    var use_dense = med.grid_idx >= Int32(0)
    var grid = grids[Int(med.grid_idx)] if use_dense else Grid_C(
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(), Int32(0), Int32(0), Int32(0),
        Point3f(Float32(0), Float32(0), Float32(0)), Point3f(Float32(0), Float32(0), Float32(0)),
        SIMD[DType.float32, 16](0), Float32(0))
    var nvdb_grid = nvdb_grids[Int(med.nvdb_idx)] if use_nvdb else NvdbGrid_C(
        UnsafePointer[UInt8, MutExternalOrigin].unsafe_dangling(), Int64(0), SIMD[DType.float32, 16](0),
        SIMD[DType.float32, 16](0), Vec3f(Float32(0), Float32(0), Float32(0)),
        Point3f(Float32(0), Float32(0), Float32(0)), Point3f(Float32(0), Float32(0), Float32(0)), Float32(0))
    var majorant_density = nvdb_grid.max_density if use_nvdb else grid.max_density
    var sigma_maj = majorant_density * sigma_t.r
    if use_dense or use_nvdb:
        # ── Heterogeneous: delta tracking ────────────────────────────────
        if sigma_maj <= Float32(0.0):
            path_ptr[].pcgState = pcg.state
            return
        # ── Segment-wise tracking with LOCAL majorants ───────────────────
        # Walk the ray one NanoVDB tree node at a time, using that node's own
        # max density as the local majorant (6.7x on disney-cloud vs one
        # global majorant -- see docs/09_volumetric_media.md, "Local
        # majorants"). Unbiased by the memorylessness of the exponential: an
        # overshoot just resumes sampling from the segment boundary under the
        # next node's majorant. uniformgrid keeps one segment spanning the
        # whole ray under the global majorant (no per-node structure to walk).
        var t = Float32(0.0)
        var collided = False
        var iters = 0
        var seg_end = Float32(-1.0)       # < t forces a majorant query on entry
        var sigma_maj_seg = Float32(0.0)
        var iray = nvdb_index_ray(nvdb_grid, ray_org, ray_dir) if use_nvdb else SIMD[DType.float32, 8](0)
        while iters < MEDIUM_TRACK_MAX_ITERS:
            iters += 1
            if t >= seg_end:
                if t >= t_surf:
                    break
                if use_nvdb:
                    var mr = nvdb_majorant_at_world(nvdb_grid, ray_org + t * ray_dir)
                    sigma_maj_seg = mr[0] * sigma_t.r
                    seg_end = min(nvdb_node_exit_t(iray, t, mr[1]), t_surf)
                else:
                    sigma_maj_seg = sigma_maj
                    seg_end = t_surf
                if sigma_maj_seg <= Float32(0.0):
                    t = seg_end
                    continue
            var u = pcg.next_float()
            var t_next = t + (-log(max(u, Float32(1e-7))) / sigma_maj_seg)
            if t_next >= seg_end:
                t = seg_end
                continue
            t = t_next
            var p_world = ray_org + t * ray_dir
            var density = nvdb_sample_density(nvdb_grid, p_world) if use_nvdb else grid_sample_density(grid, p_world)
            # ── Volumetric emission (pbrt NanoVDBMedium) ─────────────────
            # Accumulated at EVERY majorant candidate, weighted by the local
            # absorption fraction sigma_a/sigma_maj: that is the standard
            # unbiased estimator of the emitted-radiance integral along the
            # segment when distances are drawn against the majorant, and it
            # must happen before the collision test below (which breaks out
            # of the loop) so emission from the pass-through candidates is
            # not silently dropped. Temperature comes from a SECOND nanovdb
            # grid stored as an ordinary entry in the same array, so this
            # reuses nvdb_sample_density unchanged. Non-emissive media take
            # the nvdb_temp_idx < 0 branch and pay nothing.
            if med.nvdb_temp_idx >= Int32(0) and med.le_scale > Float32(0.0):
                var tgrid = nvdb_grids[Int(med.nvdb_temp_idx)]
                var tk = (nvdb_sample_density(tgrid, p_world) - med.temp_offset) * med.temp_scale
                if tk > Float32(100.0):
                    var sigma_a_real = density * med.sigma_a.r
                    path_ptr[].estimate += path_ptr[].throughput * _med_spec_illum(
                        blackbody_rgb(tk), path_ptr[].wavelengths, spectral_coeffs, spectral_res,
                        spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65
                    ) * (med.le_scale * sigma_a_real / sigma_maj_seg)
            var sigma_t_real = density * sigma_t.r
            var u2 = pcg.next_float()
            if u2 < sigma_t_real / sigma_maj_seg:
                collided = True
                break
        if not collided:
            path_ptr[].pcgState = pcg.state
            return
        t_free = t
        albedo_r = med.sigma_s.r / max(sigma_t.r, Float32(1e-7))  # density cancels — see docstring
    else:
        # ── Homogeneous: closed-form analytic transmittance ──────────────
        if sigma_t.r <= Float32(0.0):
            path_ptr[].pcgState = pcg.state
            return
        # The free-flight distance is sampled from the R channel's own
        # exponential, pdf(t) = sigma_t.r * exp(-sigma_t.r * t), so that
        # channel's transmittance is ALREADY accounted for by the sampling
        # probability. Only the RATIO of each channel's transmittance to the
        # sampled one survives as a weight:
        #     pass-through : exp(-(sigma_t_c - sigma_t.r) * t_surf)
        #     collision    : exp(-(sigma_t_c - sigma_t.r) * t) * sigma_s_c/sigma_s.r
        # (the sigma_s ratio because the analog scatter/absorb coin below
        # already applies sigma_s.r/sigma_t.r). For a GREY medium both
        # reduce to exactly 1 -- nothing to multiply at all.
        #
        # This used to multiply by the FULL exp(-sigma_t_c * t), double-
        # counting the transmittance the pdf already contains. That made
        # throughput decay exponentially per scattering event, which
        # Russian roulette then compensated for with 1/luminance factors --
        # so the medium came out both far too dark AND threw enormous
        # fireflies (measured against an analytic answer of exactly 1.0: a
        # conservative medium read 0.470 at tau=2, 6.0 at tau=4 and 1979 at
        # tau=8, with single pixels reaching 6.1e6).
        var u_free = pcg.next_float()
        t_free = -log(max(u_free, Float32(1e-7))) / sigma_t.r
        var t_seg = min(t_free, t_surf)
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
        # colour conversion.
        # Chromatic transmittance ratio -- UPSAMPLE sigma_t to the 4 hero
        # lanes FIRST via spec_refl_unbounded (the coefficient-safe smooth
        # upsampler; grey media pass through it exactly -- verified in
        # Tests/unit/test_coefficient_upsampling.mojo), THEN exponentiate
        # PER LANE. This used to compute the ratio in RGB
        # (exp(-sigma_t.g*t)/exp(-sigma_t.r*t), etc) and band-pick the
        # already-exponentiated triple -- band-picking and this order agree
        # (selection commutes with exp), but the RGB ratio itself was only
        # ever an approximation of "the medium's colour" using 3 discrete
        # samples. Smooth upsampling reconstructs a real spectral curve from
        # those same 3 samples instead, matching bdpt.mojo/sppm.mojo's
        # spectral_free_flight_weight (same fix, same derivation) so CPU
        # PT / GPU PT / VCM / SPPM all treat a medium's colour identically.
        # See docs/02_spectra_and_color.md, "Chromatic extinction".
        var sig_t_spec = spec_refl_unbounded(
            spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
            sigma_t.r, sigma_t.g, sigma_t.b, path_ptr[].wavelengths)
        var sig_t_r_ref = sigma_t.r
        path_ptr[].throughput *= SpectralSample(
            exp(-(sig_t_spec.v0 - sig_t_r_ref) * t_seg),
            exp(-(sig_t_spec.v1 - sig_t_r_ref) * t_seg),
            exp(-(sig_t_spec.v2 - sig_t_r_ref) * t_seg),
            exp(-(sig_t_spec.v3 - sig_t_r_ref) * t_seg))
        if t_free >= t_surf:
            path_ptr[].pcgState = pcg.state
            return
        # Chromatic scattering ratio; 1 for a grey medium.
        var ss_r = max(med.sigma_s.r, Float32(1e-30))
        path_ptr[].throughput *= rgb_bands_to_spectral_sample(
            Float32(1.0), med.sigma_s.g / ss_r, med.sigma_s.b / ss_r,
            path_ptr[].wavelengths)
        albedo_r = med.sigma_s.r / max(sigma_t.r, Float32(1e-7))

    # Both branches above already `return` early for the "no real collision"
    # case (homogeneous: t_free >= t_surf; heterogeneous: not collided) — so
    # reaching here always means a real scatter/absorb event at t_free.
    var p_scatter = albedo_r
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
        if n_area_lights > 0 and n_light_sampler > 0:
            var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
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
            var dist_ris = (VOL_RIS_DISTANCE and (not use_dense) and (not use_nvdb)
                and sigma_t.r > Float32(0.0) and t_surf > Float32(0.0)
                and sigma_t.g == sigma_t.r and sigma_t.b == sigma_t.r
                and med.sigma_s.g == med.sigma_s.r and med.sigma_s.b == med.sigma_s.r)
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
                var al = areaLights[light_idx]
                var lmesh = meshes[Int(al.meshIdx)]
                var lti = Int(pcg.next_uint() % UInt32(max(Int(al.n_tris), 1)))
                var r1 = pcg.next_float()
                var r2 = pcg.next_float()
                var lb = lti * 3
                var lv0 = Int(lmesh.vertexIndices[lb])
                var lv1 = Int(lmesh.vertexIndices[lb + 1])
                var lv2 = Int(lmesh.vertexIndices[lb + 2])
                var lp0 = Vec3f(lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
                var lp1 = Vec3f(lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
                var lp2 = Vec3f(lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])
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
                and _is_real_ptr(vol_used) and vol_used[i] == Int8(0))
            if vol_reuse_ok:
                vol_used[i] = Int8(1)
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
                    if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, curves, shad_ray, shad_tmax, blasNodesArr, blasPrimIdsArr, instances, spheres, n_spheres, materials=materials):
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
                            var grid_s = grids[Int(med.grid_idx)] if not use_nvdb_s else Grid_C(
                                UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(), Int32(0), Int32(0), Int32(0),
                                Point3f(Float32(0), Float32(0), Float32(0)), Point3f(Float32(0), Float32(0), Float32(0)),
                                SIMD[DType.float32, 16](0), Float32(0))
                            var nvdb_grid_s = nvdb_grids[Int(med.nvdb_idx)] if use_nvdb_s else NvdbGrid_C(
                                UnsafePointer[UInt8, MutExternalOrigin].unsafe_dangling(), Int64(0), SIMD[DType.float32, 16](0),
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
                            # and analytic spheres live in their own flat array.
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
                            var _exit_inter = InlineArray[Intersection_C, 1](fill=Intersection_C(
                                PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0)),
                                Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0)))
                            var exit_ptr = _exit_inter.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin]()
                            exit_ptr[0].hit = Int8(0)
                            traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, shad_ray,
                                               shad_tmax, exit_ptr, blasNodesArr, blasPrimIdsArr, instances)
                            test_spheres(spheres, n_spheres, shad_ray, exit_ptr)
                            # test_spheres ignores shad_tmax (it bounds only by
                            # an already-recorded closer hit), so a sphere past
                            # the light would otherwise set t_med > dist and
                            # over-attenuate instead of under-.
                            if exit_ptr[0].hit != Int8(0) and exit_ptr[0].tHit <= shad_tmax:
                                var exit_mat = materials[Int(exit_ptr[0].primId.materialIndex)]
                                if exit_mat.type == MatKind.interface:
                                    t_med = exit_ptr[0].tHit
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
                        var al_win = areaLights[Int(res.light_idx)]
                        var sel_lo = lightSamplerCdf[Int(res.light_idx)]
                        var sel_hi = lightSamplerCdf[Int(res.light_idx) + 1]
                        var sel_pdf_win = max(sel_hi - sel_lo, Float32(1e-6))
                        var mis_w = Float32(1.0)
                        if al_win.total_area > Float32(0.0):
                            var pdf_light = dist_sq * sel_pdf_win / (cos_l * al_win.total_area)
                            mis_w = power_heuristic(pdf_light, ph_a)
                        path_ptr[].estimate += path_ptr[].throughput * _med_spec_illum(
                            res.le * T, path_ptr[].wavelengths, spectral_coeffs, spectral_res,
                            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65) * (geom * ph_a * mis_w * res.state.w)

        # ── Volume scatter NEE — INFINITE (environment) light ────────────
        # Without this a medium lit ONLY by a sky dome -- which is every
        # nanovdb cloud scene in the pbrt-v4 corpus (bunny-cloud, explosion,
        # disney-cloud) and any uniformgrid scene with no area light -- got
        # NO direct lighting at scatter points at all: the block above is
        # gated on n_area_lights > 0 and samples triangle area lights only.
        # Such a medium was then lit purely by phase-sampled paths that
        # random-walk back out of it and happen to escape to the sky, which
        # is both far too dark (measured on bunny-cloud: the cloud came out
        # at ~0.49x the sky's radiance where the reference has it at
        # ~1.0-1.8x, and a 0.952-albedo medium must be roughly sky-bright)
        # and extremely high variance -- that is where the sparse bright
        # "firefly" dots on those renders came from.
        # ── Volume scatter NEE — distant / point / sphere / infinite ─────
        # The block above samples triangle AREA lights only, and is gated on
        # n_area_lights > 0. Every other light type contributed nothing at a
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
        if True:
            var scatter_w = scatter_pt.to_simd()
            var wo_v = -ray_dir
            for dl_i in range(n_distant_lights):
                _volume_nee_light(path_ptr, _sample_distant_light_nee(distantLights[dl_i]),
                    scatter_w, wo_v, med.g, pcg, use_nvdb, use_dense, grid, nvdb_grid, sigma_maj, sigma_t.r,
                    bvh2Nodes, primIds, meshes, curves, blasNodesArr, blasPrimIdsArr,
                    instances, spheres, n_spheres, materials,
                    spectral_coeffs, spectral_res, spectral_cie_x,
                    spectral_cie_y, spectral_cie_z, spectral_d65)
            for pl_i in range(n_point_lights):
                _volume_nee_light(path_ptr, _sample_point_light_nee(pointLights[pl_i], scatter_w),
                    scatter_w, wo_v, med.g, pcg, use_nvdb, use_dense, grid, nvdb_grid, sigma_maj, sigma_t.r,
                    bvh2Nodes, primIds, meshes, curves, blasNodesArr, blasPrimIdsArr,
                    instances, spheres, n_spheres, materials,
                    spectral_coeffs, spectral_res, spectral_cie_x,
                    spectral_cie_y, spectral_cie_z, spectral_d65)
            for sph_i in range(n_spheres):
                if spheres[sph_i].isAreaLight == Int8(1):
                    _volume_nee_light(path_ptr, _sample_sphere_light_nee(spheres[sph_i], n_spheres, scatter_w, pcg),
                        scatter_w, wo_v, med.g, pcg, use_nvdb, use_dense, grid, nvdb_grid, sigma_maj, sigma_t.r,
                        bvh2Nodes, primIds, meshes, curves, blasNodesArr, blasPrimIdsArr,
                        instances, spheres, n_spheres, materials,
                        spectral_coeffs, spectral_res, spectral_cie_x,
                        spectral_cie_y, spectral_cie_z, spectral_d65)
            for inf_i in range(n_infinite_lights):
                _volume_nee_light(path_ptr,
                    _sample_infinite_light_nee(infiniteLights[inf_i], Point2f(pcg.next_float(), pcg.next_float())),
                    scatter_w, wo_v, med.g, pcg, use_nvdb, use_dense, grid, nvdb_grid, sigma_maj, sigma_t.r,
                    bvh2Nodes, primIds, meshes, curves, blasNodesArr, blasPrimIdsArr,
                    instances, spheres, n_spheres, materials,
                    spectral_coeffs, spectral_res, spectral_cie_x,
                    spectral_cie_y, spectral_cie_z, spectral_d65)
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
        intersections[i].hit = Int8(0)  # no surface hit this bounce
    else:
        # Absorbed
        path_ptr[].pcgState = pcg.state
        path_ptr[].active = Int8(0)

def sample_medium_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    mediums: UnsafePointer[Medium_C, MutExternalOrigin],
    n_mediums_dp: Int64,
    grids: UnsafePointer[Grid_C, MutExternalOrigin],
    nvdb_grids: UnsafePointer[NvdbGrid_C, MutExternalOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    n_area_lights_dp: Int64,
    lightSamplerCdf: UnsafePointer[Float32, MutExternalOrigin],
    n_light_sampler_dp: Int64,
    count_dp: Int64,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin] = UnsafePointer[Sphere_C, MutExternalOrigin].unsafe_dangling(),
    n_spheres_dp: Int64 = Int64(0),
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    materials: UnsafePointer[Material_C, MutExternalOrigin] = UnsafePointer[Material_C, MutExternalOrigin].unsafe_dangling(),
    infiniteLights: UnsafePointer[InfiniteLight_C, MutExternalOrigin] = UnsafePointer[InfiniteLight_C, MutExternalOrigin].unsafe_dangling(),
    n_infinite_lights_dp: Int64 = Int64(0),
    distantLights: UnsafePointer[DistantLight_C, MutExternalOrigin] = UnsafePointer[DistantLight_C, MutExternalOrigin].unsafe_dangling(),
    n_distant_lights_dp: Int64 = Int64(0),
    pointLights: UnsafePointer[PointLight_C, MutExternalOrigin] = UnsafePointer[PointLight_C, MutExternalOrigin].unsafe_dangling(),
    n_point_lights_dp: Int64 = Int64(0),
    # Phase 7.3: only gpu_render_wavefront_kernels(...) callers that pass
    # use_vol_restir=1 AND real buffers get reuse -- see _sample_medium_core's
    # own comment for why these stay decomposed rather than one VolReservoirIO.
    use_vol_restir: Int32 = Int32(0),
    vol_read: UnsafePointer[VolReservoir, MutExternalOrigin] = UnsafePointer[VolReservoir, MutExternalOrigin].unsafe_dangling(),
    vol_write: UnsafePointer[VolReservoir, MutExternalOrigin] = UnsafePointer[VolReservoir, MutExternalOrigin].unsafe_dangling(),
    vol_used: UnsafePointer[Int8, MutExternalOrigin] = UnsafePointer[Int8, MutExternalOrigin].unsafe_dangling(),
    vol_gbuf_depth: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    vol_gbuf_world_pos: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    vol_frame_w: Int32 = Int32(0),
    vol_frame_h: Int32 = Int32(0),
):
    var n_spheres = Int(n_spheres_dp)
    var spectral_res = Int(spectral_res_dp)
    var n_mediums = Int(n_mediums_dp)
    var n_area_lights = Int(n_area_lights_dp)
    var n_light_sampler = Int(n_light_sampler_dp)
    var count = Int(count_dp)
    """GPU kernel wrapper: bounds-check, then call the SAME
    _sample_medium_core the CPU driver (render_all_tiles) calls."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    # tid IS the pixel index here only when use_vol_restir=1 -- that only
    # ever comes from gpu_render_sample (one path per pixel per dispatch),
    # mirroring shade_diffuse_gpu's identical restir_has_state contract.
    var vol_has_state = use_vol_restir != Int32(0) and _is_real_ptr(vol_read)
    _sample_medium_core(
        paths, intersections, tid, mediums, n_mediums, grids, nvdb_grids,
        bvh2Nodes, primIds, meshes, curves,
        blasNodesArr, blasPrimIdsArr, instances,
        areaLights, n_area_lights, lightSamplerCdf, n_light_sampler,
        spheres, n_spheres,
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
        materials, infiniteLights, Int(n_infinite_lights_dp),
        distantLights, Int(n_distant_lights_dp), pointLights, Int(n_point_lights_dp),
        vol_read=vol_read, vol_write=vol_write,
        pixel_idx=tid if vol_has_state else -1,
        vol_used=vol_used,
        vol_gbuf_depth=vol_gbuf_depth, vol_gbuf_world_pos=vol_gbuf_world_pos,
        vol_frame_w=vol_frame_w, vol_frame_h=vol_frame_h,
    )


def shade_hair_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    areaLightCount_dp: Int64,
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures_dp: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutExternalOrigin],
    n_distant_lights_dp: Int64,
    pointLights: UnsafePointer[PointLight_C, MutExternalOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: UnsafePointer[Float32, MutExternalOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutExternalOrigin],
    n_infinite_lights_dp: Int64,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: UnsafePointer[ShadowTask_C, MutExternalOrigin],
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
):
    var spectral_res = Int(spectral_res_dp)
    var areaLightCount = Int(areaLightCount_dp)
    var n_textures = Int(n_textures_dp)
    var n_distant_lights = Int(n_distant_lights_dp)
    var n_point_lights = Int(n_point_lights_dp)
    var n_light_sampler = Int(n_light_sampler_dp)
    var n_infinite_lights = Int(n_infinite_lights_dp)
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].pending_mat != MatKind.hair:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[tid]
    var mat = materials[Int(inter.primId.materialIndex)]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=UnsafePointer[NormalSlopeMap_C, MutExternalOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutExternalOrigin].unsafe_dangling(),
        gi_pending=UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_hair[True, False](path_ptr, inter, ctx, mat)


def shade_enqueue_shadow_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    materials: UnsafePointer[Material_C, MutExternalOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutExternalOrigin],
    areaLightCount: Int,
    textures: UnsafePointer[GpuTexture_C, MutExternalOrigin],
    n_textures: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutExternalOrigin],
    n_infinite_lights: Int,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    n_spheres: Int,
    shadow_tasks: UnsafePointer[ShadowTask_C, MutExternalOrigin],
    count: Int,
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    shadow_tasks[tid].active = Int32(0)
    var path_ptr = paths + tid
    if path_ptr[].active == 0:
        return
    var inter = intersections[tid]
    # Do NOT early-exit on miss — shade_nee_core adds env-light contribution there.
    var ls_shadow = LightSampler_C(UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(), Int32(0), Int32(0))
    var ctx_shadow = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin](),
        textures=textures, n_textures=n_textures,
        nmaps=UnsafePointer[NormalSlopeMap_C, MutExternalOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=Float32(0.0), sobol_matrices=UnsafePointer[UInt32, MutExternalOrigin].unsafe_dangling(), guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutExternalOrigin].unsafe_dangling(),
        gi_pending=UnsafePointer[GIPendingX1, MutExternalOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=UnsafePointer[DistantLight_C, MutExternalOrigin](), distant_count=0,
            point_lights=UnsafePointer[PointLight_C, MutExternalOrigin](), point_count=0,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls_shadow))
    shade_nee_core[True, True](path_ptr, inter, ctx_shadow)


# Phase 0.4 (docs/A2_restir_migration_plan.md): a task deferred by one
# material's per-pixel kernel this bounce (enqueue_shadow=True) must not be
# resolved twice, and a pixel shaded by a material that did NOT defer (still
# resolves its own shadow ray inline) must not have a stale prior-bounce
# task resolved in its place -- both need shadow_tasks[tid].active reset to
# 0 before any of this bounce's per-material kernels run, since only the one
# kernel matching pending_mat[tid] actually touches slot tid.
def reset_shadow_tasks_gpu(
    shadow_tasks: UnsafePointer[ShadowTask_C, MutExternalOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    shadow_tasks[tid].active = Int32(0)

def reset_restir_reservoirs_gpu(
    reservoirs: UnsafePointer[DIReservoir, MutExternalOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    """Clear ReSTIR DI reservoirs to "no candidate yet". Needed at scene
    upload and on every camera move: identity reprojection assumes the
    previous frame's reservoir describes THIS pixel's shading point, so a
    surviving reservoir after the camera moves would reuse a light chosen
    for a different view. The GPU film is cleared on the same events for
    exactly the same reason (gpu_clear_film)."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    reservoirs[tid] = di_reservoir_init()

def reset_restir_vol_reservoirs_gpu(
    reservoirs: UnsafePointer[VolReservoir, MutExternalOrigin],
    count_dp: Int64,
):
    """Clear volume ReSTIR reservoirs to "no candidate yet" -- the same
    identity-reprojection invalidation reason as reset_restir_reservoirs_gpu
    above, applied to Phase 7.3's per-pixel volume-scatter reservoirs."""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    reservoirs[tid] = vol_reservoir_init()


def reset_vol_used_gpu(
    used: UnsafePointer[Int8, MutExternalOrigin],
    count_dp: Int64,
):
    """Zero the per-pixel "already combined this frame" guard -- called once
    at the start of every gpu_render_sample dispatch, NOT on camera move
    (unlike reset_restir_vol_reservoirs_gpu above): this buffer has no
    cross-frame meaning at all, it only disambiguates within ONE frame's own
    bounce-round loop. See _sample_medium_core's vol_used comment."""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    used[tid] = Int8(0)


def traverse_shadow_rays_gpu(
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    shadow_tasks: UnsafePointer[ShadowTask_C, MutExternalOrigin],
    count_dp: Int64,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin] = UnsafePointer[Sphere_C, MutExternalOrigin].unsafe_dangling(),
    n_spheres_dp: Int64 = Int64(0),
    materials: UnsafePointer[Material_C, MutExternalOrigin] = UnsafePointer[Material_C, MutExternalOrigin].unsafe_dangling(),
):
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var task = shadow_tasks[tid]
    if task.active == 0:
        return
    var shadow_ray = Ray_C(Point3f(task.origin.x, task.origin.y, task.origin.z), Vec3f(task.direction.x, task.direction.y, task.direction.z))
    if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, curves, shadow_ray, task.tmax, blasNodesArr, blasPrimIdsArr, instances, spheres, n_spheres, materials=materials):
        paths[tid].estimate += task.contrib


def accumulate_film_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    film: UnsafePointer[Float32, MutExternalOrigin],
    albedo_film: UnsafePointer[Float32, MutExternalOrigin],
    count_dp: Int64,
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    # ── Output boundary: spectral transport -> RGB film ──────────────────
    var _e = spectral_sample_to_rgb(spectral_coeffs, Int(spectral_res_dp),
        spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
        paths[tid].estimate, paths[tid].wavelengths)
    film[tid*3+0] += _e[0]
    film[tid*3+1] += _e[1]
    film[tid*3+2] += _e[2]
    albedo_film[tid*3+0] += paths[tid].albedo.r
    albedo_film[tid*3+1] += paths[tid].albedo.g
    albedo_film[tid*3+2] += paths[tid].albedo.b


def clear_film_gpu(film: UnsafePointer[Float32, MutExternalOrigin], n_pixels_dp: Int64):
    var n_pixels = Int(n_pixels_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n_pixels:
        return
    film[tid*3+0] = Float32(0)
    film[tid*3+1] = Float32(0)
    film[tid*3+2] = Float32(0)


# Wavefront accumulation: thread px sums actual_batch samples from path_buf layout
# path_buf[si * n_pixels + px] and adds to film[px].  No atomics needed (one thread per pixel).
def accumulate_film_wavefront_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    film: UnsafePointer[Float32, MutExternalOrigin],
    albedo_film: UnsafePointer[Float32, MutExternalOrigin],
    n_pixels_dp: Int64, actual_batch_dp: Int64,
    spectral_coeffs: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
):
    var n_pixels = Int(n_pixels_dp)
    var actual_batch = Int(actual_batch_dp)
    var px = Int(block_idx.x * block_dim.x + thread_idx.x)
    if px >= n_pixels:
        return
    var r = Float32(0); var g = Float32(0); var b = Float32(0)
    var ar = Float32(0); var ag = Float32(0); var ab = Float32(0)
    for si in range(actual_batch):
        var p = paths[si * n_pixels + px]
        var _pe = spectral_sample_to_rgb(spectral_coeffs, Int(spectral_res_dp),
            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
            p.estimate, p.wavelengths)
        r += _pe[0]; g += _pe[1]; b += _pe[2]
        ar += p.albedo.r;  ag += p.albedo.g;  ab += p.albedo.b
    film[px*3+0] += r; film[px*3+1] += g; film[px*3+2] += b
    albedo_film[px*3+0] += ar; albedo_film[px*3+1] += ag; albedo_film[px*3+2] += ab


# Wavefront primary-ray generation: thread ti → pixel (ti % n_pixels), sample (si_start + ti // n_pixels).
# Layout: path_buf[si_local * n_pixels + px_flat] — adjacent threads touch adjacent pixels of same sample.
def gen_primary_rays_wavefront_gpu(
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
    r2c: UnsafePointer[Float32, MutExternalOrigin],
    c2w: UnsafePointer[Float32, MutExternalOrigin],
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    fw_dp: Int64, fh_dp: Int64,
    si_start: Int32, log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    filter_sigma: Float32, filter_norm_x: Float32, filter_support_x: Float32,
    filter_norm_y: Float32, filter_support_y: Float32,
    filter_type: Int32,
    count_dp: Int64, n_pixels_dp: Int64,
):
    var fw = Int(fw_dp)
    var fh = Int(fh_dp)
    var count = Int(count_dp)
    var n_pixels = Int(n_pixels_dp)
    var ti = Int(block_idx.x * block_dim.x + thread_idx.x)
    if ti >= count:
        return
    var si_local = ti // n_pixels
    var px_flat  = ti % n_pixels
    var px = Int32(px_flat % fw)
    var py = Int32(px_flat // fw)
    var si = si_start + Int32(si_local)
    var rng_seed = UInt64(rng_seed_hi) << UInt64(32) | UInt64(rng_seed_lo)
    var (ray, pcg_state, pcg_inc, sobol_idx, wavelengths) = gen_primary_ray_state(
        px, py, si, Int(log2spp), Int(n_base4),
        seed_dim0, seed_dim1, rng_seed, sobol_matrices, r2c, c2w,
        filter_norm_x, filter_sigma, filter_support_x,
        filter_norm_y, filter_support_y,
        filter_type,
    )
    paths[ti] = PathState_C(
        ray,
        SpectralSample(Float32(1.0)),
        SpectralSample(Float32(0.0)),
        RGB(Float32(0.0)),
        Int32(0), pcg_state, pcg_inc,
        Int8(1), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0), Vec3f(Float32(0.0)),
        Float32(0.0),
        Int32(-1),
        Float32(1.0),   # current_dielectric_ior (vacuum)
        Int32(3), sobol_idx,
        wavelengths,
        Float32(0.0),   # mis_null_dist
    )


# Traversal kernel that reads rays directly from PathState_C (no separate ray buffer).
def traverse_paths_gpu(
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    n_spheres_dp: Int64,
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    results: UnsafePointer[Intersection_C, MutExternalOrigin],
    curve_cand_prim: UnsafePointer[Int32, MutExternalOrigin],
    curve_cand_count: UnsafePointer[Int32, MutExternalOrigin],
    count_dp: Int64,
):
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if paths[tid].active == 0:
        return
    curve_cand_count[tid] = Int32(0)
    traverse_bvh2_core_defer_curves(
        bvh2Nodes, primIds, meshes, curves, paths[tid].ray, Float32(1.0e38), results + tid,
        curve_cand_prim + tid * CURVE_DEFER_K, curve_cand_count + tid,
        blasNodesArr, blasPrimIdsArr, instances,
    )
    test_spheres(spheres, n_spheres, paths[tid].ray, results + tid)


# Task #163 stage 3: GPU-resident replacement for the `traverse_paths_gpu`
# dispatch in gpu_render_wavefront's per-bounce loop, routing the primary
# per-bounce intersection test through hardware ray tracing via the real
# CUDA/Vulkan interop mechanism (src/vulkaninterop -- zero CPU
# synchronization) instead of vulkanrt.mojo's host-round-trip
# vulkanrt_trace_rays (measured ~94x slower in an earlier version of this
# integration; see project_vulkan_rt_backend memory). Every other
# wavefront stage (medium sampling, all shade_*_gpu kernels, film
# accumulation) is completely unchanged -- they only ever read
# path_buf/inter_buf, never re-run primary-hit traversal themselves
# (shadow/NEE rays are a separate code path inside the shade kernels and
# stay on the CUDA/software BVH, not touched here).
#
# Pack ray data from path_buf directly into the interop-shared rays buffer
# -- no host copy, this kernel writes straight into CUDA-mapped memory
# that a Vulkan compute shader also reads.
def vulkaninterop_pack_rays_kernel(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    rays: UnsafePointer[Float32, MutExternalOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var ray = paths[tid].ray
    var idx = tid * 8
    rays[idx + 0] = ray.origin.x
    rays[idx + 1] = ray.origin.y
    rays[idx + 2] = ray.origin.z
    rays[idx + 3] = Float32(1e-4)
    rays[idx + 4] = ray.direction.x
    rays[idx + 5] = ray.direction.y
    rays[idx + 6] = ray.direction.z
    rays[idx + 7] = Float32(1.0e8)

# Unpack the interop-shared results buffer (written by Vulkan's ray-query
# dispatch) directly into inter_buf's Intersection_C layout -- no host
# copy. Same materialIndex/area-light lookup and PrimId_C.type==3 encoding
# as the retired host-loop version (see finalize_scene in pbrt_parser.mojo
# for why area-light triangles need that special encoding, and
# project_vulkan_rt_backend memory for the real-bug story behind it) --
# now done per-thread on the GPU instead of a Mojo host loop, so
# mesh_material_idx/mesh_al_idx must be GPU-resident buffers here (see
# _build_mesh_light_info + the upload step in pipeline.mojo).
#
# Object instancing: a hit's hitMesh (instanceCustomIndex) is either an
# ordinary mesh index (< n_meshes, unchanged from before instancing existed)
# or mesh_count + instance_index (see vulkaninterop_rt_create_scene). For
# the latter, instance_base_mesh[instance_index] (host-precomputed as
# template_mesh_start[instances[k].blasIdx], pipeline.mojo) plus the hit's
# geometryIndex gives the real originating mesh index -- AreaLightSource is
# not supported inside ObjectBegin/ObjectEnd (pbrt_parser.mojo skips it
# there), so instance hits are always ordinary (type==0) triangles, never
# the type==3 area-light encoding.
def vulkaninterop_unpack_results_kernel(
    results: UnsafePointer[Float32, MutExternalOrigin],
    inter: UnsafePointer[Intersection_C, MutExternalOrigin],
    mesh_material_idx: UnsafePointer[Int64, MutExternalOrigin],
    mesh_al_idx: UnsafePointer[Int32, MutExternalOrigin],
    n_meshes_dp: Int64,
    count_dp: Int64,
    instance_base_mesh: UnsafePointer[Int32, MutExternalOrigin] = UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling(),
):
    var n_meshes = Int(n_meshes_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var idx = tid * 8
    var iresults = results.bitcast[Int32]()
    var hitFlag = iresults[idx + 6]
    if hitFlag == Int32(1):
        var raw_idx = Int(iresults[idx + 4])
        var tri = iresults[idx + 5]
        var geometry_idx = iresults[idx + 7]
        var instance_idx = Int32(-1)
        var mi = raw_idx
        if raw_idx >= n_meshes and _is_real_ptr(instance_base_mesh):
            instance_idx = Int32(raw_idx - n_meshes)
            mi = Int(instance_base_mesh[Int(instance_idx)]) + Int(geometry_idx)
        var mat_idx = Int64(0)
        var al = Int32(-1)
        if mi >= 0 and mi < n_meshes:
            mat_idx = mesh_material_idx[mi]
            if instance_idx < Int32(0):
                al = mesh_al_idx[mi]
        var hitT = results[idx + 0]
        var u = results[idx + 1]
        var v = results[idx + 2]
        if al >= Int32(0):
            inter[tid] = Intersection_C(
                PrimId_C(Int64(al), (Int64(mi) << 32) | Int64(tri), mat_idx, Int32(-1),
                         Int8(3), Int8(0), Int8(0), Int8(0)),
                hitT, u, v, Int8(1), Int8(0), Int8(0), Int8(0),
            )
        else:
            inter[tid] = Intersection_C(
                PrimId_C(Int64(mi), Int64(tri) * 3, mat_idx, instance_idx,
                         Int8(0), Int8(0), Int8(0), Int8(0)),
                hitT, u, v, Int8(1), Int8(0), Int8(0), Int8(0),
            )
    elif hitFlag == Int32(2):
        # Curve hit, resolved directly by intersect_batch.comp itself (real
        # ray-vs-curve narrow-phase test + rayQueryGenerateIntersectionEXT
        # on a valid hit) -- no candidate buffer, no separate CUDA resolve
        # pass. hitMesh/hitTriangle/geometryIndex already carry PrimId_C's
        # id1 (curve index)/id2 (packed piece info)/materialIndex directly
        # (see vulkaninterop_rt_create_scene's docstring), so this is a
        # straight repack, no lookups needed. u/v here are intersect_curve's
        # own (h, v) outputs, not barycentrics.
        var curve_idx = Int64(iresults[idx + 4])
        var piece_info = Int64(iresults[idx + 5])
        var mat_idx = Int64(iresults[idx + 7])
        var hitT = results[idx + 0]
        var h = results[idx + 1]
        var v = results[idx + 2]
        inter[tid] = Intersection_C(
            PrimId_C(curve_idx, piece_info, mat_idx, Int32(-1), Int8(5), Int8(0), Int8(0), Int8(0)),
            hitT, h, v, Int8(1), Int8(0), Int8(0), Int8(0),
        )
    else:
        # tHit must be reset to the "no hit yet" sentinel (matching
        # traverse_bvh2_core_defer_curves's convention), not just hit=0 --
        # vulkaninterop_test_spheres_gpu (called right after this kernel)
        # uses result[0].tHit as its starting max-distance. Leaving it at
        # whatever inter_buf held from an earlier bounce would silently
        # reject a real closer sphere hit as "farther than the (stale)
        # current best".
        var dummy_id = PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
        inter[tid] = Intersection_C(dummy_id, Float32(1.0e38), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))

# Analytic sphere test as a SEPARATE pass after the Vulkan-RT-traced mesh/
# instance hit above -- exactly mirrors how traverse_paths_gpu (the pure-
# CUDA path) composes traverse_bvh2_core_defer_curves + test_spheres
# already: test_spheres itself keeps whichever hit is closer (it reads
# result[0].tHit as its starting max-distance when result[0].hit != 0), so
# calling it here with whatever vulkaninterop_unpack_results_kernel just
# wrote is correct without any extra "closest of two" logic on this side.
# Spheres stay purely analytic (exact intersection/normals) rather than
# tessellated into the Vulkan BLAS/TLAS -- simpler, and free of any
# tessellation-precision tradeoff.
def vulkaninterop_test_spheres_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    inter: UnsafePointer[Intersection_C, MutExternalOrigin],
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    n_spheres_dp: Int64,
    count_dp: Int64,
):
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if paths[tid].active == 0:
        return
    test_spheres(spheres, n_spheres, paths[tid].ray, inter + tid)

# Host driver: enqueues pack -> interop ray-query dispatch -> unpack, ALL
# on ctx.stream() in strict program order, with NO ctx.synchronize()
# anywhere -- vulkaninterop_rt_trace's internal CUDA-signal/Vulkan-submit/
# CUDA-wait handoff (see vulkaninterop.h) is what keeps the Vulkan
# dispatch correctly ordered relative to the pack/unpack kernels on either
# side of it, entirely on the GPU timeline. This is what makes stage 3
# fast where the retired host-round-trip version was ~94x slower: no CPU
# stall, no host-memory copy, anywhere in this function.
#
# Scope: triangle/instance/curve geometry via Vulkan RT, PLUS an analytic
# sphere pass (see vulkaninterop_test_spheres_gpu) -- meshes, instances,
# spheres, AND curves are all supported now (see vulkaninterop_rt_create_
# scene's docstring for how curves are represented).
def vulkaninterop_rt_traverse_paths_gpu(
    ctx: DeviceContext,
    path_buf: DeviceBuffer[DType.uint8],
    inter_buf: DeviceBuffer[DType.uint8],
    interop_scene: VulkanInteropRtSceneHandle,
    interop_rays_buf: DeviceBuffer[DType.float32],
    interop_results_buf: DeviceBuffer[DType.float32],
    mesh_material_idx_buf: DeviceBuffer[DType.uint8],
    mesh_al_idx_buf: DeviceBuffer[DType.uint8],
    n_meshes: Int,
    n_total: Int,
    instance_base_mesh_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin] = UnsafePointer[Sphere_C, MutExternalOrigin].unsafe_dangling(),
    n_spheres: Int = 0,
) raises:
    comptime block_size = 256
    var grid = ceildiv(n_total, block_size)

    ctx.enqueue_function[vulkaninterop_pack_rays_kernel](
        path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        interop_rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        Int64(n_total),
        grid_dim=grid, block_dim=block_size,
    )

    var cuda_stream = CUDA(ctx.stream())
    _ = vulkaninterop_rt_trace(interop_scene, Int32(n_total), cuda_stream)

    var instance_base_mesh_ptr = UnsafePointer[Int32, MutExternalOrigin].unsafe_dangling()
    if instance_base_mesh_buf:
        instance_base_mesh_ptr = instance_base_mesh_buf.value().unsafe_ptr().bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin]()

    ctx.enqueue_function[vulkaninterop_unpack_results_kernel](
        interop_results_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        mesh_material_idx_buf.unsafe_ptr().bitcast[Int64]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        mesh_al_idx_buf.unsafe_ptr().bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        Int64(n_meshes),
        Int64(n_total),
        instance_base_mesh_ptr,
        grid_dim=grid, block_dim=block_size,
    )

    # Curves need no further handling here at all -- intersect_batch.comp
    # already resolved them (real narrow-phase test + generate on a valid
    # hit) as part of the SAME dispatch, and vulkaninterop_unpack_results_
    # kernel above already decoded a curve hit (hitFlag==2) into inter_buf.

    if n_spheres > 0:
        ctx.enqueue_function[vulkaninterop_test_spheres_gpu](
            path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
            inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
            spheres,
            Int64(n_spheres),
            Int64(n_total),
            grid_dim=grid, block_dim=block_size,
        )


# ── Curve-divergence-mitigation kernels (companions to traverse_paths_gpu) ────
# See traverse_bvh2_core_defer_curves for the rationale. This is a 3-kernel
# pipeline run once per bounce, only when the scene has curves:
#   1. traverse_paths_gpu records up to CURVE_DEFER_K candidate curve prims per
#      ray instead of testing them inline (above).
#   2. compact_curve_paths_gpu streams the (typically sparse) subset of rays
#      that recorded >=1 candidate into a dense array via a single atomic append
#      per curve-touching ray — cheap even though divergent, since there's no
#      expensive math in this kernel, just a compare-and-maybe-atomic.
#   3. resolve_curve_candidates_gpu processes that dense array: each thread owns
#      exactly one ray (compaction guarantees no ray appears twice), so there is
#      no race updating `results`, and warps are densely packed with real work
#      instead of ~92% idle lanes.

def reset_curve_counter_gpu(counter: UnsafePointer[Int32, MutExternalOrigin]):
    if block_idx.x == 0 and thread_idx.x == 0:
        counter[0] = Int32(0)

# The CUDA-native path (traverse_bvh2_core_defer_curves) always writes
# curve candidates at tid*CURVE_DEFER_K -- this reproduces that formula once
# at buffer-creation time so curve_cand_offset_buf is valid immediately.
# Vulkan RT rendering never touches curve_cand_prim_buf/curve_cand_count_buf/
# curve_cand_offset_buf at all anymore -- intersect_batch.comp resolves
# curve hits itself and writes them straight into the ordinary results
# buffer (see vulkaninterop_unpack_results_kernel's hitFlag==2 branch), so
# compact_curve_paths_gpu/resolve_curve_candidates_gpu never run for a
# Vulkan-RT render (see _gpu_bounce_kernels).
def init_curve_cand_offset_gpu(offset_buf: UnsafePointer[Int32, MutExternalOrigin], n_dp: Int64):
    var n = Int(n_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n:
        return
    offset_buf[tid] = Int32(tid * CURVE_DEFER_K)

def compact_curve_paths_gpu(
    curve_cand_count: UnsafePointer[Int32, MutExternalOrigin],
    n_dp: Int64,
    compact_pathIds: UnsafePointer[Int32, MutExternalOrigin],
    compact_counter: UnsafePointer[Int32, MutExternalOrigin],
):
    var n = Int(n_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n:
        return
    if curve_cand_count[tid] > Int32(0):
        var pos = Atomic.fetch_add(compact_counter, Int32(1))
        compact_pathIds[Int(pos)] = Int32(tid)

def resolve_curve_candidates_gpu(
    compact_pathIds: UnsafePointer[Int32, MutExternalOrigin],
    compact_counter: UnsafePointer[Int32, MutExternalOrigin],
    curve_cand_prim: UnsafePointer[Int32, MutExternalOrigin],
    curve_cand_count: UnsafePointer[Int32, MutExternalOrigin],
    curve_cand_offset: UnsafePointer[Int32, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    results: UnsafePointer[Intersection_C, MutExternalOrigin],
    n_dp: Int64,
):
    var n = Int(n_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n:
        return
    if tid >= Int(compact_counter[0]):
        return
    var pathId = Int(compact_pathIds[tid])
    var n_cand = Int(curve_cand_count[pathId])
    if n_cand == 0:
        return
    var base = Int(curve_cand_offset[pathId])
    var ray = paths[pathId].ray
    var ray_org = Vec3f(ray.origin.x, ray.origin.y, ray.origin.z)
    var ray_dir = Vec3f(ray.direction.x, ray.direction.y, ray.direction.z)
    var res = results[pathId]
    var best_t = res.tHit
    var best_u = res.u
    var best_v = res.v
    var best_prim = res.primId
    var best_hit = res.hit
    var changed = False
    for i in range(n_cand):
        var primIdx = Int(curve_cand_prim[base + i])
        var prim = primIds[primIdx]
        var curve = curves[Int(prim.id1)]
        var curve_hit = intersect_curve(ray_org, ray_dir, curve, Int(prim.id2) // 8, Int(prim.id2) % 8, best_t)
        if curve_hit[0]:
            best_t = curve_hit[1]
            best_u = curve_hit[2]
            best_v = curve_hit[3]
            best_prim = prim
            best_hit = Int8(1)
            changed = True
    if changed:
        results[pathId] = Intersection_C(best_prim, best_t, best_u, best_v, best_hit, 0, 0, 0)

def gpu_shade_batch(
    handlePtr: UnsafePointer[GpuSceneHandle, MutExternalOrigin],
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    count: Int64,
    intersections: UnsafePointer[Intersection_C, MutExternalOrigin]
):
    if not handlePtr:
        return
    var handle = handlePtr
    var n = Int(count)
    if n == 0:
        return

    comptime if has_accelerator():
        try:
            var path_bytes = n * size_of[PathState_C]()
            var inter_bytes = n * size_of[Intersection_C]()

            var path_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](path_bytes)
            with path_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = paths.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=path_bytes)

            var inter_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](inter_bytes)
            with inter_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = intersections.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=inter_bytes)

            # Launch shading kernel
            comptime block_size = 256
            var grid_dim = ceildiv(n, block_size)

            handle[].ctx.enqueue_function[shade_gpu](
                path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
                inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
                handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
                handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
                null_spectral_handle(),
                Int64(n),
                grid_dim=grid_dim,
                block_dim=block_size,
            )

            handle[].ctx.synchronize()

            # Transfer path back (they were updated in-place on the device)
            with path_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = paths.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=path_bytes)

        except e:
            print("GPU: Batch shading failed: " + String(e))

# GPU kernel: generate primary PathState_C for every pixel in one pass.
# Each thread handles one pixel.  All sampling is pure math — no host calls.
def gen_primary_rays_gpu(
    sobol_matrices: UnsafePointer[UInt32, MutExternalOrigin],
    r2c: UnsafePointer[Float32, MutExternalOrigin],
    c2w: UnsafePointer[Float32, MutExternalOrigin],
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    fw_dp: Int64, fh_dp: Int64,
    si: Int32, log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    filter_sigma: Float32, filter_norm_x: Float32, filter_support_x: Float32,
    filter_norm_y: Float32, filter_support_y: Float32,
    filter_type: Int32,
    count_dp: Int64,
):
    var fw = Int(fw_dp)
    var fh = Int(fh_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var px = Int32(tid % fw)
    var py = Int32(tid // fw)
    var rng_seed = UInt64(rng_seed_hi) << UInt64(32) | UInt64(rng_seed_lo)
    var (ray, pcg_state, pcg_inc, sobol_idx, wavelengths) = gen_primary_ray_state(
        px, py, si, Int(log2spp), Int(n_base4),
        seed_dim0, seed_dim1, rng_seed, sobol_matrices, r2c, c2w,
        filter_norm_x, filter_sigma, filter_support_x,
        filter_norm_y, filter_support_y,
        filter_type,
    )
    paths[tid] = PathState_C(
        ray,
        SpectralSample(Float32(1.0)),
        SpectralSample(Float32(0.0)),
        RGB(Float32(0.0)),
        Int32(0), pcg_state, pcg_inc,
        Int8(1), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0), Vec3f(Float32(0.0)),
        Float32(0.0),
        Int32(-1),
        Float32(1.0),   # current_dielectric_ior (vacuum)
        Int32(3), sobol_idx,
        wavelengths,
        Float32(0.0),   # mis_null_dist
    )


# GPU kernel: shoot one unjittered center ray per pixel and write normals + depth.
# Used to guide the à-trous denoiser with geometric edge information.
def gen_aux_buffers_gpu(
    r2c: UnsafePointer[Float32, MutExternalOrigin],
    c2w: UnsafePointer[Float32, MutExternalOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutExternalOrigin],
    primIds: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    curves: UnsafePointer[Curve_C, MutExternalOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutExternalOrigin], MutExternalOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutExternalOrigin], MutExternalOrigin],
    instances: UnsafePointer[Instance_C, MutExternalOrigin],
    spheres: UnsafePointer[Sphere_C, MutExternalOrigin],
    n_spheres_dp: Int64,
    isects_tmp: UnsafePointer[Intersection_C, MutExternalOrigin],
    normals_out: UnsafePointer[Float32, MutExternalOrigin],
    depth_out: UnsafePointer[Float32, MutExternalOrigin],
    curve_mask_out: UnsafePointer[Float32, MutExternalOrigin],
    world_pos_out: UnsafePointer[Float32, MutExternalOrigin],
    material_id_out: UnsafePointer[Int32, MutExternalOrigin],
    fw_dp: Int64, fh_dp: Int64,
):
    var n_spheres = Int(n_spheres_dp)
    var fw = Int(fw_dp)
    var fh = Int(fh_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= fw * fh:
        return
    var px = tid % fw
    var py = tid // fw
    var filmX = Float32(px) + Float32(0.5)
    var filmY = Float32(py) + Float32(0.5)

    # Raster → camera
    var cx = r2c[0]*filmX + r2c[4]*filmY + r2c[12]
    var cy = r2c[1]*filmX + r2c[5]*filmY + r2c[13]
    var cz = r2c[2]*filmX + r2c[6]*filmY + r2c[14]
    var cw = r2c[3]*filmX + r2c[7]*filmY + r2c[15]
    if cw != Float32(0.0) and cw != Float32(1.0):
        cx /= cw; cy /= cw; cz /= cw
    var cl = sqrt(cx*cx + cy*cy + cz*cz)
    if cl > Float32(0): cx /= cl; cy /= cl; cz /= cl

    # Camera → world
    var dir = Vec3f(
        c2w[0]*cx + c2w[4]*cy + c2w[8]*cz,
        c2w[1]*cx + c2w[5]*cy + c2w[9]*cz,
        c2w[2]*cx + c2w[6]*cy + c2w[10]*cz,
    )
    var dl = dir.length()
    if dl > Float32(0): dir = dir / dl
    var org = Point3f(c2w[12], c2w[13], c2w[14])

    var ray = Ray_C(org, dir)
    var dummy_id = PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    isects_tmp[tid] = Intersection_C(dummy_id, Float32(1e38), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
    traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, ray, Float32(1e38), isects_tmp + tid, blasNodesArr, blasPrimIdsArr, instances)
    test_spheres(spheres, n_spheres, ray, isects_tmp + tid)

    var normal = Vec3f(Float32(0), Float32(0), Float32(1))
    var d = Float32(1e38)

    if isects_tmp[tid].hit != Int8(0):
        d = isects_tmp[tid].tHit
        var typ = Int(isects_tmp[tid].primId.type)
        if typ == 4:
            var si = Int(isects_tmp[tid].primId.id1)
            normal = sphere_outward_normal(org + dir*d, spheres[si].center)
        elif typ == 5:
            # Approximate outward normal for the denoiser G-buffer: h alone
            # (stored in isects_tmp.u) doesn't uniquely fix the azimuthal sign,
            # so this picks one consistent side — fine for denoising, not used
            # for shading (shade_hair derives its own frame independently).
            var curve = curves[Int(isects_tmp[tid].primId.id1)]
            var piece = min(Int(curve.n_pieces) - 1, Int(isects_tmp[tid].v * Float32(curve.n_pieces)))
            var (cq0, cq1, _, _) = curve_piece_endpoints(curve, piece)
            var caxis = cq1 - cq0
            var calen = sqrt(dot(caxis, caxis))
            if calen > Float32(1e-8):
                var ctangent = caxis * (Float32(1.0) / calen)
                var cu = _curve_perp_axis(ctangent)
                var cb = cross(ctangent, cu)
                var ch = isects_tmp[tid].u
                var cs = sqrt(max(Float32(0.0), Float32(1.0) - ch*ch))
                var cn = cu*ch + cb*cs
                normal = vec3f(cn)
        elif typ == 0 or typ == 1 or typ == 2 or typ == 3:
            var mesh_idx: Int
            var base_vidx: Int
            if typ == 0:
                mesh_idx  = Int(isects_tmp[tid].primId.id1)
                base_vidx = Int(isects_tmp[tid].primId.id2)
            else:
                mesh_idx  = Int(isects_tmp[tid].primId.id2 >> 32)
                base_vidx = Int(isects_tmp[tid].primId.id2 & 0xFFFFFFFF) * 3
            var mesh = meshes[mesh_idx]
            var vi0 = Int(mesh.vertexIndices[base_vidx])
            var vi1 = Int(mesh.vertexIndices[base_vidx + 1])
            var vi2 = Int(mesh.vertexIndices[base_vidx + 2])
            var p0 = Point3f(mesh.points[vi0*4], mesh.points[vi0*4+1], mesh.points[vi0*4+2])
            var p1 = Point3f(mesh.points[vi1*4], mesh.points[vi1*4+1], mesh.points[vi1*4+2])
            var p2 = Point3f(mesh.points[vi2*4], mesh.points[vi2*4+1], mesh.points[vi2*4+2])
            var e1 = p1 - p0; var e2 = p2 - p0
            normal = Vec3f(e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x)
            var inst_idx = isects_tmp[tid].primId.instanceIdx
            if inst_idx >= Int32(0):
                var n_world = transform_normal_by_instance(instances[Int(inst_idx)].worldToObj, normal.to_simd())
                normal = vec3f(n_world)
            var nl = normal.length()
            if nl > Float32(0): normal = normal / nl
        # else: unrecognized primitive type (shouldn't happen — every hit
        # traverse_bvh2_core can produce is type 0-5) — leave normal at the
        # miss-ray default (0,0,1) rather than misreading id1/id2 as mesh data.
        if normal.dot(-dir) < Float32(0):
            normal = -normal

    normals_out[tid*3+0] = normal.x
    normals_out[tid*3+1] = normal.y
    normals_out[tid*3+2] = normal.z
    depth_out[tid] = d
    curve_mask_out[tid] = Float32(1.0) if (isects_tmp[tid].hit != Int8(0) and Int(isects_tmp[tid].primId.type) == 5) else Float32(0.0)
    store_vec3(world_pos_out, tid, (org + dir*d).to_simd())
    material_id_out[tid] = Int32(isects_tmp[tid].primId.materialIndex) if isects_tmp[tid].hit != Int8(0) else Int32(-1)


def gpu_gen_aux_buffers[Oc: Origin[mut=True]](
    handlePtr: UnsafePointer[GpuSceneHandle, MutExternalOrigin],
    c2w: UnsafePointer[Float32, Oc],
    n: Int64,
):
    """Generate unjittered normals and depth buffers for the denoiser."""
    var n_pix = Int(n)
    if n_pix == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            with handle[].c2w_buf.map_to_host() as h:
                var dst = h.unsafe_ptr().bitcast[Float32]()
                memcpy(dest=dst, src=c2w, count=16)
            comptime block_size = 256
            var grid_n = ceildiv(n_pix, block_size)
            handle[].ctx.enqueue_function[gen_aux_buffers_gpu](
                handle[].r2c_buf.unsafe_ptr().bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().bitcast[Float32](),
                handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
                handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
                handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
                handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
                handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutExternalOrigin]](),
                handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutExternalOrigin]](),
                handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
                handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
                Int64(handle[].n_spheres),
                handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
                handle[].atrous_normals_buf.unsafe_ptr().bitcast[Float32](),
                handle[].atrous_depth_buf.unsafe_ptr().bitcast[Float32](),
                handle[].atrous_curve_mask_buf.unsafe_ptr().bitcast[Float32](),
                handle[].gbuf_worldpos_buf.unsafe_ptr().bitcast[Float32](),
                handle[].gbuf_material_id_buf.unsafe_ptr().bitcast[Int32](),
                Int64(handle[].fw), Int64(handle[].fh),
                grid_dim=grid_n, block_dim=block_size,
            )
            handle[].ctx.synchronize()
        except e:
            print("GPU gen aux buffers failed: " + String(e))


# One bounce's worth of kernel dispatch (intersect -> medium -> per-material
# shade), shared verbatim between gpu_render_sample and gpu_render_wavefront
# below -- this used to be duplicated ~380 lines in each. Safe to factor out
# because it is pure host-side enqueue_function orchestration, not GPU device
# code, so it doesn't touch the class of PTX-codegen bugs that blocks sharing
# actual kernel bodies elsewhere in this codebase (see bdpt.mojo's MNEE/
# _connect duplication comments for that unrelated, still-real constraint).
def deactivate_paths_past_maxdepth_gpu(
    paths: UnsafePointer[PathState_C, MutExternalOrigin],
    n_dp: Int64, max_depth: Int32,
):
    """A NULL INTERFACE crossing (entering/leaving a medium) does not
    increment `path_ptr[].bounce` -- correctly, since pbrt does not count it
    as a bounce either -- but nothing else previously stopped a path once its
    OWN bounce count reached the scene's real max_depth: the host loop's
    fixed round count (gpu_render_sample/gpu_render_wavefront's own
    `for _ in range(Int(maxDepth))`) was the ENTIRE termination mechanism,
    and every kind of round (real scatter OR free interface pass-through)
    consumed one of those rounds equally. For any volumetric scene, that
    silently granted one fewer REAL bounce than requested for every interface
    the path had to cross -- see rendering.mojo's CPU-side twin of this
    kernel for the measurement. Dispatched as the FIRST kernel of every
    bounce round, so every later kernel's own `active != 0` check already
    skips whatever this one just deactivated."""
    var n = Int(n_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n:
        return
    # Marks, does not kill -- see rendering.mojo's twin of this for why the
    # segment leaving the last allowed vertex must still be traced.
    if paths[tid].active != Int8(0) and paths[tid].bounce >= max_depth:
        paths[tid].at_cap = Int8(1)

def _gpu_bounce_kernels(
    handle: UnsafePointer[GpuSceneHandle, MutExternalOrigin],
    n: Int, grid_dim: Int, px_scale: Float32,
    max_depth: Int32,
    use_vulkan_rt: Bool = False,
    interop_scene: VulkanInteropRtSceneHandle = UnsafePointer[UInt8, MutExternalOrigin].unsafe_dangling(),
    interop_rays_buf: Optional[DeviceBuffer[DType.float32]] = None,
    interop_results_buf: Optional[DeviceBuffer[DType.float32]] = None,
    mesh_material_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    mesh_al_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    n_meshes_vk: Int = 0,
    # ReSTIR DI: only gpu_render_sample passes these (one path per pixel, so
    # tid is a valid pixel index). gpu_render_wavefront leaves them inert.
    use_restir: Bool = False,
    restir_read: UnsafePointer[DIReservoir, MutExternalOrigin] = UnsafePointer[DIReservoir, MutExternalOrigin].unsafe_dangling(),
    restir_write: UnsafePointer[DIReservoir, MutExternalOrigin] = UnsafePointer[DIReservoir, MutExternalOrigin].unsafe_dangling(),
    # Phase 7.3: same "only gpu_render_sample passes these" contract as
    # use_restir/restir_read/restir_write above, for volume-scatter vertices.
    use_vol_restir_reuse: Bool = False,
    restir_vol_read: UnsafePointer[VolReservoir, MutExternalOrigin] = UnsafePointer[VolReservoir, MutExternalOrigin].unsafe_dangling(),
    restir_vol_write: UnsafePointer[VolReservoir, MutExternalOrigin] = UnsafePointer[VolReservoir, MutExternalOrigin].unsafe_dangling(),
    # Per-pixel "already combined this frame" guard, reset by
    # gpu_render_sample before the bounce-round loop starts -- see
    # _sample_medium_core's own comment on vol_used for the bug this fixes.
    restir_vol_used: UnsafePointer[Int8, MutExternalOrigin] = UnsafePointer[Int8, MutExternalOrigin].unsafe_dangling(),
    # Spatial reuse (2026-09-08): same G-buffers DI's own spatial reuse
    # already reads, harmless to pass unconditionally (see
    # _sample_medium_core's matching comment).
    restir_vol_gbuf_depth: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    restir_vol_gbuf_world_pos: UnsafePointer[Float32, MutExternalOrigin] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    restir_vol_frame_w: Int32 = Int32(0),
    restir_vol_frame_h: Int32 = Int32(0),
    # Object-instancing decode for Vulkan RT hits (see
    # vulkaninterop_unpack_results_kernel) -- None for scenes with no
    # instancing, matching every other Optional buffer above.
    instance_base_mesh_buf: Optional[DeviceBuffer[DType.uint8]] = None,
) raises:
    comptime block_size = 256
    handle[].ctx.enqueue_function[deactivate_paths_past_maxdepth_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        Int64(n), max_depth,
        grid_dim=grid_dim, block_dim=block_size,
    )
    if use_vulkan_rt:
        vulkaninterop_rt_traverse_paths_gpu(
            handle[].ctx,
            handle[].path_buf,
            handle[].inter_buf,
            interop_scene,
            interop_rays_buf.value(),
            interop_results_buf.value(),
            mesh_material_idx_buf.value(),
            mesh_al_idx_buf.value(),
            n_meshes_vk,
            n,
            instance_base_mesh_buf,
            handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
            handle[].n_spheres,
        )
    else:
        handle[].ctx.enqueue_function[traverse_paths_gpu](
            handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
            handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
            handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
            handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
            handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutExternalOrigin]](),
            handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutExternalOrigin]](),
            handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
            handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
            Int64(handle[].n_spheres),
            handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
            handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
            handle[].curve_cand_prim_buf.unsafe_ptr().bitcast[Int32](),
            handle[].curve_cand_count_buf.unsafe_ptr().bitcast[Int32](),
            Int64(n),
            grid_dim=grid_dim,
            block_dim=block_size,
        )
    # Deferred-candidate curve resolution only applies to the CUDA-native
    # intersection path (traverse_paths_gpu above, which defers candidates
    # into handle[]'s own curve_cand_* buffers). A Vulkan-RT render never
    # populates those buffers at all -- intersect_batch.comp resolves curve
    # hits itself, inline, as part of the SAME dispatch that traces meshes
    # (see vulkaninterop_rt_traverse_paths_gpu / vulkaninterop_unpack_
    # results_kernel's hitFlag==2 branch) -- so this whole compact+resolve
    # pass is CUDA-native-only.
    if handle[].n_curves > 0 and not use_vulkan_rt:
        handle[].ctx.enqueue_function[reset_curve_counter_gpu](
            handle[].curve_compact_counter_buf.unsafe_ptr().bitcast[Int32](),
            grid_dim=1, block_dim=1,
        )
        handle[].ctx.enqueue_function[compact_curve_paths_gpu](
            handle[].curve_cand_count_buf.unsafe_ptr().bitcast[Int32](),
            Int64(n),
            handle[].curve_compact_path_buf.unsafe_ptr().bitcast[Int32](),
            handle[].curve_compact_counter_buf.unsafe_ptr().bitcast[Int32](),
            grid_dim=grid_dim, block_dim=block_size,
        )
        handle[].ctx.enqueue_function[resolve_curve_candidates_gpu](
            handle[].curve_compact_path_buf.unsafe_ptr().bitcast[Int32](),
            handle[].curve_compact_counter_buf.unsafe_ptr().bitcast[Int32](),
            handle[].curve_cand_prim_buf.unsafe_ptr().bitcast[Int32](),
            handle[].curve_cand_count_buf.unsafe_ptr().bitcast[Int32](),
            handle[].curve_cand_offset_buf.unsafe_ptr().bitcast[Int32](),
            handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
            handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
            handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
            handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
            Int64(n),
            grid_dim=grid_dim, block_dim=block_size,
        )
    handle[].ctx.enqueue_function[sample_medium_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].mediums_buf.unsafe_ptr().bitcast[Medium_C](),
        Int64(handle[].n_mediums),
        handle[].grids_buf.unsafe_ptr().bitcast[Grid_C](),
        handle[].nvdb_grids_buf.unsafe_ptr().bitcast[NvdbGrid_C](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutExternalOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutExternalOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        Int64(handle[].n_area_lights),
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].n_light_sampler),
        Int64(n),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].spectral_res),
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
                handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
                handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        Int64(handle[].n_infinite_lights),
                handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        Int64(handle[].n_distant_lights),
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        Int64(handle[].n_point_lights),
        Int32(1) if use_vol_restir_reuse else Int32(0),
        restir_vol_read, restir_vol_write, restir_vol_used,
        restir_vol_gbuf_depth, restir_vol_gbuf_world_pos,
        restir_vol_frame_w, restir_vol_frame_h,
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_nee_preamble_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutExternalOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutExternalOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        Int64(handle[].n_area_lights),
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        Int64(handle[].n_textures),
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        Int64(handle[].n_distant_lights),
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        Int64(handle[].n_point_lights),
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].n_light_sampler),
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        Int64(handle[].n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].spectral_res),
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    # mix is a pure selector (see shade_mix_gpu's docstring) -- enqueued
    # FIRST among the per-material kernels so its pending_mat/materialIndex
    # redirect is visible to whichever real kernel the sub-material resolves
    # to, later in this SAME launch-ordered sequence.
    handle[].ctx.enqueue_function[shade_mix_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        Int64(n),
        grid_dim=grid_dim, block_dim=block_size,
    )
    # Phase 0.4 (docs/A2_restir_migration_plan.md): reset_shadow_tasks_gpu
    # and the traverse_shadow_rays_gpu resolve call below are real, verified
    # machinery -- built, buffer-sized correctly (shadow_buf now matches
    # path_buf/inter_buf's n_pixels × WAVEFRONT_BATCH sizing), and confirmed
    # via render comparison against inline NEE for a single deferred shadow
    # ray per pixel per bounce. But every per-material kernel below is left
    # at enqueue_shadow=False (shadow_tasks is threaded through as a real
    # pointer and ready, not removed) rather than flipped on, because of a
    # real, structural mismatch discovered while verifying this: ShadowTask_C
    # holds ONE task per pixel, while _shade_diffuse_nee/_nee_area_lights/
    # _shade_conductor_nee/etc. (shading.mojo) each loop over MULTIPLE light
    # types per bounce (area, sphere, distant, point, infinite), calling
    # _shadow_contribute once per candidate. Under enqueue_shadow=True that
    # write is `ctx.shadow_tasks[ctx.path_idx] = ShadowTask_C(...)` --  an
    # OVERWRITE, not an accumulation -- so only the last light type resolved
    # this bounce survives; every earlier one silently vanishes. Measured on
    # real scenes: glass-of-water (mean radiance dropped 88%), curly-hair
    # (91%), material-testball (9%) -- cornell-box (single area light, one
    # NEE candidate per pixel) was the only scene unaffected, which is what
    # hid this until a multi-light comparison render caught it.
    # This single-slot design IS correct for its actual intended consumer:
    # Phase 2 (ReSTIR DI)'s reservoir winner is BY CONSTRUCTION exactly one
    # light candidate per pixel after resampling. Do not flip any of these 6
    # kernels' enqueue_shadow to True for today's multi-candidate NEE loops
    # without first either (a) giving ShadowTask_C N slots (N = max
    # simultaneous light types, currently 5) with accumulate semantics, or
    # (b) restricting deferral to a genuinely single-candidate call site.
    handle[].ctx.enqueue_function[reset_shadow_tasks_gpu](
        handle[].shadow_buf.unsafe_ptr().bitcast[ShadowTask_C](),
        Int64(n),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_diffuse_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutExternalOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutExternalOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        Int64(handle[].n_area_lights),
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        Int64(handle[].n_textures),
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        Int64(handle[].n_distant_lights),
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        Int64(handle[].n_point_lights),
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].n_light_sampler),
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        Int64(handle[].n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].spectral_res),
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        Int32(1) if use_restir else Int32(0),
        restir_read,
        restir_write,
        handle[].atrous_normals_buf.unsafe_ptr().bitcast[Float32](),
        handle[].atrous_depth_buf.unsafe_ptr().bitcast[Float32](),
        handle[].gbuf_material_id_buf.unsafe_ptr().bitcast[Int32](),
        handle[].gbuf_worldpos_buf.unsafe_ptr().bitcast[Float32](),
        Int32(handle[].fw),
        Int32(handle[].fh),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_coated_diffuse_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutExternalOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutExternalOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        Int64(handle[].n_area_lights),
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        Int64(handle[].n_textures),
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        Int64(handle[].n_distant_lights),
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        Int64(handle[].n_point_lights),
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].n_light_sampler),
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        Int64(handle[].n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].shadow_buf.unsafe_ptr().bitcast[ShadowTask_C](),
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].spectral_res),
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_diffuse_transmit_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutExternalOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutExternalOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        Int64(handle[].n_area_lights),
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        Int64(handle[].n_textures),
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        Int64(handle[].n_distant_lights),
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        Int64(handle[].n_point_lights),
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].n_light_sampler),
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        Int64(handle[].n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].shadow_buf.unsafe_ptr().bitcast[ShadowTask_C](),
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].spectral_res),
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_conductor_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutExternalOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutExternalOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        Int64(handle[].n_area_lights),
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        Int64(handle[].n_textures),
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        Int64(handle[].n_distant_lights),
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        Int64(handle[].n_point_lights),
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].n_light_sampler),
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        Int64(handle[].n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].shadow_buf.unsafe_ptr().bitcast[ShadowTask_C](),
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].spectral_res),
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_measured_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutExternalOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutExternalOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        Int64(handle[].n_area_lights),
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        Int64(handle[].n_textures),
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        Int64(handle[].n_distant_lights),
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        Int64(handle[].n_point_lights),
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].n_light_sampler),
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        Int64(handle[].n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].shadow_buf.unsafe_ptr().bitcast[ShadowTask_C](),
        handle[].measured_brdfs_buf.unsafe_ptr().bitcast[MeasuredBRDF_C](),
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].spectral_res),
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_dielectric_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        Int64(n),
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        Int64(handle[].n_textures), px_scale,
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_thin_dielectric_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        Int64(n),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_coated_conductor_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutExternalOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutExternalOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        Int64(handle[].n_area_lights),
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        Int64(handle[].n_textures),
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        Int64(handle[].n_distant_lights),
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        Int64(handle[].n_point_lights),
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].n_light_sampler),
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        Int64(handle[].n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].shadow_buf.unsafe_ptr().bitcast[ShadowTask_C](),
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].spectral_res),
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_interface_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].medium_ifaces_buf.unsafe_ptr().bitcast[MediumInterface_C](),
        Int64(n),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[update_medium_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].medium_ifaces_buf.unsafe_ptr().bitcast[MediumInterface_C](),
        Int64(n),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_hair_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutExternalOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutExternalOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        Int64(handle[].n_area_lights),
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        Int64(handle[].n_textures),
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        Int64(handle[].n_distant_lights),
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        Int64(handle[].n_point_lights),
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].n_light_sampler),
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        Int64(handle[].n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].shadow_buf.unsafe_ptr().bitcast[ShadowTask_C](),
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].spectral_res),
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    # Phase 0.4: resolve whatever shadow rays this bounce's per-material
    # kernels deferred above. Currently a verified no-op in production --
    # every per-material kernel above still enqueues with enqueue_shadow=
    # False (see that block's own comment for why), so reset_shadow_tasks_gpu
    # zeroed every slot and nothing here ever finds task.active != 0. Kept
    # enqueued (not removed) because Phase 2 (ReSTIR DI) is expected to be
    # this resolve kernel's first real, single-candidate-per-pixel consumer.
    # Software BVH only (matches traverse_shadow_rays_gpu's own
    # implementation) even when use_vulkan_rt is set -- this machinery isn't
    # wired to Vulkan RT yet.
    handle[].ctx.enqueue_function[traverse_shadow_rays_gpu](
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutExternalOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutExternalOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
        handle[].shadow_buf.unsafe_ptr().bitcast[ShadowTask_C](),
        Int64(n),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        grid_dim=grid_dim, block_dim=block_size,
    )


# Render one sample pass into the persistent film buffer.
# Ray generation runs on GPU — no CPU-side path buffer or PCIe upload needed.
def gpu_render_sample[Oc: Origin[mut=True]](
    handlePtr: UnsafePointer[GpuSceneHandle, MutExternalOrigin],
    c2w: UnsafePointer[Float32, Oc],
    si: Int32, log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    n: Int64,
    maxDepth: Int32,
    px_scale: Float32 = Float32(0.0),
    use_restir: Bool = False,
    frame_index: Int = 0,
    use_vol_restir_reuse: Bool = False,
):
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            # Update c2w for this frame
            with handle[].c2w_buf.map_to_host() as h:
                var dst = h.unsafe_ptr().bitcast[Float32]()
                memcpy(dest=dst, src=c2w, count=16)
            comptime block_size = 256
            var grid_dim = ceildiv(n_int, block_size)
            # Generate primary rays on GPU
            handle[].ctx.enqueue_function[gen_primary_rays_gpu](
                handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
                handle[].r2c_buf.unsafe_ptr().bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().bitcast[Float32](),
                handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
                Int64(handle[].fw), Int64(handle[].fh),
                si, log2spp, n_base4,
                seed_dim0, seed_dim1,
                rng_seed_lo, rng_seed_hi,
                handle[].filter_sigma, handle[].filter_norm_x, handle[].filter_support_x,
                handle[].filter_norm_y, handle[].filter_support_y,
                handle[].filter_type,
                Int64(n_int),
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            # ReSTIR reservoir ping-pong. Spatial reuse reads neighbouring
            # pixels, which other threads are writing this frame, so the read
            # side must be the PREVIOUS frame's finished buffer. Alternating on
            # frame parity gives that without any copy.
            var restir_rd = UnsafePointer[DIReservoir, MutExternalOrigin].unsafe_dangling()
            var restir_wr = UnsafePointer[DIReservoir, MutExternalOrigin].unsafe_dangling()
            if use_restir:
                var buf_a = handle[].restir_a_buf.unsafe_ptr().bitcast[DIReservoir]()
                var buf_b = handle[].restir_b_buf.unsafe_ptr().bitcast[DIReservoir]()
                if frame_index % 2 == 0:
                    restir_rd = buf_a; restir_wr = buf_b
                else:
                    restir_rd = buf_b; restir_wr = buf_a
            # Phase 7.3: same ping-pong rule as DI's above, own buffer pair.
            # Temporal-only (no spatial neighbour reads), so the same-frame
            # in-flight-write race spatial reuse would need to worry about
            # doesn't apply here, but ping-ponging costs nothing and keeps
            # this consistent with every other reservoir buffer in the file.
            var vol_rd = UnsafePointer[VolReservoir, MutExternalOrigin].unsafe_dangling()
            var vol_wr = UnsafePointer[VolReservoir, MutExternalOrigin].unsafe_dangling()
            var vol_used_ptr = UnsafePointer[Int8, MutExternalOrigin].unsafe_dangling()
            var vol_gbuf_depth_ptr = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling()
            var vol_gbuf_world_pos_ptr = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling()
            var vol_fw = Int32(0)
            var vol_fh = Int32(0)
            if use_vol_restir_reuse:
                var vbuf_a = handle[].restir_vol_a_buf.unsafe_ptr().bitcast[VolReservoir]()
                var vbuf_b = handle[].restir_vol_b_buf.unsafe_ptr().bitcast[VolReservoir]()
                if frame_index % 2 == 0:
                    vol_rd = vbuf_a; vol_wr = vbuf_b
                else:
                    vol_rd = vbuf_b; vol_wr = vbuf_a
                vol_used_ptr = handle[].restir_vol_used_buf.unsafe_ptr().bitcast[Int8]()
                handle[].ctx.enqueue_function[reset_vol_used_gpu](
                    vol_used_ptr, Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
                )
                # Spatial reuse: the SAME G-buffers DI's own spatial reuse
                # already reads (gen_aux_buffers_gpu populates them
                # unconditionally every frame, see gpu_gen_aux_buffers).
                vol_gbuf_depth_ptr = handle[].atrous_depth_buf.unsafe_ptr().bitcast[Float32]()
                vol_gbuf_world_pos_ptr = handle[].gbuf_worldpos_buf.unsafe_ptr().bitcast[Float32]()
                vol_fw = Int32(handle[].fw)
                vol_fh = Int32(handle[].fh)
            # Padding is CONDITIONAL on the scene actually containing a
            # medium: for the vast majority of scenes (no participating
            # media), a null interface never occurs, every path already
            # reaches its true maxDepth at round maxDepth exactly, and there
            # is no per-round host sync here (unlike the CPU loop's cheap
            # `anyActive` early-exit) to make extra rounds free -- each one
            # is a real, unconditional dispatch of every kernel in
            # _gpu_bounce_kernels. Keeping the loop bound exactly `maxDepth`
            # when handle[].n_mediums == 0 makes this fix a complete no-op,
            # performance-wise, for every non-volumetric scene.
            comptime _MEDIUM_INTERFACE_MARGIN = 8
            # An SSS interior is walked one scattering event per round and
            # those steps are not charged to maxDepth (Medium_C.is_sss), so
            # the round count is what actually bounds the walk. Unlike the
            # margin above this is a large budget, and unlike the CPU loop
            # there is no `anyActive` early exit here -- every round is a real
            # dispatch. Gated on the scene actually containing an SSS medium
            # so no other scene pays for it.
            comptime _SSS_WALK_ROUNDS = 256
            var gpu_max_rounds = Int(maxDepth)
            if handle[].n_mediums > 0:
                gpu_max_rounds += _MEDIUM_INTERFACE_MARGIN
            if handle[].has_sss_medium:
                gpu_max_rounds += _SSS_WALK_ROUNDS
            for _ in range(gpu_max_rounds):
                _gpu_bounce_kernels(handle, n_int, grid_dim, px_scale, maxDepth,
                                    use_restir=use_restir,
                                    restir_read=restir_rd, restir_write=restir_wr,
                                    use_vol_restir_reuse=use_vol_restir_reuse,
                                    restir_vol_read=vol_rd, restir_vol_write=vol_wr,
                                    restir_vol_used=vol_used_ptr,
                                    restir_vol_gbuf_depth=vol_gbuf_depth_ptr,
                                    restir_vol_gbuf_world_pos=vol_gbuf_world_pos_ptr,
                                    restir_vol_frame_w=vol_fw, restir_vol_frame_h=vol_fh)
            handle[].ctx.enqueue_function[accumulate_film_gpu](
                handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
                handle[].film_buf.unsafe_ptr().bitcast[Float32](),
                handle[].albedo_film_buf.unsafe_ptr().bitcast[Float32](),
                Int64(n_int),
                        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].spectral_res),
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim,
                block_dim=block_size,
            )
        except e:
            print("GPU render sample failed: " + String(e))


# Wavefront render: generates actual_batch samples worth of primary rays for all pixels,
# runs the full bounce loop over n_pixels × actual_batch paths together, then accumulates.
# Caller loops over spp in steps of WAVEFRONT_BATCH; progress reporting is up to the caller.
def gpu_render_wavefront(
    handlePtr: UnsafePointer[GpuSceneHandle, MutExternalOrigin],
    c2w: UnsafePointer[Float32, MutExternalOrigin],
    si_start: Int32, actual_batch: Int32,
    log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    n: Int64,
    maxDepth: Int32,
    px_scale: Float32 = Float32(0.0),
    # Task #163 stage 3: when use_vulkan_rt, every bounce's primary
    # intersection test is routed through the CUDA/Vulkan interop RT
    # backend (vulkaninterop_rt_traverse_paths_gpu, zero CPU sync) instead
    # of the traverse_paths_gpu CUDA kernel below -- caller must only set
    # this for scenes with no curves/spheres (object instancing IS
    # supported now -- see vulkaninterop_rt_create_scene/
    # vulkaninterop_rt_traverse_paths_gpu's docstrings). The interop_*/
    # mesh_*_buf/n_meshes_vk params are ignored when use_vulkan_rt is False.
    use_vulkan_rt: Bool = False,
    interop_scene: VulkanInteropRtSceneHandle = UnsafePointer[UInt8, MutExternalOrigin].unsafe_dangling(),
    interop_rays_buf: Optional[DeviceBuffer[DType.float32]] = None,
    interop_results_buf: Optional[DeviceBuffer[DType.float32]] = None,
    mesh_material_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    mesh_al_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    n_meshes_vk: Int = 0,
    instance_base_mesh_buf: Optional[DeviceBuffer[DType.uint8]] = None,
):
    # --restir batch rendering does not go through this function at all
    # (docs/A2_restir_migration_plan.md, pipeline.mojo's batch GPU branch):
    # WAVEFRONT_BATCH (8) concurrent samples per pixel per dispatch made
    # reservoir reuse either meaningless (plain RIS, no persistence) or
    # actively wrong (an attempted "N readers, 1 writer" reservoir-sharing
    # scheme measured a real, unexplained divergence and was reverted --
    # see project memory). Batch --restir instead loops gpu_render_sample
    # (1 sample/pixel/dispatch, same proven-correct path
    # --interactive-frames uses) so there is only ever one reader/writer
    # per pixel. This function is unconditionally RIS-and-reuse-free.
    var n_pix = Int(n)
    var batch  = Int(actual_batch)
    var n_total = n_pix * batch
    if n_total == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            with handle[].c2w_buf.map_to_host() as h:
                var dst = h.unsafe_ptr().bitcast[Float32]()
                memcpy(dest=dst, src=c2w, count=16)
            comptime block_size = 256
            var grid_total = ceildiv(n_total, block_size)
            var grid_pix   = ceildiv(n_pix, block_size)
            handle[].ctx.enqueue_function[gen_primary_rays_wavefront_gpu](
                handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
                handle[].r2c_buf.unsafe_ptr().bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().bitcast[Float32](),
                handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
                Int64(handle[].fw), Int64(handle[].fh),
                si_start, log2spp, n_base4,
                seed_dim0, seed_dim1, rng_seed_lo, rng_seed_hi,
                handle[].filter_sigma, handle[].filter_norm_x, handle[].filter_support_x,
                handle[].filter_norm_y, handle[].filter_support_y,
                handle[].filter_type,
                Int64(n_total), Int64(n_pix),
                grid_dim=grid_total,
                block_dim=block_size,
            )
            # Padding is CONDITIONAL on the scene actually containing a
            # medium: for the vast majority of scenes (no participating
            # media), a null interface never occurs, every path already
            # reaches its true maxDepth at round maxDepth exactly, and there
            # is no per-round host sync here (unlike the CPU loop's cheap
            # `anyActive` early-exit) to make extra rounds free -- each one
            # is a real, unconditional dispatch of every kernel in
            # _gpu_bounce_kernels. Keeping the loop bound exactly `maxDepth`
            # when handle[].n_mediums == 0 makes this fix a complete no-op,
            # performance-wise, for every non-volumetric scene.
            comptime _MEDIUM_INTERFACE_MARGIN = 8
            # An SSS interior is walked one scattering event per round and
            # those steps are not charged to maxDepth (Medium_C.is_sss), so
            # the round count is what actually bounds the walk. Unlike the
            # margin above this is a large budget, and unlike the CPU loop
            # there is no `anyActive` early exit here -- every round is a real
            # dispatch. Gated on the scene actually containing an SSS medium
            # so no other scene pays for it.
            comptime _SSS_WALK_ROUNDS = 256
            var gpu_max_rounds = Int(maxDepth)
            if handle[].n_mediums > 0:
                gpu_max_rounds += _MEDIUM_INTERFACE_MARGIN
            if handle[].has_sss_medium:
                gpu_max_rounds += _SSS_WALK_ROUNDS
            for _ in range(gpu_max_rounds):
                _gpu_bounce_kernels(
                    handle, n_total, grid_total, px_scale, maxDepth,
                    use_vulkan_rt, interop_scene, interop_rays_buf, interop_results_buf,
                    mesh_material_idx_buf, mesh_al_idx_buf, n_meshes_vk,
                    instance_base_mesh_buf=instance_base_mesh_buf,
                )
            handle[].ctx.enqueue_function[accumulate_film_wavefront_gpu](
                handle[].path_buf.unsafe_ptr().bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutExternalOrigin](),
                handle[].film_buf.unsafe_ptr().bitcast[Float32](),
                handle[].albedo_film_buf.unsafe_ptr().bitcast[Float32](),
                Int64(n_pix), Int64(batch),
                        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        Int64(handle[].spectral_res),
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_pix,
                block_dim=block_size,
            )
        except e:
            print("GPU wavefront render failed: " + String(e))


def gpu_download_film(
    handlePtr: UnsafePointer[GpuSceneHandle, MutExternalOrigin],
    film: UnsafePointer[Float32, MutExternalOrigin],
    n: Int64,
):
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            handle[].ctx.synchronize()
            var film_bytes = n_int * 12
            with handle[].film_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = film.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=film_bytes)
        except e:
            print("GPU download film failed: " + String(e))


def gpu_download_albedo[Of: Origin[mut=True]](
    handlePtr: UnsafePointer[GpuSceneHandle, MutExternalOrigin],
    film: UnsafePointer[Float32, Of],
    n: Int64,
):
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            handle[].ctx.synchronize()
            var film_bytes = n_int * 12
            with handle[].albedo_film_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = film.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=film_bytes)
        except e:
            print("GPU download albedo failed: " + String(e))


# ── À-trous wavelet denoiser (Dammertz et al. 2010) ─────────────────────────
# Three kernels: normalize, variance estimate, one à-trous pass (5× ping-pong).

def normalize_beauty_albedo_gpu(
    film: UnsafePointer[Float32, MutExternalOrigin],
    albedo_film: UnsafePointer[Float32, MutExternalOrigin],
    beauty_out: UnsafePointer[Float32, MutExternalOrigin],
    albedo_out: UnsafePointer[Float32, MutExternalOrigin],
    n_pixels_dp: Int64,
    inv_weight: Float32,
    iso_scale: Float32,
    max_comp: Float32,
):
    var n_pixels = Int(n_pixels_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n_pixels:
        return
    var lr = film[tid*3+0] * inv_weight * iso_scale
    var lg = film[tid*3+1] * inv_weight * iso_scale
    var lb = film[tid*3+2] * inv_weight * iso_scale
    if lr != lr or lr < Float32(0): lr = Float32(0)
    if lg != lg or lg < Float32(0): lg = Float32(0)
    if lb != lb or lb < Float32(0): lb = Float32(0)
    var scale = Float32(1.0)
    if max_comp > Float32(0.0):
        var mx = lr if lr > lg else lg
        if lb > mx: mx = lb
        if mx > max_comp:
            scale = max_comp / mx
    beauty_out[tid*3+0] = lr * scale
    beauty_out[tid*3+1] = lg * scale
    beauty_out[tid*3+2] = lb * scale
    albedo_out[tid*3+0] = albedo_film[tid*3+0] * inv_weight
    albedo_out[tid*3+1] = albedo_film[tid*3+1] * inv_weight
    albedo_out[tid*3+2] = albedo_film[tid*3+2] * inv_weight


def estimate_variance_gpu(
    beauty: UnsafePointer[Float32, MutExternalOrigin],
    variance_out: UnsafePointer[Float32, MutExternalOrigin],
    fw_dp: Int64, fh_dp: Int64,
):
    var fw = Int(fw_dp)
    var fh = Int(fh_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= fw * fh:
        return
    var px = tid % fw
    var py = tid // fw
    var mean = Float32(0)
    var mean_sq = Float32(0)
    var count = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            var nx = px + dx; var ny = py + dy
            if nx < 0 or nx >= fw or ny < 0 or ny >= fh:
                continue
            var ni = (ny * fw + nx) * 3
            var l = Float32(0.2126)*beauty[ni] + Float32(0.7152)*beauty[ni+1] + Float32(0.0722)*beauty[ni+2]
            mean += l; mean_sq += l * l; count += 1
    var fc = Float32(count)
    mean /= fc; mean_sq /= fc
    var v = mean_sq - mean * mean
    variance_out[tid] = v if v > Float32(0) else Float32(0)


def firefly_clamp_gpu(
    beauty: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    fw_dp: Int64, fh_dp: Int64,
):
    var fw = Int(fw_dp)
    var fh = Int(fh_dp)
    """GPU counterpart of postprocess.mojo's _clamp_fireflies, which the GPU
    à-trous path never had until now -- a live divergence (see
    project_gpu_denoiser_energy_bug.md memory): without it, a single
    extreme-radiance pixel smears across the filter's full effective
    radius (up to 31px at 5 passes) exactly as it did on CPU before that
    fix existed. Same isolated-pixel test as CPU, via the SAME shared
    _firefly_clamp_pixel -- only the neighbor-gathering loop differs (one
    GPU thread per pixel vs a nested CPU loop)."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= fw * fh:
        return
    var px = tid % fw
    var py = tid // fw
    var max_n = Float32(0)
    var max_n_r = Float32(0)
    var max_n_g = Float32(0)
    var max_n_b = Float32(0)
    var has_neighbor = False
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            var nx = px + dx
            var ny = py + dy
            if nx < 0 or nx >= fw or ny < 0 or ny >= fh:
                continue
            has_neighbor = True
            var ni = (ny * fw + nx) * 3
            var lum_n = RGB(beauty[ni], beauty[ni + 1], beauty[ni + 2]).luma()
            if lum_n > max_n:
                max_n = lum_n
            if beauty[ni + 0] > max_n_r: max_n_r = beauty[ni + 0]
            if beauty[ni + 1] > max_n_g: max_n_g = beauty[ni + 1]
            if beauty[ni + 2] > max_n_b: max_n_b = beauty[ni + 2]
    var ci = tid * 3
    var c = _firefly_clamp_pixel(
        beauty[ci + 0], beauty[ci + 1], beauty[ci + 2],
        max_n, max_n_r, max_n_g, max_n_b, has_neighbor)
    output[ci + 0] = c.r
    output[ci + 1] = c.g
    output[ci + 2] = c.b


def atrous_filter_gpu(
    input: UnsafePointer[Float32, MutExternalOrigin],
    albedo: UnsafePointer[Float32, MutExternalOrigin],
    variance: UnsafePointer[Float32, MutExternalOrigin],
    normals: UnsafePointer[Float32, MutExternalOrigin],
    depth: UnsafePointer[Float32, MutExternalOrigin],
    curve_mask: UnsafePointer[Float32, MutExternalOrigin],
    output: UnsafePointer[Float32, MutExternalOrigin],
    fw_i32: Int32, fh_i32: Int32,
    step_i32: Int32,
    sigma_l: Float32,
    sigma_a: Float32,
    sigma_n: Float32,
    sigma_d: Float32,
):
    var fw = Int(fw_i32); var fh = Int(fh_i32); var step = Int(step_i32)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= fw * fh:
        return
    var px = tid % fw; var py = tid // fw

    var c = RGB(input[tid*3], input[tid*3+1], input[tid*3+2])
    if curve_mask[tid] > Float32(0.5):
        # Hair/fur: strand-to-strand self-shadowing has no reliable correlate in
        # albedo/normal/depth (adjacent strands share material and similar
        # orientation/distance), so à-trous can't tell real occlusion from noise
        # and blurs it into a flat blob. Passing raw beauty through here matches
        # pbrt's own un-denoised look for hair instead of erasing strand detail.
        output[tid*3] = c.r; output[tid*3+1] = c.g; output[tid*3+2] = c.b
        return
    var cl = c.luma()
    var var_p = variance[tid]
    var ca = RGB(albedo[tid*3], albedo[tid*3+1], albedo[tid*3+2])
    var cn = Vec3f(normals[tid*3], normals[tid*3+1], normals[tid*3+2])
    # Clamp depth before squaring to avoid Float32 overflow (background sentinel=1e38).
    var cd_clamped = min(depth[tid], Float32(1e18))
    var cd_sq = max(cd_clamped * cd_clamped, Float32(1e-6))

    var acc = RGB(Float32(0))
    var acc_w = Float32(0)

    # Per-tap weight is _atrous_tap_weight (postprocess.mojo), shared
    # VERBATIM with the CPU denoise() pass loop -- see that function's
    # own docstring for why min(var_p,var_q), not var_p alone, matters
    # (an asymmetric weight destroys energy instead of moving it; this
    # was a real, measured bug -- project_gpu_denoiser_energy_bug.md).
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            var nx = px + dx * step; var ny = py + dy * step
            if nx < 0 or nx >= fw or ny < 0 or ny >= fh:
                continue
            var ni = (ny * fw + nx) * 3
            var ni1 = ny * fw + nx
            if curve_mask[ni1] > Float32(0.5):
                continue
            var qc = RGB(input[ni], input[ni+1], input[ni+2])
            var dl = qc.luma() - cl
            var dalb = RGB(albedo[ni], albedo[ni+1], albedo[ni+2]) - ca
            var ndot = normals[ni]*cn.x + normals[ni+1]*cn.y + normals[ni+2]*cn.z
            var dd = min(depth[ni1], Float32(1e18)) - cd_clamped
            var w = _atrous_spatial_weight(dx, dy) * _atrous_tap_weight(
                dl, var_p, variance[ni1], dalb, ndot, dd, cd_sq,
                sigma_l, sigma_a, sigma_n, sigma_d)
            acc += qc * w
            acc_w += w

    if acc_w > Float32(0):
        var o = acc / acc_w
        output[tid*3] = o.r; output[tid*3+1] = o.g; output[tid*3+2] = o.b
    else:
        output[tid*3] = c.r; output[tid*3+1] = c.g; output[tid*3+2] = c.b


def gpu_atrous_denoise[Oo: Origin[mut=True]](
    handlePtr: UnsafePointer[GpuSceneHandle, MutExternalOrigin],
    output: UnsafePointer[Float32, Oo],
    n: Int64,
    frame_count: Int32,
    film_iso: Float32,
    film_max_comp: Float32,
    apply_denoise: Bool = True,
):
    var n_pix = Int(n)
    if n_pix == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            var fw = handle[].fw; var fh = handle[].fh
            comptime block_size = 256
            var grid_n = ceildiv(n_pix, block_size)
            var inv_weight = Float32(1.0) / Float32(max(Int(frame_count), 1))
            var iso_scale = film_iso / Float32(100.0)

            handle[].ctx.enqueue_function[normalize_beauty_albedo_gpu](
                handle[].film_buf.unsafe_ptr().bitcast[Float32](),
                handle[].albedo_film_buf.unsafe_ptr().bitcast[Float32](),
                handle[].atrous_ping_buf.unsafe_ptr().bitcast[Float32](),
                handle[].atrous_albedo_buf.unsafe_ptr().bitcast[Float32](),
                Int64(n_pix), inv_weight, iso_scale, film_max_comp,
                grid_dim=grid_n, block_dim=block_size,
            )
            # --no-denoise: emit the normalized beauty (atrous_ping_buf) without
            # the à-trous blur passes, so the written image is the raw render.
            if not apply_denoise:
                handle[].ctx.synchronize()
                var bytes_b = n_pix * 12
                with handle[].atrous_ping_buf.map_to_host() as h:
                    var src = h.unsafe_ptr()
                    var dst = output.bitcast[UInt8]()
                    memcpy(dest=dst, src=src, count=bytes_b)
                return
            # Firefly pre-clamp -- matches CPU's denoise() (postprocess.mojo),
            # which GPU never had before this. Without it a single extreme
            # pixel smears across the filter's full effective radius (up to
            # 31px at 5 passes). Writes into atrous_pong_buf: pass 0 below
            # then reads from THAT (clamped) buffer, reusing atrous_ping_buf
            # (whose unclamped contents are no longer needed) as scratch --
            # ping/pong roles are therefore swapped relative to before this
            # change, tracked explicitly via clamp_dst_ptr below rather than
            # implicitly through the i%2 alternation.
            handle[].ctx.enqueue_function[firefly_clamp_gpu](
                handle[].atrous_ping_buf.unsafe_ptr().bitcast[Float32](),
                handle[].atrous_pong_buf.unsafe_ptr().bitcast[Float32](),
                Int64(fw), Int64(fh),
                grid_dim=grid_n, block_dim=block_size,
            )
            var clamp_dst_ptr = handle[].atrous_pong_buf.unsafe_ptr().bitcast[Float32]()
            handle[].ctx.enqueue_function[estimate_variance_gpu](
                clamp_dst_ptr,
                handle[].atrous_variance_buf.unsafe_ptr().bitcast[Float32](),
                Int64(fw), Int64(fh),
                grid_dim=grid_n, block_dim=block_size,
            )

            var ping_ptr = handle[].atrous_ping_buf.unsafe_ptr().bitcast[Float32]()
            var pong_ptr = handle[].atrous_pong_buf.unsafe_ptr().bitcast[Float32]()
            var alb_ptr  = handle[].atrous_albedo_buf.unsafe_ptr().bitcast[Float32]()
            var var_ptr  = handle[].atrous_variance_buf.unsafe_ptr().bitcast[Float32]()
            var nrm_ptr  = handle[].atrous_normals_buf.unsafe_ptr().bitcast[Float32]()
            var dep_ptr  = handle[].atrous_depth_buf.unsafe_ptr().bitcast[Float32]()
            var cmask_ptr = handle[].atrous_curve_mask_buf.unsafe_ptr().bitcast[Float32]()
            # Ramp passes with frame_count: 1 pass at fc=1, 5 passes at fc>=5.
            # Prevents the large effective radius (31px at 5 passes) from averaging
            # lit pixels with unlit ones during fast camera movement.
            var n_passes = min(5, max(1, Int(frame_count)))
            for i in range(n_passes):
                var step = 1 << i   # 1, 2, 4, 8, 16
                # Pass 0 reads the CLAMPED buffer (pong), not ping -- see the
                # firefly-clamp comment above for why the starting side is
                # swapped from the pre-firefly-clamp version of this loop.
                var src_ptr = pong_ptr if i % 2 == 0 else ping_ptr
                var dst_ptr = ping_ptr if i % 2 == 0 else pong_ptr
                handle[].ctx.enqueue_function[atrous_filter_gpu](
                    src_ptr, alb_ptr, var_ptr, nrm_ptr, dep_ptr, cmask_ptr, dst_ptr,
                    Int32(fw), Int32(fh), Int32(step),
                    Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05),
                    grid_dim=grid_n, block_dim=block_size,
                )
            # Result is in ping if n_passes is odd, pong if even (start=pong).
            handle[].ctx.synchronize()
            var bytes = n_pix * 12
            var result_buf = handle[].atrous_ping_buf if n_passes % 2 == 1 else handle[].atrous_pong_buf
            with result_buf.map_to_host() as h:
                var src = h.unsafe_ptr()
                var dst = output.bitcast[UInt8]()
                memcpy(dest=dst, src=src, count=bytes)
        except e:
            print("GPU atrous denoise failed: " + String(e))


def gpu_clear_film(
    handlePtr: UnsafePointer[GpuSceneHandle, MutExternalOrigin],
    n: Int64,
):
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            comptime block_size = 256
            var grid_dim = ceildiv(n_int, block_size)
            handle[].ctx.enqueue_function[clear_film_gpu](
                handle[].film_buf.unsafe_ptr().bitcast[Float32](),
                Int64(n_int),
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            handle[].ctx.enqueue_function[clear_film_gpu](
                handle[].albedo_film_buf.unsafe_ptr().bitcast[Float32](),
                Int64(n_int),
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            handle[].ctx.synchronize()
        except e:
            print("GPU clear film failed: " + String(e))


def gpu_clear_restir(
    handlePtr: UnsafePointer[GpuSceneHandle, MutExternalOrigin],
    n: Int64,
):
    """Reset both ReSTIR reservoir buffers. Call wherever gpu_clear_film is
    called: identity reprojection assumes the stored reservoir belongs to this
    pixel's shading point, which a camera move invalidates. Both buffers are
    cleared because which one is "read" this frame depends on frame parity."""
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            comptime block_size = 256
            var grid_dim = ceildiv(n_int, block_size)
            handle[].ctx.enqueue_function[reset_restir_reservoirs_gpu](
                handle[].restir_a_buf.unsafe_ptr().bitcast[DIReservoir](),
                Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
            )
            handle[].ctx.enqueue_function[reset_restir_reservoirs_gpu](
                handle[].restir_b_buf.unsafe_ptr().bitcast[DIReservoir](),
                Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
            )
            handle[].ctx.synchronize()
        except e:
            print("GPU clear restir failed: " + String(e))


def gpu_clear_restir_vol(
    handlePtr: UnsafePointer[GpuSceneHandle, MutExternalOrigin],
    n: Int64,
):
    """Reset both volume ReSTIR reservoir buffers (Phase 7.3). Call wherever
    gpu_clear_restir is called -- same identity-reprojection invalidation
    rule, same both-buffers-cleared reason (frame parity)."""
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            comptime block_size = 256
            var grid_dim = ceildiv(n_int, block_size)
            handle[].ctx.enqueue_function[reset_restir_vol_reservoirs_gpu](
                handle[].restir_vol_a_buf.unsafe_ptr().bitcast[VolReservoir](),
                Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
            )
            handle[].ctx.enqueue_function[reset_restir_vol_reservoirs_gpu](
                handle[].restir_vol_b_buf.unsafe_ptr().bitcast[VolReservoir](),
                Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
            )
            handle[].ctx.synchronize()
        except e:
            print("GPU clear restir vol failed: " + String(e))


def gpu_free_scene(handlePtr: UnsafePointer[GpuSceneHandle, MutExternalOrigin]):
    if Int(handlePtr) == 0:
        return
    handlePtr.destroy_pointee()
    handlePtr.bitcast[GpuSceneHandle]().free()
