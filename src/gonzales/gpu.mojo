from std.sys import has_accelerator, has_nvidia_gpu_accelerator
from std.sys.info import size_of
from std.gpu import block_idx, thread_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.atomic import Atomic
from std.math import ceildiv, sqrt, cos, sin, log, exp
from std.memory import alloc
from .geometry import RGB, Point3f, Vec3f, vec3f, point3f, sphere_outward_normal, Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Material_C, AreaLight_C, Sphere_C, Curve_C, CURVE_N_PIECES, CURVE_DEFER_K, curve_piece_endpoints, _curve_perp_axis, intersect_curve, DistantLight_C, PointLight_C, InfiniteLight_C, PathState_C, GpuTexture_C, ShadowTask_C, LightSampler_C, light_sampler_sample, MatKind, Medium_C, MediumInterface_C, Grid_C, grid_sample_density, Instance_C, MeasuredBRDF_C, dot, cross, INV_PI, INV_FOUR_PI
from std.ffi import external_call
from .bvh import BVH2Node, SceneDescriptor2_C, traverse_bvh2_core, traverse_bvh2_core_defer_curves, any_hit_bvh2_core, test_spheres
from .transform import transform_normal_by_instance
from std.atomic import Atomic
from .rng import PCG32
from .shading import shade_core, shade_nee_core, ShadeContext, LightContext, shade_diffuse, shade_coated_diffuse, shade_diffuse_transmission, shade_mix, shade_conductor, shade_dielectric, shade_thin_dielectric, shade_coated_conductor, shade_hair, shade_interface, shade_measured
from .guide import null_guide
from .sampling import encode_morton2, sobol_get_sample_index, sobol_sample, gaussian_sample_1d, derive_pcg_seeds, gen_primary_ray_state
from .spectrum import SampledWavelengths, SpectralSample, SpectralHandle, null_spectral_handle, rgb_illuminant_to_spectral_sample, spectral_sample_to_rgb
from .vulkaninterop import VulkanInteropRtSceneHandle, vulkaninterop_rt_trace
from std.gpu.host._nvidia_cuda import CUDA

# Number of samples per pixel processed together in one wavefront bounce loop.
# path_buf and inter_buf are pre-allocated at n_pixels × WAVEFRONT_BATCH.
comptime WAVEFRONT_BATCH: Int = 8

def _cstr_eq(a: UnsafePointer[UInt8, MutAnyOrigin], b: UnsafePointer[UInt8, MutAnyOrigin]) -> Bool:
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
    var medium_ifaces_buf: DeviceBuffer[DType.uint8]  # n_medium_ifaces × sizeof(MediumInterface_C)
    var n_medium_ifaces: Int
    var grids_buf: DeviceBuffer[DType.uint8]          # n_grids × sizeof(Grid_C); Grid_C.density points into grid_density_bufs
    var n_grids: Int
    var grid_density_bufs: List[DeviceBuffer[DType.uint8]]  # kept alive; one per grid's density array
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
    var shadow_buf: DeviceBuffer[DType.uint8]       # n_pixels × sizeof(ShadowTask_C) = 48
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

def gpu_upload_scene(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    bvh2NodesCount: Int64,
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    primIdsCount: Int64,
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    blasNodeCounts: UnsafePointer[Int32, MutAnyOrigin],
    blasPrimidCounts: UnsafePointer[Int32, MutAnyOrigin],
    blasCount: Int64,
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    instanceCount: Int64,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    meshCount: Int64,
    meshPointsCounts: UnsafePointer[Int64, MutAnyOrigin],
    meshFaceIndicesCounts: UnsafePointer[Int64, MutAnyOrigin],
    meshVertexIndicesCounts: UnsafePointer[Int64, MutAnyOrigin],
    meshUvNVerts: UnsafePointer[Int64, MutAnyOrigin],
    meshNrmNVerts: UnsafePointer[Int64, MutAnyOrigin],
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    n_tex: Int32,
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    materialCount: Int64,
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int64,
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    curveCount: Int64,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    distantLightCount: Int64,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    pointLightCount: Int64,
    lightSamplerCdf: UnsafePointer[Float32, MutAnyOrigin],
    lightSamplerN: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    infiniteLightCount: Int64,
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    mediumCount: Int64,
    medium_ifaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    medium_iface_count: Int64,
    grids: UnsafePointer[Grid_C, MutAnyOrigin],
    gridCount: Int64,
    measured_brdfs: UnsafePointer[MeasuredBRDF_C, MutAnyOrigin],
    measuredBrdfCount: Int64,
    n_pixels: Int64,
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    r2c: UnsafePointer[Float32, MutAnyOrigin],
    c2w_init: UnsafePointer[Float32, MutAnyOrigin],
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
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
) -> UnsafePointer[GpuSceneHandle, MutAnyOrigin]:
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

            # Upload BVH nodes (copy only the real bytes; the buffer may be a
            # 1-element placeholder when the scene has no geometry).
            var bvh_buf = ctx.enqueue_create_buffer[DType.uint8](bvh_bytes)
            if Int(bvh2NodesCount) > 0:
                with bvh_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = bvh2Nodes.bitcast[UInt8]()
                    for i in range(Int(bvh2NodesCount) * size_of[BVH2Node]()):
                        dst[i] = src[i]

            # Upload prim IDs
            var prim_buf = ctx.enqueue_create_buffer[DType.uint8](prim_bytes)
            if Int(primIdsCount) > 0:
                with prim_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = primIds.bitcast[UInt8]()
                    for i in range(Int(primIdsCount) * size_of[PrimId_C]()):
                        dst[i] = src[i]

            # Upload object-instancing data: one device buffer per BLAS (its
            # nodes + primids), then two small "array of device pointers"
            # buffers so a kernel's blasNodesArr[i]/blasPrimIdsArr[i] resolves
            # to the right BLAS — same two-level indirection as the CPU side
            # (see bvh.mojo's _traverse_instance_leaf).
            var n_blas_int = Int(blasCount)
            var blas_nodes_bufs = List[DeviceBuffer[DType.uint8]]()
            var blas_primids_bufs = List[DeviceBuffer[DType.uint8]]()
            var blas_nodes_ptrs_host = alloc[UnsafePointer[UInt8, MutAnyOrigin]](max(n_blas_int, 1))
            var blas_primids_ptrs_host = alloc[UnsafePointer[UInt8, MutAnyOrigin]](max(n_blas_int, 1))
            for bi in range(n_blas_int):
                var bn_count = Int(blasNodeCounts[bi])
                var bn_bytes = max(bn_count, 1) * size_of[BVH2Node]()
                var bn_buf = ctx.enqueue_create_buffer[DType.uint8](bn_bytes)
                with bn_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = blasNodesArr[bi].bitcast[UInt8]()
                    for j in range(bn_count * size_of[BVH2Node]()):
                        dst[j] = src[j]
                blas_nodes_ptrs_host[bi] = bn_buf.unsafe_ptr()
                blas_nodes_bufs.append(bn_buf^)

                var bp_count = Int(blasPrimidCounts[bi])
                var bp_bytes = max(bp_count, 1) * size_of[PrimId_C]()
                var bp_buf = ctx.enqueue_create_buffer[DType.uint8](bp_bytes)
                with bp_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = blasPrimIdsArr[bi].bitcast[UInt8]()
                    for j in range(bp_count * size_of[PrimId_C]()):
                        dst[j] = src[j]
                blas_primids_ptrs_host[bi] = bp_buf.unsafe_ptr()
                blas_primids_bufs.append(bp_buf^)

            var blas_ptrs_bytes = max(n_blas_int, 1) * size_of[UnsafePointer[UInt8, MutAnyOrigin]]()
            var blas_nodes_ptrs_buf = ctx.enqueue_create_buffer[DType.uint8](blas_ptrs_bytes)
            with blas_nodes_ptrs_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().bitcast[UnsafePointer[UInt8, MutAnyOrigin]]()
                for bi in range(n_blas_int):
                    dst[bi] = blas_nodes_ptrs_host[bi]
            var blas_primids_ptrs_buf = ctx.enqueue_create_buffer[DType.uint8](blas_ptrs_bytes)
            with blas_primids_ptrs_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr().bitcast[UnsafePointer[UInt8, MutAnyOrigin]]()
                for bi in range(n_blas_int):
                    dst[bi] = blas_primids_ptrs_host[bi]
            blas_nodes_ptrs_host.free(); blas_primids_ptrs_host.free()

            var n_instances_int = Int(instanceCount)
            var inst_bytes = max(n_instances_int, 1) * size_of[Instance_C]()
            var instances_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](inst_bytes)
            if n_instances_int > 0:
                with instances_gpu_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = instances.bitcast[UInt8]()
                    for j in range(n_instances_int * size_of[Instance_C]()):
                        dst[j] = src[j]

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
                    for j in range(pts_bytes):
                        dst[j] = src[j]

                # Upload face indices
                var fi_count = Int(meshFaceIndicesCounts[i])
                var fi_bytes = fi_count * 8
                var fi_buf = ctx.enqueue_create_buffer[DType.uint8](fi_bytes)
                with fi_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = host_mesh.faceIndices.bitcast[UInt8]()
                    for j in range(fi_bytes):
                        dst[j] = src[j]

                # Upload vertex indices
                var vi_count = Int(meshVertexIndicesCounts[i])
                var vi_bytes = vi_count * 8
                var vi_buf = ctx.enqueue_create_buffer[DType.uint8](vi_bytes)
                with vi_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = host_mesh.vertexIndices.bitcast[UInt8]()
                    for j in range(vi_bytes):
                        dst[j] = src[j]

                # Upload UVs (2 floats per vertex; zeros if mesh has no UVs)
                var uv_n = Int(meshUvNVerts[i])
                var uv_bytes = max(uv_n * 2 * 4, 4)
                var uv_buf = ctx.enqueue_create_buffer[DType.uint8](uv_bytes)
                with uv_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    if uv_n > 0:
                        var src = host_mesh.uvs.bitcast[UInt8]()
                        for j in range(uv_n * 2 * 4):
                            dst[j] = src[j]
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
                        for j in range(nrm_n * 3 * 4):
                            dst[j] = src[j]
                    else:
                        for j in range(nrm_bytes):
                            dst[j] = UInt8(0)

                mesh_structs_host[i] = TriangleMesh_C(
                    pts_buf.unsafe_ptr().bitcast[Float32](),
                    fi_buf.unsafe_ptr().bitcast[Int64](),
                    vi_buf.unsafe_ptr().bitcast[Int64](),
                    uv_buf.unsafe_ptr().bitcast[Float32](),
                    nrm_buf.unsafe_ptr().bitcast[Float32]() if nrm_n > 0 else UnsafePointer[
                        Float32, MutAnyOrigin
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
                for j in range(mesh_struct_bytes):
                    dst[j] = src[j]

            mesh_structs_host.free()

            # Upload materials array (>= 1 elem to avoid a zero-size buffer)
            var mat_bytes = max(Int(materialCount), 1) * size_of[Material_C]()
            var mat_buf = ctx.enqueue_create_buffer[DType.uint8](mat_bytes)
            if Int(materialCount) > 0:
                with mat_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = materials.bitcast[UInt8]()
                    for j in range(mat_bytes):
                        dst[j] = src[j]

            ctx.synchronize()

            # Upload area lights
            var al_bytes = max(Int(areaLightCount), 1) * size_of[AreaLight_C]()
            var al_buf = ctx.enqueue_create_buffer[DType.uint8](al_bytes)
            if Int(areaLightCount) > 0:
                with al_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = areaLights.bitcast[UInt8]()
                    for j in range(Int(areaLightCount) * size_of[AreaLight_C]()):
                        dst[j] = src[j]

            # Upload spheres (analytical sphere primitives + sphere area lights)
            var sphere_bytes = max(Int(sphereCount), 1) * size_of[Sphere_C]()
            var sphere_buf = ctx.enqueue_create_buffer[DType.uint8](sphere_bytes)
            if Int(sphereCount) > 0:
                with sphere_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = spheres.bitcast[UInt8]()
                    for j in range(Int(sphereCount) * size_of[Sphere_C]()):
                        dst[j] = src[j]

            # Upload curves (native hair/fur primitives — control points only, no tessellation)
            var curve_bytes = max(Int(curveCount), 1) * size_of[Curve_C]()
            var curve_buf = ctx.enqueue_create_buffer[DType.uint8](curve_bytes)
            if Int(curveCount) > 0:
                with curve_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = curves.bitcast[UInt8]()
                    for j in range(Int(curveCount) * size_of[Curve_C]()):
                        dst[j] = src[j]

            # Upload distant (directional) lights
            var dl_bytes = max(Int(distantLightCount), 1) * size_of[DistantLight_C]()
            var dl_buf = ctx.enqueue_create_buffer[DType.uint8](dl_bytes)
            if Int(distantLightCount) > 0:
                with dl_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = distantLights.bitcast[UInt8]()
                    for j in range(Int(distantLightCount) * size_of[DistantLight_C]()):
                        dst[j] = src[j]

            # Upload point lights
            var pl_bytes = max(Int(pointLightCount), 1) * size_of[PointLight_C]()
            var pl_buf = ctx.enqueue_create_buffer[DType.uint8](pl_bytes)
            if Int(pointLightCount) > 0:
                with pl_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = pointLights.bitcast[UInt8]()
                    for j in range(Int(pointLightCount) * size_of[PointLight_C]()):
                        dst[j] = src[j]

            # Upload light sampler CDF (n+1 Float32 entries)
            var ls_entries = Int(lightSamplerN) + 1
            var ls_bytes = max(ls_entries, 2) * size_of[Float32]()
            var ls_buf = ctx.enqueue_create_buffer[DType.uint8](ls_bytes)
            if ls_entries > 0:
                with ls_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr().bitcast[Float32]()
                    for j in range(ls_entries):
                        dst[j] = lightSamplerCdf[j]

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
                il.world_to_light = w2l_buf.unsafe_ptr().bitcast[Float32]()
                il_w2l_bufs.append(w2l_buf^)
                # Upload pixels + CDF when texture is present
                if il.cdf_w > Int32(0) and Int(il.pixels_ptr) > 1:
                    var iw = Int(il.cdf_w); var ih = Int(il.cdf_h)
                    # Pixel data: iw × ih × 3 floats
                    var pix_count = iw * ih * 3
                    var pix_buf = ctx.enqueue_create_buffer[DType.uint8](pix_count * 4)
                    with pix_buf.map_to_host() as h:
                        var dst = h.unsafe_ptr().bitcast[Float32]()
                        for k in range(pix_count):
                            dst[k] = il.pixels_ptr[k]
                    il.pixels_ptr = pix_buf.unsafe_ptr().bitcast[Float32]()
                    il_pixels_bufs.append(pix_buf^)
                    # CDF data: (ih+1) marginal rows + ih×(iw+1) conditional entries
                    var cdf_count = (ih + 1) + ih * (iw + 1)
                    var cdf_buf = ctx.enqueue_create_buffer[DType.uint8](cdf_count * 4)
                    with cdf_buf.map_to_host() as h:
                        var dst = h.unsafe_ptr().bitcast[Float32]()
                        for k in range(cdf_count):
                            dst[k] = il.cdf_ptr[k]
                    il.cdf_ptr = cdf_buf.unsafe_ptr().bitcast[Float32]()
                    il_cdf_bufs.append(cdf_buf^)
                il_patched[ii] = il
            var il_bytes = max(il_count, 1) * size_of[InfiniteLight_C]()
            var il_buf = ctx.enqueue_create_buffer[DType.uint8](il_bytes)
            with il_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = il_patched.bitcast[UInt8]()
                for j in range(il_count * size_of[InfiniteLight_C]()):
                    dst[j] = src[j]
            il_patched.free()
            print("GPU: " + String(il_count) + " infinite light(s) uploaded")

            # Upload participating media (small array; >= 1 elem to avoid zero-size buffer)
            var med_bytes = max(Int(mediumCount), 1) * size_of[Medium_C]()
            var med_buf = ctx.enqueue_create_buffer[DType.uint8](med_bytes)
            if Int(mediumCount) > 0:
                with med_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = mediums.bitcast[UInt8]()
                    for j in range(Int(mediumCount) * size_of[Medium_C]()):
                        dst[j] = src[j]

            # Upload medium interfaces
            var miface_bytes = max(Int(medium_iface_count), 1) * size_of[MediumInterface_C]()
            var miface_buf = ctx.enqueue_create_buffer[DType.uint8](miface_bytes)
            if Int(medium_iface_count) > 0:
                with miface_buf.map_to_host() as host_buf:
                    var dst = host_buf.unsafe_ptr()
                    var src = medium_ifaces.bitcast[UInt8]()
                    for j in range(Int(medium_iface_count) * size_of[MediumInterface_C]()):
                        dst[j] = src[j]

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
                    for j in range(n_voxels * 4):
                        dst[j] = src[j]
                grid_structs_host[gi] = Grid_C(
                    density_buf.unsafe_ptr().bitcast[Float32](),
                    host_grid.nx, host_grid.ny, host_grid.nz,
                    host_grid.p0, host_grid.p1,
                    host_grid.world_to_medium, host_grid.max_density)
                grid_density_bufs.append(density_buf^)
            var grid_struct_bytes = max(n_grids_int, 1) * size_of[Grid_C]()
            var grids_buf = ctx.enqueue_create_buffer[DType.uint8](grid_struct_bytes)
            with grids_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = grid_structs_host.bitcast[UInt8]()
                for j in range(grid_struct_bytes):
                    dst[j] = src[j]
            grid_structs_host.free()
            if n_grids_int > 0:
                print("GPU: " + String(n_grids_int) + " heterogeneous density grid(s) uploaded")

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
                var theta_i_dptr = theta_i_buf.unsafe_ptr().bitcast[Float32]()
                measured_field_bufs.append(theta_i_buf^)

                var phi_i_buf = ctx.enqueue_create_buffer[DType.uint8](max(n_phi_i, 1) * 4)
                with phi_i_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(n_phi_i):
                        dst[k] = hm.phi_i[k]
                var phi_i_dptr = phi_i_buf.unsafe_ptr().bitcast[Float32]()
                measured_field_bufs.append(phi_i_buf^)

                var wavelengths_buf = ctx.enqueue_create_buffer[DType.uint8](max(n_wavelengths, 1) * 4)
                with wavelengths_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(n_wavelengths):
                        dst[k] = hm.wavelengths[k]
                var wavelengths_dptr = wavelengths_buf.unsafe_ptr().bitcast[Float32]()
                measured_field_bufs.append(wavelengths_buf^)

                var ndf_n = Int(hm.ndf_xs) * Int(hm.ndf_ys)
                var ndf_buf = ctx.enqueue_create_buffer[DType.uint8](max(ndf_n, 1) * 4)
                with ndf_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(ndf_n):
                        dst[k] = hm.ndf_data[k]
                var ndf_dptr = ndf_buf.unsafe_ptr().bitcast[Float32]()
                measured_field_bufs.append(ndf_buf^)

                var sigma_n = Int(hm.sigma_xs) * Int(hm.sigma_ys)
                var sigma_buf = ctx.enqueue_create_buffer[DType.uint8](max(sigma_n, 1) * 4)
                with sigma_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(sigma_n):
                        dst[k] = hm.sigma_data[k]
                var sigma_dptr = sigma_buf.unsafe_ptr().bitcast[Float32]()
                measured_field_bufs.append(sigma_buf^)

                var vndf_n = slices2 * Int(hm.vndf_xs) * Int(hm.vndf_ys)
                var vndf_buf = ctx.enqueue_create_buffer[DType.uint8](max(vndf_n, 1) * 4)
                with vndf_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(vndf_n):
                        dst[k] = hm.vndf_data[k]
                var vndf_dptr = vndf_buf.unsafe_ptr().bitcast[Float32]()
                measured_field_bufs.append(vndf_buf^)

                var vndf_marg_n = slices2 * Int(hm.vndf_ys)
                var vndf_marg_buf = ctx.enqueue_create_buffer[DType.uint8](max(vndf_marg_n, 1) * 4)
                with vndf_marg_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(vndf_marg_n):
                        dst[k] = hm.vndf_marg[k]
                var vndf_marg_dptr = vndf_marg_buf.unsafe_ptr().bitcast[Float32]()
                measured_field_bufs.append(vndf_marg_buf^)

                var vndf_cond_buf = ctx.enqueue_create_buffer[DType.uint8](max(vndf_n, 1) * 4)
                with vndf_cond_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(vndf_n):
                        dst[k] = hm.vndf_cond[k]
                var vndf_cond_dptr = vndf_cond_buf.unsafe_ptr().bitcast[Float32]()
                measured_field_bufs.append(vndf_cond_buf^)

                var lum_n = slices2 * Int(hm.lum_xs) * Int(hm.lum_ys)
                var lum_buf = ctx.enqueue_create_buffer[DType.uint8](max(lum_n, 1) * 4)
                with lum_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(lum_n):
                        dst[k] = hm.lum_data[k]
                var lum_dptr = lum_buf.unsafe_ptr().bitcast[Float32]()
                measured_field_bufs.append(lum_buf^)

                var lum_marg_n = slices2 * Int(hm.lum_ys)
                var lum_marg_buf = ctx.enqueue_create_buffer[DType.uint8](max(lum_marg_n, 1) * 4)
                with lum_marg_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(lum_marg_n):
                        dst[k] = hm.lum_marg[k]
                var lum_marg_dptr = lum_marg_buf.unsafe_ptr().bitcast[Float32]()
                measured_field_bufs.append(lum_marg_buf^)

                var lum_cond_buf = ctx.enqueue_create_buffer[DType.uint8](max(lum_n, 1) * 4)
                with lum_cond_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(lum_n):
                        dst[k] = hm.lum_cond[k]
                var lum_cond_dptr = lum_cond_buf.unsafe_ptr().bitcast[Float32]()
                measured_field_bufs.append(lum_cond_buf^)

                var spectra_n = slices3 * Int(hm.spectra_xs) * Int(hm.spectra_ys)
                var spectra_buf = ctx.enqueue_create_buffer[DType.uint8](max(spectra_n, 1) * 4)
                with spectra_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for k in range(spectra_n):
                        dst[k] = hm.spectra_data[k]
                var spectra_dptr = spectra_buf.unsafe_ptr().bitcast[Float32]()
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
                for j in range(measured_struct_bytes):
                    dst[j] = src[j]
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
            var r_shadow_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[ShadowTask_C]())
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
                var data_out = alloc[UnsafePointer[Float32, MutAnyOrigin]](1)
                var w_out = alloc[Int32](1)
                var h_out = alloc[Int32](1)
                w_out[0] = Int32(0); h_out[0] = Int32(0)
                var raw_flag = Int32(1) if tex_is_raw[ti] else Int32(0)
                var ok = external_call["load_texture_rgb", Int32,
                    UnsafePointer[UInt8, MutAnyOrigin],
                    UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin],
                    UnsafePointer[Int32, MutAnyOrigin],
                    UnsafePointer[Int32, MutAnyOrigin],
                    Int32](filename, data_out, w_out, h_out, raw_flag)
                if ok != 0 and Int(w_out[0]) > 0:
                    var tw = Int(w_out[0]); var th = Int(h_out[0])
                    # Mip pyramid: levels until 1x1, box-downsampled (in linear
                    # space, which is what load_texture_rgb returns). Anti-aliases
                    # minified textures; trilinear-sampled on the GPU via the LOD.
                    var nlev = 1; var ww = tw; var hh = th
                    while ww > 1 or hh > 1:
                        ww = max(1, ww // 2); hh = max(1, hh // 2); nlev += 1
                    var total = 0; ww = tw; hh = th
                    for _k in range(nlev):
                        total += ww * hh * 3
                        ww = max(1, ww // 2); hh = max(1, hh // 2)
                    var pyr = alloc[Float32](total)
                    var src0 = data_out[0]
                    for j in range(tw * th * 3):
                        pyr[j] = src0[j]
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
                        for j in range(total):
                            dst[j] = pyr[j]
                    pyr.free()
                    gpu_textures_host[ti] = GpuTexture_C(tex_buf.unsafe_ptr().bitcast[Float32](), Int32(tw), Int32(th), Int32(nlev))
                    _ = external_call["free_texture_rgb", Int32, UnsafePointer[Float32, MutAnyOrigin]](data_out[0])
                    tex_data_bufs.append(tex_buf^)
                else:
                    gpu_textures_host[ti] = GpuTexture_C(UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(), Int32(0), Int32(0), Int32(0))
                data_out.free(); w_out.free(); h_out.free()
            var tex_struct_bytes = max(n_textures_int, 1) * size_of[GpuTexture_C]()
            var textures_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](tex_struct_bytes)
            with textures_gpu_buf.map_to_host() as h:
                var dst = h.unsafe_ptr()
                var src = gpu_textures_host.bitcast[UInt8]()
                for j in range(tex_struct_bytes):
                    dst[j] = src[j]
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
                for i in range(N_SOBOL_GPU_WORDS):
                    dst[i] = sobol_matrices[i]

            # Upload raster_to_camera (16 floats = 64 bytes)
            var r2c_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](64)
            with r2c_gpu_buf.map_to_host() as h:
                var dst = h.unsafe_ptr().bitcast[Float32]()
                for i in range(16):
                    dst[i] = r2c[i]

            # Upload camera_to_world (16 floats = 64 bytes)
            var c2w_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](64)
            with c2w_gpu_buf.map_to_host() as h:
                var dst = h.unsafe_ptr().bitcast[Float32]()
                for i in range(16):
                    dst[i] = c2w_init[i]

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
                    for i in range(spec_coeffs_count):
                        dst[i] = spectral_coeffs[i]

            var spec_cie_count = CIE_N if spectral_res > 0 else 1
            var spec_cie_x_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](spec_cie_count * 4)
            var spec_cie_y_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](spec_cie_count * 4)
            var spec_cie_z_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](spec_cie_count * 4)
            var spec_d65_gpu_buf   = ctx.enqueue_create_buffer[DType.uint8](spec_cie_count * 4)
            if spectral_res > 0:
                with spec_cie_x_gpu_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for i in range(CIE_N):
                        dst[i] = spectral_cie_x[i]
                with spec_cie_y_gpu_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for i in range(CIE_N):
                        dst[i] = spectral_cie_y[i]
                with spec_cie_z_gpu_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for i in range(CIE_N):
                        dst[i] = spectral_cie_z[i]
                with spec_d65_gpu_buf.map_to_host() as h:
                    var dst = h.unsafe_ptr().bitcast[Float32]()
                    for i in range(CIE_N):
                        dst[i] = spectral_d65[i]

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
                medium_ifaces_buf=miface_buf^,
                n_medium_ifaces=Int(medium_iface_count),
                grids_buf=grids_buf^,
                n_grids=n_grids_int,
                grid_density_bufs=grid_density_bufs^,
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
            return UnsafePointer[GpuSceneHandle, MutAnyOrigin].unsafe_dangling()
    else:
        return UnsafePointer[GpuSceneHandle, MutAnyOrigin].unsafe_dangling()


# GPU kernel function — one thread per ray
def traverse_bvh2_gpu(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    rays: UnsafePointer[Ray_C, MutAnyOrigin],
    tMaxValues: UnsafePointer[Float32, MutAnyOrigin],
    results: UnsafePointer[Intersection_C, MutAnyOrigin],
    count: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var ray = rays[tid]
    var tMax = tMaxValues[tid]
    var result_ptr = results + tid
    traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, ray, tMax, result_ptr)

def gpu_traverse_batch(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    rays: UnsafePointer[Ray_C, MutAnyOrigin],
    tMaxValues: UnsafePointer[Float32, MutAnyOrigin],
    count: Int64,
    results: UnsafePointer[Intersection_C, MutAnyOrigin],
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
                for i in range(ray_bytes):
                    dst[i] = src[i]

            # Upload tMax values
            var tmax_bytes = n * 4
            var tmax_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](tmax_bytes)
            with tmax_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = tMaxValues.bitcast[UInt8]()
                for i in range(tmax_bytes):
                    dst[i] = src[i]

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
                ray_buf.unsafe_ptr().bitcast[Ray_C](),
                tmax_buf.unsafe_ptr().bitcast[Float32](),
                result_buf.unsafe_ptr().bitcast[Intersection_C](),
                n,
                grid_dim=grid_dim,
                block_dim=block_size,
            )

            handle[].ctx.synchronize()

            # Copy results back to host
            with result_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = results.bitcast[UInt8]()
                for i in range(result_bytes):
                    dst[i] = src[i]
        except e:
            print("GPU: Batch traversal failed: " + String(e))


def shade_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    count: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    shade_core(paths, intersections, meshes, materials, tid)



def shade_nee_preamble_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    n_distant_lights: Int,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    n_point_lights: Int,
    lightSamplerCdf: UnsafePointer[Float32, MutAnyOrigin],
    n_light_sampler: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    n_infinite_lights: Int,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    count: Int,
    px_scale: Float32,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
):
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
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        shadow_tasks=UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(),
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
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
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    n_distant_lights: Int,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    n_point_lights: Int,
    lightSamplerCdf: UnsafePointer[Float32, MutAnyOrigin],
    n_light_sampler: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    n_infinite_lights: Int,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    count: Int,
    px_scale: Float32,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
):
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
    var ctx = ShadeContext(
        path_idx=0, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        shadow_tasks=UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(),
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_diffuse[True, False](path_ptr, inter, ctx, mat)


def shade_coated_diffuse_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    n_distant_lights: Int,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    n_point_lights: Int,
    lightSamplerCdf: UnsafePointer[Float32, MutAnyOrigin],
    n_light_sampler: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    n_infinite_lights: Int,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    count: Int,
    px_scale: Float32,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
):
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
        path_idx=0, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        shadow_tasks=UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(),
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_coated_diffuse[True, False](path_ptr, inter, ctx, mat)


def shade_diffuse_transmit_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    n_distant_lights: Int,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    n_point_lights: Int,
    lightSamplerCdf: UnsafePointer[Float32, MutAnyOrigin],
    n_light_sampler: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    n_infinite_lights: Int,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    count: Int,
    px_scale: Float32,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
):
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
        path_idx=0, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        shadow_tasks=UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(),
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
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
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    count: Int,
):
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
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    n_distant_lights: Int,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    n_point_lights: Int,
    lightSamplerCdf: UnsafePointer[Float32, MutAnyOrigin],
    n_light_sampler: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    n_infinite_lights: Int,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    count: Int,
    px_scale: Float32,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
):
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
        path_idx=0, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        shadow_tasks=UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(),
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_conductor[True, False](path_ptr, inter, ctx, mat)


def shade_measured_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    n_distant_lights: Int,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    n_point_lights: Int,
    lightSamplerCdf: UnsafePointer[Float32, MutAnyOrigin],
    n_light_sampler: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    n_infinite_lights: Int,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    count: Int,
    px_scale: Float32,
    measured_brdfs: UnsafePointer[MeasuredBRDF_C, MutAnyOrigin],
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
):
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
        path_idx=0, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        shadow_tasks=UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(),
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=measured_brdfs,
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_measured[True, False](path_ptr, inter, ctx, mat)


def shade_dielectric_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    count: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].pending_mat != MatKind.dielectric:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[tid]
    var mat = materials[Int(inter.primId.materialIndex)]
    shade_dielectric(path_ptr, inter, meshes, mat)


def shade_thin_dielectric_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    count: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].pending_mat != MatKind.thin_dielectric:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[tid]
    var mat = materials[Int(inter.primId.materialIndex)]
    shade_thin_dielectric(path_ptr, inter, meshes, mat)


def shade_coated_conductor_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    n_distant_lights: Int,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    n_point_lights: Int,
    lightSamplerCdf: UnsafePointer[Float32, MutAnyOrigin],
    n_light_sampler: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    n_infinite_lights: Int,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    count: Int,
    px_scale: Float32,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
):
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
        path_idx=0, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        shadow_tasks=UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(),
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_coated_conductor[True, False](path_ptr, inter, ctx, mat)


def shade_interface_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    medium_ifaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    count: Int,
):
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
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    medium_ifaces: UnsafePointer[MediumInterface_C, MutAnyOrigin],
    count: Int,
):
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
    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var geom_n: SIMD[DType.float32, 3]
    if inter.primId.type == 4:
        # Sphere: outward normal = hit point - center. Medium-bounding
        # volumes (e.g. smoke-plume's "MediumInterface .. Shape sphere")
        # are commonly a big invisible sphere, so this case matters even
        # though spheres otherwise rarely carry materials with real shading.
        var sph = spheres[Int(inter.primId.id1)]
        var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
        var hit_pt = ray_org + inter.tHit * ray_dir
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
        var p0 = SIMD[DType.float32, 3](m.points[v0*4], m.points[v0*4+1], m.points[v0*4+2])
        var p1 = SIMD[DType.float32, 3](m.points[v1*4], m.points[v1*4+1], m.points[v1*4+2])
        var p2 = SIMD[DType.float32, 3](m.points[v2*4], m.points[v2*4+1], m.points[v2*4+2])
        geom_n = cross(p1 - p0, p2 - p0)
    if dot(ray_dir, geom_n) > Float32(0.0):
        path_ptr[].current_medium_idx = iface.outside_medium_idx
    else:
        path_ptr[].current_medium_idx = iface.inside_medium_idx


comptime MEDIUM_TRACK_MAX_ITERS: Int = 10000  # delta/ratio-tracking loop safety bound

def _spectral_beer_lambert_rgb(
    sigma_t: RGB, dist: Float32, wl: SampledWavelengths,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin],
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin],
) -> RGB:
    """Beer-Lambert transmittance exp(-sigma_t * dist), colored via a real
    per-wavelength spectral evaluation of sigma_t at this path's own hero
    wavelengths (staged spectral rendering rollout, Stage 5b -- see
    project_spectral_rendering memory) instead of the old independent-
    per-RGB-channel exp(). sigma_t is an extinction COEFFICIENT, not a
    light color, and has no dedicated spectral-upsampling target in this
    codebase -- this reuses the Jakob-Hanika illuminant-style unbounded-RGB
    conversion (rgb_illuminant_to_spectral_sample) as a documented, scoped
    approximation (same "approximate via the closest existing conversion"
    precedent as project_material_type_warnings' subsurface/measured
    material approximations), not a physically rigorous spectral-extinction
    fit."""
    var sigma_t_spec = rgb_illuminant_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, sigma_t.r, sigma_t.g, sigma_t.b, wl)
    var v0 = exp(-sigma_t_spec.v0 * dist)
    var v1 = exp(-sigma_t_spec.v1 * dist)
    var v2 = exp(-sigma_t_spec.v2 * dist)
    var v3 = exp(-sigma_t_spec.v3 * dist)
    var trans_spec = SpectralSample(v0, v1, v2, v3)
    var (tr, tg, tb) = spectral_sample_to_rgb(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, trans_spec, wl)
    return RGB(tr, tg, tb)

def _sample_medium_core(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    i: Int,
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    n_mediums: Int,
    grids: UnsafePointer[Grid_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    n_area_lights: Int,
    lightSamplerCdf: UnsafePointer[Float32, MutAnyOrigin],
    n_light_sampler: Int,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
):
    """Apply medium transmittance along the ray segment and possibly scatter
    or absorb inside the medium. On scatter, performs direct area-light NEE
    with isotropic phase function. Shared verbatim between the GPU kernel
    (sample_medium_gpu, one call per thread) and the CPU driver
    (render_all_tiles's per-sample loop) — staged spectral rendering
    rollout, Stage 5a (unify before spectralize), see
    project_spectral_rendering memory. Was previously hand-duplicated with
    real, if minor, divergences: the CPU copy never set `.bounce`/
    `.lastBsdfPdf` on a volume-scatter event (this function, matching what
    was already GPU's behavior, now does on both) and drew its NEE/
    scatter-direction random numbers in the opposite order from GPU (this
    function uses GPU's order, matching the CPU code's own long-standing
    comment that GPU was the intended canonical reference "mirrored" on the
    CPU side) — both fixed as a natural side effect of unifying into one
    body, not preserved as intentional per-backend differences.

    Homogeneous media (grid_idx < 0): closed-form analytic transmittance,
    single free-flight sample (unchanged from before heterogeneous support).

    Heterogeneous media (grid_idx >= 0, "uniformgrid"): real delta tracking
    (Woodcock tracking) against the grid's majorant — repeatedly sample
    candidate collision distances at the majorant rate, accept/reject against
    the true local density at each candidate. Null collisions leave
    throughput unchanged (that's the point of the null-collision trick —
    transmittance emerges from the accept/reject statistics, not an explicit
    exp() multiply). NEE shadow rays through the grid use the matching
    ratio-tracking transmittance estimator (grid_sample_density naturally
    returns 0 outside the grid's bounds, so this correctly stops attenuating
    once the shadow ray exits the grid's AABB — no separate bookkeeping
    needed for "did the shadow ray leave the medium").

    Both paths use the RED channel exclusively for majorant/accept-reject
    decisions (matching the pre-existing homogeneous code's own simplification):
    exact for gray/achromatic media (uniformgrid's only supported density
    representation today has no per-channel color), would need spectral MIS
    (hero-wavelength weighting) to be unbiased for a colored heterogeneous
    medium — not needed for uniformgrid, would matter for a future colored
    nanovdb medium.

    Staged spectral rendering rollout, Stage 5b (spectralize now that 5a
    unified this function — see project_spectral_rendering memory): the
    HOMOGENEOUS branch's Beer-Lambert transmittance (both the main
    throughput multiply and the NEE shadow-ray transmittance) now goes
    through _spectral_beer_lambert_rgb — real per-wavelength color at this
    path's own hero wavelengths, not independent per-RGB-channel exp().
    Deliberately scoped to just the COLOR of the transmittance result, not
    the free-flight DISTANCE-sampling decision above (still red-channel-only,
    unchanged) — matches this whole rollout's established "evaluate
    spectrally, convert back to RGB immediately, don't touch the sampling
    PDF" pattern. The heterogeneous (uniformgrid) branch is NOT spectralized:
    per this docstring's own note, uniformgrid density has no per-channel
    color to begin with, so there is nothing to spectralize there yet — this
    would matter for a future colored volumetric medium (e.g. nanovdb), not
    today's grayscale uniformgrid smoke/fire density.
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
    var ray_org = SIMD[DType.float32, 3](path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var ray_dir = SIMD[DType.float32, 3](path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)

    var t_free: Float32
    var albedo_r: Float32
    if med.grid_idx >= Int32(0):
        # ── Heterogeneous: delta tracking ────────────────────────────────
        var grid = grids[Int(med.grid_idx)]
        var sigma_maj = grid.max_density * sigma_t.r
        if sigma_maj <= Float32(0.0):
            path_ptr[].pcgState = pcg.state
            return
        var t = Float32(0.0)
        var collided = False
        var iters = 0
        while iters < MEDIUM_TRACK_MAX_ITERS:
            iters += 1
            var u = pcg.next_float()
            t += -log(max(u, Float32(1e-7))) / sigma_maj
            if t >= t_surf:
                break
            var p_world = ray_org + t * ray_dir
            var density = grid_sample_density(grid, p_world)
            var sigma_t_real = density * sigma_t.r
            var u2 = pcg.next_float()
            if u2 < sigma_t_real / sigma_maj:
                collided = True
                break
        if not collided:
            path_ptr[].pcgState = pcg.state
            return
        t_free = t
        albedo_r = med.sigma_s.r / max(sigma_t.r, Float32(1e-7))  # density cancels — see docstring
    else:
        # ── Homogeneous: closed-form analytic transmittance ──────────────
        var sigma_maj = sigma_t.r
        if sigma_maj <= Float32(0.0):
            path_ptr[].pcgState = pcg.state
            return
        var u_free = pcg.next_float()
        t_free = -log(max(u_free, Float32(1e-7))) / sigma_maj
        var t_seg = min(t_free, t_surf)
        if spectral_res > 0:
            path_ptr[].throughput *= _spectral_beer_lambert_rgb(sigma_t, t_seg, path_ptr[].wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
        else:
            path_ptr[].throughput *= RGB(exp(-sigma_t.r * t_seg), exp(-sigma_t.g * t_seg), exp(-sigma_t.b * t_seg))
        if t_free >= t_surf:
            path_ptr[].pcgState = pcg.state
            return
        albedo_r = med.sigma_s.r / sigma_maj

    # Both branches above already `return` early for the "no real collision"
    # case (homogeneous: t_free >= t_surf; heterogeneous: not collided) — so
    # reaching here always means a real scatter/absorb event at t_free.
    var p_scatter = albedo_r
    var u_mode = pcg.next_float()
    if u_mode < p_scatter:
        # Volume scatter: compute scatter point
        var scatter_pt = path_ptr[].ray.origin + path_ptr[].ray.direction * t_free
        # ── Volume scatter NEE — area light direct lighting ──────────────
        if n_area_lights > 0 and n_light_sampler > 0:
            var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
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
            var lp0 = SIMD[DType.float32, 3](lmesh.points[lv0*4], lmesh.points[lv0*4+1], lmesh.points[lv0*4+2])
            var lp1 = SIMD[DType.float32, 3](lmesh.points[lv1*4], lmesh.points[lv1*4+1], lmesh.points[lv1*4+2])
            var lp2 = SIMD[DType.float32, 3](lmesh.points[lv2*4], lmesh.points[lv2*4+1], lmesh.points[lv2*4+2])
            var sqrt_r1 = sqrt(r1)
            var light_point = lp0 * (Float32(1) - sqrt_r1) + lp1 * (sqrt_r1 * (Float32(1) - r2)) + lp2 * (sqrt_r1 * r2)
            var lcross = cross(lp1 - lp0, lp2 - lp0)
            var lcross_len = sqrt(max(Float32(1e-14), dot(lcross, lcross)))
            var light_normal = lcross * (Float32(1) / lcross_len)
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
                    if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, curves, shad_ray, shad_tmax, blasNodesArr, blasPrimIdsArr, instances):
                        var T: RGB
                        if med.grid_idx >= Int32(0):
                            # Ratio-tracking transmittance through the grid (see
                            # sample_medium_gpu's docstring). grid_sample_density
                            # returns 0 past the grid's AABB, so this naturally
                            # stops attenuating once the shadow ray exits it.
                            var grid = grids[Int(med.grid_idx)]
                            var sigma_maj_s = grid.max_density * sigma_t.r
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
                                    var ps = scatter_pt_s + shadow_dir * ts
                                    var density_s = grid_sample_density(grid, ps)
                                    Tval *= Float32(1.0) - (density_s * sigma_t.r) / sigma_maj_s
                                    if Tval < Float32(1e-4):
                                        Tval = Float32(0.0)
                                        break
                            T = RGB(Tval, Tval, Tval)
                        elif spectral_res > 0:
                            T = _spectral_beer_lambert_rgb(sigma_t, dist, path_ptr[].wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
                        else:
                            T = RGB(exp(-sigma_t.r * dist), exp(-sigma_t.g * dist), exp(-sigma_t.b * dist))
                        var geom = al.total_area * cos_l / (dist_sq * light_sel_pdf)
                        path_ptr[].estimate += path_ptr[].throughput * al.emission * T * (geom * INV_FOUR_PI)
        # Sample isotropic scatter direction
        var u1 = pcg.next_float()
        var u2 = pcg.next_float()
        path_ptr[].pcgState = pcg.state
        var cos_theta = Float32(2.0) * u1 - Float32(1.0)
        var sin_theta = sqrt(max(Float32(0.0), Float32(1.0) - cos_theta * cos_theta))
        var phi = Float32(6.28318530718) * u2
        path_ptr[].ray = Ray_C(
            scatter_pt,
            Vec3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta))
        path_ptr[].specularBounce = Int8(0)
        path_ptr[].lastBsdfPdf = INV_FOUR_PI
        path_ptr[].volume_scattered = Int8(1)
        path_ptr[].bounce += 1
        intersections[i].hit = Int8(0)  # no surface hit this bounce
    else:
        # Absorbed
        path_ptr[].pcgState = pcg.state
        path_ptr[].active = Int8(0)

def sample_medium_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    mediums: UnsafePointer[Medium_C, MutAnyOrigin],
    n_mediums: Int,
    grids: UnsafePointer[Grid_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    n_area_lights: Int,
    lightSamplerCdf: UnsafePointer[Float32, MutAnyOrigin],
    n_light_sampler: Int,
    count: Int,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
):
    """GPU kernel wrapper: bounds-check, then call the SAME
    _sample_medium_core the CPU driver (render_all_tiles) calls."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    _sample_medium_core(
        paths, intersections, tid, mediums, n_mediums, grids,
        bvh2Nodes, primIds, meshes, curves,
        blasNodesArr, blasPrimIdsArr, instances,
        areaLights, n_area_lights, lightSamplerCdf, n_light_sampler,
        spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
    )


def shade_hair_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    distantLights: UnsafePointer[DistantLight_C, MutAnyOrigin],
    n_distant_lights: Int,
    pointLights: UnsafePointer[PointLight_C, MutAnyOrigin],
    n_point_lights: Int,
    lightSamplerCdf: UnsafePointer[Float32, MutAnyOrigin],
    n_light_sampler: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    n_infinite_lights: Int,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    count: Int,
    px_scale: Float32,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
):
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
        path_idx=0, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        shadow_tasks=UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(),
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_hair[True, False](path_ptr, inter, ctx, mat)


def shade_enqueue_shadow_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    n_infinite_lights: Int,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    shadow_tasks: UnsafePointer[ShadowTask_C, MutAnyOrigin],
    count: Int,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
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
    var ls_shadow = LightSampler_C(UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(), Int32(0), Int32(0))
    var ctx_shadow = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin](),
        textures=textures, n_textures=n_textures, shadow_tasks=shadow_tasks,
        px_scale=Float32(0.0), sobol_matrices=UnsafePointer[UInt32, MutAnyOrigin].unsafe_dangling(), guide=null_guide(),
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=UnsafePointer[MeasuredBRDF_C, MutAnyOrigin].unsafe_dangling(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=UnsafePointer[DistantLight_C, MutAnyOrigin](), distant_count=0,
            point_lights=UnsafePointer[PointLight_C, MutAnyOrigin](), point_count=0,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls_shadow))
    shade_nee_core[True, True](path_ptr, inter, ctx_shadow)


def traverse_shadow_rays_gpu(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    shadow_tasks: UnsafePointer[ShadowTask_C, MutAnyOrigin],
    count: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var task = shadow_tasks[tid]
    if task.active == 0:
        return
    var shadow_ray = Ray_C(Point3f(task.origin.x, task.origin.y, task.origin.z), Vec3f(task.direction.x, task.direction.y, task.direction.z))
    if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, curves, shadow_ray, task.tmax, blasNodesArr, blasPrimIdsArr, instances):
        paths[tid].estimate += RGB(task.contrib.r, task.contrib.g, task.contrib.b)


def accumulate_film_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    film: UnsafePointer[Float32, MutAnyOrigin],
    albedo_film: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    film[tid*3+0] += paths[tid].estimate.r
    film[tid*3+1] += paths[tid].estimate.g
    film[tid*3+2] += paths[tid].estimate.b
    albedo_film[tid*3+0] += paths[tid].albedo.r
    albedo_film[tid*3+1] += paths[tid].albedo.g
    albedo_film[tid*3+2] += paths[tid].albedo.b


def clear_film_gpu(film: UnsafePointer[Float32, MutAnyOrigin], n_pixels: Int):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n_pixels:
        return
    film[tid*3+0] = Float32(0)
    film[tid*3+1] = Float32(0)
    film[tid*3+2] = Float32(0)


# Wavefront accumulation: thread px sums actual_batch samples from path_buf layout
# path_buf[si * n_pixels + px] and adds to film[px].  No atomics needed (one thread per pixel).
def accumulate_film_wavefront_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    film: UnsafePointer[Float32, MutAnyOrigin],
    albedo_film: UnsafePointer[Float32, MutAnyOrigin],
    n_pixels: Int, actual_batch: Int,
):
    var px = Int(block_idx.x * block_dim.x + thread_idx.x)
    if px >= n_pixels:
        return
    var r = Float32(0); var g = Float32(0); var b = Float32(0)
    var ar = Float32(0); var ag = Float32(0); var ab = Float32(0)
    for si in range(actual_batch):
        var p = paths[si * n_pixels + px]
        r += p.estimate.r; g += p.estimate.g; b += p.estimate.b
        ar += p.albedo.r;  ag += p.albedo.g;  ab += p.albedo.b
    film[px*3+0] += r; film[px*3+1] += g; film[px*3+2] += b
    albedo_film[px*3+0] += ar; albedo_film[px*3+1] += ag; albedo_film[px*3+2] += ab


# Wavefront primary-ray generation: thread ti → pixel (ti % n_pixels), sample (si_start + ti // n_pixels).
# Layout: path_buf[si_local * n_pixels + px_flat] — adjacent threads touch adjacent pixels of same sample.
def gen_primary_rays_wavefront_gpu(
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    r2c: UnsafePointer[Float32, MutAnyOrigin],
    c2w: UnsafePointer[Float32, MutAnyOrigin],
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    fw: Int, fh: Int,
    si_start: Int32, log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    filter_sigma: Float32, filter_norm_x: Float32, filter_support_x: Float32,
    filter_norm_y: Float32, filter_support_y: Float32,
    filter_type: Int32,
    count: Int, n_pixels: Int,
):
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


# Traversal kernel that reads rays directly from PathState_C (no separate ray buffer).
def traverse_paths_gpu(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    results: UnsafePointer[Intersection_C, MutAnyOrigin],
    curve_cand_prim: UnsafePointer[Int32, MutAnyOrigin],
    curve_cand_count: UnsafePointer[Int32, MutAnyOrigin],
    count: Int,
):
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
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    rays: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
):
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
def vulkaninterop_unpack_results_kernel(
    results: UnsafePointer[Float32, MutAnyOrigin],
    inter: UnsafePointer[Intersection_C, MutAnyOrigin],
    mesh_material_idx: UnsafePointer[Int64, MutAnyOrigin],
    mesh_al_idx: UnsafePointer[Int32, MutAnyOrigin],
    n_meshes: Int,
    count: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var idx = tid * 8
    var iresults = results.bitcast[Int32]()
    var hitFlag = iresults[idx + 6]
    if hitFlag == Int32(1):
        var mi = Int(iresults[idx + 4])
        var tri = iresults[idx + 5]
        var mat_idx = Int64(0)
        var al = Int32(-1)
        if mi >= 0 and mi < n_meshes:
            mat_idx = mesh_material_idx[mi]
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
                PrimId_C(Int64(mi), Int64(tri) * 3, mat_idx, Int32(-1),
                         Int8(0), Int8(0), Int8(0), Int8(0)),
                hitT, u, v, Int8(1), Int8(0), Int8(0), Int8(0),
            )
    else:
        inter[tid].hit = Int8(0)

# Host driver: enqueues pack -> interop ray-query dispatch -> unpack, ALL
# on ctx.stream() in strict program order, with NO ctx.synchronize()
# anywhere -- vulkaninterop_rt_trace's internal CUDA-signal/Vulkan-submit/
# CUDA-wait handoff (see vulkaninterop.h) is what keeps the Vulkan
# dispatch correctly ordered relative to the pack/unpack kernels on either
# side of it, entirely on the GPU timeline. This is what makes stage 3
# fast where the retired host-round-trip version was ~94x slower: no CPU
# stall, no host-memory copy, anywhere in this function.
#
# Scope: triangle geometry only, matching vulkaninterop_rt_create_scene --
# caller must only invoke this for scenes with no curves/spheres/object
# instancing (mirrors debug_render_vulkanrt's own scope boundary).
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
) raises:
    comptime block_size = 256
    var grid = ceildiv(n_total, block_size)

    ctx.enqueue_function[vulkaninterop_pack_rays_kernel](
        path_buf.unsafe_ptr().bitcast[PathState_C](),
        interop_rays_buf.unsafe_ptr(),
        n_total,
        grid_dim=grid, block_dim=block_size,
    )

    var cuda_stream = CUDA(ctx.stream())
    _ = vulkaninterop_rt_trace(interop_scene, Int32(n_total), cuda_stream)

    ctx.enqueue_function[vulkaninterop_unpack_results_kernel](
        interop_results_buf.unsafe_ptr(),
        inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        mesh_material_idx_buf.unsafe_ptr().bitcast[Int64](),
        mesh_al_idx_buf.unsafe_ptr().bitcast[Int32](),
        n_meshes,
        n_total,
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

def reset_curve_counter_gpu(counter: UnsafePointer[Int32, MutAnyOrigin]):
    if block_idx.x == 0 and thread_idx.x == 0:
        counter[0] = Int32(0)

def compact_curve_paths_gpu(
    curve_cand_count: UnsafePointer[Int32, MutAnyOrigin],
    n: Int,
    compact_pathIds: UnsafePointer[Int32, MutAnyOrigin],
    compact_counter: UnsafePointer[Int32, MutAnyOrigin],
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n:
        return
    if curve_cand_count[tid] > Int32(0):
        var pos = Atomic.fetch_add(compact_counter, Int32(1))
        compact_pathIds[Int(pos)] = Int32(tid)

def resolve_curve_candidates_gpu(
    compact_pathIds: UnsafePointer[Int32, MutAnyOrigin],
    compact_counter: UnsafePointer[Int32, MutAnyOrigin],
    curve_cand_prim: UnsafePointer[Int32, MutAnyOrigin],
    curve_cand_count: UnsafePointer[Int32, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    results: UnsafePointer[Intersection_C, MutAnyOrigin],
    n: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n:
        return
    if tid >= Int(compact_counter[0]):
        return
    var pathId = Int(compact_pathIds[tid])
    var n_cand = Int(curve_cand_count[pathId])
    if n_cand == 0:
        return
    var ray = paths[pathId].ray
    var ray_org = SIMD[DType.float32, 3](ray.origin.x, ray.origin.y, ray.origin.z)
    var ray_dir = SIMD[DType.float32, 3](ray.direction.x, ray.direction.y, ray.direction.z)
    var res = results[pathId]
    var best_t = res.tHit
    var best_u = res.u
    var best_v = res.v
    var best_prim = res.primId
    var best_hit = res.hit
    var changed = False
    for i in range(n_cand):
        var primIdx = Int(curve_cand_prim[pathId * CURVE_DEFER_K + i])
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
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    count: Int64,
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin]
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
                for i in range(path_bytes):
                    dst[i] = src[i]

            var inter_buf = handle[].ctx.enqueue_create_buffer[DType.uint8](inter_bytes)
            with inter_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = intersections.bitcast[UInt8]()
                for i in range(inter_bytes):
                    dst[i] = src[i]

            # Launch shading kernel
            comptime block_size = 256
            var grid_dim = ceildiv(n, block_size)

            handle[].ctx.enqueue_function[shade_gpu](
                path_buf.unsafe_ptr().bitcast[PathState_C](),
                inter_buf.unsafe_ptr().bitcast[Intersection_C](),
                handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
                handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
                n,
                grid_dim=grid_dim,
                block_dim=block_size,
            )

            handle[].ctx.synchronize()

            # Transfer path back (they were updated in-place on the device)
            with path_buf.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr()
                var dst = paths.bitcast[UInt8]()
                for i in range(path_bytes):
                    dst[i] = src[i]

        except e:
            print("GPU: Batch shading failed: " + String(e))

# GPU kernel: generate primary PathState_C for every pixel in one pass.
# Each thread handles one pixel.  All sampling is pure math — no host calls.
def gen_primary_rays_gpu(
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    r2c: UnsafePointer[Float32, MutAnyOrigin],
    c2w: UnsafePointer[Float32, MutAnyOrigin],
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    fw: Int, fh: Int,
    si: Int32, log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    filter_sigma: Float32, filter_norm_x: Float32, filter_support_x: Float32,
    filter_norm_y: Float32, filter_support_y: Float32,
    filter_type: Int32,
    count: Int,
):
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


# GPU kernel: shoot one unjittered center ray per pixel and write normals + depth.
# Used to guide the à-trous denoiser with geometric edge information.
def gen_aux_buffers_gpu(
    r2c: UnsafePointer[Float32, MutAnyOrigin],
    c2w: UnsafePointer[Float32, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    curves: UnsafePointer[Curve_C, MutAnyOrigin],
    blasNodesArr: UnsafePointer[UnsafePointer[BVH2Node, MutAnyOrigin], MutAnyOrigin],
    blasPrimIdsArr: UnsafePointer[UnsafePointer[PrimId_C, MutAnyOrigin], MutAnyOrigin],
    instances: UnsafePointer[Instance_C, MutAnyOrigin],
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    isects_tmp: UnsafePointer[Intersection_C, MutAnyOrigin],
    normals_out: UnsafePointer[Float32, MutAnyOrigin],
    depth_out: UnsafePointer[Float32, MutAnyOrigin],
    curve_mask_out: UnsafePointer[Float32, MutAnyOrigin],
    fw: Int, fh: Int,
):
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


def gpu_gen_aux_buffers(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    c2w: UnsafePointer[Float32, MutAnyOrigin],
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
                for i in range(16):
                    dst[i] = c2w[i]
            comptime block_size = 256
            var grid_n = ceildiv(n_pix, block_size)
            handle[].ctx.enqueue_function[gen_aux_buffers_gpu](
                handle[].r2c_buf.unsafe_ptr().bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().bitcast[Float32](),
                handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
                handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
                handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
                handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
                handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]](),
                handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]](),
                handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
                handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
                handle[].n_spheres,
                handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
                handle[].atrous_normals_buf.unsafe_ptr().bitcast[Float32](),
                handle[].atrous_depth_buf.unsafe_ptr().bitcast[Float32](),
                handle[].atrous_curve_mask_buf.unsafe_ptr().bitcast[Float32](),
                handle[].fw, handle[].fh,
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
def _gpu_bounce_kernels(
    handle: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    n: Int, grid_dim: Int, px_scale: Float32,
    use_vulkan_rt: Bool = False,
    interop_scene: VulkanInteropRtSceneHandle = UnsafePointer[UInt8, MutAnyOrigin].unsafe_dangling(),
    interop_rays_buf: Optional[DeviceBuffer[DType.float32]] = None,
    interop_results_buf: Optional[DeviceBuffer[DType.float32]] = None,
    mesh_material_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    mesh_al_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    n_meshes_vk: Int = 0,
) raises:
    comptime block_size = 256
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
        )
    else:
        handle[].ctx.enqueue_function[traverse_paths_gpu](
            handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
            handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
            handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
            handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
            handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]](),
            handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]](),
            handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
            handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
            handle[].n_spheres,
            handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
            handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
            handle[].curve_cand_prim_buf.unsafe_ptr().bitcast[Int32](),
            handle[].curve_cand_count_buf.unsafe_ptr().bitcast[Int32](),
            n,
            grid_dim=grid_dim,
            block_dim=block_size,
        )
    if handle[].n_curves > 0:
        handle[].ctx.enqueue_function[reset_curve_counter_gpu](
            handle[].curve_compact_counter_buf.unsafe_ptr().bitcast[Int32](),
            grid_dim=1, block_dim=1,
        )
        handle[].ctx.enqueue_function[compact_curve_paths_gpu](
            handle[].curve_cand_count_buf.unsafe_ptr().bitcast[Int32](),
            n,
            handle[].curve_compact_path_buf.unsafe_ptr().bitcast[Int32](),
            handle[].curve_compact_counter_buf.unsafe_ptr().bitcast[Int32](),
            grid_dim=grid_dim, block_dim=block_size,
        )
        handle[].ctx.enqueue_function[resolve_curve_candidates_gpu](
            handle[].curve_compact_path_buf.unsafe_ptr().bitcast[Int32](),
            handle[].curve_compact_counter_buf.unsafe_ptr().bitcast[Int32](),
            handle[].curve_cand_prim_buf.unsafe_ptr().bitcast[Int32](),
            handle[].curve_cand_count_buf.unsafe_ptr().bitcast[Int32](),
            handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
            handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
            handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
            handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
            n,
            grid_dim=grid_dim, block_dim=block_size,
        )
    handle[].ctx.enqueue_function[sample_medium_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].mediums_buf.unsafe_ptr().bitcast[Medium_C](),
        handle[].n_mediums,
        handle[].grids_buf.unsafe_ptr().bitcast[Grid_C](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        handle[].n_area_lights,
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        handle[].n_light_sampler,
        n,
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_res,
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_nee_preamble_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        handle[].n_area_lights,
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        handle[].n_textures,
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        handle[].n_distant_lights,
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        handle[].n_point_lights,
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        handle[].n_light_sampler,
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        handle[].n_infinite_lights,
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        handle[].n_spheres,
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        n, px_scale,
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_res,
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
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        n,
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_diffuse_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        handle[].n_area_lights,
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        handle[].n_textures,
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        handle[].n_distant_lights,
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        handle[].n_point_lights,
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        handle[].n_light_sampler,
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        handle[].n_infinite_lights,
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        handle[].n_spheres,
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        n, px_scale,
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_res,
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_coated_diffuse_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        handle[].n_area_lights,
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        handle[].n_textures,
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        handle[].n_distant_lights,
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        handle[].n_point_lights,
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        handle[].n_light_sampler,
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        handle[].n_infinite_lights,
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        handle[].n_spheres,
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        n, px_scale,
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_res,
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_diffuse_transmit_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        handle[].n_area_lights,
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        handle[].n_textures,
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        handle[].n_distant_lights,
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        handle[].n_point_lights,
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        handle[].n_light_sampler,
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        handle[].n_infinite_lights,
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        handle[].n_spheres,
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        n, px_scale,
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_res,
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_conductor_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        handle[].n_area_lights,
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        handle[].n_textures,
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        handle[].n_distant_lights,
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        handle[].n_point_lights,
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        handle[].n_light_sampler,
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        handle[].n_infinite_lights,
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        handle[].n_spheres,
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        n, px_scale,
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_res,
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_measured_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        handle[].n_area_lights,
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        handle[].n_textures,
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        handle[].n_distant_lights,
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        handle[].n_point_lights,
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        handle[].n_light_sampler,
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        handle[].n_infinite_lights,
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        handle[].n_spheres,
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        n, px_scale,
        handle[].measured_brdfs_buf.unsafe_ptr().bitcast[MeasuredBRDF_C](),
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_res,
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_dielectric_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        n,
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_thin_dielectric_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        n,
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_coated_conductor_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        handle[].n_area_lights,
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        handle[].n_textures,
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        handle[].n_distant_lights,
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        handle[].n_point_lights,
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        handle[].n_light_sampler,
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        handle[].n_infinite_lights,
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        handle[].n_spheres,
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        n, px_scale,
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_res,
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_interface_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].medium_ifaces_buf.unsafe_ptr().bitcast[MediumInterface_C](),
        n,
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[update_medium_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].medium_ifaces_buf.unsafe_ptr().bitcast[MediumInterface_C](),
        n,
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_hair_gpu](
        handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
        handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
        handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
        handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
        handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
        handle[].curves_buf.unsafe_ptr().bitcast[Curve_C](),
        handle[].blas_nodes_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[BVH2Node, MutAnyOrigin]](),
        handle[].blas_primids_ptrs_buf.unsafe_ptr().bitcast[UnsafePointer[PrimId_C, MutAnyOrigin]](),
        handle[].instances_buf.unsafe_ptr().bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
        handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
        handle[].n_area_lights,
        handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
        handle[].n_textures,
        handle[].distant_lights_buf.unsafe_ptr().bitcast[DistantLight_C](),
        handle[].n_distant_lights,
        handle[].point_lights_buf.unsafe_ptr().bitcast[PointLight_C](),
        handle[].n_point_lights,
        handle[].light_sampler_buf.unsafe_ptr().bitcast[Float32](),
        handle[].n_light_sampler,
        handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
        handle[].n_infinite_lights,
        handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
        handle[].n_spheres,
        handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
        n, px_scale,
        handle[].spectral_coeffs_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_res,
        handle[].spectral_cie_x_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_y_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_cie_z_buf.unsafe_ptr().bitcast[Float32](),
        handle[].spectral_d65_buf.unsafe_ptr().bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )


# Render one sample pass into the persistent film buffer.
# Ray generation runs on GPU — no CPU-side path buffer or PCIe upload needed.
def gpu_render_sample(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    c2w: UnsafePointer[Float32, MutAnyOrigin],
    si: Int32, log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    n: Int64,
    maxDepth: Int32,
    px_scale: Float32 = Float32(0.0),
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
                for i in range(16):
                    dst[i] = c2w[i]
            comptime block_size = 256
            var grid_dim = ceildiv(n_int, block_size)
            # Generate primary rays on GPU
            handle[].ctx.enqueue_function[gen_primary_rays_gpu](
                handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
                handle[].r2c_buf.unsafe_ptr().bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().bitcast[Float32](),
                handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
                handle[].fw, handle[].fh,
                si, log2spp, n_base4,
                seed_dim0, seed_dim1,
                rng_seed_lo, rng_seed_hi,
                handle[].filter_sigma, handle[].filter_norm_x, handle[].filter_support_x,
                handle[].filter_norm_y, handle[].filter_support_y,
                handle[].filter_type,
                n_int,
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            for _ in range(Int(maxDepth)):
                _gpu_bounce_kernels(handle, n_int, grid_dim, px_scale)
            handle[].ctx.enqueue_function[accumulate_film_gpu](
                handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
                handle[].film_buf.unsafe_ptr().bitcast[Float32](),
                handle[].albedo_film_buf.unsafe_ptr().bitcast[Float32](),
                n_int,
                grid_dim=grid_dim,
                block_dim=block_size,
            )
        except e:
            print("GPU render sample failed: " + String(e))


# Wavefront render: generates actual_batch samples worth of primary rays for all pixels,
# runs the full bounce loop over n_pixels × actual_batch paths together, then accumulates.
# Caller loops over spp in steps of WAVEFRONT_BATCH; progress reporting is up to the caller.
def gpu_render_wavefront(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    c2w: UnsafePointer[Float32, MutAnyOrigin],
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
    # this for scenes with no curves/spheres/instancing (see
    # vulkaninterop_rt_traverse_paths_gpu's docstring). The interop_*/
    # mesh_*_buf/n_meshes_vk params are ignored when use_vulkan_rt is False.
    use_vulkan_rt: Bool = False,
    interop_scene: VulkanInteropRtSceneHandle = UnsafePointer[UInt8, MutAnyOrigin].unsafe_dangling(),
    interop_rays_buf: Optional[DeviceBuffer[DType.float32]] = None,
    interop_results_buf: Optional[DeviceBuffer[DType.float32]] = None,
    mesh_material_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    mesh_al_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    n_meshes_vk: Int = 0,
):
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
                for i in range(16):
                    dst[i] = c2w[i]
            comptime block_size = 256
            var grid_total = ceildiv(n_total, block_size)
            var grid_pix   = ceildiv(n_pix, block_size)
            handle[].ctx.enqueue_function[gen_primary_rays_wavefront_gpu](
                handle[].sobol_buf.unsafe_ptr().bitcast[UInt32](),
                handle[].r2c_buf.unsafe_ptr().bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().bitcast[Float32](),
                handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
                handle[].fw, handle[].fh,
                si_start, log2spp, n_base4,
                seed_dim0, seed_dim1, rng_seed_lo, rng_seed_hi,
                handle[].filter_sigma, handle[].filter_norm_x, handle[].filter_support_x,
                handle[].filter_norm_y, handle[].filter_support_y,
                handle[].filter_type,
                n_total, n_pix,
                grid_dim=grid_total,
                block_dim=block_size,
            )
            for _ in range(Int(maxDepth)):
                _gpu_bounce_kernels(
                    handle, n_total, grid_total, px_scale,
                    use_vulkan_rt, interop_scene, interop_rays_buf, interop_results_buf,
                    mesh_material_idx_buf, mesh_al_idx_buf, n_meshes_vk,
                )
            handle[].ctx.enqueue_function[accumulate_film_wavefront_gpu](
                handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
                handle[].film_buf.unsafe_ptr().bitcast[Float32](),
                handle[].albedo_film_buf.unsafe_ptr().bitcast[Float32](),
                n_pix, batch,
                grid_dim=grid_pix,
                block_dim=block_size,
            )
        except e:
            print("GPU wavefront render failed: " + String(e))


def gpu_download_film(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    film: UnsafePointer[Float32, MutAnyOrigin],
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
                for i in range(film_bytes):
                    dst[i] = src[i]
        except e:
            print("GPU download film failed: " + String(e))


def gpu_download_albedo(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    film: UnsafePointer[Float32, MutAnyOrigin],
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
                for i in range(film_bytes):
                    dst[i] = src[i]
        except e:
            print("GPU download albedo failed: " + String(e))


# ── À-trous wavelet denoiser (Dammertz et al. 2010) ─────────────────────────
# Three kernels: normalize, variance estimate, one à-trous pass (5× ping-pong).

def normalize_beauty_albedo_gpu(
    film: UnsafePointer[Float32, MutAnyOrigin],
    albedo_film: UnsafePointer[Float32, MutAnyOrigin],
    beauty_out: UnsafePointer[Float32, MutAnyOrigin],
    albedo_out: UnsafePointer[Float32, MutAnyOrigin],
    n_pixels: Int,
    inv_weight: Float32,
    iso_scale: Float32,
    max_comp: Float32,
):
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
    beauty: UnsafePointer[Float32, MutAnyOrigin],
    variance_out: UnsafePointer[Float32, MutAnyOrigin],
    fw: Int, fh: Int,
):
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


def atrous_filter_gpu(
    input: UnsafePointer[Float32, MutAnyOrigin],
    albedo: UnsafePointer[Float32, MutAnyOrigin],
    variance: UnsafePointer[Float32, MutAnyOrigin],
    normals: UnsafePointer[Float32, MutAnyOrigin],
    depth: UnsafePointer[Float32, MutAnyOrigin],
    curve_mask: UnsafePointer[Float32, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    fw: Int, fh: Int,
    step: Int,
    sigma_l: Float32,
    sigma_a: Float32,
    sigma_n: Float32,
    sigma_d: Float32,
):
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
    var sigma_l2 = sigma_l * sigma_l * var_p + Float32(1e-6)
    var sigma_a2 = sigma_a * sigma_a
    var ca = RGB(albedo[tid*3], albedo[tid*3+1], albedo[tid*3+2])
    var cn = Vec3f(normals[tid*3], normals[tid*3+1], normals[tid*3+2])
    var cd = depth[tid]
    var inv_sn = Float32(1) / sigma_n
    var inv2sd = Float32(1) / (Float32(2) * sigma_d * sigma_d)
    # Clamp depth before squaring to avoid Float32 overflow (background sentinel=1e38).
    var cd_clamped = min(cd, Float32(1e18))
    var cd_sq = max(cd_clamped * cd_clamped, Float32(1e-6))

    var acc = RGB(Float32(0))
    var acc_w = Float32(0)

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
            var w_l = exp(-dl * dl / sigma_l2)
            var dalb = RGB(albedo[ni], albedo[ni+1], albedo[ni+2]) - ca
            var w_a = exp(-(dalb * dalb).sum() / sigma_a2)
            var ndot = normals[ni]*cn.x + normals[ni+1]*cn.y + normals[ni+2]*cn.z
            var normal_diff = max(Float32(0), Float32(1) - ndot)
            var dd = min(depth[ni1], Float32(1e18)) - cd_clamped
            var rel_depth2 = (dd * dd) / cd_sq
            var w_nd = exp(-(normal_diff * inv_sn + rel_depth2 * inv2sd))
            var adx = dx if dx >= 0 else -dx; var ady = dy if dy >= 0 else -dy
            var hx = Float32(0.0625) if adx == 2 else (Float32(0.25) if adx == 1 else Float32(0.375))
            var hy = Float32(0.0625) if ady == 2 else (Float32(0.25) if ady == 1 else Float32(0.375))
            var w_s = hx * hy
            var w = w_s * w_l * w_a * w_nd
            acc += qc * w
            acc_w += w

    if acc_w > Float32(0):
        var o = acc / acc_w
        output[tid*3] = o.r; output[tid*3+1] = o.g; output[tid*3+2] = o.b
    else:
        output[tid*3] = c.r; output[tid*3+1] = c.g; output[tid*3+2] = c.b


def gpu_atrous_denoise(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
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
                n_pix, inv_weight, iso_scale, film_max_comp,
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
                    for i in range(bytes_b):
                        dst[i] = src[i]
                return
            handle[].ctx.enqueue_function[estimate_variance_gpu](
                handle[].atrous_ping_buf.unsafe_ptr().bitcast[Float32](),
                handle[].atrous_variance_buf.unsafe_ptr().bitcast[Float32](),
                fw, fh,
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
                var src_ptr = ping_ptr if i % 2 == 0 else pong_ptr
                var dst_ptr = pong_ptr if i % 2 == 0 else ping_ptr
                handle[].ctx.enqueue_function[atrous_filter_gpu](
                    src_ptr, alb_ptr, var_ptr, nrm_ptr, dep_ptr, cmask_ptr, dst_ptr,
                    fw, fh, step,
                    Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05),
                    grid_dim=grid_n, block_dim=block_size,
                )
            # Result is in pong if n_passes is odd, ping if even.
            handle[].ctx.synchronize()
            var bytes = n_pix * 12
            var result_buf = handle[].atrous_pong_buf if n_passes % 2 == 1 else handle[].atrous_ping_buf
            with result_buf.map_to_host() as h:
                var src = h.unsafe_ptr()
                var dst = output.bitcast[UInt8]()
                for i in range(bytes):
                    dst[i] = src[i]
        except e:
            print("GPU atrous denoise failed: " + String(e))


def gpu_clear_film(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
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
                n_int,
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            handle[].ctx.enqueue_function[clear_film_gpu](
                handle[].albedo_film_buf.unsafe_ptr().bitcast[Float32](),
                n_int,
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            handle[].ctx.synchronize()
        except e:
            print("GPU clear film failed: " + String(e))


def gpu_free_scene(handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin]):
    if Int(handlePtr) == 0:
        return
    handlePtr.destroy_pointee()
    handlePtr.bitcast[GpuSceneHandle]().free()
