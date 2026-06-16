from std.sys import has_accelerator, has_nvidia_gpu_accelerator
from std.sys.info import size_of
from std.gpu import block_idx, thread_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer
from std.atomic import Atomic
from std.math import ceildiv, sqrt, cos, sin, log, exp
from std.memory import alloc
from .geometry import RGB, Point3f, Vec3f, Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Material_C, AreaLight_C, Sphere_C, DistantLight_C, PointLight_C, InfiniteLight_C, PathState_C, GpuTexture_C, ShadowTask_C, dot, cross
from std.ffi import external_call
from .bvh import BVH2Node, SceneDescriptor2_C, traverse_bvh2_core, any_hit_bvh2_core, test_spheres
from .rng import PCG32
from .shading import shade_core, shade_nee_core
from .sampling import encode_morton2, sobol_get_sample_index, sobol_sample, gaussian_sample_1d, derive_pcg_seeds, gen_primary_ray_state

# Number of samples per pixel processed together in one wavefront bounce loop.
# path_buf and inter_buf are pre-allocated at n_pixels × WAVEFRONT_BATCH.
comptime WAVEFRONT_BATCH: Int = 8

# GPU scene handle — holds DeviceContext and device-resident scene buffers.
# Allocated on the heap, returned as an opaque pointer.
@fieldwise_init
struct GpuSceneHandle(Movable):
    var ctx: DeviceContext
    var bvh2Nodes_buf: DeviceBuffer[DType.uint8]
    var primIds_buf: DeviceBuffer[DType.uint8]
    var meshes_buf: DeviceBuffer[DType.uint8]
    var mesh_count: Int
    var materials_buf: DeviceBuffer[DType.uint8]
    var material_count: Int
    # Keep all per-mesh device buffers alive
    var points_bufs: List[DeviceBuffer[DType.uint8]]
    var faceIndices_bufs: List[DeviceBuffer[DType.uint8]]
    var vertexIndices_bufs: List[DeviceBuffer[DType.uint8]]
    var uv_bufs: List[DeviceBuffer[DType.uint8]]
    var tex_data_bufs: List[DeviceBuffer[DType.uint8]]
    var textures_buf: DeviceBuffer[DType.uint8]  # array of GpuTexture_C
    var n_textures: Int
    var area_lights_buf: DeviceBuffer[DType.uint8]  # n_lights × sizeof(AreaLight_C) = 24
    var n_area_lights: Int
    var spheres_buf: DeviceBuffer[DType.uint8]   # n_spheres × sizeof(Sphere_C) = 36
    var n_spheres: Int
    var infinite_lights_buf: DeviceBuffer[DType.uint8]  # n_infinite × sizeof(InfiniteLight_C) = 48
    var il_pixels_bufs: List[DeviceBuffer[DType.uint8]] # per-light HDR pixel data on GPU
    var il_cdf_bufs: List[DeviceBuffer[DType.uint8]]    # per-light 2D CDF on GPU
    var il_w2l_bufs: List[DeviceBuffer[DType.uint8]]    # per-light world_to_light matrix on GPU
    var n_infinite_lights: Int
    # Persistent render buffers — sized for n_pixels × WAVEFRONT_BATCH (wavefront pass)
    # gpu_render_sample (interactive) only uses the first n_pixels slots.
    var path_buf: DeviceBuffer[DType.uint8]   # n_pixels × WAVEFRONT_BATCH × sizeof(PathState_C)=96
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
    var shadow_buf: DeviceBuffer[DType.uint8]       # n_pixels × sizeof(ShadowTask_C) = 48
    var active_count_buf: DeviceBuffer[DType.uint8] # 1 × Int32
    var active_idx_buf: DeviceBuffer[DType.uint8]   # n_pixels × Int32
    var n_pixels: Int
    # Camera and sampling data for GPU-side ray generation
    var sobol_buf: DeviceBuffer[DType.uint8]  # 2 dims × 52 UInt32 = 416 bytes
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

def gpu_available() -> Bool:
    return has_accelerator()

def gpu_upload_scene(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    bvh2NodesCount: Int64,
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    primIdsCount: Int64,
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    meshCount: Int64,
    meshPointsCounts: UnsafePointer[Int64, MutAnyOrigin],
    meshFaceIndicesCounts: UnsafePointer[Int64, MutAnyOrigin],
    meshVertexIndicesCounts: UnsafePointer[Int64, MutAnyOrigin],
    meshUvNVerts: UnsafePointer[Int64, MutAnyOrigin],
    tex_filenames: UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin],
    n_tex: Int32,
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    materialCount: Int64,
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int64,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    sphereCount: Int64,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    infiniteLightCount: Int64,
    n_pixels: Int64,
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    r2c: UnsafePointer[Float32, MutAnyOrigin],
    c2w_init: UnsafePointer[Float32, MutAnyOrigin],
    filter_sigma: Float32, filter_support_x: Float32, filter_support_y: Float32,
    filter_norm_x: Float32, filter_norm_y: Float32,
    filter_type: Int32,
    fw: Int32, fh: Int32,
) -> UnsafePointer[GpuSceneHandle, MutAnyOrigin]:
    comptime if has_accelerator():
        try:
            var ctx = DeviceContext()

            # Check GPU memory
            var mem_info = ctx.get_memory_info()
            var free_bytes = mem_info[0]
            var total_bytes = mem_info[1]

            var bvh_bytes = Int(bvh2NodesCount) * size_of[BVH2Node]()
            var prim_bytes = Int(primIdsCount) * size_of[PrimId_C]()
            var mesh_struct_bytes = Int(meshCount) * size_of[TriangleMesh_C]()
            var material_struct_bytes = Int(materialCount) * size_of[Material_C]()

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

            # Upload BVH nodes
            var bvh_buf = ctx.enqueue_create_buffer[DType.uint8](bvh_bytes)
            with bvh_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = bvh2Nodes.bitcast[UInt8]()
                for i in range(bvh_bytes):
                    dst[i] = src[i]

            # Upload prim IDs
            var prim_buf = ctx.enqueue_create_buffer[DType.uint8](prim_bytes)
            with prim_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = primIds.bitcast[UInt8]()
                for i in range(prim_bytes):
                    dst[i] = src[i]

            # Upload per-mesh vertex/index/uv data and build device-side mesh structs
            var points_bufs = List[DeviceBuffer[DType.uint8]]()
            var face_bufs = List[DeviceBuffer[DType.uint8]]()
            var vert_bufs = List[DeviceBuffer[DType.uint8]]()
            var uv_bufs   = List[DeviceBuffer[DType.uint8]]()

            var mesh_structs_host = alloc[TriangleMesh_C](Int(meshCount))

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

                mesh_structs_host[i] = TriangleMesh_C(
                    pts_buf.unsafe_ptr().bitcast[Float32](),
                    fi_buf.unsafe_ptr().bitcast[Int64](),
                    vi_buf.unsafe_ptr().bitcast[Int64](),
                    uv_buf.unsafe_ptr().bitcast[Float32](),
                )

                points_bufs.append(pts_buf^)
                face_bufs.append(fi_buf^)
                vert_bufs.append(vi_buf^)
                uv_bufs.append(uv_buf^)

            # Upload mesh struct array
            var meshes_buf = ctx.enqueue_create_buffer[DType.uint8](mesh_struct_bytes)
            with meshes_buf.map_to_host() as host_buf:
                var dst = host_buf.unsafe_ptr()
                var src = mesh_structs_host.bitcast[UInt8]()
                for j in range(mesh_struct_bytes):
                    dst[j] = src[j]

            mesh_structs_host.free()

            # Upload materials array
            var mat_bytes = Int(materialCount) * size_of[Material_C]()
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
            var r_shadow_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[ShadowTask_C]())
            var r_active_count_buf = ctx.enqueue_create_buffer[DType.uint8](4)
            var r_active_idx_buf   = ctx.enqueue_create_buffer[DType.uint8](n_pix * 4)
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
            var tex_data_bufs = List[DeviceBuffer[DType.uint8]]()
            var gpu_textures_host = alloc[GpuTexture_C](max(n_textures_int, 1))
            for ti in range(n_textures_int):
                var filename = tex_filenames[ti]
                var data_out = alloc[UnsafePointer[Float32, MutAnyOrigin]](1)
                var w_out = alloc[Int32](1)
                var h_out = alloc[Int32](1)
                w_out[0] = Int32(0); h_out[0] = Int32(0)
                var ok = external_call["load_texture_rgb", Int32,
                    UnsafePointer[UInt8, MutAnyOrigin],
                    UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin],
                    UnsafePointer[Int32, MutAnyOrigin],
                    UnsafePointer[Int32, MutAnyOrigin]](filename, data_out, w_out, h_out)
                if ok != 0 and Int(w_out[0]) > 0:
                    var tw = Int(w_out[0]); var th = Int(h_out[0])
                    var tex_bytes = tw * th * 3 * 4
                    var tex_buf = ctx.enqueue_create_buffer[DType.uint8](tex_bytes)
                    with tex_buf.map_to_host() as h:
                        var dst = h.unsafe_ptr().bitcast[Float32]()
                        var src = data_out[0]
                        for j in range(tw * th * 3):
                            dst[j] = src[j]
                    gpu_textures_host[ti] = GpuTexture_C(tex_buf.unsafe_ptr().bitcast[Float32](), Int32(tw), Int32(th))
                    _ = external_call["free_texture_rgb", Int32, UnsafePointer[Float32, MutAnyOrigin]](data_out[0])
                    tex_data_bufs.append(tex_buf^)
                else:
                    gpu_textures_host[ti] = GpuTexture_C(UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(), Int32(0), Int32(0))
                data_out.free(); w_out.free(); h_out.free()
            var tex_struct_bytes = max(n_textures_int, 1) * 16  # sizeof(GpuTexture_C) = 16
            var textures_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](tex_struct_bytes)
            with textures_gpu_buf.map_to_host() as h:
                var dst = h.unsafe_ptr()
                var src = gpu_textures_host.bitcast[UInt8]()
                for j in range(tex_struct_bytes):
                    dst[j] = src[j]
            gpu_textures_host.free()
            print("GPU: " + String(n_textures_int) + " texture(s) uploaded")

            # Upload Sobol matrices: first 2 dimensions × 52 UInt32 = 416 bytes
            var sobol_gpu_buf = ctx.enqueue_create_buffer[DType.uint8](416)
            with sobol_gpu_buf.map_to_host() as h:
                var dst = h.unsafe_ptr().bitcast[UInt32]()
                for i in range(104):
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

            # Allocate handle on heap
            var handle = alloc[GpuSceneHandle](1)
            handle.init_pointee_move(GpuSceneHandle(
                ctx=ctx^,
                bvh2Nodes_buf=bvh_buf^,
                primIds_buf=prim_buf^,
                meshes_buf=meshes_buf^,
                mesh_count=Int(meshCount),
                materials_buf=mat_buf^,
                material_count=Int(materialCount),
                points_bufs=points_bufs^,
                faceIndices_bufs=face_bufs^,
                vertexIndices_bufs=vert_bufs^,
                uv_bufs=uv_bufs^,
                tex_data_bufs=tex_data_bufs^,
                textures_buf=textures_gpu_buf^,
                n_textures=n_textures_int,
                area_lights_buf=al_buf^,
                n_area_lights=Int(areaLightCount),
                spheres_buf=sphere_buf^,
                n_spheres=Int(sphereCount),
                infinite_lights_buf=il_buf^,
                il_pixels_bufs=il_pixels_bufs^,
                il_cdf_bufs=il_cdf_bufs^,
                il_w2l_bufs=il_w2l_bufs^,
                n_infinite_lights=Int(infiniteLightCount),
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
            ))

            print("GPU: scene uploaded")
            return handle.bitcast[GpuSceneHandle]()
        except e:
            print("GPU: Failed to upload scene: " + String(e))
            return UnsafePointer[GpuSceneHandle, MutAnyOrigin].unsafe_dangling()
    else:
        return UnsafePointer[GpuSceneHandle, MutAnyOrigin].unsafe_dangling()


# GPU kernel function — one thread per ray
def traverse_bvh2_gpu(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
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
    traverse_bvh2_core(bvh2Nodes, primIds, meshes, ray, tMax, result_ptr)

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

            handle[].ctx.enqueue_function[traverse_bvh2_gpu, traverse_bvh2_gpu](
                handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
                handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
                handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
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



def shade_nee_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    materials: UnsafePointer[Material_C, MutAnyOrigin],
    areaLights: UnsafePointer[AreaLight_C, MutAnyOrigin],
    areaLightCount: Int,
    textures: UnsafePointer[GpuTexture_C, MutAnyOrigin],
    n_textures: Int,
    infiniteLights: UnsafePointer[InfiniteLight_C, MutAnyOrigin],
    n_infinite_lights: Int,
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    count: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths + tid
    if path_ptr[].active == 0:
        return
    var inter = intersections[tid]
    # Do NOT early-exit on miss — shade_nee_core adds env-light contribution there.
    shade_nee_core[True, False](path_ptr, 0, inter, bvh2Nodes, primIds, meshes, materials, areaLights, areaLightCount,
        UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin].unsafe_dangling(), textures, n_textures,
        UnsafePointer[ShadowTask_C, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[DistantLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        UnsafePointer[PointLight_C, MutAnyOrigin].unsafe_dangling(), 0,
        infiniteLights, n_infinite_lights,
        spheres, n_spheres)


def shade_enqueue_shadow_gpu(
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    intersections: UnsafePointer[Intersection_C, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
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
    shade_nee_core[True, True](path_ptr, tid, inter, bvh2Nodes, primIds, meshes, materials, areaLights, areaLightCount,
        UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin](), textures, n_textures, shadow_tasks,
        UnsafePointer[DistantLight_C, MutAnyOrigin](), 0,
        UnsafePointer[PointLight_C, MutAnyOrigin](), 0,
        infiniteLights, n_infinite_lights,
        spheres, n_spheres)


def traverse_shadow_rays_gpu(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
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
    if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, shadow_ray, task.tmax):
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
    var (ray, pcg_state, pcg_inc) = gen_primary_ray_state(
        px, py, si, Int(log2spp), Int(n_base4),
        seed_dim0, seed_dim1, rng_seed, sobol_matrices, r2c, c2w,
        filter_norm_x, filter_sigma, filter_support_x,
        filter_norm_y, filter_support_y,
        filter_type,
    )
    paths[ti] = PathState_C(
        ray,
        RGB(Float32(1.0), Float32(1.0), Float32(1.0)),
        RGB(Float32(0.0), Float32(0.0), Float32(0.0)),
        RGB(Float32(0.0), Float32(0.0), Float32(0.0)),
        Int32(0), pcg_state, pcg_inc,
        Int8(1), Int8(0), Int8(0), Int8(0),
        Float32(0.0),
        Int32(-1),
    )


# Traversal kernel that reads rays directly from PathState_C (no separate ray buffer).
def traverse_paths_gpu(
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    paths: UnsafePointer[PathState_C, MutAnyOrigin],
    results: UnsafePointer[Intersection_C, MutAnyOrigin],
    count: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if paths[tid].active == 0:
        return
    traverse_bvh2_core(bvh2Nodes, primIds, meshes, paths[tid].ray, Float32(1.0e38), results + tid)
    test_spheres(spheres, n_spheres, paths[tid].ray, results + tid)

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

            handle[].ctx.enqueue_function[shade_gpu, shade_gpu](
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
    var (ray, pcg_state, pcg_inc) = gen_primary_ray_state(
        px, py, si, Int(log2spp), Int(n_base4),
        seed_dim0, seed_dim1, rng_seed, sobol_matrices, r2c, c2w,
        filter_norm_x, filter_sigma, filter_support_x,
        filter_norm_y, filter_support_y,
        filter_type,
    )
    paths[tid] = PathState_C(
        ray,
        RGB(Float32(1.0), Float32(1.0), Float32(1.0)),
        RGB(Float32(0.0), Float32(0.0), Float32(0.0)),
        RGB(Float32(0.0), Float32(0.0), Float32(0.0)),
        Int32(0), pcg_state, pcg_inc,
        Int8(1), Int8(0), Int8(0), Int8(0),
        Float32(0.0),
        Int32(-1),
    )


# GPU kernel: shoot one unjittered center ray per pixel and write normals + depth.
# Used to guide the à-trous denoiser with geometric edge information.
def gen_aux_buffers_gpu(
    r2c: UnsafePointer[Float32, MutAnyOrigin],
    c2w: UnsafePointer[Float32, MutAnyOrigin],
    bvh2Nodes: UnsafePointer[BVH2Node, MutAnyOrigin],
    primIds: UnsafePointer[PrimId_C, MutAnyOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutAnyOrigin],
    spheres: UnsafePointer[Sphere_C, MutAnyOrigin],
    n_spheres: Int,
    isects_tmp: UnsafePointer[Intersection_C, MutAnyOrigin],
    normals_out: UnsafePointer[Float32, MutAnyOrigin],
    depth_out: UnsafePointer[Float32, MutAnyOrigin],
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
    var dx = c2w[0]*cx + c2w[4]*cy + c2w[8]*cz
    var dy = c2w[1]*cx + c2w[5]*cy + c2w[9]*cz
    var dz = c2w[2]*cx + c2w[6]*cy + c2w[10]*cz
    var dl = sqrt(dx*dx + dy*dy + dz*dz)
    if dl > Float32(0): dx /= dl; dy /= dl; dz /= dl
    var ox = c2w[12]; var oy = c2w[13]; var oz = c2w[14]

    var ray = Ray_C(Point3f(ox, oy, oz), Vec3f(dx, dy, dz))
    var dummy_id = PrimId_C(Int64(-1), Int64(-1), Int64(0), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0))
    isects_tmp[tid] = Intersection_C(dummy_id, Float32(1e38), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
    traverse_bvh2_core(bvh2Nodes, primIds, meshes, ray, Float32(1e38), isects_tmp + tid)
    test_spheres(spheres, n_spheres, ray, isects_tmp + tid)

    var nx = Float32(0); var ny = Float32(0); var nz = Float32(1)
    var d = Float32(1e38)

    if isects_tmp[tid].hit != Int8(0):
        d = isects_tmp[tid].tHit
        var typ = Int(isects_tmp[tid].primId.type)
        if typ == 4:
            var si = Int(isects_tmp[tid].primId.id1)
            var sc = spheres[si].center
            var hx = ox + dx*d - sc.x
            var hy = oy + dy*d - sc.y
            var hz = oz + dz*d - sc.z
            var hl = sqrt(hx*hx + hy*hy + hz*hz)
            if hl > Float32(0): nx = hx/hl; ny = hy/hl; nz = hz/hl
        else:
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
            var e1x = mesh.points[vi1*4]   - mesh.points[vi0*4]
            var e1y = mesh.points[vi1*4+1] - mesh.points[vi0*4+1]
            var e1z = mesh.points[vi1*4+2] - mesh.points[vi0*4+2]
            var e2x = mesh.points[vi2*4]   - mesh.points[vi0*4]
            var e2y = mesh.points[vi2*4+1] - mesh.points[vi0*4+1]
            var e2z = mesh.points[vi2*4+2] - mesh.points[vi0*4+2]
            nx = e1y*e2z - e1z*e2y
            ny = e1z*e2x - e1x*e2z
            nz = e1x*e2y - e1y*e2x
            var nl = sqrt(nx*nx + ny*ny + nz*nz)
            if nl > Float32(0): nx /= nl; ny /= nl; nz /= nl
        if nx*(-dx) + ny*(-dy) + nz*(-dz) < Float32(0):
            nx = -nx; ny = -ny; nz = -nz

    normals_out[tid*3+0] = nx
    normals_out[tid*3+1] = ny
    normals_out[tid*3+2] = nz
    depth_out[tid] = d


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
            handle[].ctx.enqueue_function[gen_aux_buffers_gpu, gen_aux_buffers_gpu](
                handle[].r2c_buf.unsafe_ptr().bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().bitcast[Float32](),
                handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
                handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
                handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
                handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
                handle[].n_spheres,
                handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
                handle[].atrous_normals_buf.unsafe_ptr().bitcast[Float32](),
                handle[].atrous_depth_buf.unsafe_ptr().bitcast[Float32](),
                handle[].fw, handle[].fh,
                grid_dim=grid_n, block_dim=block_size,
            )
            handle[].ctx.synchronize()
        except e:
            print("GPU gen aux buffers failed: " + String(e))


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
            handle[].ctx.enqueue_function[gen_primary_rays_gpu, gen_primary_rays_gpu](
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
                handle[].ctx.enqueue_function[traverse_paths_gpu, traverse_paths_gpu](
                    handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
                    handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
                    handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
                    handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
                    handle[].n_spheres,
                    handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
                    handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
                    n_int,
                    grid_dim=grid_dim,
                    block_dim=block_size,
                )
                handle[].ctx.enqueue_function[shade_nee_gpu, shade_nee_gpu](
                    handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
                    handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
                    handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
                    handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
                    handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
                    handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
                    handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
                    handle[].n_area_lights,
                    handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
                    handle[].n_textures,
                    handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
                    handle[].n_infinite_lights,
                    handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
                    handle[].n_spheres,
                    n_int,
                    grid_dim=grid_dim,
                    block_dim=block_size,
                )
            handle[].ctx.enqueue_function[accumulate_film_gpu, accumulate_film_gpu](
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
            handle[].ctx.enqueue_function[gen_primary_rays_wavefront_gpu, gen_primary_rays_wavefront_gpu](
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
                handle[].ctx.enqueue_function[traverse_paths_gpu, traverse_paths_gpu](
                    handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
                    handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
                    handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
                    handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
                    handle[].n_spheres,
                    handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
                    handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
                    n_total,
                    grid_dim=grid_total,
                    block_dim=block_size,
                )
                handle[].ctx.enqueue_function[shade_nee_gpu, shade_nee_gpu](
                    handle[].path_buf.unsafe_ptr().bitcast[PathState_C](),
                    handle[].inter_buf.unsafe_ptr().bitcast[Intersection_C](),
                    handle[].bvh2Nodes_buf.unsafe_ptr().bitcast[BVH2Node](),
                    handle[].primIds_buf.unsafe_ptr().bitcast[PrimId_C](),
                    handle[].meshes_buf.unsafe_ptr().bitcast[TriangleMesh_C](),
                    handle[].materials_buf.unsafe_ptr().bitcast[Material_C](),
                    handle[].area_lights_buf.unsafe_ptr().bitcast[AreaLight_C](),
                    handle[].n_area_lights,
                    handle[].textures_buf.unsafe_ptr().bitcast[GpuTexture_C](),
                    handle[].n_textures,
                    handle[].infinite_lights_buf.unsafe_ptr().bitcast[InfiniteLight_C](),
                    handle[].n_infinite_lights,
                    handle[].spheres_buf.unsafe_ptr().bitcast[Sphere_C](),
                    handle[].n_spheres,
                    n_total,
                    grid_dim=grid_total,
                    block_dim=block_size,
                )
            handle[].ctx.enqueue_function[accumulate_film_wavefront_gpu, accumulate_film_wavefront_gpu](
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

    var cr = input[tid*3]; var cg = input[tid*3+1]; var cb = input[tid*3+2]
    var cl = Float32(0.2126)*cr + Float32(0.7152)*cg + Float32(0.0722)*cb
    var var_p = variance[tid]
    var sigma_l2 = sigma_l * sigma_l * var_p + Float32(1e-6)
    var sigma_a2 = sigma_a * sigma_a
    var car = albedo[tid*3]; var cag = albedo[tid*3+1]; var cab = albedo[tid*3+2]
    var cnx = normals[tid*3]; var cny = normals[tid*3+1]; var cnz = normals[tid*3+2]
    var cd = depth[tid]
    var inv_sn = Float32(1) / sigma_n
    var inv2sd = Float32(1) / (Float32(2) * sigma_d * sigma_d)
    # Clamp depth before squaring to avoid Float32 overflow (background sentinel=1e38).
    var cd_clamped = min(cd, Float32(1e18))
    var cd_sq = max(cd_clamped * cd_clamped, Float32(1e-6))

    var acc_r = Float32(0); var acc_g = Float32(0); var acc_b = Float32(0)
    var acc_w = Float32(0)

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            var nx = px + dx * step; var ny = py + dy * step
            if nx < 0 or nx >= fw or ny < 0 or ny >= fh:
                continue
            var ni = (ny * fw + nx) * 3
            var ni1 = ny * fw + nx
            var ql = Float32(0.2126)*input[ni] + Float32(0.7152)*input[ni+1] + Float32(0.0722)*input[ni+2]
            var dl = ql - cl
            var w_l = exp(-dl * dl / sigma_l2)
            var dar = albedo[ni] - car; var dag = albedo[ni+1] - cag; var dab = albedo[ni+2] - cab
            var w_a = exp(-(dar*dar + dag*dag + dab*dab) / sigma_a2)
            var ndot = normals[ni]*cnx + normals[ni+1]*cny + normals[ni+2]*cnz
            var normal_diff = max(Float32(0), Float32(1) - ndot)
            var dd = min(depth[ni1], Float32(1e18)) - cd_clamped
            var rel_depth2 = (dd * dd) / cd_sq
            var w_nd = exp(-(normal_diff * inv_sn + rel_depth2 * inv2sd))
            var adx = dx if dx >= 0 else -dx; var ady = dy if dy >= 0 else -dy
            var hx = Float32(0.0625) if adx == 2 else (Float32(0.25) if adx == 1 else Float32(0.375))
            var hy = Float32(0.0625) if ady == 2 else (Float32(0.25) if ady == 1 else Float32(0.375))
            var w_s = hx * hy
            var w = w_s * w_l * w_a * w_nd
            acc_r += input[ni] * w; acc_g += input[ni+1] * w; acc_b += input[ni+2] * w
            acc_w += w

    if acc_w > Float32(0):
        output[tid*3] = acc_r / acc_w; output[tid*3+1] = acc_g / acc_w; output[tid*3+2] = acc_b / acc_w
    else:
        output[tid*3] = cr; output[tid*3+1] = cg; output[tid*3+2] = cb


def gpu_atrous_denoise(
    handlePtr: UnsafePointer[GpuSceneHandle, MutAnyOrigin],
    output: UnsafePointer[Float32, MutAnyOrigin],
    n: Int64,
    frame_count: Int32,
    film_iso: Float32,
    film_max_comp: Float32,
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

            handle[].ctx.enqueue_function[normalize_beauty_albedo_gpu, normalize_beauty_albedo_gpu](
                handle[].film_buf.unsafe_ptr().bitcast[Float32](),
                handle[].albedo_film_buf.unsafe_ptr().bitcast[Float32](),
                handle[].atrous_ping_buf.unsafe_ptr().bitcast[Float32](),
                handle[].atrous_albedo_buf.unsafe_ptr().bitcast[Float32](),
                n_pix, inv_weight, iso_scale, film_max_comp,
                grid_dim=grid_n, block_dim=block_size,
            )
            handle[].ctx.enqueue_function[estimate_variance_gpu, estimate_variance_gpu](
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
            # Ramp passes with frame_count: 1 pass at fc=1, 5 passes at fc>=5.
            # Prevents the large effective radius (31px at 5 passes) from averaging
            # lit pixels with unlit ones during fast camera movement.
            var n_passes = min(5, max(1, Int(frame_count)))
            for i in range(n_passes):
                var step = 1 << i   # 1, 2, 4, 8, 16
                var src_ptr = ping_ptr if i % 2 == 0 else pong_ptr
                var dst_ptr = pong_ptr if i % 2 == 0 else ping_ptr
                handle[].ctx.enqueue_function[atrous_filter_gpu, atrous_filter_gpu](
                    src_ptr, alb_ptr, var_ptr, nrm_ptr, dep_ptr, dst_ptr,
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
            handle[].ctx.enqueue_function[clear_film_gpu, clear_film_gpu](
                handle[].film_buf.unsafe_ptr().bitcast[Float32](),
                n_int,
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            handle[].ctx.enqueue_function[clear_film_gpu, clear_film_gpu](
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
