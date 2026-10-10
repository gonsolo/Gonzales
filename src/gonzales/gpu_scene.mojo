from .bvh import BVH2Node, BVH4, SceneView
from .curves import CURVE_DEFER_K, Curve
from .geometry import _is_real_ptr
from .guide import GuideGrid, null_guide
from .lights import AreaLight, DistantLight, InfiniteLight, PointLight, LightSampler
from .materials import MatKind, Material, MeasuredBRDF, is_specular_glass
from .footprint import CameraFootprint
from .media import Grid, MediumInterface, Medium, NvdbGrid
from .primitives import Instance, Intersection, PrimId, Sphere, TriangleMesh
from .sampling import FILTER_LUT_N, fill_filter_lut
from .render_state import FilmDims, FilterParams, GpuTexture, NormalSlopeMap, PathState, ShadowTask, SHADOW_SLOTS
from std.os import getenv
from .spectrum import SpectralHandle
from .pbrt_parser import ParsedScene_Mojo
from .host_textures import decode_host_textures
from .ptex_cache import PtexCache
from .restir_di import DIReservoir
from .restir_vol import VolReservoir
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.ffi import external_call
from std.math import ceildiv
from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.sys import has_accelerator
from std.sys.info import size_of


comptime WAVEFRONT_BATCH: Int = 8

@always_inline
def typed_ptr[T: AnyType](mut buf: DeviceBuffer[DType.uint8]) -> Pointer[T, MutUntrackedOrigin]:
    """Reinterpret a type-erased byte DeviceBuffer's pointer as Pointer[T]
    with an origin that can escape the caller (unsafe_ptr() alone ties the
    origin to the buffer's local scope; MutUntrackedOrigin is required for
    GpuSceneHandle's buffer accessor methods, whose return values are used
    well past that scope). Collapses the buf.unsafe_ptr().unsafe_bitcast[T]()
    .unsafe_origin_cast[MutUntrackedOrigin]() chain repeated at every
    GpuSceneHandle sub-struct's accessor into one call."""
    return buf.unsafe_ptr().unsafe_bitcast[T]().unsafe_origin_cast[MutUntrackedOrigin]()

@fieldwise_init
struct SpectralBuffers(Movable):
    """Device-side twin of the host SpectralHandle -- see spectrum.mojo's
    long comment on the confirmed by-value SpectralHandle miscompilation for
    why these stay separate DeviceBuffer/Int fields (not a SpectralHandle)
    on GpuSceneHandle itself; this struct only bundles them for the ONE
    place they're all constructed/read together, GpuSceneHandle, which is
    always accessed by pointer -- never passed by value."""
    var coeffs_buf: DeviceBuffer[DType.uint8]
    var cie_x_buf:  DeviceBuffer[DType.uint8]
    var cie_y_buf:  DeviceBuffer[DType.uint8]
    var cie_z_buf:  DeviceBuffer[DType.uint8]
    var d65_buf:    DeviceBuffer[DType.uint8]
    var res: Int

    @always_inline
    def unsafe_ptrs(mut self) -> Tuple[
        Pointer[Float32, MutUntrackedOrigin], Int,
        Pointer[Float32, MutUntrackedOrigin],
        Pointer[Float32, MutUntrackedOrigin],
        Pointer[Float32, MutUntrackedOrigin],
        Pointer[Float32, MutUntrackedOrigin],
    ]:
        """(coeffs, res, cie_x, cie_y, cie_z, d65) -- the individual-pointer
        shape rgb_illuminant_to_spectral_sample/spectral_sample_to_rgb/etc.
        need (never a SpectralHandle by value, see spectrum.mojo's
        miscompilation comment), collapsing the usual 6-line unpack at each
        call site to one line."""
        return (
            typed_ptr[Float32](self.coeffs_buf), self.res,
            typed_ptr[Float32](self.cie_x_buf),
            typed_ptr[Float32](self.cie_y_buf),
            typed_ptr[Float32](self.cie_z_buf),
            typed_ptr[Float32](self.d65_buf),
        )

    @staticmethod
    def upload(
        ctx: DeviceContext,
        spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
        spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
        spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
        spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
        spectral_d65: Pointer[Float32, MutUntrackedOrigin],
    ) raises -> Self:
        # Upload the spectral (Jakob-Hanika) coefficient table + CIE
        # X/Y/Z/D65 tables, if a real one was loaded (spectral.res > 0)
        # — Stage 2c-1, see project_spectral_rendering memory. Dummy
        # 1-element buffers otherwise (BDPT/SPPM GPU dispatch don't wire
        # spectral yet — Stage 3/4 — same "at least 1 elem" convention
        # already used above for zero-size scene data).
        comptime CIE_N = 95
        var spec_coeffs_count = (3 * spectral_res * spectral_res * spectral_res * 3) if spectral_res > 0 else 0
        var spec_cie_count = CIE_N if spectral_res > 0 else 0
        var spec_coeffs_gpu_buf = _gpu_upload_array[Float32](ctx, spectral_coeffs, spec_coeffs_count)
        var spec_cie_x_gpu_buf = _gpu_upload_array[Float32](ctx, spectral_cie_x, spec_cie_count)
        var spec_cie_y_gpu_buf = _gpu_upload_array[Float32](ctx, spectral_cie_y, spec_cie_count)
        var spec_cie_z_gpu_buf = _gpu_upload_array[Float32](ctx, spectral_cie_z, spec_cie_count)
        var spec_d65_gpu_buf   = _gpu_upload_array[Float32](ctx, spectral_d65, spec_cie_count)
        return Self(coeffs_buf=spec_coeffs_gpu_buf^, cie_x_buf=spec_cie_x_gpu_buf^, cie_y_buf=spec_cie_y_gpu_buf^,
                    cie_z_buf=spec_cie_z_gpu_buf^, d65_buf=spec_d65_gpu_buf^, res=spectral_res)

@fieldwise_init
struct BvhBuffers(Movable):
    var nodes_buf: DeviceBuffer[DType.uint8]
    var prim_ids_buf: DeviceBuffer[DType.uint8]

    @always_inline
    def nodes_ptr(mut self) -> Pointer[BVH2Node, MutUntrackedOrigin]:
        return typed_ptr[BVH2Node](self.nodes_buf)

    @always_inline
    def prim_ids_ptr(mut self) -> Pointer[PrimId, MutUntrackedOrigin]:
        return typed_ptr[PrimId](self.prim_ids_buf)

    @staticmethod
    def upload(ctx: DeviceContext, ref s: ParsedScene_Mojo) raises -> Self:
        # Upload BVH nodes and prim IDs (copy only the real bytes; the
        # buffer may be a 1-element placeholder when the scene has no
        # geometry).
        var bvh_buf = _gpu_upload_array[BVH2Node](ctx, s.bvh_nodes_cpu, max(Int(s.bvh_node_count_cpu), 1))   # an empty BVH still has its zero root
        var prim_buf = _gpu_upload_array[PrimId](ctx, s.prim_ids_cpu, Int(s.prim_count_cpu))
        return Self(nodes_buf=bvh_buf^, prim_ids_buf=prim_buf^)

@fieldwise_init
struct BlasBuffers(Movable):
    """Object instancing (see [[project_object_instancing]]/geometry.mojo's
    Instance docs): one device buffer per BLAS (kept alive here), plus two
    small "array of device pointers" buffers so a kernel's
    blasNodesArr[i]/blasPrimIdsArr[i] resolves to the right BLAS's buffer."""
    var nodes_bufs: List[DeviceBuffer[DType.uint8]]
    var primids_bufs: List[DeviceBuffer[DType.uint8]]
    var nodes_ptrs_buf: DeviceBuffer[DType.uint8]
    var primids_ptrs_buf: DeviceBuffer[DType.uint8]
    var n_blas: Int

    @always_inline
    def nodes_arr(mut self) -> Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin]:
        return typed_ptr[Pointer[BVH2Node, MutUntrackedOrigin]](self.nodes_ptrs_buf)

    @always_inline
    def primids_arr(mut self) -> Pointer[Pointer[PrimId, MutUntrackedOrigin], MutUntrackedOrigin]:
        return typed_ptr[Pointer[PrimId, MutUntrackedOrigin]](self.primids_ptrs_buf)

    @staticmethod
    def upload(ctx: DeviceContext, ref s: ParsedScene_Mojo) raises -> Self:
        # Upload object-instancing data: one device buffer per BLAS (its
        # nodes + primids), then two small "array of device pointers"
        # buffers so a kernel's blasNodesArr[i]/blasPrimIdsArr[i] resolves
        # to the right BLAS — same two-level indirection as the CPU side
        # (see bvh.mojo's _traverse_instance_leaf).
        var n_blas_int = Int(s.blas_count)
        var blas_nodes_bufs = List[DeviceBuffer[DType.uint8]]()
        var blas_primids_bufs = List[DeviceBuffer[DType.uint8]]()
        var blas_nodes_ptrs_host = unsafe_alloc[Pointer[UInt8, MutUntrackedOrigin]](max(n_blas_int, 1))
        var blas_primids_ptrs_host = unsafe_alloc[Pointer[UInt8, MutUntrackedOrigin]](max(n_blas_int, 1))
        for bi in range(n_blas_int):
            blas_nodes_ptrs_host[unsafe_offset=bi] = _gpu_upload_owned[BVH2Node](
                ctx, blas_nodes_bufs, s.blas_nodes_arr[unsafe_offset=bi], Int(s.blas_node_counts[unsafe_offset=bi])).unsafe_bitcast[UInt8]()
            blas_primids_ptrs_host[unsafe_offset=bi] = _gpu_upload_owned[PrimId](
                ctx, blas_primids_bufs, s.blas_primids_arr[unsafe_offset=bi], Int(s.blas_primid_counts[unsafe_offset=bi])).unsafe_bitcast[UInt8]()

        var blas_nodes_ptrs_buf = _gpu_upload_array[Pointer[UInt8, MutUntrackedOrigin]](
            ctx, blas_nodes_ptrs_host, n_blas_int)
        var blas_primids_ptrs_buf = _gpu_upload_array[Pointer[UInt8, MutUntrackedOrigin]](
            ctx, blas_primids_ptrs_host, n_blas_int)
        ctx.synchronize()   # the host pointer arrays are freed next
        blas_nodes_ptrs_host.unsafe_free(); blas_primids_ptrs_host.unsafe_free()
        return Self(nodes_bufs=blas_nodes_bufs^, primids_bufs=blas_primids_bufs^, nodes_ptrs_buf=blas_nodes_ptrs_buf^,
                    primids_ptrs_buf=blas_primids_ptrs_buf^, n_blas=n_blas_int)

@fieldwise_init
struct MeshBuffers(Movable):
    var meshes_buf: DeviceBuffer[DType.uint8]
    var mesh_count: Int
    # Keep all per-mesh device buffers alive
    var points_bufs: List[DeviceBuffer[DType.uint8]]
    var faceIndices_bufs: List[DeviceBuffer[DType.uint8]]
    var vertexIndices_bufs: List[DeviceBuffer[DType.uint8]]
    var uv_bufs: List[DeviceBuffer[DType.uint8]]
    var nrm_bufs: List[DeviceBuffer[DType.uint8]]
    var alpha_bufs: List[DeviceBuffer[DType.uint8]]   # one per distinct alpha mask
    var has_alpha: Bool   # some mesh is alpha-cut (mask or constant < 1)

    @always_inline
    def meshes_ptr(mut self) -> Pointer[TriangleMesh, MutUntrackedOrigin]:
        return typed_ptr[TriangleMesh](self.meshes_buf)

    @staticmethod
    def upload(ctx: DeviceContext, ref s: ParsedScene_Mojo) raises -> Self:
        # Upload per-mesh vertex/index/uv data and build device-side mesh structs
        var points_bufs = List[DeviceBuffer[DType.uint8]]()
        var face_bufs = List[DeviceBuffer[DType.uint8]]()
        var vert_bufs = List[DeviceBuffer[DType.uint8]]()
        var uv_bufs   = List[DeviceBuffer[DType.uint8]]()
        var nrm_bufs  = List[DeviceBuffer[DType.uint8]]()
        # Alpha masks are shared between meshes on the host (one per
        # file); upload each once, keyed by its host address.
        var alpha_bufs = List[DeviceBuffer[DType.uint8]]()
        var alpha_host_keys = List[Int]()
        var alpha_dev_ptrs = List[Pointer[UInt8, MutUntrackedOrigin]]()

        var mesh_structs_host = unsafe_alloc[TriangleMesh](max(Int(s.mesh_count), 1))
        var has_alpha = False

        for i in range(Int(s.mesh_count)):
            var host_mesh = s.meshes[unsafe_offset=i]
            if Int(s.mesh_n_tris[unsafe_offset=i]) == 0:
                # Emptied by the template merge (or just empty): nothing refers to it, so it gets no buffers.
                mesh_structs_host[unsafe_offset=i] = host_mesh
                continue
            if host_mesh.alpha_w > Int32(0) or host_mesh.alpha_const < Float32(1.0):
                has_alpha = True

            # Points (Float32), face and vertex indices (Int32), then UVs (2 floats
            # per vertex) and shading normals (3 per vertex), each a zeroed
            # 4-byte buffer when the mesh has none.
            # points are padded to 4 floats per vertex
            var pts_dptr = _gpu_upload_owned[Float32](ctx, points_bufs, host_mesh.points, Int(s.mesh_n_verts[unsafe_offset=i]) * 4)
            var fi_dptr = _gpu_upload_owned[Int32](ctx, face_bufs, host_mesh.faceIndices, Int(s.mesh_n_tris[unsafe_offset=i]))
            var vi_dptr = _gpu_upload_owned[Int32](ctx, vert_bufs, host_mesh.vertexIndices, Int(s.mesh_n_tris[unsafe_offset=i]) * 3)
            var uv_n = Int(s.mesh_uv_n_verts[unsafe_offset=i])
            var uv_dptr: Pointer[Float32, MutUntrackedOrigin]
            if uv_n > 0:
                uv_dptr = _gpu_upload_owned[Float32](ctx, uv_bufs, host_mesh.uvs, uv_n * 2)
            else:
                uv_dptr = _gpu_zeros_owned[Float32](ctx, uv_bufs, 1)
            var nrm_n = Int(s.mesh_nrm_n_verts[unsafe_offset=i])
            var nrm_dptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=1)   # "no normals"
            if nrm_n > 0:
                nrm_dptr = _gpu_upload_owned[Float32](ctx, nrm_bufs, host_mesh.normals, nrm_n * 3)
            else:
                _ = _gpu_zeros_owned[Float32](ctx, nrm_bufs, 1)

            var alpha_dptr = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling()
            if host_mesh.alpha_w > Int32(0):
                var key = Int(host_mesh.alpha)
                var found = -1
                for ai in range(len(alpha_host_keys)):
                    if alpha_host_keys[ai] == key:
                        found = ai
                        break
                if found < 0:
                    found = len(alpha_host_keys)
                    alpha_host_keys.append(key)
                    alpha_dev_ptrs.append(_gpu_upload_owned[UInt8](ctx, alpha_bufs, host_mesh.alpha,
                        Int(host_mesh.alpha_w) * Int(host_mesh.alpha_h)))
                alpha_dptr = alpha_dev_ptrs[found]
            mesh_structs_host[unsafe_offset=i] = TriangleMesh(pts_dptr, fi_dptr, vi_dptr, uv_dptr, nrm_dptr,
                alpha_dptr, host_mesh.alpha_w, host_mesh.alpha_h, host_mesh.alpha_const)
            mesh_structs_host[unsafe_offset=i].emit_two_sided = host_mesh.emit_two_sided
            if _is_real_ptr(host_mesh.materials):
                mesh_structs_host[unsafe_offset=i].materials = _gpu_upload_owned[Int32](
                    ctx, face_bufs, host_mesh.materials, Int(s.mesh_n_tris[unsafe_offset=i]))

        # Upload mesh struct array
        var meshes_buf = _gpu_upload_array[TriangleMesh](ctx, mesh_structs_host, Int(s.mesh_count))
        ctx.synchronize()   # mesh_structs_host is freed next
        mesh_structs_host.unsafe_free()
        return Self(meshes_buf=meshes_buf^, mesh_count=Int(s.mesh_count), points_bufs=points_bufs^,
                    faceIndices_bufs=face_bufs^, vertexIndices_bufs=vert_bufs^, uv_bufs=uv_bufs^,
                    nrm_bufs=nrm_bufs^, alpha_bufs=alpha_bufs^, has_alpha=has_alpha)

@fieldwise_init
struct TextureBuffers(Movable):
    var tex_data_bufs: List[DeviceBuffer[DType.uint8]]
    var textures_buf: DeviceBuffer[DType.uint8]  # array of GpuTexture
    var lut_buf: DeviceBuffer[DType.float32]     # uint8 decode tables: linear at 0, sRGB at 256
    var n_textures: Int
    # Ptex demand paging: the device copy of the cache's pool, and each paged file's tex_data_bufs index.
    var ptex_pool_buf: DeviceBuffer[DType.uint8]
    var ptex_cache: PtexCache
    var ptex_bufs: List[Int]

    @always_inline
    def textures_ptr(mut self) -> Pointer[GpuTexture, MutUntrackedOrigin]:
        return typed_ptr[GpuTexture](self.textures_buf)

    @staticmethod
    def upload(ctx: DeviceContext, ref s: ParsedScene_Mojo) raises -> Self:
        # Decode on the host (host_textures.mojo), then upload here in texture
        # order, so the GPU receives exactly the buffers a serial loop would build.
        var n_textures_int = Int(s.tex_count)
        var decoded = decode_host_textures(s.tex_filenames, n_textures_int, s.materials, Int(s.material_count), s.tex_prefetch)
        var dup_of = decoded.dup_of
        var host_tex = decoded.tex
        var tex_data_bufs = List[DeviceBuffer[DType.uint8]]()
        var gpu_textures_host = unsafe_alloc[GpuTexture](max(n_textures_int, 1))
        var lut_buf = ctx.enqueue_create_buffer[DType.float32](768)
        ctx.enqueue_copy(lut_buf, decoded.lut)
        var lut_dev = lut_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

        var pool_bytes = Int(external_call["ptex_pool_size", Int64]())
        var ptex_pool_buf = ctx.enqueue_create_buffer[DType.uint8](max(pool_bytes, 1))
        var pool_dev = ptex_pool_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        var ptex_cache = PtexCache(pool_bytes)
        var ptex_bufs = List[Int]()

        var tex_bytes = 0
        for ti in range(n_textures_int):
            if dup_of[unsafe_offset=ti] != Int32(-1):
                gpu_textures_host[unsafe_offset=ti] = gpu_textures_host[unsafe_offset=Int(dup_of[unsafe_offset=ti])]
                continue
            var ht = host_tex[unsafe_offset=ti]
            if ht.n_bytes == 0:
                gpu_textures_host[unsafe_offset=ti] = GpuTexture(Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
                    Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), pool_dev,
                    Int32(0), Int32(0), Int32(0), Int32(0), Int32(GpuTexture.FORMAT_F32))
                continue
            var lut = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
            if Int(ht.format) == GpuTexture.FORMAT_U8 or Int(ht.format) == GpuTexture.FORMAT_PTEX:
                lut = lut_dev.unsafe_offset(Int(ht.lut_off))
            if Int(ht.format) == GpuTexture.FORMAT_PTEX and ht.height >= Int32(0):
                _ = ptex_cache.add_file(s.tex_filenames[unsafe_offset=ti], ht.data, Int(ht.width))
                ptex_bufs.append(len(tex_data_bufs))
            gpu_textures_host[unsafe_offset=ti] = GpuTexture(_gpu_upload_owned[UInt8](ctx, tex_data_bufs, ht.data, ht.n_bytes),
                lut, pool_dev, ht.width, ht.height, ht.n_levels, ht.channels, ht.format)
            tex_bytes += ht.n_bytes
        var textures_gpu_buf = _gpu_upload_array[GpuTexture](ctx, gpu_textures_host, n_textures_int)
        # The uploads above are asynchronous; free their host sources once they're done.
        ctx.synchronize()
        var n_unique_tex = 0
        for ti in range(n_textures_int):
            if dup_of[unsafe_offset=ti] == Int32(-1):
                n_unique_tex += 1
        decoded.free_pixels()
        decoded.free_tables()
        gpu_textures_host.unsafe_free()
        print("GPU: " + String(n_textures_int) + " texture(s) uploaded ("
              + String(n_unique_tex) + " unique file(s) loaded, "
              + String(tex_bytes // (1024 * 1024)) + " MB)")
        return Self(tex_data_bufs=tex_data_bufs^, textures_buf=textures_gpu_buf^, lut_buf=lut_buf^, n_textures=n_textures_int,
                    ptex_pool_buf=ptex_pool_buf^, ptex_cache=ptex_cache^, ptex_bufs=ptex_bufs^)

@fieldwise_init
struct LightBuffers(Movable):
    var area_lights_buf: DeviceBuffer[DType.uint8]  # n_lights × sizeof(AreaLight)
    var n_area_lights: Int
    var area_light_cdf_bufs: List[DeviceBuffer[DType.uint8]]   # each mesh light's tri_cdf
    var distant_lights_buf: DeviceBuffer[DType.uint8]  # n_distant × sizeof(DistantLight) = 32
    var n_distant_lights: Int
    var point_lights_buf: DeviceBuffer[DType.uint8]    # n_point × sizeof(PointLight) = 16
    var n_point_lights: Int
    var light_sampler_buf: DeviceBuffer[DType.uint8]   # (n_area+1) × sizeof(Float32) CDF
    var n_light_sampler: Int                           # n_area lights (CDF has n+1 entries)
    var light_pick_buf: DeviceBuffer[DType.uint8]      # (n_all+1) × Float32, _light_pick_cdf
    var n_light_pick: Int
    var infinite_lights_buf: DeviceBuffer[DType.uint8]  # n_infinite × sizeof(InfiniteLight) (grew when the portal fields were added, see lights.mojo)
    var il_pixels_bufs: List[DeviceBuffer[DType.uint8]] # per-light HDR pixel data on GPU
    var il_cdf_bufs: List[DeviceBuffer[DType.uint8]]    # per-light 2D CDF on GPU
    var il_w2l_bufs: List[DeviceBuffer[DType.uint8]]    # per-light world_to_light matrix on GPU
    var n_infinite_lights: Int

    @always_inline
    def area_lights_ptr(mut self) -> Pointer[AreaLight, MutUntrackedOrigin]:
        return typed_ptr[AreaLight](self.area_lights_buf)

    @always_inline
    def distant_lights_ptr(mut self) -> Pointer[DistantLight, MutUntrackedOrigin]:
        return typed_ptr[DistantLight](self.distant_lights_buf)

    @always_inline
    def point_lights_ptr(mut self) -> Pointer[PointLight, MutUntrackedOrigin]:
        return typed_ptr[PointLight](self.point_lights_buf)

    @always_inline
    def light_sampler_ptr(mut self) -> Pointer[Float32, MutUntrackedOrigin]:
        return typed_ptr[Float32](self.light_sampler_buf)

    def light_pick_ptr(mut self) -> Pointer[Float32, MutUntrackedOrigin]:
        return typed_ptr[Float32](self.light_pick_buf)

    @always_inline
    def infinite_lights_ptr(mut self) -> Pointer[InfiniteLight, MutUntrackedOrigin]:
        return typed_ptr[InfiniteLight](self.infinite_lights_buf)

    @staticmethod
    def upload(ctx: DeviceContext, ref s: ParsedScene_Mojo) raises -> Self:
        # Upload area lights. tri_cdf is a HOST pointer in the parsed
        # lights, so each mesh light's CDF goes up on its own and the
        # device copy of the struct points at that.
        var al_cdf_bufs = List[DeviceBuffer[DType.uint8]]()
        var al_host = unsafe_alloc[AreaLight](max(Int(s.area_light_count), 1))
        for ali in range(Int(s.area_light_count)):
            var al_i = s.area_lights[unsafe_offset=ali]
            if al_i.kind == Int8(0) and _is_real_ptr(al_i.tri_cdf) and al_i.n_tris > Int32(0):
                al_i.tri_cdf = _gpu_upload_owned[Float32](ctx, al_cdf_bufs, al_i.tri_cdf, Int(al_i.n_tris))
            else:
                al_i.tri_cdf = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
            al_host[unsafe_offset=ali] = al_i
        var al_buf = _gpu_upload_array[AreaLight](ctx, al_host, Int(s.area_light_count))
        ctx.synchronize()   # al_host is freed next
        al_host.unsafe_free()
        # Upload distant (directional) lights
        var dl_buf = _gpu_upload_array[DistantLight](ctx, s.distant_lights, Int(s.distant_count))

        # Upload point lights
        var pl_buf = _gpu_upload_array[PointLight](ctx, s.point_lights, Int(s.point_count))

        # Upload light sampler CDF (n+1 Float32 entries), in a buffer of at
        # least 2 entries: pad a host copy, since enqueue_copy copies the whole
        # buffer. (No map_to_host anywhere here: its first use pins a ~1.3 GiB
        # host pool, ~1 s of page faults.)
        var ls_entries = Int(s.light_sampler.n) + 1
        var ls_host = unsafe_alloc[Float32](max(ls_entries, 2))
        ls_host[unsafe_offset=1] = Float32(0)
        unsafe_memcpy(dest=ls_host, src=s.light_sampler.cdf, count=ls_entries)
        var ls_buf = _gpu_upload_array[Float32](ctx, ls_host, max(ls_entries, 2))
        ctx.synchronize()   # ls_host is freed next
        ls_host.unsafe_free()
        # The parse-time power pick (pbrt_parser's _light_pick_cdf), >= 2 entries.
        var n_pick = Int(s.light_pick_n)
        var lp_buf = _gpu_upload_array[Float32](ctx, s.light_pick_cdf, max(n_pick + 1, 2))
        ctx.synchronize()

        # Upload infinite/environment lights with GPU-resident pixel/CDF data
        var il_count = Int(s.infinite_count)
        var il_pixels_bufs = List[DeviceBuffer[DType.uint8]]()
        var il_cdf_bufs    = List[DeviceBuffer[DType.uint8]]()
        var il_w2l_bufs    = List[DeviceBuffer[DType.uint8]]()
        var il_patched = unsafe_alloc[InfiniteLight](max(il_count, 1))
        for ii in range(il_count):
            var il = s.infinite_lights[unsafe_offset=ii]
            # world_to_light matrix (16 floats), then pixels + CDF when textured.
            il.world_to_light = _gpu_upload_owned[Float32](ctx, il_w2l_bufs, il.world_to_light, 16)
            if il.cdf_w > Int32(0) and _is_real_ptr(il.pixels_ptr):
                var iw = Int(il.cdf_w); var ih = Int(il.cdf_h)
                # Pixels: iw × ih × 3 floats. CDF: (ih+1) marginal rows + ih×(iw+1) conditional entries.
                il.pixels_ptr = _gpu_upload_owned[Float32](ctx, il_pixels_bufs, il.pixels_ptr, iw * ih * 3)
                il.cdf_ptr = _gpu_upload_owned[Float32](ctx, il_cdf_bufs, il.cdf_ptr, (ih + 1) + ih * (iw + 1))
            il_patched[unsafe_offset=ii] = il
        var il_buf = _gpu_upload_array[InfiniteLight](ctx, il_patched, il_count)
        ctx.synchronize()   # il_patched is freed next
        il_patched.unsafe_free()
        print("GPU: " + String(il_count) + " infinite light(s) uploaded")
        return Self(area_lights_buf=al_buf^, area_light_cdf_bufs=al_cdf_bufs^, n_area_lights=Int(s.area_light_count),
                    distant_lights_buf=dl_buf^, n_distant_lights=Int(s.distant_count),
                    point_lights_buf=pl_buf^, n_point_lights=Int(s.point_count),
                    light_sampler_buf=ls_buf^, n_light_sampler=Int(s.light_sampler.n),
                    light_pick_buf=lp_buf^, n_light_pick=n_pick,
                    infinite_lights_buf=il_buf^, il_pixels_bufs=il_pixels_bufs^, il_cdf_bufs=il_cdf_bufs^,
                    il_w2l_bufs=il_w2l_bufs^, n_infinite_lights=Int(s.infinite_count))

@fieldwise_init
struct CurveBuffers(Movable):
    var curves_buf: DeviceBuffer[DType.uint8]    # n_curves × sizeof(Curve)
    var n_curves: Int
    # Curve-divergence-mitigation scratch (see traverse_bvh2_core_defer_curves).
    # Only meaningfully sized when n_curves > 0; otherwise 1-byte dummies.
    var cand_prim_buf: DeviceBuffer[DType.uint8]      # n_pixels×WAVEFRONT_BATCH×CURVE_DEFER_K × Int32
    var cand_count_buf: DeviceBuffer[DType.uint8]     # n_pixels×WAVEFRONT_BATCH × Int32
    # Each ray's start offset into cand_prim_buf. The CUDA-native path
    # (traverse_bvh2_core_defer_curves) always writes tid*CURVE_DEFER_K-
    # strided candidates, so this is initialized ONCE to that same formula
    # (init_curve_cand_offset_gpu) and never touched again for CUDA-only
    # rendering. The Vulkan RT path OVERWRITES it every bounce with real,
    # uncapped pool offsets from its own count-then-place scheme (see
    # intersect_batch.comp) -- resolve_curve_candidates_gpu always reads
    # through this indirection so one indexing scheme serves both backends.
    var cand_offset_buf: DeviceBuffer[DType.uint8]    # n_pixels×WAVEFRONT_BATCH × Int32
    var compact_path_buf: DeviceBuffer[DType.uint8]   # n_pixels×WAVEFRONT_BATCH × Int32
    var compact_counter_buf: DeviceBuffer[DType.uint8] # 1 × Int32

    @always_inline
    def curves_ptr(mut self) -> Pointer[Curve, MutUntrackedOrigin]:
        return typed_ptr[Curve](self.curves_buf)

    @always_inline
    def cand_prim_ptr(mut self) -> Pointer[Int32, MutUntrackedOrigin]:
        return typed_ptr[Int32](self.cand_prim_buf)

    @always_inline
    def cand_count_ptr(mut self) -> Pointer[Int32, MutUntrackedOrigin]:
        return typed_ptr[Int32](self.cand_count_buf)

    @always_inline
    def cand_offset_ptr(mut self) -> Pointer[Int32, MutUntrackedOrigin]:
        return typed_ptr[Int32](self.cand_offset_buf)

    @always_inline
    def compact_path_ptr(mut self) -> Pointer[Int32, MutUntrackedOrigin]:
        return typed_ptr[Int32](self.compact_path_buf)

    @always_inline
    def compact_counter_ptr(mut self) -> Pointer[Int32, MutUntrackedOrigin]:
        return typed_ptr[Int32](self.compact_counter_buf)

    @staticmethod
    def upload(ctx: DeviceContext, ref s: ParsedScene_Mojo, n_pix: Int) raises -> Self:
        # Upload curves (native hair/fur primitives — control points only, no tessellation)
        var curve_buf = _gpu_upload_array[Curve](ctx, s.curves, Int(s.curve_count))
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
        # both gated behind `if handle[].curves.n_curves > 0` at the dispatch site,
        # so those two are still safe to leave dummy-sized.
        var n_curve_paths = n_pix * WAVEFRONT_BATCH
        var r_curve_cand_prim_buf   = ctx.enqueue_create_buffer[DType.uint8](n_curve_paths * CURVE_DEFER_K * 4)
        var r_curve_cand_count_buf  = ctx.enqueue_create_buffer[DType.uint8](n_curve_paths * 4)
        var r_curve_cand_offset_buf = ctx.enqueue_create_buffer[DType.uint8](n_curve_paths * 4)
        ctx.enqueue_function[init_curve_cand_offset_gpu](
            r_curve_cand_offset_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            Int64(n_curve_paths),
            grid_dim=ceildiv(n_curve_paths, 256), block_dim=256,
        )
        var n_curve_compact_paths = n_curve_paths if Int(s.curve_count) > 0 else 1
        var r_curve_compact_path_buf = ctx.enqueue_create_buffer[DType.uint8](n_curve_compact_paths * 4)
        var r_curve_compact_counter_buf = ctx.enqueue_create_buffer[DType.uint8](4)
        return Self(curves_buf=curve_buf^, n_curves=Int(s.curve_count), cand_prim_buf=r_curve_cand_prim_buf^,
                    cand_count_buf=r_curve_cand_count_buf^, cand_offset_buf=r_curve_cand_offset_buf^,
                    compact_path_buf=r_curve_compact_path_buf^, compact_counter_buf=r_curve_compact_counter_buf^)

@fieldwise_init
struct MediaBuffers(Movable):
    var mediums_buf: DeviceBuffer[DType.uint8]        # n_mediums × sizeof(Medium)
    var n_mediums: Int
    var has_sss_medium: Bool
    var medium_ifaces_buf: DeviceBuffer[DType.uint8]  # n_medium_ifaces × sizeof(MediumInterface)
    var n_medium_ifaces: Int
    var grids_buf: DeviceBuffer[DType.uint8]          # n_grids × sizeof(Grid); Grid.density points into grid_density_bufs
    var n_grids: Int
    var grid_density_bufs: List[DeviceBuffer[DType.uint8]]  # kept alive; one per grid's density array
    var nvdb_grids_buf: DeviceBuffer[DType.uint8]     # n_nvdb_grids × sizeof(NvdbGrid); NvdbGrid.blob points into nvdb_blob_bufs
    var n_nvdb_grids: Int
    var nvdb_blob_bufs: List[DeviceBuffer[DType.uint8]]  # kept alive; one per grid's decompressed .nvdb blob

    @staticmethod
    def upload(ctx: DeviceContext, ref s: ParsedScene_Mojo) raises -> Self:
        # Upload participating media (small array; >= 1 elem to avoid zero-size buffer)
        var med_buf = _gpu_upload_array[Medium](ctx, s.mediums, Int(s.medium_count))
        # Read the SSS flag off the host copy while it is still in reach --
        # the round-budget decision this feeds is made on the host, and
        # reading it back off the device later would need a sync.
        var has_sss_med = False
        for mi in range(Int(s.medium_count)):
            if s.mediums[unsafe_offset=mi].is_sss != Int32(0):
                has_sss_med = True
                break

        # Upload medium interfaces
        var miface_buf = _gpu_upload_array[MediumInterface](ctx, s.medium_ifaces, Int(s.medium_iface_count))

        # Upload heterogeneous density grids ("uniformgrid" media). Each
        # grid's (potentially large) density array gets its own device
        # buffer, mirroring the per-mesh points_bufs pattern; the Grid
        # struct array embeds device-resident pointers into those buffers.
        var grid_density_bufs = List[DeviceBuffer[DType.uint8]]()
        var n_grids_int = Int(s.grid_count)
        var grid_structs_host = unsafe_alloc[Grid](max(n_grids_int, 1))
        for gi in range(n_grids_int):
            var host_grid = s.grids[unsafe_offset=gi]
            var n_voxels = Int(host_grid.nx) * Int(host_grid.ny) * Int(host_grid.nz)
            grid_structs_host[unsafe_offset=gi] = Grid(
                _gpu_upload_owned[Float32](ctx, grid_density_bufs, host_grid.density, n_voxels),
                host_grid.nx, host_grid.ny, host_grid.nz,
                host_grid.p0, host_grid.p1,
                host_grid.world_to_medium, host_grid.max_density)
        var grids_buf = _gpu_upload_array[Grid](ctx, grid_structs_host, n_grids_int)
        ctx.synchronize()   # grid_structs_host is freed next
        grid_structs_host.unsafe_free()
        if n_grids_int > 0:
            print("GPU: " + String(n_grids_int) + " heterogeneous density grid(s) uploaded")

        # Upload sparse density grids ("nanovdb" media). Same shape as
        # the dense-grid upload just above: each grid's decompressed
        # blob gets its own device buffer, and the NvdbGrid struct
        # array embeds device-resident pointers into those buffers.
        var nvdb_blob_bufs = List[DeviceBuffer[DType.uint8]]()
        var n_nvdb_grids_int = Int(s.nvdb_grid_count)
        var nvdb_structs_host = unsafe_alloc[NvdbGrid](max(n_nvdb_grids_int, 1))
        for gi in range(n_nvdb_grids_int):
            var host_nvdb = s.nvdb_grids[unsafe_offset=gi]
            nvdb_structs_host[unsafe_offset=gi] = NvdbGrid(
                _gpu_upload_owned[UInt8](ctx, nvdb_blob_bufs, host_nvdb.blob, Int(host_nvdb.blob_size)),
                host_nvdb.blob_size,
                host_nvdb.world_to_medium, host_nvdb.inv_map, host_nvdb.map_vec,
                host_nvdb.index_min, host_nvdb.index_max, host_nvdb.max_density)
        var nvdb_grids_buf = _gpu_upload_array[NvdbGrid](ctx, nvdb_structs_host, n_nvdb_grids_int)
        ctx.synchronize()   # nvdb_structs_host is freed next
        nvdb_structs_host.unsafe_free()
        if n_nvdb_grids_int > 0:
            print("GPU: " + String(n_nvdb_grids_int) + " sparse (nanovdb) density grid(s) uploaded")
        return Self(mediums_buf=med_buf^, n_mediums=Int(s.medium_count), has_sss_medium=has_sss_med,
                    medium_ifaces_buf=miface_buf^, n_medium_ifaces=Int(s.medium_iface_count),
                    grids_buf=grids_buf^, n_grids=n_grids_int, grid_density_bufs=grid_density_bufs^,
                    nvdb_grids_buf=nvdb_grids_buf^, n_nvdb_grids=n_nvdb_grids_int, nvdb_blob_bufs=nvdb_blob_bufs^)


@fieldwise_init
struct MeasuredBuffers(Movable):
    var brdfs_buf: DeviceBuffer[DType.uint8]  # n_brdfs × sizeof(MeasuredBRDF); each entry's 12 pointer fields point into field_bufs
    var n_brdfs: Int
    var field_bufs: List[DeviceBuffer[DType.uint8]]  # kept alive; 12 sub-array buffers per measured material

    @staticmethod
    def upload(ctx: DeviceContext, ref s: ParsedScene_Mojo) raises -> Self:
        # Upload MeasuredBxDF tabulated-BRDF tensors ("measured" material,
        # see project_measured_bxdf memory / lovely-dazzling-meteor plan
        # Stage 3). Each measured material owns 12 flat Float32 arrays
        # (theta_i/phi_i/wavelengths + ndf/sigma/vndf/luminance/spectra
        # data+CDFs); each gets its own device buffer, mirroring the
        # per-mesh points_bufs / per-grid grid_density_bufs pattern. The
        # patched MeasuredBRDF struct array embeds device-resident
        # pointers into those buffers -- loaded from the array *inside*
        # the GPU kernel (a load, not a cross-call by-value pass) and
        # handed only to @always_inline helpers, per the by-value
        # pointer-struct hazard already documented on MeasuredBRDF.
        var measured_field_bufs = List[DeviceBuffer[DType.uint8]]()
        var n_measured_int = Int(s.measured_count)
        var measured_structs_host = unsafe_alloc[MeasuredBRDF](max(n_measured_int, 1))
        for mi in range(n_measured_int):
            var hm = s.measured_brdfs[unsafe_offset=mi]
            var slices2 = Int(hm.n_phi_i) * Int(hm.n_theta_i)
            var slices3 = slices2 * Int(hm.n_wavelengths)
            var vndf_n = slices2 * Int(hm.vndf_xs) * Int(hm.vndf_ys)
            var lum_n = slices2 * Int(hm.lum_xs) * Int(hm.lum_ys)
            var theta_i_dptr = _gpu_upload_owned[Float32](ctx, measured_field_bufs, hm.theta_i, Int(hm.n_theta_i))
            var phi_i_dptr = _gpu_upload_owned[Float32](ctx, measured_field_bufs, hm.phi_i, Int(hm.n_phi_i))
            var wavelengths_dptr = _gpu_upload_owned[Float32](ctx, measured_field_bufs, hm.wavelengths, Int(hm.n_wavelengths))
            var ndf_dptr = _gpu_upload_owned[Float32](ctx, measured_field_bufs, hm.ndf_data, Int(hm.ndf_xs) * Int(hm.ndf_ys))
            var sigma_dptr = _gpu_upload_owned[Float32](ctx, measured_field_bufs, hm.sigma_data, Int(hm.sigma_xs) * Int(hm.sigma_ys))
            var vndf_dptr = _gpu_upload_owned[Float32](ctx, measured_field_bufs, hm.vndf_data, vndf_n)
            var vndf_marg_dptr = _gpu_upload_owned[Float32](ctx, measured_field_bufs, hm.vndf_marg, slices2 * Int(hm.vndf_ys))
            var vndf_cond_dptr = _gpu_upload_owned[Float32](ctx, measured_field_bufs, hm.vndf_cond, vndf_n)
            var lum_dptr = _gpu_upload_owned[Float32](ctx, measured_field_bufs, hm.lum_data, lum_n)
            var lum_marg_dptr = _gpu_upload_owned[Float32](ctx, measured_field_bufs, hm.lum_marg, slices2 * Int(hm.lum_ys))
            var lum_cond_dptr = _gpu_upload_owned[Float32](ctx, measured_field_bufs, hm.lum_cond, lum_n)
            var spectra_dptr = _gpu_upload_owned[Float32](
                ctx, measured_field_bufs, hm.spectra_data, slices3 * Int(hm.spectra_xs) * Int(hm.spectra_ys))

            measured_structs_host[unsafe_offset=mi] = MeasuredBRDF(
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
        var measured_brdfs_buf = _gpu_upload_array[MeasuredBRDF](ctx, measured_structs_host, n_measured_int)
        ctx.synchronize()   # measured_structs_host is freed next
        measured_structs_host.unsafe_free()
        if n_measured_int > 0:
            print("GPU: " + String(n_measured_int) + " measured BRDF(s) uploaded")
        return Self(brdfs_buf=measured_brdfs_buf^, n_brdfs=n_measured_int, field_bufs=measured_field_bufs^)


# GPU scene handle — holds DeviceContext and device-resident scene buffers.
# Allocated on the heap, returned as an opaque pointer.
@fieldwise_init
struct GpuSceneHandle(Movable):
    var ctx: DeviceContext
    var bvh: BvhBuffers
    var blas: BlasBuffers
    var instances_buf: DeviceBuffer[DType.uint8]
    var n_instances: Int
    var meshes: MeshBuffers
    var materials_buf: DeviceBuffer[DType.uint8]
    var material_count: Int
    var has_glass_material: Bool
    # MNEE auto-off: [probes, probes that found glass] written by the diffuse kernel; gpu_mnee_adapt reads it after a batch.
    var mnee_stats_buf: DeviceBuffer[DType.int32]
    var mnee_adapt_done: Bool
    # one parked glass-chain emitter hit per path (SuppressTask), resolved by resolve_suppress_gpu; 32 bytes when unused
    var suppress_buf: DeviceBuffer[DType.uint8]
    # One bit per MatKind present in the scene's material table. The bounce
    # loop launches one kernel per kind; a kind the scene does not contain
    # would still cost a full-grid launch every round.
    var mat_kind_mask: Int32
    # A null `interface` material can sit in a scene with no medium at all
    # (glass stubbed out as "interface"); its crossings still need rounds.
    var has_interface_material: Bool
    # Set by each render driver for its own spp (footprint.camera_footprint).
    var cam_fp: CameraFootprint
    var textures: TextureBuffers
    var lights: LightBuffers
    var spheres_buf: DeviceBuffer[DType.uint8]   # n_spheres × sizeof(Sphere) = 36
    var n_spheres: Int
    var n_sphere_lights: Int
    var curves: CurveBuffers
    var media: MediaBuffers
    var measured: MeasuredBuffers
    # Persistent render buffers — sized for n_pixels × WAVEFRONT_BATCH (wavefront pass)
    # gpu_render_sample (interactive) only uses the first n_pixels slots.
    var path_buf: DeviceBuffer[DType.uint8]   # n_pixels × WAVEFRONT_BATCH × size_of[PathState]()
    var ptex_redo_buf: DeviceBuffer[DType.uint8]   # Ptex paging: one byte per path_buf slot, 1 = sample withheld
    var ptex_redo_si_buf: DeviceBuffer[DType.uint8]   # Ptex paging: one Int32 per path_buf slot, the sample to render again
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
    var rt_scratch_buf: DeviceBuffer[DType.uint8]   # --rt-hardware alpha passes: one 32-byte result per ray (n_pixels × WAVEFRONT_BATCH), else 32 bytes
    # --guide: the SD-tree the current training iteration reads and the shard it records into, in device memory
    # (gpu_guide.mojo). The GuideGrids hold device pointers into these buffers; null_guide() when guiding is off.
    var guide_read: GuideGrid
    var guide_write: GuideGrid
    var guide_bufs: List[DeviceBuffer[DType.uint8]]
    var filter_lut_buf: DeviceBuffer[DType.uint8]    # inverse Gaussian filter CDF per axis (sampling.fill_filter_lut); unused for other filters
    var shade_ctx_buf: DeviceBuffer[DType.uint8]    # one ShadeContext shared by all threads of a shade kernel (see gpu_shade.shared_shade_context)
    var shadow_slots: Int                           # ShadowTask slots per path in shadow_buf (SHADOW_SLOTS with --rt-hardware, else 1)
    var shadow_buf: DeviceBuffer[DType.uint8]       # n_pixels × WAVEFRONT_BATCH × sizeof(ShadowTask) = 48 -- must match path_buf/inter_buf sizing (gpu_render_sample only uses the first n_pixels slots; gpu_render_wavefront's _gpu_bounce_kernels call indexes up to n_pixels × WAVEFRONT_BATCH)
    var active_count_buf: DeviceBuffer[DType.uint8] # live count, then one count per MatKind (16 × Int32)
    var active_idx_buf: DeviceBuffer[DType.uint8]   # 4 lists of n_pixels × WAVEFRONT_BATCH Int32: live (even/odd round), RT shadow rays, paths of one material kind
    var n_pixels: Int
    # Camera and sampling data for GPU-side ray generation
    var sobol_buf: DeviceBuffer[DType.uint8]  # 1024 dims × 52 UInt32 = 212992 bytes
    var r2c_buf: DeviceBuffer[DType.uint8]    # raster_to_camera: 16 Float32 = 64 bytes
    var c2w_buf: DeviceBuffer[DType.uint8]    # camera_to_world: 16 Float32 = 64 bytes (updated each frame)
    var filter: FilterParams
    var film: FilmDims
    # Staged spectral rendering rollout (Stage 2c-1, see
    # project_spectral_rendering memory) — device-side twin of the host
    # SpectralHandle; spectral_res=0 means no real table was uploaded (dummy
    # 1-element buffers, BDPT/SPPM GPU dispatch, Stage 3/4 not wired yet).
    var spectral: SpectralBuffers

    def has_material(self, kind: Int8) -> Bool:
        return (self.mat_kind_mask >> Int32(kind)) & Int32(1) != Int32(0)

    def scene_descriptor(mut self) -> SceneView:
        """The whole device-resident scene as ONE kernel argument. Kernels
        take this by value instead of ~40 decomposed pointer/count params."""
        var (sc, sres, sx, sy, sz, sd65) = self.spectral.unsafe_ptrs()
        return SceneView(
            bvh2Nodes=self.bvh.nodes_ptr(), primIds=self.bvh.prim_ids_ptr(),
            meshes=self.meshes.meshes_ptr(), meshCount=Int64(self.meshes.mesh_count),
            materials=typed_ptr[Material](self.materials_buf), materialCount=Int64(self.material_count),
            areaLights=self.lights.area_lights_ptr(), areaLightCount=Int64(self.lights.n_area_lights),
            textures=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
            textureCount=Int64(0),
            distantLights=self.lights.distant_lights_ptr(), distantLightCount=Int64(self.lights.n_distant_lights),
            pointLights=self.lights.point_lights_ptr(), pointLightCount=Int64(self.lights.n_point_lights),
            infiniteLights=self.lights.infinite_lights_ptr(), infiniteLightCount=Int64(self.lights.n_infinite_lights),
            spheres=typed_ptr[Sphere](self.spheres_buf), sphereCount=Int64(self.n_spheres),
            curves=self.curves.curves_ptr(), curveCount=Int64(self.curves.n_curves),
            mediums=typed_ptr[Medium](self.media.mediums_buf), mediumCount=Int64(self.media.n_mediums),
            mediumInterfaces=typed_ptr[MediumInterface](self.media.medium_ifaces_buf), mediumIfaceCount=Int64(self.media.n_medium_ifaces),
            grids=typed_ptr[Grid](self.media.grids_buf), gridCount=Int64(self.media.n_grids),
            nvdbGrids=typed_ptr[NvdbGrid](self.media.nvdb_grids_buf), nvdbGridCount=Int64(self.media.n_nvdb_grids),
            lightSampler=LightSampler(cdf=self.lights.light_sampler_ptr(), n=Int32(self.lights.n_light_sampler), _pad=Int32(0)),
            blasNodesArr=self.blas.nodes_arr(), blasPrimIdsArr=self.blas.primids_arr(), blasCount=Int64(self.blas.n_blas),
            instances=typed_ptr[Instance](self.instances_buf), instanceCount=Int64(self.n_instances),
            measuredBrdfs=typed_ptr[MeasuredBRDF](self.measured.brdfs_buf), measuredBrdfCount=Int64(self.measured.n_brdfs),
            spectral=SpectralHandle(sc, sres, sx, sy, sz, sd65),
            gpuTextures=self.textures.textures_ptr(), gpuTextureCount=Int64(self.textures.n_textures),
            normalSlopeMaps=Pointer[NormalSlopeMap, MutUntrackedOrigin].unsafe_dangling(),
            vcmKeepCounts=Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
            vcmKeepInvCell=Float32(0), vcmKeepScale=Float32(1), vcmMaxDepth=Int32(9),
            vcmCamX=Float32(0), vcmCamY=Float32(0), vcmCamZ=Float32(0),
            vcmFootprint=Float32(0), vcmMergeR=Float32(0), camFp=self.cam_fp,
            sphereLightCount=Int64(self.n_sphere_lights),
            lightPickCdf=self.lights.light_pick_ptr(),
            vcmStatIn=Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
            vcmStatOut=Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
            vcmLambda=Float32(0),
            vcmBucketCap=Int32(0),
            hasGlass=Int32(1) if self.has_glass_material else Int32(0),
            mneeStats=Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
            bvh4=BVH4.none(),
        )

def gpu_available() -> Bool:
    return has_accelerator()

# Uploads count elements of T from a host array into a fresh device buffer
# (>= 1 elem so a zero-count scene never creates a 0-byte device buffer,
# which crashes on use/free). enqueue_copy copies straight from the host
# pointer; map_to_host would first copy the device buffer to host pages and
# back on exit, which cost San Miguel ~1.2 s of page faults across its
# per-mesh uploads. The copy is asynchronous: `src` must stay valid until the
# next ctx.synchronize().
def _gpu_upload_array[T: AnyType](
    ctx: DeviceContext,
    src: Pointer[T, MutUntrackedOrigin],
    count: Int,
) raises -> DeviceBuffer[DType.uint8]:
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(count, 1) * size_of[T]())
    if count > 0:
        ctx.enqueue_copy(buf, src.unsafe_bitcast[UInt8]())
    return buf^

# _gpu_upload_array into a buffer that `bufs` keeps alive; returns its device
# pointer typed as T.
def _gpu_upload_owned[T: AnyType](
    ctx: DeviceContext,
    mut bufs: List[DeviceBuffer[DType.uint8]],
    src: Pointer[T, MutUntrackedOrigin],
    count: Int,
) raises -> Pointer[T, MutUntrackedOrigin]:
    var buf = _gpu_upload_array[T](ctx, src, count)
    var dptr = typed_ptr[T](buf)
    bufs.append(buf^)
    return dptr

# A zero-filled device buffer of `count` elements of T that `bufs` keeps alive;
# returns its device pointer typed as T.
def _gpu_zeros_owned[T: AnyType](
    ctx: DeviceContext,
    mut bufs: List[DeviceBuffer[DType.uint8]],
    count: Int,
) raises -> Pointer[T, MutUntrackedOrigin]:
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(count, 1) * size_of[T]())
    ctx.enqueue_memset(buf, UInt8(0))
    var dptr = typed_ptr[T](buf)
    bufs.append(buf^)
    return dptr

def _report_gpu_memory(ctx: DeviceContext, ref s: ParsedScene_Mojo) raises:
    # Check GPU memory
    var mem_info = ctx.get_memory_info()
    var free_bytes = mem_info[0]

    # Guard against zero-size device buffers (scene with no geometry):
    # a 0-byte enqueue_create_buffer yields a misaligned/invalid device
    # pointer that crashes on use and on free. Allocate at least 1 elem.
    var bvh_bytes = max(Int(s.bvh_node_count_cpu), 1) * size_of[BVH2Node]()
    var prim_bytes = max(Int(s.prim_count_cpu), 1) * size_of[PrimId]()
    var mesh_struct_bytes = max(Int(s.mesh_count), 1) * size_of[TriangleMesh]()

    # Estimate total mesh data
    var mesh_data_bytes = 0
    for i in range(Int(s.mesh_count)):
        mesh_data_bytes += Int(s.mesh_n_verts[unsafe_offset=i]) * 4 * 4       # Float32
        mesh_data_bytes += Int(s.mesh_n_tris[unsafe_offset=i]) * 8  # Int64
        mesh_data_bytes += Int(s.mesh_n_tris[unsafe_offset=i]) * 3 * 8 # Int64

    var total_scene_bytes = bvh_bytes + prim_bytes + mesh_struct_bytes + mesh_data_bytes
    var free_mb = free_bytes // (1024 * 1024)
    var scene_mb = total_scene_bytes // (1024 * 1024)

    print("GPU: " + String(ctx.name()) + " — " + String(free_mb) + " MB free")

    if total_scene_bytes > Int(free_bytes):
        print("WARNING: Scene (" + String(scene_mb) + " MB) may exceed available GPU memory (" + String(free_mb) + " MB)!")


def gpu_upload_scene(
    psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    n_pixels: Int,
    # Decomposed rather than one by-value SpectralHandle: passing it by value
    # here once read spectral_res back as 0 and coeffs as a garbage address.
    # Never reproduced since (modular#6759 was closed unreproducible), but the
    # decomposed form costs nothing.
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    ctx_in: Optional[DeviceContext] = None,   # a context created ahead of the scene parse
) -> Pointer[GpuSceneHandle, MutUntrackedOrigin]:
    comptime if has_accelerator():
        try:
            var ctx = ctx_in.value().copy() if ctx_in else DeviceContext()
            ref s = psc[]
            _report_gpu_memory(ctx, s)
            var bvh = BvhBuffers.upload(ctx, s)
            var blas = BlasBuffers.upload(ctx, s)
            var instances_gpu_buf = _gpu_upload_array[Instance](ctx, s.instances, Int(s.instance_count))
            var meshes = MeshBuffers.upload(ctx, s)
            # >= 1 elem to avoid a zero-size buffer
            var mat_buf = _gpu_upload_array[Material](ctx, s.materials, Int(s.material_count))
            var has_iface_mat = False
            var has_glass_mat = False
            var kind_mask = Int32(0)
            for mi in range(Int(s.material_count)):
                kind_mask |= Int32(1) << Int32(s.materials[unsafe_offset=mi].type)
                if s.materials[unsafe_offset=mi].type == MatKind.interface:
                    has_iface_mat = True
                if is_specular_glass(s.materials[unsafe_offset=mi]):
                    has_glass_mat = True
            var lights = LightBuffers.upload(ctx, s)
            # analytical sphere primitives + sphere area lights
            var sphere_buf = _gpu_upload_array[Sphere](ctx, s.spheres, Int(s.sphere_count))
            var n_sphere_lights = 0
            for si in range(Int(s.sphere_count)):
                if s.spheres[unsafe_offset=si].isAreaLight != Int8(0):
                    n_sphere_lights += 1
            var media = MediaBuffers.upload(ctx, s)
            var measured = MeasuredBuffers.upload(ctx, s)

            # Allocate persistent render buffers (zeroed film)
            var n_pix = max(Int(n_pixels), 1)
            var r_suppress = ctx.enqueue_create_buffer[DType.uint8](n_pix * 32 * WAVEFRONT_BATCH if has_glass_mat else 32)
            ctx.enqueue_memset(r_suppress, UInt8(0))
            var r_mnee_stats = ctx.enqueue_create_buffer[DType.int32](2)
            ctx.enqueue_memset(r_mnee_stats, Int32(0))
            var r_path_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[PathState]() * WAVEFRONT_BATCH)
            var r_inter_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[Intersection]() * WAVEFRONT_BATCH)
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
            # --rt-hardware defers up to SHADOW_SLOTS shadow rays per path; everything else needs one slot per path.
            # The software BVH defers too (gpu.mojo sw_shadow) when every NEE candidate of a bounce has a slot and there is no medium.
            var nee_candidates = (1 if s.area_light_count > 0 else 0) + Int(s.distant_count) + Int(s.point_count) + Int(s.sphere_count) + Int(s.infinite_count)
            var sw_defer = Int(s.medium_count) == 0 and nee_candidates <= SHADOW_SLOTS and getenv("GONZALES_NO_SW_SHADOW") == ""
            var r_shadow_slots = SHADOW_SLOTS if getenv("GONZALES_RTCORE") != "" or sw_defer else 1
            var r_rt_scratch_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * 32 * WAVEFRONT_BATCH if getenv("GONZALES_RTCORE") != "" else 32)
            var r_shadow_buf = ctx.enqueue_create_buffer[DType.uint8](n_pix * size_of[ShadowTask]() * WAVEFRONT_BATCH * r_shadow_slots)
            ctx.enqueue_memset(r_shadow_buf, UInt8(0))      # every slot free (active == 0)
            var r_shade_ctx_buf = ctx.enqueue_create_buffer[DType.uint8](1024)
            var r_filter_lut_buf = ctx.enqueue_create_buffer[DType.uint8](2 * FILTER_LUT_N * 4)
            if s.filter_type == Int32(0):
                var lut_host = unsafe_alloc[Float32](2 * FILTER_LUT_N)
                fill_filter_lut(lut_host, s.filter_sigma, s.filter_support_x, s.filter_support_y)
                ctx.enqueue_copy(r_filter_lut_buf, lut_host.bitcast[UInt8]())
                ctx.synchronize()
                lut_host.unsafe_free()
            var r_active_count_buf = ctx.enqueue_create_buffer[DType.uint8](64)
            var r_active_idx_buf   = ctx.enqueue_create_buffer[DType.uint8](n_pix * 4 * WAVEFRONT_BATCH * 4)
            var curves = CurveBuffers.upload(ctx, s, n_pix)
            ctx.enqueue_memset(r_film_buf, UInt8(0))
            ctx.enqueue_memset(r_albedo_film_buf, UInt8(0))
            var textures = TextureBuffers.upload(ctx, s)
            var r_ptex_redo_buf = ctx.enqueue_create_buffer[DType.uint8](
                n_pix * WAVEFRONT_BATCH if len(textures.ptex_bufs) > 0 else 1)
            var r_ptex_redo_si_buf = ctx.enqueue_create_buffer[DType.uint8](
                n_pix * WAVEFRONT_BATCH * 4 if len(textures.ptex_bufs) > 0 else 4)

            # Upload Sobol matrices: first 1024 dimensions × 52 UInt32 = 212992 bytes
            comptime N_SOBOL_GPU_DIMS = 1024
            comptime N_SOBOL_GPU_WORDS = N_SOBOL_GPU_DIMS * 52
            var sobol_gpu_buf = _gpu_upload_array[UInt32](ctx, sobol_matrices, N_SOBOL_GPU_WORDS)

            # raster_to_camera and camera_to_world (16 floats each)
            var r2c_gpu_buf = _gpu_upload_array[Float32](ctx, s.raster_to_camera, 16)
            var c2w_gpu_buf = _gpu_upload_array[Float32](ctx, s.camera_to_world, 16)
            var spectral = SpectralBuffers.upload(ctx, spectral_coeffs, spectral_res,
                spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)

            # Every upload is asynchronous: finish them while the caller's host
            # arrays are still alive.
            ctx.synchronize()

            # Allocate handle on heap
            var handle = unsafe_alloc[GpuSceneHandle](1)
            handle.unsafe_write(GpuSceneHandle(
                ctx=ctx^,
                bvh=bvh^,
                blas=blas^,
                instances_buf=instances_gpu_buf^,
                n_instances=Int(s.instance_count),
                meshes=meshes^,
                materials_buf=mat_buf^,
                material_count=Int(s.material_count),
                has_glass_material=has_glass_mat,
                mnee_stats_buf=r_mnee_stats^,
                mnee_adapt_done=False,
                suppress_buf=r_suppress^,
                mat_kind_mask=kind_mask,
                has_interface_material=has_iface_mat,
                cam_fp=CameraFootprint.none(),
                textures=textures^,
                lights=lights^,
                spheres_buf=sphere_buf^,
                n_spheres=Int(s.sphere_count),
                n_sphere_lights=n_sphere_lights,
                curves=curves^,
                media=media^,
                measured=measured^,
                path_buf=r_path_buf^,
                ptex_redo_buf=r_ptex_redo_buf^,
                ptex_redo_si_buf=r_ptex_redo_si_buf^,
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
                shadow_slots=r_shadow_slots,
                filter_lut_buf=r_filter_lut_buf^,
                shade_ctx_buf=r_shade_ctx_buf^,
                rt_scratch_buf=r_rt_scratch_buf^,
                guide_read=null_guide(), guide_write=null_guide(), guide_bufs=List[DeviceBuffer[DType.uint8]](),
                active_count_buf=r_active_count_buf^,
                active_idx_buf=r_active_idx_buf^,
                n_pixels=n_pix,
                sobol_buf=sobol_gpu_buf^,
                r2c_buf=r2c_gpu_buf^,
                c2w_buf=c2w_gpu_buf^,
                filter=FilterParams(s.filter_sigma, s.filter_support_x, s.filter_support_y,
                                    s.filter_norm_x, s.filter_norm_y, s.filter_type),
                film=FilmDims(s.film_w, s.film_h),
                spectral=spectral^,
            ))

            print("GPU: scene uploaded")
            return handle.unsafe_bitcast[GpuSceneHandle]()
        except e:
            var msg = String(e)
            if "libnvidia" in msg or "nvidia-ml" in msg:
                print("GPU: no supported GPU driver found (requires NVIDIA or AMD)")
            else:
                print("GPU: Failed to upload scene: " + msg)
            return Pointer[GpuSceneHandle, MutUntrackedOrigin].unsafe_dangling()
    else:
        return Pointer[GpuSceneHandle, MutUntrackedOrigin].unsafe_dangling()



def init_curve_cand_offset_gpu(offset_buf: Pointer[Int32, MutUntrackedOrigin], n_dp: Int64):
    var n = Int(n_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n:
        return
    offset_buf[unsafe_offset=tid] = Int32(tid * CURVE_DEFER_K)

# ── Ptex demand paging (ptex_cache.mojo) ──
def _ptex_upload_tables(handle: Pointer[GpuSceneHandle, MutUntrackedOrigin], scratch: Pointer[UInt8, MutUntrackedOrigin]) raises:
    for k in range(len(handle[].textures.ptex_bufs)):
        if handle[].textures.ptex_cache.take_table(k, scratch):
            var sub = handle[].textures.tex_data_bufs[handle[].textures.ptex_bufs[k]].create_sub_buffer[DType.uint8](
                0, handle[].textures.ptex_cache.n_faces[k] * 8)
            handle[].ctx.enqueue_copy(sub, scratch)
            handle[].ctx.synchronize()

def _ptex_scratch(handle: Pointer[GpuSceneHandle, MutUntrackedOrigin]) -> Pointer[UInt8, MutUntrackedOrigin]:
    var max_faces = 1
    for k in range(len(handle[].textures.ptex_bufs)): max_faces = max(max_faces, handle[].textures.ptex_cache.n_faces[k])
    return unsafe_alloc[UInt8](max_faces * 8).unsafe_origin_cast[MutUntrackedOrigin]()

def gpu_ptex_begin(handle: Pointer[GpuSceneHandle, MutUntrackedOrigin]) raises -> Int:
    """Read back what the last samples asked for and start loading it in the background.
    Returns the number of faces being loaded."""
    var scratch = _ptex_scratch(handle)
    handle[].ctx.synchronize()
    for k in range(len(handle[].textures.ptex_bufs)):
        var sub = handle[].textures.tex_data_bufs[handle[].textures.ptex_bufs[k]].create_sub_buffer[DType.uint8](
            0, handle[].textures.ptex_cache.n_faces[k] * 8)
        handle[].ctx.enqueue_copy(scratch, sub)
        handle[].ctx.synchronize()
        handle[].textures.ptex_cache.feed(k, scratch)
    var n = handle[].textures.ptex_cache.begin()
    # The wanted bytes are reset now, so a face is asked for once per load.
    _ptex_upload_tables(handle, scratch)
    scratch.unsafe_free()
    return n

def gpu_ptex_finish(handle: Pointer[GpuSceneHandle, MutUntrackedOrigin], may_evict: Bool = False) raises -> Int:
    """Wait for the load in flight and put it on the device. Returns the number of faces loaded."""
    var (n, lo, hi) = handle[].textures.ptex_cache.finish(may_evict)
    handle[].ctx.synchronize()
    if hi > lo:
        var psub = handle[].textures.ptex_pool_buf.create_sub_buffer[DType.uint8](lo, hi - lo)
        handle[].ctx.enqueue_copy(psub, handle[].textures.ptex_cache.pool + lo)
        handle[].ctx.synchronize()
    var scratch = _ptex_scratch(handle)
    _ptex_upload_tables(handle, scratch)
    scratch.unsafe_free()
    return n

def gpu_ptex_withhold(handle: Pointer[GpuSceneHandle, MutUntrackedOrigin], on: Bool) raises:
    """Switch the withholding of samples that meet a face not loaded yet (texel 0 of the pool)."""
    handle[].textures.ptex_cache.pool[unsafe_offset=0] = UInt8(1) if on else UInt8(0)
    handle[].ctx.synchronize()
    var psub = handle[].textures.ptex_pool_buf.create_sub_buffer[DType.uint8](0, 3)
    handle[].ctx.enqueue_copy(psub, handle[].textures.ptex_cache.pool)
    handle[].ctx.synchronize()

def gpu_ptex_withheld(handle: Pointer[GpuSceneHandle, MutUntrackedOrigin], n_pixels: Int, batch: Int,
                      scratch: Pointer[UInt8, MutUntrackedOrigin], si_start: Int, si_of: Pointer[Int32, MutUntrackedOrigin],
                      mut px_out: List[Int32], mut si_out: List[Int32]) raises:
    """Append the (pixel, sample) pairs the last pass withheld. A slot's sample is si_start + its row,
    or si_of[slot] when si_start < 0 (a pass that rendered withheld samples again)."""
    var n = n_pixels * batch
    handle[].ctx.synchronize()
    var sub = handle[].ptex_redo_buf.create_sub_buffer[DType.uint8](0, n)
    handle[].ctx.enqueue_copy(scratch, sub)
    handle[].ctx.synchronize()
    var words = scratch.unsafe_bitcast[UInt64]()
    for w in range((n + 7) // 8):
        if words[unsafe_offset=w] == UInt64(0):
            continue
        for i in range(w * 8, min(w * 8 + 8, n)):
            if scratch[unsafe_offset=i] != UInt8(0):
                px_out.append(Int32(i % n_pixels))
                si_out.append(Int32(si_start + i // n_pixels) if si_start >= 0 else si_of[unsafe_offset=i])

def gpu_ptex_redo_upload(handle: Pointer[GpuSceneHandle, MutUntrackedOrigin], n: Int, si_of: Pointer[Int32, MutUntrackedOrigin]) raises:
    """Set the samples the next ptex_mode 2 pass renders: si_of[slot], < 0 for none."""
    var sub = handle[].ptex_redo_si_buf.create_sub_buffer[DType.uint8](0, n * 4)
    handle[].ctx.enqueue_copy(sub, si_of.unsafe_bitcast[UInt8]())
    handle[].ctx.synchronize()

def gpu_free_scene(handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin]):
    if Int(handlePtr) == 0:
        return
    handlePtr.unsafe_deinit_pointee()
    handlePtr.unsafe_bitcast[GpuSceneHandle]().unsafe_free()
