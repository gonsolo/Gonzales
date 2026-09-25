from std.collections import Array
from std.sys import has_accelerator, has_nvidia_gpu_accelerator
from std.sys.info import size_of, num_performance_cores
from max.gpu import block_idx, thread_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from max.algorithm import parallelize
from std.atomic import Atomic
from std.math import ceildiv, sqrt, cos, sin, log, exp
from std.memory.alloc import unsafe_alloc
from std.memory import unsafe_memcpy
from .geometry import RGB, Point3f, Point2f, Vec3f, vec3f, point3f, store_vec3, Material_C, MatKind, MeasuredBRDF_C, dot, cross, INV_PI, INV_FOUR_PI, _is_real_ptr, TERMINAL_SEGMENT_GRACE_ROUNDS
from .render_state import FilmDims, FilterParams, PathState_C, GpuTexture_C, NormalSlopeMap_C, ShadowTask_C
from .primitives import sphere_outward_normal, Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Sphere_C, Instance_C
from .media import Medium_C, MediumInterface_C, Grid_C, grid_sample_density, NvdbGrid_C, nvdb_sample_density, nvdb_ray_range, grid_ray_range, nvdb_index_ray, nvdb_node_exit_t, nvdb_majorant_at_world, hg_phase, hg_sample, blackbody_rgb, FreeFlight, sample_homogeneous_free_flight, sample_free_flight, medium_is_heterogeneous, medium_grid_for, medium_nvdb_for, medium_emission_spectral, MEDIUM_TRACK_MAX_ITERS, medium_transmittance_ratio_spectral, medium_sigma_s_spectral, medium_sigma_t_spectral
from .lights import AreaLight_C, DistantLight_C, PointLight_C, InfiniteLight_C, LightSampler_C, light_sampler_sample, area_light_pick_triangle
from .curves import Curve_C, CURVE_N_PIECES, CURVE_DEFER_K, curve_piece_endpoints, _curve_perp_axis, intersect_curve
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
from .sampling import power_heuristic, encode_morton2, sobol_get_sample_index, sobol_sample, derive_pcg_seeds, gen_primary_ray_state
from .spectrum import SampledWavelengths, SpectralSample, SpectralHandle, null_spectral_handle, rgb_illuminant_to_spectral_sample, rgb_bands_to_spectral_sample, spectral_sample_to_rgb, spec_refl_unbounded
from .vulkaninterop import VulkanInteropRtSceneHandle, vulkaninterop_rt_trace
from max.gpu.host._nvidia_cuda import CUDA

# Number of samples per pixel processed together in one wavefront bounce loop.
# path_buf and inter_buf are pre-allocated at n_pixels × WAVEFRONT_BATCH.
comptime WAVEFRONT_BATCH: Int = 8

def _cstr_eq(a: Pointer[UInt8, MutUntrackedOrigin], b: Pointer[UInt8, MutUntrackedOrigin]) -> Bool:
    var i = 0
    while True:
        var ca = a[unsafe_offset=i]
        var cb = b[unsafe_offset=i]
        if ca != cb:
            return False
        if ca == UInt8(0):
            return True
        i += 1

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

# (levels, total texels) of a full mip pyramid down to 1x1.
def _mip_texel_count(tw: Int, th: Int) -> Tuple[Int, Int]:
    var nlev = 1; var texels = tw * th
    var ww = tw; var hh = th
    while ww > 1 or hh > 1:
        ww = max(1, ww // 2); hh = max(1, hh // 2)
        nlev += 1; texels += ww * hh
    return (nlev, texels)

# The byte whose decoded value in `lut` (256 entries, non-decreasing) is nearest `v`.
@always_inline
def _nearest_lut_byte(lut: Pointer[Float32, MutUntrackedOrigin], v: Float32) -> UInt8:
    var lo = 0; var hi = 255
    while lo < hi:
        var mid = (lo + hi) // 2
        if lut[unsafe_offset=mid] < v:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0 and v - lut[unsafe_offset=lo - 1] <= lut[unsafe_offset=lo] - v:
        return UInt8(lo - 1)
    return UInt8(lo)

comptime _INV_LUT_SIZE: Int = 65536

@always_inline
def _dist(a: Float32, b: Float32) -> Float32:
    return a - b if a > b else b - a

# inv[q] = _nearest_lut_byte(lut, q / (_INV_LUT_SIZE - 1)): a candidate byte for
# each of _INV_LUT_SIZE evenly spaced values in [0, 1].
def _build_inverse_lut(lut: Pointer[Float32, MutUntrackedOrigin], inv: Pointer[UInt8, MutUntrackedOrigin]):
    for q in range(_INV_LUT_SIZE):
        inv[unsafe_offset=q] = _nearest_lut_byte(lut, Float32(q) / Float32(_INV_LUT_SIZE - 1))

# Same byte as _nearest_lut_byte(lut, v), in O(1). The grid is far finer than
# the LUT spacing, so the candidate is the nearest byte or a neighbour of it;
# |lut[b] - v| is unimodal in b, so walking up while strictly closer and down
# while no farther lands on the nearest byte, ties going to the lower one.
@always_inline
def _quantize_to_lut_byte(
    lut: Pointer[Float32, MutUntrackedOrigin], inv: Pointer[UInt8, MutUntrackedOrigin], v: Float32,
) -> UInt8:
    var q = Int(v * Float32(_INV_LUT_SIZE - 1) + Float32(0.5))
    if q < 0: q = 0
    if q > _INV_LUT_SIZE - 1: q = _INV_LUT_SIZE - 1
    var b = Int(inv[unsafe_offset=q])
    while b < 255 and _dist(lut[unsafe_offset=b + 1], v) < _dist(lut[unsafe_offset=b], v):
        b += 1
    while b > 0 and _dist(lut[unsafe_offset=b - 1], v) <= _dist(lut[unsafe_offset=b], v):
        b -= 1
    return UInt8(b)

# Fill `pyr` with a uint8 mip pyramid of `src` (tw x th, c channels). Each coarser
# level is the 2x2 box average in LINEAR space (bytes decoded through `lut`) --
# the same averages the float path computes -- stored as the nearest byte (via
# `inv`, the matching _build_inverse_lut table). The averages are carried in
# float between levels so rounding doesn't compound.
def _fill_u8_mips(
    pyr: Pointer[UInt8, MutUntrackedOrigin], src: Pointer[UInt8, MutUntrackedOrigin],
    tw: Int, th: Int, c: Int, lut: Pointer[Float32, MutUntrackedOrigin],
    inv: Pointer[UInt8, MutUntrackedOrigin],
):
    unsafe_memcpy(dest=pyr, src=src, count=tw * th * c)
    var prev = unsafe_alloc[Float32](tw * th * c)
    for i in range(tw * th * c):
        prev[unsafe_offset=i] = lut[unsafe_offset=Int(src[unsafe_offset=i])]
    var cur = unsafe_alloc[Float32](max(1, tw // 2) * max(1, th // 2) * c)
    var off_cur = tw * th * c
    var pw = tw; var ph = th
    while pw > 1 or ph > 1:
        var cw = max(1, pw // 2); var ch = max(1, ph // 2)
        for y in range(ch):
            for x in range(cw):
                var x0 = 2 * x; var x1 = min(2 * x + 1, pw - 1)
                var y0 = 2 * y; var y1 = min(2 * y + 1, ph - 1)
                for k in range(c):
                    var avg = (prev[unsafe_offset=(y0 * pw + x0) * c + k] + prev[unsafe_offset=(y0 * pw + x1) * c + k]
                               + prev[unsafe_offset=(y1 * pw + x0) * c + k] + prev[unsafe_offset=(y1 * pw + x1) * c + k]) * Float32(0.25)
                    cur[unsafe_offset=(y * cw + x) * c + k] = avg
                    pyr[unsafe_offset=off_cur + (y * cw + x) * c + k] = _quantize_to_lut_byte(lut, inv, avg)
        off_cur += cw * ch * c
        var tmp = prev; prev = cur; cur = tmp
        pw = cw; ph = ch
    prev.unsafe_free(); cur.unsafe_free()

# Fill `pyr` with a Float32 RGB mip pyramid of `src` (tw x th, linear RGB): level 0
# copied, each coarser level the 2x2 box average of the one before -- the float
# twin of _fill_u8_mips.
def _fill_f32_mips(
    pyr: Pointer[Float32, MutUntrackedOrigin], src: Pointer[Float32, MutUntrackedOrigin],
    tw: Int, th: Int,
):
    unsafe_memcpy(dest=pyr, src=src, count=tw * th * 3)
    var off_prev = 0; var off_cur = tw * th * 3
    var pw = tw; var ph = th
    while pw > 1 or ph > 1:
        var cw = max(1, pw // 2); var ch = max(1, ph // 2)
        for y in range(ch):
            for x in range(cw):
                var x0 = 2 * x; var x1 = min(2 * x + 1, pw - 1)
                var y0 = 2 * y; var y1 = min(2 * y + 1, ph - 1)
                for k in range(3):
                    var a = pyr[unsafe_offset=off_prev + (y0 * pw + x0) * 3 + k]
                    var b = pyr[unsafe_offset=off_prev + (y0 * pw + x1) * 3 + k]
                    var cc = pyr[unsafe_offset=off_prev + (y1 * pw + x0) * 3 + k]
                    var d = pyr[unsafe_offset=off_prev + (y1 * pw + x1) * 3 + k]
                    pyr[unsafe_offset=off_cur + (y * cw + x) * 3 + k] = (a + b + cc + d) * Float32(0.25)
        off_prev = off_cur; off_cur += cw * ch * 3
        pw = cw; ph = ch

@fieldwise_init
struct _HostTexture(TrivialRegisterPassable):
    """One texture's mip pyramid, decoded on the host and ready to upload:
    `n_bytes` bytes at `data` -- UInt8 texels for FORMAT_U8 (decoded through
    the table at `lut_off`), Float32 linear RGB for FORMAT_F32. n_bytes == 0
    means the file didn't load."""
    var data: Pointer[UInt8, MutUntrackedOrigin]
    var n_bytes: Int
    var width: Int32
    var height: Int32
    var n_levels: Int32
    var channels: Int32
    var format: Int32
    var lut_off: Int32

# Decode `filename` and build its full mip pyramid on the host: 8-bit files stay
# 8-bit (level 0 through load_texture_u8), everything else becomes Float32 RGB.
# `lut` holds both 256-entry decode tables (linear at 0, sRGB at 256) and `inv`
# their _build_inverse_lut tables (at 0 and _INV_LUT_SIZE). Reads the tables and
# touches only its own allocations, so it is safe to run on worker threads.
def _load_host_texture(
    filename: Pointer[UInt8, MutUntrackedOrigin], raw_flag: Int32,
    lut: Pointer[Float32, MutUntrackedOrigin], inv: Pointer[UInt8, MutUntrackedOrigin],
) -> _HostTexture:
    var result = _HostTexture(Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(), 0,
                              Int32(0), Int32(0), Int32(0), Int32(0), Int32(GpuTexture_C.FORMAT_F32), Int32(0))
    var w_out = unsafe_alloc[Int32](1); var h_out = unsafe_alloc[Int32](1)
    var c_out = unsafe_alloc[Int32](1); var srgb_out = unsafe_alloc[Int32](1)
    w_out[unsafe_offset=0] = Int32(0); h_out[unsafe_offset=0] = Int32(0)
    var u8_out = unsafe_alloc[Pointer[UInt8, MutUntrackedOrigin]](1)
    var ok_u8 = external_call["load_texture_u8", Int32,
        Pointer[UInt8, MutUntrackedOrigin], Int32,
        Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
        Pointer[Int32, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin],
        Pointer[Int32, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin]](
        filename, raw_flag, u8_out, w_out, h_out, c_out, srgb_out)
    if ok_u8 != 0 and Int(w_out[unsafe_offset=0]) > 0:
        var tw = Int(w_out[unsafe_offset=0]); var th = Int(h_out[unsafe_offset=0]); var c = Int(c_out[unsafe_offset=0])
        var lut_off = 256 if srgb_out[unsafe_offset=0] != Int32(0) else 0
        var (nlev, texels) = _mip_texel_count(tw, th)
        var pyr = unsafe_alloc[UInt8](texels * c)
        _fill_u8_mips(pyr, u8_out[unsafe_offset=0], tw, th, c, lut.unsafe_offset(lut_off),
                      inv.unsafe_offset((_INV_LUT_SIZE if lut_off != 0 else 0)))
        result = _HostTexture(pyr.unsafe_origin_cast[MutUntrackedOrigin](), texels * c, Int32(tw), Int32(th),
                              Int32(nlev), Int32(c), Int32(GpuTexture_C.FORMAT_U8), Int32(lut_off))
        _ = external_call["free_texture_u8", Int32, Pointer[UInt8, MutUntrackedOrigin]](u8_out[unsafe_offset=0])
    else:
        if ok_u8 != 0:
            _ = external_call["free_texture_u8", Int32, Pointer[UInt8, MutUntrackedOrigin]](u8_out[unsafe_offset=0])
        var data_out = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](1)
        w_out[unsafe_offset=0] = Int32(0); h_out[unsafe_offset=0] = Int32(0)
        var ok = external_call["load_texture_rgb", Int32,
            Pointer[UInt8, MutUntrackedOrigin],
            Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin],
            Pointer[Int32, MutUntrackedOrigin],
            Pointer[Int32, MutUntrackedOrigin],
            Int32](filename, data_out, w_out, h_out, raw_flag)
        if ok != 0 and Int(w_out[unsafe_offset=0]) > 0:
            var tw = Int(w_out[unsafe_offset=0]); var th = Int(h_out[unsafe_offset=0])
            var (nlev, texels) = _mip_texel_count(tw, th)
            var pyr = unsafe_alloc[Float32](texels * 3)
            _fill_f32_mips(pyr, data_out[unsafe_offset=0], tw, th)
            result = _HostTexture(pyr.unsafe_bitcast[UInt8]().unsafe_origin_cast[MutUntrackedOrigin](), texels * 3 * 4,
                                  Int32(tw), Int32(th), Int32(nlev), Int32(3), Int32(GpuTexture_C.FORMAT_F32), Int32(0))
            _ = external_call["free_texture_rgb", Int32, Pointer[Float32, MutUntrackedOrigin]](data_out[unsafe_offset=0])
        data_out.unsafe_free()
    w_out.unsafe_free(); h_out.unsafe_free(); c_out.unsafe_free(); srgb_out.unsafe_free(); u8_out.unsafe_free()
    if result.n_bytes == 0:
        # A missing file is already reported at parse time (parse_types.mojo,
        # scene_path); this is the one that exists and still will not decode.
        # It used to be counted in "N unique file(s) loaded" and then render
        # as the material's or light's flat default without a word.
        print("Warning: could not decode texture '" + String(unsafe_from_utf8_ptr=filename.as_imm())
              + "' -- it renders as a flat default instead.")
    return result

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

@fieldwise_init
struct BvhBuffers(Movable):
    var nodes_buf: DeviceBuffer[DType.uint8]
    var prim_ids_buf: DeviceBuffer[DType.uint8]

    @always_inline
    def nodes_ptr(mut self) -> Pointer[BVH2Node, MutUntrackedOrigin]:
        return typed_ptr[BVH2Node](self.nodes_buf)

    @always_inline
    def prim_ids_ptr(mut self) -> Pointer[PrimId_C, MutUntrackedOrigin]:
        return typed_ptr[PrimId_C](self.prim_ids_buf)

@fieldwise_init
struct BlasBuffers(Movable):
    """Object instancing (see [[project_object_instancing]]/geometry.mojo's
    Instance_C docs): one device buffer per BLAS (kept alive here), plus two
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
    def primids_arr(mut self) -> Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin]:
        return typed_ptr[Pointer[PrimId_C, MutUntrackedOrigin]](self.primids_ptrs_buf)

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

    @always_inline
    def meshes_ptr(mut self) -> Pointer[TriangleMesh_C, MutUntrackedOrigin]:
        return typed_ptr[TriangleMesh_C](self.meshes_buf)

@fieldwise_init
struct TextureBuffers(Movable):
    var tex_data_bufs: List[DeviceBuffer[DType.uint8]]
    var textures_buf: DeviceBuffer[DType.uint8]  # array of GpuTexture_C
    var lut_buf: DeviceBuffer[DType.float32]     # uint8 decode tables: linear at 0, sRGB at 256
    var n_textures: Int

    @always_inline
    def textures_ptr(mut self) -> Pointer[GpuTexture_C, MutUntrackedOrigin]:
        return typed_ptr[GpuTexture_C](self.textures_buf)

@fieldwise_init
struct LightBuffers(Movable):
    var area_lights_buf: DeviceBuffer[DType.uint8]  # n_lights × sizeof(AreaLight_C)
    var n_area_lights: Int
    var area_light_cdf_bufs: List[DeviceBuffer[DType.uint8]]   # each mesh light's tri_cdf
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

    @always_inline
    def area_lights_ptr(mut self) -> Pointer[AreaLight_C, MutUntrackedOrigin]:
        return typed_ptr[AreaLight_C](self.area_lights_buf)

    @always_inline
    def distant_lights_ptr(mut self) -> Pointer[DistantLight_C, MutUntrackedOrigin]:
        return typed_ptr[DistantLight_C](self.distant_lights_buf)

    @always_inline
    def point_lights_ptr(mut self) -> Pointer[PointLight_C, MutUntrackedOrigin]:
        return typed_ptr[PointLight_C](self.point_lights_buf)

    @always_inline
    def light_sampler_ptr(mut self) -> Pointer[Float32, MutUntrackedOrigin]:
        return typed_ptr[Float32](self.light_sampler_buf)

    @always_inline
    def infinite_lights_ptr(mut self) -> Pointer[InfiniteLight_C, MutUntrackedOrigin]:
        return typed_ptr[InfiniteLight_C](self.infinite_lights_buf)

@fieldwise_init
struct CurveBuffers(Movable):
    var curves_buf: DeviceBuffer[DType.uint8]    # n_curves × sizeof(Curve_C)
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
    def curves_ptr(mut self) -> Pointer[Curve_C, MutUntrackedOrigin]:
        return typed_ptr[Curve_C](self.curves_buf)

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
    var textures: TextureBuffers
    var lights: LightBuffers
    var spheres_buf: DeviceBuffer[DType.uint8]   # n_spheres × sizeof(Sphere_C) = 36
    var n_spheres: Int
    var curves: CurveBuffers
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
    var filter: FilterParams
    var film: FilmDims
    # Staged spectral rendering rollout (Stage 2c-1, see
    # project_spectral_rendering memory) — device-side twin of the host
    # SpectralHandle; spectral_res=0 means no real table was uploaded (dummy
    # 1-element buffers, BDPT/SPPM GPU dispatch, Stage 3/4 not wired yet).
    var spectral: SpectralBuffers

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

def gpu_upload_scene[Ompc: Origin[mut=True], Ofic: Origin[mut=True], Ovic: Origin[mut=True], Ouv: Origin[mut=True], Onv: Origin[mut=True]](
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    bvh2NodesCount: Int64,
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    primIdsCount: Int64,
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    blasNodeCounts: Pointer[Int32, MutUntrackedOrigin],
    blasPrimidCounts: Pointer[Int32, MutUntrackedOrigin],
    blasCount: Int64,
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    instanceCount: Int64,
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    meshCount: Int64,
    meshPointsCounts: Pointer[Int64, Ompc],
    meshFaceIndicesCounts: Pointer[Int64, Ofic],
    meshVertexIndicesCounts: Pointer[Int64, Ovic],
    meshUvNVerts: Pointer[Int64, Ouv],
    meshNrmNVerts: Pointer[Int64, Onv],
    tex_filenames: Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
    n_tex: Int32,
    materials: Pointer[Material_C, MutUntrackedOrigin],
    materialCount: Int64,
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    sphereCount: Int64,
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    curveCount: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    distantLightCount: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    pointLightCount: Int64,
    lightSamplerCdf: Pointer[Float32, MutUntrackedOrigin],
    lightSamplerN: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    infiniteLightCount: Int64,
    mediums: Pointer[Medium_C, MutUntrackedOrigin],
    mediumCount: Int64,
    medium_ifaces: Pointer[MediumInterface_C, MutUntrackedOrigin],
    medium_iface_count: Int64,
    grids: Pointer[Grid_C, MutUntrackedOrigin],
    gridCount: Int64,
    nvdbGrids: Pointer[NvdbGrid_C, MutUntrackedOrigin],
    nvdbGridCount: Int64,
    measured_brdfs: Pointer[MeasuredBRDF_C, MutUntrackedOrigin],
    measuredBrdfCount: Int64,
    n_pixels: Int64,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w_init: Pointer[Float32, MutUntrackedOrigin],
    filter: FilterParams,
    film: FilmDims,
    # Decomposed, NOT a single by-value `spectral: SpectralHandle` param --
    # see spectrum.mojo's long comment on the confirmed by-value SpectralHandle
    # miscompilation. This host function is part of the GPU-enabled
    # compilation unit (--target-accelerator build), and passing the handle
    # by value here reproduced the exact same corruption class (spectral_res
    # read back as 0, coeffs pointer read back as a tiny garbage address) --
    # see project_priority_backlog memory item 3 GPU-black-background bug.
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
) -> Pointer[GpuSceneHandle, MutUntrackedOrigin]:
    comptime if has_accelerator():
        try:
            var ctx = DeviceContext()

            # Check GPU memory
            var mem_info = ctx.get_memory_info()
            var free_bytes = mem_info[0]

            # Guard against zero-size device buffers (scene with no geometry):
            # a 0-byte enqueue_create_buffer yields a misaligned/invalid device
            # pointer that crashes on use and on free. Allocate at least 1 elem.
            var bvh_bytes = max(Int(bvh2NodesCount), 1) * size_of[BVH2Node]()
            var prim_bytes = max(Int(primIdsCount), 1) * size_of[PrimId_C]()
            var mesh_struct_bytes = max(Int(meshCount), 1) * size_of[TriangleMesh_C]()

            # Estimate total mesh data
            var mesh_data_bytes = 0
            for i in range(Int(meshCount)):
                mesh_data_bytes += Int(meshPointsCounts[unsafe_offset=i]) * 4       # Float32
                mesh_data_bytes += Int(meshFaceIndicesCounts[unsafe_offset=i]) * 8  # Int64
                mesh_data_bytes += Int(meshVertexIndicesCounts[unsafe_offset=i]) * 8 # Int64

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
            var blas_nodes_ptrs_host = unsafe_alloc[Pointer[UInt8, MutUntrackedOrigin]](max(n_blas_int, 1))
            var blas_primids_ptrs_host = unsafe_alloc[Pointer[UInt8, MutUntrackedOrigin]](max(n_blas_int, 1))
            for bi in range(n_blas_int):
                blas_nodes_ptrs_host[unsafe_offset=bi] = _gpu_upload_owned[BVH2Node](
                    ctx, blas_nodes_bufs, blasNodesArr[unsafe_offset=bi], Int(blasNodeCounts[unsafe_offset=bi])).unsafe_bitcast[UInt8]()
                blas_primids_ptrs_host[unsafe_offset=bi] = _gpu_upload_owned[PrimId_C](
                    ctx, blas_primids_bufs, blasPrimIdsArr[unsafe_offset=bi], Int(blasPrimidCounts[unsafe_offset=bi])).unsafe_bitcast[UInt8]()

            var blas_nodes_ptrs_buf = _gpu_upload_array[Pointer[UInt8, MutUntrackedOrigin]](
                ctx, blas_nodes_ptrs_host, n_blas_int)
            var blas_primids_ptrs_buf = _gpu_upload_array[Pointer[UInt8, MutUntrackedOrigin]](
                ctx, blas_primids_ptrs_host, n_blas_int)
            ctx.synchronize()   # the host pointer arrays are freed next
            blas_nodes_ptrs_host.unsafe_free(); blas_primids_ptrs_host.unsafe_free()

            var n_instances_int = Int(instanceCount)
            var instances_gpu_buf = _gpu_upload_array[Instance_C](ctx, instances, n_instances_int)

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

            var mesh_structs_host = unsafe_alloc[TriangleMesh_C](max(Int(meshCount), 1))

            for i in range(Int(meshCount)):
                var host_mesh = meshes[unsafe_offset=i]

                # Points (Float32), face and vertex indices (Int64), then UVs (2 floats
                # per vertex) and shading normals (3 per vertex), each a zeroed
                # 4-byte buffer when the mesh has none.
                var pts_dptr = _gpu_upload_owned[Float32](ctx, points_bufs, host_mesh.points, Int(meshPointsCounts[unsafe_offset=i]))
                var fi_dptr = _gpu_upload_owned[Int64](ctx, face_bufs, host_mesh.faceIndices, Int(meshFaceIndicesCounts[unsafe_offset=i]))
                var vi_dptr = _gpu_upload_owned[Int64](ctx, vert_bufs, host_mesh.vertexIndices, Int(meshVertexIndicesCounts[unsafe_offset=i]))
                var uv_n = Int(meshUvNVerts[unsafe_offset=i])
                var uv_dptr: Pointer[Float32, MutUntrackedOrigin]
                if uv_n > 0:
                    uv_dptr = _gpu_upload_owned[Float32](ctx, uv_bufs, host_mesh.uvs, uv_n * 2)
                else:
                    uv_dptr = _gpu_zeros_owned[Float32](ctx, uv_bufs, 1)
                var nrm_n = Int(meshNrmNVerts[unsafe_offset=i])
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
                mesh_structs_host[unsafe_offset=i] = TriangleMesh_C(pts_dptr, fi_dptr, vi_dptr, uv_dptr, nrm_dptr,
                    alpha_dptr, host_mesh.alpha_w, host_mesh.alpha_h, host_mesh.alpha_const)

            # Upload mesh struct array
            var meshes_buf = _gpu_upload_array[TriangleMesh_C](ctx, mesh_structs_host, Int(meshCount))
            ctx.synchronize()   # mesh_structs_host is freed next
            mesh_structs_host.unsafe_free()

            # Upload materials array (>= 1 elem to avoid a zero-size buffer)
            var mat_buf = _gpu_upload_array[Material_C](ctx, materials, Int(materialCount))

            ctx.synchronize()

            # Upload area lights. tri_cdf is a HOST pointer in the parsed
            # lights, so each mesh light's CDF goes up on its own and the
            # device copy of the struct points at that.
            var al_cdf_bufs = List[DeviceBuffer[DType.uint8]]()
            var al_host = unsafe_alloc[AreaLight_C](max(Int(areaLightCount), 1))
            for ali in range(Int(areaLightCount)):
                var al_i = areaLights[unsafe_offset=ali]
                if al_i.kind == Int8(0) and _is_real_ptr(al_i.tri_cdf) and al_i.n_tris > Int32(0):
                    al_i.tri_cdf = _gpu_upload_owned[Float32](ctx, al_cdf_bufs, al_i.tri_cdf, Int(al_i.n_tris))
                else:
                    al_i.tri_cdf = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
                al_host[unsafe_offset=ali] = al_i
            var al_buf = _gpu_upload_array[AreaLight_C](ctx, al_host, Int(areaLightCount))
            ctx.synchronize()   # al_host is freed next
            al_host.unsafe_free()

            # Upload spheres (analytical sphere primitives + sphere area lights)
            var sphere_buf = _gpu_upload_array[Sphere_C](ctx, spheres, Int(sphereCount))

            # Upload curves (native hair/fur primitives — control points only, no tessellation)
            var curve_buf = _gpu_upload_array[Curve_C](ctx, curves, Int(curveCount))

            # Upload distant (directional) lights
            var dl_buf = _gpu_upload_array[DistantLight_C](ctx, distantLights, Int(distantLightCount))

            # Upload point lights
            var pl_buf = _gpu_upload_array[PointLight_C](ctx, pointLights, Int(pointLightCount))

            # Upload light sampler CDF (n+1 Float32 entries), in a buffer of at
            # least 2 entries: pad a host copy, since enqueue_copy copies the whole
            # buffer. (No map_to_host anywhere here: its first use pins a ~1.3 GiB
            # host pool, ~1 s of page faults.)
            var ls_entries = Int(lightSamplerN) + 1
            var ls_host = unsafe_alloc[Float32](max(ls_entries, 2))
            ls_host[unsafe_offset=1] = Float32(0)
            unsafe_memcpy(dest=ls_host, src=lightSamplerCdf, count=ls_entries)
            var ls_buf = _gpu_upload_array[Float32](ctx, ls_host, max(ls_entries, 2))
            ctx.synchronize()   # ls_host is freed next
            ls_host.unsafe_free()

            # Upload infinite/environment lights with GPU-resident pixel/CDF data
            var il_count = Int(infiniteLightCount)
            var il_pixels_bufs = List[DeviceBuffer[DType.uint8]]()
            var il_cdf_bufs    = List[DeviceBuffer[DType.uint8]]()
            var il_w2l_bufs    = List[DeviceBuffer[DType.uint8]]()
            var il_patched = unsafe_alloc[InfiniteLight_C](max(il_count, 1))
            for ii in range(il_count):
                var il = infiniteLights[unsafe_offset=ii]
                # world_to_light matrix (16 floats), then pixels + CDF when textured.
                il.world_to_light = _gpu_upload_owned[Float32](ctx, il_w2l_bufs, il.world_to_light, 16)
                if il.cdf_w > Int32(0) and _is_real_ptr(il.pixels_ptr):
                    var iw = Int(il.cdf_w); var ih = Int(il.cdf_h)
                    # Pixels: iw × ih × 3 floats. CDF: (ih+1) marginal rows + ih×(iw+1) conditional entries.
                    il.pixels_ptr = _gpu_upload_owned[Float32](ctx, il_pixels_bufs, il.pixels_ptr, iw * ih * 3)
                    il.cdf_ptr = _gpu_upload_owned[Float32](ctx, il_cdf_bufs, il.cdf_ptr, (ih + 1) + ih * (iw + 1))
                il_patched[unsafe_offset=ii] = il
            var il_buf = _gpu_upload_array[InfiniteLight_C](ctx, il_patched, il_count)
            ctx.synchronize()   # il_patched is freed next
            il_patched.unsafe_free()
            print("GPU: " + String(il_count) + " infinite light(s) uploaded")

            # Upload participating media (small array; >= 1 elem to avoid zero-size buffer)
            var med_buf = _gpu_upload_array[Medium_C](ctx, mediums, Int(mediumCount))
            # Read the SSS flag off the host copy while it is still in reach --
            # the round-budget decision this feeds is made on the host, and
            # reading it back off the device later would need a sync.
            var has_sss_med = False
            for mi in range(Int(mediumCount)):
                if mediums[unsafe_offset=mi].is_sss != Int32(0):
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
            var grid_structs_host = unsafe_alloc[Grid_C](max(n_grids_int, 1))
            for gi in range(n_grids_int):
                var host_grid = grids[unsafe_offset=gi]
                var n_voxels = Int(host_grid.nx) * Int(host_grid.ny) * Int(host_grid.nz)
                grid_structs_host[unsafe_offset=gi] = Grid_C(
                    _gpu_upload_owned[Float32](ctx, grid_density_bufs, host_grid.density, n_voxels),
                    host_grid.nx, host_grid.ny, host_grid.nz,
                    host_grid.p0, host_grid.p1,
                    host_grid.world_to_medium, host_grid.max_density)
            var grids_buf = _gpu_upload_array[Grid_C](ctx, grid_structs_host, n_grids_int)
            ctx.synchronize()   # grid_structs_host is freed next
            grid_structs_host.unsafe_free()
            if n_grids_int > 0:
                print("GPU: " + String(n_grids_int) + " heterogeneous density grid(s) uploaded")

            # Upload sparse density grids ("nanovdb" media). Same shape as
            # the dense-grid upload just above: each grid's decompressed
            # blob gets its own device buffer, and the NvdbGrid_C struct
            # array embeds device-resident pointers into those buffers.
            var nvdb_blob_bufs = List[DeviceBuffer[DType.uint8]]()
            var n_nvdb_grids_int = Int(nvdbGridCount)
            var nvdb_structs_host = unsafe_alloc[NvdbGrid_C](max(n_nvdb_grids_int, 1))
            for gi in range(n_nvdb_grids_int):
                var host_nvdb = nvdbGrids[unsafe_offset=gi]
                nvdb_structs_host[unsafe_offset=gi] = NvdbGrid_C(
                    _gpu_upload_owned[UInt8](ctx, nvdb_blob_bufs, host_nvdb.blob, Int(host_nvdb.blob_size)),
                    host_nvdb.blob_size,
                    host_nvdb.world_to_medium, host_nvdb.inv_map, host_nvdb.map_vec,
                    host_nvdb.index_min, host_nvdb.index_max, host_nvdb.max_density)
            var nvdb_grids_buf = _gpu_upload_array[NvdbGrid_C](ctx, nvdb_structs_host, n_nvdb_grids_int)
            ctx.synchronize()   # nvdb_structs_host is freed next
            nvdb_structs_host.unsafe_free()
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
            var measured_structs_host = unsafe_alloc[MeasuredBRDF_C](max(n_measured_int, 1))
            for mi in range(n_measured_int):
                var hm = measured_brdfs[unsafe_offset=mi]
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

                measured_structs_host[unsafe_offset=mi] = MeasuredBRDF_C(
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
            var measured_brdfs_buf = _gpu_upload_array[MeasuredBRDF_C](ctx, measured_structs_host, n_measured_int)
            ctx.synchronize()   # measured_structs_host is freed next
            measured_structs_host.unsafe_free()
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
            var n_curve_compact_paths = n_curve_paths if Int(curveCount) > 0 else 1
            var r_curve_compact_path_buf = ctx.enqueue_create_buffer[DType.uint8](n_curve_compact_paths * 4)
            var r_curve_compact_counter_buf = ctx.enqueue_create_buffer[DType.uint8](4)
            ctx.enqueue_memset(r_film_buf, UInt8(0))
            ctx.enqueue_memset(r_albedo_film_buf, UInt8(0))

            # Load and upload textures
            var n_textures_int = Int(n_tex)
            # Textures referenced as normal maps hold linear data and must NOT be
            # sRGB-decoded on load. Mark those indices by scanning the materials.
            var tex_is_raw = unsafe_alloc[Bool](max(n_textures_int, 1))
            for ti in range(n_textures_int):
                tex_is_raw[unsafe_offset=ti] = False
            for mi in range(Int(materialCount)):
                var nidx = Int(materials[unsafe_offset=mi].normal_tex_idx)
                if nidx >= 0 and nidx < n_textures_int:
                    tex_is_raw[unsafe_offset=nidx] = True
            # Many scenes (e.g. landscape) declare a separate named Texture per
            # instance even when several instances share the same underlying
            # image file (batch-exported "-renamed-N" duplicates). Dedup by
            # (filename, raw-ness) so each unique file is only loaded from disk
            # and uploaded to the GPU once, instead of once per declaration.
            var dup_of = unsafe_alloc[Int32](max(n_textures_int, 1))
            for ti in range(n_textures_int):
                dup_of[unsafe_offset=ti] = Int32(-1)
                for tj in range(ti):
                    if dup_of[unsafe_offset=tj] == Int32(-1) and tex_is_raw[unsafe_offset=tj] == tex_is_raw[unsafe_offset=ti] and \
                       _cstr_eq(tex_filenames[unsafe_offset=ti], tex_filenames[unsafe_offset=tj]):
                        dup_of[unsafe_offset=ti] = Int32(tj)
                        break
            var tex_data_bufs = List[DeviceBuffer[DType.uint8]]()
            var gpu_textures_host = unsafe_alloc[GpuTexture_C](max(n_textures_int, 1))
            # 8-bit textures stay 8-bit on the GPU and decode through one of two
            # 256-entry tables (linear at 0, sRGB at 256), built by the oiio bridge
            # exactly as load_texture_rgb decodes, so level 0 matches the float path.
            var lut_host = unsafe_alloc[Float32](512)
            _ = external_call["texture_uint8_lut", NoneType, Int32, Pointer[Float32, MutUntrackedOrigin]](Int32(0), lut_host)
            _ = external_call["texture_uint8_lut", NoneType, Int32, Pointer[Float32, MutUntrackedOrigin]](Int32(1), lut_host.unsafe_offset(256))
            var lut_buf = ctx.enqueue_create_buffer[DType.float32](512)
            ctx.enqueue_copy(lut_buf, lut_host)
            var lut_dev = lut_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var inv_host = unsafe_alloc[UInt8](2 * _INV_LUT_SIZE)
            _build_inverse_lut(lut_host, inv_host)
            _build_inverse_lut(lut_host.unsafe_offset(256), inv_host.unsafe_offset(_INV_LUT_SIZE))
            # Decoding and mip building are independent per file and dominate
            # startup on texture-heavy scenes (Bistro), so run them on every core,
            # workers claiming the next texture from a shared cursor (file sizes
            # vary a lot). Uploads then happen here, in texture order, so the GPU
            # receives exactly the buffers a serial loop would build.
            var host_tex = unsafe_alloc[_HostTexture](max(n_textures_int, 1))
            var next_tex = unsafe_alloc[Int32](1)
            next_tex[unsafe_offset=0] = Int32(0)

            def decode_worker(_worker_idx: Int) {imm}:
                while True:
                    var ti = Int(Atomic.fetch_add(next_tex, Int32(1)))
                    if ti >= n_textures_int:
                        break
                    if dup_of[unsafe_offset=ti] == Int32(-1):
                        var raw_flag = Int32(1) if tex_is_raw[unsafe_offset=ti] else Int32(0)
                        host_tex[unsafe_offset=ti] = _load_host_texture(tex_filenames[unsafe_offset=ti], raw_flag, lut_host, inv_host)

            if n_textures_int > 0:
                parallelize(decode_worker, min(num_performance_cores(), n_textures_int))
            next_tex.unsafe_free()

            var tex_bytes = 0
            for ti in range(n_textures_int):
                if dup_of[unsafe_offset=ti] != Int32(-1):
                    gpu_textures_host[unsafe_offset=ti] = gpu_textures_host[unsafe_offset=Int(dup_of[unsafe_offset=ti])]
                    continue
                var ht = host_tex[unsafe_offset=ti]
                if ht.n_bytes == 0:
                    gpu_textures_host[unsafe_offset=ti] = GpuTexture_C(Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
                        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
                        Int32(0), Int32(0), Int32(0), Int32(0), Int32(GpuTexture_C.FORMAT_F32))
                    continue
                var lut = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
                if Int(ht.format) == GpuTexture_C.FORMAT_U8:
                    lut = lut_dev.unsafe_offset(Int(ht.lut_off))
                gpu_textures_host[unsafe_offset=ti] = GpuTexture_C(_gpu_upload_owned[UInt8](ctx, tex_data_bufs, ht.data, ht.n_bytes),
                    lut, ht.width, ht.height, ht.n_levels, ht.channels, ht.format)
                tex_bytes += ht.n_bytes
            var textures_gpu_buf = _gpu_upload_array[GpuTexture_C](ctx, gpu_textures_host, n_textures_int)
            # The uploads above are asynchronous; free their host sources once they're done.
            ctx.synchronize()
            for ti in range(n_textures_int):
                if dup_of[unsafe_offset=ti] == Int32(-1) and host_tex[unsafe_offset=ti].n_bytes > 0:
                    host_tex[unsafe_offset=ti].data.unsafe_free()
            host_tex.unsafe_free()
            lut_host.unsafe_free(); inv_host.unsafe_free()
            var n_unique_tex = 0
            for ti in range(n_textures_int):
                if dup_of[unsafe_offset=ti] == Int32(-1):
                    n_unique_tex += 1
            gpu_textures_host.unsafe_free()
            tex_is_raw.unsafe_free()
            dup_of.unsafe_free()
            print("GPU: " + String(n_textures_int) + " texture(s) uploaded ("
                  + String(n_unique_tex) + " unique file(s) loaded, "
                  + String(tex_bytes // (1024 * 1024)) + " MB)")

            # Upload Sobol matrices: first 1024 dimensions × 52 UInt32 = 212992 bytes
            comptime N_SOBOL_GPU_DIMS = 1024
            comptime N_SOBOL_GPU_WORDS = N_SOBOL_GPU_DIMS * 52
            var sobol_gpu_buf = _gpu_upload_array[UInt32](ctx, sobol_matrices, N_SOBOL_GPU_WORDS)

            # Upload raster_to_camera and camera_to_world (16 floats each)
            var r2c_gpu_buf = _gpu_upload_array[Float32](ctx, r2c, 16)
            var c2w_gpu_buf = _gpu_upload_array[Float32](ctx, c2w_init, 16)

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

            # Every upload is asynchronous: finish them while the caller's host
            # arrays are still alive.
            ctx.synchronize()

            # Allocate handle on heap
            var handle = unsafe_alloc[GpuSceneHandle](1)
            handle.unsafe_write(GpuSceneHandle(
                ctx=ctx^,
                bvh=BvhBuffers(
                    nodes_buf=bvh_buf^,
                    prim_ids_buf=prim_buf^,
                ),
                blas=BlasBuffers(
                    nodes_bufs=blas_nodes_bufs^,
                    primids_bufs=blas_primids_bufs^,
                    nodes_ptrs_buf=blas_nodes_ptrs_buf^,
                    primids_ptrs_buf=blas_primids_ptrs_buf^,
                    n_blas=n_blas_int,
                ),
                instances_buf=instances_gpu_buf^,
                n_instances=n_instances_int,
                meshes=MeshBuffers(
                    meshes_buf=meshes_buf^,
                    mesh_count=Int(meshCount),
                    points_bufs=points_bufs^,
                    faceIndices_bufs=face_bufs^,
                    vertexIndices_bufs=vert_bufs^,
                    uv_bufs=uv_bufs^,
                    nrm_bufs=nrm_bufs^,
                    alpha_bufs=alpha_bufs^,
                ),
                materials_buf=mat_buf^,
                material_count=Int(materialCount),
                textures=TextureBuffers(
                    tex_data_bufs=tex_data_bufs^,
                    textures_buf=textures_gpu_buf^,
                    lut_buf=lut_buf^,
                    n_textures=n_textures_int,
                ),
                lights=LightBuffers(
                    area_lights_buf=al_buf^,
                    area_light_cdf_bufs=al_cdf_bufs^,
                    n_area_lights=Int(areaLightCount),
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
                ),
                spheres_buf=sphere_buf^,
                n_spheres=Int(sphereCount),
                curves=CurveBuffers(
                    curves_buf=curve_buf^,
                    n_curves=Int(curveCount),
                    cand_prim_buf=r_curve_cand_prim_buf^,
                    cand_count_buf=r_curve_cand_count_buf^,
                    cand_offset_buf=r_curve_cand_offset_buf^,
                    compact_path_buf=r_curve_compact_path_buf^,
                    compact_counter_buf=r_curve_compact_counter_buf^,
                ),
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
                filter=filter,
                film=film,
                spectral=SpectralBuffers(
                    coeffs_buf=spec_coeffs_gpu_buf^,
                    cie_x_buf=spec_cie_x_gpu_buf^,
                    cie_y_buf=spec_cie_y_gpu_buf^,
                    cie_z_buf=spec_cie_z_gpu_buf^,
                    d65_buf=spec_d65_gpu_buf^,
                    res=spectral_res,
                ),
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



def shade_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    spectral: SpectralHandle,
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    shade_core(paths, intersections, meshes, materials, spectral, tid)



def shade_nee_preamble_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount_dp: Int64,
    textures: Pointer[GpuTexture_C, MutUntrackedOrigin],
    n_textures_dp: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    n_distant_lights_dp: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: Pointer[Float32, MutUntrackedOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    n_infinite_lights_dp: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
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
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].active == 0:
        return
    var inter = intersections[unsafe_offset=tid]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    # Do NOT early-exit on miss — shade_nee_core adds env-light contribution there.
    var ctx_no_shadow = ShadeContext(
        path_idx=0, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        shadow_tasks=Pointer[ShadowTask_C, MutUntrackedOrigin].unsafe_dangling(),
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
        gi_pending=Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
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
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount_dp: Int64,
    textures: Pointer[GpuTexture_C, MutUntrackedOrigin],
    n_textures_dp: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    n_distant_lights_dp: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: Pointer[Float32, MutUntrackedOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    n_infinite_lights_dp: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    # ReSTIR DI (Phase 2, --restir). Only gpu_render_sample ever passes
    # use_restir=True here -- gpu_render_wavefront has no ReSTIR concept at
    # all (see its own docstring: batch --restir renders via
    # gpu_render_sample instead, precisely to avoid the
    # WAVEFRONT_BATCH-concurrent-samples-per-pixel problem). All defaulted-
    # inert so gpu_render_wavefront's dispatch is unaffected.
    # Int32 rather than Bool: GPU kernel arguments must be DevicePassable and
    # Bool is not, which the compiler only reports at the enqueue site.
    use_restir: Int32 = Int32(0),
    restir_read: Pointer[DIReservoir, MutUntrackedOrigin] = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling(),
    restir_write: Pointer[DIReservoir, MutUntrackedOrigin] = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling(),
    gbuf_normal: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    gbuf_depth: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    gbuf_material_id: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
    gbuf_world_pos: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
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
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.diffuse:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var restir_on = use_restir != Int32(0)
    var ctx = ShadeContext(
        path_idx=0, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        shadow_tasks=Pointer[ShadowTask_C, MutUntrackedOrigin].unsafe_dangling(),
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=restir_on,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
        gi_pending=Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
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
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount_dp: Int64,
    textures: Pointer[GpuTexture_C, MutUntrackedOrigin],
    n_textures_dp: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    n_distant_lights_dp: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: Pointer[Float32, MutUntrackedOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    n_infinite_lights_dp: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
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
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.coated_diffuse:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
        gi_pending=Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_coated_diffuse[True, False](path_ptr, inter, ctx, mat)


def shade_diffuse_transmit_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount_dp: Int64,
    textures: Pointer[GpuTexture_C, MutUntrackedOrigin],
    n_textures_dp: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    n_distant_lights_dp: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: Pointer[Float32, MutUntrackedOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    n_infinite_lights_dp: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
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
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.diffuse_transmit:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
        gi_pending=Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
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
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.mix:
        return
    var inter = intersections[unsafe_offset=tid]
    var mat = materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var packed = mat.tex_idx
    var idx1 = Int(packed & Int32(0xFFFF))
    var idx2 = Int((packed >> 16) & Int32(0xFFFF))
    var amount = mat.roughU  # blend factor: 0 = all mat1, 1 = all mat2
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var chosen_idx = idx2 if pcg.next_float() < amount else idx1
    path_ptr[].pcgState = pcg.state
    var sub_type = materials[unsafe_offset=chosen_idx].type
    if sub_type == MatKind.mix:
        sub_type = MatKind.diffuse  # guard against mix-of-mix cycle, matches shade_mix (shading.mojo)
    intersections[unsafe_offset=tid].primId.materialIndex = Int64(chosen_idx)
    path_ptr[].pending_mat = sub_type


def shade_conductor_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount_dp: Int64,
    textures: Pointer[GpuTexture_C, MutUntrackedOrigin],
    n_textures_dp: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    n_distant_lights_dp: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: Pointer[Float32, MutUntrackedOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    n_infinite_lights_dp: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
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
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.conductor:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
        gi_pending=Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_conductor[True, False](path_ptr, inter, ctx, mat)


def shade_measured_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount_dp: Int64,
    textures: Pointer[GpuTexture_C, MutUntrackedOrigin],
    n_textures_dp: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    n_distant_lights_dp: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: Pointer[Float32, MutUntrackedOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    n_infinite_lights_dp: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
    measured_brdfs: Pointer[MeasuredBRDF_C, MutUntrackedOrigin],
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
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
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.measured:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=measured_brdfs,
        gi_pending=Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_measured[True, False](path_ptr, inter, ctx, mat)


def shade_dielectric_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    count_dp: Int64,
    textures: Pointer[GpuTexture_C, MutUntrackedOrigin],
    n_textures_dp: Int64,
    px_scale: Float32,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.dielectric:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = materials[unsafe_offset=Int(inter.primId.materialIndex)]
    # textures/px_scale are here only so a dielectric carrying "texture
    # displacement"/"normalmap" gets it applied (barcelona-pavilion's water).
    # tex_filenames is CPU-only (GPU samples the uploaded texture table), so
    # the dangling default is correct on this path.
    shade_dielectric[True](path_ptr, inter, meshes, mat, spheres,
        Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        textures, Int(n_textures_dp), px_scale)


def shade_thin_dielectric_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.thin_dielectric:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = materials[unsafe_offset=Int(inter.primId.materialIndex)]
    shade_thin_dielectric(path_ptr, inter, meshes, mat, spheres)


def shade_coated_conductor_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount_dp: Int64,
    textures: Pointer[GpuTexture_C, MutUntrackedOrigin],
    n_textures_dp: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    n_distant_lights_dp: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: Pointer[Float32, MutUntrackedOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    n_infinite_lights_dp: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
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
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.coated_conductor:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
        gi_pending=Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_coated_conductor[True, False](path_ptr, inter, ctx, mat)


def shade_interface_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    medium_ifaces: Pointer[MediumInterface_C, MutUntrackedOrigin],
    count_dp: Int64,
):
    """Passthrough (interface) material: advance ray through the surface.
    Medium update is handled by update_medium_gpu which runs after all shaders."""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.interface:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    shade_interface(path_ptr, inter)


def update_medium_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    medium_ifaces: Pointer[MediumInterface_C, MutUntrackedOrigin],
    count_dp: Int64,
):
    """Update current_medium_idx for any surface hit with a MediumInterface bound.
    Runs after all material shaders; uses the post-scatter ray direction (same
    convention as CPU rendering.mojo) to determine inside vs outside."""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].active == 0:
        return
    var inter = intersections[unsafe_offset=tid]
    if inter.hit == 0:
        return
    var mat = materials[unsafe_offset=Int(inter.primId.materialIndex)]
    if mat.medium_interface_idx < Int32(0):
        return
    var iface = medium_ifaces[unsafe_offset=Int(mat.medium_interface_idx)]
    var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)
    var geom_n: Vec3f
    if inter.primId.type == 4:
        # Sphere: outward normal = hit point - center. Medium-bounding
        # volumes (e.g. smoke-plume's "MediumInterface .. Shape sphere")
        # are commonly a big invisible sphere, so this case matters even
        # though spheres otherwise rarely carry materials with real shading.
        var sph = spheres[unsafe_offset=Int(inter.primId.id1)]
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
        var m = meshes[unsafe_offset=mi]
        var v0 = Int(m.vertexIndices[unsafe_offset=bv])
        var v1 = Int(m.vertexIndices[unsafe_offset=bv + 1])
        var v2 = Int(m.vertexIndices[unsafe_offset=bv + 2])
        var p0 = Vec3f(m.points[unsafe_offset=v0*4], m.points[unsafe_offset=v0*4+1], m.points[unsafe_offset=v0*4+2])
        var p1 = Vec3f(m.points[unsafe_offset=v1*4], m.points[unsafe_offset=v1*4+1], m.points[unsafe_offset=v1*4+2])
        var p2 = Vec3f(m.points[unsafe_offset=v2*4], m.points[unsafe_offset=v2*4+1], m.points[unsafe_offset=v2*4+2])
        geom_n = cross(p1 - p0, p2 - p0)
    if dot(ray_dir, geom_n) > Float32(0.0):
        path_ptr[].current_medium_idx = iface.outside_medium_idx
    else:
        path_ptr[].current_medium_idx = iface.inside_medium_idx



@always_inline
def _volume_nee_light(
    path_ptr: Pointer[PathState_C, MutUntrackedOrigin],
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
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres: Int,
    materials: Pointer[Material_C, MutUntrackedOrigin],
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin], spectral_res: Int,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin],
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin],
    spectral_d65: Pointer[Float32, MutUntrackedOrigin],
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
                           Pointer(to=exit_i), blasNodesArr, blasPrimIdsArr,
                           instances, spheres, n_spheres)
        var span = ls.dist if exit_i.hit == Int8(0) else exit_i.tHit
        var Th = exp(-sigma_t_r * span)
        var ph_h = hg_phase(dot(wo, edir), g)
        var mis_h = Float32(1.0) if ls.is_delta else power_heuristic(ls.pdf, ph_h)
        path_ptr[].estimate += path_ptr[].throughput * medium_emission_spectral(
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
    path_ptr[].estimate += path_ptr[].throughput * medium_emission_spectral(
        ls.Li, path_ptr[].wavelengths, spectral_coeffs, spectral_res,
        spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65) * (Te * ph * mis / ls.pdf)


def _sample_medium_core(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    i: Int,
    mediums: Pointer[Medium_C, MutUntrackedOrigin],
    n_mediums: Int,
    grids: Pointer[Grid_C, MutUntrackedOrigin],
    nvdb_grids: Pointer[NvdbGrid_C, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    n_area_lights: Int,
    lightSamplerCdf: Pointer[Float32, MutUntrackedOrigin],
    n_light_sampler: Int,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin] = Pointer[Sphere_C, MutUntrackedOrigin].unsafe_dangling(),
    n_spheres: Int = 0,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    materials: Pointer[Material_C, MutUntrackedOrigin] = Pointer[Material_C, MutUntrackedOrigin].unsafe_dangling(),
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin] = Pointer[InfiniteLight_C, MutUntrackedOrigin].unsafe_dangling(),
    n_infinite_lights: Int = 0,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin] = Pointer[DistantLight_C, MutUntrackedOrigin].unsafe_dangling(),
    n_distant_lights: Int = 0,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin] = Pointer[PointLight_C, MutUntrackedOrigin].unsafe_dangling(),
    n_point_lights: Int = 0,
    # Phase 7.3 (docs/A2_restir_migration_plan.md, project_restir_migration
    # memory): volume-scatter TEMPORAL reuse. Decomposed pointers, not one
    # `vol_io: VolReservoirIO` argument -- same defensive convention this
    # file already applies to SpectralHandle at this same kind of boundary
    # (see spectrum.mojo's comment on rgb_to_spectral_sample). `pixel_idx`
    # only means anything when this call came from gpu_render_sample (one
    # path per pixel); the wavefront batch path always leaves it at -1,
    # which the code below treats identically to "no reuse".
    vol_read: Pointer[VolReservoir, MutUntrackedOrigin] = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
    vol_write: Pointer[VolReservoir, MutUntrackedOrigin] = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
    pixel_idx: Int = -1,
    # One Int8 per PATH SLOT (indexed by `i`, this call's own index -- NOT
    # by pixel_idx), reset to 0 once at the start of this dispatch/frame by
    # the caller: guards against a single path scattering more than once
    # inside a dense medium within one frame (common -- see the long
    # comment at this buffer's read site for the real bug this fixes).
    vol_used: Pointer[Int8, MutUntrackedOrigin] = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling(),
    # Phase 7.3 spatial reuse (2026-09-08): SAME G-buffers DI's own spatial
    # reuse already reads (handle[].atrous_depth_buf/gbuf_worldpos_buf on
    # GPU, depth_int/world_pos_int on CPU) -- harmless to pass unconditionally
    # (mirrors DI's own convention), vol_temporal_spatial_combine's own
    # `_is_real_ptr`/frame_w>0/frame_h>0 checks gate the spatial pass off
    # when they're not real or the caller (batch wavefront) has no G-buffer.
    vol_gbuf_depth: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    vol_gbuf_world_pos: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
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
    var path_ptr = paths.unsafe_offset(i)
    if path_ptr[].active == 0:
        return
    var med_idx = Int(path_ptr[].current_medium_idx)
    if med_idx < 0 or med_idx >= n_mediums:
        return
    var inter = intersections[unsafe_offset=i]
    if inter.hit == 0:
        return
    var med = mediums[unsafe_offset=med_idx]
    var sigma_t = med.sigma_a + med.sigma_s
    var pcg = PCG32(path_ptr[].pcgState, path_ptr[].pcgInc)
    var t_surf = inter.tHit
    var ray_org = Vec3f(path_ptr[].ray.origin.x, path_ptr[].ray.origin.y, path_ptr[].ray.origin.z)
    var ray_dir = Vec3f(path_ptr[].ray.direction.x, path_ptr[].ray.direction.y, path_ptr[].ray.direction.z)

    # ── Free flight ────────────────────────────────────────────────────────
    # ONE call for both medium kinds: sample_free_flight (geometry.mojo) picks
    # the homogeneous closed form or heterogeneous delta tracking against a
    # local majorant, and hands both back in the same shape. The delta-tracking
    # loop used to be written out inline right here, which is precisely why it
    # was the path tracer's alone -- SPPM and BDPT/VCM called the homogeneous
    # sampler unconditionally and rendered every density field as uniform fog.
    # See sample_free_flight's own comment for that bug.
    #
    # `use_dense`/`use_nvdb`/`grid`/`nvdb_grid`/`sigma_maj` stay resolved HERE
    # too, not because the free flight needs them (it resolves its own), but
    # because the volume-scatter NEE further down ratio-tracks its shadow ray
    # against the same grid and majorant.
    var use_nvdb = med.nvdb_idx >= Int32(0)
    var use_dense = med.grid_idx >= Int32(0)
    var grid = medium_grid_for(med, grids)
    var nvdb_grid = medium_nvdb_for(med, nvdb_grids)
    var majorant_density = nvdb_grid.max_density if use_nvdb else grid.max_density
    var sigma_maj = majorant_density * sigma_t.r

    var ff = sample_free_flight(
        med, grids, nvdb_grids, ray_org, ray_dir, t_surf, pcg,
        path_ptr[].wavelengths, spectral_coeffs, spectral_res,
        spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
    # Volumetric emission (pbrt NanoVDBMedium's temperature grid) accumulated
    # over every majorant candidate along the tracked segment, already spectral
    # and already weighted by each candidate's absorption fraction. Zero for a
    # non-emissive or homogeneous medium, so this costs nothing there.
    path_ptr[].estimate += path_ptr[].throughput * ff.emission

    # Set by the homogeneous branch when a spectral table is available: the
    # LANE-AVERAGED single-scattering albedo, which the scatter/absorb coin
    # below is played on instead of red's. Negative means "not set".
    var p_scatter_spec = Float32(-1.0)
    if not (use_dense or use_nvdb):
        # ── Homogeneous-only throughput bookkeeping ────────────────────────
        # Delta tracking carries transmittance implicitly in its accept/reject
        # decisions; the closed form does not, so the analytic branch -- and
        # ONLY it -- multiplies the chromatic transmittance ratio in, on the
        # pass-through path as well as the collision one.
        var t_seg = min(ff.t_free, t_surf)
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
        # colour conversion. Upsample sigma_t to the 4 hero lanes FIRST via
        # spec_refl_unbounded (the coefficient-safe smooth upsampler; grey
        # media pass through it exactly -- verified in
        # Tests/unit/test_coefficient_upsampling.mojo), THEN exponentiate PER
        # LANE -- the same shared helper bdpt.mojo/sppm.mojo use, so CPU PT /
        # GPU PT / VCM / SPPM all treat a medium's colour identically.
        # See docs/02_spectra_and_color.md, "Chromatic extinction".
        path_ptr[].throughput *= medium_transmittance_ratio_spectral(
            med, t_seg, ff.pdf, path_ptr[].wavelengths,
            spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
        if ff.collided:
            # Chromatic scattering ratio; 1 for a grey medium.
            #
            # The extra sigma_t.r closes the estimator against the scatter/
            # absorb coin below, which is played at RED's albedo
            # (sigma_s.r/sigma_t.r) whatever lane the distance came from.
            # With the segment factor above now exp(-sigma_i*t)/p_bar rather
            # than a ratio to red's own exponential, the full product is
            #   p_bar * (sigma_s.r/sigma_t.r)            <- actually sampled
            #     * exp(-sigma_i*t)/p_bar * sigma_t.r * sigma_s_i/sigma_s.r
            #   = sigma_s_i * exp(-sigma_i*t)            <- what is wanted
            # for every lane i. p_bar cancels, which is the point: correctness
            # does not depend on WHICH lane the free flight was drawn from.
            # sigma_s(lambda) comes from the SAME smooth upsampler as the
            # sigma_t(lambda) in the exponential above. It used to be
            # BAND-PICKED off the RGB triple, which made the per-lane albedo
            # sigma_s(lambda)/sigma_t(lambda) a ratio of two inconsistent
            # conversions -- invisible at 2-3 scatters, hue-inverting over a
            # subsurface walk's hundreds (see medium_sigma_s_spectral).
            var ss_r = max(med.sigma_s.r, Float32(1e-30))
            if spectral_res > 0:
                # The scatter/absorb coin is ONE coin for all four lanes, so
                # whichever albedo it is played on becomes the reference every
                # lane is corrected against. Playing it on RED made that
                # correction sigma_s(lambda)/sigma_s.r, which for skin runs up
                # to 1.48 and is systematically >1 in blue -- and a subsurface
                # walk multiplies hundreds of them, so the product is
                # log-normal and explodes. Measured on head.pbrt the moment
                # the albedo became chromatic: max 1.3 -> 1.75e9, with 1.4% of
                # pixels above 100x the median against pbrt's 0.000%.
                #
                # Play it on the LANE MEAN instead. The correction is then
                # alpha_i/alpha_bar, centred on 1 and bounded either side, and
                # it pairs with the free-flight MIS weight (also centred on 1)
                # so neither factor drifts. Unbiased either way -- E[.] is
                # alpha_i per scatter for both -- this is purely variance.
                var sig_t_spec = medium_sigma_t_spectral(
                    med, path_ptr[].wavelengths, spectral_coeffs, spectral_res,
                    spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
                var sig_s_spec = medium_sigma_s_spectral(
                    med, path_ptr[].wavelengths, spectral_coeffs, spectral_res,
                    spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
                var a0 = sig_s_spec.v0 / max(sig_t_spec.v0, Float32(1e-30))
                var a1 = sig_s_spec.v1 / max(sig_t_spec.v1, Float32(1e-30))
                var a2 = sig_s_spec.v2 / max(sig_t_spec.v2, Float32(1e-30))
                var a3 = sig_s_spec.v3 / max(sig_t_spec.v3, Float32(1e-30))
                var abar = (a0 + a1 + a2 + a3) * Float32(0.25)
                if abar < Float32(1e-6): abar = Float32(1e-6)
                p_scatter_spec = abar
                var inv_ab = Float32(1.0) / abar
                path_ptr[].throughput *= SpectralSample(
                    sig_t_spec.v0 * a0 * inv_ab, sig_t_spec.v1 * a1 * inv_ab,
                    sig_t_spec.v2 * a2 * inv_ab, sig_t_spec.v3 * a3 * inv_ab)
            else:
                # No spectral table: lanes carry RGB, so red IS the reference
                # and this is the original expression unchanged.
                var sig_s_spec = medium_sigma_s_spectral(
                    med, path_ptr[].wavelengths, spectral_coeffs, spectral_res,
                    spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
                path_ptr[].throughput *= (sig_s_spec * (Float32(1.0) / ss_r)) * sigma_t.r

    if not ff.collided:
        path_ptr[].pcgState = pcg.state
        return
    var t_free = ff.t_free
    # Density cancels between sigma_s and sigma_t, so this is the same
    # expression for both medium kinds.
    var albedo_r = med.sigma_s.r / max(sigma_t.r, Float32(1e-7))

    # Both branches above already `return` early for the "no real collision"
    # case (homogeneous: t_free >= t_surf; heterogeneous: not collided) — so
    # reaching here always means a real scatter/absorb event at t_free.
    var p_scatter = p_scatter_spec if p_scatter_spec >= Float32(0.0) else albedo_r
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
            # VOL_RIS_DISTANCE is a compile-time kill switch, so it gates the
            # whole test at compile time rather than sitting in the runtime
            # `and` chain: with the flag off the qualification test is not
            # emitted at all, instead of being evaluated and ANDed with False.
            var dist_ris: Bool
            comptime if VOL_RIS_DISTANCE:
                dist_ris = ((not use_dense) and (not use_nvdb)
                    and sigma_t.r > Float32(0.0) and t_surf > Float32(0.0)
                    and sigma_t.g == sigma_t.r and sigma_t.b == sigma_t.r
                    and med.sigma_s.g == med.sigma_s.r and med.sigma_s.b == med.sigma_s.r)
            else:
                dist_ris = False
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
                var al = areaLights[unsafe_offset=light_idx]
                var lmesh = meshes[unsafe_offset=Int(al.meshIdx)]
                var lti = area_light_pick_triangle(al, pcg.next_float())
                var r1 = pcg.next_float()
                var r2 = pcg.next_float()
                var lb = lti * 3
                var lv0 = Int(lmesh.vertexIndices[unsafe_offset=lb])
                var lv1 = Int(lmesh.vertexIndices[unsafe_offset=lb + 1])
                var lv2 = Int(lmesh.vertexIndices[unsafe_offset=lb + 2])
                var lp0 = Vec3f(lmesh.points[unsafe_offset=lv0*4], lmesh.points[unsafe_offset=lv0*4+1], lmesh.points[unsafe_offset=lv0*4+2])
                var lp1 = Vec3f(lmesh.points[unsafe_offset=lv1*4], lmesh.points[unsafe_offset=lv1*4+1], lmesh.points[unsafe_offset=lv1*4+2])
                var lp2 = Vec3f(lmesh.points[unsafe_offset=lv2*4], lmesh.points[unsafe_offset=lv2*4+1], lmesh.points[unsafe_offset=lv2*4+2])
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
                and _is_real_ptr(vol_used) and vol_used[unsafe_offset=i] == Int8(0))
            if vol_reuse_ok:
                vol_used[unsafe_offset=i] = Int8(1)
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
                            var grid_s = grids[unsafe_offset=Int(med.grid_idx)] if not use_nvdb_s else Grid_C(
                                Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Int32(0), Int32(0), Int32(0),
                                Point3f(Float32(0), Float32(0), Float32(0)), Point3f(Float32(0), Float32(0), Float32(0)),
                                SIMD[DType.float32, 16](0), Float32(0))
                            var nvdb_grid_s = nvdb_grids[unsafe_offset=Int(med.nvdb_idx)] if use_nvdb_s else NvdbGrid_C(
                                Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(), Int64(0), SIMD[DType.float32, 16](0),
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
                            var _exit_inter = Array[Intersection_C, 1](fill=Intersection_C(
                                PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0)),
                                Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0)))
                            var exit_ptr = _exit_inter.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
                            exit_ptr[unsafe_offset=0].hit = Int8(0)
                            traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, shad_ray,
                                               shad_tmax, exit_ptr, blasNodesArr, blasPrimIdsArr, instances)
                            test_spheres(spheres, n_spheres, shad_ray, exit_ptr)
                            # test_spheres ignores shad_tmax (it bounds only by
                            # an already-recorded closer hit), so a sphere past
                            # the light would otherwise set t_med > dist and
                            # over-attenuate instead of under-.
                            if exit_ptr[unsafe_offset=0].hit != Int8(0) and exit_ptr[unsafe_offset=0].tHit <= shad_tmax:
                                var exit_mat = materials[unsafe_offset=Int(exit_ptr[unsafe_offset=0].primId.materialIndex)]
                                if exit_mat.type == MatKind.interface:
                                    t_med = exit_ptr[unsafe_offset=0].tHit
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
                        var al_win = areaLights[unsafe_offset=Int(res.light_idx)]
                        var sel_lo = lightSamplerCdf[unsafe_offset=Int(res.light_idx)]
                        var sel_hi = lightSamplerCdf[unsafe_offset=Int(res.light_idx) + 1]
                        var sel_pdf_win = max(sel_hi - sel_lo, Float32(1e-6))
                        var mis_w = Float32(1.0)
                        if al_win.total_area > Float32(0.0):
                            var pdf_light = dist_sq * sel_pdf_win / (cos_l * al_win.total_area)
                            mis_w = power_heuristic(pdf_light, ph_a)
                        path_ptr[].estimate += path_ptr[].throughput * medium_emission_spectral(
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
        var scatter_w = scatter_pt.to_simd()
        var wo_v = -ray_dir
        for dl_i in range(n_distant_lights):
            _volume_nee_light(path_ptr, _sample_distant_light_nee(distantLights[unsafe_offset=dl_i]),
                scatter_w, wo_v, med.g, pcg, use_nvdb, use_dense, grid, nvdb_grid, sigma_maj, sigma_t.r,
                bvh2Nodes, primIds, meshes, curves, blasNodesArr, blasPrimIdsArr,
                instances, spheres, n_spheres, materials,
                spectral_coeffs, spectral_res, spectral_cie_x,
                spectral_cie_y, spectral_cie_z, spectral_d65)
        for pl_i in range(n_point_lights):
            _volume_nee_light(path_ptr, _sample_point_light_nee(pointLights[unsafe_offset=pl_i], scatter_w),
                scatter_w, wo_v, med.g, pcg, use_nvdb, use_dense, grid, nvdb_grid, sigma_maj, sigma_t.r,
                bvh2Nodes, primIds, meshes, curves, blasNodesArr, blasPrimIdsArr,
                instances, spheres, n_spheres, materials,
                spectral_coeffs, spectral_res, spectral_cie_x,
                spectral_cie_y, spectral_cie_z, spectral_d65)
        for sph_i in range(n_spheres):
            if spheres[unsafe_offset=sph_i].isAreaLight == Int8(1):
                _volume_nee_light(path_ptr, _sample_sphere_light_nee(spheres[unsafe_offset=sph_i], n_spheres, scatter_w, pcg),
                    scatter_w, wo_v, med.g, pcg, use_nvdb, use_dense, grid, nvdb_grid, sigma_maj, sigma_t.r,
                    bvh2Nodes, primIds, meshes, curves, blasNodesArr, blasPrimIdsArr,
                    instances, spheres, n_spheres, materials,
                    spectral_coeffs, spectral_res, spectral_cie_x,
                    spectral_cie_y, spectral_cie_z, spectral_d65)
        for inf_i in range(n_infinite_lights):
            _volume_nee_light(path_ptr,
                _sample_infinite_light_nee(infiniteLights[unsafe_offset=inf_i], Point2f(pcg.next_float(), pcg.next_float())),
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
        intersections[unsafe_offset=i].hit = Int8(0)  # no surface hit this bounce
    else:
        # Absorbed
        path_ptr[].pcgState = pcg.state
        path_ptr[].active = Int8(0)

def sample_medium_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    mediums: Pointer[Medium_C, MutUntrackedOrigin],
    n_mediums_dp: Int64,
    grids: Pointer[Grid_C, MutUntrackedOrigin],
    nvdb_grids: Pointer[NvdbGrid_C, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    n_area_lights_dp: Int64,
    lightSamplerCdf: Pointer[Float32, MutUntrackedOrigin],
    n_light_sampler_dp: Int64,
    count_dp: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin] = Pointer[Sphere_C, MutUntrackedOrigin].unsafe_dangling(),
    n_spheres_dp: Int64 = Int64(0),
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    materials: Pointer[Material_C, MutUntrackedOrigin] = Pointer[Material_C, MutUntrackedOrigin].unsafe_dangling(),
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin] = Pointer[InfiniteLight_C, MutUntrackedOrigin].unsafe_dangling(),
    n_infinite_lights_dp: Int64 = Int64(0),
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin] = Pointer[DistantLight_C, MutUntrackedOrigin].unsafe_dangling(),
    n_distant_lights_dp: Int64 = Int64(0),
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin] = Pointer[PointLight_C, MutUntrackedOrigin].unsafe_dangling(),
    n_point_lights_dp: Int64 = Int64(0),
    # Phase 7.3: only gpu_render_wavefront_kernels(...) callers that pass
    # use_vol_restir=1 AND real buffers get reuse -- see _sample_medium_core's
    # own comment for why these stay decomposed rather than one VolReservoirIO.
    use_vol_restir: Int32 = Int32(0),
    vol_read: Pointer[VolReservoir, MutUntrackedOrigin] = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
    vol_write: Pointer[VolReservoir, MutUntrackedOrigin] = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
    vol_used: Pointer[Int8, MutUntrackedOrigin] = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling(),
    vol_gbuf_depth: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    vol_gbuf_world_pos: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    vol_frame_w: Int32 = Int32(0),
    vol_frame_h: Int32 = Int32(0),
):
    """GPU kernel wrapper: bounds-check, then call the SAME
    _sample_medium_core the CPU driver (render_all_tiles) calls."""
    var n_spheres = Int(n_spheres_dp)
    var spectral_res = Int(spectral_res_dp)
    var n_mediums = Int(n_mediums_dp)
    var n_area_lights = Int(n_area_lights_dp)
    var n_light_sampler = Int(n_light_sampler_dp)
    var count = Int(count_dp)
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
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount_dp: Int64,
    textures: Pointer[GpuTexture_C, MutUntrackedOrigin],
    n_textures_dp: Int64,
    distantLights: Pointer[DistantLight_C, MutUntrackedOrigin],
    n_distant_lights_dp: Int64,
    pointLights: Pointer[PointLight_C, MutUntrackedOrigin],
    n_point_lights_dp: Int64,
    lightSamplerCdf: Pointer[Float32, MutUntrackedOrigin],
    n_light_sampler_dp: Int64,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    n_infinite_lights_dp: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    count_dp: Int64,
    px_scale: Float32,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
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
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].pending_mat != MatKind.hair:
        return
    path_ptr[].pending_mat = Int8(0)
    var inter = intersections[unsafe_offset=tid]
    var mat = materials[unsafe_offset=Int(inter.primId.materialIndex)]
    var ls = LightSampler_C(lightSamplerCdf, Int32(n_light_sampler), Int32(0))
    var ctx = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
        textures=textures, n_textures=n_textures,
        nmaps=Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=px_scale, sobol_matrices=sobol_matrices, guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
        gi_pending=Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=distantLights, distant_count=n_distant_lights,
            point_lights=pointLights, point_count=n_point_lights,
            infinite_lights=infiniteLights, infinite_count=n_infinite_lights,
            spheres=spheres, sphere_count=n_spheres, light_sampler=ls))
    shade_hair[True, False](path_ptr, inter, ctx, mat)


def shade_enqueue_shadow_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    intersections: Pointer[Intersection_C, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    materials: Pointer[Material_C, MutUntrackedOrigin],
    areaLights: Pointer[AreaLight_C, MutUntrackedOrigin],
    areaLightCount: Int,
    textures: Pointer[GpuTexture_C, MutUntrackedOrigin],
    n_textures: Int,
    infiniteLights: Pointer[InfiniteLight_C, MutUntrackedOrigin],
    n_infinite_lights: Int,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres: Int,
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
    count: Int,
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res: Int = 0,
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    shadow_tasks[unsafe_offset=tid].active = Int32(0)
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].active == 0:
        return
    var inter = intersections[unsafe_offset=tid]
    # Do NOT early-exit on miss — shade_nee_core adds env-light contribution there.
    var ls_shadow = LightSampler_C(Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(), Int32(0), Int32(0))
    var ctx_shadow = ShadeContext(
        path_idx=tid, bvh2Nodes=bvh2Nodes, primIds=primIds, meshes=meshes, curves=curves, materials=materials,
        tex_filenames=Pointer[Pointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin](),
        textures=textures, n_textures=n_textures,
        nmaps=Pointer[NormalSlopeMap_C, MutUntrackedOrigin].unsafe_dangling(),
        shadow_tasks=shadow_tasks,
        px_scale=Float32(0.0), sobol_matrices=Pointer[UInt32, MutUntrackedOrigin].unsafe_dangling(), guide=null_guide(), use_restir=False,
        blasNodesArr=blasNodesArr, blasPrimIdsArr=blasPrimIdsArr, instances=instances,
        spectral=SpectralHandle(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65),
        measured_brdfs=Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling(),
        gi_pending=Pointer[GIPendingX1, MutUntrackedOrigin].unsafe_dangling(), gi_io=gi_reservoir_io_null(),
        lights=LightContext(
            area_lights=areaLights, area_light_count=areaLightCount,
            distant_lights=Pointer[DistantLight_C, MutUntrackedOrigin](), distant_count=0,
            point_lights=Pointer[PointLight_C, MutUntrackedOrigin](), point_count=0,
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
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    shadow_tasks[unsafe_offset=tid].active = Int32(0)

def reset_restir_reservoirs_gpu(
    reservoirs: Pointer[DIReservoir, MutUntrackedOrigin],
    count_dp: Int64,
):
    """Clear ReSTIR DI reservoirs to "no candidate yet". Needed at scene
    upload and on every camera move: identity reprojection assumes the
    previous frame's reservoir describes THIS pixel's shading point, so a
    surviving reservoir after the camera moves would reuse a light chosen
    for a different view. The GPU film is cleared on the same events for
    exactly the same reason (gpu_clear_film)."""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    reservoirs[unsafe_offset=tid] = di_reservoir_init()

def reset_restir_vol_reservoirs_gpu(
    reservoirs: Pointer[VolReservoir, MutUntrackedOrigin],
    count_dp: Int64,
):
    """Clear volume ReSTIR reservoirs to "no candidate yet" -- the same
    identity-reprojection invalidation reason as reset_restir_reservoirs_gpu
    above, applied to Phase 7.3's per-pixel volume-scatter reservoirs."""
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    reservoirs[unsafe_offset=tid] = vol_reservoir_init()


def reset_vol_used_gpu(
    used: Pointer[Int8, MutUntrackedOrigin],
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
    used[unsafe_offset=tid] = Int8(0)


def traverse_shadow_rays_gpu(
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    shadow_tasks: Pointer[ShadowTask_C, MutUntrackedOrigin],
    count_dp: Int64,
    spheres: Pointer[Sphere_C, MutUntrackedOrigin] = Pointer[Sphere_C, MutUntrackedOrigin].unsafe_dangling(),
    n_spheres_dp: Int64 = Int64(0),
    materials: Pointer[Material_C, MutUntrackedOrigin] = Pointer[Material_C, MutUntrackedOrigin].unsafe_dangling(),
):
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var task = shadow_tasks[unsafe_offset=tid]
    if task.active == 0:
        return
    var shadow_ray = Ray_C(Point3f(task.origin.x, task.origin.y, task.origin.z), Vec3f(task.direction.x, task.direction.y, task.direction.z))
    if not any_hit_bvh2_core(bvh2Nodes, primIds, meshes, curves, shadow_ray, task.tmax, blasNodesArr, blasPrimIdsArr, instances, spheres, n_spheres, materials=materials):
        paths[unsafe_offset=tid].estimate += task.contrib



@always_inline
def _clamp_sample_rgb(r: Float32, g: Float32, b: Float32, lim: Float32
                      ) -> Tuple[Float32, Float32, Float32]:
    """pbrt's `maxcomponentvalue`, applied to ONE SAMPLE as pbrt applies it.

    RGBFilm::AddSample clamps each sample's sensor RGB before it enters the
    filter accumulator. normalize_film used to be our only clamp, and it runs
    on the FINISHED pixel -- a 64-sample mean almost never crosses the
    threshold, so the clamp removed essentially nothing while pbrt was
    removing real energy from every spiky sample. Measured on
    barcelona-pavilion, where the scene asks for maxcomponentvalue 50 at iso
    500: the lit deck read 1.410x the reference, and 1.197x with the clamp
    taken out of BOTH renderers -- i.e. this asymmetry owned most of that gap.

    `lim` is already divided by the film's iso scale, because normalize_film
    multiplies by iso/100 after the fact and pbrt clamps post-sensor.
    lim <= 0 disables it."""
    if lim <= Float32(0):
        return (r, g, b)
    var mx = max(r, max(g, b))
    if mx <= lim:
        return (r, g, b)
    var k = lim / mx
    return (r * k, g * k, b * k)


def accumulate_film_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    film: Pointer[Float32, MutUntrackedOrigin],
    albedo_film: Pointer[Float32, MutUntrackedOrigin],
    count_dp: Int64,
    sample_clamp: Float32 = Float32(0.0),
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    # ── Output boundary: spectral transport -> RGB film ──────────────────
    var _e = spectral_sample_to_rgb(spectral_coeffs, Int(spectral_res_dp),
        spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
        paths[unsafe_offset=tid].estimate, paths[unsafe_offset=tid].wavelengths)
    var (_cr, _cg, _cb) = _clamp_sample_rgb(_e[0], _e[1], _e[2], sample_clamp)
    film[unsafe_offset=tid*3+0] += _cr
    film[unsafe_offset=tid*3+1] += _cg
    film[unsafe_offset=tid*3+2] += _cb
    albedo_film[unsafe_offset=tid*3+0] += paths[unsafe_offset=tid].albedo.r
    albedo_film[unsafe_offset=tid*3+1] += paths[unsafe_offset=tid].albedo.g
    albedo_film[unsafe_offset=tid*3+2] += paths[unsafe_offset=tid].albedo.b


def clear_film_gpu(film: Pointer[Float32, MutUntrackedOrigin], n_pixels_dp: Int64):
    var n_pixels = Int(n_pixels_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n_pixels:
        return
    film[unsafe_offset=tid*3+0] = Float32(0)
    film[unsafe_offset=tid*3+1] = Float32(0)
    film[unsafe_offset=tid*3+2] = Float32(0)


# Wavefront accumulation: thread px sums actual_batch samples from path_buf layout
# path_buf[si * n_pixels + px] and adds to film[px].  No atomics needed (one thread per pixel).
def accumulate_film_wavefront_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    film: Pointer[Float32, MutUntrackedOrigin],
    albedo_film: Pointer[Float32, MutUntrackedOrigin],
    n_pixels_dp: Int64, actual_batch_dp: Int64,
    sample_clamp: Float32 = Float32(0.0),
    spectral_coeffs: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_res_dp: Int64 = Int64(0),
    spectral_cie_x: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_y: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_cie_z: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    spectral_d65: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
):
    var n_pixels = Int(n_pixels_dp)
    var actual_batch = Int(actual_batch_dp)
    var px = Int(block_idx.x * block_dim.x + thread_idx.x)
    if px >= n_pixels:
        return
    var r = Float32(0); var g = Float32(0); var b = Float32(0)
    var ar = Float32(0); var ag = Float32(0); var ab = Float32(0)
    for si in range(actual_batch):
        var p = paths[unsafe_offset=si * n_pixels + px]
        var _pe = spectral_sample_to_rgb(spectral_coeffs, Int(spectral_res_dp),
            spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65,
            p.estimate, p.wavelengths)
        var (_qr, _qg, _qb) = _clamp_sample_rgb(_pe[0], _pe[1], _pe[2], sample_clamp)
        r += _qr; g += _qg; b += _qb
        ar += p.albedo.r;  ag += p.albedo.g;  ab += p.albedo.b
    film[unsafe_offset=px*3+0] += r; film[unsafe_offset=px*3+1] += g; film[unsafe_offset=px*3+2] += b
    albedo_film[unsafe_offset=px*3+0] += ar; albedo_film[unsafe_offset=px*3+1] += ag; albedo_film[unsafe_offset=px*3+2] += ab


# Wavefront primary-ray generation: thread ti → pixel (ti % n_pixels), sample (si_start + ti // n_pixels).
# Layout: path_buf[si_local * n_pixels + px_flat] — adjacent threads touch adjacent pixels of same sample.
def gen_primary_rays_wavefront_gpu(
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    paths: Pointer[PathState_C, MutUntrackedOrigin],
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
    paths[unsafe_offset=ti] = PathState_C(
        ray,
        SpectralSample(Float32(1.0)),
        SpectralSample(Float32(0.0)),
        RGB(Float32(0.0)),
        Int32(0), pcg_state, pcg_inc,
        Int8(1), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0), Vec3f(Float32(0.0)),
        Float32(0.0),
        Int32(-1),
        Float32(1.0),   # current_dielectric_ior (vacuum)
        Float32(1.0),   # previous_dielectric_ior (vacuum)
        Float32(1.0),   # eta_scale
        Int32(3), sobol_idx,
        wavelengths,
        Float32(0.0),   # mis_null_dist
        INV_FOUR_PI,    # lastEnvNeePdf (gated by lastBsdfPdf > 0; set at each scatter)
        Float32(0.0),   # cone_len: total path length, accumulated per bounce
    )


# Traversal kernel that reads rays directly from PathState_C (no separate ray buffer).
def traverse_paths_gpu(
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    results: Pointer[Intersection_C, MutUntrackedOrigin],
    curve_cand_prim: Pointer[Int32, MutUntrackedOrigin],
    curve_cand_count: Pointer[Int32, MutUntrackedOrigin],
    count_dp: Int64,
):
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if paths[unsafe_offset=tid].active == 0:
        return
    curve_cand_count[unsafe_offset=tid] = Int32(0)
    traverse_bvh2_core_defer_curves(
        bvh2Nodes, primIds, meshes, curves, paths[unsafe_offset=tid].ray, Float32(1.0e38), results.unsafe_offset(tid),
        curve_cand_prim.unsafe_offset(tid * CURVE_DEFER_K), curve_cand_count.unsafe_offset(tid),
        blasNodesArr, blasPrimIdsArr, instances,
    )
    test_spheres(spheres, n_spheres, paths[unsafe_offset=tid].ray, results.unsafe_offset(tid))


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
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    rays: Pointer[Float32, MutUntrackedOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var ray = paths[unsafe_offset=tid].ray
    var idx = tid * 8
    rays[unsafe_offset=idx + 0] = ray.origin.x
    rays[unsafe_offset=idx + 1] = ray.origin.y
    rays[unsafe_offset=idx + 2] = ray.origin.z
    rays[unsafe_offset=idx + 3] = Float32(1e-4)
    rays[unsafe_offset=idx + 4] = ray.direction.x
    rays[unsafe_offset=idx + 5] = ray.direction.y
    rays[unsafe_offset=idx + 6] = ray.direction.z
    rays[unsafe_offset=idx + 7] = Float32(1.0e8)

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
    results: Pointer[Float32, MutUntrackedOrigin],
    inter: Pointer[Intersection_C, MutUntrackedOrigin],
    mesh_material_idx: Pointer[Int64, MutUntrackedOrigin],
    mesh_al_idx: Pointer[Int32, MutUntrackedOrigin],
    n_meshes_dp: Int64,
    count_dp: Int64,
    instance_base_mesh: Pointer[Int32, MutUntrackedOrigin] = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
):
    var n_meshes = Int(n_meshes_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var idx = tid * 8
    var iresults = results.unsafe_bitcast[Int32]()
    var hitFlag = iresults[unsafe_offset=idx + 6]
    if hitFlag == Int32(1):
        var raw_idx = Int(iresults[unsafe_offset=idx + 4])
        var tri = iresults[unsafe_offset=idx + 5]
        var geometry_idx = iresults[unsafe_offset=idx + 7]
        var instance_idx = Int32(-1)
        var mi = raw_idx
        if raw_idx >= n_meshes and _is_real_ptr(instance_base_mesh):
            instance_idx = Int32(raw_idx - n_meshes)
            mi = Int(instance_base_mesh[unsafe_offset=Int(instance_idx)]) + Int(geometry_idx)
        var mat_idx = Int64(0)
        var al = Int32(-1)
        if mi >= 0 and mi < n_meshes:
            mat_idx = mesh_material_idx[unsafe_offset=mi]
            if instance_idx < Int32(0):
                al = mesh_al_idx[unsafe_offset=mi]
        var hitT = results[unsafe_offset=idx + 0]
        var u = results[unsafe_offset=idx + 1]
        var v = results[unsafe_offset=idx + 2]
        if al >= Int32(0):
            inter[unsafe_offset=tid] = Intersection_C(
                PrimId_C(Int64(al), (Int64(mi) << 32) | Int64(tri), mat_idx, Int32(-1),
                         Int8(3), Int8(0), Int8(0), Int8(0)),
                hitT, u, v, Int8(1), Int8(0), Int8(0), Int8(0),
            )
        else:
            inter[unsafe_offset=tid] = Intersection_C(
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
        var curve_idx = Int64(iresults[unsafe_offset=idx + 4])
        var piece_info = Int64(iresults[unsafe_offset=idx + 5])
        var mat_idx = Int64(iresults[unsafe_offset=idx + 7])
        var hitT = results[unsafe_offset=idx + 0]
        var h = results[unsafe_offset=idx + 1]
        var v = results[unsafe_offset=idx + 2]
        inter[unsafe_offset=tid] = Intersection_C(
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
        inter[unsafe_offset=tid] = Intersection_C(dummy_id, Float32(1.0e38), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))

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
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    inter: Pointer[Intersection_C, MutUntrackedOrigin],
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    count_dp: Int64,
):
    var n_spheres = Int(n_spheres_dp)
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if paths[unsafe_offset=tid].active == 0:
        return
    test_spheres(spheres, n_spheres, paths[unsafe_offset=tid].ray, inter.unsafe_offset(tid))

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
def accumulate_cone_gpu(
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    results: Pointer[Intersection_C, MutUntrackedOrigin],
    count_dp: Int64,
):
    """Grow each path's texture-footprint cone by the segment just traced.

    Its own kernel, enqueued after WHICHEVER traversal ran, because there are
    two (the CUDA `traverse_paths_gpu` and the Vulkan-RT one, whose unpack
    kernel never sees `paths`). Putting it inside a material's shading instead
    would skip exactly the paths that matter: a camera ray refracting through
    water onto a textured floor never touches the diffuse material's geometry
    builder on the way through the water.
    """
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    if paths[unsafe_offset=tid].active == 0:
        return
    if results[unsafe_offset=tid].hit == Int8(0):
        return
    # ONLY along a specular chain. A ray cone tracks the CAMERA's footprint,
    # and that survives mirror reflection and refraction -- but at a diffuse
    # scatter the outgoing direction is random and the cone stops meaning
    # anything about the camera. Growing it there would blur deeper bounces
    # without bound and without justification; pbrt carries differentials for
    # camera rays and specular chains for the same reason. After the first
    # non-specular scatter the cone FREEZES at its last valid width, so those
    # bounces keep the footprint they legitimately had.
    if paths[unsafe_offset=tid].bounce == Int32(0) or paths[unsafe_offset=tid].specularBounce != Int8(0):
        paths[unsafe_offset=tid].cone_len += results[unsafe_offset=tid].tHit



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
    spheres: Pointer[Sphere_C, MutUntrackedOrigin] = Pointer[Sphere_C, MutUntrackedOrigin].unsafe_dangling(),
    n_spheres: Int = 0,
) raises:
    comptime block_size = 256
    var grid = ceildiv(n_total, block_size)

    ctx.enqueue_function[vulkaninterop_pack_rays_kernel](
        path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        interop_rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n_total),
        grid_dim=grid, block_dim=block_size,
    )

    var cuda_stream = CUDA(ctx.stream())
    _ = vulkaninterop_rt_trace(interop_scene, Int32(n_total), cuda_stream)

    var instance_base_mesh_ptr = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
    if instance_base_mesh_buf:
        instance_base_mesh_ptr = instance_base_mesh_buf.value().unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()

    ctx.enqueue_function[vulkaninterop_unpack_results_kernel](
        interop_results_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        mesh_material_idx_buf.unsafe_ptr().unsafe_bitcast[Int64]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        mesh_al_idx_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
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
            path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
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

def reset_curve_counter_gpu(counter: Pointer[Int32, MutUntrackedOrigin]):
    if block_idx.x == 0 and thread_idx.x == 0:
        counter[unsafe_offset=0] = Int32(0)

# The CUDA-native path (traverse_bvh2_core_defer_curves) always writes
# curve candidates at tid*CURVE_DEFER_K -- this reproduces that formula once
# at buffer-creation time so curve_cand_offset_buf is valid immediately.
# Vulkan RT rendering never touches curve_cand_prim_buf/curve_cand_count_buf/
# curve_cand_offset_buf at all anymore -- intersect_batch.comp resolves
# curve hits itself and writes them straight into the ordinary results
# buffer (see vulkaninterop_unpack_results_kernel's hitFlag==2 branch), so
# compact_curve_paths_gpu/resolve_curve_candidates_gpu never run for a
# Vulkan-RT render (see _gpu_bounce_kernels).
def init_curve_cand_offset_gpu(offset_buf: Pointer[Int32, MutUntrackedOrigin], n_dp: Int64):
    var n = Int(n_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n:
        return
    offset_buf[unsafe_offset=tid] = Int32(tid * CURVE_DEFER_K)

def compact_curve_paths_gpu(
    curve_cand_count: Pointer[Int32, MutUntrackedOrigin],
    n_dp: Int64,
    compact_pathIds: Pointer[Int32, MutUntrackedOrigin],
    compact_counter: Pointer[Int32, MutUntrackedOrigin],
):
    var n = Int(n_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n:
        return
    if curve_cand_count[unsafe_offset=tid] > Int32(0):
        var pos = Atomic.fetch_add(compact_counter, Int32(1))
        compact_pathIds[unsafe_offset=Int(pos)] = Int32(tid)

def resolve_curve_candidates_gpu(
    compact_pathIds: Pointer[Int32, MutUntrackedOrigin],
    compact_counter: Pointer[Int32, MutUntrackedOrigin],
    curve_cand_prim: Pointer[Int32, MutUntrackedOrigin],
    curve_cand_count: Pointer[Int32, MutUntrackedOrigin],
    curve_cand_offset: Pointer[Int32, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    paths: Pointer[PathState_C, MutUntrackedOrigin],
    results: Pointer[Intersection_C, MutUntrackedOrigin],
    n_dp: Int64,
):
    var n = Int(n_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n:
        return
    if tid >= Int(compact_counter[unsafe_offset=0]):
        return
    var pathId = Int(compact_pathIds[unsafe_offset=tid])
    var n_cand = Int(curve_cand_count[unsafe_offset=pathId])
    if n_cand == 0:
        return
    var base = Int(curve_cand_offset[unsafe_offset=pathId])
    var ray = paths[unsafe_offset=pathId].ray
    var ray_org = Vec3f(ray.origin.x, ray.origin.y, ray.origin.z)
    var ray_dir = Vec3f(ray.direction.x, ray.direction.y, ray.direction.z)
    var res = results[unsafe_offset=pathId]
    var best_t = res.tHit
    var best_u = res.u
    var best_v = res.v
    var best_prim = res.primId
    var best_hit = res.hit
    var changed = False
    for i in range(n_cand):
        var primIdx = Int(curve_cand_prim[unsafe_offset=base + i])
        var prim = primIds[unsafe_offset=primIdx]
        var curve = curves[unsafe_offset=Int(prim.id1)]
        var curve_hit = intersect_curve(ray_org, ray_dir, curve, Int(prim.id2) // 8, Int(prim.id2) % 8, best_t)
        if curve_hit[0]:
            best_t = curve_hit[1]
            best_u = curve_hit[2]
            best_v = curve_hit[3]
            best_prim = prim
            best_hit = Int8(1)
            changed = True
    if changed:
        results[unsafe_offset=pathId] = Intersection_C(best_prim, best_t, best_u, best_v, best_hit, 0, 0, 0)


# GPU kernel: generate primary PathState_C for every pixel in one pass.
# Each thread handles one pixel.  All sampling is pure math — no host calls.
def gen_primary_rays_gpu(
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    paths: Pointer[PathState_C, MutUntrackedOrigin],
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
    paths[unsafe_offset=tid] = PathState_C(
        ray,
        SpectralSample(Float32(1.0)),
        SpectralSample(Float32(0.0)),
        RGB(Float32(0.0)),
        Int32(0), pcg_state, pcg_inc,
        Int8(1), Int8(0), Int8(0), Int8(0), Int8(0), Int8(0), Vec3f(Float32(0.0)),
        Float32(0.0),
        Int32(-1),
        Float32(1.0),   # current_dielectric_ior (vacuum)
        Float32(1.0),   # previous_dielectric_ior (vacuum)
        Float32(1.0),   # eta_scale
        Int32(3), sobol_idx,
        wavelengths,
        Float32(0.0),   # mis_null_dist
        INV_FOUR_PI,    # lastEnvNeePdf (gated by lastBsdfPdf > 0; set at each scatter)
        Float32(0.0),   # cone_len: total path length, accumulated per bounce
    )


# GPU kernel: shoot one unjittered center ray per pixel and write normals + depth.
# Used to guide the à-trous denoiser with geometric edge information.
def gen_aux_buffers_gpu(
    r2c: Pointer[Float32, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    bvh2Nodes: Pointer[BVH2Node, MutUntrackedOrigin],
    primIds: Pointer[PrimId_C, MutUntrackedOrigin],
    meshes: Pointer[TriangleMesh_C, MutUntrackedOrigin],
    curves: Pointer[Curve_C, MutUntrackedOrigin],
    blasNodesArr: Pointer[Pointer[BVH2Node, MutUntrackedOrigin], MutUntrackedOrigin],
    blasPrimIdsArr: Pointer[Pointer[PrimId_C, MutUntrackedOrigin], MutUntrackedOrigin],
    instances: Pointer[Instance_C, MutUntrackedOrigin],
    spheres: Pointer[Sphere_C, MutUntrackedOrigin],
    n_spheres_dp: Int64,
    isects_tmp: Pointer[Intersection_C, MutUntrackedOrigin],
    normals_out: Pointer[Float32, MutUntrackedOrigin],
    depth_out: Pointer[Float32, MutUntrackedOrigin],
    curve_mask_out: Pointer[Float32, MutUntrackedOrigin],
    world_pos_out: Pointer[Float32, MutUntrackedOrigin],
    material_id_out: Pointer[Int32, MutUntrackedOrigin],
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
    var cx = r2c[unsafe_offset=0]*filmX + r2c[unsafe_offset=4]*filmY + r2c[unsafe_offset=12]
    var cy = r2c[unsafe_offset=1]*filmX + r2c[unsafe_offset=5]*filmY + r2c[unsafe_offset=13]
    var cz = r2c[unsafe_offset=2]*filmX + r2c[unsafe_offset=6]*filmY + r2c[unsafe_offset=14]
    var cw = r2c[unsafe_offset=3]*filmX + r2c[unsafe_offset=7]*filmY + r2c[unsafe_offset=15]
    if cw != Float32(0.0) and cw != Float32(1.0):
        cx /= cw; cy /= cw; cz /= cw
    var cl = sqrt(cx*cx + cy*cy + cz*cz)
    if cl > Float32(0): cx /= cl; cy /= cl; cz /= cl

    # Camera → world
    var dir = Vec3f(
        c2w[unsafe_offset=0]*cx + c2w[unsafe_offset=4]*cy + c2w[unsafe_offset=8]*cz,
        c2w[unsafe_offset=1]*cx + c2w[unsafe_offset=5]*cy + c2w[unsafe_offset=9]*cz,
        c2w[unsafe_offset=2]*cx + c2w[unsafe_offset=6]*cy + c2w[unsafe_offset=10]*cz,
    )
    var dl = dir.length()
    if dl > Float32(0): dir = dir / dl
    var org = Point3f(c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14])

    var ray = Ray_C(org, dir)
    var dummy_id = PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    isects_tmp[unsafe_offset=tid] = Intersection_C(dummy_id, Float32(1e38), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0))
    traverse_bvh2_core(bvh2Nodes, primIds, meshes, curves, ray, Float32(1e38), isects_tmp.unsafe_offset(tid), blasNodesArr, blasPrimIdsArr, instances)
    test_spheres(spheres, n_spheres, ray, isects_tmp.unsafe_offset(tid))

    var normal = Vec3f(Float32(0), Float32(0), Float32(1))
    var d = Float32(1e38)

    if isects_tmp[unsafe_offset=tid].hit != Int8(0):
        d = isects_tmp[unsafe_offset=tid].tHit
        var typ = Int(isects_tmp[unsafe_offset=tid].primId.type)
        if typ == 4:
            var si = Int(isects_tmp[unsafe_offset=tid].primId.id1)
            normal = sphere_outward_normal(org + dir*d, spheres[unsafe_offset=si].center)
        elif typ == 5:
            # Approximate outward normal for the denoiser G-buffer: h alone
            # (stored in isects_tmp.u) doesn't uniquely fix the azimuthal sign,
            # so this picks one consistent side — fine for denoising, not used
            # for shading (shade_hair derives its own frame independently).
            var curve = curves[unsafe_offset=Int(isects_tmp[unsafe_offset=tid].primId.id1)]
            var piece = min(Int(curve.n_pieces) - 1, Int(isects_tmp[unsafe_offset=tid].v * Float32(curve.n_pieces)))
            var (cq0, cq1, _, _) = curve_piece_endpoints(curve, piece)
            var caxis = cq1 - cq0
            var calen = sqrt(dot(caxis, caxis))
            if calen > Float32(1e-8):
                var ctangent = caxis * (Float32(1.0) / calen)
                var cu = _curve_perp_axis(ctangent)
                var cb = cross(ctangent, cu)
                var ch = isects_tmp[unsafe_offset=tid].u
                var cs = sqrt(max(Float32(0.0), Float32(1.0) - ch*ch))
                var cn = cu*ch + cb*cs
                normal = vec3f(cn)
        elif typ == 0 or typ == 1 or typ == 2 or typ == 3:
            var mesh_idx: Int
            var base_vidx: Int
            if typ == 0:
                mesh_idx  = Int(isects_tmp[unsafe_offset=tid].primId.id1)
                base_vidx = Int(isects_tmp[unsafe_offset=tid].primId.id2)
            else:
                mesh_idx  = Int(isects_tmp[unsafe_offset=tid].primId.id2 >> 32)
                base_vidx = Int(isects_tmp[unsafe_offset=tid].primId.id2 & 0xFFFFFFFF) * 3
            var mesh = meshes[unsafe_offset=mesh_idx]
            var vi0 = Int(mesh.vertexIndices[unsafe_offset=base_vidx])
            var vi1 = Int(mesh.vertexIndices[unsafe_offset=base_vidx + 1])
            var vi2 = Int(mesh.vertexIndices[unsafe_offset=base_vidx + 2])
            var p0 = Point3f(mesh.points[unsafe_offset=vi0*4], mesh.points[unsafe_offset=vi0*4+1], mesh.points[unsafe_offset=vi0*4+2])
            var p1 = Point3f(mesh.points[unsafe_offset=vi1*4], mesh.points[unsafe_offset=vi1*4+1], mesh.points[unsafe_offset=vi1*4+2])
            var p2 = Point3f(mesh.points[unsafe_offset=vi2*4], mesh.points[unsafe_offset=vi2*4+1], mesh.points[unsafe_offset=vi2*4+2])
            var e1 = p1 - p0; var e2 = p2 - p0
            normal = Vec3f(e1.y*e2.z - e1.z*e2.y, e1.z*e2.x - e1.x*e2.z, e1.x*e2.y - e1.y*e2.x)
            var inst_idx = isects_tmp[unsafe_offset=tid].primId.instanceIdx
            if inst_idx >= Int32(0):
                var n_world = transform_normal_by_instance(instances[unsafe_offset=Int(inst_idx)].worldToObj, normal.to_simd())
                normal = vec3f(n_world)
            var nl = normal.length()
            if nl > Float32(0): normal = normal / nl
        # else: unrecognized primitive type (shouldn't happen — every hit
        # traverse_bvh2_core can produce is type 0-5) — leave normal at the
        # miss-ray default (0,0,1) rather than misreading id1/id2 as mesh data.
        if normal.dot(-dir) < Float32(0):
            normal = -normal

    normals_out[unsafe_offset=tid*3+0] = normal.x
    normals_out[unsafe_offset=tid*3+1] = normal.y
    normals_out[unsafe_offset=tid*3+2] = normal.z
    depth_out[unsafe_offset=tid] = d
    curve_mask_out[unsafe_offset=tid] = Float32(1.0) if (isects_tmp[unsafe_offset=tid].hit != Int8(0) and Int(isects_tmp[unsafe_offset=tid].primId.type) == 5) else Float32(0.0)
    store_vec3(world_pos_out, tid, (org + dir*d).to_simd())
    material_id_out[unsafe_offset=tid] = Int32(isects_tmp[unsafe_offset=tid].primId.materialIndex) if isects_tmp[unsafe_offset=tid].hit != Int8(0) else Int32(-1)


def gpu_gen_aux_buffers[Oc: Origin[mut=True]](
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    c2w: Pointer[Float32, Oc],
    n: Int64,
):
    """Generate unjittered normals and depth buffers for the denoiser."""
    var n_pix = Int(n)
    if n_pix == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            handle[].ctx.enqueue_copy(handle[].c2w_buf, c2w.unsafe_bitcast[UInt8]())
            comptime block_size = 256
            var grid_n = ceildiv(n_pix, block_size)
            handle[].ctx.enqueue_function[gen_aux_buffers_gpu](
                handle[].r2c_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].bvh.nodes_ptr(),
                handle[].bvh.prim_ids_ptr(),
                handle[].meshes.meshes_ptr(),
                handle[].curves.curves_ptr(),
                handle[].blas.nodes_arr(),
                handle[].blas.primids_arr(),
                handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C](),
                handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
                Int64(handle[].n_spheres),
                handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                handle[].atrous_normals_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].atrous_depth_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].atrous_curve_mask_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].gbuf_worldpos_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].gbuf_material_id_buf.unsafe_ptr().unsafe_bitcast[Int32](),
                Int64(handle[].film.width), Int64(handle[].film.height),
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
    paths: Pointer[PathState_C, MutUntrackedOrigin],
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
    if paths[unsafe_offset=tid].active != Int8(0) and paths[unsafe_offset=tid].bounce >= max_depth:
        paths[unsafe_offset=tid].at_cap = Int8(1)

def _gpu_bounce_kernels(
    handle: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    n: Int, grid_dim: Int, px_scale: Float32,
    max_depth: Int32,
    use_vulkan_rt: Bool = False,
    interop_scene: VulkanInteropRtSceneHandle = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
    interop_rays_buf: Optional[DeviceBuffer[DType.float32]] = None,
    interop_results_buf: Optional[DeviceBuffer[DType.float32]] = None,
    mesh_material_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    mesh_al_idx_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    n_meshes_vk: Int = 0,
    # ReSTIR DI: only gpu_render_sample passes these (one path per pixel, so
    # tid is a valid pixel index). gpu_render_wavefront leaves them inert.
    use_restir: Bool = False,
    restir_read: Pointer[DIReservoir, MutUntrackedOrigin] = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling(),
    restir_write: Pointer[DIReservoir, MutUntrackedOrigin] = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling(),
    # Phase 7.3: same "only gpu_render_sample passes these" contract as
    # use_restir/restir_read/restir_write above, for volume-scatter vertices.
    use_vol_restir_reuse: Bool = False,
    restir_vol_read: Pointer[VolReservoir, MutUntrackedOrigin] = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
    restir_vol_write: Pointer[VolReservoir, MutUntrackedOrigin] = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling(),
    # Per-pixel "already combined this frame" guard, reset by
    # gpu_render_sample before the bounce-round loop starts -- see
    # _sample_medium_core's own comment on vol_used for the bug this fixes.
    restir_vol_used: Pointer[Int8, MutUntrackedOrigin] = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling(),
    # Spatial reuse (2026-09-08): same G-buffers DI's own spatial reuse
    # already reads, harmless to pass unconditionally (see
    # _sample_medium_core's matching comment).
    restir_vol_gbuf_depth: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    restir_vol_gbuf_world_pos: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    restir_vol_frame_w: Int32 = Int32(0),
    restir_vol_frame_h: Int32 = Int32(0),
    # Object-instancing decode for Vulkan RT hits (see
    # vulkaninterop_unpack_results_kernel) -- None for scenes with no
    # instancing, matching every other Optional buffer above.
    instance_base_mesh_buf: Optional[DeviceBuffer[DType.uint8]] = None,
) raises:
    comptime block_size = 256
    handle[].ctx.enqueue_function[deactivate_paths_past_maxdepth_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
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
            handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
            handle[].n_spheres,
        )
    else:
        handle[].ctx.enqueue_function[traverse_paths_gpu](
            handle[].bvh.nodes_ptr(),
            handle[].bvh.prim_ids_ptr(),
            handle[].meshes.meshes_ptr(),
            handle[].curves.curves_ptr(),
            handle[].blas.nodes_arr(),
            handle[].blas.primids_arr(),
            handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C](),
            handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
            Int64(handle[].n_spheres),
            handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            handle[].curves.cand_prim_ptr(),
            handle[].curves.cand_count_ptr(),
            Int64(n),
            grid_dim=grid_dim,
            block_dim=block_size,
        )
    # Both traversal branches converge here: grow the texture-footprint cone
    # by the segment just traced, before any material shading reads it.
    handle[].ctx.enqueue_function[accumulate_cone_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
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
    if handle[].curves.n_curves > 0 and not use_vulkan_rt:
        handle[].ctx.enqueue_function[reset_curve_counter_gpu](
            handle[].curves.compact_counter_ptr(),
            grid_dim=1, block_dim=1,
        )
        handle[].ctx.enqueue_function[compact_curve_paths_gpu](
            handle[].curves.cand_count_ptr(),
            Int64(n),
            handle[].curves.compact_path_ptr(),
            handle[].curves.compact_counter_ptr(),
            grid_dim=grid_dim, block_dim=block_size,
        )
        handle[].ctx.enqueue_function[resolve_curve_candidates_gpu](
            handle[].curves.compact_path_ptr(),
            handle[].curves.compact_counter_ptr(),
            handle[].curves.cand_prim_ptr(),
            handle[].curves.cand_count_ptr(),
            handle[].curves.cand_offset_ptr(),
            handle[].bvh.prim_ids_ptr(),
            handle[].curves.curves_ptr(),
            handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            Int64(n),
            grid_dim=grid_dim, block_dim=block_size,
        )
    handle[].ctx.enqueue_function[sample_medium_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].mediums_buf.unsafe_ptr().unsafe_bitcast[Medium_C](),
        Int64(handle[].n_mediums),
        handle[].grids_buf.unsafe_ptr().unsafe_bitcast[Grid_C](),
        handle[].nvdb_grids_buf.unsafe_ptr().unsafe_bitcast[NvdbGrid_C](),
        handle[].bvh.nodes_ptr(),
        handle[].bvh.prim_ids_ptr(),
        handle[].meshes.meshes_ptr(),
        handle[].curves.curves_ptr(),
        handle[].blas.nodes_arr(),
        handle[].blas.primids_arr(),
        handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C](),
        handle[].lights.area_lights_ptr(),
        Int64(handle[].lights.n_area_lights),
        handle[].lights.light_sampler_ptr(),
        Int64(handle[].lights.n_light_sampler),
        Int64(n),
        handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].spectral.coeffs_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int64(handle[].spectral.res),
        handle[].spectral.cie_x_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_y_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_z_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.d65_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
                handle[].lights.infinite_lights_ptr(),
        Int64(handle[].lights.n_infinite_lights),
                handle[].lights.distant_lights_ptr(),
        Int64(handle[].lights.n_distant_lights),
        handle[].lights.point_lights_ptr(),
        Int64(handle[].lights.n_point_lights),
        Int32(1) if use_vol_restir_reuse else Int32(0),
        restir_vol_read, restir_vol_write, restir_vol_used,
        restir_vol_gbuf_depth, restir_vol_gbuf_world_pos,
        restir_vol_frame_w, restir_vol_frame_h,
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_nee_preamble_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].bvh.nodes_ptr(),
        handle[].bvh.prim_ids_ptr(),
        handle[].meshes.meshes_ptr(),
        handle[].curves.curves_ptr(),
        handle[].blas.nodes_arr(),
        handle[].blas.primids_arr(),
        handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
        handle[].lights.area_lights_ptr(),
        Int64(handle[].lights.n_area_lights),
        handle[].textures.textures_ptr(),
        Int64(handle[].textures.n_textures),
        handle[].lights.distant_lights_ptr(),
        Int64(handle[].lights.n_distant_lights),
        handle[].lights.point_lights_ptr(),
        Int64(handle[].lights.n_point_lights),
        handle[].lights.light_sampler_ptr(),
        Int64(handle[].lights.n_light_sampler),
        handle[].lights.infinite_lights_ptr(),
        Int64(handle[].lights.n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].spectral.coeffs_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int64(handle[].spectral.res),
        handle[].spectral.cie_x_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_y_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_z_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.d65_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    # mix is a pure selector (see shade_mix_gpu's docstring) -- enqueued
    # FIRST among the per-material kernels so its pending_mat/materialIndex
    # redirect is visible to whichever real kernel the sub-material resolves
    # to, later in this SAME launch-ordered sequence.
    handle[].ctx.enqueue_function[shade_mix_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
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
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask_C](),
        Int64(n),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_diffuse_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].bvh.nodes_ptr(),
        handle[].bvh.prim_ids_ptr(),
        handle[].meshes.meshes_ptr(),
        handle[].curves.curves_ptr(),
        handle[].blas.nodes_arr(),
        handle[].blas.primids_arr(),
        handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
        handle[].lights.area_lights_ptr(),
        Int64(handle[].lights.n_area_lights),
        handle[].textures.textures_ptr(),
        Int64(handle[].textures.n_textures),
        handle[].lights.distant_lights_ptr(),
        Int64(handle[].lights.n_distant_lights),
        handle[].lights.point_lights_ptr(),
        Int64(handle[].lights.n_point_lights),
        handle[].lights.light_sampler_ptr(),
        Int64(handle[].lights.n_light_sampler),
        handle[].lights.infinite_lights_ptr(),
        Int64(handle[].lights.n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].spectral.coeffs_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int64(handle[].spectral.res),
        handle[].spectral.cie_x_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_y_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_z_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.d65_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int32(1) if use_restir else Int32(0),
        restir_read,
        restir_write,
        handle[].atrous_normals_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].atrous_depth_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].gbuf_material_id_buf.unsafe_ptr().unsafe_bitcast[Int32](),
        handle[].gbuf_worldpos_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].film.width,
        handle[].film.height,
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_coated_diffuse_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].bvh.nodes_ptr(),
        handle[].bvh.prim_ids_ptr(),
        handle[].meshes.meshes_ptr(),
        handle[].curves.curves_ptr(),
        handle[].blas.nodes_arr(),
        handle[].blas.primids_arr(),
        handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
        handle[].lights.area_lights_ptr(),
        Int64(handle[].lights.n_area_lights),
        handle[].textures.textures_ptr(),
        Int64(handle[].textures.n_textures),
        handle[].lights.distant_lights_ptr(),
        Int64(handle[].lights.n_distant_lights),
        handle[].lights.point_lights_ptr(),
        Int64(handle[].lights.n_point_lights),
        handle[].lights.light_sampler_ptr(),
        Int64(handle[].lights.n_light_sampler),
        handle[].lights.infinite_lights_ptr(),
        Int64(handle[].lights.n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask_C](),
        handle[].spectral.coeffs_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int64(handle[].spectral.res),
        handle[].spectral.cie_x_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_y_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_z_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.d65_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_diffuse_transmit_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].bvh.nodes_ptr(),
        handle[].bvh.prim_ids_ptr(),
        handle[].meshes.meshes_ptr(),
        handle[].curves.curves_ptr(),
        handle[].blas.nodes_arr(),
        handle[].blas.primids_arr(),
        handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
        handle[].lights.area_lights_ptr(),
        Int64(handle[].lights.n_area_lights),
        handle[].textures.textures_ptr(),
        Int64(handle[].textures.n_textures),
        handle[].lights.distant_lights_ptr(),
        Int64(handle[].lights.n_distant_lights),
        handle[].lights.point_lights_ptr(),
        Int64(handle[].lights.n_point_lights),
        handle[].lights.light_sampler_ptr(),
        Int64(handle[].lights.n_light_sampler),
        handle[].lights.infinite_lights_ptr(),
        Int64(handle[].lights.n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask_C](),
        handle[].spectral.coeffs_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int64(handle[].spectral.res),
        handle[].spectral.cie_x_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_y_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_z_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.d65_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_conductor_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].bvh.nodes_ptr(),
        handle[].bvh.prim_ids_ptr(),
        handle[].meshes.meshes_ptr(),
        handle[].curves.curves_ptr(),
        handle[].blas.nodes_arr(),
        handle[].blas.primids_arr(),
        handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
        handle[].lights.area_lights_ptr(),
        Int64(handle[].lights.n_area_lights),
        handle[].textures.textures_ptr(),
        Int64(handle[].textures.n_textures),
        handle[].lights.distant_lights_ptr(),
        Int64(handle[].lights.n_distant_lights),
        handle[].lights.point_lights_ptr(),
        Int64(handle[].lights.n_point_lights),
        handle[].lights.light_sampler_ptr(),
        Int64(handle[].lights.n_light_sampler),
        handle[].lights.infinite_lights_ptr(),
        Int64(handle[].lights.n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask_C](),
        handle[].spectral.coeffs_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int64(handle[].spectral.res),
        handle[].spectral.cie_x_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_y_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_z_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.d65_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_measured_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].bvh.nodes_ptr(),
        handle[].bvh.prim_ids_ptr(),
        handle[].meshes.meshes_ptr(),
        handle[].curves.curves_ptr(),
        handle[].blas.nodes_arr(),
        handle[].blas.primids_arr(),
        handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
        handle[].lights.area_lights_ptr(),
        Int64(handle[].lights.n_area_lights),
        handle[].textures.textures_ptr(),
        Int64(handle[].textures.n_textures),
        handle[].lights.distant_lights_ptr(),
        Int64(handle[].lights.n_distant_lights),
        handle[].lights.point_lights_ptr(),
        Int64(handle[].lights.n_point_lights),
        handle[].lights.light_sampler_ptr(),
        Int64(handle[].lights.n_light_sampler),
        handle[].lights.infinite_lights_ptr(),
        Int64(handle[].lights.n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask_C](),
        handle[].measured_brdfs_buf.unsafe_ptr().unsafe_bitcast[MeasuredBRDF_C](),
        handle[].spectral.coeffs_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int64(handle[].spectral.res),
        handle[].spectral.cie_x_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_y_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_z_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.d65_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_dielectric_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].meshes.meshes_ptr(),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
        handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
        Int64(n),
        handle[].textures.textures_ptr(),
        Int64(handle[].textures.n_textures), px_scale,
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_thin_dielectric_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].meshes.meshes_ptr(),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
        handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
        Int64(n),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_coated_conductor_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].bvh.nodes_ptr(),
        handle[].bvh.prim_ids_ptr(),
        handle[].meshes.meshes_ptr(),
        handle[].curves.curves_ptr(),
        handle[].blas.nodes_arr(),
        handle[].blas.primids_arr(),
        handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
        handle[].lights.area_lights_ptr(),
        Int64(handle[].lights.n_area_lights),
        handle[].textures.textures_ptr(),
        Int64(handle[].textures.n_textures),
        handle[].lights.distant_lights_ptr(),
        Int64(handle[].lights.n_distant_lights),
        handle[].lights.point_lights_ptr(),
        Int64(handle[].lights.n_point_lights),
        handle[].lights.light_sampler_ptr(),
        Int64(handle[].lights.n_light_sampler),
        handle[].lights.infinite_lights_ptr(),
        Int64(handle[].lights.n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask_C](),
        handle[].spectral.coeffs_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int64(handle[].spectral.res),
        handle[].spectral.cie_x_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_y_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_z_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.d65_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_interface_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].meshes.meshes_ptr(),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
        handle[].medium_ifaces_buf.unsafe_ptr().unsafe_bitcast[MediumInterface_C](),
        Int64(n),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[update_medium_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].meshes.meshes_ptr(),
        handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
        handle[].medium_ifaces_buf.unsafe_ptr().unsafe_bitcast[MediumInterface_C](),
        Int64(n),
        grid_dim=grid_dim, block_dim=block_size,
    )
    handle[].ctx.enqueue_function[shade_hair_gpu](
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].inter_buf.unsafe_ptr().unsafe_bitcast[Intersection_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].bvh.nodes_ptr(),
        handle[].bvh.prim_ids_ptr(),
        handle[].meshes.meshes_ptr(),
        handle[].curves.curves_ptr(),
        handle[].blas.nodes_arr(),
        handle[].blas.primids_arr(),
        handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C](),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
        handle[].lights.area_lights_ptr(),
        Int64(handle[].lights.n_area_lights),
        handle[].textures.textures_ptr(),
        Int64(handle[].textures.n_textures),
        handle[].lights.distant_lights_ptr(),
        Int64(handle[].lights.n_distant_lights),
        handle[].lights.point_lights_ptr(),
        Int64(handle[].lights.n_point_lights),
        handle[].lights.light_sampler_ptr(),
        Int64(handle[].lights.n_light_sampler),
        handle[].lights.infinite_lights_ptr(),
        Int64(handle[].lights.n_infinite_lights),
        handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
        Int64(n), px_scale,
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask_C](),
        handle[].spectral.coeffs_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int64(handle[].spectral.res),
        handle[].spectral.cie_x_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_y_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_z_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.d65_buf.unsafe_ptr().unsafe_bitcast[Float32](),
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
        handle[].bvh.nodes_ptr(),
        handle[].bvh.prim_ids_ptr(),
        handle[].meshes.meshes_ptr(),
        handle[].curves.curves_ptr(),
        handle[].blas.nodes_arr(),
        handle[].blas.primids_arr(),
        handle[].instances_buf.unsafe_ptr().unsafe_bitcast[Instance_C](),
        handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        handle[].shadow_buf.unsafe_ptr().unsafe_bitcast[ShadowTask_C](),
        Int64(n),
        handle[].spheres_buf.unsafe_ptr().unsafe_bitcast[Sphere_C](),
        Int64(handle[].n_spheres),
        handle[].materials_buf.unsafe_ptr().unsafe_bitcast[Material_C](),
        grid_dim=grid_dim, block_dim=block_size,
    )


# Render one sample pass into the persistent film buffer.
# Ray generation runs on GPU — no CPU-side path buffer or PCIe upload needed.
def gpu_render_sample[Oc: Origin[mut=True]](
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    c2w: Pointer[Float32, Oc],
    si: Int32, log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    n: Int64,
    maxDepth: Int32,
    px_scale: Float32 = Float32(0.0),
    sample_clamp: Float32 = Float32(0.0),   # pbrt's per-sample maxcomponentvalue, iso-divided
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
            handle[].ctx.enqueue_copy(handle[].c2w_buf, c2w.unsafe_bitcast[UInt8]())
            comptime block_size = 256
            var grid_dim = ceildiv(n_int, block_size)
            # Generate primary rays on GPU
            handle[].ctx.enqueue_function[gen_primary_rays_gpu](
                handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
                handle[].r2c_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                Int64(handle[].film.width), Int64(handle[].film.height),
                si, log2spp, n_base4,
                seed_dim0, seed_dim1,
                rng_seed_lo, rng_seed_hi,
                handle[].filter.sigma, handle[].filter.norm_x, handle[].filter.support_x,
                handle[].filter.norm_y, handle[].filter.support_y,
                handle[].filter.type,
                Int64(n_int),
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            # ReSTIR reservoir ping-pong. Spatial reuse reads neighbouring
            # pixels, which other threads are writing this frame, so the read
            # side must be the PREVIOUS frame's finished buffer. Alternating on
            # frame parity gives that without any copy.
            var restir_rd = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling()
            var restir_wr = Pointer[DIReservoir, MutUntrackedOrigin].unsafe_dangling()
            if use_restir:
                var buf_a = handle[].restir_a_buf.unsafe_ptr().unsafe_bitcast[DIReservoir]()
                var buf_b = handle[].restir_b_buf.unsafe_ptr().unsafe_bitcast[DIReservoir]()
                if frame_index % 2 == 0:
                    restir_rd = buf_a; restir_wr = buf_b
                else:
                    restir_rd = buf_b; restir_wr = buf_a
            # Phase 7.3: same ping-pong rule as DI's above, own buffer pair.
            # Temporal-only (no spatial neighbour reads), so the same-frame
            # in-flight-write race spatial reuse would need to worry about
            # doesn't apply here, but ping-ponging costs nothing and keeps
            # this consistent with every other reservoir buffer in the file.
            var vol_rd = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling()
            var vol_wr = Pointer[VolReservoir, MutUntrackedOrigin].unsafe_dangling()
            var vol_used_ptr = Pointer[Int8, MutUntrackedOrigin].unsafe_dangling()
            var vol_gbuf_depth_ptr = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
            var vol_gbuf_world_pos_ptr = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
            var vol_fw = Int32(0)
            var vol_fh = Int32(0)
            if use_vol_restir_reuse:
                var vbuf_a = handle[].restir_vol_a_buf.unsafe_ptr().unsafe_bitcast[VolReservoir]()
                var vbuf_b = handle[].restir_vol_b_buf.unsafe_ptr().unsafe_bitcast[VolReservoir]()
                if frame_index % 2 == 0:
                    vol_rd = vbuf_a; vol_wr = vbuf_b
                else:
                    vol_rd = vbuf_b; vol_wr = vbuf_a
                vol_used_ptr = handle[].restir_vol_used_buf.unsafe_ptr().unsafe_bitcast[Int8]()
                handle[].ctx.enqueue_function[reset_vol_used_gpu](
                    vol_used_ptr, Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
                )
                # Spatial reuse: the SAME G-buffers DI's own spatial reuse
                # already reads (gen_aux_buffers_gpu populates them
                # unconditionally every frame, see gpu_gen_aux_buffers).
                vol_gbuf_depth_ptr = handle[].atrous_depth_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                vol_gbuf_world_pos_ptr = handle[].gbuf_worldpos_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                vol_fw = handle[].film.width
                vol_fh = handle[].film.height
            # See geometry.mojo's TERMINAL_SEGMENT_GRACE_ROUNDS: a maxdepth-
            # capped path still needs one more round to trace the ray from
            # its last real scatter and collect whatever it lands on or
            # escapes to. UNCONDITIONAL and universal, unlike the two
            # conditional margins below (a different reason: free null-
            # interface rounds / the SSS walk), which stay scene-gated.
            var gpu_max_rounds = Int(maxDepth) + TERMINAL_SEGMENT_GRACE_ROUNDS
            # Padding beyond the +1 above is CONDITIONAL on the scene
            # actually containing a medium: a null interface never occurs
            # otherwise, and there is no per-round host sync here (unlike
            # the CPU loop's cheap `anyActive` early-exit) to make extra
            # rounds free -- each one is a real, unconditional dispatch of
            # every kernel in _gpu_bounce_kernels.
            comptime _MEDIUM_INTERFACE_MARGIN = 8
            # An SSS interior is walked one scattering event per round and
            # those steps are not charged to maxDepth (Medium_C.is_sss), so
            # the round count is what actually bounds the walk. Unlike the
            # margin above this is a large budget, and unlike the CPU loop
            # there is no `anyActive` early exit here -- every round is a real
            # dispatch. Gated on the scene actually containing an SSS medium
            # so no other scene pays for it.
            comptime _SSS_WALK_ROUNDS = 256
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
                handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                handle[].film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].albedo_film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(n_int), sample_clamp,
                        handle[].spectral.coeffs_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int64(handle[].spectral.res),
        handle[].spectral.cie_x_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_y_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_z_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.d65_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        grid_dim=grid_dim,
                block_dim=block_size,
            )
        except e:
            print("GPU render sample failed: " + String(e))


# Wavefront render: generates actual_batch samples worth of primary rays for all pixels,
# runs the full bounce loop over n_pixels × actual_batch paths together, then accumulates.
# Caller loops over spp in steps of WAVEFRONT_BATCH; progress reporting is up to the caller.
def gpu_render_wavefront(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    c2w: Pointer[Float32, MutUntrackedOrigin],
    si_start: Int32, actual_batch: Int32,
    log2spp: Int32, n_base4: Int32,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed_lo: UInt32, rng_seed_hi: UInt32,
    n: Int64,
    maxDepth: Int32,
    px_scale: Float32 = Float32(0.0),
    sample_clamp: Float32 = Float32(0.0),   # pbrt's per-sample maxcomponentvalue, iso-divided
    # Task #163 stage 3: when use_vulkan_rt, every bounce's primary
    # intersection test is routed through the CUDA/Vulkan interop RT
    # backend (vulkaninterop_rt_traverse_paths_gpu, zero CPU sync) instead
    # of the traverse_paths_gpu CUDA kernel below -- caller must only set
    # this for scenes with no curves/spheres (object instancing IS
    # supported now -- see vulkaninterop_rt_create_scene/
    # vulkaninterop_rt_traverse_paths_gpu's docstrings). The interop_*/
    # mesh_*_buf/n_meshes_vk params are ignored when use_vulkan_rt is False.
    use_vulkan_rt: Bool = False,
    interop_scene: VulkanInteropRtSceneHandle = Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(),
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
            handle[].ctx.enqueue_copy(handle[].c2w_buf, c2w.unsafe_bitcast[UInt8]())
            comptime block_size = 256
            var grid_total = ceildiv(n_total, block_size)
            var grid_pix   = ceildiv(n_pix, block_size)
            handle[].ctx.enqueue_function[gen_primary_rays_wavefront_gpu](
                handle[].sobol_buf.unsafe_ptr().unsafe_bitcast[UInt32](),
                handle[].r2c_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].c2w_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                Int64(handle[].film.width), Int64(handle[].film.height),
                si_start, log2spp, n_base4,
                seed_dim0, seed_dim1, rng_seed_lo, rng_seed_hi,
                handle[].filter.sigma, handle[].filter.norm_x, handle[].filter.support_x,
                handle[].filter.norm_y, handle[].filter.support_y,
                handle[].filter.type,
                Int64(n_total), Int64(n_pix),
                grid_dim=grid_total,
                block_dim=block_size,
            )
            # See geometry.mojo's TERMINAL_SEGMENT_GRACE_ROUNDS: a maxdepth-
            # capped path still needs one more round to trace the ray from
            # its last real scatter and collect whatever it lands on or
            # escapes to. UNCONDITIONAL and universal, unlike the two
            # conditional margins below (a different reason: free null-
            # interface rounds / the SSS walk), which stay scene-gated.
            var gpu_max_rounds = Int(maxDepth) + TERMINAL_SEGMENT_GRACE_ROUNDS
            # Padding beyond the +1 above is CONDITIONAL on the scene
            # actually containing a medium: a null interface never occurs
            # otherwise, and there is no per-round host sync here (unlike
            # the CPU loop's cheap `anyActive` early-exit) to make extra
            # rounds free -- each one is a real, unconditional dispatch of
            # every kernel in _gpu_bounce_kernels.
            comptime _MEDIUM_INTERFACE_MARGIN = 8
            # An SSS interior is walked one scattering event per round and
            # those steps are not charged to maxDepth (Medium_C.is_sss), so
            # the round count is what actually bounds the walk. Unlike the
            # margin above this is a large budget, and unlike the CPU loop
            # there is no `anyActive` early exit here -- every round is a real
            # dispatch. Gated on the scene actually containing an SSS medium
            # so no other scene pays for it.
            comptime _SSS_WALK_ROUNDS = 256
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
                handle[].path_buf.unsafe_ptr().unsafe_bitcast[PathState_C]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                handle[].film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].albedo_film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(n_pix), Int64(batch), sample_clamp,
                        handle[].spectral.coeffs_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        Int64(handle[].spectral.res),
        handle[].spectral.cie_x_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_y_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.cie_z_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        handle[].spectral.d65_buf.unsafe_ptr().unsafe_bitcast[Float32](),
        grid_dim=grid_pix,
                block_dim=block_size,
            )
        except e:
            print("GPU wavefront render failed: " + String(e))


def gpu_download_film(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    film: Pointer[Float32, MutUntrackedOrigin],
    n: Int64,
):
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            # Straight device-to-host copy: map_to_host would pin a ~1.3 GiB host
            # pool on first use (~1 s of page faults). film holds n_int*3 floats,
            # the size of film_buf.
            handle[].ctx.enqueue_copy(film.unsafe_bitcast[UInt8](), handle[].film_buf)
            handle[].ctx.synchronize()
        except e:
            print("GPU download film failed: " + String(e))


def gpu_download_albedo[Of: Origin[mut=True]](
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    film: Pointer[Float32, Of],
    n: Int64,
):
    var n_int = Int(n)
    if n_int == 0:
        return
    comptime if has_accelerator():
        try:
            var handle = handlePtr
            # See gpu_download_film.
            handle[].ctx.enqueue_copy(film.unsafe_bitcast[UInt8](), handle[].albedo_film_buf)
            handle[].ctx.synchronize()
        except e:
            print("GPU download albedo failed: " + String(e))


# ── À-trous wavelet denoiser (Dammertz et al. 2010) ─────────────────────────
# Three kernels: normalize, variance estimate, one à-trous pass (5× ping-pong).

def normalize_beauty_albedo_gpu(
    film: Pointer[Float32, MutUntrackedOrigin],
    albedo_film: Pointer[Float32, MutUntrackedOrigin],
    beauty_out: Pointer[Float32, MutUntrackedOrigin],
    albedo_out: Pointer[Float32, MutUntrackedOrigin],
    n_pixels_dp: Int64,
    inv_weight: Float32,
    iso_scale: Float32,
    max_comp: Float32,
):
    var n_pixels = Int(n_pixels_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= n_pixels:
        return
    var lr = film[unsafe_offset=tid*3+0] * inv_weight * iso_scale
    var lg = film[unsafe_offset=tid*3+1] * inv_weight * iso_scale
    var lb = film[unsafe_offset=tid*3+2] * inv_weight * iso_scale
    if lr != lr or lr < Float32(0): lr = Float32(0)
    if lg != lg or lg < Float32(0): lg = Float32(0)
    if lb != lb or lb < Float32(0): lb = Float32(0)
    var scale = Float32(1.0)
    if max_comp > Float32(0.0):
        var mx = lr if lr > lg else lg
        if lb > mx: mx = lb
        if mx > max_comp:
            scale = max_comp / mx
    beauty_out[unsafe_offset=tid*3+0] = lr * scale
    beauty_out[unsafe_offset=tid*3+1] = lg * scale
    beauty_out[unsafe_offset=tid*3+2] = lb * scale
    albedo_out[unsafe_offset=tid*3+0] = albedo_film[unsafe_offset=tid*3+0] * inv_weight
    albedo_out[unsafe_offset=tid*3+1] = albedo_film[unsafe_offset=tid*3+1] * inv_weight
    albedo_out[unsafe_offset=tid*3+2] = albedo_film[unsafe_offset=tid*3+2] * inv_weight


def estimate_variance_gpu(
    beauty: Pointer[Float32, MutUntrackedOrigin],
    variance_out: Pointer[Float32, MutUntrackedOrigin],
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
            var l = Float32(0.2126)*beauty[unsafe_offset=ni] + Float32(0.7152)*beauty[unsafe_offset=ni+1] + Float32(0.0722)*beauty[unsafe_offset=ni+2]
            mean += l; mean_sq += l * l; count += 1
    var fc = Float32(count)
    mean /= fc; mean_sq /= fc
    var v = mean_sq - mean * mean
    variance_out[unsafe_offset=tid] = v if v > Float32(0) else Float32(0)


def firefly_clamp_gpu(
    beauty: Pointer[Float32, MutUntrackedOrigin],
    output: Pointer[Float32, MutUntrackedOrigin],
    fw_dp: Int64, fh_dp: Int64,
):
    """GPU counterpart of postprocess.mojo's _clamp_fireflies, which the GPU
    à-trous path never had until now -- a live divergence (see
    project_gpu_denoiser_energy_bug.md memory): without it, a single
    extreme-radiance pixel smears across the filter's full effective
    radius (up to 31px at 5 passes) exactly as it did on CPU before that
    fix existed. Same isolated-pixel test as CPU, via the SAME shared
    _firefly_clamp_pixel -- only the neighbor-gathering loop differs (one
    GPU thread per pixel vs a nested CPU loop)."""
    var fw = Int(fw_dp)
    var fh = Int(fh_dp)
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
            var lum_n = RGB(beauty[unsafe_offset=ni], beauty[unsafe_offset=ni + 1], beauty[unsafe_offset=ni + 2]).luma()
            if lum_n > max_n:
                max_n = lum_n
            if beauty[unsafe_offset=ni + 0] > max_n_r: max_n_r = beauty[unsafe_offset=ni + 0]
            if beauty[unsafe_offset=ni + 1] > max_n_g: max_n_g = beauty[unsafe_offset=ni + 1]
            if beauty[unsafe_offset=ni + 2] > max_n_b: max_n_b = beauty[unsafe_offset=ni + 2]
    var ci = tid * 3
    var c = _firefly_clamp_pixel(
        beauty[unsafe_offset=ci + 0], beauty[unsafe_offset=ci + 1], beauty[unsafe_offset=ci + 2],
        max_n, max_n_r, max_n_g, max_n_b, has_neighbor)
    output[unsafe_offset=ci + 0] = c.r
    output[unsafe_offset=ci + 1] = c.g
    output[unsafe_offset=ci + 2] = c.b


def atrous_filter_gpu(
    input: Pointer[Float32, MutUntrackedOrigin],
    albedo: Pointer[Float32, MutUntrackedOrigin],
    variance: Pointer[Float32, MutUntrackedOrigin],
    normals: Pointer[Float32, MutUntrackedOrigin],
    depth: Pointer[Float32, MutUntrackedOrigin],
    curve_mask: Pointer[Float32, MutUntrackedOrigin],
    output: Pointer[Float32, MutUntrackedOrigin],
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

    var c = RGB(input[unsafe_offset=tid*3], input[unsafe_offset=tid*3+1], input[unsafe_offset=tid*3+2])
    if curve_mask[unsafe_offset=tid] > Float32(0.5):
        # Hair/fur: strand-to-strand self-shadowing has no reliable correlate in
        # albedo/normal/depth (adjacent strands share material and similar
        # orientation/distance), so à-trous can't tell real occlusion from noise
        # and blurs it into a flat blob. Passing raw beauty through here matches
        # pbrt's own un-denoised look for hair instead of erasing strand detail.
        output[unsafe_offset=tid*3] = c.r; output[unsafe_offset=tid*3+1] = c.g; output[unsafe_offset=tid*3+2] = c.b
        return
    var cl = c.luma()
    var var_p = variance[unsafe_offset=tid]
    var ca = RGB(albedo[unsafe_offset=tid*3], albedo[unsafe_offset=tid*3+1], albedo[unsafe_offset=tid*3+2])
    var cn = Vec3f(normals[unsafe_offset=tid*3], normals[unsafe_offset=tid*3+1], normals[unsafe_offset=tid*3+2])
    # Clamp depth before squaring to avoid Float32 overflow (background sentinel=1e38).
    var cd_clamped = min(depth[unsafe_offset=tid], Float32(1e18))
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
            if curve_mask[unsafe_offset=ni1] > Float32(0.5):
                continue
            var qc = RGB(input[unsafe_offset=ni], input[unsafe_offset=ni+1], input[unsafe_offset=ni+2])
            var dl = qc.luma() - cl
            var dalb = RGB(albedo[unsafe_offset=ni], albedo[unsafe_offset=ni+1], albedo[unsafe_offset=ni+2]) - ca
            var ndot = normals[unsafe_offset=ni]*cn.x + normals[unsafe_offset=ni+1]*cn.y + normals[unsafe_offset=ni+2]*cn.z
            var dd = min(depth[unsafe_offset=ni1], Float32(1e18)) - cd_clamped
            var w = _atrous_spatial_weight(dx, dy) * _atrous_tap_weight(
                dl, var_p, variance[unsafe_offset=ni1], dalb, ndot, dd, cd_sq,
                sigma_l, sigma_a, sigma_n, sigma_d)
            acc += qc * w
            acc_w += w

    if acc_w > Float32(0):
        var o = acc / acc_w
        output[unsafe_offset=tid*3] = o.r; output[unsafe_offset=tid*3+1] = o.g; output[unsafe_offset=tid*3+2] = o.b
    else:
        output[unsafe_offset=tid*3] = c.r; output[unsafe_offset=tid*3+1] = c.g; output[unsafe_offset=tid*3+2] = c.b


def gpu_atrous_denoise[Oo: Origin[mut=True]](
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    output: Pointer[Float32, Oo],
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
            var fw = Int(handle[].film.width); var fh = Int(handle[].film.height)
            comptime block_size = 256
            var grid_n = ceildiv(n_pix, block_size)
            var inv_weight = Float32(1.0) / Float32(max(Int(frame_count), 1))
            var iso_scale = film_iso / Float32(100.0)

            handle[].ctx.enqueue_function[normalize_beauty_albedo_gpu](
                handle[].film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].albedo_film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].atrous_ping_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].atrous_albedo_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(n_pix), inv_weight, iso_scale, film_max_comp,
                grid_dim=grid_n, block_dim=block_size,
            )
            # --no-denoise: emit the normalized beauty (atrous_ping_buf) without
            # the à-trous blur passes, so the written image is the raw render.
            if not apply_denoise:
                handle[].ctx.enqueue_copy(output.unsafe_bitcast[UInt8](), handle[].atrous_ping_buf)
                handle[].ctx.synchronize()
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
                handle[].atrous_ping_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                handle[].atrous_pong_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(fw), Int64(fh),
                grid_dim=grid_n, block_dim=block_size,
            )
            var clamp_dst_ptr = handle[].atrous_pong_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            handle[].ctx.enqueue_function[estimate_variance_gpu](
                clamp_dst_ptr,
                handle[].atrous_variance_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(fw), Int64(fh),
                grid_dim=grid_n, block_dim=block_size,
            )

            var ping_ptr = handle[].atrous_ping_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var pong_ptr = handle[].atrous_pong_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var alb_ptr  = handle[].atrous_albedo_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var var_ptr  = handle[].atrous_variance_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var nrm_ptr  = handle[].atrous_normals_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var dep_ptr  = handle[].atrous_depth_buf.unsafe_ptr().unsafe_bitcast[Float32]()
            var cmask_ptr = handle[].atrous_curve_mask_buf.unsafe_ptr().unsafe_bitcast[Float32]()
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
            if n_passes % 2 == 1:
                handle[].ctx.enqueue_copy(output.unsafe_bitcast[UInt8](), handle[].atrous_ping_buf)
            else:
                handle[].ctx.enqueue_copy(output.unsafe_bitcast[UInt8](), handle[].atrous_pong_buf)
            handle[].ctx.synchronize()
        except e:
            print("GPU atrous denoise failed: " + String(e))


def gpu_clear_film(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
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
                handle[].film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(n_int),
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            handle[].ctx.enqueue_function[clear_film_gpu](
                handle[].albedo_film_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                Int64(n_int),
                grid_dim=grid_dim,
                block_dim=block_size,
            )
            handle[].ctx.synchronize()
        except e:
            print("GPU clear film failed: " + String(e))


def gpu_clear_restir(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
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
                handle[].restir_a_buf.unsafe_ptr().unsafe_bitcast[DIReservoir](),
                Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
            )
            handle[].ctx.enqueue_function[reset_restir_reservoirs_gpu](
                handle[].restir_b_buf.unsafe_ptr().unsafe_bitcast[DIReservoir](),
                Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
            )
            handle[].ctx.synchronize()
        except e:
            print("GPU clear restir failed: " + String(e))


def gpu_clear_restir_vol(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
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
                handle[].restir_vol_a_buf.unsafe_ptr().unsafe_bitcast[VolReservoir](),
                Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
            )
            handle[].ctx.enqueue_function[reset_restir_vol_reservoirs_gpu](
                handle[].restir_vol_b_buf.unsafe_ptr().unsafe_bitcast[VolReservoir](),
                Int64(n_int), grid_dim=grid_dim, block_dim=block_size,
            )
            handle[].ctx.synchronize()
        except e:
            print("GPU clear restir vol failed: " + String(e))


def gpu_free_scene(handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin]):
    if Int(handlePtr) == 0:
        return
    handlePtr.unsafe_deinit_pointee()
    handlePtr.unsafe_bitcast[GpuSceneHandle]().unsafe_free()
