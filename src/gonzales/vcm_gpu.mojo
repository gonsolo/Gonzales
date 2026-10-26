# VCM GPU megakernel driver.
# Part of the BDPT/VCM machinery that used to be one file (bdpt_*.mojo).

from std.sys import has_accelerator
from std.sys.info import size_of
from max.gpu import block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import tan, max, abs, ceildiv
from std.memory.alloc import unsafe_alloc
from .geometry import RGB, PI
from .primitives import Intersection
from .vcm_camis import CamisLightRecord
from .bvh import SceneView, _scene_bounding_sphere
from .sampling import film_filter_of
from .footprint import camera_footprint
from .transform import matrix_invert
from .pbrt_parser import ParsedScene_Mojo
from .sppm import _HSIZE
from .gpu_scene import GpuSceneHandle
from max.gpu.host._nvidia_cuda import CUDA
from .progress import Progress
from .outputs import finish_render, _sidecar_name, _write_channels
from .bdpt_vertex import _BDPT_MAX_VERTS, BDPTVertex
from .vcm_grid import (
    _VCM_VISIT_INSTRUMENT,
    _VCM_MN_STRIDE, _VCM_HEADS_SIZE, _CAMIS_LVC_PER_SLOT, _VCM_FOOTPRINT_PIXELS, _vcm_grid_inv_cell,
    _camera_typical_distance, vcm_merge_radius,
)
from .bdpt_render import _vcm_finalize_one_pixel
from .vcm_kernels import (
    _bdpt_emit_light_paths_gpu, _bdpt_splat_light_paths_gpu, _bdpt_camera_connect_gpu,
    bdpt_merge_grid_reset_gpu, bdpt_merge_grid_count_gpu, bdpt_merge_grid_insert_gpu,
    _VCM_BUDGET_REDUCE_THREADS, vcm_budget_reduce_gpu,
)

@always_inline
def _f32(buf: DeviceBuffer[DType.uint8]) -> Pointer[Float32, MutUntrackedOrigin]:
    return Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def _i32(buf: DeviceBuffer[DType.uint8]) -> Pointer[Int32, MutUntrackedOrigin]:
    return Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(buf.unsafe_ptr()))


def _vcm_write(buf: DeviceBuffer[DType.uint8], src: Pointer[Float32, MutUntrackedOrigin], n: Int) raises:
    with buf.map_to_host() as host:
        var dst = host.unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(n):
            dst[unsafe_offset=i] = src[unsafe_offset=i]


struct VcmGpu(Movable):
    """GPU VCM's buffers and per-render constants. A batch render runs all its passes on one, a session one pass per
    sample. `merging` False is BDPT: no merge grid, and merging's MIS weight is 0."""
    var handle: Pointer[GpuSceneHandle, MutUntrackedOrigin]
    var n_pix: Int
    var fw: Int
    var fh: Int
    var n_light_paths: Int
    var lvc_cap: Int
    var default_emit_med: Int32
    var scene_radius: Float32
    var merging: Bool
    var budget: Bool
    var cap: Int32
    var no_keep_mis: Bool
    var no_footprint: Bool
    var radius_scale: Float32
    var vcm_lambda: Float32
    var px_scale: Float32
    var cam: SIMD[DType.float32, 4]
    var lvc: DeviceBuffer[DType.uint8]
    var path_len: DeviceBuffer[DType.uint8]
    var heads_a: DeviceBuffer[DType.uint8]       # heads | counts | fine levels; this pass's and the last one's swap
    var heads_b: DeviceBuffer[DType.uint8]
    var stat_a: DeviceBuffer[DType.float32]      # --vcm-budget's per-bucket [Q, V], swapped like the heads
    var stat_b: DeviceBuffer[DType.float32]
    var budget_red: DeviceBuffer[DType.float32]
    var merge_next: DeviceBuffer[DType.uint8]
    var lvc_camis: DeviceBuffer[DType.uint8]
    var inter_light: DeviceBuffer[DType.uint8]
    var inter_cam: DeviceBuffer[DType.uint8]
    var accum: DeviceBuffer[DType.uint8]         # connect + t=1 splats of a batch render
    var accum_merge: DeviceBuffer[DType.uint8]   # merging, kept apart: it is added after denoising
    var albedo_accum: DeviceBuffer[DType.uint8]
    var visit_accum: DeviceBuffer[DType.uint8]
    var r2c: DeviceBuffer[DType.uint8]
    var c2w: DeviceBuffer[DType.uint8]
    var w2c: DeviceBuffer[DType.uint8]
    var c2r: DeviceBuffer[DType.uint8]

    def __init__(
        out self, handle: Pointer[GpuSceneHandle, MutUntrackedOrigin], psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
        ref sd: SceneView, n_photons_req: Int, merging: Bool = True, budget: Bool = False, cap: Int32 = Int32(0),
        no_keep_mis: Bool = False, radius_from_camera: Bool = False, cam_percentile: Float32 = Float32(0.5),
        cam_fraction_mult: Float32 = Float32(1.0), no_footprint: Bool = False, radius_scale: Float32 = Float32(1.0),
        verbose: Bool = False,
    ) raises:
        """`sd` is the host scene: its bounding sphere and media set the merge radius and the emission medium."""
        self.handle = handle
        self.fw = Int(psc[unsafe_offset=0].film_w)
        self.fh = Int(psc[unsafe_offset=0].film_h)
        self.n_pix = self.fw * self.fh
        self.n_light_paths = max(n_photons_req, self.n_pix)
        self.lvc_cap = self.n_light_paths * _BDPT_MAX_VERTS
        self.default_emit_med = Int32(-1)
        if Int(sd.mediumCount) > 0:
            for mi in range(Int(sd.mediumIfaceCount)):
                var iface = sd.mediumInterfaces[unsafe_offset=mi]
                if Int(iface.outside_medium_idx) >= 0:
                    self.default_emit_med = iface.outside_medium_idx
                    break
        var (_c, radius_bbox) = _scene_bounding_sphere(sd)
        self.scene_radius = radius_bbox
        if radius_from_camera:
            # EXPERIMENTAL: see _camera_typical_distance; the bounding sphere when the camera hits nothing.
            var cam_dist = _camera_typical_distance(psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world,
                                                    self.fw, self.fh, sd, cam_percentile)
            if cam_dist > Float32(0):
                self.scene_radius = cam_dist * cam_fraction_mult
            if verbose:
                print("vcm-radius-from-camera: bbox scene_radius=" + String(radius_bbox) + " cam_dist(p=" + String(cam_percentile)
                      + ")=" + String(cam_dist) + " -> effective scene_radius=" + String(self.scene_radius))
        self.scene_radius *= radius_scale
        self.merging = merging
        self.budget = budget
        self.cap = cap
        self.no_keep_mis = no_keep_mis
        self.no_footprint = no_footprint
        self.radius_scale = radius_scale
        self.vcm_lambda = Float32(0)
        self.px_scale = Float32(0)
        self.cam = SIMD[DType.float32, 4](0)
        ref ctx = handle[].ctx
        self.lvc = ctx.enqueue_create_buffer[DType.uint8](max(self.lvc_cap, 1) * size_of[BDPTVertex]())
        self.path_len = ctx.enqueue_create_buffer[DType.uint8](max(self.n_light_paths, 1) * size_of[Int32]())
        self.heads_a = ctx.enqueue_create_buffer[DType.uint8](_VCM_HEADS_SIZE * size_of[Int32]())
        self.heads_b = ctx.enqueue_create_buffer[DType.uint8](_VCM_HEADS_SIZE * size_of[Int32]())
        var stat_n = 2 * _HSIZE if budget else 2
        self.stat_a = ctx.enqueue_create_buffer[DType.float32](stat_n)
        self.stat_b = ctx.enqueue_create_buffer[DType.float32](stat_n)
        self.budget_red = ctx.enqueue_create_buffer[DType.float32](2)
        self.merge_next = ctx.enqueue_create_buffer[DType.uint8](max(self.lvc_cap, 1) * _VCM_MN_STRIDE * size_of[Int32]())
        # CAMIS light records, slot for slot with lvc -- one dummy element when the hybrid is compiled out.
        self.lvc_camis = ctx.enqueue_create_buffer[DType.uint8](max(self.lvc_cap * _CAMIS_LVC_PER_SLOT, 1) * size_of[CamisLightRecord]())
        self.inter_light = ctx.enqueue_create_buffer[DType.uint8](max(self.n_light_paths, 1) * size_of[Intersection]())
        self.inter_cam = ctx.enqueue_create_buffer[DType.uint8](self.n_pix * size_of[Intersection]())
        self.accum = ctx.enqueue_create_buffer[DType.uint8](self.n_pix * 3 * size_of[Float32]())
        self.accum_merge = ctx.enqueue_create_buffer[DType.uint8](self.n_pix * 3 * size_of[Float32]())
        self.albedo_accum = ctx.enqueue_create_buffer[DType.uint8](self.n_pix * 3 * size_of[Float32]())
        self.visit_accum = ctx.enqueue_create_buffer[DType.uint8](self.n_pix * 3 * size_of[Float32]())
        self.r2c = ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
        self.c2w = ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
        self.w2c = ctx.enqueue_create_buffer[DType.uint8](16 * size_of[Float32]())
        self.c2r = ctx.enqueue_create_buffer[DType.uint8](9 * size_of[Float32]())
        self.upload_camera(psc)
        self.clear()

    def upload_camera(mut self, psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin]) raises:
        """The camera's matrices: raster to camera and camera to world for camera paths, their inverses for t=1."""
        var r2c = psc[unsafe_offset=0].raster_to_camera
        var c2w = psc[unsafe_offset=0].camera_to_world
        _vcm_write(self.r2c, r2c, 16)
        _vcm_write(self.c2w, c2w, 16)
        var w2c = unsafe_alloc[Float32](16)
        _ = matrix_invert(c2w, w2c)
        _vcm_write(self.w2c, w2c, 16)
        w2c.unsafe_free()
        # The inverse of raster_to_camera's projective 3x3 (rows 0, 1, 3).
        var a0 = r2c[unsafe_offset=0]; var a1 = r2c[unsafe_offset=4]; var a2 = r2c[unsafe_offset=12]
        var b0 = r2c[unsafe_offset=1]; var b1 = r2c[unsafe_offset=5]; var b2 = r2c[unsafe_offset=13]
        var g0 = r2c[unsafe_offset=2]; var g1 = r2c[unsafe_offset=6]; var g2 = r2c[unsafe_offset=14]
        var d0 = b1*g2 - b2*g1
        var d1 = b0*g2 - b2*g0
        var d2 = b0*g1 - b1*g0
        var det = a0*d0 - a1*d1 + a2*d2
        var idet = Float32(1) / det if abs(det) > Float32(1e-20) else Float32(0)
        var c2r = unsafe_alloc[Float32](9)
        c2r[unsafe_offset=0] =  d0*idet; c2r[unsafe_offset=1] = -(a1*g2 - a2*g1)*idet; c2r[unsafe_offset=2] =  (a1*b2 - a2*b1)*idet
        c2r[unsafe_offset=3] = -d1*idet; c2r[unsafe_offset=4] =  (a0*g2 - a2*g0)*idet; c2r[unsafe_offset=5] = -(a0*b2 - a2*b0)*idet
        c2r[unsafe_offset=6] =  d2*idet; c2r[unsafe_offset=7] = -(a0*g1 - a1*g0)*idet; c2r[unsafe_offset=8] =  (a0*b1 - a1*b0)*idet
        _vcm_write(self.c2r, c2r, 9)
        c2r.unsafe_free()
        self.px_scale = Float32(2.0) * tan(psc[unsafe_offset=0].camera_fov * Float32(3.14159265 / 360.0)) / Float32(self.fh)
        self.cam = SIMD[DType.float32, 4](c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14], Float32(0))

    def clear(mut self) raises:
        """Starts the accumulation over."""
        self.accum.enqueue_fill(UInt8(0))
        self.accum_merge.enqueue_fill(UInt8(0))
        self.albedo_accum.enqueue_fill(UInt8(0))
        self.visit_accum.enqueue_fill(UInt8(0))
        self.vcm_lambda = Float32(0)

    def run_pass(mut self, psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin], si: Int,
                 film: Pointer[Float32, MutUntrackedOrigin], albedo: Pointer[Float32, MutUntrackedOrigin],
                 verbose: Bool = False) raises:
        """Pass `si`: light paths, the merge grid, camera paths into `film`/`albedo` (merging into accum_merge),
        t=1 splats. The merge radius shrinks with si (vcm_merge_radius)."""
        comptime block_size = 256
        ref ctx = self.handle[].ctx
        var gsd = self.handle[].scene_descriptor()
        var radius_i = vcm_merge_radius(self.scene_radius, si)
        # The footprint radius shrinks on the global schedule; 0 (--vcm-no-footprint) is the fixed global radius.
        var radius_0 = vcm_merge_radius(self.scene_radius, 0)
        var vcm_footprint = Float32(0) if self.no_footprint else _VCM_FOOTPRINT_PIXELS * self.radius_scale * self.px_scale * (radius_i / max(radius_0, Float32(1e-20)))
        var heads_a = _i32(self.heads_a)
        var heads_b = _i32(self.heads_b)
        var merge_heads = heads_a if si % 2 == 0 else heads_b
        var merge_next = _i32(self.merge_next)
        # Variance-aware merge MIS reads the OTHER table: last pass's counts, rescaled from its larger cells to this radius.
        var keep = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        var keep_inv_cell = Float32(0)
        var keep_scale = Float32(1)
        if si > 0 and not self.no_keep_mis and self.merging:
            var radius_prev = vcm_merge_radius(self.scene_radius, si - 1)
            var prev_tab = heads_b if si % 2 == 0 else heads_a
            keep = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(prev_tab) + _HSIZE * size_of[Int32]())
            keep_inv_cell = Float32(1.0) / max(radius_prev, Float32(1e-6))
            keep_scale = (radius_i / radius_prev) * (radius_i / radius_prev)
        var merge_r2 = radius_i * radius_i
        var merge_inv_cell = _vcm_grid_inv_cell(gsd, radius_i)
        var merge_norm = Float32(1.0) / (Float32(self.n_light_paths) * PI * max(merge_r2, Float32(1e-12)))
        var eta_vcm = PI * max(merge_r2, Float32(1e-12)) * Float32(self.n_light_paths)
        var mis_vm_weight_factor = eta_vcm if self.merging else Float32(0)
        var mis_vc_weight_factor = Float32(1.0) / eta_vcm if self.merging else Float32(0)
        if not self.merging:   # BDPT: _bdpt_merge_from_cache returns 0 without a grid
            merge_heads = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
            merge_next = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
        var base_seed = psc[unsafe_offset=0].rng_seed
        var pass_seed = base_seed ^ UInt64(si * 2654435761 + 1)
        var vsd = gsd.with_vcm(keep, keep_inv_cell, keep_scale, psc[unsafe_offset=0].max_depth, self.cam, vcm_footprint,
                               radius_i).with_vcm_cap(self.cap)
        var stat_a = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(self.stat_a.unsafe_ptr()))
        var stat_b = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(self.stat_b.unsafe_ptr()))
        if self.budget:
            # Accumulate into one table, read the other, whose lambda the last pass's reduction set (0 at first: the cap).
            if si % 2 == 0:
                self.stat_a.enqueue_fill(Float32(0))
            else:
                self.stat_b.enqueue_fill(Float32(0))
            vsd = vsd.with_vcm_budget(
                (stat_b if si % 2 == 0 else stat_a) if si > 0 else Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
                stat_a if si % 2 == 0 else stat_b, self.vcm_lambda)
        var lvc = self.lvc.unsafe_ptr().unsafe_bitcast[BDPTVertex]()
        var path_len = self.path_len.unsafe_ptr().unsafe_bitcast[Int32]()
        var lvc_camis = self.lvc_camis.unsafe_ptr().unsafe_bitcast[CamisLightRecord]()
        var inter_light = self.inter_light.unsafe_ptr().unsafe_bitcast[Intersection]()
        var filt = film_filter_of(psc[unsafe_offset=0].filter_type, psc[unsafe_offset=0].filter_sigma,
                                  psc[unsafe_offset=0].filter_support_x, psc[unsafe_offset=0].filter_support_y)
        var grid_light = ceildiv(max(self.n_light_paths, 1), block_size)
        ctx.enqueue_function[_bdpt_emit_light_paths_gpu](
            lvc, path_len, lvc_camis, mis_vc_weight_factor, mis_vm_weight_factor, inter_light,
            Int64(self.n_light_paths), self.default_emit_med, pass_seed, Int64(si), _f32(self.c2w), self.px_scale, vsd,
            grid_dim=grid_light, block_dim=block_size)
        if self.merging:
            var grid_merge_ins = ceildiv(max(self.lvc_cap, 1), block_size)
            ctx.enqueue_function[bdpt_merge_grid_reset_gpu](
                merge_heads, Int64(_HSIZE), grid_dim=ceildiv(_HSIZE, block_size), block_dim=block_size)
            ctx.enqueue_function[bdpt_merge_grid_count_gpu](
                lvc, path_len, Int64(self.lvc_cap), merge_heads, merge_inv_cell, grid_dim=grid_merge_ins, block_dim=block_size)
            ctx.enqueue_function[bdpt_merge_grid_insert_gpu](
                lvc, path_len, Int64(self.lvc_cap), merge_next, merge_heads, merge_inv_cell, vsd,
                grid_dim=grid_merge_ins, block_dim=block_size)
        ctx.enqueue_function[_bdpt_camera_connect_gpu](
            film, _f32(self.accum_merge), albedo, _f32(self.visit_accum), Int64(self.n_pix), Int64(self.fw),
            _f32(self.r2c), _f32(self.c2w), self.inter_cam.unsafe_ptr().unsafe_bitcast[Intersection](),
            lvc, path_len, lvc_camis, merge_next, merge_heads, merge_inv_cell, merge_r2, merge_norm, self.px_scale,
            mis_vc_weight_factor, mis_vm_weight_factor, Float32(self.n_light_paths), filt, base_seed, Int64(si), vsd,
            grid_dim=ceildiv(self.n_pix, block_size), block_dim=block_size)
        # t=1 light tracing, after the camera paths on the same stream.
        ctx.enqueue_function[_bdpt_splat_light_paths_gpu](
            film, lvc, path_len, Int64(self.n_light_paths), inter_light, _f32(self.w2c), _f32(self.c2r), _f32(self.c2w),
            Int64(self.fw), Int64(self.fh), self.px_scale, mis_vm_weight_factor, filt, vsd, lvc_camis,
            grid_dim=grid_light, block_dim=block_size)
        if self.budget:
            # lambda for the next pass: the budget is the merge work the fixed cap would have spent, sum Q min(n, cap).
            self.budget_red.enqueue_fill(Float32(0))
            var budget_red = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(self.budget_red.unsafe_ptr()))
            ctx.enqueue_function[vcm_budget_reduce_gpu](
                vsd.vcmStatOut, merge_heads, budget_red,
                grid_dim=_VCM_BUDGET_REDUCE_THREADS // block_size, block_dim=block_size)
            with self.budget_red.map_to_host() as red:
                var s_sum = red.unsafe_ptr()[unsafe_offset=0]
                var b_sum = red.unsafe_ptr()[unsafe_offset=1]
                self.vcm_lambda = b_sum / s_sum if s_sum > Float32(0) else Float32(0)
            if verbose:
                print("VCM (GPU): budget lambda " + String(self.vcm_lambda))


def vcm_render_gpu(
    handlePtr: Pointer[GpuSceneHandle, MutUntrackedOrigin],
    psc:      Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd:       SceneView,
    n_spp:    Int,
    n_photons_req: Int,
    no_denoise: Bool,
    verbose:  Bool,
    vcm_budget: Bool = False,
    vcm_cap: Int32 = Int32(0),
    vcm_no_keep_mis: Bool = False,
    vcm_radius_from_camera: Bool = False,
    vcm_radius_cam_percentile: Float32 = Float32(0.5),
    vcm_radius_cam_fraction_mult: Float32 = Float32(1.0),
    vcm_no_footprint: Bool = False,   # the "naive VCM" baseline: one fixed global radius
    vcm_radius_scale: Float32 = Float32(1.0),   # the whole merge radius: ceiling, grid cell, MIS density, footprint
) -> Int32:
    """A batch VCM render: n_spp passes of VcmGpu, then the image written as the scene asks."""
    var fw = Int(psc[unsafe_offset=0].film_w)
    var fh = Int(psc[unsafe_offset=0].film_h)
    var n_pix = fw * fh
    print("VCM (GPU): " + String(fw) + "x" + String(fh) + "  " + String(n_spp) + " spp  "
          + String(max(n_photons_req, n_pix)) + " light paths/pass")
    var ret = Int32(0)
    comptime if has_accelerator():
        try:
            var v = VcmGpu(handlePtr, psc, sd, n_photons_req, True, vcm_budget, vcm_cap, vcm_no_keep_mis,
                           vcm_radius_from_camera, vcm_radius_cam_percentile, vcm_radius_cam_fraction_mult,
                           vcm_no_footprint, vcm_radius_scale, verbose)
            handlePtr[].cam_fp = camera_footprint(psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world,
                                                  fw, fh, n_spp)
            var prog = Progress(n_spp, "spp", quiet=verbose)
            for si in range(n_spp):
                v.run_pass(psc, si, _f32(v.accum), _f32(v.albedo_accum), verbose)
                if verbose:
                    print("VCM (GPU): sample " + String(si + 1) + "/" + String(n_spp))
                handlePtr[].ctx.synchronize()   # the progress line reports finished work
                prog.update(si + 1)
            _ = prog.finish()
            comptime if _VCM_VISIT_INSTRUMENT:   # benchmark only: the kernels count merge visits only then
                var visit_sidecar = _sidecar_name(psc[unsafe_offset=0].film_filename, ".mergevisits.exr")
                with v.visit_accum.map_to_host() as vhost:
                    var vsrc = vhost.unsafe_ptr().unsafe_bitcast[Float32]()
                    _ = _write_channels(visit_sidecar, vsrc, Int32(fw), Int32(fh), Int32(3),
                        "naive.Y,footprint.Y,thinning.Y", Int32(fw), Int32(fh), Int32(0), Int32(0))
                visit_sidecar.unsafe_free()
            # Connect + splats and merging stay apart: finish_render denoises the first and adds the second after.
            var pixels = unsafe_alloc[Float32](n_pix * 3)
            var caustic_pixels = unsafe_alloc[Float32](n_pix * 3)
            with v.accum.map_to_host() as host_buf:
                with v.accum_merge.map_to_host() as host_buf_m:
                    var src = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                    var src_m = host_buf_m.unsafe_ptr().unsafe_bitcast[Float32]()
                    var inv_spp = psc[unsafe_offset=0].film_iso / Float32(100) / Float32(n_spp)
                    for i in range(n_pix):
                        var conn = RGB(src[unsafe_offset=i*3], src[unsafe_offset=i*3+1], src[unsafe_offset=i*3+2])
                        var mrg = RGB(src_m[unsafe_offset=i*3], src_m[unsafe_offset=i*3+1], src_m[unsafe_offset=i*3+2])
                        var (c, m) = _vcm_finalize_one_pixel(conn, mrg, inv_spp)
                        pixels[unsafe_offset=i*3] = c.r; pixels[unsafe_offset=i*3+1] = c.g; pixels[unsafe_offset=i*3+2] = c.b
                        caustic_pixels[unsafe_offset=i*3] = m.r; caustic_pixels[unsafe_offset=i*3+1] = m.g; caustic_pixels[unsafe_offset=i*3+2] = m.b
            var albedo_pixels = unsafe_alloc[Float32](n_pix * 3)
            with v.albedo_accum.map_to_host() as host_buf:
                var src = host_buf.unsafe_ptr().unsafe_bitcast[Float32]()
                var inv_spp_alb = Float32(1) / Float32(n_spp)
                for i in range(n_pix * 3):
                    albedo_pixels[unsafe_offset=i] = src[unsafe_offset=i] * inv_spp_alb
            _ = finish_render(psc, sd, pixels, albedo_pixels, no_denoise, caustic_pixels)
            pixels.unsafe_free(); caustic_pixels.unsafe_free(); albedo_pixels.unsafe_free()
        except e:
            print("VCM GPU render failed: " + String(e))
            ret = Int32(-1)
    else:
        print("VCM GPU: no accelerator")
        ret = Int32(-1)
    return ret
