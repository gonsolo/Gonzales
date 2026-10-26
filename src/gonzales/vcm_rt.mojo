# VCM through the RT cores or Vulkan RT: the wavefront driver's intersect and shadow-ray steps.

from max.gpu import block_idx, thread_idx, block_dim, MAX_THREADS_PER_BLOCK_METADATA
from std.utils import StaticTuple
from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from .bvh import SceneView
from .primitives import Intersection
from .gpu_wavefront import (
    vulkaninterop_unpack_results_kernel, rtcore_trace_unpack_gpu, rtcore_spheres_from_rays_gpu,
    rtcore_alpha_passes, rt_no_ids, rt_no_paths, rt_no_meshes,
)
from .rtcore import rtcore_active, rtcore_alpha_enabled, rtcore_trace_interop
from .vulkaninterop import VulkanInteropRtSceneHandle, vulkaninterop_rt_trace
from max.gpu.host._nvidia_cuda import CUDA
from .bdpt_camera import VCMCameraPathState
from .bdpt_light import VCMLightPathState

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def vulkaninterop_pack_light_rays_kernel(
    states: Pointer[VCMLightPathState, MutUntrackedOrigin],
    rays: Pointer[Float32, MutUntrackedOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var ro = states[unsafe_offset=tid].ro
    var rd = states[unsafe_offset=tid].rd
    var idx = tid * 8
    rays[unsafe_offset=idx + 0] = ro.x
    rays[unsafe_offset=idx + 1] = ro.y
    rays[unsafe_offset=idx + 2] = ro.z
    rays[unsafe_offset=idx + 3] = Float32(1e-4)
    rays[unsafe_offset=idx + 4] = rd.x
    rays[unsafe_offset=idx + 5] = rd.y
    rays[unsafe_offset=idx + 6] = rd.z
    rays[unsafe_offset=idx + 7] = Float32(1.0e8) if states[unsafe_offset=tid].active != Int8(0) else Float32(0.0)   # a finished path traces a zero-length ray

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def vulkaninterop_pack_camera_rays_kernel(
    states: Pointer[VCMCameraPathState, MutUntrackedOrigin],
    rays: Pointer[Float32, MutUntrackedOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var ro = states[unsafe_offset=tid].ro
    var rd = states[unsafe_offset=tid].rd
    var idx = tid * 8
    rays[unsafe_offset=idx + 0] = ro.x
    rays[unsafe_offset=idx + 1] = ro.y
    rays[unsafe_offset=idx + 2] = ro.z
    rays[unsafe_offset=idx + 3] = Float32(1e-4)
    rays[unsafe_offset=idx + 4] = rd.x
    rays[unsafe_offset=idx + 5] = rd.y
    rays[unsafe_offset=idx + 6] = rd.z
    rays[unsafe_offset=idx + 7] = Float32(1.0e8) if states[unsafe_offset=tid].active != Int8(0) else Float32(0.0)   # a finished path traces a zero-length ray

def _vcm_rt_trace_paths[camera: Bool](
    ctx: DeviceContext,
    state_buf: DeviceBuffer[DType.uint8],
    inter_buf: DeviceBuffer[DType.uint8],
    interop_scene: VulkanInteropRtSceneHandle,
    interop_rays_buf: DeviceBuffer[DType.float32],
    interop_results_buf: DeviceBuffer[DType.float32],
    mesh_material_idx_buf: DeviceBuffer[DType.uint8],
    mesh_al_idx_buf: DeviceBuffer[DType.uint8],
    n_meshes: Int,
    n_total: Int,
    # --rt-hardware (rtcore_trace_unpack_gpu): scene view, alpha scratch, instance decode.
    sd: SceneView,
    rt_scratch_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    instance_base_mesh_buf: Optional[DeviceBuffer[DType.uint8]] = None,
) raises:
    comptime block_size = 256
    var grid = ceildiv(n_total, block_size)

    var rays = interop_rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    comptime if camera:
        ctx.enqueue_function[vulkaninterop_pack_camera_rays_kernel](
            state_buf.unsafe_ptr().unsafe_bitcast[VCMCameraPathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            rays, Int64(n_total), grid_dim=grid, block_dim=block_size)
    else:
        ctx.enqueue_function[vulkaninterop_pack_light_rays_kernel](
            state_buf.unsafe_ptr().unsafe_bitcast[VCMLightPathState]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
            rays, Int64(n_total), grid_dim=grid, block_dim=block_size)

    if Int(rtcore_active()) != 0 and rt_scratch_buf:
        rtcore_trace_unpack_gpu(ctx, inter_buf, sd, interop_rays_buf, interop_results_buf, rt_scratch_buf.value(),
                                mesh_material_idx_buf, mesh_al_idx_buf, n_meshes, n_total, instance_base_mesh_buf)
        if Int(sd.sphereCount) > 0:
            ctx.enqueue_function[rtcore_spheres_from_rays_gpu](
                interop_rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                sd.spheres, Int64(Int(sd.sphereCount)), Int64(n_total),
                grid_dim=grid, block_dim=block_size,
            )
        return

    var cuda_stream = CUDA(ctx.stream())
    _ = vulkaninterop_rt_trace(interop_scene, Int32(n_total), cuda_stream)

    # GPU kernel dispatch has no default-argument fill-in (unlike ordinary
    # Mojo calls) -- must pass instance_base_mesh explicitly. VCM's Vulkan
    var instance_base_ptr = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
    if instance_base_mesh_buf:
        instance_base_ptr = instance_base_mesh_buf.value().unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    ctx.enqueue_function[vulkaninterop_unpack_results_kernel](
        interop_results_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        inter_buf.unsafe_ptr().unsafe_bitcast[Intersection]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        mesh_material_idx_buf.unsafe_ptr().unsafe_bitcast[Int64]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        mesh_al_idx_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(n_meshes),
        Int64(n_total),
        instance_base_ptr,
        rt_no_ids(), rt_no_paths(), rt_no_meshes(),
        grid_dim=grid, block_dim=block_size,
    )

@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def vulkaninterop_pack_all_shadow_rays_kernel(
    # Perf (2026-07-13, task #163 stage 5 follow-up): packs ALL
    # n_pix*_BDPT_MAX_VERTS shadow-ray slots in ONE dispatch instead of
    shadow_rays: Pointer[Float32, MutUntrackedOrigin],
    shadow_valid: Pointer[Int8, MutUntrackedOrigin],
    out_rays: Pointer[Float32, MutUntrackedOrigin],
    count_dp: Int64,
):
    var count = Int(count_dp)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= count:
        return
    var idx8 = tid * 8
    if shadow_valid[unsafe_offset=tid] == Int8(0):
        out_rays[unsafe_offset=idx8 + 0] = Float32(0)
        out_rays[unsafe_offset=idx8 + 1] = Float32(0)
        out_rays[unsafe_offset=idx8 + 2] = Float32(0)
        out_rays[unsafe_offset=idx8 + 3] = Float32(0)
        out_rays[unsafe_offset=idx8 + 4] = Float32(0)
        out_rays[unsafe_offset=idx8 + 5] = Float32(0)
        out_rays[unsafe_offset=idx8 + 6] = Float32(1)
        out_rays[unsafe_offset=idx8 + 7] = Float32(0)
        return
    out_rays[unsafe_offset=idx8 + 0] = shadow_rays[unsafe_offset=idx8 + 0]
    out_rays[unsafe_offset=idx8 + 1] = shadow_rays[unsafe_offset=idx8 + 1]
    out_rays[unsafe_offset=idx8 + 2] = shadow_rays[unsafe_offset=idx8 + 2]
    out_rays[unsafe_offset=idx8 + 3] = shadow_rays[unsafe_offset=idx8 + 3]
    out_rays[unsafe_offset=idx8 + 4] = shadow_rays[unsafe_offset=idx8 + 4]
    out_rays[unsafe_offset=idx8 + 5] = shadow_rays[unsafe_offset=idx8 + 5]
    out_rays[unsafe_offset=idx8 + 6] = shadow_rays[unsafe_offset=idx8 + 6]
    out_rays[unsafe_offset=idx8 + 7] = shadow_rays[unsafe_offset=idx8 + 7]

def vulkaninterop_rt_traverse_shadow_gpu(
    # Perf (2026-07-13): pack -> trace only, no unpack step -- the caller's
    # resolve_shadow_connect_gpu reads interop_results_buf's raw float
    ctx: DeviceContext,
    shadow_rays_buf: DeviceBuffer[DType.uint8],
    shadow_valid_buf: DeviceBuffer[DType.uint8],
    interop_scene: VulkanInteropRtSceneHandle,
    interop_rays_buf: DeviceBuffer[DType.float32],
    count: Int,
    # --rt-hardware: results come back in the interop layout (resolve_shadow_connect_gpu reads it either way); alpha scenes
    # re-trace past rejected hits first.
    interop_results_buf: Optional[DeviceBuffer[DType.float32]] = None,
    rt_scratch_buf: Optional[DeviceBuffer[DType.uint8]] = None,
    sd: Optional[SceneView] = None,
    n_meshes: Int = 0,
    instance_base_mesh_buf: Optional[DeviceBuffer[DType.uint8]] = None,
) raises:
    comptime block_size = 256
    var grid = ceildiv(count, block_size)

    ctx.enqueue_function[vulkaninterop_pack_all_shadow_rays_kernel](
        shadow_rays_buf.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        shadow_valid_buf.unsafe_ptr().unsafe_bitcast[Int8]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        interop_rays_buf.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(count),
        grid_dim=grid, block_dim=block_size,
    )

    var cuda_stream = CUDA(ctx.stream())
    if Int(rtcore_active()) != 0 and interop_results_buf and rt_scratch_buf and sd:
        _ = rtcore_trace_interop(rtcore_active(), UInt64(Int(interop_rays_buf.unsafe_ptr())), UInt64(Int(interop_results_buf.value().unsafe_ptr())), Int32(count), cuda_stream)
        if Int(rtcore_alpha_enabled()) != 0:
            var instance_base_ptr = Pointer[Int32, MutUntrackedOrigin].unsafe_dangling()
            if instance_base_mesh_buf:
                instance_base_ptr = instance_base_mesh_buf.value().unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
            rtcore_alpha_passes(ctx, interop_rays_buf, interop_results_buf.value(), rt_scratch_buf.value(), sd.value().meshes, n_meshes, instance_base_ptr, count)
    else:
        _ = vulkaninterop_rt_trace(interop_scene, Int32(count), cuda_stream)


struct VcmRt(Movable):
    """The interop scene and buffers the wavefront VCM driver traces with; `active` False: the software BVH."""
    var active: Bool
    var scene: VulkanInteropRtSceneHandle
    var rays: Optional[DeviceBuffer[DType.float32]]
    var results: Optional[DeviceBuffer[DType.float32]]
    var mesh_material_idx: Optional[DeviceBuffer[DType.uint8]]
    var mesh_al_idx: Optional[DeviceBuffer[DType.uint8]]
    var n_meshes: Int
    var rt_scratch: Optional[DeviceBuffer[DType.uint8]]          # --rt-hardware alpha scratch
    var instance_base_mesh: Optional[DeviceBuffer[DType.uint8]]  # the instance decode table

    def __init__(out self):
        self.active = False
        self.scene = VulkanInteropRtSceneHandle.unsafe_dangling()
        self.rays = None
        self.results = None
        self.mesh_material_idx = None
        self.mesh_al_idx = None
        self.n_meshes = 0
        self.rt_scratch = None
        self.instance_base_mesh = None

    def trace[camera: Bool](self, ctx: DeviceContext, states: DeviceBuffer[DType.uint8], inter: DeviceBuffer[DType.uint8],
                            n: Int, sd: SceneView) raises:
        """One intersect step of the light (camera False) or camera paths."""
        _vcm_rt_trace_paths[camera](ctx, states, inter, self.scene, self.rays.value(), self.results.value(),
                                    self.mesh_material_idx.value(), self.mesh_al_idx.value(), self.n_meshes, n, sd,
                                    self.rt_scratch, self.instance_base_mesh)

    def trace_shadow(self, ctx: DeviceContext, shadow_rays: DeviceBuffer[DType.uint8], shadow_valid: DeviceBuffer[DType.uint8],
                     n: Int, sd: SceneView) raises:
        vulkaninterop_rt_traverse_shadow_gpu(ctx, shadow_rays, shadow_valid, self.scene, self.rays.value(), n,
                                             self.results, self.rt_scratch, sd, self.n_meshes, self.instance_base_mesh)
