"""The path tracer's medium kernels."""
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from std.utils import StaticTuple
from .bvh import SceneView
from .geometry import _is_real_ptr
from .primitives import Intersection
from .render_state import PathState
from .restir_vol import VolReservoir
from max.gpu import block_dim, block_idx, thread_idx
from .volume_path import _sample_medium_core, path_medium_after_shading


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def update_medium_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
):
    """path_medium_after_shading for every live path."""
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    var path_ptr = paths.unsafe_offset(tid)
    if path_ptr[].active == 0:
        return
    path_medium_after_shading(path_ptr, intersections[unsafe_offset=tid], sd)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256)))
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))    # NVIDIA keeps its old register budget; AMD ignores this
def sample_medium_gpu(
    paths: Pointer[PathState, MutUntrackedOrigin],
    intersections: Pointer[Intersection, MutUntrackedOrigin],
    sd: SceneView,
    count_dp: Int64,
    live_idx: Pointer[Int32, MutUntrackedOrigin],
    live_count: Pointer[Int32, MutUntrackedOrigin],
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
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= Int(live_count[unsafe_offset=0]):
        return
    tid = Int(live_idx[unsafe_offset=tid])
    if paths[unsafe_offset=tid].sss.mesh >= Int32(0):
        return   # sss_walk_gpu walks these
    # tid IS the pixel index here only when use_vol_restir=1 -- that only
    # ever comes from gpu_render_sample (one path per pixel per dispatch),
    # mirroring shade_diffuse_gpu's identical restir_has_state contract.
    var vol_has_state = use_vol_restir != Int32(0) and _is_real_ptr(vol_read)
    _sample_medium_core(
        paths, intersections, tid, sd,
        vol_read=vol_read,
        vol_write=vol_write,
        pixel_idx=tid if vol_has_state else -1,
        vol_used=vol_used,
        vol_gbuf_depth=vol_gbuf_depth,
        vol_gbuf_world_pos=vol_gbuf_world_pos,
        vol_frame_w=vol_frame_w,
        vol_frame_h=vol_frame_h,
    )
