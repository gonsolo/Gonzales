# Path guiding on the GPU: the SD-tree (guide.mojo) lives on the host between training iterations and in device
# memory during one. The host builds and refines it; this file moves it.
#
#   gpu_guide_begin   upload the tree to read from (none in the first iteration) and the zeroed host shard to record
#                     into; the shade kernels see both through ShadeContext.guide / guide_write.
#   gpu_guide_end     download the shard into the host copy and release the device copies. The training loop
#                     (guide.mojo's train_guided) folds it into the tree.
#
# One shared shard suffices: guide_record adds atomically, so the CPU driver's per-thread shards are not needed.

from std.memory import Pointer
from std.sys.info import size_of
from max.gpu.host import DeviceBuffer
from .gpu_scene import GpuSceneHandle
from .guide import GuideGrid, guide_is_active, null_guide


def _device_copy[T: AnyType](
    mut handle: GpuSceneHandle, src: Pointer[T, MutUntrackedOrigin], count: Int,
) raises -> Pointer[T, MutUntrackedOrigin]:
    """Upload `count` elements into a new device buffer the handle keeps alive; returns its device address."""
    var buf = handle.ctx.enqueue_create_buffer[DType.uint8](count * size_of[T]())
    handle.ctx.enqueue_copy(buf, src.unsafe_bitcast[UInt8]())
    var dev = buf.unsafe_ptr().unsafe_bitcast[T]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    handle.guide_bufs.append(buf^)
    return dev


def gpu_guide_begin(
    handle: Pointer[GpuSceneHandle, MutUntrackedOrigin], read_tree: GuideGrid, shard: GuideGrid,
) raises:
    """Make `read_tree` (null_guide() in the first iteration) readable and the host `shard` writable by the kernels."""
    handle[].guide_bufs = List[DeviceBuffer[DType.uint8]]()
    if guide_is_active(read_tree):
        var rs = _device_copy(handle[], read_tree.snodes, Int(read_tree.n_snodes))
        var rd = _device_copy(handle[], read_tree.dnodes, Int(read_tree.n_dnodes))
        handle[].guide_read = GuideGrid(snodes=rs, n_snodes=read_tree.n_snodes, dnodes=rd, n_dnodes=read_tree.n_dnodes, bounds=read_tree.bounds)
    else:
        handle[].guide_read = null_guide()
    var ws = _device_copy(handle[], shard.snodes, Int(shard.n_snodes))
    var wd = _device_copy(handle[], shard.dnodes, Int(shard.n_dnodes))
    handle[].guide_write = GuideGrid(snodes=ws, n_snodes=shard.n_snodes, dnodes=wd, n_dnodes=shard.n_dnodes, bounds=shard.bounds)
    handle[].ctx.synchronize()


def gpu_guide_end(handle: Pointer[GpuSceneHandle, MutUntrackedOrigin], shard: GuideGrid) raises:
    """Download what the iteration recorded into the host `shard` and release the device copies."""
    handle[].ctx.synchronize()
    var n = len(handle[].guide_bufs)               # the shard's snodes and dnodes are always the last two buffers
    handle[].ctx.enqueue_copy(shard.snodes.unsafe_bitcast[UInt8](), handle[].guide_bufs[n - 2])
    handle[].ctx.enqueue_copy(shard.dnodes.unsafe_bitcast[UInt8](), handle[].guide_bufs[n - 1])
    handle[].ctx.synchronize()
    handle[].guide_read = null_guide()
    handle[].guide_write = null_guide()
    handle[].guide_bufs = List[DeviceBuffer[DType.uint8]]()
