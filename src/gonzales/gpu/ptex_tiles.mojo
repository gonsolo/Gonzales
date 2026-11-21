"""Ptex lookups as a batch kernel: the GPU check of tiled paging (test_ptex_tiles_gpu)."""
from max.gpu import block_dim, block_idx, thread_idx
from gonzales.scene.appearance.materials import GpuTexture
from gonzales.transport.texturing import _sample_ptex


def ptex_sample_batch_gpu(data: Pointer[UInt8, MutUntrackedOrigin], lut: Pointer[Float32, MutUntrackedOrigin],
                          pool: Pointer[UInt8, MutUntrackedOrigin], n_faces: Int64, udim_cols: Int32,
                          uvf: Pointer[Float32, MutUntrackedOrigin], res: Pointer[Float32, MutUntrackedOrigin], n: Int64):
    """Lookup i at (u, v, footprint) = uvf[3i..3i+2] of a paged texture, its RGB to res[3i..3i+2]."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(n):
        return
    var tex = GpuTexture(data, lut, pool, Int32(n_faces), Int32(0), -udim_cols, Int32(3), Int32(GpuTexture.FORMAT_PTEX), GpuTexture.UV_IDENTITY)
    var c = _sample_ptex(tex, 0, uvf[unsafe_offset=3 * i], uvf[unsafe_offset=3 * i + 1], uvf[unsafe_offset=3 * i + 2])
    res[unsafe_offset=3 * i] = c.r; res[unsafe_offset=3 * i + 1] = c.g; res[unsafe_offset=3 * i + 2] = c.b
