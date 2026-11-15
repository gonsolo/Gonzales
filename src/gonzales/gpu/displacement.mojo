"""Displaced-triangle intersection as a batch kernel: the GPU check of scene/shapes/displacement.mojo, and its benchmark."""
from max.gpu import block_dim, block_idx, thread_idx
from gonzales.core.geometry import Vec3f
from gonzales.scene.shapes.displacement import DisplacedMesh, intersect_displaced


def intersect_displaced_batch_gpu(dm: Pointer[DisplacedMesh, MutUntrackedOrigin], rays: Pointer[Float32, MutUntrackedOrigin],
                                  res: Pointer[Float32, MutUntrackedOrigin], n: Int64):
    """Ray i (origin, direction: 6 floats) against triangle 0 of dm: (hit, t) at res[2i]."""
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= Int(n):
        return
    var o = Vec3f(rays[unsafe_offset=6 * i], rays[unsafe_offset=6 * i + 1], rays[unsafe_offset=6 * i + 2])
    var d = Vec3f(rays[unsafe_offset=6 * i + 3], rays[unsafe_offset=6 * i + 4], rays[unsafe_offset=6 * i + 5])
    var m = dm[unsafe_offset=0]
    var h = intersect_displaced(m.tris[unsafe_offset=0], m.rm, o, d, Float32(1e30))
    res[unsafe_offset=2 * i] = Float32(1) if h.hit else Float32(0)
    res[unsafe_offset=2 * i + 1] = h.t
