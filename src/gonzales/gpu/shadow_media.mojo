"""Media shadow rays as a batch kernel: the GPU check of transport/shadow_media.mojo."""
from max.gpu import block_dim, block_idx, thread_idx
from gonzales.core.geometry import Vec3f
from gonzales.core.rng import PCG32
from gonzales.core.spectrum import sample_wavelengths
from gonzales.scene.accel.bvh import SceneView
from gonzales.scene.shapes.motion import MotionTable
from gonzales.transport.shadow_media import shadow_transmittance


def shadow_media_ray(sd: SceneView, rays: Pointer[Float32, MutUntrackedOrigin], i: Int) -> Float32:
    """Ray i (origin, direction, tmax: 7 floats) from outside every medium: its first transmittance lane."""
    var o = Vec3f(rays[unsafe_offset=7 * i], rays[unsafe_offset=7 * i + 1], rays[unsafe_offset=7 * i + 2])
    var d = Vec3f(rays[unsafe_offset=7 * i + 3], rays[unsafe_offset=7 * i + 4], rays[unsafe_offset=7 * i + 5])
    var pcg = PCG32(UInt64(i), UInt64(1))
    var tr = shadow_transmittance(o, d, rays[unsafe_offset=7 * i + 6], Int32(-1), sd.bvh2Nodes, sd.primIds, sd.meshes, sd.curves,
                                  sd.blasNodesArr, sd.blasPrimIdsArr, sd.instances, sd.spheres, Int(sd.sphereCount),
                                  sd.materials, sd.mediums, sd.mediumInterfaces, sd.grids, sd.nvdbGrids,
                                  sd.spectral, sample_wavelengths(Float32(0.3)), pcg, motion=MotionTable.none())
    return tr.v0


def shadow_media_batch_gpu(sd: SceneView, rays: Pointer[Float32, MutUntrackedOrigin], res: Pointer[Float32, MutUntrackedOrigin], n: Int64):
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i < Int(n):
        res[unsafe_offset=i] = shadow_media_ray(sd, rays, i)
