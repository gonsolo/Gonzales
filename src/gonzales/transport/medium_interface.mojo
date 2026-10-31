"""Which medium a ray is in after crossing a surface."""
from gonzales.scene.accel.bvh import SceneView
from gonzales.core.geometry import Vec3f
from gonzales.scene.shapes.geom_normal import _geom_normal
from gonzales.scene.appearance.materials import Material
from gonzales.scene.shapes.primitives import Intersection


@always_inline
def default_emit_medium(ref sd: SceneView) -> Int32:
    """The medium light paths start in: the first interface's valid outside medium."""
    if Int(sd.mediumCount) == 0:
        return Int32(-1)
    for mi in range(Int(sd.mediumIfaceCount)):
        var outside = sd.mediumInterfaces[unsafe_offset=mi].outside_medium_idx
        if Int(outside) >= 0:
            return outside
    return Int32(-1)


@always_inline
def medium_after(w: Vec3f, cur: Int32, inter: Intersection, mat: Material, ref sd: SceneView, hit: Vec3f) -> Int32:
    """The medium a ray leaving a surface hit in direction `w` travels in; `cur` when the surface binds none.
    Every integrator uses this, with the OUTGOING direction, so a reflection stays on its side."""
    if mat.medium_interface_idx < Int32(0) or sd.mediumIfaceCount == Int64(0):
        return cur
    var iface = sd.mediumInterfaces[unsafe_offset=Int(mat.medium_interface_idx)]
    return iface.toward(w, _geom_normal(inter, sd.meshes, sd.instances, sd.spheres, hit))
