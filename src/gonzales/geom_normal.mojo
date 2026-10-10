# Geometric normal of a hit (triangle, instanced triangle or analytic sphere).
# Lives apart from sppm.mojo so shading.mojo's shadow-ray media walk can use it.
from .geometry import Point3f, Vec3f, cross, dot, _is_real_ptr
from .primitives import Intersection, TriangleMesh, Instance, Sphere, sphere_outward_normal
from .transform import transform_normal, Mat4
from std.math import sqrt

@always_inline
def _geom_normal(
    inter: Intersection,
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    instances: Pointer[Instance, MutUntrackedOrigin] = Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(),
    spheres: Pointer[Sphere, MutUntrackedOrigin] = Pointer[Sphere, MutUntrackedOrigin].unsafe_dangling(),
    hit: Vec3f = Vec3f(Float32(0), Float32(0), Float32(0)),
) -> Vec3f:
    """Normalized geometric normal from triangle cross product. If this hit
    came from inside an instanced BLAS (primId.instanceIdx >= 0 — see
    bvh.mojo's traverse_bvh2_core type==6 branch), the mesh data is in that
    instance's object space, so the normal is transformed to world space
    before returning (the hit *point*, elsewhere computed as
    ray_org + ray_dir*tHit, needs no such fixup — see transform.mojo's
    transform_normal for why).

    `spheres`/`hit` give analytic spheres (primId.type == 4) their exact
    outward normal, mirroring `_shading_normal_at` right below (which this
    function predates and was never given the same treatment). Without
    them, every caller either had to special-case type==4 itself before
    calling this -- duplicating the same sphere_outward_normal(hit, center)
    one-liner at each of the (as of 2026-09-15) 21 call sites across
    bdpt_*.mojo/sppm.mojo -- or, at 2 call sites that omitted the guard
    entirely, silently got the +Y placeholder below: a live bug (a diffuse
    analytic sphere corrupted both SPPM's visible-point normal and its
    photon-bounce normal identically to how a dielectric sphere corrupted
    refraction before _shading_normal_at's own fix). See
    project_mesh_only_geometry_assumption memory."""
    var mi: Int; var bv: Int
    if inter.primId.type == 0:
        mi = Int(inter.primId.id1); bv = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mi = Int(inter.primId.id2 >> 32); bv = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    elif inter.primId.type == Int8(4) and _is_real_ptr[Sphere](spheres):
        return sphere_outward_normal(Point3f(hit[0], hit[1], hit[2]), spheres[unsafe_offset=Int(inter.primId.id1)].center)
    else:
        return Vec3f(Float32(0), Float32(1), Float32(0))
    var m = meshes[unsafe_offset=mi]
    var v0 = Int(m.vertexIndices[unsafe_offset=bv])
    var v1 = Int(m.vertexIndices[unsafe_offset=bv + 1])
    var v2 = Int(m.vertexIndices[unsafe_offset=bv + 2])
    var p0 = Vec3f(m.points[unsafe_offset=v0*4], m.points[unsafe_offset=v0*4+1], m.points[unsafe_offset=v0*4+2])
    var p1 = Vec3f(m.points[unsafe_offset=v1*4], m.points[unsafe_offset=v1*4+1], m.points[unsafe_offset=v1*4+2])
    var p2 = Vec3f(m.points[unsafe_offset=v2*4], m.points[unsafe_offset=v2*4+1], m.points[unsafe_offset=v2*4+2])
    var n = cross(p1 - p0, p2 - p0)
    if inter.primId.instanceIdx >= Int32(0):
        n = transform_normal(Mat4(instances[unsafe_offset=Int(inter.primId.instanceIdx)].world_to_obj()), n)
    var l = dot(n, n)
    if l > Float32(0.0):
        n = n * (Float32(1.0) / sqrt(l))
    return n
