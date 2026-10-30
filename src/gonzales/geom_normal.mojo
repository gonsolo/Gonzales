# Geometric normal of a hit (triangle, instanced triangle or analytic sphere).
# Lives apart from sppm.mojo so shading.mojo's shadow-ray media walk can use it.
from .geometry import Point3f, Vec3f, cross, dot, _is_real_ptr
from .primitives import Intersection, TriangleMesh, Instance, Sphere, sphere_outward_normal, mesh_vi
from .primitives import mesh_normal, mesh_pos
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
    var v0 = Int(mesh_vi(m, bv))
    var v1 = Int(mesh_vi(m, bv + 1))
    var v2 = Int(mesh_vi(m, bv + 2))
    var p0 = mesh_pos(m, v0)
    var p1 = mesh_pos(m, v1)
    var p2 = mesh_pos(m, v2)
    var n = cross(p1 - p0, p2 - p0)
    if inter.primId.instanceIdx >= Int32(0):
        n = transform_normal(Mat4(instances[unsafe_offset=Int(inter.primId.instanceIdx)].world_to_obj()), n)
    var l = dot(n, n)
    if l > Float32(0.0):
        n = n * (Float32(1.0) / sqrt(l))
    return n


@always_inline
def _shading_normal_at(
    inter: Intersection,
    meshes: Pointer[TriangleMesh, MutUntrackedOrigin],
    instances: Pointer[Instance, MutUntrackedOrigin] = Pointer[Instance, MutUntrackedOrigin].unsafe_dangling(),
    spheres: Pointer[Sphere, MutUntrackedOrigin] = Pointer[Sphere, MutUntrackedOrigin].unsafe_dangling(),
    hit: Point3f = Point3f(Float32(0)),
) -> Vec3f:
    """Barycentrically-interpolated SMOOTH shading normal at a triangle hit,
    falling back to the flat geometric normal when the mesh has no per-vertex
    normals. Refraction through a finely-tessellated curved surface (e.g. the
    wavy water sheet in water-caustic, whose PLY carries per-vertex normals)
    MUST use the smooth normal — the flat per-triangle normal refracts each
    triangle's whole patch of light in one direction, producing a blocky/
    blotchy caustic and the wrong energy distribution. This is what pbrt (and
    gonzales's own main path tracer via shading.mojo::_shading_normal) does;
    SPPM previously used only _geom_normal here, which was the discrepancy.

    `spheres`/`hit` give analytic spheres (primId.type == 4) their exact
    outward normal. Without them this returned the +Y placeholder below for
    every sphere hit, so a dielectric sphere refracted every ray identically
    regardless of where it was struck -- silently destroying any caustic it
    should cast."""
    var mi: Int; var bv: Int
    if inter.primId.type == 0:
        mi = Int(inter.primId.id1); bv = Int(inter.primId.id2)
    elif inter.primId.type == 1 or inter.primId.type == 2 or inter.primId.type == 3:
        mi = Int(inter.primId.id2 >> 32); bv = Int(inter.primId.id2 & 0xFFFFFFFF) * 3
    elif inter.primId.type == Int8(4) and _is_real_ptr[Sphere](spheres):
        return sphere_outward_normal(hit, spheres[unsafe_offset=Int(inter.primId.id1)].center)
    else:
        return Vec3f(Float32(0), Float32(1), Float32(0))
    var m = meshes[unsafe_offset=mi]
    var v0 = Int(mesh_vi(m, bv))
    var v1 = Int(mesh_vi(m, bv + 1))
    var v2 = Int(mesh_vi(m, bv + 2))
    var p0 = mesh_pos(m, v0)
    var p1 = mesh_pos(m, v1)
    var p2 = mesh_pos(m, v2)
    var gn = cross(p1 - p0, p2 - p0)
    if inter.primId.instanceIdx >= Int32(0):
        gn = transform_normal(Mat4(instances[unsafe_offset=Int(inter.primId.instanceIdx)].world_to_obj()), gn)
    var gl = dot(gn, gn)
    if gl > Float32(0.0): gn = gn * (Float32(1.0) / sqrt(gl))
    # No per-vertex normals → flat normal (sentinel addr <= 4, see GPU-nullable convention).
    if not _is_real_ptr(m.normals):
        return gn
    var w0 = Float32(1.0) - inter.u - inter.v
    var n0 = mesh_normal(m, v0)
    var n1 = mesh_normal(m, v1)
    var n2 = mesh_normal(m, v2)
    var sn = n0 * w0 + n1 * inter.u + n2 * inter.v
    if inter.primId.instanceIdx >= Int32(0):
        sn = transform_normal(Mat4(instances[unsafe_offset=Int(inter.primId.instanceIdx)].world_to_obj()), sn)
    var sl = dot(sn, sn)
    if sl <= Float32(1e-12):
        return gn
    # pbrt's convention, NOT the winding's: with per-vertex normals the
    # interpolated N IS the outward direction, and pbrt flips the GEOMETRIC
    # normal to agree with it (Triangle::InteractionFromIntersection,
    # FaceForward(n, ns)). Only glass reads this, and glass decides inside
    # vs outside from it: flipping N toward an inward winding instead
    # rendered a rough-glass sphere 1.28x pbrt, lte-orb's shell up to 2.5x.
    return sn * (Float32(1.0) / sqrt(sl))
