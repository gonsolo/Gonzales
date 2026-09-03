from std.memory import alloc
from gonzales.geometry import Point3f, Vec3f, RGB, Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Curve_C, Material_C, MatKind
from gonzales.bvh import BVH2Node, build_bvh2, traverse_bvh2_core

# ── Shared BVH-backed triangle-scene fixture ────────────────────────────────
# Builds a REAL BVH (via the same build_bvh2 that finalize_scene calls) over
# a small, caller-supplied set of triangles — not a hand-faked single leaf —
# so tests exercise the actual BVH-build and BVH2Node/PrimId_C production
# layout. Deliberately minimal: one mesh, no curves/instances/materials
# beyond a single flat one, matching what traverse_bvh2_core alone needs
# (see its own dangling-default params for blas/instances — a scene with no
# ObjectInstance usage never dereferences those).
#
# Usage:
#   var fx = make_triangle_scene([
#       Point3f(0,0,0), Point3f(1,0,0), Point3f(0,1,0),   # triangle 0
#   ])
#   var ray = Ray_C(Point3f(0.2, 0.2, -5.0), Vec3f(0,0,1))
#   var hit = fx.intersect(ray, Float32(100.0))
#   assert_true(Int(hit.hit) == 1)

@fieldwise_init
struct TriangleSceneFixture(Movable):
    var points:         UnsafePointer[Float32, MutExternalOrigin]
    var vertex_indices: UnsafePointer[Int64, MutExternalOrigin]
    var meshes:         UnsafePointer[TriangleMesh_C, MutExternalOrigin]
    var bvh_nodes:      UnsafePointer[BVH2Node, MutExternalOrigin]
    var prim_ids:       UnsafePointer[PrimId_C, MutExternalOrigin]
    var curves:         UnsafePointer[Curve_C, MutExternalOrigin]
    var materials:      UnsafePointer[Material_C, MutExternalOrigin]
    var n_tris:          Int32

    def intersect(self, ray: Ray_C, tMax: Float32) -> Intersection_C:
        var result = alloc[Intersection_C](1)
        traverse_bvh2_core(self.bvh_nodes, self.prim_ids, self.meshes, self.curves, ray, tMax, result)
        var r = result[0]
        result.free()
        return r

    def __del__(deinit self):
        self.points.free()
        self.vertex_indices.free()
        self.meshes.free()
        self.bvh_nodes.free()
        self.prim_ids.free()
        self.materials.free()

def make_triangle_scene(verts: List[Point3f]) -> TriangleSceneFixture:
    """verts must be a flat list of 3*N points (N triangles, CCW winding)."""
    var n_verts = len(verts)
    var n_tris = Int32(n_verts // 3)

    var points = alloc[Float32](n_verts * 4)
    for i in range(n_verts):
        points[i*4+0] = verts[i].x
        points[i*4+1] = verts[i].y
        points[i*4+2] = verts[i].z
        points[i*4+3] = Float32(1.0)

    var vertex_indices = alloc[Int64](n_verts)
    for i in range(n_verts):
        vertex_indices[i] = Int64(i)

    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = TriangleMesh_C(
        points,
        UnsafePointer[Int64, MutExternalOrigin].unsafe_dangling(),  # faceIndices, unused
        vertex_indices,
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),  # uvs
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),  # normals
    )

    var bounds = alloc[Float32](Int(n_tris) * 6)
    for t in range(Int(n_tris)):
        var p0 = verts[t*3+0]; var p1 = verts[t*3+1]; var p2 = verts[t*3+2]
        bounds[t*6+0] = min(p0.x, min(p1.x, p2.x))
        bounds[t*6+1] = min(p0.y, min(p1.y, p2.y))
        bounds[t*6+2] = min(p0.z, min(p1.z, p2.z))
        bounds[t*6+3] = max(p0.x, max(p1.x, p2.x))
        bounds[t*6+4] = max(p0.y, max(p1.y, p2.y))
        bounds[t*6+5] = max(p0.z, max(p1.z, p2.z))

    var max_nodes = Int(n_tris) * 2 + 4
    var bvh_nodes = alloc[BVH2Node](max_nodes)
    var order = alloc[Int32](Int(n_tris))
    _ = build_bvh2(bounds, n_tris, bvh_nodes, order)
    bounds.free()

    var prim_ids = alloc[PrimId_C](Int(n_tris))
    for k in range(Int(n_tris)):
        var orig = Int(order[k])
        prim_ids[k] = PrimId_C(Int64(0), Int64(orig * 3), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    order.free()

    var materials = alloc[Material_C](1)
    materials[0] = Material_C(
        MatKind.diffuse, Int8(0), Int8(0), Int8(0),
        RGB(Float32(0.8)),           # albedo
        RGB(Float32(0.0)),           # emission
        Int32(-1),                   # tex_idx
        Float32(0.0), Float32(0.0),  # roughU/V
        Int32(-1),                   # normal_tex_idx
        Int32(-1),                   # rough_tex_idx
        Int32(-1),                   # medium_interface_idx
        RGB(Float32(0.0)), RGB(Float32(0.0)),  # checker_tex1/2
        Float32(1.0), Float32(1.0),  # checker_uscale/vscale
        Int32(-1),                   # measured_idx
    )

    return TriangleSceneFixture(
        points, vertex_indices, meshes, bvh_nodes, prim_ids,
        UnsafePointer[Curve_C, MutExternalOrigin].unsafe_dangling(),
        materials, n_tris,
    )
