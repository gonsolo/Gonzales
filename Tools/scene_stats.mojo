"""Parse a scene and print what it holds and how many bytes each part takes in memory today:
the measurement behind the out-of-core geometry work. Prints one `key value` pair per line.

    build/scene_stats scene.pbrt
"""
from gonzales.bvh import BVH2Node
from gonzales.curves import Curve
from gonzales.geometry import _is_real_ptr
from gonzales.primitives import Instance, PrimId, TriangleMesh, mesh_points_floats
from gonzales.scene_loader import mojo_parse_scene_any
from std.memory.alloc import unsafe_alloc
from std.sys import argv
from std.sys.info import size_of
from std.time import perf_counter_ns


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("Usage: scene_stats scene.pbrt")
        return
    var path = String(args[1])
    var n = path.byte_length()
    var cpath = unsafe_alloc[UInt8](n + 1)
    for i in range(n): cpath[unsafe_offset=i] = path.unsafe_ptr()[unsafe_offset=i]
    cpath[unsafe_offset=n] = UInt8(0)
    var t0 = perf_counter_ns()
    var psc = mojo_parse_scene_any(cpath.unsafe_origin_cast[MutUntrackedOrigin]())
    var seconds = Float64(perf_counter_ns() - t0) / 1.0e9

    var verts = 0; var tris = 0; var uv_verts = 0; var nrm_verts = 0
    for m in range(Int(psc[unsafe_offset=0].mesh_count)):
        verts += Int(psc[unsafe_offset=0].mesh_n_verts[unsafe_offset=m]); tris += Int(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=m])
        uv_verts += Int(psc[unsafe_offset=0].mesh_uv_n_verts[unsafe_offset=m]); nrm_verts += Int(psc[unsafe_offset=0].mesh_nrm_n_verts[unsafe_offset=m])
    var mat_tris = 0; var live_meshes = 0; var quad_tris = 0; var point_bytes = 0
    for m in range(Int(psc[unsafe_offset=0].mesh_count)):
        if Int(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=m]) > 0:
            live_meshes += 1
            point_bytes += 4 * mesh_points_floats(Int(psc[unsafe_offset=0].mesh_n_verts[unsafe_offset=m]), psc[unsafe_offset=0].meshes[unsafe_offset=m].quads)
            if _is_real_ptr(psc[unsafe_offset=0].meshes[unsafe_offset=m].materials):
                mat_tris += Int(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=m])
            if psc[unsafe_offset=0].meshes[unsafe_offset=m].quads != Int32(0):
                quad_tris += Int(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=m])
    var blas_nodes = 0; var blas_prims = 0
    for b in range(Int(psc[unsafe_offset=0].blas_count)):
        blas_nodes += Int(psc[unsafe_offset=0].blas_node_counts[unsafe_offset=b]); blas_prims += Int(psc[unsafe_offset=0].blas_primid_counts[unsafe_offset=b])

    print("parse_seconds", seconds)
    print("meshes", Int(psc[unsafe_offset=0].mesh_count))
    print("meshes_with_triangles", live_meshes)
    print("vertices", verts)
    print("triangles", tris)
    print("curves", Int(psc[unsafe_offset=0].curve_count))
    print("templates", Int(psc[unsafe_offset=0].blas_count))
    print("instances", Int(psc[unsafe_offset=0].instance_count))
    print("tlas_nodes", Int(psc[unsafe_offset=0].bvh_node_count_cpu))
    print("tlas_prims", Int(psc[unsafe_offset=0].prim_count_cpu))
    print("blas_nodes", blas_nodes)
    print("blas_prims", blas_prims)
    # Bytes as held today: points are 4 floats; a triangle has three vertex ids and a face id (Int32), and in a
    # merged template a material id as well.
    print("bytes_points", point_bytes)
    print("quad_triangles", quad_tris)
    # A quad soup keeps no vertex indices, and a face id and a material a quad instead of a triangle.
    print("bytes_indices", (tris - quad_tris) * 16 + (mat_tris - quad_tris) * 4 + quad_tris * 4)
    print("bytes_uv_normals", uv_verts * 8 + nrm_verts * 4)
    print("bytes_uvs", uv_verts * 8)
    print("bytes_normals", nrm_verts * 4)
    # How many triangles carry the two halves of a unit quad as their texture coordinates: (0,0)(1,0)(1,1)
    # and (0,0)(1,1)(0,1). Those need not be stored.
    var uv_a = 0; var uv_b = 0; var uv_other = 0
    for m in range(Int(psc[unsafe_offset=0].mesh_count)):
        if Int(psc[unsafe_offset=0].mesh_uv_n_verts[unsafe_offset=m]) == 0:
            continue
        var mesh = psc[unsafe_offset=0].meshes[unsafe_offset=m]
        for t in range(Int(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=m])):
            var i0 = Int(mesh.vertexIndices[unsafe_offset=t * 3]); var i1 = Int(mesh.vertexIndices[unsafe_offset=t * 3 + 1])
            var i2 = Int(mesh.vertexIndices[unsafe_offset=t * 3 + 2])
            var u0 = mesh.uvs[unsafe_offset=i0 * 2]; var v0 = mesh.uvs[unsafe_offset=i0 * 2 + 1]
            var u1 = mesh.uvs[unsafe_offset=i1 * 2]; var v1 = mesh.uvs[unsafe_offset=i1 * 2 + 1]
            var u2 = mesh.uvs[unsafe_offset=i2 * 2]; var v2 = mesh.uvs[unsafe_offset=i2 * 2 + 1]
            if u0 == Float32(0) and v0 == Float32(0) and u1 == Float32(1) and v1 == Float32(0) and u2 == Float32(1) and v2 == Float32(1):
                uv_a += 1
            elif u0 == Float32(0) and v0 == Float32(0) and u1 == Float32(1) and v1 == Float32(1) and u2 == Float32(0) and v2 == Float32(1):
                uv_b += 1
            else:
                uv_other += 1
    # ...and how many vertices have the corner their index names: vertex v of a quad soup is corner v mod 4.
    var uv_corner = 0; var uv_verts_seen = 0
    for m in range(Int(psc[unsafe_offset=0].mesh_count)):
        var nuv = Int(psc[unsafe_offset=0].mesh_uv_n_verts[unsafe_offset=m])
        var mesh = psc[unsafe_offset=0].meshes[unsafe_offset=m]
        for v in range(nuv):
            var c = v % 4
            var eu = Float32(1) if (c == 1 or c == 2) else Float32(0)
            var ev = Float32(1) if c >= 2 else Float32(0)
            if mesh.uvs[unsafe_offset=v * 2] == eu and mesh.uvs[unsafe_offset=v * 2 + 1] == ev:
                uv_corner += 1
        uv_verts_seen += nuv
    print("uv_verts", uv_verts_seen)
    print("uv_verts_at_their_corner", uv_corner)
    print("uv_tris_first_half", uv_a)
    print("uv_tris_second_half", uv_b)
    print("uv_tris_other", uv_other)
    print("bytes_mesh_structs", Int(psc[unsafe_offset=0].mesh_count) * size_of[TriangleMesh]())
    print("bytes_curves", Int(psc[unsafe_offset=0].curve_count) * size_of[Curve]())
    print("bytes_instances", Int(psc[unsafe_offset=0].instance_count) * size_of[Instance]())
    print("bytes_tlas", Int(psc[unsafe_offset=0].bvh_node_count_cpu) * size_of[BVH2Node]() + Int(psc[unsafe_offset=0].prim_count_cpu) * size_of[PrimId]())
    print("bytes_blas", blas_nodes * size_of[BVH2Node]() + blas_prims * size_of[PrimId]())
    # The BVH over the instances follows the templates' in the BLAS arrays (pbrt_parser.mojo).
    var inst_nodes = 0
    if psc[unsafe_offset=0].instance_count > Int32(0):
        inst_nodes = Int(psc[unsafe_offset=0].blas_node_counts[unsafe_offset=Int(psc[unsafe_offset=0].blas_count)])
    print("bytes_instance_bvh", inst_nodes * size_of[BVH2Node]())
