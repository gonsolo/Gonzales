"""Host-side accounting of the per-mesh device buffers of a cached scene (what MeshBuffers.upload would allocate).

    build/scene_mem scene.pbrt cache.gsc
"""
from gonzales.scene_cache import scene_cache_load
from gonzales.scene_loader import mojo_parse_scene_any
from gonzales.primitives import mesh_points_floats, TriangleMesh
from std.memory.alloc import unsafe_alloc
from std.sys import argv


def _cstr(s: String) -> Pointer[UInt8, MutUntrackedOrigin]:
    var n = s.byte_length()
    var p = unsafe_alloc[UInt8](n + 1)
    for i in range(n): p[unsafe_offset=i] = s.unsafe_ptr()[unsafe_offset=i]
    p[unsafe_offset=n] = UInt8(0)
    return p.unsafe_origin_cast[MutUntrackedOrigin]()


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("Usage: scene_mem scene.pbrt (cache.gsc | -)")
        return
    var psc = mojo_parse_scene_any(_cstr(String(args[1]))) if String(args[2]) == "-" else scene_cache_load(_cstr(String(args[2])), _cstr(String(args[1])))
    if Int(psc) == 0 or Int(psc) == 1:
        print("cache not usable")
        return
    var nm = Int(psc[unsafe_offset=0].mesh_count)
    var n_buf = 0
    var bytes = 0
    var pad256 = 0
    var pad512 = 0
    var pad4k = 0
    var n_tiny = 0           # buffers of 16 bytes or less
    var b_pts = 0
    var b_face = 0
    var b_vert = 0
    var b_uv = 0
    var b_nrm = 0
    var b_mat = 0
    var n_meshes_used = 0
    var q_meshes = 0
    var q_bytes = 0
    var i_meshes = 0
    var i_bytes = 0
    var i_tris = 0
    var q_tris = 0
    var nt_tmpl = Int(psc[unsafe_offset=0].blas_count)
    var in_tmpl = 0
    var tmpl_nonempty = 0
    var tmpl_tris = 0
    for t in range(nt_tmpl):
        for i in range(Int(psc[unsafe_offset=0].template_mesh_start[unsafe_offset=t]), Int(psc[unsafe_offset=0].template_mesh_end[unsafe_offset=t])):
            in_tmpl += 1
            var k = Int(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=i])
            if k > 0:
                tmpl_nonempty += 1; tmpl_tris += k
    print("templates", nt_tmpl, "template meshes", in_tmpl, "non-empty", tmpl_nonempty, "triangles in them", tmpl_tris)
    var sizes = unsafe_alloc[Int](8 * max(nm, 1))
    var ns = 0
    for i in range(nm):
        var nt = Int(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=i])
        if nt == 0:
            continue
        n_meshes_used += 1
        var m = psc[unsafe_offset=0].meshes[unsafe_offset=i]
        var is_quads = m.quads != Int32(0)
        var per_face = nt // 2 if is_quads else nt
        var uv_n = Int(psc[unsafe_offset=0].mesh_uv_n_verts[unsafe_offset=i])
        var nrm_n = Int(psc[unsafe_offset=0].mesh_nrm_n_verts[unsafe_offset=i])
        var s0 = mesh_points_floats(Int(psc[unsafe_offset=0].mesh_n_verts[unsafe_offset=i]), m.quads) * 4
        var s1 = per_face * 4
        var s2 = 4 if is_quads else nt * 12
        var s3 = uv_n * 8 if uv_n > 0 else 4
        var s4 = nrm_n * 4 if nrm_n > 0 else 4
        if is_quads:
            q_meshes += 1; q_bytes += s0 + s1 + s2 + s3 + s4; q_tris += nt
        else:
            i_meshes += 1; i_bytes += s0 + s1 + s2 + s3 + s4; i_tris += nt
        b_pts += s0; b_face += s1; b_vert += s2; b_uv += s3; b_nrm += s4
        sizes[unsafe_offset=ns] = s0; sizes[unsafe_offset=ns + 1] = s1; sizes[unsafe_offset=ns + 2] = s2
        sizes[unsafe_offset=ns + 3] = s3; sizes[unsafe_offset=ns + 4] = s4
        ns += 5
        if _is_real(m):
            b_mat += per_face * 4
            sizes[unsafe_offset=ns] = per_face * 4
            ns += 1
    for k in range(ns):
        var b = max(sizes[unsafe_offset=k], 1)
        n_buf += 1
        bytes += b
        if b <= 16: n_tiny += 1
        pad256 += (b + 255) // 256 * 256
        pad512 += (b + 511) // 512 * 512
        pad4k += (b + 4095) // 4096 * 4096
    print("meshes with triangles:", n_meshes_used, "of", nm, "; buffers:", n_buf, "; tiny (<=16 B):", n_tiny)
    print("points", b_pts // 1048576, "MB; face ids", b_face // 1048576, "MB; vertex idx", b_vert // 1048576,
          "MB; uv", b_uv // 1048576, "MB; normals", b_nrm // 1048576, "MB; per-face materials", b_mat // 1048576, "MB")
    print("quad meshes:", q_meshes, q_tris, "triangles,", q_bytes // 1048576, "MB; indexed meshes:", i_meshes, i_tris, "triangles,", i_bytes // 1048576, "MB")
    print("payload", bytes // 1048576, "MB; rounded to 256 B:", pad256 // 1048576, "MB, 512 B:", pad512 // 1048576, "MB, 4 KB:", pad4k // 1048576, "MB")
    print("instances", Int(psc[unsafe_offset=0].instance_count), "x 52 B =", Int(psc[unsafe_offset=0].instance_count) * 52 // 1048576, "MB;",
          "prims", Int(psc[unsafe_offset=0].prim_count), "bvh nodes", Int(psc[unsafe_offset=0].bvh_node_count))


def _is_real(m: TriangleMesh) -> Bool:
    return Int(m.materials) > 4
