"""Write the triangles of a scene (or of one template, or of the ordinary meshes) as nine float32 each.

    build/dump_tris scene.pbrt out.f32 [all | ordinary | template N]
"""
from gonzales.primitives import mesh_vi, mesh_pos
from gonzales.scene_loader import mojo_parse_scene_any
from std.memory.alloc import unsafe_alloc
from std.memory import bitcast
from std.sys import argv


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("Usage: dump_tris scene.pbrt out.f32 [all | ordinary | template N]")
        return
    var path = String(args[1])
    var n = path.byte_length()
    var cpath = unsafe_alloc[UInt8](n + 1)
    for i in range(n): cpath[unsafe_offset=i] = path.unsafe_ptr()[unsafe_offset=i]
    cpath[unsafe_offset=n] = UInt8(0)
    var psc = mojo_parse_scene_any(cpath.unsafe_origin_cast[MutUntrackedOrigin]())
    var mode = String("all")
    if len(args) > 3: mode = String(args[3])
    var nm = Int(psc[unsafe_offset=0].mesh_count)
    var first = 0
    var last = nm
    var skip_templates = False
    if mode == "template" and len(args) > 4:
        var t = Int(String(args[4]))
        first = Int(psc[unsafe_offset=0].template_mesh_start[unsafe_offset=t])
        last = Int(psc[unsafe_offset=0].template_mesh_end[unsafe_offset=t])
    if mode == "ordinary":
        skip_templates = True
    var is_template = unsafe_alloc[Bool](max(nm, 1))
    for i in range(nm): is_template[unsafe_offset=i] = False
    for t in range(Int(psc[unsafe_offset=0].blas_count)):
        for i in range(Int(psc[unsafe_offset=0].template_mesh_start[unsafe_offset=t]), Int(psc[unsafe_offset=0].template_mesh_end[unsafe_offset=t])):
            is_template[unsafe_offset=i] = True
    var total = 0
    for i in range(first, last):
        if skip_templates and is_template[unsafe_offset=i]:
            continue
        total += Int(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=i])
    var out = List[UInt8](length=total * 36, fill=UInt8(0))
    var w = 0
    for i in range(first, last):
        if skip_templates and is_template[unsafe_offset=i]:
            continue
        var mesh = psc[unsafe_offset=0].meshes[unsafe_offset=i]
        for t in range(Int(psc[unsafe_offset=0].mesh_n_tris[unsafe_offset=i])):
            for k in range(3):
                var p = mesh_pos(mesh, mesh_vi(mesh, t * 3 + k))
                var f = SIMD[DType.float32, 4](p.x, p.y, p.z, Float32(0))
                var b = bitcast[DType.uint32, 4](f)
                for a in range(3):
                    for q in range(4): out[w + a * 4 + q] = UInt8((b[a] >> UInt32(8 * q)) & UInt32(0xff))
                w += 12
    print("triangles", total, "templates", Int(psc[unsafe_offset=0].blas_count), "instances", Int(psc[unsafe_offset=0].instance_count))
    with open(String(args[2]), "w") as f:
        f.write_bytes(out)
