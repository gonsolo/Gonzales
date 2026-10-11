"""Build an RT-core BLAS from triangles in a file and write the bytes and the primitive-id map.

    build/rt_build_dump tris.f32 out.blas out.gid

tris.f32: nine float32 per triangle. out.gid: int32 per primitive id (-1 = unused)."""
from gonzales.rt_build import rt_build_blas, RtBlas
from std.memory.alloc import unsafe_alloc
from std.sys import argv


def main() raises:
    var args = argv()
    if len(args) < 4:
        print("Usage: rt_build_dump tris.f32 out.blas out.gid")
        return
    var tris_path = String(args[1])
    var data = open(tris_path, "r").read_bytes()
    var n = len(data) // 36
    var tv = unsafe_alloc[Float32](max(n * 9, 1))
    var src = data.unsafe_ptr().bitcast[Float32]()
    for i in range(n * 9): tv[unsafe_offset=i] = src[unsafe_offset=i]
    var tight = True
    var sah = True
    for i in range(4, len(args)):
        if String(args[i]) == "loose": tight = False
        if String(args[i]) == "median": sah = False
    var quads = False
    for i in range(4, len(args)):
        if String(args[i]) == "quads": quads = True
    var r = rt_build_blas(tv, n, tight, sah, quads)
    print("triangles", n, "nodes", r.n_nodes, "blocks", r.n_blocks, "bytes", r.size)
    var ob = List[UInt8](length=r.size, fill=UInt8(0))
    for i in range(r.size): ob[i] = r.bytes[unsafe_offset=i]
    with open(String(args[2]), "w") as f:
        f.write_bytes(ob)
    var og = List[UInt8](length=r.n_gid * 4, fill=UInt8(0))
    for i in range(r.n_gid):
        var v = UInt32(r.gid_to_tri[unsafe_offset=i])
        for q in range(4): og[i * 4 + q] = UInt8((v >> UInt32(8 * q)) & UInt32(0xff))
    with open(String(args[3]), "w") as f:
        f.write_bytes(og)
