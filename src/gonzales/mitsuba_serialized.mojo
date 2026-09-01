from std.memory import alloc
from std.subprocess import run
from std.os.path import exists

# ── Mitsuba ".serialized" binary mesh reader ────────────────────────────────
#
# Format (verified byte-for-byte against two independent sources: the
# original mitsuba-renderer/mitsuba C++ TriMesh::loadCompressed, and
# ~/src/mitsuba3's src/shapes/serialized.cpp plugin docstring):
#
#   uint16 format (0x041C), uint16 version (3 or 4), then ONE zlib-deflate
#   stream (for the single-mesh / shapeIndex==0 case handled here) holding:
#     uint32 flags (bit0 normals, bit1 texcoords, bit3 colors, bit4
#                   faceNormals, bit12 single-precision, bit13 double-
#                   precision)
#     [version==4 only] null-terminated name string
#     uint64 vertexCount, uint64 triangleCount
#     positions (3*vertexCount, single or double per the precision flag)
#     [if flagged] normals, texcoords, colors (same precision, same count-per-
#     vertex layout)
#     triangle indices (3*triangleCount, ALWAYS uint32 regardless of the
#     precision flag)
#
# Multi-mesh files (shapeIndex>0, with an end-of-file table-of-contents) are
# out of scope -- every mesh in the torus-caustic scene (the motivating case)
# is a single-mesh file. No in-memory zlib inflate exists in gonzales (the
# .pbrt.gz include support shells out to the `gzip` CLI on whole files, not
# reusable for an embedded byte range), so this shells out to `python3` for
# the zlib step only, then parses the raw inflated bytes in Mojo -- same
# "shell out for decompression, parse natively" split as that precedent.

struct MitsubaMesh(Movable):
    var positions: List[Float32]  # 3 per vertex, object space
    var normals:   List[Float32]  # 3 per vertex, or empty
    var uvs:       List[Float32]  # 2 per vertex, or empty
    var indices:   List[Int32]    # 3 per triangle
    var n_verts:   Int32
    var n_tris:    Int32

    def __init__(out self):
        self.positions = List[Float32]()
        self.normals   = List[Float32]()
        self.uvs       = List[Float32]()
        self.indices   = List[Int32]()
        self.n_verts   = Int32(0)
        self.n_tris    = Int32(0)


def _mit_ser_u32(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Int:
    return Int((buf + pos).bitcast[UInt32]()[0])

def _mit_ser_u64(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Int:
    return Int((buf + pos).bitcast[UInt64]()[0])

def _mit_ser_f32(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Float32:
    return (buf + pos).bitcast[Float32]()[0]

def _mit_ser_f64(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Float64:
    return (buf + pos).bitcast[Float64]()[0]


def load_mitsuba_serialized(path: String) -> MitsubaMesh:
    var mesh = MitsubaMesh()

    var raw_list: List[UInt8]
    try:
        var f0 = open(path, "r")
        raw_list = f0.read_bytes()
        f0.close()
    except:
        return mesh^
    var raw_n = len(raw_list)
    if raw_n < 4:
        return mesh^
    var raw = alloc[UInt8](raw_n)
    for i in range(raw_n):
        raw[i] = raw_list[i]
    var version = Int(raw.bitcast[UInt16]()[1])
    raw.free()

    var inflated_path = path + ".inflated"
    if not exists(inflated_path):
        try:
            _ = run("python3 -c \"import zlib; d=open('" + path +
                    "','rb').read(); open('" + inflated_path +
                    "','wb').write(zlib.decompress(d[4:]))\"")
        except:
            pass
    if not exists(inflated_path):
        return mesh^

    var body_list: List[UInt8]
    try:
        var f = open(inflated_path, "r")
        body_list = f.read_bytes()
        f.close()
    except:
        return mesh^
    var n = len(body_list)
    if n < 4:
        return mesh^
    var body = alloc[UInt8](n)
    for i in range(n):
        body[i] = body_list[i]

    var off = 0
    var flags = _mit_ser_u32(body, off); off += 4
    var has_normals = (flags & 0x0001) != 0
    var has_uv      = (flags & 0x0002) != 0
    var has_colors  = (flags & 0x0008) != 0
    var is_double   = (flags & 0x2000) != 0

    if version == 4:
        while off < n and body[off] != UInt8(0):
            off += 1
        off += 1

    var vc = _mit_ser_u64(body, off); off += 8
    var tc = _mit_ser_u64(body, off); off += 8
    mesh.n_verts = Int32(vc)
    mesh.n_tris  = Int32(tc)
    var fsize = 8 if is_double else 4

    mesh.positions.reserve(vc * 3)
    for _ in range(vc * 3):
        if is_double:
            mesh.positions.append(Float32(_mit_ser_f64(body, off)))
        else:
            mesh.positions.append(_mit_ser_f32(body, off))
        off += fsize

    if has_normals:
        mesh.normals.reserve(vc * 3)
        for _ in range(vc * 3):
            if is_double:
                mesh.normals.append(Float32(_mit_ser_f64(body, off)))
            else:
                mesh.normals.append(_mit_ser_f32(body, off))
            off += fsize

    if has_uv:
        mesh.uvs.reserve(vc * 2)
        for _ in range(vc * 2):
            if is_double:
                mesh.uvs.append(Float32(_mit_ser_f64(body, off)))
            else:
                mesh.uvs.append(_mit_ser_f32(body, off))
            off += fsize

    if has_colors:
        off += vc * 3 * fsize  # unused in v1, skip past

    mesh.indices.reserve(tc * 3)
    for _ in range(tc * 3):
        mesh.indices.append(Int32(_mit_ser_u32(body, off)))
        off += 4

    body.free()
    return mesh^
