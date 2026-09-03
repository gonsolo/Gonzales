from std.math import abs
from std.memory import alloc
from std.os import remove
from std.testing import assert_true, TestSuite
from gonzales.ply import load_ply

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# ── byte-packing helpers ────────────────────────────────────────────────────
# For the binary_little_endian/binary_big_endian format tests below — these
# helpers append the little/big-endian byte representation of a value to a
# growable buffer, exactly the way a real binary PLY exporter would. (ASCII
# format tests further down just append plain decimal text instead.)

def _append_str(mut data: List[UInt8], s: String):
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        data.append(bytes[i])

def _append_f32_le(mut data: List[UInt8], v: Float32):
    var tmp = alloc[UInt8](4)
    tmp.bitcast[Float32]()[0] = v
    for i in range(4):
        data.append(tmp[i])
    tmp.free()

def _append_f32_be(mut data: List[UInt8], v: Float32):
    var tmp = alloc[UInt8](4)
    tmp.bitcast[Float32]()[0] = v
    data.append(tmp[3]); data.append(tmp[2]); data.append(tmp[1]); data.append(tmp[0])
    tmp.free()

def _append_f64_le(mut data: List[UInt8], v: Float64):
    var tmp = alloc[UInt8](8)
    tmp.bitcast[Float64]()[0] = v
    for i in range(8):
        data.append(tmp[i])
    tmp.free()

def _append_i32_le(mut data: List[UInt8], v: Int32):
    var tmp = alloc[UInt8](4)
    tmp.bitcast[Int32]()[0] = v
    for i in range(4):
        data.append(tmp[i])
    tmp.free()

def _append_i32_be(mut data: List[UInt8], v: Int32):
    var tmp = alloc[UInt8](4)
    tmp.bitcast[Int32]()[0] = v
    data.append(tmp[3]); data.append(tmp[2]); data.append(tmp[1]); data.append(tmp[0])
    tmp.free()

def _append_u16_le(mut data: List[UInt8], v: UInt16):
    var tmp = alloc[UInt8](2)
    tmp.bitcast[UInt16]()[0] = v
    data.append(tmp[0]); data.append(tmp[1])
    tmp.free()

def _write_file(path: String, data: List[UInt8]) raises:
    var f = open(path, "w")
    f.write_all(Span(data))
    f.close()

def _path_cstr(path: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var n = path.byte_length()
    var buf = alloc[UInt8](n + 1)
    for i in range(n):
        buf[i] = path.as_bytes()[i]
    buf[n] = UInt8(0)
    return buf

# ── load_ply out-parameter bundle ───────────────────────────────────────────
# Mirrors exactly how pbrt_parser.mojo calls load_ply (see the `is_ply`
# branch of its shape handler): 8 out-pointers, each a 1-element heap
# allocation that load_ply fills in.

@fieldwise_init
struct _PlyResult(Movable):
    var pts:         UnsafePointer[UnsafePointer[Float32, MutExternalOrigin], MutExternalOrigin]
    var n_verts:     UnsafePointer[Int32, MutExternalOrigin]
    var idx:         UnsafePointer[UnsafePointer[Int32, MutExternalOrigin], MutExternalOrigin]
    var n_tris:      UnsafePointer[Int32, MutExternalOrigin]
    var uvs:         UnsafePointer[UnsafePointer[Float32, MutExternalOrigin], MutExternalOrigin]
    var has_uvs:     UnsafePointer[Int32, MutExternalOrigin]
    var normals:     UnsafePointer[UnsafePointer[Float32, MutExternalOrigin], MutExternalOrigin]
    var has_normals: UnsafePointer[Int32, MutExternalOrigin]

def _alloc_ply_result() -> _PlyResult:
    var r = _PlyResult(
        alloc[UnsafePointer[Float32, MutExternalOrigin]](1),
        alloc[Int32](1),
        alloc[UnsafePointer[Int32, MutExternalOrigin]](1),
        alloc[Int32](1),
        alloc[UnsafePointer[Float32, MutExternalOrigin]](1),
        alloc[Int32](1),
        alloc[UnsafePointer[Float32, MutExternalOrigin]](1),
        alloc[Int32](1),
    )
    r.uvs[0] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling()
    r.has_uvs[0] = Int32(0)
    r.normals[0] = UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling()
    r.has_normals[0] = Int32(0)
    return r^

def _load(path: String, mut r: _PlyResult) -> Int32:
    var path_ptr = _path_cstr(path)
    var ok = load_ply(
        path_ptr, r.pts, r.n_verts, r.idx, r.n_tris, r.uvs, r.has_uvs,
        r.normals, r.has_normals,
    )
    path_ptr.free()
    return ok

def _free(mut r: _PlyResult):
    r.pts.free(); r.n_verts.free(); r.idx.free(); r.n_tris.free()
    r.uvs.free(); r.has_uvs.free(); r.normals.free(); r.has_normals.free()

def _cleanup(path: String):
    try:
        remove(path)
    except:
        pass

# ── ascii format ─────────────────────────────────────────────────────────────
# FIXED: load_ply's header parser used to only distinguish "binary_big_endian"
# from everything else, silently treating `format ascii 1.0` as little-endian
# binary (no textual number-parsing path existed at all). Now `is_ascii` is
# detected at the format line and the vertex/face data section is parsed as
# whitespace-separated decimal text via _ply_word_to_float/_ply_word_to_int,
# reusing the same per-property role/order logic as the binary path.

def test_ascii_triangle() raises:
    var path = String("/tmp/gonzales_test_ascii_triangle.ply")
    var data = List[UInt8]()
    _append_str(data, String(
        "ply\nformat ascii 1.0\n"
        + "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        + "element face 1\nproperty list uchar int vertex_indices\n"
        + "end_header\n"
        + "0.0 0.0 0.0\n"
        + "1.0 0.0 0.0\n"
        + "0.0 1.0 0.0\n"
        + "3 0 1 2\n"
    ))
    _write_file(path, data)

    var r = _alloc_ply_result()
    var ok = _load(path, r)
    assert_true(ok == Int32(1))
    assert_true(r.n_verts[0] == Int32(3))
    assert_true(r.n_tris[0] == Int32(1))
    var pts = r.pts[0]
    assert_true(_close(pts[0], Float32(0.0)) and _close(pts[1], Float32(0.0)) and _close(pts[2], Float32(0.0)))
    assert_true(_close(pts[3], Float32(1.0)) and _close(pts[4], Float32(0.0)) and _close(pts[5], Float32(0.0)))
    assert_true(_close(pts[6], Float32(0.0)) and _close(pts[7], Float32(1.0)) and _close(pts[8], Float32(0.0)))
    var idx = r.idx[0]
    assert_true(idx[0] == Int32(0) and idx[1] == Int32(1) and idx[2] == Int32(2))

    pts.free(); idx.free()
    _free(r)
    _cleanup(path)

def test_ascii_negative_and_exponent_values() raises:
    """Exercises _ply_word_to_float's sign/fraction/exponent parsing on
    values a real exporter would plausibly emit."""
    var path = String("/tmp/gonzales_test_ascii_signs.ply")
    var data = List[UInt8]()
    _append_str(data, String(
        "ply\nformat ascii 1.0\n"
        + "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        + "element face 1\nproperty list uchar int vertex_indices\n"
        + "end_header\n"
        + "-1.5 2.25 0.0\n"
        + "1.0e2 -3.5e-1 0.0\n"
        + "0.0 0.0 -0.001\n"
        + "3 0 1 2\n"
    ))
    _write_file(path, data)

    var r = _alloc_ply_result()
    var ok = _load(path, r)
    assert_true(ok == Int32(1))
    var pts = r.pts[0]
    assert_true(_close(pts[0], Float32(-1.5)) and _close(pts[1], Float32(2.25)))
    assert_true(_close(pts[3], Float32(100.0)) and _close(pts[4], Float32(-0.35)))
    assert_true(_close(pts[8], Float32(-0.001)))

    pts.free(); r.idx[0].free()
    _free(r)
    _cleanup(path)

def test_ascii_quad_face_triangulated_into_fan() raises:
    """Same fan-triangulation convention as the binary quad test, but via
    the ASCII face-list path (count token followed by `count` index tokens
    on one line)."""
    var path = String("/tmp/gonzales_test_ascii_quad.ply")
    var data = List[UInt8]()
    _append_str(data, String(
        "ply\nformat ascii 1.0\n"
        + "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
        + "element face 1\nproperty list uchar int vertex_indices\n"
        + "end_header\n"
        + "0.0 0.0 0.0\n"
        + "1.0 0.0 0.0\n"
        + "1.0 1.0 0.0\n"
        + "0.0 1.0 0.0\n"
        + "4 0 1 2 3\n"
    ))
    _write_file(path, data)

    var r = _alloc_ply_result()
    var ok = _load(path, r)
    assert_true(ok == Int32(1))
    assert_true(r.n_verts[0] == Int32(4))
    assert_true(r.n_tris[0] == Int32(2))
    var idx = r.idx[0]
    assert_true(idx[0] == Int32(0) and idx[1] == Int32(1) and idx[2] == Int32(2))
    assert_true(idx[3] == Int32(0) and idx[4] == Int32(2) and idx[5] == Int32(3))

    r.pts[0].free(); idx.free()
    _free(r)
    _cleanup(path)

# ── binary_little_endian: minimal triangle ──────────────────────────────────

def test_binary_le_triangle() raises:
    var path = String("/tmp/gonzales_test_le_triangle.ply")
    var data = List[UInt8]()
    _append_str(data, String(
        "ply\nformat binary_little_endian 1.0\n"
        + "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        + "element face 1\nproperty list uchar int vertex_indices\n"
        + "end_header\n"
    ))
    _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0))
    _append_f32_le(data, Float32(1.0)); _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0))
    _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(1.0)); _append_f32_le(data, Float32(0.0))
    data.append(UInt8(3))
    _append_i32_le(data, Int32(0)); _append_i32_le(data, Int32(1)); _append_i32_le(data, Int32(2))
    _write_file(path, data)

    var r = _alloc_ply_result()
    var ok = _load(path, r)
    assert_true(ok == Int32(1))
    assert_true(r.n_verts[0] == Int32(3))
    assert_true(r.n_tris[0] == Int32(1))
    var pts = r.pts[0]
    assert_true(_close(pts[0], Float32(0.0)) and _close(pts[1], Float32(0.0)) and _close(pts[2], Float32(0.0)))
    assert_true(_close(pts[3], Float32(1.0)) and _close(pts[4], Float32(0.0)) and _close(pts[5], Float32(0.0)))
    assert_true(_close(pts[6], Float32(0.0)) and _close(pts[7], Float32(1.0)) and _close(pts[8], Float32(0.0)))
    var idx = r.idx[0]
    assert_true(idx[0] == Int32(0) and idx[1] == Int32(1) and idx[2] == Int32(2))
    assert_true(r.has_uvs[0] == Int32(0))
    assert_true(r.has_normals[0] == Int32(0))

    pts.free(); idx.free()
    _free(r)
    _cleanup(path)

# ── binary_big_endian: same triangle, exercises the _be decode path ────────

def test_binary_be_triangle() raises:
    var path = String("/tmp/gonzales_test_be_triangle.ply")
    var data = List[UInt8]()
    _append_str(data, String(
        "ply\nformat binary_big_endian 1.0\n"
        + "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        + "element face 1\nproperty list uchar int vertex_indices\n"
        + "end_header\n"
    ))
    _append_f32_be(data, Float32(2.0)); _append_f32_be(data, Float32(0.0)); _append_f32_be(data, Float32(0.0))
    _append_f32_be(data, Float32(0.0)); _append_f32_be(data, Float32(2.0)); _append_f32_be(data, Float32(0.0))
    _append_f32_be(data, Float32(0.0)); _append_f32_be(data, Float32(0.0)); _append_f32_be(data, Float32(2.0))
    data.append(UInt8(3))
    _append_i32_be(data, Int32(0)); _append_i32_be(data, Int32(1)); _append_i32_be(data, Int32(2))
    _write_file(path, data)

    var r = _alloc_ply_result()
    var ok = _load(path, r)
    assert_true(ok == Int32(1))
    assert_true(r.n_verts[0] == Int32(3))
    assert_true(r.n_tris[0] == Int32(1))
    var pts = r.pts[0]
    assert_true(_close(pts[0], Float32(2.0)))
    assert_true(_close(pts[4], Float32(2.0)))
    assert_true(_close(pts[8], Float32(2.0)))
    var idx = r.idx[0]
    assert_true(idx[0] == Int32(0) and idx[1] == Int32(1) and idx[2] == Int32(2))

    pts.free(); idx.free()
    _free(r)
    _cleanup(path)

# ── double-precision (float64) vertex properties ────────────────────────────

def test_binary_le_double_precision_positions() raises:
    var path = String("/tmp/gonzales_test_le_double.ply")
    var data = List[UInt8]()
    _append_str(data, String(
        "ply\nformat binary_little_endian 1.0\n"
        + "element vertex 3\nproperty double x\nproperty double y\nproperty double z\n"
        + "element face 1\nproperty list uchar int vertex_indices\n"
        + "end_header\n"
    ))
    _append_f64_le(data, Float64(0.0)); _append_f64_le(data, Float64(0.0)); _append_f64_le(data, Float64(0.0))
    _append_f64_le(data, Float64(3.5)); _append_f64_le(data, Float64(0.0)); _append_f64_le(data, Float64(0.0))
    _append_f64_le(data, Float64(0.0)); _append_f64_le(data, Float64(3.5)); _append_f64_le(data, Float64(0.0))
    data.append(UInt8(3))
    _append_i32_le(data, Int32(0)); _append_i32_le(data, Int32(1)); _append_i32_le(data, Int32(2))
    _write_file(path, data)

    var r = _alloc_ply_result()
    var ok = _load(path, r)
    assert_true(ok == Int32(1))
    assert_true(r.n_verts[0] == Int32(3))
    var pts = r.pts[0]
    assert_true(_close(pts[3], Float32(3.5)))
    assert_true(_close(pts[7], Float32(3.5)))

    pts.free(); r.idx[0].free()
    _free(r)
    _cleanup(path)

# ── quad face: triangulated via the (0, i+1, i+2) fan the code implements ──

def test_quad_face_triangulated_into_fan() raises:
    var path = String("/tmp/gonzales_test_quad.ply")
    var data = List[UInt8]()
    _append_str(data, String(
        "ply\nformat binary_little_endian 1.0\n"
        + "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
        + "element face 1\nproperty list uchar int vertex_indices\n"
        + "end_header\n"
    ))
    _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0))
    _append_f32_le(data, Float32(1.0)); _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0))
    _append_f32_le(data, Float32(1.0)); _append_f32_le(data, Float32(1.0)); _append_f32_le(data, Float32(0.0))
    _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(1.0)); _append_f32_le(data, Float32(0.0))
    data.append(UInt8(4))
    _append_i32_le(data, Int32(0)); _append_i32_le(data, Int32(1))
    _append_i32_le(data, Int32(2)); _append_i32_le(data, Int32(3))
    _write_file(path, data)

    var r = _alloc_ply_result()
    var ok = _load(path, r)
    assert_true(ok == Int32(1))
    assert_true(r.n_verts[0] == Int32(4))
    # cnt-2 = 2 triangles: fan (face_idx[0], face_idx[ti+1], face_idx[ti+2])
    assert_true(r.n_tris[0] == Int32(2))
    var idx = r.idx[0]
    assert_true(idx[0] == Int32(0) and idx[1] == Int32(1) and idx[2] == Int32(2))
    assert_true(idx[3] == Int32(0) and idx[4] == Int32(2) and idx[5] == Int32(3))

    r.pts[0].free(); idx.free()
    _free(r)
    _cleanup(path)

# ── ushort face-index width (property list uchar ushort vertex_indices) ────

def test_face_list_ushort_index_width() raises:
    var path = String("/tmp/gonzales_test_ushort_idx.ply")
    var data = List[UInt8]()
    _append_str(data, String(
        "ply\nformat binary_little_endian 1.0\n"
        + "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        + "element face 1\nproperty list uchar ushort vertex_indices\n"
        + "end_header\n"
    ))
    _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0))
    _append_f32_le(data, Float32(1.0)); _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0))
    _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(1.0)); _append_f32_le(data, Float32(0.0))
    data.append(UInt8(3))
    _append_u16_le(data, UInt16(0)); _append_u16_le(data, UInt16(1)); _append_u16_le(data, UInt16(2))
    _write_file(path, data)

    var r = _alloc_ply_result()
    var ok = _load(path, r)
    assert_true(ok == Int32(1))
    assert_true(r.n_tris[0] == Int32(1))
    var idx = r.idx[0]
    assert_true(idx[0] == Int32(0) and idx[1] == Int32(1) and idx[2] == Int32(2))

    r.pts[0].free(); idx.free()
    _free(r)
    _cleanup(path)

# ── extra vertex properties: normals + UVs are read and attached ───────────

def test_normals_and_uvs_attached() raises:
    var path = String("/tmp/gonzales_test_normals_uvs.ply")
    var data = List[UInt8]()
    _append_str(data, String(
        "ply\nformat binary_little_endian 1.0\n"
        + "element vertex 3\n"
        + "property float x\nproperty float y\nproperty float z\n"
        + "property float nx\nproperty float ny\nproperty float nz\n"
        + "property float u\nproperty float v\n"
        + "element face 1\nproperty list uchar int vertex_indices\n"
        + "end_header\n"
    ))
    # vertex 0: pos(0,0,0) normal(0,0,1) uv(0,0)
    _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0))
    _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(1.0))
    _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0))
    # vertex 1: pos(1,0,0) normal(0,0,1) uv(1,0)
    _append_f32_le(data, Float32(1.0)); _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0))
    _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(1.0))
    _append_f32_le(data, Float32(1.0)); _append_f32_le(data, Float32(0.0))
    # vertex 2: pos(0,1,0) normal(0,0,1) uv(0,1)
    _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(1.0)); _append_f32_le(data, Float32(0.0))
    _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(1.0))
    _append_f32_le(data, Float32(0.0)); _append_f32_le(data, Float32(1.0))
    data.append(UInt8(3))
    _append_i32_le(data, Int32(0)); _append_i32_le(data, Int32(1)); _append_i32_le(data, Int32(2))
    _write_file(path, data)

    var r = _alloc_ply_result()
    var ok = _load(path, r)
    assert_true(ok == Int32(1))
    assert_true(r.has_normals[0] == Int32(1))
    assert_true(r.has_uvs[0] == Int32(1))
    var nrm = r.normals[0]
    assert_true(_close(nrm[0], Float32(0.0)) and _close(nrm[1], Float32(0.0)) and _close(nrm[2], Float32(1.0)))
    assert_true(_close(nrm[6], Float32(0.0)) and _close(nrm[7], Float32(0.0)) and _close(nrm[8], Float32(1.0)))
    var uvs = r.uvs[0]
    assert_true(_close(uvs[0], Float32(0.0)) and _close(uvs[1], Float32(0.0)))
    assert_true(_close(uvs[2], Float32(1.0)) and _close(uvs[3], Float32(0.0)))
    assert_true(_close(uvs[4], Float32(0.0)) and _close(uvs[5], Float32(1.0)))

    r.pts[0].free(); r.idx[0].free(); nrm.free(); uvs.free()
    _free(r)
    _cleanup(path)

# ── malformed header (missing vertex/face elements) fails gracefully ───────

def test_missing_elements_returns_failure() raises:
    var path = String("/tmp/gonzales_test_bad_header.ply")
    var data = List[UInt8]()
    _append_str(data, String("ply\nformat binary_little_endian 1.0\nend_header\n"))
    _write_file(path, data)

    var r = _alloc_ply_result()
    var ok = _load(path, r)
    assert_true(ok == Int32(0))
    _free(r)
    _cleanup(path)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
