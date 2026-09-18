from std.math import abs
from std.memory.alloc import unsafe_alloc
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
    var tmp = unsafe_alloc[UInt8](4)
    tmp.unsafe_bitcast[Float32]()[unsafe_offset=0] = v
    for i in range(4):
        data.append(tmp[unsafe_offset=i])
    tmp.unsafe_free()

def _append_f32_be(mut data: List[UInt8], v: Float32):
    var tmp = unsafe_alloc[UInt8](4)
    tmp.unsafe_bitcast[Float32]()[unsafe_offset=0] = v
    data.append(tmp[unsafe_offset=3]); data.append(tmp[unsafe_offset=2]); data.append(tmp[unsafe_offset=1]); data.append(tmp[unsafe_offset=0])
    tmp.unsafe_free()

def _append_f64_le(mut data: List[UInt8], v: Float64):
    var tmp = unsafe_alloc[UInt8](8)
    tmp.unsafe_bitcast[Float64]()[unsafe_offset=0] = v
    for i in range(8):
        data.append(tmp[unsafe_offset=i])
    tmp.unsafe_free()

def _append_i32_le(mut data: List[UInt8], v: Int32):
    var tmp = unsafe_alloc[UInt8](4)
    tmp.unsafe_bitcast[Int32]()[unsafe_offset=0] = v
    for i in range(4):
        data.append(tmp[unsafe_offset=i])
    tmp.unsafe_free()

def _append_i32_be(mut data: List[UInt8], v: Int32):
    var tmp = unsafe_alloc[UInt8](4)
    tmp.unsafe_bitcast[Int32]()[unsafe_offset=0] = v
    data.append(tmp[unsafe_offset=3]); data.append(tmp[unsafe_offset=2]); data.append(tmp[unsafe_offset=1]); data.append(tmp[unsafe_offset=0])
    tmp.unsafe_free()

def _append_u16_le(mut data: List[UInt8], v: UInt16):
    var tmp = unsafe_alloc[UInt8](2)
    tmp.unsafe_bitcast[UInt16]()[unsafe_offset=0] = v
    data.append(tmp[unsafe_offset=0]); data.append(tmp[unsafe_offset=1])
    tmp.unsafe_free()

def _write_file(path: String, data: List[UInt8]) raises:
    var f = open(path, "w")
    f.write_all(Span(data))
    f.close()

def _path_cstr(path: String) -> Pointer[UInt8, MutUntrackedOrigin]:
    var n = path.byte_length()
    var buf = unsafe_alloc[UInt8](n + 1)
    for i in range(n):
        buf[unsafe_offset=i] = path.as_bytes()[i]
    buf[unsafe_offset=n] = UInt8(0)
    return buf

# ── load_ply out-parameter bundle ───────────────────────────────────────────
# Mirrors exactly how pbrt_parser.mojo calls load_ply (see the `is_ply`
# branch of its shape handler): 8 out-pointers, each a 1-element heap
# allocation that load_ply fills in.

@fieldwise_init
struct _PlyResult(Movable):
    var pts:         Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin]
    var n_verts:     Pointer[Int32, MutUntrackedOrigin]
    var idx:         Pointer[Pointer[Int32, MutUntrackedOrigin], MutUntrackedOrigin]
    var n_tris:      Pointer[Int32, MutUntrackedOrigin]
    var uvs:         Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin]
    var has_uvs:     Pointer[Int32, MutUntrackedOrigin]
    var normals:     Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin]
    var has_normals: Pointer[Int32, MutUntrackedOrigin]

def _alloc_ply_result() -> _PlyResult:
    var r = _PlyResult(
        unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](1),
        unsafe_alloc[Int32](1),
        unsafe_alloc[Pointer[Int32, MutUntrackedOrigin]](1),
        unsafe_alloc[Int32](1),
        unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](1),
        unsafe_alloc[Int32](1),
        unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](1),
        unsafe_alloc[Int32](1),
    )
    r.uvs[unsafe_offset=0] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
    r.has_uvs[unsafe_offset=0] = Int32(0)
    r.normals[unsafe_offset=0] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
    r.has_normals[unsafe_offset=0] = Int32(0)
    return r^

def _load(path: String, mut r: _PlyResult) -> Int32:
    var path_ptr = _path_cstr(path)
    var ok = load_ply(
        path_ptr, r.pts, r.n_verts, r.idx, r.n_tris, r.uvs, r.has_uvs,
        r.normals, r.has_normals,
    )
    path_ptr.unsafe_free()
    return ok

def _free(mut r: _PlyResult):
    r.pts.unsafe_free(); r.n_verts.unsafe_free(); r.idx.unsafe_free(); r.n_tris.unsafe_free()
    r.uvs.unsafe_free(); r.has_uvs.unsafe_free(); r.normals.unsafe_free(); r.has_normals.unsafe_free()

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
    assert_true(r.n_verts[unsafe_offset=0] == Int32(3))
    assert_true(r.n_tris[unsafe_offset=0] == Int32(1))
    var pts = r.pts[unsafe_offset=0]
    assert_true(_close(pts[unsafe_offset=0], Float32(0.0)) and _close(pts[unsafe_offset=1], Float32(0.0)) and _close(pts[unsafe_offset=2], Float32(0.0)))
    assert_true(_close(pts[unsafe_offset=3], Float32(1.0)) and _close(pts[unsafe_offset=4], Float32(0.0)) and _close(pts[unsafe_offset=5], Float32(0.0)))
    assert_true(_close(pts[unsafe_offset=6], Float32(0.0)) and _close(pts[unsafe_offset=7], Float32(1.0)) and _close(pts[unsafe_offset=8], Float32(0.0)))
    var idx = r.idx[unsafe_offset=0]
    assert_true(idx[unsafe_offset=0] == Int32(0) and idx[unsafe_offset=1] == Int32(1) and idx[unsafe_offset=2] == Int32(2))

    pts.unsafe_free(); idx.unsafe_free()
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
    var pts = r.pts[unsafe_offset=0]
    assert_true(_close(pts[unsafe_offset=0], Float32(-1.5)) and _close(pts[unsafe_offset=1], Float32(2.25)))
    assert_true(_close(pts[unsafe_offset=3], Float32(100.0)) and _close(pts[unsafe_offset=4], Float32(-0.35)))
    assert_true(_close(pts[unsafe_offset=8], Float32(-0.001)))

    pts.unsafe_free(); r.idx[unsafe_offset=0].unsafe_free()
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
    assert_true(r.n_verts[unsafe_offset=0] == Int32(4))
    assert_true(r.n_tris[unsafe_offset=0] == Int32(2))
    var idx = r.idx[unsafe_offset=0]
    assert_true(idx[unsafe_offset=0] == Int32(0) and idx[unsafe_offset=1] == Int32(1) and idx[unsafe_offset=2] == Int32(2))
    assert_true(idx[unsafe_offset=3] == Int32(0) and idx[unsafe_offset=4] == Int32(2) and idx[unsafe_offset=5] == Int32(3))

    r.pts[unsafe_offset=0].unsafe_free(); idx.unsafe_free()
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
    assert_true(r.n_verts[unsafe_offset=0] == Int32(3))
    assert_true(r.n_tris[unsafe_offset=0] == Int32(1))
    var pts = r.pts[unsafe_offset=0]
    assert_true(_close(pts[unsafe_offset=0], Float32(0.0)) and _close(pts[unsafe_offset=1], Float32(0.0)) and _close(pts[unsafe_offset=2], Float32(0.0)))
    assert_true(_close(pts[unsafe_offset=3], Float32(1.0)) and _close(pts[unsafe_offset=4], Float32(0.0)) and _close(pts[unsafe_offset=5], Float32(0.0)))
    assert_true(_close(pts[unsafe_offset=6], Float32(0.0)) and _close(pts[unsafe_offset=7], Float32(1.0)) and _close(pts[unsafe_offset=8], Float32(0.0)))
    var idx = r.idx[unsafe_offset=0]
    assert_true(idx[unsafe_offset=0] == Int32(0) and idx[unsafe_offset=1] == Int32(1) and idx[unsafe_offset=2] == Int32(2))
    assert_true(r.has_uvs[unsafe_offset=0] == Int32(0))
    assert_true(r.has_normals[unsafe_offset=0] == Int32(0))

    pts.unsafe_free(); idx.unsafe_free()
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
    assert_true(r.n_verts[unsafe_offset=0] == Int32(3))
    assert_true(r.n_tris[unsafe_offset=0] == Int32(1))
    var pts = r.pts[unsafe_offset=0]
    assert_true(_close(pts[unsafe_offset=0], Float32(2.0)))
    assert_true(_close(pts[unsafe_offset=4], Float32(2.0)))
    assert_true(_close(pts[unsafe_offset=8], Float32(2.0)))
    var idx = r.idx[unsafe_offset=0]
    assert_true(idx[unsafe_offset=0] == Int32(0) and idx[unsafe_offset=1] == Int32(1) and idx[unsafe_offset=2] == Int32(2))

    pts.unsafe_free(); idx.unsafe_free()
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
    assert_true(r.n_verts[unsafe_offset=0] == Int32(3))
    var pts = r.pts[unsafe_offset=0]
    assert_true(_close(pts[unsafe_offset=3], Float32(3.5)))
    assert_true(_close(pts[unsafe_offset=7], Float32(3.5)))

    pts.unsafe_free(); r.idx[unsafe_offset=0].unsafe_free()
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
    assert_true(r.n_verts[unsafe_offset=0] == Int32(4))
    # cnt-2 = 2 triangles: fan (face_idx[0], face_idx[ti+1], face_idx[ti+2])
    assert_true(r.n_tris[unsafe_offset=0] == Int32(2))
    var idx = r.idx[unsafe_offset=0]
    assert_true(idx[unsafe_offset=0] == Int32(0) and idx[unsafe_offset=1] == Int32(1) and idx[unsafe_offset=2] == Int32(2))
    assert_true(idx[unsafe_offset=3] == Int32(0) and idx[unsafe_offset=4] == Int32(2) and idx[unsafe_offset=5] == Int32(3))

    r.pts[unsafe_offset=0].unsafe_free(); idx.unsafe_free()
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
    assert_true(r.n_tris[unsafe_offset=0] == Int32(1))
    var idx = r.idx[unsafe_offset=0]
    assert_true(idx[unsafe_offset=0] == Int32(0) and idx[unsafe_offset=1] == Int32(1) and idx[unsafe_offset=2] == Int32(2))

    r.pts[unsafe_offset=0].unsafe_free(); idx.unsafe_free()
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
    assert_true(r.has_normals[unsafe_offset=0] == Int32(1))
    assert_true(r.has_uvs[unsafe_offset=0] == Int32(1))
    var nrm = r.normals[unsafe_offset=0]
    assert_true(_close(nrm[unsafe_offset=0], Float32(0.0)) and _close(nrm[unsafe_offset=1], Float32(0.0)) and _close(nrm[unsafe_offset=2], Float32(1.0)))
    assert_true(_close(nrm[unsafe_offset=6], Float32(0.0)) and _close(nrm[unsafe_offset=7], Float32(0.0)) and _close(nrm[unsafe_offset=8], Float32(1.0)))
    var uvs = r.uvs[unsafe_offset=0]
    assert_true(_close(uvs[unsafe_offset=0], Float32(0.0)) and _close(uvs[unsafe_offset=1], Float32(0.0)))
    assert_true(_close(uvs[unsafe_offset=2], Float32(1.0)) and _close(uvs[unsafe_offset=3], Float32(0.0)))
    assert_true(_close(uvs[unsafe_offset=4], Float32(0.0)) and _close(uvs[unsafe_offset=5], Float32(1.0)))

    r.pts[unsafe_offset=0].unsafe_free(); r.idx[unsafe_offset=0].unsafe_free(); nrm.unsafe_free(); uvs.unsafe_free()
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
