from std.memory import alloc

comptime PLY_X    = 0
comptime PLY_Y    = 1
comptime PLY_Z    = 2
comptime PLY_SKIP = 3
comptime PLY_MAX_PROPS = 32

fn _ply_read_line(
    buf:      UnsafePointer[UInt8, MutAnyOrigin],
    size:     Int,
    pos:      Int,
    line_buf: UnsafePointer[UInt8, MutAnyOrigin],
    max:      Int,
) -> Int:
    var p = pos
    var i = 0
    while p < size and buf[p] != UInt8(10):
        if i < max - 1:
            line_buf[i] = buf[p]
            i += 1
        p += 1
    line_buf[i] = UInt8(0)
    if p < size:
        p += 1
    return p

fn _ply_word_start(line: UnsafePointer[UInt8, MutAnyOrigin], n: Int) -> Int:
    var i = 0
    var w = 0
    while line[i] != UInt8(0):
        while line[i] == UInt8(32) or line[i] == UInt8(9):
            i += 1
        if line[i] == UInt8(0):
            break
        if w == n:
            return i
        while line[i] != UInt8(0) and line[i] != UInt8(32) and line[i] != UInt8(9):
            i += 1
        w += 1
    return -1

fn _ply_word_eq(line: UnsafePointer[UInt8, MutAnyOrigin], n: Int, literal: StringLiteral) -> Bool:
    var si = _ply_word_start(line, n)
    if si < 0:
        return False
    var lp = literal.unsafe_ptr()
    var j = 0
    while lp[j] != UInt8(0):
        if line[si + j] != lp[j]:
            return False
        j += 1
    var next = line[si + j]
    return next == UInt8(0) or next == UInt8(32) or next == UInt8(9)

fn _ply_word_to_int(line: UnsafePointer[UInt8, MutAnyOrigin], n: Int) -> Int:
    var si = _ply_word_start(line, n)
    if si < 0:
        return -1
    var v = 0
    var i = si
    while line[i] >= UInt8(48) and line[i] <= UInt8(57):
        v = v * 10 + Int(line[i]) - 48
        i += 1
    return v

# Returns byte size of a PLY scalar type name (e.g. "float", "double", "uchar", "int").
# Returns 4 for unknown types (safe default for float/int).
fn _ply_type_size(line: UnsafePointer[UInt8, MutAnyOrigin], word_n: Int) -> Int:
    if _ply_word_eq(line, word_n, "float64") or _ply_word_eq(line, word_n, "double"):
        return 8
    if _ply_word_eq(line, word_n, "int8")   or _ply_word_eq(line, word_n, "char"):
        return 1
    if _ply_word_eq(line, word_n, "uint8")  or _ply_word_eq(line, word_n, "uchar"):
        return 1
    if _ply_word_eq(line, word_n, "int16")  or _ply_word_eq(line, word_n, "short"):
        return 2
    if _ply_word_eq(line, word_n, "uint16") or _ply_word_eq(line, word_n, "ushort"):
        return 2
    return 4  # float32, float, int32, int, uint32, uint

@always_inline
fn _ply_f32_le(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Float32:
    return (buf + pos).bitcast[Float32]()[0]

@always_inline
fn _ply_i32_le(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Int32:
    return (buf + pos).bitcast[Int32]()[0]

@always_inline
fn _ply_u8_at(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Int:
    return Int(buf[pos])

fn _ply_f32_be(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Float32:
    var tmp = alloc[UInt8](4)
    tmp[0] = buf[pos + 3]; tmp[1] = buf[pos + 2]
    tmp[2] = buf[pos + 1]; tmp[3] = buf[pos + 0]
    var v = tmp.bitcast[Float32]()[0]
    tmp.free()
    return v

fn _ply_i32_be(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Int32:
    var tmp = alloc[UInt8](4)
    tmp[0] = buf[pos + 3]; tmp[1] = buf[pos + 2]
    tmp[2] = buf[pos + 1]; tmp[3] = buf[pos + 0]
    var v = tmp.bitcast[Int32]()[0]
    tmp.free()
    return v

# Read a 64-bit double and return as Float32 (for double-precision PLY positions).
fn _ply_f64_le(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Float32:
    var tmp = alloc[UInt8](8)
    for k in range(8):
        tmp[k] = buf[pos + k]
    var d = tmp.bitcast[Float64]()[0]
    tmp.free()
    return Float32(d)

fn _ply_f64_be(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Float32:
    var tmp = alloc[UInt8](8)
    for k in range(8):
        tmp[k] = buf[pos + 7 - k]
    var d = tmp.bitcast[Float64]()[0]
    tmp.free()
    return Float32(d)

# Read a count from a face list field. type_size is 1, 2, or 4.
fn _ply_read_count(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int, type_size: Int, le: Bool) -> Int:
    if type_size == 1:
        return Int(buf[pos])
    if type_size == 2:
        if le:
            return Int((buf + pos).bitcast[UInt16]()[0])
        else:
            return Int(UInt16(buf[pos]) << 8 | UInt16(buf[pos + 1]))
    # 4-byte count
    if le:
        return Int(_ply_i32_le(buf, pos))
    else:
        return Int(_ply_i32_be(buf, pos))

fn mojo_load_ply(
    path_cstr:   UnsafePointer[UInt8, MutAnyOrigin],
    out_pts:     UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin],
    out_n_verts: UnsafePointer[Int32, MutAnyOrigin],
    out_idx:     UnsafePointer[UnsafePointer[Int32, MutAnyOrigin], MutAnyOrigin],
    out_n_tris:  UnsafePointer[Int32, MutAnyOrigin],
) -> Int32:
    var path_str = String(unsafe_from_utf8_ptr=path_cstr.as_immutable())
    var file_buf: UnsafePointer[UInt8, MutAnyOrigin]
    var file_size: Int
    try:
        var f = open(path_str, "r")
        var bytes = f.read_bytes()
        f.close()
        file_size = len(bytes)
        file_buf = alloc[UInt8](file_size + 1)
        for i in range(file_size):
            file_buf[i] = bytes[i]
        file_buf[file_size] = UInt8(0)
    except:
        return Int32(0)

    var line_buf = alloc[UInt8](512)
    var pos = 0

    pos = _ply_read_line(file_buf, file_size, pos, line_buf, 512)
    if not _ply_word_eq(line_buf, 0, "ply"):
        line_buf.free(); file_buf.free()
        return Int32(0)

    var is_le = True       # little-endian
    var n_verts = 0
    var n_faces = 0

    # Per-vertex property role (PLY_X/Y/Z/SKIP) and byte size
    var prop_roles = alloc[Int32](PLY_MAX_PROPS)
    var prop_sizes = alloc[Int32](PLY_MAX_PROPS)  # byte size of each property
    var prop_is_double = alloc[Int32](PLY_MAX_PROPS)  # 1 if float64/double
    var n_props = 0

    var face_count_size = 1   # bytes for face vertex-count field (uchar=1 by default)
    var face_idx_size   = 4   # bytes per face vertex index (int=4 by default)
    var hstate = 0            # 0=other, 1=vertex, 2=face

    while pos < file_size:
        pos = _ply_read_line(file_buf, file_size, pos, line_buf, 512)
        if _ply_word_eq(line_buf, 0, "end_header"):
            break
        if _ply_word_eq(line_buf, 0, "format"):
            if _ply_word_eq(line_buf, 1, "binary_big_endian"):
                is_le = False
        elif _ply_word_eq(line_buf, 0, "element"):
            if _ply_word_eq(line_buf, 1, "vertex"):
                n_verts = _ply_word_to_int(line_buf, 2)
                hstate = 1; n_props = 0
            elif _ply_word_eq(line_buf, 1, "face"):
                n_faces = _ply_word_to_int(line_buf, 2)
                hstate = 2
            else:
                hstate = 0
        elif _ply_word_eq(line_buf, 0, "property"):
            if hstate == 1 and not _ply_word_eq(line_buf, 1, "list") and n_props < PLY_MAX_PROPS:
                # "property <type> <name>"
                var sz = _ply_type_size(line_buf, 1)
                var is_dbl = Int32(1) if (_ply_word_eq(line_buf, 1, "float64") or _ply_word_eq(line_buf, 1, "double")) else Int32(0)
                prop_sizes[n_props] = Int32(sz)
                prop_is_double[n_props] = is_dbl
                if _ply_word_eq(line_buf, 2, "x"):
                    prop_roles[n_props] = Int32(PLY_X)
                elif _ply_word_eq(line_buf, 2, "y"):
                    prop_roles[n_props] = Int32(PLY_Y)
                elif _ply_word_eq(line_buf, 2, "z"):
                    prop_roles[n_props] = Int32(PLY_Z)
                else:
                    prop_roles[n_props] = Int32(PLY_SKIP)
                n_props += 1
            elif hstate == 2 and _ply_word_eq(line_buf, 1, "list"):
                # "property list <count_type> <index_type> vertex_indices"
                face_count_size = _ply_type_size(line_buf, 2)
                face_idx_size   = _ply_type_size(line_buf, 3)

    if n_verts <= 0 or n_faces <= 0:
        line_buf.free(); prop_roles.free(); prop_sizes.free()
        prop_is_double.free(); file_buf.free()
        return Int32(0)

    var pts     = alloc[Float32](n_verts * 3)
    var max_idx = n_faces * 6   # worst case: quads → 2 triangles each
    var idx_buf = alloc[Int32](max_idx)
    var n_tris  = 0

    for v in range(n_verts):
        var vx = Float32(0); var vy = Float32(0); var vz = Float32(0)
        for pi in range(n_props):
            var sz   = Int(prop_sizes[pi])
            var role = Int(prop_roles[pi])
            var is_d = Int(prop_is_double[pi]) == 1
            if role == PLY_X or role == PLY_Y or role == PLY_Z:
                var val: Float32
                if is_d:
                    val = _ply_f64_le(file_buf, pos) if is_le else _ply_f64_be(file_buf, pos)
                else:
                    val = _ply_f32_le(file_buf, pos) if is_le else _ply_f32_be(file_buf, pos)
                if role == PLY_X:
                    vx = val
                elif role == PLY_Y:
                    vy = val
                else:
                    vz = val
            pos += sz
        pts[v*3+0] = vx; pts[v*3+1] = vy; pts[v*3+2] = vz

    for _ in range(n_faces):
        var cnt = _ply_read_count(file_buf, pos, face_count_size, is_le)
        pos += face_count_size
        if cnt < 3 or pos + cnt * face_idx_size > file_size:
            pos += cnt * face_idx_size
            continue
        var face_idx = alloc[Int32](cnt)
        for fi in range(cnt):
            if face_idx_size == 4:
                face_idx[fi] = _ply_i32_le(file_buf, pos) if is_le else _ply_i32_be(file_buf, pos)
            elif face_idx_size == 2:
                face_idx[fi] = Int32((file_buf + pos).bitcast[Int16]()[0]) if is_le else Int32(Int16(file_buf[pos]) << 8 | Int16(file_buf[pos+1]))
            else:
                face_idx[fi] = Int32(file_buf[pos])
            pos += face_idx_size
        for ti in range(cnt - 2):
            if n_tris * 3 + 2 < max_idx:
                idx_buf[n_tris*3+0] = face_idx[0]
                idx_buf[n_tris*3+1] = face_idx[ti + 1]
                idx_buf[n_tris*3+2] = face_idx[ti + 2]
                n_tris += 1
        face_idx.free()

    line_buf.free(); prop_roles.free(); prop_sizes.free()
    prop_is_double.free(); file_buf.free()

    out_pts[0]     = pts
    out_n_verts[0] = Int32(n_verts)
    out_idx[0]     = idx_buf
    out_n_tris[0]  = Int32(n_tris)
    return Int32(1)
