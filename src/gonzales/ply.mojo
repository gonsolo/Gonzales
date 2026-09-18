from std.memory import alloc

comptime PLY_X    = 0
comptime PLY_Y    = 1
comptime PLY_Z    = 2
comptime PLY_SKIP = 3
comptime PLY_U    = 4
comptime PLY_V    = 5
comptime PLY_NX   = 6
comptime PLY_NY   = 7
comptime PLY_NZ   = 8
comptime PLY_MAX_PROPS = 32

def _ply_read_line(
    buf:      Pointer[UInt8, MutUntrackedOrigin],
    size:     Int,
    pos:      Int,
    line_buf: Pointer[UInt8, MutUntrackedOrigin],
    max:      Int,
) -> Int:
    var p = pos
    var i = 0
    while p < size and buf[unsafe_offset=p] != UInt8(10):
        if i < max - 1:
            line_buf[unsafe_offset=i] = buf[unsafe_offset=p]
            i += 1
        p += 1
    line_buf[unsafe_offset=i] = UInt8(0)
    if p < size:
        p += 1
    return p

def _ply_word_start(line: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
    var i = 0
    var w = 0
    while line[unsafe_offset=i] != UInt8(0):
        while line[unsafe_offset=i] == UInt8(32) or line[unsafe_offset=i] == UInt8(9):
            i += 1
        if line[unsafe_offset=i] == UInt8(0):
            break
        if w == n:
            return i
        while line[unsafe_offset=i] != UInt8(0) and line[unsafe_offset=i] != UInt8(32) and line[unsafe_offset=i] != UInt8(9):
            i += 1
        w += 1
    return -1

def _ply_word_eq(line: Pointer[UInt8, MutUntrackedOrigin], n: Int, literal: StringLiteral) -> Bool:
    var si = _ply_word_start(line, n)
    if si < 0:
        return False
    var lp = literal.unsafe_ptr()
    var j = 0
    while lp[unsafe_offset=j] != UInt8(0):
        if line[unsafe_offset=si + j] != lp[unsafe_offset=j]:
            return False
        j += 1
    var next = line[unsafe_offset=si + j]
    return next == UInt8(0) or next == UInt8(32) or next == UInt8(9)

def _ply_word_to_int(line: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Int:
    var si = _ply_word_start(line, n)
    if si < 0:
        return -1
    var v = 0
    var i = si
    while line[unsafe_offset=i] >= UInt8(48) and line[unsafe_offset=i] <= UInt8(57):
        v = v * 10 + Int(line[unsafe_offset=i]) - 48
        i += 1
    return v

# Parses the n-th whitespace-separated token on an ASCII PLY vertex/face
# line as a float (sign, integer part, optional fraction, optional exponent).
# Self-contained (no lexer.mojo dependency) to match this file's existing
# hand-rolled word-parsing style.
def _ply_word_to_float(line: Pointer[UInt8, MutUntrackedOrigin], n: Int) -> Float32:
    var si = _ply_word_start(line, n)
    if si < 0:
        return Float32(0.0)
    var i = si
    var neg = False
    if line[unsafe_offset=i] == UInt8(45):  # '-'
        neg = True
        i += 1
    var int_part = Float64(0)
    while line[unsafe_offset=i] >= UInt8(48) and line[unsafe_offset=i] <= UInt8(57):
        int_part = int_part * 10 + Float64(Int(line[unsafe_offset=i]) - 48)
        i += 1
    var v = int_part
    if line[unsafe_offset=i] == UInt8(46):  # '.'
        i += 1
        var scale = Float64(0.1)
        while line[unsafe_offset=i] >= UInt8(48) and line[unsafe_offset=i] <= UInt8(57):
            v += Float64(Int(line[unsafe_offset=i]) - 48) * scale
            scale *= 0.1
            i += 1
    if line[unsafe_offset=i] == UInt8(101) or line[unsafe_offset=i] == UInt8(69):  # 'e' / 'E'
        i += 1
        var exp_neg = False
        if line[unsafe_offset=i] == UInt8(45):
            exp_neg = True
            i += 1
        elif line[unsafe_offset=i] == UInt8(43):  # '+'
            i += 1
        var exp_val = 0
        while line[unsafe_offset=i] >= UInt8(48) and line[unsafe_offset=i] <= UInt8(57):
            exp_val = exp_val * 10 + Int(line[unsafe_offset=i]) - 48
            i += 1
        var factor = Float64(1.0)
        for _ in range(exp_val):
            factor *= 10.0
        v = (v / factor) if exp_neg else (v * factor)
    if neg:
        v = -v
    return Float32(v)

# Returns byte size of a PLY scalar type name (e.g. "float", "double", "uchar", "int").
# Returns 4 for unknown types (safe default for float/int).
def _ply_type_size(line: Pointer[UInt8, MutUntrackedOrigin], word_n: Int) -> Int:
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
def _ply_f32_le(buf: Pointer[UInt8, MutUntrackedOrigin], pos: Int) -> Float32:
    return (buf.unsafe_offset(pos)).unsafe_bitcast[Float32]()[unsafe_offset=0]

@always_inline
def _ply_i32_le(buf: Pointer[UInt8, MutUntrackedOrigin], pos: Int) -> Int32:
    return (buf.unsafe_offset(pos)).unsafe_bitcast[Int32]()[unsafe_offset=0]

@always_inline
def _ply_u8_at(buf: Pointer[UInt8, MutUntrackedOrigin], pos: Int) -> Int:
    return Int(buf[unsafe_offset=pos])

def _ply_f32_be(buf: Pointer[UInt8, MutUntrackedOrigin], pos: Int) -> Float32:
    var tmp = alloc[UInt8](4)
    tmp[unsafe_offset=0] = buf[unsafe_offset=pos + 3]; tmp[unsafe_offset=1] = buf[unsafe_offset=pos + 2]
    tmp[unsafe_offset=2] = buf[unsafe_offset=pos + 1]; tmp[unsafe_offset=3] = buf[unsafe_offset=pos + 0]
    var v = tmp.unsafe_bitcast[Float32]()[unsafe_offset=0]
    tmp.unsafe_free()
    return v

def _ply_i32_be(buf: Pointer[UInt8, MutUntrackedOrigin], pos: Int) -> Int32:
    var tmp = alloc[UInt8](4)
    tmp[unsafe_offset=0] = buf[unsafe_offset=pos + 3]; tmp[unsafe_offset=1] = buf[unsafe_offset=pos + 2]
    tmp[unsafe_offset=2] = buf[unsafe_offset=pos + 1]; tmp[unsafe_offset=3] = buf[unsafe_offset=pos + 0]
    var v = tmp.unsafe_bitcast[Int32]()[unsafe_offset=0]
    tmp.unsafe_free()
    return v

# Read a 64-bit double and return as Float32 (for double-precision PLY positions).
def _ply_f64_le(buf: Pointer[UInt8, MutUntrackedOrigin], pos: Int) -> Float32:
    var tmp = alloc[UInt8](8)
    for k in range(8):
        tmp[unsafe_offset=k] = buf[unsafe_offset=pos + k]
    var d = tmp.unsafe_bitcast[Float64]()[unsafe_offset=0]
    tmp.unsafe_free()
    return Float32(d)

def _ply_f64_be(buf: Pointer[UInt8, MutUntrackedOrigin], pos: Int) -> Float32:
    var tmp = alloc[UInt8](8)
    for k in range(8):
        tmp[unsafe_offset=k] = buf[unsafe_offset=pos + 7 - k]
    var d = tmp.unsafe_bitcast[Float64]()[unsafe_offset=0]
    tmp.unsafe_free()
    return Float32(d)

# Read a count from a face list field. type_size is 1, 2, or 4.
def _ply_read_count(buf: Pointer[UInt8, MutUntrackedOrigin], pos: Int, type_size: Int, le: Bool) -> Int:
    if type_size == 1:
        return Int(buf[unsafe_offset=pos])
    if type_size == 2:
        if le:
            return Int((buf.unsafe_offset(pos)).unsafe_bitcast[UInt16]()[unsafe_offset=0])
        else:
            return Int(UInt16(buf[unsafe_offset=pos]) << 8 | UInt16(buf[unsafe_offset=pos + 1]))
    # 4-byte count
    if le:
        return Int(_ply_i32_le(buf, pos))
    else:
        return Int(_ply_i32_be(buf, pos))

def load_ply(
    path_cstr:   Pointer[UInt8, MutUntrackedOrigin],
    out_pts:     Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin],
    out_n_verts: Pointer[Int32, MutUntrackedOrigin],
    out_idx:     Pointer[Pointer[Int32, MutUntrackedOrigin], MutUntrackedOrigin],
    out_n_tris:  Pointer[Int32, MutUntrackedOrigin],
    out_uvs:     Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin],
    out_has_uvs: Pointer[Int32, MutUntrackedOrigin],
    out_normals: Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin],
    out_has_normals: Pointer[Int32, MutUntrackedOrigin],
) -> Int32:
    var path_str = String(unsafe_from_utf8_ptr=path_cstr.as_imm())
    var file_buf: Pointer[UInt8, MutUntrackedOrigin]
    var file_size: Int
    try:
        var f = open(path_str, "r")
        var bytes = f.read_bytes()
        f.close()
        file_size = len(bytes)
        file_buf = alloc[UInt8](file_size + 1)
        for i in range(file_size):
            file_buf[unsafe_offset=i] = bytes[i]
        file_buf[unsafe_offset=file_size] = UInt8(0)
    except:
        # Every failure path here names the file AND the reason. A mesh that
        # fails to load doesn't crash the render -- it just isn't there, and
        # a scene missing its subject still looks like a plausible image
        # (ganesha rendered its backdrop, light and floor, minus the statue).
        print("PLY load FAILED (cannot open/read):", path_str)
        return Int32(0)

    var line_buf = alloc[UInt8](512)
    var pos = 0

    pos = _ply_read_line(file_buf, file_size, pos, line_buf, 512)
    if not _ply_word_eq(line_buf, 0, "ply"):
        # Most likely a still-compressed file: gzip's magic (0x1f 0x8b) is
        # not "ply". The .ply.gz decompression lives in pbrt_parser.mojo's
        # plymesh handler, which passes the decompressed sibling's path.
        if file_size >= 2 and file_buf[unsafe_offset=0] == UInt8(0x1f) and file_buf[unsafe_offset=1] == UInt8(0x8b):
            print("PLY load FAILED (file is still gzip-compressed):", path_str)
        else:
            print("PLY load FAILED (missing 'ply' magic, not a PLY file):", path_str)
        line_buf.unsafe_free(); file_buf.unsafe_free()
        return Int32(0)

    var is_le = True       # little-endian
    var is_ascii = False
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
            elif _ply_word_eq(line_buf, 1, "ascii"):
                is_ascii = True
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
                prop_sizes[unsafe_offset=n_props] = Int32(sz)
                prop_is_double[unsafe_offset=n_props] = is_dbl
                if _ply_word_eq(line_buf, 2, "x"):
                    prop_roles[unsafe_offset=n_props] = Int32(PLY_X)
                elif _ply_word_eq(line_buf, 2, "y"):
                    prop_roles[unsafe_offset=n_props] = Int32(PLY_Y)
                elif _ply_word_eq(line_buf, 2, "z"):
                    prop_roles[unsafe_offset=n_props] = Int32(PLY_Z)
                elif _ply_word_eq(line_buf, 2, "u") or _ply_word_eq(line_buf, 2, "s"):
                    prop_roles[unsafe_offset=n_props] = Int32(PLY_U)
                elif _ply_word_eq(line_buf, 2, "v") or _ply_word_eq(line_buf, 2, "t"):
                    prop_roles[unsafe_offset=n_props] = Int32(PLY_V)
                elif _ply_word_eq(line_buf, 2, "nx"):
                    prop_roles[unsafe_offset=n_props] = Int32(PLY_NX)
                elif _ply_word_eq(line_buf, 2, "ny"):
                    prop_roles[unsafe_offset=n_props] = Int32(PLY_NY)
                elif _ply_word_eq(line_buf, 2, "nz"):
                    prop_roles[unsafe_offset=n_props] = Int32(PLY_NZ)
                else:
                    prop_roles[unsafe_offset=n_props] = Int32(PLY_SKIP)
                n_props += 1
            elif hstate == 2 and _ply_word_eq(line_buf, 1, "list"):
                # "property list <count_type> <index_type> vertex_indices"
                face_count_size = _ply_type_size(line_buf, 2)
                face_idx_size   = _ply_type_size(line_buf, 3)

    if n_verts <= 0 or n_faces <= 0:
        print("PLY load FAILED (header declares", n_verts, "vertices and",
              n_faces, "faces):", path_str)
        line_buf.unsafe_free(); prop_roles.unsafe_free(); prop_sizes.unsafe_free()
        prop_is_double.unsafe_free(); file_buf.unsafe_free()
        return Int32(0)

    var pts     = alloc[Float32](n_verts * 3)
    var uvs_buf = alloc[Float32](n_verts * 2)
    var nrm_buf = alloc[Float32](n_verts * 3)
    var max_idx = n_faces * 6   # worst case: quads → 2 triangles each
    var idx_buf = alloc[Int32](max_idx)
    var n_tris  = 0
    var found_uvs = False
    var found_normals = False

    # Check if we have any U/V or normal properties
    for pi in range(n_props):
        var role = Int(prop_roles[unsafe_offset=pi])
        if role == PLY_U or role == PLY_V:
            found_uvs = True
        elif role == PLY_NX or role == PLY_NY or role == PLY_NZ:
            found_normals = True

    for v in range(n_verts):
        var vx = Float32(0); var vy = Float32(0); var vz = Float32(0)
        var vu = Float32(0); var vv = Float32(0)
        var vnx = Float32(0); var vny = Float32(0); var vnz = Float32(0)
        if is_ascii:
            pos = _ply_read_line(file_buf, file_size, pos, line_buf, 512)
        for pi in range(n_props):
            var sz   = Int(prop_sizes[unsafe_offset=pi])
            var role = Int(prop_roles[unsafe_offset=pi])
            var is_d = Int(prop_is_double[unsafe_offset=pi]) == 1
            if role != PLY_SKIP:
                var val: Float32
                if is_ascii:
                    val = _ply_word_to_float(line_buf, pi)
                elif is_d:
                    val = _ply_f64_le(file_buf, pos) if is_le else _ply_f64_be(file_buf, pos)
                else:
                    val = _ply_f32_le(file_buf, pos) if is_le else _ply_f32_be(file_buf, pos)
                if role == PLY_X:
                    vx = val
                elif role == PLY_Y:
                    vy = val
                elif role == PLY_Z:
                    vz = val
                elif role == PLY_U:
                    vu = val
                elif role == PLY_V:
                    vv = val
                elif role == PLY_NX:
                    vnx = val
                elif role == PLY_NY:
                    vny = val
                elif role == PLY_NZ:
                    vnz = val
            if not is_ascii:
                pos += sz
        pts[unsafe_offset=v*3+0] = vx; pts[unsafe_offset=v*3+1] = vy; pts[unsafe_offset=v*3+2] = vz
        uvs_buf[unsafe_offset=v*2+0] = vu; uvs_buf[unsafe_offset=v*2+1] = vv
        nrm_buf[unsafe_offset=v*3+0] = vnx; nrm_buf[unsafe_offset=v*3+1] = vny; nrm_buf[unsafe_offset=v*3+2] = vnz

    for _ in range(n_faces):
        var cnt: Int
        if is_ascii:
            pos = _ply_read_line(file_buf, file_size, pos, line_buf, 512)
            cnt = _ply_word_to_int(line_buf, 0)
        else:
            cnt = _ply_read_count(file_buf, pos, face_count_size, is_le)
            pos += face_count_size
        if cnt < 3 or (not is_ascii and pos + cnt * face_idx_size > file_size):
            if not is_ascii:
                pos += cnt * face_idx_size
            continue
        var face_idx = alloc[Int32](cnt)
        for fi in range(cnt):
            if is_ascii:
                face_idx[unsafe_offset=fi] = Int32(_ply_word_to_int(line_buf, fi + 1))
            elif face_idx_size == 4:
                face_idx[unsafe_offset=fi] = _ply_i32_le(file_buf, pos) if is_le else _ply_i32_be(file_buf, pos)
            elif face_idx_size == 2:
                face_idx[unsafe_offset=fi] = Int32(file_buf.unsafe_offset(pos).unsafe_bitcast[Int16]()[unsafe_offset=0]) if is_le else Int32(Int16(file_buf[unsafe_offset=pos]) << 8 | Int16(file_buf[unsafe_offset=pos+1]))
            else:
                face_idx[unsafe_offset=fi] = Int32(file_buf[unsafe_offset=pos])
            if not is_ascii:
                pos += face_idx_size
        for ti in range(cnt - 2):
            if n_tris * 3 + 2 < max_idx:
                idx_buf[unsafe_offset=n_tris*3+0] = face_idx[unsafe_offset=0]
                idx_buf[unsafe_offset=n_tris*3+1] = face_idx[unsafe_offset=ti + 1]
                idx_buf[unsafe_offset=n_tris*3+2] = face_idx[unsafe_offset=ti + 2]
                n_tris += 1
        face_idx.unsafe_free()

    line_buf.unsafe_free(); prop_roles.unsafe_free(); prop_sizes.unsafe_free()
    prop_is_double.unsafe_free(); file_buf.unsafe_free()

    out_pts[unsafe_offset=0]     = pts
    out_n_verts[unsafe_offset=0] = Int32(n_verts)
    out_idx[unsafe_offset=0]     = idx_buf
    out_n_tris[unsafe_offset=0]  = Int32(n_tris)

    if found_uvs:
        out_uvs[unsafe_offset=0]     = uvs_buf
        out_has_uvs[unsafe_offset=0] = Int32(1)
    else:
        uvs_buf.unsafe_free()
        out_uvs[unsafe_offset=0]     = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
        out_has_uvs[unsafe_offset=0] = Int32(0)

    if found_normals:
        out_normals[unsafe_offset=0]     = nrm_buf
        out_has_normals[unsafe_offset=0] = Int32(1)
    else:
        nrm_buf.unsafe_free()
        out_normals[unsafe_offset=0]     = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
        out_has_normals[unsafe_offset=0] = Int32(0)

    return Int32(1)
