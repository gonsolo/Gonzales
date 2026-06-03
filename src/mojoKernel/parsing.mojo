from std.ffi import external_call
from std.math import tan
from std.memory import alloc
from .geometry import Ray_C, Intersection_C, PrimId_C, TriangleMesh_C, Material_C, AreaLight_C
from .transform import mojo_matrix_multiply, mojo_matrix_invert, mojo_transform_points
from .bvh import BVH2Node, SceneDescriptor2_C, mojo_build_bvh2
from .sampling import mojo_gaussian_norm

# ── pbrt Scanner Helpers ──────────────────────────────────────────────

@always_inline
fn _is_ws(b: UInt8) -> Bool:
    return b == UInt8(32) or b == UInt8(9) or b == UInt8(10) or b == UInt8(13)

@always_inline
fn _is_digit(b: UInt8) -> Bool:
    return b >= UInt8(48) and b <= UInt8(57)

fn mojo_scan_int(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    result: UnsafePointer[Int32, MutAnyOrigin],
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    if cur >= len:
        return Int32(0)

    var negative = False
    if bytes[cur] == UInt8(45):       # '-'
        negative = True
        cur += 1
    if cur >= len or not _is_digit(bytes[cur]):
        return Int32(0)
    var value = Int32(0)
    while cur < len and _is_digit(bytes[cur]):
        value = value * Int32(10) + Int32(bytes[cur]) - Int32(48)
        cur += 1
    if negative:
        value = -value
    cursor[0] = Int32(cur)
    result[0] = value
    return Int32(1)

fn mojo_scan_float(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    result: UnsafePointer[Float32, MutAnyOrigin],
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    if cur >= len:
        return Int32(0)

    var leading_negative = bytes[cur] == UInt8(45)

    var int_negative = False
    if cur < len and bytes[cur] == UInt8(45):
        int_negative = True
        cur += 1
    var int_part = Int32(0)
    var int_seen = False
    while cur < len and _is_digit(bytes[cur]):
        int_part = int_part * Int32(10) + Int32(bytes[cur]) - Int32(48)
        cur += 1
        int_seen = True
    if int_negative:
        int_part = -int_part

    var dval = Float64(int_part)

    if cur < len and bytes[cur] == UInt8(46):   # '.'
        cur += 1
        var tenth = Float64(0.1)
        while cur < len and _is_digit(bytes[cur]):
            var d = Float64(Int32(bytes[cur]) - Int32(48))
            if dval < Float64(0.0):
                dval -= tenth * d
            else:
                dval += tenth * d
            tenth *= Float64(0.1)
            cur += 1
    elif not int_seen:
        return Int32(0)

    if cur < len and bytes[cur] == UInt8(101):  # 'e'
        cur += 1
        while cur < len and _is_ws(bytes[cur]):
            cur += 1
        var exp_negative = False
        if cur < len and bytes[cur] == UInt8(45):
            exp_negative = True
            cur += 1
        var exp_val = Int32(0)
        while cur < len and _is_digit(bytes[cur]):
            exp_val = exp_val * Int32(10) + Int32(bytes[cur]) - Int32(48)
            cur += 1
        if exp_negative:
            exp_val = -exp_val
        var factor = Float64(1.0)
        var abs_exp = exp_val if exp_val >= 0 else -exp_val
        for _ in range(Int(abs_exp)):
            factor *= Float64(10.0)
        if exp_val < 0:
            dval /= factor
        else:
            dval *= factor

    var f = Float32(dval)
    if leading_negative and int_part == Int32(0):
        f = -f

    cursor[0] = Int32(cur)
    result[0] = f
    return Int32(1)

fn mojo_count_floats(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: Int32,
) -> Int32:
    var cur = Int(cursor)
    var len = Int(length)
    var count = Int32(0)
    while True:
        while cur < len and _is_ws(bytes[cur]):
            cur += 1
        if cur >= len:
            break
        if cur < len and bytes[cur] == UInt8(45):   # optional '-'
            cur += 1
        var int_seen = False
        while cur < len and _is_digit(bytes[cur]):
            int_seen = True
            cur += 1
        if cur < len and bytes[cur] == UInt8(46):   # '.'
            cur += 1
            while cur < len and _is_digit(bytes[cur]):
                cur += 1
        elif not int_seen:
            break
        if cur < len and bytes[cur] == UInt8(101):  # 'e'
            cur += 1
            if cur < len and bytes[cur] == UInt8(45):
                cur += 1
            while cur < len and _is_digit(bytes[cur]):
                cur += 1
        count += Int32(1)
    return count

fn mojo_scan_floats(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    result: UnsafePointer[Float32, MutAnyOrigin],
    max_count: Int32,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    var count = Int32(0)
    while count < max_count:
        while cur < len and _is_ws(bytes[cur]):
            cur += 1
        if cur >= len:
            break
        var leading_negative = bytes[cur] == UInt8(45)
        var int_negative = False
        if cur < len and bytes[cur] == UInt8(45):
            int_negative = True
            cur += 1
        var int_part = Int32(0)
        var int_seen = False
        while cur < len and _is_digit(bytes[cur]):
            int_part = int_part * Int32(10) + Int32(bytes[cur]) - Int32(48)
            cur += 1
            int_seen = True
        if int_negative:
            int_part = -int_part
        var dval = Float64(int_part)
        if cur < len and bytes[cur] == UInt8(46):
            cur += 1
            var tenth = Float64(0.1)
            while cur < len and _is_digit(bytes[cur]):
                var d = Float64(Int32(bytes[cur]) - Int32(48))
                if dval < Float64(0.0):
                    dval -= tenth * d
                else:
                    dval += tenth * d
                tenth *= Float64(0.1)
                cur += 1
        elif not int_seen:
            break
        if cur < len and bytes[cur] == UInt8(101):
            cur += 1
            while cur < len and _is_ws(bytes[cur]):
                cur += 1
            var exp_negative = False
            if cur < len and bytes[cur] == UInt8(45):
                exp_negative = True
                cur += 1
            var exp_val = Int32(0)
            while cur < len and _is_digit(bytes[cur]):
                exp_val = exp_val * Int32(10) + Int32(bytes[cur]) - Int32(48)
                cur += 1
            if exp_negative:
                exp_val = -exp_val
            var factor = Float64(1.0)
            var abs_exp = exp_val if exp_val >= 0 else -exp_val
            for _ in range(Int(abs_exp)):
                factor *= Float64(10.0)
            if exp_val < 0:
                dval /= factor
            else:
                dval *= factor
        var f = Float32(dval)
        if leading_negative and int_part == Int32(0):
            f = -f
        result[Int(count)] = f
        count += Int32(1)
    cursor[0] = Int32(cur)
    return count

fn mojo_count_ints(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: Int32,
) -> Int32:
    var cur = Int(cursor)
    var len = Int(length)
    var count = Int32(0)
    while True:
        while cur < len and _is_ws(bytes[cur]):
            cur += 1
        if cur >= len:
            break
        if cur < len and bytes[cur] == UInt8(45):
            cur += 1
        if cur >= len or not _is_digit(bytes[cur]):
            break
        while cur < len and _is_digit(bytes[cur]):
            cur += 1
        count += Int32(1)
    return count

fn mojo_scan_ints(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    result: UnsafePointer[Int32, MutAnyOrigin],
    max_count: Int32,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    var count = Int32(0)
    while count < max_count:
        while cur < len and _is_ws(bytes[cur]):
            cur += 1
        if cur >= len:
            break
        var negative = False
        if bytes[cur] == UInt8(45):
            negative = True
            cur += 1
        if cur >= len or not _is_digit(bytes[cur]):
            break
        var value = Int32(0)
        while cur < len and _is_digit(bytes[cur]):
            value = value * Int32(10) + Int32(bytes[cur]) - Int32(48)
            cur += 1
        if negative:
            value = -value
        result[Int(count)] = value
        count += Int32(1)
    cursor[0] = Int32(cur)
    return count

fn mojo_scan_char(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    expected: UInt8,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    cursor[0] = Int32(cur)
    if cur >= len or bytes[cur] != expected:
        return Int32(0)
    cursor[0] = Int32(cur + 1)
    return Int32(1)

fn mojo_peek_char(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    expected: UInt8,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    cursor[0] = Int32(cur)
    if cur >= len or bytes[cur] != expected:
        return Int32(0)
    return Int32(1)

fn mojo_scan_token(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    delims: UnsafePointer[UInt8, MutAnyOrigin],
    n_delims: Int32,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    max_buf: Int32,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    cursor[0] = Int32(cur)
    if cur >= len:
        if max_buf > 0:
            buf[0] = UInt8(0)
        return Int32(-1)
    var n = Int(n_delims)
    var written = Int32(0)
    while cur < len:
        var b = bytes[cur]
        var is_delim = False
        for i in range(n):
            if delims[i] == b:
                is_delim = True
                break
        if is_delim:
            break
        if written < max_buf - 1:
            buf[Int(written)] = b
        written += Int32(1)
        cur += 1
    var cap = Int(written) if Int(written) < Int(max_buf) - 1 else Int(max_buf) - 1
    if max_buf > 0:
        buf[cap] = UInt8(0)
    cursor[0] = Int32(cur)
    return written

fn mojo_parse_quoted_string(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    max_buf: Int32,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    cursor[0] = Int32(cur)
    if cur >= len or bytes[cur] != UInt8(34):   # '"' = 34
        return Int32(-1)
    cur += 1  # opening '"'
    var written = Int32(0)
    while cur < len and bytes[cur] != UInt8(34):
        if written < max_buf - 1:
            buf[Int(written)] = bytes[cur]
        written += Int32(1)
        cur += 1
    if cur < len:
        cur += 1  # closing '"'
    if max_buf > 0:
        var cap = Int(written) if Int(written) < Int(max_buf) - 1 else Int(max_buf) - 1
        buf[cap] = UInt8(0)
    cursor[0] = Int32(cur)
    return written

fn mojo_parse_param_header(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    type_buf: UnsafePointer[UInt8, MutAnyOrigin],
    type_max: Int32,
    name_buf: UnsafePointer[UInt8, MutAnyOrigin],
    name_max: Int32,
    is_array: UnsafePointer[Int32, MutAnyOrigin],
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    cursor[0] = Int32(cur)
    if cur >= len or bytes[cur] != UInt8(34):
        return Int32(0)
    cur += 1  # opening '"'
    # type: read until whitespace or '"'
    var t = Int32(0)
    while cur < len and not _is_ws(bytes[cur]) and bytes[cur] != UInt8(34):
        if t < type_max - 1:
            type_buf[Int(t)] = bytes[cur]
        t += Int32(1)
        cur += 1
    if type_max > 0:
        var cap = Int(t) if Int(t) < Int(type_max) - 1 else Int(type_max) - 1
        type_buf[cap] = UInt8(0)
    # skip separator whitespace
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    # name: read until '"'
    var n = Int32(0)
    while cur < len and bytes[cur] != UInt8(34):
        if n < name_max - 1:
            name_buf[Int(n)] = bytes[cur]
        n += Int32(1)
        cur += 1
    if name_max > 0:
        var cap = Int(n) if Int(n) < Int(name_max) - 1 else Int(name_max) - 1
        name_buf[cap] = UInt8(0)
    if cur < len and bytes[cur] == UInt8(34):
        cur += 1  # closing '"'
    # skip ws, check for '['
    while cur < len and _is_ws(bytes[cur]):
        cur += 1
    if cur < len and bytes[cur] == UInt8(91):   # '[' = 91
        cur += 1
        is_array[0] = Int32(1)
    else:
        is_array[0] = Int32(0)
    cursor[0] = Int32(cur)
    return Int32(1)


# ── PbrtScanner_Mojo ──────────────────────────────────────────────────────────

struct PbrtScanner_Mojo:
    var buffer: UnsafePointer[UInt8, MutAnyOrigin]
    var total_bytes: Int32
    var cursor: Int32
    var is_at_end: Int32


@always_inline
fn _scanner_cursor_ptr(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin]()


@always_inline
fn _scanner_call_int(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], result: UnsafePointer[Int32, MutAnyOrigin]) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_scan_int(handle[0].buffer, handle[0].total_bytes, cur, result)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


@always_inline
fn _scanner_call_float(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], result: UnsafePointer[Float32, MutAnyOrigin]) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_scan_float(handle[0].buffer, handle[0].total_bytes, cur, result)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


fn mojo_scanner_new(path: UnsafePointer[UInt8, MutAnyOrigin]) -> UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]:
    var mode = alloc[UInt8](3)
    mode[0] = UInt8(114)   # 'r'
    mode[1] = UInt8(98)    # 'b'
    mode[2] = UInt8(0)
    var fp = external_call["fopen", UnsafePointer[UInt8, MutAnyOrigin],
        UnsafePointer[UInt8, MutAnyOrigin], UnsafePointer[UInt8, MutAnyOrigin]](path, mode)
    mode.free()
    var handle = alloc[PbrtScanner_Mojo](1)
    if not fp:
        handle[0].buffer = UnsafePointer[UInt8, MutAnyOrigin]()
        handle[0].total_bytes = Int32(0)
        handle[0].cursor = Int32(0)
        handle[0].is_at_end = Int32(1)
        return handle
    _ = external_call["fseek", Int32, UnsafePointer[UInt8, MutAnyOrigin], Int64, Int32](fp, Int64(0), Int32(2))
    var size = external_call["ftell", Int64, UnsafePointer[UInt8, MutAnyOrigin]](fp)
    _ = external_call["fseek", Int32, UnsafePointer[UInt8, MutAnyOrigin], Int64, Int32](fp, Int64(0), Int32(0))
    var buf = alloc[UInt8](Int(size) + 1)
    _ = external_call["fread", Int, UnsafePointer[UInt8, MutAnyOrigin], Int, Int, UnsafePointer[UInt8, MutAnyOrigin]](buf, 1, Int(size), fp)
    _ = external_call["fclose", Int32, UnsafePointer[UInt8, MutAnyOrigin]](fp)
    buf[Int(size)] = UInt8(0)
    handle[0].buffer = buf
    handle[0].total_bytes = Int32(size)
    handle[0].cursor = Int32(0)
    handle[0].is_at_end = Int32(0)
    return handle


fn mojo_scanner_new_from_bytes(bytes: UnsafePointer[UInt8, MutAnyOrigin], length: Int32) -> UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]:
    var handle = alloc[PbrtScanner_Mojo](1)
    var buf = alloc[UInt8](Int(length) + 1)
    for i in range(Int(length)):
        buf[i] = bytes[i]
    buf[Int(length)] = UInt8(0)
    handle[0].buffer = buf
    handle[0].total_bytes = length
    handle[0].cursor = Int32(0)
    handle[0].is_at_end = Int32(0)
    return handle


fn mojo_scanner_free(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]):
    if handle[0].buffer:
        handle[0].buffer.free()
    handle.free()


fn mojo_scanner_is_at_end(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]) -> Int32:
    return handle[0].is_at_end


fn mojo_scanner_scan_location(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]) -> Int32:
    return handle[0].cursor


fn mojo_scanner_peek_char(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], expected: UInt8) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_peek_char(handle[0].buffer, handle[0].total_bytes, cur, expected)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


fn mojo_scanner_scan_char(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], expected: UInt8) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_scan_char(handle[0].buffer, handle[0].total_bytes, cur, expected)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


fn mojo_scanner_scan_int(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], result: UnsafePointer[Int32, MutAnyOrigin]) -> Int32:
    return _scanner_call_int(handle, result)


fn mojo_scanner_scan_float(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], result: UnsafePointer[Float32, MutAnyOrigin]) -> Int32:
    return _scanner_call_float(handle, result)


fn mojo_scanner_count_floats(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]) -> Int32:
    return mojo_count_floats(handle[0].buffer, handle[0].total_bytes, handle[0].cursor)


fn mojo_scanner_scan_floats(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], dst: UnsafePointer[Float32, MutAnyOrigin], max_count: Int32) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_scan_floats(handle[0].buffer, handle[0].total_bytes, cur, dst, max_count)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


fn mojo_scanner_count_ints(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]) -> Int32:
    return mojo_count_ints(handle[0].buffer, handle[0].total_bytes, handle[0].cursor)


fn mojo_scanner_scan_ints(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], dst: UnsafePointer[Int32, MutAnyOrigin], max_count: Int32) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_scan_ints(handle[0].buffer, handle[0].total_bytes, cur, dst, max_count)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


fn mojo_scanner_parse_quoted_string(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin], buf: UnsafePointer[UInt8, MutAnyOrigin], max_buf: Int32) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_parse_quoted_string(handle[0].buffer, handle[0].total_bytes, cur, buf, max_buf)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


fn mojo_scanner_parse_param_header(
    handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
    type_buf: UnsafePointer[UInt8, MutAnyOrigin], type_max: Int32,
    name_buf: UnsafePointer[UInt8, MutAnyOrigin], name_max: Int32,
    is_array: UnsafePointer[Int32, MutAnyOrigin],
) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_parse_param_header(handle[0].buffer, handle[0].total_bytes, cur,
                                      type_buf, type_max, name_buf, name_max, is_array)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


fn mojo_scanner_scan_token(
    handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
    delims: UnsafePointer[UInt8, MutAnyOrigin], n_delims: Int32,
    buf: UnsafePointer[UInt8, MutAnyOrigin], max_buf: Int32,
) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = mojo_scan_token(handle[0].buffer, handle[0].total_bytes, cur,
                              delims, n_delims, buf, max_buf)
    handle[0].cursor = cur[0]
    if ret < 0:
        handle[0].is_at_end = Int32(1)
    cur.free()
    return ret


# ── ParsedScene constants ─────────────────────────────────────────────────────

comptime PSC_MAX_MESHES = 64
comptime PSC_MAX_NAMED  = 64
comptime PSC_CTM_DEPTH  = 16
comptime PSC_ATTR_DEPTH = 8
comptime PSC_NAME_MAX   = 64
comptime PSC_FILE_MAX   = 256

# ── Output struct ─────────────────────────────────────────────────────────────

struct ParsedScene_Mojo:
    var raster_to_camera: UnsafePointer[Float32, MutAnyOrigin]   # 16 floats, column-major
    var camera_to_world:  UnsafePointer[Float32, MutAnyOrigin]   # 16 floats, column-major
    var materials:        UnsafePointer[Material_C, MutAnyOrigin]
    var material_count:   Int32
    var area_lights:      UnsafePointer[AreaLight_C, MutAnyOrigin]
    var area_light_count: Int32
    var meshes:           UnsafePointer[TriangleMesh_C, MutAnyOrigin]
    var mesh_pts:         UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin]
    var mesh_vis:         UnsafePointer[UnsafePointer[Int64, MutAnyOrigin], MutAnyOrigin]
    var mesh_fis:         UnsafePointer[UnsafePointer[Int64, MutAnyOrigin], MutAnyOrigin]
    var mesh_n_verts:     UnsafePointer[Int32, MutAnyOrigin]
    var mesh_n_tris:      UnsafePointer[Int32, MutAnyOrigin]
    var mesh_count:       Int32
    var bvh_nodes:        UnsafePointer[BVH2Node, MutAnyOrigin]
    var prim_ids:         UnsafePointer[PrimId_C, MutAnyOrigin]
    var bvh_node_count:   Int32
    var prim_count:       Int32
    var film_w:           Int32
    var film_h:           Int32
    var film_iso:         Float32
    var film_max_comp:    Float32
    var film_filename:    UnsafePointer[UInt8, MutAnyOrigin]      # null-terminated
    var filter_sigma:     Float32
    var filter_support_x: Float32
    var filter_support_y: Float32
    var filter_norm_x:    Float32
    var filter_norm_y:    Float32
    var filter_weight:    Float32
    var samples_per_pixel: Int32
    var log2_spp:         Int32
    var n_base4_digits:   Int32
    var max_depth:        Int32
    var rng_seed:         UInt64

# ── Internal parse state ──────────────────────────────────────────────────────

struct _PscState:
    var ctm:       UnsafePointer[Float32, MutAnyOrigin]
    var ctm_stack: UnsafePointer[Float32, MutAnyOrigin]
    var ctm_depth: Int32

    var attr_mat:    UnsafePointer[Int32, MutAnyOrigin]
    var attr_alight: UnsafePointer[Int32, MutAnyOrigin]
    var attr_al_rgb: UnsafePointer[Float32, MutAnyOrigin]
    var attr_depth:  Int32

    var named_names:  UnsafePointer[UInt8, MutAnyOrigin]
    var named_albedo: UnsafePointer[Float32, MutAnyOrigin]
    var n_named:      Int32

    var cur_mat_idx: Int32
    var in_alight:   Int32
    var al_r: Float32
    var al_g: Float32
    var al_b: Float32

    var film_w: Int32
    var film_h: Int32
    var film_iso: Float32
    var film_max_comp: Float32
    var film_filename: UnsafePointer[UInt8, MutAnyOrigin]
    var filter_sigma: Float32
    var filter_support_x: Float32
    var filter_support_y: Float32
    var samples_per_pixel: Int32
    var camera_fov: Float32
    var cam2w_raw: UnsafePointer[Float32, MutAnyOrigin]
    var max_depth: Int32

    var n_meshes:      Int32
    var mesh_pts_list: UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin]
    var mesh_vis_list: UnsafePointer[UnsafePointer[Int64, MutAnyOrigin], MutAnyOrigin]
    var mesh_fis_list: UnsafePointer[UnsafePointer[Int64, MutAnyOrigin], MutAnyOrigin]
    var mesh_nv:       UnsafePointer[Int32, MutAnyOrigin]
    var mesh_nt:       UnsafePointer[Int32, MutAnyOrigin]
    var mesh_mat_idx:  UnsafePointer[Int32, MutAnyOrigin]
    var mesh_is_al:    UnsafePointer[Int32, MutAnyOrigin]
    var mesh_al_rgb:   UnsafePointer[Float32, MutAnyOrigin]


# ── Utility functions ─────────────────────────────────────────────────────────

fn _psc_strcmp(a: UnsafePointer[UInt8, MutAnyOrigin], b: UnsafePointer[UInt8, MutAnyOrigin]) -> Int32:
    return external_call["strcmp", Int32,
        UnsafePointer[UInt8, MutAnyOrigin], UnsafePointer[UInt8, MutAnyOrigin]](a, b)

fn _psc_streq(a: UnsafePointer[UInt8, MutAnyOrigin], b: StringLiteral) -> Bool:
    var bp = b.unsafe_ptr()
    var i = 0
    while True:
        var ai = a[i]
        var bi = bp[i]
        if ai != bi:
            return False
        if ai == UInt8(0):
            return True
        i += 1
    return False

fn _psc_strncpy(dst: UnsafePointer[UInt8, MutAnyOrigin],
                src: UnsafePointer[UInt8, MutAnyOrigin], n: Int32):
    var i = Int32(0)
    while i < n - Int32(1) and src[Int(i)] != UInt8(0):
        dst[Int(i)] = src[Int(i)]
        i += 1
    dst[Int(i)] = UInt8(0)

fn _psc_identity(m: UnsafePointer[Float32, MutAnyOrigin]):
    for i in range(16):
        m[i] = Float32(0)
    m[0] = Float32(1)
    m[5] = Float32(1)
    m[10] = Float32(1)
    m[15] = Float32(1)

fn _psc_matcopy(dst: UnsafePointer[Float32, MutAnyOrigin],
                src: UnsafePointer[Float32, MutAnyOrigin]):
    for i in range(16):
        dst[i] = src[i]

fn _psc_row_to_col(col_out: UnsafePointer[Float32, MutAnyOrigin],
                   row_in:  UnsafePointer[Float32, MutAnyOrigin]):
    for row in range(4):
        for col in range(4):
            col_out[col * 4 + row] = row_in[row * 4 + col]

fn _psc_type_is_float(t: UnsafePointer[UInt8, MutAnyOrigin]) -> Bool:
    var c = t[0]
    if c == UInt8(102): return True  # 'f' float
    if c == UInt8(114): return True  # 'r' rgb
    if c == UInt8(99):  return True  # 'c' color
    if c == UInt8(110): return True  # 'n' normal
    if c == UInt8(112): return True  # 'p' point/point2/point3
    if c == UInt8(118): return True  # 'v' vector3
    if c == UInt8(115) and t[1] == UInt8(112): return True  # "sp" spectrum
    if c == UInt8(98) and t[1] == UInt8(108): return True   # "bl" blackbody
    return False

fn _psc_type_is_int(t: UnsafePointer[UInt8, MutAnyOrigin]) -> Bool:
    return t[0] == UInt8(105)  # 'i' integer

fn _psc_type_is_str(t: UnsafePointer[UInt8, MutAnyOrigin]) -> Bool:
    if t[0] == UInt8(116): return True  # 't' texture
    if t[0] == UInt8(115) and t[1] == UInt8(116): return True  # "string"
    return False

fn _psc_skip_value(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                   type_buf: UnsafePointer[UInt8, MutAnyOrigin],
                   is_array: Int32):
    var tmp_f = alloc[Float32](65536)
    var tmp_i = alloc[Int32](16384)
    var tmp_s = alloc[UInt8](1024)
    if _psc_type_is_float(type_buf):
        if is_array:
            _ = mojo_scanner_scan_floats(handle, tmp_f, 65536)
        else:
            _ = mojo_scanner_scan_float(handle, tmp_f)
    elif _psc_type_is_int(type_buf):
        if is_array:
            _ = mojo_scanner_scan_ints(handle, tmp_i, 16384)
        else:
            _ = mojo_scanner_scan_int(handle, tmp_i)
    elif _psc_type_is_str(type_buf):
        _ = mojo_scanner_parse_quoted_string(handle, tmp_s, 1024)
        if is_array:
            while mojo_scanner_parse_quoted_string(handle, tmp_s, 1024) >= 0:
                pass
    else:
        var nl_buf = alloc[UInt8](1)
        nl_buf[0] = UInt8(10)
        _ = mojo_scanner_scan_token(handle, nl_buf, 1, tmp_s, 1024)
        nl_buf.free()
    tmp_f.free()
    tmp_i.free()
    tmp_s.free()

fn _psc_skip_params(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]):
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        _psc_skip_value(handle, type_buf, is_array)
        if is_array:
            _ = mojo_scanner_scan_char(handle, UInt8(93))  # ']'
        ia[0] = Int32(0)
        found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    type_buf.free()
    name_buf.free()

fn _psc_skip_line(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin]):
    var nl_buf = alloc[UInt8](1)
    nl_buf[0] = UInt8(10)
    var buf = alloc[UInt8](4096)
    _ = mojo_scanner_scan_token(handle, nl_buf, 1, buf, 4096)
    nl_buf.free()
    buf.free()

fn _psc_scan_one_float(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                       is_array: Int32) -> Float32:
    var vp = alloc[Float32](1)
    vp[0] = Float32(0)
    _ = mojo_scanner_scan_float(handle, vp)
    var v = vp[0]
    vp.free()
    if is_array:
        _ = mojo_scanner_scan_char(handle, UInt8(93))  # ']'
    return v

fn _psc_scan_one_int(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                     is_array: Int32) -> Int32:
    var ip = alloc[Int32](1)
    ip[0] = Int32(0)
    _ = mojo_scanner_scan_int(handle, ip)
    var v = ip[0]
    ip.free()
    if is_array:
        _ = mojo_scanner_scan_char(handle, UInt8(93))  # ']'
    return v

fn _psc_scan_one_str(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                     dst: UnsafePointer[UInt8, MutAnyOrigin], dst_max: Int32,
                     is_array: Int32):
    _ = mojo_scanner_parse_quoted_string(handle, dst, dst_max)
    if is_array:
        _ = mojo_scanner_scan_char(handle, UInt8(93))  # ']'

fn _psc_scan_rgb(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                 rgb: UnsafePointer[Float32, MutAnyOrigin],
                 is_array: Int32):
    _ = mojo_scanner_scan_float(handle, rgb + 0)
    _ = mojo_scanner_scan_float(handle, rgb + 1)
    _ = mojo_scanner_scan_float(handle, rgb + 2)
    if is_array:
        _ = mojo_scanner_scan_char(handle, UInt8(93))  # ']'

fn _psc_state_new() -> UnsafePointer[_PscState, MutAnyOrigin]:
    var s = alloc[_PscState](1)

    s[0].ctm       = alloc[Float32](16)
    s[0].ctm_stack = alloc[Float32](PSC_CTM_DEPTH * 16)
    s[0].ctm_depth = Int32(0)
    _psc_identity(s[0].ctm)

    s[0].attr_mat    = alloc[Int32](PSC_ATTR_DEPTH)
    s[0].attr_alight = alloc[Int32](PSC_ATTR_DEPTH)
    s[0].attr_al_rgb = alloc[Float32](PSC_ATTR_DEPTH * 3)
    s[0].attr_depth  = Int32(0)

    s[0].named_names  = alloc[UInt8](PSC_MAX_NAMED * PSC_NAME_MAX)
    s[0].named_albedo = alloc[Float32](PSC_MAX_NAMED * 3)
    s[0].n_named      = Int32(0)

    s[0].cur_mat_idx = Int32(-1)
    s[0].in_alight   = Int32(0)
    s[0].al_r = Float32(0); s[0].al_g = Float32(0); s[0].al_b = Float32(0)

    s[0].film_w    = Int32(512)
    s[0].film_h    = Int32(512)
    s[0].film_iso  = Float32(100)
    s[0].film_max_comp = Float32(0)
    s[0].film_filename = alloc[UInt8](PSC_FILE_MAX)
    # default filename "gonzales.exr"
    var defname = "gonzales.exr"
    for i in range(12):
        s[0].film_filename[i] = defname.unsafe_ptr()[i]
    s[0].film_filename[12] = UInt8(0)

    s[0].filter_sigma     = Float32(0.5)
    s[0].filter_support_x = Float32(1.5)
    s[0].filter_support_y = Float32(1.5)
    s[0].samples_per_pixel = Int32(1)
    s[0].camera_fov = Float32(30)
    s[0].cam2w_raw  = alloc[Float32](16)
    _psc_identity(s[0].cam2w_raw)
    s[0].max_depth = Int32(5)

    s[0].n_meshes      = Int32(0)
    s[0].mesh_pts_list = alloc[UnsafePointer[Float32, MutAnyOrigin]](PSC_MAX_MESHES)
    s[0].mesh_vis_list = alloc[UnsafePointer[Int64, MutAnyOrigin]](PSC_MAX_MESHES)
    s[0].mesh_fis_list = alloc[UnsafePointer[Int64, MutAnyOrigin]](PSC_MAX_MESHES)
    s[0].mesh_nv       = alloc[Int32](PSC_MAX_MESHES)
    s[0].mesh_nt       = alloc[Int32](PSC_MAX_MESHES)
    s[0].mesh_mat_idx  = alloc[Int32](PSC_MAX_MESHES)
    s[0].mesh_is_al    = alloc[Int32](PSC_MAX_MESHES)
    s[0].mesh_al_rgb   = alloc[Float32](PSC_MAX_MESHES * 3)

    return s

fn _psc_state_free(s: UnsafePointer[_PscState, MutAnyOrigin]):
    s[0].ctm.free()
    s[0].ctm_stack.free()
    s[0].attr_mat.free()
    s[0].attr_alight.free()
    s[0].attr_al_rgb.free()
    s[0].named_names.free()
    s[0].named_albedo.free()
    s[0].film_filename.free()
    s[0].cam2w_raw.free()
    s[0].mesh_pts_list.free()
    s[0].mesh_vis_list.free()
    s[0].mesh_fis_list.free()
    s[0].mesh_nv.free()
    s[0].mesh_nt.free()
    s[0].mesh_mat_idx.free()
    s[0].mesh_is_al.free()
    s[0].mesh_al_rgb.free()
    s.free()

fn _psc_ctm_push(s: UnsafePointer[_PscState, MutAnyOrigin]):
    var d = Int(s[0].ctm_depth)
    if d < PSC_CTM_DEPTH:
        for i in range(16):
            s[0].ctm_stack[d * 16 + i] = s[0].ctm[i]
        s[0].ctm_depth += 1

fn _psc_ctm_pop(s: UnsafePointer[_PscState, MutAnyOrigin]):
    if s[0].ctm_depth > 0:
        s[0].ctm_depth -= 1
        var d = Int(s[0].ctm_depth)
        for i in range(16):
            s[0].ctm[i] = s[0].ctm_stack[d * 16 + i]

fn _psc_handle_integrator(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                          s: UnsafePointer[_PscState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = mojo_scanner_parse_quoted_string(handle, sbuf, 64)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "maxdepth") and _psc_type_is_int(type_buf):
            s[0].max_depth = _psc_scan_one_int(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

fn _psc_handle_sampler(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                       s: UnsafePointer[_PscState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = mojo_scanner_parse_quoted_string(handle, sbuf, 64)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if (_psc_streq(name_buf, "pixelsamples") or _psc_streq(name_buf, "samples")) and _psc_type_is_int(type_buf):
            s[0].samples_per_pixel = _psc_scan_one_int(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

fn _psc_handle_filter(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                      s: UnsafePointer[_PscState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = mojo_scanner_parse_quoted_string(handle, sbuf, 64)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "xradius") and _psc_type_is_float(type_buf):
            s[0].filter_support_x = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "yradius") and _psc_type_is_float(type_buf):
            s[0].filter_support_y = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "sigma") and _psc_type_is_float(type_buf):
            s[0].filter_sigma = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

fn _psc_handle_film(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                    s: UnsafePointer[_PscState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = mojo_scanner_parse_quoted_string(handle, sbuf, 64)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "xresolution") and _psc_type_is_int(type_buf):
            s[0].film_w = _psc_scan_one_int(handle, is_array)
        elif _psc_streq(name_buf, "yresolution") and _psc_type_is_int(type_buf):
            s[0].film_h = _psc_scan_one_int(handle, is_array)
        elif _psc_streq(name_buf, "filename") and _psc_type_is_str(type_buf):
            _psc_scan_one_str(handle, s[0].film_filename, PSC_FILE_MAX, is_array)
        elif _psc_streq(name_buf, "iso") and _psc_type_is_float(type_buf):
            s[0].film_iso = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "maxcomponentvalue") and _psc_type_is_float(type_buf):
            s[0].film_max_comp = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

fn _psc_handle_camera(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                      s: UnsafePointer[_PscState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = mojo_scanner_parse_quoted_string(handle, sbuf, 64)
    _psc_matcopy(s[0].cam2w_raw, s[0].ctm)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "fov") and _psc_type_is_float(type_buf):
            s[0].camera_fov = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

fn _psc_handle_transform(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                         s: UnsafePointer[_PscState, MutAnyOrigin]):
    _ = mojo_scanner_scan_char(handle, UInt8(91))  # '['
    for i in range(16):
        _ = mojo_scanner_scan_float(handle, s[0].ctm + i)
    _ = mojo_scanner_scan_char(handle, UInt8(93))  # ']'

fn _psc_handle_world_begin(s: UnsafePointer[_PscState, MutAnyOrigin]):
    _psc_identity(s[0].ctm)
    s[0].ctm_depth = Int32(0)

fn _psc_handle_make_named_material(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                                   s: UnsafePointer[_PscState, MutAnyOrigin]):
    var mat_name = alloc[UInt8](PSC_NAME_MAX)
    _ = mojo_scanner_parse_quoted_string(handle, mat_name, PSC_NAME_MAX)

    var rgb = alloc[Float32](3)
    rgb[0] = Float32(0.5); rgb[1] = Float32(0.5); rgb[2] = Float32(0.5)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "reflectance") and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, rgb, is_array)
        elif _psc_streq(name_buf, "L") and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, rgb, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()

    var idx = Int(s[0].n_named)
    if idx < PSC_MAX_NAMED:
        _psc_strncpy(s[0].named_names + idx * PSC_NAME_MAX, mat_name, PSC_NAME_MAX)
        s[0].named_albedo[idx * 3 + 0] = rgb[0]
        s[0].named_albedo[idx * 3 + 1] = rgb[1]
        s[0].named_albedo[idx * 3 + 2] = rgb[2]
        s[0].n_named += 1

    mat_name.free(); type_buf.free(); name_buf.free(); rgb.free()

fn _psc_handle_named_material(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                               s: UnsafePointer[_PscState, MutAnyOrigin]):
    var mat_name = alloc[UInt8](PSC_NAME_MAX)
    _ = mojo_scanner_parse_quoted_string(handle, mat_name, PSC_NAME_MAX)
    s[0].cur_mat_idx = Int32(-1)
    for i in range(Int(s[0].n_named)):
        if _psc_strcmp(s[0].named_names + i * PSC_NAME_MAX, mat_name) == 0:
            s[0].cur_mat_idx = Int32(i)
            break
    mat_name.free()

fn _psc_handle_attribute_begin(s: UnsafePointer[_PscState, MutAnyOrigin]):
    _psc_ctm_push(s)
    var d = Int(s[0].attr_depth)
    if d < PSC_ATTR_DEPTH:
        s[0].attr_mat[d]        = s[0].cur_mat_idx
        s[0].attr_alight[d]     = s[0].in_alight
        s[0].attr_al_rgb[d*3+0] = s[0].al_r
        s[0].attr_al_rgb[d*3+1] = s[0].al_g
        s[0].attr_al_rgb[d*3+2] = s[0].al_b
        s[0].attr_depth += 1

fn _psc_handle_attribute_end(s: UnsafePointer[_PscState, MutAnyOrigin]):
    _psc_ctm_pop(s)
    if s[0].attr_depth > 0:
        s[0].attr_depth -= 1
        var d = Int(s[0].attr_depth)
        s[0].cur_mat_idx = s[0].attr_mat[d]
        s[0].in_alight   = s[0].attr_alight[d]
        s[0].al_r = s[0].attr_al_rgb[d*3+0]
        s[0].al_g = s[0].attr_al_rgb[d*3+1]
        s[0].al_b = s[0].attr_al_rgb[d*3+2]
        s[0].in_alight = Int32(0)

fn _psc_handle_area_light_source(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                                 s: UnsafePointer[_PscState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = mojo_scanner_parse_quoted_string(handle, sbuf, 64)
    s[0].in_alight = Int32(1)
    var rgb = alloc[Float32](3)
    rgb[0] = Float32(1); rgb[1] = Float32(1); rgb[2] = Float32(1)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "L") and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, rgb, is_array)
        elif _psc_streq(name_buf, "scale") and _psc_type_is_float(type_buf):
            var sc = _psc_scan_one_float(handle, is_array)
            rgb[0] *= sc; rgb[1] *= sc; rgb[2] *= sc
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    s[0].al_r = rgb[0]; s[0].al_g = rgb[1]; s[0].al_b = rgb[2]
    sbuf.free(); type_buf.free(); name_buf.free(); rgb.free()

fn _psc_handle_shape(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                     s: UnsafePointer[_PscState, MutAnyOrigin]):
    var shape_type = alloc[UInt8](64)
    _ = mojo_scanner_parse_quoted_string(handle, shape_type, 64)

    if not _psc_streq(shape_type, "trianglemesh"):
        shape_type.free()
        _psc_skip_params(handle)
        return
    shape_type.free()

    var n_meshes = Int(s[0].n_meshes)
    if n_meshes >= PSC_MAX_MESHES:
        _psc_skip_params(handle)
        return

    var tmp_f = alloc[Float32](65536)
    var tmp_i = alloc[Int32](16384)
    var n_pts  = Int32(0)
    var n_idx  = Int32(0)

    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        var is_P = (_psc_streq(name_buf, "P") and _psc_type_is_float(type_buf))
        var is_I = (_psc_streq(name_buf, "indices") and _psc_type_is_int(type_buf))

        if is_P:
            if is_array:
                n_pts = mojo_scanner_scan_floats(handle, tmp_f, 65536)
                _ = mojo_scanner_scan_char(handle, UInt8(93))  # ']'
            else:
                _ = mojo_scanner_scan_float(handle, tmp_f)
                n_pts = Int32(3)
        elif is_I:
            if is_array:
                n_idx = mojo_scanner_scan_ints(handle, tmp_i, 16384)
                _ = mojo_scanner_scan_char(handle, UInt8(93))  # ']'
            else:
                _ = mojo_scanner_scan_int(handle, tmp_i)
                n_idx = Int32(1)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    type_buf.free(); name_buf.free()

    var n_verts = n_pts / Int32(3)
    var n_tris  = n_idx / Int32(3)

    if n_verts <= 0 or n_tris <= 0:
        tmp_f.free(); tmp_i.free()
        return

    # Allocate raw stride-4 points (x,y,z,1)
    var raw_pts = alloc[Float32](Int(n_verts) * 4)
    for v in range(Int(n_verts)):
        raw_pts[v*4+0] = tmp_f[v*3+0]
        raw_pts[v*4+1] = tmp_f[v*3+1]
        raw_pts[v*4+2] = tmp_f[v*3+2]
        raw_pts[v*4+3] = Float32(1)

    # Transform by current CTM
    var fin_pts = alloc[Float32](Int(n_verts) * 4)
    mojo_transform_points(s[0].ctm, raw_pts, n_verts, fin_pts)
    raw_pts.free()

    # Build vertex-index array (Int64) and face-index array
    var vis = alloc[Int64](Int(n_tris) * 3)
    var fis = alloc[Int64](Int(n_tris))
    for t in range(Int(n_tris)):
        vis[t*3+0] = Int64(tmp_i[t*3+0])
        vis[t*3+1] = Int64(tmp_i[t*3+1])
        vis[t*3+2] = Int64(tmp_i[t*3+2])
        fis[t] = Int64(3)

    tmp_f.free(); tmp_i.free()

    s[0].mesh_pts_list[n_meshes] = fin_pts
    s[0].mesh_vis_list[n_meshes] = vis
    s[0].mesh_fis_list[n_meshes] = fis
    s[0].mesh_nv[n_meshes] = n_verts
    s[0].mesh_nt[n_meshes] = n_tris
    s[0].mesh_mat_idx[n_meshes] = s[0].cur_mat_idx
    s[0].mesh_is_al[n_meshes]   = s[0].in_alight
    s[0].mesh_al_rgb[n_meshes*3+0] = s[0].al_r
    s[0].mesh_al_rgb[n_meshes*3+1] = s[0].al_g
    s[0].mesh_al_rgb[n_meshes*3+2] = s[0].al_b
    s[0].n_meshes += 1

fn _psc_parse(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
              s: UnsafePointer[_PscState, MutAnyOrigin]):
    var kw_buf = alloc[UInt8](256)
    var ws_delims = alloc[UInt8](4)
    ws_delims[0] = UInt8(32); ws_delims[1] = UInt8(9)
    ws_delims[2] = UInt8(10); ws_delims[3] = UInt8(13)

    while mojo_scanner_is_at_end(handle) == 0:
        var n = mojo_scanner_scan_token(handle, ws_delims, 4, kw_buf, 256)
        if n < 0:
            break
        if n == 0:
            continue

        if kw_buf[0] == UInt8(35):  # '#'
            _psc_skip_line(handle)
            continue

        if _psc_streq(kw_buf, "Integrator"):
            _psc_handle_integrator(handle, s)
        elif _psc_streq(kw_buf, "Sampler"):
            _psc_handle_sampler(handle, s)
        elif _psc_streq(kw_buf, "PixelFilter"):
            _psc_handle_filter(handle, s)
        elif _psc_streq(kw_buf, "Film"):
            _psc_handle_film(handle, s)
        elif _psc_streq(kw_buf, "Camera"):
            _psc_handle_camera(handle, s)
        elif _psc_streq(kw_buf, "Transform"):
            _psc_handle_transform(handle, s)
        elif _psc_streq(kw_buf, "WorldBegin"):
            _psc_handle_world_begin(s)
        elif _psc_streq(kw_buf, "WorldEnd"):
            break
        elif _psc_streq(kw_buf, "MakeNamedMaterial"):
            _psc_handle_make_named_material(handle, s)
        elif _psc_streq(kw_buf, "NamedMaterial"):
            _psc_handle_named_material(handle, s)
        elif _psc_streq(kw_buf, "Shape"):
            _psc_handle_shape(handle, s)
        elif _psc_streq(kw_buf, "AttributeBegin"):
            _psc_handle_attribute_begin(s)
        elif _psc_streq(kw_buf, "AttributeEnd"):
            _psc_handle_attribute_end(s)
        elif _psc_streq(kw_buf, "AreaLightSource"):
            _psc_handle_area_light_source(handle, s)
        else:
            _ = mojo_scanner_parse_quoted_string(handle, kw_buf, 256)
            _psc_skip_params(handle)

    kw_buf.free()
    ws_delims.free()

fn _psc_make_perspective(fov_deg: Float32, near: Float32,
                         dst: UnsafePointer[Float32, MutAnyOrigin]):
    var half_rad = fov_deg * Float32(3.14159265358979323846) / Float32(360)
    var inv_tan = Float32(1) / tan(half_rad)
    var far = fov_deg
    var t22 = far / (far - near)
    var t23 = -(far * near) / (far - near)
    for i in range(16):
        dst[i] = Float32(0)
    dst[0]  = inv_tan          # M[0,0]
    dst[5]  = inv_tan          # M[1,1]
    dst[10] = t22              # M[2,2]
    dst[11] = Float32(1)       # M[3,2]
    dst[14] = t23              # M[2,3]

fn _psc_make_screen_to_raster(fw: Int32, fh: Int32,
                               smin_x: Float32, smax_x: Float32,
                               smin_y: Float32, smax_y: Float32,
                               dst: UnsafePointer[Float32, MutAnyOrigin]):
    var sx = Float32(fw) / (smax_x - smin_x)
    var sy = Float32(fh) / (smin_y - smax_y)
    var tx = -smin_x * sx
    var ty = -smax_y * sy
    for i in range(16):
        dst[i] = Float32(0)
    dst[0]  = sx           # M[0,0]
    dst[5]  = sy           # M[1,1]
    dst[10] = Float32(1)   # M[2,2]
    dst[15] = Float32(1)   # M[3,3]
    dst[12] = tx           # M[0,3]
    dst[13] = ty           # M[1,3]

fn _psc_finalize(s: UnsafePointer[_PscState, MutAnyOrigin],
                 psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]):

    # ---- Camera matrices ----
    var c2w = alloc[Float32](16)
    _ = mojo_matrix_invert(s[0].cam2w_raw, c2w)
    psc[0].camera_to_world = c2w

    var cts = alloc[Float32](16)
    _psc_make_perspective(s[0].camera_fov, Float32(0.01), cts)

    var frame = Float32(s[0].film_w) / Float32(s[0].film_h)
    var smin_x: Float32; var smax_x: Float32
    var smin_y: Float32; var smax_y: Float32
    if frame >= Float32(1):
        smin_x = -frame; smax_x = frame; smin_y = Float32(-1); smax_y = Float32(1)
    else:
        smin_x = Float32(-1); smax_x = Float32(1)
        smin_y = -Float32(1)/frame; smax_y = Float32(1)/frame

    var str_mat = alloc[Float32](16)
    _psc_make_screen_to_raster(s[0].film_w, s[0].film_h,
                                smin_x, smax_x, smin_y, smax_y, str_mat)

    var rts = alloc[Float32](16)
    _ = mojo_matrix_invert(str_mat, rts)    # rasterToScreen

    var cts_inv = alloc[Float32](16)
    _ = mojo_matrix_invert(cts, cts_inv)    # inverse(cameraToScreen)

    var r2c = alloc[Float32](16)
    mojo_matrix_multiply(cts_inv, rts, r2c) # rasterToCamera
    psc[0].raster_to_camera = r2c

    cts.free(); str_mat.free(); rts.free(); cts_inv.free()

    # ---- Materials ----
    var n_regular = Int(s[0].n_named)

    var n_al = Int32(0)
    for i in range(Int(s[0].n_meshes)):
        if s[0].mesh_is_al[i] != 0:
            n_al += 1

    var n_mats = n_regular + Int(n_al)
    var mats = alloc[Material_C](max(n_mats, 1))
    for i in range(n_regular):
        mats[i].type = Int8(1)  # diffuse
        mats[i].albedoR = s[0].named_albedo[i*3+0]
        mats[i].albedoG = s[0].named_albedo[i*3+1]
        mats[i].albedoB = s[0].named_albedo[i*3+2]
        mats[i].emissionR = Float32(0)
        mats[i].emissionG = Float32(0)
        mats[i].emissionB = Float32(0)

    # ---- Meshes + area lights ----
    var n_meshes = Int(s[0].n_meshes)
    var meshes  = alloc[TriangleMesh_C](max(n_meshes, 1))
    var out_pts = alloc[UnsafePointer[Float32, MutAnyOrigin]](max(n_meshes, 1))
    var out_vis = alloc[UnsafePointer[Int64, MutAnyOrigin]](max(n_meshes, 1))
    var out_fis = alloc[UnsafePointer[Int64, MutAnyOrigin]](max(n_meshes, 1))
    var out_nv  = alloc[Int32](max(n_meshes, 1))
    var out_nt  = alloc[Int32](max(n_meshes, 1))

    var al_list = alloc[AreaLight_C](max(Int(n_al), 1))
    var al_count = Int32(0)

    var al_mat_base = n_regular

    for i in range(n_meshes):
        out_pts[i] = s[0].mesh_pts_list[i]
        out_vis[i] = s[0].mesh_vis_list[i]
        out_fis[i] = s[0].mesh_fis_list[i]
        out_nv[i]  = s[0].mesh_nv[i]
        out_nt[i]  = s[0].mesh_nt[i]
        meshes[i].points       = out_pts[i]
        meshes[i].vertexIndices = out_vis[i]
        meshes[i].faceIndices   = out_fis[i]

        if s[0].mesh_is_al[i] != 0:
            var al_idx = Int(al_count)
            var er = s[0].mesh_al_rgb[i*3+0]
            var eg = s[0].mesh_al_rgb[i*3+1]
            var eb = s[0].mesh_al_rgb[i*3+2]
            al_list[al_idx].meshIdx     = Int32(i)
            al_list[al_idx].triBaseVidx = Int32(0)
            al_list[al_idx].emissionR   = er
            al_list[al_idx].emissionG   = eg
            al_list[al_idx].emissionB   = eb
            al_list[al_idx]._pad        = Int32(0)

            mats[al_mat_base + al_idx].type = Int8(2)  # arealight
            mats[al_mat_base + al_idx].albedoR = Float32(0)
            mats[al_mat_base + al_idx].albedoG = Float32(0)
            mats[al_mat_base + al_idx].albedoB = Float32(0)
            mats[al_mat_base + al_idx].emissionR = er
            mats[al_mat_base + al_idx].emissionG = eg
            mats[al_mat_base + al_idx].emissionB = eb
            al_count += 1

    # ---- BVH construction ----
    var total_tris = Int32(0)
    for i in range(n_meshes):
        total_tris += s[0].mesh_nt[i]

    var prim_bounds = alloc[Float32](Int(total_tris) * 6)
    var tri_mesh    = alloc[Int32](Int(total_tris))
    var tri_local   = alloc[Int32](Int(total_tris))

    var flat_idx = Int32(0)
    for mi in range(n_meshes):
        var pts = s[0].mesh_pts_list[mi]
        var vis = s[0].mesh_vis_list[mi]
        var nt  = Int(s[0].mesh_nt[mi])
        for ti in range(nt):
            var v0 = Int(vis[ti*3+0]) * 4
            var v1 = Int(vis[ti*3+1]) * 4
            var v2 = Int(vis[ti*3+2]) * 4
            var x0 = pts[v0]; var y0 = pts[v0+1]; var z0 = pts[v0+2]
            var x1 = pts[v1]; var y1 = pts[v1+1]; var z1 = pts[v1+2]
            var x2 = pts[v2]; var y2 = pts[v2+1]; var z2 = pts[v2+2]
            var b = Int(flat_idx) * 6
            prim_bounds[b+0] = min(x0, min(x1, x2))
            prim_bounds[b+1] = min(y0, min(y1, y2))
            prim_bounds[b+2] = min(z0, min(z1, z2))
            prim_bounds[b+3] = max(x0, max(x1, x2))
            prim_bounds[b+4] = max(y0, max(y1, y2))
            prim_bounds[b+5] = max(z0, max(z1, z2))
            tri_mesh[Int(flat_idx)]  = Int32(mi)
            tri_local[Int(flat_idx)] = Int32(ti)
            flat_idx += 1

    var max_bvh_nodes = Int(total_tris) * 2 + 4
    var bvh_nodes = alloc[BVH2Node](max_bvh_nodes)
    var bvh_order = alloc[Int32](Int(total_tris))
    var node_count = mojo_build_bvh2(prim_bounds, total_tris, bvh_nodes, bvh_order)

    prim_bounds.free()

    # Build PrimId array in BVH order
    var prim_ids = alloc[PrimId_C](Int(total_tris))

    var mesh_al_idx = alloc[Int32](max(n_meshes, 1))
    var running_al = Int32(0)
    for mi in range(n_meshes):
        if s[0].mesh_is_al[mi] != 0:
            mesh_al_idx[mi] = running_al
            running_al += 1
        else:
            mesh_al_idx[mi] = Int32(-1)

    for k in range(Int(total_tris)):
        var orig = Int(bvh_order[k])
        var mi   = Int(tri_mesh[orig])
        var ti   = Int(tri_local[orig])
        if s[0].mesh_is_al[mi] != 0:
            var al_idx = Int(mesh_al_idx[mi])
            prim_ids[k].type          = Int8(3)
            prim_ids[k].id1           = Int64(al_idx)
            prim_ids[k].id2           = (Int64(mi) << 32) | Int64(ti)
            prim_ids[k].materialIndex = Int64(al_mat_base + al_idx)
        else:
            var mat_idx = Int(s[0].mesh_mat_idx[mi])
            prim_ids[k].type          = Int8(0)
            prim_ids[k].id1           = Int64(mi)
            prim_ids[k].id2           = Int64(ti * 3)
            prim_ids[k].materialIndex = Int64(max(mat_idx, 0))
        prim_ids[k]._pad0 = Int8(0); prim_ids[k]._pad1 = Int8(0)
        prim_ids[k]._pad2 = Int8(0); prim_ids[k]._pad3 = Int8(0)
        prim_ids[k]._pad4 = Int8(0); prim_ids[k]._pad5 = Int8(0)
        prim_ids[k]._pad6 = Int8(0)

    tri_mesh.free(); tri_local.free(); bvh_order.free(); mesh_al_idx.free()

    # ---- Sampler params ----
    var spp = s[0].samples_per_pixel
    var log2_spp = Int32(0)
    var tmp_spp = spp
    while tmp_spp > Int32(1):
        tmp_spp >>= 1
        log2_spp += 1
    var log4_spp = (log2_spp + Int32(1)) / Int32(2)
    var dim = max(s[0].film_w, s[0].film_h)
    var log2_dim = Int32(0)
    var tmp_dim = dim
    while tmp_dim > Int32(1):
        tmp_dim >>= 1
        log2_dim += 1
    var n_base4 = log2_dim + log4_spp

    # ---- Filter norms ----
    var norm_x = mojo_gaussian_norm(s[0].filter_support_x, s[0].filter_sigma)
    var norm_y = mojo_gaussian_norm(s[0].filter_support_y, s[0].filter_sigma)
    var fweight = (Float32(2) * norm_x - Float32(1)) * (Float32(2) * norm_y - Float32(1))

    # ---- RNG seed from time ----
    var rng_seed = UInt64(external_call["time", Int64, UnsafePointer[Int64, MutAnyOrigin]](
        UnsafePointer[Int64, MutAnyOrigin]()))

    # ---- Film filename copy ----
    var fname = alloc[UInt8](PSC_FILE_MAX)
    _psc_strncpy(fname, s[0].film_filename, PSC_FILE_MAX)

    # ---- Fill output struct ----
    psc[0].materials        = mats
    psc[0].material_count   = Int32(n_mats)
    psc[0].area_lights      = al_list
    psc[0].area_light_count = al_count
    psc[0].meshes           = meshes
    psc[0].mesh_pts         = out_pts
    psc[0].mesh_vis         = out_vis
    psc[0].mesh_fis         = out_fis
    psc[0].mesh_n_verts     = out_nv
    psc[0].mesh_n_tris      = out_nt
    psc[0].mesh_count       = Int32(n_meshes)
    psc[0].bvh_nodes        = bvh_nodes
    psc[0].prim_ids         = prim_ids
    psc[0].bvh_node_count   = node_count
    psc[0].prim_count       = total_tris
    psc[0].film_w           = s[0].film_w
    psc[0].film_h           = s[0].film_h
    psc[0].film_iso         = s[0].film_iso
    psc[0].film_max_comp    = s[0].film_max_comp
    psc[0].film_filename    = fname
    psc[0].filter_sigma     = s[0].filter_sigma
    psc[0].filter_support_x = s[0].filter_support_x
    psc[0].filter_support_y = s[0].filter_support_y
    psc[0].filter_norm_x    = norm_x
    psc[0].filter_norm_y    = norm_y
    psc[0].filter_weight    = fweight
    psc[0].samples_per_pixel = spp
    psc[0].log2_spp         = log2_spp
    psc[0].n_base4_digits   = n_base4
    psc[0].max_depth        = s[0].max_depth
    psc[0].rng_seed         = rng_seed

# ── Exported API ──────────────────────────────────────────────────────────────

fn mojo_parse_scene(path: UnsafePointer[UInt8, MutAnyOrigin]
                    ) -> UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]:
    var handle = mojo_scanner_new(path)
    if not handle:
        var psc = alloc[ParsedScene_Mojo](1)
        psc[0].mesh_count = Int32(0)
        psc[0].prim_count = Int32(0)
        psc[0].bvh_node_count = Int32(0)
        psc[0].material_count = Int32(0)
        psc[0].area_light_count = Int32(0)
        return psc

    var s = _psc_state_new()
    _psc_parse(handle, s)
    mojo_scanner_free(handle)

    var psc = alloc[ParsedScene_Mojo](1)
    _psc_finalize(s, psc)
    _psc_state_free(s)
    return psc

fn mojo_parsed_free(psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]):
    if not psc:
        return
    var n = Int(psc[0].mesh_count)
    for i in range(n):
        psc[0].mesh_pts[i].free()
        psc[0].mesh_vis[i].free()
        psc[0].mesh_fis[i].free()
    if psc[0].mesh_count > 0:
        psc[0].mesh_pts.free()
        psc[0].mesh_vis.free()
        psc[0].mesh_fis.free()
        psc[0].mesh_n_verts.free()
        psc[0].mesh_n_tris.free()
    if psc[0].meshes:
        psc[0].meshes.free()
    if psc[0].materials:
        psc[0].materials.free()
    if psc[0].area_lights:
        psc[0].area_lights.free()
    if psc[0].bvh_nodes:
        psc[0].bvh_nodes.free()
    if psc[0].prim_ids:
        psc[0].prim_ids.free()
    if psc[0].raster_to_camera:
        psc[0].raster_to_camera.free()
    if psc[0].camera_to_world:
        psc[0].camera_to_world.free()
    if psc[0].film_filename:
        psc[0].film_filename.free()
    psc.free()

fn mojo_parsed_scene_descriptor(
    psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]
) -> UnsafePointer[SceneDescriptor2_C, MutAnyOrigin]:
    var sd = alloc[SceneDescriptor2_C](1)
    sd[0].bvh2Nodes      = psc[0].bvh_nodes
    sd[0].primIds        = psc[0].prim_ids
    sd[0].meshes         = psc[0].meshes
    sd[0].meshCount      = Int64(psc[0].mesh_count)
    sd[0].materials      = psc[0].materials
    sd[0].materialCount  = Int64(psc[0].material_count)
    sd[0].areaLights     = psc[0].area_lights
    sd[0].areaLightCount = Int64(psc[0].area_light_count)
    return sd
