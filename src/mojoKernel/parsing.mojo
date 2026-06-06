from std.ffi import external_call
from .ply import mojo_load_ply
from std.math import tan, sqrt, acos, atan2, sin, abs, exp
from std.memory import alloc
from .geometry import RGB, SampledSpectrum, Point3f, Vec3f, Ray_C, Material_C, AreaLight_C, Sphere_C, DistantLight_C, PointLight_C, InfiniteLight_C, TriangleMesh_C, PrimId_C, PI, TWO_PI, Medium_C, MediumInterface_C
from .transform import mojo_matrix_multiply, mojo_matrix_invert, mojo_transform_points
from .bvh import BVH2Node, SceneDescriptor2_C, mojo_build_bvh2
from .sampling import mojo_gaussian_norm

# ── pbrt Scanner Helpers ──────────────────────────────────────────────

@always_inline
fn _is_ws(b: UInt8) -> Bool:
    return b == UInt8(32) or b == UInt8(9) or b == UInt8(10) or b == UInt8(13)

# Skip whitespace and pbrt line comments (# … \n).
@always_inline
fn _skip_ws_comments(bytes: UnsafePointer[UInt8, MutAnyOrigin], length: Int, pos: Int) -> Int:
    var cur = pos
    while cur < length:
        var b = bytes[cur]
        if _is_ws(b):
            cur += 1
        elif b == UInt8(35):  # '#'
            while cur < length and bytes[cur] != UInt8(10):
                cur += 1
        else:
            break
    return cur

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
    cur = _skip_ws_comments(bytes, len, cur)
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
    cur = _skip_ws_comments(bytes, len, cur)
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
    var handle = alloc[PbrtScanner_Mojo](1)
    var path_str = String(unsafe_from_utf8_ptr=path.as_immutable())
    try:
        var f = open(path_str, "r")
        var bytes = f.read_bytes()
        f.close()
        var size = len(bytes)
        var buf = alloc[UInt8](size + 1)
        for i in range(size):
            buf[i] = bytes[i]
        buf[size] = UInt8(0)
        handle[0].buffer = buf
        handle[0].total_bytes = Int32(size)
        handle[0].cursor = Int32(0)
        handle[0].is_at_end = Int32(0)
    except:
        handle[0].buffer = UnsafePointer[UInt8, MutAnyOrigin]()
        handle[0].total_bytes = Int32(0)
        handle[0].cursor = Int32(0)
        handle[0].is_at_end = Int32(1)
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

comptime PSC_MAX_MESHES = 1024
comptime PSC_MAX_NAMED  = 64
comptime PSC_CTM_DEPTH  = 16
comptime PSC_ATTR_DEPTH = 8
comptime PSC_NAME_MAX   = 64
comptime PSC_FILE_MAX   = 256
comptime PSC_MAX_TEX    = 64

# ── Hair curve tessellation constants ─────────────────────────────────────────
# B-spline curves (Shape "curve") are tessellated into cross-ribbon triangles.
# HAIR_EVAL_N points are sampled uniformly along each strand (B-spline),
# giving HAIR_EVAL_N-1 ribbon segments. Each segment becomes 2 perpendicular
# quads (4 triangles) forming a cross-shaped profile visible from any angle.
comptime HAIR_EVAL_N    = 8          # sample points along each strand
comptime HAIR_MAX_VTX   = 15_000_000 # lazy buffer: max accumulated hair vertices
comptime HAIR_MAX_TRI   = 10_000_000 # lazy buffer: max accumulated hair triangles

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
    var mesh_uv_n_verts:  UnsafePointer[Int32, MutAnyOrigin]  # per-mesh UV vertex count; 0 = no UVs
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
    var tex_filenames:    UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin]
    var tex_count:        Int32
    var distant_lights:   UnsafePointer[DistantLight_C, MutAnyOrigin]
    var distant_count:    Int32
    var point_lights:     UnsafePointer[PointLight_C, MutAnyOrigin]
    var point_count:      Int32
    var infinite_lights:  UnsafePointer[InfiniteLight_C, MutAnyOrigin]
    var infinite_count:   Int32
    var spheres:          UnsafePointer[Sphere_C, MutAnyOrigin]
    var sphere_count:     Int32
    var mediums:          UnsafePointer[Medium_C, MutAnyOrigin]
    var medium_count:     Int32
    var medium_ifaces:    UnsafePointer[MediumInterface_C, MutAnyOrigin]
    var medium_iface_count: Int32

# ── Internal parse state ──────────────────────────────────────────────────────

struct _PscState:
    var ctm:       UnsafePointer[Float32, MutAnyOrigin]
    var ctm_stack: UnsafePointer[Float32, MutAnyOrigin]
    var ctm_depth: Int32

    var attr_mat:    UnsafePointer[Int32, MutAnyOrigin]
    var attr_alight: UnsafePointer[Int32, MutAnyOrigin]
    var attr_al_rgb: UnsafePointer[RGB, MutAnyOrigin]
    var attr_depth:  Int32

    var named_names:  UnsafePointer[UInt8, MutAnyOrigin]
    var named_albedo: UnsafePointer[RGB, MutAnyOrigin]
    var named_type:   UnsafePointer[Int8, MutAnyOrigin]
    var named_ior:    UnsafePointer[Float32, MutAnyOrigin]
    var named_roughU: UnsafePointer[Float32, MutAnyOrigin]
    var named_roughV: UnsafePointer[Float32, MutAnyOrigin]
    var n_named:      Int32

    var cur_mat_idx: Int32
    var in_alight:   Int32
    var al: RGB

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
    var mesh_al_rgb:   UnsafePointer[RGB, MutAnyOrigin]
    var mesh_uvs_list: UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin]
    var mesh_has_uvs:  UnsafePointer[Int8, MutAnyOrigin]
    var scene_dir:     UnsafePointer[UInt8, MutAnyOrigin]

    var named_tex_idx: UnsafePointer[Int32, MutAnyOrigin]
    var named_normal_tex_idx: UnsafePointer[Int32, MutAnyOrigin]  # per-material normal map tex idx
    var named_mix1:    UnsafePointer[UInt8, MutAnyOrigin]   # mix mat name1 (PSC_MAX_NAMED * PSC_NAME_MAX)
    var named_mix2:    UnsafePointer[UInt8, MutAnyOrigin]   # mix mat name2
    var named_amount:  UnsafePointer[Float32, MutAnyOrigin] # mix blend amount
    var named_transmittance: UnsafePointer[RGB, MutAnyOrigin] # diffusetransmission transmittance
    # Hair curve accumulator (lazy-allocated on first Shape "curve")
    var hair_pts:    UnsafePointer[Float32, MutAnyOrigin]  # 3 floats per vertex (world-space)
    var hair_idx:    UnsafePointer[Int32, MutAnyOrigin]    # 3 ints per triangle
    var hair_nv:     Int32
    var hair_nt:     Int32
    var hair_mat:    Int32   # cur_mat_idx when first curve was added
    var hair_inited: Int32   # 0 = not yet allocated
    var tex_names:     UnsafePointer[UInt8, MutAnyOrigin]
    var tex_files:     UnsafePointer[UInt8, MutAnyOrigin]
    var n_textures:    Int32

    # Non-area lights accumulated during parse
    var dist_dirs:     UnsafePointer[Float32, MutAnyOrigin]   # n_distant * 3 floats
    var dist_rgb:      UnsafePointer[Float32, MutAnyOrigin]   # n_distant * 3 floats
    var n_distant:     Int32
    var pt_pos:        UnsafePointer[Float32, MutAnyOrigin]   # n_point * 3 floats
    var pt_rgb:        UnsafePointer[Float32, MutAnyOrigin]   # n_point * 3 floats
    var n_point:       Int32
    var inf_tex_idx:   UnsafePointer[Int32, MutAnyOrigin]     # n_infinite indices
    var inf_rgb:       UnsafePointer[Float32, MutAnyOrigin]   # n_infinite * 3 floats
    var n_infinite:    Int32

    # Analytical sphere primitives
    var sph_cx:   UnsafePointer[Float32, MutAnyOrigin]  # center x (n_spheres)
    var sph_cy:   UnsafePointer[Float32, MutAnyOrigin]  # center y
    var sph_cz:   UnsafePointer[Float32, MutAnyOrigin]  # center z
    var sph_r:    UnsafePointer[Float32, MutAnyOrigin]  # radius
    var sph_mat:  UnsafePointer[Int32, MutAnyOrigin]    # material index
    var sph_al:   UnsafePointer[Int8, MutAnyOrigin]     # isAreaLight flag
    var sph_rgb:  UnsafePointer[RGB, MutAnyOrigin]      # emission RGB
    var n_spheres: Int32

    # Homogeneous media (cap 32)
    var med_names:  UnsafePointer[UInt8, MutAnyOrigin]   # 32 * 64 bytes
    var med_sa:     UnsafePointer[Float32, MutAnyOrigin]  # n_mediums * 3
    var med_ss:     UnsafePointer[Float32, MutAnyOrigin]  # n_mediums * 3
    var med_g:      UnsafePointer[Float32, MutAnyOrigin]  # n_mediums
    var n_mediums:  Int32
    # Medium interfaces bound to materials (cap 256)
    var miface_inside:  UnsafePointer[Int32, MutAnyOrigin]  # n_ifaces
    var miface_outside: UnsafePointer[Int32, MutAnyOrigin]  # n_ifaces
    var miface_mat:     UnsafePointer[Int32, MutAnyOrigin]  # which mat_idx each iface belongs to
    var n_ifaces: Int32
    var cur_inside_medium: Int32   # attribute-state: current MediumInterface inside
    var cur_outside_medium: Int32  # attribute-state: current MediumInterface outside
    var mesh_inside_med:  UnsafePointer[Int32, MutAnyOrigin]  # per-mesh inside medium idx
    var mesh_outside_med: UnsafePointer[Int32, MutAnyOrigin]  # per-mesh outside medium idx
    var sph_inside_med:   UnsafePointer[Int32, MutAnyOrigin]  # per-sphere inside medium idx
    var sph_outside_med:  UnsafePointer[Int32, MutAnyOrigin]  # per-sphere outside medium idx
    var attr_inside_med:  UnsafePointer[Int32, MutAnyOrigin]  # attribute stack
    var attr_outside_med: UnsafePointer[Int32, MutAnyOrigin]  # attribute stack


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

fn _psc_strncmp(a: UnsafePointer[UInt8, MutAnyOrigin], b: StringLiteral, n: Int) -> Int:
    """Compare first n bytes of a against literal b. Returns 0 if equal."""
    for i in range(n):
        var ca = Int(a[i])
        var cb = Int(b.unsafe_ptr()[i])
        if ca != cb:
            return ca - cb
        if ca == 0:
            return 0
    return 0

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
    s[0].attr_al_rgb = alloc[RGB](PSC_ATTR_DEPTH)
    s[0].named_transmittance = alloc[RGB](PSC_MAX_NAMED)
    # Hair accumulator: lazily allocated on first Shape "curve"
    s[0].hair_inited = Int32(0)
    s[0].hair_nv = Int32(0)
    s[0].hair_nt = Int32(0)
    s[0].hair_mat = Int32(-1)
    s[0].hair_pts = UnsafePointer[Float32, MutAnyOrigin]()
    s[0].hair_idx = UnsafePointer[Int32, MutAnyOrigin]()
    s[0].attr_depth  = Int32(0)

    s[0].named_names  = alloc[UInt8](PSC_MAX_NAMED * PSC_NAME_MAX)
    s[0].named_albedo = alloc[RGB](PSC_MAX_NAMED)
    s[0].named_type   = alloc[Int8](PSC_MAX_NAMED)
    s[0].named_ior    = alloc[Float32](PSC_MAX_NAMED)
    s[0].named_roughU = alloc[Float32](PSC_MAX_NAMED)
    s[0].named_roughV = alloc[Float32](PSC_MAX_NAMED)
    s[0].n_named      = Int32(0)

    s[0].cur_mat_idx = Int32(-1)
    s[0].in_alight   = Int32(0)
    s[0].al = RGB(Float32(0), Float32(0), Float32(0))

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
    s[0].mesh_al_rgb   = alloc[RGB](PSC_MAX_MESHES)
    s[0].mesh_uvs_list = alloc[UnsafePointer[Float32, MutAnyOrigin]](PSC_MAX_MESHES)
    s[0].mesh_has_uvs  = alloc[Int8](PSC_MAX_MESHES)
    for i in range(PSC_MAX_MESHES):
        s[0].mesh_uvs_list[i] = UnsafePointer[Float32, MutAnyOrigin]()
        s[0].mesh_has_uvs[i]  = Int8(0)
    s[0].scene_dir     = alloc[UInt8](PSC_FILE_MAX * 2)
    s[0].scene_dir[0]  = UInt8(0)

    s[0].named_tex_idx = alloc[Int32](PSC_MAX_NAMED)
    for i in range(PSC_MAX_NAMED):
        s[0].named_tex_idx[i] = Int32(-1)
    s[0].named_normal_tex_idx = alloc[Int32](PSC_MAX_NAMED)
    for i in range(PSC_MAX_NAMED):
        s[0].named_normal_tex_idx[i] = Int32(-1)
    s[0].named_mix1   = alloc[UInt8](PSC_MAX_NAMED * PSC_NAME_MAX)
    s[0].named_mix2   = alloc[UInt8](PSC_MAX_NAMED * PSC_NAME_MAX)
    s[0].named_amount  = alloc[Float32](PSC_MAX_NAMED)
    for i in range(PSC_MAX_NAMED):
        s[0].named_mix1[i * PSC_NAME_MAX] = UInt8(0)
        s[0].named_mix2[i * PSC_NAME_MAX] = UInt8(0)
        s[0].named_amount[i] = Float32(0.5)
    s[0].tex_names  = alloc[UInt8](PSC_MAX_TEX * PSC_NAME_MAX)
    s[0].tex_files  = alloc[UInt8](PSC_MAX_TEX * PSC_FILE_MAX * 2)
    s[0].n_textures = Int32(0)

    comptime MAX_LIGHTS = 64
    s[0].dist_dirs   = alloc[Float32](MAX_LIGHTS * 3)
    s[0].dist_rgb    = alloc[Float32](MAX_LIGHTS * 3)
    s[0].n_distant   = Int32(0)
    s[0].pt_pos      = alloc[Float32](MAX_LIGHTS * 3)
    s[0].pt_rgb      = alloc[Float32](MAX_LIGHTS * 3)
    s[0].n_point     = Int32(0)
    s[0].inf_tex_idx = alloc[Int32](MAX_LIGHTS)
    s[0].inf_rgb     = alloc[Float32](MAX_LIGHTS * 3)
    s[0].n_infinite  = Int32(0)

    comptime MAX_SPHERES = 64
    s[0].sph_cx  = alloc[Float32](MAX_SPHERES)
    s[0].sph_cy  = alloc[Float32](MAX_SPHERES)
    s[0].sph_cz  = alloc[Float32](MAX_SPHERES)
    s[0].sph_r   = alloc[Float32](MAX_SPHERES)
    s[0].sph_mat = alloc[Int32](MAX_SPHERES)
    s[0].sph_al  = alloc[Int8](MAX_SPHERES)
    s[0].sph_rgb = alloc[RGB](MAX_SPHERES)
    s[0].n_spheres = Int32(0)

    # Mediums
    s[0].med_names = alloc[UInt8](32 * 64)
    s[0].med_sa    = alloc[Float32](32 * 3)
    s[0].med_ss    = alloc[Float32](32 * 3)
    s[0].med_g     = alloc[Float32](32)
    s[0].n_mediums = Int32(0)
    s[0].miface_inside  = alloc[Int32](256)
    s[0].miface_outside = alloc[Int32](256)
    s[0].miface_mat     = alloc[Int32](256)
    s[0].n_ifaces = Int32(0)
    s[0].cur_inside_medium = Int32(-1)
    s[0].cur_outside_medium = Int32(-1)
    s[0].mesh_inside_med  = alloc[Int32](PSC_MAX_MESHES)
    s[0].mesh_outside_med = alloc[Int32](PSC_MAX_MESHES)
    s[0].sph_inside_med   = alloc[Int32](256)
    s[0].sph_outside_med  = alloc[Int32](256)
    s[0].attr_inside_med  = alloc[Int32](PSC_ATTR_DEPTH)
    s[0].attr_outside_med = alloc[Int32](PSC_ATTR_DEPTH)

    return s

fn _psc_state_free(s: UnsafePointer[_PscState, MutAnyOrigin]):
    s[0].ctm.free()
    s[0].ctm_stack.free()
    s[0].attr_mat.free()
    s[0].attr_alight.free()
    s[0].attr_al_rgb.free()
    s[0].named_transmittance.free()
    if s[0].hair_inited != 0:
        s[0].hair_pts.free()
        s[0].hair_idx.free()
    s[0].named_names.free()
    s[0].named_albedo.free()
    s[0].named_type.free()
    s[0].named_ior.free()
    s[0].named_roughU.free()
    s[0].named_roughV.free()
    s[0].film_filename.free()
    s[0].cam2w_raw.free()
    s[0].mesh_pts_list.free()
    s[0].mesh_vis_list.free()
    s[0].mesh_fis_list.free()
    s[0].mesh_nv.free()
    s[0].mesh_nt.free()
    s[0].mesh_mat_idx.free()
    s[0].mesh_is_al.free()
    s[0].mesh_inside_med.free()
    s[0].mesh_outside_med.free()
    s[0].sph_inside_med.free()
    s[0].sph_outside_med.free()
    s[0].attr_inside_med.free()
    s[0].attr_outside_med.free()
    s[0].mesh_al_rgb.free()
    s[0].mesh_uvs_list.free()
    s[0].mesh_has_uvs.free()
    s[0].scene_dir.free()
    s[0].named_tex_idx.free()
    s[0].named_normal_tex_idx.free()
    s[0].named_mix1.free()
    s[0].named_mix2.free()
    s[0].named_amount.free()
    s[0].tex_names.free()
    s[0].tex_files.free()
    s[0].dist_dirs.free(); s[0].dist_rgb.free()
    s[0].pt_pos.free();    s[0].pt_rgb.free()
    s[0].inf_tex_idx.free(); s[0].inf_rgb.free()
    s[0].sph_cx.free(); s[0].sph_cy.free(); s[0].sph_cz.free()
    s[0].sph_r.free();  s[0].sph_mat.free()
    s[0].sph_al.free(); s[0].sph_rgb.free()
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
    # transmittance for DiffuseTransmission (default 0.25 per PBRT)
    var trans_rgb = alloc[Float32](3)
    trans_rgb[0] = Float32(0.25); trans_rgb[1] = Float32(0.25); trans_rgb[2] = Float32(0.25)
    # named-spectrum conductor optical constants (R/G/B at 630/530/450 nm)
    var metal_eta = alloc[Float32](3)
    metal_eta[0] = Float32(0.5); metal_eta[1] = Float32(0.5); metal_eta[2] = Float32(0.5)
    var metal_k   = alloc[Float32](3)
    metal_k[0] = Float32(0.5); metal_k[1] = Float32(0.5); metal_k[2] = Float32(0.5)
    var has_spectral_conductor = False
    # hair material melanin parameters
    var eumelanin   = Float32(0.0)
    var pheomelanin = Float32(0.0)
    var sigma_a_rgb = alloc[Float32](3)
    sigma_a_rgb[0] = Float32(-1); sigma_a_rgb[1] = Float32(-1); sigma_a_rgb[2] = Float32(-1)
    var has_sigma_a = False
    var mat_type = Int8(1)  # default: diffuse
    var mat_ior  = Float32(1.5)
    var mat_roughU = Float32(0.0)
    var mat_roughV = Float32(0.0)
    var tex_idx_for_mat = Int32(-1)
    var normal_tex_idx_for_mat = Int32(-1)
    var mix_name1 = alloc[UInt8](PSC_NAME_MAX)
    var mix_name2 = alloc[UInt8](PSC_NAME_MAX)
    var mix_amount = Float32(0.5)
    mix_name1[0] = UInt8(0); mix_name2[0] = UInt8(0)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var str_val  = alloc[UInt8](64)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "type") and _psc_type_is_str(type_buf):
            _ = mojo_scanner_parse_quoted_string(handle, str_val, 64)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
            if _psc_streq(str_val, "conductor"):
                mat_type = Int8(3)
            elif _psc_streq(str_val, "dielectric"):
                mat_type = Int8(4)
            elif _psc_streq(str_val, "coateddiffuse"):
                mat_type = Int8(5)
            elif _psc_streq(str_val, "diffusetransmission"):
                mat_type = Int8(6)
            elif _psc_streq(str_val, "coatedconductor"):
                mat_type = Int8(7)
            elif _psc_streq(str_val, "mix"):
                mat_type = Int8(8)
            elif _psc_streq(str_val, "thindielectric"):
                mat_type = Int8(9)
            elif _psc_streq(str_val, "hair"):
                mat_type = Int8(11)
            else:
                mat_type = Int8(1)
        elif (_psc_streq(name_buf, "eta") or _psc_streq(name_buf, "intIOR")) and _psc_type_is_float(type_buf):
            var tmp = alloc[Float32](1)
            _ = mojo_scanner_scan_float(handle, tmp)
            mat_ior = tmp[0]
            tmp.free()
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
        elif (_psc_streq(name_buf, "uroughness") or _psc_streq(name_buf, "roughness")) and _psc_type_is_float(type_buf):
            mat_roughU = _psc_scan_one_float(handle, is_array)
            if _psc_streq(name_buf, "roughness"):
                mat_roughV = mat_roughU
        elif _psc_streq(name_buf, "vroughness") and _psc_type_is_float(type_buf):
            mat_roughV = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "reflectance") and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, rgb, is_array)
        elif _psc_streq(name_buf, "transmittance") and _psc_type_is_float(type_buf):
            # DiffuseTransmission transmittance — stored in trans_rgb, later -> mat.emission
            _psc_scan_rgb(handle, trans_rgb, is_array)
        elif _psc_streq(name_buf, "eumelanin") and _psc_type_is_float(type_buf):
            # Hair material: melanin concentration -> stored in rgb[0] (temp)
            eumelanin = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "pheomelanin") and _psc_type_is_float(type_buf):
            pheomelanin = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "sigma_a") and _psc_type_is_float(type_buf):
            # Hair material: direct absorption coefficients (R,G,B)
            _psc_scan_rgb(handle, sigma_a_rgb, is_array)
            has_sigma_a = True
        elif (_psc_streq(name_buf, "eta") or _psc_streq(name_buf, "k")) and type_buf[0] == UInt8(115):  # 's' = spectrum
            # Named-spectrum conductor: "spectrum eta" ["metal-Ag-eta"] etc.
            # Read the metal name string, look up precomputed F0 per channel.
            var mname = alloc[UInt8](64)
            _ = mojo_scanner_parse_quoted_string(handle, mname, 64)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
            # Precomputed Fresnel F0 = ((eta-1)^2+k^2)/((eta+1)^2+k^2) for common metals
            # Channels: R≈630nm, G≈530nm, B≈450nm  (from NIST/Filament spectral data)
            if _psc_streq(name_buf, "eta"):
                if _psc_strncmp(mname, "metal-Ag", 8) == 0:
                    metal_eta[0] = Float32(0.136); metal_eta[1] = Float32(0.130); metal_eta[2] = Float32(0.144)
                elif _psc_strncmp(mname, "metal-Al", 8) == 0:
                    metal_eta[0] = Float32(1.300); metal_eta[1] = Float32(0.826); metal_eta[2] = Float32(0.644)
                elif _psc_strncmp(mname, "metal-Au", 8) == 0:
                    metal_eta[0] = Float32(0.194); metal_eta[1] = Float32(0.608); metal_eta[2] = Float32(1.426)
                elif _psc_strncmp(mname, "metal-Cu", 8) == 0:
                    metal_eta[0] = Float32(0.272); metal_eta[1] = Float32(1.120); metal_eta[2] = Float32(1.160)
                has_spectral_conductor = True
            else:  # "k"
                if _psc_strncmp(mname, "metal-Ag", 8) == 0:
                    metal_k[0] = Float32(3.880); metal_k[1] = Float32(3.070); metal_k[2] = Float32(2.560)
                elif _psc_strncmp(mname, "metal-Al", 8) == 0:
                    metal_k[0] = Float32(7.480); metal_k[1] = Float32(6.280); metal_k[2] = Float32(5.580)
                elif _psc_strncmp(mname, "metal-Au", 8) == 0:
                    metal_k[0] = Float32(3.060); metal_k[1] = Float32(2.120); metal_k[2] = Float32(1.846)
                elif _psc_strncmp(mname, "metal-Cu", 8) == 0:
                    metal_k[0] = Float32(3.240); metal_k[1] = Float32(2.605); metal_k[2] = Float32(2.433)
                has_spectral_conductor = True
            mname.free()
        elif _psc_streq(name_buf, "reflectance") and type_buf[0] == UInt8(116):  # 't' = texture
            _ = mojo_scanner_parse_quoted_string(handle, str_val, 64)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
            # Look up str_val in tex_names
            for ti in range(Int(s[0].n_textures)):
                if _psc_strcmp(s[0].tex_names + ti * PSC_NAME_MAX, str_val) == 0:
                    tex_idx_for_mat = Int32(ti)
                    break
        elif _psc_streq(name_buf, "L") and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, rgb, is_array)
        elif (_psc_streq(name_buf, "normalmap") or _psc_streq(name_buf, "bumpmap")) and type_buf[0] == UInt8(116):  # 't' = texture
            _ = mojo_scanner_parse_quoted_string(handle, str_val, 64)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
            for ti in range(Int(s[0].n_textures)):
                if _psc_strcmp(s[0].tex_names + ti * PSC_NAME_MAX, str_val) == 0:
                    normal_tex_idx_for_mat = Int32(ti)
                    break
        elif _psc_streq(name_buf, "amount") and _psc_type_is_float(type_buf):
            mix_amount = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "materials") and _psc_type_is_str(type_buf):
            # Two quoted material names for mix
            var tmp1 = alloc[UInt8](PSC_NAME_MAX)
            var tmp2 = alloc[UInt8](PSC_NAME_MAX)
            _ = mojo_scanner_parse_quoted_string(handle, tmp1, PSC_NAME_MAX)
            _ = mojo_scanner_parse_quoted_string(handle, tmp2, PSC_NAME_MAX)
            _psc_strncpy(mix_name1, tmp1, PSC_NAME_MAX)
            _psc_strncpy(mix_name2, tmp2, PSC_NAME_MAX)
            tmp1.free(); tmp2.free()
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
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
        # For named-spectrum conductors: compute Fresnel F0 per channel
        # F0 = ((eta-1)^2 + k^2) / ((eta+1)^2 + k^2)   (normal-incidence)
        if has_spectral_conductor and mat_type == Int8(3):
            var f0r = ((metal_eta[0]-Float32(1.0))*(metal_eta[0]-Float32(1.0)) + metal_k[0]*metal_k[0]) / \
                      ((metal_eta[0]+Float32(1.0))*(metal_eta[0]+Float32(1.0)) + metal_k[0]*metal_k[0])
            var f0g = ((metal_eta[1]-Float32(1.0))*(metal_eta[1]-Float32(1.0)) + metal_k[1]*metal_k[1]) / \
                      ((metal_eta[1]+Float32(1.0))*(metal_eta[1]+Float32(1.0)) + metal_k[1]*metal_k[1])
            var f0b = ((metal_eta[2]-Float32(1.0))*(metal_eta[2]-Float32(1.0)) + metal_k[2]*metal_k[2]) / \
                      ((metal_eta[2]+Float32(1.0))*(metal_eta[2]+Float32(1.0)) + metal_k[2]*metal_k[2])
            s[0].named_albedo[idx] = RGB(f0r, f0g, f0b)
        elif mat_type == Int8(11):
            # Physically-based melanin -> RGB albedo.
            # PBRT-v4 sigma_a coefficients at R@630nm, G@530nm, B@450nm:
            #   eumelanin:   [0.419, 0.697, 1.37]
            #   pheomelanin: [0.187, 0.400, 1.05]
            # Color = exp(-sigma_a * scale) where scale = 2.0 is a good typical
            # effective path length through a hair cross-section ribbon.
            var ce = eumelanin; var cp = pheomelanin
            var scale = Float32(2.0)
            var al_r = exp(-(ce * Float32(0.419) + cp * Float32(0.187)) * scale)
            var al_g = exp(-(ce * Float32(0.697) + cp * Float32(0.400)) * scale)
            var al_b = exp(-(ce * Float32(1.370) + cp * Float32(1.050)) * scale)
            s[0].named_albedo[idx] = RGB(al_r, al_g, al_b)
        else:
            s[0].named_albedo[idx] = RGB(rgb[0], rgb[1], rgb[2])
        s[0].named_transmittance[idx] = RGB(trans_rgb[0], trans_rgb[1], trans_rgb[2])
        s[0].named_type[idx] = mat_type
        s[0].named_ior[idx]  = mat_ior
        s[0].named_roughU[idx] = mat_roughU
        s[0].named_roughV[idx] = mat_roughV
        s[0].named_tex_idx[idx] = tex_idx_for_mat
        s[0].named_normal_tex_idx[idx] = normal_tex_idx_for_mat
        _psc_strncpy(s[0].named_mix1 + idx * PSC_NAME_MAX, mix_name1, PSC_NAME_MAX)
        _psc_strncpy(s[0].named_mix2 + idx * PSC_NAME_MAX, mix_name2, PSC_NAME_MAX)
        s[0].named_amount[idx] = mix_amount
        s[0].n_named += 1

    mat_name.free(); type_buf.free(); name_buf.free(); str_val.free(); rgb.free()
    trans_rgb.free(); metal_eta.free(); metal_k.free()
    mix_name1.free(); mix_name2.free()

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
        s[0].attr_mat[d]    = s[0].cur_mat_idx
        s[0].attr_alight[d] = s[0].in_alight
        s[0].attr_al_rgb[d] = s[0].al
        s[0].attr_inside_med[d]  = s[0].cur_inside_medium
        s[0].attr_outside_med[d] = s[0].cur_outside_medium
        s[0].attr_depth += 1

fn _psc_handle_attribute_end(s: UnsafePointer[_PscState, MutAnyOrigin]):
    _psc_ctm_pop(s)
    if s[0].attr_depth > 0:
        s[0].attr_depth -= 1
        var d = Int(s[0].attr_depth)
        s[0].cur_mat_idx = s[0].attr_mat[d]
        s[0].in_alight   = s[0].attr_alight[d]
        s[0].al = s[0].attr_al_rgb[d]
        s[0].in_alight = Int32(0)
        s[0].cur_inside_medium  = s[0].attr_inside_med[d]
        s[0].cur_outside_medium = s[0].attr_outside_med[d]

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
    s[0].al = RGB(rgb[0], rgb[1], rgb[2])
    sbuf.free(); type_buf.free(); name_buf.free(); rgb.free()

fn _psc_mesh_store(
    s:       UnsafePointer[_PscState, MutAnyOrigin],
    tmp_f:   UnsafePointer[Float32, MutAnyOrigin],
    tmp_i:   UnsafePointer[Int32, MutAnyOrigin],
    n_verts: Int32,
    n_tris:  Int32,
):
    var n_meshes = Int(s[0].n_meshes)
    var raw_pts = alloc[Float32](Int(n_verts) * 4)
    for v in range(Int(n_verts)):
        raw_pts[v*4+0] = tmp_f[v*3+0]
        raw_pts[v*4+1] = tmp_f[v*3+1]
        raw_pts[v*4+2] = tmp_f[v*3+2]
        raw_pts[v*4+3] = Float32(1)
    var fin_pts = alloc[Float32](Int(n_verts) * 4)
    mojo_transform_points(s[0].ctm, raw_pts, n_verts, fin_pts)
    raw_pts.free()
    var vis = alloc[Int64](Int(n_tris) * 3)
    var fis = alloc[Int64](Int(n_tris))
    for t in range(Int(n_tris)):
        vis[t*3+0] = Int64(tmp_i[t*3+0])
        vis[t*3+1] = Int64(tmp_i[t*3+1])
        vis[t*3+2] = Int64(tmp_i[t*3+2])
        fis[t] = Int64(3)
    s[0].mesh_pts_list[n_meshes] = fin_pts
    s[0].mesh_vis_list[n_meshes] = vis
    s[0].mesh_fis_list[n_meshes] = fis
    s[0].mesh_nv[n_meshes] = n_verts
    s[0].mesh_nt[n_meshes] = n_tris
    s[0].mesh_mat_idx[n_meshes] = s[0].cur_mat_idx
    s[0].mesh_inside_med[n_meshes]  = s[0].cur_inside_medium
    s[0].mesh_outside_med[n_meshes] = s[0].cur_outside_medium
    s[0].mesh_is_al[n_meshes]   = s[0].in_alight
    s[0].mesh_al_rgb[n_meshes]  = s[0].al
    s[0].n_meshes += 1

# ── Hair curve helpers ────────────────────────────────────────────────────────

fn _psc_ensure_hair_buf(s: UnsafePointer[_PscState, MutAnyOrigin]) -> Bool:
    """Lazily allocate the hair accumulation buffers. Returns False if at mesh limit."""
    if s[0].hair_inited != 0:
        return True
    if s[0].n_meshes >= Int32(PSC_MAX_MESHES):
        return False
    s[0].hair_pts = alloc[Float32](HAIR_MAX_VTX * 3)
    s[0].hair_idx = alloc[Int32](HAIR_MAX_TRI * 3)
    s[0].hair_nv  = Int32(0)
    s[0].hair_nt  = Int32(0)
    s[0].hair_mat = s[0].cur_mat_idx
    s[0].hair_inited = Int32(1)
    return True

fn _psc_bspline3(cp: UnsafePointer[Float32, MutAnyOrigin], n_cp: Int,
                  u: Float32, pt_out: UnsafePointer[Float32, MutAnyOrigin]):
    """Evaluate uniform cubic B-spline at global parameter u in [0,1].
    cp: interleaved xyz control points, n_cp: number of 3D control points."""
    var n_seg = n_cp - 3
    if n_seg <= 0:
        pt_out[0] = cp[0]; pt_out[1] = cp[1]; pt_out[2] = cp[2]; return
    var s_f = u * Float32(n_seg)
    var seg = Int(s_f)
    if seg >= n_seg: seg = n_seg - 1
    var t = s_f - Float32(seg)
    var t2 = t * t
    var t3 = t2 * t
    # Uniform cubic B-spline basis at local t:
    var b0 = (Float32(1) - Float32(3)*t + Float32(3)*t2 - t3) / Float32(6)
    var b1 = (Float32(4) - Float32(6)*t2 + Float32(3)*t3) / Float32(6)
    var b2 = (Float32(1) + Float32(3)*t + Float32(3)*t2 - Float32(3)*t3) / Float32(6)
    var b3 = t3 / Float32(6)
    for k in range(3):
        pt_out[k] = b0*cp[seg*3+k] + b1*cp[(seg+1)*3+k] + b2*cp[(seg+2)*3+k] + b3*cp[(seg+3)*3+k]

fn _psc_hair_add_quad(s: UnsafePointer[_PscState, MutAnyOrigin],
        p0x: Float32, p0y: Float32, p0z: Float32,
        p1x: Float32, p1y: Float32, p1z: Float32,
        ux: Float32, uy: Float32, uz: Float32, hw: Float32):
    """Append one rectangular quad (2 triangles) to the hair accumulator."""
    if s[0].hair_nv + Int32(4) > Int32(HAIR_MAX_VTX): return
    if s[0].hair_nt + Int32(2) > Int32(HAIR_MAX_TRI): return
    var bv = Int(s[0].hair_nv)
    var bt = Int(s[0].hair_nt)
    s[0].hair_pts[bv*3+0] = p0x+ux*hw; s[0].hair_pts[bv*3+1] = p0y+uy*hw; s[0].hair_pts[bv*3+2] = p0z+uz*hw
    s[0].hair_pts[(bv+1)*3+0] = p0x-ux*hw; s[0].hair_pts[(bv+1)*3+1] = p0y-uy*hw; s[0].hair_pts[(bv+1)*3+2] = p0z-uz*hw
    s[0].hair_pts[(bv+2)*3+0] = p1x+ux*hw; s[0].hair_pts[(bv+2)*3+1] = p1y+uy*hw; s[0].hair_pts[(bv+2)*3+2] = p1z+uz*hw
    s[0].hair_pts[(bv+3)*3+0] = p1x-ux*hw; s[0].hair_pts[(bv+3)*3+1] = p1y-uy*hw; s[0].hair_pts[(bv+3)*3+2] = p1z-uz*hw
    s[0].hair_idx[bt*3+0] = Int32(bv);   s[0].hair_idx[bt*3+1] = Int32(bv+1); s[0].hair_idx[bt*3+2] = Int32(bv+2)
    s[0].hair_idx[(bt+1)*3+0] = Int32(bv+1); s[0].hair_idx[(bt+1)*3+1] = Int32(bv+3); s[0].hair_idx[(bt+1)*3+2] = Int32(bv+2)
    s[0].hair_nv += Int32(4)
    s[0].hair_nt += Int32(2)

fn _psc_handle_curve_shape(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                            s: UnsafePointer[_PscState, MutAnyOrigin]):
    """Parse Shape "curve" B-spline strand and append cross-ribbon triangles."""
    var MAX_CP = 512
    var cp_buf = alloc[Float32](MAX_CP * 3)
    var n_cp = Int32(0)
    var width = Float32(0.002)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_type_is_float(type_buf) and _psc_streq(name_buf, "P"):
            var n_f = mojo_scanner_scan_floats(handle, cp_buf, MAX_CP * 3)
            n_cp = n_f / Int32(3)
            if is_array: _ = mojo_scanner_scan_char(handle, UInt8(93))
        elif _psc_type_is_float(type_buf) and (_psc_streq(name_buf, "width") or _psc_streq(name_buf, "width0")):
            width = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array: _ = mojo_scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    type_buf.free(); name_buf.free(); ia.free()
    if n_cp < Int32(4):
        cp_buf.free(); return
    if not _psc_ensure_hair_buf(s):
        cp_buf.free(); return
    # Evaluate HAIR_EVAL_N uniformly-spaced points along the B-spline
    var eval_pts = alloc[Float32](HAIR_EVAL_N * 3)
    for step in range(HAIR_EVAL_N):
        var u = Float32(step) / Float32(HAIR_EVAL_N - 1)
        _psc_bspline3(cp_buf, Int(n_cp), u, eval_pts + step * 3)
    cp_buf.free()
    # Apply current CTM to evaluation points
    var raw4 = alloc[Float32](HAIR_EVAL_N * 4)
    var xfm4 = alloc[Float32](HAIR_EVAL_N * 4)
    for step in range(HAIR_EVAL_N):
        raw4[step*4+0] = eval_pts[step*3+0]
        raw4[step*4+1] = eval_pts[step*3+1]
        raw4[step*4+2] = eval_pts[step*3+2]
        raw4[step*4+3] = Float32(1)
    mojo_transform_points(s[0].ctm, raw4, Int32(HAIR_EVAL_N), xfm4)
    raw4.free()
    # Tessellate segments into cross-ribbon geometry
    var hw = width / Float32(2)
    for seg in range(HAIR_EVAL_N - 1):
        var p0x = xfm4[seg*4+0]; var p0y = xfm4[seg*4+1]; var p0z = xfm4[seg*4+2]
        var p1x = xfm4[(seg+1)*4+0]; var p1y = xfm4[(seg+1)*4+1]; var p1z = xfm4[(seg+1)*4+2]
        var tx = p1x-p0x; var ty = p1y-p0y; var tz = p1z-p0z
        var tlen = sqrt(tx*tx + ty*ty + tz*tz)
        if tlen < Float32(1e-10): continue
        tx /= tlen; ty /= tlen; tz /= tlen
        # First perpendicular: T x Y (or T x X if near-parallel)
        var ux: Float32; var uy: Float32; var uz: Float32
        if abs(ty) < Float32(0.9):
            ux = -tz; uy = Float32(0); uz = tx   # T x Y = (-Tz, 0, Tx)
        else:
            ux = Float32(0); uy = tz; uz = -ty    # T x X = (0, Tz, -Ty)
        var ulen = sqrt(ux*ux + uy*uy + uz*uz)
        ux /= ulen; uy /= ulen; uz /= ulen
        # Second perpendicular: V = T x U
        var vx = ty*uz - tz*uy; var vy = tz*ux - tx*uz; var vz = tx*uy - ty*ux
        _psc_hair_add_quad(s, p0x,p0y,p0z, p1x,p1y,p1z, ux,uy,uz, hw)
        _psc_hair_add_quad(s, p0x,p0y,p0z, p1x,p1y,p1z, vx,vy,vz, hw)
    xfm4.free(); eval_pts.free()

fn _psc_flush_hair(s: UnsafePointer[_PscState, MutAnyOrigin]):
    """Flush all accumulated hair strands into a single scene mesh."""
    if s[0].hair_inited == 0 or s[0].hair_nt == 0:
        return
    var saved_mat = s[0].cur_mat_idx
    s[0].cur_mat_idx = s[0].hair_mat
    _psc_mesh_store(s, s[0].hair_pts, s[0].hair_idx, s[0].hair_nv, s[0].hair_nt)
    s[0].cur_mat_idx = saved_mat
    # Mark as flushed — but DON'T free: buffers will be freed by _psc_state_free
    # (hair_inited stays 1 so the free path in _psc_state_free still runs)
    s[0].hair_nv = Int32(0)
    s[0].hair_nt = Int32(0)


fn _psc_handle_make_named_medium(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                                  s: UnsafePointer[_PscState, MutAnyOrigin]):
    """Parse MakeNamedMedium \"name\" [params].
    Supports type = \"homogeneous\" with rgb sigma_a, rgb sigma_s, float g, float scale.
    """
    var name_buf = alloc[UInt8](64)
    _ = mojo_scanner_parse_quoted_string(handle, name_buf, 64)
    var type_buf  = alloc[UInt8](64)
    var val_buf   = alloc[UInt8](64)
    var sa = alloc[Float32](3); sa[0] = Float32(0); sa[1] = Float32(0); sa[2] = Float32(0)
    var ss = alloc[Float32](3); ss[0] = Float32(0); ss[1] = Float32(0); ss[2] = Float32(0)
    var g_val = Float32(0)
    var scale = Float32(1)
    var is_hom = False
    var ia = alloc[Int32](1)
    while True:
        ia[0] = Int32(0)
        var ok = mojo_scanner_parse_param_header(handle, type_buf, Int32(64), val_buf, Int32(64), ia)
        if ok == Int32(0):
            break
        var is_arr = ia[0]
        if _psc_streq(val_buf, "type"):
            var tmp = alloc[UInt8](64)
            _ = mojo_scanner_parse_quoted_string(handle, tmp, 64)
            if _psc_streq(tmp, "homogeneous"):
                is_hom = True
            tmp.free()
        elif _psc_streq(val_buf, "sigma_a") and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, sa, is_arr)
        elif _psc_streq(val_buf, "sigma_s") and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, ss, is_arr)
        elif _psc_streq(val_buf, "g") and _psc_type_is_float(type_buf):
            g_val = _psc_scan_one_float(handle, is_arr)
        elif _psc_streq(val_buf, "scale") and _psc_type_is_float(type_buf):
            scale = _psc_scan_one_float(handle, is_arr)
        else:
            _psc_skip_value(handle, type_buf, is_arr)
            if is_arr:
                _ = mojo_scanner_scan_char(handle, UInt8(93))  # ']' 
    if is_hom:
        var n = Int(s[0].n_mediums)
        if n < 32:
            var dst = s[0].med_names + n * 64
            var ni = 0
            while name_buf[ni] != UInt8(0) and ni < 63:
                dst[ni] = name_buf[ni]; ni += 1
            dst[ni] = UInt8(0)
            s[0].med_sa[n*3]   = sa[0] * scale
            s[0].med_sa[n*3+1] = sa[1] * scale
            s[0].med_sa[n*3+2] = sa[2] * scale
            s[0].med_ss[n*3]   = ss[0] * scale
            s[0].med_ss[n*3+1] = ss[1] * scale
            s[0].med_ss[n*3+2] = ss[2] * scale
            s[0].med_g[n] = g_val
            s[0].n_mediums = Int32(n + 1)
    name_buf.free(); sa.free(); ss.free(); type_buf.free(); val_buf.free()


fn _psc_lookup_medium(s: UnsafePointer[_PscState, MutAnyOrigin],
                       name: UnsafePointer[UInt8, MutAnyOrigin]) -> Int32:
    """Return medium index for name, or -1 for empty string / not found."""
    if name[0] == UInt8(0):
        return Int32(-1)
    for i in range(Int(s[0].n_mediums)):
        var entry = s[0].med_names + i * 64
        var j = 0
        var match_ok = True
        while name[j] != UInt8(0) or entry[j] != UInt8(0):
            if name[j] != entry[j]:
                match_ok = False
                break
            j += 1
        if match_ok:
            return Int32(i)
    return Int32(-1)


fn _psc_handle_medium_interface(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                                 s: UnsafePointer[_PscState, MutAnyOrigin]):
    """Parse MediumInterface \"inside_name\" \"outside_name\" and bind to cur_mat_idx."""
    var inside_buf  = alloc[UInt8](64)
    var outside_buf = alloc[UInt8](64)
    _ = mojo_scanner_parse_quoted_string(handle, inside_buf, 64)
    _ = mojo_scanner_parse_quoted_string(handle, outside_buf, 64)
    # Set attribute-state medium interface (applied to shapes emitted in this scope)
    s[0].cur_inside_medium  = _psc_lookup_medium(s, inside_buf)
    s[0].cur_outside_medium = _psc_lookup_medium(s, outside_buf)
    inside_buf.free(); outside_buf.free()

fn _psc_handle_sphere_shape(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                             s: UnsafePointer[_PscState, MutAnyOrigin]):
    """Parse Shape "sphere" and store as analytical Sphere_C.
    The sphere center is the CTM translation column; radius is the parsed float.
    """
    var radius = Float32(1.0)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_type_is_float(type_buf) and _psc_streq(name_buf, "radius"):
            radius = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    type_buf.free(); name_buf.free()

    var n = Int(s[0].n_spheres)
    # World-space center = CTM * (0,0,0,1) = translation column (indices 12,13,14)
    var ctm = s[0].ctm
    var cx = ctm[12]
    var cy = ctm[13]
    var cz = ctm[14]
    # Scale radius by the uniform scale of the CTM (length of first column vector)
    var sx = sqrt(ctm[0]*ctm[0] + ctm[1]*ctm[1] + ctm[2]*ctm[2])
    if sx < Float32(1e-6): sx = Float32(1.0)
    radius *= sx
    s[0].sph_cx[n]  = cx
    s[0].sph_cy[n]  = cy
    s[0].sph_cz[n]  = cz
    s[0].sph_r[n]   = radius
    s[0].sph_mat[n] = s[0].cur_mat_idx
    s[0].sph_inside_med[n]  = s[0].cur_inside_medium
    s[0].sph_outside_med[n] = s[0].cur_outside_medium
    s[0].sph_al[n]  = Int8(1) if s[0].in_alight != 0 else Int8(0)
    s[0].sph_rgb[n] = s[0].al
    s[0].n_spheres  = Int32(n + 1)

fn _psc_handle_shape(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                     s: UnsafePointer[_PscState, MutAnyOrigin]):
    var shape_type = alloc[UInt8](64)
    _ = mojo_scanner_parse_quoted_string(handle, shape_type, 64)

    var is_tri = _psc_streq(shape_type, "trianglemesh")
    var is_ply = _psc_streq(shape_type, "plymesh")
    var is_curve = _psc_streq(shape_type, "curve")
    var is_sphere = _psc_streq(shape_type, "sphere")
    shape_type.free()

    if is_curve:
        _psc_handle_curve_shape(handle, s)
        return

    if is_sphere:
        _psc_handle_sphere_shape(handle, s)
        return

    if not is_tri and not is_ply:
        _psc_skip_params(handle)
        return

    var n_meshes = Int(s[0].n_meshes)
    if n_meshes >= PSC_MAX_MESHES:
        _psc_skip_params(handle)
        return

    if is_ply:
        var ply_filename = alloc[UInt8](PSC_FILE_MAX)
        ply_filename[0] = UInt8(0)
        var type_buf = alloc[UInt8](64)
        var name_buf = alloc[UInt8](128)
        var ia = alloc[Int32](1)
        ia[0] = Int32(0)
        var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
        while found != 0:
            var is_array = ia[0]
            if _psc_streq(name_buf, "filename") and _psc_type_is_str(type_buf):
                _ = mojo_scanner_parse_quoted_string(handle, ply_filename, PSC_FILE_MAX)
                if is_array:
                    _ = mojo_scanner_scan_char(handle, UInt8(93))
            else:
                _psc_skip_value(handle, type_buf, is_array)
                if is_array:
                    _ = mojo_scanner_scan_char(handle, UInt8(93))
            ia[0] = Int32(0)
            found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
        ia.free()
        type_buf.free(); name_buf.free()

        var full_path = alloc[UInt8](PSC_FILE_MAX * 2)
        var dir_len = 0
        while s[0].scene_dir[dir_len] != UInt8(0):
            full_path[dir_len] = s[0].scene_dir[dir_len]
            dir_len += 1
        var fn_i = 0
        while ply_filename[fn_i] != UInt8(0) and dir_len + fn_i < PSC_FILE_MAX * 2 - 1:
            full_path[dir_len + fn_i] = ply_filename[fn_i]
            fn_i += 1
        full_path[dir_len + fn_i] = UInt8(0)
        ply_filename.free()

        var ply_pts     = alloc[UnsafePointer[Float32, MutAnyOrigin]](1)
        var ply_nv      = alloc[Int32](1)
        var ply_idx     = alloc[UnsafePointer[Int32, MutAnyOrigin]](1)
        var ply_nt      = alloc[Int32](1)
        var ply_uvs     = alloc[UnsafePointer[Float32, MutAnyOrigin]](1)
        var ply_has_uvs = alloc[Int32](1)
        ply_uvs[0] = UnsafePointer[Float32, MutAnyOrigin]()
        ply_has_uvs[0] = Int32(0)
        var ok = mojo_load_ply(full_path, ply_pts, ply_nv, ply_idx, ply_nt, ply_uvs, ply_has_uvs)
        full_path.free()
        if ok == 0:
            ply_pts.free(); ply_nv.free(); ply_idx.free(); ply_nt.free()
            ply_uvs.free(); ply_has_uvs.free()
            return
        var nv = ply_nv[0]
        var nt = ply_nt[0]
        if nv <= 0 or nt <= 0:
            ply_pts[0].free(); ply_idx[0].free()
            if ply_has_uvs[0] != 0:
                ply_uvs[0].free()
            ply_pts.free(); ply_nv.free(); ply_idx.free(); ply_nt.free()
            ply_uvs.free(); ply_has_uvs.free()
            return
        var tmp_f2 = ply_pts[0]
        var tmp_i2 = ply_idx[0]
        _psc_mesh_store(s, tmp_f2, tmp_i2, nv, nt)
        # Store UVs for this mesh
        var cur_mesh_idx = Int(s[0].n_meshes) - 1
        if ply_has_uvs[0] != 0:
            s[0].mesh_uvs_list[cur_mesh_idx] = ply_uvs[0]
            s[0].mesh_has_uvs[cur_mesh_idx]  = Int8(1)
        else:
            s[0].mesh_uvs_list[cur_mesh_idx] = UnsafePointer[Float32, MutAnyOrigin]()
        tmp_f2.free(); tmp_i2.free()
        ply_pts.free(); ply_nv.free(); ply_idx.free(); ply_nt.free()
        ply_uvs.free(); ply_has_uvs.free()
        return

    var tmp_f = alloc[Float32](65536)
    var tmp_i = alloc[Int32](16384)
    var tmp_uv = alloc[Float32](65536)
    var n_pts  = Int32(0)
    var n_idx  = Int32(0)
    var n_uv   = Int32(0)

    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        var is_P  = (_psc_streq(name_buf, "P") and _psc_type_is_float(type_buf))
        var is_I  = (_psc_streq(name_buf, "indices") and _psc_type_is_int(type_buf))
        var is_UV = ((_psc_streq(name_buf, "uv") or _psc_streq(name_buf, "st")) and _psc_type_is_float(type_buf))

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
        elif is_UV:
            if is_array:
                n_uv = mojo_scanner_scan_floats(handle, tmp_uv, 65536)
                _ = mojo_scanner_scan_char(handle, UInt8(93))  # ']'
            else:
                _ = mojo_scanner_scan_float(handle, tmp_uv)
                n_uv = Int32(2)
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
        tmp_f.free(); tmp_i.free(); tmp_uv.free()
        return

    _psc_mesh_store(s, tmp_f, tmp_i, n_verts, n_tris)
    # Store UVs for inline trianglemesh
    var cur_mesh_idx = Int(s[0].n_meshes) - 1
    if n_uv >= n_verts * Int32(2):
        var uv_copy = alloc[Float32](Int(n_verts) * 2)
        for ui in range(Int(n_verts) * 2):
            uv_copy[ui] = tmp_uv[ui]
        s[0].mesh_uvs_list[cur_mesh_idx] = uv_copy
        s[0].mesh_has_uvs[cur_mesh_idx]  = Int8(1)
    else:
        s[0].mesh_uvs_list[cur_mesh_idx] = UnsafePointer[Float32, MutAnyOrigin]()
    tmp_f.free(); tmp_i.free(); tmp_uv.free()

fn _psc_handle_texture(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                       s: UnsafePointer[_PscState, MutAnyOrigin]):
    # Texture <name> <type> <class> [params]
    var tex_name = alloc[UInt8](PSC_NAME_MAX)
    _ = mojo_scanner_parse_quoted_string(handle, tex_name, PSC_NAME_MAX)
    var tex_type = alloc[UInt8](64)
    _ = mojo_scanner_parse_quoted_string(handle, tex_type, 64)
    var tex_class = alloc[UInt8](64)
    _ = mojo_scanner_parse_quoted_string(handle, tex_class, 64)
    if not _psc_streq(tex_class, "imagemap"):
        tex_name.free(); tex_type.free(); tex_class.free()
        _psc_skip_params(handle)
        return

    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var str_val  = alloc[UInt8](PSC_FILE_MAX * 2)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "filename") and _psc_type_is_str(type_buf):
            _ = mojo_scanner_parse_quoted_string(handle, str_val, PSC_FILE_MAX * 2)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
            var idx = Int(s[0].n_textures)
            if idx < PSC_MAX_TEX:
                # Store name
                _psc_strncpy(s[0].tex_names + idx * PSC_NAME_MAX, tex_name, Int32(PSC_NAME_MAX))
                # Build full path: scene_dir + filename
                var full = s[0].tex_files + idx * PSC_FILE_MAX * 2
                var dir_len = 0
                while s[0].scene_dir[dir_len] != UInt8(0):
                    full[dir_len] = s[0].scene_dir[dir_len]
                    dir_len += 1
                var fn_i = 0
                while str_val[fn_i] != UInt8(0) and dir_len + fn_i < PSC_FILE_MAX * 2 - 1:
                    full[dir_len + fn_i] = str_val[fn_i]
                    fn_i += 1
                full[dir_len + fn_i] = UInt8(0)
                s[0].n_textures += 1
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    tex_name.free(); tex_type.free(); tex_class.free()
    type_buf.free(); name_buf.free(); str_val.free()

fn _psc_handle_light_source(handle: UnsafePointer[PbrtScanner_Mojo, MutAnyOrigin],
                             s: UnsafePointer[_PscState, MutAnyOrigin]):
    var ltype = alloc[UInt8](64)
    _ = mojo_scanner_parse_quoted_string(handle, ltype, 64)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var str_val  = alloc[UInt8](PSC_FILE_MAX * 2)
    var rgb      = alloc[Float32](3)
    var xyz      = alloc[Float32](3)
    rgb[0] = Float32(1); rgb[1] = Float32(1); rgb[2] = Float32(1)
    xyz[0] = Float32(0); xyz[1] = Float32(0); xyz[2] = Float32(1000)  # default: from above
    var scale    = Float32(1.0)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if (_psc_streq(name_buf, "L") or _psc_streq(name_buf, "I")) and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, rgb, is_array)
        elif _psc_streq(name_buf, "scale") and _psc_type_is_float(type_buf):
            scale = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "from") and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, xyz, is_array)   # reuse xyz for point-light position
        elif _psc_streq(name_buf, "filename") and _psc_type_is_str(type_buf):
            _ = mojo_scanner_parse_quoted_string(handle, str_val, PSC_FILE_MAX * 2)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = mojo_scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = mojo_scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()

    comptime MAX_LIGHTS = 64
    if _psc_streq(ltype, "distant"):
        var idx = Int(s[0].n_distant)
        if idx < MAX_LIGHTS:
            # direction = -from (from describes where light comes from)
            var len = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2])
            if len < Float32(0.0001): len = Float32(1.0)
            s[0].dist_dirs[idx*3+0] = -xyz[0] / len
            s[0].dist_dirs[idx*3+1] = -xyz[1] / len
            s[0].dist_dirs[idx*3+2] = -xyz[2] / len
            s[0].dist_rgb[idx*3+0] = rgb[0] * scale
            s[0].dist_rgb[idx*3+1] = rgb[1] * scale
            s[0].dist_rgb[idx*3+2] = rgb[2] * scale
            s[0].n_distant += 1
    elif _psc_streq(ltype, "point"):
        var idx = Int(s[0].n_point)
        if idx < MAX_LIGHTS:
            # Apply current CTM to position
            var raw = alloc[Float32](4)
            raw[0] = xyz[0]; raw[1] = xyz[1]; raw[2] = xyz[2]; raw[3] = Float32(1)
            var fin = alloc[Float32](4)
            mojo_transform_points(s[0].ctm, raw, Int32(1), fin)
            s[0].pt_pos[idx*3+0] = fin[0]
            s[0].pt_pos[idx*3+1] = fin[1]
            s[0].pt_pos[idx*3+2] = fin[2]
            s[0].pt_rgb[idx*3+0] = rgb[0] * scale
            s[0].pt_rgb[idx*3+1] = rgb[1] * scale
            s[0].pt_rgb[idx*3+2] = rgb[2] * scale
            s[0].n_point += 1
            raw.free(); fin.free()
    elif _psc_streq(ltype, "infinite"):
        var idx = Int(s[0].n_infinite)
        if idx < MAX_LIGHTS:
            # Try to find texture by filename
            var tex_i = Int32(-1)
            if str_val[0] != UInt8(0):
                # Build full path and register as a texture
                var full = alloc[UInt8](PSC_FILE_MAX * 2)
                var dir_len = 0
                while s[0].scene_dir[dir_len] != UInt8(0):
                    full[dir_len] = s[0].scene_dir[dir_len]
                    dir_len += 1
                var fn_i = 0
                while str_val[fn_i] != UInt8(0) and dir_len + fn_i < PSC_FILE_MAX * 2 - 1:
                    full[dir_len + fn_i] = str_val[fn_i]
                    fn_i += 1
                full[dir_len + fn_i] = UInt8(0)
                # Register as a texture entry
                var ti = Int(s[0].n_textures)
                if ti < PSC_MAX_TEX:
                    # Name = "__inf_N"
                    s[0].tex_names[ti * PSC_NAME_MAX + 0] = UInt8(95)  # '_'
                    s[0].tex_names[ti * PSC_NAME_MAX + 1] = UInt8(95)
                    s[0].tex_names[ti * PSC_NAME_MAX + 2] = UInt8(105) # 'i'
                    s[0].tex_names[ti * PSC_NAME_MAX + 3] = UInt8(110) # 'n'
                    s[0].tex_names[ti * PSC_NAME_MAX + 4] = UInt8(102) # 'f'
                    s[0].tex_names[ti * PSC_NAME_MAX + 5] = UInt8(0)
                    var dst = s[0].tex_files + ti * PSC_FILE_MAX * 2
                    for ci in range(dir_len + fn_i + 1):
                        dst[ci] = full[ci]
                    tex_i = Int32(ti)
                    s[0].n_textures += 1
                full.free()
            s[0].inf_tex_idx[idx] = tex_i
            s[0].inf_rgb[idx*3+0] = rgb[0] * scale
            s[0].inf_rgb[idx*3+1] = rgb[1] * scale
            s[0].inf_rgb[idx*3+2] = rgb[2] * scale
            s[0].n_infinite += 1

    ltype.free(); type_buf.free(); name_buf.free(); str_val.free(); rgb.free(); xyz.free()

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
        elif _psc_streq(kw_buf, "LightSource"):
            _psc_handle_light_source(handle, s)
        elif _psc_streq(kw_buf, "Texture"):
            _psc_handle_texture(handle, s)
        elif _psc_streq(kw_buf, "Include") or _psc_streq(kw_buf, "Import"):
            # Read filename, resolve relative to scene_dir, parse sub-file recursively
            var inc_name = alloc[UInt8](PSC_FILE_MAX)
            _ = mojo_scanner_parse_quoted_string(handle, inc_name, PSC_FILE_MAX)
            var inc_path = alloc[UInt8](PSC_FILE_MAX * 2)
            var dlen = 0
            while s[0].scene_dir[dlen] != UInt8(0):
                inc_path[dlen] = s[0].scene_dir[dlen]
                dlen += 1
            var fi = 0
            while inc_name[fi] != UInt8(0) and dlen + fi < PSC_FILE_MAX * 2 - 1:
                inc_path[dlen + fi] = inc_name[fi]
                fi += 1
            inc_path[dlen + fi] = UInt8(0)
            var sub_handle = mojo_scanner_new(inc_path)
            if mojo_scanner_is_at_end(sub_handle) == 0:
                _psc_parse(sub_handle, s)
            mojo_scanner_free(sub_handle)
            inc_name.free(); inc_path.free()
        elif _psc_streq(kw_buf, "Material"):
            # Anonymous inline material: treat like MakeNamedMaterial under a temp name "_mat"
            # then immediately set cur_mat_idx to it.
            _psc_handle_make_named_material(handle, s)
            # The name was parsed as part of make_named_material; find the last added
            s[0].cur_mat_idx = s[0].n_named - Int32(1)
        elif _psc_streq(kw_buf, "MakeNamedMedium"):
            _psc_handle_make_named_medium(handle, s)
        elif _psc_streq(kw_buf, "MediumInterface"):
            _psc_handle_medium_interface(handle, s)
        elif _psc_streq(kw_buf, "ConcatTransform"):
            # ConcatTransform [16 floats] — multiply CTM on right
            _ = mojo_scanner_scan_char(handle, UInt8(91))  # '['
            var tmp = alloc[Float32](16)
            for i in range(16):
                _ = mojo_scanner_scan_float(handle, tmp + i)
            _ = mojo_scanner_scan_char(handle, UInt8(93))  # ']'
            var result = alloc[Float32](16)
            mojo_matrix_multiply(s[0].ctm, tmp, result)
            for i in range(16):
                s[0].ctm[i] = result[i]
            tmp.free(); result.free()
        else:
            _ = mojo_scanner_parse_quoted_string(handle, kw_buf, 256)
            _psc_skip_params(handle)

    kw_buf.free()
    ws_delims.free()

fn _psc_make_perspective(fov_deg: Float32, near: Float32,
                         dst: UnsafePointer[Float32, MutAnyOrigin]):
    var half_rad = fov_deg * PI / Float32(360)
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
        var mt = s[0].named_type[i]
        var ior = s[0].named_ior[i]
        mats[i].type = mt
        mats[i].tex_idx = s[0].named_tex_idx[i]
        mats[i].roughU  = s[0].named_roughU[i]
        mats[i].roughV  = s[0].named_roughV[i]
        mats[i].normal_tex_idx = s[0].named_normal_tex_idx[i]
        mats[i].medium_interface_idx = Int32(-1)
        if mt == Int8(4):  # dielectric: albedo.r holds IOR
            mats[i].albedo = RGB(ior, Float32(0), Float32(0))
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))
        elif mt == Int8(5):  # coated diffuse: emission.r holds IOR
            mats[i].albedo = s[0].named_albedo[i]
            mats[i].emission = RGB(ior, Float32(0), Float32(0))
        elif mt == Int8(7):  # coated conductor: emission.r holds IOR (clearcoat eta)
            mats[i].albedo = s[0].named_albedo[i]
            mats[i].emission = RGB(ior, Float32(0), Float32(0))
        elif mt == Int8(9):  # thin dielectric: albedo.r holds IOR (same as type 4)
            mats[i].albedo = RGB(ior, Float32(0), Float32(0))
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))
        elif mt == Int8(8):  # mix: resolve sub-material names to indices
            var m1_name = s[0].named_mix1 + i * PSC_NAME_MAX
            var m2_name = s[0].named_mix2 + i * PSC_NAME_MAX
            var idx1 = Int32(0)  # default to first material
            var idx2 = Int32(0)
            for j in range(n_regular):
                if _psc_strcmp(s[0].named_names + j * PSC_NAME_MAX, m1_name) == 0:
                    idx1 = Int32(j)
                if _psc_strcmp(s[0].named_names + j * PSC_NAME_MAX, m2_name) == 0:
                    idx2 = Int32(j)
            # Pack both indices into tex_idx: high 16 bits = idx2, low 16 bits = idx1
            mats[i].tex_idx = (idx2 << 16) | (idx1 & Int32(0xFFFF))
            mats[i].roughU  = s[0].named_amount[i]  # blend factor
            mats[i].albedo = s[0].named_albedo[i]
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))
        elif mt == Int8(6):  # diffusetransmission: emission holds the transmittance
            mats[i].albedo = s[0].named_albedo[i]
            mats[i].emission = s[0].named_transmittance[i]
        elif mt == Int8(11):  # hair: simplified as diffuse with melanin-computed color
            mats[i].albedo = s[0].named_albedo[i]
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))
            mats[i].type = Int8(1)  # redirect to diffuse shader
        else:
            mats[i].albedo = s[0].named_albedo[i]
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))

    # ---- Meshes + area lights ----
    var n_meshes = Int(s[0].n_meshes)
    var meshes  = alloc[TriangleMesh_C](max(n_meshes, 1))
    var out_pts = alloc[UnsafePointer[Float32, MutAnyOrigin]](max(n_meshes, 1))
    var out_vis = alloc[UnsafePointer[Int64, MutAnyOrigin]](max(n_meshes, 1))
    var out_fis = alloc[UnsafePointer[Int64, MutAnyOrigin]](max(n_meshes, 1))
    var out_nv    = alloc[Int32](max(n_meshes, 1))
    var out_nt    = alloc[Int32](max(n_meshes, 1))
    var out_uv_nv = alloc[Int32](max(n_meshes, 1))

    var al_list = alloc[AreaLight_C](max(Int(n_al), 1))
    var al_count = Int32(0)

    var al_mat_base = n_regular

    for i in range(n_meshes):
        out_pts[i] = s[0].mesh_pts_list[i]
        out_vis[i] = s[0].mesh_vis_list[i]
        out_fis[i] = s[0].mesh_fis_list[i]
        out_nv[i]  = s[0].mesh_nv[i]
        out_nt[i]  = s[0].mesh_nt[i]
        meshes[i].points        = out_pts[i]
        meshes[i].vertexIndices = out_vis[i]
        meshes[i].faceIndices   = out_fis[i]
        meshes[i].uvs           = s[0].mesh_uvs_list[i]
        out_uv_nv[i] = s[0].mesh_nv[i] if s[0].mesh_has_uvs[i] != Int8(0) else Int32(0)

        if s[0].mesh_is_al[i] != 0:
            var al_idx = Int(al_count)
            var em = s[0].mesh_al_rgb[i]
            # Compute total light mesh area from its triangles
            var al_nt   = Int(s[0].mesh_nt[i])
            var al_pts  = s[0].mesh_pts_list[i]
            var al_vis  = s[0].mesh_vis_list[i]
            var t_area  = Float32(0.0)
            for ti in range(al_nt):
                var vi0 = Int(al_vis[ti*3+0]) * 4
                var vi1 = Int(al_vis[ti*3+1]) * 4
                var vi2 = Int(al_vis[ti*3+2]) * 4
                var ex = al_pts[vi1+0] - al_pts[vi0+0]
                var ey = al_pts[vi1+1] - al_pts[vi0+1]
                var ez = al_pts[vi1+2] - al_pts[vi0+2]
                var fx = al_pts[vi2+0] - al_pts[vi0+0]
                var fy = al_pts[vi2+1] - al_pts[vi0+1]
                var fz = al_pts[vi2+2] - al_pts[vi0+2]
                var cx = ey*fz - ez*fy
                var cy = ez*fx - ex*fz
                var cz = ex*fy - ey*fx
                t_area += Float32(0.5) * sqrt(cx*cx + cy*cy + cz*cz)
            al_list[al_idx].meshIdx    = Int32(i)
            al_list[al_idx].n_tris     = Int32(al_nt)
            al_list[al_idx].emission   = em
            al_list[al_idx].total_area = t_area

            mats[al_mat_base + al_idx].type     = Int8(2)  # arealight
            mats[al_mat_base + al_idx].albedo   = RGB(Float32(0), Float32(0), Float32(0))
            mats[al_mat_base + al_idx].emission = em
            mats[al_mat_base + al_idx].tex_idx  = Int32(-1)
            mats[al_mat_base + al_idx].roughU   = Float32(0)
            mats[al_mat_base + al_idx].roughV   = Float32(0)
            mats[al_mat_base + al_idx].normal_tex_idx = Int32(-1)
            mats[al_mat_base + al_idx].medium_interface_idx = Int32(-1)
            al_count += 1

    # ---- BVH construction ----
    # Build per-shape medium interfaces.
    # Phase 1: count how many shapes need medium interfaces.
    var n_with_mi = 0
    for mi in range(n_meshes):
        if s[0].mesh_inside_med[mi] >= Int32(0) or s[0].mesh_outside_med[mi] >= Int32(0):
            n_with_mi += 1
    for si in range(Int(s[0].n_spheres)):
        if s[0].sph_inside_med[si] >= Int32(0) or s[0].sph_outside_med[si] >= Int32(0):
            n_with_mi += 1

    if n_with_mi > 0:
        # Phase 2: expand mats array to hold duplicates (one per shape with MI)
        var expanded_n = n_mats + n_with_mi
        var new_mats = alloc[Material_C](expanded_n)
        for ci in range(n_mats):
            new_mats[ci] = mats[ci]
        mats.free()
        mats = new_mats

        var iface_buf = alloc[MediumInterface_C](n_with_mi)
        var dup_idx = n_mats  # next slot for duplicated material
        var iface_idx = 0

        # Phase 3: for each mesh with MI, create a duplicated material + interface
        for mi in range(n_meshes):
            var ins = s[0].mesh_inside_med[mi]
            var out = s[0].mesh_outside_med[mi]
            if ins < Int32(0) and out < Int32(0):
                continue
            var orig_mat = Int(s[0].mesh_mat_idx[mi])
            if orig_mat < 0:
                continue
            # Duplicate the material
            mats[dup_idx] = mats[orig_mat]
            mats[dup_idx].medium_interface_idx = Int32(iface_idx)
            iface_buf[iface_idx] = MediumInterface_C(ins, out)
            # Point this mesh at the duplicated material
            s[0].mesh_mat_idx[mi] = Int32(dup_idx)
            dup_idx += 1
            iface_idx += 1

        # Phase 4: same for spheres
        for si in range(Int(s[0].n_spheres)):
            var ins = s[0].sph_inside_med[si]
            var out = s[0].sph_outside_med[si]
            if ins < Int32(0) and out < Int32(0):
                continue
            var orig_mat = Int(s[0].sph_mat[si])
            if orig_mat < 0:
                continue
            mats[dup_idx] = mats[orig_mat]
            mats[dup_idx].medium_interface_idx = Int32(iface_idx)
            iface_buf[iface_idx] = MediumInterface_C(ins, out)
            s[0].sph_mat[si] = Int32(dup_idx)
            dup_idx += 1
            iface_idx += 1

        n_mats = dup_idx
        psc[0].medium_ifaces = iface_buf
        psc[0].medium_iface_count = Int32(iface_idx)

    else:
        psc[0].medium_ifaces = UnsafePointer[MediumInterface_C, MutAnyOrigin]()
        psc[0].medium_iface_count = Int32(0)

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

    # ---- Texture filename table ----
    var n_tex = Int(s[0].n_textures)
    var tex_ptrs = alloc[UnsafePointer[UInt8, MutAnyOrigin]](max(n_tex, 1))
    for ti in range(n_tex):
        var src = s[0].tex_files + ti * PSC_FILE_MAX * 2
        # Compute length
        var slen = 0
        while src[slen] != UInt8(0):
            slen += 1
        var copy = alloc[UInt8](slen + 1)
        for ci in range(slen):
            copy[ci] = src[ci]
        copy[slen] = UInt8(0)
        tex_ptrs[ti] = copy

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
    psc[0].mesh_uv_n_verts  = out_uv_nv
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
    psc[0].tex_filenames    = tex_ptrs
    psc[0].tex_count        = Int32(n_tex)

    # ---- Non-area lights ----
    var nd = Int(s[0].n_distant)
    if nd > 0:
        var dl_buf = alloc[DistantLight_C](nd)
        for i in range(nd):
            dl_buf[i] = DistantLight_C(
                Vec3f(s[0].dist_dirs[i*3+0], s[0].dist_dirs[i*3+1], s[0].dist_dirs[i*3+2]),
                Float32(0),
                RGB(s[0].dist_rgb[i*3+0], s[0].dist_rgb[i*3+1], s[0].dist_rgb[i*3+2]),
                Float32(0))
        psc[0].distant_lights = dl_buf
    else:
        psc[0].distant_lights = UnsafePointer[DistantLight_C, MutAnyOrigin]()
    psc[0].distant_count = Int32(nd)

    var np2 = Int(s[0].n_point)
    if np2 > 0:
        var pl_buf = alloc[PointLight_C](np2)
        for i in range(np2):
            pl_buf[i] = PointLight_C(
                Point3f(s[0].pt_pos[i*3+0], s[0].pt_pos[i*3+1], s[0].pt_pos[i*3+2]),
                Float32(0),
                RGB(s[0].pt_rgb[i*3+0], s[0].pt_rgb[i*3+1], s[0].pt_rgb[i*3+2]),
                Float32(0))
        psc[0].point_lights = pl_buf
    else:
        psc[0].point_lights = UnsafePointer[PointLight_C, MutAnyOrigin]()
    psc[0].point_count = Int32(np2)

    var ni = Int(s[0].n_infinite)
    if ni > 0:
        var il_buf = alloc[InfiniteLight_C](ni)
        for i in range(ni):
            var tidx = s[0].inf_tex_idx[i]
            var sc = RGB(s[0].inf_rgb[i*3+0], s[0].inf_rgb[i*3+1], s[0].inf_rgb[i*3+2])
            var cdf_w = Int32(0); var cdf_h = Int32(0)
            var cdf_ptr = UnsafePointer[Float32, MutAnyOrigin]()
            if tidx >= Int32(0):
                # Build 2D importance-sampling CDF from env-map luminance via load_texture_rgb
                var fname = psc[0].tex_filenames[Int(tidx)]
                var pixels_ptr = alloc[UnsafePointer[Float32, MutAnyOrigin]](1)
                var iw_out = alloc[Int32](1); var ih_out = alloc[Int32](1)
                iw_out[0] = Int32(0); ih_out[0] = Int32(0)
                var load_ok = external_call["load_texture_rgb", Int32,
                    UnsafePointer[UInt8, MutAnyOrigin],
                    UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin],
                    UnsafePointer[Int32, MutAnyOrigin], UnsafePointer[Int32, MutAnyOrigin]](
                    fname, pixels_ptr, iw_out, ih_out)
                var iw = Int(iw_out[0]); var ih = Int(ih_out[0])
                iw_out.free(); ih_out.free()
                if load_ok == Int32(0) and iw > 0 and ih > 0:
                    var pixels = pixels_ptr[0]
                    # CDF layout: (ih+1) marginal + ih*(iw+1) conditional floats
                    var cdf_size = (ih + 1) + ih * (iw + 1)
                    var cdf_buf = alloc[Float32](cdf_size)
                    # Compute per-row luminance sums (marginal pdf)
                    var row_sums = alloc[Float32](ih)
                    for ry in range(ih):
                        # Sin-weighted solid angle for lat-long map
                        var sin_theta = sin(PI * (Float32(ry) + Float32(0.5)) / Float32(ih))
                        var row_sum = Float32(0.0)
                        for rx in range(iw):
                            var r2 = pixels[(ry * iw + rx) * 3 + 0]
                            var g2 = pixels[(ry * iw + rx) * 3 + 1]
                            var b2 = pixels[(ry * iw + rx) * 3 + 2]
                            var lum = Float32(0.2126) * r2 + Float32(0.7152) * g2 + Float32(0.0722) * b2
                            row_sum += lum * sin_theta
                        row_sums[ry] = row_sum
                    # Build marginal CDF (ih+1 values, starts at 0)
                    cdf_buf[0] = Float32(0.0)
                    for ry in range(ih):
                        cdf_buf[ry + 1] = cdf_buf[ry] + row_sums[ry]
                    var total = cdf_buf[ih]
                    if total > Float32(0.0):
                        var inv_total = Float32(1.0) / total
                        for ry in range(ih + 1):
                            cdf_buf[ry] *= inv_total
                    # Build per-row conditional CDFs (ih * (iw+1) values)
                    for ry in range(ih):
                        var sin_theta = sin(PI * (Float32(ry) + Float32(0.5)) / Float32(ih))
                        var base = (ih + 1) + ry * (iw + 1)
                        cdf_buf[base] = Float32(0.0)
                        for rx in range(iw):
                            var r2 = pixels[(ry * iw + rx) * 3 + 0]
                            var g2 = pixels[(ry * iw + rx) * 3 + 1]
                            var b2 = pixels[(ry * iw + rx) * 3 + 2]
                            var lum = Float32(0.2126) * r2 + Float32(0.7152) * g2 + Float32(0.0722) * b2
                            cdf_buf[base + rx + 1] = cdf_buf[base + rx] + lum * sin_theta
                        var row_total = cdf_buf[base + iw]
                        if row_total > Float32(0.0):
                            var inv_rt = Float32(1.0) / row_total
                            for rx in range(iw + 1):
                                cdf_buf[base + rx] *= inv_rt
                    row_sums.free()
                    _ = external_call["free_texture_rgb", Int32,
                        UnsafePointer[Float32, MutAnyOrigin]](pixels)
                    cdf_ptr = cdf_buf
                    cdf_w = Int32(iw); cdf_h = Int32(ih)
                pixels_ptr.free()
            il_buf[i] = InfiniteLight_C(sc, tidx, cdf_w, cdf_h, cdf_ptr)
        psc[0].infinite_lights = il_buf
    else:
        psc[0].infinite_lights = UnsafePointer[InfiniteLight_C, MutAnyOrigin]()
    psc[0].infinite_count = Int32(ni)

    # ---- Analytical spheres ----
    var ns = Int(s[0].n_spheres)
    if ns > 0:
        var sph_buf = alloc[Sphere_C](ns)
        for i in range(ns):
            var em = SampledSpectrum(s[0].sph_rgb[i].r, s[0].sph_rgb[i].g, s[0].sph_rgb[i].b)
            sph_buf[i] = Sphere_C(
                Point3f(s[0].sph_cx[i], s[0].sph_cy[i], s[0].sph_cz[i]),
                s[0].sph_r[i],
                s[0].sph_mat[i],
                s[0].sph_al[i],
                Int8(0), Int8(0), Int8(0),
                em)
        psc[0].spheres = sph_buf
    else:
        psc[0].spheres = UnsafePointer[Sphere_C, MutAnyOrigin]()
    psc[0].sphere_count = Int32(ns)

    # ---- Media ----
    var nm = Int(s[0].n_mediums)
    if nm > 0:
        var med_buf = alloc[Medium_C](nm)
        for i in range(nm):
            var sa = SampledSpectrum(s[0].med_sa[i*3], s[0].med_sa[i*3+1], s[0].med_sa[i*3+2])
            var ss = SampledSpectrum(s[0].med_ss[i*3], s[0].med_ss[i*3+1], s[0].med_ss[i*3+2])
            med_buf[i] = Medium_C(sa, ss, s[0].med_g[i],
                                  Float32(0), Float32(0), Float32(0))
        psc[0].mediums = med_buf
    else:
        psc[0].mediums = UnsafePointer[Medium_C, MutAnyOrigin]()
    psc[0].medium_count = Int32(nm)


# ── Exported API ──────────────────────────────────────────────────────────────

fn mojo_parse_scene(path: UnsafePointer[UInt8, MutAnyOrigin]
                    ) -> UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]:
    external_call["createTextureSystem", NoneType]()
    var handle = mojo_scanner_new(path)
    if not handle:
        var psc = alloc[ParsedScene_Mojo](1)
        psc[0].mesh_count = Int32(0)
        psc[0].prim_count = Int32(0)
        psc[0].bvh_node_count = Int32(0)
        psc[0].material_count = Int32(0)
        psc[0].area_light_count = Int32(0)
        psc[0].tex_count = Int32(0)
        psc[0].distant_count = Int32(0)
        psc[0].point_count = Int32(0)
        psc[0].infinite_count = Int32(0)
        psc[0].sphere_count = Int32(0)
        return psc

    var s = _psc_state_new()
    var pi = 0
    while path[pi] != UInt8(0):
        pi += 1
    var last_slash = -1
    for ki in range(pi):
        if path[ki] == UInt8(47):
            last_slash = ki
    if last_slash >= 0:
        for ki in range(last_slash + 1):
            s[0].scene_dir[ki] = path[ki]
        s[0].scene_dir[last_slash + 1] = UInt8(0)
    else:
        s[0].scene_dir[0] = UInt8(0)
    _psc_parse(handle, s)
    mojo_scanner_free(handle)
    # Flush accumulated hair curve geometry into a single mesh (if any)
    _psc_flush_hair(s)

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
        psc[0].mesh_uv_n_verts.free()
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
    if psc[0].tex_filenames:
        var nt = Int(psc[0].tex_count)
        for ti in range(nt):
            psc[0].tex_filenames[ti].free()
        psc[0].tex_filenames.free()
    if psc[0].distant_count > 0:
        psc[0].distant_lights.free()
    if psc[0].point_count > 0:
        psc[0].point_lights.free()
    if psc[0].infinite_count > 0:
        psc[0].infinite_lights.free()
    if psc[0].sphere_count > 0:
        psc[0].spheres.free()
    psc.free()

def mojo_parsed_scene_descriptor(
    psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]
) -> UnsafePointer[SceneDescriptor2_C, MutAnyOrigin]:
    var sd = alloc[SceneDescriptor2_C](1)
    sd[0].bvh2Nodes        = psc[0].bvh_nodes
    sd[0].primIds          = psc[0].prim_ids
    sd[0].meshes           = psc[0].meshes
    sd[0].meshCount        = Int64(psc[0].mesh_count)
    sd[0].materials        = psc[0].materials
    sd[0].materialCount    = Int64(psc[0].material_count)
    sd[0].areaLights       = psc[0].area_lights
    sd[0].areaLightCount   = Int64(psc[0].area_light_count)
    sd[0].textures         = psc[0].tex_filenames
    sd[0].textureCount     = Int64(psc[0].tex_count)
    sd[0].distantLights    = psc[0].distant_lights
    sd[0].distantLightCount = Int64(psc[0].distant_count)
    sd[0].pointLights      = psc[0].point_lights
    sd[0].pointLightCount  = Int64(psc[0].point_count)
    sd[0].infiniteLights   = psc[0].infinite_lights
    sd[0].infiniteLightCount = Int64(psc[0].infinite_count)
    sd[0].spheres          = psc[0].spheres
    sd[0].sphereCount      = Int64(psc[0].sphere_count)
    sd[0].mediums          = psc[0].mediums
    sd[0].mediumCount      = Int64(psc[0].medium_count)
    sd[0].mediumInterfaces = psc[0].medium_ifaces
    sd[0].mediumIfaceCount = Int64(psc[0].medium_iface_count)
    return sd
