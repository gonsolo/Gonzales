from std.ffi import external_call
from std.time import perf_counter_ns
from .ply import load_ply
from std.math import tan, sqrt, acos, atan2, sin, abs, exp
from std.memory import alloc
from .geometry import RGB, SampledSpectrum, Point3f, Vec3f, Ray_C, Material_C, AreaLight_C, Sphere_C, DistantLight_C, PointLight_C, InfiniteLight_C, TriangleMesh_C, PrimId_C, PI, TWO_PI, Medium_C, MediumInterface_C
from .transform import matrix_multiply, matrix_invert, transform_points, transform_normals
from .bvh import BVH2Node, SceneDescriptor2_C, build_bvh2
from .sampling import gaussian_norm

# ── pbrt Scanner Helpers ──────────────────────────────────────────────

@always_inline
def is_whitespace(b: UInt8) -> Bool:
    return b == UInt8(32) or b == UInt8(9) or b == UInt8(10) or b == UInt8(13)

# Skip whitespace and pbrt line comments (# … \n).
@always_inline
def skip_whitespace_and_comments(bytes: UnsafePointer[UInt8, MutAnyOrigin], length: Int, pos: Int) -> Int:
    var cur = pos
    while cur < length:
        var b = bytes[cur]
        if is_whitespace(b):
            cur += 1
        elif b == UInt8(35):  # '#'
            while cur < length and bytes[cur] != UInt8(10):
                cur += 1
        else:
            break
    return cur

@always_inline
def is_digit(b: UInt8) -> Bool:
    return b >= UInt8(48) and b <= UInt8(57)

def scan_int(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    result: UnsafePointer[Int32, MutAnyOrigin],
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and is_whitespace(bytes[cur]):
        cur += 1
    if cur >= len:
        return Int32(0)

    var negative = False
    if bytes[cur] == UInt8(45):       # '-'
        negative = True
        cur += 1
    if cur >= len or not is_digit(bytes[cur]):
        return Int32(0)
    var value = Int32(0)
    while cur < len and is_digit(bytes[cur]):
        value = value * Int32(10) + Int32(bytes[cur]) - Int32(48)
        cur += 1
    if negative:
        value = -value
    cursor[0] = Int32(cur)
    result[0] = value
    return Int32(1)

def scan_float(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    result: UnsafePointer[Float32, MutAnyOrigin],
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and is_whitespace(bytes[cur]):
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
    while cur < len and is_digit(bytes[cur]):
        int_part = int_part * Int32(10) + Int32(bytes[cur]) - Int32(48)
        cur += 1
        int_seen = True
    if int_negative:
        int_part = -int_part

    var dval = Float64(int_part)

    if cur < len and bytes[cur] == UInt8(46):   # '.'
        cur += 1
        var tenth = Float64(0.1)
        while cur < len and is_digit(bytes[cur]):
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
        while cur < len and is_whitespace(bytes[cur]):
            cur += 1
        var exp_negative = False
        if cur < len and bytes[cur] == UInt8(45):
            exp_negative = True
            cur += 1
        var exp_val = Int32(0)
        while cur < len and is_digit(bytes[cur]):
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

def count_floats(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: Int32,
) -> Int32:
    var cur = Int(cursor)
    var len = Int(length)
    var count = Int32(0)
    while True:
        while cur < len and is_whitespace(bytes[cur]):
            cur += 1
        if cur >= len:
            break
        if cur < len and bytes[cur] == UInt8(45):   # optional '-'
            cur += 1
        var int_seen = False
        while cur < len and is_digit(bytes[cur]):
            int_seen = True
            cur += 1
        if cur < len and bytes[cur] == UInt8(46):   # '.'
            cur += 1
            while cur < len and is_digit(bytes[cur]):
                cur += 1
        elif not int_seen:
            break
        if cur < len and bytes[cur] == UInt8(101):  # 'e'
            cur += 1
            if cur < len and bytes[cur] == UInt8(45):
                cur += 1
            while cur < len and is_digit(bytes[cur]):
                cur += 1
        count += Int32(1)
    return count

def scan_floats(
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
        while cur < len and is_whitespace(bytes[cur]):
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
        while cur < len and is_digit(bytes[cur]):
            int_part = int_part * Int32(10) + Int32(bytes[cur]) - Int32(48)
            cur += 1
            int_seen = True
        if int_negative:
            int_part = -int_part
        var dval = Float64(int_part)
        if cur < len and bytes[cur] == UInt8(46):
            cur += 1
            var tenth = Float64(0.1)
            while cur < len and is_digit(bytes[cur]):
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
            while cur < len and is_whitespace(bytes[cur]):
                cur += 1
            var exp_negative = False
            if cur < len and bytes[cur] == UInt8(45):
                exp_negative = True
                cur += 1
            var exp_val = Int32(0)
            while cur < len and is_digit(bytes[cur]):
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

def count_ints(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: Int32,
) -> Int32:
    var cur = Int(cursor)
    var len = Int(length)
    var count = Int32(0)
    while True:
        while cur < len and is_whitespace(bytes[cur]):
            cur += 1
        if cur >= len:
            break
        if cur < len and bytes[cur] == UInt8(45):
            cur += 1
        if cur >= len or not is_digit(bytes[cur]):
            break
        while cur < len and is_digit(bytes[cur]):
            cur += 1
        count += Int32(1)
    return count

def scan_ints(
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
        while cur < len and is_whitespace(bytes[cur]):
            cur += 1
        if cur >= len:
            break
        var negative = False
        if bytes[cur] == UInt8(45):
            negative = True
            cur += 1
        if cur >= len or not is_digit(bytes[cur]):
            break
        var value = Int32(0)
        while cur < len and is_digit(bytes[cur]):
            value = value * Int32(10) + Int32(bytes[cur]) - Int32(48)
            cur += 1
        if negative:
            value = -value
        result[Int(count)] = value
        count += Int32(1)
    cursor[0] = Int32(cur)
    return count

def scan_char(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    expected: UInt8,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and is_whitespace(bytes[cur]):
        cur += 1
    cursor[0] = Int32(cur)
    if cur >= len or bytes[cur] != expected:
        return Int32(0)
    cursor[0] = Int32(cur + 1)
    return Int32(1)

def peek_char(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    expected: UInt8,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and is_whitespace(bytes[cur]):
        cur += 1
    cursor[0] = Int32(cur)
    if cur >= len or bytes[cur] != expected:
        return Int32(0)
    return Int32(1)

def scan_token(
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
    cur = skip_whitespace_and_comments(bytes, len, cur)
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

def parse_quoted_string(
    bytes: UnsafePointer[UInt8, MutAnyOrigin],
    length: Int32,
    cursor: UnsafePointer[Int32, MutAnyOrigin],
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    max_buf: Int32,
) -> Int32:
    var cur = Int(cursor[0])
    var len = Int(length)
    while cur < len and is_whitespace(bytes[cur]):
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

def parse_param_header(
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
    cur = skip_whitespace_and_comments(bytes, len, cur)
    cursor[0] = Int32(cur)
    if cur >= len or bytes[cur] != UInt8(34):
        return Int32(0)
    cur += 1  # opening '"'
    # type: read until whitespace or '"'
    var t = Int32(0)
    while cur < len and not is_whitespace(bytes[cur]) and bytes[cur] != UInt8(34):
        if t < type_max - 1:
            type_buf[Int(t)] = bytes[cur]
        t += Int32(1)
        cur += 1
    if type_max > 0:
        var cap = Int(t) if Int(t) < Int(type_max) - 1 else Int(type_max) - 1
        type_buf[cap] = UInt8(0)
    # skip separator whitespace
    while cur < len and is_whitespace(bytes[cur]):
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
    while cur < len and is_whitespace(bytes[cur]):
        cur += 1
    if cur < len and bytes[cur] == UInt8(91):   # '[' = 91
        cur += 1
        is_array[0] = Int32(1)
    else:
        is_array[0] = Int32(0)
    cursor[0] = Int32(cur)
    return Int32(1)


# ── PbrtScanner ──────────────────────────────────────────────────────────

struct PbrtScanner:
    var buffer: UnsafePointer[UInt8, MutAnyOrigin]
    var total_bytes: Int32
    var cursor: Int32
    var is_at_end: Int32


@always_inline
def scanner_cursor_ptr(handle: UnsafePointer[PbrtScanner, MutAnyOrigin]) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin].unsafe_dangling()


@always_inline
def scanner_call_int(handle: UnsafePointer[PbrtScanner, MutAnyOrigin], result: UnsafePointer[Int32, MutAnyOrigin]) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = scan_int(handle[0].buffer, handle[0].total_bytes, cur, result)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


@always_inline
def scanner_call_float(handle: UnsafePointer[PbrtScanner, MutAnyOrigin], result: UnsafePointer[Float32, MutAnyOrigin]) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = scan_float(handle[0].buffer, handle[0].total_bytes, cur, result)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


def scanner_open(path: UnsafePointer[UInt8, MutAnyOrigin]) -> UnsafePointer[PbrtScanner, MutAnyOrigin]:
    var handle = alloc[PbrtScanner](1)
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
        handle[0].buffer = UnsafePointer[UInt8, MutAnyOrigin].unsafe_dangling()
        handle[0].total_bytes = Int32(0)
        handle[0].cursor = Int32(0)
        handle[0].is_at_end = Int32(1)
    return handle


def scanner_from_bytes(bytes: UnsafePointer[UInt8, MutAnyOrigin], length: Int32) -> UnsafePointer[PbrtScanner, MutAnyOrigin]:
    var handle = alloc[PbrtScanner](1)
    var buf = alloc[UInt8](Int(length) + 1)
    for i in range(Int(length)):
        buf[i] = bytes[i]
    buf[Int(length)] = UInt8(0)
    handle[0].buffer = buf
    handle[0].total_bytes = length
    handle[0].cursor = Int32(0)
    handle[0].is_at_end = Int32(0)
    return handle


def scanner_free(handle: UnsafePointer[PbrtScanner, MutAnyOrigin]):
    if Int(handle[0].buffer) > 1:
        handle[0].buffer.free()
    handle.free()


def scanner_is_at_end(handle: UnsafePointer[PbrtScanner, MutAnyOrigin]) -> Int32:
    return handle[0].is_at_end


def scanner_location(handle: UnsafePointer[PbrtScanner, MutAnyOrigin]) -> Int32:
    return handle[0].cursor


def scanner_peek_char(handle: UnsafePointer[PbrtScanner, MutAnyOrigin], expected: UInt8) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = peek_char(handle[0].buffer, handle[0].total_bytes, cur, expected)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


def scanner_scan_char(handle: UnsafePointer[PbrtScanner, MutAnyOrigin], expected: UInt8) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = scan_char(handle[0].buffer, handle[0].total_bytes, cur, expected)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


def scanner_scan_int(handle: UnsafePointer[PbrtScanner, MutAnyOrigin], result: UnsafePointer[Int32, MutAnyOrigin]) -> Int32:
    return scanner_call_int(handle, result)


def scanner_scan_float(handle: UnsafePointer[PbrtScanner, MutAnyOrigin], result: UnsafePointer[Float32, MutAnyOrigin]) -> Int32:
    return scanner_call_float(handle, result)


def scanner_count_floats(handle: UnsafePointer[PbrtScanner, MutAnyOrigin]) -> Int32:
    return count_floats(handle[0].buffer, handle[0].total_bytes, handle[0].cursor)


def scanner_scan_floats(handle: UnsafePointer[PbrtScanner, MutAnyOrigin], dst: UnsafePointer[Float32, MutAnyOrigin], max_count: Int32) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = scan_floats(handle[0].buffer, handle[0].total_bytes, cur, dst, max_count)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


def scanner_count_ints(handle: UnsafePointer[PbrtScanner, MutAnyOrigin]) -> Int32:
    return count_ints(handle[0].buffer, handle[0].total_bytes, handle[0].cursor)


def scanner_scan_ints(handle: UnsafePointer[PbrtScanner, MutAnyOrigin], dst: UnsafePointer[Int32, MutAnyOrigin], max_count: Int32) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = scan_ints(handle[0].buffer, handle[0].total_bytes, cur, dst, max_count)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


def scanner_parse_quoted_string(handle: UnsafePointer[PbrtScanner, MutAnyOrigin], buf: UnsafePointer[UInt8, MutAnyOrigin], max_buf: Int32) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = parse_quoted_string(handle[0].buffer, handle[0].total_bytes, cur, buf, max_buf)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


def scanner_parse_param_header(
    handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
    type_buf: UnsafePointer[UInt8, MutAnyOrigin], type_max: Int32,
    name_buf: UnsafePointer[UInt8, MutAnyOrigin], name_max: Int32,
    is_array: UnsafePointer[Int32, MutAnyOrigin],
) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = parse_param_header(handle[0].buffer, handle[0].total_bytes, cur,
                                      type_buf, type_max, name_buf, name_max, is_array)
    handle[0].cursor = cur[0]
    cur.free()
    return ret


def scanner_scan_token(
    handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
    delims: UnsafePointer[UInt8, MutAnyOrigin], n_delims: Int32,
    buf: UnsafePointer[UInt8, MutAnyOrigin], max_buf: Int32,
) -> Int32:
    var cur = alloc[Int32](1)
    cur[0] = handle[0].cursor
    var ret = scan_token(handle[0].buffer, handle[0].total_bytes, cur,
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
    var mesh_nrm_n_verts: UnsafePointer[Int32, MutAnyOrigin]  # per-mesh normal vertex count; 0 = no shading normals
    var mesh_count:       Int32
    var bvh_nodes:        UnsafePointer[BVH2Node, MutAnyOrigin]
    var prim_ids:         UnsafePointer[PrimId_C, MutAnyOrigin]
    var bvh_node_count:   Int32
    var prim_count:       Int32
    var film_w:           Int32
    var film_h:           Int32
    var camera_fov:       Float32
    var film_iso:         Float32
    var film_max_comp:    Float32
    var film_filename:    UnsafePointer[UInt8, MutAnyOrigin]      # null-terminated
    var filter_sigma:     Float32
    var filter_support_x: Float32
    var filter_support_y: Float32
    var filter_norm_x:    Float32
    var filter_norm_y:    Float32
    var filter_weight:    Float32
    var filter_type:      Int32
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
# These types are private to parsing.mojo.

@fieldwise_init
struct AttributeState(Copyable, ImplicitlyCopyable, Movable):
    var mat_idx:        Int32
    var is_alight:      Bool
    var al_rgb:         RGB
    var inside_medium:  Int32
    var outside_medium: Int32
    var reverse_orient: Bool   # PBRT ReverseOrientation: flip surface normals

struct NamedMaterial(Copyable, ImplicitlyCopyable, Movable):
    var name:           String
    var albedo:         RGB
    var kind:           Int8
    var ior:            Float32
    var roughness_u:    Float32
    var roughness_v:    Float32
    var tex_idx:        Int32
    var normal_tex_idx: Int32
    var mix_name1:      String
    var mix_name2:      String
    var mix_amount:     Float32
    var transmittance:  RGB

    def __init__(out self, name: String):
        self.name           = name
        self.albedo         = RGB(Float32(0.8), Float32(0.8), Float32(0.8))
        self.kind           = Int8(0)
        self.ior            = Float32(1.5)
        self.roughness_u    = Float32(0)
        self.roughness_v    = Float32(0)
        self.tex_idx        = Int32(-1)
        self.normal_tex_idx = Int32(-1)
        self.mix_name1      = String("")
        self.mix_name2      = String("")
        self.mix_amount     = Float32(0.5)
        self.transmittance  = RGB(Float32(1), Float32(1), Float32(1))

struct MeshAccum(Copyable, Movable):
    var points:         List[Float32]  # 4 floats per vertex (xyz + pad)
    var vert_idxs:      List[Int64]    # flat vertex indices
    var face_idxs:      List[Int64]    # flat face indices
    var uvs:            List[Float32]  # 2 floats per vertex (or empty)
    var normals:        List[Float32]  # 3 floats per vertex (or empty) — world-space shading normals
    var mat_idx:        Int32
    var is_area_light:  Bool
    var al_rgb:         RGB
    var inside_medium:  Int32
    var outside_medium: Int32

    def __init__(out self, mat_idx: Int32, inside_medium: Int32, outside_medium: Int32):
        self.points        = List[Float32]()
        self.vert_idxs     = List[Int64]()
        self.face_idxs     = List[Int64]()
        self.uvs           = List[Float32]()
        self.normals       = List[Float32]()
        self.mat_idx       = mat_idx
        self.is_area_light = False
        self.al_rgb        = RGB(Float32(0), Float32(0), Float32(0))
        self.inside_medium  = inside_medium
        self.outside_medium = outside_medium

struct SceneParseState(Movable):
    # Current transform matrix and stack
    var ctm:       InlineArray[Float32, 16]
    var ctm_stack: List[InlineArray[Float32, 16]]

    # Attribute stack (material, area-light, medium per nesting level)
    var attr_stack: List[AttributeState]
    var cur_attr:   AttributeState

    # Named materials (MakeNamedMaterial / Material directives)
    var named_materials: List[NamedMaterial]

    # Mesh accumulation
    var meshes: List[MeshAccum]

    # Hair curve accumulator (lazily populated on first Shape "curve")
    var hair:         Optional[MeshAccum]

    # Non-area lights
    var distant_dirs: List[Float32]   # 3 floats per light
    var distant_rgbs: List[Float32]   # 3 floats per light
    var point_pos:    List[Float32]   # 3 floats per light
    var point_rgbs:   List[Float32]   # 3 floats per light
    var inf_tex_idx:  List[Int32]     # 1 per infinite light
    var inf_rgb:      List[Float32]   # 3 floats per light
    var inf_ctm:      List[Float32]   # 16 floats per light

    # Analytical sphere primitives
    var spheres_cx:  List[Float32]
    var spheres_cy:  List[Float32]
    var spheres_cz:  List[Float32]
    var spheres_r:   List[Float32]
    var spheres_mat: List[Int32]
    var spheres_al:  List[Bool]
    var spheres_rgb: List[RGB]
    var spheres_inside_med:  List[Int32]
    var spheres_outside_med: List[Int32]

    # Homogeneous media
    var med_names: List[String]
    var med_sa:    List[Float32]   # 3 floats per medium
    var med_ss:    List[Float32]   # 3 floats per medium
    var med_g:     List[Float32]   # 1 per medium

    # Medium interfaces
    var miface_inside:  List[Int32]
    var miface_outside: List[Int32]
    var miface_mat:     List[Int32]

    # Textures
    var tex_names: List[String]
    var tex_files: List[String]

    # Film / camera / sampler settings
    var film_w:           Int32
    var film_h:           Int32
    var film_iso:         Float32
    var film_max_comp:    Float32
    var film_filename:    String
    var filter_sigma:     Float32
    var filter_support_x: Float32
    var filter_support_y: Float32
    var filter_type:      Int32      # 0=gaussian 1=triangle 2=box
    var samples_per_pixel: Int32
    var camera_fov:       Float32
    var cam2w_raw:        InlineArray[Float32, 16]
    var max_depth:        Int32
    var scene_dir:        String
    var object_depth:     Int32

    def __init__(out self):
        # Identity CTM
        self.ctm = InlineArray[Float32, 16](fill=Float32(0))
        self.ctm[0] = Float32(1); self.ctm[5] = Float32(1)
        self.ctm[10] = Float32(1); self.ctm[15] = Float32(1)
        self.ctm_stack = List[InlineArray[Float32, 16]]()

        self.cur_attr  = AttributeState(Int32(-1), False,
                             RGB(Float32(0),Float32(0),Float32(0)),
                             Int32(-1), Int32(-1), False)
        self.attr_stack = List[AttributeState]()

        self.named_materials = List[NamedMaterial]()
        self.meshes          = List[MeshAccum]()
        self.hair            = None

        self.distant_dirs = List[Float32]()
        self.distant_rgbs = List[Float32]()
        self.point_pos    = List[Float32]()
        self.point_rgbs   = List[Float32]()
        self.inf_tex_idx  = List[Int32]()
        self.inf_rgb      = List[Float32]()
        self.inf_ctm      = List[Float32]()

        self.spheres_cx  = List[Float32]()
        self.spheres_cy  = List[Float32]()
        self.spheres_cz  = List[Float32]()
        self.spheres_r   = List[Float32]()
        self.spheres_mat = List[Int32]()
        self.spheres_al  = List[Bool]()
        self.spheres_rgb = List[RGB]()
        self.spheres_inside_med  = List[Int32]()
        self.spheres_outside_med = List[Int32]()

        self.med_names = List[String]()
        self.med_sa    = List[Float32]()
        self.med_ss    = List[Float32]()
        self.med_g     = List[Float32]()

        self.miface_inside  = List[Int32]()
        self.miface_outside = List[Int32]()
        self.miface_mat     = List[Int32]()

        self.tex_names = List[String]()
        self.tex_files = List[String]()

        self.film_w           = Int32(512)
        self.film_h           = Int32(512)
        self.film_iso         = Float32(100)
        self.film_max_comp    = Float32(0)
        self.film_filename    = String("gonzales.exr")
        self.filter_sigma     = Float32(0.5)
        self.filter_support_x = Float32(1.5)
        self.filter_support_y = Float32(1.5)
        self.filter_type      = Int32(0)
        self.samples_per_pixel = Int32(1)
        self.camera_fov       = Float32(30)
        self.cam2w_raw        = InlineArray[Float32, 16](fill=Float32(0))
        self.cam2w_raw[0] = Float32(1); self.cam2w_raw[5] = Float32(1)
        self.cam2w_raw[10] = Float32(1); self.cam2w_raw[15] = Float32(1)
        self.max_depth  = Int32(5)
        self.scene_dir  = String("")
        self.object_depth = Int32(0)


# ── Utility functions ─────────────────────────────────────────────────────────

def psc_strcmp(a: UnsafePointer[UInt8, MutAnyOrigin], b: UnsafePointer[UInt8, MutAnyOrigin]) -> Int32:
    var i = 0
    while True:
        var ca = Int32(a[i]); var cb = Int32(b[i])
        if ca != cb: return ca - cb
        if ca == Int32(0): return Int32(0)
        i += 1
    # unreachable

def _psc_streq(a: UnsafePointer[UInt8, MutAnyOrigin], b: StringLiteral) -> Bool:
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
    # unreachable — loop always returns via the ai==0 or ai!=bi branches above

def _psc_strncpy(dst: UnsafePointer[UInt8, MutAnyOrigin],
                src: UnsafePointer[UInt8, MutAnyOrigin], n: Int32):
    var i = Int32(0)
    while i < n - Int32(1) and src[Int(i)] != UInt8(0):
        dst[Int(i)] = src[Int(i)]
        i += 1
    dst[Int(i)] = UInt8(0)

def _psc_strncmp(a: UnsafePointer[UInt8, MutAnyOrigin], b: StringLiteral, n: Int) -> Int:
    """Compare first n bytes of a against literal b. Returns 0 if equal."""
    for i in range(n):
        var ca = Int(a[i])
        var cb = Int(b.unsafe_ptr()[i])
        if ca != cb:
            return ca - cb
        if ca == 0:
            return 0
    return 0

def _psc_identity(m: UnsafePointer[Float32, MutAnyOrigin]):
    for i in range(16):
        m[i] = Float32(0)
    m[0] = Float32(1)
    m[5] = Float32(1)
    m[10] = Float32(1)
    m[15] = Float32(1)

def _psc_matcopy(dst: UnsafePointer[Float32, MutAnyOrigin],
                src: UnsafePointer[Float32, MutAnyOrigin]):
    for i in range(16):
        dst[i] = src[i]

# ---- Transform keyword helpers (Translate / Scale / Rotate) ----
# All of these RIGHT-multiply the CTM: CTM_new = CTM_old × M
# CTM is stored column-major; identity has 1 at indices 0,5,10,15.

def _psc_ctm_concat(s: UnsafePointer[SceneParseState, MutAnyOrigin],
                   t: UnsafePointer[Float32, MutAnyOrigin]):
    """Compute s.ctm = s.ctm × t and store back."""
    var result = alloc[Float32](16)
    matrix_multiply(s[0].ctm.unsafe_ptr(), t, result)
    for i in range(16):
        s[0].ctm[i] = result[i]
    result.free()

def ctm_push(mut s: SceneParseState):
    s.ctm_stack.append(s.ctm)

def ctm_pop(mut s: SceneParseState):
    if len(s.ctm_stack) > 0:
        s.ctm = s.ctm_stack[len(s.ctm_stack) - 1]
        _ = s.ctm_stack.pop()

def scene_parse_state_new() -> SceneParseState:
    return SceneParseState()

def _psc_handle_translate(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                         s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    """Translate tx ty tz  →  CTM = CTM × T(tx,ty,tz)"""
    var v = alloc[Float32](3)
    v[0] = Float32(0); v[1] = Float32(0); v[2] = Float32(0)
    _ = scanner_scan_float(handle, v + 0)
    _ = scanner_scan_float(handle, v + 1)
    _ = scanner_scan_float(handle, v + 2)
    var t = alloc[Float32](16)
    _psc_identity(t)
    t[12] = v[0]; t[13] = v[1]; t[14] = v[2]   # col-major: col3 = (tx,ty,tz,1)
    _psc_ctm_concat(s, t)
    v.free(); t.free()

def _psc_handle_scale_kw(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                        s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    """Scale sx sy sz  →  CTM = CTM × S(sx,sy,sz)"""
    var v = alloc[Float32](3)
    v[0] = Float32(1); v[1] = Float32(1); v[2] = Float32(1)
    _ = scanner_scan_float(handle, v + 0)
    _ = scanner_scan_float(handle, v + 1)
    _ = scanner_scan_float(handle, v + 2)
    var t = alloc[Float32](16)
    _psc_identity(t)
    t[0] = v[0]; t[5] = v[1]; t[10] = v[2]     # col-major: diagonal
    _psc_ctm_concat(s, t)
    v.free(); t.free()

def _psc_handle_rotate(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                      s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    """Rotate angle ax ay az  →  CTM = CTM × R(angle, axis)"""
    from std.math import sin as _sin, cos as _cos, sqrt as _sqrt
    var rv = alloc[Float32](4)  # angle, ax, ay, az
    rv[0] = Float32(0); rv[1] = Float32(0); rv[2] = Float32(0); rv[3] = Float32(1)
    _ = scanner_scan_float(handle, rv + 0)
    _ = scanner_scan_float(handle, rv + 1)
    _ = scanner_scan_float(handle, rv + 2)
    _ = scanner_scan_float(handle, rv + 3)
    var angle = rv[0] * PI / Float32(180)
    var ax = rv[1]; var ay = rv[2]; var az = rv[3]
    var ln = _sqrt(ax*ax + ay*ay + az*az)
    if ln > Float32(1e-12): ax /= ln; ay /= ln; az /= ln
    var c = _cos(angle); var sv = _sin(angle); var mc = Float32(1) - c
    var t = alloc[Float32](16)
    # Column-major rotation matrix (standard Rodrigues)
    t[0]  = c + ax*ax*mc;       t[1]  = ay*ax*mc + az*sv;   t[2]  = az*ax*mc - ay*sv;   t[3]  = Float32(0)
    t[4]  = ax*ay*mc - az*sv;   t[5]  = c + ay*ay*mc;       t[6]  = az*ay*mc + ax*sv;   t[7]  = Float32(0)
    t[8]  = ax*az*mc + ay*sv;   t[9]  = ay*az*mc - ax*sv;   t[10] = c + az*az*mc;        t[11] = Float32(0)
    t[12] = Float32(0);         t[13] = Float32(0);          t[14] = Float32(0);          t[15] = Float32(1)
    _psc_ctm_concat(s, t)
    rv.free(); t.free()

def _psc_handle_lookat(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                      s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    """LookAt ex ey ez  lx ly lz  ux uy uz
    Constructs a world-to-camera view matrix and right-multiplies the CTM.
    Matches PBRT's LookAt semantics exactly."""
    from std.math import sqrt as _sqrt
    var v = alloc[Float32](9)
    for i in range(9): _ = scanner_scan_float(handle, v + i)
    # eye, look, up
    var ex = v[0]; var ey = v[1]; var ez = v[2]
    var lx = v[3]; var ly = v[4]; var lz = v[5]
    var ux = v[6]; var uy = v[7]; var uz = v[8]
    v.free()

    # dir = normalize(look - eye)
    var dx = lx - ex; var dy = ly - ey; var dz = lz - ez
    var dl = _sqrt(dx*dx + dy*dy + dz*dz)
    if dl > Float32(1e-12): dx /= dl; dy /= dl; dz /= dl

    # right = normalize(cross(normalize(up), dir))
    var ul = _sqrt(ux*ux + uy*uy + uz*uz)
    if ul > Float32(1e-12): ux /= ul; uy /= ul; uz /= ul
    # cross(up, dir)
    var rx = uy*dz - uz*dy
    var ry = uz*dx - ux*dz
    var rz = ux*dy - uy*dx
    var rl = _sqrt(rx*rx + ry*ry + rz*rz)
    if rl > Float32(1e-12): rx /= rl; ry /= rl; rz /= rl

    # newUp = cross(dir, right)
    var nx = dy*rz - dz*ry
    var ny = dz*rx - dx*rz
    var nz = dx*ry - dy*rx

    # PBRT LookAt world-to-camera matrix (row-major):
    # [ rx  ry  rz  -dot(r,eye) ]
    # [ nx  ny  nz  -dot(n,eye) ]
    # [ dx  dy  dz  -dot(d,eye) ]
    # [ 0   0   0    1          ]
    # Store in column-major for our CTM convention:
    var t = alloc[Float32](16)
    t[0]  = rx;  t[1]  = nx;  t[2]  = dx;  t[3]  = Float32(0)
    t[4]  = ry;  t[5]  = ny;  t[6]  = dy;  t[7]  = Float32(0)
    t[8]  = rz;  t[9]  = nz;  t[10] = dz;  t[11] = Float32(0)
    t[12] = -(rx*ex + ry*ey + rz*ez)
    t[13] = -(nx*ex + ny*ey + nz*ez)
    t[14] = -(dx*ex + dy*ey + dz*ez)
    t[15] = Float32(1)
    _psc_ctm_concat(s, t)
    t.free()

def _psc_row_to_col(col_out: UnsafePointer[Float32, MutAnyOrigin],
                   row_in:  UnsafePointer[Float32, MutAnyOrigin]):
    for row in range(4):
        for col in range(4):
            col_out[col * 4 + row] = row_in[row * 4 + col]

def _psc_type_is_float(t: UnsafePointer[UInt8, MutAnyOrigin]) -> Bool:
    var c = t[0]
    if c == UInt8(102): return True  # 'f' float
    if c == UInt8(114): return True  # 'r' rgb
    if c == UInt8(99):  return True  # 'c' color
    if c == UInt8(110): return True  # 'n' normal
    if c == UInt8(112): return True  # 'p' point/point2/point3
    if c == UInt8(118): return True  # 'v' vector3
    if c == UInt8(115) and t[1] == UInt8(112): return True  # "sp" spectrum
    # NOTE: "blackbody" is intentionally NOT listed here; it is 1 float (temperature),
    # not 3 floats, so _psc_scan_rgb must NOT be called for it.
    return False

def _psc_type_is_blackbody(t: UnsafePointer[UInt8, MutAnyOrigin]) -> Bool:
    return t[0] == UInt8(98) and t[1] == UInt8(108)  # 'b','l'

# Mitchell-Charity blackbody colour approximation (normalised, max=1).
# Reference: http://www.tannerhelland.com/4435/
@always_inline
def _psc_blackbody_to_rgb(temp: Float32, rgb: UnsafePointer[Float32, MutAnyOrigin]):
    from std.math import log as _log, pow as _pow
    var t100 = temp / Float32(100.0)
    var r: Float32; var g: Float32; var b: Float32
    # Red channel
    if temp <= Float32(6600):
        r = Float32(1.0)
    else:
        r = Float32(329.698727446) * _pow(t100 - Float32(60), Float32(-0.1332047592)) / Float32(255)
        r = max(Float32(0), min(Float32(1), r))
    # Green channel
    if temp <= Float32(6600):
        g = (Float32(99.4708025861) * _log(t100) - Float32(161.1195681661)) / Float32(255)
        g = max(Float32(0), min(Float32(1), g))
    else:
        g = Float32(288.1221695283) * _pow(t100 - Float32(60), Float32(-0.0755148492)) / Float32(255)
        g = max(Float32(0), min(Float32(1), g))
    # Blue channel
    if temp >= Float32(6600):
        b = Float32(1.0)
    elif temp <= Float32(1900):
        b = Float32(0.0)
    else:
        b = (Float32(138.5177312231) * _log(t100 - Float32(10)) - Float32(305.0447927307)) / Float32(255)
        b = max(Float32(0), min(Float32(1), b))
    # Normalise so max component = 1 (scale is applied separately)
    var mx = max(r, max(g, b))
    if mx > Float32(0):
        r /= mx; g /= mx; b /= mx
    rgb[0] = r; rgb[1] = g; rgb[2] = b

def _psc_type_is_int(t: UnsafePointer[UInt8, MutAnyOrigin]) -> Bool:
    return t[0] == UInt8(105)  # 'i' integer

def _psc_type_is_str(t: UnsafePointer[UInt8, MutAnyOrigin]) -> Bool:
    if t[0] == UInt8(116): return True  # 't' texture
    if t[0] == UInt8(115) and t[1] == UInt8(116): return True  # "string"
    return False

def _psc_skip_value(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                   type_buf: UnsafePointer[UInt8, MutAnyOrigin],
                   is_array: Int32):
    var tmp_f = alloc[Float32](65536)
    var tmp_i = alloc[Int32](16384)
    var tmp_s = alloc[UInt8](1024)
    if _psc_type_is_float(type_buf):
        if is_array:
            _ = scanner_scan_floats(handle, tmp_f, 65536)
        else:
            _ = scanner_scan_float(handle, tmp_f)
    elif _psc_type_is_int(type_buf):
        if is_array:
            _ = scanner_scan_ints(handle, tmp_i, 16384)
        else:
            _ = scanner_scan_int(handle, tmp_i)
    elif _psc_type_is_str(type_buf):
        _ = scanner_parse_quoted_string(handle, tmp_s, 1024)
        if is_array:
            while scanner_parse_quoted_string(handle, tmp_s, 1024) >= 0:
                pass
    else:
        var nl_buf = alloc[UInt8](1)
        nl_buf[0] = UInt8(10)
        _ = scanner_scan_token(handle, nl_buf, 1, tmp_s, 1024)
        nl_buf.free()
    tmp_f.free()
    tmp_i.free()
    tmp_s.free()

def _psc_skip_params(handle: UnsafePointer[PbrtScanner, MutAnyOrigin]):
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        _psc_skip_value(handle, type_buf, is_array)
        if is_array:
            _ = scanner_scan_char(handle, UInt8(93))  # ']'
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    type_buf.free()
    name_buf.free()

def _psc_skip_line(handle: UnsafePointer[PbrtScanner, MutAnyOrigin]):
    var nl_buf = alloc[UInt8](1)
    nl_buf[0] = UInt8(10)
    var buf = alloc[UInt8](4096)
    _ = scanner_scan_token(handle, nl_buf, 1, buf, 4096)
    nl_buf.free()
    buf.free()

def _psc_scan_one_float(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                       is_array: Int32) -> Float32:
    var vp = alloc[Float32](1)
    vp[0] = Float32(0)
    _ = scanner_scan_float(handle, vp)
    var v = vp[0]
    vp.free()
    if is_array:
        _ = scanner_scan_char(handle, UInt8(93))  # ']'
    return v

def _psc_scan_one_int(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                     is_array: Int32) -> Int32:
    var ip = alloc[Int32](1)
    ip[0] = Int32(0)
    _ = scanner_scan_int(handle, ip)
    var v = ip[0]
    ip.free()
    if is_array:
        _ = scanner_scan_char(handle, UInt8(93))  # ']'
    return v

def _psc_scan_one_str(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                     dst: UnsafePointer[UInt8, MutAnyOrigin], dst_max: Int32,
                     is_array: Int32):
    _ = scanner_parse_quoted_string(handle, dst, dst_max)
    if is_array:
        _ = scanner_scan_char(handle, UInt8(93))  # ']'

def _psc_scan_rgb(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                 rgb: UnsafePointer[Float32, MutAnyOrigin],
                 is_array: Int32):
    _ = scanner_scan_float(handle, rgb + 0)
    _ = scanner_scan_float(handle, rgb + 1)
    _ = scanner_scan_float(handle, rgb + 2)
    if is_array:
        _ = scanner_scan_char(handle, UInt8(93))  # ']'

def _psc_handle_integrator(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                          s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "maxdepth") and _psc_type_is_int(type_buf):
            s[0].max_depth = _psc_scan_one_int(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

def _psc_handle_sampler(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                       s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if (_psc_streq(name_buf, "pixelsamples") or _psc_streq(name_buf, "samples")) and _psc_type_is_int(type_buf):
            s[0].samples_per_pixel = _psc_scan_one_int(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

def _psc_handle_filter(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                      s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    if _psc_streq(sbuf, "triangle") or _psc_streq(sbuf, "tent"):
        s[0].filter_type = Int32(1)
    elif _psc_streq(sbuf, "box"):
        s[0].filter_type = Int32(2)
    else:
        s[0].filter_type = Int32(0)  # gaussian (default)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
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
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

def _psc_handle_film(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                    s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "xresolution") and _psc_type_is_int(type_buf):
            s[0].film_w = _psc_scan_one_int(handle, is_array)
        elif _psc_streq(name_buf, "yresolution") and _psc_type_is_int(type_buf):
            s[0].film_h = _psc_scan_one_int(handle, is_array)
        elif _psc_streq(name_buf, "filename") and _psc_type_is_str(type_buf):
            var fn_tmp = alloc[UInt8](PSC_FILE_MAX)
            _psc_scan_one_str(handle, fn_tmp, Int32(PSC_FILE_MAX), is_array)
            s[0].film_filename = String(unsafe_from_utf8_ptr=fn_tmp.as_immutable())
            fn_tmp.free()
        elif _psc_streq(name_buf, "iso") and _psc_type_is_float(type_buf):
            s[0].film_iso = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "maxcomponentvalue") and _psc_type_is_float(type_buf):
            s[0].film_max_comp = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

def _psc_handle_camera(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                      s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    # Copy current CTM into cam2w_raw
    for i in range(16): s[0].cam2w_raw[i] = s[0].ctm[i]
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "fov") and _psc_type_is_float(type_buf):
            s[0].camera_fov = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    sbuf.free(); type_buf.free(); name_buf.free()

def _psc_handle_transform(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                         s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    _ = scanner_scan_char(handle, UInt8(91))  # '['
    var tmp = alloc[Float32](1)
    for i in range(16):
        _ = scanner_scan_float(handle, tmp)
        s[0].ctm[i] = tmp[0]
    tmp.free()
    _ = scanner_scan_char(handle, UInt8(93))  # ']'

def _psc_handle_world_begin(s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    for i in range(16): s[0].ctm[i] = Float32(0)
    s[0].ctm[0] = Float32(1); s[0].ctm[5] = Float32(1)
    s[0].ctm[10] = Float32(1); s[0].ctm[15] = Float32(1)
    s[0].ctm_stack.clear()

def _psc_handle_make_named_material(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                                   s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var mat_name = alloc[UInt8](PSC_NAME_MAX)
    _ = scanner_parse_quoted_string(handle, mat_name, PSC_NAME_MAX)

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
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "type") and _psc_type_is_str(type_buf):
            _ = scanner_parse_quoted_string(handle, str_val, 64)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
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
        elif (_psc_streq(name_buf, "eta") or _psc_streq(name_buf, "k")) and type_buf[0] == UInt8(114):  # 'r' rgb eta/k for conductor
            if _psc_streq(name_buf, "eta"):
                _psc_scan_rgb(handle, metal_eta, is_array)
            else:
                _psc_scan_rgb(handle, metal_k, is_array)
            has_spectral_conductor = True
        elif (_psc_streq(name_buf, "eta") or _psc_streq(name_buf, "intIOR")) and type_buf[0] == UInt8(102):  # 'f' float IOR for dielectric
            var tmp = alloc[Float32](1)
            _ = scanner_scan_float(handle, tmp)
            mat_ior = tmp[0]
            tmp.free()
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
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
            _ = scanner_parse_quoted_string(handle, mname, 64)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
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
            _ = scanner_parse_quoted_string(handle, str_val, 64)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
            # Look up str_val in tex_names
            var str_name = String(unsafe_from_utf8_ptr=str_val.as_immutable())
            for ti in range(len(s[0].tex_names)):
                if s[0].tex_names[ti] == str_name:
                    tex_idx_for_mat = Int32(ti)
                    break
        elif _psc_streq(name_buf, "L") and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, rgb, is_array)
        elif (_psc_streq(name_buf, "normalmap") or _psc_streq(name_buf, "bumpmap")) and type_buf[0] == UInt8(116):  # 't' = texture
            _ = scanner_parse_quoted_string(handle, str_val, 64)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
            var norm_name = String(unsafe_from_utf8_ptr=str_val.as_immutable())
            for ti in range(len(s[0].tex_names)):
                if s[0].tex_names[ti] == norm_name:
                    normal_tex_idx_for_mat = Int32(ti)
                    break
        elif _psc_streq(name_buf, "amount") and _psc_type_is_float(type_buf):
            mix_amount = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "materials") and _psc_type_is_str(type_buf):
            # Two quoted material names for mix
            var tmp1 = alloc[UInt8](PSC_NAME_MAX)
            var tmp2 = alloc[UInt8](PSC_NAME_MAX)
            _ = scanner_parse_quoted_string(handle, tmp1, PSC_NAME_MAX)
            _ = scanner_parse_quoted_string(handle, tmp2, PSC_NAME_MAX)
            _psc_strncpy(mix_name1, tmp1, PSC_NAME_MAX)
            _psc_strncpy(mix_name2, tmp2, PSC_NAME_MAX)
            tmp1.free(); tmp2.free()
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()

    # Store into named_materials List
    var nm = NamedMaterial(String(unsafe_from_utf8_ptr=mat_name.as_immutable()))
    # For named-spectrum conductors: compute Fresnel F0 per channel
    if has_spectral_conductor and mat_type == Int8(3):
        var f0r = ((metal_eta[0]-Float32(1.0))*(metal_eta[0]-Float32(1.0)) + metal_k[0]*metal_k[0]) / \
                  ((metal_eta[0]+Float32(1.0))*(metal_eta[0]+Float32(1.0)) + metal_k[0]*metal_k[0])
        var f0g = ((metal_eta[1]-Float32(1.0))*(metal_eta[1]-Float32(1.0)) + metal_k[1]*metal_k[1]) / \
                  ((metal_eta[1]+Float32(1.0))*(metal_eta[1]+Float32(1.0)) + metal_k[1]*metal_k[1])
        var f0b = ((metal_eta[2]-Float32(1.0))*(metal_eta[2]-Float32(1.0)) + metal_k[2]*metal_k[2]) / \
                  ((metal_eta[2]+Float32(1.0))*(metal_eta[2]+Float32(1.0)) + metal_k[2]*metal_k[2])
        nm.albedo = RGB(f0r, f0g, f0b)
    elif mat_type == Int8(11):
        var ce = eumelanin; var cp = pheomelanin
        var scale2 = Float32(2.0)
        nm.albedo = RGB(
            exp(-(ce * Float32(0.419) + cp * Float32(0.187)) * scale2),
            exp(-(ce * Float32(0.697) + cp * Float32(0.400)) * scale2),
            exp(-(ce * Float32(1.370) + cp * Float32(1.050)) * scale2),
        )
    else:
        nm.albedo = RGB(rgb[0], rgb[1], rgb[2])
    nm.transmittance  = RGB(trans_rgb[0], trans_rgb[1], trans_rgb[2])
    nm.kind           = mat_type
    nm.ior            = mat_ior
    nm.roughness_u    = mat_roughU
    nm.roughness_v    = mat_roughV
    nm.tex_idx        = tex_idx_for_mat
    nm.normal_tex_idx = normal_tex_idx_for_mat
    nm.mix_name1      = String(unsafe_from_utf8_ptr=mix_name1.as_immutable())
    nm.mix_name2      = String(unsafe_from_utf8_ptr=mix_name2.as_immutable())
    nm.mix_amount     = mix_amount
    s[0].named_materials.append(nm^)

    mat_name.free(); type_buf.free(); name_buf.free(); str_val.free(); rgb.free()
    trans_rgb.free(); metal_eta.free(); metal_k.free()
    mix_name1.free(); mix_name2.free()

def _psc_handle_named_material(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                               s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var mat_name = alloc[UInt8](PSC_NAME_MAX)
    _ = scanner_parse_quoted_string(handle, mat_name, PSC_NAME_MAX)
    s[0].cur_attr.mat_idx = Int32(-1)
    var name_str = String(unsafe_from_utf8_ptr=mat_name.as_immutable())
    for i in range(len(s[0].named_materials)):
        if s[0].named_materials[i].name == name_str:
            s[0].cur_attr.mat_idx = Int32(i)
            break
    mat_name.free()

def _psc_handle_attribute_begin(s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    ctm_push(s[0])
    s[0].attr_stack.append(s[0].cur_attr)

def _psc_handle_attribute_end(s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    ctm_pop(s[0])
    if len(s[0].attr_stack) > 0:
        s[0].cur_attr = s[0].attr_stack[len(s[0].attr_stack) - 1]
        _ = s[0].attr_stack.pop()

def _psc_handle_area_light_source(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                                 s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var sbuf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, sbuf, 64)
    s[0].cur_attr.is_alight = True
    s[0].cur_attr.al_rgb = RGB(Float32(1), Float32(1), Float32(1))
    var rgb = alloc[Float32](3)
    rgb[0] = Float32(1); rgb[1] = Float32(1); rgb[2] = Float32(1)
    var scale = Float32(1.0)  # accumulated separately; applied after loop
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "L") and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, rgb, is_array)
        elif _psc_streq(name_buf, "L") and _psc_type_is_blackbody(type_buf):
            var temp = _psc_scan_one_float(handle, is_array)
            _psc_blackbody_to_rgb(temp, rgb)
        elif _psc_streq(name_buf, "scale") and _psc_type_is_float(type_buf):
            scale *= _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    rgb[0] *= scale; rgb[1] *= scale; rgb[2] *= scale
    s[0].cur_attr.al_rgb = RGB(rgb[0], rgb[1], rgb[2])
    sbuf.free(); type_buf.free(); name_buf.free(); rgb.free()

def store_mesh(
    s:       UnsafePointer[SceneParseState, MutAnyOrigin],
    tmp_f:   UnsafePointer[Float32, MutAnyOrigin],
    tmp_i:   UnsafePointer[Int32, MutAnyOrigin],
    n_verts: Int32,
    n_tris:  Int32,
):
    var raw_pts = alloc[Float32](Int(n_verts) * 4)
    for v in range(Int(n_verts)):
        raw_pts[v*4+0] = tmp_f[v*3+0]
        raw_pts[v*4+1] = tmp_f[v*3+1]
        raw_pts[v*4+2] = tmp_f[v*3+2]
        raw_pts[v*4+3] = Float32(1)
    var fin_pts = alloc[Float32](Int(n_verts) * 4)
    transform_points(s[0].ctm.unsafe_ptr(), raw_pts, n_verts, fin_pts)
    raw_pts.free()
    # Build MeshAccum with List fields
    var ma = MeshAccum(
        s[0].cur_attr.mat_idx,
        s[0].cur_attr.inside_medium,
        s[0].cur_attr.outside_medium,
    )
    ma.is_area_light = s[0].cur_attr.is_alight
    ma.al_rgb = s[0].cur_attr.al_rgb
    ma.points.reserve(Int(n_verts) * 4)
    for v in range(Int(n_verts) * 4):
        ma.points.append(fin_pts[v])
    fin_pts.free()
    ma.vert_idxs.reserve(Int(n_tris) * 3)
    ma.face_idxs.reserve(Int(n_tris))
    # PBRT ReverseOrientation flips the geometric (winding) normal. Reverse the
    # triangle winding by swapping the 2nd and 3rd vertex so cross(p1-p0,p2-p0)
    # negates; shading-normal interpolation stays consistent (same indices).
    var rev = s[0].cur_attr.reverse_orient
    for t in range(Int(n_tris)):
        ma.vert_idxs.append(Int64(tmp_i[t*3+0]))
        if rev:
            ma.vert_idxs.append(Int64(tmp_i[t*3+2]))
            ma.vert_idxs.append(Int64(tmp_i[t*3+1]))
        else:
            ma.vert_idxs.append(Int64(tmp_i[t*3+1]))
            ma.vert_idxs.append(Int64(tmp_i[t*3+2]))
        ma.face_idxs.append(Int64(3))
    s[0].meshes.append(ma^)

# ── Hair curve helpers ────────────────────────────────────────────────────────

def ensure_hair_buffer(s: UnsafePointer[SceneParseState, MutAnyOrigin]) -> Bool:
    """Lazily init the hair MeshAccum. Returns False if at mesh limit."""
    if s[0].hair:
        return True
    s[0].hair = MeshAccum(
        s[0].cur_attr.mat_idx,
        s[0].cur_attr.inside_medium,
        s[0].cur_attr.outside_medium,
    )
    return True

def bspline3_eval(cp: UnsafePointer[Float32, MutAnyOrigin], n_cp: Int,
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

def hair_add_quad(s: UnsafePointer[SceneParseState, MutAnyOrigin],
        p0x: Float32, p0y: Float32, p0z: Float32,
        p1x: Float32, p1y: Float32, p1z: Float32,
        ux: Float32, uy: Float32, uz: Float32, hw: Float32):
    """Append one rectangular quad (2 triangles) to the hair accumulator."""
    ref h = s[0].hair.value()
    var bv = len(h.points) // 4
    # 4 vertices (xyz + pad)
    h.points.append(p0x+ux*hw); h.points.append(p0y+uy*hw)
    h.points.append(p0z+uz*hw); h.points.append(Float32(1))
    h.points.append(p0x-ux*hw); h.points.append(p0y-uy*hw)
    h.points.append(p0z-uz*hw); h.points.append(Float32(1))
    h.points.append(p1x+ux*hw); h.points.append(p1y+uy*hw)
    h.points.append(p1z+uz*hw); h.points.append(Float32(1))
    h.points.append(p1x-ux*hw); h.points.append(p1y-uy*hw)
    h.points.append(p1z-uz*hw); h.points.append(Float32(1))
    # 2 triangles
    h.vert_idxs.append(Int64(bv));   h.vert_idxs.append(Int64(bv+1)); h.vert_idxs.append(Int64(bv+2))
    h.face_idxs.append(Int64(3))
    h.vert_idxs.append(Int64(bv+1)); h.vert_idxs.append(Int64(bv+3)); h.vert_idxs.append(Int64(bv+2))
    h.face_idxs.append(Int64(3))

def handle_curve_shape(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                            s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    """Parse Shape "curve" B-spline strand and append cross-ribbon triangles."""
    var MAX_CP = 512
    var cp_buf = alloc[Float32](MAX_CP * 3)
    var n_cp = Int32(0)
    var width = Float32(0.002)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_type_is_float(type_buf) and _psc_streq(name_buf, "P"):
            var n_f = scanner_scan_floats(handle, cp_buf, Int32(MAX_CP * 3))
            n_cp = n_f / Int32(3)
            if is_array: _ = scanner_scan_char(handle, UInt8(93))
        elif _psc_type_is_float(type_buf) and (_psc_streq(name_buf, "width") or _psc_streq(name_buf, "width0")):
            width = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array: _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    type_buf.free(); name_buf.free(); ia.free()
    if n_cp < Int32(4):
        cp_buf.free(); return
    if not ensure_hair_buffer(s):
        cp_buf.free(); return
    # Evaluate HAIR_EVAL_N uniformly-spaced points along the B-spline
    var eval_pts = alloc[Float32](HAIR_EVAL_N * 3)
    for step in range(HAIR_EVAL_N):
        var u = Float32(step) / Float32(HAIR_EVAL_N - 1)
        bspline3_eval(cp_buf, Int(n_cp), u, eval_pts + step * 3)
    cp_buf.free()
    # Apply current CTM to evaluation points
    var raw4 = alloc[Float32](HAIR_EVAL_N * 4)
    var xfm4 = alloc[Float32](HAIR_EVAL_N * 4)
    for step in range(HAIR_EVAL_N):
        raw4[step*4+0] = eval_pts[step*3+0]
        raw4[step*4+1] = eval_pts[step*3+1]
        raw4[step*4+2] = eval_pts[step*3+2]
        raw4[step*4+3] = Float32(1)
    transform_points(s[0].ctm.unsafe_ptr(), raw4, Int32(HAIR_EVAL_N), xfm4)
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
        hair_add_quad(s, p0x,p0y,p0z, p1x,p1y,p1z, ux,uy,uz, hw)
        hair_add_quad(s, p0x,p0y,p0z, p1x,p1y,p1z, vx,vy,vz, hw)
    xfm4.free(); eval_pts.free()

def flush_hair(s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    """Flush all accumulated hair strands into the meshes List."""
    if not s[0].hair:
        return
    var h = s[0].hair.take()
    if len(h.face_idxs) == 0:
        return
    s[0].meshes.append(h^)


def handle_named_medium(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                                  s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    """Parse MakeNamedMedium \"name\" [params].
    Supports type = \"homogeneous\" with rgb sigma_a, rgb sigma_s, float g, float scale.
    """
    var name_buf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, name_buf, 64)
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
        var ok = scanner_parse_param_header(handle, type_buf, Int32(64), val_buf, Int32(64), ia)
        if ok == Int32(0):
            break
        var is_arr = ia[0]
        if _psc_streq(val_buf, "type"):
            var tmp = alloc[UInt8](64)
            _ = scanner_parse_quoted_string(handle, tmp, 64)
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
                _ = scanner_scan_char(handle, UInt8(93))  # ']' 
    if is_hom:
        var name_str = String(unsafe_from_utf8_ptr=name_buf.as_immutable())
        s[0].med_names.append(name_str)
        s[0].med_sa.append(sa[0] * scale)
        s[0].med_sa.append(sa[1] * scale)
        s[0].med_sa.append(sa[2] * scale)
        s[0].med_ss.append(ss[0] * scale)
        s[0].med_ss.append(ss[1] * scale)
        s[0].med_ss.append(ss[2] * scale)
        s[0].med_g.append(g_val)
    name_buf.free(); sa.free(); ss.free(); type_buf.free(); val_buf.free()


def lookup_medium(s: UnsafePointer[SceneParseState, MutAnyOrigin],
                  name: UnsafePointer[UInt8, MutAnyOrigin]) -> Int32:
    """Return medium index for name, or -1 for empty string / not found."""
    if name[0] == UInt8(0):
        return Int32(-1)
    var name_str = String(unsafe_from_utf8_ptr=name.as_immutable())
    for i in range(len(s[0].med_names)):
        if s[0].med_names[i] == name_str:
            return Int32(i)
    return Int32(-1)


def handle_medium_interface(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                            s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    """Parse MediumInterface \"inside_name\" \"outside_name\" and bind to cur_attr."""
    var inside_buf  = alloc[UInt8](64)
    var outside_buf = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, inside_buf, 64)
    _ = scanner_parse_quoted_string(handle, outside_buf, 64)
    s[0].cur_attr.inside_medium  = lookup_medium(s, inside_buf)
    s[0].cur_attr.outside_medium = lookup_medium(s, outside_buf)
    inside_buf.free(); outside_buf.free()

def handle_sphere_shape(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                             s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    """Parse Shape "sphere" and store as analytical Sphere_C.
    The sphere center is the CTM translation column; radius is the parsed float.
    """
    var radius = Float32(1.0)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_type_is_float(type_buf) and _psc_streq(name_buf, "radius"):
            radius = _psc_scan_one_float(handle, is_array)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    type_buf.free(); name_buf.free()

    var n = Int(Int32(len(s[0].spheres_cx)))
    # World-space center = CTM * (0,0,0,1) = translation column (indices 12,13,14)
    var cx = s[0].ctm[12]
    var cy = s[0].ctm[13]
    var cz = s[0].ctm[14]
    # Scale radius by the uniform scale of the CTM (length of first column vector)
    var sx = sqrt(s[0].ctm[0]*s[0].ctm[0] + s[0].ctm[1]*s[0].ctm[1] + s[0].ctm[2]*s[0].ctm[2])
    if sx < Float32(1e-6): sx = Float32(1.0)
    radius *= sx
    s[0].spheres_cx.append(cx)
    s[0].spheres_cy.append(cy)
    s[0].spheres_cz.append(cz)
    s[0].spheres_r.append(radius)
    s[0].spheres_mat.append(s[0].cur_attr.mat_idx)
    s[0].spheres_inside_med.append(s[0].cur_attr.inside_medium)
    s[0].spheres_outside_med.append(s[0].cur_attr.outside_medium)
    s[0].spheres_al.append(s[0].cur_attr.is_alight)
    s[0].spheres_rgb.append(s[0].cur_attr.al_rgb)

def handle_shape(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                     s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var shape_type = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, shape_type, 64)

    var is_tri = _psc_streq(shape_type, "trianglemesh")
    var is_ply = _psc_streq(shape_type, "plymesh")
    var is_curve = _psc_streq(shape_type, "curve")
    var is_sphere = _psc_streq(shape_type, "sphere")
    shape_type.free()

    if is_curve:
        handle_curve_shape(handle, s)
        return

    if is_sphere:
        handle_sphere_shape(handle, s)
        return

    if not is_tri and not is_ply:
        _psc_skip_params(handle)
        return

    if is_ply:
        var ply_filename = alloc[UInt8](PSC_FILE_MAX)
        ply_filename[0] = UInt8(0)
        var type_buf = alloc[UInt8](64)
        var name_buf = alloc[UInt8](128)
        var ia = alloc[Int32](1)
        ia[0] = Int32(0)
        var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
        while found != 0:
            var is_array = ia[0]
            if _psc_streq(name_buf, "filename") and _psc_type_is_str(type_buf):
                _ = scanner_parse_quoted_string(handle, ply_filename, PSC_FILE_MAX)
                if is_array:
                    _ = scanner_scan_char(handle, UInt8(93))
            else:
                _psc_skip_value(handle, type_buf, is_array)
                if is_array:
                    _ = scanner_scan_char(handle, UInt8(93))
            ia[0] = Int32(0)
            found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
        ia.free()
        type_buf.free(); name_buf.free()

        var full_path = alloc[UInt8](PSC_FILE_MAX * 2)
        var dir_len = s[0].scene_dir.byte_length()
        for ki in range(dir_len):
            full_path[ki] = s[0].scene_dir.unsafe_ptr()[ki]
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
        var ply_nrm     = alloc[UnsafePointer[Float32, MutAnyOrigin]](1)
        var ply_has_nrm = alloc[Int32](1)
        ply_uvs[0] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
        ply_has_uvs[0] = Int32(0)
        ply_nrm[0] = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
        ply_has_nrm[0] = Int32(0)
        var ok = load_ply(full_path, ply_pts, ply_nv, ply_idx, ply_nt, ply_uvs, ply_has_uvs, ply_nrm, ply_has_nrm)
        if ok == 0:
            print("PLY load FAILED:", String(unsafe_from_utf8_ptr=full_path.as_immutable()))
            full_path.free()
            ply_pts.free(); ply_nv.free(); ply_idx.free(); ply_nt.free()
            ply_uvs.free(); ply_has_uvs.free(); ply_nrm.free(); ply_has_nrm.free()
            return
        full_path.free()
        var nv = ply_nv[0]
        var nt = ply_nt[0]
        if nv <= 0 or nt <= 0:
            ply_pts[0].free(); ply_idx[0].free()
            if ply_has_uvs[0] != 0:
                ply_uvs[0].free()
            if ply_has_nrm[0] != 0:
                ply_nrm[0].free()
            ply_pts.free(); ply_nv.free(); ply_idx.free(); ply_nt.free()
            ply_uvs.free(); ply_has_uvs.free(); ply_nrm.free(); ply_has_nrm.free()
            return
        var tmp_f2 = ply_pts[0]
        var tmp_i2 = ply_idx[0]
        store_mesh(s, tmp_f2, tmp_i2, nv, nt)
        # Store UVs into last mesh's uvs List
        if ply_has_uvs[0] != 0:
            var uv_ptr = ply_uvs[0]
            var n_uv_floats = Int(nv) * 2
            for uvi in range(n_uv_floats):
                s[0].meshes[len(s[0].meshes) - 1].uvs.append(uv_ptr[uvi])
            uv_ptr.free()
        # Store shading normals (transformed to world space by inverse-transpose CTM)
        if ply_has_nrm[0] != 0:
            var nrm_ptr = ply_nrm[0]
            var ctm_inv = alloc[Float32](16)
            _ = matrix_invert(s[0].ctm.unsafe_ptr(), ctm_inv)
            var nrm_world = alloc[Float32](Int(nv) * 3)
            transform_normals(ctm_inv, nrm_ptr, nv, nrm_world)
            ref last_mesh = s[0].meshes[len(s[0].meshes) - 1]
            last_mesh.normals.reserve(Int(nv) * 3)
            for ni in range(Int(nv)):
                # Renormalize after transform
                var nx = nrm_world[ni*3+0]; var ny = nrm_world[ni*3+1]; var nz = nrm_world[ni*3+2]
                var nlen = sqrt(nx*nx + ny*ny + nz*nz)
                if nlen > Float32(1e-12):
                    var inv = Float32(1.0) / nlen
                    nx *= inv; ny *= inv; nz *= inv
                last_mesh.normals.append(nx)
                last_mesh.normals.append(ny)
                last_mesh.normals.append(nz)
            nrm_world.free(); ctm_inv.free(); nrm_ptr.free()
        tmp_f2.free(); tmp_i2.free()
        ply_pts.free(); ply_nv.free(); ply_idx.free(); ply_nt.free()
        ply_uvs.free(); ply_has_uvs.free(); ply_nrm.free(); ply_has_nrm.free()
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
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        var is_P  = (_psc_streq(name_buf, "P") and _psc_type_is_float(type_buf))
        var is_I  = (_psc_streq(name_buf, "indices") and _psc_type_is_int(type_buf))
        var is_UV = ((_psc_streq(name_buf, "uv") or _psc_streq(name_buf, "st")) and _psc_type_is_float(type_buf))

        if is_P:
            if is_array:
                n_pts = scanner_scan_floats(handle, tmp_f, 65536)
                _ = scanner_scan_char(handle, UInt8(93))  # ']'
            else:
                _ = scanner_scan_float(handle, tmp_f)
                n_pts = Int32(3)
        elif is_I:
            if is_array:
                n_idx = scanner_scan_ints(handle, tmp_i, 16384)
                _ = scanner_scan_char(handle, UInt8(93))  # ']'
            else:
                _ = scanner_scan_int(handle, tmp_i)
                n_idx = Int32(1)
        elif is_UV:
            if is_array:
                n_uv = scanner_scan_floats(handle, tmp_uv, 65536)
                _ = scanner_scan_char(handle, UInt8(93))  # ']'
            else:
                _ = scanner_scan_float(handle, tmp_uv)
                n_uv = Int32(2)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    type_buf.free(); name_buf.free()

    var n_verts = n_pts / Int32(3)
    var n_tris  = n_idx / Int32(3)

    if n_verts <= 0 or n_tris <= 0:
        tmp_f.free(); tmp_i.free(); tmp_uv.free()
        return

    store_mesh(s, tmp_f, tmp_i, n_verts, n_tris)
    # Store UVs for inline trianglemesh
    if n_uv >= n_verts * Int32(2):
        for ui in range(Int(n_verts) * 2):
            s[0].meshes[len(s[0].meshes) - 1].uvs.append(tmp_uv[ui])
    tmp_f.free(); tmp_i.free(); tmp_uv.free()

def handle_texture(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                       s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    # Texture <name> <type> <class> [params]
    var tex_name = alloc[UInt8](PSC_NAME_MAX)
    _ = scanner_parse_quoted_string(handle, tex_name, PSC_NAME_MAX)
    var tex_type = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, tex_type, 64)
    var tex_class = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, tex_class, 64)
    if not _psc_streq(tex_class, "imagemap"):
        tex_name.free(); tex_type.free(); tex_class.free()
        _psc_skip_params(handle)
        return

    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var str_val  = alloc[UInt8](PSC_FILE_MAX * 2)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if _psc_streq(name_buf, "filename") and _psc_type_is_str(type_buf):
            _ = scanner_parse_quoted_string(handle, str_val, PSC_FILE_MAX * 2)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
            var name_str = String(unsafe_from_utf8_ptr=tex_name.as_immutable())
            var file_str = s[0].scene_dir + String(unsafe_from_utf8_ptr=str_val.as_immutable())
            s[0].tex_names.append(name_str)
            s[0].tex_files.append(file_str)
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()
    tex_name.free(); tex_type.free(); tex_class.free()
    type_buf.free(); name_buf.free(); str_val.free()

def handle_light_source(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                             s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var ltype = alloc[UInt8](64)
    _ = scanner_parse_quoted_string(handle, ltype, 64)
    var type_buf = alloc[UInt8](64)
    var name_buf = alloc[UInt8](128)
    var str_val  = alloc[UInt8](PSC_FILE_MAX * 2)
    str_val[0] = UInt8(0)  # must init: alloc doesn't zero-fill
    var rgb      = alloc[Float32](3)
    var xyz      = alloc[Float32](3)
    rgb[0] = Float32(1); rgb[1] = Float32(1); rgb[2] = Float32(1)
    xyz[0] = Float32(0); xyz[1] = Float32(0); xyz[2] = Float32(1000)  # default: from above
    var scale    = Float32(1.0)
    var ia = alloc[Int32](1)
    ia[0] = Int32(0)
    var found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    while found != 0:
        var is_array = ia[0]
        if (_psc_streq(name_buf, "L") or _psc_streq(name_buf, "I")) and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, rgb, is_array)
        elif (_psc_streq(name_buf, "L") or _psc_streq(name_buf, "I")) and _psc_type_is_blackbody(type_buf):
            var temp = _psc_scan_one_float(handle, is_array)
            _psc_blackbody_to_rgb(temp, rgb)
        elif _psc_streq(name_buf, "scale") and _psc_type_is_float(type_buf):
            scale = _psc_scan_one_float(handle, is_array)
        elif _psc_streq(name_buf, "from") and _psc_type_is_float(type_buf):
            _psc_scan_rgb(handle, xyz, is_array)   # reuse xyz for point-light position
        elif _psc_streq(name_buf, "filename") and _psc_type_is_str(type_buf):
            _ = scanner_parse_quoted_string(handle, str_val, PSC_FILE_MAX * 2)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        else:
            _psc_skip_value(handle, type_buf, is_array)
            if is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        ia[0] = Int32(0)
        found = scanner_parse_param_header(handle, type_buf, 64, name_buf, 128, ia)
    ia.free()

    if _psc_streq(ltype, "distant"):
        # direction = -from (from describes where light comes from)
        var dlen = sqrt(xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2])
        if dlen < Float32(0.0001): dlen = Float32(1.0)
        s[0].distant_dirs.append(-xyz[0] / dlen)
        s[0].distant_dirs.append(-xyz[1] / dlen)
        s[0].distant_dirs.append(-xyz[2] / dlen)
        s[0].distant_rgbs.append(rgb[0] * scale)
        s[0].distant_rgbs.append(rgb[1] * scale)
        s[0].distant_rgbs.append(rgb[2] * scale)
    elif _psc_streq(ltype, "point"):
        # Apply current CTM to position
        var raw = alloc[Float32](4)
        raw[0] = xyz[0]; raw[1] = xyz[1]; raw[2] = xyz[2]; raw[3] = Float32(1)
        var fin = alloc[Float32](4)
        transform_points(s[0].ctm.unsafe_ptr(), raw, Int32(1), fin)
        s[0].point_pos.append(fin[0])
        s[0].point_pos.append(fin[1])
        s[0].point_pos.append(fin[2])
        s[0].point_rgbs.append(rgb[0] * scale)
        s[0].point_rgbs.append(rgb[1] * scale)
        s[0].point_rgbs.append(rgb[2] * scale)
        raw.free(); fin.free()
    elif _psc_streq(ltype, "infinite"):
        if str_val[0] != UInt8(0):
            var file_str = s[0].scene_dir + String(unsafe_from_utf8_ptr=str_val.as_immutable())
            s[0].tex_names.append(String("__inf"))
            s[0].tex_files.append(file_str)
            s[0].inf_tex_idx.append(Int32(len(s[0].tex_names) - 1))
        else:
            s[0].inf_tex_idx.append(Int32(-1))
        s[0].inf_rgb.append(rgb[0] * scale)
        s[0].inf_rgb.append(rgb[1] * scale)
        s[0].inf_rgb.append(rgb[2] * scale)
        # Store the light's CTM for env-map direction transform
        for ci in range(16):
            s[0].inf_ctm.append(s[0].ctm[ci])

    ltype.free(); type_buf.free(); name_buf.free(); str_val.free(); rgb.free(); xyz.free()

def parse_scene_file(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
              s: UnsafePointer[SceneParseState, MutAnyOrigin]):
    var kw_buf = alloc[UInt8](256)
    var ws_delims = alloc[UInt8](4)
    ws_delims[0] = UInt8(32); ws_delims[1] = UInt8(9)
    ws_delims[2] = UInt8(10); ws_delims[3] = UInt8(13)

    while scanner_is_at_end(handle) == 0:
        var n = scanner_scan_token(handle, ws_delims, 4, kw_buf, 256)
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
        elif _psc_streq(kw_buf, "Translate"):
            _psc_handle_translate(handle, s)
        elif _psc_streq(kw_buf, "Scale"):
            _psc_handle_scale_kw(handle, s)
        elif _psc_streq(kw_buf, "Rotate"):
            _psc_handle_rotate(handle, s)
        elif _psc_streq(kw_buf, "LookAt"):
            _psc_handle_lookat(handle, s)
        elif _psc_streq(kw_buf, "WorldBegin"):
            _psc_handle_world_begin(s)
        elif _psc_streq(kw_buf, "WorldEnd"):
            break
        elif _psc_streq(kw_buf, "MakeNamedMaterial"):
            _psc_handle_make_named_material(handle, s)
        elif _psc_streq(kw_buf, "NamedMaterial"):
            _psc_handle_named_material(handle, s)
        elif _psc_streq(kw_buf, "ObjectBegin"):
            # Start of an object definition — skip shapes inside
            var obj_name = alloc[UInt8](PSC_NAME_MAX)
            _ = scanner_parse_quoted_string(handle, obj_name, PSC_NAME_MAX)
            obj_name.free()
            s[0].object_depth += 1
        elif _psc_streq(kw_buf, "ObjectEnd"):
            if s[0].object_depth > 0:
                s[0].object_depth -= 1
        elif _psc_streq(kw_buf, "ObjectInstance"):
            # TODO: implement proper object instancing
            var obj_name = alloc[UInt8](PSC_NAME_MAX)
            _ = scanner_parse_quoted_string(handle, obj_name, PSC_NAME_MAX)
            obj_name.free()
        elif _psc_streq(kw_buf, "Shape"):
            if s[0].object_depth == 0:
                handle_shape(handle, s)
            else:
                _psc_skip_params(handle)
        elif _psc_streq(kw_buf, "AttributeBegin"):
            _psc_handle_attribute_begin(s)
        elif _psc_streq(kw_buf, "AttributeEnd"):
            _psc_handle_attribute_end(s)
        elif _psc_streq(kw_buf, "TransformBegin"):
            ctm_push(s[0])
        elif _psc_streq(kw_buf, "TransformEnd"):
            ctm_pop(s[0])
        elif _psc_streq(kw_buf, "ReverseOrientation"):
            s[0].cur_attr.reverse_orient = not s[0].cur_attr.reverse_orient
        elif _psc_streq(kw_buf, "AreaLightSource"):
            if s[0].object_depth == 0:
                _psc_handle_area_light_source(handle, s)
            else:
                _psc_skip_params(handle)
        elif _psc_streq(kw_buf, "LightSource"):
            if s[0].object_depth == 0:
                handle_light_source(handle, s)
            else:
                _psc_skip_params(handle)
        elif _psc_streq(kw_buf, "Texture"):
            handle_texture(handle, s)
        elif _psc_streq(kw_buf, "Include") or _psc_streq(kw_buf, "Import"):
            # Read filename, resolve relative to scene_dir, parse sub-file recursively
            var inc_name = alloc[UInt8](PSC_FILE_MAX)
            _ = scanner_parse_quoted_string(handle, inc_name, PSC_FILE_MAX)
            var inc_path = alloc[UInt8](PSC_FILE_MAX * 2)
            var dlen = s[0].scene_dir.byte_length()
            for ki in range(dlen):
                inc_path[ki] = s[0].scene_dir.unsafe_ptr()[ki]
            var fi = 0
            while inc_name[fi] != UInt8(0) and dlen + fi < PSC_FILE_MAX * 2 - 1:
                inc_path[dlen + fi] = inc_name[fi]
                fi += 1
            inc_path[dlen + fi] = UInt8(0)
            var sub_handle = scanner_open(inc_path)
            if scanner_is_at_end(sub_handle) == 0:
                parse_scene_file(sub_handle, s)
            scanner_free(sub_handle)
            inc_name.free(); inc_path.free()
        elif _psc_streq(kw_buf, "Material"):
            # Anonymous inline material: treat like MakeNamedMaterial under a temp name "_mat"
            # then immediately set cur_mat_idx to it.
            _psc_handle_make_named_material(handle, s)
            # The name was parsed as part of make_named_material; find the last added
            s[0].cur_attr.mat_idx = Int32(len(s[0].named_materials)) - Int32(1)
        elif _psc_streq(kw_buf, "MakeNamedMedium"):
            handle_named_medium(handle, s)
        elif _psc_streq(kw_buf, "MediumInterface"):
            handle_medium_interface(handle, s)
        elif _psc_streq(kw_buf, "ConcatTransform"):
            # ConcatTransform [16 floats] — multiply CTM on right
            _ = scanner_scan_char(handle, UInt8(91))  # '['
            var tmp = alloc[Float32](16)
            for i in range(16):
                _ = scanner_scan_float(handle, tmp + i)
            _ = scanner_scan_char(handle, UInt8(93))  # ']'
            var result = alloc[Float32](16)
            matrix_multiply(s[0].ctm.unsafe_ptr(), tmp, result)
            for i in range(16):
                s[0].ctm[i] = result[i]
            tmp.free(); result.free()
        else:
            _ = scanner_parse_quoted_string(handle, kw_buf, 256)
            _psc_skip_params(handle)

    kw_buf.free()
    ws_delims.free()

def make_perspective_matrix(fov_deg: Float32, near: Float32,
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

def make_screen_to_raster(fw: Int32, fh: Int32,
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

def finalize_scene(s: UnsafePointer[SceneParseState, MutAnyOrigin],
                 psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]):

    # ---- Camera matrices ----
    var c2w = alloc[Float32](16)
    var cam2w_tmp = alloc[Float32](16)
    for i in range(16): cam2w_tmp[i] = s[0].cam2w_raw[i]
    _ = matrix_invert(cam2w_tmp, c2w)
    cam2w_tmp.free()
    psc[0].camera_to_world = c2w


    var cts = alloc[Float32](16)
    make_perspective_matrix(s[0].camera_fov, Float32(0.01), cts)

    var frame = Float32(s[0].film_w) / Float32(s[0].film_h)
    var smin_x: Float32; var smax_x: Float32
    var smin_y: Float32; var smax_y: Float32
    if frame >= Float32(1):
        smin_x = -frame; smax_x = frame; smin_y = Float32(-1); smax_y = Float32(1)
    else:
        smin_x = Float32(-1); smax_x = Float32(1)
        smin_y = -Float32(1)/frame; smax_y = Float32(1)/frame

    var str_mat = alloc[Float32](16)
    make_screen_to_raster(s[0].film_w, s[0].film_h,
                                smin_x, smax_x, smin_y, smax_y, str_mat)

    var rts = alloc[Float32](16)
    _ = matrix_invert(str_mat, rts)    # rasterToScreen

    var cts_inv = alloc[Float32](16)
    _ = matrix_invert(cts, cts_inv)    # inverse(cameraToScreen)

    var r2c = alloc[Float32](16)
    matrix_multiply(cts_inv, rts, r2c) # rasterToCamera
    psc[0].raster_to_camera = r2c

    cts.free(); str_mat.free(); rts.free(); cts_inv.free()

    # ---- Materials ----
    var n_regular = len(s[0].named_materials)

    var n_al = 0
    for i in range(len(s[0].meshes)):
        if s[0].meshes[i].is_area_light:
            n_al += 1

    var n_mats = n_regular + n_al
    var mats = alloc[Material_C](max(n_mats, 1))
    for i in range(n_regular):
        var nm3 = s[0].named_materials[i]
        var mt = nm3.kind
        var ior = nm3.ior
        mats[i].type = mt
        mats[i].tex_idx = nm3.tex_idx
        mats[i].roughU  = nm3.roughness_u
        mats[i].roughV  = nm3.roughness_v
        mats[i].normal_tex_idx = nm3.normal_tex_idx
        mats[i].medium_interface_idx = Int32(-1)
        if mt == Int8(4):  # dielectric: albedo.r holds IOR
            mats[i].albedo = RGB(ior, Float32(0), Float32(0))
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))
        elif mt == Int8(5):  # coated diffuse: emission.r holds IOR
            mats[i].albedo = nm3.albedo
            mats[i].emission = RGB(ior, Float32(0), Float32(0))
        elif mt == Int8(7):  # coated conductor: emission.r holds IOR (clearcoat eta)
            mats[i].albedo = nm3.albedo
            mats[i].emission = RGB(ior, Float32(0), Float32(0))
        elif mt == Int8(9):  # thin dielectric: albedo.r holds IOR (same as type 4)
            mats[i].albedo = RGB(ior, Float32(0), Float32(0))
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))
        elif mt == Int8(8):  # mix: resolve sub-material names to indices
            var idx1 = Int32(0)
            var idx2 = Int32(0)
            for j in range(n_regular):
                if s[0].named_materials[j].name == nm3.mix_name1: idx1 = Int32(j)
                if s[0].named_materials[j].name == nm3.mix_name2: idx2 = Int32(j)
            mats[i].tex_idx = (idx2 << 16) | (idx1 & Int32(0xFFFF))
            mats[i].roughU  = nm3.mix_amount
            mats[i].albedo  = nm3.albedo
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))
        elif mt == Int8(6):  # diffusetransmission: emission holds the transmittance
            mats[i].albedo = nm3.albedo
            mats[i].emission = nm3.transmittance
        elif mt == Int8(11):  # hair: simplified as diffuse with melanin-computed color
            mats[i].albedo = nm3.albedo
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))
            mats[i].type = Int8(1)  # redirect to diffuse shader
        else:
            mats[i].albedo = nm3.albedo
            mats[i].emission = RGB(Float32(0), Float32(0), Float32(0))

    # ---- Meshes + area lights ----
    var n_meshes = len(s[0].meshes)
    var meshes   = alloc[TriangleMesh_C](max(n_meshes, 1))
    var out_pts  = alloc[UnsafePointer[Float32, MutAnyOrigin]](max(n_meshes, 1))
    var out_vis  = alloc[UnsafePointer[Int64, MutAnyOrigin]](max(n_meshes, 1))
    var out_fis  = alloc[UnsafePointer[Int64, MutAnyOrigin]](max(n_meshes, 1))
    var out_nv    = alloc[Int32](max(n_meshes, 1))
    var out_nt    = alloc[Int32](max(n_meshes, 1))
    var out_uv_nv = alloc[Int32](max(n_meshes, 1))
    var out_nrm_nv = alloc[Int32](max(n_meshes, 1))

    var al_list  = alloc[AreaLight_C](max(n_al, 1))
    var al_count = Int32(0)
    var al_mat_base = n_regular

    for i in range(n_meshes):
        ref ma = s[0].meshes[i]
        var nv = len(ma.points) // 4
        var nt = len(ma.face_idxs)
        # Copy points to C array
        var pts_c = alloc[Float32](nv * 4)
        for vi in range(nv * 4): pts_c[vi] = ma.points[vi]
        # Copy vertex indices to C array
        var vis_c = alloc[Int64](nt * 3)
        for ti2 in range(nt * 3): vis_c[ti2] = ma.vert_idxs[ti2]
        # Copy face indices
        var fis_c = alloc[Int64](nt)
        for ti2 in range(nt): fis_c[ti2] = ma.face_idxs[ti2]
        out_pts[i] = pts_c
        out_vis[i] = vis_c
        out_fis[i] = fis_c
        out_nv[i]  = Int32(nv)
        out_nt[i]  = Int32(nt)
        meshes[i].points        = pts_c
        meshes[i].vertexIndices = vis_c
        meshes[i].faceIndices   = fis_c
        # Copy UVs if present
        if len(ma.uvs) >= nv * 2:
            var uv_c = alloc[Float32](nv * 2)
            for ui in range(nv * 2): uv_c[ui] = ma.uvs[ui]
            meshes[i].uvs = uv_c
            out_uv_nv[i] = Int32(nv)
        else:
            meshes[i].uvs = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
            out_uv_nv[i] = Int32(0)
        # Copy shading normals if present
        if len(ma.normals) >= nv * 3:
            var nrm_c = alloc[Float32](nv * 3)
            for ni in range(nv * 3): nrm_c[ni] = ma.normals[ni]
            meshes[i].normals = nrm_c
            out_nrm_nv[i] = Int32(nv)
        else:
            # NULL pointer (not unsafe_dangling, which is address 4) so the
            # shading-normal guard `Int(mesh.normals) <= 1` works correctly.
            meshes[i].normals = UnsafePointer[Float32, MutAnyOrigin]()
            out_nrm_nv[i] = Int32(0)

        if ma.is_area_light:
            var al_idx = Int(al_count)
            var em = ma.al_rgb
            var t_area  = Float32(0.0)
            for ti in range(nt):
                var vi0 = Int(vis_c[ti*3+0]) * 4
                var vi1 = Int(vis_c[ti*3+1]) * 4
                var vi2 = Int(vis_c[ti*3+2]) * 4
                var ex = pts_c[vi1+0] - pts_c[vi0+0]
                var ey = pts_c[vi1+1] - pts_c[vi0+1]
                var ez = pts_c[vi1+2] - pts_c[vi0+2]
                var fx = pts_c[vi2+0] - pts_c[vi0+0]
                var fy = pts_c[vi2+1] - pts_c[vi0+1]
                var fz = pts_c[vi2+2] - pts_c[vi0+2]
                var cxv = ey*fz - ez*fy
                var cyv = ez*fx - ex*fz
                var czv = ex*fy - ey*fx
                t_area += Float32(0.5) * sqrt(cxv*cxv + cyv*cyv + czv*czv)
            al_list[al_idx].meshIdx    = Int32(i)
            al_list[al_idx].n_tris     = Int32(nt)
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
    var n_with_mi = 0
    for mi in range(n_meshes):
        if s[0].meshes[mi].inside_medium >= Int32(0) or s[0].meshes[mi].outside_medium >= Int32(0):
            n_with_mi += 1
    for si in range(len(s[0].spheres_cx)):
        if s[0].spheres_inside_med[si] >= Int32(0) or s[0].spheres_outside_med[si] >= Int32(0):
            n_with_mi += 1

    if n_with_mi > 0:
        var expanded_n = n_mats + n_with_mi
        var new_mats = alloc[Material_C](expanded_n)
        for ci in range(n_mats):
            new_mats[ci] = mats[ci]
        mats.free()
        mats = new_mats

        var iface_buf = alloc[MediumInterface_C](n_with_mi)
        var dup_idx = n_mats
        var iface_idx = 0

        for mi in range(n_meshes):
            var ins = s[0].meshes[mi].inside_medium
            var out = s[0].meshes[mi].outside_medium
            if ins < Int32(0) and out < Int32(0): continue
            var orig_mat = Int(s[0].meshes[mi].mat_idx)
            if orig_mat < 0: continue
            mats[dup_idx] = mats[orig_mat]
            mats[dup_idx].medium_interface_idx = Int32(iface_idx)
            iface_buf[iface_idx] = MediumInterface_C(ins, out)
            s[0].meshes[mi].mat_idx = Int32(dup_idx)
            dup_idx += 1
            iface_idx += 1

        for si in range(len(s[0].spheres_cx)):
            var ins = s[0].spheres_inside_med[si]
            var out = s[0].spheres_outside_med[si]
            if ins < Int32(0) and out < Int32(0): continue
            var orig_mat = Int(s[0].spheres_mat[si])
            if orig_mat < 0: continue
            mats[dup_idx] = mats[orig_mat]
            mats[dup_idx].medium_interface_idx = Int32(iface_idx)
            iface_buf[iface_idx] = MediumInterface_C(ins, out)
            s[0].spheres_mat[si] = Int32(dup_idx)
            dup_idx += 1
            iface_idx += 1

        n_mats = dup_idx
        psc[0].medium_ifaces = iface_buf
        psc[0].medium_iface_count = Int32(iface_idx)
    else:
        psc[0].medium_ifaces = UnsafePointer[MediumInterface_C, MutAnyOrigin].unsafe_dangling()
        psc[0].medium_iface_count = Int32(0)

    var total_tris = Int32(0)
    for i in range(n_meshes):
        total_tris += Int32(len(s[0].meshes[i].face_idxs))

    var prim_bounds = alloc[Float32](Int(total_tris) * 6)
    var tri_mesh    = alloc[Int32](Int(total_tris))
    var tri_local   = alloc[Int32](Int(total_tris))

    var flat_idx = Int32(0)
    for mi in range(n_meshes):
        var pts = out_pts[mi]
        var vis = out_vis[mi]
        var nt  = Int(out_nt[mi])
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
    var node_count = build_bvh2(prim_bounds, total_tris, bvh_nodes, bvh_order)

    prim_bounds.free()

    # Build PrimId array in BVH order
    var prim_ids = alloc[PrimId_C](Int(total_tris))

    var mesh_al_idx = alloc[Int32](max(n_meshes, 1))
    var running_al = Int32(0)
    for mi in range(n_meshes):
        if s[0].meshes[mi].is_area_light:
            mesh_al_idx[mi] = running_al
            running_al += 1
        else:
            mesh_al_idx[mi] = Int32(-1)

    for k in range(Int(total_tris)):
        var orig = Int(bvh_order[k])
        var mi   = Int(tri_mesh[orig])
        var ti   = Int(tri_local[orig])
        if s[0].meshes[mi].is_area_light:
            var al_idx = Int(mesh_al_idx[mi])
            prim_ids[k].type          = Int8(3)
            prim_ids[k].id1           = Int64(al_idx)
            prim_ids[k].id2           = (Int64(mi) << 32) | Int64(ti)
            prim_ids[k].materialIndex = Int64(al_mat_base + al_idx)
        else:
            var mat_idx = Int(s[0].meshes[mi].mat_idx)
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
    var norm_x = gaussian_norm(s[0].filter_support_x, s[0].filter_sigma)
    var norm_y = gaussian_norm(s[0].filter_support_y, s[0].filter_sigma)
    # Importance-sampling the filter gives uniform sample weights.
    var fweight = Float32(1.0)

    # ---- RNG seed from time ----
    var rng_seed = UInt64(perf_counter_ns())

    # ---- Film filename copy ----
    var fname = alloc[UInt8](PSC_FILE_MAX)
    var fnstr = s[0].film_filename
    var fnlen = min(fnstr.byte_length(), PSC_FILE_MAX - 1)
    for fi in range(fnlen): fname[fi] = fnstr.unsafe_ptr()[fi]
    fname[fnlen] = UInt8(0)

    # ---- Texture filename table ----
    var n_tex = len(s[0].tex_names)
    var tex_ptrs = alloc[UnsafePointer[UInt8, MutAnyOrigin]](max(n_tex, 1))
    for ti in range(n_tex):
        var fstr = s[0].tex_files[ti]
        var slen = fstr.byte_length()
        var copy = alloc[UInt8](slen + 1)
        for ci in range(slen): copy[ci] = fstr.unsafe_ptr()[ci]
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
    psc[0].mesh_nrm_n_verts = out_nrm_nv
    psc[0].mesh_count       = Int32(n_meshes)
    psc[0].bvh_nodes        = bvh_nodes
    psc[0].prim_ids         = prim_ids
    psc[0].bvh_node_count   = node_count
    psc[0].prim_count       = total_tris
    psc[0].film_w           = s[0].film_w
    psc[0].film_h           = s[0].film_h
    psc[0].camera_fov       = s[0].camera_fov
    psc[0].film_iso         = s[0].film_iso
    psc[0].film_max_comp    = s[0].film_max_comp
    psc[0].film_filename    = fname
    psc[0].filter_sigma     = s[0].filter_sigma
    psc[0].filter_support_x = s[0].filter_support_x
    psc[0].filter_support_y = s[0].filter_support_y
    psc[0].filter_type      = s[0].filter_type
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
    var nd = len(s[0].distant_dirs) // 3
    if nd > 0:
        var dl_buf = alloc[DistantLight_C](nd)
        for i in range(nd):
            dl_buf[i] = DistantLight_C(
                Vec3f(s[0].distant_dirs[i*3+0], s[0].distant_dirs[i*3+1], s[0].distant_dirs[i*3+2]),
                Float32(0),
                RGB(s[0].distant_rgbs[i*3+0], s[0].distant_rgbs[i*3+1], s[0].distant_rgbs[i*3+2]),
                Float32(0))
        psc[0].distant_lights = dl_buf
    else:
        psc[0].distant_lights = UnsafePointer[DistantLight_C, MutAnyOrigin].unsafe_dangling()
    psc[0].distant_count = Int32(nd)

    var np2 = len(s[0].point_pos) // 3
    if np2 > 0:
        var pl_buf = alloc[PointLight_C](np2)
        for i in range(np2):
            pl_buf[i] = PointLight_C(
                Point3f(s[0].point_pos[i*3+0], s[0].point_pos[i*3+1], s[0].point_pos[i*3+2]),
                Float32(0),
                RGB(s[0].point_rgbs[i*3+0], s[0].point_rgbs[i*3+1], s[0].point_rgbs[i*3+2]),
                Float32(0))
        psc[0].point_lights = pl_buf
    else:
        psc[0].point_lights = UnsafePointer[PointLight_C, MutAnyOrigin].unsafe_dangling()
    psc[0].point_count = Int32(np2)

    var ni = len(s[0].inf_tex_idx)
    if ni > 0:
        var il_bytes = max(ni, 1) * 40
        var il_buf = alloc[InfiniteLight_C](ni)
        for i in range(ni):
            var tidx = s[0].inf_tex_idx[i]
            var sc = RGB(s[0].inf_rgb[i*3+0], s[0].inf_rgb[i*3+1], s[0].inf_rgb[i*3+2])
            var cdf_w = Int32(0); var cdf_h = Int32(0)
            var cdf_ptr = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
            var raw_pixels = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()  # kept alive for CPU lookup
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
                if load_ok != Int32(0) and iw > 0 and ih > 0:  # != 0 = success
                    var pixels = pixels_ptr[0]
                    # OIIO returns this PFM bottom-to-top, but the equal-area
                    # lookup (and PBRT) index it top-to-bottom — so the envmap
                    # was upside down (reflections/background sampled the wrong
                    # rows). Flip rows in place before CDF build & shade lookups.
                    var tmp_row = alloc[Float32](iw * 3)
                    for r in range(ih // 2):
                        var r2 = ih - 1 - r
                        var off1 = r * iw * 3
                        var off2 = r2 * iw * 3
                        for c in range(iw * 3):
                            tmp_row[c] = pixels[off1 + c]
                            pixels[off1 + c] = pixels[off2 + c]
                            pixels[off2 + c] = tmp_row[c]
                    tmp_row.free()
                    raw_pixels = pixels  # keep alive for shade-time lookup
                    # CDF layout: (ih+1) marginal + ih*(iw+1) conditional floats
                    var cdf_size = (ih + 1) + ih * (iw + 1)
                    var cdf_buf = alloc[Float32](cdf_size)
                    # Compute per-row luminance sums (marginal pdf)
                    var row_sums = alloc[Float32](ih)
                    for ry in range(ih):
                        # Equal-area mapping: uniform solid angle per pixel
                        var row_sum = Float32(0.0)
                        for rx in range(iw):
                            var r2 = pixels[(ry * iw + rx) * 3 + 0]
                            var g2 = pixels[(ry * iw + rx) * 3 + 1]
                            var b2 = pixels[(ry * iw + rx) * 3 + 2]
                            var lum = Float32(0.2126) * r2 + Float32(0.7152) * g2 + Float32(0.0722) * b2
                            row_sum += lum
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
                        var base = (ih + 1) + ry * (iw + 1)
                        cdf_buf[base] = Float32(0.0)
                        for rx in range(iw):
                            var r2 = pixels[(ry * iw + rx) * 3 + 0]
                            var g2 = pixels[(ry * iw + rx) * 3 + 1]
                            var b2 = pixels[(ry * iw + rx) * 3 + 2]
                            var lum = Float32(0.2126) * r2 + Float32(0.7152) * g2 + Float32(0.0722) * b2
                            cdf_buf[base + rx + 1] = cdf_buf[base + rx] + lum
                        var row_total = cdf_buf[base + iw]
                        if row_total > Float32(0.0):
                            var inv_rt = Float32(1.0) / row_total
                            for rx in range(iw + 1):
                                cdf_buf[base + rx] *= inv_rt
                    row_sums.free()
                    cdf_ptr = cdf_buf
                    cdf_w = Int32(iw); cdf_h = Int32(ih)
                pixels_ptr.free()
            # Compute world-to-light transform (inverse of the light's CTM)
            var w2l = alloc[Float32](16)
            var light_ctm_base = i * 16
            var is_identity = True
            for ci in range(16):
                var expected = Float32(1) if (ci == 0 or ci == 5 or ci == 10 or ci == 15) else Float32(0)
                if s[0].inf_ctm[light_ctm_base + ci] != expected:
                    is_identity = False
                    break
            if is_identity:
                _psc_identity(w2l)
            else:
                var light_ctm_tmp = alloc[Float32](16)
                for ci in range(16): light_ctm_tmp[ci] = s[0].inf_ctm[light_ctm_base + ci]
                _ = matrix_invert(light_ctm_tmp, w2l)
                light_ctm_tmp.free()
            il_buf[i] = InfiniteLight_C(sc, tidx, cdf_w, cdf_h, cdf_ptr, raw_pixels, w2l)
        psc[0].infinite_lights = il_buf
    else:
        psc[0].infinite_lights = UnsafePointer[InfiniteLight_C, MutAnyOrigin].unsafe_dangling()
    psc[0].infinite_count = Int32(ni)

    # ---- Analytical spheres ----
    var ns = len(s[0].spheres_cx)
    if ns > 0:
        var sph_buf = alloc[Sphere_C](ns)
        for i in range(ns):
            var em = SampledSpectrum(s[0].spheres_rgb[i].r, s[0].spheres_rgb[i].g, s[0].spheres_rgb[i].b)
            var al_flag = Int8(1) if s[0].spheres_al[i] else Int8(0)
            sph_buf[i] = Sphere_C(
                Point3f(s[0].spheres_cx[i], s[0].spheres_cy[i], s[0].spheres_cz[i]),
                s[0].spheres_r[i],
                s[0].spheres_mat[i],
                al_flag,
                Int8(0), Int8(0), Int8(0),
                em)
        psc[0].spheres = sph_buf
    else:
        psc[0].spheres = UnsafePointer[Sphere_C, MutAnyOrigin].unsafe_dangling()
    psc[0].sphere_count = Int32(ns)

    # ---- Media ----
    var nm = len(s[0].med_g)
    if nm > 0:
        var med_buf = alloc[Medium_C](nm)
        for i in range(nm):
            var sa = SampledSpectrum(s[0].med_sa[i*3], s[0].med_sa[i*3+1], s[0].med_sa[i*3+2])
            var ss = SampledSpectrum(s[0].med_ss[i*3], s[0].med_ss[i*3+1], s[0].med_ss[i*3+2])
            med_buf[i] = Medium_C(sa, ss, s[0].med_g[i],
                                  Float32(0), Float32(0), Float32(0))
        psc[0].mediums = med_buf
    else:
        psc[0].mediums = UnsafePointer[Medium_C, MutAnyOrigin].unsafe_dangling()
    psc[0].medium_count = Int32(nm)


def resize_film(psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin],
               new_w: Int32, new_h: Int32):
    """Override film resolution and recompute raster_to_camera in place."""
    psc[0].film_w = new_w
    psc[0].film_h = new_h

    var frame = Float32(new_w) / Float32(new_h)
    var smin_x: Float32; var smax_x: Float32
    var smin_y: Float32; var smax_y: Float32
    if frame >= Float32(1):
        smin_x = -frame; smax_x = frame; smin_y = Float32(-1); smax_y = Float32(1)
    else:
        smin_x = Float32(-1); smax_x = Float32(1)
        smin_y = -Float32(1)/frame; smax_y = Float32(1)/frame

    var str_mat = alloc[Float32](16)
    make_screen_to_raster(new_w, new_h, smin_x, smax_x, smin_y, smax_y, str_mat)
    var rts = alloc[Float32](16)
    _ = matrix_invert(str_mat, rts)
    var cts = alloc[Float32](16)
    make_perspective_matrix(psc[0].camera_fov, Float32(0.01), cts)
    var cts_inv = alloc[Float32](16)
    _ = matrix_invert(cts, cts_inv)
    if Int(psc[0].raster_to_camera) > 1:
        psc[0].raster_to_camera.free()
    var r2c = alloc[Float32](16)
    matrix_multiply(cts_inv, rts, r2c)
    psc[0].raster_to_camera = r2c
    cts.free(); str_mat.free(); rts.free(); cts_inv.free()


# ── Exported API ──────────────────────────────────────────────────────────────

def mojo_parse_scene(path: UnsafePointer[UInt8, MutAnyOrigin]
                    ) -> UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]:
    external_call["createTextureSystem", NoneType]()
    var handle = scanner_open(path)
    if handle[0].is_at_end != Int32(0):
        print("Error: cannot open scene file:", String(unsafe_from_utf8_ptr=path.as_immutable()))
        scanner_free(handle)
        return UnsafePointer[ParsedScene_Mojo, MutAnyOrigin].unsafe_dangling()

    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    var pi = 0
    while path[pi] != UInt8(0):
        pi += 1
    var last_slash = -1
    for ki in range(pi):
        if path[ki] == UInt8(47):
            last_slash = ki
    if last_slash >= 0:
        var dir_tmp = alloc[UInt8](last_slash + 2)
        for ki in range(last_slash + 1):
            dir_tmp[ki] = path[ki]
        dir_tmp[last_slash + 1] = UInt8(0)
        s_ptr[0].scene_dir = String(unsafe_from_utf8_ptr=dir_tmp.as_immutable())
        dir_tmp.free()
    parse_scene_file(handle, s_ptr)
    scanner_free(handle)
    flush_hair(s_ptr)

    var psc = alloc[ParsedScene_Mojo](1)
    finalize_scene(s_ptr, psc)
    _ = s_ptr.take_pointee()  # destroys all List fields
    s_ptr.free()
    return psc

def mojo_parsed_free(psc: UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]):
    if Int(psc) == 0:
        return
    var n = Int(psc[0].mesh_count)
    for i in range(n):
        psc[0].mesh_pts[i].free()
        psc[0].mesh_vis[i].free()
        psc[0].mesh_fis[i].free()
        if psc[0].mesh_uv_n_verts[i] > Int32(0):
            psc[0].meshes[i].uvs.free()
        if psc[0].mesh_nrm_n_verts[i] > Int32(0):
            psc[0].meshes[i].normals.free()
    if psc[0].mesh_count > 0:
        psc[0].mesh_pts.free()
        psc[0].mesh_vis.free()
        psc[0].mesh_fis.free()
        psc[0].mesh_n_verts.free()
        psc[0].mesh_n_tris.free()
        psc[0].mesh_uv_n_verts.free()
        psc[0].mesh_nrm_n_verts.free()
        psc[0].meshes.free()
    if psc[0].material_count > 0:
        psc[0].materials.free()
    if psc[0].area_light_count > 0:
        psc[0].area_lights.free()
    if psc[0].bvh_node_count > 0:
        psc[0].bvh_nodes.free()
    if psc[0].prim_count > 0:
        psc[0].prim_ids.free()
    if Int(psc[0].raster_to_camera) > 1:
        psc[0].raster_to_camera.free()
    if Int(psc[0].camera_to_world) > 1:
        psc[0].camera_to_world.free()
    if Int(psc[0].film_filename) > 1:
        psc[0].film_filename.free()
    if psc[0].tex_count > 0:
        var nt = Int(psc[0].tex_count)
        for ti in range(nt):
            psc[0].tex_filenames[ti].free()
        psc[0].tex_filenames.free()
    if psc[0].distant_count > 0:
        psc[0].distant_lights.free()
    if psc[0].point_count > 0:
        psc[0].point_lights.free()
    if psc[0].infinite_count > 0:
        var ni = Int(psc[0].infinite_count)
        for ii in range(ni):
            var il = psc[0].infinite_lights[ii]
            if il.cdf_w > Int32(0):
                il.cdf_ptr.free()
                _ = external_call["free_texture_rgb", Int32,
                    UnsafePointer[Float32, MutAnyOrigin]](il.pixels_ptr)
            il.world_to_light.free()
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
