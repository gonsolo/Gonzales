from std.memory import alloc
from std.math import log as _log_math, pow as _pow_math

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


# ── String / type utilities (shared by material_builder and light_builder) ────

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
    var t100 = temp / Float32(100.0)
    var r: Float32; var g: Float32; var b: Float32
    # Red channel
    if temp <= Float32(6600):
        r = Float32(1.0)
    else:
        r = Float32(329.698727446) * _pow_math(t100 - Float32(60), Float32(-0.1332047592)) / Float32(255)
        r = max(Float32(0), min(Float32(1), r))
    # Green channel
    if temp <= Float32(6600):
        g = (Float32(99.4708025861) * _log_math(t100) - Float32(161.1195681661)) / Float32(255)
        g = max(Float32(0), min(Float32(1), g))
    else:
        g = Float32(288.1221695283) * _pow_math(t100 - Float32(60), Float32(-0.0755148492)) / Float32(255)
        g = max(Float32(0), min(Float32(1), g))
    # Blue channel
    if temp >= Float32(6600):
        b = Float32(1.0)
    elif temp <= Float32(1900):
        b = Float32(0.0)
    else:
        b = (Float32(138.5177312231) * _log_math(t100 - Float32(10)) - Float32(305.0447927307)) / Float32(255)
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
