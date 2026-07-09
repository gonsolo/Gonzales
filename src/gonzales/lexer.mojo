from std.memory import alloc
from std.math import log as _log_math, pow as _pow_math
from .geometry import RGB

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


def scanner_free(handle: UnsafePointer[PbrtScanner, MutAnyOrigin]):
    if Int(handle[0].buffer) > 1:
        handle[0].buffer.free()
    handle.free()


def scanner_is_at_end(handle: UnsafePointer[PbrtScanner, MutAnyOrigin]) -> Int32:
    return handle[0].is_at_end


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
    var tmp_s = alloc[UInt8](1024)
    if _psc_type_is_float(type_buf):
        if is_array:
            # Size to fit exactly — a fixed cap here would silently stop
            # short and desync the scanner cursor mid-array, same bug as
            # the mesh/curve scratch buffers (see pbrt_parser.mojo).
            var cnt = scanner_count_floats(handle)
            var tmp = alloc[Float32](Int(cnt) if cnt > Int32(0) else 1)
            _ = scanner_scan_floats(handle, tmp, cnt)
            tmp.free()
        else:
            var tmp = alloc[Float32](1)
            _ = scanner_scan_float(handle, tmp)
            tmp.free()
    elif _psc_type_is_int(type_buf):
        if is_array:
            var cnt = scanner_count_ints(handle)
            var tmp = alloc[Int32](Int(cnt) if cnt > Int32(0) else 1)
            _ = scanner_scan_ints(handle, tmp, cnt)
            tmp.free()
        else:
            var tmp = alloc[Int32](1)
            _ = scanner_scan_int(handle, tmp)
            tmp.free()
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
    tmp_s.free()

struct ParamScanner(Movable):
    """Wraps the alloc-buffers / parse-header-loop / free scaffolding shared
    by every PBRT directive's parameter loop. Construct once per directive,
    then `while ps.next(handle): ...`; ps.type_buf/ps.name_buf/ps.is_array
    refresh each call. Unrecognized params: `ps.skip(handle)`."""
    var type_buf: UnsafePointer[UInt8, MutAnyOrigin]
    var name_buf: UnsafePointer[UInt8, MutAnyOrigin]
    var ia: UnsafePointer[Int32, MutAnyOrigin]
    var type_cap: Int32
    var name_cap: Int32
    var is_array: Int32

    def __init__(out self, type_cap: Int32 = Int32(64), name_cap: Int32 = Int32(128)):
        self.type_cap = type_cap
        self.name_cap = name_cap
        self.type_buf = alloc[UInt8](Int(type_cap))
        self.name_buf = alloc[UInt8](Int(name_cap))
        self.ia = alloc[Int32](1)
        self.is_array = Int32(0)

    def next(mut self, handle: UnsafePointer[PbrtScanner, MutAnyOrigin]) -> Bool:
        self.ia[0] = Int32(0)
        var found = scanner_parse_param_header(handle, self.type_buf, self.type_cap,
                                                self.name_buf, self.name_cap, self.ia)
        self.is_array = self.ia[0]
        return found != 0

    def name_is(self, n: StringLiteral) -> Bool:
        return _psc_streq(self.name_buf, n)

    def is_float(self) -> Bool:
        return _psc_type_is_float(self.type_buf)

    def is_int(self) -> Bool:
        return _psc_type_is_int(self.type_buf)

    def is_str(self) -> Bool:
        return _psc_type_is_str(self.type_buf)

    def skip(self, handle: UnsafePointer[PbrtScanner, MutAnyOrigin]):
        _psc_skip_value(handle, self.type_buf, self.is_array)
        if self.is_array:
            _ = scanner_scan_char(handle, UInt8(93))  # ']'

    def __del__(deinit self):
        self.type_buf.free()
        self.name_buf.free()
        self.ia.free()

def _psc_skip_params(handle: UnsafePointer[PbrtScanner, MutAnyOrigin]):
    var ps = ParamScanner()
    while ps.next(handle):
        ps.skip(handle)

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

# pbrt "bool" values are bare (unquoted) `true`/`false` tokens, e.g.
# `"bool remaproughness" [ false ]` — scanned like the catch-all _psc_skip_value
# else-branch, but the token is compared instead of discarded.
def _psc_scan_one_bool(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                      is_array: Int32) -> Bool:
    var buf = alloc[UInt8](32)
    var nl_buf = alloc[UInt8](1)
    nl_buf[0] = UInt8(10)
    _ = scanner_scan_token(handle, nl_buf, 1, buf, 32)
    var v = _psc_streq(buf, "true")
    nl_buf.free()
    buf.free()
    if is_array:
        _ = scanner_scan_char(handle, UInt8(93))  # ']'
    return v

def _psc_scan_rgb(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                 rgb: UnsafePointer[Float32, MutAnyOrigin],
                 is_array: Int32):
    _ = scanner_scan_float(handle, rgb + 0)
    _ = scanner_scan_float(handle, rgb + 1)
    _ = scanner_scan_float(handle, rgb + 2)
    if is_array:
        _ = scanner_scan_char(handle, UInt8(93))  # ']'

comptime SPECTRUM_SCAN_SCRATCH_MAX: Int = 256  # generous bound for wavelength/value pairs

def _psc_scan_spectrum_scalar(
    handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
    is_array: Int32,
    name_dst: UnsafePointer[UInt8, MutAnyOrigin],
    name_max: Int32,
) -> Tuple[Float32, Bool]:
    """Robustly scan a "spectrum"-typed parameter's value into a single
    representative scalar. pbrt allows a "spectrum" param to be EITHER a
    named-spectrum string reference ("metal-Au-eta") OR an inline piecewise
    array [wavelen0 val0 wavelen1 val1 ...] under the SAME declared type --
    calling a single-shape scanner (a fixed quoted-string parse, or a
    fixed-arity float scan) unconditionally silently mishandles whichever
    form it didn't expect. A numeric array fed to a quoted-string parse
    bails immediately without consuming anything, leaving the array sitting
    in the token stream to be mis-parsed as bogus subsequent scene
    directives -- no crash, no warning, just wrong data.

    Consolidates a bug found and independently fixed twice in this codebase
    (medium sigma_a/sigma_s in pbrt_parser.mojo, conductor/dielectric eta/k
    in material_builder.mojo) into one place, so it can't recur a third
    time. Returns (mean, True) for the numeric-array form -- not a real
    spectral upsample, just the mean of the value samples (odd indices),
    matching this codebase's existing scope for "good enough for the
    near-constant spectra actually seen in this scene corpus." Returns
    (0.0, False) for the named-string form, with the name copied into
    name_dst for the caller's own lookup. Consumes the trailing array ']'
    itself (matches _psc_scan_one_float/_psc_scan_rgb's convention, NOT
    _psc_skip_value's, which leaves that to its caller)."""
    var cnt = scanner_count_floats(handle)
    if cnt > Int32(0):
        var cap = Int(cnt) if Int(cnt) < SPECTRUM_SCAN_SCRATCH_MAX else SPECTRUM_SCAN_SCRATCH_MAX
        var tmp = alloc[Float32](cap)
        var n = scanner_scan_floats(handle, tmp, Int32(cap))
        var sum = Float32(0.0)
        var count = 0
        var vi = 1
        while vi < Int(n):
            sum += tmp[vi]
            count += 1
            vi += 2
        var mean = sum / Float32(max(count, 1))
        tmp.free()
        if name_max > Int32(0):
            name_dst[0] = UInt8(0)
        if is_array:
            _ = scanner_scan_char(handle, UInt8(93))
        return (mean, True)
    else:
        _ = scanner_parse_quoted_string(handle, name_dst, name_max)
        if is_array:
            var tmp_s = alloc[UInt8](Int(name_max) if name_max > Int32(0) else 1)
            while scanner_parse_quoted_string(handle, tmp_s, name_max) >= 0:
                pass
            tmp_s.free()
            _ = scanner_scan_char(handle, UInt8(93))
        return (Float32(0.0), False)

def _psc_scan_float4(handle: UnsafePointer[PbrtScanner, MutAnyOrigin],
                    dst: UnsafePointer[Float32, MutAnyOrigin],
                    is_array: Int32):
    _ = scanner_scan_float(handle, dst + 0)
    _ = scanner_scan_float(handle, dst + 1)
    _ = scanner_scan_float(handle, dst + 2)
    _ = scanner_scan_float(handle, dst + 3)
    if is_array:
        _ = scanner_scan_char(handle, UInt8(93))  # ']'

# ── Generic parameter dictionary ─────────────────────────────────────────────
# pbrt's own parser design: scan every directive's params ONCE into a generic
# name -> typed-value structure (no per-name knowledge at scan time), then
# have each material/light/shape/texture/camera/film/... constructor query it
# by name with typed accessors and defaults -- mirrors pbrt's own
# ParameterDictionary. Consumers no longer hand-write "elif name_is(X) and
# type_buf[0]==Y: <bespoke scan>" chains; they call _psc_collect_params once,
# then params.get_float(name, default) etc. Handles the float-vs-texture
# duality ("reflectance" as either a "float"/"rgb" value or a "texture"
# reference) and the spectrum string-vs-array duality (via
# _psc_scan_spectrum_scalar) uniformly, for every parameter, not just the
# ones someone thought to special-case.
#
# Deliberately NOT used for bulk geometry data (triangle mesh points/indices/
# uvs/normals, curve control points, volume grid density arrays) -- those are
# large numeric buffers with their own hardened incremental-growth scratch
# buffers (see pbrt_parser.mojo's mesh/curve parsing), a different concern
# from small named scalar/string/spectrum config values. Materials, lights,
# textures, camera, film, sampler, and integrator directives all fit the
# small-named-params pattern and use this.

@fieldwise_init
struct ParamValue(Copyable, Movable):
    """Holds a scanned parameter's value. Exactly one of the two lists is
    populated, depending on the declared pbrt type: `floats` for
    float/integer/rgb/color/normal/point/vector/spectrum(numeric-array)
    types, `strs` for string/texture/spectrum(named-string)/bool types (bool
    stores its raw "true"/"false" token as a 1-element string list)."""
    var floats: List[Float32]
    var strs: List[String]

@fieldwise_init
struct ParsedParam(Copyable, Movable):
    var name: String
    var value: ParamValue

struct ParameterDictionary(Movable):
    """Small linear list of (name, ParamValue) pairs -- directives have at
    most a couple dozen params, so linear-scan lookup is simpler than a Dict
    and just as fast at this size."""
    var params: List[ParsedParam]

    def __init__(out self):
        self.params = List[ParsedParam]()

    def get_float(self, name: StringLiteral, default: Float32) -> Float32:
        for p in self.params:
            if p.name == name and len(p.value.floats) > 0:
                return p.value.floats[0]
        return default

    def get_int(self, name: StringLiteral, default: Int32) -> Int32:
        for p in self.params:
            if p.name == name and len(p.value.floats) > 0:
                return Int32(p.value.floats[0])
        return default

    def get_bool(self, name: StringLiteral, default: Bool) -> Bool:
        for p in self.params:
            if p.name == name and len(p.value.strs) > 0:
                return p.value.strs[0] == "true"
        return default

    def get_rgb(self, name: StringLiteral, default: RGB) -> RGB:
        for p in self.params:
            if p.name == name and len(p.value.floats) >= 3:
                return RGB(p.value.floats[0], p.value.floats[1], p.value.floats[2])
        return default

    def get_rgb_or_blackbody(self, name: StringLiteral, default: RGB) -> RGB:
        """Light emission color params ("L"/"I") accept EITHER an explicit
        RGB triple ("rgb L" [r g b], 3 floats) OR a blackbody temperature
        ("blackbody L" [6500], 1 float, Kelvin) -- both land in the same
        dictionary entry's `.floats`, distinguished by count."""
        for p in self.params:
            if p.name == name:
                if len(p.value.floats) >= 3:
                    return RGB(p.value.floats[0], p.value.floats[1], p.value.floats[2])
                elif len(p.value.floats) == 1:
                    var rgb_out = alloc[Float32](3)
                    _psc_blackbody_to_rgb(p.value.floats[0], rgb_out)
                    var result = RGB(rgb_out[0], rgb_out[1], rgb_out[2])
                    rgb_out.free()
                    return result
        return default

    def get_string(self, name: StringLiteral, default: String) -> String:
        for p in self.params:
            if p.name == name and len(p.value.strs) > 0:
                return p.value.strs[0]
        return default

    def has(self, name: StringLiteral) -> Bool:
        for p in self.params:
            if p.name == name:
                return True
        return False

    def get_floats(self, name: StringLiteral) -> List[Float32]:
        """Every float sample for `name`, in declaration order (e.g. mesh-
        level small arrays like a 2-value min/max, or spectrum sample pairs
        the caller wants to interpret itself rather than via
        get_spectrum_scalar's mean-of-samples convention)."""
        for p in self.params:
            if p.name == name:
                return p.value.floats.copy()
        return List[Float32]()

    def get_strings(self, name: StringLiteral) -> List[String]:
        for p in self.params:
            if p.name == name:
                return p.value.strs.copy()
        return List[String]()

    def get_spectrum_scalar(self, name: StringLiteral, default: Float32) -> Tuple[Float32, String]:
        """(numeric_mean, "") for the inline-array spectrum form, or
        (default, named_string) for the named-spectrum-reference form --
        mirrors _psc_scan_spectrum_scalar's own (value, is_numeric) split,
        just replayed against the already-collected dictionary entry instead
        of scanning live."""
        for p in self.params:
            if p.name == name:
                if len(p.value.floats) > 0:
                    var sum = Float32(0.0)
                    var count = 0
                    var vi = 1
                    while vi < len(p.value.floats):
                        sum += p.value.floats[vi]
                        count += 1
                        vi += 2
                    return (sum / Float32(max(count, 1)), String(""))
                elif len(p.value.strs) > 0:
                    return (default, p.value.strs[0])
        return (default, String(""))

def _psc_collect_params(handle: UnsafePointer[PbrtScanner, MutAnyOrigin]) -> ParameterDictionary:
    """Generic, type-driven (not name-driven) scan of every parameter in the
    current directive into a ParameterDictionary. Reuses the same type
    classification (_psc_type_is_float/_psc_type_is_int/_psc_type_is_str)
    and quantity-scanning primitives (scanner_count_floats/scan_floats,
    scanner_parse_quoted_string) already proven safe by _psc_skip_value and
    _psc_scan_spectrum_scalar -- nothing here is scanned differently based on
    the parameter's NAME, only its declared TYPE, so there is no
    "someone forgot this (name,type) combination" failure mode; an unused
    name is simply never looked up afterward (matches real pbrt's own
    behavior for parameters an object type doesn't query)."""
    var dict = ParameterDictionary()
    var ps = ParamScanner()
    while ps.next(handle):
        var pv = ParamValue(List[Float32](), List[String]())
        var is_spectrum = ps.type_buf[0] == UInt8(115) and ps.type_buf[1] == UInt8(112)  # "sp"
        if _psc_type_is_blackbody(ps.type_buf):
            # "blackbody" is a single temperature value (Kelvin), not an RGB
            # triple -- stored as a 1-element float list, same shape as a
            # scalar "float"-typed param. Consumers that accept EITHER a
            # direct RGB ("rgb L" [r g b], 3 floats) OR a blackbody
            # temperature ("blackbody L" [6500], 1 float) for the same
            # logical parameter distinguish the two by count, same pattern
            # used for "eta"'s RGB-vs-scalar duality in material_builder.mojo.
            var tmp = alloc[Float32](1)
            var n = scanner_scan_float(handle, tmp)
            if n > Int32(0):
                pv.floats.append(tmp[0])
            tmp.free()
            if ps.is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        elif is_spectrum:
            var name_buf = alloc[UInt8](256)
            var (mean, is_numeric) = _psc_scan_spectrum_scalar(handle, ps.is_array, name_buf, Int32(256))
            if is_numeric:
                pv.floats.append(mean)
            else:
                pv.strs.append(String(unsafe_from_utf8_ptr=name_buf.as_immutable()))
            name_buf.free()
        elif _psc_type_is_float(ps.type_buf):
            if ps.is_array:
                var cnt = scanner_count_floats(handle)
                var tmp = alloc[Float32](Int(cnt) if cnt > Int32(0) else 1)
                var n = scanner_scan_floats(handle, tmp, cnt)
                for i in range(Int(n)):
                    pv.floats.append(tmp[i])
                tmp.free()
                _ = scanner_scan_char(handle, UInt8(93))
            else:
                var tmp = alloc[Float32](1)
                var n = scanner_scan_float(handle, tmp)
                if n > Int32(0):
                    pv.floats.append(tmp[0])
                tmp.free()
        elif _psc_type_is_int(ps.type_buf):
            if ps.is_array:
                var cnt = scanner_count_ints(handle)
                var tmp = alloc[Int32](Int(cnt) if cnt > Int32(0) else 1)
                var n = scanner_scan_ints(handle, tmp, cnt)
                for i in range(Int(n)):
                    pv.floats.append(Float32(tmp[i]))
                tmp.free()
                _ = scanner_scan_char(handle, UInt8(93))
            else:
                var tmp = alloc[Int32](1)
                var n = scanner_scan_int(handle, tmp)
                if n > Int32(0):
                    pv.floats.append(Float32(tmp[0]))
                tmp.free()
        elif _psc_type_is_str(ps.type_buf):
            var tmp_s = alloc[UInt8](512)
            var r = scanner_parse_quoted_string(handle, tmp_s, 512)
            if r >= 0:
                pv.strs.append(String(unsafe_from_utf8_ptr=tmp_s.as_immutable()))
            if ps.is_array:
                while True:
                    var r2 = scanner_parse_quoted_string(handle, tmp_s, 512)
                    if r2 < 0:
                        break
                    pv.strs.append(String(unsafe_from_utf8_ptr=tmp_s.as_immutable()))
                _ = scanner_scan_char(handle, UInt8(93))
            tmp_s.free()
        else:
            # bool (bare true/false token) or anything else not covered
            # above -- scan the raw token and keep it as a string so
            # get_bool can compare it; mirrors _psc_skip_value's own
            # else-branch scanning, just retained instead of discarded.
            var tmp_s = alloc[UInt8](32)
            var nl_buf = alloc[UInt8](1)
            nl_buf[0] = UInt8(10)
            _ = scanner_scan_token(handle, nl_buf, 1, tmp_s, 32)
            nl_buf.free()
            pv.strs.append(String(unsafe_from_utf8_ptr=tmp_s.as_immutable()))
            tmp_s.free()
            if ps.is_array:
                _ = scanner_scan_char(handle, UInt8(93))
        var name_str = String(unsafe_from_utf8_ptr=ps.name_buf.as_immutable())
        dict.params.append(ParsedParam(name_str, pv^))
    return dict^
