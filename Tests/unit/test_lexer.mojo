from std.math import abs
from std.memory import alloc
from std.testing import assert_true, assert_false, assert_equal, TestSuite
from gonzales.lexer import (
    scan_int, scan_float, scan_floats, scan_ints, count_floats, count_ints,
    scan_char, scan_token, parse_quoted_string,
)

comptime EPS: Float32 = 1e-5

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _buf(s: String) -> UnsafePointer[UInt8, MutAnyOrigin]:
    """Copies a Mojo string literal into a fresh UInt8 buffer — mirrors the
    same String->buffer pattern __init__.mojo uses for argv, just for tests
    against the byte-level scan_* primitives directly (no PbrtScanner/file
    needed)."""
    var n = s.byte_length()
    var b = alloc[UInt8](n)
    for i in range(n):
        b[i] = s.as_bytes()[i]
    return b

# ── scan_int ─────────────────────────────────────────────────────────────

def test_scan_int_positive() raises:
    var buf = _buf("42")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var result = alloc[Int32](1)
    var ok = scan_int(buf, Int32(2), cur, result)
    assert_true(ok == Int32(1))
    assert_true(result[0] == Int32(42))
    assert_true(cur[0] == Int32(2))
    buf.free(); cur.free(); result.free()

def test_scan_int_negative() raises:
    var buf = _buf("-17")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var result = alloc[Int32](1)
    var ok = scan_int(buf, Int32(3), cur, result)
    assert_true(ok == Int32(1))
    assert_true(result[0] == Int32(-17))
    buf.free(); cur.free(); result.free()

def test_scan_int_skips_leading_whitespace() raises:
    var buf = _buf("   7")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var result = alloc[Int32](1)
    var ok = scan_int(buf, Int32(4), cur, result)
    assert_true(ok == Int32(1))
    assert_true(result[0] == Int32(7))
    buf.free(); cur.free(); result.free()

def test_scan_int_failure_leaves_cursor_unchanged() raises:
    """On failure (no digit found), the cursor must stay exactly where it
    started — callers rely on this to detect "not an int" without having
    consumed anything from the stream."""
    var buf = _buf("abc")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var result = alloc[Int32](1)
    var ok = scan_int(buf, Int32(3), cur, result)
    assert_true(ok == Int32(0))
    assert_true(cur[0] == Int32(0))
    buf.free(); cur.free(); result.free()

# ── scan_float ───────────────────────────────────────────────────────────

def test_scan_float_decimal() raises:
    var buf = _buf("3.14")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var result = alloc[Float32](1)
    var ok = scan_float(buf, Int32(4), cur, result)
    assert_true(ok == Int32(1))
    assert_true(_close(result[0], Float32(3.14)))
    buf.free(); cur.free(); result.free()

def test_scan_float_negative() raises:
    var buf = _buf("-0.5")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var result = alloc[Float32](1)
    var ok = scan_float(buf, Int32(4), cur, result)
    assert_true(ok == Int32(1))
    assert_true(_close(result[0], Float32(-0.5)))
    buf.free(); cur.free(); result.free()

def test_scan_float_exponent() raises:
    var buf = _buf("1.5e-2")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var result = alloc[Float32](1)
    var ok = scan_float(buf, Int32(6), cur, result)
    assert_true(ok == Int32(1))
    assert_true(_close(result[0], Float32(0.015)))
    buf.free(); cur.free(); result.free()

def test_scan_float_bare_integer() raises:
    var buf = _buf("42")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var result = alloc[Float32](1)
    var ok = scan_float(buf, Int32(2), cur, result)
    assert_true(ok == Int32(1))
    assert_true(_close(result[0], Float32(42.0)))
    buf.free(); cur.free(); result.free()

# ── count_floats / scan_floats (and the truncation semantics behind #42) ────

def test_count_floats_does_not_advance_cursor() raises:
    """Count_floats takes the cursor by value — it must be a pure peek,
    leaving the caller free to scan the same range afterward."""
    var buf = _buf("1.5 2.5 3.5")
    var n = count_floats(buf, Int32(11), Int32(0))
    assert_true(n == Int32(3))

def test_scan_floats_reads_all_when_capacity_suffices() raises:
    var buf = _buf("1.5 2.5 3.5")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var dst = alloc[Float32](8)
    var n = scan_floats(buf, Int32(11), cur, dst, Int32(8))
    assert_true(n == Int32(3))
    assert_true(_close(dst[0], Float32(1.5)))
    assert_true(_close(dst[1], Float32(2.5)))
    assert_true(_close(dst[2], Float32(3.5)))
    assert_true(cur[0] == Int32(11))  # cursor lands at end of buffer
    buf.free(); cur.free(); dst.free()

def test_scan_floats_truncates_at_max_count_and_leaves_cursor_mid_buffer() raises:
    """Pins down the exact truncation behavior that made the pre-fix
    mesh/curve scratch buffers silently desync the scanner (see the #42
    fix, pbrt_parser.mojo): when max_count is smaller than the actual
    array size, scan_floats stops early and the cursor is left BEFORE the
    remaining unconsumed values, not at the end of the array. This is why
    callers must size the destination via count_floats first rather than
    assume a fixed cap always fits."""
    var buf = _buf("1.5 2.5 3.5")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var dst = alloc[Float32](2)
    var n = scan_floats(buf, Int32(11), cur, dst, Int32(2))
    assert_true(n == Int32(2))
    assert_true(_close(dst[0], Float32(1.5)))
    assert_true(_close(dst[1], Float32(2.5)))
    assert_true(cur[0] < Int32(11))       # did NOT reach the end...
    assert_true(cur[0] == Int32(7))       # ...cursor sits right before "3.5"
    buf.free(); cur.free(); dst.free()

# ── count_ints / scan_ints ──────────────────────────────────────────────────

def test_count_ints_does_not_advance_cursor() raises:
    var buf = _buf("1 2 3 4")
    var n = count_ints(buf, Int32(7), Int32(0))
    assert_true(n == Int32(4))

def test_scan_ints_truncates_at_max_count() raises:
    var buf = _buf("1 2 3 4")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var dst = alloc[Int32](2)
    var n = scan_ints(buf, Int32(7), cur, dst, Int32(2))
    assert_true(n == Int32(2))
    assert_true(dst[0] == Int32(1))
    assert_true(dst[1] == Int32(2))
    assert_true(cur[0] < Int32(7))
    buf.free(); cur.free(); dst.free()

# ── scan_char ────────────────────────────────────────────────────────────

def test_scan_char_matches_and_advances() raises:
    var buf = _buf("  ]rest")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var ok = scan_char(buf, Int32(7), cur, UInt8(93))  # ']'
    assert_true(ok == Int32(1))
    assert_true(cur[0] == Int32(3))
    buf.free(); cur.free()

def test_scan_char_no_match_advances_past_whitespace_only() raises:
    """On a non-match, whitespace before the checked position is still
    skipped (per the implementation), but the mismatching byte itself is
    not consumed."""
    var buf = _buf("  x")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var ok = scan_char(buf, Int32(3), cur, UInt8(93))  # ']', buffer has 'x'
    assert_true(ok == Int32(0))
    assert_true(cur[0] == Int32(2))
    buf.free(); cur.free()

# ── scan_token ───────────────────────────────────────────────────────────

def test_scan_token_reads_until_delimiter() raises:
    var buf = _buf("hello world")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var delims = alloc[UInt8](1); delims[0] = UInt8(32)  # ' '
    var out = alloc[UInt8](32)
    var n = scan_token(buf, Int32(11), cur, delims, Int32(1), out, Int32(32))
    assert_true(n == Int32(5))
    assert_true(String(unsafe_from_utf8_ptr=out.as_immutable()) == String("hello"))
    assert_true(cur[0] == Int32(5))
    buf.free(); cur.free(); delims.free(); out.free()

def test_scan_token_at_end_returns_negative() raises:
    var buf = _buf("")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var delims = alloc[UInt8](1); delims[0] = UInt8(32)
    var out = alloc[UInt8](8)
    var n = scan_token(buf, Int32(0), cur, delims, Int32(1), out, Int32(8))
    assert_true(n == Int32(-1))
    buf.free(); cur.free(); delims.free(); out.free()

# ── parse_quoted_string ──────────────────────────────────────────────────

def test_parse_quoted_string_basic() raises:
    var buf = _buf('"hello"')
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var out = alloc[UInt8](32)
    var n = parse_quoted_string(buf, Int32(7), cur, out, Int32(32))
    assert_true(n == Int32(5))
    assert_true(String(unsafe_from_utf8_ptr=out.as_immutable()) == String("hello"))
    assert_true(cur[0] == Int32(7))
    buf.free(); cur.free(); out.free()

def test_parse_quoted_string_requires_opening_quote() raises:
    var buf = _buf("hello")
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var out = alloc[UInt8](32)
    var n = parse_quoted_string(buf, Int32(5), cur, out, Int32(32))
    assert_true(n == Int32(-1))
    buf.free(); cur.free(); out.free()

def test_parse_quoted_string_truncates_at_max_buf() raises:
    var buf = _buf('"abcdef"')
    var cur = alloc[Int32](1); cur[0] = Int32(0)
    var out = alloc[UInt8](4)
    var n = parse_quoted_string(buf, Int32(8), cur, out, Int32(4))
    assert_true(n == Int32(6))  # reports the true length...
    assert_true(String(unsafe_from_utf8_ptr=out.as_immutable()) == String("abc"))  # ...but out is capped
    buf.free(); cur.free(); out.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
