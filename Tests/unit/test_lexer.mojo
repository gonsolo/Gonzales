from std.math import abs
from std.memory import alloc
from std.testing import assert_true, assert_false, assert_equal, TestSuite
from gonzales.lexer import (
    scan_int, scan_float, scan_floats, scan_ints, count_floats, count_ints,
    scan_char, scan_token, parse_quoted_string,
    _psc_streq, _psc_strncmp, _psc_strncpy,
    _psc_type_is_float, _psc_type_is_int, _psc_type_is_str, _psc_type_is_blackbody,
    _psc_blackbody_to_rgb,
    PbrtScanner, scanner_free, ParamScanner,
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

def _buf0(s: String) -> UnsafePointer[UInt8, MutAnyOrigin]:
    """Like _buf, but null-terminated — needed by the _psc_streq/_psc_strncpy/
    _psc_type_is_* helpers, which (unlike scan_int/scan_float/...) take no
    explicit length and instead scan/index until a 0 byte or a fixed offset."""
    var n = s.byte_length()
    var b = alloc[UInt8](n + 1)
    for i in range(n):
        b[i] = s.as_bytes()[i]
    b[n] = UInt8(0)
    return b

def _scanner_from_string(s: String) -> UnsafePointer[PbrtScanner, MutAnyOrigin]:
    """Builds a PbrtScanner directly over an in-memory buffer, mirroring the
    helper of the same name in test_parser_integration.mojo — needed to drive
    ParamScanner, which is only ever handed a PbrtScanner handle, never raw
    bytes."""
    var n = s.byte_length()
    var buf = alloc[UInt8](n + 1)
    for i in range(n):
        buf[i] = s.as_bytes()[i]
    buf[n] = UInt8(0)
    var handle = alloc[PbrtScanner](1)
    handle[0].buffer = buf
    handle[0].total_bytes = Int32(n)
    handle[0].cursor = Int32(0)
    handle[0].is_at_end = Int32(0)
    return handle

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

# ── _psc_streq ───────────────────────────────────────────────────────────

def test_psc_streq_equal_strings_match() raises:
    var buf = _buf0("hello")
    assert_true(_psc_streq(buf, "hello"))
    buf.free()

def test_psc_streq_different_strings_do_not_match() raises:
    var buf = _buf0("hello")
    assert_false(_psc_streq(buf, "world"))
    buf.free()

def test_psc_streq_stops_at_null_terminator() raises:
    """_psc_streq must stop comparing as soon as it hits the buffer's null
    terminator rather than reading (or caring about) whatever garbage bytes
    follow it — build a buffer where the bytes past the logical string
    spell something else entirely and confirm the match still succeeds."""
    var buf = alloc[UInt8](6)
    buf[0] = UInt8(104)  # 'h'
    buf[1] = UInt8(105)  # 'i'
    buf[2] = UInt8(0)    # terminator
    buf[3] = UInt8(88)   # 'X' — garbage past the logical string
    buf[4] = UInt8(88)   # 'X'
    buf[5] = UInt8(0)
    assert_true(_psc_streq(buf, "hi"))
    buf.free()

# ── _psc_strncmp ─────────────────────────────────────────────────────────

def test_psc_strncmp_zero_when_first_n_bytes_match() raises:
    """Returns 0 (equal) when only the first n bytes are compared, even
    though a's full string continues on past what the literal describes."""
    var buf = _buf0("hello world")
    assert_true(_psc_strncmp(buf, "hello", 5) == 0)
    buf.free()

def test_psc_strncmp_nonzero_when_bytes_differ_within_n() raises:
    var buf = _buf0("hexlo")
    assert_true(_psc_strncmp(buf, "hello", 5) != 0)
    buf.free()

# ── _psc_strncpy ─────────────────────────────────────────────────────────

def test_psc_strncpy_truncates_and_null_terminates() raises:
    """Bounded copy: at most n-1 source bytes are copied, and the destination
    is always null-terminated within the n-byte capacity."""
    var src = _buf0("hello")
    var dst = alloc[UInt8](3)
    _psc_strncpy(dst, src, Int32(3))
    assert_true(dst[0] == UInt8(104))  # 'h'
    assert_true(dst[1] == UInt8(101))  # 'e'
    assert_true(dst[2] == UInt8(0))
    src.free(); dst.free()

def test_psc_strncpy_full_copy_when_capacity_suffices() raises:
    var src = _buf0("hi")
    var dst = alloc[UInt8](8)
    _psc_strncpy(dst, src, Int32(8))
    assert_true(dst[0] == UInt8(104))  # 'h'
    assert_true(dst[1] == UInt8(105))  # 'i'
    assert_true(dst[2] == UInt8(0))
    src.free(); dst.free()

# ── _psc_type_is_* ───────────────────────────────────────────────────────

def test_psc_type_is_float_recognizes_float_like_tags() raises:
    """Every one of these tags names a param type stored as 3 floats (or 1,
    for plain 'float'); 'sp' is the 2-byte prefix of 'spectrum'."""
    var b_f = _buf0("float"); assert_true(_psc_type_is_float(b_f)); b_f.free()
    var b_r = _buf0("rgb"); assert_true(_psc_type_is_float(b_r)); b_r.free()
    var b_c = _buf0("color"); assert_true(_psc_type_is_float(b_c)); b_c.free()
    var b_n = _buf0("normal"); assert_true(_psc_type_is_float(b_n)); b_n.free()
    var b_p = _buf0("point3"); assert_true(_psc_type_is_float(b_p)); b_p.free()
    var b_v = _buf0("vector3"); assert_true(_psc_type_is_float(b_v)); b_v.free()
    var b_sp = _buf0("spectrum"); assert_true(_psc_type_is_float(b_sp)); b_sp.free()

def test_psc_type_is_float_rejects_unrelated_tags() raises:
    """'blackbody' is intentionally excluded (1 float, not 3) per the inline
    NOTE in lexer.mojo, as are 'integer' and 'string'."""
    var b_i = _buf0("integer"); assert_false(_psc_type_is_float(b_i)); b_i.free()
    var b_s = _buf0("string"); assert_false(_psc_type_is_float(b_s)); b_s.free()
    var b_bb = _buf0("blackbody"); assert_false(_psc_type_is_float(b_bb)); b_bb.free()

def test_psc_type_is_int_recognizes_integer_tag() raises:
    var buf = _buf0("integer")
    assert_true(_psc_type_is_int(buf))
    buf.free()

def test_psc_type_is_int_rejects_unrelated_tags() raises:
    var b_f = _buf0("float"); assert_false(_psc_type_is_int(b_f)); b_f.free()
    var b_s = _buf0("string"); assert_false(_psc_type_is_int(b_s)); b_s.free()

def test_psc_type_is_str_recognizes_string_and_texture_tags() raises:
    var b_s = _buf0("string"); assert_true(_psc_type_is_str(b_s)); b_s.free()
    var b_t = _buf0("texture"); assert_true(_psc_type_is_str(b_t)); b_t.free()

def test_psc_type_is_str_rejects_unrelated_tags() raises:
    var b_f = _buf0("float"); assert_false(_psc_type_is_str(b_f)); b_f.free()
    var b_i = _buf0("integer"); assert_false(_psc_type_is_str(b_i)); b_i.free()

def test_psc_type_is_blackbody_recognizes_tag() raises:
    var buf = _buf0("blackbody")
    assert_true(_psc_type_is_blackbody(buf))
    buf.free()

def test_psc_type_is_blackbody_rejects_unrelated_tags() raises:
    var b_f = _buf0("float"); assert_false(_psc_type_is_blackbody(b_f)); b_f.free()
    var b_s = _buf0("string"); assert_false(_psc_type_is_blackbody(b_s)); b_s.free()

# ── _psc_blackbody_to_rgb ────────────────────────────────────────────────

def test_psc_blackbody_to_rgb_low_temp_is_red_dominant() raises:
    """~1000K (candlelight) sits at the warm end of the Planckian locus; the
    Mitchell-Charity approximation should give a clearly red-dominant color
    (and in fact zero blue, per the temp <= 1900 branch)."""
    var rgb = alloc[Float32](3)
    _psc_blackbody_to_rgb(Float32(1000.0), rgb)
    assert_true(rgb[0] > rgb[2])
    rgb.free()

def test_psc_blackbody_to_rgb_high_temp_is_blue_dominant() raises:
    """~10000K+ sits at the cool end; unlike the low-temp case, blue should
    now equal-or-exceed red — a directional sanity check on the empirical
    fit, not an exact numeric match (no simple closed form exists)."""
    var rgb = alloc[Float32](3)
    _psc_blackbody_to_rgb(Float32(12000.0), rgb)
    assert_true(rgb[2] >= rgb[0])
    rgb.free()

# ── ParamScanner ─────────────────────────────────────────────────────────

def test_param_scanner_walks_typed_params_then_reports_exhausted() raises:
    """Mirrors the `"typename" [ value ]` loop every PBRT directive parses
    via ParamScanner (added this session to dedupe that boilerplate);
    confirms next() walks header-by-header reporting the right name/type/
    is_array, and returns False once the params run out."""
    var body = String('"integer maxdepth" [ 4 ] "float radius" [ 2.5 ]')
    var handle = _scanner_from_string(body)
    var ps = ParamScanner()

    assert_true(ps.next(handle))
    assert_true(ps.name_is("maxdepth"))
    assert_true(ps.is_int())
    assert_false(ps.is_float())
    assert_true(ps.is_array != Int32(0))
    ps.skip(handle)

    assert_true(ps.next(handle))
    assert_true(ps.name_is("radius"))
    assert_true(ps.is_float())
    assert_false(ps.is_int())
    assert_true(ps.is_array != Int32(0))
    ps.skip(handle)

    assert_false(ps.next(handle))

    scanner_free(handle)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
