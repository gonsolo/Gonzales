from std.math import abs
from std.memory import alloc
from std.testing import assert_true, TestSuite
from gonzales.lexer import PbrtScanner, scanner_free
from gonzales.parse_types import SceneParseState
from gonzales.pbrt_parser import handle_curve_shape, handle_named_medium
from gonzales.material_builder import _psc_handle_make_named_material

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _scanner_from_string(s: String) -> UnsafePointer[PbrtScanner, MutExternalOrigin]:
    """Builds a PbrtScanner directly over an in-memory buffer — no temp file
    needed. Handlers are called mid-stream, exactly as parse_scene_file's
    directive dispatch does: the leading keyword (e.g. "MakeNamedMedium") is
    already consumed by the caller, so the snippet starts right after it."""
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

# ── handle_curve_shape: cp_cap growth past the old fixed 512-point cap ──────
# Task #42 fixed a silent-truncation bug where fixed scratch buffers (mesh
# points/indices, curve control points) would stop scan_floats early and
# desync the rest of the file's parse. The mesh case's default cap (65536
# floats) is impractical to exceed in a literal test string, but the curve
# case's cap (512 points) is not — build a curve with more control points
# than that and confirm every point is consumed correctly end-to-end.

def test_curve_shape_grows_past_default_control_point_cap() raises:
    comptime N_POINTS = 600  # > the 512-point default cp_cap
    var body = String('"point3 P" [ ')
    for i in range(N_POINTS):
        body += String(Float64(i)) + String(" 0 0  ")
    body += String('] "float width" [ 0.1 ]')

    var handle = _scanner_from_string(body)
    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    handle_curve_shape(handle, s_ptr)

    var expected_segments = N_POINTS - 3
    assert_true(len(s_ptr[0].curves_w0) == expected_segments)
    assert_true(len(s_ptr[0].curves_w1) == expected_segments)
    # 4 control points per segment, 3 floats each
    assert_true(len(s_ptr[0].curves_cp) == expected_segments * 4 * 3)
    scanner_free(handle)
    _ = s_ptr.take_pointee(); s_ptr.free()

# ── handle_named_medium: bracket-wrapped "string type" desync ──────────────
# Pre-existing bug (documented inline at handle_named_medium): "string type"
# is sometimes bracket-wrapped (["uniformgrid"]) — every other param already
# consumes its closing ']' when array-valued, but "type" didn't, which
# desynced the scanner right after the first param and silently dropped
# every subsequent one (density/nx/ny/nz/sigma_a/sigma_s all skipped).

def test_named_medium_bracket_wrapped_type_does_not_desync() raises:
    var body = String(
        '"fog" '
        + '"string type" [ "uniformgrid" ] '
        + '"float sigma_a" [ 1.0 1.0 1.0 ] '
        + '"float sigma_s" [ 2.0 2.0 2.0 ] '
        + '"integer nx" [ 2 ] "integer ny" [ 2 ] "integer nz" [ 2 ] '
        + '"float density" [ 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 ]'
    )
    var handle = _scanner_from_string(body)
    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    handle_named_medium(handle, s_ptr)

    assert_true(len(s_ptr[0].med_names) == 1)
    assert_true(s_ptr[0].med_names[0] == String("fog"))
    assert_true(s_ptr[0].grid_nx[0] == Int32(2))
    assert_true(s_ptr[0].grid_ny[0] == Int32(2))
    assert_true(s_ptr[0].grid_nz[0] == Int32(2))
    assert_true(_close(s_ptr[0].med_sa[0], Float32(1.0)))
    assert_true(_close(s_ptr[0].med_ss[0], Float32(2.0)))
    assert_true(len(s_ptr[0].grid_density) == 8)
    assert_true(_close(s_ptr[0].grid_density[0], Float32(0.1)))
    assert_true(_close(s_ptr[0].grid_density[7], Float32(0.8)))
    scanner_free(handle)
    _ = s_ptr.take_pointee(); s_ptr.free()

# ── MakeNamedMaterial: normalmap filename buffer overflow ──────────────────
# Pre-existing bug (fixed separately from the ParamScanner dedup): str_val
# was allocated with only 64 bytes but the normalmap/bumpmap branch scans up
# to PSC_FILE_MAX*2 = 512 bytes into it. Confirms a long path round-trips
# intact — NOTE this does not reliably re-catch the original overflow on
# revert: verified by hand that reverting str_val to 64 bytes does NOT fail
# this test (the allocator's real chunk size silently absorbs the overrun in
# this short-lived process), which is exactly why heap overflows are
# dangerous — "works until it doesn't." The lexer-level truncation mechanics
# behind this bug class are pinned deterministically in test_lexer.mojo
# instead; this test only guards the filename-handling logic itself.

def test_named_material_long_normalmap_path_is_not_truncated() raises:
    var long_name = String("a/")
    for _ in range(10):
        long_name += String("very-long-directory-name-segment/")
    long_name += String("texture.png")
    assert_true(long_name.byte_length() > 300)
    var body = (
        '"wood" "string type" [ "coateddiffuse" ] '
        + '"string normalmap" [ "' + long_name + '" ]'
    )
    var handle = _scanner_from_string(body)
    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    _psc_handle_make_named_material(handle, s_ptr, False)

    assert_true(len(s_ptr[0].named_materials) == 1)
    assert_true(s_ptr[0].named_materials[0].normal_tex_idx == Int32(0))
    assert_true(len(s_ptr[0].tex_files) == 1)
    assert_true(s_ptr[0].tex_files[0].endswith(long_name))
    scanner_free(handle)
    _ = s_ptr.take_pointee(); s_ptr.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
