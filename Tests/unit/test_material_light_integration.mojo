from std.math import abs
from std.memory import alloc
from std.testing import assert_true, TestSuite
from gonzales.lexer import PbrtScanner, scanner_free
from gonzales.parse_types import SceneParseState
from gonzales.light_builder import handle_light_source, _psc_handle_area_light_source
from gonzales.material_builder import _psc_handle_make_named_material
from gonzales.geometry import MatKind

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _scanner_from_string(s: String) -> UnsafePointer[PbrtScanner, MutAnyOrigin]:
    """Same helper as test_parser_integration.mojo: builds a PbrtScanner over
    an in-memory buffer, positioned right after the directive keyword, exactly
    as parse_scene_file's directive dispatch hands off to a handler."""
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

# ── handle_light_source: distant ────────────────────────────────────────────
# direction = -from, normalized (see comment in light_builder.mojo: "from"
# describes where the light comes FROM, so the direction it travels is the
# negation). from=(0,0,5) -> direction=(0,0,-1).

def test_light_source_distant_direction_and_rgb() raises:
    var body = String(
        '"distant" "point3 from" [ 0 0 5 ] "rgb L" [ 2 3 4 ]'
    )
    var handle = _scanner_from_string(body)
    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    handle_light_source(handle, s_ptr)

    assert_true(len(s_ptr[0].distant_dirs) == 3)
    assert_true(_close(s_ptr[0].distant_dirs[0], Float32(0.0)))
    assert_true(_close(s_ptr[0].distant_dirs[1], Float32(0.0)))
    assert_true(_close(s_ptr[0].distant_dirs[2], Float32(-1.0)))
    assert_true(_close(s_ptr[0].distant_rgbs[0], Float32(2.0)))
    assert_true(_close(s_ptr[0].distant_rgbs[1], Float32(3.0)))
    assert_true(_close(s_ptr[0].distant_rgbs[2], Float32(4.0)))
    scanner_free(handle)
    _ = s_ptr.take_pointee(); s_ptr.free()

# ── handle_light_source: point ──────────────────────────────────────────────
# "from" position is passed through the current CTM; a freshly-constructed
# SceneParseState's CTM is identity, so the position must round-trip exactly.

def test_light_source_point_position_and_rgb() raises:
    var body = String(
        '"point" "point3 from" [ 1 2 3 ] "rgb I" [ 0.5 0.5 0.5 ]'
    )
    var handle = _scanner_from_string(body)
    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    handle_light_source(handle, s_ptr)

    assert_true(len(s_ptr[0].point_pos) == 3)
    assert_true(_close(s_ptr[0].point_pos[0], Float32(1.0)))
    assert_true(_close(s_ptr[0].point_pos[1], Float32(2.0)))
    assert_true(_close(s_ptr[0].point_pos[2], Float32(3.0)))
    assert_true(_close(s_ptr[0].point_rgbs[0], Float32(0.5)))
    assert_true(_close(s_ptr[0].point_rgbs[1], Float32(0.5)))
    assert_true(_close(s_ptr[0].point_rgbs[2], Float32(0.5)))
    scanner_free(handle)
    _ = s_ptr.take_pointee(); s_ptr.free()

# ── handle_light_source: infinite, no filename ──────────────────────────────
# Without a filename, inf_tex_idx must be -1 (constant-color env light) and
# inf_rgb holds the (scaled) constant color.

def test_light_source_infinite_no_filename() raises:
    var body = String('"infinite" "rgb L" [ 0.1 0.2 0.3 ]')
    var handle = _scanner_from_string(body)
    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    handle_light_source(handle, s_ptr)

    assert_true(len(s_ptr[0].inf_tex_idx) == 1)
    assert_true(s_ptr[0].inf_tex_idx[0] == Int32(-1))
    assert_true(_close(s_ptr[0].inf_rgb[0], Float32(0.1)))
    assert_true(_close(s_ptr[0].inf_rgb[1], Float32(0.2)))
    assert_true(_close(s_ptr[0].inf_rgb[2], Float32(0.3)))
    scanner_free(handle)
    _ = s_ptr.take_pointee(); s_ptr.free()

# ── handle_light_source: infinite, with filename ────────────────────────────
# With a filename, a "__inf" texture entry is registered and inf_tex_idx
# points at it; the file path is scene_dir + filename (scene_dir is "" for a
# freshly-constructed SceneParseState).

def test_light_source_infinite_with_filename() raises:
    var body = String(
        '"infinite" "string filename" [ "textures/sky.exr" ] "rgb L" [ 1 1 1 ]'
    )
    var handle = _scanner_from_string(body)
    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    handle_light_source(handle, s_ptr)

    assert_true(len(s_ptr[0].inf_tex_idx) == 1)
    assert_true(s_ptr[0].inf_tex_idx[0] == Int32(0))
    assert_true(len(s_ptr[0].tex_files) == 1)
    assert_true(s_ptr[0].tex_files[0] == String("textures/sky.exr"))
    assert_true(_close(s_ptr[0].inf_rgb[0], Float32(1.0)))
    scanner_free(handle)
    _ = s_ptr.take_pointee(); s_ptr.free()

# ── _psc_handle_area_light_source: rgb L combined with float scale ─────────

def test_area_light_source_rgb_l_scaled() raises:
    var body = String('"diffuse" "rgb L" [ 2 4 6 ] "float scale" [ 0.5 ]')
    var handle = _scanner_from_string(body)
    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    _psc_handle_area_light_source(handle, s_ptr)

    assert_true(s_ptr[0].cur_attr.is_alight)
    assert_true(_close(s_ptr[0].cur_attr.al_rgb.r, Float32(1.0)))
    assert_true(_close(s_ptr[0].cur_attr.al_rgb.g, Float32(2.0)))
    assert_true(_close(s_ptr[0].cur_attr.al_rgb.b, Float32(3.0)))
    scanner_free(handle)
    _ = s_ptr.take_pointee(); s_ptr.free()

# ── _psc_handle_make_named_material: conductor with explicit rgb eta/k ─────
# has_spectral_conductor path computes per-channel Fresnel F0 = ((eta-1)^2+k^2)
# / ((eta+1)^2+k^2) and stores it in albedo.

def test_named_material_conductor_rgb_eta_k_fresnel() raises:
    var body = (
        '"copper" "string type" [ "conductor" ] '
        + '"rgb eta" [ 0.2 0.9 1.1 ] "rgb k" [ 3.0 2.5 2.0 ]'
    )
    var handle = _scanner_from_string(body)
    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    _psc_handle_make_named_material(handle, s_ptr, False)

    assert_true(len(s_ptr[0].named_materials) == 1)
    assert_true(s_ptr[0].named_materials[0].kind == MatKind.conductor)
    assert_true(_close(s_ptr[0].named_materials[0].albedo.r, Float32(0.923372)))
    assert_true(_close(s_ptr[0].named_materials[0].albedo.g, Float32(0.634888)))
    assert_true(_close(s_ptr[0].named_materials[0].albedo.b, Float32(0.476813)))
    scanner_free(handle)
    _ = s_ptr.take_pointee(); s_ptr.free()

# ── _psc_handle_make_named_material: dielectric with float eta ─────────────

def test_named_material_dielectric_float_eta() raises:
    var body = '"glass" "string type" [ "dielectric" ] "float eta" [ 1.33 ]'
    var handle = _scanner_from_string(body)
    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    _psc_handle_make_named_material(handle, s_ptr, False)

    assert_true(len(s_ptr[0].named_materials) == 1)
    assert_true(s_ptr[0].named_materials[0].kind == MatKind.dielectric)
    assert_true(_close(s_ptr[0].named_materials[0].ior, Float32(1.33)))
    scanner_free(handle)
    _ = s_ptr.take_pointee(); s_ptr.free()

# ── _psc_handle_make_named_material: mix of two named materials ────────────

def test_named_material_mix_names_and_amount() raises:
    var body = (
        '"blend" "string type" [ "mix" ] '
        + '"string materials" [ "matA" "matB" ] "float amount" [ 0.25 ]'
    )
    var handle = _scanner_from_string(body)
    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    _psc_handle_make_named_material(handle, s_ptr, False)

    assert_true(len(s_ptr[0].named_materials) == 1)
    assert_true(s_ptr[0].named_materials[0].kind == MatKind.mix)
    assert_true(s_ptr[0].named_materials[0].mix_name1 == String("matA"))
    assert_true(s_ptr[0].named_materials[0].mix_name2 == String("matB"))
    assert_true(_close(s_ptr[0].named_materials[0].mix_amount, Float32(0.25)))
    scanner_free(handle)
    _ = s_ptr.take_pointee(); s_ptr.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
