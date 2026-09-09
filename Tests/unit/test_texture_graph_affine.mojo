"""pbrt "scale"/"mix" texture graphs folded into gonzales's affine form.

Every shape here is taken from a real corpus scene -- the comments name which.
The property under test is always the same: a graph over ONE imagemap must
resolve to `albedo = tex_bias + tex_scale * texel`, reproducing pbrt's
(1-amount)*tex1 + amount*tex2, while a graph multiplying two DIFFERENT
imagemaps must refuse to resolve (it would need a second lookup per shading
point) and fall back to flat albedo rather than rendering something wrong.
"""
from std.math import abs
from std.memory import alloc
from std.testing import assert_true, TestSuite
from gonzales.lexer import PbrtScanner, scanner_free
from gonzales.parse_types import SceneParseState
from gonzales.pbrt_parser import handle_texture
from gonzales.material_builder import _psc_handle_make_named_material

comptime _EPS = Float32(1e-5)


def _scanner_from_string(body: String) -> UnsafePointer[PbrtScanner, MutExternalOrigin]:
    var n = body.byte_length()
    var buf = alloc[UInt8](n + 1)
    for i in range(n):
        buf[i] = body.as_bytes()[i]
    buf[n] = UInt8(0)
    var handle = alloc[PbrtScanner](1)
    handle[0].buffer = buf
    handle[0].total_bytes = Int32(n)
    handle[0].cursor = Int32(0)
    handle[0].is_at_end = Int32(0)
    return handle


def _state() -> UnsafePointer[SceneParseState, MutExternalOrigin]:
    var s_ptr = alloc[SceneParseState](1)
    s_ptr.init_pointee_move(SceneParseState())
    return s_ptr


def _tex(s: UnsafePointer[SceneParseState, MutExternalOrigin], decl: String):
    var h = _scanner_from_string(decl)
    handle_texture(h, s)
    scanner_free(h)


def _mat(s: UnsafePointer[SceneParseState, MutExternalOrigin], decl: String):
    var h = _scanner_from_string(decl)
    _psc_handle_make_named_material(h, s, False)
    scanner_free(h)


def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < _EPS


def test_plain_imagemap_is_identity_affine() raises:
    """The no-graph baseline: scale 1, bias 0, so the fold is a no-op."""
    var s = _state()
    _tex(s, '"img" "spectrum" "imagemap" "string filename" [ "t.png" ]')
    _mat(s, '"m" "string type" [ "diffuse" ] "texture reflectance" [ "img" ]')
    var nm = s[0].named_materials[0]
    assert_true(nm.tex_idx == Int32(0))
    assert_true(_close(nm.tex_scale.r, Float32(1)))
    assert_true(_close(nm.tex_bias.r, Float32(0)))
    _ = s.take_pointee(); s.free()


def test_scale_of_imagemap_folds_into_scale() raises:
    """killeroos: `"texture tex" ["grid"] "float scale" [0.5]`."""
    var s = _state()
    _tex(s, '"img" "spectrum" "imagemap" "string filename" [ "t.png" ]')
    _tex(s, '"sc" "spectrum" "scale" "texture tex" [ "img" ] "float scale" [ 0.5 ]')
    _mat(s, '"m" "string type" [ "diffuse" ] "texture reflectance" [ "sc" ]')
    var nm = s[0].named_materials[0]
    assert_true(nm.tex_idx == Int32(0))
    assert_true(_close(nm.tex_scale.r, Float32(0.5)))
    assert_true(_close(nm.tex_bias.r, Float32(0)))
    _ = s.take_pointee(); s.free()


def test_mix_texture_with_constant_matches_pbrt_lerp() raises:
    """watercolor: `"texture tex1" [img] "rgb tex2" [c] "float amount" [a]`.

    pbrt evaluates (1-a)*tex1 + a*tex2, so the texture keeps weight (1-a) and
    the constant becomes a fixed offset a*c."""
    var s = _state()
    _tex(s, '"img" "spectrum" "imagemap" "string filename" [ "t.png" ]')
    _tex(s, '"mx" "spectrum" "mix" "texture tex1" [ "img" ]'
            + ' "rgb tex2" [ 0.8 0.4 0.2 ] "float amount" [ 0.25 ]')
    _mat(s, '"m" "string type" [ "diffuse" ] "texture reflectance" [ "mx" ]')
    var nm = s[0].named_materials[0]
    assert_true(nm.tex_idx == Int32(0))
    assert_true(_close(nm.tex_scale.r, Float32(0.75)))
    assert_true(_close(nm.tex_bias.r, Float32(0.25) * Float32(0.8)))
    assert_true(_close(nm.tex_bias.g, Float32(0.25) * Float32(0.4)))
    assert_true(_close(nm.tex_bias.b, Float32(0.25) * Float32(0.2)))
    _ = s.take_pointee(); s.free()


def test_mix_argument_order_is_not_symmetric() raises:
    """Swapping tex1/tex2 must swap which side the amount weights -- the
    obvious way to get pbrt's lerp backwards, and invisible in a render."""
    var s = _state()
    _tex(s, '"img" "spectrum" "imagemap" "string filename" [ "t.png" ]')
    _tex(s, '"mx" "spectrum" "mix" "rgb tex1" [ 0.8 0.4 0.2 ]'
            + ' "texture tex2" [ "img" ] "float amount" [ 0.25 ]')
    _mat(s, '"m" "string type" [ "diffuse" ] "texture reflectance" [ "mx" ]')
    var nm = s[0].named_materials[0]
    assert_true(nm.tex_idx == Int32(0))
    # Texture is now tex2, so it carries weight `amount`, not `1 - amount`.
    assert_true(_close(nm.tex_scale.r, Float32(0.25)))
    assert_true(_close(nm.tex_bias.r, Float32(0.75) * Float32(0.8)))
    _ = s.take_pointee(); s.free()


def test_mix_of_constants_driven_by_texture() raises:
    """villa/kroken: `"rgb tex1" [c1] "rgb tex2" [c2] "texture amount" [a]`.

    Expands to c1 + A(uv)*(c2 - c1): the amount map becomes the sampled
    texture, scaled by the colour difference and offset by c1."""
    var s = _state()
    _tex(s, '"amt" "float" "imagemap" "string filename" [ "a.png" ]')
    _tex(s, '"mx" "spectrum" "mix" "rgb tex1" [ 0.1 0.2 0.3 ]'
            + ' "rgb tex2" [ 0.9 0.7 0.5 ] "texture amount" [ "amt" ]')
    _mat(s, '"m" "string type" [ "diffuse" ] "texture reflectance" [ "mx" ]')
    var nm = s[0].named_materials[0]
    assert_true(nm.tex_idx == Int32(0))
    assert_true(_close(nm.tex_scale.r, Float32(0.8)))
    assert_true(_close(nm.tex_scale.g, Float32(0.5)))
    assert_true(_close(nm.tex_scale.b, Float32(0.2)))
    assert_true(_close(nm.tex_bias.r, Float32(0.1)))
    assert_true(_close(nm.tex_bias.b, Float32(0.3)))
    _ = s.take_pointee(); s.free()


def test_constant_tinted_by_texture_scale() raises:
    """kroken's book covers: `"rgb tex" [c] "texture scale" [t]` -- the
    operands are the mirror image of the killeroos shape, and reading only
    the string `tex` / float `scale` misses this entirely."""
    var s = _state()
    _tex(s, '"proj" "float" "imagemap" "string filename" [ "p.png" ]')
    _tex(s, '"cover" "spectrum" "scale" "rgb tex" [ 0.4 0.3 0.2 ]'
            + ' "texture scale" [ "proj" ]')
    _mat(s, '"m" "string type" [ "diffuse" ] "texture reflectance" [ "cover" ]')
    var nm = s[0].named_materials[0]
    assert_true(nm.tex_idx == Int32(0))
    assert_true(_close(nm.tex_scale.r, Float32(0.4)))
    assert_true(_close(nm.tex_scale.g, Float32(0.3)))
    assert_true(_close(nm.tex_bias.r, Float32(0)))
    _ = s.take_pointee(); s.free()


def test_nested_scale_of_mix_composes() raises:
    """watercolor chains these (a mix whose tex1 is itself a scaled imagemap).
    Affine composes, so the whole chain must still collapse to one lookup."""
    var s = _state()
    _tex(s, '"img" "spectrum" "imagemap" "string filename" [ "t.png" ]')
    _tex(s, '"sc" "spectrum" "scale" "texture tex" [ "img" ] "float scale" [ 2.0 ]')
    _tex(s, '"mx" "spectrum" "mix" "texture tex1" [ "sc" ]'
            + ' "rgb tex2" [ 1.0 1.0 1.0 ] "float amount" [ 0.5 ]')
    _mat(s, '"m" "string type" [ "diffuse" ] "texture reflectance" [ "mx" ]')
    var nm = s[0].named_materials[0]
    assert_true(nm.tex_idx == Int32(0))
    # 0.5 * (2 * texel) + 0.5 * 1
    assert_true(_close(nm.tex_scale.r, Float32(1.0)))
    assert_true(_close(nm.tex_bias.r, Float32(0.5)))
    _ = s.take_pointee(); s.free()


def test_product_of_two_textures_refuses_to_resolve() raises:
    """kroken's Brick_Color-muddled multiplies two independent imagemaps.
    One lookup cannot express that, so the graph must NOT resolve -- the
    material falls back to flat albedo (and warns) instead of silently
    rendering one of the two maps as if it were the product."""
    var s = _state()
    _tex(s, '"a" "spectrum" "imagemap" "string filename" [ "a.png" ]')
    _tex(s, '"b" "float" "imagemap" "string filename" [ "b.png" ]')
    _tex(s, '"prod" "spectrum" "scale" "texture tex" [ "a" ] "texture scale" [ "b" ]')
    _mat(s, '"m" "string type" [ "diffuse" ] "texture reflectance" [ "prod" ]')
    var nm = s[0].named_materials[0]
    assert_true(nm.tex_idx == Int32(-1))
    _ = s.take_pointee(); s.free()


def test_mix_over_two_different_textures_refuses_to_resolve() raises:
    """The mix twin of the case above: blending two different imagemaps by a
    constant still needs both lookups."""
    var s = _state()
    _tex(s, '"a" "spectrum" "imagemap" "string filename" [ "a.png" ]')
    _tex(s, '"b" "spectrum" "imagemap" "string filename" [ "b.png" ]')
    _tex(s, '"mx" "spectrum" "mix" "texture tex1" [ "a" ]'
            + ' "texture tex2" [ "b" ] "float amount" [ 0.5 ]')
    _mat(s, '"m" "string type" [ "diffuse" ] "texture reflectance" [ "mx" ]')
    var nm = s[0].named_materials[0]
    assert_true(nm.tex_idx == Int32(-1))
    _ = s.take_pointee(); s.free()


def test_mix_of_two_constants_collapses_to_flat_albedo() raises:
    """No texture anywhere in the graph: it must fold to a plain colour, not
    leave a dangling texture index."""
    var s = _state()
    _tex(s, '"mx" "spectrum" "mix" "rgb tex1" [ 0.0 0.0 0.0 ]'
            + ' "rgb tex2" [ 1.0 0.5 0.25 ] "float amount" [ 0.5 ]')
    _mat(s, '"m" "string type" [ "diffuse" ] "texture reflectance" [ "mx" ]')
    var nm = s[0].named_materials[0]
    assert_true(nm.tex_idx == Int32(-1))
    assert_true(_close(nm.albedo.r, Float32(0.5)))
    assert_true(_close(nm.albedo.g, Float32(0.25)))
    assert_true(_close(nm.albedo.b, Float32(0.125)))
    _ = s.take_pointee(); s.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
