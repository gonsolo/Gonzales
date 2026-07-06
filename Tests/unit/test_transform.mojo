from std.math import abs
from std.memory import alloc
from std.testing import assert_true, assert_false, TestSuite
from gonzales.transform import matrix_multiply, matrix_invert, transform_points, transform_normals

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _identity(m: UnsafePointer[Float32, MutAnyOrigin]):
    for i in range(16):
        m[i] = Float32(0)
    m[0] = Float32(1)
    m[5] = Float32(1)
    m[10] = Float32(1)
    m[15] = Float32(1)

def _translation(m: UnsafePointer[Float32, MutAnyOrigin], tx: Float32, ty: Float32, tz: Float32):
    # Column-major: flat[col*4+row] = matrix[row,col]. Translation lives in
    # column 3 (indices 12,13,14) — matches _psc_handle_translate in pbrt_parser.mojo.
    _identity(m)
    m[12] = tx; m[13] = ty; m[14] = tz

def _scale(m: UnsafePointer[Float32, MutAnyOrigin], sx: Float32, sy: Float32, sz: Float32):
    _identity(m)
    m[0] = sx; m[5] = sy; m[10] = sz

def _mat_close(a: UnsafePointer[Float32, MutAnyOrigin], b: UnsafePointer[Float32, MutAnyOrigin]) -> Bool:
    for i in range(16):
        if not _close(a[i], b[i]):
            return False
    return True

# ── matrix_multiply ─────────────────────────────────────────────────────────

def test_matrix_multiply_identity_is_neutral() raises:
    """Identity * M must return M unchanged — the base case CTM concatenation
    relies on (a fresh CTM starts as the identity)."""
    var id = alloc[Float32](16); _identity(id)
    var m = alloc[Float32](16)
    _translation(m, Float32(1.0), Float32(2.0), Float32(3.0))
    var result = alloc[Float32](16)
    matrix_multiply(id, m, result)
    assert_true(_mat_close(result, m))
    id.free(); m.free(); result.free()

def test_matrix_multiply_translation_composition() raises:
    """T(a) * T(b) must equal T(a+b) — translations compose additively."""
    var t1 = alloc[Float32](16); _translation(t1, Float32(1.0), Float32(2.0), Float32(3.0))
    var t2 = alloc[Float32](16); _translation(t2, Float32(4.0), Float32(-1.0), Float32(0.5))
    var result = alloc[Float32](16)
    matrix_multiply(t1, t2, result)
    var expected = alloc[Float32](16)
    _translation(expected, Float32(5.0), Float32(1.0), Float32(3.5))
    assert_true(_mat_close(result, expected))
    t1.free(); t2.free(); result.free(); expected.free()

def test_matrix_multiply_matches_hand_computed_case() raises:
    """A hand-computed 4x4 * 4x4 case, independent of any translate/scale
    shortcut, to pin down the column-major index arithmetic itself."""
    # a = row-major [[1,2,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] stored column-major
    var a = alloc[Float32](16); _identity(a)
    a[4] = Float32(2.0)  # row0,col1 = 2  -> flat[col*4+row] = flat[1*4+0] = flat[4]
    var b = alloc[Float32](16); _identity(b)
    b[12] = Float32(3.0); b[13] = Float32(5.0); b[14] = Float32(7.0)
    var result = alloc[Float32](16)
    matrix_multiply(a, b, result)
    # Expect: a * b = translate by (3 + 2*5, 5, 7) = (13, 5, 7) in col 3,
    # since row0 of a is [1,2,0,0] dotted with b's translation column (3,5,7,1).
    var expected = alloc[Float32](16); _identity(expected)
    expected[4] = Float32(2.0)
    expected[12] = Float32(13.0); expected[13] = Float32(5.0); expected[14] = Float32(7.0)
    assert_true(_mat_close(result, expected))
    a.free(); b.free(); result.free(); expected.free()

# ── matrix_invert ────────────────────────────────────────────────────────────

def test_matrix_invert_of_identity_is_identity() raises:
    var id = alloc[Float32](16); _identity(id)
    var result = alloc[Float32](16)
    var ok = matrix_invert(id, result)
    assert_true(ok == Int32(1))
    assert_true(_mat_close(result, id))
    id.free(); result.free()

def test_matrix_invert_translation() raises:
    """Inverse of T(tx,ty,tz) is exactly T(-tx,-ty,-tz)."""
    var t = alloc[Float32](16); _translation(t, Float32(2.0), Float32(-3.0), Float32(5.0))
    var inv = alloc[Float32](16)
    var ok = matrix_invert(t, inv)
    assert_true(ok == Int32(1))
    var expected = alloc[Float32](16); _translation(expected, Float32(-2.0), Float32(3.0), Float32(-5.0))
    assert_true(_mat_close(inv, expected))
    t.free(); inv.free(); expected.free()

def test_matrix_invert_round_trip_matches_original() raises:
    """Matrix_invert(matrix_invert(M)) == M for an invertible translate+scale
    composition, i.e. inversion is its own involution."""
    var s = alloc[Float32](16); _scale(s, Float32(2.0), Float32(4.0), Float32(0.5))
    var t = alloc[Float32](16); _translation(t, Float32(1.0), Float32(2.0), Float32(3.0))
    var m = alloc[Float32](16)
    matrix_multiply(t, s, m)  # composed invertible matrix
    var inv1 = alloc[Float32](16)
    var ok1 = matrix_invert(m, inv1)
    assert_true(ok1 == Int32(1))
    var inv2 = alloc[Float32](16)
    var ok2 = matrix_invert(inv1, inv2)
    assert_true(ok2 == Int32(1))
    assert_true(_mat_close(inv2, m))
    s.free(); t.free(); m.free(); inv1.free(); inv2.free()

def test_matrix_invert_times_original_is_identity() raises:
    """M * M^-1 == identity, the defining property of matrix inversion."""
    var s = alloc[Float32](16); _scale(s, Float32(2.0), Float32(4.0), Float32(0.5))
    var t = alloc[Float32](16); _translation(t, Float32(1.0), Float32(2.0), Float32(3.0))
    var m = alloc[Float32](16)
    matrix_multiply(t, s, m)
    var inv = alloc[Float32](16)
    var ok = matrix_invert(m, inv)
    assert_true(ok == Int32(1))
    var product = alloc[Float32](16)
    matrix_multiply(m, inv, product)
    var id = alloc[Float32](16); _identity(id)
    assert_true(_mat_close(product, id))
    s.free(); t.free(); m.free(); inv.free(); product.free(); id.free()

def test_matrix_invert_singular_writes_identity_and_reports_failure() raises:
    """A singular (all-zero) matrix must fail cleanly: return 0 and leave the
    identity in `result`, never garbage — callers rely on this fallback."""
    var singular = alloc[Float32](16)
    for i in range(16):
        singular[i] = Float32(0)
    var result = alloc[Float32](16)
    var ok = matrix_invert(singular, result)
    assert_true(ok == Int32(0))
    var id = alloc[Float32](16); _identity(id)
    assert_true(_mat_close(result, id))
    singular.free(); result.free(); id.free()

# ── transform_points ─────────────────────────────────────────────────────────

def test_transform_points_identity_leaves_points_unchanged() raises:
    var id = alloc[Float32](16); _identity(id)
    var pts_in = alloc[Float32](4)
    pts_in[0] = Float32(1.0); pts_in[1] = Float32(2.0); pts_in[2] = Float32(3.0); pts_in[3] = Float32(1.0)
    var pts_out = alloc[Float32](4)
    transform_points(id, pts_in, Int32(1), pts_out)
    assert_true(_close(pts_out[0], Float32(1.0)))
    assert_true(_close(pts_out[1], Float32(2.0)))
    assert_true(_close(pts_out[2], Float32(3.0)))
    id.free(); pts_in.free(); pts_out.free()

def test_transform_points_translation_moves_by_exact_vector() raises:
    var t = alloc[Float32](16); _translation(t, Float32(10.0), Float32(-5.0), Float32(2.0))
    var pts_in = alloc[Float32](4)
    pts_in[0] = Float32(1.0); pts_in[1] = Float32(1.0); pts_in[2] = Float32(1.0); pts_in[3] = Float32(1.0)
    var pts_out = alloc[Float32](4)
    transform_points(t, pts_in, Int32(1), pts_out)
    assert_true(_close(pts_out[0], Float32(11.0)))
    assert_true(_close(pts_out[1], Float32(-4.0)))
    assert_true(_close(pts_out[2], Float32(3.0)))
    t.free(); pts_in.free(); pts_out.free()

def test_transform_points_scale_scales_coordinates_exactly() raises:
    var s = alloc[Float32](16); _scale(s, Float32(2.0), Float32(3.0), Float32(-1.0))
    var pts_in = alloc[Float32](8)
    pts_in[0] = Float32(1.0); pts_in[1] = Float32(2.0); pts_in[2] = Float32(3.0); pts_in[3] = Float32(1.0)
    pts_in[4] = Float32(-2.0); pts_in[5] = Float32(0.5); pts_in[6] = Float32(4.0); pts_in[7] = Float32(1.0)
    var pts_out = alloc[Float32](8)
    transform_points(s, pts_in, Int32(2), pts_out)
    assert_true(_close(pts_out[0], Float32(2.0)))
    assert_true(_close(pts_out[1], Float32(6.0)))
    assert_true(_close(pts_out[2], Float32(-3.0)))
    assert_true(_close(pts_out[4], Float32(-4.0)))
    assert_true(_close(pts_out[5], Float32(1.5)))
    assert_true(_close(pts_out[6], Float32(-4.0)))
    s.free(); pts_in.free(); pts_out.free()

# ── transform_normals ────────────────────────────────────────────────────────

def test_transform_normals_identity_leaves_normal_unchanged() raises:
    """Inv_matrix here is the inverse of the forward transform; for the
    identity transform the inverse is itself, so the normal passes through."""
    var id = alloc[Float32](16); _identity(id)
    var n_in = alloc[Float32](3)
    n_in[0] = Float32(0.0); n_in[1] = Float32(1.0); n_in[2] = Float32(0.0)
    var n_out = alloc[Float32](3)
    transform_normals(id, n_in, Int32(1), n_out)
    assert_true(_close(n_out[0], Float32(0.0)))
    assert_true(_close(n_out[1], Float32(1.0)))
    assert_true(_close(n_out[2], Float32(0.0)))
    id.free(); n_in.free(); n_out.free()

def test_transform_normals_uniform_scale_inverse_rescales_normal() raises:
    """Normals transform by (M^-1)^T. For a uniform scale S=diag(k,k,k), the
    caller passes inv_matrix = S^-1 = diag(1/k,1/k,1/k); transform_normals
    applies its transpose, so the normal is scaled by exactly 1/k."""
    var k = Float32(2.0)
    var inv_s = alloc[Float32](16); _scale(inv_s, Float32(1.0) / k, Float32(1.0) / k, Float32(1.0) / k)
    var n_in = alloc[Float32](3)
    n_in[0] = Float32(0.0); n_in[1] = Float32(0.0); n_in[2] = Float32(1.0)
    var n_out = alloc[Float32](3)
    transform_normals(inv_s, n_in, Int32(1), n_out)
    assert_true(_close(n_out[0], Float32(0.0)))
    assert_true(_close(n_out[1], Float32(0.0)))
    assert_true(_close(n_out[2], Float32(0.5)))
    inv_s.free(); n_in.free(); n_out.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
