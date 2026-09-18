from std.math import abs
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_false, TestSuite
from gonzales.transform import matrix_multiply, matrix_invert, transform_points, transform_normals

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

def _identity(m: Pointer[Float32, MutUntrackedOrigin]):
    for i in range(16):
        m[unsafe_offset=i] = Float32(0)
    m[unsafe_offset=0] = Float32(1)
    m[unsafe_offset=5] = Float32(1)
    m[unsafe_offset=10] = Float32(1)
    m[unsafe_offset=15] = Float32(1)

def _translation(m: Pointer[Float32, MutUntrackedOrigin], tx: Float32, ty: Float32, tz: Float32):
    # Column-major: flat[col*4+row] = matrix[row,col]. Translation lives in
    # column 3 (indices 12,13,14) — matches _psc_handle_translate in pbrt_parser.mojo.
    _identity(m)
    m[unsafe_offset=12] = tx; m[unsafe_offset=13] = ty; m[unsafe_offset=14] = tz

def _scale(m: Pointer[Float32, MutUntrackedOrigin], sx: Float32, sy: Float32, sz: Float32):
    _identity(m)
    m[unsafe_offset=0] = sx; m[unsafe_offset=5] = sy; m[unsafe_offset=10] = sz

def _mat_close(a: Pointer[Float32, MutUntrackedOrigin], b: Pointer[Float32, MutUntrackedOrigin]) -> Bool:
    for i in range(16):
        if not _close(a[unsafe_offset=i], b[unsafe_offset=i]):
            return False
    return True

# ── matrix_multiply ─────────────────────────────────────────────────────────

def test_matrix_multiply_identity_is_neutral() raises:
    """Identity * M must return M unchanged — the base case CTM concatenation
    relies on (a fresh CTM starts as the identity)."""
    var id = unsafe_alloc[Float32](16); _identity(id)
    var m = unsafe_alloc[Float32](16)
    _translation(m, Float32(1.0), Float32(2.0), Float32(3.0))
    var result = unsafe_alloc[Float32](16)
    matrix_multiply(id, m, result)
    assert_true(_mat_close(result, m))
    id.unsafe_free(); m.unsafe_free(); result.unsafe_free()

def test_matrix_multiply_translation_composition() raises:
    """T(a) * T(b) must equal T(a+b) — translations compose additively."""
    var t1 = unsafe_alloc[Float32](16); _translation(t1, Float32(1.0), Float32(2.0), Float32(3.0))
    var t2 = unsafe_alloc[Float32](16); _translation(t2, Float32(4.0), Float32(-1.0), Float32(0.5))
    var result = unsafe_alloc[Float32](16)
    matrix_multiply(t1, t2, result)
    var expected = unsafe_alloc[Float32](16)
    _translation(expected, Float32(5.0), Float32(1.0), Float32(3.5))
    assert_true(_mat_close(result, expected))
    t1.unsafe_free(); t2.unsafe_free(); result.unsafe_free(); expected.unsafe_free()

def test_matrix_multiply_matches_hand_computed_case() raises:
    """A hand-computed 4x4 * 4x4 case, independent of any translate/scale
    shortcut, to pin down the column-major index arithmetic itself."""
    # a = row-major [[1,2,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] stored column-major
    var a = unsafe_alloc[Float32](16); _identity(a)
    a[unsafe_offset=4] = Float32(2.0)  # row0,col1 = 2  -> flat[col*4+row] = flat[1*4+0] = flat[4]
    var b = unsafe_alloc[Float32](16); _identity(b)
    b[unsafe_offset=12] = Float32(3.0); b[unsafe_offset=13] = Float32(5.0); b[unsafe_offset=14] = Float32(7.0)
    var result = unsafe_alloc[Float32](16)
    matrix_multiply(a, b, result)
    # Expect: a * b = translate by (3 + 2*5, 5, 7) = (13, 5, 7) in col 3,
    # since row0 of a is [1,2,0,0] dotted with b's translation column (3,5,7,1).
    var expected = unsafe_alloc[Float32](16); _identity(expected)
    expected[unsafe_offset=4] = Float32(2.0)
    expected[unsafe_offset=12] = Float32(13.0); expected[unsafe_offset=13] = Float32(5.0); expected[unsafe_offset=14] = Float32(7.0)
    assert_true(_mat_close(result, expected))
    a.unsafe_free(); b.unsafe_free(); result.unsafe_free(); expected.unsafe_free()

# ── matrix_invert ────────────────────────────────────────────────────────────

def test_matrix_invert_of_identity_is_identity() raises:
    var id = unsafe_alloc[Float32](16); _identity(id)
    var result = unsafe_alloc[Float32](16)
    var ok = matrix_invert(id, result)
    assert_true(ok == Int32(1))
    assert_true(_mat_close(result, id))
    id.unsafe_free(); result.unsafe_free()

def test_matrix_invert_translation() raises:
    """Inverse of T(tx,ty,tz) is exactly T(-tx,-ty,-tz)."""
    var t = unsafe_alloc[Float32](16); _translation(t, Float32(2.0), Float32(-3.0), Float32(5.0))
    var inv = unsafe_alloc[Float32](16)
    var ok = matrix_invert(t, inv)
    assert_true(ok == Int32(1))
    var expected = unsafe_alloc[Float32](16); _translation(expected, Float32(-2.0), Float32(3.0), Float32(-5.0))
    assert_true(_mat_close(inv, expected))
    t.unsafe_free(); inv.unsafe_free(); expected.unsafe_free()

def test_matrix_invert_round_trip_matches_original() raises:
    """Matrix_invert(matrix_invert(M)) == M for an invertible translate+scale
    composition, i.e. inversion is its own involution."""
    var s = unsafe_alloc[Float32](16); _scale(s, Float32(2.0), Float32(4.0), Float32(0.5))
    var t = unsafe_alloc[Float32](16); _translation(t, Float32(1.0), Float32(2.0), Float32(3.0))
    var m = unsafe_alloc[Float32](16)
    matrix_multiply(t, s, m)  # composed invertible matrix
    var inv1 = unsafe_alloc[Float32](16)
    var ok1 = matrix_invert(m, inv1)
    assert_true(ok1 == Int32(1))
    var inv2 = unsafe_alloc[Float32](16)
    var ok2 = matrix_invert(inv1, inv2)
    assert_true(ok2 == Int32(1))
    assert_true(_mat_close(inv2, m))
    s.unsafe_free(); t.unsafe_free(); m.unsafe_free(); inv1.unsafe_free(); inv2.unsafe_free()

def test_matrix_invert_times_original_is_identity() raises:
    """M * M^-1 == identity, the defining property of matrix inversion."""
    var s = unsafe_alloc[Float32](16); _scale(s, Float32(2.0), Float32(4.0), Float32(0.5))
    var t = unsafe_alloc[Float32](16); _translation(t, Float32(1.0), Float32(2.0), Float32(3.0))
    var m = unsafe_alloc[Float32](16)
    matrix_multiply(t, s, m)
    var inv = unsafe_alloc[Float32](16)
    var ok = matrix_invert(m, inv)
    assert_true(ok == Int32(1))
    var product = unsafe_alloc[Float32](16)
    matrix_multiply(m, inv, product)
    var id = unsafe_alloc[Float32](16); _identity(id)
    assert_true(_mat_close(product, id))
    s.unsafe_free(); t.unsafe_free(); m.unsafe_free(); inv.unsafe_free(); product.unsafe_free(); id.unsafe_free()

def test_matrix_invert_singular_writes_identity_and_reports_failure() raises:
    """A singular (all-zero) matrix must fail cleanly: return 0 and leave the
    identity in `result`, never garbage — callers rely on this fallback."""
    var singular = unsafe_alloc[Float32](16)
    for i in range(16):
        singular[unsafe_offset=i] = Float32(0)
    var result = unsafe_alloc[Float32](16)
    var ok = matrix_invert(singular, result)
    assert_true(ok == Int32(0))
    var id = unsafe_alloc[Float32](16); _identity(id)
    assert_true(_mat_close(result, id))
    singular.unsafe_free(); result.unsafe_free(); id.unsafe_free()

# ── transform_points ─────────────────────────────────────────────────────────

def test_transform_points_identity_leaves_points_unchanged() raises:
    var id = unsafe_alloc[Float32](16); _identity(id)
    var pts_in = unsafe_alloc[Float32](4)
    pts_in[unsafe_offset=0] = Float32(1.0); pts_in[unsafe_offset=1] = Float32(2.0); pts_in[unsafe_offset=2] = Float32(3.0); pts_in[unsafe_offset=3] = Float32(1.0)
    var pts_out = unsafe_alloc[Float32](4)
    transform_points(id, pts_in, Int32(1), pts_out)
    assert_true(_close(pts_out[unsafe_offset=0], Float32(1.0)))
    assert_true(_close(pts_out[unsafe_offset=1], Float32(2.0)))
    assert_true(_close(pts_out[unsafe_offset=2], Float32(3.0)))
    id.unsafe_free(); pts_in.unsafe_free(); pts_out.unsafe_free()

def test_transform_points_translation_moves_by_exact_vector() raises:
    var t = unsafe_alloc[Float32](16); _translation(t, Float32(10.0), Float32(-5.0), Float32(2.0))
    var pts_in = unsafe_alloc[Float32](4)
    pts_in[unsafe_offset=0] = Float32(1.0); pts_in[unsafe_offset=1] = Float32(1.0); pts_in[unsafe_offset=2] = Float32(1.0); pts_in[unsafe_offset=3] = Float32(1.0)
    var pts_out = unsafe_alloc[Float32](4)
    transform_points(t, pts_in, Int32(1), pts_out)
    assert_true(_close(pts_out[unsafe_offset=0], Float32(11.0)))
    assert_true(_close(pts_out[unsafe_offset=1], Float32(-4.0)))
    assert_true(_close(pts_out[unsafe_offset=2], Float32(3.0)))
    t.unsafe_free(); pts_in.unsafe_free(); pts_out.unsafe_free()

def test_transform_points_scale_scales_coordinates_exactly() raises:
    var s = unsafe_alloc[Float32](16); _scale(s, Float32(2.0), Float32(3.0), Float32(-1.0))
    var pts_in = unsafe_alloc[Float32](8)
    pts_in[unsafe_offset=0] = Float32(1.0); pts_in[unsafe_offset=1] = Float32(2.0); pts_in[unsafe_offset=2] = Float32(3.0); pts_in[unsafe_offset=3] = Float32(1.0)
    pts_in[unsafe_offset=4] = Float32(-2.0); pts_in[unsafe_offset=5] = Float32(0.5); pts_in[unsafe_offset=6] = Float32(4.0); pts_in[unsafe_offset=7] = Float32(1.0)
    var pts_out = unsafe_alloc[Float32](8)
    transform_points(s, pts_in, Int32(2), pts_out)
    assert_true(_close(pts_out[unsafe_offset=0], Float32(2.0)))
    assert_true(_close(pts_out[unsafe_offset=1], Float32(6.0)))
    assert_true(_close(pts_out[unsafe_offset=2], Float32(-3.0)))
    assert_true(_close(pts_out[unsafe_offset=4], Float32(-4.0)))
    assert_true(_close(pts_out[unsafe_offset=5], Float32(1.5)))
    assert_true(_close(pts_out[unsafe_offset=6], Float32(-4.0)))
    s.unsafe_free(); pts_in.unsafe_free(); pts_out.unsafe_free()

# ── transform_normals ────────────────────────────────────────────────────────

def test_transform_normals_identity_leaves_normal_unchanged() raises:
    """Inv_matrix here is the inverse of the forward transform; for the
    identity transform the inverse is itself, so the normal passes through."""
    var id = unsafe_alloc[Float32](16); _identity(id)
    var n_in = unsafe_alloc[Float32](3)
    n_in[unsafe_offset=0] = Float32(0.0); n_in[unsafe_offset=1] = Float32(1.0); n_in[unsafe_offset=2] = Float32(0.0)
    var n_out = unsafe_alloc[Float32](3)
    transform_normals(id, n_in, Int32(1), n_out)
    assert_true(_close(n_out[unsafe_offset=0], Float32(0.0)))
    assert_true(_close(n_out[unsafe_offset=1], Float32(1.0)))
    assert_true(_close(n_out[unsafe_offset=2], Float32(0.0)))
    id.unsafe_free(); n_in.unsafe_free(); n_out.unsafe_free()

def test_transform_normals_uniform_scale_inverse_rescales_normal() raises:
    """Normals transform by (M^-1)^T. For a uniform scale S=diag(k,k,k), the
    caller passes inv_matrix = S^-1 = diag(1/k,1/k,1/k); transform_normals
    applies its transpose, so the normal is scaled by exactly 1/k."""
    var k = Float32(2.0)
    var inv_s = unsafe_alloc[Float32](16); _scale(inv_s, Float32(1.0) / k, Float32(1.0) / k, Float32(1.0) / k)
    var n_in = unsafe_alloc[Float32](3)
    n_in[unsafe_offset=0] = Float32(0.0); n_in[unsafe_offset=1] = Float32(0.0); n_in[unsafe_offset=2] = Float32(1.0)
    var n_out = unsafe_alloc[Float32](3)
    transform_normals(inv_s, n_in, Int32(1), n_out)
    assert_true(_close(n_out[unsafe_offset=0], Float32(0.0)))
    assert_true(_close(n_out[unsafe_offset=1], Float32(0.0)))
    assert_true(_close(n_out[unsafe_offset=2], Float32(0.5)))
    inv_s.unsafe_free(); n_in.unsafe_free(); n_out.unsafe_free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
