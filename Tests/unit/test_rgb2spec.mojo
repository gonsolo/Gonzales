# Tests for the real Jakob & Hanika 2019 spectral upsampling in rgb2spec.mojo
# — both the on-demand solver (gauss_newton_solve) and the dense prebaked
# table (build_spectrum_table / rgb_to_coeffs_table_lookup / save+load). The
# table test uses a small resolution (16, not the shipped 64) to keep the
# suite fast; res=64 is validated separately (see project_spectral_rendering
# memory) and shipped as src/gonzales/data/rgb2spectrum_table.bin.
from std.math import abs
from std.testing import assert_true, TestSuite
from gonzales.rgb2spec import (
    build_quadrature_tables, gauss_newton_solve, eval_sigmoid_spectrum,
    RGBSigmoidCoeffs, _QuadratureTables, build_spectrum_table,
    rgb_to_coeffs_table_lookup, save_spectrum_table, load_spectrum_table,
)

comptime TEST_RES = 16

def _rgb_from_coeffs(coeffs: RGBSigmoidCoeffs, tables: _QuadratureTables) -> Tuple[Float32, Float32, Float32]:
    var out0 = Float32(0.0); var out1 = Float32(0.0); var out2 = Float32(0.0)
    for i in range(len(tables.lambda_tbl)):
        var lam = Float32(tables.lambda_tbl[i])
        var s = eval_sigmoid_spectrum(coeffs, lam)
        out0 += Float32(tables.rgb_tbl0[i]) * s
        out1 += Float32(tables.rgb_tbl1[i]) * s
        out2 += Float32(tables.rgb_tbl2[i]) * s
    return (out0, out1, out2)

def _close(a: Float32, b: Float32, tol: Float32) -> Bool:
    return abs(a - b) < tol

# ── on-demand solver ─────────────────────────────────────────────────────

def test_gauss_newton_solve_recovers_grey() raises:
    var tables = build_quadrature_tables()
    var coeffs = gauss_newton_solve(0.5, 0.5, 0.5, tables)
    var (r, g, b) = _rgb_from_coeffs(coeffs, tables)
    assert_true(_close(r, Float32(0.5), Float32(0.01)))
    assert_true(_close(g, Float32(0.5), Float32(0.01)))
    assert_true(_close(b, Float32(0.5), Float32(0.01)))

def test_gauss_newton_solve_recovers_saturated_red() raises:
    var tables = build_quadrature_tables()
    var coeffs = gauss_newton_solve(0.8, 0.05, 0.05, tables)
    var (r, g, b) = _rgb_from_coeffs(coeffs, tables)
    assert_true(_close(r, Float32(0.8), Float32(0.01)))
    assert_true(_close(g, Float32(0.05), Float32(0.01)))
    assert_true(_close(b, Float32(0.05), Float32(0.01)))

def test_eval_sigmoid_spectrum_stays_in_unit_range() raises:
    var tables = build_quadrature_tables()
    var coeffs = gauss_newton_solve(0.9, 0.1, 0.4, tables)
    var lam = Float32(360.0)
    while lam <= Float32(830.0):
        var s = eval_sigmoid_spectrum(coeffs, lam)
        assert_true(s >= Float32(0.0) and s <= Float32(1.0))
        lam += Float32(10.0)

# ── dense prebaked table ─────────────────────────────────────────────────

def test_table_lookup_matches_direct_solve_for_neutral_colors() raises:
    var tables = build_quadrature_tables()
    var table = build_spectrum_table(TEST_RES)

    var direct = gauss_newton_solve(0.5, 0.5, 0.5, tables)
    var (dr, dg, db) = _rgb_from_coeffs(direct, tables)
    var looked_up = rgb_to_coeffs_table_lookup(table, TEST_RES, 0.5, 0.5, 0.5)
    var (lr, lg, lb) = _rgb_from_coeffs(looked_up, tables)

    assert_true(_close(dr, lr, Float32(0.02)))
    assert_true(_close(dg, lg, Float32(0.02)))
    assert_true(_close(db, lb, Float32(0.02)))

def test_table_lookup_is_approximately_correct_for_saturated_color() raises:
    var tables = build_quadrature_tables()
    var table = build_spectrum_table(TEST_RES)

    var looked_up = rgb_to_coeffs_table_lookup(table, TEST_RES, 0.8, 0.05, 0.05)
    var (lr, lg, lb) = _rgb_from_coeffs(looked_up, tables)

    # Coarse (res=16) grid — loose tolerance; the shipped res=64 table is
    # sub-1%-accurate (verified separately), this only checks the lookup
    # machinery itself is wired correctly, not final production accuracy.
    assert_true(_close(lr, Float32(0.8), Float32(0.02)))
    assert_true(_close(lg, Float32(0.05), Float32(0.02)))
    assert_true(_close(lb, Float32(0.05), Float32(0.02)))

def test_table_lookup_black_is_near_zero() raises:
    var tables = build_quadrature_tables()
    var table = build_spectrum_table(TEST_RES)
    var looked_up = rgb_to_coeffs_table_lookup(table, TEST_RES, 0.0, 0.0, 0.0)
    var (r, g, b) = _rgb_from_coeffs(looked_up, tables)
    assert_true(abs(r) < Float32(0.02))
    assert_true(abs(g) < Float32(0.02))
    assert_true(abs(b) < Float32(0.02))

def test_save_and_load_spectrum_table_round_trips() raises:
    var table = build_spectrum_table(TEST_RES)
    var path = "/tmp/gonzales_test_rgb2spectrum_table.bin"
    save_spectrum_table(table, TEST_RES, path)

    var loaded = load_spectrum_table(path)
    assert_true(loaded[0])
    assert_true(loaded[1] == TEST_RES)
    assert_true(len(loaded[2]) == len(table))
    for i in range(len(table)):
        assert_true(table[i] == loaded[2][i])

def test_load_spectrum_table_rejects_bad_path() raises:
    var loaded = load_spectrum_table("/nonexistent/path/does_not_exist.bin")
    assert_true(not loaded[0])

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
