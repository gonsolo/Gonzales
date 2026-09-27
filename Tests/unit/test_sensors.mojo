# Tests for sensors.mojo's named camera-sensor colorimetry (PBRT-v4's
# PixelSensor non-default path). See sensors.mojo's header for the design
# and the algebraic derivation this file's self-check verifies.
#
# Historical note: an earlier version of named_sensor_srgb_matrix multiplied
# its returned matrix by an extra CIE_Y_INTEGRAL factor, which passed an
# independent from-scratch Python re-derivation of the SAME (wrong) premise
# bit-for-bit, yet rendered explosion.pbrt's nikon_d850 sensor ~161x too
# bright. test_sensor_matrix_self_check below is the check that actually
# catches that class of bug: it feeds the exact CIE X/Y/Z matching functions
# in as a fake "sensor" and asserts the fitted correction matrix comes out
# close to the IDENTITY, not CIE_Y_INTEGRAL*Identity -- a "sensor" whose
# response IS the CIE curves must reproduce plain cie1931 rendering exactly.
# A second bug (buf.r hard-clamped to 0 before the wb multiply, in
# rendering.mojo/gpu_denoise.mojo) is exercised by the reference-matrix
# check below indirectly (it lives downstream of named_sensor_srgb_matrix,
# so isn't visible to a pure calibration test) -- see project_explosion_*
# memory for the render-level repro (a flat "rgb L" test light).

from std.math import abs
from std.testing import assert_true, TestSuite
from gonzales.sensors import (
    named_sensor_is_supported, named_sensor_srgb_matrix, scan_film_sensor_params,
    _CIE_X, _CIE_Y, _CIE_Z, _swatch, N_SWATCHES,
    _project_reflectance_sensor, _project_reflectance_xyz,
    _solve_xyz_from_sensor_rgb, _mat3_inverse, _mat3_mul,
)
from gonzales.rgb2spec import (
    CIE_LAMBDA_MIN, CIE_LAMBDA_MAX, CIE_SAMPLES,
    _XYZ_TO_SRGB00, _XYZ_TO_SRGB01, _XYZ_TO_SRGB02,
    _XYZ_TO_SRGB10, _XYZ_TO_SRGB11, _XYZ_TO_SRGB12,
    _XYZ_TO_SRGB20, _XYZ_TO_SRGB21, _XYZ_TO_SRGB22,
)

comptime EPS: Float32 = 1e-3

def _close(a: Float32, b: Float32, tol: Float32) -> Bool:
    return abs(a - b) < tol

def test_named_sensor_is_supported() raises:
    assert_true(named_sensor_is_supported(String("nikon_d850")))
    assert_true(named_sensor_is_supported(String("canon_eos_100d")))
    assert_true(named_sensor_is_supported(String("canon_eos_5d_mkiv")))
    assert_true(not named_sensor_is_supported(String("cie1931")))
    assert_true(not named_sensor_is_supported(String("some_unknown_camera")))

def test_scan_film_sensor_params_found() raises:
    var text = String(
        "Film \"rgb\"\n  \"integer xresolution\" 320\n"
        + "  \"string sensor\" \"nikon_d850\"\n  \"float whitebalance\" 6000\n"
        + "  \"float iso\" 100\n"
    )
    var res = scan_film_sensor_params(text)
    assert_true(res[0] == String("nikon_d850"))
    assert_true(_close(res[1], Float32(6000.0), Float32(1e-6)))

def test_scan_film_sensor_params_default() raises:
    var text = String("Film \"rgb\"\n  \"integer xresolution\" 320\n  \"float iso\" 100\n")
    var res = scan_film_sensor_params(text)
    assert_true(res[0] == String("cie1931"))
    assert_true(_close(res[1], Float32(0.0), Float32(1e-6)))

def test_sensor_matrix_self_check() raises:
    """A 'sensor' whose r/g/b response curves ARE the exact CIE X/Y/Z
    matching functions must reproduce plain cie1931 rendering exactly, so
    named_sensor_srgb_matrix's whole calibration+composition pipeline, run
    on that fake sensor, must come out close to the IDENTITY matrix -- see
    this file's header. Reimplements named_sensor_srgb_matrix's body
    directly against sensors.mojo's exposed building blocks (rather than
    routing through named_sensor_curves, which only knows real sensor
    names) so this test exercises the exact same calibration code the real
    named sensors use."""
    var cie_x_dense = _CIE_X(); var cie_y_dense = _CIE_Y(); var cie_z_dense = _CIE_Z()
    var r_curve = List[Float64](capacity=CIE_SAMPLES * 2)
    var g_curve = List[Float64](capacity=CIE_SAMPLES * 2)
    var b_curve = List[Float64](capacity=CIE_SAMPLES * 2)
    for i in range(CIE_SAMPLES):
        var l = CIE_LAMBDA_MIN + Float64(i) * (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN) / Float64(CIE_SAMPLES - 1)
        r_curve.append(l); r_curve.append(cie_x_dense[i])
        g_curve.append(l); g_curve.append(cie_y_dense[i])
        b_curve.append(l); b_curve.append(cie_z_dense[i])

    var wb_temp = 6500.0
    var rgb_camera = List[Float64](capacity=N_SWATCHES * 3)
    var xyz_output = List[Float64](capacity=N_SWATCHES * 3)
    for _ in range(N_SWATCHES * 3):
        rgb_camera.append(0.0); xyz_output.append(0.0)
    for s in range(N_SWATCHES):
        var refl = _swatch(s)
        var rgb = _project_reflectance_sensor(refl, wb_temp, r_curve, g_curve, b_curve)
        rgb_camera[s * 3 + 0] = rgb[0]; rgb_camera[s * 3 + 1] = rgb[1]; rgb_camera[s * 3 + 2] = rgb[2]
        var xyz = _project_reflectance_xyz(refl)
        xyz_output[s * 3 + 0] = xyz[0]; xyz_output[s * 3 + 1] = xyz[1]; xyz_output[s * 3 + 2] = xyz[2]
    var f_mat = _solve_xyz_from_sensor_rgb(rgb_camera, xyz_output)

    var s_mat = List[Float64](capacity=9)
    s_mat.append(Float64(_XYZ_TO_SRGB00)); s_mat.append(Float64(_XYZ_TO_SRGB01)); s_mat.append(Float64(_XYZ_TO_SRGB02))
    s_mat.append(Float64(_XYZ_TO_SRGB10)); s_mat.append(Float64(_XYZ_TO_SRGB11)); s_mat.append(Float64(_XYZ_TO_SRGB12))
    s_mat.append(Float64(_XYZ_TO_SRGB20)); s_mat.append(Float64(_XYZ_TO_SRGB21)); s_mat.append(Float64(_XYZ_TO_SRGB22))
    var s_inv = _mat3_inverse(s_mat)
    var wb_mat = _mat3_mul(_mat3_mul(s_mat, f_mat), s_inv)

    # NOTE: rgb_camera used wb_temp-weighted illuminant (via
    # _project_reflectance_sensor) while xyz_output used the fixed D65
    # output illuminant (via _project_reflectance_xyz) -- at 6500K these are
    # close enough that F should still be close to identity (within ~15%,
    # loose on purpose: this test's job is to catch a gross CIE_Y_INTEGRAL-
    # sized [~107x] scale bug, not to pin exact swatch-fit residuals).
    for r in range(3):
        for c in range(3):
            var expected = Float32(1.0) if r == c else Float32(0.0)
            assert_true(_close(Float32(wb_mat[r * 3 + c]), expected, Float32(0.2)))

def test_named_sensor_srgb_matrix_reference() raises:
    """Pins named_sensor_srgb_matrix's nikon_d850/6000K output against a
    value independently cross-checked in Python (a from-scratch
    reimplementation of pbrt's PixelSensor calibration against the same
    extracted data, run outside this codebase) -- see project_explosion_*
    memory. Loose tolerance: this guards against a structural regression
    (wrong sign, transposed matrix, missing swatch), not last-decimal
    drift."""
    var res = named_sensor_srgb_matrix(String("nikon_d850"), Float32(6000.0))
    assert_true(res[0])
    var wb = res[1]
    # Expected values (S @ F @ inv(S), NO CIE_Y_INTEGRAL factor):
    #   [[1.5067, 1.0024, 0.5857], [0.2572, 1.0914, -0.4574], [-0.0356, -0.1636, 2.1839]]
    assert_true(_close(wb[0], Float32(1.5067), Float32(0.05)))
    assert_true(_close(wb[1], Float32(1.0024), Float32(0.05)))
    assert_true(_close(wb[2], Float32(0.5857), Float32(0.05)))
    assert_true(_close(wb[3], Float32(0.2572), Float32(0.05)))
    assert_true(_close(wb[4], Float32(1.0914), Float32(0.05)))
    assert_true(_close(wb[5], Float32(-0.4574), Float32(0.05)))
    assert_true(_close(wb[6], Float32(-0.0356), Float32(0.05)))
    assert_true(_close(wb[7], Float32(-0.1636), Float32(0.05)))
    assert_true(_close(wb[8], Float32(2.1839), Float32(0.05)))
    var xt = res[2].copy()
    assert_true(len(xt) == CIE_SAMPLES)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
