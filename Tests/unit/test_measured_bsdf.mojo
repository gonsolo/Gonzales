from std.math import abs
from std.testing import assert_true, assert_equal, TestSuite
from gonzales.measured_bsdf import (
    load_measured_brdf_full, _pl2d_strides2, _pl2d_strides3,
    _pl2d_build_cdf, _pl2d_build_scaled_verbatim,
)

comptime EPS: Float32 = 1e-4

def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) < EPS

# ── Stride computation (PiecewiseLinear2D's backward per-axis stride loop,
#    util/sampling.h:1354-1364 — see measured_bsdf.mojo's port) ─────────────

def test_strides2_isotropic_phi_gets_zero_stride() raises:
    """Nphi<=1: phi's stride must be 0 (no second slice to blend — the
    degenerate-axis fast path pbrt's own Sample/Invert/Evaluate special-case
    with `param_size[dim]==1`), theta's stride is 1 (first/only param dim
    stepped)."""
    var (stride_phi, stride_theta) = _pl2d_strides2(1, 8)
    assert_equal(stride_phi, 0)
    assert_equal(stride_theta, 1)

def test_strides2_matches_hand_derivation_for_real_shapes() raises:
    """Nphi=2, Ntheta=8 (both >1): theta (last axis) gets stride=1, phi (first
    axis) gets stride=slices-after-theta=8."""
    var (stride_phi, stride_theta) = _pl2d_strides2(2, 8)
    assert_equal(stride_theta, 1)
    assert_equal(stride_phi, 8)

def test_strides3_matches_hand_derivation_for_real_shapes() raises:
    """Nphi=1 (isotropic), Ntheta=8, Nlambda=195 -- matches this project's
    real sportscar .bsdf file dimensions (verified via a standalone Python
    field-table dump of cc_blue_agat_spec.bsdf during design). lambda (last
    axis) stride=1, theta stride=Nlambda=195, phi stride=0 (isotropic)."""
    var (stride_phi, stride_theta, stride_lambda) = _pl2d_strides3(1, 8, 195)
    assert_equal(stride_lambda, 1)
    assert_equal(stride_theta, 195)
    assert_equal(stride_phi, 0)

# ── PiecewiseLinear2D construction (util/sampling.h:1370-1411 for build_cdf,
#    :1412-1436 for the scaled-verbatim else-branch) ─────────────────────────

def test_build_scaled_verbatim_uses_inv_patch_size_normalization() raises:
    """Ndf/sigma/spectra's build_cdf=false,normalize=false branch is NOT a
    verbatim copy -- pbrt still scales by 1/((xs-1)*(ys-1))
    (HProd(m_inv_patch_size)), read directly from util/sampling.h:1416/1432.
    A 2x2 grid of all-1.0s scaled by 1/((2-1)*(2-1)) = 1/1 = 1.0 stays 1.0;
    verify the *formula* with a non-trivial shape instead so a regression to
    "verbatim copy" would actually be caught."""
    var raw = alloc[Float32](3 * 4)  # xs=4, ys=3 -> norm = 1/((4-1)*(3-1)) = 1/6
    for i in range(12):
        raw[i] = Float32(6.0)
    var out = _pl2d_build_scaled_verbatim(raw, 4, 3, 1)
    for i in range(12):
        assert_true(_close(out[i], Float32(1.0)))
    raw.free(); out.free()

def test_build_cdf_marginal_ends_at_one_and_is_monotonic() raises:
    """Per pbrt's construction (util/sampling.h:1398-1405): the marginal CDF
    is normalized by 1/marginal_cdf[ys-1], so its last entry must be exactly
    1.0 after normalization, and (being a CDF of nonnegative density) every
    entry must be monotonically non-decreasing."""
    var xs = 4; var ys = 5
    var raw = alloc[Float32](xs * ys)
    # Arbitrary nonnegative density, not uniform -- exercises real weighting.
    for y in range(ys):
        for x in range(xs):
            raw[y * xs + x] = Float32(1.0 + Float64(x)) * Float32(1.0 + Float64(y))
    var (data_out, marginal, conditional) = _pl2d_build_cdf(raw, xs, ys, 1)

    assert_true(_close(marginal[ys - 1], Float32(1.0)))
    for y in range(ys - 1):
        assert_true(marginal[y + 1] >= marginal[y] - EPS)
    # Conditional CDF's last column of each row must also end at exactly
    # what the marginal step for that row implies -- weaker, cheaper check:
    # just confirm every row's conditional CDF is itself monotonic.
    for y in range(ys):
        for x in range(xs - 1):
            var i = y * xs + x
            assert_true(conditional[i + 1] >= conditional[i] - EPS)

    raw.free(); data_out.free(); marginal.free(); conditional.free()

def test_build_cdf_handles_multiple_slices_independently() raises:
    """Slices>1 (the vndf/luminance real case, sliced by phi*theta): each
    slice's marginal CDF must independently end at 1.0 -- a bug that only
    processes slice 0 (e.g. a forgotten per-slice base-offset) would leave
    slice 1's marginal at 0."""
    var xs = 3; var ys = 3
    var raw = alloc[Float32](xs * ys * 2)
    for i in range(xs * ys * 2):
        raw[i] = Float32(1.0 + Float32(i))
    var (data_out, marginal, conditional) = _pl2d_build_cdf(raw, xs, ys, 2)
    assert_true(_close(marginal[ys - 1], Float32(1.0)))
    assert_true(_close(marginal[ys + ys - 1], Float32(1.0)))
    raw.free(); data_out.free(); marginal.free(); conditional.free()

# ── Full tensor-file loader (load_measured_brdf_full) ────────────────────────
# Builds a minimal-but-structurally-valid synthetic ".bsdf" tensor file
# in-memory (no dependency on any real pbrt-v4-scenes asset, which would be
# tens of MB and unsuitable to commit) exercising the SAME field-table parse
# + shape validation + CDF construction path load_measured_brdf_full uses on
# a real file. Field/shape/dtype conventions verified against a real
# sportscar .bsdf file's field table during design (see the module's other
# tests' docstrings for the real-world numbers this mirrors).

def _put_u16(mut buf: List[UInt8], v: UInt16):
    var p = UnsafePointer(to=v).bitcast[UInt8]()
    buf.append(p[0]); buf.append(p[1])

def _put_u32(mut buf: List[UInt8], v: UInt32):
    var p = UnsafePointer(to=v).bitcast[UInt8]()
    for i in range(4): buf.append(p[i])

def _put_u64(mut buf: List[UInt8], v: UInt64):
    var p = UnsafePointer(to=v).bitcast[UInt8]()
    for i in range(8): buf.append(p[i])

def _put_f32(mut buf: List[UInt8], v: Float32):
    var p = UnsafePointer(to=v).bitcast[UInt8]()
    for i in range(4): buf.append(p[i])

def _lit_len(s: StringLiteral) -> Int:
    var p = s.unsafe_ptr()
    var n = 0
    while p[n] != UInt8(0):
        n += 1
    return n

def _put_name(mut buf: List[UInt8], name: StringLiteral):
    var n = _lit_len(name)
    var np = name.unsafe_ptr()
    _put_u16(buf, UInt16(n))
    for i in range(n):
        buf.append(np[i])

def _write_synthetic_bsdf(path: String) raises:
    """Minimal valid isotropic file: Nphi=1, Ntheta=2, Nlambda=2, ndf/sigma
    2x2, vndf/luminance 1x2x2x2, spectra 1x2x2x2x2 -- the smallest shapes
    satisfying every cross-field assert in load_measured_brdf_full (mirrors
    MeasuredBxDFData::Create's own validation, bxdfs.cpp:902-931)."""
    # ---- Header ----
    var header = List[UInt8]()
    comptime magic_lit = "tensor_file"
    var magic_p = magic_lit.unsafe_ptr()
    for i in range(_lit_len(magic_lit)):
        header.append(magic_p[i])
    header.append(UInt8(0))
    header.append(UInt8(1)); header.append(UInt8(0))  # version 1.0
    _put_u32(header, UInt32(9))  # n_fields

    # ---- Field table (name, ndim, dtype, offset, shape...) ----
    comptime DT_U8 = 1
    comptime DT_F32 = 10
    var table = List[UInt8]()

    var theta_i_data = List[Float32](); theta_i_data.append(Float32(0.0)); theta_i_data.append(Float32(1.0))
    var phi_i_data = List[Float32](); phi_i_data.append(Float32(0.0))
    var wavelengths_data = List[Float32](); wavelengths_data.append(Float32(400.0)); wavelengths_data.append(Float32(700.0))
    var ndf_data = List[Float32]()
    for _i in range(4): ndf_data.append(Float32(1.0))
    var sigma_data = List[Float32]()
    for _i in range(4): sigma_data.append(Float32(1.0))
    var vndf_data = List[Float32]()
    for i in range(8): vndf_data.append(Float32(1.0 + Float32(i)))
    var luminance_data = List[Float32]()
    for i in range(8): luminance_data.append(Float32(1.0 + Float32(i)))
    var spectra_data = List[Float32]()
    for i in range(16): spectra_data.append(Float32(0.5 + Float32(i) * Float32(0.1)))
    var jacobian_data = List[UInt8](); jacobian_data.append(UInt8(1))

    # Layout raw data blocks back-to-back after the header+table; compute the
    # table size up front so absolute offsets are known before writing entries.
    # name_len(2) + name + ndim(2) + dtype(1) + offset(8) + shape*8, per field.
    comptime HEADER_LEN = 12 + 2 + 4
    var table_len = 0
    table_len += 2 + _lit_len("theta_i") + 2 + 1 + 8 + 1 * 8
    table_len += 2 + _lit_len("phi_i") + 2 + 1 + 8 + 1 * 8
    table_len += 2 + _lit_len("wavelengths") + 2 + 1 + 8 + 1 * 8
    table_len += 2 + _lit_len("ndf") + 2 + 1 + 8 + 2 * 8
    table_len += 2 + _lit_len("sigma") + 2 + 1 + 8 + 2 * 8
    table_len += 2 + _lit_len("vndf") + 2 + 1 + 8 + 4 * 8
    table_len += 2 + _lit_len("luminance") + 2 + 1 + 8 + 4 * 8
    table_len += 2 + _lit_len("spectra") + 2 + 1 + 8 + 5 * 8
    table_len += 2 + _lit_len("jacobian") + 2 + 1 + 8 + 1 * 8

    var data_start = HEADER_LEN + table_len
    var off_theta_i = data_start
    var off_phi_i = off_theta_i + len(theta_i_data) * 4
    var off_wavelengths = off_phi_i + len(phi_i_data) * 4
    var off_ndf = off_wavelengths + len(wavelengths_data) * 4
    var off_sigma = off_ndf + len(ndf_data) * 4
    var off_vndf = off_sigma + len(sigma_data) * 4
    var off_luminance = off_vndf + len(vndf_data) * 4
    var off_spectra = off_luminance + len(luminance_data) * 4
    var off_jacobian = off_spectra + len(spectra_data) * 4

    _put_name(table, "theta_i")
    _put_u16(table, UInt16(1)); table.append(UInt8(DT_F32)); _put_u64(table, UInt64(off_theta_i))
    _put_u64(table, UInt64(2))

    _put_name(table, "phi_i")
    _put_u16(table, UInt16(1)); table.append(UInt8(DT_F32)); _put_u64(table, UInt64(off_phi_i))
    _put_u64(table, UInt64(1))

    _put_name(table, "wavelengths")
    _put_u16(table, UInt16(1)); table.append(UInt8(DT_F32)); _put_u64(table, UInt64(off_wavelengths))
    _put_u64(table, UInt64(2))

    _put_name(table, "ndf")
    _put_u16(table, UInt16(2)); table.append(UInt8(DT_F32)); _put_u64(table, UInt64(off_ndf))
    _put_u64(table, UInt64(2)); _put_u64(table, UInt64(2))  # shape[0]=ys, shape[1]=xs

    _put_name(table, "sigma")
    _put_u16(table, UInt16(2)); table.append(UInt8(DT_F32)); _put_u64(table, UInt64(off_sigma))
    _put_u64(table, UInt64(2)); _put_u64(table, UInt64(2))

    _put_name(table, "vndf")
    _put_u16(table, UInt16(4)); table.append(UInt8(DT_F32)); _put_u64(table, UInt64(off_vndf))
    _put_u64(table, UInt64(1)); _put_u64(table, UInt64(2)); _put_u64(table, UInt64(2)); _put_u64(table, UInt64(2))

    _put_name(table, "luminance")
    _put_u16(table, UInt16(4)); table.append(UInt8(DT_F32)); _put_u64(table, UInt64(off_luminance))
    _put_u64(table, UInt64(1)); _put_u64(table, UInt64(2)); _put_u64(table, UInt64(2)); _put_u64(table, UInt64(2))

    _put_name(table, "spectra")
    _put_u16(table, UInt16(5)); table.append(UInt8(DT_F32)); _put_u64(table, UInt64(off_spectra))
    _put_u64(table, UInt64(1)); _put_u64(table, UInt64(2)); _put_u64(table, UInt64(2)); _put_u64(table, UInt64(2)); _put_u64(table, UInt64(2))

    _put_name(table, "jacobian")
    _put_u16(table, UInt16(1)); table.append(UInt8(DT_U8)); _put_u64(table, UInt64(off_jacobian))
    _put_u64(table, UInt64(1))

    var buf = List[UInt8]()
    for i in range(len(header)): buf.append(header[i])
    for i in range(len(table)): buf.append(table[i])
    for v in theta_i_data: _put_f32(buf, v)
    for v in phi_i_data: _put_f32(buf, v)
    for v in wavelengths_data: _put_f32(buf, v)
    for v in ndf_data: _put_f32(buf, v)
    for v in sigma_data: _put_f32(buf, v)
    for v in vndf_data: _put_f32(buf, v)
    for v in luminance_data: _put_f32(buf, v)
    for v in spectra_data: _put_f32(buf, v)
    for v in jacobian_data: buf.append(v)

    var f = open(path, "w")
    f.write_bytes(Span(buf))
    f.close()

def test_load_measured_brdf_full_parses_synthetic_file() raises:
    var path = "/tmp/gonzales_test_measured_synthetic.bsdf"
    _write_synthetic_bsdf(path)
    var (ok, mb) = load_measured_brdf_full(path)
    assert_true(ok)
    assert_equal(Int(mb.isotropic), 1)
    assert_equal(Int(mb.n_theta_i), 2)
    assert_equal(Int(mb.n_phi_i), 1)
    assert_equal(Int(mb.n_wavelengths), 2)
    assert_true(_close(mb.theta_i[0], Float32(0.0)))
    assert_true(_close(mb.theta_i[1], Float32(1.0)))
    assert_true(_close(mb.wavelengths[0], Float32(400.0)))
    # vndf/luminance CDFs must be normalized (marginal ends at 1.0) for each
    # of the slices*ys entries -- slices = n_phi_i*n_theta_i = 2 here.
    assert_true(_close(mb.vndf_marg[mb.vndf_ys - 1], Float32(1.0)))
    assert_true(_close(mb.vndf_marg[2 * mb.vndf_ys - 1], Float32(1.0)))
    assert_true(_close(mb.lum_marg[mb.lum_ys - 1], Float32(1.0)))

def test_load_measured_brdf_full_fails_gracefully_on_missing_file() raises:
    var (ok, _mb) = load_measured_brdf_full("/tmp/gonzales_test_measured_does_not_exist.bsdf")
    assert_true(not ok)

def test_load_measured_brdf_full_fails_gracefully_on_garbage_file() raises:
    var path = "/tmp/gonzales_test_measured_garbage.bsdf"
    var f = open(path, "w")
    var junk = List[UInt8]()
    for _i in range(64): junk.append(UInt8(0xAB))
    f.write_bytes(Span(junk))
    f.close()
    var (ok, _mb) = load_measured_brdf_full(path)
    assert_true(not ok)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
