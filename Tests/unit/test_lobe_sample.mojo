# lobe_sample, the sampling half of the BxDF interface (bxdf.mojo), checked
# against lobe_eval, the evaluation half every other strategy uses. A sampler
# and an evaluator that disagree are the drift that put VCM's diffusetransmission
# connections out of MIS (8fcd6be0) and made pbrt's LayeredBxDF::f() disagree
# with its own Sample_f() (mmp/pbrt-v4#551).
from std.math import abs, sqrt, cos, sin
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import RGB, Vec3f, Material_C, MatKind, LobeKind, MeasuredBRDF_C, PI
from gonzales.curves import Curve_C
from gonzales.bxdf import LobeCtx, LobeTables, lobe_eval, lobe_sample, lobe_scoped
from gonzales.rng import PCG32
from gonzales.spectrum import SampledWavelengths, sample_wavelengths_uniform, SpectralContext, spectral_handle
from gonzales.rgb2spec import build_spectrum_table, build_cie_xyz_tables, SpectrumTable

comptime _curves = Pointer[Curve_C, MutUntrackedOrigin].unsafe_dangling()
comptime _mbrdfs = Pointer[MeasuredBRDF_C, MutUntrackedOrigin].unsafe_dangling()
comptime TEST_RES = 16

comptime MAT_DT = 0
comptime MAT_COAT_ROUGH = 1
comptime MAT_COAT_SMOOTH = 2


def _test_ctx() -> SpectralContext:
    var table = build_spectrum_table(TEST_RES)
    var cie = build_cie_xyz_tables()
    return SpectralContext(SpectrumTable(table^, TEST_RES), cie^)


def _material(type: Int8, albedo: RGB, emission: RGB, rough: Float32) -> Material_C:
    return Material_C(type, Int8(0), Int8(0), Int8(0), albedo, emission,
        Int32(-1), rough, rough, Int32(-1), Int32(-1), Float32(1.0), Int32(-1), Int32(-1),
        RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(1.0), Float32(1.0), Int32(-1),
        RGB(Float32(1.0)), RGB(Float32(0.0)), RGB(Float32(1.0)))


def _tables() -> LobeTables:
    var mats = unsafe_alloc[Material_C](3)
    # diffusetransmission: reflectance 0.3, transmittance (in .emission) 0.5
    mats[unsafe_offset=MAT_DT] = _material(MatKind.diffuse_transmit, RGB(Float32(0.3)), RGB(Float32(0.5)), Float32(0))
    # coateddiffuse: eta in .emission.r, coat alpha in roughU/V
    mats[unsafe_offset=MAT_COAT_ROUGH] = _material(MatKind.coated_diffuse, RGB(Float32(0.8)), RGB(Float32(1.5)), Float32(0.3))
    mats[unsafe_offset=MAT_COAT_SMOOTH] = _material(MatKind.coated_diffuse, RGB(Float32(0.8)), RGB(Float32(1.5)), Float32(0))
    return LobeTables(mats, _curves, _mbrdfs)


def _ctx(kind: Int32, mat_idx: Int, alb: RGB, cos_o: Float32, adjoint: Bool) -> LobeCtx:
    var wo = Vec3f(sqrt(max(Float32(0), Float32(1) - cos_o * cos_o)), Float32(0), cos_o)
    return LobeCtx(kind, True, False, Vec3f(0.0, 0.0, 1.0), wo, alb, Int32(mat_idx),
                   Float32(0), Float32(0), Int32(-1), Float32(0), Float32(0), False, adjoint)


def _uniform_sphere(u: Float32, v: Float32) -> Vec3f:
    var z = Float32(1) - Float32(2) * u
    var r = sqrt(max(Float32(0), Float32(1) - z * z))
    var phi = Float32(2) * PI * v
    return Vec3f(r * cos(phi), r * sin(phi), z)


def _check_consistency(name: String, c: LobeCtx, tab: LobeTables) raises:
    """E[weight] over non-delta samples == integral of lobe_eval's f*cos, and
    every sample's pdf_fwd == lobe_eval's pdf_fwd at that direction."""
    var ctx = _test_ctx()
    var h = spectral_handle(ctx)
    var wl = sample_wavelengths_uniform(Float32(0.37))
    var rng = PCG32(UInt64(7), UInt64(11))
    comptime N = 60000
    var sampled = Float64(0)
    var uniform = Float64(0)
    var worst_pdf = Float32(0)
    for _ in range(N):
        var s = lobe_sample(c, rng.next_float(), rng.next_float(), rng.next_float(), rng.next_float(), tab,
                            h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65, wl)
        if s.valid and not s.is_delta:
            sampled += Float64(s.weight.v0)
            var le_s = lobe_eval[want_pdfs=True](c, s.wi, tab, h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65, wl)
            var rel = abs(le_s.pdf_fwd - s.pdf_fwd) / max(le_s.pdf_fwd, Float32(1e-6))
            worst_pdf = max(worst_pdf, rel)
        var wi = _uniform_sphere(rng.next_float(), rng.next_float())
        var le = lobe_eval[want_pdfs=True](c, wi, tab, h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65, wl)
        uniform += Float64(le.f_cos.v0) * Float64(4.0 * 3.14159265358979)
    sampled /= Float64(N)
    uniform /= Float64(N)
    var err = abs(sampled - uniform) / max(uniform, Float64(1e-9))
    print(name, "albedo from lobe_sample", sampled, "from lobe_eval", uniform,
          "rel err", err, "worst pdf mismatch", worst_pdf)
    assert_true(err < Float64(0.03), name + ": lobe_sample and lobe_eval disagree")
    assert_true(worst_pdf < Float32(1e-4), name + ": sampled pdf_fwd is not lobe_eval's")
    _ = ctx^


def test_lambertian_sample_matches_eval() raises:
    var tab = _tables()
    for cos_o in [Float32(0.9), Float32(0.3)]:
        _check_consistency("lambertian", _ctx(LobeKind.lambertian, -1, RGB(Float32(0.6)), cos_o, False), tab)


def test_diffuse_transmit_sample_matches_eval() raises:
    var tab = _tables()
    for cos_o in [Float32(0.9), Float32(-0.4)]:   # wo on either side of n
        _check_consistency("diffuse_transmit", _ctx(LobeKind.diffuse_transmit, MAT_DT, RGB(Float32(0.3)), cos_o, False), tab)


def test_layered_sample_matches_eval() raises:
    var tab = _tables()
    for adjoint in [False, True]:
        for cos_o in [Float32(0.9), Float32(0.4)]:
            _check_consistency("layered rough", _ctx(LobeKind.layered, MAT_COAT_ROUGH, RGB(Float32(0.8)), cos_o, adjoint), tab)
            _check_consistency("layered smooth", _ctx(LobeKind.layered, MAT_COAT_SMOOTH, RGB(Float32(0.8)), cos_o, adjoint), tab)


def test_ggx_sample_matches_eval() raises:
    var tab = _tables()
    # alpha >= 0.25: a narrower lobe makes the uniform-sphere reference
    # itself too noisy to judge (at 0.1 it misses the lobe ~99.5% of the time).
    for alpha in [Float32(0.25), Float32(0.5)]:
        for cos_o in [Float32(0.9), Float32(0.4)]:
            var c = _ctx(LobeKind.ggx, -1, RGB(Float32(0.9), Float32(0.6), Float32(0.3)), cos_o, False)
            c.param = alpha
            _check_consistency("ggx alpha " + String(alpha), c, tab)


def test_ggx_mirror_is_delta() raises:
    var tab = _tables()
    var ctx = _test_ctx()
    var h = spectral_handle(ctx)
    var wl = sample_wavelengths_uniform(Float32(0.37))
    var c = _ctx(LobeKind.ggx, -1, RGB(Float32(0.9)), Float32(0.6), False)
    c.param = Float32(0)
    var s = lobe_sample(c, Float32(0.5), Float32(0.5), Float32(0.5), Float32(0.5), tab, h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65, wl)
    assert_true(s.valid and s.is_delta, "a smooth conductor samples a delta mirror")
    # mirror of wo about +z
    assert_true(abs(s.wi[0] + c.wo[0]) < Float32(1e-5) and abs(s.wi[2] - c.wo[2]) < Float32(1e-5), "mirror direction")
    assert_true(s.pdf_fwd == Float32(0) and s.pdf_rev == Float32(0), "a delta event carries no density")
    _ = ctx^


def test_eval_scoped_is_lobe_scoped() raises:
    """lobe_eval's scoped flag must be lobe_scoped's answer, for every kind:
    VCM's NEE reads the former and _connect the latter, and when they
    disagreed for diffuse_transmit a leaf-lit floor read 1.8x pbrt."""
    var tab = _tables()
    var ctx = _test_ctx()
    var h = spectral_handle(ctx)
    var wl = sample_wavelengths_uniform(Float32(0.37))
    var wi = Vec3f(0.3, 0.1, 0.9)
    for kind_mat in [(LobeKind.lambertian, -1), (LobeKind.ggx, -1),
                     (LobeKind.diffuse_transmit, MAT_DT), (LobeKind.layered, MAT_COAT_ROUGH)]:
        var c = _ctx(kind_mat[0], kind_mat[1], RGB(Float32(0.5)), Float32(0.8), False)
        c.param = Float32(0.3)
        var le = lobe_eval[want_pdfs=True](c, wi, tab, h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65, wl)
        assert_true(le.scoped == lobe_scoped(c), "lobe_eval.scoped != lobe_scoped for kind " + String(kind_mat[0]))
    _ = ctx^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
