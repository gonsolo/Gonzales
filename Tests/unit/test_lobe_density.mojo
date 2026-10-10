# Density checks for lobe_sample / lobe_eval that test_lobe_sample.mojo cannot
# make: that test compares f-weighted integrals, which are blind to a pdf that
# is wrong where f is small, and MIS divides by the pdf directly.
#   1. integral of pdf_fwd over the sphere == P(sample is valid and non-delta)
#   2. the histogram of sampled cos(theta) == the mass lobe_eval's pdf_fwd puts
#      in each bin
#   3. pdf_rev == pdf_fwd of the swapped query (wo <-> wi), the reverse density
#      BDPT/VCM feed into their MIS carries
from std.math import abs, sqrt, cos, sin
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, TestSuite
from gonzales.geometry import RGB, Vec3f, PI
from gonzales.materials import Material, MatKind, LobeKind, MeasuredBRDF
from gonzales.curves import Curve
from gonzales.bxdf import LobeCtx, LobeTables, lobe_eval, lobe_sample
from gonzales.rng import PCG32
from gonzales.spectrum import sample_wavelengths, SpectralContext, spectral_handle
from gonzales.rgb2spec import build_spectrum_table, build_cie_xyz_tables, SpectrumTable

comptime _curves = Pointer[Curve, MutUntrackedOrigin].unsafe_dangling()
comptime _mbrdfs = Pointer[MeasuredBRDF, MutUntrackedOrigin].unsafe_dangling()
comptime TEST_RES = 16
comptime NB = 10

comptime MAT_DT = 0
comptime MAT_COAT_ROUGH = 1
comptime MAT_COAT_SMOOTH = 2
comptime MAT_GLASS = 3


def _test_ctx() -> SpectralContext:
    var table = build_spectrum_table(TEST_RES)
    var cie = build_cie_xyz_tables()
    return SpectralContext(SpectrumTable(table^, TEST_RES), cie^)


def _material(type: Int8, albedo: RGB, emission: RGB, rough: Float32) -> Material:
    return Material(type, Int8(0), Int8(0), Int8(0), albedo, emission,
        Int32(-1), rough, rough, Int32(-1), Int32(-1), Float32(1.0), Int32(-1), Int32(-1),
        RGB(Float32(0.0)), RGB(Float32(0.0)), Float32(1.0), Float32(1.0), Int32(-1),
        RGB(Float32(1.0)), RGB(Float32(0.0)), RGB(Float32(1.0)))


def _tables() -> LobeTables:
    var mats = unsafe_alloc[Material](4)
    mats[unsafe_offset=MAT_DT] = _material(MatKind.diffuse_transmit, RGB(Float32(0.3)), RGB(Float32(0.5)), Float32(0))
    mats[unsafe_offset=MAT_COAT_ROUGH] = _material(MatKind.coated_diffuse, RGB(Float32(0.8)), RGB(Float32(1.5), Float32(0.01), Float32(0)), Float32(0.3))
    mats[unsafe_offset=MAT_COAT_SMOOTH] = _material(MatKind.coated_diffuse, RGB(Float32(0.8)), RGB(Float32(1.5), Float32(0.01), Float32(0)), Float32(0))
    mats[unsafe_offset=MAT_GLASS] = _material(MatKind.dielectric, RGB(Float32(1.5)), RGB(Float32(0.0)), Float32(0.3))
    return LobeTables(mats, _curves, _mbrdfs)


def _ctx(kind: Int32, mat_idx: Int, alb: RGB, cos_o: Float32, param: Float32) -> LobeCtx:
    var wo = Vec3f(sqrt(max(Float32(0), Float32(1) - cos_o * cos_o)), Float32(0), cos_o)
    return LobeCtx(kind, True, False, Vec3f(0.0, 0.0, 1.0), wo, alb, Int32(mat_idx),
                   param, Float32(0), Int32(-1), Float32(0), Float32(0), False, False)


def _uniform_sphere(u: Float32, v: Float32) -> Vec3f:
    var z = Float32(1) - Float32(2) * u
    var r = sqrt(max(Float32(0), Float32(1) - z * z))
    var phi = Float32(2) * PI * v
    return Vec3f(r * cos(phi), r * sin(phi), z)


def _bin(z: Float32) -> Int:
    return min(NB - 1, max(0, Int((Float64(z) + 1.0) * 0.5 * Float64(NB))))


def _check_density(name: String, c: LobeCtx, tab: LobeTables, n: Int, rev_tol: Float64,
                   is_density: Bool = True) raises -> Int:
    """Returns the number of failed checks (and prints each one). is_density=False
    skips checks 1 and 2: layered's pdf is pbrt's weights-only estimate (90% walk
    estimate + 10% uniform), not the density of its sampler."""
    var ctx = _test_ctx()
    var h = spectral_handle(ctx)
    var wl = sample_wavelengths(Float32(0.37))
    var rng = PCG32(UInt64(3), UInt64(5))
    var hist = List[Float64]()
    var mass = List[Float64]()
    var mass2 = List[Float64]()
    for _ in range(NB):
        hist.append(0.0)
        mass.append(0.0)
        mass2.append(0.0)
    var sampled_nd = Float64(0)
    var worst_rev = Float64(0)
    for _ in range(n):
        var s = lobe_sample(c, rng.next_float(), rng.next_float(), rng.next_float(), rng.next_float(), tab,
                            h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65, wl)
        if s.valid and not s.is_delta:
            sampled_nd += 1.0
            hist[_bin(s.wi[2])] += 1.0
            if s.pdf_rev > Float32(0):
                var c2 = c
                c2.wo = s.wi
                var le2 = lobe_eval[want_pdfs=True](c2, c.wo, tab, h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65, wl)
                var rel = Float64(abs(le2.pdf_fwd - s.pdf_rev) / max(s.pdf_rev, Float32(1e-6)))
                worst_rev = max(worst_rev, rel)
        var wi = _uniform_sphere(rng.next_float(), rng.next_float())
        var le = lobe_eval[want_pdfs=True](c, wi, tab, h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65, wl)
        var x = Float64(le.pdf_fwd) * Float64(4.0 * 3.14159265358979)
        var b = _bin(wi[2])
        mass[b] += x
        mass2[b] += x * x
    var nf = Float64(n)
    var total = Float64(0)
    var total2 = Float64(0)
    for b in range(NB):
        total += mass[b]
        total2 += mass2[b]
    var p_mass = total / nf
    var p_samp = sampled_nd / nf
    var sig_tot = sqrt(max(Float64(0), (total2 / nf - p_mass * p_mass) / nf) + p_samp * (1.0 - p_samp) / nf)
    var fails = 0
    if is_density and abs(p_mass - p_samp) > 5.0 * sig_tot + 0.01:
        print("FAIL", name, ": integral of pdf", p_mass, "!= P(valid non-delta sample)", p_samp, "sigma", sig_tot)
        fails += 1
    var worst_z = Float64(0)
    var worst_b = 0
    for b in range(NB):
        var ph = hist[b] / nf
        var pm = mass[b] / nf
        var sig = sqrt(max(Float64(0), (mass2[b] / nf - pm * pm) / nf) + ph * (1.0 - ph) / nf)
        var z = abs(ph - pm) / max(sig, Float64(1e-5))
        if abs(ph - pm) > 5.0 * sig + 0.002 and z > worst_z:
            worst_z = z
            worst_b = b
    if is_density and worst_z > Float64(0):
        print("FAIL", name, ": cos(theta) bin", worst_b, "sampled mass", hist[worst_b] / nf,
              "pdf mass", mass[worst_b] / nf, "(", worst_z, "sigma )")
        fails += 1
    if worst_rev > rev_tol:
        print("FAIL", name, ": pdf_rev differs from the swapped pdf_fwd by", worst_rev)
        fails += 1
    print(name, "| int pdf", p_mass, "P(sample)", p_samp, "| worst pdf_rev rel", worst_rev)
    _ = ctx^
    return fails


def test_lambertian_density() raises:
    var tab = _tables()
    var fails = 0
    for cos_o in [Float32(0.9), Float32(0.3), Float32(0.05)]:
        fails += _check_density("lambertian cos_o " + String(cos_o),
            _ctx(LobeKind.lambertian, -1, RGB(Float32(0.6)), cos_o, Float32(0)), tab, 50000, 1e-3)
    assert_true(fails == 0, "lambertian density checks failed")


def test_diffuse_transmit_density() raises:
    var tab = _tables()
    var fails = 0
    for cos_o in [Float32(0.9), Float32(-0.4)]:
        fails += _check_density("diffuse_transmit cos_o " + String(cos_o),
            _ctx(LobeKind.diffuse_transmit, MAT_DT, RGB(Float32(0.3)), cos_o, Float32(0)), tab, 50000, 1e-3)
    assert_true(fails == 0, "diffuse_transmit density checks failed")


def test_ggx_density() raises:
    var tab = _tables()
    var fails = 0
    for alpha in [Float32(0.25), Float32(0.5), Float32(1.0)]:
        for cos_o in [Float32(0.9), Float32(0.4), Float32(0.1)]:
            fails += _check_density("ggx alpha " + String(alpha) + " cos_o " + String(cos_o),
                _ctx(LobeKind.ggx, -1, RGB(Float32(0.9), Float32(0.6), Float32(0.3)), cos_o, alpha), tab, 50000, 2e-3)
    assert_true(fails == 0, "ggx density checks failed")


def test_rough_dielectric_density() raises:
    var tab = _tables()
    var fails = 0
    for alpha in [Float32(0.3), Float32(0.6)]:
        for cos_o in [Float32(0.9), Float32(0.4), Float32(-0.9), Float32(-0.4)]:
            fails += _check_density("rough_dielectric alpha " + String(alpha) + " cos_o " + String(cos_o),
                _ctx(LobeKind.rough_dielectric, MAT_GLASS, RGB(Float32(1.0)), cos_o, alpha), tab, 50000, 2e-3)
    assert_true(fails == 0, "rough_dielectric density checks failed")


def test_layered_density() raises:
    var tab = _tables()
    var fails = 0
    for cos_o in [Float32(0.9), Float32(0.4)]:
        fails += _check_density("layered rough cos_o " + String(cos_o),
            _ctx(LobeKind.layered, MAT_COAT_ROUGH, RGB(Float32(0.8)), cos_o, Float32(0)), tab, 40000, 1e-3, False)
        fails += _check_density("layered smooth cos_o " + String(cos_o),
            _ctx(LobeKind.layered, MAT_COAT_SMOOTH, RGB(Float32(0.8)), cos_o, Float32(0)), tab, 40000, 1e-3, False)
    assert_true(fails == 0, "layered density checks failed")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
