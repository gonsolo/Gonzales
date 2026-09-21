"""White-furnace energy conservation for the rough conductor, measured on the
BxDF itself rather than through a render.

A conductor with reflectance 1 absorbs nothing, so

    integral over the hemisphere of f(wo,wi) cos(wi) dwi  ==  1

at EVERY roughness and EVERY outgoing angle. Single-scattering GGX cannot
reach that on its own -- it drops the light that bounces between microfacets
before escaping -- which is what the Kulla-Conty compensation lobe adds back.

Scenes/furnace/conductor-a*.pbrt asks the same question of the whole
renderer. This file asks it of the four functions underneath, so that a
future failure says WHICH half broke:

  * test_conductor_eval_white_furnace   -- f is right (quadrature over f)
  * test_conductor_sample_white_furnace -- the SAMPLING WEIGHT is right
  * test_conductor_sample_matches_pdf   -- and the two agree, which is the
                                           property MIS actually depends on

The second of those is not redundant with the first. The sampling weight was
the bare Fresnel with G2/G1 dropped for as long as this renderer has existed,
and no test of f could have seen it: f was correct the whole time. It took
isolating the strategy on the furnace scene (NEE off, MIS forced to 1) to
catch a strategy reading P(wi above the horizon) instead of the directional
albedo -- 1.6x too bright at alpha=1. A unit test belongs on the weight.
"""
from std.math import abs, sqrt, cos, sin, acos
from std.testing import assert_true, TestSuite
from gonzales.geometry import RGB, Vec3f, dot, PI, INV_PI
from gonzales.rng import PCG32
from gonzales.bxdf import (
    GeomContext, Material_C,
    bxdf_sample_conductor, bxdf_eval_conductor_ggx, bxdf_pdf_conductor_ggx,
    ggx_albedo, ggx_albedo_avg, _eval_conductor_ggx_spectral,
)
from gonzales.spectrum import SampledWavelengths, null_spectral_handle

# All-equal wavelengths + null_spectral_handle: the upsampler degrades to the
# RGB passthrough (v0,v1,v2 = r,g,b), so a white f0 comes back as exactly 1
# on every lane and the spectral evaluator's furnace integral is directly
# comparable to the RGB one's.
comptime NULL_WL = SampledWavelengths(Float32(0.0), Float32(0.0), Float32(0.0),
                                      Float32(0.0), Float32(0.0))

# Plain functions rather than comptime arrays: a comptime Array[Float32, N]
# is not ImplicitlyCopyable, so indexing it from a runtime loop will not
# materialize.
def _alpha_at(i: Int) -> Float32:
    """The roughness sweep -- the same five the furnace scenes render."""
    if i == 0: return Float32(0.1)
    if i == 1: return Float32(0.2)
    if i == 2: return Float32(0.4)
    if i == 3: return Float32(0.7)
    return Float32(1.0)

def _mu_o_at(i: Int) -> Float32:
    """Normal, moderate and grazing view angles. The furnace SCENE only ever
    sees mu_o near 1 (its crop is the middle of a quad viewed head-on), so
    the off-normal columns are coverage this file adds on top of it."""
    if i == 0: return Float32(1.0)
    if i == 1: return Float32(0.7)
    return Float32(0.4)


def _white_material(alpha: Float32) -> Material_C:
    """reflectance 1 -- the only setting for which the answer is arithmetic."""
    return Material_C(Int8(0), Int8(0), Int8(0), Int8(0), RGB(Float32(1.0)),
        RGB(Float32(0.0)), Int32(-1), alpha, alpha, Int32(-1), Int32(-1),
        Float32(1.0), Int32(-1), Int32(-1), RGB(Float32(0.0)), RGB(Float32(0.0)),
        Float32(1.0), Float32(1.0), Int32(-1), RGB(Float32(1.0)),
        RGB(Float32(0.0)), RGB(Float32(1.0)))


def _wo_at(mu: Float32) -> Vec3f:
    var st = sqrt(max(Float32(0.0), Float32(1.0) - mu * mu))
    return Vec3f(st, Float32(0.0), mu)


def _gc(wo: Vec3f) -> GeomContext:
    var n = Vec3f(Float32(0.0), Float32(0.0), Float32(1.0))
    return GeomContext(n, n, Vec3f(Float32(0.0), Float32(0.0), Float32(0.0)), wo,
        Vec3f(Float32(1.0), Float32(0.0), Float32(0.0)),
        Vec3f(Float32(0.0), Float32(1.0), Float32(0.0)),
        RGB(Float32(0.0)), Float32(0.0))


def test_conductor_eval_white_furnace() raises:
    """Quadrature of f*cos over the hemisphere, straight off the evaluator
    the NEE and photon-gather paths use. 64 x 256 cells on (cos theta, phi);
    the integrand is smooth away from the specular peak, and the peak is
    where cos theta is coarsest in solid angle, so this converges even at
    alpha = 0.1."""
    var n = Vec3f(Float32(0.0), Float32(0.0), Float32(1.0))
    comptime NT = 64
    comptime NP = 256
    for ai in range(5):
        var alpha = _alpha_at(ai)
        for mi in range(3):
            var wo = _wo_at(_mu_o_at(mi))
            var total = Float32(0.0)
            for ti in range(NT):
                # uniform in cos(theta): dwi = dcos * dphi
                var mu_i = (Float32(ti) + Float32(0.5)) / Float32(NT)
                var st = sqrt(max(Float32(0.0), Float32(1.0) - mu_i * mu_i))
                for pi_ in range(NP):
                    var phi = Float32(2.0) * PI * (Float32(pi_) + Float32(0.5)) / Float32(NP)
                    var wi = Vec3f(st * cos(phi), st * sin(phi), mu_i)
                    var f = bxdf_eval_conductor_ggx(n, wo, wi, alpha, RGB(Float32(1.0)))
                    total += f.r * mu_i
            total *= (Float32(2.0) * PI) / (Float32(NT) * Float32(NP))
            # 5%: the quadrature itself is the loose end at low alpha, not
            # the BxDF -- a 64-cell cos(theta) grid cannot resolve a lobe
            # whose width is alpha.
            assert_true(abs(total - Float32(1.0)) < Float32(0.05))


def test_conductor_eval_spectral_white_furnace() raises:
    """The SPECTRAL evaluator must conserve energy too -- it is a SECOND
    implementation of f_ss + f_ms, not a wrapper over the RGB one.

    It earned its own test the hard way. After Kulla-Conty landed, SPPM's
    conductor furnace sat at 0.84 with the deficit FLAT across passes and
    photon counts, i.e. bias, while the RGB evaluator's own furnace test
    passed. The path tracer hid it: PT combines this evaluator's NEE with
    BSDF sampling under MIS, so an error in one strategy is partly masked by
    the other. SPPM has no second strategy at a visible point, so it reads
    the spectral evaluator RAW -- which is exactly why SPPM is the better
    witness for this function and why the furnace belongs on it directly."""
    var n = Vec3f(Float32(0.0), Float32(0.0), Float32(1.0))
    var h = null_spectral_handle()
    comptime NT = 64
    comptime NP = 256
    for ai in range(5):
        var alpha = _alpha_at(ai)
        for mi in range(3):
            var wo = _wo_at(_mu_o_at(mi))
            var total = Float32(0.0)
            for ti in range(NT):
                var mu_i = (Float32(ti) + Float32(0.5)) / Float32(NT)
                var st = sqrt(max(Float32(0.0), Float32(1.0) - mu_i * mu_i))
                for pi_ in range(NP):
                    var phi = Float32(2.0) * PI * (Float32(pi_) + Float32(0.5)) / Float32(NP)
                    var wi = Vec3f(st * cos(phi), st * sin(phi), mu_i)
                    # returns f*cos already -- no cosine applied here
                    var fs = _eval_conductor_ggx_spectral(
                        n, wo, wi, alpha, RGB(Float32(1.0)), h.coeffs, h.res,
                        h.cie_x, h.cie_y, h.cie_z, h.d65, NULL_WL)
                    total += fs.v0
            total *= (Float32(2.0) * PI) / (Float32(NT) * Float32(NP))
            if abs(total - Float32(1.0)) >= Float32(0.05):
                print("  SPECTRAL alpha", alpha, " mu_o", _mu_o_at(mi), " integral", total)
            assert_true(abs(total - Float32(1.0)) < Float32(0.05))


def test_conductor_sample_white_furnace() raises:
    """The same integral, estimated the way the integrators actually reach
    it: E[weight] over bxdf_sample_conductor. This is the test that would
    have caught the missing G2/G1."""
    var rng = PCG32(UInt64(0x9E3779B97F4A7C15), UInt64(1))
    comptime N = 200000
    for ai in range(5):
        var alpha = _alpha_at(ai)
        var mat = _white_material(alpha)
        for mi in range(3):
            var gc = _gc(_wo_at(_mu_o_at(mi)))
            var total = Float32(0.0)
            for _ in range(N):
                var bs = bxdf_sample_conductor(gc, mat, rng.next_float(), rng.next_float())
                if bs.is_valid != Int8(0):
                    total += bs.f.r
            var e = total / Float32(N)
            # Name the offending cell: "the conductor loses energy" is not
            # actionable, "it loses 12% at alpha=1.0, mu_o=0.4 and nothing at
            # mu_o=1.0" points straight at an incidence-dependent term.
            if abs(e - Float32(1.0)) >= Float32(0.03):
                print("  alpha", alpha, " mu_o", _mu_o_at(mi), " E[weight]", e)
            assert_true(abs(e - Float32(1.0)) < Float32(0.03))


def test_conductor_sample_matches_pdf() raises:
    """The pdf must be the density bxdf_sample_conductor draws from.

    bxdf_pdf_conductor_ggx must be the density bxdf_sample_conductor draws
    from -- MIS divides one by the other and is biased the moment they drift.
    Checked by re-deriving the weight from the evaluator and the pdf and
    comparing it against the one the sampler returned."""
    var n = Vec3f(Float32(0.0), Float32(0.0), Float32(1.0))
    var rng = PCG32(UInt64(0x2545F4914F6CDD1D), UInt64(2))
    for ai in range(5):
        var alpha = _alpha_at(ai)
        var mat = _white_material(alpha)
        for mi in range(3):
            var wo = _wo_at(_mu_o_at(mi))
            var gc = _gc(wo)
            for _ in range(2000):
                var bs = bxdf_sample_conductor(gc, mat, rng.next_float(), rng.next_float())
                if bs.is_valid == Int8(0):
                    continue
                var cos_i = dot(bs.wi, n)
                if cos_i <= Float32(1e-4):
                    continue
                var pdf = bxdf_pdf_conductor_ggx(n, wo, bs.wi, alpha)
                if pdf <= Float32(1e-6):
                    continue
                var f = bxdf_eval_conductor_ggx(n, wo, bs.wi, alpha, RGB(Float32(1.0)))
                var expect = f.r * cos_i / pdf
                assert_true(abs(bs.f.r - expect) < Float32(1e-3) * max(Float32(1.0), expect))


def test_ggx_albedo_table_is_consistent() raises:
    """E_avg(alpha) must be the cosine-weighted mean of E(mu, alpha), because
    the compensation lobe's normalisation 1/(pi(1-E_avg)) is what makes
    f_ss + f_ms integrate to 1. Two independent polynomial fits; if they ever
    drift apart the furnace goes off by exactly that drift."""
    comptime NM = 2000
    for ai in range(5):
        var alpha = _alpha_at(ai)
        var acc = Float32(0.0)
        for i in range(NM):
            var mu = (Float32(i) + Float32(0.5)) / Float32(NM)
            acc += ggx_albedo(mu, alpha) * Float32(2.0) * mu
        acc /= Float32(NM)
        assert_true(abs(acc - ggx_albedo_avg(alpha)) < Float32(0.01))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
