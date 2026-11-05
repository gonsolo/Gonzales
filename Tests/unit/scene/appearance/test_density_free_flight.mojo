# A chromatic density field: the lane-mixture tracker and the ratio tracker against per-lane Beer-Lambert.

from std.math import abs, exp
from std.testing import assert_true, TestSuite
from gonzales.core.geometry import RGB, Vec3f, Point3f
from gonzales.core.rng import PCG32
from gonzales.core.spectrum import load_spectral_context, spectral_handle, sample_wavelengths, SpectralSample
from gonzales.scene.appearance.media import (
    Grid, Medium, NvdbGrid, sample_free_flight, medium_ratio_track, medium_sigma_t_spectral,
)

comptime DATA_DIR = "src/gonzales/data"
comptime N = 50000
comptime DENSITY: Float32 = 0.5
comptime T_SURF: Float32 = 3.0


def _identity() -> SIMD[DType.float32, 16]:
    var m = SIMD[DType.float32, 16](0)
    m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1
    return m


def _medium() -> Medium:
    return Medium(RGB(Float32(0.05)), RGB(Float32(0.20), Float32(0.45), Float32(0.95)), Float32(0.0),
                  Int32(0), Int32(-1), Int32(-1), Float32(0.0), Float32(0.0), Float32(1.0), Int32(0))


def _lane(s: SpectralSample, i: Int) -> Float32:
    return s.get(i)


def _near(est: Float32, want: Float32) -> Bool:
    return abs(est - want) < Float32(0.03) * max(want, Float32(0.05))


def test_density_field_flight_and_ratio_track_match_beer_lambert_per_lane() raises:
    var loaded = load_spectral_context(DATA_DIR)
    if not loaded[0]:
        print("SKIP: no spectrum table")
        return
    var ctx = loaded[1].copy()
    var h = spectral_handle(ctx)
    var dens: List[Float32] = [DENSITY]
    # A loose majorant (4x the density) makes most candidates null events.
    var grids: List[Grid] = [Grid(dens.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), Int32(1), Int32(1), Int32(1),
        Point3f(Float32(-10), Float32(-10), Float32(-10)), Point3f(Float32(10), Float32(10), Float32(10)), _identity(), Float32(2.0))]
    var nvdb = List[NvdbGrid]()
    var med = _medium()
    var wl = sample_wavelengths(Float32(0.37))
    var sig = medium_sigma_t_spectral(med, wl, h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65)
    var o = Vec3f(Float32(0), Float32(0), Float32(0))
    var d = Vec3f(Float32(0), Float32(0), Float32(1))
    var pcg = PCG32(UInt64(7), UInt64(3))
    var pass_w = SpectralSample(Float32(0))
    var coll_w = SpectralSample(Float32(0))
    var pass_rgb = RGB(Float32(0))
    var ratio = SpectralSample(Float32(0))
    for _ in range(N):
        var ff = sample_free_flight(med, grids.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            nvdb.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), o, d, T_SURF, pcg, wl,
            h.coeffs, h.res, h.cie_x, h.cie_y, h.cie_z, h.d65)
        if ff.collided:
            coll_w += ff.lane_weight
        else:
            pass_w += ff.lane_weight
            pass_rgb = pass_rgb + ff.weight
        ratio += medium_ratio_track(med, grids.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            nvdb.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), o, d, Float32(0), T_SURF, sig, pcg)
    var inv = Float32(1) / Float32(N)
    for i in range(4):
        var tr = exp(-DENSITY * _lane(sig, i) * T_SURF)
        assert_true(_near(_lane(pass_w, i) * inv, tr), "pass-through lane weight != per-lane transmittance")
        assert_true(_near(_lane(coll_w, i) * inv, Float32(1) - tr), "collision lane weight != per-lane extinction")
        assert_true(_near(_lane(ratio, i) * inv, tr), "ratio tracking != per-lane transmittance")
    var st = med.sigma_a + med.sigma_s
    assert_true(_near(pass_rgb.r * inv, exp(-DENSITY * st.r * T_SURF)), "RGB pass-through weight, red")
    assert_true(_near(pass_rgb.b * inv, exp(-DENSITY * st.b * T_SURF)), "RGB pass-through weight, blue")
    assert_true(len(dens) == 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
