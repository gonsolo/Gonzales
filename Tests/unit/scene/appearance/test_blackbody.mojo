"""Blackbody emission as pbrt delivers it: unit luminance for ordinary lights, the peak-normalised spectrum's
luminance for a portal light. Reference values integrate Planck against the CIE 1931 y-bar fit at 1 nm."""
from std.math import abs
from std.testing import assert_true, TestSuite
from gonzales.scene.appearance.media import blackbody_rgb, blackbody_peak_luminance


def test_blackbody_rgb_has_unit_luminance() raises:
    var c = blackbody_rgb(Float32(6500))
    var y = Float32(0.2126) * c.r + Float32(0.7152) * c.g + Float32(0.0722) * c.b
    assert_true(abs(y - Float32(1)) < Float32(0.01))


def test_peak_normalised_luminance() raises:
    assert_true(abs(blackbody_peak_luminance(Float32(6500)) - Float32(0.8856)) < Float32(0.005))
    assert_true(abs(blackbody_peak_luminance(Float32(5500)) - Float32(0.9795)) < Float32(0.005))
    assert_true(abs(blackbody_peak_luminance(Float32(3000)) - Float32(0.4164)) < Float32(0.005))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
