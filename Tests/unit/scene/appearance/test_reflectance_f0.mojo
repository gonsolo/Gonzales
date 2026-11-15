"""pbrt's conductor "reflectance": F0 = r in air, the metal's Fresnel against the coat under one."""
from std.math import abs
from std.testing import assert_true, TestSuite
from gonzales.core.geometry import RGB
from gonzales.scene.appearance.materials import reflectance_f0


def test_bare_metal_reflects_r() raises:
    var f = reflectance_f0(RGB(0.1, 0.5, 0.9), 1)
    assert_true(abs(f.r - 0.1) < 1e-5 and abs(f.g - 0.5) < 1e-5 and abs(f.b - 0.9) < 1e-4)


def test_coat_lowers_f0() raises:
    # eta 1/1.5, k 2/1.5: (1/9 + 16/9) / (25/9 + 16/9) = 17/41
    assert_true(abs(reflectance_f0(RGB(0.5), 1.5).g - Float32(17.0 / 41.0)) < 1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
