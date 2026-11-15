"""pbrt's portal light is one-sided: light comes in against portalFrame.z = p03 x p01."""
from std.testing import assert_true, assert_false, TestSuite
from gonzales.core.geometry import Point3f, Vec3f
from gonzales.scene.appearance.portal_light import portal_passes


def _p(x: Float32, y: Float32, z: Float32) -> Point3f:
    return Point3f(x, y, z)


def test_upward_portal_lights_the_room_below() raises:
    # p03 = +x, p01 = -z: portalFrame.z = x cross -z = +y, so a ray from below toward the sky passes
    var p0 = _p(0, 1, 1); var p1 = _p(0, 1, 0); var p3 = _p(1, 1, 1)
    assert_true(portal_passes(_p(0.5, 0, 0.5), Vec3f(0, 1, 0), p0, p1, p3, True))
    assert_false(portal_passes(_p(0.5, 0, 0.5), Vec3f(0, 1, 0), p0, p1, p3, False))
    assert_true(portal_passes(_p(0.5, 2, 0.5), Vec3f(0, -1, 0), p0, p1, p3, False))   # light tracing in


def test_reversed_winding_is_dark_from_below() raises:
    # watercolor's second window: the same quad wound the other way faces down
    var p0 = _p(0, 1, 1); var p1 = _p(1, 1, 1); var p3 = _p(0, 1, 0)
    assert_false(portal_passes(_p(0.5, 0, 0.5), Vec3f(0, 1, 0), p0, p1, p3, True))


def test_miss_beside_the_quad() raises:
    var p0 = _p(0, 1, 1); var p1 = _p(0, 1, 0); var p3 = _p(1, 1, 1)
    assert_false(portal_passes(_p(3, 0, 0.5), Vec3f(0, 1, 0), p0, p1, p3, True))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
