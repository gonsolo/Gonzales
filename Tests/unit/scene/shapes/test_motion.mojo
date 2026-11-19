from std.math import abs, sqrt, sin, cos
from std.testing import assert_true, TestSuite
from gonzales.scene.shapes.motion import InstanceMotion, MotionTable

comptime M64 = SIMD[DType.float64, 16]


def _identity() -> M64:
    var m = M64(0)
    m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1
    return m


def _rot_z(deg: Float64) -> M64:
    var a = deg * 3.141592653589793 / 180.0
    var m = _identity()
    m[0] = cos(a); m[1] = sin(a); m[4] = -sin(a); m[5] = cos(a)
    return m


def _close(a: SIMD[DType.float32, 16], b: M64) -> Bool:
    for i in range(16):
        if abs(Float64(a[i]) - b[i]) > 1e-4:
            return False
    return True


def test_endpoints_and_translation() raises:
    var start = _identity(); start[12] = 1; start[13] = 2; start[14] = 3
    start[0] = 2   # a scale: Decompose must give it back
    var end = start; end[12] = 5; end[13] = -2; end[14] = 3
    var m = InstanceMotion(start, end, _identity())
    assert_true(_close(m.obj_to_world(0), start))
    assert_true(_close(m.obj_to_world(1), end))
    var mid = start; mid[12] = 3; mid[13] = 0
    assert_true(_close(m.obj_to_world(0.5), mid))


def test_rotation_slerps() raises:
    var m = InstanceMotion(_identity(), _rot_z(90), _identity())
    assert_true(_close(m.obj_to_world(0.5), _rot_z(45)))
    assert_true(abs(m.rotation_angle() - Float32(3.14159265 / 2)) < 1e-4)


def test_base_is_applied_first() raises:
    # A shape baked with its start CTM: the instance transform starts at the identity.
    var start = _rot_z(30); start[12] = 10
    var end = _rot_z(60); end[12] = -10
    var base = _rot_z(-30); base[12] = -10 * cos(-30 * 3.141592653589793 / 180.0); base[13] = -10 * sin(-30 * 3.141592653589793 / 180.0)
    var m = InstanceMotion(start, end, base)
    assert_true(_close(m.obj_to_world(0), _identity()))


def test_table_time() raises:
    var t = MotionTable.none()
    t.t0 = 2; t.t1 = 4
    assert_true(t.at(3).dt() == 0.5)
    assert_true(t.at(1).dt() == 0)
    assert_true(t.at(5).dt() == 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
