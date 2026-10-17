from std.os import setenv, unsetenv
from std.testing import assert_equal, assert_not_equal, TestSuite
from gonzales.scene_cache import scene_cache_variant


def test_culling_changes_the_variant() raises:
    _ = unsetenv("GONZALES_CULL_SIZE")
    _ = unsetenv("GONZALES_CAMERA_SCENE")
    var plain = scene_cache_variant()
    _ = setenv("GONZALES_CULL_SIZE", "0", True)
    assert_equal(scene_cache_variant(), plain)
    _ = setenv("GONZALES_CULL_SIZE", "0.0015", True)
    var culled = scene_cache_variant()
    assert_not_equal(culled, plain)
    _ = setenv("GONZALES_CAMERA_SCENE", "other.pbrt", True)
    assert_not_equal(scene_cache_variant(), culled)
    _ = unsetenv("GONZALES_CULL_SIZE")
    _ = unsetenv("GONZALES_CAMERA_SCENE")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
