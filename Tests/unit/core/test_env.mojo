from std.os import unsetenv
from std.testing import assert_equal, assert_false, assert_true, TestSuite
from gonzales.core.env import ENV_SWITCHES, env_int, env_on, env_registered, env_set, env_str


def test_names_are_unique_and_described() raises:
    comptime for i in range(len(ENV_SWITCHES)):
        comptime s = ENV_SWITCHES[i]
        assert_true(s.desc != "")
        comptime for j in range(i):
            comptime t = ENV_SWITCHES[j]
            assert_true(s.name != t.name, s.name)


def test_registered() raises:
    assert_true(env_registered("GONZALES_SCENE_CACHE"))
    assert_true(env_registered("GZ_VCM_TIMING"))
    assert_false(env_registered("GONZALES_SCENE_CACHES"))


def test_accessors() raises:
    _ = unsetenv("GONZALES_QUAD_LEAF")
    assert_false(env_on["GONZALES_QUAD_LEAF"]())
    assert_equal(env_str["GONZALES_QUAD_LEAF"]("8"), "8")
    assert_equal(env_int["GONZALES_QUAD_LEAF"](8), 8)
    env_set["GONZALES_QUAD_LEAF"]("")
    assert_false(env_on["GONZALES_QUAD_LEAF"]())
    assert_equal(env_str["GONZALES_QUAD_LEAF"]("8"), "")
    assert_equal(env_int["GONZALES_QUAD_LEAF"](8), 8)
    env_set["GONZALES_QUAD_LEAF"]("2")
    assert_true(env_on["GONZALES_QUAD_LEAF"]())
    assert_equal(env_int["GONZALES_QUAD_LEAF"](8), 2)
    _ = unsetenv("GONZALES_QUAD_LEAF")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
