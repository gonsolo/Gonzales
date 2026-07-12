from std.sys import has_accelerator
from std.testing import assert_true, TestSuite
from gonzales.vulkanrt import vulkanrt_smoke_test

# Proves the full Mojo -> C++ bridge -> Vulkan -> RT-core round trip for
# task #162's first step: a headless VK_KHR_ray_query dispatch against a
# hardcoded single-triangle BLAS/TLAS, traced by a ray known to hit it.
# See src/vulkanrt/vulkanrt.cpp for the implementation.

def test_vulkanrt_smoke_test_hits_triangle() raises:
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return

    assert_true(vulkanrt_smoke_test() == 1)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
