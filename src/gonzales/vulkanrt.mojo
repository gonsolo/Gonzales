from std.ffi import external_call

# Task #162 step 1: Mojo-side FFI wrapper for the headless Vulkan
# ray-query smoke test (src/vulkanrt/vulkanrt.cpp). Completes the
# Mojo -> C++ bridge -> Vulkan -> RT-core round trip this step exists to
# prove -- see project_vulkan_rt_backend memory for the full architecture
# plan this is the first concrete step of.
#
# Builds a single hardcoded triangle + a ray known to hit it entirely on
# the C++ side (no gonzales scene data crosses the FFI boundary yet), runs
# one ray-query compute dispatch on the GPU, and reports whether the
# hardware-traced hit matches the expected result.
# Returns 1 on a correct hit, 0 on any failure (extension unsupported,
# device creation failed, wrong/missing hit). Diagnostics go to stderr.
def vulkanrt_smoke_test() -> Int32:
    return external_call["vulkanrt_smoke_test", Int32]()
