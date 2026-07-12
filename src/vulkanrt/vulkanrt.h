#ifndef GONZALES_VULKANRT_H
#define GONZALES_VULKANRT_H

#ifdef __cplusplus
extern "C" {
#endif

// Task #162 step 1 smoke test: stand up a headless Vulkan instance/device
// requesting VK_KHR_ray_query + VK_KHR_acceleration_structure +
// VK_KHR_deferred_host_operations, build a single-triangle BLAS/TLAS, run
// one ray-query compute dispatch against it, and confirm the ray (hardcoded
// to pass through the triangle) actually reports a hit. No gonzales scene
// data involved yet -- this only proves the Mojo -> C++ bridge -> Vulkan ->
// RT-core round trip works end to end before building the real intersection
// backend on top of it.
//
// Returns 1 if the ray-query hit was reported correctly, 0 on any failure
// (extension unsupported, device creation failed, wrong/missing hit, etc).
// Prints diagnostics to stderr on failure.
int vulkanrt_smoke_test(void);

#ifdef __cplusplus
}
#endif

#endif // GONZALES_VULKANRT_H
