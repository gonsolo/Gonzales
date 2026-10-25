// optixcurves.h without CUDA or the OptiX SDK: no handle, so the curves stay on the software clusters.
#include "optixcurves.h"

extern "C" {
void *oxc_create(void) { return nullptr; }
int oxc_build(void *, uint64_t, uint64_t, uint32_t, uint64_t, uint64_t, void *) { return 1; }
int oxc_trace_closest(void *, uint64_t, uint64_t, uint32_t, void *) { return 1; }
int oxc_trace_shadow(void *, uint64_t, uint64_t, uint64_t, int32_t, int32_t, uint64_t, uint32_t, void *) { return 1; }
uint64_t oxc_scratch(void *, size_t) { return 0; }
size_t oxc_bytes(void *) { return 0; }
void oxc_destroy(void *) {}
void oxc_set_active(void *) {}
void *oxc_active(void) { return nullptr; }
}
