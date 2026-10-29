#ifndef GONZALES_XERT_H
#define GONZALES_XERT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Hardware ray tracing on Intel Xe2 GPUs: acceleration structures from the driver's builder (ZE_extension_rtas),
// traced by a Mojo kernel through IGC's ray-query built-ins. Level Zero is loaded at run time.

// Builds a bottom-level structure over n_tris triangles, 9 floats each (unindexed); the primitive id of a hit is the
// triangle's index. Returns a buffer of *out_size bytes to copy into device memory as is (free with xert_free), or
// NULL without an Intel GPU or on failure.
void* xert_build_blas(const float* tri_verts, int64_t n_tris, int64_t* out_size);

// The alignment the device needs for a structure, 0 without an Intel GPU.
int64_t xert_alignment(void);

void xert_free(void* p);

#ifdef __cplusplus
}
#endif

#endif
