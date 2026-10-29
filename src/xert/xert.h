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

// The same over n_geoms meshes: geometry g has n_tris[g] triangles at tri_verts[g]; a hit reports the geometry and the
// triangle within it. bounds receives the box (lower x y z, upper x y z).
void* xert_build_blas_geoms(int32_t n_geoms, const float* const* tri_verts, const int64_t* n_tris, int64_t* out_size,
                            float* bounds);

// A top-level structure: instance k places bottom-level structure blas_of[k], copied to device address blas_addr[b]
// with object-space box blas_bounds[6b], by the column-major 3x4 object-to-world transform xf[12k]. A hit reports k.
void* xert_build_tlas(int64_t n_inst, const float* xf, const int32_t* blas_of, const uint64_t* blas_addr,
                      const float* blas_bounds, int64_t* out_size);

// The alignment the device needs for a structure, 0 without an Intel GPU.
int64_t xert_alignment(void);

void xert_free(void* p);

#ifdef __cplusplus
}
#endif

#endif
