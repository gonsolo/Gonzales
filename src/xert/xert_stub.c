#include "xert.h"

#include <stdlib.h>

// Built where the Level Zero headers are missing: no Intel hardware ray tracing.
void* xert_build_blas(const float* tri_verts, int64_t n_tris, int64_t* out_size) { return NULL; }

void* xert_build_blas_geoms(int32_t n_geoms, const float* const* tri_verts, const int64_t* n_tris, int64_t* out_size,
                            float* bounds) { return NULL; }

void* xert_build_tlas(int64_t n_inst, const float* xf, const int32_t* blas_of, const uint64_t* blas_addr,
                      const float* blas_bounds, int64_t* out_size) { return NULL; }

int64_t xert_alignment(void) { return 0; }

void xert_free(void* p) { free(p); }
