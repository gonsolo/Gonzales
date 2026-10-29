#include "xert.h"

#include <stdlib.h>

// Built where the Level Zero headers are missing: no Intel hardware ray tracing.
void* xert_build_blas(const float* tri_verts, int64_t n_tris, int64_t* out_size) { return NULL; }

int64_t xert_alignment(void) { return 0; }

void xert_free(void* p) { free(p); }
