// NanoVDB-headers-free stand-in for nvdb.cc.
//
// nanovdb.mojo (the loader FFI wrappers, not the pure-Mojo accessor) is
// imported transitively by the core renderer (geometry.mojo, for
// nvdb_sample_density) as of the "nanovdb" pbrt medium type landing, not
// just by the opt-in Tools/nvdb_diff.mojo harness. So the Mojo link line
// now always pulls in -lnvdbbridge, and a machine without the NanoVDB
// headers (CI's container; anyone who hasn't installed them locally) could
// not build gonzales AT ALL -- exactly the failure mode CUDA-less builds
// used to hit before vulkaninterop_stub.cpp existed (see that file's own
// comment). Same fix, same shape: export the same symbols, report "no
// grid" from every one of them. nvdb_load returning NULL is ALREADY the
// documented "load failed" contract every caller handles (see nvdb.h and
// nanovdb.mojo's own docs) -- MakeNamedMedium "nanovdb" media just warn and
// render as empty air on a build using this stub, exactly like a real
// build hitting a bad/missing file path. Every renderer feature that does
// NOT touch a "nanovdb" medium is completely unaffected.
#include "nvdb.h"

#include <stddef.h>

extern "C" {

void* nvdb_load(const char*, int) { return nullptr; }
void* nvdb_load_named(const char*, const char*) { return nullptr; }
const void* nvdb_data(void*) { return nullptr; }
unsigned long nvdb_size(void*) { return 0ul; }

float nvdb_get_value(void*, int, int, int) { return 0.0f; }
unsigned long nvdb_active_count(void*) { return 0ul; }
void nvdb_active_coord(void*, unsigned long, int* out3) {
    if (out3) { out3[0] = 0; out3[1] = 0; out3[2] = 0; }
}

int nvdb_grid_type(void*) { return 0; }
int nvdb_grid_class(void*) { return 0; }
void nvdb_index_bbox(void*, int* mn, int* mx) {
    // min > max: an empty range, matching the real bridge's own convention
    // for "no grid" so nvdb_sample_density's bounds check rejects every
    // coordinate instead of accepting an arbitrary one.
    if (mn) { mn[0] = mn[1] = mn[2] = 0; }
    if (mx) { mx[0] = mx[1] = mx[2] = -1; }
}
void nvdb_world_bbox(void*, double*, double*) {}
void nvdb_voxel_size(void*, double*) {}
void nvdb_value_range(void*, float* out_min, float* out_max) {
    if (out_min) *out_min = 0.0f;
    if (out_max) *out_max = 0.0f;
}
void nvdb_grid_name(void*, char* out, int cap) {
    if (out && cap > 0) out[0] = '\0';
}

void nvdb_map_matf(void*, float* out9) {
    if (!out9) return;
    for (int i = 0; i < 9; ++i) out9[i] = 0.0f;
}
void nvdb_map_invmatf(void*, float* out9) {
    if (!out9) return;
    for (int i = 0; i < 9; ++i) out9[i] = 0.0f;
}
void nvdb_map_vecf(void*, float* out3) {
    if (out3) { out3[0] = out3[1] = out3[2] = 0.0f; }
}

void nvdb_free(void*) {}

} // extern "C"
