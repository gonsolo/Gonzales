#include "nvdb.h"

#include <cstring>
#include <string>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>

namespace {

struct Handle {
    nanovdb::GridHandle<nanovdb::HostBuffer> h;
};

// Only Float grids are wired up so far; everything else is rejected at load
// rather than silently mis-read as float further down.
const nanovdb::FloatGrid* float_grid(Handle* H) {
    return H ? H->h.grid<float>() : nullptr;
}

} // namespace

extern "C" {

void* nvdb_load(const char* path, int n) {
    if (!path) return nullptr;
    try {
        auto gh = nanovdb::io::readGrid<nanovdb::HostBuffer>(std::string(path), n);
        if (!gh.gridMetaData()) return nullptr;
        auto* H = new Handle{std::move(gh)};
        if (!float_grid(H)) {   // not a float grid -- refuse rather than misread
            delete H;
            return nullptr;
        }
        return H;
    } catch (...) {
        return nullptr;
    }
}

const void* nvdb_data(void* handle) {
    auto* H = static_cast<Handle*>(handle);
    return H ? H->h.data() : nullptr;
}

unsigned long nvdb_size(void* handle) {
    auto* H = static_cast<Handle*>(handle);
    return H ? (unsigned long)H->h.bufferSize() : 0ul;
}

int nvdb_grid_type(void* handle) {
    auto* H = static_cast<Handle*>(handle);
    if (!H || !H->h.gridMetaData()) return 0;
    return (int)H->h.gridMetaData()->gridType();
}

int nvdb_grid_class(void* handle) {
    auto* H = static_cast<Handle*>(handle);
    if (!H || !H->h.gridMetaData()) return 0;
    return (int)H->h.gridMetaData()->gridClass();
}

void nvdb_index_bbox(void* handle, int* mn, int* mx) {
    auto* g = float_grid(static_cast<Handle*>(handle));
    if (!g || !mn || !mx) return;
    auto bb = g->indexBBox();
    mn[0] = bb.min()[0]; mn[1] = bb.min()[1]; mn[2] = bb.min()[2];
    mx[0] = bb.max()[0]; mx[1] = bb.max()[1]; mx[2] = bb.max()[2];
}

void nvdb_world_bbox(void* handle, double* mn, double* mx) {
    auto* g = float_grid(static_cast<Handle*>(handle));
    if (!g || !mn || !mx) return;
    auto bb = g->worldBBox();
    mn[0] = bb.min()[0]; mn[1] = bb.min()[1]; mn[2] = bb.min()[2];
    mx[0] = bb.max()[0]; mx[1] = bb.max()[1]; mx[2] = bb.max()[2];
}

void nvdb_voxel_size(void* handle, double* out3) {
    auto* g = float_grid(static_cast<Handle*>(handle));
    if (!g || !out3) return;
    auto v = g->voxelSize();
    out3[0] = v[0]; out3[1] = v[1]; out3[2] = v[2];
}

void nvdb_value_range(void* handle, float* out_min, float* out_max) {
    auto* g = float_grid(static_cast<Handle*>(handle));
    if (!g) return;
    // Root-node stats: the majorant over the whole grid. Per-leaf majorants
    // come from the blob itself during traversal, which is strictly tighter
    // than this and than uniformgrid's single global max_density.
    if (out_min) *out_min = g->tree().root().minimum();
    if (out_max) *out_max = g->tree().root().maximum();
}

void nvdb_grid_name(void* handle, char* out, int cap) {
    auto* H = static_cast<Handle*>(handle);
    if (!out || cap <= 0) return;
    out[0] = '\0';
    if (!H || !H->h.gridMetaData()) return;
    std::strncpy(out, H->h.gridMetaData()->shortGridName(), (size_t)cap - 1);
    out[cap - 1] = '\0';
}

void nvdb_free(void* handle) {
    delete static_cast<Handle*>(handle);
}

} // extern "C"
