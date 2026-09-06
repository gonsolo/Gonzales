// C bridge to NanoVDB's file reader.
//
// Why a bridge rather than parsing .nvdb in Mojo: the grid blob inside a
// .nvdb file is codec-compressed (bunny_cloud.nvdb is ZIP, 146.6MB of grid
// packed into 76MB on disk), and gonzales has no in-memory zlib -- its
// only decompression today shells out to the gzip CLI for whole .pbrt.gz
// files, which cannot address a byte range inside a larger binary. The
// files are also written by NanoVDB 32.3.3 while the headers here are
// 32.9.0, so letting NanoVDB's own reader handle codec + version skew is
// both less code and more robust than hand-rolling it.
//
// What crosses the boundary is deliberately minimal: this returns the
// UNCOMPRESSED grid blob as a flat byte range. All traversal of that blob
// happens in Mojo, using the PNanoVDB-style flat offset arithmetic, so the
// exact same accessor works on CPU and GPU with no second implementation
// (the mistake this codebase already paid for with the VCM backends).
#ifdef __cplusplus
extern "C" {
#endif

// Opens `path` and reads grid `n` (0 = first). Returns an opaque handle, or
// null on failure. The handle owns the decompressed blob.
void* nvdb_load(const char* path, int n);

// Flat, decompressed grid blob: memcpy-able straight to the GPU.
const void* nvdb_data(void* handle);
unsigned long nvdb_size(void* handle);

// Metadata, all read from the grid itself (not the file header, so it is
// still correct for a grid handed over by any other route later).
// grid_type: 1 = Float (the only one wired up so far).
// grid_class: 2 = FogVolume.
int nvdb_grid_type(void* handle);
int nvdb_grid_class(void* handle);
void nvdb_index_bbox(void* handle, int* out_min3, int* out_max3);
void nvdb_world_bbox(void* handle, double* out_min3, double* out_max3);
void nvdb_voxel_size(void* handle, double* out3);
// Value range over the whole grid -- the majorant delta/ratio tracking needs.
void nvdb_value_range(void* handle, float* out_min, float* out_max);
// Grid name, copied into `out` (NUL-terminated, truncated to cap).
void nvdb_grid_name(void* handle, char* out, int cap);

void nvdb_free(void* handle);

#ifdef __cplusplus
}
#endif
