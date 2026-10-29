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

// The same over n_geoms indexed meshes: geometry g has n_verts[g] vertices (3 floats) at verts[g] and n_tris[g] triangles
// (3 vertex indices) at tris[g]; shared edges let the builder pair triangles into quads. A hit reports the geometry and
// the triangle within it. With n_procs > 0 a last geometry holds that many procedural primitives with the boxes
// proc_bounds (6 floats each), which a ray query reports as candidates. bounds receives the box (lower x y z, upper x y z).
void* xert_build_blas_geoms(int32_t n_geoms, const float* const* verts, const int64_t* n_verts, const uint32_t* const* tris,
                            const int64_t* n_tris, int64_t n_procs, const float* proc_bounds, int64_t* out_size,
                            float* bounds);

// A top-level structure: instance k places bottom-level structure blas_of[k], copied to device address blas_addr[b]
// with object-space box blas_bounds[6b], by the column-major 3x4 object-to-world transform xf[12k]. A hit reports k;
// blas_of[k] < 0 leaves instance k out.
void* xert_build_tlas(int64_t n_inst, const float* xf, const int32_t* blas_of, const uint64_t* blas_addr,
                      const float* blas_bounds, int64_t* out_size);

// The scene the wavefront passes trace, set once its structures are on the device: the TLAS's device address (0 = none),
// the instance holding the ordinary meshes and the device table of their mesh numbers, the scene's mesh count, the
// shadow slots to trace in hardware (0 = none) and whether alpha cutouts need the any-hit passes.
void xert_set_scene(uint64_t accel, int64_t ord_inst, uint64_t ord_mesh, int64_t n_meshes, int32_t shadow_slots, int32_t alpha);
uint64_t xert_scene_accel(void);
int64_t xert_scene_ord_inst(void);
uint64_t xert_scene_ord_mesh(void);
int64_t xert_scene_n_meshes(void);
int32_t xert_shadow_slots(void);
int32_t xert_alpha(void);

// The curves of the active scene, device addresses: the Curve array, the curve leaves (PrimId, type 5) of every bottom-level
// structure one after the other, the first leaf of each structure, the structure of each instance and the materials (shadow
// rays pass interface curves). All 0: no curves.
void xert_set_curves(uint64_t curves, uint64_t leaves, uint64_t leaf_base, uint64_t inst_blas, uint64_t materials);
uint64_t xert_curve_table(int32_t which);

// Hit counts while profiling, n device words, one per instance (the ordinary geometry's last); 0 = off.
void xert_set_heat(uint64_t counts, int64_t n);
uint64_t xert_heat(void);
int64_t xert_heat_n(void);

// The alignment the device needs for a structure, 0 without an Intel GPU.
int64_t xert_alignment(void);

void xert_free(void* p);

#ifdef __cplusplus
}
#endif

#endif
