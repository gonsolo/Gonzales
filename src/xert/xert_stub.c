#include "xert.h"

#include <stdlib.h>

// Built where the Level Zero headers are missing: no Intel hardware ray tracing.
void* xert_build_blas(const float* tri_verts, int64_t n_tris, int64_t* out_size) { return NULL; }

void* xert_build_blas_geoms(int32_t n_geoms, const float* const* verts, const int64_t* n_verts, const uint32_t* const* tris,
                            const int64_t* n_tris, int64_t n_procs, const float* proc_bounds, int64_t* out_size,
                            float* bounds) { return NULL; }

void* xert_build_tlas(int64_t n_inst, const float* xf, const int32_t* blas_of, const uint64_t* blas_addr,
                      const float* blas_bounds, int64_t* out_size) { return NULL; }

int64_t xert_alignment(void) { return 0; }

void xert_free(void* p) { free(p); }

static struct {
  uint64_t accel, ord_mesh;
  int64_t ord_inst, n_meshes;
  int32_t shadow_slots, alpha;
} scene;

void xert_set_scene(uint64_t accel, int64_t ord_inst, uint64_t ord_mesh, int64_t n_meshes, int32_t shadow_slots, int32_t alpha) {
  scene.accel = accel;
  scene.ord_inst = ord_inst;
  scene.ord_mesh = ord_mesh;
  scene.n_meshes = n_meshes;
  scene.shadow_slots = shadow_slots;
  scene.alpha = alpha;
}

uint64_t xert_scene_accel(void) { return scene.accel; }
int64_t xert_scene_ord_inst(void) { return scene.ord_inst; }
uint64_t xert_scene_ord_mesh(void) { return scene.ord_mesh; }
int64_t xert_scene_n_meshes(void) { return scene.n_meshes; }
int32_t xert_shadow_slots(void) { return scene.shadow_slots; }
int32_t xert_alpha(void) { return scene.alpha; }

static uint64_t curve_tables[5];

void xert_set_curves(uint64_t curves, uint64_t leaves, uint64_t leaf_base, uint64_t inst_blas, uint64_t materials) {
  curve_tables[0] = curves;
  curve_tables[1] = leaves;
  curve_tables[2] = leaf_base;
  curve_tables[3] = inst_blas;
  curve_tables[4] = materials;
}

uint64_t xert_curve_table(int32_t which) { return which >= 0 && which < 5 ? curve_tables[which] : 0; }

static struct {
  uint64_t counts;
  int64_t n;
} heat;

void xert_set_heat(uint64_t counts, int64_t n) {
  heat.counts = counts;
  heat.n = n;
}

uint64_t xert_heat(void) { return heat.counts; }
int64_t xert_heat_n(void) { return heat.n; }
