#include "xert.h"

#include <dlfcn.h>
#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ZE_FNS(X)                                                                                                     \
  X(zeInit) X(zeDriverGet) X(zeDeviceGet) X(zeDeviceGetProperties) X(zeRTASBuilderCreateExt)                         \
  X(zeRTASBuilderGetBuildPropertiesExt) X(zeRTASBuilderBuildExt) X(zeRTASParallelOperationCreateExt)                 \
  X(zeRTASParallelOperationJoinExt) X(zeRTASParallelOperationDestroyExt)

static struct {
#define X(f) __typeof__(&f) f;
  ZE_FNS(X)
#undef X
  int ready;
  ze_driver_handle_t driver;
  ze_rtas_format_ext_t format;
  uint32_t alignment;
  ze_rtas_builder_ext_handle_t builder;
} ze;

static int xert_init(void) {
  if (ze.ready)
    return ze.ready > 0;
  ze.ready = -1;
  // Spatial splits store triangle pairs in several leaves (Moana's BLASes 5.6 -> 9.6 GB); GONZALES_XERT_SPLITS=1 keeps them.
  const char* splits = getenv("GONZALES_XERT_SPLITS");
  if (!(splits && splits[0] == '1'))
    setenv("ZE_RTAS_SPATIAL_SPLITS", "0", 0);
  void* lib = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!lib)
    return 0;
#define X(f) if (!(ze.f = (__typeof__(&f))dlsym(lib, #f))) return 0;
  ZE_FNS(X)
#undef X
  uint32_t n = 1;
  ze_device_handle_t device;
  if (ze.zeInit(ZE_INIT_FLAG_GPU_ONLY) || ze.zeDriverGet(&n, &ze.driver) || n == 0)
    return 0;
  n = 1;
  if (ze.zeDeviceGet(ze.driver, &n, &device) || n == 0)
    return 0;
  ze_rtas_device_ext_properties_t rtas = {ZE_STRUCTURE_TYPE_RTAS_DEVICE_EXT_PROPERTIES};
  ze_device_properties_t props = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, &rtas};
  if (ze.zeDeviceGetProperties(device, &props) || rtas.rtasFormat == ZE_RTAS_FORMAT_EXT_INVALID)
    return 0;
  ze.format = rtas.rtasFormat;
  ze.alignment = rtas.rtasBufferAlignment;
  ze_rtas_builder_ext_desc_t desc = {ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXT_DESC};
  desc.builderVersion = ZE_RTAS_BUILDER_EXT_VERSION_CURRENT;
  if (ze.zeRTASBuilderCreateExt(ze.driver, &desc, &ze.builder))
    return 0;
  ze.ready = 1;
  return 1;
}

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

int64_t xert_alignment(void) { return xert_init() ? ze.alignment : 0; }

void xert_free(void* p) { free(p); }

static void* xert_build(const ze_rtas_builder_geometry_info_ext_t** geoms, uint32_t n_geoms, int64_t* out_size, float* box) {
  ze_rtas_builder_build_op_ext_desc_t op = {ZE_STRUCTURE_TYPE_RTAS_BUILDER_BUILD_OP_EXT_DESC};
  op.rtasFormat = ze.format;
  op.buildQuality = ZE_RTAS_BUILDER_BUILD_QUALITY_HINT_EXT_HIGH;
  op.ppGeometries = geoms;
  op.numGeometries = n_geoms;
  ze_rtas_builder_ext_properties_t props = {ZE_STRUCTURE_TYPE_RTAS_BUILDER_EXT_PROPERTIES};
  if (ze.zeRTASBuilderGetBuildPropertiesExt(ze.builder, &op, &props))
    return NULL;
  size_t bytes = (props.rtasBufferSizeBytesMaxRequired + ze.alignment - 1) / ze.alignment * ze.alignment;
  void* rtas = aligned_alloc(ze.alignment, bytes);
  void* scratch = malloc(props.scratchBufferSizeBytes + 1);
  ze_rtas_parallel_operation_ext_handle_t par = NULL;
  ze_rtas_aabb_ext_t bounds;
  size_t used = 0;
  // The driver dereferences the parallel operation even for a synchronous build.
  ze_result_t r = ze.zeRTASParallelOperationCreateExt(ze.driver, &par);
  if (r == ZE_RESULT_SUCCESS)
    r = ze.zeRTASBuilderBuildExt(ze.builder, &op, scratch, props.scratchBufferSizeBytes, rtas, bytes, par, NULL,
                                 &bounds, &used);
  if (r == ZE_RESULT_EXT_RTAS_BUILD_DEFERRED)
    r = ze.zeRTASParallelOperationJoinExt(par);
  if (par)
    ze.zeRTASParallelOperationDestroyExt(par);
  free(scratch);
  if (r != ZE_RESULT_SUCCESS) {
    fprintf(stderr, "xert: build failed 0x%x\n", r);
    free(rtas);
    return NULL;
  }
  *out_size = (int64_t)used;
  memcpy(box, &bounds, sizeof(bounds));
  return rtas;
}

static void xert_proc_bounds(ze_rtas_geometry_aabbs_ext_cb_params_t* params) {
  const float* b = (const float*)params->pGeomUserPtr + 6 * (size_t)params->primID;
  memcpy(params->pBoundsOut, b, 6 * sizeof(float) * params->primIDCount);
}

void* xert_build_blas_geoms(int32_t n_geoms, const float* const* verts, const int64_t* n_verts, const uint32_t* const* tris,
                            const int64_t* n_tris, int64_t n_procs, const float* proc_bounds, int64_t* out_size,
                            float* bounds) {
  if (!xert_init() || n_geoms + (n_procs > 0) <= 0)
    return NULL;
  ze_rtas_builder_triangles_geometry_info_ext_t* meshes = calloc(n_geoms > 0 ? n_geoms : 1, sizeof(*meshes));
  const ze_rtas_builder_geometry_info_ext_t** geoms = malloc((n_geoms + 1) * sizeof(*geoms));
  for (int32_t g = 0; g < n_geoms; g++) {
    ze_rtas_builder_triangles_geometry_info_ext_t* m = &meshes[g];
    m->geometryType = ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXT_TRIANGLES;
    m->geometryMask = 0xFF;
    m->triangleFormat = ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXT_TRIANGLE_INDICES_UINT32;
    m->triangleCount = (uint32_t)n_tris[g];
    m->triangleStride = 3 * sizeof(uint32_t);
    m->pTriangleBuffer = (void*)tris[g];
    m->vertexFormat = ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXT_FLOAT3;
    m->vertexCount = (uint32_t)n_verts[g];
    m->vertexStride = 3 * sizeof(float);
    m->pVertexBuffer = (void*)verts[g];
    geoms[g] = (const ze_rtas_builder_geometry_info_ext_t*)m;
  }
  ze_rtas_builder_procedural_geometry_info_ext_t procs = {ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXT_PROCEDURAL};
  if (n_procs > 0) {
    procs.geometryMask = 0xFF;
    procs.primCount = (uint32_t)n_procs;
    procs.pfnGetBoundsCb = xert_proc_bounds;
    procs.pGeomUserPtr = (void*)proc_bounds;
    geoms[n_geoms] = (const ze_rtas_builder_geometry_info_ext_t*)&procs;
  }
  void* rtas = xert_build(geoms, n_geoms + (n_procs > 0), out_size, bounds);
  free(geoms);
  free(meshes);
  return rtas;
}

void* xert_build_blas(const float* tri_verts, int64_t n_tris, int64_t* out_size) {
  if (n_tris <= 0 || n_tris > UINT32_MAX / 3)
    return NULL;
  uint32_t* tris = malloc(3 * n_tris * sizeof(uint32_t));
  for (int64_t i = 0; i < 3 * n_tris; i++)
    tris[i] = (uint32_t)i;
  int64_t n_verts = 3 * n_tris;
  const uint32_t* tri_list = tris;
  float bounds[6];
  void* rtas = xert_build_blas_geoms(1, &tri_verts, &n_verts, &tri_list, &n_tris, 0, NULL, out_size, bounds);
  free(tris);
  return rtas;
}

void* xert_build_tlas(int64_t n_inst, const float* xf, const int32_t* blas_of, const uint64_t* blas_addr,
                      const float* blas_bounds, int64_t* out_size) {
  if (!xert_init() || n_inst <= 0 || n_inst > UINT32_MAX)
    return NULL;
  ze_rtas_builder_instance_geometry_info_ext_t* inst = calloc(n_inst, sizeof(*inst));
  const ze_rtas_builder_geometry_info_ext_t** geoms = malloc(n_inst * sizeof(*geoms));
  int64_t n = 0;
  for (int64_t k = 0; k < n_inst; k++) {
    int32_t b = blas_of[k];
    if (b < 0)
      continue;
    ze_rtas_builder_instance_geometry_info_ext_t* g = &inst[n];
    g->geometryType = ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXT_INSTANCE;
    g->geometryMask = 0xFF;
    g->transformFormat = ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXT_FLOAT3X4_COLUMN_MAJOR;
    g->instanceUserID = (uint32_t)k;
    g->pTransform = (void*)(xf + 12 * k);
    g->pBounds = (ze_rtas_aabb_ext_t*)(blas_bounds + 6 * b);
    g->pAccelerationStructure = (void*)blas_addr[b];
    geoms[n++] = (const ze_rtas_builder_geometry_info_ext_t*)g;
  }
  float bounds[6];
  void* rtas = n > 0 ? xert_build(geoms, (uint32_t)n, out_size, bounds) : NULL;
  free(geoms);
  free(inst);
  return rtas;
}
