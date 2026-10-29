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

int64_t xert_alignment(void) { return xert_init() ? ze.alignment : 0; }

void xert_free(void* p) { free(p); }

static void* xert_build(const ze_rtas_builder_geometry_info_ext_t** geoms, uint32_t n_geoms, int64_t* out_size) {
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
  return rtas;
}

void* xert_build_blas(const float* tri_verts, int64_t n_tris, int64_t* out_size) {
  if (!xert_init() || n_tris <= 0 || n_tris > UINT32_MAX / 3)
    return NULL;
  ze_rtas_triangle_indices_uint32_ext_t* tris = malloc(n_tris * sizeof(*tris));
  for (int64_t i = 0; i < n_tris; i++)
    tris[i] = (ze_rtas_triangle_indices_uint32_ext_t){3 * i, 3 * i + 1, 3 * i + 2};
  ze_rtas_builder_triangles_geometry_info_ext_t mesh = {ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXT_TRIANGLES};
  mesh.geometryMask = 0xFF;
  mesh.triangleFormat = ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXT_TRIANGLE_INDICES_UINT32;
  mesh.triangleCount = (uint32_t)n_tris;
  mesh.triangleStride = sizeof(*tris);
  mesh.pTriangleBuffer = tris;
  mesh.vertexFormat = ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXT_FLOAT3;
  mesh.vertexCount = (uint32_t)(3 * n_tris);
  mesh.vertexStride = 3 * sizeof(float);
  mesh.pVertexBuffer = (void*)tri_verts;
  const ze_rtas_builder_geometry_info_ext_t* geoms[1] = {(const ze_rtas_builder_geometry_info_ext_t*)&mesh};
  void* rtas = xert_build(geoms, 1, out_size);
  free(tris);
  return rtas;
}
