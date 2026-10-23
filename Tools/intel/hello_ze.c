#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#define CK(x) do { ze_result_t r = (x); if (r) { fprintf(stderr, "%s: 0x%x\n", #x, r); exit(1); } } while (0)

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s kernel.spv\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); size_t len = ftell(f); rewind(f);
    unsigned char *spv = malloc(len); if (fread(spv, 1, len, f) != len) return 2; fclose(f);
    CK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t nd = 1; ze_driver_handle_t drv; CK(zeDriverGet(&nd, &drv));
    uint32_t ndev = 1; ze_device_handle_t dev; CK(zeDeviceGet(drv, &ndev, &dev));
    if (!ndev) { fprintf(stderr, "no Level Zero GPU\n"); return 1; }
    ze_device_properties_t dp = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES}; CK(zeDeviceGetProperties(dev, &dp));
    printf("device: %s, %u EUs\n", dp.name, dp.numSlices * dp.numSubslicesPerSlice * dp.numEUsPerSubslice);
    ze_context_desc_t cd = {ZE_STRUCTURE_TYPE_CONTEXT_DESC}; ze_context_handle_t ctx; CK(zeContextCreate(drv, &cd, &ctx));
    ze_module_desc_t md = {ZE_STRUCTURE_TYPE_MODULE_DESC, 0, ZE_MODULE_FORMAT_IL_SPIRV, len, spv, "", 0};
    ze_module_handle_t mod; ze_module_build_log_handle_t log;
    ze_result_t r = zeModuleCreate(ctx, dev, &md, &mod, &log);
    if (r) { size_t n = 0; zeModuleBuildLogGetString(log, &n, 0); char *s = malloc(n); zeModuleBuildLogGetString(log, &n, s); fprintf(stderr, "build 0x%x: %s\n", r, s); return 1; }
    uint32_t nk = 1; const char *name; CK(zeModuleGetKernelNames(mod, &nk, &name));
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, 0, 0, name}; ze_kernel_handle_t k; CK(zeKernelCreate(mod, &kd, &k));
    int64_t n = 1000; float *p;
    ze_device_mem_alloc_desc_t dd = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC}; ze_host_mem_alloc_desc_t hd = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    CK(zeMemAllocShared(ctx, &dd, &hd, n * sizeof(float), 64, dev, (void **)&p));
    uint32_t gx = 64, gy = 1, gz = 1; CK(zeKernelSetGroupSize(k, gx, gy, gz));
    CK(zeKernelSetArgumentValue(k, 0, sizeof(p), &p)); CK(zeKernelSetArgumentValue(k, 1, sizeof(n), &n));
    ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, 0, 0, 0, 0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS};
    ze_command_list_handle_t cl; CK(zeCommandListCreateImmediate(ctx, dev, &qd, &cl));
    ze_group_count_t gc = {(uint32_t)((n + gx - 1) / gx), 1, 1};
    CK(zeCommandListAppendLaunchKernel(cl, k, &gc, 0, 0, 0)); CK(zeCommandListHostSynchronize(cl, UINT64_MAX));
    int bad = 0; for (int64_t i = 0; i < n; i++) bad += p[i] != 2.0f * i;
    printf("p[999] = %g, %d wrong\n", p[999], bad);
    return bad != 0;
}
