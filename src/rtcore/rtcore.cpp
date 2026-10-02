#include "rtcore.h"
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
struct RtCore {
    CUcontext ctx = nullptr;
    CUmodule module = nullptr;
    CUfunction fn = nullptr;
    CUdeviceptr as = 0;
    size_t asSize = 0;
};

bool check(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return true;
    const char* name = "?"; cuGetErrorName(r, &name);
    fprintf(stderr, "rtcore: %s failed: %s\n", what, name);
    return false;
}
} // namespace

extern "C" void* rtcore_create(const char* cubinPath, const uint8_t* asBytes, int64_t asSize, uint64_t asVkAddress) {
    if (!check(cuInit(0), "cuInit")) return nullptr;
    RtCore* rc = new RtCore();
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    if (!ctx) {
        CUdevice dev;
        if (!check(cuDeviceGet(&dev, 0), "cuDeviceGet") || !check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain") ||
            !check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")) { delete rc; return nullptr; }
    }
    rc->ctx = ctx;
    if (!check(cuModuleLoad(&rc->module, cubinPath), "cuModuleLoad") ||
        !check(cuModuleGetFunction(&rc->fn, rc->module, "k"), "cuModuleGetFunction")) { delete rc; return nullptr; }
    rc->asSize = (size_t)asSize;
    if (!check(cuMemAlloc(&rc->as, rc->asSize + 4096), "cuMemAlloc")) { delete rc; return nullptr; }
    std::vector<uint8_t> bytes(asBytes, asBytes + asSize);
    int patched = 0;
    for (size_t o = 0; o + 8 <= bytes.size(); o += 8) {
        uint64_t v; memcpy(&v, &bytes[o], 8);
        if (v == asVkAddress) { uint64_t n = rc->as; memcpy(&bytes[o], &n, 8); patched++; }
    }
    if (patched == 0) fprintf(stderr, "rtcore: warning: no self address found in the acceleration structure\n");
    if (!check(cuMemcpyHtoD(rc->as, bytes.data(), bytes.size()), "cuMemcpyHtoD")) { delete rc; return nullptr; }
    return rc;
}

extern "C" int rtcore_trace(void* handle, uint64_t rays, uint64_t results, int32_t rayCount, void* stream) {
    if (!handle || rayCount <= 0) return 0;
    RtCore* rc = (RtCore*)handle;
    // Kernel parameters, the layout docs/rtcore/exec/make_cubin.py gives the skeleton kernel.
    unsigned int n = (unsigned int)rayCount, raysBytes = n * 32u, resultsBytes = n * 32u;
    unsigned long long as = rc->as, r = rays, o = results, c10 = 0, c18 = 0;
    void* params[] = {&n, &as, &r, &raysBytes, &o, &resultsBytes, &c10, &c18};
    unsigned int blocks = (n + 63u) / 64u;
    return check(cuLaunchKernel(rc->fn, blocks, 1, 1, 64, 1, 1, 0, (CUstream)stream, params, nullptr), "cuLaunchKernel") ? 1 : 0;
}

extern "C" void rtcore_destroy(void* handle) {
    if (!handle) return;
    RtCore* rc = (RtCore*)handle;
    if (rc->as) cuMemFree(rc->as);
    if (rc->module) cuModuleUnload(rc->module);
    delete rc;
}
