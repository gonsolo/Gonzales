#include "rtcore.h"
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

namespace {
struct RtCore {
    CUcontext ctx = nullptr;
    CUmodule module = nullptr;
    CUfunction fn = nullptr;
    CUmodule convertModule = nullptr;
    CUfunction convertFn = nullptr;
    CUdeviceptr prefix = 0;
    int nMeshes = 0;
    CUdeviceptr scratch = 0;
    size_t scratchRays = 0;
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
    // The conversion kernel lives next to the trace cubin (build_rt_cubin.py writes both).
    std::string convertPath(cubinPath);
    size_t slash = convertPath.find_last_of('/');
    convertPath = (slash == std::string::npos ? std::string() : convertPath.substr(0, slash + 1)) + "rt_convert.cubin";
    if (cuModuleLoad(&rc->convertModule, convertPath.c_str()) == CUDA_SUCCESS)
        cuModuleGetFunction(&rc->convertFn, rc->convertModule, "convert");
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

extern "C" int rtcore_set_meshes(void* handle, const int32_t* prefix, int32_t nMeshes) {
    if (!handle || nMeshes <= 0) return 0;
    RtCore* rc = (RtCore*)handle;
    if (rc->prefix) cuMemFree(rc->prefix);
    if (!check(cuMemAlloc(&rc->prefix, (size_t)(nMeshes + 1) * 4), "cuMemAlloc prefix") ||
        !check(cuMemcpyHtoD(rc->prefix, prefix, (size_t)(nMeshes + 1) * 4), "cuMemcpyHtoD prefix")) return 0;
    rc->nMeshes = nMeshes;
    return 1;
}

extern "C" int rtcore_trace_interop(void* handle, uint64_t rays, uint64_t results, int32_t rayCount, void* stream) {
    if (!handle || rayCount <= 0) return 0;
    RtCore* rc = (RtCore*)handle;
    if (!rc->convertFn || !rc->prefix) { fprintf(stderr, "rtcore: rtcore_trace_interop needs rt_convert.cubin and rtcore_set_meshes\n"); return 0; }
    if (rc->scratchRays < (size_t)rayCount) {
        if (rc->scratch) cuMemFree(rc->scratch);
        if (!check(cuMemAlloc(&rc->scratch, (size_t)rayCount * 32), "cuMemAlloc scratch")) return 0;
        rc->scratchRays = (size_t)rayCount;
    }
    if (!rtcore_trace(handle, rays, (uint64_t)rc->scratch, rayCount, stream)) return 0;
    unsigned int n = (unsigned int)rayCount;
    unsigned long long raw = rc->scratch, out = results, pre = rc->prefix; int nm = rc->nMeshes;
    void* params[] = {&raw, &out, &pre, &nm, &n};
    return check(cuLaunchKernel(rc->convertFn, (n + 255u) / 256u, 1, 1, 256, 1, 1, 0, (CUstream)stream, params, nullptr), "cuLaunchKernel convert") ? 1 : 0;
}

static void* g_active = nullptr;
extern "C" void rtcore_set_active(void* handle) { g_active = handle; }
extern "C" void* rtcore_active(void) { return g_active; }

// Shadow rays on the RT cores need opaque geometry (the hardware trace ignores alpha cutouts): the caller decides, and
// passes the number of deferred-ray slots per path it wants used (0 = shadow rays stay on the software BVH).
static int g_shadow = 0;
extern "C" void rtcore_set_shadow(int slots) { g_shadow = slots; }
extern "C" int rtcore_shadow_enabled(void) { return g_active ? g_shadow : 0; }

extern "C" void rtcore_destroy(void* handle) {
    if (!handle) return;
    RtCore* rc = (RtCore*)handle;
    if (g_active == handle) g_active = nullptr;
    if (rc->prefix) cuMemFree(rc->prefix);
    if (rc->scratch) cuMemFree(rc->scratch);
    if (rc->convertModule) cuModuleUnload(rc->convertModule);
    if (rc->as) cuMemFree(rc->as);
    if (rc->module) cuModuleUnload(rc->module);
    delete rc;
}
