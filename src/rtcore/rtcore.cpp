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
    // Decode tables for rtcore_trace_interop. A "domain" is one TLAS instance (or the single merged BLAS); its pieces are
    // the mesh ranges inside that instance's triangle numbering.
    CUdeviceptr domBase = 0, domN = 0, pre = 0, rawv = 0, geom = 0;
    int nDomains = 0;
    int hasTlas = 0;
    CUdeviceptr scratch = 0;
    size_t scratchRays = 0;
    CUdeviceptr as = 0;      // the allocation
    CUdeviceptr root = 0;    // 64 KB aligned address of the root structure
    size_t asSize = 0;
};

bool check(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return true;
    const char* name = "?"; cuGetErrorName(r, &name);
    fprintf(stderr, "rtcore: %s failed: %s\n", what, name);
    return false;
}
} // namespace

namespace {
RtCore* openCore(const char* cubinPath) {
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
    return rc;
}
} // namespace

// Several acceleration structures: structure 0 is the root (a TLAS when there is more than one, otherwise the single
// BLAS). They are placed in one allocation, each at a 64 KB aligned offset (the TLAS stores BLAS addresses shifted right
// by 16), and every stored reference is relocated: 8-byte words equal to another structure's Vulkan address, and, inside
// the TLAS, the 32-bit `address >> 16` fields (at unaligned offsets).
extern "C" void* rtcore_create_scene(const char* cubinPath, int32_t nAs, const uint8_t* const* asBytes, const int64_t* asSizes,
                                     const uint64_t* asVkAddresses) {
    if (nAs <= 0) return nullptr;
    RtCore* rc = openCore(cubinPath);
    if (!rc) return nullptr;
    std::vector<size_t> offset((size_t)nAs);
    size_t total = 0;
    for (int i = 0; i < nAs; i++) { offset[(size_t)i] = total; total += ((size_t)asSizes[i] + 0xffffu) & ~(size_t)0xffffu; }
    rc->asSize = total + 4096;
    if (!check(cuMemAlloc(&rc->as, rc->asSize + 0x10000), "cuMemAlloc")) { delete rc; return nullptr; }
    CUdeviceptr base = (rc->as + 0xffffu) & ~(CUdeviceptr)0xffffu;     // cuMemAlloc is 256-byte aligned only
    rc->hasTlas = nAs > 1;
    int patched = 0;
    std::vector<std::vector<uint8_t>> bytes((size_t)nAs);
    for (int i = 0; i < nAs; i++) bytes[(size_t)i].assign(asBytes[i], asBytes[i] + asSizes[i]);
    for (int i = 0; i < nAs; i++) {
        std::vector<uint8_t>& b = bytes[(size_t)i];
        for (int j = 0; j < nAs; j++) {
            uint64_t nva = base + offset[(size_t)j];
            for (size_t o = 0; o + 8 <= b.size(); o += 8) {
                uint64_t v; memcpy(&v, &b[o], 8);
                if (v == asVkAddresses[j]) { memcpy(&b[o], &nva, 8); patched++; }
            }
            if (i == 0 && j > 0) {
                uint32_t os = (uint32_t)(asVkAddresses[j] >> 16), ns = (uint32_t)(nva >> 16);
                for (size_t o = 0; o + 4 <= b.size(); o++) {
                    uint32_t v; memcpy(&v, &b[o], 4);
                    if (v == os) { memcpy(&b[o], &ns, 4); patched++; }
                }
            }
        }
    }
    if (patched == 0) fprintf(stderr, "rtcore: warning: no self address found in the acceleration structure\n");
    for (int i = 0; i < nAs; i++)
        if (!check(cuMemcpyHtoD(base + offset[(size_t)i], bytes[(size_t)i].data(), bytes[(size_t)i].size()), "cuMemcpyHtoD")) { delete rc; return nullptr; }
    rc->root = base;
    return rc;
}

extern "C" void* rtcore_create(const char* cubinPath, const uint8_t* asBytes, int64_t asSize, uint64_t asVkAddress) {
    return rtcore_create_scene(cubinPath, 1, &asBytes, &asSize, &asVkAddress);
}

extern "C" int rtcore_trace(void* handle, uint64_t rays, uint64_t results, int32_t rayCount, void* stream) {
    if (!handle || rayCount <= 0) return 0;
    RtCore* rc = (RtCore*)handle;
    // Kernel parameters, the layout docs/rtcore/exec/make_cubin.py gives the skeleton kernel.
    unsigned int n = (unsigned int)rayCount, raysBytes = n * 32u, resultsBytes = n * 32u;
    unsigned long long as = rc->root, r = rays, o = results, c10 = 0, c18 = 0;
    void* params[] = {&n, &as, &r, &raysBytes, &o, &resultsBytes, &c10, &c18};
    unsigned int blocks = (n + 63u) / 64u;
    return check(cuLaunchKernel(rc->fn, blocks, 1, 1, 64, 1, 1, 0, (CUstream)stream, params, nullptr), "cuLaunchKernel") ? 1 : 0;
}

namespace {
bool upload(CUdeviceptr* dst, const int32_t* src, size_t n, const char* what) {
    if (*dst) cuMemFree(*dst);
    *dst = 0;
    return check(cuMemAlloc(dst, n * 4 + 4), what) && check(cuMemcpyHtoD(*dst, src, n * 4), what);
}
} // namespace

// Decode tables. Domain d (the d-th TLAS instance; domain 0 for a single merged BLAS) has domN[d] pieces (domN[d] = -1: a
// template instance whose BLAS has one geometry per mesh -- the hardware reports the geometry, rawv[domBase[d]] is the raw mesh
// index to report); its entries start
// at domBase[d] in `pre` (domN[d] + 1 local triangle starts, ascending) and in `rawv` / `geom` (domN[d] entries, then one
// unused): a hit on triangle `tri` of domain d lies in the last piece j with pre[domBase[d] + j] <= tri and is reported as
// raw mesh rawv[domBase[d] + j], local triangle tri - pre[...], geometry geom[...].
extern "C" int rtcore_set_domains(void* handle, int32_t nDomains, const int32_t* domBase, const int32_t* domN, const int32_t* pre,
                                  const int32_t* rawv, const int32_t* geom, int32_t nEntries) {
    if (!handle || nDomains <= 0 || nEntries <= 0) return 0;
    RtCore* rc = (RtCore*)handle;
    if (!upload(&rc->domBase, domBase, (size_t)nDomains, "domBase") || !upload(&rc->domN, domN, (size_t)nDomains, "domN") ||
        !upload(&rc->pre, pre, (size_t)nEntries, "pre") || !upload(&rc->rawv, rawv, (size_t)nEntries, "rawv") ||
        !upload(&rc->geom, geom, (size_t)nEntries, "geom")) return 0;
    rc->nDomains = nDomains;
    return 1;
}

// One merged BLAS: a single domain whose pieces are the meshes (raw mesh j, geometry 0).
extern "C" int rtcore_set_meshes(void* handle, const int32_t* prefix, int32_t nMeshes) {
    if (nMeshes <= 0) return 0;
    std::vector<int32_t> raw((size_t)nMeshes + 1), geom((size_t)nMeshes + 1, 0);
    for (int j = 0; j <= nMeshes; j++) raw[(size_t)j] = j;
    int32_t base = 0;
    return rtcore_set_domains(handle, 1, &base, &nMeshes, prefix, raw.data(), geom.data(), nMeshes + 1);
}

extern "C" int rtcore_trace_interop(void* handle, uint64_t rays, uint64_t results, int32_t rayCount, void* stream) {
    if (!handle || rayCount <= 0) return 0;
    RtCore* rc = (RtCore*)handle;
    if (!rc->convertFn || !rc->nDomains) { fprintf(stderr, "rtcore: rtcore_trace_interop needs rt_convert.cubin and rtcore_set_meshes / rtcore_set_domains\n"); return 0; }
    if (rc->scratchRays < (size_t)rayCount) {
        if (rc->scratch) cuMemFree(rc->scratch);
        if (!check(cuMemAlloc(&rc->scratch, (size_t)rayCount * 32), "cuMemAlloc scratch")) return 0;
        rc->scratchRays = (size_t)rayCount;
    }
    if (!rtcore_trace(handle, rays, (uint64_t)rc->scratch, rayCount, stream)) return 0;
    unsigned int n = (unsigned int)rayCount;
    unsigned long long raw = rc->scratch, out = results, db = rc->domBase, dn = rc->domN, pre = rc->pre, rv = rc->rawv, gm = rc->geom;
    int nd = rc->nDomains, tl = rc->hasTlas;
    void* params[] = {&raw, &out, &db, &dn, &pre, &rv, &gm, &nd, &tl, &n};
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
    for (CUdeviceptr p : {rc->domBase, rc->domN, rc->pre, rc->rawv, rc->geom}) if (p) cuMemFree(p);
    if (rc->scratch) cuMemFree(rc->scratch);
    if (rc->convertModule) cuModuleUnload(rc->convertModule);
    if (rc->as) cuMemFree(rc->as);
    if (rc->module) cuModuleUnload(rc->module);
    delete rc;
}
