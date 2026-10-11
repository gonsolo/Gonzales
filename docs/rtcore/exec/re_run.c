// Runs the RT-core trace on a (patched) BLAS and writes the raw results. Usage: re_run <cubin> <blas.bin> <rays.bin> <out.bin> [hexoffset=hexbytes ...]
// (based on re_sens.c)
// Sensitivity map of an acceleration structure: which 8-byte windows of a BLAS change what the RT-core trace returns.
// Usage: re_sens <cubin> <blas.bin> <rays.bin> [start end]   (blas.bin.addr holds Vulkan's address and size)
// Each probe runs in a fresh CUDA context (a fault kills the context). Prints the windows whose zeroing changes results.
// Build: gcc -O1 re_sens.c -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -o /tmp/re_sens
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void* slurp(const char* path, size_t* n) { FILE* f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET); void* d = malloc(*n); fread(d, 1, *n, f); fclose(f); return d; }
// returns 0 ok, 1 fault; fills res (nrays*32 bytes)
static int run(const char* cubin, unsigned char* blob, size_t bn, unsigned long long oldA, void* rays, size_t rn, unsigned char* res) {
    CUdevice dev; CUcontext ctx;
    if (cuDeviceGet(&dev, 0) || cuCtxCreate(&ctx, NULL, 0, dev)) return 1;
    CUmodule mod; CUfunction fn;
    if (cuModuleLoad(&mod, cubin) || cuModuleGetFunction(&fn, mod, "k")) { cuCtxDestroy(ctx); return 1; }
    size_t total = bn + (1 << 17); CUdeviceptr base; cuMemAlloc(&base, total); cuMemsetD8(base, 0, total);
    unsigned long long newA = (base + 0x1ffff) & ~0x1ffffULL;      // 128 KB aligned like Vulkan's
    unsigned char* b = malloc(bn); memcpy(b, blob, bn);
    for (size_t o = 0; o + 8 <= bn; o += 8) { unsigned long long v; memcpy(&v, b + o, 8); if (v == oldA) memcpy(b + o, &newA, 8); }
    cuMemcpyHtoD(newA, b, bn); free(b);
    unsigned nrays = rn / 32; CUdeviceptr dR, dO; size_t resn = (size_t)nrays * 32;
    cuMemAlloc(&dR, rn); cuMemcpyHtoD(dR, rays, rn); cuMemAlloc(&dO, resn); cuMemsetD8(dO, 0xAB, resn);
    unsigned rc = nrays, rb = rn, sb = resn; unsigned long long a = newA, r = dR, o = dO, c10 = 0, c18 = 0;
    void* params[] = {&rc, &a, &r, &rb, &o, &sb, &c10, &c18};
    cuLaunchKernel(fn, (nrays + 63) / 64, 1, 1, 64, 1, 1, 0, NULL, params, NULL);
    CUresult rs = cuCtxSynchronize();
    if (!rs) cuMemcpyDtoH(res, dO, resn);
    cuCtxDestroy(ctx);
    return rs ? 1 : 0;
}
int main(int argc, char** argv) {
    size_t bn, rn; unsigned char* blob = slurp(argv[2], &bn); void* rays = slurp(argv[3], &rn);
    char p[600]; snprintf(p, sizeof p, "%s.addr", argv[2]); unsigned long long oldA; long long sz; FILE* f = fopen(p, "r"); fscanf(f, "%llu %lld", &oldA, &sz); fclose(f);
    for (int a = 5; a < argc; a++) { char* eq = strchr(argv[a], '='); if (!eq) continue; size_t off = strtoul(argv[a], 0, 16); const char* h = eq + 1;
        for (size_t i = 0; h[i] && h[i + 1]; i += 2) { char t[3] = {h[i], h[i + 1], 0}; blob[off + i / 2] = (unsigned char)strtoul(t, 0, 16); } }
    cuInit(0);
    unsigned char* res = malloc(rn);
    int fault = run(argv[1], blob, bn, oldA, rays, rn, res);
    if (fault) { printf("fault\n"); return 2; }
    f = fopen(argv[4], "wb"); fwrite(res, 1, rn, f); fclose(f); printf("ok\n"); return 0;
}
