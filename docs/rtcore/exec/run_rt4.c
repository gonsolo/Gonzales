// Multi-BLAS version of run_rt2: places the TLAS and every BLAS saved by as_probe in one CUDA allocation (BLASes at 64 KB
// aligned offsets, the TLAS stores their addresses shifted right by 16), relocates all stored addresses and traces.
// Usage: run_rt4 <cubin> <asprobe dir> <number of blas files>
// Build: gcc -O1 run_rt4.c -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -o /tmp/run_rt4
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorName(r_, &s); fprintf(stderr, "%s -> %s\n", #x, s); return 1; } } while (0)
static void* slurp(const char* path, size_t* n) { FILE* f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET); void* d = malloc(*n); fread(d, 1, *n, f); fclose(f); return d; }
int main(int argc, char** argv) {
    char p[512]; int nb = atoi(argv[3]); size_t rn;
    snprintf(p, sizeof p, "%s/rays.bin", argv[2]); void* rays = slurp(p, &rn); unsigned nrays = rn / 32;
    struct { void* d; size_t n; unsigned long long va, nva; } as[16];  // as[0] = TLAS, as[1..nb] = BLAS
    for (int i = 0; i <= nb; i++) { char nm[32]; if (i == 0) snprintf(nm, sizeof nm, "tlas"); else if (i == 1) snprintf(nm, sizeof nm, "blas"); else snprintf(nm, sizeof nm, "blas%d", i - 1);
        snprintf(p, sizeof p, "%s/%s.bin", argv[2], nm); as[i].d = slurp(p, &as[i].n);
        snprintf(p, sizeof p, "%s/%s.addr", argv[2], nm); FILE* f = fopen(p, "r"); long long sz; fscanf(f, "%llu %lld", &as[i].va, &sz); fclose(f); }
    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev, 0)); CUcontext ctx; CK(cuCtxCreate(&ctx, NULL, 0, dev));
    CUmodule mod; CK(cuModuleLoad(&mod, argv[1])); CUfunction fn; CK(cuModuleGetFunction(&fn, mod, "k"));
    size_t total = (size_t)(nb + 1) * (1 << 20) + (1 << 20); CUdeviceptr base; CK(cuMemAlloc(&base, total)); CK(cuMemsetD8(base, 0, total));
    for (int i = 0; i <= nb; i++) as[i].nva = base + (unsigned long long)i * (1 << 20);   // 1 MB apart, so 64 KB aligned
    int patched = 0;
    for (int i = 0; i <= nb; i++) { unsigned char* b = (unsigned char*)as[i].d;
        for (int j = 0; j <= nb; j++) {
            for (size_t o = 0; o + 8 <= as[i].n; o += 8) { unsigned long long v; memcpy(&v, b + o, 8); if (v == as[j].va) { memcpy(b + o, &as[j].nva, 8); patched++; } }
            unsigned int os = (unsigned int)(as[j].va >> 16), ns = (unsigned int)(as[j].nva >> 16);
            if (i == 0 && j > 0) for (size_t o = 0; o + 4 <= as[i].n; o++) { unsigned int v; memcpy(&v, b + o, 4); if (v == os) { memcpy(b + o, &ns, 4); patched++; } } } }
    printf("relocated %d references\n", patched);
    for (int i = 0; i <= nb; i++) CK(cuMemcpyHtoD(as[i].nva, as[i].d, as[i].n));
    CUdeviceptr dRays, dRes; size_t resn = (size_t)nrays * 32; CK(cuMemAlloc(&dRays, rn)); CK(cuMemcpyHtoD(dRays, rays, rn)); CK(cuMemAlloc(&dRes, resn)); CK(cuMemsetD8(dRes, 0xAB, resn));
    unsigned rc = nrays, rb = rn, sb = resn; unsigned long long a = as[0].nva, r = dRays, o = dRes, c10 = 0, c18 = 0;
    void* params[] = {&rc, &a, &r, &rb, &o, &sb, &c10, &c18};
    CK(cuLaunchKernel(fn, (nrays + 63) / 64, 1, 1, 64, 1, 1, 0, NULL, params, NULL));
    CUresult res = cuCtxSynchronize(); const char* s = "OK"; if (res) cuGetErrorName(res, &s); printf("launch result: %s\n", s); if (res) return 2;
    void* h = malloc(resn); CK(cuMemcpyDtoH(h, dRes, resn)); FILE* fo = fopen("/tmp/ncu/res.bin", "wb"); fwrite(h, 1, resn, fo); fclose(fo); return 0;
}
