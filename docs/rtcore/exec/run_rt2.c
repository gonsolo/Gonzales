// Like run_rt, but maps the acceleration structures at the SAME virtual addresses Vulkan gave them (CUDA virtual
// memory management, fixed address), because the driver stores absolute addresses inside the structures.
// Usage: run_rt2 <cubin> <asprobe dir> <root: tlas|blas>
// Build: gcc -O1 run_rt2.c -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -o /tmp/run_rt2
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorName(r_, &s); fprintf(stderr, "%s -> %s\n", #x, s); return 1; } } while (0)
static void* slurp(const char* path, size_t* n) { FILE* f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET); void* d = malloc(*n); fread(d, 1, *n, f); fclose(f); return d; }
int main(int argc, char** argv) {
    char p[512]; size_t rn, tn, bn;
    snprintf(p, sizeof p, "%s/rays.bin", argv[2]); void* rays = slurp(p, &rn);
    snprintf(p, sizeof p, "%s/tlas.bin", argv[2]); void* tlas = slurp(p, &tn);
    snprintf(p, sizeof p, "%s/blas.bin", argv[2]); void* blas = slurp(p, &bn);
    unsigned long long tva, bva; long long x;
    snprintf(p, sizeof p, "%s/tlas.addr", argv[2]); FILE* f = fopen(p, "r"); fscanf(f, "%llu %lld", &tva, &x); fclose(f);
    snprintf(p, sizeof p, "%s/blas.addr", argv[2]); f = fopen(p, "r"); fscanf(f, "%llu %lld", &bva, &x); fclose(f);
    unsigned nrays = rn / 32;
    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev, 0)); CUcontext ctx; CK(cuCtxCreate(&ctx, NULL, 0, dev));
    CUmodule mod; CK(cuModuleLoad(&mod, argv[1])); CUfunction fn; CK(cuModuleGetFunction(&fn, mod, "k"));
    // CUDA cannot be asked for Vulkan's addresses, so relocate: place both structures in one CUDA buffer and patch the
    // absolute addresses the driver stored inside them (each structure's own address at +0xd0, and the TLAS's
    // reference to the BLAS at +0x200). Any other absolute pointer would show up as a fault or a miss.
    CUdeviceptr base; CK(cuMemAlloc(&base, 1 << 20)); CK(cuMemsetD8(base, 0, 1 << 20));
    unsigned long long newT = base, newB = base + (getenv("BLAS_OFF") ? strtoull(getenv("BLAS_OFF"), 0, 0) : 8192);
    unsigned long long oldT = tva, oldB = bva;
    unsigned char* tb = (unsigned char*)tlas; unsigned char* bb = (unsigned char*)blas; int patched = 0;
    for (size_t o = 0; o + 8 <= tn; o += 8) { unsigned long long v = *(unsigned long long*)(tb + o);
        if (v == oldT) { *(unsigned long long*)(tb + o) = newT; patched++; } else if (v == oldB) { *(unsigned long long*)(tb + o) = newB; patched++; } }
    for (size_t o = 0; o + 8 <= bn; o += 8) { unsigned long long v = *(unsigned long long*)(bb + o);
        if (v == oldB) { *(unsigned long long*)(bb + o) = newB; patched++; } else if (v == oldT) { *(unsigned long long*)(bb + o) = newT; patched++; } }
    printf("relocated %d pointers; TLAS at 0x%llx BLAS at 0x%llx\n", patched, newT, newB);
    CK(cuMemcpyHtoD(newT, tlas, tn)); CK(cuMemcpyHtoD(newB, blas, bn)); tva = newT; bva = newB;
    CUdeviceptr dRays, dRes; size_t resn = (size_t)nrays * 32;
    CK(cuMemAlloc(&dRays, rn)); CK(cuMemcpyHtoD(dRays, rays, rn)); CK(cuMemAlloc(&dRes, resn)); CK(cuMemsetD8(dRes, 0xAB, resn));
    unsigned rc = nrays, rb = rn, sb = resn; unsigned long long a = (argv[3][0] == 'b') ? bva : tva, r = dRays, o = dRes, c10 = 0, c18 = 0;
    void* params[] = {&rc, &a, &r, &rb, &o, &sb, &c10, &c18};
    printf("root AS at 0x%llx, %u rays\n", a, nrays);
    CK(cuLaunchKernel(fn, (nrays + 63) / 64, 1, 1, 64, 1, 1, 0, NULL, params, NULL));
    CUresult res = cuCtxSynchronize(); const char* s = "OK"; if (res) cuGetErrorName(res, &s);
    printf("launch result: %s\n", s); if (res) return 2;
    float* hres = malloc(resn); CK(cuMemcpyDtoH(hres, dRes, resn));
    int hits = 0; for (unsigned i = 0; i < nrays; i++) hits += (((unsigned*)hres)[i * 8 + 6] == 1);
    printf("hits=%d/%u\n", hits, nrays);
    for (unsigned i = 0; i < 8; i++) { float* q = hres + i * 8; printf("ray %u: t=%g u=%g v=%g mesh=%d tri=%d flag=%u\n", i, q[0], q[1], q[2], ((int*)q)[4], ((int*)q)[5], ((unsigned*)q)[6]); }
    return 0;
}
