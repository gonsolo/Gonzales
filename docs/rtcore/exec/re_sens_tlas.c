// Sensitivity map of a TLAS (or any of its BLASes) with all structures relocated into one CUDA buffer.
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
// returns 0 ok, 1 fault. blobs[0] is the root (TLAS); all blobs are placed 128 KB aligned and every stored address
// (full 64-bit words, and address >> 16 as an unaligned 32-bit field) is rewritten to the new place.
static int runN(const char* cubin, unsigned char** blobs, size_t* sizes, unsigned long long* olds, int nb, void* rays, size_t rn, unsigned char* res) {
    CUdevice dev; CUcontext ctx;
    if (cuDeviceGet(&dev, 0) || cuCtxCreate(&ctx, NULL, 0, dev)) return 1;
    CUmodule mod; CUfunction fn;
    if (cuModuleLoad(&mod, cubin) || cuModuleGetFunction(&fn, mod, "k")) { cuCtxDestroy(ctx); return 1; }
    size_t total = (1 << 17); for (int i = 0; i < nb; i++) total += ((sizes[i] + 0x1ffff) & ~0x1ffffULL) + (1 << 17);
    CUdeviceptr base; cuMemAlloc(&base, total); cuMemsetD8(base, 0, total);
    unsigned long long news[64]; unsigned long long cur = (base + 0x1ffff) & ~0x1ffffULL;
    for (int i = 0; i < nb; i++) { news[i] = cur; cur += ((sizes[i] + 0x1ffff) & ~0x1ffffULL); }
    for (int i = 0; i < nb; i++) {
        unsigned char* b = malloc(sizes[i]); memcpy(b, blobs[i], sizes[i]);
        for (int j = 0; j < nb; j++) {
            for (size_t o = 0; o + 8 <= sizes[i]; o += 8) { unsigned long long v; memcpy(&v, b + o, 8); if (v == olds[j]) memcpy(b + o, &news[j], 8); }
            unsigned o32 = (unsigned)(olds[j] >> 16), n32 = (unsigned)(news[j] >> 16);
            for (size_t o = 0; o + 4 <= sizes[i]; o++) { unsigned v; memcpy(&v, b + o, 4); if (v == o32 && o % 8 != 0) memcpy(b + o, &n32, 4); }
        }
        cuMemcpyHtoD(news[i], b, sizes[i]); free(b);
    }
    unsigned nrays = rn / 32; CUdeviceptr dR, dO; size_t resn = (size_t)nrays * 32;
    cuMemAlloc(&dR, rn); cuMemcpyHtoD(dR, rays, rn); cuMemAlloc(&dO, resn); cuMemsetD8(dO, 0xAB, resn);
    unsigned rc = nrays, rb = rn, sb = resn; unsigned long long a = news[0], r = dR, o = dO, c10 = 0, c18 = 0;
    void* params[] = {&rc, &a, &r, &rb, &o, &sb, &c10, &c18};
    cuLaunchKernel(fn, (nrays + 63) / 64, 1, 1, 64, 1, 1, 0, NULL, params, NULL);
    CUresult rs = cuCtxSynchronize();
    if (!rs) cuMemcpyDtoH(res, dO, resn);
    cuCtxDestroy(ctx);
    return rs ? 1 : 0;
}
int main(int argc, char** argv) {
    // re_sens_tlas <cubin> <rays.bin> <lo> <hi> <which blob index to zero in> <tlas.bin> <blas...>
    size_t rn; void* rays = slurp(argv[2], &rn);
    size_t lo = strtoul(argv[3], 0, 0), hi = strtoul(argv[4], 0, 0); int which = atoi(argv[5]);
    int nb = argc - 6; unsigned char* blobs[64]; size_t sizes[64]; unsigned long long olds[64];
    for (int i = 0; i < nb; i++) { blobs[i] = slurp(argv[6 + i], &sizes[i]); char p[700]; snprintf(p, sizeof p, "%s.addr", argv[6 + i]); long long sz; FILE* f = fopen(p, "r"); fscanf(f, "%llu %lld", &olds[i], &sz); fclose(f); }
    cuInit(0);
    unsigned char* base = malloc(rn), *cur = malloc(rn);
    if (runN(argv[1], blobs, sizes, olds, nb, rays, rn, base)) { printf("baseline faulted\n"); return 1; }
    unsigned nrays = rn / 32; int hits = 0; for (unsigned i = 0; i < nrays; i++) hits += (((unsigned*)base)[i * 8 + 3] != 0xffffffffu);
    printf("baseline: %u rays, %d hits\n", nrays, hits);
    unsigned char* b = blobs[which]; size_t bn = sizes[which];
    for (size_t o = lo; o + 8 <= hi && o + 8 <= bn; o += 8) {
        unsigned char save[8]; memcpy(save, b + o, 8); memset(b + o, 0, 8);
        int fault = runN(argv[1], blobs, sizes, olds, nb, rays, rn, cur);
        memcpy(b + o, save, 8);
        if (fault) { printf("%04zx fault\n", o); continue; }
        int diff = 0; for (unsigned i = 0; i < nrays; i++) if (memcmp(base + 32 * i, cur + 32 * i, 32)) diff++;
        if (diff) { int h = 0; for (unsigned i = 0; i < nrays; i++) h += (((unsigned*)cur)[i * 8 + 3] != 0xffffffffu); printf("%04zx changed %d rays (hits now %d)\n", o, diff, h); }
    }
    return 0;
}
