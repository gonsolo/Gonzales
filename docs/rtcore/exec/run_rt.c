// Launches the patched driver ray-query code as a CUDA kernel against a Vulkan-built acceleration structure
// whose bytes were saved by as_probe. Usage: run_rt <cubin> <asprobe dir> <as file: blas.bin|tlas.bin>
// Build: gcc -O1 run_rt.c -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -o /tmp/run_rt
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorName(r_, &s); fprintf(stderr, "%s -> %s\n", #x, s); return 1; } } while (0)
static void* slurp(const char* path, size_t* n) { FILE* f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET); void* d = malloc(*n); fread(d, 1, *n, f); fclose(f); return d; }
int main(int argc, char** argv) {
    char p[512]; size_t asn, raysn;
    snprintf(p, sizeof p, "%s/%s", argv[2], argv[3]); void* as = slurp(p, &asn);
    snprintf(p, sizeof p, "%s/rays.bin", argv[2]); void* rays = slurp(p, &raysn);
    unsigned nrays = raysn / 32;
    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev, 0)); CUcontext ctx; CK(cuCtxCreate(&ctx, NULL, 0, dev));
    CUmodule mod; CK(cuModuleLoad(&mod, argv[1])); CUfunction fn; CK(cuModuleGetFunction(&fn, mod, "k"));
    CUdeviceptr dAs, dRays, dRes; size_t resn = (size_t)nrays * 32;
    CK(cuMemAlloc(&dAs, asn + 4096)); CK(cuMemcpyHtoD(dAs, as, asn));
    CK(cuMemAlloc(&dRays, raysn)); CK(cuMemcpyHtoD(dRays, rays, raysn));
    CK(cuMemAlloc(&dRes, resn)); CK(cuMemsetD8(dRes, 0xAB, resn));
    unsigned rc = nrays, rb = raysn, sb = resn; unsigned long long a = dAs, r = dRays, o = dRes, c10 = 0, c18 = 0;
    void* params[] = {&rc, &a, &r, &rb, &o, &sb, &c10, &c18};
    printf("AS at 0x%llx (%zu bytes), %u rays\n", a, asn, nrays);
    CK(cuLaunchKernel(fn, (nrays + 63) / 64, 1, 1, 64, 1, 1, 0, NULL, params, NULL));
    CUresult res = cuCtxSynchronize(); const char* s = "OK"; if (res) cuGetErrorName(res, &s);
    printf("launch result: %s\n", s); if (res) return 2;
    float* h = malloc(resn); CK(cuMemcpyDtoH(h, dRes, resn));
    int hits = 0; for (unsigned i = 0; i < nrays; i++) { unsigned flag = ((unsigned*)h)[i * 8 + 6]; hits += (flag == 1); }
    printf("hits=%d/%u\n", hits, nrays);
    for (unsigned i = 0; i < 8; i++) { float* q = h + i * 8; printf("ray %u: t=%g u=%g v=%g mesh=%d tri=%d flag=%u\n", i, q[0], q[1], q[2], ((int*)q)[4], ((int*)q)[5], ((unsigned*)q)[6]); }
    return 0;
}
