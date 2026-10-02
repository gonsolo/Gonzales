// Loads a cubin with kernel `k` and launches it with 4 u64 parameters + int; reports the CUDA error.
// Build: gcc -O1 run_cubin.c -I/opt/cuda/include -L/opt/cuda/lib64 -lcuda -o /tmp/run_cubin
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorName(r_, &s); fprintf(stderr, "%s -> %s\n", #x, s); return 1; } } while (0)
int main(int argc, char** argv) {
    CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev, 0)); CUcontext ctx; CK(cuCtxCreate(&ctx, NULL, 0, dev));
    CUmodule mod; CK(cuModuleLoad(&mod, argv[1])); CUfunction fn; CK(cuModuleGetFunction(&fn, mod, "k"));
    int regs; cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn); printf("regs=%d\n", regs);
    CUdeviceptr buf; CK(cuMemAlloc(&buf, 1 << 20)); cuMemsetD8(buf, 0, 1 << 20);
    unsigned long long a = buf, b = 0, c = 0, e = 0; int n = 1;
    void* params[] = {&a, &b, &c, &e, &n};
    CK(cuLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, NULL, params, NULL));
    CUresult r = cuCtxSynchronize(); const char* s = "OK"; if (r) cuGetErrorName(r, &s);
    printf("launch result: %s\n", s); return 0;
}
