// Builds a small triangle scene with the Vulkan ray-query backend, writes the driver's BLAS/TLAS bytes and a
// set of test rays with Vulkan's own hit results to files, so a CUDA kernel can be compared against them.
// Build: gcc -O1 as_probe.c -I../../../src/vulkanrt -L../../../build -lvulkanrt -lm -Wl,-rpath,/home/gonsolo/work/gonzales/build -o /tmp/as_probe
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "vulkanrt.h"
int main(int argc, char** argv) {
    const char* out = argc > 1 ? argv[1] : "/tmp/ncu/asprobe"; char cmd[300]; snprintf(cmd, sizeof cmd, "mkdir -p %s", out); system(cmd);
    int g = argc > 2 ? atoi(argv[2]) : 4, nrays = 256;
    int nv = (g + 1) * (g + 1), nt = g * g * 2;
    float* pts = malloc(16 * nv); int64_t* idx = malloc(8 * 3 * nt);
    for (int y = 0; y <= g; y++) for (int x = 0; x <= g; x++) { float* p = pts + 4 * (y * (g + 1) + x);
        p[0] = x; p[1] = y; p[2] = 0.3f * sinf(x * 0.9f) * cosf(y * 0.7f); p[3] = 1; }
    int k = 0;
    for (int y = 0; y < g; y++) for (int x = 0; x < g; x++) { int a = y * (g + 1) + x, b = a + 1, c = a + g + 1, d = c + 1;
        idx[k++] = a; idx[k++] = b; idx[k++] = c; idx[k++] = b; idx[k++] = d; idx[k++] = c; }
    VulkanRtMesh m = {0}; m.points = pts; m.vertexIndices = idx; int64_t pc = nv, ic = 3 * nt;
    void* scene = vulkanrt_build_scene(&m, 1, &pc, &ic);
    if (!scene) { fprintf(stderr, "build failed\n"); return 1; }
    for (int kind = 0; kind < 2; kind++) {
        uint64_t addr = 0; int64_t sz = vulkanrt_debug_read_as(scene, kind, 0, NULL, 0, &addr);
        uint8_t* buf = malloc(sz); vulkanrt_debug_read_as(scene, kind, 0, buf, sz, &addr);
        char fn[400]; snprintf(fn, sizeof fn, "%s/%s.bin", out, kind ? "tlas" : "blas"); FILE* f = fopen(fn, "wb"); fwrite(buf, 1, sz, f); fclose(f);
        { char fa[400]; snprintf(fa, sizeof fa, "%s/%s.addr", out, kind ? "tlas" : "blas"); FILE* fx = fopen(fa, "w"); fprintf(fx, "%llu %lld\n", (unsigned long long)addr, (long long)sz); fclose(fx); }
        printf("%s: %lld bytes, vk address 0x%llx\n", kind ? "TLAS" : "BLAS", (long long)sz, (unsigned long long)addr);
    }
    float* rays = malloc(32 * nrays); srand(7);
    for (int i = 0; i < nrays; i++) { float* r = rays + 8 * i;
        r[0] = g * (rand() / (float)RAND_MAX); r[1] = g * (rand() / (float)RAND_MAX); r[2] = 5; r[3] = 0;
        r[4] = 0.1f * (rand() / (float)RAND_MAX - .5f); r[5] = 0.1f * (rand() / (float)RAND_MAX - .5f); r[6] = -1; r[7] = 100; }
    float* t = malloc(4 * nrays); float* u = malloc(4 * nrays); float* v = malloc(4 * nrays);
    int32_t* mesh = malloc(4 * nrays); int32_t* tri = malloc(4 * nrays); uint8_t* hit = malloc(nrays);
    vulkanrt_trace_rays(scene, nrays, rays, t, u, v, mesh, tri, hit);
    char fn[400]; snprintf(fn, sizeof fn, "%s/rays.bin", out); FILE* f = fopen(fn, "wb"); fwrite(rays, 32, nrays, f); fclose(f);
    snprintf(fn, sizeof fn, "%s/ref.txt", out); f = fopen(fn, "w"); int h = 0;
    for (int i = 0; i < nrays; i++) { fprintf(f, "%d %d %.6f %.6f %.6f %d\n", i, hit[i], t[i], u[i], v[i], tri[i]); h += hit[i]; }
    fclose(f); printf("%d/%d rays hit\n", h, nrays); return 0;
}
