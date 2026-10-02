// Builds a small triangle scene with the Vulkan ray-query backend, writes the driver's BLAS/TLAS bytes and a
// set of test rays with Vulkan's own hit results to files, so a CUDA kernel can be compared against them.
// Build: gcc -O1 as_probe.c -I../../../src/vulkanrt -L../../../build -lvulkanrt -lm -Wl,-rpath,/home/gonsolo/work/gonzales/build -o /tmp/as_probe
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "vulkanrt.h"
int main(int argc, char** argv) {
    const char* out = argc > 1 ? argv[1] : "/tmp/ncu/asprobe"; char cmd[300]; snprintf(cmd, sizeof cmd, "mkdir -p %s", out); system(cmd);
    int g = argc > 2 ? atoi(argv[2]) : 4, nrays = argc > 3 ? atoi(argv[3]) : 256, nm = argc > 4 ? atoi(argv[4]) : 1;
    int nv = (g + 1) * (g + 1), nt = g * g * 2;
    float* pts = malloc(16 * nv); int64_t* idx = malloc(8 * 3 * nt);
    for (int y = 0; y <= g; y++) for (int x = 0; x <= g; x++) { float* p = pts + 4 * (y * (g + 1) + x);
        p[0] = x; p[1] = y; p[2] = 0.3f * sinf(x * 0.9f) * cosf(y * 0.7f); p[3] = 1; }
    int k = 0;
    for (int y = 0; y < g; y++) for (int x = 0; x < g; x++) { int a = y * (g + 1) + x, b = a + 1, c = a + g + 1, d = c + 1;
        idx[k++] = a; idx[k++] = b; idx[k++] = c; idx[k++] = b; idx[k++] = d; idx[k++] = c; }
    VulkanRtMesh ms[8]; int64_t pcs[8], ics[8]; float* pp[8];
    for (int q = 0; q < nm; q++) { pp[q] = malloc(16 * nv); for (int i = 0; i < nv; i++) { pp[q][4 * i] = pts[4 * i] + q * (g + 3); pp[q][4 * i + 1] = pts[4 * i + 1]; pp[q][4 * i + 2] = pts[4 * i + 2] + 0.5f * q; pp[q][4 * i + 3] = 1; }
        memset(&ms[q], 0, sizeof ms[q]); ms[q].points = pp[q]; ms[q].vertexIndices = idx; pcs[q] = nv; ics[q] = 3 * nt; }
    struct timespec ta, tb; clock_gettime(CLOCK_MONOTONIC, &ta);
    void* scene = vulkanrt_build_scene(ms, nm, pcs, ics);
    clock_gettime(CLOCK_MONOTONIC, &tb); fprintf(stderr, "vulkanrt_build_scene: %.1f ms for %d triangles x %d mesh(es)\n", (tb.tv_sec - ta.tv_sec) * 1e3 + (tb.tv_nsec - ta.tv_nsec) / 1e6, nt, nm);
    if (!scene) { fprintf(stderr, "build failed\n"); return 1; }
    for (int kind = 0; kind < 2 + (nm - 1); kind++) {
        int bi = kind >= 2 ? kind - 1 : 0; int kk = kind == 1 ? 1 : 0; if (kind >= 2) bi = kind - 1;
        uint64_t addr = 0; int64_t sz = vulkanrt_debug_read_as(scene, kk, bi, NULL, 0, &addr);
        uint8_t* buf = malloc(sz); vulkanrt_debug_read_as(scene, kk, bi, buf, sz, &addr);
        char fn[400]; char nm_[32]; if (kind == 1) snprintf(nm_, sizeof nm_, "tlas"); else if (kind == 0) snprintf(nm_, sizeof nm_, "blas"); else snprintf(nm_, sizeof nm_, "blas%d", kind - 1);
        snprintf(fn, sizeof fn, "%s/%s.bin", out, nm_); FILE* f = fopen(fn, "wb"); fwrite(buf, 1, sz, f); fclose(f);
        { char fa[400]; snprintf(fa, sizeof fa, "%s/%s.addr", out, nm_); FILE* fx = fopen(fa, "w"); fprintf(fx, "%llu %lld\n", (unsigned long long)addr, (long long)sz); fclose(fx); }
        printf("%s: %lld bytes, vk address 0x%llx\n", kind ? "TLAS" : "BLAS", (long long)sz, (unsigned long long)addr);
    }
    float* rays = malloc(32 * nrays); srand(7);
    for (int i = 0; i < nrays; i++) { float* r = rays + 8 * i;
        r[0] = (g + 3) * nm * (rand() / (float)RAND_MAX) - 1; r[1] = g * (rand() / (float)RAND_MAX); r[2] = 5; r[3] = 0;
        r[4] = 0.1f * (rand() / (float)RAND_MAX - .5f); r[5] = 0.1f * (rand() / (float)RAND_MAX - .5f); r[6] = -1; r[7] = 100; }
    float* t = malloc(4 * nrays); float* u = malloc(4 * nrays); float* v = malloc(4 * nrays);
    int32_t* mesh = malloc(4 * nrays); int32_t* tri = malloc(4 * nrays); uint8_t* hit = malloc(nrays);
    vulkanrt_trace_rays(scene, nrays, rays, t, u, v, mesh, tri, hit);
    char fn[400]; snprintf(fn, sizeof fn, "%s/rays.bin", out); FILE* f = fopen(fn, "wb"); fwrite(rays, 32, nrays, f); fclose(f);
    snprintf(fn, sizeof fn, "%s/ref.txt", out); f = fopen(fn, "w"); int h = 0;
    for (int i = 0; i < nrays; i++) { fprintf(f, "%d %d %.6f %.6f %.6f %d %d\n", i, hit[i], t[i], u[i], v[i], tri[i], mesh[i]); h += hit[i]; }
    fclose(f); printf("%d/%d rays hit\n", h, nrays); return 0;
}
