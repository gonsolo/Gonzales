// Traces batches of rays through the Vulkan ray-query backend in a loop, so a
// graphics profiler (Nsight Graphics GPU Trace) has dispatches to capture.
// Build: gcc -O2 trace_loop.c -I../../src/vulkanrt -L../../build -lvulkanrt -Wl,-rpath,../../build -o /tmp/trace_loop
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "vulkanrt.h"
int main(int argc, char** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 200;
    int n = 1 << 20;
    int g = 64;  // g*g quads on the z=0 plane
    int nv = (g + 1) * (g + 1), nt = g * g * 2;
    float* pts = malloc(sizeof(float) * 4 * nv);
    int64_t* idx = malloc(sizeof(int64_t) * 3 * nt);
    for (int y = 0; y <= g; y++) for (int x = 0; x <= g; x++) {
        float* p = pts + 4 * (y * (g + 1) + x);
        p[0] = x; p[1] = y; p[2] = 0.1f * sinf(x * 0.5f) * cosf(y * 0.5f); p[3] = 1;
    }
    int k = 0;
    for (int y = 0; y < g; y++) for (int x = 0; x < g; x++) {
        int a = y * (g + 1) + x, b = a + 1, c = a + g + 1, d = c + 1;
        idx[k++] = a; idx[k++] = b; idx[k++] = c; idx[k++] = b; idx[k++] = d; idx[k++] = c;
    }
    VulkanRtMesh m = {0};
    m.points = pts; m.vertexIndices = idx;
    int64_t pc = nv, ic = 3 * nt;
    void* scene = vulkanrt_build_scene(&m, 1, &pc, &ic);
    if (!scene) { fprintf(stderr, "build failed\n"); return 1; }
    float* rays = malloc(sizeof(float) * 8 * n);
    srand(1);
    for (int i = 0; i < n; i++) {
        float* r = rays + 8 * i;
        r[0] = g * (rand() / (float)RAND_MAX); r[1] = g * (rand() / (float)RAND_MAX); r[2] = 5;
        r[3] = 0; r[4] = 0.2f * (rand() / (float)RAND_MAX - 0.5f); r[5] = 0.2f * (rand() / (float)RAND_MAX - 0.5f); r[6] = -1; r[7] = 100;
    }
    float* t = malloc(4 * n); float* u = malloc(4 * n); float* v = malloc(4 * n);
    int32_t* mesh = malloc(4 * n); int32_t* tri = malloc(4 * n); uint8_t* hit = malloc(n);
    for (int it = 0; it < iters; it++) {
        if (!vulkanrt_trace_rays(scene, n, rays, t, u, v, mesh, tri, hit)) { fprintf(stderr, "trace failed\n"); return 1; }
        if (it == 0) { int h = 0; for (int i = 0; i < n; i++) h += hit[i]; printf("hits %d / %d\n", h, n); }
    }
    vulkanrt_destroy_scene(scene);
    return 0;
}
