// Builds a BLAS from triangles given as 9 floats each in a binary file (argv[1]), writes the driver's BLAS bytes to argv[2].
// Build: gcc -O1 as_dump.c -I../../../src/vulkanrt -L../../../build -lvulkanrt -lm -Wl,-rpath,/home/gonsolo/work/gonzales/build -o /tmp/as_dump
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "vulkanrt.h"
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: as_dump tris.bin out.bin\n"); return 1; }
    FILE* f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f) / 36; fseek(f, 0, SEEK_SET);
    float* tri = malloc(36 * n); fread(tri, 36, n, f); fclose(f);
    float* pts = malloc(16 * 3 * n); int32_t* idx = malloc(4 * 3 * n);
    for (long t = 0; t < n; t++) for (int c = 0; c < 3; c++) {
        float* p = pts + 4 * (3 * t + c); p[0] = tri[9 * t + 3 * c]; p[1] = tri[9 * t + 3 * c + 1]; p[2] = tri[9 * t + 3 * c + 2]; p[3] = 1;
        idx[3 * t + c] = 3 * t + c; }
    VulkanRtMesh m; memset(&m, 0, sizeof m); m.points = pts; m.vertexIndices = idx;
    int64_t pc = 3 * n, ic = 3 * n;
    void* scene = vulkanrt_build_scene(&m, 1, &pc, &ic);
    if (!scene) { fprintf(stderr, "build failed\n"); return 1; }
    uint64_t addr = 0; int64_t sz = vulkanrt_debug_read_as(scene, 0, 0, NULL, 0, &addr);
    uint8_t* buf = malloc(sz); vulkanrt_debug_read_as(scene, 0, 0, buf, sz, &addr);
    f = fopen(argv[2], "wb"); fwrite(buf, 1, sz, f); fclose(f);
    { char fa[600]; snprintf(fa, sizeof fa, "%s.addr", argv[2]); FILE* fx = fopen(fa, "w"); fprintf(fx, "%llu %lld\n", (unsigned long long)addr, (long long)sz); fclose(fx); }
    int64_t last = sz - 1; while (last > 0 && buf[last] == 0) last--;
    printf("%ld triangles: buffer %lld bytes, last non-zero byte at %lld\n", n, (long long)sz, (long long)last);
    return 0;
}
