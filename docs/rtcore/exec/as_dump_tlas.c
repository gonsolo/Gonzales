// Builds a scene of K single-triangle meshes (mesh k = unit triangle at x = 3k) with the Vulkan backend and writes the TLAS
// (tlas.bin) and every BLAS (blas<k>.bin) with .addr files into a directory.
// Build: gcc -O1 as_dump_tlas.c -I../../../src/vulkanrt -L../../../build -lvulkanrt -lm -Wl,-rpath,/home/gonsolo/work/gonzales/build -o /tmp/as_dump_tlas
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "vulkanrt.h"
static void save(void* scene, int kind, int index, const char* path) {
    uint64_t addr = 0; int64_t sz = vulkanrt_debug_read_as(scene, kind, index, NULL, 0, &addr);
    uint8_t* buf = malloc(sz); vulkanrt_debug_read_as(scene, kind, index, buf, sz, &addr);
    FILE* f = fopen(path, "wb"); fwrite(buf, 1, sz, f); fclose(f);
    char fa[600]; snprintf(fa, sizeof fa, "%s.addr", path); f = fopen(fa, "w"); fprintf(f, "%llu %lld\n", (unsigned long long)addr, (long long)sz); fclose(f);
    printf("%s: %lld bytes at 0x%llx\n", path, (long long)sz, (unsigned long long)addr);
}
int main(int argc, char** argv) {
    int k = atoi(argv[2]); const char* dir = argv[1];
    VulkanRtMesh* ms = calloc(k, sizeof *ms); int64_t* pcs = malloc(8 * k); int64_t* ics = malloc(8 * k);
    for (int m = 0; m < k; m++) {
        float* p = malloc(48); int32_t* idx = malloc(12);
        float v[9] = {3.0f * m, 0, 0, 3.0f * m + 1, 0, 0, 3.0f * m, 1, 0};
        for (int c = 0; c < 3; c++) { p[4 * c] = v[3 * c]; p[4 * c + 1] = v[3 * c + 1]; p[4 * c + 2] = v[3 * c + 2]; p[4 * c + 3] = 1; idx[c] = c; }
        ms[m].points = p; ms[m].vertexIndices = idx; pcs[m] = 3; ics[m] = 3;
    }
    void* scene = vulkanrt_build_scene(ms, k, pcs, ics);
    if (!scene) { fprintf(stderr, "build failed\n"); return 1; }
    char path[700]; snprintf(path, sizeof path, "%s/tlas.bin", dir); save(scene, 1, 0, path);
    for (int m = 0; m < k; m++) { snprintf(path, sizeof path, "%s/blas%d.bin", dir, m); save(scene, 0, m, path); }
    return 0;
}
