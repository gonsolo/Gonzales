// Converts the raw records rtcore's trace kernel writes (t, u, v, hit word, ...) into the Result layout the Vulkan
// interop path produces, so the existing vulkaninterop_unpack_results_kernel can consume them unchanged:
//   float t, u, v, pad; int mesh, triangle, hitFlag, geometryIndex      (32 bytes per ray)
// The trace runs on ONE merged triangle geometry, so the global triangle index is mapped back to (mesh, local triangle)
// with the prefix sums of the per-mesh triangle counts. Built to build/rt_convert.cubin by build_rt_cubin.py.
extern "C" __global__ void convert(const unsigned int* rt, float* out, const int* prefix, int nMeshes, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned int* r = rt + 8 * i;
    float* o = out + 8 * i;
    int* oi = (int*)o;
    unsigned int word = r[3];
    if (word == 0xffffffffu) {
        o[0] = -1.0f; o[1] = 0.0f; o[2] = 0.0f; o[3] = 0.0f;
        oi[4] = -1; oi[5] = -1; oi[6] = 0; oi[7] = 0;
        return;
    }
    int tri = (int)(word & 0x1fffffffu);
    int lo = 0, hi = nMeshes;              // largest m with prefix[m] <= tri
    while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (prefix[mid] <= tri) lo = mid; else hi = mid; }
    // The hardware returns u with a negative sign on some hits (same magnitude Vulkan reports positive; presumably a
    // back-face indicator). Barycentrics are never negative, so the sign is dropped.
    o[0] = __uint_as_float(r[0]); o[1] = fabsf(__uint_as_float(r[1])); o[2] = __uint_as_float(r[2]); o[3] = 0.0f;
    oi[4] = lo; oi[5] = tri - prefix[lo]; oi[6] = 1; oi[7] = 0;
}
