// Converts the raw records rtcore's trace kernel writes (t, u, v, hit word, ...) into the Result layout the Vulkan
// interop path produces, so the existing vulkaninterop_unpack_results_kernel can consume them unchanged:
//   float t, u, v, pad; int mesh, triangle, hitFlag, geometryIndex      (32 bytes per ray)
// The trace runs on ONE merged triangle geometry, so the global triangle index is mapped back to (mesh, local triangle)
// with the prefix sums of the per-mesh triangle counts. Built to build/rt_convert.cubin by build_rt_cubin.py.
extern "C" __global__ void convert(const unsigned int* rt, float* out, const int* domBase, const int* domN, const int* pre,
                                   const int* rawv, const int* geom, int nDomains, int hasTlas, unsigned int count) {
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
    // Hit word: a BLAS with a single geometry has bit 31 set and the triangle number in the low 29 bits. With several
    // geometries bit 31 is clear, k = word >> 29 selects the split s = 28 - 4k: the geometry index is in bits 28..s and the
    // triangle number within that geometry in the low s bits (measured on a 9-geometry template BLAS).
    int tri, geomHw = 0;
    if (word & 0x80000000u) tri = (int)(word & 0x1fffffffu);
    else { int s = 28 - 4 * (int)(word >> 29); geomHw = (int)((word >> s) & ((1u << (29 - s)) - 1u)); tri = (int)(word & ((1u << s) - 1u)); }
    int d = hasTlas ? (int)r[4] - 1 : 0;                 // r[4] = hit instance index + 1
    if (d < 0 || d >= nDomains) d = 0;
    int base = domBase[d];
    int lo = 0, hi = domN[d];                            // largest j with pre[base + j] <= tri
    if (hi < 0) {                                        // direct domain (a template instance): the hardware gave the geometry
        o[0] = __uint_as_float(r[0]); o[1] = fabsf(__uint_as_float(r[1])); o[2] = __uint_as_float(r[2]); o[3] = 0.0f;
        oi[4] = rawv[base]; oi[5] = tri; oi[6] = 1; oi[7] = geomHw;
        return;
    }
    while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (pre[base + mid] <= tri) lo = mid; else hi = mid; }
    // The hardware returns u with a negative sign on some hits (same magnitude Vulkan reports positive; presumably a
    // back-face indicator). Barycentrics are never negative, so the sign is dropped.
    o[0] = __uint_as_float(r[0]); o[1] = fabsf(__uint_as_float(r[1])); o[2] = __uint_as_float(r[2]); o[3] = 0.0f;
    oi[4] = rawv[base + lo]; oi[5] = tri - pre[base + lo]; oi[6] = 1; oi[7] = geom[base + lo];
}
