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
    // Hit word: the top three bits select the layout of the low 29 (measured with templates of 1..140000 geometries of
    // 2..5,000,000 triangles). Kinds 0..4: the triangle number within its geometry is in the low s = 28 - 4 * kind bits
    // and the geometry index above it. Kinds 5 and up (6 seen): one running triangle number over all geometries, used
    // when the two fields do not fit. A single-geometry BLAS is the same with geometry 0, so its triangle is the low 29.
    unsigned int kind = word >> 29, low = word & 0x1fffffffu;
    int d = hasTlas ? (int)r[4] - 1 : 0;                 // r[4] = hit instance index + 1
    if (d < 0 || d >= nDomains) d = 0;
    int base = domBase[d];
    int lo = 0, hi = domN[d];                            // largest j with pre[base + j] <= tri
    if (hi < 0) {                                        // direct domain (a template instance): the hardware gave the geometry
        int tri, geomHw;
        if (kind >= 5) {                                 // running number: this template's prefix sums are at pre[pre[base]..]
            const int* tp = pre + pre[base];
            int a = 0, b = geom[base];
            while (b - a > 1) { int mid = (a + b) >> 1; if (tp[mid] <= (int)low) a = mid; else b = mid; }
            geomHw = a; tri = (int)low - tp[a];
        } else {
            int s = 28 - 4 * (int)kind;
            geomHw = (int)(low >> s); tri = (int)(low & ((1u << s) - 1u));
        }
        o[0] = __uint_as_float(r[0]); o[1] = fabsf(__uint_as_float(r[1])); o[2] = __uint_as_float(r[2]); o[3] = 0.0f;
        oi[4] = rawv[base]; oi[5] = tri; oi[6] = 1; oi[7] = geomHw;
        return;
    }
    int tri = (int)low;
    while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (pre[base + mid] <= tri) lo = mid; else hi = mid; }
    // The hardware returns u with a negative sign on some hits (same magnitude Vulkan reports positive; presumably a
    // back-face indicator). Barycentrics are never negative, so the sign is dropped.
    o[0] = __uint_as_float(r[0]); o[1] = fabsf(__uint_as_float(r[1])); o[2] = __uint_as_float(r[2]); o[3] = 0.0f;
    oi[4] = rawv[base + lo]; oi[5] = tri - pre[base + lo]; oi[6] = 1; oi[7] = geom[base + lo];
}
