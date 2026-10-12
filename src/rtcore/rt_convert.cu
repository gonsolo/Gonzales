// Converts the raw records rtcore's trace kernel writes (t, u, v, hit word, ...) into the Result layout the Vulkan
// interop path produces, so the existing vulkaninterop_unpack_results_kernel can consume them unchanged:
//   float t, u, v, pad; int mesh, triangle, hitFlag, geometryIndex      (32 bytes per ray)
// The trace runs on ONE merged triangle geometry, so the global triangle index is mapped back to (mesh, local triangle)
// with the prefix sums of the per-mesh triangle counts. Built to build/rt_convert.cubin by build_rt_cubin.py.
// Compact id table: per group of 8 ids a head word; wide groups (bit 31) index 8 raw ints, others a base plus 8 byte deltas.
__device__ __forceinline__ int gid_lookup(const int* gid, const unsigned int* gHead, const unsigned char* gDel, const int* gWide, unsigned int id) {
    if (!gHead) return gid[id];
    unsigned int g = id >> 3, k = id & 7u, h = gHead[g];
    if (h & 0x80000000u) return gWide[(size_t)(h & 0x7fffffffu) * 8 + k];
    return (int)h + (int)gDel[(size_t)g * 8 + k];
}

extern "C" __global__ void convert(const unsigned int* rt, float* out, const int* domBase, const int* domN, const int* pre,
                                   const int* rawv, const int* geom, int nDomains, int hasTlas, unsigned int count, const int* gid,
                                   const int* instBlas, int instStride, int nInst, const int* tpreOff, const int* tCnt, int nTemplates, int rawBase,
                                   const unsigned int* gHead, const unsigned char* gDel, const int* gWide) {
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
    int tk = -1;                                         // a template instance: its BLAS from the per-instance table
    if (nInst > 0 && d >= 0 && d < nInst) tk = instBlas[(size_t)d * (size_t)instStride];
    bool direct = tk >= 0 && tk < nTemplates;
    if (!direct && (d < 0 || d >= nDomains || nInst > 0)) d = 0;
    int base = direct ? 0 : domBase[d];
    int lo = 0, hi = direct ? -1 : domN[d];              // largest j with pre[base + j] <= tri
    if (hi < 0) {                                        // direct domain (a template instance): the hardware gave the geometry
        int tri, geomHw;
        if (gid || gHead) { low = (unsigned int)gid_lookup(gid, gHead, gDel, gWide, low); kind = 5; }   // own structure: primitive id -> running triangle number in the template
        if (kind >= 5) {                                 // running number: this template's prefix sums are at pre[pre[base]..]
            const int* tp = direct ? pre + tpreOff[tk] : pre + pre[base];
            int a = 0, b = direct ? tCnt[tk] : geom[base];
            while (b - a > 1) { int mid = (a + b) >> 1; if (tp[mid] <= (int)low) a = mid; else b = mid; }
            geomHw = a; tri = (int)low - tp[a];
        } else {
            int s = 28 - 4 * (int)kind;
            geomHw = (int)(low >> s); tri = (int)(low & ((1u << s) - 1u));
        }
        o[0] = __uint_as_float(r[0]); o[1] = fabsf(__uint_as_float(r[1])); o[2] = __uint_as_float(r[2]); o[3] = 0.0f;
        oi[4] = direct ? rawBase + d : rawv[base]; oi[5] = tri; oi[6] = 1; oi[7] = geomHw;
        return;
    }
    int tri = (int)low;
    if (gid || gHead) tri = gid_lookup(gid, gHead, gDel, gWide, (unsigned int)tri);                             // hand-built structures number primitives by block slot
    while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (pre[base + mid] <= tri) lo = mid; else hi = mid; }
    // The hardware returns u with a negative sign on some hits (same magnitude Vulkan reports positive; presumably a
    // back-face indicator). Barycentrics are never negative, so the sign is dropped.
    o[0] = __uint_as_float(r[0]); o[1] = fabsf(__uint_as_float(r[1])); o[2] = __uint_as_float(r[2]); o[3] = 0.0f;
    oi[4] = rawv[base + lo]; oi[5] = tri - pre[base + lo]; oi[6] = 1; oi[7] = geom[base + lo];
}
