// Software BVH baseline for the RT-core experiment: same mesh and rays as as_probe, CPU-built binary BVH
// (median split on the longest centroid axis, <=4 triangles per leaf), GPU closest-hit traversal with a per-thread
// stack and Moller-Trumbore. Validates against Vulkan's reference hits, then times repeated launches.
// Build: nvcc -O3 -arch=sm_86 sw_bvh_baseline.cu -o /tmp/sw_bvh
// Usage: sw_bvh <as_probe dir> <grid> <rays>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
struct Tri { float v[9]; };
struct Node { float lo[3], hi[3]; int left, right_or_first, count; };  // count>0: leaf (first=right_or_first)
static std::vector<Tri> tris; static std::vector<int> order; static std::vector<Node> nodes;
static void centroid(const Tri& t, float* c) { for (int a = 0; a < 3; a++) c[a] = (t.v[a] + t.v[3 + a] + t.v[6 + a]) / 3.f; }
static int build(int b, int e) {
    int id = nodes.size(); nodes.push_back(Node());
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f}, clo[3] = {1e30f, 1e30f, 1e30f}, chi[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = b; i < e; i++) { const Tri& t = tris[order[i]]; float c[3]; centroid(t, c);
        for (int a = 0; a < 3; a++) { for (int k = 0; k < 3; k++) { lo[a] = std::min(lo[a], t.v[3 * k + a]); hi[a] = std::max(hi[a], t.v[3 * k + a]); }
            clo[a] = std::min(clo[a], c[a]); chi[a] = std::max(chi[a], c[a]); } }
    Node n; for (int a = 0; a < 3; a++) { n.lo[a] = lo[a]; n.hi[a] = hi[a]; }
    if (e - b <= 4) { n.left = -1; n.right_or_first = b; n.count = e - b; nodes[id] = n; return id; }
    int ax = 0; for (int a = 1; a < 3; a++) if (chi[a] - clo[a] > chi[ax] - clo[ax]) ax = a;
    int m = (b + e) / 2;
    std::nth_element(order.begin() + b, order.begin() + m, order.begin() + e, [&](int x, int y) { float cx[3], cy[3]; centroid(tris[x], cx); centroid(tris[y], cy); return cx[ax] < cy[ax]; });
    n.count = 0; nodes[id] = n;
    int l = build(b, m); int r = build(m, e); nodes[id].left = l; nodes[id].right_or_first = r; return id;
}
__device__ __forceinline__ bool box(const Node& n, const float* o, const float* inv, float tmax, float& tn) {
    float t0 = 0.f, t1 = tmax;
    for (int a = 0; a < 3; a++) { float ta = (n.lo[a] - o[a]) * inv[a], tb = (n.hi[a] - o[a]) * inv[a]; float lo = fminf(ta, tb), hi = fmaxf(ta, tb); t0 = fmaxf(t0, lo); t1 = fminf(t1, hi); }
    tn = t0; return t0 <= t1;
}
__global__ void trace(const Node* nodes, const Tri* tris, const int* order, const float* rays, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const float* r = rays + 8 * i; float o[3] = {r[0], r[1], r[2]}, d[3] = {r[4], r[5], r[6]}, inv[3] = {1.f / d[0], 1.f / d[1], 1.f / d[2]};
    float tmax = r[7], bu = 0, bv = 0; int bt = -1; int stack[40]; int sp = 0; stack[sp++] = 0;
    while (sp) { const Node& nd = nodes[stack[--sp]]; float tn;
        if (!box(nd, o, inv, tmax, tn)) continue;
        if (nd.count > 0) { for (int k = 0; k < nd.count; k++) { int ti = order[nd.right_or_first + k]; const float* v = tris[ti].v;
                float e1[3] = {v[3] - v[0], v[4] - v[1], v[5] - v[2]}, e2[3] = {v[6] - v[0], v[7] - v[1], v[8] - v[2]};
                float p[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
                float det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2]; if (fabsf(det) < 1e-12f) continue; float id = 1.f / det;
                float s[3] = {o[0] - v[0], o[1] - v[1], o[2] - v[2]}; float u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * id; if (u < 0 || u > 1) continue;
                float q[3] = {s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]};
                float w = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * id; if (w < 0 || u + w > 1) continue;
                float t = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * id; if (t > r[3] && t < tmax) { tmax = t; bu = u; bv = w; bt = ti; } } }
        else { stack[sp++] = nd.right_or_first; stack[sp++] = nd.left; } }
    out[4 * i] = bt >= 0 ? tmax : -1.f; out[4 * i + 1] = bu; out[4 * i + 2] = bv; ((int*)out)[4 * i + 3] = bt;
}
int main(int argc, char** argv) {
    int g = atoi(argv[2]); int nr = atoi(argv[3]); char p[512];
    int nv = (g + 1) * (g + 1); std::vector<float> pts(4 * nv);
    for (int y = 0; y <= g; y++) for (int x = 0; x <= g; x++) { float* q = &pts[4 * (y * (g + 1) + x)]; q[0] = x; q[1] = y; q[2] = 0.3f * sinf(x * 0.9f) * cosf(y * 0.7f); q[3] = 1; }
    for (int y = 0; y < g; y++) for (int x = 0; x < g; x++) { int a = y * (g + 1) + x, b = a + 1, c = a + g + 1, d = c + 1; int ids[6] = {a, b, c, b, d, c};
        for (int t = 0; t < 2; t++) { Tri tr; for (int k = 0; k < 3; k++) for (int ax = 0; ax < 3; ax++) tr.v[3 * k + ax] = pts[4 * ids[3 * t + k] + ax]; tris.push_back(tr); } }
    order.resize(tris.size()); for (size_t i = 0; i < order.size(); i++) order[i] = i;
    build(0, tris.size()); printf("%zu triangles, %zu nodes\n", tris.size(), nodes.size());
    snprintf(p, sizeof p, "%s/rays.bin", argv[1]); std::vector<float> rays(8 * nr); FILE* f = fopen(p, "rb"); fread(rays.data(), 32, nr, f); fclose(f);
    Node* dn; Tri* dt; int* dord; float* dr; float* dout;
    cudaMalloc(&dn, nodes.size() * sizeof(Node)); cudaMemcpy(dn, nodes.data(), nodes.size() * sizeof(Node), cudaMemcpyHostToDevice);
    cudaMalloc(&dt, tris.size() * sizeof(Tri)); cudaMemcpy(dt, tris.data(), tris.size() * sizeof(Tri), cudaMemcpyHostToDevice);
    cudaMalloc(&dord, order.size() * 4); cudaMemcpy(dord, order.data(), order.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&dr, 32 * (size_t)nr); cudaMemcpy(dr, rays.data(), 32 * (size_t)nr, cudaMemcpyHostToDevice); cudaMalloc(&dout, 16 * (size_t)nr);
    trace<<<(nr + 63) / 64, 64>>>(dn, dt, dord, dr, dout, nr); cudaDeviceSynchronize();
    std::vector<float> res(4 * nr); cudaMemcpy(res.data(), dout, 16 * (size_t)nr, cudaMemcpyDeviceToHost);
    snprintf(p, sizeof p, "%s/ref.txt", argv[1]); f = fopen(p, "r"); int bad = 0, hits = 0;
    for (int i = 0; i < nr; i++) { int idx, hit, tri; float t, u, v; fscanf(f, "%d %d %f %f %f %d", &idx, &hit, &t, &u, &v, &tri);
        bool h = res[4 * i] >= 0; hits += h; if (h != (bool)hit || (hit && (fabsf(res[4 * i] - t) > 1e-4f || ((int*)res.data())[4 * i + 3] != tri))) bad++; }
    fclose(f); printf("validation: %d hits, %d mismatches vs Vulkan\n", hits, bad);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); int reps = 50; cudaEventRecord(e0);
    for (int r = 0; r < reps; r++) trace<<<(nr + 63) / 64, 64>>>(dn, dt, dord, dr, dout, nr);
    cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1);
    printf("software BVH: %.3f ms/launch = %.1f Mrays/s\n", ms / reps, nr / (ms / reps) / 1000.0); return 0;
}
