// The ported march on the CPU behind a midpoint-split BVH over the sub-square boxes: checks the port against the Mojo
// prototype before the NVIDIA card is in.
//
//   cpu_render surface.dump hits.bin
#include "disp_load.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

using namespace gz;

struct Node {
        float lo[3], hi[3];
        int left, count;   // count > 0: leaf with prims [left, left + count) of the order; else children left, left + 1
};

static void build(std::vector<Node> &nodes, std::vector<int> &ord, const float *bx, int ni, int b, int e) {
        Node &nd = nodes[ni];
        for (int k = 0; k < 3; ++k)
                nd.lo[k] = 1e30f, nd.hi[k] = -1e30f;
        for (int i = b; i < e; ++i)
                for (int k = 0; k < 3; ++k)
                        nd.lo[k] = std::min(nd.lo[k], bx[6 * ord[i] + k]), nd.hi[k] = std::max(nd.hi[k], bx[6 * ord[i] + 3 + k]);
        if (e - b <= 4) {
                nd.left = b, nd.count = e - b;
                return;
        }
        int ax = 0;
        float ext = -1;
        for (int k = 0; k < 3; ++k)
                if (nd.hi[k] - nd.lo[k] > ext)
                        ext = nd.hi[k] - nd.lo[k], ax = k;
        int m = (b + e) / 2;
        std::nth_element(ord.begin() + b, ord.begin() + m, ord.begin() + e,
                         [&](int x, int y) { return bx[6 * x + ax] + bx[6 * x + 3 + ax] < bx[6 * y + ax] + bx[6 * y + 3 + ax]; });
        int l = int(nodes.size());
        nodes.push_back({}), nodes.push_back({});
        nodes[ni].left = l, nodes[ni].count = 0;
        build(nodes, ord, bx, l, b, m);
        build(nodes, ord, bx, l + 1, m, e);
}

static bool slab(const float *lo, const float *hi, V3 o, V3 inv, float tmax, float &t0) {
        float a[3] = {o.x, o.y, o.z}, iv[3] = {inv.x, inv.y, inv.z}, tn = 0.f, tf = tmax;
        for (int k = 0; k < 3; ++k) {
                float p = (lo[k] - a[k]) * iv[k], q = (hi[k] - a[k]) * iv[k];
                tn = std::max(tn, std::min(p, q)), tf = std::min(tf, std::max(p, q));
        }
        t0 = tn;
        return tn <= tf;
}

int main(int argc, char **argv) {
        if (argc < 3) {
                std::fprintf(stderr, "usage: %s surface.dump hits.bin\n", argv[0]);
                return 1;
        }
        Dump d = load_dump(argv[1]);
        std::vector<int> ord(d.nprim);
        for (int i = 0; i < d.nprim; ++i)
                ord[i] = i;
        std::vector<Node> nodes(1);
        nodes.reserve(size_t(d.nprim));
        build(nodes, ord, d.boxes, 0, 0, d.nprim);
        int W = d.cam.w, H = d.cam.h;
        std::vector<PixHit> hits(size_t(W) * H);
        std::atomic<int> next{0};
        std::atomic<long> steps{0};
        auto t0 = std::chrono::steady_clock::now();
        auto work = [&]() {
                long st = 0;
                for (int y; (y = next++) < H;)
                        for (int x = 0; x < W; ++x) {
                                V3 o = d.cam.eye, dir = d.cam.dir(x, y), inv = v3(1.f / dir.x, 1.f / dir.y, 1.f / dir.z);
                                PixHit h{1e30f, -1, 0.f, 0.f};
                                int stack[64], sp = 0;
                                stack[sp++] = 0;
                                while (sp) {
                                        const Node &nd = nodes[stack[--sp]];
                                        float te;
                                        if (!slab(nd.lo, nd.hi, o, inv, h.t, te))
                                                continue;
                                        if (nd.count == 0) {
                                                stack[sp++] = nd.left, stack[sp++] = nd.left + 1;
                                                continue;
                                        }
                                        for (int i = nd.left; i < nd.left + nd.count; ++i) {
                                                MHit m = intersect_march(d.s, ord[i], o, dir, h.t);
                                                st += m.steps;
                                                if (m.hit && m.t < h.t)
                                                        h = {m.t, prim_patch(d.s, ord[i]).patch, m.s, m.u};
                                        }
                                }
                                hits[size_t(y) * W + x] = h;
                        }
                steps += st;
        };
        std::vector<std::thread> th;
        for (unsigned k = 0; k < std::max(1u, std::thread::hardware_concurrency()); ++k)
                th.emplace_back(work);
        for (auto &t : th)
                t.join();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        long nh = 0;
        for (auto &h : hits)
                nh += h.patch >= 0;
        write_hits(argv[2], hits);
        std::printf("cpu march: %d x %d, %d sub-squares, %ld hits, %.1f ms, steps/ray %.2f\n", W, H, d.nprim, nh, ms, double(steps) / (double(W) * H));
        return 0;
}
