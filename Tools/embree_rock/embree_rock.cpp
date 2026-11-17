// Embree CPU baseline: Moana's lava rock as a Catmull-Clark subdivision surface displaced by its Ptex heights.
#include <embree4/rtcore.h>
#include <Ptexture.h>
#include <png.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct V3 { float x, y, z; };
static V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static V3 norm(V3 a) { return a * (1.0f / std::sqrt(dot(a, a))); }

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::atomic<long long> mem_cur{0}, mem_peak{0};
static bool monitor(void *, ssize_t bytes, bool) {
        long long c = mem_cur += bytes, p = mem_peak.load();
        while (c > p && !mem_peak.compare_exchange_weak(p, c)) {}
        return true;
}

struct Disp {
        PtexTexture *tex;
        float scale;                        // EXAG * normal sign
        float width;                        // Ptex filter width in face units (0: finest level)
        RTCGeometry geom;
        bool probe;                         // check the uv and normal conventions against rtcInterpolate
        std::atomic<long long> points{0}, agree{0}, checked{0};
        std::atomic<int> maxdev_bits{0};
};

static PtexFilter *filter_for(PtexTexture *tex) {
        thread_local PtexTexture *t = nullptr;
        thread_local PtexFilter *f = nullptr;
        if (t != tex) {
                if (f)
                        f->release();
                f = PtexFilter::getFilter(tex, PtexFilter::Options(PtexFilter::f_bilinear));
                t = tex;
        }
        return f;
}

static void displace(const RTCDisplacementFunctionNArguments *a) {
        Disp *d = static_cast<Disp *>(a->geometryUserPtr);
        PtexFilter *f = filter_for(d->tex);
        for (unsigned i = 0; i < a->N; ++i) {
                float h = 0, u = std::clamp(a->u[i], 0.0f, 1.0f), v = std::clamp(a->v[i], 0.0f, 1.0f);
                if (d->probe && i == 0) {   // Ng vs Ptex's dP/du x dP/dv, and the callback's P vs rtcInterpolate's
                        float P[3], du[3], dv[3];
                        rtcInterpolate1(d->geom, a->primID, a->u[i], a->v[i], RTC_BUFFER_TYPE_VERTEX, 0, P, du, dv, 3);
                        V3 n = cross({du[0], du[1], du[2]}, {dv[0], dv[1], dv[2]});
                        d->agree += dot(n, {a->Ng_x[i], a->Ng_y[i], a->Ng_z[i]}) > 0;
                        d->checked += 1;
                        float dev = std::sqrt(dot(V3{P[0], P[1], P[2]} - V3{a->P_x[i], a->P_y[i], a->P_z[i]},
                                                  V3{P[0], P[1], P[2]} - V3{a->P_x[i], a->P_y[i], a->P_z[i]}));
                        int bits; std::memcpy(&bits, &dev, 4);
                        int old = d->maxdev_bits.load();
                        while (bits > old && !d->maxdev_bits.compare_exchange_weak(old, bits)) {}
                }
                f->eval(&h, 0, 1, a->primID, u, v, d->width, 0, 0, d->width);
                const float s = d->scale * h;
                a->P_x[i] += s * a->Ng_x[i];
                a->P_y[i] += s * a->Ng_y[i];
                a->P_z[i] += s * a->Ng_z[i];
        }
        d->points += a->N;
}

static std::vector<float> list_f(const char *env, const char *def) {
        std::vector<float> r;
        std::stringstream ss(getenv(env) ? getenv(env) : def);
        for (std::string s; std::getline(ss, s, ',');)
                r.push_back(std::stof(s));
        return r;
}

static void write_png(const std::string &path, const std::vector<unsigned char> &rgb, int w, int h) {
        png_image img;
        std::memset(&img, 0, sizeof img);
        img.version = PNG_IMAGE_VERSION;
        img.width = w; img.height = h; img.format = PNG_FORMAT_RGB;
        if (!png_image_write_to_file(&img, path.c_str(), 0, rgb.data(), 3 * w, nullptr))
                fprintf(stderr, "cannot write %s\n", path.c_str());
}

static RTCGeometry make_geom(RTCDevice dev, const std::vector<float> &pos, const std::vector<int> &cnt, const std::vector<int> &idx,
                             float rate, Disp *d) {
        RTCGeometry g = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_SUBDIVISION);
        auto *fb = static_cast<unsigned *>(rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_FACE, 0, RTC_FORMAT_UINT, 4, cnt.size()));
        auto *ib = static_cast<unsigned *>(rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT, 4, idx.size()));
        auto *vb = static_cast<float *>(rtcSetNewGeometryBuffer(g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 12, pos.size() / 3));
        std::memcpy(fb, cnt.data(), cnt.size() * 4);
        std::memcpy(ib, idx.data(), idx.size() * 4);
        std::memcpy(vb, pos.data(), pos.size() * 4);
        rtcSetGeometryTessellationRate(g, rate);
        if (d) {
                d->geom = g;
                rtcSetGeometryUserData(g, d);
                rtcSetGeometryDisplacementFunction(g, displace);
        }
        rtcCommitGeometry(g);
        return g;
}

int main() {
        const char *disp_path = getenv("DISP") ? getenv("DISP") : "/mnt/storage/moana/base/ex/island/textures/isLavaRocks/Displacement/rockfacebg0001_geo.ptx";
        const char *col_path = getenv("COLOR") ? getenv("COLOR") : "/mnt/storage/moana/base/ex/island/textures/isLavaRocks/Color/rockfacebg0001_geo.ptx";
        const std::string out = getenv("OUT") ? getenv("OUT") : ".";
        const auto exags = list_f("EXAGS", "1,8,16,32");
        const auto rates = list_f("RATES", "16,32,64,128");
        const auto ws = list_f("WIDTHS", "960,960"), hs = list_f("HEIGHTS", "540,720");
        const int reps = getenv("REPS") ? atoi(getenv("REPS")) : 3;
        const float fwidth = getenv("FILTER_WIDTH") ? atof(getenv("FILTER_WIDTH")) : 0.0f;

        Ptex::String err;
        PtexCache *cache = PtexCache::create(64, 4ull << 30);
        PtexTexture *tex = cache->get(disp_path, err), *ctex = cache->get(col_path, err);
        if (!tex || !ctex) { fprintf(stderr, "Ptex: %s\n", err.c_str()); return 1; }
        PtexPtr<PtexMetaData> meta(tex->getMetaData());
        const float *p = nullptr; const int *c = nullptr, *v = nullptr; int np = 0, nc = 0, nv = 0;
        meta->getValue("PtexVertPositions", p, np);
        meta->getValue("PtexFaceVertCounts", c, nc);
        meta->getValue("PtexFaceVertIndices", v, nv);
        if (!p || !c || !v || nc != tex->numFaces()) { fprintf(stderr, "no cage in %s\n", disp_path); return 1; }
        std::vector<float> pos(p, p + np);
        std::vector<int> cnt(c, c + nc), idx(v, v + nv);
        meta.reset();   // before the cache goes
        int quads = 0; long long texels = 0; int maxres = 0;
        for (int f = 0; f < nc; ++f) {
                quads += cnt[f] == 4;
                Ptex::Res r = tex->getFaceInfo(f).res;
                texels += (long long)r.u() * r.v();
                maxres = std::max({maxres, (int)r.u(), (int)r.v()});
        }
        V3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
        for (int i = 0; i < np / 3; ++i) {
                lo = {std::min(lo.x, pos[3 * i]), std::min(lo.y, pos[3 * i + 1]), std::min(lo.z, pos[3 * i + 2])};
                hi = {std::max(hi.x, pos[3 * i]), std::max(hi.y, pos[3 * i + 1]), std::max(hi.z, pos[3 * i + 2])};
        }
        printf("cage: %d verts, %d faces (%d quads), %zu indices; displacement Ptex %lld texels, max face res %d, data type %d\n",
               np / 3, nc, quads, idx.size(), texels, maxres, (int)tex->dataType());
        printf("box %g %g %g .. %g %g %g\n", lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);

        RTCDevice dev = rtcNewDevice(getenv("EMBREE_CONFIG"));
        if (!dev) { fprintf(stderr, "no Embree device\n"); return 1; }
        rtcSetDeviceMemoryMonitorFunction(dev, monitor, nullptr);
        printf("Embree %ld.%ld.%ld, subdivision supported %ld, threads %u, CPU threads via TBB\n",
               rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_VERSION_MAJOR), rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_VERSION_MINOR),
               rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_VERSION_PATCH),
               rtcGetDeviceProperty(dev, RTC_DEVICE_PROPERTY_SUBDIVISION_GEOMETRY_SUPPORTED), std::thread::hardware_concurrency());

        // Conventions: Embree's quad uv (p0 base, u to p1, v to p3) against the cage, and the sign of Embree's Ng.
        float sign = 1;
        {
                Disp d{tex, 0.0f, fwidth, nullptr, true};
                RTCScene s = rtcNewScene(dev);
                RTCGeometry g = make_geom(dev, pos, cnt, idx, 4, &d);
                rtcAttachGeometry(s, g);
                double t0 = now();
                rtcCommitScene(s);
                int ok = 0, off = 0;
                for (int f = 0; f < nc; ++f) {
                        if (cnt[f] != 4) continue;
                        const float cu[4] = {0, 1, 1, 0}, cv[4] = {0, 0, 1, 1};
                        float dist[4][4];   // limit corner k to cage vertex j
                        for (int k = 0; k < 4; ++k) {
                                float P[3];
                                rtcInterpolate1(g, f, cu[k], cv[k], RTC_BUFFER_TYPE_VERTEX, 0, P, nullptr, nullptr, 3);
                                for (int j = 0; j < 4; ++j) {
                                        const float *q = &pos[3 * idx[off + j]];
                                        dist[k][j] = std::sqrt((P[0] - q[0]) * (P[0] - q[0]) + (P[1] - q[1]) * (P[1] - q[1]) + (P[2] - q[2]) * (P[2] - q[2]));
                                }
                        }
                        int best = 0; float bd = 1e30f;   // the best of the quad's 8 symmetries (0: the identity)
                        for (int m = 0; m < 8; ++m) {
                                float sum = 0;
                                for (int k = 0; k < 4; ++k)
                                        sum += dist[k][m < 4 ? (k + m) % 4 : (m - k + 4) % 4];
                                if (sum < bd) { bd = sum; best = m; }
                        }
                        ok += best == 0;
                        off += cnt[f];
                }
                float maxdev; int bits = d.maxdev_bits.load(); std::memcpy(&maxdev, &bits, 4);
                sign = 2.0f * d.agree >= d.checked ? 1.0f : -1.0f;
                printf("probe (rate 4, %.2f s, warms the Ptex cache): uv corners match cage p0..p3 on %d of %d quads; "
                       "callback P vs rtcInterpolate max dev %g; Ng agrees with dP/du x dP/dv on %lld of %lld -> sign %+g\n",
                       now() - t0, ok, quads, maxdev, d.agree.load(), d.checked.load(), sign);
                rtcReleaseGeometry(g);
                rtcReleaseScene(s);
        }

        const V3 cen = (lo + hi) * 0.5f, ext = hi - lo;
        const float diag = std::sqrt(dot(ext, ext));
        const V3 eye = cen + V3{0.45f, 0.55f, 0.7f} * (diag * 1.1f);
        const V3 fwd = norm(cen - eye), right = norm(cross(fwd, {0, 1, 0})), up = cross(right, fwd);
        const V3 light = norm(right * 0.75f + up * 0.55f - fwd * 0.25f);
        const float tan_h = std::tan(40.0f * float(M_PI) / 180.0f * 0.5f);

        printf("\n%5s %5s %9s %10s %10s %9s | %s\n", "exag", "rate", "build_s", "mem_MiB", "peak_MiB", "Mpts", "per size: render_ms Mrays/s hit%");
        const long long base = mem_cur.load();
        for (float exag : exags) {
                for (float rate : rates) {
                        Disp d{tex, sign * exag, fwidth, nullptr, false};
                        mem_peak = mem_cur.load();
                        const long long before = mem_cur.load();
                        RTCScene s = rtcNewScene(dev);
                        RTCGeometry g = make_geom(dev, pos, cnt, idx, rate, &d);
                        rtcAttachGeometry(s, g);
                        rtcReleaseGeometry(g);
                        double t0 = now();
                        rtcCommitScene(s);
                        const double build = now() - t0;
                        if (rtcGetDeviceError(dev) != RTC_ERROR_NONE) { fprintf(stderr, "Embree error at commit\n"); return 1; }
                        printf("%5g %5g %9.3f %10.1f %10.1f %9.1f |", exag, rate, build, (mem_cur - before) / 1048576.0,
                               (mem_peak - before) / 1048576.0, d.points / 1e6);
                        for (size_t si = 0; si < ws.size(); ++si) {
                                const int W = int(ws[si]), H = int(hs[si]);
                                std::vector<float> shade(W * H, 0.0f), hu(W * H), hv(W * H);
                                std::vector<int> prim(W * H, -1);
                                double best = 1e30;
                                for (int r = 0; r < std::max(reps, 1); ++r) {
                                        double t1 = now();
                                        tbb::parallel_for(tbb::blocked_range2d<int>(0, H, 8, 0, W, 8), [&](const tbb::blocked_range2d<int> &b) {
                                                for (int y = b.rows().begin(); y < b.rows().end(); ++y)
                                                        for (int x = b.cols().begin(); x < b.cols().end(); ++x) {
                                                                const float px = (2.0f * (x + 0.5f) / W - 1.0f) * tan_h;
                                                                const float py = (1.0f - 2.0f * (y + 0.5f) / H) * tan_h * H / W;
                                                                const V3 dir = norm(fwd + right * px + up * py);
                                                                RTCRayHit rh;
                                                                rh.ray = {eye.x, eye.y, eye.z, 0.0f, dir.x, dir.y, dir.z, 0.0f, 1e30f, 0xffffffff, 0, 0};
                                                                rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
                                                                rtcIntersect1(s, &rh);
                                                                const int i = y * W + x;
                                                                if (rh.hit.geomID == RTC_INVALID_GEOMETRY_ID) { shade[i] = 0; prim[i] = -1; continue; }
                                                                V3 n = norm({rh.hit.Ng_x, rh.hit.Ng_y, rh.hit.Ng_z});
                                                                if (dot(n, dir) > 0) n = n * -1.0f;
                                                                shade[i] = 0.08f + 0.75f * std::max(dot(n, light), 0.0f);
                                                                prim[i] = rh.hit.primID; hu[i] = rh.hit.u; hv[i] = rh.hit.v;
                                                        }
                                        });
                                        best = std::min(best, now() - t1);
                                }
                                long long hits = 0;
                                for (int pi : prim) hits += pi >= 0;
                                printf(" %dx%d %8.1f ms %7.2f %5.1f%% |", W, H, best * 1e3, W * H / best / 1e6, 100.0 * hits / (W * H));
                                // Untimed: grey (sqrt encoded) and the colour Ptex as albedo (linear, x1.6, 1/2.2 encoded), as the GPU tool does.
                                std::vector<unsigned char> grey(3 * W * H), col(3 * W * H);
                                tbb::parallel_for(0, H, [&](int y) {
                                        PtexFilter *cf = filter_for(ctex);
                                        for (int x = 0; x < W; ++x) {
                                                const int i = y * W + x;
                                                const unsigned char gv = (unsigned char)(std::sqrt(std::clamp(shade[i], 0.0f, 1.0f)) * 255 + 0.5f);
                                                grey[3 * i] = grey[3 * i + 1] = grey[3 * i + 2] = gv;
                                                float rgb[3] = {0, 0, 0};
                                                if (prim[i] >= 0)
                                                        cf->eval(rgb, 0, 3, prim[i], std::clamp(hu[i], 0.0f, 1.0f), std::clamp(hv[i], 0.0f, 1.0f), 0, 0, 0, 0);
                                                for (int k = 0; k < 3; ++k) {
                                                        const float lin = std::pow(std::clamp(rgb[k], 0.0f, 1.0f), 2.2f) * shade[i] * 1.6f;
                                                        col[3 * i + k] = (unsigned char)(std::pow(std::clamp(lin, 0.0f, 1.0f), 1.0f / 2.2f) * 255 + 0.5f);
                                                }
                                        }
                                });
                                char name[128];
                                snprintf(name, sizeof name, "/embree_exag%g_rate%g_%dx%d", exag, rate, W, H);
                                write_png(out + name + ".png", grey, W, H);
                                write_png(out + name + "_color.png", col, W, H);
                        }
                        printf("\n");
                        fflush(stdout);
                        rtcReleaseScene(s);
                }
        }
        printf("Embree bytes still allocated after releasing all scenes: %lld (base %lld)\n", mem_cur.load(), base);
        rtcReleaseDevice(dev);
        tex->release(); ctex->release(); cache->release();
        return 0;
}
