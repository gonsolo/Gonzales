// Writes a displaced Catmull-Clark test asset as Moana writes them: a quad cage in the Ptex metadata (PtexVertPositions,
// PtexFaceVertCounts, PtexFaceVertIndices) and one uint8 height face per quad, sampled from a tileable displacement
// texture along the cage's global parametrization.
//
//   make_disp_ptex out.ptx torus|plane texture.png faceres_log2 cells_u cells_v tiles_u tiles_v [R r]

#include <OpenImageIO/imageio.h>
#include <Ptexture.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct Tex {
        int w = 0, h = 0;
        std::vector<float> v;
        float at(float s, float t) const {   // bilinear, wrapping
                float x = s * w - 0.5f, y = t * h - 0.5f;
                int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
                float fx = x - x0, fy = y - y0;
                auto p = [&](int i, int j) { return v[((j % h + h) % h) * w + (i % w + w) % w]; };
                return (p(x0, y0) * (1 - fx) + p(x0 + 1, y0) * fx) * (1 - fy) + (p(x0, y0 + 1) * (1 - fx) + p(x0 + 1, y0 + 1) * fx) * fy;
        }
};

static bool load(const char *path, Tex &t) {
        auto in = OIIO::ImageInput::open(path);
        if (!in)
                return false;
        const OIIO::ImageSpec &s = in->spec();
        t.w = s.width, t.h = s.height;
        std::vector<float> all(size_t(t.w) * t.h * s.nchannels);
        in->read_image(0, 0, 0, s.nchannels, OIIO::TypeDesc::FLOAT, all.data());
        t.v.resize(size_t(t.w) * t.h);
        float lo = 1e30f, hi = -1e30f;
        for (size_t i = 0; i < t.v.size(); ++i)
                t.v[i] = all[i * s.nchannels], lo = std::min(lo, t.v[i]), hi = std::max(hi, t.v[i]);
        for (float &x : t.v)   // to [0, 1]
                x = hi > lo ? (x - lo) / (hi - lo) : 0.f;
        return true;
}

int main(int argc, char **argv) {
        if (argc < 9) {
                std::fprintf(stderr, "usage: %s out.ptx torus|plane texture faceres_log2 cells_u cells_v tiles_u tiles_v [R r]\n", argv[0]);
                return 1;
        }
        std::string shape = argv[2];
        int lr = std::atoi(argv[4]), nu = std::atoi(argv[5]), nv = std::atoi(argv[6]);
        float tu = std::atof(argv[7]), tv = std::atof(argv[8]);
        float R = argc > 9 ? std::atof(argv[9]) : 10.f, r = argc > 10 ? std::atof(argv[10]) : 4.f;
        Tex tex;
        if (!load(argv[3], tex)) {
                std::fprintf(stderr, "cannot read %s\n", argv[3]);
                return 1;
        }
        bool wrap = shape == "torus";
        int vu = wrap ? nu : nu + 1, vv = wrap ? nv : nv + 1;
        std::vector<float> pos;
        for (int j = 0; j < vv; ++j)
                for (int i = 0; i < vu; ++i) {
                        float s = float(i) / nu, t = float(j) / nv;
                        if (wrap) {
                                float a = 2 * float(M_PI) * s, b = 2 * float(M_PI) * t;
                                pos.insert(pos.end(), {(R + r * std::cos(b)) * std::cos(a), (R + r * std::cos(b)) * std::sin(a), r * std::sin(b)});
                        } else {
                                pos.insert(pos.end(), {(s - 0.5f) * R, (t - 0.5f) * R * nv / nu, 0.f});
                        }
                }
        std::vector<int> idx, cnt;
        auto vid = [&](int i, int j) { return (j % vv) * vu + (i % vu); };
        for (int j = 0; j < nv; ++j)
                for (int i = 0; i < nu; ++i) {
                        idx.insert(idx.end(), {vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)});
                        cnt.push_back(4);
                }
        int nf = nu * nv;
        // Adjacency: edge e of a face runs from its vertex e to e + 1, as Ptex numbers them.
        std::map<std::pair<int, int>, std::pair<int, int>> edges;
        std::vector<std::array<int, 4>> af(nf), ae(nf);
        for (int f = 0; f < nf; ++f)
                for (int e = 0; e < 4; ++e) {
                        af[f][e] = -1, ae[f][e] = 0;
                        int a = idx[4 * f + e], b = idx[4 * f + (e + 1) % 4];
                        edges[{a, b}] = {f, e};
                }
        for (int f = 0; f < nf; ++f)
                for (int e = 0; e < 4; ++e) {
                        auto it = edges.find({idx[4 * f + (e + 1) % 4], idx[4 * f + e]});
                        if (it != edges.end())
                                af[f][e] = it->second.first, ae[f][e] = it->second.second;
                }
        Ptex::String err;
        PtexWriter *w = PtexWriter::open(argv[1], Ptex::mt_quad, Ptex::dt_uint8, 1, -1, nf, err, true);
        if (!w) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
        }
        int res = 1 << lr;
        std::vector<unsigned char> data(size_t(res) * res);
        for (int f = 0; f < nf; ++f) {
                int i = f % nu, j = f / nu;
                for (int y = 0; y < res; ++y)
                        for (int x = 0; x < res; ++x) {
                                float s = (i + (x + 0.5f) / res) / nu, t = (j + (y + 0.5f) / res) / nv;
                                float h = tex.at(s * tu, t * tv);
                                data[size_t(y) * res + x] = (unsigned char)std::lround(std::min(std::max(h, 0.f), 1.f) * 255.f);
                        }
                int adjf[4] = {af[f][0], af[f][1], af[f][2], af[f][3]}, adje[4] = {ae[f][0], ae[f][1], ae[f][2], ae[f][3]};
                w->writeFace(f, Ptex::FaceInfo(Ptex::Res(lr, lr), adjf, adje), data.data(), res);
        }
        w->writeMeta("PtexVertPositions", pos.data(), (int)pos.size());
        w->writeMeta("PtexFaceVertCounts", cnt.data(), (int)cnt.size());
        w->writeMeta("PtexFaceVertIndices", idx.data(), (int)idx.size());
        bool ok = w->close(err);
        w->release();
        if (!ok) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
        }
        std::printf("%s: %d faces of %d x %d texels, %zu cage vertices\n", argv[1], nf, res, res, pos.size() / 3);
        return 0;
}
