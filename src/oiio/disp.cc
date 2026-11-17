// Displaced Catmull-Clark surfaces from a Ptex displacement file: its meta data holds the cage (Ptex's mesh convention),
// osd.cc turns the cage into Bezier patches, and each face's heights get a quadtree of planes with residuals.
#include <Ptexture.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

extern "C" void *osd_build(int, const float *, int, const int *, const int *, int, const int *, const float *, int,
                           const int *, const float *, int, int *);
extern "C" void osd_take_exact(void *, float *, float *, unsigned char *, float *, int *);
extern "C" void *ptex_open(const char *);
extern "C" void ptex_close(void *);
extern "C" void ptex_read_face(void *, int, int, int, unsigned char *);

namespace {

struct Disp {
        std::vector<float> cp, fm, sq, cage;   // cp: Gregory patches with Fp inside, fm: their Fm points (12 floats a patch)
        std::vector<unsigned char> greg;
        std::vector<int> face, hdr;            // hdr: per face its node offset, log2 width, log2 height
        std::vector<unsigned char> pyr;        // per face, levels 2 up: cells of (centre, half u slope / 2, half v slope / 2, residual)
        std::vector<int> chain_off;            // per face: its bordered height chain in `chain`
        std::vector<unsigned char> chain;      // per face, full res to 1x1: (w + 2) x (h + 2) heights, borders from the neighbours
        float box[6] = {1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f};
};

}  // namespace

extern "C" {

// The cage and heights of `ptx` as Bezier patches and pyramids; counts: patches, faces, pyramid bytes. Null on failure.
void *disp_open(const char *ptx, int isolation, long long *counts) {
        Ptex::String err;
        PtexPtr<PtexTexture> tex(PtexTexture::open(ptx, err));
        if (!tex)
                return nullptr;
        PtexPtr<PtexMetaData> meta(tex->getMetaData());
        const float *pos = nullptr;
        const int *vcount = nullptr, *vidx = nullptr;
        int npos = 0, nfc = 0, nidx = 0;
        meta->getValue("PtexVertPositions", pos, npos);
        meta->getValue("PtexFaceVertCounts", vcount, nfc);
        meta->getValue("PtexFaceVertIndices", vidx, nidx);
        if (!pos || !vcount || !vidx || nfc != tex->numFaces())
                return nullptr;
        auto d = std::make_unique<Disp>();
        d->cage.assign(pos, pos + npos);
        for (int i = 0; i < npos; ++i) {
                d->box[i % 3] = std::min(d->box[i % 3], pos[i]);
                d->box[3 + i % 3] = std::max(d->box[3 + i % 3], pos[i]);
        }
        int np = 0;
        void *h = osd_build(npos / 3, pos, nfc, vcount, vidx, 0, nullptr, nullptr, 0, nullptr, nullptr, isolation, &np);
        if (!h)
                return nullptr;
        d->cp.resize(static_cast<size_t>(np) * 48);
        d->fm.resize(static_cast<size_t>(np) * 12);
        d->greg.resize(np);
        d->sq.resize(static_cast<size_t>(np) * 3);
        d->face.resize(np);
        osd_take_exact(h, d->cp.data(), d->fm.data(), d->greg.data(), d->sq.data(), d->face.data());

        for (int f = 0; f < nfc; ++f) {   // per face: plane + residual nodes from 4 x 4-texel cells up (Xu, Zhou, Budge 2026)
                const Ptex::Res r = tex->getFaceInfo(f).res;
                const int ul = r.ulog2, vl = r.vlog2, w = 1 << ul, hh = 1 << vl;
                d->hdr.insert(d->hdr.end(), {static_cast<int>(d->pyr.size()), ul, vl});
                std::vector<float> lvl(static_cast<size_t>(w) * hh);
                float px[4];
                for (int y = 0; y < hh; ++y)
                        for (int x = 0; x < w; ++x) {
                                tex->getPixel(f, x, y, px, 0, 1);
                                lvl[static_cast<size_t>(y) * w + x] = std::round(std::min(std::max(px[0], 0.0f), 1.0f) * 255.0f);
                        }
                auto height = [&](float X, float Y) {   // the bilinear height at texel position (X, Y), clamped like face_height
                        const float x = std::min(std::max(X - 0.5f, 0.0f), float(w - 1)), y = std::min(std::max(Y - 0.5f, 0.0f), float(hh - 1));
                        const int x0 = std::min(int(x), w - 1), y0 = std::min(int(y), hh - 1), x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, hh - 1);
                        const float fx = x - x0, fy = y - y0;
                        const float a = lvl[y0 * w + x0] * (1 - fx) + lvl[y0 * w + x1] * fx, b = lvl[y1 * w + x0] * (1 - fx) + lvl[y1 * w + x1] * fx;
                        return a * (1 - fy) + b * fy;
                };
                for (int k = 2; (w >> (k - 1)) > 1 || (hh >> (k - 1)) > 1; ++k) {
                        const int cw = std::max(w >> k, 1), ch = std::max(hh >> k, 1), tw = w / cw, th = hh / ch;
                        for (int cy = 0; cy < ch; ++cy)
                                for (int cx = 0; cx < cw; ++cx) {
                                        const float X0 = cx * tw, X1 = X0 + tw, Y0 = cy * th, Y1 = Y0 + th, xc = X0 + 0.5f * tw, yc = Y0 + 0.5f * th;
                                        double sh = 0, sx = 0, sy = 0, sxx = 0, syy = 0;   // least squares over the cell's texel centres
                                        for (int y = int(Y0); y < int(Y1); ++y)
                                                for (int x = int(X0); x < int(X1); ++x) {
                                                        const double h = lvl[y * w + x], ex = (x + 0.5 - xc) / (0.5 * tw), ey = (y + 0.5 - yc) / (0.5 * th);
                                                        sh += h; sx += h * ex; sy += h * ey; sxx += ex * ex; syy += ey * ey;
                                                }
                                        const double nn = double(tw) * th;
                                        const int qa = std::min(std::max(int(std::lround(sh / nn)), 0), 255);
                                        const int qx = sxx > 0 ? std::min(std::max(int(std::lround(sx / sxx / 2)), -127), 127) : 0;
                                        const int qy = syy > 0 ? std::min(std::max(int(std::lround(sy / syy / 2)), -127), 127) : 0;
                                        // Residual: the piecewise bilinear surface minus the plane peaks on the region's grid of
                                        // texel-centre lines and its own edges, so these points bound it exactly.
                                        std::vector<float> xs = {X0}, ys = {Y0};
                                        for (int x = int(X0); x < int(X1); ++x) xs.push_back(x + 0.5f);
                                        for (int y = int(Y0); y < int(Y1); ++y) ys.push_back(y + 0.5f);
                                        xs.push_back(X1); ys.push_back(Y1);
                                        double dev = 0, lo = 1e30, hi = -1e30;
                                        for (float Y : ys)
                                                for (float X : xs) {
                                                        const double hgt = height(X, Y);
                                                        const double pl = qa + 2.0 * qx * (X - xc) / (0.5 * tw) + 2.0 * qy * (Y - yc) / (0.5 * th);
                                                        dev = std::max(dev, std::abs(hgt - pl));
                                                        lo = std::min(lo, hgt); hi = std::max(hi, hgt);
                                                }
                                        int na = qa, nx = qx, ny = qy, nr = int(std::ceil(dev + 1e-3));
                                        if (nr > 255) {   // a flat node over the cell's range instead
                                                na = int(std::floor((lo + hi) / 2)); nx = ny = 0; nr = int(std::ceil(std::max(hi - na, na - lo) + 1e-3));
                                        }
                                        d->pyr.insert(d->pyr.end(), {static_cast<unsigned char>(na), static_cast<unsigned char>(static_cast<signed char>(nx)),
                                                                     static_cast<unsigned char>(static_cast<signed char>(ny)), static_cast<unsigned char>(nr)});
                                }
                }
        }
        void *pt = ptex_open(ptx);   // oiio.cc: Ptex chains with the neighbours' texels in the borders (rotation-aware)
        std::vector<unsigned char> rgb;
        for (int f = 0; pt && f < nfc; ++f) {
                const Ptex::Res r = tex->getFaceInfo(f).res;
                size_t texels = 0;
                for (int ul = r.ulog2, vl = r.vlog2;; ul = std::max(ul - 1, 0), vl = std::max(vl - 1, 0)) {
                        texels += static_cast<size_t>((1 << ul) + 2) * ((1 << vl) + 2);
                        if (ul == 0 && vl == 0)
                                break;
                }
                rgb.resize(texels * 3);
                ptex_read_face(pt, f, r.ulog2, r.vlog2, rgb.data());
                d->chain_off.push_back(static_cast<int>(d->chain.size()));
                for (size_t i = 0; i < texels; ++i)
                        d->chain.push_back(rgb[i * 3]);   // uint8 data: channel 0 is the height byte
        }
        if (pt)
                ptex_close(pt);
        counts[0] = np;
        counts[1] = nfc;
        counts[2] = static_cast<long long>(d->pyr.size());
        counts[3] = static_cast<long long>(d->chain.size());
        return d.release();
}

// The Gregory data (fm 12 floats, greg 1 byte a patch); call before disp_take, which frees everything.
void disp_take_gregory(void *h, float *fm, unsigned char *greg) {
        auto *d = static_cast<Disp *>(h);
        std::memcpy(fm, d->fm.data(), d->fm.size() * sizeof(float));
        std::memcpy(greg, d->greg.data(), d->greg.size());
}

// The bordered height chains (offsets: 1 int a face); call before disp_take, which frees everything.
void disp_take_chains(void *h, int *off, unsigned char *chain) {
        auto *d = static_cast<Disp *>(h);
        std::memcpy(off, d->chain_off.data(), d->chain_off.size() * sizeof(int));
        std::memcpy(chain, d->chain.data(), d->chain.size());
}

// Copy everything out (cp 48 floats, sq 3 floats, face 1 int a patch; hdr 3 ints a face; box 6 floats) and free it.
void disp_take(void *h, float *cp, float *sq, int *face, int *hdr, unsigned char *pyr, float *box) {
        auto *d = static_cast<Disp *>(h);
        std::memcpy(cp, d->cp.data(), d->cp.size() * sizeof(float));
        std::memcpy(sq, d->sq.data(), d->sq.size() * sizeof(float));
        std::memcpy(face, d->face.data(), d->face.size() * sizeof(int));
        std::memcpy(hdr, d->hdr.data(), d->hdr.size() * sizeof(int));
        std::memcpy(pyr, d->pyr.data(), d->pyr.size());
        std::memcpy(box, d->box, sizeof d->box);
        delete d;
}

// The cage of `ptx`: with pos null only sizes (vertices, faces, indices), else the positions, face sizes and indices too.
int disp_cage(const char *ptx, float *pos, int *counts, int *idx, int *sizes) {
        Ptex::String err;
        PtexPtr<PtexTexture> tex(PtexTexture::open(ptx, err));
        if (!tex)
                return 0;
        PtexPtr<PtexMetaData> meta(tex->getMetaData());
        const float *p = nullptr;
        const int *c = nullptr, *v = nullptr;
        int np = 0, nc = 0, nv = 0;
        meta->getValue("PtexVertPositions", p, np);
        meta->getValue("PtexFaceVertCounts", c, nc);
        meta->getValue("PtexFaceVertIndices", v, nv);
        if (!p || !c || !v)
                return 0;
        sizes[0] = np / 3; sizes[1] = nc; sizes[2] = nv;
        if (pos) {
                std::memcpy(pos, p, np * sizeof(float));
                std::memcpy(counts, c, nc * sizeof(int));
                std::memcpy(idx, v, nv * sizeof(int));
        }
        return 1;
}

// The colour Ptex as bordered RGB8 chains per face (oiio.cc's ptex_read_face), for GPU lookups; total bytes in *n, null
// if it cannot be read. Its faces must match the displacement's.
void *disp_color_chains(const char *ptx, long long *n) {
        Ptex::String err;
        PtexPtr<PtexTexture> tex(PtexTexture::open(ptx, err));
        void *pt = tex ? ptex_open(ptx) : nullptr;
        if (!pt)
                return nullptr;
        auto *c = new Disp;
        for (int f = 0; f < tex->numFaces(); ++f) {
                const Ptex::Res r = tex->getFaceInfo(f).res;
                size_t texels = 0;
                for (int ul = r.ulog2, vl = r.vlog2;; ul = std::max(ul - 1, 0), vl = std::max(vl - 1, 0)) {
                        texels += static_cast<size_t>((1 << ul) + 2) * ((1 << vl) + 2);
                        if (ul == 0 && vl == 0)
                                break;
                }
                c->chain_off.push_back(static_cast<int>(c->chain.size()));
                const size_t at = c->chain.size();
                c->chain.resize(at + texels * 3);
                ptex_read_face(pt, f, r.ulog2, r.vlog2, c->chain.data() + at);
        }
        ptex_close(pt);
        *n = static_cast<long long>(c->chain.size());
        return c;
}

// Copy the colour chains out (per face its offset in RGB texels, then the bytes) and free them.
void disp_color_chains_take(void *h, int *off, unsigned char *rgb) {
        auto *c = static_cast<Disp *>(h);
        for (size_t f = 0; f < c->chain_off.size(); ++f)
                off[f] = c->chain_off[f] / 3;
        std::memcpy(rgb, c->chain.data(), c->chain.size());
        delete c;
}

static PtexCache *color_cache = nullptr;
static PtexTexture *color_tex = nullptr;

// A colour Ptex for disp_color; 0 if it cannot be read.
int disp_color_open(const char *ptx) {
        Ptex::String err;
        if (!color_cache)
                color_cache = PtexCache::create(16, 1 << 30);
        color_tex = color_cache->get(ptx, err);
        return color_tex ? 1 : 0;
}

// Linear RGB of the colour Ptex at face f, (u, v), filtered bilinearly over a width w in face coordinates.
void disp_color(int f, float u, float v, float w, float *rgb) {
        thread_local PtexFilter *filter = nullptr;
        if (!filter)
                filter = PtexFilter::getFilter(color_tex, PtexFilter::Options(PtexFilter::f_bilinear));
        float c[3] = {0, 0, 0};
        filter->eval(c, 0, 3, f, u, v, w, 0, 0, w);
        for (int i = 0; i < 3; ++i)
                rgb[i] = std::pow(std::min(std::max(c[i], 0.0f), 1.0f), 2.2f);   // 8-bit colour Ptex is gamma encoded
}

}  // extern "C"
