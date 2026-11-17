// Displaced Catmull-Clark surfaces from a Ptex displacement file: its meta data holds the cage (Ptex's mesh convention),
// osd.cc turns the cage into Bezier patches, and each face's heights get min/max pyramids aligned with its quadtree.
#include <Ptexture.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

extern "C" void *osd_build(int, const float *, int, const int *, const int *, int, const int *, const float *, int,
                           const int *, const float *, int, int *);
extern "C" void osd_take(void *, float *, float *, int *);

namespace {

struct Disp {
        std::vector<float> cp, sq, cage;
        std::vector<int> face, hdr;            // hdr: per face its pyramid offset, log2 width, log2 height
        std::vector<unsigned char> pyr;        // per face: heights, then per coarser level the min and max cells
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
        d->sq.resize(static_cast<size_t>(np) * 3);
        d->face.resize(np);
        osd_take(h, d->cp.data(), d->sq.data(), d->face.data());

        for (int f = 0; f < nfc; ++f) {
                const Ptex::Res r = tex->getFaceInfo(f).res;
                const int ul = r.ulog2, vl = r.vlog2, w = 1 << ul, hh = 1 << vl;
                d->hdr.insert(d->hdr.end(), {static_cast<int>(d->pyr.size()), ul, vl});
                std::vector<unsigned char> lvl(static_cast<size_t>(w) * hh);
                float px[4];
                for (int y = 0; y < hh; ++y)
                        for (int x = 0; x < w; ++x) {
                                tex->getPixel(f, x, y, px, 0, 1);
                                lvl[static_cast<size_t>(y) * w + x] =
                                    static_cast<unsigned char>(std::min(std::max(px[0], 0.0f), 1.0f) * 255.0f + 0.5f);
                        }
                d->pyr.insert(d->pyr.end(), lvl.begin(), lvl.end());
                std::vector<unsigned char> lo = lvl, hi = lvl;
                int cw = w, ch = hh;
                while (cw > 1 || ch > 1) {   // cell (x, y) of the next level covers cells 2x..2x+1, 2y..2y+1
                        const int nw = std::max(cw / 2, 1), nh = std::max(ch / 2, 1);
                        std::vector<unsigned char> nlo(static_cast<size_t>(nw) * nh), nhi(nlo.size());
                        for (int y = 0; y < nh; ++y)
                                for (int x = 0; x < nw; ++x) {
                                        unsigned char a = 255, b = 0;
                                        for (int dy = 0; dy < 2; ++dy)
                                                for (int dx = 0; dx < 2; ++dx) {
                                                        const int sx = std::min(2 * x + dx, cw - 1), sy = std::min(2 * y + dy, ch - 1);
                                                        a = std::min(a, lo[static_cast<size_t>(sy) * cw + sx]);
                                                        b = std::max(b, hi[static_cast<size_t>(sy) * cw + sx]);
                                                }
                                        nlo[static_cast<size_t>(y) * nw + x] = a;
                                        nhi[static_cast<size_t>(y) * nw + x] = b;
                                }
                        d->pyr.insert(d->pyr.end(), nlo.begin(), nlo.end());
                        d->pyr.insert(d->pyr.end(), nhi.begin(), nhi.end());
                        lo.swap(nlo);
                        hi.swap(nhi);
                        cw = nw;
                        ch = nh;
                }
        }
        counts[0] = np;
        counts[1] = nfc;
        counts[2] = static_cast<long long>(d->pyr.size());
        return d.release();
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

}  // extern "C"
