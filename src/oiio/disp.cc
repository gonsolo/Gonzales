// Displaced Catmull-Clark surfaces from a Ptex displacement file: its meta data holds the cage (Ptex's mesh convention),
// osd.cc turns the cage into Bezier patches, and each face's heights get a quadtree of planes with residuals.
#include <Ptexture.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>
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
        std::vector<unsigned char> drop;       // per face: the finest levels left out for the view
        std::vector<int> face, hdr;            // hdr: per face its node offset, log2 width, log2 height
        std::vector<unsigned char> pyr;        // per face, levels 2 up: cells of (centre, half u slope / 2, half v slope / 2, residual, min, max)
        std::vector<int> chain_off;            // per face: its bordered height chain in `chain`
        std::vector<unsigned char> chain;      // per face, full res to 1x1: (w + 2) x (h + 2) heights, borders from the neighbours
        float box[6] = {1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f};
};

struct Cage {
        std::vector<float> pos;
        std::vector<int> vcount, vidx;
};

struct DispCache {
        PtexTexture *tex = nullptr;
        Cage cage;
        std::unique_ptr<Disp> base;
        std::vector<size_t> chain_off, pyr_first, pyr_off;   // pyr_off[pyr_first[f] + k]: face f's nodes at drop k
        std::vector<unsigned char> chain, pyr;
        ~DispCache() {
                if (tex)
                        tex->release();
        }
};

// The cage from the meta data, posed by `corners` (12 floats a face) if given.
bool read_cage(PtexTexture *tex, const float *corners, Cage &c) {
        PtexPtr<PtexMetaData> meta(tex->getMetaData());
        const float *pos = nullptr;
        const int *vcount = nullptr, *vidx = nullptr;
        int npos = 0, nfc = 0, nidx = 0;
        meta->getValue("PtexVertPositions", pos, npos);
        meta->getValue("PtexFaceVertCounts", vcount, nfc);
        meta->getValue("PtexFaceVertIndices", vidx, nidx);
        if (!pos || !vcount || !vidx || nfc != tex->numFaces())
                return false;
        c.pos.assign(pos, pos + npos), c.vcount.assign(vcount, vcount + nfc), c.vidx.assign(vidx, vidx + nidx);
        if (corners)
                for (int f = 0, at = 0; f < nfc; at += vcount[f], ++f) {
                        if (vcount[f] != 4)
                                return false;
                        for (int k = 0; k < 4; ++k)
                                for (int i = 0; i < 3; ++i)
                                        c.pos[3 * vidx[at + k] + i] = corners[12 * f + 3 * k + i];
                }
        return true;
}

// The cage, its box and its patches: everything but the heights.
std::unique_ptr<Disp> disp_base(const Cage &c, int isolation) {
        const float *pos = c.pos.data();
        const int npos = int(c.pos.size()), nfc = int(c.vcount.size());
        auto d = std::make_unique<Disp>();
        d->cage = c.pos;
        for (int i = 0; i < npos; ++i) {
                d->box[i % 3] = std::min(d->box[i % 3], pos[i]);
                d->box[3 + i % 3] = std::max(d->box[3 + i % 3], pos[i]);
        }
        int np = 0;
        void *h = osd_build(npos / 3, pos, nfc, c.vcount.data(), c.vidx.data(), 0, nullptr, nullptr, 0, nullptr, nullptr, isolation, &np);
        if (!h)
                return nullptr;
        d->cp.resize(static_cast<size_t>(np) * 48);
        d->fm.resize(static_cast<size_t>(np) * 12);
        d->greg.resize(np);
        d->sq.resize(static_cast<size_t>(np) * 3);
        d->face.resize(np);
        osd_take_exact(h, d->cp.data(), d->fm.data(), d->greg.data(), d->sq.data(), d->face.data());
        return d;
}

// The finest levels of face f (vertices from `at` on) left out for `view`.
int face_drop(const float *pos, const int *vidx, int at, int nv, Ptex::Res r, const float *view) {
        int drop = 0;
        if (!view || view[4] <= 0)
                return 0;
        float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f}, edge = 0;
        for (int k = 0; k < nv; ++k) {
                const float *a = pos + 3 * vidx[at + k], *b = pos + 3 * vidx[at + (k + 1) % nv];
                edge = std::max(edge, std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])));
                for (int i = 0; i < 3; ++i)
                        lo[i] = std::min(lo[i], a[i]), hi[i] = std::max(hi[i], a[i]);
        }
        float d2 = 0;   // to the face's box
        for (int i = 0; i < 3; ++i) {
                const float e = std::max(std::max(lo[i] - view[i], view[i] - hi[i]), 0.0f);
                d2 += e * e;
        }
        float pixel = view[3] * std::max(std::sqrt(d2), 1e-3f * edge);
        if (view[8] >= -1) {   // outside the view cone (axis view[5..7], cosine view[8]) a face counts view[9] times farther
                float c[3], r = 0;
                for (int i = 0; i < 3; ++i)
                        c[i] = 0.5f * (lo[i] + hi[i]) - view[i], r += 0.25f * (hi[i] - lo[i]) * (hi[i] - lo[i]);
                r = std::sqrt(r);
                const float d = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
                if (d > r) {
                        const float sa = r / d, ca = std::sqrt(std::max(1 - sa * sa, 0.0f)), sh = std::sqrt(std::max(1 - view[8] * view[8], 0.0f));
                        if ((c[0] * view[5] + c[1] * view[6] + c[2] * view[7]) / d < view[8] * ca - sh * sa)
                                pixel *= view[9];
                }
        }
        float texel = edge / float(1 << std::max(r.ulog2, r.vlog2));
        while (drop < std::max(r.ulog2, r.vlog2) && 2 * texel <= view[4] * pixel)
                texel *= 2, ++drop;
        return drop;
}

// Face f's plane + residual nodes from 4 x 4-texel cells up (Xu, Zhou, Budge 2026) at (ul, vl), appended to pyr.
void face_pyramid(PtexTexture *tex, int f, int ul, int vl, std::vector<unsigned char> &pyr) {
        const int w = 1 << ul, hh = 1 << vl;
        std::vector<float> lvl(static_cast<size_t>(w) * hh);
        float px[4];
        for (int y = 0; y < hh; ++y)
                for (int x = 0; x < w; ++x) {
                        tex->getPixel(f, x, y, px, 0, 1, Ptex::Res(ul, vl));
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
                                const int mn = std::min(std::max(int(std::floor(lo)), 0), 255), mx = std::min(std::max(int(std::ceil(hi)), 0), 255);
                                pyr.insert(pyr.end(), {static_cast<unsigned char>(na), static_cast<unsigned char>(static_cast<signed char>(nx)),
                                                             static_cast<unsigned char>(static_cast<signed char>(ny)), static_cast<unsigned char>(nr),
                                                             static_cast<unsigned char>(mn), static_cast<unsigned char>(mx)});
                        }
        }
}

size_t chain_texels(int ul, int vl) {
        size_t texels = 0;
        for (;; ul = std::max(ul - 1, 0), vl = std::max(vl - 1, 0)) {
                texels += static_cast<size_t>((1 << ul) + 2) * ((1 << vl) + 2);
                if (ul == 0 && vl == 0)
                        return texels;
        }
}

// Face f's bordered height chain from (ul, vl) to 1 x 1, appended to chain.
void face_chain(void *pt, int f, int ul, int vl, std::vector<unsigned char> &chain) {
        const size_t texels = chain_texels(ul, vl);
        std::vector<unsigned char> rgb(texels * 3);
        ptex_read_face(pt, f, ul, vl, rgb.data());
        for (size_t i = 0; i < texels; ++i)
                chain.push_back(rgb[i * 3]);   // uint8 data: channel 0 is the height byte
}

// Border texels at vertices and between faces of different resolution, from the faces' kept levels.
void fix_borders(Disp *d, PtexTexture *tex, const std::vector<int> &rul, const std::vector<int> &rvl, int nfc) {
        // Each face's diagonal border texel such that its four corner texels average to the vertex's height, the mean of the
        // corner texels of all faces around it: then all of them agree at the vertex and on the half texel of edge beside it.
        auto texel = [&](int f, int x, int y) -> unsigned char & {
                const int w = 1 << rul[f];
                return d->chain[d->chain_off[f] + static_cast<size_t>(y + 1) * (w + 2) + x + 1];
        };
        auto corner = [&](int f, int c, int ox, int oy) {   // (ox, oy): 0 for the texel inside, 1 one texel out
                const int w = 1 << rul[f], h = 1 << rvl[f];
                const int x = (c == 1 || c == 2) ? w - 1 + ox : -ox, y = c >= 2 ? h - 1 + oy : -oy;
                return std::pair<int, int>(x, y);
        };
        auto vertex_fix = [&] {
                for (int f = 0; !d->chain_off.empty() && f < nfc; ++f)
                        for (int c = 0; c < 4; ++c) {
                                int sum = 0, n = 0, g = f, k = c;
                                bool closed = false;
                                for (int step = 0; step < 32; ++step) {   // around the vertex: across edge k, which starts at corner k
                                        const Ptex::FaceInfo &gi = tex->getFaceInfo(g);
                                        if (gi.isSubface())
                                                break;
                                        const auto p = corner(g, k, 0, 0);
                                        sum += texel(g, p.first, p.second); ++n;
                                        const int af = gi.adjface(k), ae = gi.adjedge(k);
                                        if (af < 0)
                                                break;
                                        g = af; k = (ae + 1) % 4;
                                        if (g == f && k == c) {
                                                closed = true;
                                                break;
                                        }
                                }
                                if (!closed)
                                        continue;
                                const auto p = corner(f, c, 0, 0), a = corner(f, c, 1, 0), b = corner(f, c, 0, 1), q = corner(f, c, 1, 1);
                                const int dv = int(std::lround(4.0 * sum / n - texel(f, p.first, p.second) - texel(f, a.first, a.second) - texel(f, b.first, b.second)));
                                texel(f, q.first, q.second) = static_cast<unsigned char>(std::min(std::max(dv, 0), 255));
                        }
        };
        vertex_fix();
        // Along an edge between faces of different resolution, the finer face's border texels are set so that its edge heights
        // (the mean of edge and border texel) follow the coarser face's edge heights, which the surfaces meet at.
        auto edge_n = [&](int f, int e) { return 1 << (e % 2 == 0 ? rul[f] : rvl[f]); };
        auto edge_at = [&](int f, int e, int j, int out) -> unsigned char & {   // texel j along edge e (counter-clockwise); out: 1 for the border
                const int w = 1 << rul[f], h = 1 << rvl[f];
                if (e == 0) return texel(f, j, out ? -1 : 0);
                if (e == 1) return texel(f, out ? w : w - 1, j);
                if (e == 2) return texel(f, w - 1 - j, out ? h : h - 1);
                return texel(f, out ? -1 : 0, h - 1 - j);
        };
        int mixed_edges = 0, clamped = 0, worst = 0;
        for (int f = 0; !d->chain_off.empty() && f < nfc; ++f) {
                const Ptex::FaceInfo &fi = tex->getFaceInfo(f);
                if (fi.isSubface())
                        continue;
                for (int e = 0; e < 4; ++e) {
                        const int a = fi.adjface(e), ae = fi.adjedge(e);
                        if (a < 0 || tex->getFaceInfo(a).isSubface())
                                continue;
                        const int nf = edge_n(f, e), na = edge_n(a, ae);
                        if (nf <= na || (getenv("DISP_EDGE") && getenv("DISP_EDGE")[0] == '0'))
                                continue;
                        ++mixed_edges;
                        for (int j = 0; j < nf; ++j) {   // the neighbour's edge runs the other way
                                const float x = std::min(std::max((1.0f - (j + 0.5f) / nf) * na - 0.5f, -0.5f), na - 0.5f);   // -1, na: the corners
                                const int i0 = std::min(int(std::floor(x)), na - 1), i1 = i0 + 1;
                                const float t = x - i0;
                                const float e0 = 0.5f * (edge_at(a, ae, i0, 0) + edge_at(a, ae, i0, 1)), e1 = 0.5f * (edge_at(a, ae, i1, 0) + edge_at(a, ae, i1, 1));
                                const float target = e0 + (e1 - e0) * t;
                                const int want = int(std::lround(2.0f * target - edge_at(f, e, j, 0)));
                                clamped += want < 0 || want > 255; worst = std::max(worst, std::max(-want, want - 255));
                                edge_at(f, e, j, 1) = static_cast<unsigned char>(std::min(std::max(want, 0), 255));
                        }
                }
        }
        vertex_fix();   // again, with the finer faces' new border texels
        if (mixed_edges > 0 && getenv("DISP_VERBOSE"))
                std::printf("edges between faces of different resolution: %d, %d border texels clamped (worst by %d)\n", mixed_edges, clamped, worst);
}

}  // namespace

extern "C" {

// The cage's box (6 floats) of `ptx` from its meta data; 0 on failure.
int disp_bounds(const char *ptx, float *box) {
        Ptex::String err;
        PtexPtr<PtexTexture> tex(PtexTexture::open(ptx, err));
        if (!tex)
                return 0;
        PtexPtr<PtexMetaData> meta(tex->getMetaData());
        const float *pos = nullptr;
        int npos = 0;
        meta->getValue("PtexVertPositions", pos, npos);
        if (!pos || npos < 3)
                return 0;
        for (int i = 0; i < 3; ++i)
                box[i] = 1e30f, box[3 + i] = -1e30f;
        for (int i = 0; i < npos; ++i)
                box[i % 3] = std::min(box[i % 3], pos[i]), box[3 + i % 3] = std::max(box[3 + i % 3], pos[i]);
        return 1;
}

// The cage and heights of `ptx` as Bezier patches and pyramids; counts: patches, faces, pyramid bytes. Null on failure.
// view (eye x, y, z, pixel angle, keep; keep <= 0: off; then the view cone's axis x, y, z, cosine (< -1: none) and the
// factor faces outside it count as farther) or null: a face keeps its finer levels only while its texels at its nearest point
// to the eye are at least `keep` pixels; the coarser level then stands in for the face's resolution everywhere.
// corners or null: the mesh's own positions, 4 a Ptex face in the face's (u, v) order (0,0) (1,0) (1,1) (0,1), for meshes
// posed differently from the cage stored in the Ptex (one Ptex shared by a rig's poses).
void *disp_open(const char *ptx, int isolation, long long *counts, const float *view, const float *corners) {
        Ptex::String err;
        PtexPtr<PtexTexture> tex(PtexTexture::open(ptx, err));
        Cage c;
        if (!tex || !read_cage(tex.get(), corners, c))
                return nullptr;
        const int nfc = int(c.vcount.size());
        auto d = disp_base(c, isolation);
        if (!d)
                return nullptr;
        std::vector<int> rul(nfc), rvl(nfc);   // the resolution kept per face
        long long full = 0, kept = 0;
        for (int f = 0, at = 0; f < nfc; at += c.vcount[f], ++f) {
                const Ptex::Res r = tex->getFaceInfo(f).res;
                const int drop = face_drop(c.pos.data(), c.vidx.data(), at, c.vcount[f], r, view);
                rul[f] = std::max(r.ulog2 - drop, 0), rvl[f] = std::max(r.vlog2 - drop, 0);
                d->drop.push_back(static_cast<unsigned char>(std::max(r.ulog2, r.vlog2) - std::max(rul[f], rvl[f])));
                full += (1LL << r.ulog2) << r.vlog2, kept += (1LL << rul[f]) << rvl[f];
        }
        for (int f = 0; f < nfc; ++f) {
                d->hdr.insert(d->hdr.end(), {static_cast<int>(d->pyr.size()), rul[f], rvl[f]});
                face_pyramid(tex.get(), f, rul[f], rvl[f], d->pyr);
        }
        void *pt = ptex_open(ptx);   // oiio.cc: Ptex chains with the neighbours' texels in the borders (rotation-aware)
        for (int f = 0; pt && f < nfc; ++f) {
                d->chain_off.push_back(static_cast<int>(d->chain.size()));
                face_chain(pt, f, rul[f], rvl[f], d->chain);
        }
        if (pt)
                ptex_close(pt);
        fix_borders(d.get(), tex.get(), rul, rvl, nfc);
        counts[0] = static_cast<long long>(d->face.size());
        counts[1] = nfc;
        counts[2] = static_cast<long long>(d->pyr.size());
        counts[3] = static_cast<long long>(d->chain.size());
        if (view && view[4] > 0)
                std::printf("height levels for the view: %lld of %lld texels kept (%.1f%%)\n", kept, full, 100.0 * kept / std::max(full, 1LL));
        return d.release();
}


// Streaming: what disp_open reads for any view, once (patches, full chains, nodes at every level), and per view a Disp
// cut from it, equal to disp_open's.
void *disp_cache_open(const char *ptx, int isolation, const float *corners) {
        Ptex::String err;
        auto C = std::make_unique<DispCache>();
        C->tex = PtexTexture::open(ptx, err);
        if (!C->tex || !read_cage(C->tex, corners, C->cage))
                return nullptr;
        C->base = disp_base(C->cage, isolation);
        if (!C->base)
                return nullptr;
        const int nfc = int(C->cage.vcount.size());
        std::vector<std::vector<unsigned char>> fchain(nfc), fpyr(nfc);
        std::vector<std::vector<size_t>> fpoff(nfc);
        std::atomic<int> next{0};
        std::atomic<bool> ok{true};
        auto work = [&] {
                Ptex::String e;
                PtexPtr<PtexTexture> tex(PtexTexture::open(ptx, e));
                void *pt = ptex_open(ptx);
                if (!tex || !pt) {
                        ok = false;
                        if (pt)
                                ptex_close(pt);
                        return;
                }
                for (int f; (f = next++) < nfc;) {
                        const Ptex::Res r = tex->getFaceInfo(f).res;
                        face_chain(pt, f, r.ulog2, r.vlog2, fchain[f]);
                        for (int k = 0; k <= std::max(r.ulog2, r.vlog2); ++k) {
                                fpoff[f].push_back(fpyr[f].size());
                                face_pyramid(tex.get(), f, std::max(r.ulog2 - k, 0), std::max(r.vlog2 - k, 0), fpyr[f]);
                        }
                        fpoff[f].push_back(fpyr[f].size());
                }
                ptex_close(pt);
        };
        std::vector<std::thread> pool;
        for (unsigned i = 0; i < std::max(1u, std::thread::hardware_concurrency()); ++i)
                pool.emplace_back(work);
        for (auto &t : pool)
                t.join();
        if (!ok)
                return nullptr;
        for (int f = 0; f < nfc; ++f) {
                C->chain_off.push_back(C->chain.size());
                C->chain.insert(C->chain.end(), fchain[f].begin(), fchain[f].end());
                C->pyr_first.push_back(C->pyr_off.size());
                for (size_t o : fpoff[f])
                        C->pyr_off.push_back(C->pyr.size() + o);
                C->pyr.insert(C->pyr.end(), fpyr[f].begin(), fpyr[f].end());
                std::vector<unsigned char>().swap(fchain[f]), std::vector<unsigned char>().swap(fpyr[f]);
        }
        return C.release();
}

long long disp_cache_bytes(void *h) {
        auto *C = static_cast<DispCache *>(h);
        const Disp &b = *C->base;
        return static_cast<long long>(C->chain.size() + C->pyr.size() + 8 * (C->chain_off.size() + C->pyr_first.size() + C->pyr_off.size()) +
                                      4 * (b.cp.size() + b.fm.size() + b.sq.size() + b.face.size() + b.cage.size()) + b.greg.size());
}

void disp_cache_close(void *h) { delete static_cast<DispCache *>(h); }

// Per face its four neighbours (-1: none), from the Ptex adjacency.
void disp_cache_adj(void *h, int *adj) {
        auto *C = static_cast<DispCache *>(h);
        for (int f = 0; f < C->tex->numFaces(); ++f)
                for (int e = 0; e < 4; ++e)
                        adj[4 * f + e] = C->tex->getFaceInfo(f).adjface(e);
}

// The Disp disp_open(ptx, isolation, counts, view, corners) returns, from its cache; free it with disp_take.
void *disp_cache_view(void *h, long long *counts, const float *view) {
        auto *C = static_cast<DispCache *>(h);
        const Cage &c = C->cage;
        const int nfc = int(c.vcount.size());
        auto d = std::make_unique<Disp>(*C->base);
        std::vector<int> rul(nfc), rvl(nfc);
        long long full = 0, kept = 0;
        for (int f = 0, at = 0; f < nfc; at += c.vcount[f], ++f) {
                const Ptex::Res r = C->tex->getFaceInfo(f).res;
                const int drop = face_drop(c.pos.data(), c.vidx.data(), at, c.vcount[f], r, view);
                rul[f] = std::max(r.ulog2 - drop, 0), rvl[f] = std::max(r.vlog2 - drop, 0);
                d->drop.push_back(static_cast<unsigned char>(std::max(r.ulog2, r.vlog2) - std::max(rul[f], rvl[f])));
                full += (1LL << r.ulog2) << r.vlog2, kept += (1LL << rul[f]) << rvl[f];
                d->hdr.insert(d->hdr.end(), {static_cast<int>(d->pyr.size()), rul[f], rvl[f]});
                const size_t *po = &C->pyr_off[C->pyr_first[f] + drop];
                d->pyr.insert(d->pyr.end(), C->pyr.begin() + po[0], C->pyr.begin() + po[1]);
                size_t skip = 0;   // the finer levels in front of (rul, rvl) in the full chain
                for (int k = 0; k < drop; ++k)
                        skip += static_cast<size_t>((1 << std::max(r.ulog2 - k, 0)) + 2) * ((1 << std::max(r.vlog2 - k, 0)) + 2);
                const auto from = C->chain.begin() + C->chain_off[f] + skip;
                d->chain_off.push_back(static_cast<int>(d->chain.size()));
                d->chain.insert(d->chain.end(), from, from + chain_texels(rul[f], rvl[f]));
        }
        fix_borders(d.get(), C->tex, rul, rvl, nfc);
        counts[0] = static_cast<long long>(d->face.size());
        counts[1] = nfc;
        counts[2] = static_cast<long long>(d->pyr.size());
        counts[3] = static_cast<long long>(d->chain.size());
        if (view && view[4] > 0)
                std::printf("height levels for the view: %lld of %lld texels kept (%.1f%%)\n", kept, full, 100.0 * kept / std::max(full, 1LL));
        return d.release();
}


// The Gregory data (fm 12 floats, greg 1 byte a patch); call before disp_take, which frees everything.
// Per face the finest height levels left out for the view (1 byte a face); call before disp_take.
void disp_take_drop(void *h, unsigned char *drop) {
        auto *d = static_cast<Disp *>(h);
        std::memcpy(drop, d->drop.data(), d->drop.size());
}

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
                c->hdr.insert(c->hdr.end(), {r.ulog2, r.vlog2});
                const size_t at = c->chain.size();
                c->chain.resize(at + texels * 3);
                ptex_read_face(pt, f, r.ulog2, r.vlog2, c->chain.data() + at);
        }
        ptex_close(pt);
        *n = static_cast<long long>(c->chain.size());
        return c;
}

// Per face the colour chain's log2 width and height (2 ints); call before disp_color_chains_take.
void disp_color_res(void *h, int *res) {
        auto *c = static_cast<Disp *>(h);
        std::memcpy(res, c->hdr.data(), c->hdr.size() * sizeof(int));
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
