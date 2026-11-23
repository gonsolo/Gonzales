// The march traversal (src/gonzales/scene/shapes/march.mojo) and the parts of displaced_patches.mojo it reads, ported
// one to one for CUDA and the host: the second vendor's intersection program and its CPU check.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

namespace gz {

struct V3 {
        float x, y, z;
};
HD V3 v3(float x, float y, float z) { return {x, y, z}; }
HD V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
HD V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
HD V3 operator-(V3 a) { return {-a.x, -a.y, -a.z}; }
HD V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
HD float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
HD V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
HD float fmin_(float a, float b) { return a < b ? a : b; }
HD float fmax_(float a, float b) { return a > b ? a : b; }
HD int imin(int a, int b) { return a < b ? a : b; }
HD int imax(int a, int b) { return a > b ? a : b; }

constexpr int LEVELS = 7;
constexpr int NODE_B = 6;

// DispSurface: the arrays the march reads.
struct Surf {
        const float *sq;          // per patch: u0, v0, size in its face
        const int32_t *face;      // per patch
        const int32_t *hdr;       // per face: node offset, log2 width, log2 height
        const uint8_t *pyr;       // height nodes
        const int32_t *chain_off; // per face: its bordered heights
        const uint8_t *chain;
        const float *corners;     // corner grids (corner_off)
        int32_t lv_patch[8], lv_prim[8], lv_corner[8];
        float scale;
        V3 cam;
        float pixel;
};

struct Frame {   // BLFrame: corners 00 10 11 01 anchored at the ray origin, unit normals
        float px[4], py[4], pz[4], nx[4], ny[4], nz[4];
};

HD int class_of(const int32_t *v, int x) {
        int c = 0;
        for (int l = 1; l < LEVELS; ++l)
                if (v[l] <= x)
                        c = l;
        return c;
}

HD int corner_words(int n) { return 5 + (5 * (n + 1) * (n + 1) + 1) / 2; }

struct PP {
        int patch, k, level;
};
HD PP prim_patch(const Surf &d, int prim) {
        int l = class_of(d.lv_prim, prim), loc = prim - d.lv_prim[l];
        return {d.lv_patch[l] + (loc >> (2 * l)), loc & ((1 << (2 * l)) - 1), l};
}

struct SQ {
        float u0, v0, size;
};
HD SQ sub_square(int la, int k) {
        int n = 1 << la, ku = k % n, kv = k / n;
        float sz = 1.f / float(n);
        return {float(ku) * sz, float(kv) * sz, sz};
}

HD V3 oct_dec(uint16_t e0, uint16_t e1) {
        float u = (float(e0) - 32768.f) * float(1.0 / 32767), v = (float(e1) - 32768.f) * float(1.0 / 32767);
        float z = 1.f - fabsf(u) - fabsf(v);
        if (z < 0.f) {
                float uu = (1.f - fabsf(v)) * (u >= 0.f ? 1.f : -1.f);
                v = (1.f - fabsf(u)) * (v >= 0.f ? 1.f : -1.f);
                u = uu;
        }
        float r = (e0 == 0 && e1 == 0) ? 0.f : 1.f / sqrtf(u * u + v * v + z * z);
        return {u * r, v * r, z * r};
}

HD int edge_idx(int n, int e, int k) {
        int i = e == 0 ? k : (e == 1 ? n : (e == 2 ? n - k : 0));
        int j = e == 0 ? 0 : (e == 1 ? k : (e == 2 ? n : n - k));
        return j * (n + 1) + i;
}

HD Frame bl_frame(const Surf &d, int pi, int k, V3 org) {
        int l = class_of(d.lv_patch, pi), n = 1 << l;
        int ku = k & (n - 1), kv = k >> l;
        Frame f;
        const float *g = d.corners + d.lv_corner[l] + (pi - d.lv_patch[l]) * corner_words(n);
        float st = g[3];
        V3 b = v3(g[0], g[1], g[2]);
        uint32_t ew = reinterpret_cast<const uint32_t *>(g + 4)[0];
        const uint16_t *q = reinterpret_cast<const uint16_t *>(g + 5);
        for (int c = 0; c < 4; ++c) {
                int o = 5 * ((kv + (c >= 2 ? 1 : 0)) * (n + 1) + ku + ((c == 1 || c == 2) ? 1 : 0));
                f.px[c] = (b.x + float(q[o]) * st) - org.x;
                f.py[c] = (b.y + float(q[o + 1]) * st) - org.y;
                f.pz[c] = (b.z + float(q[o + 2]) * st) - org.z;
                V3 nv = oct_dec(q[o + 3], q[o + 4]);
                f.nx[c] = nv.x, f.ny[c] = nv.y, f.nz[c] = nv.z;
        }
        if (ew != 0) {
                for (int c = 0; c < 4; ++c) {
                        int i = ku + ((c == 1 || c == 2) ? 1 : 0), j = kv + (c >= 2 ? 1 : 0);
                        int e = j == 0 ? 0 : (i == n ? 1 : (j == n ? 2 : (i == 0 ? 3 : -1)));
                        if (e < 0)
                                continue;
                        int kk = e == 0 ? i : (e == 1 ? j : (e == 2 ? n - i : n - j));
                        int r = 1 << int((ew >> (4 * e)) & 15u), m = kk & (r - 1);
                        if (m == 0)
                                continue;
                        int o0 = 5 * edge_idx(n, e, kk - m), o1 = 5 * edge_idx(n, e, kk - m + r);
                        float a = float(m) / float(r);
                        V3 p0 = v3((b.x + float(q[o0]) * st) - org.x, (b.y + float(q[o0 + 1]) * st) - org.y, (b.z + float(q[o0 + 2]) * st) - org.z);
                        V3 p1 = v3((b.x + float(q[o1]) * st) - org.x, (b.y + float(q[o1 + 1]) * st) - org.y, (b.z + float(q[o1 + 2]) * st) - org.z);
                        V3 n0 = oct_dec(q[o0 + 3], q[o0 + 4]), n1 = oct_dec(q[o1 + 3], q[o1 + 4]);
                        f.px[c] = p0.x + (p1.x - p0.x) * a, f.py[c] = p0.y + (p1.y - p0.y) * a, f.pz[c] = p0.z + (p1.z - p0.z) * a;
                        f.nx[c] = n0.x + (n1.x - n0.x) * a, f.ny[c] = n0.y + (n1.y - n0.y) * a, f.nz[c] = n0.z + (n1.z - n0.z) * a;
                }
        }
        return f;
}

HD float bilerp(const float *q, float a, float b) { return (q[0] + (q[1] - q[0]) * a) * (1.f - b) + (q[3] + (q[2] - q[3]) * a) * b; }

HD float texel(const Surf &d, int off, int w, int x, int y) { return float(d.chain[off + (y + 1) * (w + 2) + x + 1]); }

struct R2 {
        float lo, hi;
};

HD R2 plane_range(const Surf &d, int node, float X0, float X1, float Y0, float Y1, float xc, float yc, float hw, float hh) {
        float a = float(d.pyr[node]);
        float bx = 2.f * float(int8_t(d.pyr[node + 1])), by = 2.f * float(int8_t(d.pyr[node + 2]));
        float r = float(d.pyr[node + 3]);
        float ex0 = bx * (X0 - xc) / hw, ex1 = bx * (X1 - xc) / hw, ey0 = by * (Y0 - yc) / hh, ey1 = by * (Y1 - yc) / hh;
        float mn = float(d.pyr[node + 4]), mx = float(d.pyr[node + 5]);
        return {fmax_(a + fmin_(ex0, ex1) + fmin_(ey0, ey1) - r, mn), fmin_(a + fmax_(ex0, ex1) + fmax_(ey0, ey1) + r, mx)};
}

HD R2 border_range(const Surf &d, int off, int w, int h, int x0, int x1, int y0, int y1) {
        float lo = 1e30f, hi = -1e30f;
        if (x0 < 0 || x1 >= w)
                for (int y = y0; y <= y1; ++y) {
                        if (x0 < 0) {
                                float t = texel(d, off, w, -1, y);
                                lo = fmin_(lo, t), hi = fmax_(hi, t);
                        }
                        if (x1 >= w) {
                                float t = texel(d, off, w, w, y);
                                lo = fmin_(lo, t), hi = fmax_(hi, t);
                        }
                }
        if (y0 < 0 || y1 >= h)
                for (int x = x0; x <= x1; ++x) {
                        if (y0 < 0) {
                                float t = texel(d, off, w, x, -1);
                                lo = fmin_(lo, t), hi = fmax_(hi, t);
                        }
                        if (y1 >= h) {
                                float t = texel(d, off, w, x, h);
                                lo = fmin_(lo, t), hi = fmax_(hi, t);
                        }
                }
        return {lo, hi};
}

HD R2 height_range(const Surf &d, int f, float u0, float u1, float v0, float v1) {
        int ul = d.hdr[3 * f + 1], vl = d.hdr[3 * f + 2], w = 1 << ul, h = 1 << vl;
        int x0 = imin(imax(int(floorf(u0 * float(w) - 0.5f)), -1), w);
        int x1 = imin(imax(int(floorf(u1 * float(w) - 0.5f)) + 1, -1), w);
        int y0 = imin(imax(int(floorf(v0 * float(h) - 0.5f)), -1), h);
        int y1 = imin(imax(int(floorf(v1 * float(h) - 0.5f)) + 1, -1), h);
        int off = d.chain_off[f];
        if (x1 - x0 <= 4 && y1 - y0 <= 4) {
                float lo = 1e30f, hi = -1e30f;
                for (int y = y0; y <= y1; ++y)
                        for (int x = x0; x <= x1; ++x) {
                                float t = texel(d, off, w, imin(imax(x, -1), w), imin(imax(y, -1), h));
                                lo = fmin_(lo, t), hi = fmax_(hi, t);
                        }
                return {lo, hi};
        }
        R2 edge = border_range(d, off, w, h, x0, x1, y0, y1);
        float X0 = u0 * float(w), X1 = u1 * float(w), Y0 = v0 * float(h), Y1 = v1 * float(h);
        int node = d.hdr[3 * f], k = 2, cw = 1, ch = 1, cx0 = 0, cx1 = 0, cy0 = 0, cy1 = 0;
        float tw = 1.f, th = 1.f;
        while (true) {
                cw = imax(w >> k, 1), ch = imax(h >> k, 1);
                tw = float(w / cw), th = float(h / ch);
                cx0 = imin(int(X0 / tw), cw - 1), cx1 = imax(imin(int(ceilf(X1 / tw)) - 1, cw - 1), cx0);
                cy0 = imin(int(Y0 / th), ch - 1), cy1 = imax(imin(int(ceilf(Y1 / th)) - 1, ch - 1), cy0);
                if ((cx1 - cx0 <= 1 && cy1 - cy0 <= 1) || (cw == 1 && ch == 1))
                        break;
                node += NODE_B * cw * ch;
                ++k;
        }
        float lo = edge.lo, hi = edge.hi;
        for (int cy = cy0; cy <= cy1; ++cy)
                for (int cx = cx0; cx <= cx1; ++cx) {
                        float bx0 = float(cx) * tw, by0 = float(cy) * th;
                        R2 r = plane_range(d, node + NODE_B * (cy * cw + cx), fmax_(X0, bx0), fmin_(X1, bx0 + tw), fmax_(Y0, by0),
                                           fmin_(Y1, by0 + th), bx0 + 0.5f * tw, by0 + 0.5f * th, 0.5f * tw, 0.5f * th);
                        lo = fmin_(lo, r.lo), hi = fmax_(hi, r.hi);
                }
        return {lo, hi};
}

HD R2 imul(float a0, float a1, float b0, float b1) {
        float p = a0 * b0, q = a0 * b1, r = a1 * b0, s = a1 * b1;
        return {fmin_(fmin_(p, q), fmin_(r, s)), fmax_(fmax_(p, q), fmax_(r, s))};
}

struct Cell {   // _bl_cell: box, base extent, slab along n
        V3 lo, hi;
        float base, s0, s1;
};

HD void bilerp4(const float *q, const float *a, const float *b, float *out) {
        for (int i = 0; i < 4; ++i)
                out[i] = bilerp(q, a[i], b[i]);
}
HD float min4(const float *v) { return fmin_(fmin_(v[0], v[1]), fmin_(v[2], v[3])); }
HD float max4(const float *v) { return fmax_(fmax_(v[0], v[1]), fmax_(v[2], v[3])); }

HD Cell bl_cell(const Surf &d, int pi, int f, const Frame &fr, float s0, float t0, float inv, float d0, float d1, float d2, float d3, V3 n) {
        float a[4] = {d0 * inv - s0 * inv, (d0 + d2) * inv - s0 * inv, (d0 + d2) * inv - s0 * inv, d0 * inv - s0 * inv};
        float b[4] = {d1 * inv - t0 * inv, d1 * inv - t0 * inv, (d1 + d3) * inv - t0 * inv, (d1 + d3) * inv - t0 * inv};
        float x[4], y[4], z[4];
        bilerp4(fr.px, a, b, x), bilerp4(fr.py, a, b, y), bilerp4(fr.pz, a, b, z);
        V3 lo = v3(min4(x), min4(y), min4(z)), hi = v3(max4(x), max4(y), max4(z));
        float base = fmax_(fmax_(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z);
        float dp[4];
        for (int i = 0; i < 4; ++i)
                dp[i] = x[i] * n.x + y[i] * n.y + z[i] * n.z;
        if (d.scale <= 0.f)
                return {lo, hi, base, min4(dp), max4(dp)};
        float nx[4], ny[4], nz[4], nn[4];
        bilerp4(fr.nx, a, b, nx), bilerp4(fr.ny, a, b, ny), bilerp4(fr.nz, a, b, nz);
        for (int i = 0; i < 4; ++i)
                nn[i] = nx[i] * n.x + ny[i] * n.y + nz[i] * n.z;
        float sz = d.sq[3 * pi + 2], u0 = d.sq[3 * pi], v0 = d.sq[3 * pi + 1];
        R2 hr = height_range(d, f, u0 + d0 * sz, u0 + (d0 + d2) * sz, v0 + d1 * sz, v0 + (d1 + d3) * sz);
        float h0 = hr.lo * d.scale, h1 = hr.hi * d.scale;
        R2 ox = imul(h0, h1, min4(nx), max4(nx)), oy = imul(h0, h1, min4(ny), max4(ny)), oz = imul(h0, h1, min4(nz), max4(nz));
        float ax[4], ay[4], az[4];
        for (int i = 0; i < 4; ++i)
                ax[i] = fabsf(x[i]), ay[i] = fabsf(y[i]), az[i] = fabsf(z[i]);
        float pad = 1e-6f * (max4(ax) + max4(ay) + max4(az));
        R2 on = imul(h0, h1, min4(nn), max4(nn));
        return {v3(lo.x + ox.lo - pad, lo.y + oy.lo - pad, lo.z + oz.lo - pad), v3(hi.x + ox.hi + pad, hi.y + oy.hi + pad, hi.z + oz.hi + pad), base,
                min4(dp) + on.lo, max4(dp) + on.hi};
}

struct Hit4 {
        bool hit;
        float t, u, v;
};

HD Hit4 tri_hit(V3 a, V3 b, V3 c, V3 dir, float t_max) {
        V3 e1 = b - a, e2 = c - a, pv = cross(dir, e2);
        float det = dot(e1, pv);
        if (det == 0.f)
                return {false, t_max, 0.f, 0.f};
        float inv = 1.f / det;
        V3 tv = -a;
        float u = dot(tv, pv) * inv;
        V3 qv = cross(tv, e1);
        float v = dot(dir, qv) * inv, t = dot(e2, qv) * inv;
        const float e = 1e-3f;
        if (u < -e || v < -e || u + v > 1.f + e || t <= 0.f || t >= t_max)
                return {false, t_max, 0.f, 0.f};
        V3 res = a + e1 * u + e2 * v - dir * t;
        float tol = e * (sqrtf(dot(e1, e1)) + sqrtf(dot(e2, e2))) + 1e-6f * (fabsf(a.x) + fabsf(a.y) + fabsf(a.z));
        if (dot(res, res) > tol * tol)
                return {false, t_max, 0.f, 0.f};
        return {true, t, u, v};
}

HD Hit4 bilinear_hit(V3 q00, V3 q10, V3 q11, V3 q01, V3 dir, float t_max) {
        V3 e10 = q10 - q00, e11 = q11 - q10, e00 = q01 - q00;
        V3 qn = cross(q10 - q00, q01 - q11);
        float a = dot(cross(q00, dir), e00), c = dot(qn, dir), b = dot(cross(q10, dir), e11) - a - c;
        float det = b * b - 4.f * a * c, t = t_max, u = 0.f, v = 0.f;
        bool hit = false;
        if (det < 0.f)
                return {false, t, u, v};
        det = sqrtf(det);
        float u1, u2;
        if (c == 0.f) {
                u1 = -a / b, u2 = -1.f;
        } else {
                u1 = (-b - (b >= 0.f ? det : -det)) * 0.5f;
                u2 = u1 != 0.f ? a / u1 : -1.f;
                u1 = u1 / c;
        }
        const float e = 1e-3f;
        for (int r = 0; r < 2; ++r) {
                float ur = r == 0 ? u1 : u2;
                if (ur < -e || ur > 1.f + e)
                        continue;
                V3 pa = q00 + e10 * ur, pb = e00 + (e11 - e00) * ur, n = cross(dir, pb);
                float dd = dot(n, n);
                if (dd <= 0.f)
                        continue;
                n = cross(n, pa);
                float tr = dot(n, pb) / dd, vr = dot(n, dir);
                if (vr >= -e * dd && vr <= (1.f + e) * dd && tr > 0.f && tr < t) {
                        V3 res = pa + pb * (vr / dd) - dir * tr;
                        float tol = e * (sqrtf(dot(e10, e10)) + sqrtf(dot(e00, e00)) + sqrtf(dot(e11, e11))) + 1e-6f * (fabsf(pa.x) + fabsf(pa.y) + fabsf(pa.z));
                        if (dot(res, res) > tol * tol)
                                continue;
                        t = tr, u = fmin_(fmax_(ur, 0.f), 1.f), v = fmin_(fmax_(vr / dd, 0.f), 1.f), hit = true;
                }
        }
        return {hit, t, u, v};
}

HD float height_at(const Surf &d, int off, int w, int h, float u0, float v0, float sz, float s, float t) {   // one lane of _heights8
        float x = fmin_(fmax_((s * sz + u0) * float(w) - 0.5f, -0.5f), float(w) - 0.5f);
        float y = fmin_(fmax_((t * sz + v0) * float(h) - 0.5f, -0.5f), float(h) - 0.5f);
        int x0 = imin(int(floorf(x)), w - 1), y0 = imin(int(floorf(y)), h - 1);
        float fx = x - float(x0), fy = y - float(y0);
        float a = texel(d, off, w, x0, y0) * (1.f - fx) + texel(d, off, w, x0 + 1, y0) * fx;
        float b = texel(d, off, w, x0, y0 + 1) * (1.f - fx) + texel(d, off, w, x0 + 1, y0 + 1) * fx;
        return a * (1.f - fy) + b * fy;
}

HD V3 bl_point(const Surf &d, const Frame &fr, float s0, float t0, float inv, float s, float t, float hgt) {
        float a = (s - s0) * inv, b = (t - t0) * inv;
        V3 p = v3(bilerp(fr.px, a, b), bilerp(fr.py, a, b), bilerp(fr.pz, a, b));
        if (d.scale <= 0.f)
                return p;
        V3 n = v3(bilerp(fr.nx, a, b), bilerp(fr.ny, a, b), bilerp(fr.nz, a, b));
        return p + n * (hgt * d.scale);
}

HD Hit4 bl_leaf(const Surf &d, int pi, int f, const Frame &fr, float s0q, float t0q, float inv_s, float d0, float d1, float d2, float d3, V3 dir,
                float t_max) {
        float s0 = d0, s1 = d0 + d2, t0 = d1, t1 = d1 + d3, sm = s0 + 0.5f * d2, tm = t0 + 0.5f * d3;
        float cs[8] = {s0, s1, s1, s0, sm, s1, sm, s0}, ct[8] = {t0, t0, t1, t1, t0, tm, t1, tm};
        int off = d.chain_off[f], w = 1 << d.hdr[3 * f + 1], h = 1 << d.hdr[3 * f + 2];
        float u0 = d.sq[3 * pi], v0 = d.sq[3 * pi + 1], sz = d.sq[3 * pi + 2];
        V3 cp[8];
        for (int k = 0; k < 8; ++k)
                cp[k] = bl_point(d, fr, s0q, t0q, inv_s, cs[k], ct[k], height_at(d, off, w, h, u0, v0, sz, cs[k], ct[k]));
        // _leaf_hit: the bilinear patch through the corners and the edge fans through the midpoints.
        float best = t_max, hs = 0.f, ht = 0.f;
        bool hit = false;
        Hit4 bh = bilinear_hit(cp[0], cp[1], cp[2], cp[3], dir, best);
        if (bh.hit)
                best = bh.t, hit = true, hs = s0 + bh.u * (s1 - s0), ht = t0 + bh.v * (t1 - t0);
        for (int k = 0; k < 4; ++k) {
                int ka = k, kb = (k + 1) % 4, km = 4 + k;
                Hit4 th = tri_hit(cp[ka], cp[km], cp[kb], dir, best);
                if (th.hit) {
                        best = th.t, hit = true;
                        float w0 = 1.f - th.u - th.v;
                        hs = w0 * cs[ka] + th.u * cs[km] + th.v * cs[kb], ht = w0 * ct[ka] + th.u * ct[km] + th.v * ct[kb];
                }
        }
        return {hit, best, hs, ht};
}

HD V3 mean_normal(const Frame &fr) {
        V3 n = cross(v3(fr.px[2] - fr.px[0], fr.py[2] - fr.py[0], fr.pz[2] - fr.pz[0]), v3(fr.px[3] - fr.px[1], fr.py[3] - fr.py[1], fr.pz[3] - fr.pz[1]));
        float nl = sqrtf(dot(n, n));
        return nl > 0.f ? n * (1.f / nl) : v3(0, 0, 0);
}

HD float leaf_fp(V3 lo, V3 hi, V3 cam, float pixel) {   // _leaf_size(...)[1]
        V3 c = (lo + hi) * 0.5f - cam, e = hi - lo;
        float r = 0.5f * sqrtf(dot(e, e));
        float dist = fmax_(sqrtf(dot(c, c)) - r, r * 0.1f + 1e-4f);
        return dist * pixel;
}

// march.mojo

HD V3 pn_p(const Frame &fr, float a, float b) { return v3(bilerp(fr.px, a, b), bilerp(fr.py, a, b), bilerp(fr.pz, a, b)); }
HD V3 pn_n(const Frame &fr, float a, float b) { return v3(bilerp(fr.nx, a, b), bilerp(fr.ny, a, b), bilerp(fr.nz, a, b)); }

HD float psi(const Frame &fr, V3 dir, float a, float b) { return dot(pn_p(fr, a, b), cross(pn_n(fr, a, b), dir)); }

HD void invert(const Frame &fr, V3 x, float a0, float b0, float &ao, float &bo) {
        float a = a0, b = b0;
        for (int it = 0; it < 8; ++it) {
                V3 p = pn_p(fr, a, b) - x, n = pn_n(fr, a, b);
                V3 pa = pn_p(fr, 1.f, b) - pn_p(fr, 0.f, b), pb = pn_p(fr, a, 1.f) - pn_p(fr, a, 0.f);
                V3 na = pn_n(fr, 1.f, b) - pn_n(fr, 0.f, b), nb = pn_n(fr, a, 1.f) - pn_n(fr, a, 0.f);
                V3 fv = cross(p, n), ja = cross(pa, n) + cross(p, na), jb = cross(pb, n) + cross(p, nb);
                float m00 = dot(ja, ja), m01 = dot(ja, jb), m11 = dot(jb, jb), det = m00 * m11 - m01 * m01;
                if (!(fabsf(det) > 0.f))
                        break;
                float ga = dot(ja, fv), gb = dot(jb, fv);
                float da = -(m11 * ga - m01 * gb) / det, db = -(m00 * gb - m01 * ga) / det;
                a += da, b += db;
                if (da * da + db * db < 1e-10f)
                        break;
        }
        ao = a, bo = b;
}

HD float ray_t(const Frame &fr, V3 dir, float a, float b) {
        V3 p = pn_p(fr, a, b), n = pn_n(fr, a, b);
        float dn = dot(dir, n), nn = dot(n, n), det = dn * dn - nn;
        if (det == 0.f)
                return dot(dir, p);
        return (dot(dir, p) * (-nn) + dn * dot(n, p)) / det;
}

HD void edge_pt(int e, float x, float &a, float &b) {
        a = e == 0 ? x : (e == 1 ? 1.f : (e == 2 ? x : 0.f));
        b = e == 0 ? 0.f : (e == 1 ? x : (e == 2 ? 1.f : x));
}

struct Bd {
        float fa, fb, ft, fe, la, lb, lt, cnt;
};

HD Bd boundary(const Frame &fr, V3 dir, float t0, float t1) {
        Bd r{0, 0, 1e30f, 0, 0, 0, -1e30f, 0};
        int cnt = 0;
        for (int e = 0; e < 4; ++e) {
                float a, b;
                edge_pt(e, 0.f, a, b);
                float f0 = psi(fr, dir, a, b);
                edge_pt(e, 0.5f, a, b);
                float fm = psi(fr, dir, a, b);
                edge_pt(e, 1.f, a, b);
                float f1 = psi(fr, dir, a, b);
                float c2 = 2.f * (f0 + f1) - 4.f * fm, c1 = f1 - f0 - c2;
                float rr[2] = {-1.f, -1.f};
                if (fabsf(c2) < 1e-12f * (fabsf(c1) + fabsf(f0))) {
                        if (c1 != 0.f)
                                rr[0] = -f0 / c1;
                } else {
                        float disc = c1 * c1 - 4.f * c2 * f0;
                        if (disc >= 0.f) {
                                float sq = sqrtf(disc), q = -0.5f * (c1 + (c1 >= 0.f ? sq : -sq));
                                rr[0] = q / c2, rr[1] = q != 0.f ? f0 / q : -1.f;
                        }
                }
                for (int g = 0; g < 2; ++g) {
                        if (rr[g] >= 0.f && rr[g] <= 1.f) {
                                edge_pt(e, rr[g], a, b);
                                float tt = ray_t(fr, dir, a, b);
                                if (tt >= t0 && tt <= t1) {
                                        ++cnt;
                                        if (tt < r.ft)
                                                r.fa = a, r.fb = b, r.ft = tt, r.fe = float(e);
                                        if (tt > r.lt)
                                                r.la = a, r.lb = b, r.lt = tt;
                                }
                        }
                }
        }
        r.cnt = float(cnt);
        return r;
}

struct MHit {
        bool hit;
        float t, s, u;
        int steps;
        float size;
};

// intersect_march: the nearest hit of sub-square prim, dir unit length.
HD MHit intersect_march(const Surf &d, int prim, V3 org, V3 dir, float t_max) {
        PP pk = prim_patch(d, prim);
        int pi = pk.patch;
        SQ sq = sub_square(pk.level, pk.k);
        float S = sq.size, inv_s = 1.f / S;
        int f = d.face[pi];
        float texels = S * d.sq[3 * pi + 2] * float(1 << imax(d.hdr[3 * f + 1], d.hdr[3 * f + 2]));
        Frame fr = bl_frame(d, pi, pk.k, org);
        V3 nrm = mean_normal(fr);
        V3 inv = v3(1.f / dir.x, 1.f / dir.y, 1.f / dir.z);
        Cell c = bl_cell(d, pi, f, fr, sq.u0, sq.v0, inv_s, sq.u0, sq.v0, S, S, nrm);
        // _span
        float spa, spb;
        {
                V3 lo = c.lo, hi = c.hi;
                float size = fmax_(fmax_(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z), bp = size * 5e-3f + 1e-6f;
                float ax = (lo.x - bp) * inv.x, bx = (hi.x + bp) * inv.x, ay = (lo.y - bp) * inv.y, by = (hi.y + bp) * inv.y;
                float az = (lo.z - bp) * inv.z, bz = (hi.z + bp) * inv.z;
                float t0 = fmax_(fmax_(fmin_(ax, bx), fmin_(ay, by)), fmax_(fmin_(az, bz), 0.f));
                float t1 = fmin_(fmin_(fmax_(ax, bx), fmax_(ay, by)), fmin_(fmax_(az, bz), t_max));
                float dd = dot(nrm, dir);
                if (fabsf(dd) > 1e-20f) {
                        float pad = bp + 1e-5f * (fabsf(c.s0) + fabsf(c.s1));
                        float ta = (c.s0 - pad) / dd, tb = (c.s1 + pad) / dd;
                        t0 = fmax_(t0, fmin_(ta, tb)), t1 = fmin_(t1, fmax_(ta, tb));
                }
                spa = t0, spb = t1;
        }
        if (spb < spa)
                return {false, t_max, 0.f, 0.f, 0, S};
        float lx = fmin_(fmin_(fr.px[0], fr.px[1]), fmin_(fr.px[2], fr.px[3])), hx = fmax_(fmax_(fr.px[0], fr.px[1]), fmax_(fr.px[2], fr.px[3]));
        float ly = fmin_(fmin_(fr.py[0], fr.py[1]), fmin_(fr.py[2], fr.py[3])), hy = fmax_(fmax_(fr.py[0], fr.py[1]), fmax_(fr.py[2], fr.py[3]));
        float lz = fmin_(fmin_(fr.pz[0], fr.pz[1]), fmin_(fr.pz[2], fr.pz[3])), hz = fmax_(fmax_(fr.pz[0], fr.pz[1]), fmax_(fr.pz[2], fr.pz[3]));
        float ext = fmax_(fmax_(hx - lx, hy - ly), hz - lz);
        float fp = leaf_fp(c.lo, c.hi, d.cam - org, d.pixel);
        int nc = 1;
        while (texels / float(nc) > 1.f && ext / float(nc) > fp)
                nc *= 2;
        float nf = float(nc);
        int i = 0, j = 0, i1 = 0, j1 = 0, side = -1;
        if (nc > 1) {
                float a0, b0, a1, b1;
                invert(fr, dir * spa, 0.5f, 0.5f, a0, b0);
                invert(fr, dir * spb, a0, b0, a1, b1);
                const float e = 1e-4f;
                bool in0 = a0 >= -e && a0 <= 1 + e && b0 >= -e && b0 <= 1 + e;
                bool in1 = a1 >= -e && a1 <= 1 + e && b1 >= -e && b1 <= 1 + e;
                if (!in0 || !in1) {
                        Bd bd = boundary(fr, dir, spa, spb);
                        if (!in0) {
                                if (bd.cnt == 0.f)
                                        return {false, t_max, 0.f, 0.f, 1, S};
                                a0 = bd.fa, b0 = bd.fb, side = int(bd.fe);
                        }
                        if (!in1 && bd.cnt > 0.f)
                                a1 = bd.la, b1 = bd.lb;
                }
                i = imin(imax(int(floorf(fmin_(fmax_(a0, 0.f), 1.f) * nf)), 0), nc - 1);
                j = imin(imax(int(floorf(fmin_(fmax_(b0, 0.f), 1.f) * nf)), 0), nc - 1);
                i1 = imin(imax(int(floorf(fmin_(fmax_(a1, 0.f), 1.f) * nf)), 0), nc - 1);
                j1 = imin(imax(int(floorf(fmin_(fmax_(b1, 0.f), 1.f) * nf)), 0), nc - 1);
        }
        float cw = S / nf;
        int steps = 0;
        for (int it = 0; it < 4 * nc; ++it) {
                ++steps;
                Hit4 lh = bl_leaf(d, pi, f, fr, sq.u0, sq.v0, inv_s, sq.u0 + float(i) * cw, sq.v0 + float(j) * cw, cw, cw, dir, t_max);
                if (lh.hit)
                        return {true, lh.t, lh.u, lh.v, steps, cw};
                if (i == i1 && j == j1)
                        break;
                float a_lo = float(i) / nf, a_hi = float(i + 1) / nf, b_lo = float(j) / nf, b_hi = float(j + 1) / nf;
                bool s00 = psi(fr, dir, a_lo, b_lo) <= 0.f, s10 = psi(fr, dir, a_hi, b_lo) <= 0.f;
                bool s11 = psi(fr, dir, a_hi, b_hi) <= 0.f, s01 = psi(fr, dir, a_lo, b_hi) <= 0.f;
                int best_e = -1, best_d = 1 << 30;
                for (int ed = 0; ed < 4; ++ed) {
                        bool crosses = ed == 0 ? (s00 != s10) : (ed == 1 ? (s10 != s11) : (ed == 2 ? (s11 != s01) : (s01 != s00)));
                        if (crosses && ed != side) {
                                int ni = i + (ed == 1 ? 1 : (ed == 3 ? -1 : 0)), nj = j + (ed == 2 ? 1 : (ed == 0 ? -1 : 0));
                                int dd = abs(i1 - ni) + abs(j1 - nj);
                                if (dd < best_d)
                                        best_d = dd, best_e = ed;
                        }
                }
                if (best_e < 0)
                        break;
                int ni = i + (best_e == 1 ? 1 : (best_e == 3 ? -1 : 0)), nj = j + (best_e == 2 ? 1 : (best_e == 0 ? -1 : 0));
                if (ni < 0 || ni >= nc || nj < 0 || nj >= nc)
                        break;
                i = ni, j = nj, side = (best_e + 2) % 4;
        }
        return {false, t_max, 0.f, 0.f, steps, cw};
}

} // namespace gz
