// Reads the surface dump of Tools/disp_proto.mojo (DUMP=file) and writes hit files for its HITS=file shading.
#pragma once
#include "disp_march.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace gz {

struct Cam {
        V3 eye, fwd, right, up, light;
        float tan_h;
        int w, h;
        V3 dir(int x, int y) const {
                float px = (2.f * (float(x) + 0.5f) / float(w) - 1.f) * tan_h;
                float py = (1.f - 2.f * (float(y) + 0.5f) / float(h)) * tan_h * float(h) / float(w);
                V3 d = fwd + right * px + up * py;
                return d * (1.f / sqrtf(dot(d, d)));
        }
};

struct Dump {
        std::vector<uint8_t> bytes;
        Surf s{};
        Cam cam{};
        int np = 0, nf = 0, npyr = 0, nchain = 0, ncw = 0, nprim = 0;
        const float *boxes = nullptr;
        // Byte ranges of the arrays inside `bytes`, for device copies.
        size_t off_sq, off_face, off_hdr, off_coff, off_corners, off_boxes, off_pyr, off_chain;
};

inline Dump load_dump(const char *path) {
        Dump d;
        FILE *f = std::fopen(path, "rb");
        if (!f)
                throw std::runtime_error(std::string("cannot open ") + path);
        std::fseek(f, 0, SEEK_END);
        long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        d.bytes.resize(size_t(n));
        if (std::fread(d.bytes.data(), 1, size_t(n), f) != size_t(n))
                throw std::runtime_error("short read");
        std::fclose(f);
        size_t o = 0;
        auto i32 = [&]() { int32_t v; std::memcpy(&v, d.bytes.data() + o, 4); o += 4; return v; };
        auto f32 = [&]() { float v; std::memcpy(&v, d.bytes.data() + o, 4); o += 4; return v; };
        auto v3f = [&]() { float x = f32(), y = f32(), z = f32(); return v3(x, y, z); };
        if (i32() != 0x53445A47 || i32() != 1)
                throw std::runtime_error("not a surface dump (version 1)");
        d.np = i32(), d.nf = i32(), d.npyr = i32(), d.nchain = i32(), d.ncw = i32(), d.nprim = i32(), d.cam.w = i32(), d.cam.h = i32();
        d.s.scale = f32();
        d.s.cam = v3f();
        d.s.pixel = f32();
        d.cam.tan_h = f32();
        d.cam.eye = v3f(), d.cam.fwd = v3f(), d.cam.right = v3f(), d.cam.up = v3f(), d.cam.light = v3f();
        for (int l = 0; l < 8; ++l)
                d.s.lv_patch[l] = i32();
        for (int l = 0; l < 8; ++l)
                d.s.lv_prim[l] = i32();
        for (int l = 0; l < 8; ++l)
                d.s.lv_corner[l] = i32();
        auto take = [&](size_t len) { size_t at = o; o += len; return at; };
        d.off_sq = take(12 * size_t(d.np)), d.off_face = take(4 * size_t(d.np)), d.off_hdr = take(12 * size_t(d.nf)), d.off_coff = take(4 * size_t(d.nf));
        d.off_corners = take(4 * size_t(d.ncw)), d.off_boxes = take(24 * size_t(d.nprim)), d.off_pyr = take(size_t(d.npyr)), d.off_chain = take(size_t(d.nchain));
        if (o != d.bytes.size())
                throw std::runtime_error("dump size mismatch");
        const uint8_t *b = d.bytes.data();
        d.s.sq = reinterpret_cast<const float *>(b + d.off_sq);
        d.s.face = reinterpret_cast<const int32_t *>(b + d.off_face);
        d.s.hdr = reinterpret_cast<const int32_t *>(b + d.off_hdr);
        d.s.chain_off = reinterpret_cast<const int32_t *>(b + d.off_coff);
        d.s.corners = reinterpret_cast<const float *>(b + d.off_corners);
        d.s.pyr = b + d.off_pyr;
        d.s.chain = b + d.off_chain;
        d.boxes = reinterpret_cast<const float *>(b + d.off_boxes);
        return d;
}

struct PixHit {   // the HITS record (OdHit): t, patch (-1: none), s, t
        float t;
        int32_t patch;
        float s, u;
};

inline void write_hits(const char *path, const std::vector<PixHit> &h) {
        FILE *f = std::fopen(path, "wb");
        if (!f || std::fwrite(h.data(), sizeof(PixHit), h.size(), f) != h.size())
                throw std::runtime_error(std::string("cannot write ") + path);
        std::fclose(f);
}

} // namespace gz
