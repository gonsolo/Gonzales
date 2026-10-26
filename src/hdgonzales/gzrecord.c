// LD_PRELOAD shim: records every call of the C API (gonzales.h), with its arrays, to $GZ_RECORD and forwards it to
// libgonzales.so ($GZ_LIB). gzreplay renders the recording without the host application.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gonzales.h"
#include "gzrecord.h"

static FILE *out;

static void *lib(void) {
    static void *h;
    if (!h) {
        const char *p = getenv("GZ_LIB");
        h = dlopen(p ? p : "libgonzales.so", RTLD_NOW | RTLD_GLOBAL);
        if (!h) {
            fprintf(stderr, "gzrecord: %s\n", dlerror());
            exit(1);
        }
        out = fopen(getenv("GZ_RECORD"), "wb");
    }
    return h;
}

#define FWD(name) static __typeof__(name) *real; if (!real) real = (__typeof__(name) *)dlsym(lib(), #name)

static void op(uint32_t o) { fwrite(&o, 4, 1, out); }
static void i32(int v) { fwrite(&v, 4, 1, out); }
static void f32(float v) { fwrite(&v, 4, 1, out); }
static void arr(const void *p, size_t bytes) {
    uint64_t n = p ? bytes : UINT64_MAX;
    fwrite(&n, 8, 1, out);
    if (p) fwrite(p, 1, bytes, out);
}
static void str(const char *s) { arr(s, s ? strlen(s) + 1 : 0); }

void *gz_renderer_new(const char *data_dir) { FWD(gz_renderer_new); op(GZ_RENDERER_NEW); str(data_dir); return real(data_dir); }
void *gz_scene_new(void) { FWD(gz_scene_new); op(GZ_SCENE_NEW); return real(); }
int gz_scene_add_diffuse(void *s, float r, float g, float b, const char *tex, const char *nmap) {
    FWD(gz_scene_add_diffuse); op(GZ_ADD_DIFFUSE); f32(r); f32(g); f32(b); str(tex); str(nmap); return real(s, r, g, b, tex, nmap);
}
int gz_scene_add_coated(void *s, float r, float g, float b, const char *tex, const char *nmap, float alpha, float ior) {
    FWD(gz_scene_add_coated); op(GZ_ADD_COATED); f32(r); f32(g); f32(b); str(tex); str(nmap); f32(alpha); f32(ior);
    return real(s, r, g, b, tex, nmap, alpha, ior);
}
void gz_scene_add_mesh(void *s, const float *xform, const float *pts, int nv, const int *idx, int nt, const float *nrm,
                       const float *uvs, int material) {
    FWD(gz_scene_add_mesh); op(GZ_ADD_MESH); arr(xform, 64); arr(pts, 12 * (size_t)nv); i32(nv); arr(idx, 12 * (size_t)nt); i32(nt);
    arr(nrm, 12 * (size_t)nv); arr(uvs, 8 * (size_t)nv); i32(material);
    real(s, xform, pts, nv, idx, nt, nrm, uvs, material);
}
void gz_scene_add_area_light(void *s, const float *xform, const float *pts, int nv, const int *idx, int nt, float r, float g, float b) {
    FWD(gz_scene_add_area_light); op(GZ_ADD_AREA_LIGHT); arr(xform, 64); arr(pts, 12 * (size_t)nv); i32(nv); arr(idx, 12 * (size_t)nt);
    i32(nt); f32(r); f32(g); f32(b);
    real(s, xform, pts, nv, idx, nt, r, g, b);
}
void gz_scene_add_dome(void *s, float r, float g, float b) { FWD(gz_scene_add_dome); op(GZ_ADD_DOME); f32(r); f32(g); f32(b); real(s, r, g, b); }
void gz_scene_add_dome_map(void *s, const char *path, float r, float g, float b) {
    FWD(gz_scene_add_dome_map); op(GZ_ADD_DOME_MAP); str(path); f32(r); f32(g); f32(b); real(s, path, r, g, b);
}
void gz_scene_add_distant(void *s, float dx, float dy, float dz, float r, float g, float b) {
    FWD(gz_scene_add_distant); op(GZ_ADD_DISTANT); f32(dx); f32(dy); f32(dz); f32(r); f32(g); f32(b); real(s, dx, dy, dz, r, g, b);
}
void gz_scene_add_point(void *s, float x, float y, float z, float r, float g, float b) {
    FWD(gz_scene_add_point); op(GZ_ADD_POINT); f32(x); f32(y); f32(z); f32(r); f32(g); f32(b); real(s, x, y, z, r, g, b);
}
void gz_scene_add_sphere_light(void *s, float x, float y, float z, float radius, float r, float g, float b) {
    FWD(gz_scene_add_sphere_light); op(GZ_ADD_SPHERE_LIGHT); f32(x); f32(y); f32(z); f32(radius); f32(r); f32(g); f32(b);
    real(s, x, y, z, radius, r, g, b);
}
int gz_scene_add_hair(void *s, float sr, float sg, float sb, float bm, float bn, float ior, float alpha) {
    FWD(gz_scene_add_hair); op(GZ_ADD_HAIR); f32(sr); f32(sg); f32(sb); f32(bm); f32(bn); f32(ior); f32(alpha);
    return real(s, sr, sg, sb, bm, bn, ior, alpha);
}
int gz_scene_add_dielectric(void *s, float ior, float alpha) {
    FWD(gz_scene_add_dielectric); op(GZ_ADD_DIELECTRIC); f32(ior); f32(alpha); return real(s, ior, alpha);
}
void gz_scene_add_curves(void *s, const float *cp, const float *widths, int n, int material, const int *materials) {
    FWD(gz_scene_add_curves); op(GZ_ADD_CURVES); arr(cp, 48 * (size_t)n); arr(widths, 8 * (size_t)n); i32(n); i32(material);
    arr(materials, 4 * (size_t)n);
    real(s, cp, widths, n, material, materials);
}
void gz_scene_set_subsurface(void *s, int material, float weight, float rr, float rg, float rb, float aniso, int vdh) {
    FWD(gz_scene_set_subsurface); op(GZ_SET_SUBSURFACE); i32(material); f32(weight); f32(rr); f32(rg); f32(rb); f32(aniso); i32(vdh);
    real(s, material, weight, rr, rg, rb, aniso, vdh);
}
void gz_scene_set_emission(void *s, int material, const char *texture) {
    FWD(gz_scene_set_emission); op(GZ_SET_EMISSION); i32(material); str(texture); real(s, material, texture);
}
void gz_scene_set_camera(void *s, const float *w2c, float fov, int w, int h) {
    FWD(gz_scene_set_camera); op(GZ_SET_CAMERA); arr(w2c, 64); f32(fov); i32(w); i32(h); real(s, w2c, fov, w, h);
}
void gz_scene_set_lens(void *s, float radius, float focus) { FWD(gz_scene_set_lens); op(GZ_SET_LENS); f32(radius); f32(focus); real(s, radius, focus); }
void *gz_session_new(void *renderer, void *scene, int spp, int depth) {
    FWD(gz_session_new); op(GZ_SESSION_NEW); i32(spp); i32(depth); fflush(out); return real(renderer, scene, spp, depth);
}
int gz_session_set_camera(void *s, const float *w2c, float fov, int w, int h) {
    FWD(gz_session_set_camera); op(GZ_SESSION_CAMERA); arr(w2c, 64); f32(fov); i32(w); i32(h); return real(s, w2c, fov, w, h);
}
void gz_session_set_samples(void *s, int spp, int depth) { FWD(gz_session_set_samples); op(GZ_SESSION_SAMPLES); i32(spp); i32(depth); real(s, spp, depth); }
void gz_session_set_lens(void *s, float radius, float focus) { FWD(gz_session_set_lens); op(GZ_SESSION_LENS); f32(radius); f32(focus); real(s, radius, focus); }
void gz_session_set_filter_glossy(void *s, float f) { FWD(gz_session_set_filter_glossy); op(GZ_SESSION_FILTER_GLOSSY); f32(f); real(s, f); }
void gz_session_set_adaptive(void *s, float t, int m) { FWD(gz_session_set_adaptive); op(GZ_SESSION_ADAPTIVE); f32(t); i32(m); real(s, t, m); }
void gz_session_set_clamp(void *s, float c) { FWD(gz_session_set_clamp); op(GZ_SESSION_CLAMP); f32(c); real(s, c); }
int gz_session_render(void *s, int n) { FWD(gz_session_render); return real(s, n); }
int gz_session_resolve(void *s, float *rgb, int denoise) {
    FWD(gz_session_resolve); op(GZ_SESSION_RESOLVE); i32(denoise); fflush(out); return real(s, rgb, denoise);
}
void gz_session_free(void *s) { FWD(gz_session_free); real(s); }
