// Renders a gzrecord recording: the same C API calls, then all samples at once and one resolve. Writes a PFM.
// Usage: gzreplay RECORDING OUT.pfm [SPP]; GZ_DENOISE=0/1, GZ_FILTER_GLOSSY, GZ_CLAMP, GZ_BOUNCES=d,g,t and GZ_INTEGRATOR=0/1/2
// (path, BDPT, VCM) override the recorded settings
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>
#include "gonzales.h"
#include "gzrecord.h"

static FILE *in;
static int i32(void) { int v = 0; if (fread(&v, 4, 1, in) != 1) exit(2); return v; }
static float f32(void) { float v = 0; if (fread(&v, 4, 1, in) != 1) exit(2); return v; }
static void *arr(void) {
    uint64_t n;
    if (fread(&n, 8, 1, in) != 1) exit(2);
    if (n == UINT64_MAX) return NULL;
    void *p = malloc(n ? n : 1);
    if (n && fread(p, 1, n, in) != n) exit(2);
    return p;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: gzreplay RECORDING OUT.pfm [SPP]\n"); return 1; }
    in = fopen(argv[1], "rb");
    if (!in) { perror(argv[1]); return 1; }
    void *renderer = NULL, *scene = NULL, *session = NULL;
    float w2c[16] = {0}, fov = 0, rad = 0, focus = 0, glossy = 0, thr = 0, clamp = 0;
    int bounces[3] = {-1, -1, -1};
    int integrator = 0;
    int w = 0, h = 0, spp = 0, depth = 0, min_spp = 0, denoise = 0;
    double t0 = now(), t_session = 0;
    uint32_t op;
    while (fread(&op, 4, 1, in) == 1) {
        float a, b, c, d, e, f, g;
        char *s1, *s2;
        float *x, *p, *nrm, *uv;
        int *idx, nv, nt, n, m;
        switch (op) {
        case GZ_RENDERER_NEW: s1 = arr(); renderer = gz_renderer_new(getenv("GONZALES_DATA_DIR") ? getenv("GONZALES_DATA_DIR") : s1); break;
        case GZ_SCENE_NEW: scene = gz_scene_new(); break;
        case GZ_ADD_DIFFUSE: a = f32(); b = f32(); c = f32(); s1 = arr(); s2 = arr(); gz_scene_add_diffuse(scene, a, b, c, s1, s2); break;
        case GZ_ADD_COATED: a = f32(); b = f32(); c = f32(); s1 = arr(); s2 = arr(); d = f32(); e = f32();
            gz_scene_add_coated(scene, a, b, c, s1, s2, d, e); break;
        case GZ_ADD_MESH: x = arr(); p = arr(); nv = i32(); idx = arr(); nt = i32(); nrm = arr(); uv = arr(); m = i32();
            gz_scene_add_mesh(scene, x, p, nv, idx, nt, nrm, uv, m); free(x); free(p); free(idx); free(nrm); free(uv); break;
        case GZ_ADD_AREA_LIGHT: x = arr(); p = arr(); nv = i32(); idx = arr(); nt = i32(); a = f32(); b = f32(); c = f32();
            gz_scene_add_area_light(scene, x, p, nv, idx, nt, a, b, c); free(x); free(p); free(idx); break;
        case GZ_ADD_DOME: a = f32(); b = f32(); c = f32(); gz_scene_add_dome(scene, a, b, c); break;
        case GZ_ADD_DOME_MAP: s1 = arr(); a = f32(); b = f32(); c = f32(); gz_scene_add_dome_map(scene, s1, a, b, c); break;
        case GZ_ADD_DISTANT: a = f32(); b = f32(); c = f32(); d = f32(); e = f32(); f = f32(); gz_scene_add_distant(scene, a, b, c, d, e, f); break;
        case GZ_ADD_POINT: a = f32(); b = f32(); c = f32(); d = f32(); e = f32(); f = f32(); gz_scene_add_point(scene, a, b, c, d, e, f); break;
        case GZ_ADD_SPHERE_LIGHT: a = f32(); b = f32(); c = f32(); d = f32(); e = f32(); f = f32(); g = f32();
            gz_scene_add_sphere_light(scene, a, b, c, d, e, f, g); break;
        case GZ_ADD_HAIR: a = f32(); b = f32(); c = f32(); d = f32(); e = f32(); f = f32(); g = f32(); gz_scene_add_hair(scene, a, b, c, d, e, f, g); break;
        case GZ_ADD_DIELECTRIC: a = f32(); b = f32(); gz_scene_add_dielectric(scene, a, b); break;
        case GZ_ADD_CURVES: p = arr(); x = arr(); n = i32(); m = i32(); idx = arr();
            gz_scene_add_curves(scene, p, x, n, m, idx); free(p); free(x); free(idx); break;
        case GZ_SET_SUBSURFACE: m = i32(); a = f32(); b = f32(); c = f32(); d = f32(); e = f32(); n = i32();
            gz_scene_set_subsurface(scene, m, a, b, c, d, e, n); break;
        case GZ_SET_EMISSION: { m = i32(); char *t = arr(); gz_scene_set_emission(scene, m, t); free(t); } break;
        case GZ_SET_CAMERA: x = arr(); a = f32(); nv = i32(); nt = i32(); gz_scene_set_camera(scene, x, a, nv, nt); free(x); break;
        case GZ_SET_LENS: a = f32(); b = f32(); gz_scene_set_lens(scene, a, b); break;
        case GZ_SESSION_NEW: spp = i32(); depth = i32(); break;
        case GZ_SESSION_CAMERA: x = arr(); for (int k = 0; k < 16; ++k) w2c[k] = x[k]; free(x); fov = f32(); w = i32(); h = i32(); break;
        case GZ_SESSION_SAMPLES: spp = i32(); depth = i32(); break;
        case GZ_SESSION_LENS: rad = f32(); focus = f32(); break;
        case GZ_SESSION_FILTER_GLOSSY: glossy = f32(); break;
        case GZ_SESSION_ADAPTIVE: thr = f32(); min_spp = i32(); break;
        case GZ_SESSION_CLAMP: clamp = f32(); break;
        case GZ_SESSION_BOUNCES: bounces[0] = i32(); bounces[1] = i32(); bounces[2] = i32(); break;
        case GZ_SESSION_INTEGRATOR: integrator = i32(); break;
        case GZ_SESSION_RESOLVE: denoise = i32(); break;
        default: fprintf(stderr, "gzreplay: unknown op %u\n", op); return 2;
        }
    }
    if (argc > 3) spp = atoi(argv[3]);
    double t_scene = now();
    session = gz_session_new(renderer, scene, spp, depth);
    if (!session) { fprintf(stderr, "gzreplay: no session\n"); return 1; }
    t_session = now();
    gz_session_set_samples(session, spp, depth);
    gz_session_set_lens(session, rad, focus);
    if (getenv("GZ_FILTER_GLOSSY")) glossy = atof(getenv("GZ_FILTER_GLOSSY"));
    gz_session_set_filter_glossy(session, glossy);
    gz_session_set_adaptive(session, thr, min_spp);
    if (getenv("GZ_CLAMP")) clamp = atof(getenv("GZ_CLAMP"));
    gz_session_set_clamp(session, clamp);
    if (getenv("GZ_BOUNCES")) sscanf(getenv("GZ_BOUNCES"), "%d,%d,%d", &bounces[0], &bounces[1], &bounces[2]);
    if (bounces[0] >= 0) gz_session_set_bounces(session, bounces[0], bounces[1], bounces[2]);
    if (getenv("GZ_INTEGRATOR")) integrator = atoi(getenv("GZ_INTEGRATOR"));
    gz_session_set_integrator(session, integrator);
    if (gz_session_set_camera(session, w2c, fov, w, h) != 0) return 1;
    int done = gz_session_render(session, spp);
    double t_render = now();
    float *rgb = malloc(sizeof(float) * 3 * (size_t)w * h);
    if (getenv("GZ_DENOISE")) denoise = atoi(getenv("GZ_DENOISE"));   // override the recorded choice
    if (done < 0 || gz_session_resolve(session, rgb, denoise) != 0) return 1;
    double t_resolve = now();
    FILE *o = fopen(argv[2], "wb");
    fprintf(o, "PF\n%d %d\n-1.0\n", w, h);
    for (int y = h - 1; y >= 0; --y) fwrite(rgb + 3 * (size_t)y * w, 4, 3 * (size_t)w, o);
    fclose(o);
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    printf("gzreplay: %dx%d, %d spp: scene %.2f s, session %.2f s, render %.2f s, resolve %.2f s, total %.2f s, peak RSS %ld MB\n",
           w, h, done, t_scene - t0, t_session - t_scene, t_render - t_session, t_resolve - t_render, t_resolve - t0, ru.ru_maxrss / 1024);
    gz_session_free(session);
    return 0;
}
