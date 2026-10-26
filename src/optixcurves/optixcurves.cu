// OptiX programs for the curve rays next to the RT cores (optixcurves.h). A segment is one locally-linear piece of a
// curve, an OptiX round linear curve; a hit is reported as curves.intersect_curve does (h across the fibre, v along it).
#include <optix.h>
#include "optixcurves_params.h"

extern "C" {
__constant__ OxcParams params;
}

struct Curve {          // curves.mojo
    float cp[12];
    float width0, width1;
    int material, n_pieces, flat;
};

struct CurveSeg {       // curves.mojo: a one-piece curve as its piece; params.curves has bit 0 set for these
    float p0[3], p1[3];
    float width0, width1;
    int material, flat;
};

static __device__ const CurveSeg *segs() {
    unsigned long long a = (unsigned long long)params.curves;
    return a & 1 ? (const CurveSeg *)(a - 1) : nullptr;
}

static __device__ float3 f3(const float *p) { return make_float3(p[0], p[1], p[2]); }
static __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __device__ float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
static __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static __device__ float3 cross(float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

static __device__ float3 bspline(const Curve &c, float t) {
    float t2 = t * t, t3 = t2 * t;
    float b0 = (1 - 3 * t + 3 * t2 - t3) / 6, b1 = (4 - 6 * t2 + 3 * t3) / 6, b2 = (1 + 3 * t + 3 * t2 - 3 * t3) / 6, b3 = t3 / 6;
    return f3(c.cp) * b0 + f3(c.cp + 3) * b1 + f3(c.cp + 6) * b2 + f3(c.cp + 9) * b3;
}

static __device__ float3 perp_axis(float3 t) {   // curves._curve_perp_axis
    float3 v = fabsf(t.y) < 0.9f ? make_float3(-t.z, 0, t.x) : make_float3(0, t.z, -t.y);
    float l = sqrtf(dot(v, v));
    return l > 1e-12f ? v * (1 / l) : make_float3(1, 0, 0);
}

// The curve, PrimId.id2 and material of segment k: from refs, or without them segment k is curve k's only piece.
static __device__ int3 seg_ref(unsigned int k) {
    if (params.refs) return make_int3(params.refs[3 * k], params.refs[3 * k + 1], params.refs[3 * k + 2]);
    const CurveSeg *sg = segs();
    return make_int3(int(k), 1, sg ? sg[k].material : ((const Curve *)params.curves)[k].material);
}

// (h, v) of a hit at distance t on segment k, as curves.intersect_curve computes them.
static __device__ float2 hit_hv(unsigned int k, float3 o, float3 d, float t) {
    int3 r = seg_ref(k);
    int piece = r.y / 8;
    float n = 1, rad;
    float3 q0, q1;
    if (const CurveSeg *sg = segs()) {
        q0 = f3(sg[r.x].p0); q1 = f3(sg[r.x].p1);
        rad = (sg[r.x].width0 + sg[r.x].width1) * 0.25f;
    } else {
        const Curve &c = ((const Curve *)params.curves)[r.x];
        n = float(c.n_pieces);
        float t0 = piece / n, t1 = (piece + 1) / n;
        q0 = bspline(c, t0); q1 = bspline(c, t1);
        rad = (c.width0 + (c.width1 - c.width0) * (t0 + t1) * 0.5f) * 0.5f;
    }
    float3 axis = q1 - q0;
    float len = sqrtf(dot(axis, axis));
    float3 ad = axis * (1 / len);
    float3 rel = o + d * t - q0;
    float s = dot(rel, ad);
    float3 off = rel - ad * s;
    float3 wo = d * -1.0f;
    float3 wp = wo - ad * dot(wo, ad);
    float l = sqrtf(dot(wp, wp));
    float3 toward = l > 1e-6f ? wp * (1 / l) : cross(ad, perp_axis(ad));
    float h = fminf(1.0f, fmaxf(-1.0f, dot(off, cross(toward, ad)) / rad));
    return make_float2(h, (piece + fminf(1.0f, fmaxf(0.0f, s / len))) / n);
}

extern "C" __global__ void __raygen__closest() {
    const unsigned int i = optixGetLaunchIndex().x;
    const float *ray = params.rays + 8 * i;
    unsigned int *res = params.results + 8 * i;
    unsigned int flag = res[6];
    if (flag == 3) return;                                      // the software BVH decides this one
    float tmax = flag == 1 ? __uint_as_float(res[0]) : ray[7];
    float3 o = f3(ray), d = f3(ray + 4);
    unsigned int k = 0xffffffffu, tb = 0;
    optixTrace(params.handle, o, d, params.tmin, tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, k, tb);
    if (k == 0xffffffffu) return;
    float t = __uint_as_float(tb);
    float2 hv = hit_hv(k, o, d, t);
    int3 r = seg_ref(k);
    res[0] = __float_as_uint(t);
    res[1] = __float_as_uint(hv.x);
    res[2] = __float_as_uint(hv.y);
    res[3] = 0xffffffffu;                                       // instance -1
    res[4] = r.x;
    res[5] = r.y;
    res[7] = r.z;
    res[6] = 2;
}

extern "C" __global__ void __raygen__shadow() {
    const unsigned int j = optixGetLaunchIndex().x;
    if (int(j) >= params.counts[params.slot]) return;
    int e = params.list[params.slot == 0 ? j : params.cap - 1 - j];
    if (e < 0) return;                                          // bit 31: the software BVH decides this one
    const float *ray = params.rays + 8 * e;
    unsigned int k = 0, tb = 0;                                 // no closest hit: only a miss changes k
    optixTrace(params.handle, f3(ray), f3(ray + 4), params.tmin, ray[7], 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, k, tb);
    params.blocked[e] = k != 0xffffffffu;
}

extern "C" __global__ void __miss__() { optixSetPayload_0(0xffffffffu); }

extern "C" __global__ void __closesthit__() {
    optixSetPayload_0(optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}
