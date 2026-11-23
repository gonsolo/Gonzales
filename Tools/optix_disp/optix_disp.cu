// OptiX programs: the sub-squares as custom primitives, the march as their intersection program, the hit's patch and
// (s, t) written per pixel for the prototype to shade (HITS=file).
#include <optix.h>
#include "optix_disp_params.h"

extern "C" {
__constant__ OdParams params;
}

extern "C" __global__ void __raygen__() {
        uint3 ix = optixGetLaunchIndex();
        int x = int(ix.x), y = int(ix.y);
        gz::V3 d = params.cam.dir(x, y);
        unsigned int p0 = __float_as_uint(1e30f), p1 = 0xffffffffu, p2 = 0, p3 = 0, p4 = 0;
        optixTrace(params.handle, make_float3(params.cam.eye.x, params.cam.eye.y, params.cam.eye.z), make_float3(d.x, d.y, d.z), 0.f, 1e30f, 0.f,
                   255, OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0, p1, p2, p3, p4);
        size_t k = size_t(y) * params.cam.w + x;
        params.hits[k] = {__uint_as_float(p0), int32_t(p1), __uint_as_float(p2), __uint_as_float(p3)};
        params.steps[k] = p4;
}

extern "C" __global__ void __intersection__march() {
        float3 o = optixGetWorldRayOrigin(), d = optixGetWorldRayDirection();
        gz::MHit m = gz::intersect_march(params.s, int(optixGetPrimitiveIndex()), gz::v3(o.x, o.y, o.z), gz::v3(d.x, d.y, d.z), optixGetRayTmax());
        optixSetPayload_4(optixGetPayload_4() + unsigned(m.steps));
        if (m.hit)
                optixReportIntersection(m.t, 0, __float_as_uint(m.s), __float_as_uint(m.u));
}

extern "C" __global__ void __closesthit__() {
        optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
        optixSetPayload_1(unsigned(gz::prim_patch(params.s, int(optixGetPrimitiveIndex())).patch));
        optixSetPayload_2(optixGetAttribute_0());
        optixSetPayload_3(optixGetAttribute_1());
}

extern "C" __global__ void __miss__() {}
