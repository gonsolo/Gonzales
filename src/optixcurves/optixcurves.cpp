// optixcurves.h on OptiX 7+: chunks of segments in compacted GASes under one IAS, the programs of optixcurves.cu
// (embedded as optixcurves_ir.h by the build).
#include "optixcurves.h"
#include "optixcurves_params.h"
#include "optixcurves_ir.h"
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr unsigned int kBuildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;   // the built-in intersector must repeat them
constexpr uint32_t kChunk = 1u << 22;                                       // segments a GAS: ~0.5 KB build scratch each

template <typename T> struct Record {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct Oxc {
    OptixDeviceContext ctx = nullptr;
    OptixModule module = nullptr, is = nullptr;
    OptixProgramGroup groups[4] = {};
    OptixPipeline closest = nullptr, shadow = nullptr;
    CUdeviceptr sbt_mem = 0, params = 0, ias = 0, scratch = 0;
    size_t scratch_size = 0;
    std::vector<CUdeviceptr> gases;
    size_t bytes = 0;
    OptixShaderBindingTable sbt_closest = {}, sbt_shadow = {};
    OxcParams p = {};
};

void *g_active = nullptr;

bool ok(OptixResult r, const char *what) {
    if (r != OPTIX_SUCCESS) fprintf(stderr, "optixcurves: %s: %s\n", what, optixGetErrorString(r));
    return r == OPTIX_SUCCESS;
}
bool ok(CUresult r, const char *what) {
    if (r != CUDA_SUCCESS) fprintf(stderr, "optixcurves: %s: CUDA error %d\n", what, int(r));
    return r == CUDA_SUCCESS;
}

OptixPipelineCompileOptions compile_options() {
    OptixPipelineCompileOptions po = {};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    po.numPayloadValues = 2;
    po.numAttributeValues = 2;
    po.pipelineLaunchParamsVariableName = "params";
    po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR;
    return po;
}

bool make_pipeline(Oxc &o, int raygen, OptixPipeline &pipe, OptixShaderBindingTable &sbt, CUdeviceptr records) {
    OptixPipelineCompileOptions po = compile_options();
    OptixPipelineLinkOptions lo = {};
    lo.maxTraceDepth = 1;
    OptixProgramGroup g[3] = {o.groups[raygen], o.groups[2], o.groups[3]};
    if (!ok(optixPipelineCreate(o.ctx, &po, &lo, g, 3, nullptr, nullptr, &pipe), "pipeline")) return false;
    if (!ok(optixPipelineSetStackSize(pipe, 0, 0, 1024, 2), "stack size")) return false;
    sbt.raygenRecord = records + raygen * sizeof(Record<int>);
    sbt.missRecordBase = records + 2 * sizeof(Record<int>);
    sbt.missRecordStrideInBytes = sizeof(Record<int>);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = records + 3 * sizeof(Record<int>);
    sbt.hitgroupRecordStrideInBytes = sizeof(Record<int>);
    sbt.hitgroupRecordCount = 1;
    return true;
}

bool launch(Oxc &o, OptixPipeline pipe, const OptixShaderBindingTable &sbt, uint32_t n, CUstream s) {
    if (n == 0) return true;
    return ok(cuMemcpyHtoDAsync(o.params, &o.p, sizeof(OxcParams), s), "params") &&
           ok(optixLaunch(pipe, s, o.params, sizeof(OxcParams), &sbt, n, 1, 1), "launch");
}

}  // namespace

extern "C" void *oxc_create(void) {
    CUcontext cu = nullptr;
    if (cuCtxGetCurrent(&cu) != CUDA_SUCCESS || !cu || optixInit() != OPTIX_SUCCESS) return nullptr;
    auto *o = new Oxc();
    OptixDeviceContextOptions co = {};
    OptixModuleCompileOptions mo = {};
    OptixPipelineCompileOptions po = compile_options();
    OptixBuiltinISOptions iso = {};
    iso.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
    iso.buildFlags = kBuildFlags;
    OptixProgramGroupDesc d[4] = {};
    d[0].kind = d[1].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    d[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    d[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    OptixProgramGroupOptions go = {};
    Record<int> rec[4] = {};
    bool good = ok(optixDeviceContextCreate(cu, &co, &o->ctx), "context") &&
                ok(optixModuleCreate(o->ctx, &mo, &po, (const char *)oxc_ir, sizeof(oxc_ir), nullptr, nullptr, &o->module), "module") &&
                ok(optixBuiltinISModuleGet(o->ctx, &mo, &po, &iso, &o->is), "curve intersector");
    if (good) {
        d[0].raygen = {o->module, "__raygen__closest"};
        d[1].raygen = {o->module, "__raygen__shadow"};
        d[2].miss = {o->module, "__miss__"};
        d[3].hitgroup.moduleCH = o->module;
        d[3].hitgroup.entryFunctionNameCH = "__closesthit__";
        d[3].hitgroup.moduleIS = o->is;
        good = ok(optixProgramGroupCreate(o->ctx, d, 4, &go, nullptr, nullptr, o->groups), "program groups");
    }
    for (int k = 0; good && k < 4; k++) good = ok(optixSbtRecordPackHeader(o->groups[k], &rec[k]), "SBT record");
    good = good && ok(cuMemAlloc(&o->sbt_mem, sizeof(rec)), "SBT") && ok(cuMemcpyHtoD(o->sbt_mem, rec, sizeof(rec)), "SBT") &&
           ok(cuMemAlloc(&o->params, sizeof(OxcParams)), "params") &&
           make_pipeline(*o, 0, o->closest, o->sbt_closest, o->sbt_mem) && make_pipeline(*o, 1, o->shadow, o->sbt_shadow, o->sbt_mem);
    if (!good) {
        oxc_destroy(o);
        return nullptr;
    }
    o->p.tmin = 1e-4f;                                          // curves.intersect_curve's
    return o;
}

extern "C" int oxc_build(void *h, uint64_t verts, uint64_t radii, uint32_t n, uint64_t refs, uint64_t curves, void *stream) {
    auto &o = *(Oxc *)h;
    CUstream s = (CUstream)stream;
    o.p.refs = (const int *)refs;
    o.p.curves = (const void *)curves;
    std::vector<uint32_t> local(std::min(n, kChunk));
    for (uint32_t i = 0; i < local.size(); i++) local[i] = 2 * i;
    CUdeviceptr index = 0, size = 0;
    if (!ok(cuMemAlloc(&index, 4 * std::max<size_t>(local.size(), 1)), "index") || !ok(cuMemAlloc(&size, 8), "size") ||
        !ok(cuMemcpyHtoD(index, local.data(), 4 * local.size()), "index"))
        return 1;
    std::vector<OptixInstance> inst;
    int err = 0;
    for (uint32_t a = 0; a < n && !err; a += kChunk) {
        uint32_t k = std::min(kChunk, n - a);
        CUdeviceptr v = verts + 24ull * a, w = radii + 8ull * a, temp = 0, out = 0, compact = 0;
        OptixBuildInput bi = {};
        bi.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
        bi.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
        bi.curveArray.numPrimitives = k;
        bi.curveArray.vertexBuffers = &v;
        bi.curveArray.numVertices = 2 * k;
        bi.curveArray.vertexStrideInBytes = 12;
        bi.curveArray.widthBuffers = &w;
        bi.curveArray.widthStrideInBytes = 4;
        bi.curveArray.indexBuffer = index;
        bi.curveArray.indexStrideInBytes = 4;
        bi.curveArray.flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        bi.curveArray.primitiveIndexOffset = a;
        OptixAccelBuildOptions ao = {};
        ao.buildFlags = kBuildFlags;
        ao.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes bs;
        OptixTraversableHandle gas;
        OptixAccelEmitDesc em = {size, OPTIX_PROPERTY_TYPE_COMPACTED_SIZE};
        size_t csize = 0;
        err = !(ok(optixAccelComputeMemoryUsage(o.ctx, &ao, &bi, 1, &bs), "GAS sizes") &&
                ok(cuMemAlloc(&temp, bs.tempSizeInBytes), "GAS scratch") && ok(cuMemAlloc(&out, bs.outputSizeInBytes), "GAS") &&
                ok(optixAccelBuild(o.ctx, s, &ao, &bi, 1, temp, bs.tempSizeInBytes, out, bs.outputSizeInBytes, &gas, &em, 1), "GAS build") &&
                ok(cuMemcpyDtoHAsync(&csize, size, 8, s), "compacted size") && ok(cuStreamSynchronize(s), "GAS build") &&
                ok(cuMemAlloc(&compact, csize), "compacted GAS") &&
                ok(optixAccelCompact(o.ctx, s, gas, compact, csize, &gas), "compaction") && ok(cuStreamSynchronize(s), "compaction"));
        if (temp) cuMemFree(temp);
        if (out) cuMemFree(out);
        if (err) {
            if (compact) cuMemFree(compact);
            break;
        }
        o.gases.push_back(compact);
        o.bytes += csize;
        OptixInstance in = {};
        in.transform[0] = in.transform[5] = in.transform[10] = 1;
        in.visibilityMask = 255;
        in.traversableHandle = gas;
        inst.push_back(in);
    }
    cuMemFree(index);
    cuMemFree(size);
    if (err) return 1;
    CUdeviceptr dinst = 0, temp = 0;
    OptixBuildInput ii = {};
    ii.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ii.instanceArray.numInstances = inst.size();
    OptixAccelBuildOptions ao = {};
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs;
    err = !(ok(cuMemAlloc(&dinst, inst.size() * sizeof(OptixInstance)), "instances") &&
            ok(cuMemcpyHtoD(dinst, inst.data(), inst.size() * sizeof(OptixInstance)), "instances"));
    ii.instanceArray.instances = dinst;
    err = err || !(ok(optixAccelComputeMemoryUsage(o.ctx, &ao, &ii, 1, &bs), "IAS sizes") &&
                   ok(cuMemAlloc(&temp, bs.tempSizeInBytes), "IAS scratch") && ok(cuMemAlloc(&o.ias, bs.outputSizeInBytes), "IAS") &&
                   ok(optixAccelBuild(o.ctx, s, &ao, &ii, 1, temp, bs.tempSizeInBytes, o.ias, bs.outputSizeInBytes, &o.p.handle, nullptr, 0), "IAS build") &&
                   ok(cuStreamSynchronize(s), "IAS build"));
    if (temp) cuMemFree(temp);
    if (dinst) cuMemFree(dinst);
    o.bytes += bs.outputSizeInBytes;
    return err;
}

extern "C" int oxc_trace_closest(void *h, uint64_t rays, uint64_t results, uint32_t n, void *stream) {
    auto &o = *(Oxc *)h;
    o.p.rays = (const float *)rays;
    o.p.results = (unsigned int *)results;
    return !launch(o, o.closest, o.sbt_closest, n, (CUstream)stream);
}

extern "C" int oxc_trace_shadow(void *h, uint64_t rays, uint64_t list, uint64_t counts, int32_t cap, int32_t slot, uint64_t blocked,
                                uint32_t n, void *stream) {
    auto &o = *(Oxc *)h;
    o.p.rays = (const float *)rays;
    o.p.list = (const int *)list;
    o.p.counts = (const int *)counts;
    o.p.cap = cap;
    o.p.slot = slot;
    o.p.blocked = (unsigned char *)blocked;
    return !launch(o, o.shadow, o.sbt_shadow, n, (CUstream)stream);
}

extern "C" uint64_t oxc_scratch(void *h, size_t bytes) {
    auto &o = *(Oxc *)h;
    if (bytes > o.scratch_size) {
        if (o.scratch) cuMemFree(o.scratch);
        o.scratch_size = 0;
        if (!ok(cuMemAlloc(&o.scratch, bytes), "scratch")) return o.scratch = 0;
        o.scratch_size = bytes;
    }
    return o.scratch;
}

extern "C" size_t oxc_bytes(void *h) { return h ? ((Oxc *)h)->bytes : 0; }

extern "C" void oxc_destroy(void *h) {
    if (!h) return;
    auto *o = (Oxc *)h;
    if (g_active == h) g_active = nullptr;
    if (o->closest) optixPipelineDestroy(o->closest);
    if (o->shadow) optixPipelineDestroy(o->shadow);
    for (auto g : o->groups)
        if (g) optixProgramGroupDestroy(g);
    if (o->module) optixModuleDestroy(o->module);
    for (auto g : o->gases) cuMemFree(g);
    if (o->ias) cuMemFree(o->ias);
    if (o->sbt_mem) cuMemFree(o->sbt_mem);
    if (o->params) cuMemFree(o->params);
    if (o->scratch) cuMemFree(o->scratch);
    if (o->ctx) optixDeviceContextDestroy(o->ctx);
    delete o;
}

extern "C" void oxc_set_active(void *h) { g_active = h; }
extern "C" void *oxc_active(void) { return g_active; }
