// The march on NVIDIA's RT cores through OptiX: the prototype's surface dump as custom primitives, rendered from its
// camera; hits written for the prototype's shading (HITS=file).
//
//   optix_disp surface.dump optix_disp.optixir hits.bin [k]   (k: k x k rays a pixel, timing only; no hits written)
#include "disp_load.h"
#include "optix_disp_params.h"
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>

using namespace gz;

template <typename T> struct Record {
        alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
};

static void ck(OptixResult r, const char *what) {
        if (r != OPTIX_SUCCESS) {
                std::fprintf(stderr, "optix_disp: %s: %s\n", what, optixGetErrorString(r));
                std::exit(1);
        }
}
static void ck(CUresult r, const char *what) {
        if (r != CUDA_SUCCESS) {
                const char *n = "?";
                cuGetErrorName(r, &n);
                std::fprintf(stderr, "optix_disp: %s: %s\n", what, n);
                std::exit(1);
        }
}

static CUdeviceptr up(const void *p, size_t n) {
        CUdeviceptr d = 0;
        ck(cuMemAlloc(&d, std::max<size_t>(n, 4)), "alloc");
        if (n)
                ck(cuMemcpyHtoD(d, p, n), "upload");
        return d;
}

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv) {
        if (argc < 4) {
                std::fprintf(stderr, "usage: %s surface.dump optix_disp.optixir hits.bin\n", argv[0]);
                return 1;
        }
        Dump dm = load_dump(argv[1]);
        std::ifstream irf(argv[2], std::ios::binary);
        std::vector<char> ir((std::istreambuf_iterator<char>(irf)), std::istreambuf_iterator<char>());
        if (ir.empty()) {
                std::fprintf(stderr, "optix_disp: cannot read %s\n", argv[2]);
                return 1;
        }
        CUdevice dev;
        CUcontext cu;
        ck(cuInit(0), "cuInit");
        ck(cuDeviceGet(&dev, 0), "device");
        ck(cuDevicePrimaryCtxRetain(&cu, dev), "context");
        ck(cuCtxSetCurrent(cu), "context");
        char name[256];
        cuDeviceGetName(name, sizeof(name), dev);
        ck(optixInit(), "optixInit");
        OptixDeviceContext ctx;
        OptixDeviceContextOptions co = {};
        ck(optixDeviceContextCreate(cu, &co, &ctx), "context");

        // The surface on the device: one copy of the dump, the arrays inside it.
        CUdeviceptr all = up(dm.bytes.data(), dm.bytes.size());
        auto at = [&](size_t off) { return all + off; };
        Surf s = dm.s;
        s.sq = (const float *)at(dm.off_sq), s.face = (const int32_t *)at(dm.off_face), s.hdr = (const int32_t *)at(dm.off_hdr);
        s.chain_off = (const int32_t *)at(dm.off_coff), s.corners = (const float *)at(dm.off_corners), s.pyr = (const uint8_t *)at(dm.off_pyr);
        s.chain = (const uint8_t *)at(dm.off_chain);

        // One GAS of custom primitives: the sub-square boxes.
        CUdeviceptr aabbs = at(dm.off_boxes);
        unsigned int flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        OptixBuildInput bi = {};
        bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        bi.customPrimitiveArray.aabbBuffers = &aabbs;
        bi.customPrimitiveArray.numPrimitives = unsigned(dm.nprim);
        bi.customPrimitiveArray.strideInBytes = 24;
        bi.customPrimitiveArray.flags = &flags;
        bi.customPrimitiveArray.numSbtRecords = 1;
        OptixAccelBuildOptions ao = {};
        ao.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        ao.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes bs;
        ck(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs), "GAS sizes");
        CUdeviceptr temp, out, csz, gas_mem;
        ck(cuMemAlloc(&temp, bs.tempSizeInBytes), "GAS scratch");
        ck(cuMemAlloc(&out, bs.outputSizeInBytes), "GAS");
        ck(cuMemAlloc(&csz, 8), "size");
        OptixAccelEmitDesc em = {csz, OPTIX_PROPERTY_TYPE_COMPACTED_SIZE};
        OptixTraversableHandle gas;
        double tb = now_ms();
        ck(optixAccelBuild(ctx, 0, &ao, &bi, 1, temp, bs.tempSizeInBytes, out, bs.outputSizeInBytes, &gas, &em, 1), "GAS build");
        size_t compact = 0;
        ck(cuMemcpyDtoH(&compact, csz, 8), "compacted size");
        ck(cuMemAlloc(&gas_mem, compact), "compacted GAS");
        ck(optixAccelCompact(ctx, 0, gas, gas_mem, compact, &gas), "compaction");
        ck(cuCtxSynchronize(), "build");
        double build_ms = now_ms() - tb;
        cuMemFree(temp), cuMemFree(out), cuMemFree(csz);

        // Programs and pipeline.
        OptixModuleCompileOptions mo = {};
        OptixPipelineCompileOptions po = {};
        po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        po.numPayloadValues = 5;
        po.numAttributeValues = 2;
        po.pipelineLaunchParamsVariableName = "params";
        po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        OptixModule mod;
        char log[4096];
        size_t ll = sizeof(log);
        ck(optixModuleCreate(ctx, &mo, &po, ir.data(), ir.size(), log, &ll, &mod), "module");
        OptixProgramGroupDesc gd[3] = {};
        gd[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN, gd[0].raygen = {mod, "__raygen__"};
        gd[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS, gd[1].miss = {mod, "__miss__"};
        gd[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        gd[2].hitgroup.moduleCH = mod, gd[2].hitgroup.entryFunctionNameCH = "__closesthit__";
        gd[2].hitgroup.moduleIS = mod, gd[2].hitgroup.entryFunctionNameIS = "__intersection__march";
        OptixProgramGroupOptions go = {};
        OptixProgramGroup pg[3];
        ll = sizeof(log);
        ck(optixProgramGroupCreate(ctx, gd, 3, &go, log, &ll, pg), "program groups");
        OptixPipelineLinkOptions lo = {};
        lo.maxTraceDepth = 1;
        OptixPipeline pipe;
        ll = sizeof(log);
        ck(optixPipelineCreate(ctx, &po, &lo, pg, 3, log, &ll, &pipe), "pipeline");
        OptixStackSizes ss = {};
        for (auto &g : pg)
                ck(optixUtilAccumulateStackSizes(g, &ss, pipe), "stack sizes");
        unsigned dcT, dcS, cont;
        ck(optixUtilComputeStackSizes(&ss, 1, 0, 0, &dcT, &dcS, &cont), "stack sizes");
        ck(optixPipelineSetStackSize(pipe, dcT, dcS, cont, 1), "stack size");
        Record<int> rec[3] = {};
        for (int k = 0; k < 3; ++k)
                ck(optixSbtRecordPackHeader(pg[k], &rec[k]), "SBT record");
        CUdeviceptr sbt_mem = up(rec, sizeof(rec));
        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord = sbt_mem;
        sbt.missRecordBase = sbt_mem + sizeof(Record<int>), sbt.missRecordStrideInBytes = sizeof(Record<int>), sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = sbt_mem + 2 * sizeof(Record<int>), sbt.hitgroupRecordStrideInBytes = sizeof(Record<int>), sbt.hitgroupRecordCount = 1;

        int spp = argc > 4 ? std::max(1, std::atoi(argv[4])) : 1;
        int W = dm.cam.w * spp, H = dm.cam.h * spp;
        OdParams p = {};
        p.handle = gas;
        p.s = s;
        p.cam = {dm.cam.eye, dm.cam.fwd, dm.cam.right, dm.cam.up, dm.cam.tan_h, W, H};
        CUdeviceptr hits, steps, pd;
        ck(cuMemAlloc(&hits, sizeof(OdHit) * size_t(W) * H), "hits");
        ck(cuMemAlloc(&steps, 4 * size_t(W) * H), "steps");
        p.hits = (OdHit *)hits, p.steps = (unsigned *)steps;
        pd = up(&p, sizeof(p));
        double best = 1e30;
        for (int r = 0; r < 5; ++r) {   // the fastest of four after a warm-up, as the prototype times
                ck(cuCtxSynchronize(), "sync");
                double t0 = now_ms();
                ck(optixLaunch(pipe, 0, pd, sizeof(p), &sbt, unsigned(W), unsigned(H), 1), "launch");
                ck(cuCtxSynchronize(), "trace");
                if (r > 0)
                        best = std::min(best, now_ms() - t0);
        }
        std::vector<PixHit> hh(size_t(W) * H);
        std::vector<unsigned> st(size_t(W) * H);
        ck(cuMemcpyDtoH(hh.data(), hits, hh.size() * sizeof(PixHit)), "hits");
        ck(cuMemcpyDtoH(st.data(), steps, st.size() * 4), "steps");
        long nh = 0;
        double tot = 0;
        for (size_t k = 0; k < hh.size(); ++k)
                nh += hh[k].patch >= 0, tot += st[k];
        if (spp == 1)
                write_hits(argv[3], hh);
        std::printf("optix march (%s): %d x %d, %d sub-squares, GAS %zu KiB (build %.1f ms), surface %zu MiB, render %.3f ms, %.1f Mrays/s, "
                    "%ld hits, steps/ray %.2f\n",
                    name, W, H, dm.nprim, compact / 1024, build_ms, dm.bytes.size() >> 20, best, double(W) * H / (best * 1e3), nh, tot / (double(W) * H));
        return 0;
}
