// Curves on the RT hardware through OptiX: the segments (locally-linear curve pieces) as OptiX round linear curves,
// traced in batches on the caller's ray buffers next to the RT-core triangle rays (rtcore.h). Driver API only; device
// pointers are CUdeviceptr values, streams CUstream. Every call but oxc_create returns 0 on success.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void *oxc_create(void);       // on the current CUDA context; null without OptiX
// n segments: verts 6 floats (both ends), radii 2 floats; refs 3 ints (curve, PrimId.id2, material), or 0 when segment
// k is curve k's one piece in its own material, and curves (the Curve array) stay the caller's and must outlive the handle. verts and radii may be freed after the call.
int oxc_build(void *h, uint64_t verts, uint64_t radii, uint32_t n, uint64_t refs, uint64_t curves, void *stream);
// Closest: every ray of rays/results (the RT-core layouts) not decided elsewhere gets its curve hit in front of the
// triangle hit written into results.
int oxc_trace_closest(void *h, uint64_t rays, uint64_t results, uint32_t n, void *stream);
// Shadow: the rays named by one slot's survivor list (launched for up to n) get blocked[ray] = 1 when a curve is in the
// way, 0 otherwise.
int oxc_trace_shadow(void *h, uint64_t rays, uint64_t list, uint64_t counts, int32_t cap, int32_t slot, uint64_t blocked,
                     uint32_t n, void *stream);
uint64_t oxc_scratch(void *h, size_t bytes);   // a device buffer of at least `bytes`, kept by the handle; 0 on failure
size_t oxc_bytes(void *h);    // the acceleration structures
void oxc_destroy(void *h);
void oxc_set_active(void *h); // the session's handle, for the trace call sites
void *oxc_active(void);

#ifdef __cplusplus
}
#endif
