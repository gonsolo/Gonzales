# ReSTIR DI (Bitterli et al. 2020's RIS applied to direct light sampling),
# Phase 2 of docs/A2_restir_migration_plan.md. This file holds the payload
# type and pure target-function math only -- no dependency on shading.mojo
# (which would create a circular import, since shading.mojo is the caller).
# The shading-integration glue (candidate generation via the light sampler,
# the deferred-shadow-ray resolve) lives in shading.mojo itself, next to
# _shade_diffuse_nee, and imports from this file one-directionally.
#
# Scope actually implemented: plain RIS (2.1 payload, 2.2 initial
# candidates, 2.6 one deferred shadow ray for the winner, 2.7 MIS against
# BSDF sampling), temporal reuse (2.3), and spatial reuse (2.5) for the
# diffuse material's area-light NEE at the PRIMARY bounce only, CPU path
# tracer only (--restir, no --gpu/--guide/--sppm/--vcm combination yet).
# Both reuse passes need persistent per-pixel state across real
# interactive-viewer frames -- see project_restir_migration.md memory for
# the headless `--interactive-frames` debug path that makes them
# verifiable without a real window.

from std.math import sqrt
from .geometry import RGB, dot, INV_PI
from .reservoir import ReservoirState, reservoir_state_init

@fieldwise_init
struct DIReservoir(TrivialRegisterPassable):
    """ReSTIR DI's reservoir payload: the current best light candidate
    (light_idx < 0 means none found yet) plus its RIS bookkeeping
    (reservoir.mojo's ReservoirState -- w_sum/m/w)."""
    var light_idx:     Int32
    var sample_point:  SIMD[DType.float32, 3]
    var light_normal:  SIMD[DType.float32, 3]
    var le:            RGB
    var state:         ReservoirState

@always_inline
def di_reservoir_init() -> DIReservoir:
    return DIReservoir(
        light_idx=Int32(-1),
        sample_point=SIMD[DType.float32, 3](Float32(0)),
        light_normal=SIMD[DType.float32, 3](Float32(0)),
        le=RGB(Float32(0)),
        state=reservoir_state_init(),
    )

@always_inline
def di_target_pdf(
    hit_point: SIMD[DType.float32, 3], normal: SIMD[DType.float32, 3], alb: RGB,
    sample_point: SIMD[DType.float32, 3], light_normal: SIMD[DType.float32, 3], le: RGB,
) -> Float32:
    """RIS target function p̂(candidate): luminance of the UNSHADOWED
    Lambertian contribution BSDF x G x Le (2.1 -- visibility deliberately
    excluded here, resolved once for the reservoir's eventual winner).
    Scalar (luminance) target rather than per-channel, matching standard
    ReSTIR practice -- keeps w_sum/w single floats instead of per-channel
    reservoirs."""
    var to_light = sample_point - hit_point
    var dist_sq = dot(to_light, to_light)
    if dist_sq <= Float32(1e-8):
        return Float32(0.0)
    var dist = sqrt(dist_sq)
    var wi = to_light * (Float32(1.0) / dist)
    var cos_s = dot(normal, wi)
    var cos_l = -dot(light_normal, wi)
    if cos_s <= Float32(0.0) or cos_l <= Float32(0.0):
        return Float32(0.0)
    var g = cos_s * cos_l / dist_sq
    var contrib = alb * le * (INV_PI * g)
    return contrib.r * Float32(0.2126) + contrib.g * Float32(0.7152) + contrib.b * Float32(0.0722)

@fieldwise_init
struct ReservoirIO(TrivialRegisterPassable):
    """Bundles everything di_temporal_step needs beyond the current pixel's
    own hit data, so Phase 2.3/2.5's plumbing through the
    shade_core_cpu_nee -> ... -> di_temporal_step call chain (rendering.mojo,
    shading.mojo) is ONE trailing defaulted parameter instead of five. `read`
    is the previous frame's fully-resolved reservoirs (safe to read from any
    thread -- render_all_tiles parallelizes per-tile, so this must never be
    written to during the frame that reads it); `write` is this frame's
    target (each pixel written by exactly one thread, at interactive mode's
    1 spp/frame, so no two threads ever touch the same slot). pipeline.mojo's
    render_interactive swaps which physical buffer is `read` vs `write` each
    frame rather than copying. gbuf_* are Phase 0.3's world-pos/material-ID
    G-buffer extension (gbuf_normal/gbuf_depth already existed for the
    denoiser; gbuf_material_id is new, wired only when use_restir) -- used
    for 2.5's neighbor rejection (normal dot / depth delta / material match).
    All pointers default to `.unsafe_dangling()` and frame_w/frame_h to 0;
    di_temporal_step checks `_is_real_ptr`/`> 0` before ever touching them,
    so callers that don't pass a real ReservoirIO (the batch --restir path)
    get exactly the old "no temporal, no spatial reuse" behavior for free."""
    var read:  UnsafePointer[DIReservoir, MutAnyOrigin]
    var write: UnsafePointer[DIReservoir, MutAnyOrigin]
    var gbuf_normal:      UnsafePointer[Float32, MutAnyOrigin]
    var gbuf_depth:       UnsafePointer[Float32, MutAnyOrigin]
    var gbuf_material_id: UnsafePointer[Int32, MutAnyOrigin]
    var frame_w: Int32
    var frame_h: Int32

@always_inline
def reservoir_io_null() -> ReservoirIO:
    return ReservoirIO(
        read=UnsafePointer[DIReservoir, MutAnyOrigin].unsafe_dangling(),
        write=UnsafePointer[DIReservoir, MutAnyOrigin].unsafe_dangling(),
        gbuf_normal=UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
        gbuf_depth=UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
        gbuf_material_id=UnsafePointer[Int32, MutAnyOrigin].unsafe_dangling(),
        frame_w=Int32(0), frame_h=Int32(0),
    )
