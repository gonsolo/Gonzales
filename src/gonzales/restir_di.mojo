# ReSTIR DI (Bitterli et al. 2020's RIS applied to direct light sampling),
# Phase 2 of docs/A2_restir_migration_plan.md. This file holds the payload
# type and pure target-function math only -- no dependency on shading.mojo
# (which would create a circular import, since shading.mojo is the caller).
# The shading-integration glue (candidate generation via the light sampler,
# the deferred-shadow-ray resolve) lives in shading.mojo itself, next to
# _shade_diffuse_nee, and imports from this file one-directionally.
#
# Scope actually implemented this session: plain RIS (2.1 payload, 2.2
# initial candidates, 2.6 one deferred shadow ray for the winner, 2.7 MIS
# against BSDF sampling) for the diffuse material's area-light NEE at the
# PRIMARY bounce only, CPU path tracer only (--restir, no --gpu/--guide/
# --sppm/--vcm combination yet). Temporal reuse (2.3) and spatial reuse
# (2.5) need persistent per-pixel GPU state across real interactive-viewer
# frames and are NOT implemented here -- see project_restir_migration.md
# memory for why (no headless way to verify them to this session's bar).

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
