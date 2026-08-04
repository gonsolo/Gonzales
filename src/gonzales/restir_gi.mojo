# ReSTIR GI (Ouyang et al. 2021's path-reuse reservoir applied to indirect
# lighting), Phase 4 of docs/A2_restir_migration_plan.md. Mirrors
# restir_di.mojo's shape exactly: payload type + pure target-function math
# only, no dependency on shading.mojo. Phase 4.1 scope only -- the payload
# struct and its target function -- not yet wired into any render path
# (shading-integration glue, the reconnection shift's Jacobian application,
# and delta-BSDF rejection at the call site are separate, later commits).
#
# Design note on why Lo is reused unmodified across the shift (no BSDF
# re-evaluation at the reconnection vertex): ReSTIR GI treats the
# reconnection vertex x2's own outgoing radiance Lo (leaving x2 back toward
# whichever x1 reconnects to it) as approximately view-independent -- valid
# exactly when x2's BSDF is non-delta (a delta BSDF's response changes
# completely with viewing direction, which is precisely why Phase 4.3's
# delta-BSDF rejection exists: never choose a delta vertex as a reconnection
# point). Only the FIRST segment (x1 -> x2) is genuinely re-evaluated under
# the shift -- x1's own BSDF, cosine, and the inverse-square/Jacobian term --
# exactly analogous to how Phase 2's light reconnection re-evaluates x1's
# BSDF toward a fixed light point while reusing the light's own Le verbatim.

from std.math import sqrt
from .geometry import RGB, dot, INV_PI
from .reservoir import ReservoirState, reservoir_state_init

@fieldwise_init
struct GIReservoir(TrivialRegisterPassable):
    """ReSTIR GI's reservoir payload: a one-bounce-reconnection path suffix.

    `recon_point`/`recon_normal` are the reconnection vertex x2's world-space
    position and shading normal -- kept fixed in world space across temporal
    and spatial reuse, exactly like DIReservoir's `sample_point` is a light
    point fixed in world space. `lo` is the accumulated outgoing radiance
    leaving x2 back toward the path's ORIGINAL x1 (all further bounces'
    contribution already integrated) -- reused as-is under the shift (see
    module docstring). `recon_is_delta` records whether x2's own BSDF is a
    delta distribution; a reconnection vertex chosen there can never be
    validly reused from a different x1 (Phase 4.3), so any candidate/neighbor
    carrying this flag set must be rejected before ever computing a target
    pdf against it -- callers must check this themselves, since a pure
    geometry function can't know a material's BSDF kind.
    `valid` mirrors DIReservoir's `light_idx < 0` sentinel (0 = no winner
    yet; every other field is meaningless until a candidate is streamed)."""
    var recon_point:   SIMD[DType.float32, 3]
    var recon_normal:  SIMD[DType.float32, 3]
    var lo:            RGB
    var recon_is_delta: Int8
    var valid:          Int8
    var _pad0:          Int8
    var _pad1:          Int8
    var state:          ReservoirState

@always_inline
def gi_reservoir_init() -> GIReservoir:
    return GIReservoir(
        recon_point=SIMD[DType.float32, 3](Float32(0)),
        recon_normal=SIMD[DType.float32, 3](Float32(0)),
        lo=RGB(Float32(0)),
        recon_is_delta=Int8(0),
        valid=Int8(0),
        _pad0=Int8(0),
        _pad1=Int8(0),
        state=reservoir_state_init(),
    )

@always_inline
def gi_target_pdf(
    hit_point: SIMD[DType.float32, 3], normal: SIMD[DType.float32, 3], alb: RGB,
    recon_point: SIMD[DType.float32, 3], recon_normal: SIMD[DType.float32, 3], lo: RGB,
) -> Float32:
    """RIS target function p̂(candidate) for a one-bounce-reconnection GI
    sample: luminance of the UNSHADOWED Lambertian-at-x1 contribution
    f(x1) x G(x1,x2) x Lo(x2), where f(x1) = alb/pi (matching Phase 2's own
    diffuse-only-at-x1 scope -- x1's BSDF re-evaluation for non-diffuse
    materials is future work, same restriction Phase 2 started with). Same
    shape as di_target_pdf, with the reconnection vertex's position/normal/
    Lo standing in for a light sample's point/normal/Le -- visibility
    between x1 and x2 deliberately excluded here, to be resolved once for
    the reservoir's eventual winner, same as Phase 2.1."""
    var to_recon = recon_point - hit_point
    var dist_sq = dot(to_recon, to_recon)
    if dist_sq <= Float32(1e-8):
        return Float32(0.0)
    var dist = sqrt(dist_sq)
    var wi = to_recon * (Float32(1.0) / dist)
    var cos_x1 = dot(normal, wi)
    var cos_x2 = -dot(recon_normal, wi)
    if cos_x1 <= Float32(0.0) or cos_x2 <= Float32(0.0):
        return Float32(0.0)
    var g = cos_x1 * cos_x2 / dist_sq
    var contrib = alb * lo * (INV_PI * g)
    return contrib.r * Float32(0.2126) + contrib.g * Float32(0.7152) + contrib.b * Float32(0.0722)
