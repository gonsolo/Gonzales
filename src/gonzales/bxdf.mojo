from .geometry import RGB, MatKind

# ── BxDF flags ────────────────────────────────────────────────────────────────
struct BxDFFlags:
    comptime delta    = Int8(1)   # Dirac delta (perfect mirror / glass)
    comptime diffuse  = Int8(2)   # cosine lobe
    comptime glossy   = Int8(4)   # GGX microfacet lobe
    comptime reflect  = Int8(8)
    comptime transmit = Int8(16)

@always_inline
def bxdf_is_delta(flags: Int8) -> Bool:
    return (Int(flags) & Int(BxDFFlags.delta)) != 0

# ── BxDF sample result ────────────────────────────────────────────────────────
# Convention:
#   Delta BSDFs:     f = throughput multiplier, pdf = 1.0
#                    integrator does: throughput *= f  (no cos/pdf division)
#   Non-delta BSDFs: f = BxDF value f(wo,wi), pdf = sampling pdf
#                    integrator does: throughput *= f * cos_wi / pdf
@fieldwise_init
struct BxDFSample(TrivialRegisterPassable):
    var wi:       SIMD[DType.float32, 3]  # sampled incident direction (world-space)
    var f:        RGB                      # BxDF value or throughput multiplier
    var pdf:      Float32                  # sampling PDF; 1.0 for delta BSDFs
    var flags:    Int8                     # BxDFFlags bitmask
    var is_valid: Int8                     # 0 = degenerate (TIR, back-face, etc.)
    var _pad0:    Int8
    var _pad1:    Int8

# ── Local geometry at a surface hit ──────────────────────────────────────────
# Populated once per bounce by _build_geom_context_minimal / _build_geom_context_full.
# Passed by value to all BxDF and NEE functions — LLVM/NVPTX eliminates the
# struct when all fields are inlined at @always_inline call sites.
@fieldwise_init
struct GeomContext(TrivialRegisterPassable):
    var normal:     SIMD[DType.float32, 3]  # shading normal, faceforward to geo_normal
    var geo_normal: SIMD[DType.float32, 3]  # geometric normal, faceforward to wo
    var hit_point:  SIMD[DType.float32, 3]  # world-space surface hit point
    var wo:         SIMD[DType.float32, 3]  # outgoing direction = -ray.direction
    var tangent:    SIMD[DType.float32, 3]  # shading tangent (Frisvad frame)
    var bitangent:  SIMD[DType.float32, 3]  # shading bitangent
    var alb:        RGB                      # surface albedo (texture or mat.albedo)
    var pixel_uv:   Float32                  # mip LOD footprint (0 on CPU path)

# ── Pre-drawn Sobol samples for one bounce ────────────────────────────────────
# Drawn by _draw_sobol_8 at the start of each non-delta bounce.
# Named fields replace the current unnamed u_light, u_bary1, ... parameters.
@fieldwise_init
struct SobolSamples8(TrivialRegisterPassable):
    var light: Float32   # light CDF selection
    var bary1: Float32   # area light barycentric r1
    var bary2: Float32   # area light barycentric r2
    var env1:  Float32   # env-map u1 (BSDF-MIS cosine sample / CDF u)
    var env2:  Float32   # env-map u2 (BSDF-MIS cosine sample / CDF v)
    var scat1: Float32   # BSDF scatter u1
    var scat2: Float32   # BSDF scatter u2
    var rr:    Float32   # Russian roulette
