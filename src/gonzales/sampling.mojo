from std.math import sqrt, log, exp, cos, sin, atan2, acos
from std.memory import alloc
from .geometry import Vec3f, Ray_C, Point3f, dot, Frame, PI, TWO_PI, INV_PI
from .spectrum import SampledWavelengths, sample_wavelengths_uniform

# ── Multiple-importance sampling ───────────────────────────────────────────────
# See: docs/04_sampling.md — Multiple Importance Sampling

@always_inline
# <<listing: power_heuristic>>
def power_heuristic(pdf_f: Float32, pdf_g: Float32) -> Float32:
    """Balance heuristic with β=2 (Veach 1997).
    Combines two sampling strategies f and g into a single weight:
        w(f) = f² / (f² + g²)
    See: docs/04_sampling.md — Power Heuristic.
    """
    var f2 = pdf_f * pdf_f
    var g2 = pdf_g * pdf_g
    var denom = f2 + g2
    if denom <= Float32(0.0):
        return Float32(0.0)
    return f2 / denom

# ── Directional sampling ───────────────────────────────────────────────────────
# See: docs/04_sampling.md — Directional Distributions
# <</listing>>

@always_inline
# <<listing: sample_cosine_hemisphere>>
def sample_cosine_hemisphere(u1: Float32, u2: Float32) -> Vec3f:
    """Cosine-weighted hemisphere sampling via Malley's method.
    Maps uniform (u1,u2) ∈ [0,1)² to a direction proportional to cos θ.
    PDF = cos(θ) / π.  z = cos θ (hemisphere axis).
    See: docs/04_sampling.md — Cosine-Weighted Hemisphere.
    """
    var r     = sqrt(u1)
    var phi   = TWO_PI * u2
    var x     = r * cos(phi)
    var y     = r * sin(phi)
    var z_sq  = Float32(1.0) - u1
    var z     = sqrt(z_sq if z_sq > Float32(0.0) else Float32(0.0))
# <</listing>>
    return Vec3f(x, y, z)

@always_inline
def sample_cosine_hemisphere_world(
    u1: Float32, u2: Float32,
    normal: SIMD[DType.float32, 3],
) -> Tuple[SIMD[DType.float32, 3], Float32]:
    """Cosine-weighted hemisphere sample in world space, coupled with its pdf.
    Returns (direction, pdf) where pdf = cos(θ)/π — they cannot drift apart.
    Uses a Duff et al. (2017) orthonormal frame from `normal`.
    """
    var r   = sqrt(u1)
    var phi = TWO_PI * u2
    var x   = r * cos(phi)
    var y   = r * sin(phi)
    var z2  = Float32(1.0) - u1
    var z   = sqrt(z2 if z2 > Float32(0.0) else Float32(0.0))
    var pdf = z / PI  # cos(θ)/π — coupled, cannot diverge from the sampled direction

    # Duff et al. 2017 orthonormal frame
    var frame = Frame.from_z(Vec3f(normal[0], normal[1], normal[2]))
    var tangent   = SIMD[DType.float32, 3](frame.x.x, frame.x.y, frame.x.z)
    var bitangent = SIMD[DType.float32, 3](frame.y.x, frame.y.y, frame.y.z)

    var dir  = tangent * x + bitangent * y + normal * z
    var dlen = dot(dir, dir)
    if dlen > Float32(0.0):
        dir = dir * (Float32(1.0) / sqrt(dlen))
    return Tuple[SIMD[DType.float32, 3], Float32](dir, pdf)

@always_inline
# <<listing: sample_ggx_vndf>>
def sample_ggx_vndf(
    wo_local: Vec3f,           # outgoing direction in the stretched frame
    alpha_x: Float32,          # GGX roughness along tangent
    alpha_y: Float32,          # GGX roughness along bitangent
    u1: Float32, u2: Float32,  # uniform random numbers
) -> Vec3f:
    """Visible Normal Distribution Function sampling (Heitz 2018).
    Returns a half-vector wh in the anisotropic local frame such that
    wh is sampled proportional to D(wh) |wh·wo| / (wh normalisation).
    The caller reflects wo around wh to get the outgoing direction.

    Reference: E. Heitz, "Sampling the GGX Distribution of Visible Normals",
    JCGT 7(4), 2018. https://jcgt.org/published/0007/04/01/
    See: docs/05_reflection_models.md — GGX VNDF Sampling.
    """
    # 1. Stretch incoming direction by (alpha_x, alpha_y)
    var wos = Vec3f(wo_local.x * alpha_x, wo_local.y * alpha_y, wo_local.z)
    var wos_len = wos.length()
    var vh = wos * (Float32(1.0) / wos_len) if wos_len > Float32(0.0) else Vec3f(0.0, 0.0, 1.0)

    # 2. Orthonormal basis around the stretched half-vector (Duff et al. 2017)
    var vh_frame = Frame.from_z(vh)
    var bt1 = vh_frame.x
    var bt2 = vh_frame.y

    # 3. Sample point on visible hemisphere disk
    var r_disk = sqrt(u1)
    var phi    = TWO_PI * u2
    var tx     = r_disk * cos(phi)
    var ty_raw = r_disk * sin(phi)
    var s_corr = Float32(0.5) * (Float32(1.0) + vh.z)
    var ty     = tx * sqrt(Float32(1.0) - s_corr) + ty_raw * sqrt(s_corr)
    var tz_sq  = Float32(1.0) - tx * tx - ty * ty
    var tz     = sqrt(tz_sq if tz_sq > Float32(0.0) else Float32(0.0))
    var nh_local = bt1 * tx + bt2 * ty + vh * tz

    # 4. Unstretch to get the half-vector in the anisotropic frame
    var wh = Vec3f(alpha_x * nh_local.x, alpha_y * nh_local.y,
                   max(Float32(0.0), nh_local.z))
    return wh.normalize() if wh.length_sq() > Float32(0.0) else Vec3f(0.0, 0.0, 1.0)
# <</listing>>

# ── ZSobolSampler + GaussianFilter ──────────────────────────────────────────

@fieldwise_init
struct TileSamplerParams_C(TrivialRegisterPassable):
    var sobolMatrices: UnsafePointer[UInt32, MutAnyOrigin]
    var rngSeed: UInt64
    var sobolSeed: Int32
    var log2SamplesPerPixel: Int32
    var nBase4Digits: Int32
    var samplesPerPixel: Int32
    var filterSigma: Float32
    var filterSupportX: Float32
    var filterSupportY: Float32
    var filterNormX: Float32
    var filterNormY: Float32
    var filterWeight: Float32
    var filterType: Int32   # 0=gaussian 1=triangle 2=box
    var sampleIndexOffset: Int32  # shift into the Sobol sequence (for multi-pass)

@always_inline
def reverse_bits32(v_in: UInt32) -> UInt32:
    var v = v_in
    v = ((v >> 1) & UInt32(0x55555555)) | ((v & UInt32(0x55555555)) << 1)
    v = ((v >> 2) & UInt32(0x33333333)) | ((v & UInt32(0x33333333)) << 2)
    v = ((v >> 4) & UInt32(0x0f0f0f0f)) | ((v & UInt32(0x0f0f0f0f)) << 4)
    v = ((v >> 8) & UInt32(0x00ff00ff)) | ((v & UInt32(0x00ff00ff)) << 8)
    return (v >> 16) | (v << 16)

@always_inline
def fast_owen_scramble(value_in: UInt32, seed: UInt32) -> UInt32:
    var v = reverse_bits32(value_in)
    v ^= v * UInt32(0x3d20adea)
    v += seed
    v *= (seed >> 16) | UInt32(1)
    v ^= v * UInt32(0x05526c56)
    v ^= v * UInt32(0x53a22864)
    return reverse_bits32(v)

@always_inline
def mix_bits_u64(v: UInt64) -> UInt32:
    var v32 = UInt32(v & UInt64(0xFFFFFFFF))
    v32 ^= UInt32(v >> 32)
    v32 ^= v32 >> 16
    v32 *= UInt32(0x85ebca77)
    v32 ^= v32 >> 13
    v32 *= UInt32(0xc2b2ae35)
    v32 ^= v32 >> 16
    return v32

@always_inline
def encode_morton2(x: UInt32, y: UInt32) -> UInt64:
    var x64 = UInt64(x)
    var y64 = UInt64(y)
    x64 = (x64 | (x64 << 16)) & UInt64(0x0000FFFF0000FFFF)
    x64 = (x64 | (x64 << 8))  & UInt64(0x00FF00FF00FF00FF)
    x64 = (x64 | (x64 << 4))  & UInt64(0x0F0F0F0F0F0F0F0F)
    x64 = (x64 | (x64 << 2))  & UInt64(0x3333333333333333)
    x64 = (x64 | (x64 << 1))  & UInt64(0x5555555555555555)
    y64 = (y64 | (y64 << 16)) & UInt64(0x0000FFFF0000FFFF)
    y64 = (y64 | (y64 << 8))  & UInt64(0x00FF00FF00FF00FF)
    y64 = (y64 | (y64 << 4))  & UInt64(0x0F0F0F0F0F0F0F0F)
    y64 = (y64 | (y64 << 2))  & UInt64(0x3333333333333333)
    y64 = (y64 | (y64 << 1))  & UInt64(0x5555555555555555)
    return x64 | (y64 << 1)

# Compact permutation encoding: each of 24 permutations of {0,1,2,3} stored in one UInt8.
@always_inline
def sobol_perm_lookup(p_idx: Int, digit: Int) -> Int:
    var enc = InlineArray[UInt8, 24](fill=UInt8(0))
    enc[ 0]=27; enc[ 1]=30; enc[ 2]=39; enc[ 3]=45; enc[ 4]=57; enc[ 5]=54
    enc[ 6]=75; enc[ 7]=78; enc[ 8]=99; enc[ 9]=108; enc[10]=120; enc[11]=114
    enc[12]=147; enc[13]=156; enc[14]=135; enc[15]=141; enc[16]=177; enc[17]=180
    enc[18]=216; enc[19]=210; enc[20]=228; enc[21]=225; enc[22]=201; enc[23]=198
    return Int((Int(enc[p_idx]) >> (2 * (3 - digit))) & 3)

@always_inline
def sobol_get_sample_index(
    morton_idx: UInt64, dim: Int, log2spp: Int, n_base4: Int,
) -> UInt64:
    var sample_index: UInt64 = 0
    var pow2_samples = (log2spp & 1) == 1
    var last_digit = 1 if pow2_samples else 0
    var digit_index = n_base4 - 1
    while digit_index >= last_digit:
        var digit_shift = 2 * digit_index - (1 if pow2_samples else 0)
        var digit = Int((morton_idx >> UInt64(digit_shift)) & UInt64(3))
        var higher_digits = morton_idx >> UInt64(digit_shift + 2)
        var hash_val = mix_bits_u64(higher_digits ^ (UInt64(0x55555555) * UInt64(dim)))
        var p_idx = Int((hash_val >> 24) % UInt32(24))
        digit = sobol_perm_lookup(p_idx, digit)
        sample_index |= UInt64(digit) << UInt64(digit_shift)
        digit_index -= 1
    if pow2_samples:
        var digit = Int(morton_idx & UInt64(1))
        var hash_val = mix_bits_u64((morton_idx >> 1) ^ (UInt64(0x55555555) * UInt64(dim)))
        digit ^= Int(hash_val & UInt32(1))
        sample_index |= UInt64(digit)
    return sample_index

@always_inline
def sobol_sample(
    index: Int, dim: Int, seed: UInt32,
    matrices: UnsafePointer[UInt32, MutAnyOrigin],
) -> Float32:
    var acc: UInt32 = 0
    var cur = index
    var base = dim * 52
    for bit in range(52):
        if cur & 1 != 0:
            acc ^= matrices[base + bit]
        cur >>= 1
        if cur == 0:
            break
    var scrambled = fast_owen_scramble(acc, seed)
    return min(Float32(scrambled) * Float32(2.32830643653869628906e-10), Float32(0.9999999))

# Polynomial erfinv — no Newton refinement, sufficient accuracy for filter sampling.
@always_inline
def gaussian_erfinv(y: Float32) -> Float32:
    var abs_y = y if y >= Float32(0.0) else -y
    if abs_y <= Float32(0.7):
        var z = y * y
        var num = Float32(0.886226899) + z * (Float32(-1.645349621) + z * (Float32(0.914624893) + z * Float32(-0.140543331)))
        var den = Float32(1.0) + z * (Float32(-2.118377725) + z * (Float32(1.442710462) + z * (Float32(-0.329097515) + z * Float32(0.012229801))))
        return y * num / den
    elif abs_y < Float32(1.0):
        var z = sqrt(-log((Float32(1.0) - abs_y) / Float32(2.0)))
        var num = Float32(-1.970840454) + z * (Float32(-1.624906493) + z * (Float32(3.429567803) + z * Float32(1.641345311)))
        var den = Float32(1.0) + z * (Float32(3.543889200) + z * Float32(1.637067800))
        var sign_y = Float32(1.0) if y >= Float32(0.0) else Float32(-1.0)
        return sign_y * num / den
    else:
        return Float32(3.4e38) if y > Float32(0.0) else Float32(-3.4e38)

# Importance-sample a 1D Gaussian filter.
@always_inline
def gaussian_sample_1d(u: Float32, norm: Float32, sigma: Float32, radius: Float32) -> Float32:
    var u_s = (Float32(1.0) - norm) + u * (Float32(2.0) * norm - Float32(1.0))
    var x = sigma * sqrt(Float32(2.0)) * gaussian_erfinv(Float32(2.0) * u_s - Float32(1.0))
    return max(-radius, min(radius, x))

# erf via Abramowitz & Stegun 7.1.26 (max error ≤ 1.5e-7).
def _erf(x: Float32) -> Float32:
    var sign = Float32(1) if x >= Float32(0) else Float32(-1)
    var ax = x if x >= Float32(0) else -x
    var t = Float32(1) / (Float32(1) + Float32(0.3275911) * ax)
    var poly = ((((Float32(1.061405429) * t
                - Float32(1.453152027)) * t
               + Float32(1.421413741)) * t
              - Float32(0.284496736)) * t
             + Float32(0.254829592)) * t
    return sign * (Float32(1) - poly * exp(-ax * ax))


# Gaussian filter normalization.
def gaussian_norm(support: Float32, sigma: Float32) -> Float32:
    var x = support / (sigma * sqrt(Float32(2)))
    return Float32(0.5) * (Float32(1) + _erf(x))


# Hash pixel + sample index into a unique PCG (state, inc) pair.
@always_inline
def derive_pcg_seeds(px: Int32, py: Int32, si: Int32, seed: UInt64) -> Tuple[UInt64, UInt64]:
    var h = UInt64(px) * UInt64(2654435761) ^ UInt64(py) * UInt64(1664525) ^ UInt64(si) * UInt64(22695477) ^ seed
    h ^= h >> 30; h *= UInt64(0xbf58476d1ce4e5b9)
    h ^= h >> 27; h *= UInt64(0x94d049bb133111eb)
    h ^= h >> 31
    var state = h
    h ^= h >> 30; h *= UInt64(0xbf58476d1ce4e5b9)
    h ^= h >> 27
    return (state, h | UInt64(1))


@always_inline
def triangle_sample_1d(u: Float32, radius: Float32) -> Float32:
    if u < Float32(0.5):
        return radius * (sqrt(Float32(2.0) * u) - Float32(1.0))
    else:
        return radius * (Float32(1.0) - sqrt(Float32(2.0) * (Float32(1.0) - u)))

# ── Shared primary-ray generation ─────────────────────────────────────────────
# Encapsulates the duplicated Sobol + Gaussian-filter + rasterToCamera +
# cameraToWorld math used by both the CPU tile renderer and the two GPU
# gen_primary_rays kernels.
#
# Returns the world-space Ray_C and the PCG seed pair for the path.
# px/py are integer pixel coords; si is the sample index (Int32).
@always_inline
def gen_primary_ray_state(
    px: Int32, py: Int32, si: Int32,
    log2spp: Int, n_base4: Int,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed: UInt64,
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
    r2c: UnsafePointer[Float32, MutAnyOrigin],   # rasterToCamera  (16 Float32, col-major)
    c2w: UnsafePointer[Float32, MutAnyOrigin],   # cameraToWorld   (16 Float32, col-major)
    filter_norm_x: Float32, filter_sigma: Float32, filter_support_x: Float32,
    filter_norm_y: Float32, filter_support_y: Float32,
    filter_type: Int32 = Int32(0),
) -> Tuple[Ray_C, UInt64, UInt64, UInt64, SampledWavelengths]:
    """Shared Sobol + filter + camera-transform primary ray generator.
    Returns (ray, pcg_state, pcg_inc, sobol_idx, wavelengths).
    """
    var morton_base = encode_morton2(UInt32(px), UInt32(py)) << UInt64(log2spp)
    var morton_idx  = morton_base | UInt64(si)
    var sobol_idx   = sobol_get_sample_index(morton_idx, 0, log2spp, n_base4)
    var u0 = sobol_sample(Int(sobol_idx), 0, seed_dim0, sobol_matrices)
    var u1 = sobol_sample(Int(sobol_idx), 1, seed_dim1, sobol_matrices)
    var deltaX: Float32
    var deltaY: Float32
    if filter_type == Int32(1):
        deltaX = triangle_sample_1d(u0, filter_support_x)
        deltaY = triangle_sample_1d(u1, filter_support_y)
    elif filter_type == Int32(2):
        deltaX = (u0 - Float32(0.5)) * Float32(2.0) * filter_support_x
        deltaY = (u1 - Float32(0.5)) * Float32(2.0) * filter_support_y
    else:
        deltaX = gaussian_sample_1d(u0, filter_norm_x, filter_sigma, filter_support_x)
        deltaY = gaussian_sample_1d(u1, filter_norm_y, filter_sigma, filter_support_y)
    var filmX = Float32(px) + Float32(0.5) + deltaX
    var filmY = Float32(py) + Float32(0.5) + deltaY

    # rasterToCamera (column-major 4×4)
    var cx = r2c[0]*filmX + r2c[4]*filmY + r2c[12]
    var cy = r2c[1]*filmX + r2c[5]*filmY + r2c[13]
    var cz = r2c[2]*filmX + r2c[6]*filmY + r2c[14]
    var cw = r2c[3]*filmX + r2c[7]*filmY + r2c[15]
    if cw != Float32(0.0) and cw != Float32(1.0):
        cx /= cw; cy /= cw; cz /= cw
    var camLen = sqrt(cx*cx + cy*cy + cz*cz)
    if camLen > Float32(0.0):
        cx /= camLen; cy /= camLen; cz /= camLen

    # cameraToWorld rotation (upper-left 3×3 of col-major 4×4)
    var dx = c2w[0]*cx + c2w[4]*cy + c2w[8]*cz
    var dy = c2w[1]*cx + c2w[5]*cy + c2w[9]*cz
    var dz = c2w[2]*cx + c2w[6]*cy + c2w[10]*cz
    var dirLen = sqrt(dx*dx + dy*dy + dz*dz)
    if dirLen > Float32(0.0):
        dx /= dirLen; dy /= dirLen; dz /= dirLen

    var orgX = c2w[12]; var orgY = c2w[13]; var orgZ = c2w[14]
    var (pcg_state, pcg_inc) = derive_pcg_seeds(px, py, si, rng_seed)

    # Hero-wavelength sample — its own Sobol dimension (2), reserved once per
    # path at primary-ray generation (never resampled per bounce, so hero-
    # wavelength coherence holds across the whole path). Scrambling seed
    # mirrors the per-bounce-dimension convention (shading.mojo's
    # _draw_sobol_8: mix_bits_u64(pcgInc ^ dim)), not an externally threaded
    # seed_dim2, since this dimension has no film-position-style external
    # jitter parameters to interact with.
    var u_wave = sobol_sample(Int(sobol_idx), 2, mix_bits_u64(pcg_inc ^ UInt64(2)), sobol_matrices)
    var wavelengths = sample_wavelengths_uniform(u_wave)

    return (Ray_C(Point3f(orgX, orgY, orgZ), Vec3f(dx, dy, dz)), pcg_state, pcg_inc, sobol_idx, wavelengths)
