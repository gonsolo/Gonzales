from std.collections import Array
from std.math import sqrt, log, exp, cos, sin, atan2, acos
from std.memory.alloc import unsafe_alloc
from .geometry import Vec3f, Point3f, dot, cross, Frame, PI, TWO_PI, INV_PI
from .primitives import Ray
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
    normal: Vec3f,
) -> Tuple[Vec3f, Float32]:
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
    var tangent   = Vec3f(frame.x.x, frame.x.y, frame.x.z)
    var bitangent = Vec3f(frame.y.x, frame.y.y, frame.y.z)

    var dir  = tangent * x + bitangent * y + normal * z
    var dlen = dot(dir, dir)
    if dlen > Float32(0.0):
        dir = dir * (Float32(1.0) / sqrt(dlen))
    return Tuple[Vec3f, Float32](dir, pdf)

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

    # 2. Orthonormal basis around the stretched half-vector. This frame is
    # NOT free to be any basis perpendicular to vh, which is why the
    # general-purpose Frame.from_z that used to stand here was wrong: step 3
    # squashes the disk ANISOTROPICALLY, and the axis it squashes along has
    # to be the one pointing back toward the macro-normal. Heitz builds it
    # as T1 = normalize(cross(z, vh)) -- perpendicular to both -- leaving
    # T2 = cross(vh, T1) in the plane that contains z and vh, which is the
    # axis the squash means. A Duff frame is perpendicular to vh but at an
    # arbitrary azimuth, so it rotated the squash by an arbitrary angle.
    # Same blind spot as the squash formula below: at vh.z -> 1 the squash
    # vanishes, so both were invisible to every head-on furnace scene.
    var bt1: Vec3f
    var lensq = vh.x * vh.x + vh.y * vh.y
    if lensq > Float32(1e-12):
        var inv = Float32(1.0) / sqrt(lensq)
        bt1 = Vec3f(-vh.y * inv, vh.x * inv, Float32(0.0))
    else:
        bt1 = Vec3f(Float32(1.0), Float32(0.0), Float32(0.0))
    var bt2 = cross(vh, bt1)

    # 3. Sample point on visible hemisphere disk
    var r_disk = sqrt(u1)
    var phi    = TWO_PI * u2
    var tx     = r_disk * cos(phi)
    var ty_raw = r_disk * sin(phi)
    # Heitz 2018 eq. (Listing 3): squash the disk's LOWER half onto the
    # visible hemisphere -- lerp(sqrt(1 - tx^2), ty, s), NOT a sqrt-weighted
    # blend of tx and ty. Both forms collapse to ty = ty_raw at vh.z == 1,
    # which is exactly why this stood for so long: every furnace scene in
    # Scenes/furnace views its quad head-on, so mu_o ~ 1 and the wrong branch
    # was never exercised. Off-normal it is badly wrong -- the strategy
    # delivered 0.348 where the GGX directional albedo is 0.499 at alpha=1,
    # mu_o=0.4, i.e. 30% of a grazing rough conductor's energy simply gone,
    # and sample_ggx_vndf is shared, so the coat walk lost it too.
    # Tests/unit/test_conductor_energy.mojo sweeps mu_o for this reason.
    var s_corr = Float32(0.5) * (Float32(1.0) + vh.z)
    var tx2    = Float32(1.0) - tx * tx
    var ty     = (Float32(1.0) - s_corr) * sqrt(tx2 if tx2 > Float32(0.0) else Float32(0.0)) + s_corr * ty_raw
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
    var sobolMatrices: Pointer[UInt32, MutUntrackedOrigin]
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
    var enc = Array[UInt8, 24](fill=UInt8(0))
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
    matrices: Pointer[UInt32, MutUntrackedOrigin],
) -> Float32:
    var acc: UInt32 = 0
    var cur = index
    var base = dim * 52
    for bit in range(52):
        if cur & 1 != 0:
            acc ^= matrices[unsafe_offset=base + bit]
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

# ── Pixel reconstruction filters: THE film-filter implementation ─────────────
# One definition for all three integrators. The path tracer used to be the
# only one to apply a filter at all -- SPPM and VCM jittered uniformly inside
# the pixel, silently ignoring the scene's PixelFilter -- and its Gaussian was
# not pbrt's. That second point is why gonzales' path tracer looked visibly
# SOFTER than pbrt with identical parameters (measured on cornell-box, sigma
# 2.5 / radius 5: blurring VCM's unfiltered emitter with pbrt's kernel
# reproduces pbrt's render row by row, RMS 0.040; with the old kernel it
# reproduces the old path tracer, RMS 0.058; the cross-matches are 0.29/0.34).
#
# Filter types follow the parser: 0 = gaussian, 1 = triangle, 2 = box.

@always_inline
def gaussian_filter_sample_1d(u: Float32, sigma: Float32, radius: Float32) -> Float32:
    """Sample one axis of pbrt-v4's GaussianFilter, whose kernel is

        f(x) = max(0, g(x) - g(radius)),   g(x) = exp(-x^2 / (2 sigma^2))

    on [-radius, radius]. The `- g(radius)` term is what the old sampler here
    left out: it takes the kernel smoothly to zero at its edge, and with a
    wide filter it removes a LOT -- g(r)/g(0) is e^-2 = 13.5% at sigma 2.5,
    radius 5, and 84% for glowing_hair's sigma 2.5, radius 1.5. A plain
    Gaussian truncated at the radius keeps those tails and blurs visibly more.

    The CDF has a closed form but no closed-form inverse, so it is inverted by
    Newton's method with a bisection bracket: every step stays inside
    [lo, hi], and a step that would leave it, or a near-zero density at the
    edges, falls back to bisection. It converges in a handful of iterations
    from the truncated-Gaussian inverse as the starting point. Exact, so the
    sample weight f/pdf is constant -- the film needs no per-sample weights."""
    if radius <= Float32(0.0) or sigma <= Float32(0.0):
        return Float32(0.0)
    var s2 = sigma * sqrt(Float32(2.0))
    var inv2s2 = Float32(1.0) / (Float32(2.0) * sigma * sigma)
    var gr = exp(-(radius * radius) * inv2s2)
    var c = sigma * Float32(1.2533141373155003)   # sigma * sqrt(pi/2)
    var er = _erf(radius / s2)
    var z = Float32(2.0) * (c * er - gr * radius)  # integral of f over [-r, r]
    if z <= Float32(1e-12):
        return (u - Float32(0.5)) * Float32(2.0) * radius
    var target = u * z
    var lo = -radius
    var hi = radius
    # Start from the truncated-Gaussian inverse: close whenever g(r) is small,
    # and always inside the bracket.
    var norm = Float32(0.5) * (Float32(1.0) + er)
    var u_s = (Float32(1.0) - norm) + u * (Float32(2.0) * norm - Float32(1.0))
    var x = max(lo, min(hi, s2 * gaussian_erfinv(Float32(2.0) * u_s - Float32(1.0))))
    for _ in range(16):
        var fcdf = c * (_erf(x / s2) + er) - gr * (x + radius) - target
        if fcdf > Float32(0.0):
            hi = x
        else:
            lo = x
        var dens = exp(-(x * x) * inv2s2) - gr
        var xn: Float32
        if dens > Float32(1e-7):
            xn = x - fcdf / dens
        else:
            xn = Float32(0.5) * (lo + hi)
        if xn <= lo or xn >= hi:
            xn = Float32(0.5) * (lo + hi)
        var step = xn - x
        x = xn
        if abs(step) < Float32(1e-6) * radius:
            break
    return x

@always_inline
def filter_sample_2d(u0: Float32, u1: Float32, filter_type: Int32, sigma: Float32,
                     radius_x: Float32, radius_y: Float32) -> Tuple[Float32, Float32]:
    """Film-plane offset from the pixel CENTRE for one camera sample, drawn in
    proportion to the scene's PixelFilter. THE one filter sampler: the path
    tracer's primary rays, SPPM's visible points and VCM's camera subpaths all
    call this, so they cannot disagree about how sharp an image is."""
    if filter_type == Int32(1):
        return (triangle_sample_1d(u0, radius_x), triangle_sample_1d(u1, radius_y))
    elif filter_type == Int32(2):
        return ((u0 - Float32(0.5)) * Float32(2.0) * radius_x,
                (u1 - Float32(0.5)) * Float32(2.0) * radius_y)
    return (gaussian_filter_sample_1d(u0, sigma, radius_x),
            gaussian_filter_sample_1d(u1, sigma, radius_y))

# The film filter as ONE value: (type, sigma, radius_x, radius_y), type as a
# float. One SIMD argument threads through every camera-ray kernel instead of
# four scalars -- fewer places for a call site to pass them in the wrong
# order, and a GPU kernel argument that is DevicePassable as-is. Deliberately
# never defaulted anywhere: a camera-ray site that forgets it must fail to
# compile, not silently box-filter (which is exactly how SPPM and VCM ignored
# the scene's PixelFilter for their whole history).
comptime FilmFilter = SIMD[DType.float32, 4]

@always_inline
def film_filter_of(filter_type: Int32, sigma: Float32, radius_x: Float32,
                   radius_y: Float32) -> FilmFilter:
    return FilmFilter(Float32(filter_type), sigma, radius_x, radius_y)

@always_inline
def film_filter_offset(u0: Float32, u1: Float32, ff: FilmFilter) -> Tuple[Float32, Float32]:
    """filter_sample_2d on a packed FilmFilter."""
    return filter_sample_2d(u0, u1, ff[0].cast[DType.int32](), ff[1], ff[2], ff[3])

@always_inline
def filter_eval_2d(dx: Float32, dy: Float32, filter_type: Int32, sigma: Float32,
                   radius_x: Float32, radius_y: Float32) -> Float32:
    """The filter kernel itself at offset (dx, dy), unnormalised -- the same
    shape filter_sample_2d samples. Used to spread a light-traced splat over
    the pixels its footprint covers (pbrt's RGBFilm::AddSplat); divide by
    filter_integral_2d for a normalised weight."""
    if abs(dx) >= radius_x or abs(dy) >= radius_y:
        return Float32(0.0)
    if filter_type == Int32(1):
        return (radius_x - abs(dx)) * (radius_y - abs(dy))
    elif filter_type == Int32(2):
        return Float32(1.0)
    var inv2s2 = Float32(1.0) / (Float32(2.0) * sigma * sigma)
    var fx = exp(-(dx * dx) * inv2s2) - exp(-(radius_x * radius_x) * inv2s2)
    var fy = exp(-(dy * dy) * inv2s2) - exp(-(radius_y * radius_y) * inv2s2)
    return max(fx, Float32(0.0)) * max(fy, Float32(0.0))

@always_inline
def filter_integral_2d(filter_type: Int32, sigma: Float32, radius_x: Float32,
                       radius_y: Float32) -> Float32:
    """Integral of filter_eval_2d over its support (pbrt's Filter::Integral)."""
    if filter_type == Int32(1):
        return radius_x * radius_x * radius_y * radius_y
    elif filter_type == Int32(2):
        return Float32(4.0) * radius_x * radius_y
    var s2 = sigma * sqrt(Float32(2.0))
    var c = sigma * Float32(1.2533141373155003)
    var inv2s2 = Float32(1.0) / (Float32(2.0) * sigma * sigma)
    var ix = Float32(2.0) * (c * _erf(radius_x / s2) - exp(-(radius_x * radius_x) * inv2s2) * radius_x)
    var iy = Float32(2.0) * (c * _erf(radius_y / s2) - exp(-(radius_y * radius_y) * inv2s2) * radius_y)
    return ix * iy

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
# Returns the world-space Ray and the PCG seed pair for the path.
# px/py are integer pixel coords; si is the sample index (Int32).
@always_inline
@always_inline
def camera_ray_from_film_xy[Oc2w: Origin[mut=True] = MutUntrackedOrigin](
    filmX: Float32, filmY: Float32,
    r2c: Pointer[Float32, MutUntrackedOrigin],   # rasterToCamera  (16 Float32, col-major)
    c2w: Pointer[Float32, Oc2w],                 # cameraToWorld   (16 Float32, col-major)
) -> Tuple[Vec3f, Point3f, Float32]:
    """The raster->camera->world transform alone: film-space (filmX, filmY)
    to a normalized world-space ray direction, the camera's world origin, and
    the PRE-normalize camera-space direction length (camLen below) -- a
    texture-footprint cone needs that to turn r2c's own per-pixel derivative
    into a pixel angle (|r2c[0..2]| / camLen; camLen is preserved by c2w's
    rotation, so the camera-space and world-space lengths are the same
    scalar). Callers that only want the ray discard it.

    Deliberately just the matrix math, not primary-ray generation as a whole
    -- gen_primary_ray_state below wraps this with Sobol sampling, the scene's
    reconstruction filter and hero-wavelength selection, which SPPM/VCM/the
    --pixel and Vulkan-RT debug paths each want done differently (uniform PCG
    jitter, a bare pixel centre, or none at all). Pulling in the whole thing
    would force gen_primary_ray_state's Sobol+filter+wavelength conventions
    onto callers that deliberately use different ones.
    
    Was this same ~12-line block copy-pasted at four sites (sppm.mojo,
    bdpt.mojo, and twice in pipeline.mojo -- one of which said so in its own
    comment: "same raster_to_camera/camera_to_world math as debug_trace_pixel
    above") with no shared function, despite one existing one call away."""
    var cx = r2c[unsafe_offset=0]*filmX + r2c[unsafe_offset=4]*filmY + r2c[unsafe_offset=12]
    var cy = r2c[unsafe_offset=1]*filmX + r2c[unsafe_offset=5]*filmY + r2c[unsafe_offset=13]
    var cz = r2c[unsafe_offset=2]*filmX + r2c[unsafe_offset=6]*filmY + r2c[unsafe_offset=14]
    var cw = r2c[unsafe_offset=3]*filmX + r2c[unsafe_offset=7]*filmY + r2c[unsafe_offset=15]
    var w_div = Float32(1.0)
    if cw != Float32(0.0) and cw != Float32(1.0):
        cx /= cw; cy /= cw; cz /= cw
        w_div = abs(cw)
    var camLen = sqrt(cx*cx + cy*cy + cz*cz)
    if camLen > Float32(0.0):
        cx /= camLen; cy /= camLen; cz /= camLen
    # The length returned is the PRE-divide one. r2c is projective (pbrt's
    # rasterToCamera lands on the near plane, z = 0.01), and its first column
    # is the per-pixel derivative BEFORE the divide by w, so a pixel's angle
    # is |r2c[0..2]| / (camLen * w). Returning the post-divide camLen made
    # that ratio 1/near = 100x too large: SPPM's bump footprint (its only
    # consumer) spanned whole texture tiles, and barcelona's displaced deck
    # rendered as smooth blobs instead of gravel.
    var camLenPre = camLen * w_div

    var dx = c2w[unsafe_offset=0]*cx + c2w[unsafe_offset=4]*cy + c2w[unsafe_offset=8]*cz
    var dy = c2w[unsafe_offset=1]*cx + c2w[unsafe_offset=5]*cy + c2w[unsafe_offset=9]*cz
    var dz = c2w[unsafe_offset=2]*cx + c2w[unsafe_offset=6]*cy + c2w[unsafe_offset=10]*cz
    var dirLen = sqrt(dx*dx + dy*dy + dz*dz)
    if dirLen > Float32(0.0):
        dx /= dirLen; dy /= dirLen; dz /= dirLen

    var org = Point3f(c2w[unsafe_offset=12], c2w[unsafe_offset=13], c2w[unsafe_offset=14])
    return (Vec3f(dx, dy, dz), org, camLenPre)


def gen_primary_ray_state[Oc2w: Origin[mut=True] = MutUntrackedOrigin](
    px: Int32, py: Int32, si: Int32,
    log2spp: Int, n_base4: Int,
    seed_dim0: UInt32, seed_dim1: UInt32,
    rng_seed: UInt64,
    sobol_matrices: Pointer[UInt32, MutUntrackedOrigin],
    r2c: Pointer[Float32, MutUntrackedOrigin],   # rasterToCamera  (16 Float32, col-major)
    c2w: Pointer[Float32, Oc2w],   # cameraToWorld   (16 Float32, col-major)
    filter_norm_x: Float32, filter_sigma: Float32, filter_support_x: Float32,
    filter_norm_y: Float32, filter_support_y: Float32,
    filter_type: Int32 = Int32(0),
) -> Tuple[Ray, UInt64, UInt64, UInt64, SampledWavelengths]:
    """Shared Sobol + filter + camera-transform primary ray generator.
    Returns (ray, pcg_state, pcg_inc, sobol_idx, wavelengths).
    """
    var morton_base = encode_morton2(UInt32(px), UInt32(py)) << UInt64(log2spp)
    var morton_idx  = morton_base | UInt64(si)
    var sobol_idx   = sobol_get_sample_index(morton_idx, 0, log2spp, n_base4)
    var u0 = sobol_sample(Int(sobol_idx), 0, seed_dim0, sobol_matrices)
    var u1 = sobol_sample(Int(sobol_idx), 1, seed_dim1, sobol_matrices)
    # filter_norm_x/_y are no longer read: they normalised the old truncated
    # Gaussian, and filter_sample_2d computes pbrt's kernel from sigma and the
    # radii directly. Left in the signature so the GPU kernels that forward
    # them keep their argument layout.
    var (deltaX, deltaY) = filter_sample_2d(u0, u1, filter_type, filter_sigma,
                                            filter_support_x, filter_support_y)
    var filmX = Float32(px) + Float32(0.5) + deltaX
    var filmY = Float32(py) + Float32(0.5) + deltaY

    var (dir, org, _camLen) = camera_ray_from_film_xy(filmX, filmY, r2c, c2w)
    var dx = dir.x; var dy = dir.y; var dz = dir.z
    var orgX = org.x; var orgY = org.y; var orgZ = org.z
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

    return (Ray(Point3f(orgX, orgY, orgZ), Vec3f(dx, dy, dz)), pcg_state, pcg_inc, sobol_idx, wavelengths)
