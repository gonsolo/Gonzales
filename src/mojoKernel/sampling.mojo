from std.math import sqrt, log, exp
from std.memory import alloc

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

@always_inline
fn reverse_bits32(v_in: UInt32) -> UInt32:
    var v = v_in
    v = ((v >> 1) & UInt32(0x55555555)) | ((v & UInt32(0x55555555)) << 1)
    v = ((v >> 2) & UInt32(0x33333333)) | ((v & UInt32(0x33333333)) << 2)
    v = ((v >> 4) & UInt32(0x0f0f0f0f)) | ((v & UInt32(0x0f0f0f0f)) << 4)
    v = ((v >> 8) & UInt32(0x00ff00ff)) | ((v & UInt32(0x00ff00ff)) << 8)
    return (v >> 16) | (v << 16)

@always_inline
fn fast_owen_scramble(value_in: UInt32, seed: UInt32) -> UInt32:
    var v = reverse_bits32(value_in)
    v ^= v * UInt32(0x3d20adea)
    v += seed
    v *= (seed >> 16) | UInt32(1)
    v ^= v * UInt32(0x05526c56)
    v ^= v * UInt32(0x53a22864)
    return reverse_bits32(v)

@always_inline
fn mix_bits_u64(v: UInt64) -> UInt32:
    var v32 = UInt32(v & UInt64(0xFFFFFFFF))
    v32 ^= UInt32(v >> 32)
    v32 ^= v32 >> 16
    v32 *= UInt32(0x85ebca77)
    v32 ^= v32 >> 13
    v32 *= UInt32(0xc2b2ae35)
    v32 ^= v32 >> 16
    return v32

@always_inline
fn encode_morton2(x: UInt32, y: UInt32) -> UInt64:
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
fn sobol_perm_lookup(p_idx: Int, digit: Int) -> Int:
    var enc = InlineArray[UInt8, 24](fill=UInt8(0))
    enc[ 0]=27; enc[ 1]=30; enc[ 2]=39; enc[ 3]=45; enc[ 4]=57; enc[ 5]=54
    enc[ 6]=75; enc[ 7]=78; enc[ 8]=99; enc[ 9]=108; enc[10]=120; enc[11]=114
    enc[12]=147; enc[13]=156; enc[14]=135; enc[15]=141; enc[16]=177; enc[17]=180
    enc[18]=216; enc[19]=210; enc[20]=228; enc[21]=225; enc[22]=201; enc[23]=198
    return Int((Int(enc[p_idx]) >> (2 * (3 - digit))) & 3)

@always_inline
fn sobol_get_sample_index(
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
fn sobol_sample(
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
fn gaussian_erfinv(y: Float32) -> Float32:
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
fn gaussian_sample_1d(u: Float32, norm: Float32, sigma: Float32, radius: Float32) -> Float32:
    var u_s = (Float32(1.0) - norm) + u * (Float32(2.0) * norm - Float32(1.0))
    var x = sigma * sqrt(Float32(2.0)) * gaussian_erfinv(Float32(2.0) * u_s - Float32(1.0))
    return max(-radius, min(radius, x))

# erf via Abramowitz & Stegun 7.1.26 (max error ≤ 1.5e-7).
fn _erf(x: Float32) -> Float32:
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
@export
fn mojo_gaussian_norm(support: Float32, sigma: Float32) -> Float32:
    var x = support / (sigma * sqrt(Float32(2)))
    return Float32(0.5) * (Float32(1) + _erf(x))


# Hash pixel + sample index into a unique PCG (state, inc) pair.
@always_inline
fn derive_pcg_seeds(px: Int32, py: Int32, si: Int32, seed: UInt64) -> Tuple[UInt64, UInt64]:
    var h = UInt64(px) * UInt64(2654435761) ^ UInt64(py) * UInt64(1664525) ^ UInt64(si) * UInt64(22695477) ^ seed
    h ^= h >> 30; h *= UInt64(0xbf58476d1ce4e5b9)
    h ^= h >> 27; h *= UInt64(0x94d049bb133111eb)
    h ^= h >> 31
    var state = h
    h ^= h >> 30; h *= UInt64(0xbf58476d1ce4e5b9)
    h ^= h >> 27
    return (state, h | UInt64(1))
