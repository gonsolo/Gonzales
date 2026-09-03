from std.ffi import external_call
from std.math import exp
from std.memory import alloc
from std.collections import List
from .geometry import RGB

comptime _FIREFLY_CLAMP_K = Float32(4.0)  # isolated-pixel threshold multiplier, see _clamp_fireflies

# Firefly pre-clamp: without this, the bilateral filter below smears a
# single extreme-radiance pixel across its whole (2r+1)x(2r+1) window,
# since the filter's edge-stopping terms are guided by albedo/normal/depth
# only -- all of which can look identical on a smooth caustic-lit surface,
# so a hot pixel's huge beauty value passes through with near-full weight
# and gets blurred into a visible, blocky square the exact size of the
# filter kernel (found while investigating task #152's decoupled photon
# budget -- --no-denoise renders show ordinary fine-grained MC speckle,
# confirming the squares are a denoiser artifact, not a merge/grid one).
#
# Classic isolated-pixel despeckle: flag a pixel whose luminance exceeds
# _FIREFLY_CLAMP_K times the MAX luminance of its own 8 immediate
# neighbors (not a blurred/averaged threshold -- a single legitimately
# bright neighbor is enough to prove the region isn't isolated), and
# scale it down to that threshold, preserving hue/chroma (scale all three
# channels equally). A genuine bright but spatially-coherent feature
# (light source, specular highlight, caustic focus) has bright neighbors
# too, so the local max stays high and it's untouched; only a true
# single-pixel spike -- no coherent neighbor to back it up -- gets
# clamped. Biased (slightly dims genuine isolated micro-highlights), but
# this is the standard practical mitigation production renderers use for
# exactly this failure mode.
@always_inline
def _firefly_clamp_pixel(
    r0: Float32, g0: Float32, b0: Float32,
    max_n: Float32, max_n_r: Float32, max_n_g: Float32, max_n_b: Float32,
    has_neighbor: Bool,
) -> RGB:
    """Per-pixel clamp math shared by the CPU firefly pre-clamp (below) and
    firefly_clamp_gpu (gpu.mojo) -- pure scalar math, no CPU/GPU-specific
    ops, so it compiles into both a plain function call and a GPU kernel
    body unchanged. Takes the pixel's own color plus its 8-neighborhood's
    precomputed maxima (luminance and per-channel), so the CALLER owns the
    neighbor-gathering loop, which differs between a full nested CPU loop
    and one GPU thread per pixel."""
    var lum0 = RGB(r0, g0, b0).luma()
    var threshold = _FIREFLY_CLAMP_K * max_n
    var r = r0; var g = g0; var b = b0
    # No in-bounds neighbor at all (e.g. a 1x1 image) -- "isolated" is
    # meaningless without anything to compare against, so pass the pixel
    # through unchanged rather than clamping to 0.
    if has_neighbor and lum0 > threshold and lum0 > Float32(1e-6):
        var scale = threshold / lum0
        r *= scale; g *= scale; b *= scale
    # Chrominance clamp: a pixel can have UNREMARKABLE overall luminance
    # yet one channel wildly out of proportion to its neighbors -- colored,
    # not bright, Monte Carlo noise. Found concretely from a
    # hero-wavelength spectral NEE estimate occasionally drawing an
    # unlucky wavelength set for a genuinely-colored light. Same
    # isolated-pixel reasoning as the luminance clamp, per channel: a
    # genuinely-colored coherent feature has similarly-colored neighbors,
    # so its channel max stays high and it's untouched.
    if has_neighbor:
        if max_n_r > Float32(1e-6) and r > _FIREFLY_CLAMP_K * max_n_r:
            r = _FIREFLY_CLAMP_K * max_n_r
        if max_n_g > Float32(1e-6) and g > _FIREFLY_CLAMP_K * max_n_g:
            g = _FIREFLY_CLAMP_K * max_n_g
        if max_n_b > Float32(1e-6) and b > _FIREFLY_CLAMP_K * max_n_b:
            b = _FIREFLY_CLAMP_K * max_n_b
    return RGB(r, g, b)

def _clamp_fireflies[Ob: Origin[mut=True]](
    beauty: UnsafePointer[Float32, Ob],
    width: Int32, height: Int32,
) -> UnsafePointer[Float32, MutExternalOrigin]:
    var w = Int(width)
    var h = Int(height)
    var out = alloc[Float32](w * h * 3)
    for py in range(h):
        for px in range(w):
            var ci = (py * w + px) * 3
            var max_n = Float32(0)
            var max_n_r = Float32(0)
            var max_n_g = Float32(0)
            var max_n_b = Float32(0)
            var has_neighbor = False
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    var nx = px + dx
                    var ny = py + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    has_neighbor = True
                    var ni = (ny * w + nx) * 3
                    var lum_n = RGB(beauty[ni], beauty[ni + 1], beauty[ni + 2]).luma()
                    if lum_n > max_n:
                        max_n = lum_n
                    if beauty[ni + 0] > max_n_r: max_n_r = beauty[ni + 0]
                    if beauty[ni + 1] > max_n_g: max_n_g = beauty[ni + 1]
                    if beauty[ni + 2] > max_n_b: max_n_b = beauty[ni + 2]
            var c = _firefly_clamp_pixel(
                beauty[ci + 0], beauty[ci + 1], beauty[ci + 2],
                max_n, max_n_r, max_n_g, max_n_b, has_neighbor)
            out[ci + 0] = c.r
            out[ci + 1] = c.g
            out[ci + 2] = c.b
    return out

@always_inline
def _atrous_tap_weight(
    dl: Float32,
    var_p: Float32, var_q: Float32,
    dalb: RGB,
    ndot: Float32,
    ddiff: Float32, d0_sq: Float32,
    sigma_l: Float32, sigma_a: Float32, sigma_n: Float32, sigma_d: Float32,
) -> Float32:
    """One à-trous tap's edge-stopping weight (excluding the fixed spatial
    B3-spline factor, which the caller's hierarchical kernel supplies) --
    shared verbatim by the CPU pass loop (denoise, below) and
    atrous_filter_gpu (gpu.mojo), pure scalar/RGB math with no
    CPU/GPU-specific ops.

    `min(var_p, var_q)`, NOT `var_p` alone: this is the fix for a real bug
    (see project_gpu_denoiser_energy_bug.md memory) where keying the
    luminance tolerance to only the CENTRE pixel's variance makes the
    weight asymmetric (w(p,q) != w(q,p)), and since the caller normalizes
    by its own weight sum, an asymmetric weight DESTROYS energy instead of
    moving it -- a small bright feature sits at high spatial variance, so
    it accepts dark neighbours and dims, while those neighbours (low
    variance) reject it and never brighten. min() rather than max() so
    mixing happens only where BOTH pixels are noisy -- a clean pixel is
    never contaminated by a noisy neighbour."""
    var sigma_l2 = sigma_l * sigma_l * min(var_p, var_q) + Float32(1e-6)
    var w_l = exp(-dl * dl / sigma_l2)
    var sigma_a2 = sigma_a * sigma_a
    var w_a = exp(-(dalb * dalb).sum() / sigma_a2)
    var normal_diff = max(Float32(0), Float32(1) - ndot)
    var inv_sn = Float32(1) / sigma_n
    var inv2sd = Float32(1) / (Float32(2) * sigma_d * sigma_d)
    var rel_depth2 = (ddiff * ddiff) / d0_sq
    var w_nd = exp(-(normal_diff * inv_sn + rel_depth2 * inv2sd))
    return w_l * w_a * w_nd

@always_inline
def _atrous_spatial_weight(dx: Int, dy: Int) -> Float32:
    """Separable B3-spline kernel {3/8, 1/4, 1/16} at a 5x5 tap offset,
    shared by the CPU pass loop and atrous_filter_gpu -- the fixed
    hierarchical weight à-trous dilates by `step` each pass instead of a
    spatial Gaussian, which is what lets 5 cheap 5x5 passes reach an
    effective 31px radius."""
    var adx = dx if dx >= 0 else -dx
    var ady = dy if dy >= 0 else -dy
    var hx = Float32(0.0625) if adx == 2 else (Float32(0.25) if adx == 1 else Float32(0.375))
    var hy = Float32(0.0625) if ady == 2 else (Float32(0.25) if ady == 1 else Float32(0.375))
    return hx * hy

# Joint bilateral denoiser guided by albedo, normals, and depth.
# beauty/albedo: width*height*3 floats, R,G,B interleaved, row-major.
# normals: width*height*3 unit-vector floats (Nx,Ny,Nz) from unjittered first-hit geometry.
# depth:   width*height floats, unjittered first-hit ray distance.
# n_passes: number of à-trous iterations; pass i dilates taps by step=2^i,
#           so n_passes=5 reaches an effective 31px radius from five cheap
#           5x5 taps instead of one huge dense window.
# sigma_l: per-tap luminance tolerance (see _atrous_tap_weight for why it's
#          keyed to min(var_p,var_q), not either pixel alone).
# sigma_a: albedo range std-dev.
# sigma_n: normal edge sharpness (weight = exp(-(1-dot(n0,n1))/sigma_n)).
# sigma_d: depth relative std-dev (weight = exp(-((d1-d0)/d0)^2 / (2*sigma_d^2))).
#
# à-trous ("with holes") wavelet filter (Dammertz et al. 2010), the same
# algorithm and edge-stopping weights as atrous_filter_gpu (gpu.mojo) --
# unified 2026-08-03 after finding the GPU version was both a better
# algorithm AND, before a companion fix, had a real energy-destroying bug;
# see project_gpu_denoiser_energy_bug.md memory. CPU still lacks GPU's
# curve_mask hair passthrough (hair/fur was never specially handled on the
# CPU path either before or after this change -- not a regression, just an
# still-open gap, see the memory file).
def denoise[Ob: Origin[mut=True], Oa: Origin[mut=True], On: Origin[mut=True], Od: Origin[mut=True], Oo: Origin[mut=True]](
    beauty:  UnsafePointer[Float32, Ob],
    albedo:  UnsafePointer[Float32, Oa],
    normals: UnsafePointer[Float32, On],
    depth:   UnsafePointer[Float32, Od],
    width: Int32, height: Int32,
    output: UnsafePointer[Float32, Oo],
    n_passes: Int32,
    sigma_l: Float32,
    sigma_a: Float32,
    sigma_n: Float32,
    sigma_d: Float32,
):
    var w = Int(width)
    var h = Int(height)
    var n = w * h

    var clamped = _clamp_fireflies(beauty, width, height)

    # Per-pixel luminance variance over a 3x3 neighborhood, estimated ONCE
    # (not re-estimated per pass, matching estimate_variance_gpu) and used,
    # via min(var_p,var_q), by every à-trous pass below.
    var variance = alloc[Float32](n)
    for py in range(h):
        for px in range(w):
            var pi = py * w + px
            var mean = Float32(0)
            var mean_sq = Float32(0)
            var count = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    var nx = px + dx
                    var ny = py + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    var ni = (ny * w + nx) * 3
                    var l = RGB(clamped[ni], clamped[ni + 1], clamped[ni + 2]).luma()
                    mean += l; mean_sq += l * l; count += 1
            var fc = Float32(count)
            mean /= fc; mean_sq /= fc
            var v = mean_sq - mean * mean
            variance[pi] = v if v > Float32(0) else Float32(0)

    var ping = clamped
    var pong = alloc[Float32](n * 3)
    var np = Int(n_passes)
    for i in range(np):
        var step = 1 << i
        var src = ping if i % 2 == 0 else pong
        var dst = pong if i % 2 == 0 else ping
        for py in range(h):
            for px in range(w):
                var ci = (py * w + px) * 3
                var pi = py * w + px
                var c = RGB(src[ci], src[ci + 1], src[ci + 2])
                var cl = c.luma()
                var var_p = variance[pi]
                var a0 = RGB(albedo[ci + 0], albedo[ci + 1], albedo[ci + 2])
                var n0x = normals[ci + 0]
                var n0y = normals[ci + 1]
                var n0z = normals[ci + 2]
                var d0_clamped = min(depth[pi], Float32(1e18))
                var d0_sq = max(d0_clamped * d0_clamped, Float32(1e-6))

                var acc = RGB(Float32(0))
                var acc_w = Float32(0)
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        var nx = px + dx * step
                        var ny = py + dy * step
                        if nx < 0 or nx >= w or ny < 0 or ny >= h:
                            continue
                        var ni = (ny * w + nx) * 3
                        var npi = ny * w + nx
                        var qc = RGB(src[ni], src[ni + 1], src[ni + 2])
                        var dl = qc.luma() - cl
                        var dalb = RGB(albedo[ni + 0], albedo[ni + 1], albedo[ni + 2]) - a0
                        var ndot = normals[ni+0]*n0x + normals[ni+1]*n0y + normals[ni+2]*n0z
                        var ddiff = min(depth[npi], Float32(1e18)) - d0_clamped
                        var wt = _atrous_spatial_weight(dx, dy) * _atrous_tap_weight(
                            dl, var_p, variance[npi], dalb, ndot, ddiff, d0_sq,
                            sigma_l, sigma_a, sigma_n, sigma_d)
                        acc += qc * wt
                        acc_w += wt

                var result = (acc / acc_w) if acc_w > Float32(0) else c
                dst[ci + 0] = result.r
                dst[ci + 1] = result.g
                dst[ci + 2] = result.b

    var result_buf = pong if np % 2 == 1 else ping
    for i in range(n * 3):
        output[i] = result_buf[i]
    variance.free()
    clamped.free()
    pong.free()


# Write a float RGB buffer via the OpenImageIO bridge.
# EXR/HDR → float32; PNG/JPG/etc. → Reinhard tonemap + sRGB gamma → uint8.
# pixels: width*height*3 floats, R,G,B interleaved, row-major.
# filename: null-terminated UTF-8 string.
# Returns 1 on success, 0 on failure.
def write_image[Opx: Origin[mut=True]](
    pixels: UnsafePointer[Float32, Opx],
    width: Int32, height: Int32,
    filename: UnsafePointer[UInt8, MutExternalOrigin],
    tile_w: Int32, tile_h: Int32,
) -> Int32:
    return external_call["write_image_rgb", Int32,
        UnsafePointer[UInt8, MutExternalOrigin],
        UnsafePointer[Float32, MutExternalOrigin],
        Int32, Int32, Int32, Int32,
    ](filename, pixels.unsafe_origin_cast[MutExternalOrigin](), width, height, tile_w, tile_h)

# Same as write_image, but tags the output with an OpenEXR dataWindow/
# displayWindow pair: `pixels` holds only the (width x height) data-window
# pixels, offset at (x, y) within a (full_width x full_height) display
# window starting at (0, 0) — matching pbrt's own Film "float cropwindow"
# output convention (verified against a real pbrt-v4 cropped render: same
# ceil()-based pixel bounds, same dataWindow/displayWindow split). PNG/JPG/
# etc. have no data/display-window concept, so the C++ bridge only applies
# the windowing for EXR/HDR output.
def write_image_windowed(
    pixels: UnsafePointer[Float32, MutExternalOrigin],
    width: Int32, height: Int32,
    full_width: Int32, full_height: Int32,
    x: Int32, y: Int32,
    filename: UnsafePointer[UInt8, MutExternalOrigin],
    tile_w: Int32, tile_h: Int32,
) -> Int32:
    return external_call["write_image_rgb_windowed", Int32,
        UnsafePointer[UInt8, MutExternalOrigin],
        UnsafePointer[Float32, MutExternalOrigin],
        Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32,
    ](filename, pixels, width, height, full_width, full_height, x, y, tile_w, tile_h)

# Film "float cropwindow" support: writes `pixels` (a full_w×full_h RGB float
# buffer) restricted to the [crop_x0, crop_x0+crop_w) x [crop_y0, crop_y0+
# crop_h) sub-rectangle, tagged with the full frame as an OpenEXR display
# window (see write_image_windowed) — matching pbrt's own cropped output
# convention exactly. No-op passthrough (zero extra allocation, no windowing
# metadata) when the crop rectangle is the full frame — the overwhelmingly
# common case (scenes without a cropwindow).
def write_image_cropped[Opx: Origin[mut=True]](
    pixels: UnsafePointer[Float32, Opx],
    full_w: Int32, full_h: Int32,
    crop_x0: Int32, crop_y0: Int32, crop_w: Int32, crop_h: Int32,
    filename: UnsafePointer[UInt8, MutExternalOrigin],
    tile_w: Int32, tile_h: Int32,
) -> Int32:
    if crop_x0 == Int32(0) and crop_y0 == Int32(0) and crop_w == full_w and crop_h == full_h:
        return write_image(pixels, full_w, full_h, filename, tile_w, tile_h)
    var n = Int(crop_w) * Int(crop_h) * 3
    var cropped = alloc[Float32](n)
    for row in range(Int(crop_h)):
        var src_row_off = (Int(crop_y0) + row) * Int(full_w) + Int(crop_x0)
        var dst_row_off = row * Int(crop_w)
        for col in range(Int(crop_w)):
            var s = (src_row_off + col) * 3
            var d = (dst_row_off + col) * 3
            cropped[d + 0] = pixels[s + 0]
            cropped[d + 1] = pixels[s + 1]
            cropped[d + 2] = pixels[s + 2]
    var ret = write_image_windowed(cropped, crop_w, crop_h, full_w, full_h, crop_x0, crop_y0,
                                    filename, tile_w, tile_h)
    cropped.free()
    return ret
