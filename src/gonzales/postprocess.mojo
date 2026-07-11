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
def _clamp_fireflies(
    beauty: UnsafePointer[Float32, MutAnyOrigin],
    width: Int32, height: Int32,
) -> UnsafePointer[Float32, MutAnyOrigin]:
    var w = Int(width)
    var h = Int(height)
    var out = alloc[Float32](w * h * 3)
    for py in range(h):
        for px in range(w):
            var ci = (py * w + px) * 3
            var lum0 = Float32(0.2126) * beauty[ci] + Float32(0.7152) * beauty[ci + 1] + Float32(0.0722) * beauty[ci + 2]
            var max_n = Float32(0)
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
                    var lum_n = Float32(0.2126) * beauty[ni] + Float32(0.7152) * beauty[ni + 1] + Float32(0.0722) * beauty[ni + 2]
                    if lum_n > max_n:
                        max_n = lum_n
            var threshold = _FIREFLY_CLAMP_K * max_n
            # No in-bounds neighbor at all (e.g. a 1x1 image) -- "isolated"
            # is meaningless without anything to compare against, so pass
            # the pixel through unchanged rather than clamping to 0.
            if has_neighbor and lum0 > threshold and lum0 > Float32(1e-6):
                var scale = threshold / lum0
                out[ci + 0] = beauty[ci + 0] * scale
                out[ci + 1] = beauty[ci + 1] * scale
                out[ci + 2] = beauty[ci + 2] * scale
            else:
                out[ci + 0] = beauty[ci + 0]
                out[ci + 1] = beauty[ci + 1]
                out[ci + 2] = beauty[ci + 2]
    return out

# Joint bilateral denoiser guided by albedo, normals, and depth.
# beauty/albedo: width*height*3 floats, R,G,B interleaved, row-major.
# normals: width*height*3 unit-vector floats (Nx,Ny,Nz) from unjittered first-hit geometry.
# depth:   width*height floats, unjittered first-hit ray distance.
# sigma_s: spatial Gaussian std-dev (pixels).
# sigma_r: albedo range std-dev.
# sigma_n: normal edge sharpness (weight = exp(-(1-dot(n0,n1))/sigma_n)).
# sigma_d: depth relative std-dev (weight = exp(-((d1-d0)/d0)^2 / (2*sigma_d^2))).
def denoise(
    beauty:  UnsafePointer[Float32, MutAnyOrigin],
    albedo:  UnsafePointer[Float32, MutAnyOrigin],
    normals: UnsafePointer[Float32, MutAnyOrigin],
    depth:   UnsafePointer[Float32, MutAnyOrigin],
    width: Int32, height: Int32,
    output: UnsafePointer[Float32, MutAnyOrigin],
    radius: Int32,
    sigma_s: Float32,
    sigma_r: Float32,
    sigma_n: Float32,
    sigma_d: Float32,
):
    var w = Int(width)
    var h = Int(height)
    var r = Int(radius)
    var inv2ss = Float32(1) / (Float32(2) * sigma_s * sigma_s)
    var inv2sr = Float32(1) / (Float32(2) * sigma_r * sigma_r)
    var inv_sn = Float32(1) / sigma_n
    var inv2sd = Float32(1) / (Float32(2) * sigma_d * sigma_d)
    var diam = 2 * r + 1

    var clamped = _clamp_fireflies(beauty, width, height)

    # Precompute spatial weights for the (2r+1)×(2r+1) window.
    var sw = List[Float32](capacity=diam * diam)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            var dist2 = Float32(dx * dx + dy * dy)
            sw.append(exp(-dist2 * inv2ss))

    for py in range(h):
        for px in range(w):
            var ci  = (py * w + px) * 3
            var pi  = py * w + px
            var a0 = RGB(albedo[ci + 0], albedo[ci + 1], albedo[ci + 2])
            var n0x = normals[ci + 0]
            var n0y = normals[ci + 1]
            var n0z = normals[ci + 2]
            var d0  = depth[pi]
            var d0_clamped = min(d0, Float32(1e18))
            var d0_sq = max(d0_clamped * d0_clamped, Float32(1e-6))

            var acc = RGB(Float32(0))
            var acc_w = Float32(0)

            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    var nx = px + dx
                    var ny = py + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    var ni  = (ny * w + nx) * 3
                    var npi = ny * w + nx
                    var dalb = RGB(albedo[ni + 0], albedo[ni + 1], albedo[ni + 2]) - a0
                    var albedo_dist2 = (dalb * dalb).sum()
                    # Normal: 1 - dot(n0, n_neighbor) is 0 for same dir, 2 for opposite.
                    var ndot = normals[ni+0]*n0x + normals[ni+1]*n0y + normals[ni+2]*n0z
                    var normal_diff = max(Float32(0), Float32(1) - ndot)
                    # Depth: relative difference squared.
                    var ddiff = min(depth[npi], Float32(1e18)) - d0_clamped
                    var rel_depth2 = (ddiff * ddiff) / d0_sq
                    # Merge all three edge-stopping terms into one exp.
                    var wt = sw[(dy + r) * diam + (dx + r)] * exp(
                        -(albedo_dist2 * inv2sr + normal_diff * inv_sn + rel_depth2 * inv2sd)
                    )
                    acc += RGB(clamped[ni + 0], clamped[ni + 1], clamped[ni + 2]) * wt
                    acc_w += wt

            var result = (acc / acc_w) if acc_w > Float32(0) else RGB(clamped[ci + 0], clamped[ci + 1], clamped[ci + 2])
            output[ci + 0] = result.r
            output[ci + 1] = result.g
            output[ci + 2] = result.b
    clamped.free()
    # sw freed automatically when it goes out of scope


# Write a float RGB buffer via the OpenImageIO bridge.
# EXR/HDR → float32; PNG/JPG/etc. → Reinhard tonemap + sRGB gamma → uint8.
# pixels: width*height*3 floats, R,G,B interleaved, row-major.
# filename: null-terminated UTF-8 string.
# Returns 1 on success, 0 on failure.
def write_image(
    pixels: UnsafePointer[Float32, MutAnyOrigin],
    width: Int32, height: Int32,
    filename: UnsafePointer[UInt8, MutAnyOrigin],
    tile_w: Int32, tile_h: Int32,
) -> Int32:
    return external_call["write_image_rgb", Int32,
        UnsafePointer[UInt8, MutAnyOrigin],
        UnsafePointer[Float32, MutAnyOrigin],
        Int32, Int32, Int32, Int32,
    ](filename, pixels, width, height, tile_w, tile_h)

# Same as write_image, but tags the output with an OpenEXR dataWindow/
# displayWindow pair: `pixels` holds only the (width x height) data-window
# pixels, offset at (x, y) within a (full_width x full_height) display
# window starting at (0, 0) — matching pbrt's own Film "float cropwindow"
# output convention (verified against a real pbrt-v4 cropped render: same
# ceil()-based pixel bounds, same dataWindow/displayWindow split). PNG/JPG/
# etc. have no data/display-window concept, so the C++ bridge only applies
# the windowing for EXR/HDR output.
def write_image_windowed(
    pixels: UnsafePointer[Float32, MutAnyOrigin],
    width: Int32, height: Int32,
    full_width: Int32, full_height: Int32,
    x: Int32, y: Int32,
    filename: UnsafePointer[UInt8, MutAnyOrigin],
    tile_w: Int32, tile_h: Int32,
) -> Int32:
    return external_call["write_image_rgb_windowed", Int32,
        UnsafePointer[UInt8, MutAnyOrigin],
        UnsafePointer[Float32, MutAnyOrigin],
        Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32,
    ](filename, pixels, width, height, full_width, full_height, x, y, tile_w, tile_h)

# Film "float cropwindow" support: writes `pixels` (a full_w×full_h RGB float
# buffer) restricted to the [crop_x0, crop_x0+crop_w) x [crop_y0, crop_y0+
# crop_h) sub-rectangle, tagged with the full frame as an OpenEXR display
# window (see write_image_windowed) — matching pbrt's own cropped output
# convention exactly. No-op passthrough (zero extra allocation, no windowing
# metadata) when the crop rectangle is the full frame — the overwhelmingly
# common case (scenes without a cropwindow).
def write_image_cropped(
    pixels: UnsafePointer[Float32, MutAnyOrigin],
    full_w: Int32, full_h: Int32,
    crop_x0: Int32, crop_y0: Int32, crop_w: Int32, crop_h: Int32,
    filename: UnsafePointer[UInt8, MutAnyOrigin],
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
