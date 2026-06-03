from std.ffi import external_call
from std.math import exp
from std.memory import alloc

# Joint bilateral denoiser guided by albedo.
# beauty and albedo are width*height*3 float arrays (R,G,B interleaved, row-major).
# output must hold width*height*3 floats. radius is the filter half-width (window = 2r+1).
@export
fn mojo_denoise(
    beauty: UnsafePointer[Float32, MutAnyOrigin],
    albedo: UnsafePointer[Float32, MutAnyOrigin],
    width: Int32, height: Int32,
    output: UnsafePointer[Float32, MutAnyOrigin],
    radius: Int32,
    sigma_s: Float32,
    sigma_r: Float32,
):
    var w = Int(width)
    var h = Int(height)
    var r = Int(radius)
    var inv2ss = Float32(1) / (Float32(2) * sigma_s * sigma_s)
    var inv2sr = Float32(1) / (Float32(2) * sigma_r * sigma_r)
    var diam = 2 * r + 1

    # Precompute spatial weights for the (2r+1)×(2r+1) window.
    var sw = alloc[Float32](diam * diam)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            var dist2 = Float32(dx * dx + dy * dy)
            sw[(dy + r) * diam + (dx + r)] = exp(-dist2 * inv2ss)

    for py in range(h):
        for px in range(w):
            var ci = (py * w + px) * 3
            var a0r = albedo[ci + 0]
            var a0g = albedo[ci + 1]
            var a0b = albedo[ci + 2]

            var acc_r = Float32(0)
            var acc_g = Float32(0)
            var acc_b = Float32(0)
            var acc_w = Float32(0)

            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    var nx = px + dx
                    var ny = py + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    var ni = (ny * w + nx) * 3
                    var dar = albedo[ni + 0] - a0r
                    var dag = albedo[ni + 1] - a0g
                    var dab = albedo[ni + 2] - a0b
                    var albedo_dist2 = dar * dar + dag * dag + dab * dab
                    # Merge spatial and range into one exp call.
                    var wt = sw[(dy + r) * diam + (dx + r)] * exp(-albedo_dist2 * inv2sr)
                    acc_r += beauty[ni + 0] * wt
                    acc_g += beauty[ni + 1] * wt
                    acc_b += beauty[ni + 2] * wt
                    acc_w += wt

            if acc_w > Float32(0):
                output[ci + 0] = acc_r / acc_w
                output[ci + 1] = acc_g / acc_w
                output[ci + 2] = acc_b / acc_w
            else:
                output[ci + 0] = beauty[ci + 0]
                output[ci + 1] = beauty[ci + 1]
                output[ci + 2] = beauty[ci + 2]

    sw.free()


# Write a float RGB buffer via the OpenImageIO bridge.
# EXR/HDR → float32; PNG/JPG/etc. → Reinhard tonemap + sRGB gamma → uint8.
# pixels: width*height*3 floats, R,G,B interleaved, row-major.
# filename: null-terminated UTF-8 string.
# Returns 1 on success, 0 on failure.
@export
fn mojo_write_image(
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
