from std.ffi import external_call
from std.math import exp
from std.collections import List

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
            var a0r = albedo[ci + 0]
            var a0g = albedo[ci + 1]
            var a0b = albedo[ci + 2]
            var n0x = normals[ci + 0]
            var n0y = normals[ci + 1]
            var n0z = normals[ci + 2]
            var d0  = depth[pi]
            var d0_clamped = min(d0, Float32(1e18))
            var d0_sq = max(d0_clamped * d0_clamped, Float32(1e-6))

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
                    var ni  = (ny * w + nx) * 3
                    var npi = ny * w + nx
                    var dar = albedo[ni + 0] - a0r
                    var dag = albedo[ni + 1] - a0g
                    var dab = albedo[ni + 2] - a0b
                    var albedo_dist2 = dar * dar + dag * dag + dab * dab
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
