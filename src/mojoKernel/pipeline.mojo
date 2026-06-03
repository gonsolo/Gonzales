from std.memory import alloc
from .parsing import ParsedScene_Mojo, mojo_parse_scene, mojo_parsed_free, mojo_parsed_scene_descriptor
from .rendering import mojo_render_all_tiles, mojo_normalize_film
from .geometry import TileResult_C
from .postprocess import mojo_denoise, mojo_write_exr
from .sampling import TileSamplerParams_C
from .bvh import BVH2Node

# Generate Sobol matrices from the Joe-Kuo data file.
# Returns a heap-allocated pointer to 21201*52 UInt32 values, or null on error.
fn _generate_sobol_matrices(path: String) -> UnsafePointer[UInt32, MutAnyOrigin]:
    var file_buf: UnsafePointer[UInt8, MutAnyOrigin]
    var file_size: Int
    try:
        var f = open(path, "r")
        var bytes = f.read_bytes()
        f.close()
        file_size = len(bytes)
        file_buf = alloc[UInt8](file_size + 1)
        for i in range(file_size):
            file_buf[i] = bytes[i]
        file_buf[file_size] = UInt8(0)
    except:
        print("Error: cannot open Sobol data file: " + path)
        return UnsafePointer[UInt32, MutAnyOrigin]()

    # Allocate matrices: 21201 dimensions × 52 bits
    comptime N_DIMS = 21201
    comptime N_BITS = 52
    var matrices = alloc[UInt32](N_DIMS * N_BITS)
    # Zero-initialize
    for i in range(N_DIMS * N_BITS):
        matrices[i] = UInt32(0)

    # Dimension 0: all ones (standard Sobol)
    for j in range(N_BITS):
        matrices[j] = UInt32(1) << UInt32(31 - j)

    # Parse remaining dimensions from file
    var pos = 0
    var flen = Int(file_size)
    var dim = 1

    # Helper: skip whitespace (spaces and tabs, but not newlines)
    # Read one line at a time and parse
    while pos < flen and dim < N_DIMS:
        # Skip leading whitespace including newlines
        while pos < flen and (file_buf[pos] == UInt8(32) or file_buf[pos] == UInt8(9) or file_buf[pos] == UInt8(10) or file_buf[pos] == UInt8(13)):
            pos += 1
        if pos >= flen:
            break
        # Skip comment lines starting with '#'
        if file_buf[pos] == UInt8(35):  # '#'
            while pos < flen and file_buf[pos] != UInt8(10):
                pos += 1
            continue

        # Parse: s a m1 m2 ... ms
        # s = number of direction numbers
        var s = Int32(0)
        while pos < flen and file_buf[pos] >= UInt8(48) and file_buf[pos] <= UInt8(57):
            s = s * Int32(10) + Int32(file_buf[pos]) - Int32(48)
            pos += 1
        # skip whitespace
        while pos < flen and (file_buf[pos] == UInt8(32) or file_buf[pos] == UInt8(9)):
            pos += 1

        # a = polynomial
        var a = UInt32(0)
        while pos < flen and file_buf[pos] >= UInt8(48) and file_buf[pos] <= UInt8(57):
            a = a * UInt32(10) + UInt32(file_buf[pos]) - UInt32(48)
            pos += 1
        # skip whitespace
        while pos < flen and (file_buf[pos] == UInt8(32) or file_buf[pos] == UInt8(9)):
            pos += 1

        # m values
        var m = InlineArray[UInt32, 52](fill=UInt32(0))
        var num_m = Int(s)
        if num_m > N_BITS:
            num_m = N_BITS
        for i in range(num_m):
            var v = UInt32(0)
            while pos < flen and file_buf[pos] >= UInt8(48) and file_buf[pos] <= UInt8(57):
                v = v * UInt32(10) + UInt32(file_buf[pos]) - UInt32(48)
                pos += 1
            m[i] = v
            while pos < flen and (file_buf[pos] == UInt8(32) or file_buf[pos] == UInt8(9)):
                pos += 1

        # Skip to end of line
        while pos < flen and file_buf[pos] != UInt8(10):
            pos += 1

        # Compute direction numbers v[i] = m[i] << (32 - i - 1)
        var base = dim * N_BITS
        for i in range(Int(s)):
            if i >= N_BITS:
                break
            matrices[base + i] = m[i] << UInt32(31 - i)

        # Recurrence for i >= s
        for i in range(Int(s), N_BITS):
            var v_prev = matrices[base + i - Int(s)]
            var vi = v_prev ^ (v_prev >> UInt32(s))
            var j = 1
            var poly = a
            while j <= Int(s) - 1:
                if (poly & UInt32(1)) != UInt32(0):
                    vi ^= matrices[base + i - j]
                poly >>= 1
                j += 1
            matrices[base + i] = vi

        dim += 1

    file_buf.free()

    if dim < 2:
        print("Warning: Sobol file had fewer dimensions than expected")

    print("Sobol matrices loaded: " + String(dim) + " dimensions")
    return matrices


@export
fn mojo_parse_and_render(
    path: UnsafePointer[UInt8, MutAnyOrigin],
    sobol_matrices: UnsafePointer[UInt32, MutAnyOrigin],
) -> Int32:
    var psc = mojo_parse_scene(path)
    if not psc:
        return Int32(-1)

    var fw = psc[0].film_w
    var fh = psc[0].film_h
    var n_pixels = Int(fw) * Int(fh)

    var sd = mojo_parsed_scene_descriptor(psc)

    # Build sampler params
    var sp = TileSamplerParams_C(
        sobolMatrices=sobol_matrices,
        rngSeed=psc[0].rng_seed,
        sobolSeed=Int32(0),
        log2SamplesPerPixel=psc[0].log2_spp,
        nBase4Digits=psc[0].n_base4_digits,
        samplesPerPixel=psc[0].samples_per_pixel,
        filterSigma=psc[0].filter_sigma,
        filterSupportX=psc[0].filter_support_x,
        filterSupportY=psc[0].filter_support_y,
        filterNormX=psc[0].filter_norm_x,
        filterNormY=psc[0].filter_norm_y,
        filterWeight=psc[0].filter_weight,
    )

    var results = alloc[TileResult_C](n_pixels)
    var zero = TileResult_C(
        estimateR=Float32(0), estimateG=Float32(0), estimateB=Float32(0),
        albedoR=Float32(0), albedoG=Float32(0), albedoB=Float32(0),
        filterWeight=Float32(0), pixelX=Int32(0), pixelY=Int32(0))
    for i in range(n_pixels):
        results[i] = zero

    var sp_ptr = alloc[TileSamplerParams_C](1)
    sp_ptr[0] = sp
    mojo_render_all_tiles(
        psc[0].raster_to_camera, psc[0].camera_to_world,
        Int32(0), Int32(0), fw, fh,
        Int32(32), Int32(32),
        sp_ptr, sd, results, psc[0].max_depth)
    sp_ptr.free()

    var beauty = alloc[Float32](n_pixels * 3)
    var albedo = alloc[Float32](n_pixels * 3)
    mojo_normalize_film(results, Int32(n_pixels),
                        psc[0].film_iso, psc[0].film_max_comp,
                        beauty, albedo)
    results.free()

    var denoised = alloc[Float32](n_pixels * 3)
    mojo_denoise(beauty, albedo, fw, fh, denoised, Int32(7), Float32(5.0), Float32(0.2))

    _ = mojo_write_exr(denoised, fw, fh, psc[0].film_filename, Int32(32), Int32(32))
    var albedo_name_buf = alloc[UInt8](16)
    var an = "albedo.exr"
    var an_ptr = an.unsafe_ptr()
    for i in range(10):
        albedo_name_buf[i] = an_ptr[i]
    albedo_name_buf[10] = UInt8(0)
    _ = mojo_write_exr(albedo, fw, fh, albedo_name_buf, Int32(32), Int32(32))
    albedo_name_buf.free()

    beauty.free(); albedo.free(); denoised.free()
    sd.free()
    mojo_parsed_free(psc)
    return Int32(0)
