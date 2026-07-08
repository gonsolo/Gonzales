# Reads the pbrt-v4 "measured" material's ".bsdf" tensor file (the tabulated
# BRDF format from Dupuy & Jakob 2018, "An Adaptive Parameterization for
# Efficient Material Acquisition and Rendering") to extract a coarse,
# achromatic hemispherical-reflectance estimate — NOT the full spectral
# tabulated BxDF. PBRT's actual MeasuredBxDF does multi-dimensional spline
# interpolation + importance sampling over the file's per-incident-angle
# "vndf"/"spectra" tensors (bxdfs.cpp, ~500 lines); reproducing that exactly
# is out of scope. Instead this reads only the much smaller "luminance"
# tensor (a coarse per-direction reflected-energy table already baked into
# the file for the real sampler's own CDF construction) and averages it,
# giving `_psc_handle_make_named_material` a physically-plausible brightness
# to render "measured" materials as an approximate rough conductor instead of
# a flat 50%-grey diffuse.
from std.memory import alloc

@always_inline
def _mbsdf_u16(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Int:
    return Int((buf + pos).bitcast[UInt16]()[0])

@always_inline
def _mbsdf_u32(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Int:
    return Int((buf + pos).bitcast[UInt32]()[0])

@always_inline
def _mbsdf_u64(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Int:
    return Int((buf + pos).bitcast[UInt64]()[0])

@always_inline
def _mbsdf_f32(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int) -> Float32:
    return (buf + pos).bitcast[Float32]()[0]

def _mbsdf_field_eq(buf: UnsafePointer[UInt8, MutAnyOrigin], pos: Int, length: Int, literal: StringLiteral) -> Bool:
    var lp = literal.unsafe_ptr()
    var j = 0
    while lp[j] != UInt8(0):
        if j >= length or buf[pos + j] != lp[j]:
            return False
        j += 1
    return j == length

comptime MEASURED_BSDF_DTYPE_FLOAT32 = 10

def load_measured_bsdf_reflectance(path: String) -> Tuple[Bool, Float32]:
    """Returns (ok, mean_luminance) — ok=False on any parse/format failure,
    in which case the caller should keep its existing diffuse fallback."""
    var file_buf: UnsafePointer[UInt8, MutAnyOrigin]
    var file_size: Int
    try:
        var f = open(path, "r")
        var bytes = f.read_bytes()
        f.close()
        file_size = len(bytes)
        if file_size < 18:
            return (False, Float32(0.0))
        file_buf = alloc[UInt8](file_size)
        for i in range(file_size):
            file_buf[i] = bytes[i]
    except:
        return (False, Float32(0.0))

    # 12-byte magic is "tensor_file" (11 chars) + one trailing null byte —
    # checked separately since _mbsdf_field_eq's comparison loop stops at its
    # own literal's null terminator and can't see past it.
    if not _mbsdf_field_eq(file_buf, 0, 11, "tensor_file") or file_buf[11] != UInt8(0):
        file_buf.free()
        return (False, Float32(0.0))
    if file_buf[12] != UInt8(1):  # version major must be 1
        file_buf.free()
        return (False, Float32(0.0))

    var n_fields = _mbsdf_u32(file_buf, 14)
    var pos = 18
    var found_offset = -1
    var found_dtype = -1
    var found_count = 1
    for _ in range(n_fields):
        if pos + 2 > file_size:
            break
        var name_len = _mbsdf_u16(file_buf, pos); pos += 2
        var name_pos = pos; pos += name_len
        if pos + 2 + 1 + 8 > file_size:
            break
        var ndim = _mbsdf_u16(file_buf, pos); pos += 2
        var dtype = Int(file_buf[pos]); pos += 1
        var data_offset = _mbsdf_u64(file_buf, pos); pos += 8
        var count = 1
        for d in range(ndim):
            if pos + 8 > file_size:
                break
            count *= _mbsdf_u64(file_buf, pos)
            pos += 8
        if _mbsdf_field_eq(file_buf, name_pos, name_len, "luminance"):
            found_offset = data_offset
            found_dtype = dtype
            found_count = count
            break

    if found_offset < 0 or found_dtype != MEASURED_BSDF_DTYPE_FLOAT32 or found_count <= 0:
        file_buf.free()
        return (False, Float32(0.0))
    if found_offset + found_count * 4 > file_size:
        file_buf.free()
        return (False, Float32(0.0))

    var total = Float32(0.0)
    for i in range(found_count):
        total += _mbsdf_f32(file_buf, found_offset + i * 4)
    file_buf.free()
    return (True, total / Float32(found_count))
