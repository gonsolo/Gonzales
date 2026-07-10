# Reads the pbrt-v4 "measured" material's ".bsdf" tensor file (the tabulated
# BRDF format from Dupuy & Jakob 2018, "An Adaptive Parameterization for
# Efficient Material Acquisition and Rendering").
#
# load_measured_bsdf_reflectance below extracts a coarse, achromatic
# hemispherical-reflectance estimate — NOT the full spectral tabulated BxDF —
# used by material_builder.mojo's interim rough-conductor approximation.
#
# load_measured_brdf_full is the real thing: it parses every field
# (theta_i/phi_i/wavelengths/ndf/sigma/vndf/luminance/spectra/jacobian) and
# builds the PiecewiseLinear2D marginal/conditional CDF structures pbrt's own
# MeasuredBxDFData::Create does (bxdfs.cpp) / PiecewiseLinear2D's constructor
# does (util/sampling.h) — read directly from the pbrt-v4 reference source at
# /home/gonsolo/src/pbrt-v4/src/pbrt/{bxdfs.cpp,util/sampling.h} to keep this
# a faithful port rather than a reconstruction from memory. See bxdf.mojo's
# bxdf_eval_measured/bxdf_sample_measured/bxdf_pdf_measured (Stage 2) for the
# consumers of the MeasuredBRDF_C this returns.
from std.memory import alloc
from .geometry import MeasuredBRDF_C

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

comptime MEASURED_BSDF_DTYPE_UINT8   = 1
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

# ── Full tensor-file field-table parse ───────────────────────────────────────

struct _MbsdfFieldInfo(Copyable, Movable):
    var found:  Bool
    var offset: Int
    var dtype:  Int
    var ndim:   Int
    var shape:  List[Int]

    def __init__(out self):
        self.found = False
        self.offset = 0
        self.dtype = 0
        self.ndim = 0
        self.shape = List[Int]()

struct _MbsdfFields(Movable):
    """Every field load_measured_brdf_full needs, found in a single pass over
    the field table (mirrors Tensor::Tensor's own single-pass read,
    bxdfs.cpp:729-811, but only materializing the raw bytes for fields we
    actually consume -- "version"/"description"/"valid" and any other extra
    fields real .bsdf files carry are silently skipped, matching how pbrt's
    own Tensor::field() lookup-by-name ignores unrequested fields)."""
    var theta_i:     _MbsdfFieldInfo
    var phi_i:       _MbsdfFieldInfo
    var wavelengths: _MbsdfFieldInfo
    var ndf:         _MbsdfFieldInfo
    var sigma:       _MbsdfFieldInfo
    var vndf:        _MbsdfFieldInfo
    var luminance:   _MbsdfFieldInfo
    var spectra:     _MbsdfFieldInfo
    var jacobian:    _MbsdfFieldInfo

    def __init__(out self):
        self.theta_i = _MbsdfFieldInfo()
        self.phi_i = _MbsdfFieldInfo()
        self.wavelengths = _MbsdfFieldInfo()
        self.ndf = _MbsdfFieldInfo()
        self.sigma = _MbsdfFieldInfo()
        self.vndf = _MbsdfFieldInfo()
        self.luminance = _MbsdfFieldInfo()
        self.spectra = _MbsdfFieldInfo()
        self.jacobian = _MbsdfFieldInfo()

def _mbsdf_scan_fields(file_buf: UnsafePointer[UInt8, MutAnyOrigin], file_size: Int) -> _MbsdfFields:
    var out = _MbsdfFields()
    var n_fields = _mbsdf_u32(file_buf, 14)
    var pos = 18
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
        var shape = List[Int]()
        var ok_shape = True
        for _d in range(ndim):
            if pos + 8 > file_size:
                ok_shape = False
                break
            shape.append(_mbsdf_u64(file_buf, pos))
            pos += 8
        if not ok_shape:
            break

        var info = _MbsdfFieldInfo()
        info.found = True
        info.offset = data_offset
        info.dtype = dtype
        info.ndim = ndim
        info.shape = shape.copy()

        if _mbsdf_field_eq(file_buf, name_pos, name_len, "theta_i"):
            out.theta_i = info.copy()
        elif _mbsdf_field_eq(file_buf, name_pos, name_len, "phi_i"):
            out.phi_i = info.copy()
        elif _mbsdf_field_eq(file_buf, name_pos, name_len, "wavelengths"):
            out.wavelengths = info.copy()
        elif _mbsdf_field_eq(file_buf, name_pos, name_len, "ndf"):
            out.ndf = info.copy()
        elif _mbsdf_field_eq(file_buf, name_pos, name_len, "sigma"):
            out.sigma = info.copy()
        elif _mbsdf_field_eq(file_buf, name_pos, name_len, "vndf"):
            out.vndf = info.copy()
        elif _mbsdf_field_eq(file_buf, name_pos, name_len, "luminance"):
            out.luminance = info.copy()
        elif _mbsdf_field_eq(file_buf, name_pos, name_len, "spectra"):
            out.spectra = info.copy()
        elif _mbsdf_field_eq(file_buf, name_pos, name_len, "jacobian"):
            out.jacobian = info.copy()
    return out^

def _mbsdf_copy_f32(file_buf: UnsafePointer[UInt8, MutAnyOrigin], offset: Int, count: Int) -> UnsafePointer[Float32, MutAnyOrigin]:
    var out = alloc[Float32](max(count, 1))
    for i in range(count):
        out[i] = _mbsdf_f32(file_buf, offset + i * 4)
    return out

# ── PiecewiseLinear2D construction (util/sampling.h:1336-1438) ──────────────
# Two shapes needed: build_cdf=true,normalize=true (vndf/luminance, both
# PiecewiseLinear2D<2>) and build_cdf=false,normalize=false (ndf/sigma as
# PiecewiseLinear2D<0>; spectra as PiecewiseLinear2D<3>) -- note the
# normalize=false branch STILL rescales by 1/((xSize-1)*(ySize-1))
# (HProd(m_inv_patch_size)), it is not a verbatim copy; read
# util/sampling.h:1412-1436 directly if this looks surprising.

def _pl2d_build_cdf(
    raw: UnsafePointer[Float32, MutAnyOrigin], xs: Int, ys: Int, slices: Int
) -> Tuple[UnsafePointer[Float32, MutAnyOrigin], UnsafePointer[Float32, MutAnyOrigin], UnsafePointer[Float32, MutAnyOrigin]]:
    """Returns (data_out, marginal_cdf, conditional_cdf) for a
    build_cdf=true,normalize=true PiecewiseLinear2D (vndf/luminance).
    Per-slice: conditional CDF is a running trapezoidal integral across x for
    each row; marginal CDF is a trapezoidal integral (down y) of each row's
    total (its conditional CDF's last entry); both + the copied data are then
    rescaled by 1/marginal_cdf[ys-1]. Float64 accumulators match pbrt's own
    `double sum` (util/sampling.h:1380,1391)."""
    var n_values = xs * ys
    var data_out = alloc[Float32](slices * n_values)
    var marginal = alloc[Float32](slices * ys)
    var conditional = alloc[Float32](slices * n_values)
    for slice_i in range(slices):
        var base = slice_i * n_values
        var mbase = slice_i * ys
        for y in range(ys):
            var sum = Float64(0.0)
            var i = y * xs
            conditional[base + i] = Float32(0.0)
            for _x in range(xs - 1):
                sum += Float64(0.5) * (Float64(raw[base + i]) + Float64(raw[base + i + 1]))
                conditional[base + i + 1] = Float32(sum)
                i += 1
        marginal[mbase] = Float32(0.0)
        var msum = Float64(0.0)
        for y in range(ys - 1):
            msum += Float64(0.5) * (Float64(conditional[base + (y + 1) * xs - 1]) +
                                     Float64(conditional[base + (y + 2) * xs - 1]))
            marginal[mbase + y + 1] = Float32(msum)
        var norm = Float32(1.0) / marginal[mbase + ys - 1]
        for i in range(n_values):
            conditional[base + i] *= norm
        for i in range(ys):
            marginal[mbase + i] *= norm
        for i in range(n_values):
            data_out[base + i] = raw[base + i] * norm
    return (data_out, marginal, conditional)

def _pl2d_build_scaled_verbatim(
    raw: UnsafePointer[Float32, MutAnyOrigin], xs: Int, ys: Int, slices: Int
) -> UnsafePointer[Float32, MutAnyOrigin]:
    """build_cdf=false,normalize=false PiecewiseLinear2D data (ndf/sigma/
    spectra): scaled by 1/((xs-1)*(ys-1)), not a verbatim copy -- see the
    module-level note above."""
    var n_values = xs * ys
    var data_out = alloc[Float32](slices * n_values)
    var norm = Float32(1.0) / (Float32(xs - 1) * Float32(ys - 1))
    for i in range(slices * n_values):
        data_out[i] = raw[i] * norm
    return data_out

@always_inline
def _pl2d_strides2(n_phi: Int, n_theta: Int) -> Tuple[Int, Int]:
    """Per-param-axis strides for a Dimension=2 PiecewiseLinear2D with param
    order (phi, theta) -- matches the backward (last-axis-first) stride loop
    in PiecewiseLinear2D's constructor (util/sampling.h:1354-1364)."""
    var slices = 1
    var stride_theta = slices if n_theta > 1 else 0
    slices *= n_theta
    var stride_phi = slices if n_phi > 1 else 0
    return (stride_phi, stride_theta)

@always_inline
def _pl2d_strides3(n_phi: Int, n_theta: Int, n_lambda: Int) -> Tuple[Int, Int, Int]:
    """Same as _pl2d_strides2 but Dimension=3, param order (phi, theta,
    lambda) -- spectra's own strides, distinct from vndf/luminance's even
    though phi_i/theta_i are shared, since Dimension differs."""
    var slices = 1
    var stride_lambda = slices if n_lambda > 1 else 0
    slices *= n_lambda
    var stride_theta = slices if n_theta > 1 else 0
    slices *= n_theta
    var stride_phi = slices if n_phi > 1 else 0
    return (stride_phi, stride_theta, stride_lambda)

def load_measured_brdf_full(path: String) -> Tuple[Bool, MeasuredBRDF_C]:
    """The real MeasuredBxDF tensor-file loader -- parses every field and
    builds the PiecewiseLinear2D CDF structures, matching
    MeasuredBxDFData::Create (bxdfs.cpp:889-988) exactly, including its
    field-shape validation. Returns (False, <dangling dummy>) on any
    parse/format/shape mismatch, OR on a non-isotropic file (phi_i.shape[0] >
    2) -- anisotropic support is deferred (see project plan); no scene in the
    current corpus needs it. Caller must not use the returned MeasuredBRDF_C
    when ok=False."""

    def _fail() -> Tuple[Bool, MeasuredBRDF_C]:
        var dangling = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
        return (False, MeasuredBRDF_C(
            Int32(0), Int32(0), Int32(0), Int32(0),
            dangling, dangling, dangling,
            dangling, Int32(0), Int32(0),
            dangling, Int32(0), Int32(0),
            dangling, dangling, dangling, Int32(0), Int32(0),
            dangling, dangling, dangling, Int32(0), Int32(0),
            Int32(0), Int32(0),
            dangling, Int32(0), Int32(0),
            Int32(0), Int32(0), Int32(0),
        ))

    var file_buf: UnsafePointer[UInt8, MutAnyOrigin]
    var file_size: Int
    try:
        var f = open(path, "r")
        var bytes = f.read_bytes()
        f.close()
        file_size = len(bytes)
        if file_size < 18:
            return _fail()
        file_buf = alloc[UInt8](file_size)
        for i in range(file_size):
            file_buf[i] = bytes[i]
    except:
        return _fail()

    if not _mbsdf_field_eq(file_buf, 0, 11, "tensor_file") or file_buf[11] != UInt8(0):
        file_buf.free()
        return _fail()
    if file_buf[12] != UInt8(1):
        file_buf.free()
        return _fail()

    var fields = _mbsdf_scan_fields(file_buf, file_size)

    # Shape/dtype validation -- mirrors MeasuredBxDFData::Create's big
    # boolean assert (bxdfs.cpp:902-931) exactly.
    var ok = (
        fields.theta_i.found and len(fields.theta_i.shape) == 1 and fields.theta_i.dtype == MEASURED_BSDF_DTYPE_FLOAT32 and
        fields.phi_i.found and len(fields.phi_i.shape) == 1 and fields.phi_i.dtype == MEASURED_BSDF_DTYPE_FLOAT32 and
        fields.wavelengths.found and len(fields.wavelengths.shape) == 1 and fields.wavelengths.dtype == MEASURED_BSDF_DTYPE_FLOAT32 and
        fields.ndf.found and len(fields.ndf.shape) == 2 and fields.ndf.dtype == MEASURED_BSDF_DTYPE_FLOAT32 and
        fields.sigma.found and len(fields.sigma.shape) == 2 and fields.sigma.dtype == MEASURED_BSDF_DTYPE_FLOAT32 and
        fields.vndf.found and len(fields.vndf.shape) == 4 and fields.vndf.dtype == MEASURED_BSDF_DTYPE_FLOAT32 and
        fields.luminance.found and len(fields.luminance.shape) == 4 and fields.luminance.dtype == MEASURED_BSDF_DTYPE_FLOAT32 and
        fields.spectra.found and len(fields.spectra.shape) == 5 and fields.spectra.dtype == MEASURED_BSDF_DTYPE_FLOAT32 and
        fields.jacobian.found and len(fields.jacobian.shape) == 1 and fields.jacobian.shape[0] == 1 and fields.jacobian.dtype == MEASURED_BSDF_DTYPE_UINT8
    )
    if not ok:
        file_buf.free()
        return _fail()

    var n_phi_i = fields.phi_i.shape[0]
    var n_theta_i = fields.theta_i.shape[0]
    var n_wavelengths = fields.wavelengths.shape[0]

    ok = (
        fields.vndf.shape[0] == n_phi_i and fields.vndf.shape[1] == n_theta_i and
        fields.luminance.shape[0] == n_phi_i and fields.luminance.shape[1] == n_theta_i and
        fields.luminance.shape[2] == fields.luminance.shape[3] and
        fields.spectra.shape[0] == n_phi_i and fields.spectra.shape[1] == n_theta_i and
        fields.spectra.shape[2] == n_wavelengths and
        fields.spectra.shape[3] == fields.spectra.shape[4] and
        fields.luminance.shape[2] == fields.spectra.shape[3] and
        fields.luminance.shape[3] == fields.spectra.shape[4]
    )
    if not ok:
        file_buf.free()
        return _fail()

    if n_phi_i > 2:
        # Anisotropic .bsdf file -- not supported yet (see docstring).
        file_buf.free()
        return _fail()

    var theta_i = _mbsdf_copy_f32(file_buf, fields.theta_i.offset, n_theta_i)
    var phi_i = _mbsdf_copy_f32(file_buf, fields.phi_i.offset, n_phi_i)
    var wavelengths = _mbsdf_copy_f32(file_buf, fields.wavelengths.offset, n_wavelengths)

    # ndf / sigma: PiecewiseLinear2D<0> -- xSize=shape[1], ySize=shape[0]
    # (bxdfs.cpp:949-954 passes (shape[1], shape[0]) as (xSize, ySize)).
    var ndf_xs = fields.ndf.shape[1]; var ndf_ys = fields.ndf.shape[0]
    var ndf_raw = _mbsdf_copy_f32(file_buf, fields.ndf.offset, ndf_xs * ndf_ys)
    var ndf_data = _pl2d_build_scaled_verbatim(ndf_raw, ndf_xs, ndf_ys, 1)
    ndf_raw.free()

    var sigma_xs = fields.sigma.shape[1]; var sigma_ys = fields.sigma.shape[0]
    var sigma_raw = _mbsdf_copy_f32(file_buf, fields.sigma.offset, sigma_xs * sigma_ys)
    var sigma_data = _pl2d_build_scaled_verbatim(sigma_raw, sigma_xs, sigma_ys, 1)
    sigma_raw.free()

    # vndf / luminance: PiecewiseLinear2D<2>, param axes (phi_i, theta_i).
    # xSize=shape[3], ySize=shape[2] (bxdfs.cpp:957-966).
    var (stride2_phi, stride2_theta) = _pl2d_strides2(n_phi_i, n_theta_i)
    var slices2 = n_phi_i * n_theta_i

    var vndf_xs = fields.vndf.shape[3]; var vndf_ys = fields.vndf.shape[2]
    var vndf_raw = _mbsdf_copy_f32(file_buf, fields.vndf.offset, slices2 * vndf_xs * vndf_ys)
    var (vndf_data, vndf_marg, vndf_cond) = _pl2d_build_cdf(vndf_raw, vndf_xs, vndf_ys, slices2)
    vndf_raw.free()

    var lum_xs = fields.luminance.shape[3]; var lum_ys = fields.luminance.shape[2]
    var lum_raw = _mbsdf_copy_f32(file_buf, fields.luminance.offset, slices2 * lum_xs * lum_ys)
    var (lum_data, lum_marg, lum_cond) = _pl2d_build_cdf(lum_raw, lum_xs, lum_ys, slices2)
    lum_raw.free()

    # spectra: PiecewiseLinear2D<3>, param axes (phi_i, theta_i, wavelengths).
    # xSize=shape[4], ySize=shape[3] (bxdfs.cpp:975-980).
    var (stride3_phi, stride3_theta, stride3_lambda) = _pl2d_strides3(n_phi_i, n_theta_i, n_wavelengths)
    var slices3 = n_phi_i * n_theta_i * n_wavelengths

    var spectra_xs = fields.spectra.shape[4]; var spectra_ys = fields.spectra.shape[3]
    var spectra_raw = _mbsdf_copy_f32(file_buf, fields.spectra.offset, slices3 * spectra_xs * spectra_ys)
    var spectra_data = _pl2d_build_scaled_verbatim(spectra_raw, spectra_xs, spectra_ys, slices3)
    spectra_raw.free()

    file_buf.free()

    var mb = MeasuredBRDF_C(
        Int32(1), Int32(n_theta_i), Int32(n_phi_i), Int32(n_wavelengths),
        theta_i, phi_i, wavelengths,
        ndf_data, Int32(ndf_xs), Int32(ndf_ys),
        sigma_data, Int32(sigma_xs), Int32(sigma_ys),
        vndf_data, vndf_marg, vndf_cond, Int32(vndf_xs), Int32(vndf_ys),
        lum_data, lum_marg, lum_cond, Int32(lum_xs), Int32(lum_ys),
        Int32(stride2_phi), Int32(stride2_theta),
        spectra_data, Int32(spectra_xs), Int32(spectra_ys),
        Int32(stride3_phi), Int32(stride3_theta), Int32(stride3_lambda),
    )
    return (True, mb)
