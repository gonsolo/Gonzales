# PiecewiseLinear2D primitives + MeasuredBxDF f/Sample_f/PDF -- the actual
# Dupuy & Jakob tabulated-BRDF evaluation/sampling algorithm, ported directly
# from the real pbrt-v4 reference source
# (/home/gonsolo/src/pbrt-v4/src/pbrt/{bxdfs.{h,cpp},util/sampling.h}) rather
# than reconstructed from memory. See measured_bsdf.mojo for the loader that
# builds the MeasuredBRDF_C this file consumes (theta_i/phi_i/wavelengths
# grids, ndf/sigma/vndf/luminance/spectra tensors + vndf/luminance's derived
# marginal/conditional CDFs).
#
# Only THREE concrete PiecewiseLinear2D shapes are ever needed (matching
# pbrt's own PiecewiseLinear2D<Dimension> instantiations in
# MeasuredBxDFData::Create): Dimension=0 (ndf/sigma, Evaluate-only),
# Dimension=2 (vndf/luminance, param axes phi_i,theta_i, Sample+Invert+
# Evaluate-capable), Dimension=3 (spectra, param axes phi_i,theta_i,
# wavelengths, Evaluate-only). Rather than a fully generic
# PiecewiseLinear2D[Dimension] template (Mojo's comptime story makes C++'s
# variadic-template SFINAE approach awkward to translate faithfully), this
# implements the 5 concrete entry points pbrt's algorithm actually calls,
# sharing the one genuinely Dimension-dependent piece (the per-param-axis
# multilinear blend, pbrt's `lookup<Dim>`) via _pl_lookup2/_pl_lookup3.
#
# The PL2D primitives below (_pl_find_interval_array through _pl2d_invert2)
# are deliberately NOT @always_inline, unlike every other BxDF helper in
# this codebase, to avoid duplicating their ~40-line bodies at every one of
# bxdf_eval_measured/bxdf_sample_measured's 8 call sites (4x per wavelength
# each). This is a genuine code-size win kept on its own merits, but it was
# NOT the fix for shade_measured_gpu's CUDA_ERROR_INVALID_PTX bug -- see
# _bxdf_eval_measured_core's docstring below for the real root cause
# (std.math.atan2, not kernel size).
#
# These functions take only plain UnsafePointer[Float32]/Int/Float32
# arguments -- never MeasuredBRDF_C itself -- so making them real (non-
# inlined) calls carries none of the by-value TrivialRegisterPassable-struct
# corruption risk documented elsewhere (modular/modular#6759).
from std.math import sqrt, sin, cos, acos, abs, min, max, floor
from .geometry import RGB, MeasuredBRDF_C, safe_sqrt, PI, dot
from .spectrum import SampledWavelengths, SpectralSample, spectral_sample_to_rgb, rgb_illuminant_to_spectral_sample
from .rgb2spec import cie_d65_runtime
from .bvh import LightSample, _atan2f
from .sampling import power_heuristic

# ── Local-frame warps (bxdfs.h:1058-1065) ────────────────────────────────────

@always_inline
def _measured_theta2u(theta: Float32) -> Float32:
    return sqrt(theta * (Float32(2.0) / PI))

@always_inline
def _measured_phi2u(phi: Float32) -> Float32:
    return phi * (Float32(1.0) / (Float32(2.0) * PI)) + Float32(0.5)

@always_inline
def _measured_u2theta(u: Float32) -> Float32:
    return u * u * (PI / Float32(2.0))

@always_inline
def _measured_u2phi(u: Float32) -> Float32:
    return (Float32(2.0) * u - Float32(1.0)) * PI

# ── FindInterval (util/math.h:507-519) — clamped binary search ──────────────

def _pl_find_interval_array(vals: UnsafePointer[Float32, MutAnyOrigin], sz: Int, x: Float32) -> Int:
    """FindInterval specialized to the predicate `vals[idx] <= x` (parameter-
    axis lookup: phi_i/theta_i/wavelengths)."""
    var size = sz - 2
    var first = 1
    while size > 0:
        var half = size // 2
        var middle = first + half
        if vals[middle] <= x:
            first = middle + 1
            size = size - (half + 1)
        else:
            size = half
    var result = first - 1
    if result < 0: result = 0
    if result > sz - 2: result = sz - 2
    return result

def _pl_param_wt(vals: UnsafePointer[Float32, MutAnyOrigin], size: Int, x: Float32) -> Tuple[Int, Float32]:
    """Per-axis (index, weight-of-upper-sample) for the multilinear param
    blend -- mirrors Sample/Invert/Evaluate's shared `if (m_param_size[dim]
    == 1) { w0=1,w1=0 }` fast path plus the general FindInterval+lerp case
    (util/sampling.h:1461-1478 and siblings)."""
    if size <= 1:
        return (0, Float32(0.0))
    var idx = _pl_find_interval_array(vals, size, x)
    var p0 = vals[idx]
    var p1 = vals[idx + 1]
    var w1 = (x - p0) / (p1 - p0)
    if w1 < Float32(0.0): w1 = Float32(0.0)
    if w1 > Float32(1.0): w1 = Float32(1.0)
    return (idx, w1)

# ── Per-param-axis multilinear blend (pbrt's `lookup<Dim>`, sampling.h:1711-1728) ──

def _pl_lookup2(
    data: UnsafePointer[Float32, MutAnyOrigin], i0: Int, size: Int,
    stride_phi: Int, stride_theta: Int,
    w_phi0: Float32, w_phi1: Float32, w_theta0: Float32, w_theta1: Float32,
) -> Float32:
    """lookup<2>: outer blend over theta (the LAST param axis, matching
    pbrt's back-to-front recursion), inner blend over phi (the first)."""
    var i_t0 = i0
    var i_t1 = i0 + stride_theta * size
    var v_t0 = data[i_t0] * w_phi0 + data[i_t0 + stride_phi * size] * w_phi1
    var v_t1 = data[i_t1] * w_phi0 + data[i_t1 + stride_phi * size] * w_phi1
    return v_t0 * w_theta0 + v_t1 * w_theta1

def _pl_lookup3(
    data: UnsafePointer[Float32, MutAnyOrigin], i0: Int, size: Int,
    stride_phi: Int, stride_theta: Int, stride_lambda: Int,
    w_phi0: Float32, w_phi1: Float32, w_theta0: Float32, w_theta1: Float32,
    w_lam0: Float32, w_lam1: Float32,
) -> Float32:
    """lookup<3>: outer blend over wavelength (last axis), then the same
    lookup<2> as above for phi/theta."""
    var i_l0 = i0
    var i_l1 = i0 + stride_lambda * size
    var v_l0 = _pl_lookup2(data, i_l0, size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var v_l1 = _pl_lookup2(data, i_l1, size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    return v_l0 * w_lam0 + v_l1 * w_lam1

# ── Evaluate (util/sampling.h:1644-1700) ─────────────────────────────────────

def _pl2d_eval0(
    data: UnsafePointer[Float32, MutAnyOrigin], xs: Int, ys: Int,
    px: Float32, py: Float32,
) -> Float32:
    """PiecewiseLinear2D<0>::Evaluate — no param axes (ndf/sigma)."""
    var inv_px = Float32(xs - 1)
    var inv_py = Float32(ys - 1)
    var fx = px * inv_px
    var fy = py * inv_py
    var ix = Int(fx); var iy = Int(fy)
    if ix > xs - 2: ix = xs - 2
    if iy > ys - 2: iy = ys - 2
    if ix < 0: ix = 0
    if iy < 0: iy = 0
    var w1x = fx - Float32(ix)
    var w1y = fy - Float32(iy)
    var w0x = Float32(1.0) - w1x
    var w0y = Float32(1.0) - w1y
    var idx = ix + iy * xs
    var v00 = data[idx]; var v10 = data[idx + 1]
    var v01 = data[idx + xs]; var v11 = data[idx + xs + 1]
    return (w0y * (w0x * v00 + w1x * v10) + w1y * (w0x * v01 + w1x * v11)) * inv_px * inv_py

def _pl2d_eval2(
    data: UnsafePointer[Float32, MutAnyOrigin], xs: Int, ys: Int,
    stride_phi: Int, stride_theta: Int,
    phi_i: UnsafePointer[Float32, MutAnyOrigin], n_phi: Int,
    theta_i: UnsafePointer[Float32, MutAnyOrigin], n_theta: Int,
    px: Float32, py: Float32, phi_o: Float32, theta_o: Float32,
) -> Float32:
    """PiecewiseLinear2D<2>::Evaluate (luminance, in this port)."""
    var (idx_phi, w_phi1) = _pl_param_wt(phi_i, n_phi, phi_o)
    var w_phi0 = Float32(1.0) - w_phi1
    var (idx_theta, w_theta1) = _pl_param_wt(theta_i, n_theta, theta_o)
    var w_theta0 = Float32(1.0) - w_theta1
    var slice_offset = idx_phi * stride_phi + idx_theta * stride_theta
    var size = xs * ys

    var inv_px = Float32(xs - 1)
    var inv_py = Float32(ys - 1)
    var fx = px * inv_px
    var fy = py * inv_py
    var ix = Int(fx); var iy = Int(fy)
    if ix > xs - 2: ix = xs - 2
    if iy > ys - 2: iy = ys - 2
    if ix < 0: ix = 0
    if iy < 0: iy = 0
    var w1x = fx - Float32(ix)
    var w1y = fy - Float32(iy)
    var w0x = Float32(1.0) - w1x
    var w0y = Float32(1.0) - w1y

    var base = slice_offset * size + ix + iy * xs
    var v00 = _pl_lookup2(data, base, size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var v10 = _pl_lookup2(data, base + 1, size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var v01 = _pl_lookup2(data, base + xs, size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var v11 = _pl_lookup2(data, base + xs + 1, size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    return (w0y * (w0x * v00 + w1x * v10) + w1y * (w0x * v01 + w1x * v11)) * inv_px * inv_py

def _pl2d_eval3(
    data: UnsafePointer[Float32, MutAnyOrigin], xs: Int, ys: Int,
    stride_phi: Int, stride_theta: Int, stride_lambda: Int,
    phi_i: UnsafePointer[Float32, MutAnyOrigin], n_phi: Int,
    theta_i: UnsafePointer[Float32, MutAnyOrigin], n_theta: Int,
    wavelengths: UnsafePointer[Float32, MutAnyOrigin], n_lambda: Int,
    px: Float32, py: Float32, phi_o: Float32, theta_o: Float32, lam: Float32,
) -> Float32:
    """PiecewiseLinear2D<3>::Evaluate (spectra)."""
    var (idx_phi, w_phi1) = _pl_param_wt(phi_i, n_phi, phi_o)
    var w_phi0 = Float32(1.0) - w_phi1
    var (idx_theta, w_theta1) = _pl_param_wt(theta_i, n_theta, theta_o)
    var w_theta0 = Float32(1.0) - w_theta1
    var (idx_lam, w_lam1) = _pl_param_wt(wavelengths, n_lambda, lam)
    var w_lam0 = Float32(1.0) - w_lam1
    var slice_offset = idx_phi * stride_phi + idx_theta * stride_theta + idx_lam * stride_lambda
    var size = xs * ys

    var inv_px = Float32(xs - 1)
    var inv_py = Float32(ys - 1)
    var fx = px * inv_px
    var fy = py * inv_py
    var ix = Int(fx); var iy = Int(fy)
    if ix > xs - 2: ix = xs - 2
    if iy > ys - 2: iy = ys - 2
    if ix < 0: ix = 0
    if iy < 0: iy = 0
    var w1x = fx - Float32(ix)
    var w1y = fy - Float32(iy)
    var w0x = Float32(1.0) - w1x
    var w0y = Float32(1.0) - w1y

    var base = slice_offset * size + ix + iy * xs
    var v00 = _pl_lookup3(data, base, size, stride_phi, stride_theta, stride_lambda, w_phi0, w_phi1, w_theta0, w_theta1, w_lam0, w_lam1)
    var v10 = _pl_lookup3(data, base + 1, size, stride_phi, stride_theta, stride_lambda, w_phi0, w_phi1, w_theta0, w_theta1, w_lam0, w_lam1)
    var v01 = _pl_lookup3(data, base + xs, size, stride_phi, stride_theta, stride_lambda, w_phi0, w_phi1, w_theta0, w_theta1, w_lam0, w_lam1)
    var v11 = _pl_lookup3(data, base + xs + 1, size, stride_phi, stride_theta, stride_lambda, w_phi0, w_phi1, w_theta0, w_theta1, w_lam0, w_lam1)
    return (w0y * (w0x * v00 + w1x * v10) + w1y * (w0x * v01 + w1x * v11)) * inv_px * inv_py

# ── Sample (util/sampling.h:1446-1548) ───────────────────────────────────────

def _pl2d_sample2(
    data: UnsafePointer[Float32, MutAnyOrigin],
    marg: UnsafePointer[Float32, MutAnyOrigin],
    cond: UnsafePointer[Float32, MutAnyOrigin],
    xs: Int, ys: Int, stride_phi: Int, stride_theta: Int,
    phi_i: UnsafePointer[Float32, MutAnyOrigin], n_phi: Int,
    theta_i: UnsafePointer[Float32, MutAnyOrigin], n_theta: Int,
    u0: Float32, u1: Float32, phi_o: Float32, theta_o: Float32,
) -> Tuple[Float32, Float32, Float32]:
    """PiecewiseLinear2D<2>::Sample -- returns (px, py, pdf). Used for vndf
    and luminance, which share Dimension=2/param-axis shape (differ only in
    which data/marg/cond arrays are passed)."""
    comptime ONE_MINUS_EPS = Float32(0.99999994)
    var sx = u0; var sy = u1
    if sx > ONE_MINUS_EPS: sx = ONE_MINUS_EPS
    if sx < Float32(1.0) - ONE_MINUS_EPS: sx = Float32(1.0) - ONE_MINUS_EPS
    if sy > ONE_MINUS_EPS: sy = ONE_MINUS_EPS
    if sy < Float32(1.0) - ONE_MINUS_EPS: sy = Float32(1.0) - ONE_MINUS_EPS

    var (idx_phi, w_phi1) = _pl_param_wt(phi_i, n_phi, phi_o)
    var w_phi0 = Float32(1.0) - w_phi1
    var (idx_theta, w_theta1) = _pl_param_wt(theta_i, n_theta, theta_o)
    var w_theta0 = Float32(1.0) - w_theta1
    var slice_offset = idx_phi * stride_phi + idx_theta * stride_theta
    var slice_size = xs * ys

    # Sample the row (marginal CDF over y). fetch_marginal(idx) inlined into
    # the binary search directly (Mojo's nested-def closures can't infer a
    # capture convention for the pointer/float locals this needs).
    var moff = slice_offset * ys

    var row = 0
    var size = ys - 2
    var first = 1
    while size > 0:
        var half = size // 2
        var middle = first + half
        var fm = _pl_lookup2(marg, moff + middle, ys, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
        if fm < sy:
            first = middle + 1
            size = size - (half + 1)
        else:
            size = half
    row = first - 1
    if row < 0: row = 0
    if row > ys - 2: row = ys - 2
    sy -= _pl_lookup2(marg, moff + row, ys, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)

    var offset = row * xs + slice_offset * slice_size
    var r0 = _pl_lookup2(cond, offset + xs - 1, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var r1 = _pl_lookup2(cond, offset + (xs * 2 - 1), slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)

    var is_const = abs(r0 - r1) < Float32(1e-4) * (r0 + r1)
    if is_const:
        sy = Float32(2.0) * sy
    else:
        sy = r0 - safe_sqrt(r0 * r0 - Float32(2.0) * sy * (r0 - r1))
    if is_const:
        sy /= (r0 + r1)
    else:
        sy /= (r0 - r1)

    # Sample the column (conditional CDF over x, blended by row-fraction sy).
    sx *= (Float32(1.0) - sy) * r0 + sy * r1

    var col = 0
    size = xs - 2
    first = 1
    while size > 0:
        var half = size // 2
        var middle = first + half
        var fc0 = _pl_lookup2(cond, offset + middle, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
        var fc1 = _pl_lookup2(cond, offset + xs + middle, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
        var fc = (Float32(1.0) - sy) * fc0 + sy * fc1
        if fc < sx:
            first = middle + 1
            size = size - (half + 1)
        else:
            size = half
    col = first - 1
    if col < 0: col = 0
    if col > xs - 2: col = xs - 2
    var fcol0 = _pl_lookup2(cond, offset + col, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var fcol1 = _pl_lookup2(cond, offset + xs + col, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    sx -= (Float32(1.0) - sy) * fcol0 + sy * fcol1

    offset += col

    var v00 = _pl_lookup2(data, offset, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var v10 = _pl_lookup2(data, offset + 1, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var v01 = _pl_lookup2(data, offset + xs, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var v11 = _pl_lookup2(data, offset + xs + 1, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var c0 = (Float32(1.0) - sy) * v00 + sy * v01
    var c1 = (Float32(1.0) - sy) * v10 + sy * v11

    var is_const2 = abs(c0 - c1) < Float32(1e-4) * (c0 + c1)
    if is_const2:
        sx = Float32(2.0) * sx
    else:
        sx = c0 - safe_sqrt(c0 * c0 - Float32(2.0) * sx * (c0 - c1))
    if is_const2:
        sx /= (c0 + c1)
    else:
        sx /= (c0 - c1)

    var patch_x = Float32(1.0) / Float32(xs - 1)
    var patch_y = Float32(1.0) / Float32(ys - 1)
    var px = (Float32(col) + sx) * patch_x
    var py = (Float32(row) + sy) * patch_y
    var pdf = ((Float32(1.0) - sx) * c0 + sx * c1) * Float32(xs - 1) * Float32(ys - 1)
    return (px, py, pdf)

# ── Invert (util/sampling.h:1551-1638) ───────────────────────────────────────

def _pl2d_invert2(
    data: UnsafePointer[Float32, MutAnyOrigin],
    marg: UnsafePointer[Float32, MutAnyOrigin],
    cond: UnsafePointer[Float32, MutAnyOrigin],
    xs: Int, ys: Int, stride_phi: Int, stride_theta: Int,
    phi_i: UnsafePointer[Float32, MutAnyOrigin], n_phi: Int,
    theta_i: UnsafePointer[Float32, MutAnyOrigin], n_theta: Int,
    px_in: Float32, py_in: Float32, phi_o: Float32, theta_o: Float32,
) -> Tuple[Float32, Float32, Float32]:
    """PiecewiseLinear2D<2>::Invert -- returns (ix, iy, pdf), the exact
    inverse of _pl2d_sample2. Used only for vndf (f()/PDF() call
    vndf.Invert, never luminance.Invert)."""
    var (idx_phi, w_phi1) = _pl_param_wt(phi_i, n_phi, phi_o)
    var w_phi0 = Float32(1.0) - w_phi1
    var (idx_theta, w_theta1) = _pl_param_wt(theta_i, n_theta, theta_o)
    var w_theta0 = Float32(1.0) - w_theta1
    var slice_offset = idx_phi * stride_phi + idx_theta * stride_theta
    var slice_size = xs * ys

    var sx = px_in * Float32(xs - 1)
    var sy = py_in * Float32(ys - 1)
    var pos_x = Int(sx); var pos_y = Int(sy)
    if pos_x > xs - 2: pos_x = xs - 2
    if pos_y > ys - 2: pos_y = ys - 2
    if pos_x < 0: pos_x = 0
    if pos_y < 0: pos_y = 0
    sx -= Float32(pos_x)
    sy -= Float32(pos_y)

    var offset = pos_x + pos_y * xs + slice_offset * slice_size

    var v00 = _pl_lookup2(data, offset, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var v10 = _pl_lookup2(data, offset + 1, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var v01 = _pl_lookup2(data, offset + xs, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var v11 = _pl_lookup2(data, offset + xs + 1, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)

    var w1x = sx; var w0x = Float32(1.0) - w1x
    var w1y = sy; var w0y = Float32(1.0) - w1y
    var c0 = w0y * v00 + w1y * v01
    var c1 = w0y * v10 + w1y * v11
    var pdf = w0x * c0 + w1x * c1

    sx *= c0 + Float32(0.5) * sx * (c1 - c0)

    var v0 = _pl_lookup2(cond, offset, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var v1 = _pl_lookup2(cond, offset + xs, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    sx += (Float32(1.0) - sy) * v0 + sy * v1

    var offset2 = pos_y * xs + slice_offset * slice_size
    var r0 = _pl_lookup2(cond, offset2 + xs - 1, slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    var r1 = _pl_lookup2(cond, offset2 + (xs * 2 - 1), slice_size, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)
    sx /= (Float32(1.0) - sy) * r0 + sy * r1

    sy *= r0 + Float32(0.5) * sy * (r1 - r0)

    var offset3 = pos_y + slice_offset * ys
    sy += _pl_lookup2(marg, offset3, ys, stride_phi, stride_theta, w_phi0, w_phi1, w_theta0, w_theta1)

    return (sx, sy, pdf * Float32(xs - 1) * Float32(ys - 1))

# ── MeasuredBxDF f / Sample_f / PDF (bxdfs.cpp:999-1120) ─────────────────────
# All work in a LOCAL z-up shading frame (wo_l/wi_l with .z == CosTheta),
# matching the convention every other GGX-family function in bxdf.mojo
# already uses (see bxdf_sample_conductor's wo_l construction) — pbrt's own
# formulas assume a z-up frame throughout (wo.z, SphericalTheta, etc).

@always_inline
def _measured_spherical_theta(w: SIMD[DType.float32, 3]) -> Float32:
    var cz = w[2]
    if cz < Float32(-1.0): cz = Float32(-1.0)
    if cz > Float32(1.0): cz = Float32(1.0)
    return acos(cz)

def _bxdf_eval_measured_core(
    isotropic: Int32,
    vndf_data: UnsafePointer[Float32, MutAnyOrigin], vndf_marg: UnsafePointer[Float32, MutAnyOrigin], vndf_cond: UnsafePointer[Float32, MutAnyOrigin], vndf_xs: Int32, vndf_ys: Int32,
    lum_data: UnsafePointer[Float32, MutAnyOrigin], lum_xs: Int32, lum_ys: Int32,
    stride2_phi: Int32, stride2_theta: Int32,
    spectra_data: UnsafePointer[Float32, MutAnyOrigin], spectra_xs: Int32, spectra_ys: Int32,
    stride3_phi: Int32, stride3_theta: Int32, stride3_lambda: Int32,
    ndf_data: UnsafePointer[Float32, MutAnyOrigin], ndf_xs: Int32, ndf_ys: Int32,
    sigma_data: UnsafePointer[Float32, MutAnyOrigin], sigma_xs: Int32, sigma_ys: Int32,
    phi_i: UnsafePointer[Float32, MutAnyOrigin], n_phi_i: Int32,
    theta_i: UnsafePointer[Float32, MutAnyOrigin], n_theta_i: Int32,
    wavelengths_tab: UnsafePointer[Float32, MutAnyOrigin], n_wavelengths: Int32,
    wo_l: SIMD[DType.float32, 3], wi_l: SIMD[DType.float32, 3],
    wavelengths: SampledWavelengths,
) -> Tuple[SpectralSample, Float32]:
    """MeasuredBxDF::f + PDF combined (bxdfs.cpp:999-1034, :1087-1120) —
    both need the same wm/theta/phi/u_wm/vndf.Invert setup, so this computes
    them once and returns (fr_spectral, pdf) together, matching
    bxdf_eval_any's shape. wo_l/wi_l are LOCAL-frame (z = shading normal).

    Deliberately NOT @always_inline, and takes MeasuredBRDF_C's fields
    unpacked (never the struct itself) rather than `mb: MeasuredBRDF_C` --
    see bxdf_eval_measured's docstring below for why. Returns the RAW
    spectral reflectance-like value, NOT converted to RGB -- converting a
    bare reflectance spectrum to RGB in isolation implicitly assumes an
    equal-energy illuminant, whose white point does not match sRGB's D65
    reference; every other material in this codebase avoids that bias by
    multiplying reflectance-spectral x illuminant-spectral BEFORE the
    one-time spectral_sample_to_rgb conversion (see
    _nee_weight_simple_spectral in bxdf.mojo). Callers here must do the
    same -- see _nee_weight_measured below.

    Uses `_atan2f` (bvh.mojo), NOT std.math.atan2, for phi_o/phi_m. This is
    the actual fix for shade_measured_gpu's long-standing CUDA_ERROR_INVALID_PTX
    (root-caused 2026-07-10 via progressive stubbing, after an earlier,
    WRONG hypothesis that it was an inlining/kernel-size issue -- see
    project_gpu_ptx_environment_break memory for the full trail): plain
    std.math.atan2 depends on an unresolved CUDA libdevice extern, exactly
    as already documented on `_atan2f` itself ("avoids the unresolved
    libdevice extern on GPU") and already worked around for hair's own
    phi computation in bvh.mojo. This file used raw `atan2` because that
    prior fix/precedent wasn't known when this port was written."""
    var wo = wo_l; var wi = wi_l
    # Same-hemisphere check (SameHemisphere: wo.z*wi.z > 0).
    if wo[2] * wi[2] <= Float32(0.0):
        return (SpectralSample(Float32(0.0)), Float32(0.0))
    if wo[2] < Float32(0.0):
        wo = -wo; wi = -wi

    var wm = wo + wi
    var wm_len2 = dot(wm, wm)
    if wm_len2 == Float32(0.0):
        return (SpectralSample(Float32(0.0)), Float32(0.0))
    wm = wm * (Float32(1.0) / sqrt(wm_len2))

    var theta_o = _measured_spherical_theta(wo)
    var phi_o = _atan2f(wo[1], wo[0])
    var theta_m = _measured_spherical_theta(wm)
    var phi_m = _atan2f(wm[1], wm[0])

    var u_wo_x = _measured_theta2u(theta_o)
    var u_wo_y = _measured_phi2u(phi_o)
    var u_wm_x = _measured_theta2u(theta_m)
    var u_wm_y_raw = _measured_phi2u(phi_m - phi_o) if Int(isotropic) != 0 else _measured_phi2u(phi_m)
    var u_wm_y = u_wm_y_raw - floor(u_wm_y_raw)

    var (sample_x, sample_y, vndf_pdf) = _pl2d_invert2(
        vndf_data, vndf_marg, vndf_cond, Int(vndf_xs), Int(vndf_ys),
        Int(stride2_phi), Int(stride2_theta),
        phi_i, Int(n_phi_i), theta_i, Int(n_theta_i),
        u_wm_x, u_wm_y, phi_o, theta_o,
    )

    var cos_wi = wi[2]
    if cos_wi <= Float32(0.0):
        return (SpectralSample(Float32(0.0)), Float32(0.0))

    var fr = SpectralSample(Float32(0.0))
    var fr0 = _pl2d_eval3(spectra_data, Int(spectra_xs), Int(spectra_ys),
        Int(stride3_phi), Int(stride3_theta), Int(stride3_lambda),
        phi_i, Int(n_phi_i), theta_i, Int(n_theta_i),
        wavelengths_tab, Int(n_wavelengths), sample_x, sample_y, phi_o, theta_o, wavelengths.get(0))
    var fr1 = _pl2d_eval3(spectra_data, Int(spectra_xs), Int(spectra_ys),
        Int(stride3_phi), Int(stride3_theta), Int(stride3_lambda),
        phi_i, Int(n_phi_i), theta_i, Int(n_theta_i),
        wavelengths_tab, Int(n_wavelengths), sample_x, sample_y, phi_o, theta_o, wavelengths.get(1))
    var fr2 = _pl2d_eval3(spectra_data, Int(spectra_xs), Int(spectra_ys),
        Int(stride3_phi), Int(stride3_theta), Int(stride3_lambda),
        phi_i, Int(n_phi_i), theta_i, Int(n_theta_i),
        wavelengths_tab, Int(n_wavelengths), sample_x, sample_y, phi_o, theta_o, wavelengths.get(2))
    var fr3 = _pl2d_eval3(spectra_data, Int(spectra_xs), Int(spectra_ys),
        Int(stride3_phi), Int(stride3_theta), Int(stride3_lambda),
        phi_i, Int(n_phi_i), theta_i, Int(n_theta_i),
        wavelengths_tab, Int(n_wavelengths), sample_x, sample_y, phi_o, theta_o, wavelengths.get(3))
    fr = SpectralSample(max(fr0, Float32(0.0)), max(fr1, Float32(0.0)), max(fr2, Float32(0.0)), max(fr3, Float32(0.0)))

    var ndf_val = _pl2d_eval0(ndf_data, Int(ndf_xs), Int(ndf_ys), u_wm_x, u_wm_y)
    var sigma_val = _pl2d_eval0(sigma_data, Int(sigma_xs), Int(sigma_ys), u_wo_x, u_wo_y)
    if sigma_val <= Float32(0.0):
        return (SpectralSample(Float32(0.0)), Float32(0.0))

    var fr_scaled = fr * (ndf_val / (Float32(4.0) * sigma_val * cos_wi))

    # PDF (bxdfs.cpp:1087-1120) — reuses the same u_wm/vndf.Invert result.
    var lum_val = _pl2d_eval2(lum_data, Int(lum_xs), Int(lum_ys),
        Int(stride2_phi), Int(stride2_theta),
        phi_i, Int(n_phi_i), theta_i, Int(n_theta_i), sample_x, sample_y, phi_o, theta_o)
    var sin_theta_m = sqrt(max(Float32(0.0), wm[0] * wm[0] + wm[1] * wm[1]))
    var jacobian = Float32(4.0) * dot(wo, wm) * max(Float32(2.0) * PI * PI * u_wm_x * sin_theta_m, Float32(1e-6))
    var pdf = Float32(0.0)
    if jacobian > Float32(0.0):
        pdf = vndf_pdf * lum_val / jacobian

    # Returned raw (not RGB, not clamped past the per-wavelength max(fr_i,0)
    # above): the caller must composite with an illuminant spectrum before
    # the one-time spectral_sample_to_rgb conversion -- see this function's
    # docstring and _nee_weight_measured. Converting here in isolation was
    # the previous (buggy) behavior: it implicitly assumed an equal-energy
    # illuminant, whose white point doesn't match sRGB's D65 reference, which
    # biased neutral/blue reflectances toward violet.
    return (fr_scaled, pdf)

@always_inline
def bxdf_eval_measured(
    mb: MeasuredBRDF_C,
    wo_l: SIMD[DType.float32, 3], wi_l: SIMD[DType.float32, 3],
    wavelengths: SampledWavelengths,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin],
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin],
) -> Tuple[SpectralSample, Float32]:
    """Thin @always_inline wrapper: unpacks `mb`'s fields (a plain local
    read, never a by-value cross-call pass -- safe under inlining, per the
    by-value TrivialRegisterPassable-struct hazard documented on
    MeasuredBRDF_C / modular/modular#6759) and forwards to
    _bxdf_eval_measured_core, the real (non-inlined) implementation. Exists
    so every call site keeps the ergonomic `mb: MeasuredBRDF_C` signature
    while the actual ~90-line body compiles once instead of once per call
    site -- shade_measured's NEE path alone calls this 5 times (one per
    light type), and @always_inline on the full body made shade_measured_gpu
    fail to produce valid PTX on this machine's toolchain (confirmed via
    bisection down to this exact function; unused spectral_* params kept
    for call-site compatibility with the rest of the measured-BxDF API)."""
    return _bxdf_eval_measured_core(
        mb.isotropic,
        mb.vndf_data, mb.vndf_marg, mb.vndf_cond, mb.vndf_xs, mb.vndf_ys,
        mb.lum_data, mb.lum_xs, mb.lum_ys,
        mb.stride2_phi, mb.stride2_theta,
        mb.spectra_data, mb.spectra_xs, mb.spectra_ys,
        mb.stride3_phi, mb.stride3_theta, mb.stride3_lambda,
        mb.ndf_data, mb.ndf_xs, mb.ndf_ys,
        mb.sigma_data, mb.sigma_xs, mb.sigma_ys,
        mb.phi_i, mb.n_phi_i,
        mb.theta_i, mb.n_theta_i,
        mb.wavelengths, mb.n_wavelengths,
        wo_l, wi_l, wavelengths,
    )

@always_inline
def bxdf_pdf_measured(
    mb: MeasuredBRDF_C,
    wo_l: SIMD[DType.float32, 3], wi_l: SIMD[DType.float32, 3],
) -> Float32:
    """MeasuredBxDF::PDF alone (bxdfs.cpp:1087-1120), for MIS call sites that
    already have f from elsewhere and just need the competing pdf (NEE)."""
    var wo = wo_l; var wi = wi_l
    if wo[2] * wi[2] <= Float32(0.0):
        return Float32(0.0)
    if wo[2] < Float32(0.0):
        wo = -wo; wi = -wi
    var wm = wo + wi
    var wm_len2 = dot(wm, wm)
    if wm_len2 == Float32(0.0):
        return Float32(0.0)
    wm = wm * (Float32(1.0) / sqrt(wm_len2))

    var theta_o = _measured_spherical_theta(wo)
    var phi_o = _atan2f(wo[1], wo[0])
    var theta_m = _measured_spherical_theta(wm)
    var phi_m = _atan2f(wm[1], wm[0])
    var u_wm_x = _measured_theta2u(theta_m)
    var u_wm_y_raw = _measured_phi2u(phi_m - phi_o) if Int(mb.isotropic) != 0 else _measured_phi2u(phi_m)
    var u_wm_y = u_wm_y_raw - floor(u_wm_y_raw)

    var (sample_x, sample_y, vndf_pdf) = _pl2d_invert2(
        mb.vndf_data, mb.vndf_marg, mb.vndf_cond, Int(mb.vndf_xs), Int(mb.vndf_ys),
        Int(mb.stride2_phi), Int(mb.stride2_theta),
        mb.phi_i, Int(mb.n_phi_i), mb.theta_i, Int(mb.n_theta_i),
        u_wm_x, u_wm_y, phi_o, theta_o,
    )
    var lum_val = _pl2d_eval2(mb.lum_data, Int(mb.lum_xs), Int(mb.lum_ys),
        Int(mb.stride2_phi), Int(mb.stride2_theta),
        mb.phi_i, Int(mb.n_phi_i), mb.theta_i, Int(mb.n_theta_i), sample_x, sample_y, phi_o, theta_o)
    var sin_theta_m = sqrt(max(Float32(0.0), wm[0] * wm[0] + wm[1] * wm[1]))
    var jacobian = Float32(4.0) * dot(wo, wm) * max(Float32(2.0) * PI * PI * u_wm_x * sin_theta_m, Float32(1e-6))
    if jacobian <= Float32(0.0):
        return Float32(0.0)
    return vndf_pdf * lum_val / jacobian

@always_inline
def bxdf_sample_measured(
    mb: MeasuredBRDF_C,
    wo_l: SIMD[DType.float32, 3],
    u0: Float32, u1: Float32,
    wavelengths: SampledWavelengths,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin],
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin],
) -> Tuple[SIMD[DType.float32, 3], RGB, Float32, Bool]:
    """MeasuredBxDF::Sample_f (bxdfs.cpp:1036-1085) — returns (wi_l, f, pdf,
    valid). wo_l is LOCAL-frame; wi_l is returned in the SAME local frame
    (caller transforms back to world, matching bxdf_sample_conductor's own
    tangent/bitangent/normal reconstruction)."""
    var wo = wo_l
    var flip_wi = False
    if wo[2] <= Float32(0.0):
        wo = -wo
        flip_wi = True

    var theta_o = _measured_spherical_theta(wo)
    var phi_o = _atan2f(wo[1], wo[0])

    var (lum_x, lum_y, lum_pdf) = _pl2d_sample2(
        mb.lum_data, mb.lum_marg, mb.lum_cond, Int(mb.lum_xs), Int(mb.lum_ys),
        Int(mb.stride2_phi), Int(mb.stride2_theta),
        mb.phi_i, Int(mb.n_phi_i), mb.theta_i, Int(mb.n_theta_i),
        u0, u1, phi_o, theta_o,
    )
    var (u_wm_x, u_wm_y, vndf_pdf) = _pl2d_sample2(
        mb.vndf_data, mb.vndf_marg, mb.vndf_cond, Int(mb.vndf_xs), Int(mb.vndf_ys),
        Int(mb.stride2_phi), Int(mb.stride2_theta),
        mb.phi_i, Int(mb.n_phi_i), mb.theta_i, Int(mb.n_theta_i),
        lum_x, lum_y, phi_o, theta_o,
    )

    var phi_m = _measured_u2phi(u_wm_y)
    var theta_m = _measured_u2theta(u_wm_x)
    if Int(mb.isotropic) != 0:
        phi_m += phi_o
    var sin_theta_m = sin(theta_m)
    var cos_theta_m = cos(theta_m)
    var wm = SIMD[DType.float32, 3](sin_theta_m * cos(phi_m), sin_theta_m * sin(phi_m), cos_theta_m)

    var wo_dot_wm = dot(wo, wm)
    var wi = wm * (Float32(2.0) * wo_dot_wm) - wo
    if wi[2] <= Float32(0.0):
        return (wi, RGB(Float32(0.0)), Float32(0.0), False)

    var fr0 = _pl2d_eval3(mb.spectra_data, Int(mb.spectra_xs), Int(mb.spectra_ys),
        Int(mb.stride3_phi), Int(mb.stride3_theta), Int(mb.stride3_lambda),
        mb.phi_i, Int(mb.n_phi_i), mb.theta_i, Int(mb.n_theta_i),
        mb.wavelengths, Int(mb.n_wavelengths), lum_x, lum_y, phi_o, theta_o, wavelengths.get(0))
    var fr1 = _pl2d_eval3(mb.spectra_data, Int(mb.spectra_xs), Int(mb.spectra_ys),
        Int(mb.stride3_phi), Int(mb.stride3_theta), Int(mb.stride3_lambda),
        mb.phi_i, Int(mb.n_phi_i), mb.theta_i, Int(mb.n_theta_i),
        mb.wavelengths, Int(mb.n_wavelengths), lum_x, lum_y, phi_o, theta_o, wavelengths.get(1))
    var fr2 = _pl2d_eval3(mb.spectra_data, Int(mb.spectra_xs), Int(mb.spectra_ys),
        Int(mb.stride3_phi), Int(mb.stride3_theta), Int(mb.stride3_lambda),
        mb.phi_i, Int(mb.n_phi_i), mb.theta_i, Int(mb.n_theta_i),
        mb.wavelengths, Int(mb.n_wavelengths), lum_x, lum_y, phi_o, theta_o, wavelengths.get(2))
    var fr3 = _pl2d_eval3(mb.spectra_data, Int(mb.spectra_xs), Int(mb.spectra_ys),
        Int(mb.stride3_phi), Int(mb.stride3_theta), Int(mb.stride3_lambda),
        mb.phi_i, Int(mb.n_phi_i), mb.theta_i, Int(mb.n_theta_i),
        mb.wavelengths, Int(mb.n_wavelengths), lum_x, lum_y, phi_o, theta_o, wavelengths.get(3))
    var fr = SpectralSample(max(fr0, Float32(0.0)), max(fr1, Float32(0.0)), max(fr2, Float32(0.0)), max(fr3, Float32(0.0)))

    var u_wo_x = _measured_theta2u(theta_o)
    var u_wo_y = _measured_phi2u(phi_o)
    var ndf_val = _pl2d_eval0(mb.ndf_data, Int(mb.ndf_xs), Int(mb.ndf_ys), u_wm_x, u_wm_y)
    var sigma_val = _pl2d_eval0(mb.sigma_data, Int(mb.sigma_xs), Int(mb.sigma_ys), u_wo_x, u_wo_y)
    var abs_cos_wi = abs(wi[2])
    if sigma_val <= Float32(0.0) or abs_cos_wi <= Float32(0.0):
        return (wi, RGB(Float32(0.0)), Float32(0.0), False)
    var fr_scaled = fr * (ndf_val / (Float32(4.0) * sigma_val * abs_cos_wi))

    # This is a bounce-continuation sample: there is no specific light to
    # composite with yet (unlike _nee_weight_measured, which has ls.Li), so
    # -- matching every other material's indirect-bounce convention -- we
    # composite against a neutral D65 illuminant shape before the one-time
    # RGB conversion. Converting the bare reflectance alone (the previous,
    # buggy behavior) implicitly assumes an equal-energy illuminant, whose
    # white point doesn't match sRGB's D65 reference and biased neutral/blue
    # reflectances toward violet -- see bxdf_eval_measured's docstring.
    var illum0 = cie_d65_runtime(spectral_d65, wavelengths.get(0))
    var illum1 = cie_d65_runtime(spectral_d65, wavelengths.get(1))
    var illum2 = cie_d65_runtime(spectral_d65, wavelengths.get(2))
    var illum3 = cie_d65_runtime(spectral_d65, wavelengths.get(3))
    var fr_lit = SpectralSample(fr_scaled.v0 * illum0, fr_scaled.v1 * illum1, fr_scaled.v2 * illum2, fr_scaled.v3 * illum3)
    var (r, g, b) = spectral_sample_to_rgb(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, fr_lit, wavelengths)

    var jacobian = Float32(4.0) * wo_dot_wm * max(Float32(2.0) * PI * PI * u_wm_x * sin_theta_m, Float32(1e-6))
    if jacobian <= Float32(0.0):
        return (wi, RGB(Float32(0.0)), Float32(0.0), False)
    var pdf = (vndf_pdf / jacobian) * lum_pdf

    if flip_wi:
        wi = -wi

    # No post-conversion max(,0) clamp -- see bxdf_eval_measured's comment
    # above the same pattern for why (out-of-sRGB-gamut negative red is
    # legitimate for a saturated measured blue, and clamping it away here
    # inflates the accumulated red channel, reading as violet).
    return (wi, RGB(r, g, b), pdf, True)

# ── NEE weight (mirrors bxdf.mojo's _nee_weight_simple / _nee_weight_hair) ──

# NOT @always_inline (unlike bxdf.mojo's _nee_weight_simple/_via_spectral,
# which stay inline since conductor/diffuse's total NEE bulk fits fine) --
# called 5x from _shade_measured_nee (one per light type), and that call
# site is itself now a real function specifically to keep shade_measured_gpu
# small (see _shade_measured_nee's docstring in shading.mojo); leaving this
# inline would still unroll its full spectral-compositing body 5x into
# _shade_measured_nee's own compiled body. De-inlining it directly reduces
# that to 5 real calls instead.
def _nee_weight_measured(
    ls: LightSample,
    mb: MeasuredBRDF_C,
    tangent: SIMD[DType.float32, 3], bitangent: SIMD[DType.float32, 3], normal: SIMD[DType.float32, 3],
    wo: SIMD[DType.float32, 3],
    wavelengths: SampledWavelengths,
    spectral_coeffs: UnsafePointer[Float32, MutAnyOrigin], spectral_res: Int,
    spectral_cie_x: UnsafePointer[Float32, MutAnyOrigin],
    spectral_cie_y: UnsafePointer[Float32, MutAnyOrigin],
    spectral_cie_z: UnsafePointer[Float32, MutAnyOrigin],
    spectral_d65: UnsafePointer[Float32, MutAnyOrigin],
) -> RGB:
    """Measured's own version of _nee_weight_simple (bxdf.mojo) — can't share
    that function's flat (mat_kind, alb, alpha) signature since it needs the
    full MeasuredBRDF_C descriptor + wavelengths, same reason hair has its
    own _nee_weight_hair. Projects wo/ls.wi into the local (tangent,
    bitangent, normal) frame bxdf_eval_measured expects.

    Mirrors _nee_weight_simple_via_spectral (bxdf.mojo): bxdf_eval_measured
    returns a RAW spectral reflectance (not RGB -- see its docstring), which
    must be composited with the light's own spectral radiance (via
    rgb_illuminant_to_spectral_sample on ls.Li) BEFORE the one-time
    spectral_sample_to_rgb conversion. Converting the bare reflectance alone
    would (as it previously did) implicitly assume an equal-energy
    illuminant, biasing neutral/blue reflectances toward violet."""
    if not ls.valid:
        return RGB(Float32(0.0))
    var cos_s = dot(normal, ls.wi)
    if cos_s <= Float32(0.0):
        return RGB(Float32(0.0))
    var wo_l = SIMD[DType.float32, 3](dot(wo, tangent), dot(wo, bitangent), dot(wo, normal))
    var wi_l = SIMD[DType.float32, 3](dot(ls.wi, tangent), dot(ls.wi, bitangent), dot(ls.wi, normal))
    var (fr_spectral, pdf_bsdf) = bxdf_eval_measured(mb, wo_l, wi_l, wavelengths, spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65)
    if fr_spectral.v0 <= Float32(0.0) and fr_spectral.v1 <= Float32(0.0) and fr_spectral.v2 <= Float32(0.0) and fr_spectral.v3 <= Float32(0.0):
        return RGB(Float32(0.0))
    var li_spectral = rgb_illuminant_to_spectral_sample(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, ls.Li.r, ls.Li.g, ls.Li.b, wavelengths)
    var product = fr_spectral * li_spectral
    var (r, g, b) = spectral_sample_to_rgb(spectral_coeffs, spectral_res, spectral_cie_x, spectral_cie_y, spectral_cie_z, spectral_d65, product, wavelengths)
    var f_times_li = RGB(r, g, b)
    if ls.is_delta:
        return f_times_li * cos_s
    if ls.pdf <= Float32(0.0):
        return RGB(Float32(0.0))
    var mis_w = power_heuristic(ls.pdf, pdf_bsdf)
    return f_times_li * (cos_s * mis_w / ls.pdf)
