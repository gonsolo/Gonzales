"""Turning accumulated samples into a film: the sensor, normalisation, a VCM pixel. CPU and GPU share it."""
from std.math import max
from gonzales.core.geometry import RGB
from gonzales.transport.render_state import TileResult


# Shoot one unjittered center ray per pixel; record geometric normal and depth.
# normals_out: n_pixels*3 floats (Nx,Ny,Nz unit vectors; background = (0,0,1)).
# depth_out:   n_pixels floats (first-hit tHit; background = 1e38).
# Normalize TileResult[] → per-pixel float RGB arrays.
def apply_film_sensor[Ob: Origin[mut=True]](
    buf: Pointer[Float32, Ob],
    n_pixels: Int,
    exposure_time: Float32,
    wb: SIMD[DType.float32, 16],
):
    """The rest of pbrt's PixelSensor, applied to a finished RGB buffer.

    ISO is already folded in by whichever normalizer produced `buf` (CPU
    normalize_film, GPU normalize_beauty_albedo_gpu), so this adds only the
    remaining two terms: the exposure half of pbrt's imaging ratio
    (exposuretime * iso / 100), and the sensor white-balance/colour-response
    matrix built by pbrt_parser (_film_white_balance_matrix, or
    sensors.named_sensor_srgb_matrix for a supported named sensor). Kept as
    ONE post-pass over the final buffer rather than duplicated into both
    normalizers, so the CPU and GPU paths cannot drift apart on it -- the
    same discipline _sample_medium_core already applies to medium sampling.

    This is the ONLY place `buf` gets clamped to non-negative: both
    normalizers upstream now do a NaN guard only, on purpose. `buf` can
    legitimately go negative per-channel here (sRGB, or a named sensor's
    substituted-curve equivalent, both have out-of-gamut colours that dot
    negative against the XYZ->sRGB matrix), and for a non-identity `wb` that
    negative value is real information the matrix multiply below needs --
    clamping it away first and THEN mixing channels through `wb` silently
    drops a colour-correcting term, which measurably discolours the image
    (caught on explosion.pbrt's nikon_d850: a flat mid-grey "rgb L" test
    light rendered R:G:B = 0.80:0.56:0.52 instead of pbrt's near-neutral
    0.47:0.50:0.53, traced to exactly this -- buf.r was hard 0.0 going into
    the wb multiply, where it should have been negative)."""
    var wb_identity = (wb[0] == Float32(1) and wb[4] == Float32(1) and wb[8] == Float32(1)
                       and wb[1] == Float32(0) and wb[2] == Float32(0) and wb[3] == Float32(0)
                       and wb[5] == Float32(0) and wb[6] == Float32(0) and wb[7] == Float32(0))
    var trivial = wb_identity and exposure_time == Float32(1.0)
    for i in range(n_pixels):
        var r = buf[unsafe_offset=i*3+0]
        var g = buf[unsafe_offset=i*3+1]
        var b = buf[unsafe_offset=i*3+2]
        if not trivial:
            r *= exposure_time; g *= exposure_time; b *= exposure_time
            if not wb_identity:
                var nr = wb[0]*r + wb[1]*g + wb[2]*b
                var ng = wb[3]*r + wb[4]*g + wb[5]*b
                var nb = wb[6]*r + wb[7]*g + wb[8]*b
                r = nr; g = ng; b = nb
        buf[unsafe_offset=i*3+0] = max(Float32(0), r)
        buf[unsafe_offset=i*3+1] = max(Float32(0), g)
        buf[unsafe_offset=i*3+2] = max(Float32(0), b)


def normalize_film[Ores: Origin[mut=True], Obo: Origin[mut=True], Oao: Origin[mut=True]](
    results: Pointer[TileResult, Ores],
    count: Int32,
    iso: Float32,
    max_component_value: Float32,
    beauty_out: Pointer[Float32, Obo],
    albedo_out: Pointer[Float32, Oao],
):
    var scale = iso / Float32(100)
    for i in range(Int(count)):
        var r = results[unsafe_offset=i]
        var w = r.filterWeight
        if w == Float32(0):
            beauty_out[unsafe_offset=i * 3 + 0] = Float32(0)
            beauty_out[unsafe_offset=i * 3 + 1] = Float32(0)
            beauty_out[unsafe_offset=i * 3 + 2] = Float32(0)
            albedo_out[unsafe_offset=i * 3 + 0] = Float32(0)
            albedo_out[unsafe_offset=i * 3 + 1] = Float32(0)
            albedo_out[unsafe_offset=i * 3 + 2] = Float32(0)
            continue
        var b = RGB(r.estimate.r / w * scale, r.estimate.g / w * scale, r.estimate.b / w * scale)
        # NaN guard only -- NOT a negative clamp. A negative component here is
        # real (out-of-gamut colour) information apply_film_sensor's wb matrix
        # multiply still needs; clamping it away here discolours the image
        # for any non-identity wb (named sensor or explicit whitebalance).
        # See apply_film_sensor's docstring for the measured symptom.
        if b.r != b.r: b.r = Float32(0)
        if b.g != b.g: b.g = Float32(0)
        if b.b != b.b: b.b = Float32(0)
        b = b.sensor_clamped(max_component_value)
        beauty_out[unsafe_offset=i * 3 + 0] = b.r
        beauty_out[unsafe_offset=i * 3 + 1] = b.g
        beauty_out[unsafe_offset=i * 3 + 2] = b.b
        var a = r.albedo / w
        albedo_out[unsafe_offset=i * 3 + 0] = a.r
        albedo_out[unsafe_offset=i * 3 + 1] = a.g
        albedo_out[unsafe_offset=i * 3 + 2] = a.b


def _vcm_finalize_one_pixel(
    connect: RGB, merge: RGB, inv_spp: Float32,
) -> Tuple[RGB, RGB]:
    """Averages one pixel's accumulated connect+splat and vertex-merging
    sums over the render's spp samples, returning them SEPARATELY --"""
    var c = connect * inv_spp
    var m = merge * inv_spp
    if c.r != c.r: c.r = Float32(0)
    if c.g != c.g: c.g = Float32(0)
    if c.b != c.b: c.b = Float32(0)
    if m.r != m.r: m.r = Float32(0)
    if m.g != m.g: m.g = Float32(0)
    if m.b != m.b: m.b = Float32(0)
    return (c, m)
