# What a finished render writes, for every integrator.
#
# The film file itself carries the image plus pbrt-v4 GBufferFilm-style layers
# -- R G B, Albedo.R/G/B, N.X/Y/Z (world-space normal at the first hit), Z
# (camera distance) -- so tev/Blender/denoisers get everything in one file,
# and the same buffers are also written as sidecars next to it for viewers
# that only read RGB (GIMP): <name>.albedo.exr, <name>.normal.exr,
# <name>.depth.exr, and <name>.noisy.exr (the image before denoising) when the
# denoiser ran. Always on.
#
# Seven render drivers (PT CPU/GPU, VCM CPU/GPU/wavefront, SPPM CPU/GPU) each
# used to write only the image, and the GPU path tracer an `albedo.exr` with a
# fixed name in the working directory, which every render overwrote.

from std.ffi import external_call
from std.math import ceil
from std.memory.alloc import unsafe_alloc
from .bvh import SceneView, render_aux_buffers
from .pbrt_parser import ParsedScene_Mojo
from .postprocess import denoise, write_image_cropwindow
from .geometry import RGB, _is_real_ptr


def _cstr(s: String) -> Pointer[UInt8, MutUntrackedOrigin]:
    var n = s.byte_length()
    var buf = unsafe_alloc[UInt8](n + 1)
    var p = s.unsafe_ptr()
    for i in range(n):
        buf[unsafe_offset=i] = p[unsafe_offset=i]
    buf[unsafe_offset=n] = UInt8(0)
    return buf


def _sidecar_name(film: Pointer[UInt8, MutUntrackedOrigin], suffix: String) -> Pointer[UInt8, MutUntrackedOrigin]:
    """`film` with its extension replaced by `suffix` ("x/foo.exr" +
    ".albedo.exr" -> "x/foo.albedo.exr"). Sidecars are always EXR, whatever
    the film's own format."""
    var n = 0
    while film[unsafe_offset=n] != UInt8(0):
        n += 1
    var dot = n
    var i = n - 1
    while i >= 0:
        var c = film[unsafe_offset=i]
        if c == UInt8(ord("/")):
            break
        if c == UInt8(ord(".")):
            dot = i
            break
        i -= 1
    var sb = suffix.unsafe_ptr()
    var m = suffix.byte_length()
    var buf = unsafe_alloc[UInt8](dot + m + 1)
    for k in range(dot):
        buf[unsafe_offset=k] = film[unsafe_offset=k]
    for k in range(m):
        buf[unsafe_offset=dot + k] = sb[unsafe_offset=k]
    buf[unsafe_offset=dot + m] = UInt8(0)
    return buf


def _write_channels(
    filename: Pointer[UInt8, MutUntrackedOrigin],
    data: Pointer[Float32, MutUntrackedOrigin],
    w: Int32, h: Int32, nch: Int32, names: String,
    full_w: Int32, full_h: Int32, x: Int32, y: Int32,
) -> Int32:
    var cn = _cstr(names)
    var r = external_call["write_image_channels", Int32,
        Pointer[UInt8, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin],
        Int32, Int32, Int32, Pointer[UInt8, MutUntrackedOrigin],
        Int32, Int32, Int32, Int32, Int32, Int32,
    ](filename, data, w, h, nch, cn, full_w, full_h, x, y, Int32(32), Int32(32))
    cn.unsafe_free()
    return r


def write_render_outputs[Ob: Origin[mut=True], On: Origin[mut=True], Oa: Origin[mut=True], Onm: Origin[mut=True], Od: Origin[mut=True]](
    psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    image: Pointer[Float32, Ob],     # what the film shows (denoised, if the denoiser ran)
    noisy: Pointer[Float32, On],     # the image before denoising; read only if `denoised`
    denoised: Bool,
    albedo: Pointer[Float32, Oa],
    normals: Pointer[Float32, Onm],
    depth: Pointer[Float32, Od],
) -> Int32:
    """The film file with its layers, then the sidecars, all under the film's
    crop window (pbrt's ceil()-based bounds, as write_image_cropwindow)."""
    var fw = psc[unsafe_offset=0].film_w
    var fh = psc[unsafe_offset=0].film_h
    var film = psc[unsafe_offset=0].film_filename
    var cx0 = psc[unsafe_offset=0].crop_x0; var cy0 = psc[unsafe_offset=0].crop_y0
    var cx1 = psc[unsafe_offset=0].crop_x1; var cy1 = psc[unsafe_offset=0].crop_y1
    var x0 = Int(ceil(cx0 * Float32(fw))); var y0 = Int(ceil(cy0 * Float32(fh)))
    var x1 = Int(ceil(cx1 * Float32(fw))); var y1 = Int(ceil(cy1 * Float32(fh)))
    var cw = x1 - x0; var ch = y1 - y0
    comptime NCH = 10
    var main = unsafe_alloc[Float32](cw * ch * NCH)
    var dep = unsafe_alloc[Float32](cw * ch)
    for row in range(ch):
        for col in range(cw):
            var s = (y0 + row) * Int(fw) + (x0 + col)
            var d = row * cw + col
            for c in range(3):
                main[unsafe_offset=d * NCH + c] = image[unsafe_offset=s * 3 + c]
                main[unsafe_offset=d * NCH + 3 + c] = albedo[unsafe_offset=s * 3 + c]
                main[unsafe_offset=d * NCH + 6 + c] = normals[unsafe_offset=s * 3 + c]
            main[unsafe_offset=d * NCH + 9] = depth[unsafe_offset=s]
            dep[unsafe_offset=d] = depth[unsafe_offset=s]
    var ret = _write_channels(film, main, Int32(cw), Int32(ch), Int32(NCH),
        "R,G,B,Albedo.R,Albedo.G,Albedo.B,N.X,N.Y,N.Z,Z", fw, fh, Int32(x0), Int32(y0))
    var n_albedo = _sidecar_name(film, ".albedo.exr")
    _ = write_image_cropwindow(albedo, fw, fh, cx0, cy0, cx1, cy1, n_albedo, Int32(32), Int32(32))
    var n_normal = _sidecar_name(film, ".normal.exr")
    _ = write_image_cropwindow(normals, fw, fh, cx0, cy0, cx1, cy1, n_normal, Int32(32), Int32(32))
    var n_depth = _sidecar_name(film, ".depth.exr")
    _ = _write_channels(n_depth, dep, Int32(cw), Int32(ch), Int32(1), "Y", fw, fh, Int32(x0), Int32(y0))
    if denoised:
        var n_noisy = _sidecar_name(film, ".noisy.exr")
        _ = write_image_cropwindow(noisy, fw, fh, cx0, cy0, cx1, cy1, n_noisy, Int32(32), Int32(32))
        n_noisy.unsafe_free()
    n_albedo.unsafe_free(); n_normal.unsafe_free(); n_depth.unsafe_free()
    main.unsafe_free(); dep.unsafe_free()
    return ret


def finish_render[Op: Origin[mut=True], Oa: Origin[mut=True]](
    psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin],
    ref sd: SceneView,
    pixels: Pointer[Float32, Op],
    albedo: Pointer[Float32, Oa],
    no_denoise: Bool,
    add_after_denoise: Pointer[Float32, MutUntrackedOrigin] = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
) -> Int32:
    """The tail every VCM and SPPM driver shared: normals and depth from the
    integrator-agnostic first-hit pass (render_aux_buffers), the denoiser
    unless --no-denoise, then write_render_outputs. The caller still owns
    `pixels`, `albedo` and `add_after_denoise`.

    `add_after_denoise` (SPPM's two call sites, a real pointer only there --
    see project_water_caustic_sppm_gap memory) is added to the buffer AFTER
    denoise() runs on `pixels` and BEFORE the max-component sensor clamp,
    which now happens here rather than in the caller's own finalize step:
    SPPM's photon-gather/caustic term is a legitimately high-frequency
    spatial signal (a caustic thread IS 1-2px bright lines on a dim
    background) that denoise()'s spatial-variance edge-stopping cannot
    tell apart from noise -- routing only the NEE/direct "global" term
    through the denoiser, and adding the untouched caustic term back after,
    keeps real GI denoising for ordinary SPPM scenes while no longer
    smearing a caustic's brightest pixels down ~85x. Clamping the SUM once,
    here, rather than clamping global and caustic separately beforehand,
    matters for the ~48 corpus scenes that set `maxcomponentvalue`: clamping
    each half independently against the same absolute limit would
    under-clamp their sum. VCM's three call sites never pass a real pointer
    here, so they are unaffected -- and already clamp before calling in."""
    var fw = psc[unsafe_offset=0].film_w
    var fh = psc[unsafe_offset=0].film_h
    var n_pix = Int(fw) * Int(fh)
    var normals = unsafe_alloc[Float32](n_pix * 3)
    var depth = unsafe_alloc[Float32](n_pix)
    var sd_local = sd
    render_aux_buffers(psc[unsafe_offset=0].raster_to_camera, psc[unsafe_offset=0].camera_to_world,
                       Int32(0), Int32(0), fw, fh, Pointer(to=sd_local), normals, depth)
    var out = unsafe_alloc[Float32](n_pix * 3)
    if no_denoise:
        for i in range(n_pix * 3):
            out[unsafe_offset=i] = pixels[unsafe_offset=i]
    else:
        denoise(pixels, albedo, normals, depth, fw, fh, out,
                Int32(5), Float32(4.0), Float32(0.1), Float32(0.3), Float32(0.05))
    var has_extra = _is_real_ptr(add_after_denoise)
    var noisy_ref = pixels.unsafe_origin_cast[MutUntrackedOrigin]()
    var noisy_owned = Pointer[Float32, MutUntrackedOrigin].unsafe_dangling()
    if has_extra:
        var max_comp = psc[unsafe_offset=0].film_max_comp
        noisy_owned = unsafe_alloc[Float32](n_pix * 3)
        for i in range(n_pix):
            var og = RGB(out[unsafe_offset=i*3+0] + add_after_denoise[unsafe_offset=i*3+0],
                          out[unsafe_offset=i*3+1] + add_after_denoise[unsafe_offset=i*3+1],
                          out[unsafe_offset=i*3+2] + add_after_denoise[unsafe_offset=i*3+2])
            og = og.sensor_clamped(max_comp)
            out[unsafe_offset=i*3+0] = og.r; out[unsafe_offset=i*3+1] = og.g; out[unsafe_offset=i*3+2] = og.b
            var ng = RGB(pixels[unsafe_offset=i*3+0] + add_after_denoise[unsafe_offset=i*3+0],
                         pixels[unsafe_offset=i*3+1] + add_after_denoise[unsafe_offset=i*3+1],
                         pixels[unsafe_offset=i*3+2] + add_after_denoise[unsafe_offset=i*3+2])
            ng = ng.sensor_clamped(max_comp)
            noisy_owned[unsafe_offset=i*3+0] = ng.r; noisy_owned[unsafe_offset=i*3+1] = ng.g; noisy_owned[unsafe_offset=i*3+2] = ng.b
        noisy_ref = noisy_owned
    var ret = write_render_outputs(psc, out, noisy_ref, not no_denoise, albedo, normals, depth)
    if has_extra:
        noisy_owned.unsafe_free()
    normals.unsafe_free(); depth.unsafe_free(); out.unsafe_free()
    return ret
