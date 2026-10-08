from std.sys import argv, exit
from std.time import perf_counter_ns
from std.os import getenv, setenv
from std.memory.alloc import unsafe_alloc
from gonzales.pipeline import _generate_sobol_matrices, parse_and_render, render_interactive, debug_trace_pixel, debug_render_vulkanrt
from gonzales.spectrum import load_spectral_context, spectral_handle, SpectralContext
from gonzales.pbrt_parser import ParsedScene_Mojo
from gonzales.ply_prefetch import PlyPrefetch, scan_plys
from gonzales.os_thread import OsThread, ThreadArg
from gonzales.scene_loader import mojo_parse_scene_any
from gonzales.gpu_scene import gpu_available
from max.gpu.host import DeviceContext
from gonzales.sensors import scan_film_sensor_params

def _parse_int32(s: String, start: Int) -> Int32:
    var v = Int32(0)
    var n = s.byte_length()
    var j = start
    while j < n:
        var c = Int32(s.as_bytes()[j])
        if c < Int32(48) or c > Int32(57):
            break
        v = v * Int32(10) + c - Int32(48)
        j += 1
    return v

def _parse_float32(s: String) -> Float32:
    var v = Float32(0)
    var n = s.byte_length()
    var j = 0
    var frac = Float32(0)
    var frac_div = Float32(1)
    var after_dot = False
    while j < n:
        var c = Int(s.as_bytes()[j])
        if c == 46:  # '.'
            after_dot = True
        elif c >= 48 and c <= 57:
            if after_dot:
                frac_div *= Float32(10)
                frac += Float32(c - 48) / frac_div
            else:
                v = v * Float32(10) + Float32(c - 48)
        j += 1
    return v + frac

def _parse_res(s: String, start: Int) -> Tuple[Int32, Int32]:
    var n = s.byte_length()
    var j = start
    var wv = Int32(0)
    while j < n and s.as_bytes()[j] != UInt8(120):  # 'x'
        wv = wv * Int32(10) + Int32(s.as_bytes()[j]) - Int32(48)
        j += 1
    j += 1
    var hv = Int32(0)
    while j < n:
        hv = hv * Int32(10) + Int32(s.as_bytes()[j]) - Int32(48)
        j += 1
    return (wv, hv)

# Creates the CUDA context (about 130 ms of driver start-up) off the parse's critical path.
def _create_context_thread(arg: ThreadArg) -> ThreadArg:
    var slot = arg.unsafe_bitcast[Optional[DeviceContext]]()
    try:
        slot[unsafe_offset=0] = Optional[DeviceContext](DeviceContext())
    except:
        pass
    return arg

# The Sobol matrices and the spectral tables, loaded while the scene is parsed.
# Load the real Jakob-Hanika spectral upsampling table once (staged
# spectral rendering rollout, see project_spectral_rendering memory)
# and keep the owning SpectralContext alive for the whole render — the
# SpectralHandle threaded everywhere else is just raw pointers into it.
# Missing table -> null_spectral_handle() default everywhere downstream
# (same "unwired yet" behavior as before this table existed), not a
# fatal error, so a stale/missing data dir doesn't block rendering.
# A named sensor (e.g. "nikon_d850") needs its measured response curves
# baked into the CieXyzTables from the start (see sensors.mojo's header)
# -- cheap enough to just re-read the scene file here for a pre-scan,
# since the real parse (which also reads "string sensor"/"float
# whitebalance", but too late for this) hasn't happened yet.
struct _StartupBundle(Movable):
    var data_dir: String
    var scene_path: String
    var sobol: Optional[Pointer[UInt32, MutUntrackedOrigin]]
    var spectral: Optional[SpectralContext]
    var spectral_ok: Bool

    def __init__(out self, data_dir: String, scene_path: String):
        self.data_dir = data_dir
        self.scene_path = scene_path
        self.sobol = None
        self.spectral = None
        self.spectral_ok = False

    def run(mut self):
        self.sobol = _generate_sobol_matrices(self.data_dir + "/new-joe-kuo-6.21201")
        var sensor_name = String("cie1931")
        var sensor_wb = Float32(0.0)
        try:
            var scene_text = open(self.scene_path, "r").read()
            var scan = scan_film_sensor_params(scene_text)
            sensor_name = scan[0]
            sensor_wb = scan[1]
        except:
            pass
        var loaded = load_spectral_context(self.data_dir, sensor_name, sensor_wb)
        self.spectral_ok = loaded[0]
        self.spectral = loaded[1].copy()

def _load_tables_thread(arg: ThreadArg) -> ThreadArg:
    arg.unsafe_bitcast[_StartupBundle]()[unsafe_offset=0].run()
    return arg

def _ply_loader_thread(arg: ThreadArg) -> ThreadArg:
    arg.unsafe_bitcast[PlyPrefetch]()[unsafe_offset=0].run_loader()
    return arg

def main() raises:
    var t0 = perf_counter_ns()

    var args = argv()
    var scene_path = String("")
    var interactive = False
    var use_gpu = False
    var fullscreen = False
    var override_w = Int32(0)
    var override_h = Int32(0)
    var pixel_x = Int32(-1)
    var pixel_y = Int32(-1)
    var no_denoise = False
    var verbose = False
    var spp_override = Int32(0)
    var seed_override = Int64(-1)
    var use_sppm = False
    var sppm_passes = Int32(64)
    var sppm_photons = Int32(-1)   # -1 = not passed on CLI; fall back to scene/pbrt-matching default
    var sppm_radius = Float32(-1)  # -1 = not passed on CLI; fall back to scene/pbrt-matching default
    var use_guide = False
    var use_vcm = False
    var vcm_spp = Int32(-1)  # -1 = not passed on CLI; fall back to scene pixelsamples
    var vcm_photons = Int32(-1)  # -1 = not passed on CLI; fall back to n_pix (today's default)
    var vcm_budget = False       # --vcm-budget: per-cell photon budget instead of the fixed cap
    var vcm_cap = Int32(0)       # --vcm-cap N: merge-bucket cap (0 = _PHOTON_BUCKET_CAP)
    var vcm_no_keep_mis = False  # --vcm-no-keep-mis: thin, but hide it from merging's MIS
    var vcm_radius_from_camera = False  # --vcm-radius-from-camera: EXPERIMENTAL merge-radius
                                 # ceiling from median primary-ray depth instead of the whole
                                 # scene's bounding sphere (vcm_render_gpu); see bdpt_*.mojo's
                                 # _camera_typical_distance docstring
    var vcm_radius_cam_percentile = Float32(0.5)      # --vcm-radius-cam-percentile P (0..1)
    var vcm_radius_cam_fraction_mult = Float32(1.0)   # --vcm-radius-cam-fraction-mult M
    var vcm_radius_scale = Float32(1.0)   # --vcm-radius-scale S: scales the whole merge radius (ceiling, grid, MIS, footprint)
    var vcm_no_footprint = False  # --vcm-no-footprint: the "naive VCM" baseline, one fixed
                                 # global radius instead of per-vertex footprint scaling
    var use_vulkan_rt = False
    var use_vulkan_rt_shade = False
    var use_vcm_wavefront = False
    var use_restir = False  # docs/A2_restir_migration_plan.md Phase 2/3: ReSTIR DI
                             # (area-light NEE only, primary bounce), CPU + GPU batch
    var use_restir_gi = False  # Phase 4: ReSTIR GI, one-bounce reconnection, scoped to
                             # diffuse x1 AND diffuse x2 (see shading.mojo's
                             # _gi_generate_recon_candidate). CPU batch only so far, no
                             # temporal/spatial reuse yet (matches --restir's own batch-
                             # mode scope) -- requires --restir, no effect alone.
    var use_sms_restir = False  # Phase 6: ReSTIR SMS, temporal reuse for glass-caustic
                             # MNEE probing (shading.mojo's sms_temporal_step). CPU +
                             # --interactive only, INDEPENDENT of --restir (unlike
                             # --restir-gi) -- no effect in batch mode, since a single
                             # SMS candidate with no reservoir combine is mathematically
                             # identical to plain per-frame MNEE.
    var use_vol_restir_reuse = False  # Phase 7.3: volume-scatter TEMPORAL reuse (no
                             # spatial -- see project_restir_migration memory's "7.3"
                             # section). GPU only so far (gpu_render_sample's per-pixel
                             # dispatch, both true --interactive-frames and batch
                             # --gpu, which switches dispatch mode the same way
                             # --restir already does). INDEPENDENT of --restir --
                             # this is the medium sampler's own NEE, not DI's.
    var headless_frames = Int32(0)  # --interactive-frames: run render_interactive's
                             # per-frame loop N times with no window/camera polling,
                             # then write the result like a normal batch render --
                             # lets interactive-only features (temporal ReSTIR reuse)
                             # be verified by ordinary render-diffing. See
                             # project_restir_migration.md memory for why this exists.
    var i = 1
    while i < len(args):
        var arg = String(args[i])
        if arg == "--help" or arg == "-h":
            print("Usage: gonzales [--interactive] [--gpu] [--fullscreen] [--no-denoise] [--verbose] [--spp N] [--seed N] [--resolution WxH] [--width W] [--height H] [--pixel X Y] [--vulkan-rt] scene.pbrt")
            return
        elif arg == "--interactive":
            interactive = True
        elif arg == "--gpu":
            use_gpu = True
        elif arg == "--no-denoise":
            no_denoise = True
        elif arg == "--fullscreen":
            fullscreen = True
            interactive = True
        elif arg == "--verbose":
            verbose = True
        elif arg == "--spp" and i + 1 < len(args):
            i += 1
            spp_override = _parse_int32(String(args[i]), 0)
        elif arg.startswith("--spp="):
            spp_override = _parse_int32(arg, 6)
        elif arg == "--seed" and i + 1 < len(args):
            i += 1
            seed_override = Int64(_parse_int32(String(args[i]), 0))
        elif arg.startswith("--seed="):
            seed_override = Int64(_parse_int32(arg, 7))
        elif arg == "--resolution" and i + 1 < len(args):
            i += 1
            var wh = _parse_res(String(args[i]), 0)
            override_w = wh[0]
            override_h = wh[1]
        elif arg.startswith("--resolution="):
            var wh = _parse_res(arg, 13)
            override_w = wh[0]
            override_h = wh[1]
        elif arg == "--width" and i + 1 < len(args):
            i += 1
            override_w = Int32(atol(String(args[i])))
        elif arg == "--height" and i + 1 < len(args):
            i += 1
            override_h = Int32(atol(String(args[i])))
        elif arg == "--pixel" and i + 2 < len(args):
            i += 1
            pixel_x = Int32(atol(String(args[i])))
            i += 1
            pixel_y = Int32(atol(String(args[i])))
        elif arg == "--guide":
            use_guide = True
        elif arg == "--vulkan-rt":
            use_vulkan_rt = True
        elif arg == "--vulkan-rt-shade":
            use_vulkan_rt_shade = True
        elif arg == "--rt-hardware":
            # Wavefront GPU path tracing with the RT cores driven from a CUDA kernel (docs/rtcore/NOTES.md); implies
            # --vulkan-rt-shade (same buffers, Vulkan keeps building the acceleration structure).
            use_vulkan_rt_shade = True
            use_vcm_wavefront = True          # with --vcm: the staged VCM driver is the one with the trace hook
            _ = setenv("GONZALES_RTCORE", "1", True)
        elif arg == "--vcm":
            use_vcm = True
        elif arg == "--vcm-wavefront":
            use_vcm_wavefront = True
        elif arg == "--restir":
            use_restir = True
        elif arg == "--restir-gi":
            use_restir_gi = True
        elif arg == "--sms-restir":
            use_sms_restir = True
        elif arg == "--vol-restir-reuse":
            use_vol_restir_reuse = True
        elif arg == "--interactive-frames" and i + 1 < len(args):
            i += 1
            headless_frames = _parse_int32(String(args[i]), 0)
            interactive = True
        elif arg == "--vcm-spp" and i + 1 < len(args):
            i += 1
            vcm_spp = _parse_int32(String(args[i]), 0)
        elif arg == "--vcm-photons" and i + 1 < len(args):
            i += 1
            vcm_photons = _parse_int32(String(args[i]), 0)
        elif arg == "--vcm-budget":
            vcm_budget = True
        elif arg == "--vcm-no-keep-mis":
            vcm_no_keep_mis = True
        elif arg == "--vcm-no-footprint":
            vcm_no_footprint = True
        elif arg == "--vcm-radius-from-camera":
            vcm_radius_from_camera = True
        elif arg == "--vcm-radius-cam-percentile" and i + 1 < len(args):
            i += 1
            vcm_radius_cam_percentile = _parse_float32(String(args[i]))
        elif arg == "--vcm-radius-cam-fraction-mult" and i + 1 < len(args):
            i += 1
            vcm_radius_cam_fraction_mult = _parse_float32(String(args[i]))
        elif arg == "--vcm-radius-scale" and i + 1 < len(args):
            i += 1
            vcm_radius_scale = _parse_float32(String(args[i]))
        elif arg == "--vcm-cap" and i + 1 < len(args):
            i += 1
            vcm_cap = _parse_int32(String(args[i]), 0)
        elif arg == "--sppm":
            use_sppm = True
        elif arg == "--sppm-passes" and i + 1 < len(args):
            i += 1
            sppm_passes = _parse_int32(String(args[i]), 0)
        elif arg == "--sppm-photons" and i + 1 < len(args):
            i += 1
            sppm_photons = _parse_int32(String(args[i]), 0)
        elif arg == "--sppm-radius" and i + 1 < len(args):
            i += 1
            # Parse float radius
            var rs = String(args[i])
            var rv = Float32(0)
            var rn = rs.byte_length()
            var rj = 0
            var rfrac = Float32(0)
            var rfrac_div = Float32(1)
            var after_dot = False
            while rj < rn:
                var rc = Int(rs.as_bytes()[rj])
                if rc == 46:  # '.'
                    after_dot = True
                elif rc >= 48 and rc <= 57:
                    if after_dot:
                        rfrac_div *= Float32(10)
                        rfrac += Float32(rc - 48) / rfrac_div
                    else:
                        rv = rv * Float32(10) + Float32(rc - 48)
                rj += 1
            sppm_radius = rv + rfrac
        else:
            scene_path = arg
        i += 1

    if scene_path.byte_length() == 0:
        print("Usage: gonzales [--interactive] [--gpu] [--fullscreen] [--no-denoise] [--verbose] [--spp N] [--seed N] [--resolution WxH] [--width W] [--height H] scene.pbrt")
        return

    if not scene_path.endswith(".pbrt") and not scene_path.endswith(".xml"):
        print("Error: expected a .pbrt or .xml (Mitsuba) scene file, got:", scene_path)
        return

    var data_dir = getenv("GONZALES_DATA_DIR", "src/gonzales/data")

    var path_len = scene_path.byte_length()
    var path_cstr = unsafe_alloc[UInt8](path_len + 1)
    for k in range(path_len):
        path_cstr[unsafe_offset=k] = scene_path.as_bytes()[k]
    path_cstr[unsafe_offset=path_len] = UInt8(0)

    # The Sobol matrices, the spectral tables, the CUDA context, the PLY meshes and the scene parse do not depend on each
    # other, so they run side by side. The parse (the longest) stays on this thread; the rest run on threads of their own
    # (OsThread), only for the drivers that parse in parse_and_render.
    var will_parse = not (pixel_x >= 0 and pixel_y >= 0) and not use_vulkan_rt and not interactive
    var want_ctx = will_parse and use_gpu and gpu_available()

    var ctx_slot = unsafe_alloc[Optional[DeviceContext]](1)
    ctx_slot.unsafe_write(Optional[DeviceContext](None))
    var ctx_thread = OsThread()
    if want_ctx:
        ctx_thread.start(_create_context_thread, ctx_slot.unsafe_bitcast[UInt8]().unsafe_origin_cast[MutUntrackedOrigin]())

    var bundle = unsafe_alloc[_StartupBundle](1)
    bundle.unsafe_write(_StartupBundle(data_dir, scene_path))
    var bundle_thread = OsThread()
    bundle_thread.start(_load_tables_thread, bundle.unsafe_bitcast[UInt8]().unsafe_origin_cast[MutUntrackedOrigin]())

    var scene_dir = String("")
    var last_slash = -1
    for ki in range(path_len):
        if path_cstr[unsafe_offset=ki] == UInt8(47):
            last_slash = ki
    if last_slash >= 0:
        var dir_tmp = unsafe_alloc[UInt8](last_slash + 2)
        for ki in range(last_slash + 1):
            dir_tmp[unsafe_offset=ki] = path_cstr[unsafe_offset=ki]
        dir_tmp[unsafe_offset=last_slash + 1] = UInt8(0)
        scene_dir = String(unsafe_from_utf8_ptr=dir_tmp.as_imm())
        dir_tmp.unsafe_free()
    var plys = unsafe_alloc[PlyPrefetch](1)
    plys.unsafe_write(PlyPrefetch())
    var ply_threads = List[OsThread]()
    if will_parse and scene_path.endswith(".pbrt"):
        var scene_text = String("")
        try:
            scene_text = open(scene_path, "r").read()
        except:
            pass
        plys[unsafe_offset=0] = scan_plys(scene_text.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](), scene_text.byte_length(), scene_dir)
        for _ in range(min(6, plys[unsafe_offset=0].n)):
            ply_threads.append(OsThread())
            ply_threads[len(ply_threads) - 1].start(_ply_loader_thread, plys.unsafe_bitcast[UInt8]().unsafe_origin_cast[MutUntrackedOrigin]())

    var preparsed = Pointer[ParsedScene_Mojo, MutUntrackedOrigin].unsafe_dangling()
    if will_parse:
        preparsed = mojo_parse_scene_any(path_cstr, verbose, plys)

    for ti in range(len(ply_threads)):
        ply_threads[ti].join()
    plys[unsafe_offset=0].free_all()
    _ = plys.unsafe_take_pointee()
    plys.unsafe_free()
    ctx_thread.join()
    bundle_thread.join()
    var gpu_ctx = ctx_slot.unsafe_take_pointee()
    ctx_slot.unsafe_free()
    var loaded = bundle.unsafe_take_pointee()
    bundle.unsafe_free()

    if not loaded.sobol:
        return
    var sobol = loaded.sobol.value()
    var spectral_ctx = loaded.spectral.take()
    var spectral = spectral_handle(spectral_ctx)
    if not loaded.spectral_ok:
        print("Warning: could not load spectral table from " + data_dir + " -- spectral rendering disabled")

    if pixel_x >= 0 and pixel_y >= 0:
        debug_trace_pixel(path_cstr, pixel_x, pixel_y, override_w=override_w, override_h=override_h)
    elif use_vulkan_rt:
        debug_render_vulkanrt(path_cstr, verbose)
    elif interactive:
        render_interactive(path_cstr, sobol, use_gpu, spectral=spectral, fullscreen=fullscreen, override_w=override_w, override_h=override_h, spp_override=spp_override, seed_override=seed_override, verbose=verbose, use_restir=use_restir, use_restir_gi=use_restir_gi, use_sms_restir=use_sms_restir, use_vol_restir_reuse=use_vol_restir_reuse, headless_frames=headless_frames)
    else:
        var rc = parse_and_render(path_cstr, sobol, use_gpu, spectral=spectral, preparsed=preparsed, gpu_ctx=gpu_ctx, override_w=override_w, override_h=override_h, no_denoise=no_denoise, spp_override=spp_override, seed_override=seed_override, verbose=verbose, use_sppm=use_sppm, sppm_passes=sppm_passes, sppm_photons=sppm_photons, sppm_radius=sppm_radius, use_guide=use_guide, use_vcm=use_vcm, vcm_spp=vcm_spp, vcm_photons=vcm_photons, vcm_budget=vcm_budget, vcm_cap=vcm_cap, vcm_no_keep_mis=vcm_no_keep_mis, vcm_radius_from_camera=vcm_radius_from_camera, vcm_radius_cam_percentile=vcm_radius_cam_percentile, vcm_radius_cam_fraction_mult=vcm_radius_cam_fraction_mult, vcm_no_footprint=vcm_no_footprint, vcm_radius_scale=vcm_radius_scale, use_vulkan_rt_shade=use_vulkan_rt_shade, use_vcm_wavefront=use_vcm_wavefront, use_restir=use_restir, use_restir_gi=use_restir_gi, use_sms_restir=use_sms_restir, use_vol_restir_reuse=use_vol_restir_reuse)
        var elapsed_s = Float64(perf_counter_ns() - t0) / 1_000_000_000.0
        print("Gonzales Total Execution Time:", elapsed_s, "s")
        if rc != Int32(0):
            path_cstr.unsafe_free()
            sobol.unsafe_free()
            exit(Int(rc))

    # Keep the owning SpectralContext alive until every render path above
    # has finished. The comment at its declaration says exactly this, but
    # nothing enforced it: `spectral_ctx` is not mentioned again after
    # `spectral_handle(spectral_ctx)`, and that handle is only raw
    # MutUntrackedOrigin pointers INTO it, so the origin erasure hides the
    # dependency and ASAP destruction was free to drop the tables before
    # rendering ever read them.
    #
    # Symptom: --vcm rendered all-NaN in 6 of 8 runs of one binary with a
    # pinned --seed (and the plain path tracer did not, because spectral
    # evaluation is only wired into some paths so far -- the staged
    # rollout). Whether it showed depended on whether the freed pages had
    # been reused, which is why it looked like flakiness rather than a bug.
    # Same pattern, in the tests, was fixed in 06a08d1f.
    _ = spectral_ctx^

    path_cstr.unsafe_free()
    sobol.unsafe_free()
