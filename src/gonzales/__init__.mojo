from std.sys import argv, exit
from std.time import perf_counter_ns
from std.os import getenv
from std.memory import alloc
from gonzales.pipeline import _generate_sobol_matrices, parse_and_render, render_interactive, debug_trace_pixel, debug_render_vulkanrt
from gonzales.spectrum import load_spectral_context, spectral_handle

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

def main() raises:
    var t0 = perf_counter_ns()

    var args = argv()
    var scene_path = String("")
    var interactive = False
    var use_gpu = False
    var fullscreen = False
    var override_w = Int32(0)
    var override_h = Int32(0)
    var override_spp = Int32(0)
    var pixel_x = Int32(-1)
    var pixel_y = Int32(-1)
    var no_denoise = False
    var verbose = False
    var spp_override = Int32(0)
    var use_sppm = False
    var sppm_passes = Int32(64)
    var sppm_photons = Int32(-1)   # -1 = not passed on CLI; fall back to scene/pbrt-matching default
    var sppm_radius = Float32(-1)  # -1 = not passed on CLI; fall back to scene/pbrt-matching default
    var use_guide = False
    var use_vcm = False
    var vcm_spp = Int32(-1)  # -1 = not passed on CLI; fall back to scene pixelsamples
    var vcm_photons = Int32(-1)  # -1 = not passed on CLI; fall back to n_pix (today's default)
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
            print("Usage: gonzales [--interactive] [--gpu] [--fullscreen] [--no-denoise] [--verbose] [--spp N] [--resolution WxH] [--width W] [--height H] [--pixel X Y] [--vulkan-rt] scene.pbrt")
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
        print("Usage: gonzales [--interactive] [--gpu] [--fullscreen] [--no-denoise] [--verbose] [--spp N] [--resolution WxH] [--width W] [--height H] scene.pbrt")
        return

    if not scene_path.endswith(".pbrt"):
        print("Error: expected a .pbrt scene file, got:", scene_path)
        return

    var data_dir = getenv("GONZALES_DATA_DIR", "src/gonzales/data")
    var sobol_opt = _generate_sobol_matrices(data_dir + "/new-joe-kuo-6.21201")
    if not sobol_opt:
        return
    var sobol = sobol_opt.value()

    # Load the real Jakob-Hanika spectral upsampling table once (staged
    # spectral rendering rollout, see project_spectral_rendering memory)
    # and keep the owning SpectralContext alive for the whole render — the
    # SpectralHandle threaded everywhere else is just raw pointers into it.
    # Missing table -> null_spectral_handle() default everywhere downstream
    # (same "unwired yet" behavior as before this table existed), not a
    # fatal error, so a stale/missing data dir doesn't block rendering.
    var spectral_ctx_result = load_spectral_context(data_dir)
    var spectral_ctx = spectral_ctx_result[1].copy()
    var spectral = spectral_handle(spectral_ctx)
    if not spectral_ctx_result[0]:
        print("Warning: could not load spectral table from " + data_dir + " -- spectral rendering disabled")

    var path_len = scene_path.byte_length()
    var path_cstr = alloc[UInt8](path_len + 1)
    for k in range(path_len):
        path_cstr[k] = scene_path.as_bytes()[k]
    path_cstr[path_len] = UInt8(0)

    if pixel_x >= 0 and pixel_y >= 0:
        debug_trace_pixel(path_cstr, pixel_x, pixel_y)
    elif use_vulkan_rt:
        debug_render_vulkanrt(path_cstr, verbose)
    elif interactive:
        render_interactive(path_cstr, sobol, use_gpu, spectral=spectral, fullscreen=fullscreen, override_w=override_w, override_h=override_h, spp_override=spp_override, verbose=verbose, use_restir=use_restir, use_restir_gi=use_restir_gi, use_sms_restir=use_sms_restir, headless_frames=headless_frames)
    else:
        var rc = parse_and_render(path_cstr, sobol, use_gpu, spectral=spectral, override_w=override_w, override_h=override_h, no_denoise=no_denoise, spp_override=spp_override, verbose=verbose, use_sppm=use_sppm, sppm_passes=sppm_passes, sppm_photons=sppm_photons, sppm_radius=sppm_radius, use_guide=use_guide, use_vcm=use_vcm, vcm_spp=vcm_spp, vcm_photons=vcm_photons, use_vulkan_rt_shade=use_vulkan_rt_shade, use_vcm_wavefront=use_vcm_wavefront, use_restir=use_restir, use_restir_gi=use_restir_gi, use_sms_restir=use_sms_restir)
        var elapsed_s = Float64(perf_counter_ns() - t0) / 1_000_000_000.0
        print("Gonzales Total Execution Time:", elapsed_s, "s")
        if rc != Int32(0):
            path_cstr.free()
            sobol.free()
            exit(Int(rc))

    path_cstr.free()
    sobol.free()
