from std.sys import argv
from std.time import perf_counter_ns
from std.os import getenv
from std.memory import alloc
from gonzales.pipeline import _generate_sobol_matrices, parse_and_render, render_interactive, debug_trace_pixel

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
    var sppm_photons = Int32(200000)
    var sppm_radius = Float32(0.05)
    var use_guide = False
    var use_bdpt = False
    var bdpt_spp = Int32(64)
    var i = 1
    while i < len(args):
        var arg = String(args[i])
        if arg == "--help" or arg == "-h":
            print("Usage: gonzales [--interactive] [--gpu] [--fullscreen] [--no-denoise] [--verbose] [--spp N] [--resolution WxH] [--width W] [--height H] [--pixel X Y] scene.pbrt")
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
        elif arg == "--bdpt":
            use_bdpt = True
        elif arg == "--bdpt-spp" and i + 1 < len(args):
            i += 1
            bdpt_spp = _parse_int32(String(args[i]), 0)
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

    var path_len = scene_path.byte_length()
    var path_cstr = alloc[UInt8](path_len + 1)
    for k in range(path_len):
        path_cstr[k] = scene_path.as_bytes()[k]
    path_cstr[path_len] = UInt8(0)

    if pixel_x >= 0 and pixel_y >= 0:
        debug_trace_pixel(path_cstr, pixel_x, pixel_y)
    elif interactive:
        render_interactive(path_cstr, sobol, use_gpu, fullscreen, override_w, override_h, spp_override, verbose)
    else:
        _ = parse_and_render(path_cstr, sobol, use_gpu, override_w, override_h, no_denoise, spp_override, verbose, use_sppm, sppm_passes, sppm_photons, sppm_radius, use_guide, use_bdpt, bdpt_spp)
        var elapsed_s = Float64(perf_counter_ns() - t0) / 1_000_000_000.0
        print("Gonzales Total Execution Time:", elapsed_s, "s")

    path_cstr.free()
    sobol.free()
