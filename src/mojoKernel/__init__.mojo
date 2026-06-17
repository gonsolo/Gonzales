from std.sys import argv
from std.time import perf_counter_ns
from std.os import getenv
from std.memory import alloc
from mojoKernel.pipeline import _generate_sobol_matrices, parse_and_render, render_interactive

def main() raises:
    var t0 = perf_counter_ns()

    var args = argv()
    var scene_path = String("")
    var interactive = False
    var use_gpu = False
    var fullscreen = False
    var override_w = Int32(0)
    var override_h = Int32(0)
    var i = 1
    while i < len(args):
        var arg = String(args[i])
        if arg == "--help" or arg == "-h":
            print("Usage: gonzales [--interactive] [--gpu] [--fullscreen] [--width W] [--height H] scene.pbrt")
            return
        elif arg == "--interactive":
            interactive = True
        elif arg == "--gpu":
            use_gpu = True
        elif arg == "--fullscreen":
            fullscreen = True
            interactive = True
        elif arg == "--width" and i + 1 < len(args):
            i += 1
            override_w = Int32(atol(String(args[i])))
        elif arg == "--height" and i + 1 < len(args):
            i += 1
            override_h = Int32(atol(String(args[i])))
        else:
            scene_path = arg
        i += 1

    if scene_path.byte_length() == 0:
        print("Usage: gonzales [--interactive] [--gpu] [--fullscreen] [--width W] [--height H] scene.pbrt")
        return

    if not scene_path.endswith(".pbrt"):
        print("Error: expected a .pbrt scene file, got:", scene_path)
        return

    var data_dir = getenv("GONZALES_DATA_DIR", "src/mojoKernel/data")
    var sobol_opt = _generate_sobol_matrices(data_dir + "/new-joe-kuo-6.21201")
    if not sobol_opt:
        return
    var sobol = sobol_opt.value()

    var path_len = scene_path.byte_length()
    var path_cstr = alloc[UInt8](path_len + 1)
    for k in range(path_len):
        path_cstr[k] = scene_path.as_bytes()[k]
    path_cstr[path_len] = UInt8(0)

    if interactive:
        render_interactive(path_cstr, sobol, use_gpu, fullscreen, override_w, override_h)
    else:
        _ = parse_and_render(path_cstr, sobol, use_gpu, override_w, override_h)
        var elapsed_s = Float64(perf_counter_ns() - t0) / 1_000_000_000.0
        print("Gonzales Total Execution Time:", elapsed_s, "s")

    path_cstr.free()
    sobol.free()
