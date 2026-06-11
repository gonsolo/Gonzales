from std.sys import argv
from std.time import perf_counter_ns
from std.os import getenv
from std.memory import alloc
from mojoKernel.pipeline import _generate_sobol_matrices, mojo_parse_and_render, mojo_render_interactive

def main() raises:
    var t0 = perf_counter_ns()

    var args = argv()
    var scene_path = String("")
    var interactive = False
    var use_gpu = False
    var i = 1
    while i < len(args):
        var arg = String(args[i])
        if arg == "--help" or arg == "-h":
            print("Usage: gonzales [--interactive] [--gpu] scene.pbrt")
            return
        elif arg == "--interactive":
            interactive = True
        elif arg == "--gpu":
            use_gpu = True
        else:
            scene_path = arg
        i += 1

    if scene_path.byte_length() == 0:
        print("Usage: gonzales [--interactive] [--gpu] scene.pbrt")
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
        mojo_render_interactive(path_cstr, sobol, use_gpu)
    else:
        _ = mojo_parse_and_render(path_cstr, sobol, use_gpu)
        var elapsed_s = Float64(perf_counter_ns() - t0) / 1_000_000_000.0
        print("Gonzales Total Execution Time:", elapsed_s, "s")

    path_cstr.free()
    sobol.free()
