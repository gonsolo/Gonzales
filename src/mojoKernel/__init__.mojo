from std.sys import argv
from std.time import perf_counter_ns
from std.os import getenv
from std.memory import alloc
from mojoKernel.pipeline import _generate_sobol_matrices, mojo_parse_and_render

fn main() raises:
    var t0 = perf_counter_ns()

    var args = argv()
    var scene_path = String("")
    var i = 1
    while i < len(args):
        var arg = String(args[i])
        if arg == "--help" or arg == "-h":
            print("Usage: gonzales scene.pbrt")
            return
        scene_path = arg
        i += 1

    if len(scene_path) == 0:
        print("Usage: gonzales scene.pbrt")
        return

    var data_dir = getenv("GONZALES_DATA_DIR", "src/mojoKernel/data")
    var sobol = _generate_sobol_matrices(data_dir + "/new-joe-kuo-6.21201")
    if not sobol:
        return

    var path_len = len(scene_path)
    var path_cstr = alloc[UInt8](path_len + 1)
    for k in range(path_len):
        path_cstr[k] = scene_path.as_bytes()[k]
    path_cstr[path_len] = UInt8(0)

    _ = mojo_parse_and_render(path_cstr, sobol)

    path_cstr.free()
    sobol.free()

    var elapsed_s = Float64(perf_counter_ns() - t0) / 1_000_000_000.0
    print("Gonzales Total Execution Time:", elapsed_s, "s")
