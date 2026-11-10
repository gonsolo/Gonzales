"""Every environment switch Gonzales reads, one line each. Mojo reads them only through env_str/env_on/env_int/env_set,
which check the name against ENV_SWITCHES at compile time; entries marked (C), (C++) or (Python) are read outside Mojo."""
from std.os import getenv, setenv
from std.sys import stderr


@fieldwise_init
struct EnvSwitch(ImplicitlyCopyable):
    var name: StaticString
    var desc: StaticString


comptime ENV_SWITCHES: List[EnvSwitch] = [
    # Files and directories
    EnvSwitch("GONZALES_DATA_DIR", "directory of the spectral tables and data (default src/gonzales/data)"),
    EnvSwitch("GONZALES_SCENE_CACHE", "file: parsed-scene cache, written once, read on later starts; .rt and .heat go next to it"),
    EnvSwitch("GONZALES_RT_CACHE", "file: RT-core structure cache (default <GONZALES_SCENE_CACHE>.rt)"),
    EnvSwitch("GONZALES_XERT_HEAT", "file: Intel Xe2 per-instance hit counts for VRAM placement (default <GONZALES_SCENE_CACHE>.heat)"),
    EnvSwitch("GONZALES_CAMERA_SCENE", "pbrt file: camera and maxdepth from it, so one scene cache serves several views"),
    EnvSwitch("GONZALES_RTCORE_CUBIN", "file: RT-core trace cubin (default rt_trace.cubin next to the binary)"),
    EnvSwitch("GONZALES_HYDRA_DIR", "(Python) Blender add-on: directory of the Hydra plugin"),
    # Features
    EnvSwitch("GONZALES_NO_DENOISE", "set: --interactive-frames output is not denoised"),
    EnvSwitch("GONZALES_NO_MNEE", "set: the GPU path tracer runs no MNEE through smooth glass"),
    EnvSwitch("GONZALES_NO_SW_SHADOW", "set: the software BVH traces shadow rays inline instead of deferring them"),
    EnvSwitch("GONZALES_NO_SW_COATED", "set: coated materials trace their shadow rays inline even when others defer them"),
    EnvSwitch("GONZALES_SSS_RT", "0: subsurface walk segments traced in software instead of on the RT cores"),
    EnvSwitch("GONZALES_TEX_SPP", "Ptex streaming footprint as at N samples a pixel (default 16; 1 = the pixel's own)"),
    EnvSwitch("GONZALES_CULL_SIZE", "fraction of image height: drop instances smaller than that from the camera (default 0 = none)"),
    EnvSwitch("GONZALES_MERGE_TEMPLATES", "0: object-template meshes stay apart (set for the Vulkan backends; default merged)"),
    # RT cores and acceleration structures
    EnvSwitch("GONZALES_RTCORE", "set: trace on the NVIDIA RT cores (--rt-hardware sets it; Hydra sessions read it)"),
    EnvSwitch("GONZALES_RTCORE_VULKAN", "set: --rt-hardware builds its structures with Vulkan, not the own builders"),
    EnvSwitch("GONZALES_RTCORE_NOSHADOW", "set: no shadow rays deferred to the RT cores"),
    EnvSwitch("GONZALES_RT_PRUNE", "1: software structures drop instance triangles, the RT cores trace them; 8+bits: single steps"),
    EnvSwitch("GONZALES_RT_ABLATE", "loose,noquads: switch single RT builder steps off (size ablation)"),
    EnvSwitch("GONZALES_RT_NO_ORD", "set: ordinary (non-instanced) meshes left out of the own RT structures (debug)"),
    EnvSwitch("GONZALES_CURVE_BVH", "0: curves stay in the scene BVH, no curve structure next to the RT cores"),
    EnvSwitch("GONZALES_OPTIX_CURVES", "0: curves as software clusters instead of OptiX curves"),
    EnvSwitch("GONZALES_RT_COMPACT", "(C++) 0: Vulkan acceleration structures are not compacted"),
    EnvSwitch("GONZALES_XERT_SPLITS", "(C) 1: the Intel RT builder keeps spatial splits (bigger BLASes)"),
    EnvSwitch("GONZALES_WIDE_BVH", "1: 8-wide quantised nodes for quad templates and the instance tree"),
    EnvSwitch("GONZALES_QUAD_LEAF", "quads a leaf in quad templates (default 8)"),
    EnvSwitch("GONZALES_INST_LEAF", "instances a leaf in the instance BVH (default 8)"),
    # Memory placement and budgets
    EnvSwitch("GONZALES_MESH_HOST", "KB: mesh arrays from this size stay in host memory (0 = none; default automatic)"),
    EnvSwitch("GONZALES_ENV_HOST", "KB: environment-light pixels from this size stay in host memory (default automatic)"),
    EnvSwitch("GONZALES_ENV_DOWN", "1: environment map at half resolution on the GPU; other values: never (default automatic)"),
    EnvSwitch("GONZALES_MESH_VRAM_MB", "MiB of the host-placed mesh arrays kept on the card, hottest first"),
    EnvSwitch("GONZALES_PTEX_BUDGET_MB", "MiB for Ptex texels (default 2048)"),
    EnvSwitch("GONZALES_PTEX_LOG2RES", "(C++) log2 of a uniform per-face Ptex resolution cap; no paging"),
    EnvSwitch("GONZALES_XERT_VRAM_MB", "Intel Xe2: MiB of RT structures and mesh arrays in VRAM (default automatic)"),
    EnvSwitch("GONZALES_XERT_RESERVE_MB", "Intel Xe2 automatic VRAM budget: MiB left free (default 2800)"),
    # Logs, timing, dumps
    EnvSwitch("GONZALES_TIMING", "set: stage times of Hydra sessions and the curve setup"),
    EnvSwitch("GZ_VCM_TIMING", "1: GPU time of each VCM wavefront phase; 2: also each launch as it runs"),
    EnvSwitch("GONZALES_FRAME_LOG", "set: stage times of each interactive frame"),
    EnvSwitch("GONZALES_BENCH", "N: --interactive-frames ray counts and times after N warm-up frames"),
    EnvSwitch("GONZALES_BENCH_TRACE", "set with GONZALES_BENCH: trace and shade timed apart"),
    EnvSwitch("GONZALES_TRAVERSE_COUNT", "set: traversal steps and ns a ray of the software BVH"),
    EnvSwitch("GONZALES_ROUND_LOG", "set: live paths after each bounce round"),
    EnvSwitch("GONZALES_ADAPTIVE_LOG", "set: active pixels after each adaptive-sampling update of a Hydra session"),
    EnvSwitch("GONZALES_GPU_LOG", "set: free GPU memory after each upload stage, uploads from 8 MB"),
    EnvSwitch("GONZALES_SCENE_CACHE_LOG", "set: scene-cache arrays of 100 MB and more"),
    EnvSwitch("GONZALES_RT_ALPHA_LOG", "set: alpha-test hits left after the RT-core retrace passes"),
    EnvSwitch("GONZALES_RT_AS_LOG", "(C++) set: sizes of the Vulkan acceleration structures"),
    EnvSwitch("GONZALES_SHOW_GPU", "(C++) set: the viewer starts with its graphics-card overlay (key G)"),
    EnvSwitch("GONZALES_GEOM_DUMP", "prefix: triangles and instances of the RT structures written to files"),
    EnvSwitch("GONZALES_RT_DUMP", "directory: the own RT-core structures written there"),
    EnvSwitch("GONZALES_RAY_DUMP", "prefix: one GONZALES_BENCH frame's packed rays to files, bare RT-core trace timed"),
    EnvSwitch("GONZALES_TEST_MOVE", "float: the --interactive-frames camera slides sideways this much a frame"),
    EnvSwitch("VULKANRT_DEBUG_AABB", "(C++) 1|2[:first mesh]: Vulkan RT builds triangle boxes instead (2 opaque; docs/rtcore)"),
    # gzrecord/gzreplay (C), around libgonzales
    EnvSwitch("GZ_LIB", "(C) gzrecord: the libgonzales it forwards to (default libgonzales.so)"),
    EnvSwitch("GZ_RECORD", "(C) gzrecord: file the Hydra calls are recorded to"),
    EnvSwitch("GZ_DEPTH", "(C) gzreplay: max depth instead of the recorded one"),
    EnvSwitch("GZ_FILTER_GLOSSY", "(C) gzreplay: filter-glossy value instead of the recorded one"),
    EnvSwitch("GZ_CLAMP", "(C) gzreplay: indirect clamp instead of the recorded one"),
    EnvSwitch("GZ_BOUNCES", "(C) gzreplay: diffuse,glossy,transmission bounce limits"),
    EnvSwitch("GZ_INTEGRATOR", "(C) gzreplay: integrator number instead of the recorded one"),
    EnvSwitch("GZ_DENOISE", "(C) gzreplay: 0/1 overrides the recorded denoise choice"),
]


def _registered[name: StaticString]() -> Bool:
    comptime for i in range(len(ENV_SWITCHES)):
        comptime if ENV_SWITCHES[i].name == name:
            return True
    return False


# The value; `default` only when unset (set but empty stays empty).
def env_str[name: StaticString](default: String = "") -> String:
    comptime assert _registered[name](), "environment switch not in ENV_SWITCHES (core/env.mojo): " + name
    return getenv(name, default)


def env_on[name: StaticString]() -> Bool:
    return env_str[name]() != ""


# `default` when unset or empty; raises on a value that is no integer.
def env_int[name: StaticString](default: Int) raises -> Int:
    var v = env_str[name]()
    return default if v == "" else atol(v)


def env_registered(name: String) -> Bool:
    comptime for i in range(len(ENV_SWITCHES)):
        comptime known = ENV_SWITCHES[i].name
        if name == known:
            return True
    return False


# Warns once for each GONZALES_* or GZ_* variable at startup that is no switch (a typo).
def env_warn_unknown():
    var text: String
    try:
        with open("/proc/self/environ", "r") as f:
            text = f.read()
    except:
        return
    for entry in text.split("\0"):
        if not (entry.startswith("GONZALES_") or entry.startswith("GZ_")):
            continue
        var name = String(entry.split("=")[0])
        if not env_registered(name):
            print("Warning:", name, "is set but is no Gonzales environment switch (gonzales --list-env)", file=stderr)


def env_set[name: StaticString](value: String):
    comptime assert _registered[name](), "environment switch not in ENV_SWITCHES (core/env.mojo): " + name
    _ = setenv(name, value, True)


# --list-env: every switch, its value when set, and what it does.
def env_list():
    comptime for i in range(len(ENV_SWITCHES)):
        comptime s = ENV_SWITCHES[i]
        var v = getenv(s.name)
        print(s.name + ("=" + v if v != "" else ""))
        print("    " + s.desc)
