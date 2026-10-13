from .pbrt_parser import ParsedScene_Mojo, mojo_parse_scene, resize_film
from .ply_prefetch import PlyPrefetch
from .mitsuba_parser import mojo_parse_mitsuba_scene
from .scene_cache import scene_cache_load, scene_cache_save
from std.os import getenv
from .geometry import _is_real_ptr
from std.memory.alloc import unsafe_alloc

# Dispatches on scene-file extension so pipeline.mojo's four call sites don't
# need to know which parser owns which format. Lives outside both
# pbrt_parser.mojo and mitsuba_parser.mojo to avoid a circular import:
# mitsuba_parser.mojo already depends on pbrt_parser.mojo for
# ParsedScene_Mojo/finalize_scene, so the reverse dependency has to live
# elsewhere.
# GONZALES_CAMERA_SCENE=<pbrt file>: take camera and maxdepth from it, so one scene cache serves several views.
def _camera_override(psc: Pointer[ParsedScene_Mojo, MutUntrackedOrigin]):
    var env = getenv("GONZALES_CAMERA_SCENE")
    if env.byte_length() == 0:
        return
    var cpath = unsafe_alloc[UInt8](env.byte_length() + 1)
    for k in range(env.byte_length()):
        cpath[unsafe_offset=k] = env.as_bytes()[k]
    cpath[unsafe_offset=env.byte_length()] = UInt8(0)
    var cam = mojo_parse_scene(cpath, False, Pointer[PlyPrefetch, MutUntrackedOrigin].unsafe_dangling())
    cpath.unsafe_free()
    for i in range(16):
        psc[unsafe_offset=0].camera_to_world[unsafe_offset=i] = cam[unsafe_offset=0].camera_to_world[unsafe_offset=i]
    psc[unsafe_offset=0].camera_fov = cam[unsafe_offset=0].camera_fov
    psc[unsafe_offset=0].max_depth = cam[unsafe_offset=0].max_depth
    resize_film(psc, psc[unsafe_offset=0].film_w, psc[unsafe_offset=0].film_h)

def mojo_parse_scene_any(path: Pointer[UInt8, MutUntrackedOrigin],
                         verbose: Bool = False,
                         plys: Pointer[PlyPrefetch, MutUntrackedOrigin] = Pointer[PlyPrefetch, MutUntrackedOrigin].unsafe_dangling(),
                        ) -> Pointer[ParsedScene_Mojo, MutUntrackedOrigin]:
    var pi = 0
    while path[unsafe_offset=pi] != UInt8(0):
        pi += 1
    var is_xml = (pi >= 4 and path[unsafe_offset=pi - 4] == UInt8(46) and path[unsafe_offset=pi - 3] == UInt8(120)
                  and path[unsafe_offset=pi - 2] == UInt8(109) and path[unsafe_offset=pi - 1] == UInt8(108))
    if is_xml:
        return mojo_parse_mitsuba_scene(path, verbose)
    var cache_env = getenv("GONZALES_SCENE_CACHE")
    if cache_env.byte_length() == 0:
        return mojo_parse_scene(path, verbose, plys)
    var cpath = unsafe_alloc[UInt8](cache_env.byte_length() + 1)
    for k in range(cache_env.byte_length()):
        cpath[unsafe_offset=k] = cache_env.as_bytes()[k]
    cpath[unsafe_offset=cache_env.byte_length()] = UInt8(0)
    var cached = scene_cache_load(cpath, path)
    if _is_real_ptr[ParsedScene_Mojo](cached):
        cpath.unsafe_free()
        _camera_override(cached)
        return cached
    var psc = mojo_parse_scene(path, verbose, plys)
    _ = scene_cache_save(psc, cpath, path)
    cpath.unsafe_free()
    return psc
