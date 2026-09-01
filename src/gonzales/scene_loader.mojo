from .pbrt_parser import ParsedScene_Mojo, mojo_parse_scene
from .mitsuba_parser import mojo_parse_mitsuba_scene

# Dispatches on scene-file extension so pipeline.mojo's four call sites don't
# need to know which parser owns which format. Lives outside both
# pbrt_parser.mojo and mitsuba_parser.mojo to avoid a circular import:
# mitsuba_parser.mojo already depends on pbrt_parser.mojo for
# ParsedScene_Mojo/finalize_scene, so the reverse dependency has to live
# elsewhere.
def mojo_parse_scene_any(path: UnsafePointer[UInt8, MutAnyOrigin],
                         verbose: Bool = False,
                        ) -> UnsafePointer[ParsedScene_Mojo, MutAnyOrigin]:
    var pi = 0
    while path[pi] != UInt8(0):
        pi += 1
    var is_xml = (pi >= 4 and path[pi - 4] == UInt8(46) and path[pi - 3] == UInt8(120)
                  and path[pi - 2] == UInt8(109) and path[pi - 1] == UInt8(108))
    if is_xml:
        return mojo_parse_mitsuba_scene(path, verbose)
    return mojo_parse_scene(path, verbose)
