"""Arbitrary bytes into the whole pbrt scene parser, up to where finalize_scene would build the BVH."""
from std.memory.alloc import unsafe_alloc
from gonzales.scene.description.lexer import PbrtScanner, scanner_free
from gonzales.scene.description.parse_types import SceneParseState
from gonzales.scene.description.pbrt_parser import parse_scene_file
from gonzales.scene.shapes.ply_prefetch import PlyPrefetch, scan_plys


@export
def LLVMFuzzerTestOneInput(data: Pointer[UInt8, MutUntrackedOrigin], size: Int) abi("C") -> Int32:
    var buf = unsafe_alloc[UInt8](size + 1)
    for i in range(size):
        buf[unsafe_offset=i] = data[unsafe_offset=i]
    buf[unsafe_offset=size] = UInt8(0)
    var h = unsafe_alloc[PbrtScanner](1)
    h[unsafe_offset=0].buffer = buf
    h[unsafe_offset=0].total_bytes = Int32(size)
    h[unsafe_offset=0].cursor = Int32(0)
    h[unsafe_offset=0].is_at_end = Int32(0)
    var s = unsafe_alloc[SceneParseState](1)
    s.unsafe_write(SceneParseState())
    # Scanned but not loaded: a mesh the fuzzer names is simply missing.
    var plys = unsafe_alloc[PlyPrefetch](1)
    plys.unsafe_write(scan_plys(h[unsafe_offset=0].buffer, Int(h[unsafe_offset=0].total_bytes), s[unsafe_offset=0].scene_dir))
    s[unsafe_offset=0].ply_cache = plys
    parse_scene_file(h, s)
    scanner_free(h)
    _ = s.unsafe_take_pointee()
    s.unsafe_free()
    _ = plys.unsafe_take_pointee()
    plys.unsafe_free()
    return Int32(0)
