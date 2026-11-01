"""Arbitrary bytes into the pbrt parameter parser: crashes and memory errors."""
from std.memory.alloc import unsafe_alloc
from gonzales.scene.description.lexer import PbrtScanner, scanner_free, _psc_collect_params


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
    _ = _psc_collect_params(h)
    scanner_free(h)
    return Int32(0)
