"""Generates parameters in every syntax from the fuzz bytes; aborts unless each parses back to its name and value."""
from std.memory.alloc import unsafe_alloc
from std.os import abort
from std.math import abs
from gonzales.scene.description.lexer import PbrtScanner, scanner_free, _psc_collect_params


def _sep(k: Int) -> String:
    if k == 0: return " "
    if k == 1: return "\n"
    if k == 2: return "\t"
    return " # note \"bool x\" [ true ]\n"


def _float_text(v: Int, form: Int) -> String:
    """v + 0.5, in every way a number may be written."""
    var d = String(v) + ".5"
    var m = String(v * 10 + 5)
    var k = form % 7
    if k == 0: return d
    if k == 1: return "+" + d
    if k == 2: return m + "e-1"
    if k == 3: return m + "E-1"
    if k == 4: return d + "e+0"
    if k == 5: return d + "E0"
    return String(v * 10 + 5) + "0e-2"


@export
def LLVMFuzzerTestOneInput(data: Pointer[UInt8, MutUntrackedOrigin], size: Int) abi("C") -> Int32:
    var text = String("")
    var kinds = List[Int]()
    var vals = List[Int]()
    var i = 0
    var n = 0
    while i + 4 <= size and n < 8:
        var t = Int(data[unsafe_offset=i]) % 5
        var f = Int(data[unsafe_offset=i + 1]) % 3
        var sp = _sep(Int(data[unsafe_offset=i + 2]) % 4)
        var v = Int(data[unsafe_offset=i + 3])
        var name = "p" + String(n)
        var body: String
        var ty: String
        if t == 0:
            ty = "bool"; body = "true" if v & 1 == 1 else "false"
            if f == 2: body = "\"" + body + "\""
        elif t == 1:
            ty = "float"; body = _float_text(v, Int(data[unsafe_offset=i + 1]) >> 2)
        elif t == 2:
            ty = "integer"; body = String(v)
        elif t == 3:
            ty = "string"; body = "\"s" + String(v) + "\""
        else:
            ty = "rgb"; body = _float_text(v, Int(data[unsafe_offset=i + 1]) >> 2) + " 1 2"
        var value: String
        if t == 4 or f == 1:
            value = "[ " + body + " ]"
        elif f == 2 and t != 0:
            value = "[" + body + "]"
        else:
            value = body
        text += "\"" + ty + " " + name + "\" " + value + sp
        kinds.append(t); vals.append(v)
        n += 1; i += 4
    var m = text.byte_length()
    var buf = unsafe_alloc[UInt8](m + 1)
    for k in range(m):
        buf[unsafe_offset=k] = text.as_bytes()[k]
    buf[unsafe_offset=m] = UInt8(0)
    var h = unsafe_alloc[PbrtScanner](1)
    h[unsafe_offset=0].buffer = buf
    h[unsafe_offset=0].total_bytes = Int32(m)
    h[unsafe_offset=0].cursor = Int32(0)
    h[unsafe_offset=0].is_at_end = Int32(0)
    var d = _psc_collect_params(h)
    if len(d.params) != n:
        abort("param count: " + text)
    for k in range(n):
        ref p = d.params[k]
        if p.name != "p" + String(k):
            abort("name: " + text)
        var t = kinds[k]
        var v = vals[k]
        if t == 0:
            if len(p.value.strs) != 1 or p.value.strs[0] != ("true" if v & 1 == 1 else "false"):
                abort("bool: " + text)
        elif t == 1:
            if len(p.value.floats) != 1 or abs(p.value.floats[0] - (Float32(v) + Float32(0.5))) > Float32(1e-3):
                abort("float: " + text)
        elif t == 2:
            if len(p.value.ints) != 1 or Int(p.value.ints[0]) != v:
                abort("integer: " + text)
        elif t == 3:
            if len(p.value.strs) != 1 or p.value.strs[0] != "s" + String(v):
                abort("string: " + text)
        else:
            if len(p.value.floats) != 3 or abs(p.value.floats[0] - (Float32(v) + Float32(0.5))) > Float32(1e-3):
                abort("rgb: " + text)
    scanner_free(h)
    return Int32(0)
