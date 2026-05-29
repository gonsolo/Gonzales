# Sobol matrix generator — Mojo port of Sources/SobolGenerator/main.swift
# Reads Joe-Kuo direction numbers, writes 1024×52 UInt32 matrix as little-endian binary.
# Usage: uv run mojo run Sources/SobolGenerator/gen_sobol.mojo <dir-numbers> <output.bin>
from std.ffi import external_call
from std.memory import alloc
from std.sys import argv

comptime N_DIMS = 1024
comptime MATRIX_SIZE = 52

fn open_file(path: UnsafePointer[UInt8, MutAnyOrigin],
             mode: UnsafePointer[UInt8, MutAnyOrigin]) -> UnsafePointer[UInt8, MutAnyOrigin]:
    return external_call["fopen", UnsafePointer[UInt8, MutAnyOrigin],
        UnsafePointer[UInt8, MutAnyOrigin], UnsafePointer[UInt8, MutAnyOrigin]](path, mode)

fn make_cstr(s: String) -> UnsafePointer[UInt8, MutAnyOrigin]:
    var n = len(s)
    var buf = alloc[UInt8](n + 1)
    var src = s.unsafe_ptr()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = UInt8(0)
    return buf

fn is_digit(c: UInt8) -> Bool:
    return c >= UInt8(48) and c <= UInt8(57)

fn is_ws(c: UInt8) -> Bool:
    return c == UInt8(32) or c == UInt8(9) or c == UInt8(13)

fn parse_uint(buf: UnsafePointer[UInt8, MutAnyOrigin], mut pos: Int, end: Int) -> Int:
    while pos < end and is_ws(buf[pos]):
        pos += 1
    var val = 0
    while pos < end and is_digit(buf[pos]):
        val = val * 10 + Int(buf[pos]) - 48
        pos += 1
    return val

fn skip_line(buf: UnsafePointer[UInt8, MutAnyOrigin], mut pos: Int, end: Int):
    while pos < end and buf[pos] != UInt8(10):
        pos += 1
    if pos < end:
        pos += 1

fn run(in_path: String, out_path: String) -> Int:
    # ── read direction numbers file ──
    var in_cstr = make_cstr(in_path)
    var rb_mode = make_cstr("rb")
    var fp_in = open_file(in_cstr, rb_mode)
    in_cstr.free()
    rb_mode.free()
    if not fp_in:
        print("Error: cannot open", in_path)
        return 1

    _ = external_call["fseek", Int32,
        UnsafePointer[UInt8, MutAnyOrigin], Int64, Int32](fp_in, Int64(0), Int32(2))
    var fsize = Int(external_call["ftell", Int64,
        UnsafePointer[UInt8, MutAnyOrigin]](fp_in))
    _ = external_call["fseek", Int32,
        UnsafePointer[UInt8, MutAnyOrigin], Int64, Int32](fp_in, Int64(0), Int32(0))
    var text = alloc[UInt8](fsize + 1)
    _ = external_call["fread", Int,
        UnsafePointer[UInt8, MutAnyOrigin], Int, Int, UnsafePointer[UInt8, MutAnyOrigin]](
        text, 1, fsize, fp_in)
    _ = external_call["fclose", Int32, UnsafePointer[UInt8, MutAnyOrigin]](fp_in)
    text[fsize] = UInt8(0)

    # ── allocate and zero the matrix table ──
    var n_total = N_DIMS * MATRIX_SIZE
    var matrices = alloc[UInt32](n_total)
    for i in range(n_total):
        matrices[i] = UInt32(0)

    # dim 0: Van der Corput (identity)
    for i in range(32):
        matrices[i] = UInt32(1) << UInt32(31 - i)

    # ── parse direction numbers and generate dims 1..N_DIMS-1 ──
    var pos = 0
    var dim = 0

    # skip header line (starts with 'd')
    while pos < fsize and text[pos] != UInt8(10):
        pos += 1
    if pos < fsize:
        pos += 1

    while pos < fsize and dim < N_DIMS - 1:
        # skip blank / whitespace-only lines
        while pos < fsize and (text[pos] == UInt8(10) or is_ws(text[pos])):
            pos += 1
        if pos >= fsize:
            break

        # parse: d  s  a  m1 m2 ... ms  (tab-separated, skip 'd' column)
        # skip 'd' token
        while pos < fsize and not is_ws(text[pos]) and text[pos] != UInt8(10):
            pos += 1
        var s = parse_uint(text, pos, fsize)
        var a = parse_uint(text, pos, fsize)

        var m = alloc[Int32](s + 1)
        for idx in range(s):
            m[idx] = Int32(parse_uint(text, pos, fsize))
        skip_line(text, pos, fsize)

        # generate direction numbers via Joe-Kuo recurrence
        var v = alloc[UInt32](MATRIX_SIZE + 1)
        for idx in range(MATRIX_SIZE + 1):
            v[idx] = UInt32(0)
        for idx in range(1, s + 1):
            v[idx] = UInt32(m[idx - 1]) << UInt32(32 - idx)
        for idx in range(s + 1, MATRIX_SIZE + 1):
            v[idx] = v[idx - s] ^ (v[idx - s] >> UInt32(s))
            for jdx in range(1, s):
                if (a >> (s - 1 - jdx)) & 1 == 1:
                    v[idx] = v[idx] ^ v[idx - jdx]
        m.free()

        # store as matrix columns for this dimension
        var base = (dim + 1) * MATRIX_SIZE
        for idx in range(MATRIX_SIZE):
            matrices[base + idx] = v[idx + 1]
        v.free()
        dim += 1

    text.free()

    # ── write binary output (little-endian UInt32 values) ──
    var out_cstr = make_cstr(out_path)
    var wb_mode = make_cstr("wb")
    var fp_out = open_file(out_cstr, wb_mode)
    out_cstr.free()
    wb_mode.free()
    if not fp_out:
        print("Error: cannot open", out_path)
        matrices.free()
        return 1

    var written = external_call["fwrite", Int,
        UnsafePointer[UInt32, MutAnyOrigin], Int, Int, UnsafePointer[UInt8, MutAnyOrigin]](
        matrices, 4, n_total, fp_out)
    _ = external_call["fclose", Int32, UnsafePointer[UInt8, MutAnyOrigin]](fp_out)
    matrices.free()

    if written != n_total:
        print("Error: wrote", written, "of", n_total, "entries")
        return 1

    print("Generated", out_path, "—", N_DIMS, "dims ×", MATRIX_SIZE, "entries")
    return 0

fn main():
    var args = argv()
    if len(args) < 3:
        print("Usage: gen_sobol <direction-numbers-file> <output.bin>")
        _ = external_call["exit", Int32, Int32](Int32(1))
        return
    var rc = run(args[1], args[2])
    if rc != 0:
        _ = external_call["exit", Int32, Int32](Int32(rc))
