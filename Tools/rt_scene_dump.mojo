"""Build a test scene of BLASes and instances with the own builders and write the files the hardware harness reads.

    build/rt_scene_dump spec.bin outdir

spec.bin: int32 n_blas, int32 n_inst; per BLAS: int32 n_tris then nine float32 per triangle; per instance: int32 blas,
twelve float32 (world-to-object, the top three rows of a column-major matrix). Writes outdir/tlas.bin, blas<k>.bin,
blas<k>.gid (int32 per primitive id) and refs.txt (hex record pointer offset and structure index)."""
from gonzales.rt_build import rt_build_blas, rt_build_tlas, RtBlas, RtTlas
from std.memory.alloc import unsafe_alloc
from std.sys import argv


def _i32(data: List[UInt8], off: Int) -> Int:
    return Int(data[off]) | (Int(data[off + 1]) << 8) | (Int(data[off + 2]) << 16) | (Int(data[off + 3]) << 24)


def _write(path: String, ptr: Pointer[UInt8, MutUntrackedOrigin], n: Int) raises:
    var l = List[UInt8](length=n, fill=UInt8(0))
    for i in range(n): l[i] = ptr[unsafe_offset=i]
    with open(path, "w") as f:
        f.write_bytes(l)


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("Usage: rt_scene_dump spec.bin outdir")
        return
    var data = open(String(args[1]), "r").read_bytes()
    var dir = String(args[2])
    var nb = _i32(data, 0)
    var ni = _i32(data, 4)
    var off = 8
    var boxes = unsafe_alloc[Float64](6 * max(nb, 1))
    var id_base = 0
    for b in range(nb):
        var nt = _i32(data, off); off += 4
        var tv = unsafe_alloc[Float32](max(nt * 9, 1))
        var src = (data.unsafe_ptr() + off).bitcast[Float32]()
        for i in range(nt * 9): tv[unsafe_offset=i] = src[unsafe_offset=i]
        off += nt * 36
        var r = rt_build_blas(tv, nt, True, True, False, id_base)
        for q in range(6): boxes[unsafe_offset=b * 6 + q] = r.box[unsafe_offset=q]
        _write(dir + "/blas" + String(b) + ".bin", r.bytes, r.size)
        var g = List[UInt8](length=r.n_gid * 4, fill=UInt8(0))
        for i in range(r.n_gid):
            var v = UInt32(r.gid_to_tri[unsafe_offset=i])
            for q in range(4): g[i * 4 + q] = UInt8((v >> UInt32(8 * q)) & UInt32(0xff))
        with open(dir + "/blas" + String(b) + ".gid", "w") as f:
            f.write_bytes(g)
        id_base += r.n_gid
    var w2o = unsafe_alloc[Float32](max(ni * 12, 1))
    var binst = unsafe_alloc[Int32](max(ni, 1))
    for k in range(ni):
        binst[unsafe_offset=k] = Int32(_i32(data, off)); off += 4
        var src = (data.unsafe_ptr() + off).bitcast[Float32]()
        for i in range(12): w2o[unsafe_offset=k * 12 + i] = src[unsafe_offset=i]
        off += 48
    var t = rt_build_tlas(w2o, binst, ni, boxes, nb)
    print("blas", nb, "instances", ni, "tlas bytes", t.size, "records", t.n_records)
    _write(dir + "/tlas.bin", t.bytes, t.size)
    var refs = String("")
    for r in range(t.n_records):
        var ro = t.rec_off + r * 64 + 8
        var idx = Int(t.bytes[unsafe_offset=ro - 8 + 8])        # first byte of the pointer field = BLAS index (small tests)
        refs += hex(ro) + " " + String(idx + 1) + "\n"
    with open(dir + "/refs.txt", "w") as f:
        f.write(refs)
