"""Tiled Ptex paging: with every tile loaded, a tiled lookup reads exactly what the full, untiled face does."""
from std.ffi import external_call
from std.math import abs
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_equal, TestSuite
from gonzales.core.geometry import RGB
from gonzales.core.rng import PCG32
from gonzales.scene.appearance.materials import GpuTexture
from gonzales.scene.appearance.tile_cache import (TileCache, PTEX_SLOT_BYTES, TILE_REQ_COUNT, TILE_REQ_CAP, TILE_USED_OFF,
                                                  TILE_SLOTS, TILE_SLOTS_OFF, TILE_REQS)
from gonzales.transport.texturing import _sample_ptex

comptime _Bytes = Pointer[UInt8, MutUntrackedOrigin]
comptime SIDE = 1024


def _cstr(s: String) -> _Bytes:
    var p = unsafe_alloc[UInt8](s.byte_length() + 1)
    for i in range(s.byte_length()):
        p[unsafe_offset=i] = s.as_bytes()[i]
    p[unsafe_offset=s.byte_length()] = UInt8(0)
    return p


def _write_tile(path: String) raises:
    var px = unsafe_alloc[Float32](SIDE * SIDE * 3)
    var pcg = PCG32(3, 7)
    for y in range(SIDE):
        for x in range(SIDE):
            var i = (y * SIDE + x) * 3
            px[unsafe_offset=i] = Float32(x) / Float32(SIDE)
            px[unsafe_offset=i + 1] = Float32(y) / Float32(SIDE)
            px[unsafe_offset=i + 2] = pcg.next_float()
    var c = _cstr(path)
    var ok = external_call["write_image_rgb", Int32, _Bytes, Pointer[Float32, MutUntrackedOrigin], Int32, Int32, Int32, Int32](
        c, px, Int32(SIDE), Int32(SIDE), Int32(0), Int32(0))
    c.unsafe_free(); px.unsafe_free()
    assert_equal(ok, Int32(1))


def _pack(name: _Bytes) raises -> Tuple[_Bytes, Int]:
    var nbytes = unsafe_alloc[Int64](1); var nfaces = unsafe_alloc[Int32](1); var paged = unsafe_alloc[Int32](1)
    assert_equal(external_call["ptex_packed_size", Int32, _Bytes, Pointer[Int64, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin],
                               Pointer[Int32, MutUntrackedOrigin]](name, nbytes, nfaces, paged), Int32(1))
    var blob = unsafe_alloc[UInt8](Int(nbytes[unsafe_offset=0]))
    _ = external_call["ptex_packed_take", Int32, _Bytes, _Bytes](name, blob)
    var n = Int(nfaces[unsafe_offset=0])
    nbytes.unsafe_free(); nfaces.unsafe_free(); paged.unsafe_free()
    return (blob, n)


def test_all_tiles_loaded_match_the_full_face() raises:
    _write_tile("/tmp/gonzales_tile_test.1001.exr")
    var name = _cstr("udim:raw:1:1:/tmp/gonzales_tile_test.<UDIM>.exr")
    # Untiled, every face at full resolution.
    _ = external_call["ptex_set_tiles", NoneType, Int32](Int32(0))
    _ = external_call["ptex_set_paging", NoneType, Int32](Int32(0))
    _ = external_call["ptex_set_plan", NoneType, Int32, Int32, Int64](Int32(12), Int32(12), Int64(0))
    var (full, n_faces) = _pack(name)
    # Tiled with a 4-texel resident tail: 128^2 faces page five levels, the finest as 2 x 2 tiles.
    _ = external_call["ptex_set_plan", NoneType, Int32, Int32, Int64](Int32(12), Int32(2), Int64(1 << 30))
    _ = external_call["ptex_set_paging", NoneType, Int32](Int32(1))
    _ = external_call["ptex_set_tiles", NoneType, Int32](Int32(1))
    var (tiled, n2) = _pack(name)
    _ = external_call["ptex_set_tiles", NoneType, Int32](Int32(0))
    _ = external_call["ptex_set_paging", NoneType, Int32](Int32(0))
    assert_equal(n_faces, 64)
    assert_equal(n2, 64)
    assert_true((Int(tiled[unsafe_offset=7]) & 16) != 0)
    # Load every tile into a host pool and point its page entry at it.
    var tc = TileCache(1 << 30)
    _ = tc.add_file(name, tiled, n_faces)
    var info = tiled.unsafe_offset(n_faces * 8)
    var first = Int(info.unsafe_bitcast[UInt32]()[unsafe_offset=0])
    var last = first
    for f in range(n_faces):   # the page tables end where the last face's ends
        var e = info.unsafe_offset(f * 8)
        var pages = 0
        for l in range(Int(e[unsafe_offset=6])):
            var ul = max(Int(e[unsafe_offset=4]) - l, 0); var vl = max(Int(e[unsafe_offset=5]) - l, 0)
            pages += (1 << max(ul - 6, 0)) * (1 << max(vl - 6, 0))
        last = max(last, Int(e.unsafe_bitcast[UInt32]()[unsafe_offset=0]) + pages * 4)
    var n_pages = (last - first) // 4
    assert_true(n_pages > 0)
    var slots_off = (TILE_REQS + n_pages + 63) // 64 * 64   # a used byte per slot before the slots
    var pool = unsafe_alloc[UInt8](slots_off + n_pages * PTEX_SLOT_BYTES + 64)
    for i in range(slots_off): pool[unsafe_offset=i] = UInt8(0)
    pool.unsafe_offset(TILE_REQ_CAP).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(0)
    pool.unsafe_offset(TILE_USED_OFF).unsafe_bitcast[UInt64]()[unsafe_offset=0] = UInt64(TILE_REQS)
    pool.unsafe_offset(TILE_SLOTS).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(n_pages)
    pool.unsafe_offset(TILE_SLOTS_OFF).unsafe_bitcast[UInt64]()[unsafe_offset=0] = UInt64(slots_off)
    var tex_h = external_call["ptex_open", Int, _Bytes](name)
    assert_true(tex_h != 0)
    var level = unsafe_alloc[UInt8](((1 << 7) + 2) * ((1 << 7) + 2) * 3)
    for k in range(n_pages):
        var entry = first + k * 4
        var (face, ul, vl, tx, ty) = tc._decode(0, entry)
        _ = external_call["ptex_read_level", NoneType, Int, Int32, Int32, Int32, _Bytes](tex_h, Int32(face), Int32(ul), Int32(vl), level)
        _ = external_call["ptex_crop_tile", NoneType, _Bytes, Int32, Int32, Int32, Int32, _Bytes](
            level, Int32(ul), Int32(vl), Int32(tx), Int32(ty), pool.unsafe_offset(slots_off + k * PTEX_SLOT_BYTES))
        tiled.unsafe_offset(entry).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(16 + k)
    _ = external_call["ptex_close", NoneType, Int](tex_h)
    var lut = unsafe_alloc[Float32](256)
    for i in range(256): lut[unsafe_offset=i] = Float32(i) / Float32(255)
    var a = GpuTexture(full, lut, pool, Int32(n_faces), Int32(-1), Int32(-1), Int32(3), Int32(GpuTexture.FORMAT_PTEX), GpuTexture.UV_IDENTITY)
    var b = GpuTexture(tiled, lut, pool, Int32(n_faces), Int32(0), Int32(-1), Int32(3), Int32(GpuTexture.FORMAT_PTEX), GpuTexture.UV_IDENTITY)
    var pcg = PCG32(11, 5)
    var widths = List[Float32]()
    widths.append(0.00005); widths.append(0.0004); widths.append(0.002); widths.append(0.01); widths.append(0.05)
    for i in range(4000):
        var u = pcg.next_float(); var v = pcg.next_float()
        var fp = widths[i % len(widths)]
        var ca = _sample_ptex(a, 0, u, v, fp)
        var cb = _sample_ptex(b, 0, u, v, fp)
        var d = abs(ca.r - cb.r) + abs(ca.g - cb.g) + abs(ca.b - cb.b)
        assert_true(d < 1e-5, "u " + String(u) + " v " + String(v) + " fp " + String(fp) + ": " + String(d))
    # Nothing was asked for: every needed tile was there.
    assert_equal(pool.unsafe_offset(TILE_REQ_COUNT).unsafe_bitcast[UInt32]()[unsafe_offset=0], UInt32(0))
    full.unsafe_free(); tiled.unsafe_free(); pool.unsafe_free(); level.unsafe_free(); lut.unsafe_free(); name.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
