"""Tiled Ptex paging on the GPU: tiles copied into their slots and page entries set by the cache's kernels, then
sampled on the device, read what the full, untiled face does on the host."""
from std.ffi import external_call
from std.math import abs, ceildiv
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, assert_equal, TestSuite
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from gonzales.core.rng import PCG32
from gonzales.scene.appearance.materials import GpuTexture
from gonzales.scene.appearance.tile_cache import (TileCache, PTEX_SLOT_BYTES, TILE_REQ_COUNT, TILE_REQ_CAP, TILE_USED_OFF,
                                                  TILE_SLOTS, TILE_SLOTS_OFF, TILE_REQS)
from gonzales.transport.texturing import _sample_ptex
from gonzales.gpu.scene import tile_copy_gpu, tile_patch_gpu
from gonzales.gpu.ptex_tiles import ptex_sample_batch_gpu

comptime _Bytes = Pointer[UInt8, MutUntrackedOrigin]
comptime SIDE = 1024
comptime N = 4096


def _cstr(s: String) -> _Bytes:
    var p = unsafe_alloc[UInt8](s.byte_length() + 1)
    for i in range(s.byte_length()):
        p[unsafe_offset=i] = s.as_bytes()[i]
    p[unsafe_offset=s.byte_length()] = UInt8(0)
    return p


def _pack(name: _Bytes) raises -> Tuple[_Bytes, Int, Int]:
    var nbytes = unsafe_alloc[Int64](1); var nfaces = unsafe_alloc[Int32](1); var paged = unsafe_alloc[Int32](1)
    assert_equal(external_call["ptex_packed_size", Int32, _Bytes, Pointer[Int64, MutUntrackedOrigin], Pointer[Int32, MutUntrackedOrigin],
                               Pointer[Int32, MutUntrackedOrigin]](name, nbytes, nfaces, paged), Int32(1))
    var nb = Int(nbytes[unsafe_offset=0])
    var blob = unsafe_alloc[UInt8](nb)
    _ = external_call["ptex_packed_take", Int32, _Bytes, _Bytes](name, blob)
    var n = Int(nfaces[unsafe_offset=0])
    nbytes.unsafe_free(); nfaces.unsafe_free(); paged.unsafe_free()
    return (blob, n, nb)


def test_gpu_tiles_match_the_full_face() raises:
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return
    var px = unsafe_alloc[Float32](SIDE * SIDE * 3)
    var pcg = PCG32(3, 7)
    for k in range(SIDE * SIDE):
        px[unsafe_offset=3 * k] = Float32(k % SIDE) / Float32(SIDE)
        px[unsafe_offset=3 * k + 1] = Float32(k // SIDE) / Float32(SIDE)
        px[unsafe_offset=3 * k + 2] = pcg.next_float()
    var path = _cstr("/tmp/gonzales_tile_gpu_test.1001.exr")
    assert_equal(external_call["write_image_rgb", Int32, _Bytes, Pointer[Float32, MutUntrackedOrigin], Int32, Int32, Int32, Int32](
        path, px, Int32(SIDE), Int32(SIDE), Int32(0), Int32(0)), Int32(1))
    path.unsafe_free(); px.unsafe_free()
    var name = _cstr("udim:raw:1:1:/tmp/gonzales_tile_gpu_test.<UDIM>.exr")
    _ = external_call["ptex_set_plan", NoneType, Int32, Int32, Int64](Int32(12), Int32(12), Int64(0))
    var (full, n_faces, _nf) = _pack(name)
    _ = external_call["ptex_set_plan", NoneType, Int32, Int32, Int64](Int32(12), Int32(2), Int64(1 << 30))
    _ = external_call["ptex_set_paging", NoneType, Int32](Int32(1))
    _ = external_call["ptex_set_tiles", NoneType, Int32](Int32(1))
    var (tiled, _n2, tiled_bytes) = _pack(name)
    _ = external_call["ptex_set_tiles", NoneType, Int32](Int32(0))
    _ = external_call["ptex_set_paging", NoneType, Int32](Int32(0))
    # Every page, its tile staged on the host in page order.
    var tc = TileCache(1 << 30)
    _ = tc.add_file(name, tiled, n_faces)
    var info = tiled.unsafe_offset(n_faces * 8)
    var first = Int(info.unsafe_bitcast[UInt32]()[unsafe_offset=0])
    var n_pages = (tiled_bytes - first) // 4
    var staging = unsafe_alloc[UInt8](n_pages * PTEX_SLOT_BYTES)
    var level = unsafe_alloc[UInt8](((1 << 7) + 2) * ((1 << 7) + 2) * 3)
    var tex_h = external_call["ptex_open", Int, _Bytes](name)
    var patches = List[UInt32]()
    var slots = List[UInt32]()
    for k in range(n_pages):
        var entry = first + k * 4
        var (face, ul, vl, tx, ty) = tc._decode(0, entry)
        _ = external_call["ptex_read_level", NoneType, Int, Int32, Int32, Int32, _Bytes](tex_h, Int32(face), Int32(ul), Int32(vl), level)
        _ = external_call["ptex_crop_tile", NoneType, _Bytes, Int32, Int32, Int32, Int32, _Bytes](
            level, Int32(ul), Int32(vl), Int32(tx), Int32(ty), staging.unsafe_offset(k * PTEX_SLOT_BYTES))
        patches.append(UInt32(0)); patches.append(UInt32(entry)); patches.append(UInt32(16 + k))
        slots.append(UInt32(k))
    _ = external_call["ptex_close", NoneType, Int](tex_h)
    # The pool as the cache lays it out: header, used bytes, slots.
    var slots_off = (TILE_REQS + n_pages + 63) // 64 * 64
    var pool_bytes = slots_off + n_pages * PTEX_SLOT_BYTES
    var hdr = unsafe_alloc[UInt8](slots_off)
    for i in range(slots_off): hdr[unsafe_offset=i] = UInt8(0)
    hdr.unsafe_offset(TILE_REQ_CAP).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(64)
    hdr.unsafe_offset(TILE_USED_OFF).unsafe_bitcast[UInt64]()[unsafe_offset=0] = UInt64(TILE_REQS)
    hdr.unsafe_offset(TILE_SLOTS).unsafe_bitcast[UInt32]()[unsafe_offset=0] = UInt32(n_pages)
    hdr.unsafe_offset(TILE_SLOTS_OFF).unsafe_bitcast[UInt64]()[unsafe_offset=0] = UInt64(slots_off)
    var lut = unsafe_alloc[Float32](256)
    for i in range(256): lut[unsafe_offset=i] = Float32(i) / Float32(255)
    var uvf = unsafe_alloc[Float32](3 * N)
    var widths = List[Float32]()
    widths.append(0.00005); widths.append(0.0004); widths.append(0.002); widths.append(0.01); widths.append(0.05)
    var rng = PCG32(11, 5)
    for i in range(N):
        uvf[unsafe_offset=3 * i] = rng.next_float(); uvf[unsafe_offset=3 * i + 1] = rng.next_float()
        uvf[unsafe_offset=3 * i + 2] = widths[i % len(widths)]

    var ctx = DeviceContext()
    var d_blob = ctx.enqueue_create_buffer[DType.uint8](tiled_bytes)
    ctx.enqueue_copy(d_blob, tiled)
    var d_pool = ctx.enqueue_create_buffer[DType.uint8](pool_bytes)
    ctx.enqueue_copy(d_pool.create_sub_buffer[DType.uint8](0, slots_off), hdr)
    var d_stage = ctx.enqueue_create_buffer[DType.uint8](n_pages * PTEX_SLOT_BYTES)
    ctx.enqueue_copy(d_stage, staging)
    var d_slots = ctx.enqueue_create_buffer[DType.uint32](n_pages)
    ctx.enqueue_copy(d_slots, slots.unsafe_ptr())
    var d_patch = ctx.enqueue_create_buffer[DType.uint32](3 * n_pages)
    ctx.enqueue_copy(d_patch, patches.unsafe_ptr())
    var tables = List[UInt64](); tables.append(UInt64(Int(d_blob.unsafe_ptr())))
    var d_tables = ctx.enqueue_create_buffer[DType.uint64](1)
    ctx.enqueue_copy(d_tables, tables.unsafe_ptr())
    var d_lut = ctx.enqueue_create_buffer[DType.float32](256)
    ctx.enqueue_copy(d_lut, lut)
    var d_uvf = ctx.enqueue_create_buffer[DType.float32](3 * N)
    ctx.enqueue_copy(d_uvf, uvf)
    var d_res = ctx.enqueue_create_buffer[DType.float32](3 * N)
    comptime W = PTEX_SLOT_BYTES // 4
    ctx.enqueue_function[tile_copy_gpu](d_pool.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                                        d_stage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                                        d_slots.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), Int64(slots_off), Int64(n_pages),
                                        grid_dim=ceildiv(n_pages * W, 256), block_dim=256)
    ctx.enqueue_function[tile_patch_gpu](d_tables.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                                         d_patch.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), Int64(n_pages),
                                         grid_dim=ceildiv(n_pages, 256), block_dim=256)
    ctx.enqueue_function[ptex_sample_batch_gpu](d_blob.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                                           d_lut.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                                           d_pool.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), Int64(n_faces), Int32(1),
                                           d_uvf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                                           d_res.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), Int64(N),
                                           grid_dim=ceildiv(N, 64), block_dim=64)
    ctx.synchronize()
    var cpu = GpuTexture(full, lut, hdr, Int32(n_faces), Int32(-1), Int32(-1), Int32(3), Int32(GpuTexture.FORMAT_PTEX), GpuTexture.UV_IDENTITY)
    var bad = 0
    with d_res.map_to_host() as g:
        for i in range(N):
            var c = _sample_ptex(cpu, 0, uvf[unsafe_offset=3 * i], uvf[unsafe_offset=3 * i + 1], uvf[unsafe_offset=3 * i + 2])
            var d = abs(g[3 * i] - c.r) + abs(g[3 * i + 1] - c.g) + abs(g[3 * i + 2] - c.b)
            if d > Float32(1e-5):
                if bad < 6:
                    print("lookup", i, "gpu", g[3 * i], g[3 * i + 1], g[3 * i + 2], "cpu", c.r, c.g, c.b)
                bad += 1
    var asked = 0
    with d_pool.create_sub_buffer[DType.uint8](0, 16).map_to_host() as h:
        asked = Int(h.unsafe_ptr().unsafe_offset(TILE_REQ_COUNT).unsafe_bitcast[UInt32]()[unsafe_offset=0])
    print("tiled lookups on the GPU:", N, "mismatches", bad, "requests", asked)
    assert_equal(bad, 0)
    assert_equal(asked, 0)
    full.unsafe_free(); tiled.unsafe_free(); staging.unsafe_free(); level.unsafe_free(); hdr.unsafe_free(); lut.unsafe_free()
    uvf.unsafe_free(); name.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
