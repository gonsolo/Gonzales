from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite
from max.gpu.host import DeviceContext
from gonzales.curves import Curve
from gonzales.materials import Material
from gonzales.primitives import PrimId
from gonzales.xert import (
    xert_build_blas, xert_build_blas_geoms, xert_build_tlas, xert_alignment, xert_free, xert_trace_rays_gpu,
    xert_trace_interop_gpu, XERT_MISS,
)

# Intel Xe2 hardware ray tracing: the driver builds a BLAS for the quad [-1, 1]^2 at z = 0 and a grid of rays straight
# down from z = 2 must hit exactly the grid points inside it, at t = 2, with barycentrics that give back the hit point.
# Skips without an Intel GPU (xert_alignment() == 0).

comptime N = 64

def test_xert_quad() raises:
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return
    if xert_alignment() == 0:
        print("SKIP: no Intel GPU with ray tracing")
        return
    var corners: List[Float32] = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0]
    var tv = unsafe_alloc[Float32](18)
    for k in range(18):
        tv[unsafe_offset=k] = corners[k]
    var size = Int64(0)
    var rtas = xert_build_blas(tv, 2, Pointer(to=size).unsafe_origin_cast[MutUntrackedOrigin]())
    assert_true(Int(rtas) != 0, "xert_build_blas failed")
    var ctx = DeviceContext()
    var accel = ctx.enqueue_create_buffer[DType.uint8](Int(size))
    with accel.map_to_host() as h:
        for k in range(Int(size)):
            h[k] = rtas[unsafe_offset=k]
    xert_free(rtas)
    var rays = ctx.enqueue_create_buffer[DType.float32](8 * N * N)
    with rays.map_to_host() as h:
        for i in range(N * N):
            var r: List[Float32] = [Float32(i % N) / Float32(N - 1) * 3.0 - 1.5, Float32(i // N) / Float32(N - 1) * 3.0 - 1.5,
                                    2.0, 0.0, 0.0, 0.0, -1.0, 1.0e30]
            for k in range(8):
                h[8 * i + k] = r[k]
    var results = ctx.enqueue_create_buffer[DType.uint32](8 * N * N)
    ctx.enqueue_function[xert_trace_rays_gpu](accel.unsafe_ptr(), rays.unsafe_ptr(), results.unsafe_ptr(), Int64(N * N), UInt32(0),
                                              grid_dim=(N * N + 63) // 64, block_dim=64)
    var hits = 0; var prims = List[Int](length=2, fill=0); var bad = 0
    with results.map_to_host() as h:
        with rays.map_to_host() as rh:
            for i in range(N * N):
                var word = h[8 * i + 3]
                if word == XERT_MISS:
                    continue
                hits += 1
                var t = bitcast[DType.float32](h[8 * i])
                var u = bitcast[DType.float32](h[8 * i + 1])
                var v = bitcast[DType.float32](h[8 * i + 2])
                var p = Int(word)
                if p > 1 or abs(t - 2.0) > 1.0e-5:
                    bad += 1
                    continue
                prims[p] += 1
                # Corners of triangle p: (v0, v1, v2) = corners[9p ...]; the hit point is (1 - u - v) v0 + u v1 + v v2.
                var x = (1 - u - v) * corners[9 * p] + u * corners[9 * p + 3] + v * corners[9 * p + 6]
                var y = (1 - u - v) * corners[9 * p + 1] + u * corners[9 * p + 4] + v * corners[9 * p + 7]
                if abs(x - rh[8 * i]) > 1.0e-4 or abs(y - rh[8 * i + 1]) > 1.0e-4:
                    bad += 1
    _ = accel^
    tv.unsafe_free()
    assert_equal(hits, 42 * 42)   # x = k / 63 * 3 - 1.5 lies in [-1, 1] for k = 11 ... 52
    assert_equal(bad, 0)
    assert_true(prims[0] > hits // 3 and prims[1] > hits // 3, "both triangles hit")

# Instances: a BLAS of two meshes (the quad's two triangles) placed at -2.5 and +2.5 as template instances 0 and 1, and at 0
# the ordinary meshes 7 and 9 as instances 2 and 3, a one-mesh BLAS each. Every hit's interop words must name the instance
# (n_meshes + k) or the ordinary mesh, the mesh within the template and triangle 0.
def test_xert_instances() raises:
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return
    var align = Int(xert_alignment())
    if align == 0:
        print("SKIP: no Intel GPU with ray tracing")
        return
    var corners: List[Float32] = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0]
    var tv = unsafe_alloc[Float32](18)
    for k in range(18):
        tv[unsafe_offset=k] = corners[k]
    var geoms = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](2)
    geoms[unsafe_offset=0] = tv
    geoms[unsafe_offset=1] = tv.unsafe_offset(9)
    var idx = unsafe_alloc[UInt32](3)
    for k in range(3):
        idx[unsafe_offset=k] = UInt32(k)
    var tris = unsafe_alloc[Pointer[UInt32, MutUntrackedOrigin]](2)
    tris[unsafe_offset=0] = idx
    tris[unsafe_offset=1] = idx
    var n_verts = unsafe_alloc[Int64](2)
    var n_tris = unsafe_alloc[Int64](2)
    for g in range(2):
        n_verts[unsafe_offset=g] = 3
        n_tris[unsafe_offset=g] = 1
    var bounds = unsafe_alloc[Float32](18)
    var sizes = List[Int]()
    var ptrs = List[Pointer[UInt8, MutUntrackedOrigin]]()
    for b in range(3):    # the template, then mesh 7 (triangle 0) and mesh 9 (triangle 1)
        var size = Int64(0)
        var r = xert_build_blas_geoms(Int32(2 if b == 0 else 1), geoms.unsafe_offset(max(b - 1, 0)), n_verts, tris, n_tris, 0,
                                      Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
                                      Pointer(to=size).unsafe_origin_cast[MutUntrackedOrigin](), bounds.unsafe_offset(6 * b))
        assert_true(Int(r) != 0, "xert_build_blas_geoms failed")
        ptrs.append(r); sizes.append((Int(size) + align - 1) // align * align)
    var ctx = DeviceContext()
    # The TLAS stores the BLASes' device addresses, which must be aligned.
    var blas_buf = ctx.enqueue_create_buffer[DType.uint8](sizes[0] + sizes[1] + sizes[2] + align)
    var pad = (align - Int(UInt(Int(blas_buf.unsafe_ptr())) % UInt(align))) % align
    var addr = unsafe_alloc[UInt64](3)
    var off = pad
    with blas_buf.map_to_host() as h:
        for b in range(3):
            for k in range(sizes[b]):
                h[off + k] = ptrs[b][unsafe_offset=k]
            addr[unsafe_offset=b] = UInt64(UInt(Int(blas_buf.unsafe_ptr())) + UInt(off))
            off += sizes[b]
            xert_free(ptrs[b])
    var xf = unsafe_alloc[Float32](48)
    var blas_of = unsafe_alloc[Int32](4)
    var shift: List[Float32] = [-2.5, 2.5, 0.0, 0.0]
    for k in range(4):
        for q in range(12):
            xf[unsafe_offset=12 * k + q] = Float32(1) if q == 0 or q == 4 or q == 8 else Float32(0)
        xf[unsafe_offset=12 * k + 9] = shift[k]
        blas_of[unsafe_offset=k] = Int32(max(k - 1, 0))
    var tlas_size = Int64(0)
    var tlas = xert_build_tlas(4, xf, blas_of, addr, bounds, Pointer(to=tlas_size).unsafe_origin_cast[MutUntrackedOrigin]())
    assert_true(Int(tlas) != 0, "xert_build_tlas failed")
    var tlas_buf = ctx.enqueue_create_buffer[DType.uint8](Int(tlas_size) + align)
    var tpad = (align - Int(UInt(Int(tlas_buf.unsafe_ptr())) % UInt(align))) % align
    with tlas_buf.map_to_host() as h:
        for k in range(Int(tlas_size)):
            h[tpad + k] = tlas[unsafe_offset=k]
    xert_free(tlas)
    var ord_mesh = ctx.enqueue_create_buffer[DType.int32](2)
    with ord_mesh.map_to_host() as h:
        h[0] = 7
        h[1] = 9
    comptime NX = 97
    comptime NY = 33
    comptime N_MESHES = 10
    var rays = ctx.enqueue_create_buffer[DType.float32](8 * NX * NY)
    with rays.map_to_host() as h:
        for i in range(NX * NY):
            var r: List[Float32] = [Float32(i % NX) / Float32(NX - 1) * 8.0 - 4.0 + 0.013, Float32(i // NX) / Float32(NY - 1) * 3.0 - 1.5 + 0.007,
                                    2.0, 0.0, 0.0, 0.0, -1.0, 1.0e30]
            for k in range(8):
                h[8 * i + k] = r[k]
    var results = ctx.enqueue_create_buffer[DType.uint32](8 * NX * NY)
    var heat = ctx.enqueue_create_buffer[DType.uint32](4)
    heat.enqueue_fill(0)
    ctx.enqueue_function[xert_trace_interop_gpu](tlas_buf.unsafe_ptr().unsafe_offset(tpad), rays.unsafe_ptr(), results.unsafe_ptr(),
                                                 Int64(NX * NY), UInt32(0), Int64(N_MESHES), Int64(2), ord_mesh.unsafe_ptr(),
                                                 Pointer[Curve, MutUntrackedOrigin].unsafe_dangling(), Pointer[PrimId, MutUntrackedOrigin].unsafe_dangling(),
                                                 Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(), Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
                                                 Pointer[Material, MutUntrackedOrigin].unsafe_dangling(),
                                                 heat.unsafe_ptr(),
                                                 grid_dim=(NX * NY + 63) // 64, block_dim=64)
    var hits = 0; var expected = 0; var bad = 0
    with results.map_to_host() as h:
        with rays.map_to_host() as rh:
            for i in range(NX * NY):
                var x = rh[8 * i]
                var y = rh[8 * i + 1]
                var want_k = -1
                for k in range(3):
                    if abs(x - shift[k]) <= 1.0 and abs(y) <= 1.0:
                        want_k = k
                var hit = h[8 * i + 6] == 1
                if want_k < 0:
                    if hit:
                        bad += 1
                    continue
                expected += 1
                if not hit:
                    bad += 1
                    continue
                hits += 1
                var lx = x - shift[want_k]
                var want_geom = 0 if lx > y else 1   # triangle 0 is (-1,-1) (1,-1) (1,1): below the diagonal
                var raw = Int(h[8 * i + 4])
                var geom = Int(h[8 * i + 7])
                var want_raw = (7 if want_geom == 0 else 9) if want_k == 2 else N_MESHES + want_k
                var ok = raw == want_raw and Int(h[8 * i + 5]) == 0 and abs(bitcast[DType.float32](h[8 * i]) - 2.0) < 1.0e-5
                if want_k != 2:
                    ok = ok and geom == want_geom
                if not ok:
                    bad += 1
    _ = blas_buf^
    _ = tlas_buf^
    _ = ord_mesh^
    var counted = 0
    with heat.map_to_host() as h:
        for k in range(4):
            counted += Int(h[k])
    assert_equal(counted, hits, "hit counts per instance")
    tv.unsafe_free(); geoms.unsafe_free(); idx.unsafe_free(); tris.unsafe_free(); n_verts.unsafe_free(); n_tris.unsafe_free(); bounds.unsafe_free(); addr.unsafe_free(); xf.unsafe_free(); blas_of.unsafe_free()
    print("instances:", hits, "of", expected, "hits,", bad, "wrong")
    assert_true(expected > 1000, "enough rays inside")
    assert_equal(bad, 0)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
