"""RMIP displacement on the GPU: the same rays as the CPU, through device copies laid out as the renderer's."""
from std.math import abs, sin, cos
from std.memory.alloc import unsafe_alloc
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from gonzales.gpu.displacement import intersect_displaced_batch_gpu
from max.gpu.host import DeviceContext, DeviceBuffer
from gonzales.core.geometry import Vec3f
from gonzales.core.rng import PCG32
from gonzales.gpu.scene import _gpu_upload_owned
from gonzales.scene.shapes.displacement import DisplacedMesh, DisplacedTri, Rmip, rmip_build, displaced_tri, intersect_displaced

comptime F2 = SIMD[DType.float32, 2]
comptime N = 512


def test_gpu_matches_cpu() raises:
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return
    var rng = PCG32(11, 1)
    var w = 64; var hh = 48
    var hm = unsafe_alloc[Float32](w * hh)
    for j in range(hh):
        for i in range(w):
            hm[unsafe_offset=j * w + i] = Float32(0.3) * sin(Float32(i) * 0.4) * cos(Float32(j) * 0.3) + Float32(0.2) * rng.next_float()
    var rm = rmip_build(hm, w, hh)
    var p0 = Vec3f(0, 0, 0); var p1 = Vec3f(4, 0.3, 0); var p2 = Vec3f(0.5, 3.5, 0.2)
    var n0 = Vec3f(-0.4, 0, 1).normalize(); var n1 = Vec3f(0.4, 0.1, 1).normalize(); var n2 = Vec3f(0, 0.4, 1).normalize()
    var tri = unsafe_alloc[DisplacedTri](1)
    tri[unsafe_offset=0] = displaced_tri(p0, p1, p2, n0, n1, n2, F2(3, 4), F2(33, 6), F2(8, 30), rm)
    var rays = unsafe_alloc[Float32](6 * N)
    for k in range(N):
        var a = rng.next_float(); var b = rng.next_float() * (Float32(1) - a)
        var target = p0 * (Float32(1) - a - b) + p1 * a + p2 * b
        var dir = Vec3f(rng.next_float() * 2 - 1, rng.next_float() * 2 - 1, rng.next_float() * 1.5 - 0.3).normalize()
        var o = target + dir * Float32(6)
        rays[unsafe_offset=6 * k] = o.x; rays[unsafe_offset=6 * k + 1] = o.y; rays[unsafe_offset=6 * k + 2] = o.z
        rays[unsafe_offset=6 * k + 3] = -dir.x; rays[unsafe_offset=6 * k + 4] = -dir.y; rays[unsafe_offset=6 * k + 5] = -dir.z
    var ctx = DeviceContext()
    var bufs = List[DeviceBuffer[DType.uint8]]()
    var r = rm
    r.h = _gpu_upload_owned[Float32](ctx, bufs, rm.h, w * hh)
    r.tab = _gpu_upload_owned[Float32](ctx, bufs, rm.tab, 2 * Int(rm.rx) * Int(rm.ry) * Int(rm.lx) * Int(rm.ly))
    var dmh = unsafe_alloc[DisplacedMesh](1)
    dmh[unsafe_offset=0] = DisplacedMesh(_gpu_upload_owned[DisplacedTri](ctx, bufs, tri, 1), r)
    var dmd = _gpu_upload_owned[DisplacedMesh](ctx, bufs, dmh, 1)
    var rd = _gpu_upload_owned[Float32](ctx, bufs, rays, 6 * N)
    var out = ctx.enqueue_create_buffer[DType.float32](2 * N)
    ctx.synchronize()
    ctx.enqueue_function[intersect_displaced_batch_gpu](dmd, rd, out.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), Int64(N),
                                      grid_dim=(N + 63) // 64, block_dim=64)
    ctx.synchronize()
    var hits = 0; var bad = 0
    with out.map_to_host() as host:
        var g = host.unsafe_ptr()
        for k in range(N):
            var o = Vec3f(rays[unsafe_offset=6 * k], rays[unsafe_offset=6 * k + 1], rays[unsafe_offset=6 * k + 2])
            var d = Vec3f(rays[unsafe_offset=6 * k + 3], rays[unsafe_offset=6 * k + 4], rays[unsafe_offset=6 * k + 5])
            var c = intersect_displaced(tri[unsafe_offset=0], rm, o, d, Float32(1e30))
            if c.hit: hits += 1
            if (g[unsafe_offset=2 * k] != Float32(0)) != c.hit or (c.hit and abs(g[unsafe_offset=2 * k + 1] - c.t) > Float32(1e-3) * c.t):
                bad += 1
    print("gpu displacement hits", hits, "mismatches", bad)
    assert_true(hits > 100 and bad <= 2)
    hm.unsafe_free(); rm.tab.unsafe_free(); tri.unsafe_free(); rays.unsafe_free(); dmh.unsafe_free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
