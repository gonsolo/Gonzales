from std.memory.alloc import unsafe_alloc
from std.sys import has_accelerator
from std.os.path import exists
from std.testing import assert_true, TestSuite
from std.math import abs, sin, cos
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host._nvidia_cuda import CUDA
from gonzales.primitives import TriangleMesh
from gonzales.vulkanrt import (
    vulkanrt_build_scene, vulkanrt_trace_rays, vulkanrt_destroy_scene, vulkanrt_debug_read_as,
)
from gonzales.rtcore import (
    rtcore_create, rtcore_trace, rtcore_destroy,
    RTCORE_MISS, RTCORE_PRIM_MASK,
)

# Hardware ray tracing from a CUDA kernel: a Vulkan-built BLAS is traced by the patched driver ray-query code
# (build/rt_trace.cubin, made by docs/rtcore/exec/build_rt_cubin.py) on Mojo's own DeviceContext stream, and every
# ray's hit/miss, t, u, v and triangle index must equal what Vulkan's own ray-query shader reports.
# Skips without a GPU, without the cubin, or when the bridge is the CUDA-less stub.

comptime GRID = 12
comptime NRAYS = 4096

def test_rtcore_matches_vulkan_ray_query() raises:
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return
    var cubin = String("build/rt_trace.cubin")
    if not exists(cubin):
        print("SKIP: build/rt_trace.cubin missing (run docs/rtcore/exec/build_rt_cubin.py)")
        return

    # A wavy grid of GRID*GRID quads (two triangles each) on z ~ 0.
    var nv = (GRID + 1) * (GRID + 1)
    var nt = GRID * GRID * 2
    var pts = unsafe_alloc[Float32](4 * nv)
    for y in range(GRID + 1):
        for x in range(GRID + 1):
            var o = 4 * (y * (GRID + 1) + x)
            pts[unsafe_offset=o + 0] = Float32(x)
            pts[unsafe_offset=o + 1] = Float32(y)
            pts[unsafe_offset=o + 2] = Float32(0.3) * sin(Float32(x) * Float32(0.9)) * cos(Float32(y) * Float32(0.7))
            pts[unsafe_offset=o + 3] = Float32(1)
    var idx = unsafe_alloc[Int32](3 * nt)
    var k = 0
    for y in range(GRID):
        for x in range(GRID):
            var a = Int32(y * (GRID + 1) + x)
            var b = a + 1
            var c = a + Int32(GRID + 1)
            var d = c + 1
            idx[unsafe_offset=k] = a; idx[unsafe_offset=k + 1] = b; idx[unsafe_offset=k + 2] = c
            idx[unsafe_offset=k + 3] = b; idx[unsafe_offset=k + 4] = d; idx[unsafe_offset=k + 5] = c
            k += 6
    var meshes = unsafe_alloc[TriangleMesh](1)
    meshes[unsafe_offset=0] = TriangleMesh(
        pts, Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(), idx,
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    )
    var point_counts = unsafe_alloc[Int64](1)
    point_counts[unsafe_offset=0] = Int64(nv)
    var idx_counts = unsafe_alloc[Int64](1)
    idx_counts[unsafe_offset=0] = Int64(3 * nt)

    var scene = vulkanrt_build_scene(meshes, Int64(1), point_counts, idx_counts)
    assert_true(Int(scene) != 0)

    # Reference: Vulkan ray query on the same rays.
    var rays = unsafe_alloc[Float32](8 * NRAYS)
    var state = UInt32(12345)
    for i in range(NRAYS):
        state = state * UInt32(1664525) + UInt32(1013904223)
        var rx = Float32(state >> UInt32(8)) / Float32(16777216)
        state = state * UInt32(1664525) + UInt32(1013904223)
        var ry = Float32(state >> UInt32(8)) / Float32(16777216)
        state = state * UInt32(1664525) + UInt32(1013904223)
        var dx = (Float32(state >> UInt32(8)) / Float32(16777216) - Float32(0.5)) * Float32(0.1)
        rays[unsafe_offset=8 * i + 0] = rx * Float32(GRID) * Float32(1.1) - Float32(0.5)
        rays[unsafe_offset=8 * i + 1] = ry * Float32(GRID) * Float32(1.1) - Float32(0.5)
        rays[unsafe_offset=8 * i + 2] = Float32(5)
        rays[unsafe_offset=8 * i + 3] = Float32(0)
        rays[unsafe_offset=8 * i + 4] = dx
        rays[unsafe_offset=8 * i + 5] = Float32(0)
        rays[unsafe_offset=8 * i + 6] = Float32(-1)
        rays[unsafe_offset=8 * i + 7] = Float32(100)
    var ref_t = unsafe_alloc[Float32](NRAYS)
    var ref_u = unsafe_alloc[Float32](NRAYS)
    var ref_v = unsafe_alloc[Float32](NRAYS)
    var ref_mesh = unsafe_alloc[Int32](NRAYS)
    var ref_tri = unsafe_alloc[Int32](NRAYS)
    var ref_hit = unsafe_alloc[UInt8](NRAYS)
    assert_true(Int(vulkanrt_trace_rays(scene, Int32(NRAYS), rays, ref_t, ref_u, ref_v, ref_mesh, ref_tri, ref_hit)) == 1)

    # The acceleration structure's bytes.
    var as_address = unsafe_alloc[UInt64](1)
    var as_size = vulkanrt_debug_read_as(scene, Int32(0), Int32(0), Pointer[UInt8, MutUntrackedOrigin].unsafe_dangling(), Int64(0), as_address)
    assert_true(as_size > 0)
    var as_bytes = unsafe_alloc[UInt8](Int(as_size))
    _ = vulkanrt_debug_read_as(scene, Int32(0), Int32(0), as_bytes, as_size, as_address)

    var ctx = DeviceContext()
    var rt = rtcore_create(cubin.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](), as_bytes, as_size, as_address[unsafe_offset=0])
    if Int(rt) == 0:
        print("SKIP: rtcore unavailable (CUDA-less stub or load failure)")
        vulkanrt_destroy_scene(scene)
        return

    var d_rays = ctx.enqueue_create_buffer[DType.float32](8 * NRAYS)
    var d_res = ctx.enqueue_create_buffer[DType.float32](8 * NRAYS)
    with d_rays.map_to_host() as h:
        for i in range(8 * NRAYS):
            h[i] = rays[unsafe_offset=i]
    var stream = CUDA(ctx.stream())
    var rc = rtcore_trace(rt, UInt64(Int(d_rays.unsafe_ptr())), UInt64(Int(d_res.unsafe_ptr())), Int32(NRAYS), stream)
    assert_true(Int(rc) == 1)
    ctx.synchronize()

    var hits = 0
    var bad = 0
    with d_res.map_to_host() as r:
        var words = r.unsafe_ptr().unsafe_bitcast[UInt32]()
        for i in range(NRAYS):
            var word = words[unsafe_offset=8 * i + 3]
            if ref_hit[unsafe_offset=i] == 1:
                hits += 1
                var ok = (word != RTCORE_MISS) and (Int32(word & RTCORE_PRIM_MASK) == ref_tri[unsafe_offset=i])
                ok = ok and abs(r[8 * i + 0] - ref_t[unsafe_offset=i]) < Float32(1e-4)
                ok = ok and abs(r[8 * i + 1] - ref_u[unsafe_offset=i]) < Float32(1e-4)
                ok = ok and abs(r[8 * i + 2] - ref_v[unsafe_offset=i]) < Float32(1e-4)
                if not ok:
                    bad += 1
            else:
                if word != RTCORE_MISS:
                    bad += 1
    print("rtcore:", hits, "hits of", NRAYS, "rays,", bad, "mismatches")
    assert_true(hits > NRAYS // 2)
    assert_true(bad == 0)
    rtcore_destroy(rt)
    vulkanrt_destroy_scene(scene)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
