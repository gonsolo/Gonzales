from std.memory import alloc
from std.sys import has_accelerator
from std.testing import assert_true, TestSuite
from std.math import abs
from gonzales.geometry import TriangleMesh_C, intersect_triangle
from gonzales.vulkanrt import (
    vulkanrt_build_scene, vulkanrt_trace_rays, vulkanrt_destroy_scene,
)

# Task #162 step 3: proves vulkanrt_trace_rays batches many rays into ONE
# dispatch (unlike test_vulkanrt_scene.mojo's vulkanrt_trace_ray, one queue
# submit per ray -- far too slow for a real renderer) against a real
# 2-mesh scene, and cross-checks the hardware's barycentric convention
# against gonzales's own Moller-Trumbore intersect_triangle (geometry.mojo)
# for the same ray/triangle -- the numbers must agree, not just "some
# hit was reported", to trust this backend's output downstream.

comptime EPS: Float32 = 1e-3

def test_vulkanrt_trace_rays_batches_and_matches_cpu_barycentrics() raises:
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return

    # Mesh 0: triangle (0,0,0)-(1,0,0)-(0,1,0) at z=0.
    var pts0 = alloc[Float32](12)
    var tri0 = [Float32(0), 0, 0, 1,  1, 0, 0, 1,  0, 1, 0, 1]
    for i in range(12):
        pts0[i] = tri0[i]
    var idx0 = alloc[Int64](3)
    idx0[0] = 0; idx0[1] = 1; idx0[2] = 2

    # Mesh 1: triangle (5,0,0)-(6,0,0)-(5,1,0) at z=0, far away in x.
    var pts1 = alloc[Float32](12)
    var tri1 = [Float32(5), 0, 0, 1,  6, 0, 0, 1,  5, 1, 0, 1]
    for i in range(12):
        pts1[i] = tri1[i]
    var idx1 = alloc[Int64](3)
    idx1[0] = 0; idx1[1] = 1; idx1[2] = 2

    var meshes = alloc[TriangleMesh_C](2)
    meshes[0] = TriangleMesh_C(
        pts0, UnsafePointer[Int64, MutAnyOrigin].unsafe_dangling(), idx0,
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    )
    meshes[1] = TriangleMesh_C(
        pts1, UnsafePointer[Int64, MutAnyOrigin].unsafe_dangling(), idx1,
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    )

    var point_counts = alloc[Int64](2)
    point_counts[0] = 3; point_counts[1] = 3
    var idx_counts = alloc[Int64](2)
    idx_counts[0] = 3; idx_counts[1] = 3

    var scene = vulkanrt_build_scene(meshes, Int64(2), point_counts, idx_counts)
    assert_true(Int(scene) != 0)

    # 4 rays in one batch: hits mesh 0, hits mesh 1, misses both, repeats
    # mesh 0's ray at a different offset.
    comptime N = 4
    var rays = alloc[Float32](N * 8)
    # ray 0: hits mesh 0 at (0.25, 0.25)
    rays[0] = 0.25; rays[1] = 0.25; rays[2] = -1.0; rays[3] = 0.001
    rays[4] = 0.0; rays[5] = 0.0; rays[6] = 1.0; rays[7] = 10.0
    # ray 1: hits mesh 1 at (5.25, 0.25)
    rays[8] = 5.25; rays[9] = 0.25; rays[10] = -1.0; rays[11] = 0.001
    rays[12] = 0.0; rays[13] = 0.0; rays[14] = 1.0; rays[15] = 10.0
    # ray 2: misses both
    rays[16] = 100.0; rays[17] = 100.0; rays[18] = -1.0; rays[19] = 0.001
    rays[20] = 0.0; rays[21] = 0.0; rays[22] = 1.0; rays[23] = 10.0
    # ray 3: hits mesh 0 at (0.1, 0.1)
    rays[24] = 0.1; rays[25] = 0.1; rays[26] = -1.0; rays[27] = 0.001
    rays[28] = 0.0; rays[29] = 0.0; rays[30] = 1.0; rays[31] = 10.0

    var out_t = alloc[Float32](N)
    var out_u = alloc[Float32](N)
    var out_v = alloc[Float32](N)
    var out_mesh = alloc[Int32](N)
    var out_tri = alloc[Int32](N)
    var out_hit = alloc[UInt8](N)

    var rc = vulkanrt_trace_rays(scene, Int32(N), rays, out_t, out_u, out_v,
                                  out_mesh, out_tri, out_hit)
    assert_true(Int(rc) == 1)

    assert_true(Int(out_hit[0]) == 1)
    assert_true(Int(out_mesh[0]) == 0)
    assert_true(Int(out_tri[0]) == 0)

    assert_true(Int(out_hit[1]) == 1)
    assert_true(Int(out_mesh[1]) == 1)
    assert_true(Int(out_tri[1]) == 0)

    assert_true(Int(out_hit[2]) == 0)

    assert_true(Int(out_hit[3]) == 1)
    assert_true(Int(out_mesh[3]) == 0)

    # Cross-check the GPU's barycentrics against gonzales's own CPU
    # Moller-Trumbore for the exact same ray/triangle -- must agree, not
    # merely "a hit happened".
    var cpu0 = intersect_triangle(
        SIMD[DType.float32, 3](0.25, 0.25, -1.0),
        SIMD[DType.float32, 3](0.0, 0.0, 1.0),
        SIMD[DType.float32, 3](0.0, 0.0, 0.0),
        SIMD[DType.float32, 3](1.0, 0.0, 0.0),
        SIMD[DType.float32, 3](0.0, 1.0, 0.0),
        Float32(10.0),
    )
    assert_true(cpu0[0])
    assert_true(abs(out_t[0] - cpu0[1]) < EPS)
    assert_true(abs(out_u[0] - cpu0[2]) < EPS)
    assert_true(abs(out_v[0] - cpu0[3]) < EPS)

    var cpu3 = intersect_triangle(
        SIMD[DType.float32, 3](0.1, 0.1, -1.0),
        SIMD[DType.float32, 3](0.0, 0.0, 1.0),
        SIMD[DType.float32, 3](0.0, 0.0, 0.0),
        SIMD[DType.float32, 3](1.0, 0.0, 0.0),
        SIMD[DType.float32, 3](0.0, 1.0, 0.0),
        Float32(10.0),
    )
    assert_true(cpu3[0])
    assert_true(abs(out_t[3] - cpu3[1]) < EPS)
    assert_true(abs(out_u[3] - cpu3[2]) < EPS)
    assert_true(abs(out_v[3] - cpu3[3]) < EPS)

    vulkanrt_destroy_scene(scene)

    out_t.free(); out_u.free(); out_v.free()
    out_mesh.free(); out_tri.free(); out_hit.free()
    rays.free()
    point_counts.free(); idx_counts.free()
    meshes.free()
    pts0.free(); idx0.free(); pts1.free(); idx1.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
