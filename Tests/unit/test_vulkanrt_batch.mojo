from std.memory.alloc import unsafe_alloc
from std.sys import has_accelerator
from std.testing import assert_true, TestSuite
from std.math import abs
from gonzales.geometry import TriangleMesh_C, intersect_triangle, Vec3f
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
    var pts0 = unsafe_alloc[Float32](12)
    var tri0 = [Float32(0), 0, 0, 1,  1, 0, 0, 1,  0, 1, 0, 1]
    for i in range(12):
        pts0[unsafe_offset=i] = tri0[i]
    var idx0 = unsafe_alloc[Int64](3)
    idx0[unsafe_offset=0] = 0; idx0[unsafe_offset=1] = 1; idx0[unsafe_offset=2] = 2

    # Mesh 1: triangle (5,0,0)-(6,0,0)-(5,1,0) at z=0, far away in x.
    var pts1 = unsafe_alloc[Float32](12)
    var tri1 = [Float32(5), 0, 0, 1,  6, 0, 0, 1,  5, 1, 0, 1]
    for i in range(12):
        pts1[unsafe_offset=i] = tri1[i]
    var idx1 = unsafe_alloc[Int64](3)
    idx1[unsafe_offset=0] = 0; idx1[unsafe_offset=1] = 1; idx1[unsafe_offset=2] = 2

    var meshes = unsafe_alloc[TriangleMesh_C](2)
    meshes[unsafe_offset=0] = TriangleMesh_C(
        pts0, Pointer[Int64, MutUntrackedOrigin].unsafe_dangling(), idx0,
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    )
    meshes[unsafe_offset=1] = TriangleMesh_C(
        pts1, Pointer[Int64, MutUntrackedOrigin].unsafe_dangling(), idx1,
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
        Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
    )

    var point_counts = unsafe_alloc[Int64](2)
    point_counts[unsafe_offset=0] = 3; point_counts[unsafe_offset=1] = 3
    var idx_counts = unsafe_alloc[Int64](2)
    idx_counts[unsafe_offset=0] = 3; idx_counts[unsafe_offset=1] = 3

    var scene = vulkanrt_build_scene(meshes, Int64(2), point_counts, idx_counts)
    assert_true(Int(scene) != 0)

    # 4 rays in one batch: hits mesh 0, hits mesh 1, misses both, repeats
    # mesh 0's ray at a different offset.
    comptime N = 4
    var rays = unsafe_alloc[Float32](N * 8)
    # ray 0: hits mesh 0 at (0.25, 0.25)
    rays[unsafe_offset=0] = 0.25; rays[unsafe_offset=1] = 0.25; rays[unsafe_offset=2] = -1.0; rays[unsafe_offset=3] = 0.001
    rays[unsafe_offset=4] = 0.0; rays[unsafe_offset=5] = 0.0; rays[unsafe_offset=6] = 1.0; rays[unsafe_offset=7] = 10.0
    # ray 1: hits mesh 1 at (5.25, 0.25)
    rays[unsafe_offset=8] = 5.25; rays[unsafe_offset=9] = 0.25; rays[unsafe_offset=10] = -1.0; rays[unsafe_offset=11] = 0.001
    rays[unsafe_offset=12] = 0.0; rays[unsafe_offset=13] = 0.0; rays[unsafe_offset=14] = 1.0; rays[unsafe_offset=15] = 10.0
    # ray 2: misses both
    rays[unsafe_offset=16] = 100.0; rays[unsafe_offset=17] = 100.0; rays[unsafe_offset=18] = -1.0; rays[unsafe_offset=19] = 0.001
    rays[unsafe_offset=20] = 0.0; rays[unsafe_offset=21] = 0.0; rays[unsafe_offset=22] = 1.0; rays[unsafe_offset=23] = 10.0
    # ray 3: hits mesh 0 at (0.1, 0.1)
    rays[unsafe_offset=24] = 0.1; rays[unsafe_offset=25] = 0.1; rays[unsafe_offset=26] = -1.0; rays[unsafe_offset=27] = 0.001
    rays[unsafe_offset=28] = 0.0; rays[unsafe_offset=29] = 0.0; rays[unsafe_offset=30] = 1.0; rays[unsafe_offset=31] = 10.0

    var out_t = unsafe_alloc[Float32](N)
    var out_u = unsafe_alloc[Float32](N)
    var out_v = unsafe_alloc[Float32](N)
    var out_mesh = unsafe_alloc[Int32](N)
    var out_tri = unsafe_alloc[Int32](N)
    var out_hit = unsafe_alloc[UInt8](N)

    var rc = vulkanrt_trace_rays(scene, Int32(N), rays, out_t, out_u, out_v,
                                  out_mesh, out_tri, out_hit)
    assert_true(Int(rc) == 1)

    assert_true(Int(out_hit[unsafe_offset=0]) == 1)
    assert_true(Int(out_mesh[unsafe_offset=0]) == 0)
    assert_true(Int(out_tri[unsafe_offset=0]) == 0)

    assert_true(Int(out_hit[unsafe_offset=1]) == 1)
    assert_true(Int(out_mesh[unsafe_offset=1]) == 1)
    assert_true(Int(out_tri[unsafe_offset=1]) == 0)

    assert_true(Int(out_hit[unsafe_offset=2]) == 0)

    assert_true(Int(out_hit[unsafe_offset=3]) == 1)
    assert_true(Int(out_mesh[unsafe_offset=3]) == 0)

    # Cross-check the GPU's barycentrics against gonzales's own CPU
    # Moller-Trumbore for the exact same ray/triangle -- must agree, not
    # merely "a hit happened".
    var cpu0 = intersect_triangle(
        Vec3f(0.25, 0.25, -1.0),
        Vec3f(0.0, 0.0, 1.0),
        Vec3f(0.0, 0.0, 0.0),
        Vec3f(1.0, 0.0, 0.0),
        Vec3f(0.0, 1.0, 0.0),
        Float32(10.0),
    )
    assert_true(cpu0[0])
    assert_true(abs(out_t[unsafe_offset=0] - cpu0[1]) < EPS)
    assert_true(abs(out_u[unsafe_offset=0] - cpu0[2]) < EPS)
    assert_true(abs(out_v[unsafe_offset=0] - cpu0[3]) < EPS)

    var cpu3 = intersect_triangle(
        Vec3f(0.1, 0.1, -1.0),
        Vec3f(0.0, 0.0, 1.0),
        Vec3f(0.0, 0.0, 0.0),
        Vec3f(1.0, 0.0, 0.0),
        Vec3f(0.0, 1.0, 0.0),
        Float32(10.0),
    )
    assert_true(cpu3[0])
    assert_true(abs(out_t[unsafe_offset=3] - cpu3[1]) < EPS)
    assert_true(abs(out_u[unsafe_offset=3] - cpu3[2]) < EPS)
    assert_true(abs(out_v[unsafe_offset=3] - cpu3[3]) < EPS)

    vulkanrt_destroy_scene(scene)

    out_t.unsafe_free(); out_u.unsafe_free(); out_v.unsafe_free()
    out_mesh.unsafe_free(); out_tri.unsafe_free(); out_hit.unsafe_free()
    rays.unsafe_free()
    point_counts.unsafe_free(); idx_counts.unsafe_free()
    meshes.unsafe_free()
    pts0.unsafe_free(); idx0.unsafe_free(); pts1.unsafe_free(); idx1.unsafe_free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
