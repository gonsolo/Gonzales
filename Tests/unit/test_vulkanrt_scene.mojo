from std.memory import alloc
from std.sys import has_accelerator
from std.testing import assert_true, TestSuite
from gonzales.geometry import TriangleMesh_C
from gonzales.vulkanrt import vulkanrt_build_scene, vulkanrt_trace_ray, vulkanrt_destroy_scene

# Task #162 step 2: proves vulkanrt_build_scene/vulkanrt_trace_ray produce a
# real Vulkan BLAS/TLAS from gonzales's own TriangleMesh_C data (not the
# hardcoded single triangle test_vulkanrt_smoke.mojo exercises), and that
# tracing against it reports the correct (mesh, triangle) hit -- the same
# (meshIdx, triIdx) numbering gonzales's own PrimId_C uses.
#
# Two separate meshes, one triangle each, placed far apart in x so a probe
# ray can unambiguously land on one or the other (or neither).

def test_vulkanrt_build_scene_traces_correct_mesh_and_triangle() raises:
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
        pts0, UnsafePointer[Int64, MutExternalOrigin].unsafe_dangling(), idx0,
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    )
    meshes[1] = TriangleMesh_C(
        pts1, UnsafePointer[Int64, MutExternalOrigin].unsafe_dangling(), idx1,
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutExternalOrigin].unsafe_dangling(),
    )

    var point_counts = alloc[Int64](2)
    point_counts[0] = 3; point_counts[1] = 3
    var idx_counts = alloc[Int64](2)
    idx_counts[0] = 3; idx_counts[1] = 3

    var scene = vulkanrt_build_scene(meshes, Int64(2), point_counts, idx_counts)
    assert_true(Int(scene) != 0)

    var out_t = alloc[Float32](1)
    var out_mesh = alloc[Int32](1)
    var out_tri = alloc[Int32](1)

    # Ray hits mesh 0's triangle.
    var hit0 = vulkanrt_trace_ray(scene, 0.25, 0.25, -1.0, 0, 0, 1, 0.001, 10.0,
                                   out_t, out_mesh, out_tri)
    assert_true(Int(hit0) == 1)
    assert_true(Int(out_mesh[0]) == 0)
    assert_true(Int(out_tri[0]) == 0)
    assert_true(out_t[0] > Float32(0.9) and out_t[0] < Float32(1.1))

    # Ray hits mesh 1's triangle.
    var hit1 = vulkanrt_trace_ray(scene, 5.25, 0.25, -1.0, 0, 0, 1, 0.001, 10.0,
                                   out_t, out_mesh, out_tri)
    assert_true(Int(hit1) == 1)
    assert_true(Int(out_mesh[0]) == 1)
    assert_true(Int(out_tri[0]) == 0)

    # Ray hits neither.
    var hit_miss = vulkanrt_trace_ray(scene, 100.0, 100.0, -1.0, 0, 0, 1, 0.001, 10.0,
                                       out_t, out_mesh, out_tri)
    assert_true(Int(hit_miss) == 0)

    vulkanrt_destroy_scene(scene)

    out_t.free(); out_mesh.free(); out_tri.free()
    point_counts.free(); idx_counts.free()
    meshes.free()
    pts0.free(); idx0.free(); pts1.free(); idx1.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
