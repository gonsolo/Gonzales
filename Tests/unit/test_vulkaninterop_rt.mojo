from std.memory import alloc
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.host._nvidia_cuda import CUDA
from std.testing import assert_true, TestSuite
from gonzales.geometry import TriangleMesh_C
from gonzales.vulkaninterop import (
    vulkaninterop_rt_create_scene, vulkaninterop_rt_get_rays_ptr,
    vulkaninterop_rt_get_results_ptr, vulkaninterop_rt_trace,
    vulkaninterop_rt_destroy_scene, vulkaninterop_rt_write_test_rays_kernel,
)

# Task #163 stage 2: proves real VK_KHR_ray_query tracing through the
# CUDA/Vulkan GPU-side interop mechanism, exercised from gonzales's own
# Mojo GPU kernels -- a real Mojo enqueue_function kernel writes ray data
# directly into the interop-shared rays buffer (no host copy),
# vulkaninterop_rt_trace hands off to a real ray-query dispatch against a
# real 2-mesh BLAS/TLAS scene entirely on the GPU timeline (no
# ctx.synchronize() until the very end, to inspect results for this test's
# assertions), and the results are read back and checked against known
# geometry -- same 2-mesh scene and expected hits as
# Tests/unit/test_vulkanrt_scene.mojo, cross-validating that the interop
# path produces identical results to vulkanrt.mojo's non-interop path.

def test_vulkaninterop_rt_trace_matches_known_geometry() raises:
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

    comptime maxRays = 16
    var no_templates = UnsafePointer[Int64, MutAnyOrigin].unsafe_dangling()
    var no_instances = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
    var no_instance_tmpl = UnsafePointer[Int32, MutAnyOrigin].unsafe_dangling()
    var no_curve_aabbs = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
    var no_curve_prim_idx = UnsafePointer[Int64, MutAnyOrigin].unsafe_dangling()
    var scene = vulkaninterop_rt_create_scene(
        meshes, Int64(2), point_counts, idx_counts,
        Int64(0), no_templates, no_templates,
        Int64(0), no_instances, no_instance_tmpl,
        Int64(0), no_curve_aabbs, no_curve_prim_idx,
        Int64(maxRays))
    assert_true(Int(scene) != 0)

    var raysPtr = vulkaninterop_rt_get_rays_ptr(scene)
    var resultsPtr = vulkaninterop_rt_get_results_ptr(scene)
    assert_true(Int(raysPtr) != 0)
    assert_true(Int(resultsPtr) != 0)

    var ctx = DeviceContext()
    var raysBuf = DeviceBuffer[DType.float32](ctx, raysPtr, maxRays * 8, owning=False)
    var resultsBuf = DeviceBuffer[DType.float32](ctx, resultsPtr, maxRays * 8, owning=False)

    # Write 3 known test rays directly into the interop buffer via a real
    # Mojo kernel -- no host round trip.
    ctx.enqueue_function[vulkaninterop_rt_write_test_rays_kernel](
        raysBuf.unsafe_ptr(), grid_dim=1, block_dim=3)

    var cuda_stream = CUDA(ctx.stream())
    var rc = vulkaninterop_rt_trace(scene, Int32(3), cuda_stream)
    assert_true(Int(rc) == 1)

    ctx.synchronize()
    with resultsBuf.map_to_host() as host:
        var fptr = host.unsafe_ptr()
        var iptr = fptr.bitcast[Int32]()

        # ray 0: hits mesh 0's triangle at u=v=0.25 (Moller-Trumbore
        # convention, same as vulkanrt.mojo's own verified barycentrics).
        assert_true(iptr[0*8 + 6] == 1)          # hitFlag
        assert_true(iptr[0*8 + 4] == 0)          # hitMesh
        assert_true(iptr[0*8 + 5] == 0)          # hitTriangle
        assert_true(fptr[0*8 + 0] > Float32(0.9) and fptr[0*8 + 0] < Float32(1.1))  # hitT ~ 1.0
        assert_true(fptr[0*8 + 1] > Float32(0.2) and fptr[0*8 + 1] < Float32(0.3))  # u ~ 0.25
        assert_true(fptr[0*8 + 2] > Float32(0.2) and fptr[0*8 + 2] < Float32(0.3))  # v ~ 0.25

        # ray 1: hits mesh 1's triangle.
        assert_true(iptr[1*8 + 6] == 1)
        assert_true(iptr[1*8 + 4] == 1)
        assert_true(iptr[1*8 + 5] == 0)

        # ray 2: misses both.
        assert_true(iptr[2*8 + 6] == 0)

    vulkaninterop_rt_destroy_scene(scene)

    point_counts.free(); idx_counts.free()
    meshes.free()
    pts0.free(); idx0.free(); pts1.free(); idx1.free()

def test_vulkaninterop_rt_instancing_places_template_correctly() raises:
    comptime if not has_accelerator():
        print("SKIP: no GPU accelerator on this machine")
        return

    # One template mesh (object space): triangle (0,0,0)-(1,0,0)-(0,1,0),
    # z=0 -- same shape as test_vulkaninterop_rt_trace_matches_known_geometry's
    # mesh 0, but placed via TWO instances instead of being an ordinary
    # top-level mesh: instance 0 at identity (same world position as before)
    # and instance 1 translated +5 in x (same world position as that test's
    # mesh 1). Exercises per-instance transform correctness, the
    # instanceCustomIndex = mesh_count + instance_index encoding, and
    # geometryIndex reporting -- see vulkaninterop_rt_create_scene's
    # docstring.
    var pts0 = alloc[Float32](12)
    var tri0 = [Float32(0), 0, 0, 1,  1, 0, 0, 1,  0, 1, 0, 1]
    for i in range(12):
        pts0[i] = tri0[i]
    var idx0 = alloc[Int64](3)
    idx0[0] = 0; idx0[1] = 1; idx0[2] = 2

    var meshes = alloc[TriangleMesh_C](1)
    meshes[0] = TriangleMesh_C(
        pts0, UnsafePointer[Int64, MutAnyOrigin].unsafe_dangling(), idx0,
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
        UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling(),
    )
    var point_counts = alloc[Int64](1)
    point_counts[0] = 3
    var idx_counts = alloc[Int64](1)
    idx_counts[0] = 3

    var template_start = alloc[Int64](1); template_start[0] = 0
    var template_end   = alloc[Int64](1); template_end[0] = 1

    # Column-major 4x4 identity and +5-in-x translation (gonzales's own
    # transform.mojo convention: M[col*4+row], translation in column 3).
    var o2w = alloc[Float32](32)
    for i in range(32): o2w[i] = Float32(0)
    o2w[0] = 1; o2w[5] = 1; o2w[10] = 1; o2w[15] = 1               # instance 0: identity
    o2w[16] = 1; o2w[21] = 1; o2w[26] = 1; o2w[31] = 1; o2w[28] = 5  # instance 1: +5 in x
    var inst_tmpl_idx = alloc[Int32](2)
    inst_tmpl_idx[0] = 0; inst_tmpl_idx[1] = 0

    comptime maxRays = 16
    var no_curve_aabbs = UnsafePointer[Float32, MutAnyOrigin].unsafe_dangling()
    var no_curve_prim_idx = UnsafePointer[Int64, MutAnyOrigin].unsafe_dangling()
    var scene = vulkaninterop_rt_create_scene(
        meshes, Int64(1), point_counts, idx_counts,
        Int64(1), template_start, template_end,
        Int64(2), o2w, inst_tmpl_idx,
        Int64(0), no_curve_aabbs, no_curve_prim_idx,
        Int64(maxRays))
    assert_true(Int(scene) != 0)

    var raysPtr = vulkaninterop_rt_get_rays_ptr(scene)
    var resultsPtr = vulkaninterop_rt_get_results_ptr(scene)

    var ctx = DeviceContext()
    var raysBuf = DeviceBuffer[DType.float32](ctx, raysPtr, maxRays * 8, owning=False)
    var resultsBuf = DeviceBuffer[DType.float32](ctx, resultsPtr, maxRays * 8, owning=False)

    # Same 3 test rays as the ordinary-mesh test: ray0 hits the (0,0,0)-
    # (1,0,0)-(0,1,0) triangle -- now instance 0's world position -- ray1
    # hits the +5-in-x-shifted triangle -- now instance 1's world position
    # -- ray2 misses both.
    ctx.enqueue_function[vulkaninterop_rt_write_test_rays_kernel](
        raysBuf.unsafe_ptr(), grid_dim=1, block_dim=3)

    var cuda_stream = CUDA(ctx.stream())
    var rc = vulkaninterop_rt_trace(scene, Int32(3), cuda_stream)
    assert_true(Int(rc) == 1)

    ctx.synchronize()
    with resultsBuf.map_to_host() as host:
        var fptr = host.unsafe_ptr()
        var iptr = fptr.bitcast[Int32]()

        # ray 0 -> instance 0: instanceCustomIndex = mesh_count(1) + 0 = 1.
        assert_true(iptr[0*8 + 6] == 1)              # hitFlag
        assert_true(iptr[0*8 + 4] == 1)               # hitMesh (raw instanceCustomIndex)
        assert_true(iptr[0*8 + 5] == 0)               # hitTriangle
        assert_true(iptr[0*8 + 7] == 0)               # geometryIndex (template has 1 mesh)
        assert_true(fptr[0*8 + 0] > Float32(0.9) and fptr[0*8 + 0] < Float32(1.1))  # hitT ~ 1.0

        # ray 1 -> instance 1: instanceCustomIndex = mesh_count(1) + 1 = 2.
        assert_true(iptr[1*8 + 6] == 1)
        assert_true(iptr[1*8 + 4] == 2)
        assert_true(iptr[1*8 + 5] == 0)
        assert_true(iptr[1*8 + 7] == 0)

        # ray 2: misses both instances.
        assert_true(iptr[2*8 + 6] == 0)

    vulkaninterop_rt_destroy_scene(scene)

    point_counts.free(); idx_counts.free()
    meshes.free()
    pts0.free(); idx0.free()
    template_start.free(); template_end.free()
    o2w.free(); inst_tmpl_idx.free()

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
