# Reproducer for the --gpu --vcm CUDA_ERROR_ILLEGAL_ADDRESS (gonzales).
#
# Shape under test: a BVH traversal whose result cell is ALSO the result
# cell of a traversal still on the stack further up -- exactly what
# _visible_transmittance did when reached via _connect from
# _bdpt_trace_camera_and_connect.
#
#   build: mojo build -I src repro_alias.mojo -o repro_alias
#   run:   ./repro_alias cpu       # same traversal on CPU: validates the data
#          ./repro_alias single    # GPU, one traversal, no aliasing
#          ./repro_alias aliased   # GPU, nested traversal sharing one cell
#          ./repro_alias private   # GPU, nested traversal, private inner cell
from std.sys import has_accelerator, argv
from std.sys.info import size_of
from std.gpu import block_idx, thread_idx, block_dim
from max.gpu.host import DeviceContext
from gonzales.geometry import (
    Point3f, Vec3f, Ray_C, PrimId_C, Intersection_C, TriangleMesh_C, Curve_C,
)
from gonzales.bvh import BVH2Node, traverse_bvh2_core

comptime NO_CURVES = UnsafePointer[Curve_C, MutExternalOrigin].unsafe_dangling

def _nested_shared(
    bvh: UnsafePointer[BVH2Node, MutExternalOrigin],
    prims: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    cell: UnsafePointer[Intersection_C, MutExternalOrigin],
    org: Point3f, dir: Vec3f,
) -> Float32:
    """Traverses through the CALLER'S cell -- the aliasing under test."""
    var ray = Ray_C(org, dir)
    cell[0].hit = Int8(0)
    traverse_bvh2_core(bvh, prims, meshes, NO_CURVES(), ray, Float32(50), cell)
    return cell[0].tHit if cell[0].hit != Int8(0) else Float32(-1)

def _nested_private(
    bvh: UnsafePointer[BVH2Node, MutExternalOrigin],
    prims: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    org: Point3f, dir: Vec3f,
) -> Float32:
    """Same, but with its own local cell -- the fix."""
    var mine = InlineArray[Intersection_C, 1](fill=Intersection_C(
        PrimId_C(Int64(-1), Int64(-1), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0)),
        Float32(0), Float32(0), Float32(0), Int8(0), Int8(0), Int8(0), Int8(0)))
    var cell = mine.unsafe_ptr().unsafe_origin_cast[MutExternalOrigin]()
    var ray = Ray_C(org, dir)
    cell[0].hit = Int8(0)
    traverse_bvh2_core(bvh, prims, meshes, NO_CURVES(), ray, Float32(50), cell)
    return cell[0].tHit if cell[0].hit != Int8(0) else Float32(-1)

def k_single(
    bvh: UnsafePointer[BVH2Node, MutExternalOrigin],
    prims: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    scratch: UnsafePointer[Intersection_C, MutExternalOrigin],
    out_buf: UnsafePointer[Float32, MutExternalOrigin],
):
    if Int(block_idx.x * block_dim.x + thread_idx.x) != 0: return
    var org = Point3f(Float32(0), Float32(0), Float32(-2))
    var dir = Vec3f(Float32(0), Float32(0), Float32(1))
    var ray = Ray_C(org, dir)
    scratch[0].hit = Int8(0)
    traverse_bvh2_core(bvh, prims, meshes, NO_CURVES(), ray, Float32(1e38), scratch)
    out_buf[0] = scratch[0].tHit if scratch[0].hit != Int8(0) else Float32(-1)
    out_buf[1] = Float32(0)

def k_aliased(
    bvh: UnsafePointer[BVH2Node, MutExternalOrigin],
    prims: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    scratch: UnsafePointer[Intersection_C, MutExternalOrigin],
    out_buf: UnsafePointer[Float32, MutExternalOrigin],
):
    if Int(block_idx.x * block_dim.x + thread_idx.x) != 0: return
    var org = Point3f(Float32(0), Float32(0), Float32(-2))
    var dir = Vec3f(Float32(0), Float32(0), Float32(1))
    var ray = Ray_C(org, dir)
    scratch[0].hit = Int8(0)
    traverse_bvh2_core(bvh, prims, meshes, NO_CURVES(), ray, Float32(1e38), scratch)
    var inter = scratch[0]                     # copy out, as the renderer does
    var t1 = _nested_shared(bvh, prims, meshes, scratch, org, dir)
    out_buf[0] = inter.tHit if inter.hit != Int8(0) else Float32(-1)
    out_buf[1] = t1

def k_private(
    bvh: UnsafePointer[BVH2Node, MutExternalOrigin],
    prims: UnsafePointer[PrimId_C, MutExternalOrigin],
    meshes: UnsafePointer[TriangleMesh_C, MutExternalOrigin],
    scratch: UnsafePointer[Intersection_C, MutExternalOrigin],
    out_buf: UnsafePointer[Float32, MutExternalOrigin],
):
    if Int(block_idx.x * block_dim.x + thread_idx.x) != 0: return
    var org = Point3f(Float32(0), Float32(0), Float32(-2))
    var dir = Vec3f(Float32(0), Float32(0), Float32(1))
    var ray = Ray_C(org, dir)
    scratch[0].hit = Int8(0)
    traverse_bvh2_core(bvh, prims, meshes, NO_CURVES(), ray, Float32(1e38), scratch)
    var inter = scratch[0]
    var t1 = _nested_private(bvh, prims, meshes, org, dir)
    out_buf[0] = inter.tHit if inter.hit != Int8(0) else Float32(-1)
    out_buf[1] = t1

def main() raises:
    var mode = String("aliased")
    var av = argv()
    if len(av) > 1: mode = String(av[1])

    # one triangle at z=0, points stride 4 floats
    var pts = alloc[Float32](12)
    for i in range(12): pts[i] = Float32(0)
    pts[0]=-1.0; pts[1]=-1.0; pts[2]=0.0
    pts[4]= 1.0; pts[5]=-1.0; pts[6]=0.0
    pts[8]= 0.0; pts[9]= 1.0; pts[10]=0.0
    var vidx = alloc[Int64](3)
    vidx[0]=0; vidx[1]=1; vidx[2]=2
    var mesh_h = alloc[TriangleMesh_C](1)
    mesh_h[0] = TriangleMesh_C(pts, vidx, vidx,
        UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=1),
        UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=1))
    var prim_h = alloc[PrimId_C](1)
    prim_h[0] = PrimId_C(Int64(0), Int64(0), Int64(0), Int32(-1), Int8(0), Int8(0), Int8(0), Int8(0))
    var bvh_h = alloc[BVH2Node](1)
    bvh_h[0] = BVH2Node(Point3f(-1.1,-1.1,-0.1), Point3f(1.1,1.1,0.1), Int32(0), Int32(1))

    if mode == "cpu":
        var cell = alloc[Intersection_C](1)
        var ray = Ray_C(Point3f(0,0,-2), Vec3f(0,0,1))
        cell[0].hit = Int8(0)
        traverse_bvh2_core(bvh_h, prim_h, mesh_h, NO_CURVES(), ray, Float32(1e38), cell)
        print("CPU: hit=", Int(cell[0].hit), " tHit=", cell[0].tHit, " (expect hit=1 tHit=2)")
        var t1 = _nested_shared(bvh_h, prim_h, mesh_h, cell, Point3f(0,0,-2), Vec3f(0,0,1))
        print("CPU nested(shared cell): t=", t1, " -- data is valid if these are 2.0")
        return

    comptime if not has_accelerator():
        print("SKIP: no GPU")
        return
    var ctx = DeviceContext()
    var pts_d = ctx.enqueue_create_buffer[DType.float32](12)
    with pts_d.map_to_host() as h:
        var p = h.unsafe_ptr()
        for i in range(12): p[unsafe_offset=i] = pts[i]
    var vi_d = ctx.enqueue_create_buffer[DType.int64](3)
    with vi_d.map_to_host() as h:
        var p = h.unsafe_ptr()
        for i in range(3): p[unsafe_offset=i] = vidx[i]
    ctx.synchronize()
    var mesh_d = ctx.enqueue_create_buffer[DType.uint8](size_of[TriangleMesh_C]())
    with mesh_d.map_to_host() as h:
        h.unsafe_ptr().bitcast[TriangleMesh_C]()[0] = TriangleMesh_C(
            pts_d.unsafe_ptr().bitcast[Float32]().unsafe_origin_cast[MutExternalOrigin](),
            vi_d.unsafe_ptr().bitcast[Int64]().unsafe_origin_cast[MutExternalOrigin](),
            vi_d.unsafe_ptr().bitcast[Int64]().unsafe_origin_cast[MutExternalOrigin](),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=1),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=1))
    var prim_d = ctx.enqueue_create_buffer[DType.uint8](size_of[PrimId_C]())
    with prim_d.map_to_host() as h:
        h.unsafe_ptr().bitcast[PrimId_C]()[0] = prim_h[0]
    var bvh_d = ctx.enqueue_create_buffer[DType.uint8](size_of[BVH2Node]())
    with bvh_d.map_to_host() as h:
        h.unsafe_ptr().bitcast[BVH2Node]()[0] = bvh_h[0]
    var scr_d = ctx.enqueue_create_buffer[DType.uint8](size_of[Intersection_C]())
    var out_d = ctx.enqueue_create_buffer[DType.float32](2)
    out_d.enqueue_fill(Float32(-99))
    ctx.synchronize()

    var B = bvh_d.unsafe_ptr().bitcast[BVH2Node]()
    var P = prim_d.unsafe_ptr().bitcast[PrimId_C]()
    var M = mesh_d.unsafe_ptr().bitcast[TriangleMesh_C]()
    var S = scr_d.unsafe_ptr().bitcast[Intersection_C]()
    var O = out_d.unsafe_ptr()

    print("mode:", mode)
    if mode == "single":
        ctx.enqueue_function[k_single](B, P, M, S, O, grid_dim=1, block_dim=1)
    elif mode == "private":
        ctx.enqueue_function[k_private](B, P, M, S, O, grid_dim=1, block_dim=1)
    else:
        ctx.enqueue_function[k_aliased](B, P, M, S, O, grid_dim=1, block_dim=1)
    ctx.synchronize()
    with out_d.map_to_host() as h:
        var p = h.unsafe_ptr()
        print("  outer t=", p[unsafe_offset=0], "  nested t=", p[unsafe_offset=1])
    print("  OK: no illegal access")
    # Keep-alives. Mojo destroys ASAP: pts_d and vi_d are last *used* when
    # their device addresses get stored into mesh_d's TriangleMesh_C, so
    # without this the compiler is free to free those buffers before the
    # kernel runs, leaving those stored pointers dangling -- which faults as
    # CUDA_ERROR_ILLEGAL_ADDRESS and looks exactly like the bug under test.
    _ = pts_d^
    _ = vi_d^
    _ = mesh_d^
    _ = prim_d^
    _ = bvh_d^
    _ = scr_d^
    _ = out_d^
