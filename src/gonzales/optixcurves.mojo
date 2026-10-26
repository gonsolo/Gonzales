# The curves on the RT hardware through OptiX (src/optixcurves/optixcurves.h): FFI, and the build from the curve PrimIds.
from std.ffi import external_call
from std.math import ceildiv
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host._nvidia_cuda import CUDA, CUstream
from .curves import Curve, curve_at, curve_piece_endpoints
from .materials import Material, MatKind
from .primitives import PrimId

comptime OxcHandle = Pointer[UInt8, MutUntrackedOrigin]


def oxc_active() -> OxcHandle:
    return external_call["oxc_active", OxcHandle]()


def oxc_set_active(h: OxcHandle):
    external_call["oxc_set_active", NoneType, OxcHandle](h)


def oxc_destroy(h: OxcHandle):
    external_call["oxc_destroy", NoneType, OxcHandle](h)


def oxc_bytes(h: OxcHandle) -> Int:
    return external_call["oxc_bytes", Int, OxcHandle](h)


def oxc_scratch(h: OxcHandle, bytes: Int) -> UInt64:
    return external_call["oxc_scratch", UInt64, OxcHandle, Int](h, bytes)


def oxc_trace_closest(h: OxcHandle, rays: UInt64, results: UInt64, n: Int, stream: CUstream) -> Int32:
    return external_call["oxc_trace_closest", Int32, OxcHandle, UInt64, UInt64, UInt32, CUstream](h, rays, results, UInt32(n), stream)


def oxc_trace_shadow(h: OxcHandle, rays: UInt64, list: UInt64, counts: UInt64, cap: Int, slot: Int, blocked: UInt64, n: Int,
                     stream: CUstream) -> Int32:
    return external_call["oxc_trace_shadow", Int32, OxcHandle, UInt64, UInt64, UInt64, Int32, Int32, UInt64, UInt32, CUstream](
        h, rays, list, counts, Int32(cap), Int32(slot), blocked, UInt32(n), stream)


# A segment per PrimId: one curve piece, both ends and radii, and its ref (curve, id2, material). bad[0] is set when a
# PrimId covers several pieces or the curve is a null (interface) material, which the OptiX path does not handle;
# bad[1] when some segment k is not curve k's one piece in its own material, so the refs are needed.
def _segments_kernel(prim_ids: Pointer[PrimId, MutUntrackedOrigin], curves: Pointer[Curve, MutUntrackedOrigin],
                     materials: Pointer[Material, MutUntrackedOrigin], n_dp: Int64, verts: Pointer[Float32, MutUntrackedOrigin],
                     radii: Pointer[Float32, MutUntrackedOrigin], refs: Pointer[Int32, MutUntrackedOrigin], bad: Pointer[Int32, MutUntrackedOrigin]):
    var k = Int(block_idx.x * block_dim.x + thread_idx.x)
    if k >= Int(n_dp):
        return
    var p = prim_ids[unsafe_offset=k]
    if p.id2 % 8 != Int64(1) or (p.materialIndex >= Int64(0) and materials[unsafe_offset=Int(p.materialIndex)].type == MatKind.interface):
        bad[unsafe_offset=0] = Int32(1)
    var curve = curve_at(curves, Int(p.id1))
    if p.id1 != Int64(k) or p.id2 != Int64(1) or p.materialIndex != Int64(curve.materialIndex):
        bad[unsafe_offset=1] = Int32(1)
    var e = curve_piece_endpoints(curve, Int(p.id2 // 8))
    var v = verts.unsafe_offset(6 * k)
    v[unsafe_offset=0] = e[0][0]; v[unsafe_offset=1] = e[0][1]; v[unsafe_offset=2] = e[0][2]
    v[unsafe_offset=3] = e[1][0]; v[unsafe_offset=4] = e[1][1]; v[unsafe_offset=5] = e[1][2]
    radii[unsafe_offset=2 * k] = e[2]; radii[unsafe_offset=2 * k + 1] = e[3]
    var r = refs.unsafe_offset(3 * k)
    r[unsafe_offset=0] = Int32(p.id1); r[unsafe_offset=1] = Int32(p.id2); r[unsafe_offset=2] = Int32(p.materialIndex)


def gpu_build_optix_curves(ctx: DeviceContext, prim_ids: Pointer[PrimId, MutUntrackedOrigin], curves: Pointer[Curve, MutUntrackedOrigin],
                           materials: Pointer[Material, MutUntrackedOrigin], n: Int, mut refs: DeviceBuffer[DType.uint8]) raises -> Tuple[OxcHandle, Bool]:
    """The OptiX curves over the curve PrimIds [0, n), if OptiX is there and handles these curves; `refs` gets the
    segments' refs, which the handle reads and the caller keeps, or one byte when OptiX can derive them."""
    var h = external_call["oxc_create", OxcHandle]()
    if Int(h) == 0:
        return (h, False)
    var verts = ctx.enqueue_create_buffer[DType.float32](6 * n)
    var radii = ctx.enqueue_create_buffer[DType.float32](2 * n)
    refs = ctx.enqueue_create_buffer[DType.uint8](12 * n)
    var bad = ctx.enqueue_create_buffer[DType.int32](2)
    bad.enqueue_fill(Int32(0))
    var rp = refs.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    ctx.enqueue_function[_segments_kernel](prim_ids, curves, materials, Int64(n),
                                           verts.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                                           radii.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](), rp,
                                           bad.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
                                           grid_dim=ceildiv(n, 256), block_dim=256)
    var flags = SIMD[DType.int32, 2](0)
    ctx.enqueue_copy(Pointer(to=flags).unsafe_bitcast[Int32]().unsafe_origin_cast[MutUntrackedOrigin](), bad)
    ctx.synchronize()
    var refs_addr = UInt64(Int(rp))
    if flags[1] == Int32(0):
        refs = ctx.enqueue_create_buffer[DType.uint8](1)
        refs_addr = UInt64(0)
    var failed = flags[0] != Int32(0) or external_call["oxc_build", Int32, OxcHandle, UInt64, UInt64, UInt32, UInt64, UInt64, CUstream](
        h, UInt64(Int(verts.unsafe_ptr())), UInt64(Int(radii.unsafe_ptr())), UInt32(n), refs_addr, UInt64(Int(curves)),
        CUDA(ctx.stream())) != Int32(0)
    _ = verts^; _ = radii^; _ = bad^                       # the build synchronized
    if failed:
        oxc_destroy(h)
    return (h, not failed)
