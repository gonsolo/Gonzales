from std.ffi import external_call
from std.math import ceildiv
from std.memory import OpaquePointer, bitcast
from std.sys.info import is_amd_gpu, is_apple_gpu, is_gpu, is_nvidia_gpu
from max.gpu import global_idx
from max.gpu.host import DeviceContext

# Hardware ray tracing on Intel Xe2 (src/xert): the driver builds the acceleration structures, kernels trace them with
# IGC's ray-query built-ins (opencl-c-intel.h, cl_intel_rt_production), linked by name from IGC's built-in library.

def xert_build_blas(tri_verts: Pointer[Float32, MutUntrackedOrigin], n_tris: Int64,
                    out_size: Pointer[Int64, MutUntrackedOrigin]) -> Pointer[UInt8, MutUntrackedOrigin]:
    return external_call["xert_build_blas", Pointer[UInt8, MutUntrackedOrigin],
        Pointer[Float32, MutUntrackedOrigin], Int64, Pointer[Int64, MutUntrackedOrigin]](tri_verts, n_tris, out_size)

def xert_build_blas_geoms(n_geoms: Int32, tri_verts: Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin],
                          n_tris: Pointer[Int64, MutUntrackedOrigin], out_size: Pointer[Int64, MutUntrackedOrigin],
                          bounds: Pointer[Float32, MutUntrackedOrigin]) -> Pointer[UInt8, MutUntrackedOrigin]:
    return external_call["xert_build_blas_geoms", Pointer[UInt8, MutUntrackedOrigin], Int32,
        Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin], Pointer[Int64, MutUntrackedOrigin],
        Pointer[Int64, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]](n_geoms, tri_verts, n_tris, out_size, bounds)

def xert_build_tlas(n_inst: Int64, xf: Pointer[Float32, MutUntrackedOrigin], blas_of: Pointer[Int32, MutUntrackedOrigin],
                    blas_addr: Pointer[UInt64, MutUntrackedOrigin], blas_bounds: Pointer[Float32, MutUntrackedOrigin],
                    out_size: Pointer[Int64, MutUntrackedOrigin]) -> Pointer[UInt8, MutUntrackedOrigin]:
    return external_call["xert_build_tlas", Pointer[UInt8, MutUntrackedOrigin], Int64, Pointer[Float32, MutUntrackedOrigin],
        Pointer[Int32, MutUntrackedOrigin], Pointer[UInt64, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin],
        Pointer[Int64, MutUntrackedOrigin]](n_inst, xf, blas_of, blas_addr, blas_bounds, out_size)

def xert_alignment() -> Int64:
    return external_call["xert_alignment", Int64]()

def xert_free(p: Pointer[UInt8, MutUntrackedOrigin]):
    external_call["xert_free", NoneType, Pointer[UInt8, MutUntrackedOrigin]](p)

def xert_set_scene(accel: UInt64, ord_inst: Int64, ord_mesh: UInt64, n_meshes: Int64, shadow_slots: Int32, alpha: Int32):
    external_call["xert_set_scene", NoneType, UInt64, Int64, UInt64, Int64, Int32, Int32](
        accel, ord_inst, ord_mesh, n_meshes, shadow_slots, alpha)

def xert_active() -> Bool:
    return external_call["xert_scene_accel", UInt64]() != 0

def xert_shadow_slots() -> Int:
    return Int(external_call["xert_shadow_slots", Int32]())

def xert_alpha() -> Bool:
    return external_call["xert_alpha", Int32]() != 0

comptime XERT_ANY_HIT = UInt32(0x04)   # intel_ray_flags_accept_first_hit_and_end_search

# intel_ray_desc_t: origin, direction, t range, 8-bit instance mask, intel_ray_flags_t.
@fieldwise_init
struct XeRayDesc(TrivialRegisterPassable):
    var ox: Float32
    var oy: Float32
    var oz: Float32
    var dx: Float32
    var dy: Float32
    var dz: Float32
    var tmin: Float32
    var tmax: Float32
    var mask: UInt32
    var flags: UInt32

# One hit of xert_trace: t = -1 and prim = -1 on a miss; u, v weight the triangle's second and third vertex; geom is the
# geometry within the bottom-level structure, inst the instance (its user id) when tracing a top-level structure.
@fieldwise_init
struct XeHit(TrivialRegisterPassable):
    var t: Float32
    var u: Float32
    var v: Float32
    var prim: Int32
    var geom: Int32
    var inst: Int32

# The closest (or with XERT_ANY_HIT any) hit in [tmin, tmax].
@always_inline
def xert_trace(accel: MutPointer[UInt8, MutAnyOrigin], ray: XeRayDesc) -> XeHit:
    comptime if is_gpu() and not is_nvidia_gpu() and not is_amd_gpu() and not is_apple_gpu():   # Intel: no is_intel_gpu in released Mojo
        var desc = ray
        var q = external_call["intel_ray_query_init", OpaquePointer[MutAnyOrigin]](
            Pointer(to=desc), accel.unsafe_address_space_cast[AddressSpace.GLOBAL]())
        external_call["intel_ray_query_start_traversal", NoneType](q)
        external_call["intel_ray_query_sync", NoneType](q)
        var hit = XeHit(-1, 0, 0, -1, -1, -1)
        if external_call["intel_has_committed_hit", Bool](q):
            hit.t = external_call["intel_get_hit_distance", Float32](q, UInt32(0))
            hit.prim = Int32(external_call["intel_get_hit_primitive_id", UInt32](q, UInt32(0)))
            hit.geom = Int32(external_call["intel_get_hit_geometry_id", UInt32](q, UInt32(0)))
            hit.inst = Int32(external_call["intel_get_hit_instance_user_id", UInt32](q, UInt32(0)))
            var uv = SIMD[DType.float32, 2](0)
            external_call["intel_get_hit_barycentrics", NoneType](Pointer(to=uv), q, UInt32(0))   # struct return: sret first
            hit.u = uv[0]
            hit.v = uv[1]
        external_call["intel_ray_query_abandon", NoneType](q)
        return hit
    else:
        return XeHit(-1, 0, 0, -1, -1, -1)

# The rtcore_trace layout: rays are 8 floats (ox oy oz t_min dx dy dz t_max), results 8 words (t u v hit word, 4 unused)
# with the hit word XERT_MISS or the primitive id.
comptime XERT_MISS = UInt32(0xFFFFFFFF)

def xert_trace_rays_gpu(accel: MutPointer[UInt8, MutAnyOrigin], rays: MutPointer[Float32, MutAnyOrigin],
                        results: MutPointer[UInt32, MutAnyOrigin], n: Int64, flags: UInt32):
    var i = global_idx.x
    if i >= Int(n):
        return
    var r = rays.unsafe_load[width=8](8 * i)
    var hit = xert_trace(accel, XeRayDesc(r[0], r[1], r[2], r[4], r[5], r[6], r[3], r[7], 0xFF, flags))
    var o = results.unsafe_offset(8 * i)
    o[unsafe_offset=0] = bitcast[DType.uint32](hit.t)
    o[unsafe_offset=1] = bitcast[DType.uint32](hit.u)
    o[unsafe_offset=2] = bitcast[DType.uint32](hit.v)
    o[unsafe_offset=3] = XERT_MISS if hit.prim < 0 else UInt32(hit.prim)

# The interop layout of vulkaninterop_unpack_results_kernel: t u v, -1, mesh (or n_meshes + instance), triangle, hit flag,
# mesh within the instance's template. Instance ord_inst holds the ordinary meshes, its geometry g is mesh ord_mesh[g].
def xert_trace_interop_gpu(accel: MutPointer[UInt8, MutAnyOrigin], rays: MutPointer[Float32, MutAnyOrigin],
                           results: MutPointer[UInt32, MutAnyOrigin], n: Int64, flags: UInt32, n_meshes: Int64,
                           ord_inst: Int64, ord_mesh: MutPointer[Int32, MutAnyOrigin]):
    var i = global_idx.x
    if i >= Int(n):
        return
    var r = rays.unsafe_load[width=8](8 * i)
    var hit = xert_trace(accel, XeRayDesc(r[0], r[1], r[2], r[4], r[5], r[6], r[3], r[7], 0xFF, flags))
    var o = results.unsafe_offset(8 * i)
    if hit.prim < 0:
        o[unsafe_offset=6] = 0
        return
    var ordinary = Int(hit.inst) == Int(ord_inst)
    o[unsafe_offset=0] = bitcast[DType.uint32](hit.t)
    o[unsafe_offset=1] = bitcast[DType.uint32](hit.u)
    o[unsafe_offset=2] = bitcast[DType.uint32](hit.v)
    o[unsafe_offset=3] = UInt32(0xFFFFFFFF)
    o[unsafe_offset=4] = UInt32(ord_mesh[unsafe_offset=Int(hit.geom)]) if ordinary else UInt32(Int(n_meshes) + Int(hit.inst))
    o[unsafe_offset=5] = UInt32(hit.prim)
    o[unsafe_offset=6] = 1
    o[unsafe_offset=7] = 0 if ordinary else UInt32(hit.geom)

# Enqueues the trace of m rays (device addresses, interop layout) against the active scene; any_hit for shadow rays.
def xert_trace_interop(ctx: DeviceContext, rays: Int, results: Int, m: Int, any_hit: Bool) raises:
    ctx.enqueue_function[xert_trace_interop_gpu](
        MutPointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(external_call["xert_scene_accel", UInt64]())),
        MutPointer[Float32, MutAnyOrigin](unsafe_from_address=rays), MutPointer[UInt32, MutAnyOrigin](unsafe_from_address=results),
        Int64(m), XERT_ANY_HIT if any_hit else UInt32(0), external_call["xert_scene_n_meshes", Int64](),
        external_call["xert_scene_ord_inst", Int64](),
        MutPointer[Int32, MutAnyOrigin](unsafe_from_address=Int(external_call["xert_scene_ord_mesh", UInt64]())),
        grid_dim=ceildiv(m, 256), block_dim=256)
