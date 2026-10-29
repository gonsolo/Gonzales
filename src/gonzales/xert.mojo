from std.atomic import Atomic
from std.ffi import external_call
from std.os import getenv
from std.math import ceildiv, sqrt
from std.memory import OpaquePointer, bitcast
from std.sys.info import is_amd_gpu, is_apple_gpu, is_gpu, is_nvidia_gpu
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from .curves import Curve, curve_at, intersect_curve
from .geometry import Vec3f
from .bvh import _shadow_is_null_material
from .materials import Material
from .primitives import PrimId

# Hardware ray tracing on Intel Xe2 (src/xert): the driver builds the acceleration structures, kernels trace them with
# IGC's ray-query built-ins (opencl-c-intel.h, cl_intel_rt_production), linked by name from IGC's built-in library.

def xert_build_blas(tri_verts: Pointer[Float32, MutUntrackedOrigin], n_tris: Int64,
                    out_size: Pointer[Int64, MutUntrackedOrigin]) -> Pointer[UInt8, MutUntrackedOrigin]:
    return external_call["xert_build_blas", Pointer[UInt8, MutUntrackedOrigin],
        Pointer[Float32, MutUntrackedOrigin], Int64, Pointer[Int64, MutUntrackedOrigin]](tri_verts, n_tris, out_size)

def xert_build_blas_geoms(n_geoms: Int32, verts: Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin],
                          n_verts: Pointer[Int64, MutUntrackedOrigin], tris: Pointer[Pointer[UInt32, MutUntrackedOrigin], MutUntrackedOrigin],
                          n_tris: Pointer[Int64, MutUntrackedOrigin], n_procs: Int64, proc_bounds: Pointer[Float32, MutUntrackedOrigin],
                          out_size: Pointer[Int64, MutUntrackedOrigin], bounds: Pointer[Float32, MutUntrackedOrigin]) -> Pointer[UInt8, MutUntrackedOrigin]:
    return external_call["xert_build_blas_geoms", Pointer[UInt8, MutUntrackedOrigin], Int32,
        Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin], Pointer[Int64, MutUntrackedOrigin],
        Pointer[Pointer[UInt32, MutUntrackedOrigin], MutUntrackedOrigin], Pointer[Int64, MutUntrackedOrigin], Int64,
        Pointer[Float32, MutUntrackedOrigin], Pointer[Int64, MutUntrackedOrigin], Pointer[Float32, MutUntrackedOrigin]](
        n_geoms, verts, n_verts, tris, n_tris, n_procs, proc_bounds, out_size, bounds)

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

def xert_set_curves(curves: UInt64, leaves: UInt64, leaf_base: UInt64, inst_blas: UInt64, materials: UInt64):
    external_call["xert_set_curves", NoneType, UInt64, UInt64, UInt64, UInt64, UInt64](curves, leaves, leaf_base, inst_blas, materials)

def xert_curve_table(which: Int32) -> UInt64:
    return external_call["xert_curve_table", UInt64, Int32](which)

def xert_set_heat(counts: UInt64, n: Int64):
    external_call["xert_set_heat", NoneType, UInt64, Int64](counts, n)

# Per-instance hit counts: GONZALES_XERT_HEAT, else next to the scene cache ("" = none).
def xert_heat_path() -> String:
    var path = getenv("GONZALES_XERT_HEAT")
    if path == "" and getenv("GONZALES_SCENE_CACHE") != "":
        path = getenv("GONZALES_SCENE_CACHE") + ".heat"
    return path

# Writes the profiled hit counts (one UInt32 per instance) to xert_heat_path().
def xert_heat_save(ctx: DeviceContext) raises:
    var addr = Int(external_call["xert_heat", UInt64]())
    var n = Int(external_call["xert_heat_n", Int64]())
    if addr == 0:
        return
    var counts = List[UInt8](length=4 * n, fill=0)
    ctx.enqueue_copy(counts.unsafe_ptr(), DeviceBuffer[DType.uint8](ctx, MutPointer[UInt8, MutAnyOrigin](unsafe_from_address=addr), 4 * n, owning=False))
    ctx.synchronize()
    with open(xert_heat_path(), "w") as f:
        f.write_bytes(Span(counts))
    print("RT hardware (Intel): hit counts written to", xert_heat_path())

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
# geometry within the bottom-level structure, inst the instance (its user id) when tracing a top-level structure. A curve hit
# has curve = its leaf in XeCurves.leaves (else -1) and u, v = intersect_curve's h, v.
@fieldwise_init
struct XeHit(TrivialRegisterPassable):
    var t: Float32
    var u: Float32
    var v: Float32
    var prim: Int32
    var geom: Int32
    var inst: Int32
    var curve: Int32

# The tables of xert_set_curves: the procedural primitive p of the structure of instance k is leaf
# leaf_base[inst_blas[k]] + p, a curve PrimId (curve, first piece * 8 + piece count). leaves null: no curves.
@fieldwise_init
struct XeCurves(TrivialRegisterPassable):
    var curves: Pointer[Curve, MutUntrackedOrigin]
    var leaves: Pointer[PrimId, MutUntrackedOrigin]
    var leaf_base: Pointer[Int32, MutUntrackedOrigin]
    var inst_blas: Pointer[Int32, MutUntrackedOrigin]
    var materials: Pointer[Material, MutUntrackedOrigin]

# The closest (or with XERT_ANY_HIT any) hit in [tmin, tmax]. The hardware commits triangles itself; every curve box it
# reports is tested here in the instance's object space and committed when closer.
@always_inline
def xert_trace(accel: MutPointer[UInt8, MutAnyOrigin], ray: XeRayDesc, cv: XeCurves) -> XeHit:
    comptime if is_gpu() and not is_nvidia_gpu() and not is_amd_gpu() and not is_apple_gpu():   # Intel: no is_intel_gpu in released Mojo
        var desc = ray
        var q = external_call["intel_ray_query_init", OpaquePointer[MutAnyOrigin]](
            Pointer(to=desc), accel.unsafe_address_space_cast[AddressSpace.GLOBAL]())
        external_call["intel_ray_query_start_traversal", NoneType](q)
        external_call["intel_ray_query_sync", NoneType](q)
        var curve = Int32(-1)
        var ch_u = Float32(0)
        var ch_v = Float32(0)
        while not external_call["intel_is_traversal_done", Bool](q):
            var level = external_call["intel_get_hit_bvh_level", UInt32](q, UInt32(1))
            var org = SIMD[DType.float32, 4](0)
            var dir = SIMD[DType.float32, 4](0)
            external_call["intel_get_ray_origin", NoneType](Pointer(to=org), q, level)   # struct return: sret first
            external_call["intel_get_ray_direction", NoneType](Pointer(to=dir), q, level)
            var k = Int(external_call["intel_get_hit_instance_user_id", UInt32](q, UInt32(1)))
            var leaf = Int(cv.leaf_base[unsafe_offset=Int(cv.inst_blas[unsafe_offset=k])]) + Int(
                external_call["intel_get_hit_primitive_id", UInt32](q, UInt32(1)))
            var p = cv.leaves[unsafe_offset=leaf]
            var any_hit = (ray.flags & XERT_ANY_HIT) != 0
            if any_hit and _shadow_is_null_material(cv.materials, p.materialIndex):   # as the software curve walk
                external_call["intel_ray_query_start_traversal", NoneType](q)
                external_call["intel_ray_query_sync", NoneType](q)
                continue
            var t_max = min(ray.tmax, external_call["intel_get_hit_distance", Float32](q, UInt32(0)))
            var dlen = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
            var unit = Vec3f(dir[0], dir[1], dir[2]) * (Float32(1) / dlen)
            var h = intersect_curve(Vec3f(org[0], org[1], org[2]), unit, curve_at(cv.curves, Int(p.id1)), Int(p.id2) // 8,
                                    Int(p.id2) % 8, t_max * dlen)
            var t = h[1] / dlen
            if h[0] and t >= ray.tmin and t < t_max:
                var uv = SIMD[DType.float32, 2](h[2], h[3])
                external_call["intel_ray_query_commit_potential_hit_override", NoneType](q, t, Pointer(to=uv))   # struct argument: by pointer
                curve = Int32(leaf); ch_u = h[2]; ch_v = h[3]
                if any_hit:
                    break
            external_call["intel_ray_query_start_traversal", NoneType](q)
            external_call["intel_ray_query_sync", NoneType](q)
        var hit = XeHit(-1, 0, 0, -1, -1, -1, -1)
        if external_call["intel_has_committed_hit", Bool](q):
            hit.t = external_call["intel_get_hit_distance", Float32](q, UInt32(0))
            hit.prim = Int32(external_call["intel_get_hit_primitive_id", UInt32](q, UInt32(0)))
            hit.geom = Int32(external_call["intel_get_hit_geometry_id", UInt32](q, UInt32(0)))
            hit.inst = Int32(external_call["intel_get_hit_instance_user_id", UInt32](q, UInt32(0)))
            if external_call["intel_get_hit_candidate", UInt32](q, UInt32(0)) == 1:   # intel_candidate_type_procedural
                hit.curve = curve; hit.u = ch_u; hit.v = ch_v
            else:
                var uv = SIMD[DType.float32, 2](0)
                external_call["intel_get_hit_barycentrics", NoneType](Pointer(to=uv), q, UInt32(0))   # struct return: sret first
                hit.u = uv[0]
                hit.v = uv[1]
        external_call["intel_ray_query_abandon", NoneType](q)
        return hit
    else:
        return XeHit(-1, 0, 0, -1, -1, -1, -1)

# The rtcore_trace layout: rays are 8 floats (ox oy oz t_min dx dy dz t_max), results 8 words (t u v hit word, 4 unused)
# with the hit word XERT_MISS or the primitive id.
comptime XERT_MISS = UInt32(0xFFFFFFFF)

def xert_trace_rays_gpu(accel: MutPointer[UInt8, MutAnyOrigin], rays: MutPointer[Float32, MutAnyOrigin],
                        results: MutPointer[UInt32, MutAnyOrigin], n: Int64, flags: UInt32):
    var i = global_idx.x
    if i >= Int(n):
        return
    var r = rays.unsafe_load[width=8](8 * i)
    var none = XeCurves(Pointer[Curve, MutUntrackedOrigin].unsafe_dangling(), Pointer[PrimId, MutUntrackedOrigin].unsafe_dangling(),
                        Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(), Pointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
                        Pointer[Material, MutUntrackedOrigin].unsafe_dangling())
    var hit = xert_trace(accel, XeRayDesc(r[0], r[1], r[2], r[4], r[5], r[6], r[3], r[7], 0xFF, flags), none)
    var o = results.unsafe_offset(8 * i)
    o[unsafe_offset=0] = bitcast[DType.uint32](hit.t)
    o[unsafe_offset=1] = bitcast[DType.uint32](hit.u)
    o[unsafe_offset=2] = bitcast[DType.uint32](hit.v)
    o[unsafe_offset=3] = XERT_MISS if hit.prim < 0 else UInt32(hit.prim)

# The interop layout of vulkaninterop_unpack_results_kernel: t u v, -1, mesh (or n_meshes + instance), triangle, hit flag,
# mesh within the instance's template. Instances from ord_inst on hold the ordinary meshes, instance ord_inst + j mesh
# ord_mesh[j]. A
# curve hit is written as rtcore_curve_clip_gpu writes one: t h v, instance (-1: none), curve, pieces, flag 4 (2: not
# instanced), material.
def xert_trace_interop_gpu(accel: MutPointer[UInt8, MutAnyOrigin], rays: MutPointer[Float32, MutAnyOrigin],
                           results: MutPointer[UInt32, MutAnyOrigin], n: Int64, flags: UInt32, n_meshes: Int64,
                           ord_inst: Int64, ord_mesh: MutPointer[Int32, MutAnyOrigin], curves: Pointer[Curve, MutUntrackedOrigin],
                           leaves: Pointer[PrimId, MutUntrackedOrigin], leaf_base: Pointer[Int32, MutUntrackedOrigin],
                           inst_blas: Pointer[Int32, MutUntrackedOrigin], materials: Pointer[Material, MutUntrackedOrigin],
                           heat: MutPointer[UInt32, MutAnyOrigin]):
    var i = global_idx.x
    if i >= Int(n):
        return
    var r = rays.unsafe_load[width=8](8 * i)
    var hit = xert_trace(accel, XeRayDesc(r[0], r[1], r[2], r[4], r[5], r[6], r[3], r[7], 0xFF, flags),
                         XeCurves(curves, leaves, leaf_base, inst_blas, materials))
    var o = results.unsafe_offset(8 * i)
    if hit.prim < 0:
        o[unsafe_offset=6] = 0
        return
    if Int(heat) != 0:
        _ = Atomic.fetch_add(heat.unsafe_offset(Int(hit.inst)), UInt32(1))
    var ordinary = ord_inst >= 0 and Int(hit.inst) >= Int(ord_inst)
    if hit.curve >= 0:
        var p = leaves[unsafe_offset=Int(hit.curve)]
        o[unsafe_offset=0] = bitcast[DType.uint32](hit.t)
        o[unsafe_offset=1] = bitcast[DType.uint32](hit.u)
        o[unsafe_offset=2] = bitcast[DType.uint32](hit.v)
        o[unsafe_offset=3] = UInt32(0xFFFFFFFF) if ordinary else UInt32(hit.inst)
        o[unsafe_offset=4] = UInt32(Int(p.id1))
        o[unsafe_offset=5] = UInt32(Int(p.id2))
        o[unsafe_offset=6] = 2 if ordinary else 4
        o[unsafe_offset=7] = UInt32(Int(p.materialIndex))
        return
    o[unsafe_offset=0] = bitcast[DType.uint32](hit.t)
    o[unsafe_offset=1] = bitcast[DType.uint32](hit.u)
    o[unsafe_offset=2] = bitcast[DType.uint32](hit.v)
    o[unsafe_offset=3] = UInt32(0xFFFFFFFF)
    o[unsafe_offset=4] = UInt32(ord_mesh[unsafe_offset=Int(hit.inst) - Int(ord_inst)]) if ordinary else UInt32(Int(n_meshes) + Int(hit.inst))
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
        Pointer[Curve, MutUntrackedOrigin](unsafe_from_address=Int(xert_curve_table(0))),
        Pointer[PrimId, MutUntrackedOrigin](unsafe_from_address=Int(xert_curve_table(1))),
        Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(xert_curve_table(2))),
        Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(xert_curve_table(3))),
        Pointer[Material, MutUntrackedOrigin](unsafe_from_address=Int(xert_curve_table(4))),
        MutPointer[UInt32, MutAnyOrigin](unsafe_from_address=Int(external_call["xert_heat", UInt64]())),
        grid_dim=ceildiv(m, 256), block_dim=256)
