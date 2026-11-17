# Prototype: Moana's lava rock (Catmull-Clark cage + Ptex displacement) ray cast without tessellation, against a
# tessellated reference. Writes PPM images and prints timings, traversal steps, cracks and memory.
from std.ffi import external_call
from std.math import pow, max, min, sqrt, abs, tan, acos, sin, cos
from std.memory.alloc import unsafe_alloc
from std.memory import unsafe_memcpy
from std.os import getenv
from std.sys.info import size_of
from std.time import perf_counter_ns
from max.algorithm import parallelize
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import ceildiv
from gonzales.core.geometry import Vec3f, cross, dot
from gonzales.scene.shapes.patches import BezierPatch
from gonzales.scene.shapes.displaced_patches import DispSurface, patch_box, sub_box, sub_count, patch_at, intersect_displaced, displaced_normal, displaced_point, displaced_point_n, chain_color
from gonzales.scene.accel.xert import XeRayDesc, xert_alignment, xert_build_blas_geoms, xert_free
from std.memory import OpaquePointer
from std.sys.info import is_gpu
from max.gpu.memory import AddressSpace
from gonzales.app.viewer import CameraState, ViewerHandle, viewer_create, viewer_destroy, viewer_poll_events, viewer_should_close, \
    viewer_get_camera_state, viewer_set_camera_state, viewer_update_framebuffer, viewer_take_key
from gonzales.core.geometry import Point3f
from gonzales.scene.shapes.patches import patch_eval

comptime PTX = "/mnt/storage/moana/base/ex/island/textures/isLavaRocks/Displacement/rockfacebg0001_geo.ptx"


@fieldwise_init
struct BNode(TrivialRegisterPassable):
    var lo: Vec3f
    var hi: Vec3f
    var a: Int32       # interior: left child; leaf: first primitive in the order
    var b: Int32       # interior: right child; leaf: -count


def _build(mut nodes: List[BNode], mut order: List[Int32], lo: List[Vec3f], hi: List[Vec3f], first: Int, n: Int, leaf_n: Int = 4) -> Int:
    """A midpoint-split BVH over boxes; returns the node index."""
    var blo = Vec3f(1e30, 1e30, 1e30); var bhi = Vec3f(-1e30, -1e30, -1e30)
    var clo = Vec3f(1e30, 1e30, 1e30); var chi = Vec3f(-1e30, -1e30, -1e30)
    for i in range(first, first + n):
        var k = Int(order[i])
        blo = Vec3f(min(blo.x, lo[k].x), min(blo.y, lo[k].y), min(blo.z, lo[k].z))
        bhi = Vec3f(max(bhi.x, hi[k].x), max(bhi.y, hi[k].y), max(bhi.z, hi[k].z))
        var c = (lo[k] + hi[k]) * Float32(0.5)
        clo = Vec3f(min(clo.x, c.x), min(clo.y, c.y), min(clo.z, c.z))
        chi = Vec3f(max(chi.x, c.x), max(chi.y, c.y), max(chi.z, c.z))
    var me = len(nodes)
    nodes.append(BNode(blo, bhi, Int32(first), Int32(-n)))
    if n <= leaf_n:
        return me
    var e = chi - clo
    var axis = 0 if e.x >= e.y and e.x >= e.z else (1 if e.y >= e.z else 2)
    var mid = (clo.x + chi.x) * Float32(0.5) if axis == 0 else ((clo.y + chi.y) * Float32(0.5) if axis == 1 else (clo.z + chi.z) * Float32(0.5))
    var i = first; var j = first + n - 1
    while i <= j:
        var k = Int(order[i])
        var c = (lo[k] + hi[k]) * Float32(0.5)
        var cv = c.x if axis == 0 else (c.y if axis == 1 else c.z)
        if cv < mid:
            i += 1
        else:
            var t = order[i]; order[i] = order[j]; order[j] = t
            j -= 1
    var nl = i - first
    if nl == 0 or nl == n:
        nl = n // 2
    var l = _build(nodes, order, lo, hi, first, nl, leaf_n)
    var r = _build(nodes, order, lo, hi, first + nl, n - nl, leaf_n)
    nodes[me] = BNode(blo, bhi, Int32(l), Int32(r))
    return me


@always_inline
def _box(lo: Vec3f, hi: Vec3f, org: Vec3f, inv: Vec3f, t_max: Float32) -> Bool:
    var ax = (lo.x - org.x) * inv.x; var bx = (hi.x - org.x) * inv.x
    var ay = (lo.y - org.y) * inv.y; var by = (hi.y - org.y) * inv.y
    var az = (lo.z - org.z) * inv.z; var bz = (hi.z - org.z) * inv.z
    var t0 = max(max(min(ax, bx), min(ay, by)), max(min(az, bz), Float32(0)))
    var t1 = min(min(max(ax, bx), max(ay, by)), min(max(az, bz), t_max))
    return t0 <= t1


@fieldwise_init
struct Hit(TrivialRegisterPassable):
    var t: Float32
    var prim: Int
    var s: Float32
    var u: Float32
    var steps: Int
    var size: Float32


@always_inline
def _box_t(lo: Vec3f, hi: Vec3f, org: Vec3f, inv: Vec3f, t_max: Float32) -> Tuple[Bool, Float32]:
    var ax = (lo.x - org.x) * inv.x; var bx = (hi.x - org.x) * inv.x
    var ay = (lo.y - org.y) * inv.y; var by = (hi.y - org.y) * inv.y
    var az = (lo.z - org.z) * inv.z; var bz = (hi.z - org.z) * inv.z
    var t0 = max(max(min(ax, bx), min(ay, by)), max(min(az, bz), Float32(0)))
    var t1 = min(min(max(ax, bx), max(ay, by)), min(max(az, bz), t_max))
    return (t0 <= t1, t0)


def trace_patches(d: DispSurface, nodes: Pointer[BNode, MutUntrackedOrigin], order: Pointer[Int32, MutUntrackedOrigin], org: Vec3f, dir: Vec3f,
                  boxes_only: Bool = False) -> Hit:
    """Front to back over the sub-square BVH; the hit names the patch."""
    var inv = Vec3f(Float32(1) / dir.x, Float32(1) / dir.y, Float32(1) / dir.z)
    var h = Hit(Float32(1e30), -1, Float32(0), Float32(0), 0, Float32(0))
    var stack = Array[Int32, 64](fill=Int32(0))
    var sp = 1
    while sp > 0:
        sp -= 1
        var nd = nodes[unsafe_offset=Int(stack[sp])]
        if not _box(nd.lo, nd.hi, org, inv, h.t):
            continue
        if nd.b < 0:
            for k in range(Int(nd.a), Int(nd.a) - Int(nd.b)):
                var prim = Int(order[unsafe_offset=k])
                if boxes_only:
                    var bt = _box_t(nd.lo, nd.hi, org, inv, h.t)
                    if bt[0] and bt[1] < h.t:
                        h.t = bt[1]; h.prim = prim // sub_count(d); h.steps += 1
                    continue
                var r = intersect_displaced(d, prim, org, dir, h.t)
                h.steps += r[4]
                if r[0] and r[1] < h.t:
                    h.t = r[1]; h.prim = prim // sub_count(d); h.s = r[2]; h.u = r[3]; h.size = r[5]
        elif sp < 62:
            var na = nodes[unsafe_offset=Int(nd.a)]; var nb = nodes[unsafe_offset=Int(nd.b)]
            var ta = _box_t(na.lo, na.hi, org, inv, h.t); var tb = _box_t(nb.lo, nb.hi, org, inv, h.t)
            var near = nd.a; var far = nd.b; var tn = ta; var tf = tb
            if tb[0] and (not ta[0] or tb[1] < ta[1]):
                near = nd.b; far = nd.a; tn = tb; tf = ta
            if tf[0]:
                stack[sp] = far; sp += 1
            if tn[0]:
                stack[sp] = near; sp += 1
    return h


def _sub_bvh(d: DispSurface, np: Int, mut nodes: List[BNode], mut order: List[Int32]):
    """The BVH over every patch's sub-squares for the surface's height scale (boxes in parallel)."""
    var nsub = sub_count(d)
    var n = np * nsub
    var lo = List[Vec3f](length=n, fill=Vec3f(0, 0, 0)); var hi = List[Vec3f](length=n, fill=Vec3f(0, 0, 0))
    var lo_p = lo.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](); var hi_p = hi.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    def boxes(p: Int) {imm}:
        for k in range(nsub):
            var b = sub_box(d, p * nsub + k)
            lo_p[unsafe_offset=p * nsub + k] = b[0]; hi_p[unsafe_offset=p * nsub + k] = b[1]
    parallelize(boxes, np)
    nodes.clear(); order.clear()
    for i in range(n):
        order.append(Int32(i))
    _ = _build(nodes, order, lo, hi, 0, n, 1)   # one sub-square a leaf: the front-to-back order reaches each
    _ = len(lo); _ = len(hi)


@fieldwise_init
struct GCam(TrivialRegisterPassable, DevicePassable):
    """The camera and light of a GPU render."""
    var eye: Vec3f
    var fwd: Vec3f
    var right: Vec3f
    var up: Vec3f
    var light: Vec3f
    var tan_h: Float32
    var w: Int32
    var h: Int32
    var tile: Int32   # 1 (TILE=0: 0): threads walk 4 x 4 pixel tiles, a subgroup a tile
    var prof: Int32   # PROF bits: 1 no shading, 2 the sub-square boxes only (their entry is the hit)
    comptime device_type: AnyType = Self

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "GCam"


@always_inline
def _pix(i: Int, W: Int, tile: Int32) -> Tuple[Int, Int]:
    if tile == 0:
        return (i % W, i // W)
    var t = i // 16; var k = i % 16; var tw = W // 4
    return ((t % tw) * 4 + k % 4, (t // tw) * 4 + k // 4)


def render_gpu(d: DispSurface, nodes: Pointer[BNode, MutUntrackedOrigin], order: Pointer[Int32, MutUntrackedOrigin],
               img: Pointer[Float32, MutUntrackedOrigin], steps: Pointer[Int32, MutUntrackedOrigin], cam: GCam):
    """One pixel a thread: the same traversal and shading as the CPU rows."""
    var i = global_idx.x
    var W = Int(cam.w); var H = Int(cam.h)
    if i >= W * H:
        return
    var xy = _pix(i, W, cam.tile); var x = xy[0]; var y = xy[1]
    var px = (Float32(2) * (Float32(x) + Float32(0.5)) / Float32(W) - Float32(1)) * cam.tan_h
    var py = (Float32(1) - Float32(2) * (Float32(y) + Float32(0.5)) / Float32(H)) * cam.tan_h * Float32(H) / Float32(W)
    var dir = cam.fwd + cam.right * px + cam.up * py
    dir = dir * (Float32(1) / sqrt(dot(dir, dir)))
    var hh = trace_patches(d, nodes, order, cam.eye, dir, cam.prof & 2 != 0)
    steps[unsafe_offset=y * W + x] = Int32(hh.steps)
    var shade = Float32(0)
    if hh.prim >= 0 and cam.prof != 0:
        shade = Float32(0.5)
    elif hh.prim >= 0:
        var n = displaced_normal(d, hh.prim, hh.s, hh.u, max(hh.size, Float32(1.0 / 512.0)))
        if dot(n, dir) > Float32(0):
            n = -n
        shade = Float32(0.08) + Float32(0.75) * max(dot(n, cam.light), Float32(0))
    img[unsafe_offset=y * W + x] = shade


@always_inline
def _is_real(p: ViewerHandle) -> Bool:
    return Int(p) > 16


def _up[T: AnyType](ctx: DeviceContext, mut bufs: List[DeviceBuffer[DType.uint8]], src: Pointer[T, MutUntrackedOrigin], n: Int) raises -> Pointer[T, MutUntrackedOrigin]:
    """n elements to the device, kept alive in bufs; the device pointer."""
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(n, 1) * size_of[T]())
    if n > 0:
        ctx.enqueue_copy(buf, src.unsafe_bitcast[UInt8]())
    var p = buf.unsafe_ptr().unsafe_bitcast[T]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    bufs.append(buf^)
    return p


def trace_tris(pts: Pointer[Vec3f, MutUntrackedOrigin], tri: Pointer[Int32, MutUntrackedOrigin], nodes: Pointer[BNode, MutUntrackedOrigin],
               order: Pointer[Int32, MutUntrackedOrigin], org: Vec3f, dir: Vec3f) -> Hit:
    var inv = Vec3f(Float32(1) / dir.x, Float32(1) / dir.y, Float32(1) / dir.z)
    var h = Hit(Float32(1e30), -1, Float32(0), Float32(0), 0, Float32(0))
    var stack = Array[Int32, 512](fill=Int32(0))
    var sp = 1
    while sp > 0:
        sp -= 1
        var nd = nodes[unsafe_offset=Int(stack[sp])]
        if not _box(nd.lo, nd.hi, org, inv, h.t):
            continue
        if nd.b < 0:
            for k in range(Int(nd.a), Int(nd.a) - Int(nd.b)):
                var ti = Int(order[unsafe_offset=k])
                var p0 = pts[unsafe_offset=Int(tri[unsafe_offset=3 * ti])]
                var e1 = pts[unsafe_offset=Int(tri[unsafe_offset=3 * ti + 1])] - p0
                var e2 = pts[unsafe_offset=Int(tri[unsafe_offset=3 * ti + 2])] - p0
                var pv = cross(dir, e2)
                var det = dot(e1, pv)
                if abs(det) < Float32(1e-12):
                    continue
                var idet = Float32(1) / det
                var tv = org - p0
                var bu = dot(tv, pv) * idet
                if bu < Float32(0) or bu > Float32(1):
                    continue
                var qv = cross(tv, e1)
                var bv = dot(dir, qv) * idet
                if bv < Float32(0) or bu + bv > Float32(1):
                    continue
                var t = dot(e2, qv) * idet
                if t > Float32(1e-4) and t < h.t:
                    h.t = t; h.prim = ti
        elif sp < 510:
            stack[sp] = nd.a; stack[sp + 1] = nd.b
            sp += 2
    return h


@always_inline
def _rq_trace(accel: MutPointer[UInt8, MutAnyOrigin], d: DispSurface, org: Vec3f, dir: Vec3f, procedural: Bool) -> Hit:
    """Closest hit through Intel's ray queries: the hardware commits triangles; each sub-square it reports runs the
    subdivision loop, whose hit is committed (prim: the triangle, or the patch with its s, u and leaf size)."""
    var h = Hit(Float32(1e30), -1, Float32(0), Float32(0), 0, Float32(0))
    comptime if is_gpu():
        var desc = XeRayDesc(org.x, org.y, org.z, dir.x, dir.y, dir.z, Float32(0), Float32(1e30), UInt32(0xFF), UInt32(0))
        var q = external_call["intel_ray_query_init", OpaquePointer[MutAnyOrigin]](
            Pointer(to=desc), accel.unsafe_address_space_cast[AddressSpace.GLOBAL]())
        external_call["intel_ray_query_start_traversal", NoneType](q)
        external_call["intel_ray_query_sync", NoneType](q)
        while not external_call["intel_is_traversal_done", Bool](q):
            var prim = Int(external_call["intel_get_hit_primitive_id", UInt32](q, UInt32(1)))
            var r = intersect_displaced(d, prim, org, dir, h.t)
            h.steps += r[4]
            if r[0] and r[1] < h.t:
                h.t = r[1]; h.prim = prim // sub_count(d); h.s = r[2]; h.u = r[3]; h.size = r[5]
                var uv = SIMD[DType.float32, 2](r[2], r[3])
                external_call["intel_ray_query_commit_potential_hit_override", NoneType](q, r[1], Pointer(to=uv))
            external_call["intel_ray_query_start_traversal", NoneType](q)
            external_call["intel_ray_query_sync", NoneType](q)
        if not procedural and external_call["intel_has_committed_hit", Bool](q):
            h.t = external_call["intel_get_hit_distance", Float32](q, UInt32(0))
            h.prim = Int(external_call["intel_get_hit_primitive_id", UInt32](q, UInt32(0)))
            var uv = SIMD[DType.float32, 2](0)
            external_call["intel_get_hit_barycentrics", NoneType](Pointer(to=uv), q, UInt32(0))   # struct return: sret first
            h.s = uv[0]; h.u = uv[1]
        external_call["intel_ray_query_abandon", NoneType](q)
    return h


def render_hw(accel: MutPointer[UInt8, MutAnyOrigin], d: DispSurface, tri_base: Pointer[Int32, MutUntrackedOrigin],
              grid_n: Pointer[Int32, MutUntrackedOrigin], ext: Pointer[Float32, MutUntrackedOrigin], np: Int32, procedural: Int32,
              img: Pointer[Float32, MutUntrackedOrigin], steps: Pointer[Int32, MutUntrackedOrigin], cam: GCam):
    """render_gpu through the hardware: a hit triangle maps back to its patch and (s, t), shaded as ours."""
    var i = global_idx.x
    var W = Int(cam.w); var H = Int(cam.h)
    if i >= W * H:
        return
    var xy = _pix(i, W, cam.tile); var x = xy[0]; var y = xy[1]
    var px = (Float32(2) * (Float32(x) + Float32(0.5)) / Float32(W) - Float32(1)) * cam.tan_h
    var py = (Float32(1) - Float32(2) * (Float32(y) + Float32(0.5)) / Float32(H)) * cam.tan_h * Float32(H) / Float32(W)
    var dir = cam.fwd + cam.right * px + cam.up * py
    dir = dir * (Float32(1) / sqrt(dot(dir, dir)))
    var hh = _rq_trace(accel, d, cam.eye, dir, procedural != 0)
    steps[unsafe_offset=y * W + x] = Int32(hh.steps)
    var shade = Float32(0)
    if hh.prim >= 0:
        var p = hh.prim; var s = hh.s; var t = hh.u; var size = hh.size
        if procedural == 0:
            var lo = 0; var hi = Int(np)
            while hi - lo > 1:
                var mid = (lo + hi) // 2
                if Int(tri_base[unsafe_offset=mid]) <= hh.prim:
                    lo = mid
                else:
                    hi = mid
            p = lo
            var n = Int(grid_n[unsafe_offset=p]); var loc = hh.prim - Int(tri_base[unsafe_offset=p]); var qd = loc // 2
            var gi = Float32(qd % n); var gj = Float32(qd // n); var inv_n = Float32(1) / Float32(n)
            if loc % 2 == 0:   # (i, j) (i+1, j) (i+1, j+1)
                s = (gi + hh.s + hh.u) * inv_n; t = (gj + hh.u) * inv_n
            else:              # (i, j) (i+1, j+1) (i, j+1)
                s = (gi + hh.s) * inv_n; t = (gj + hh.s + hh.u) * inv_n
            size = d.pixel * hh.t / ext[unsafe_offset=p]   # the footprint in patch units
        var n = displaced_normal(d, p, s, t, max(size, Float32(1.0 / 512.0)))
        if dot(n, dir) > Float32(0):
            n = -n
        shade = Float32(0.08) + Float32(0.75) * max(dot(n, cam.light), Float32(0))
    img[unsafe_offset=y * W + x] = shade


def _hw_render(ctx: DeviceContext, mut bufs: List[DeviceBuffer[DType.uint8]], d: DispSurface, dd: DispSurface, np: Int, name: String,
               gc: GCam, mut img: List[Float32], mut hitmask: List[UInt8], mut steps: List[Int], mut maxs: List[Int]) raises:
    """Intel hardware ray tracing. REF_HW=8/16/32/64/texel: the displaced surface tessellated n x n a patch (or a quad a
    texel), heights from the level matching n; else our sub-squares as procedural primitives (HW=1)."""
    var rmode = getenv("REF_HW"); var procedural = rmode.byte_length() == 0
    var W = Int(gc.w); var H = Int(gc.h)
    var tb = unsafe_alloc[Int32](np + 1); var gn = unsafe_alloc[Int32](np); var ext = unsafe_alloc[Float32](np)
    for p in range(np):
        var bp = d.patches[unsafe_offset=p]
        ext[unsafe_offset=p] = max(max(bp.x.reduce_max() - bp.x.reduce_min(), bp.y.reduce_max() - bp.y.reduce_min()), bp.z.reduce_max() - bp.z.reduce_min())
    var size = Int64(0); var bounds = unsafe_alloc[Float32](6)
    var rtas: Pointer[UInt8, MutUntrackedOrigin]
    var t0 = perf_counter_ns(); var t1 = t0; var in_bytes = 0; var nprim = 0
    if not procedural:
        var nt = 0; var nv = 0
        var vo = unsafe_alloc[Int](np)
        for p in range(np):
            var n: Int
            if rmode == "texel":
                var f = Int(d.face[unsafe_offset=p])
                var res = 1 << max(Int(d.hdr[unsafe_offset=3 * f + 1]), Int(d.hdr[unsafe_offset=3 * f + 2]))
                n = max(Int(d.sq[unsafe_offset=3 * p + 2] * Float32(res) + Float32(0.5)), 1)
            else:
                n = atol(rmode)
            gn[unsafe_offset=p] = Int32(n); tb[unsafe_offset=p] = Int32(nt); vo[unsafe_offset=p] = nv
            nt += 2 * n * n; nv += (n + 1) * (n + 1)
        tb[unsafe_offset=np] = Int32(nt)
        in_bytes = 12 * nv + 12 * nt; nprim = nt
        print(name + ": tessellation", rmode, ":", nt, "triangles,", nv, "vertices,", in_bytes // (1024 * 1024), "MiB vertices + indices")
        if nt >= (1 << 31) - 1 or in_bytes > (24 << 30):
            print(name + ": too large to build here")
            return
        var vb = unsafe_alloc[Float32](3 * nv); var ib = unsafe_alloc[UInt32](3 * nt)
        def fill(p: Int) {imm}:
            var n = Int(gn[unsafe_offset=p]); var v0 = vo[unsafe_offset=p]; var k0 = Int(tb[unsafe_offset=p])
            for j in range(n + 1):
                for i in range(n + 1):
                    var q = displaced_point_n(d, p, Float32(i) / Float32(n), Float32(j) / Float32(n), n)
                    var o = 3 * (v0 + j * (n + 1) + i)
                    vb[unsafe_offset=o] = q.x; vb[unsafe_offset=o + 1] = q.y; vb[unsafe_offset=o + 2] = q.z
            for j in range(n):
                for i in range(n):
                    var a = UInt32(v0 + j * (n + 1) + i); var k = 3 * (k0 + 2 * (j * n + i))
                    ib[unsafe_offset=k] = a; ib[unsafe_offset=k + 1] = a + 1; ib[unsafe_offset=k + 2] = a + UInt32(n + 2)
                    ib[unsafe_offset=k + 3] = a; ib[unsafe_offset=k + 4] = a + UInt32(n + 2); ib[unsafe_offset=k + 5] = a + UInt32(n + 1)
        parallelize(fill, np)
        var vp = unsafe_alloc[Pointer[Float32, MutUntrackedOrigin]](1); vp[unsafe_offset=0] = vb
        var ip = unsafe_alloc[Pointer[UInt32, MutUntrackedOrigin]](1); ip[unsafe_offset=0] = ib
        var nvp = unsafe_alloc[Int64](1); nvp[unsafe_offset=0] = Int64(nv)
        var ntp = unsafe_alloc[Int64](1); ntp[unsafe_offset=0] = Int64(nt)
        t0 = perf_counter_ns()
        rtas = xert_build_blas_geoms(1, vp, nvp, ip, ntp, 0, Pointer[Float32, MutUntrackedOrigin].unsafe_dangling(),
                                     Pointer(to=size).unsafe_origin_cast[MutUntrackedOrigin](), bounds)
        t1 = perf_counter_ns()
        vb.unsafe_free(); ib.unsafe_free(); vp.unsafe_free(); ip.unsafe_free(); nvp.unsafe_free(); ntp.unsafe_free(); vo.unsafe_free()
    else:
        var npr = np * sub_count(d); nprim = npr
        var pb = unsafe_alloc[Float32](6 * npr)
        def boxes(p: Int) {imm}:
            for k in range(sub_count(d)):
                var b = sub_box(d, p * sub_count(d) + k); var o = 6 * (p * sub_count(d) + k)
                pb[unsafe_offset=o] = b[0].x; pb[unsafe_offset=o + 1] = b[0].y; pb[unsafe_offset=o + 2] = b[0].z
                pb[unsafe_offset=o + 3] = b[1].x; pb[unsafe_offset=o + 4] = b[1].y; pb[unsafe_offset=o + 5] = b[1].z
        parallelize(boxes, np)
        t0 = perf_counter_ns()
        rtas = xert_build_blas_geoms(0, Pointer[Pointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
                                     Pointer[Int64, MutUntrackedOrigin].unsafe_dangling(), Pointer[Pointer[UInt32, MutUntrackedOrigin], MutUntrackedOrigin].unsafe_dangling(),
                                     Pointer[Int64, MutUntrackedOrigin].unsafe_dangling(), Int64(npr), pb,
                                     Pointer(to=size).unsafe_origin_cast[MutUntrackedOrigin](), bounds)
        t1 = perf_counter_ns()
        pb.unsafe_free()
    if Int(rtas) == 0:
        raise Error("RTAS build failed")
    var accel = _up(ctx, bufs, rtas, Int(size))
    xert_free(rtas)
    if Int(accel) % Int(xert_alignment()) != 0:
        raise Error("RTAS buffer not aligned")
    print(name + ":", "procedural sub-squares" if procedural else "triangles", nprim, ", RTAS", Int(size) // 1024, "KiB, build",
          (t1 - t0) // 1000000, "ms")
    var tb_d = _up(ctx, bufs, tb, np + 1); var gn_d = _up(ctx, bufs, gn, np); var ext_d = _up(ctx, bufs, ext, np)
    var img_b = ctx.enqueue_create_buffer[DType.float32](W * H)
    var st_b = ctx.enqueue_create_buffer[DType.int32](W * H)
    var img_d = img_b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var st_d = st_b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var acc_d = accel.unsafe_origin_cast[MutAnyOrigin]()
    var pr = Int32(1 if procedural else 0)
    ctx.synchronize()
    for hgt in [540, H]:   # the fastest of three launches after one with the compile
        var g = GCam(gc.eye, gc.fwd, gc.right, gc.up, gc.light, gc.tan_h, gc.w, Int32(hgt), gc.tile, gc.prof)
        var best = 1 << 62
        for _ in range(4):
            var ta = perf_counter_ns()
            ctx.enqueue_function[render_hw](acc_d, dd, tb_d, gn_d, ext_d, Int32(np), pr, img_d, st_d, g, grid_dim=ceildiv(W * hgt, 64), block_dim=64)
            ctx.synchronize()
            best = min(best, Int(perf_counter_ns()) - Int(ta))
        print(name + ": HW", W, "x", hgt, "render", Float32(best) * Float32(1e-6), "ms,", Float32(W * hgt) / (Float32(best) * Float32(1e-9)) / Float32(1e6), "Mrays/s")
    with img_b.map_to_host() as hi:
        for k in range(W * H):
            img[k] = hi[k]
            if hi[k] > Float32(0):
                hitmask[k] = UInt8(1)
    with st_b.map_to_host() as hs:
        for k in range(W * H):
            steps[k // W] += Int(hs[k]); maxs[k // W] = max(maxs[k // W], Int(hs[k]))
    tb.unsafe_free(); gn.unsafe_free(); ext.unsafe_free(); bounds.unsafe_free()


def _grid_levels() -> Int32:
    """GRID=4/8/16: sub-squares a patch side as BVH primitives (default 16)."""
    var g = getenv("GRID")
    return Int32(4) if g == "4" else (Int32(6) if g == "8" else Int32(8))


def _leaf_mode() -> Int32:
    """LEAF=box (footprint-sized boxes), exact (bilinear leaves at a texel or a pixel; default), texel (at a texel only) or crop
    (exact, boxes cropped per step)."""
    var l = getenv("LEAF")
    return Int32(0) if l == "box" else (Int32(2) if l == "texel" else (Int32(4) if l == "crop" else Int32(1)))


def _env_f(name: String, dflt: Float32) raises -> Float32:
    var s = getenv(String(name))
    return Float32(atof(s)) if s.byte_length() > 0 else dflt


def _write_ppm_rgb(path: String, img: List[Float32], w: Int, h: Int) raises:
    var bytes = List[UInt8]()
    var head = String("P6\n") + String(w) + " " + String(h) + "\n255\n"
    for c in head.as_bytes():
        bytes.append(c)
    for i in range(3 * w * h):
        var v = min(max(img[i], Float32(0)), Float32(1))
        bytes.append(UInt8(Int(pow(v, Float32(1.0 / 2.2)) * Float32(255) + Float32(0.5))))
    with open(path, "w") as f:
        f.write_bytes(Span(bytes))


def _write_ppm(path: String, img: List[Float32], w: Int, h: Int) raises:
    var bytes = List[UInt8]()
    var head = String("P6\n") + String(w) + " " + String(h) + "\n255\n"
    for c in head.as_bytes():
        bytes.append(c)
    for i in range(w * h):
        var v = min(max(img[i], Float32(0)), Float32(1))
        var g = UInt8(Int(sqrt(v) * Float32(255) + Float32(0.5)))
        bytes.append(g); bytes.append(g); bytes.append(g)
    with open(path, "w") as f:
        f.write_bytes(Span(bytes))


def _unit_normal(d: DispSurface, pi: Int, s: Float32, t: Float32) -> Vec3f:
    var e = patch_eval(patch_at(d, pi, s, t), s, t)
    var n = cross(e[1], e[2])
    var l = sqrt(dot(n, n))
    return n * (Float32(1) / l) if l > Float32(0) else Vec3f(0, 0, 0)


def _edge_gap(d: DispSurface, np: Int) raises:
    """Angle between the two patches' normals along shared edges (matched by their end points), split by Gregory edges."""
    var ends = Dict[String, Int]()   # key of an edge -> patch * 4 + edge
    var cs: List[Float32] = [0, 1, 1, 0]; var ct: List[Float32] = [0, 0, 1, 1]   # corners (s, t) counter-clockwise
    var worst_g = Float32(0); var worst_r = Float32(0); var sum_g = Float64(0); var sum_r = Float64(0); var n_g = 0; var n_r = 0
    for p in range(np):
        for e in range(4):
            var a = patch_eval(d.patches[unsafe_offset=p], cs[e], ct[e])[0]; var b = patch_eval(d.patches[unsafe_offset=p], cs[(e + 1) % 4], ct[(e + 1) % 4])[0]
            var ka = String(Int(a.x * 1000)) + "," + String(Int(a.y * 1000)) + "," + String(Int(a.z * 1000))
            var kb = String(Int(b.x * 1000)) + "," + String(Int(b.y * 1000)) + "," + String(Int(b.z * 1000))
            var key = ka + "|" + kb if ka < kb else kb + "|" + ka
            if key not in ends:
                ends[key] = p * 4 + e
                continue
            var q = ends[key] // 4; var f = ends[key] % 4
            var gpair = d.greg[unsafe_offset=p] != 0 or d.greg[unsafe_offset=q] != 0
            var qa = patch_eval(d.patches[unsafe_offset=q], cs[f], ct[f])[0]
            var same = dot(qa - a, qa - a) < dot(qa - b, qa - b)   # do the two edges run the same way?
            for k in range(1, 8):
                var x = Float32(k) / Float32(8)
                var ps = cs[e] + (cs[(e + 1) % 4] - cs[e]) * x; var pt = ct[e] + (ct[(e + 1) % 4] - ct[e]) * x
                var y = x if same else Float32(1) - x
                var qs = cs[f] + (cs[(f + 1) % 4] - cs[f]) * y; var qt = ct[f] + (ct[(f + 1) % 4] - ct[f]) * y
                var c = abs(dot(_unit_normal(d, p, ps, pt), _unit_normal(d, q, qs, qt)))
                var ang = Float32(57.29578) * acos(min(c, Float32(1)))
                if gpair:
                    worst_g = max(worst_g, ang); sum_g += Float64(ang); n_g += 1
                else:
                    worst_r = max(worst_r, ang); sum_r += Float64(ang); n_r += 1
    print("normal gap across edges touching Gregory patches: max", worst_g, "deg, mean", sum_g / Float64(max(n_g, 1)), "over", n_g, "samples")
    print("normal gap across regular edges: max", worst_r, "deg, mean", sum_r / Float64(max(n_r, 1)), "over", n_r, "samples")


@fieldwise_init
struct DCam(TrivialRegisterPassable, DevicePassable):
    """The interactive frame: camera, key light, exposure, size and accumulation index."""
    var eye: Vec3f
    var fwd: Vec3f
    var right: Vec3f
    var up: Vec3f
    var light: Vec3f
    var tan_h: Float32
    var exposure: Float32
    var w: Int32
    var h: Int32
    var frame: Int32       # samples already in the accumulation; 0 restarts it
    var headlight: Int32
    var color: Int32
    comptime device_type: AnyType = Self

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "DCam"


@always_inline
def _hash01(a: UInt32) -> Float32:
    var x = a * UInt32(747796405) + UInt32(2891336453)
    x = ((x >> ((x >> 28) + 4)) ^ x) * UInt32(277803737)
    x = (x >> 22) ^ x
    return Float32(Int(x >> 8)) * Float32(1.0 / 16777216.0)


def demo_gpu(d: DispSurface, nodes: Pointer[BNode, MutUntrackedOrigin], order: Pointer[Int32, MutUntrackedOrigin],
             crgb: Pointer[UInt8, MutUntrackedOrigin], coff: Pointer[Int32, MutUntrackedOrigin], accum: Pointer[Float32, MutUntrackedOrigin], cam: DCam):
    """One jittered sample a pixel, averaged into accum (linear RGB)."""
    var i = global_idx.x
    var W = Int(cam.w); var H = Int(cam.h)
    if i >= W * H:
        return
    var x = i % W; var y = i // W
    var seed = UInt32(i) * UInt32(9781) + UInt32(Int(cam.frame)) * UInt32(6271) + UInt32(1)
    var jx = _hash01(seed); var jy = _hash01(seed ^ UInt32(0x9E3779B9))
    var px = (Float32(2) * (Float32(x) + jx) / Float32(W) - Float32(1)) * cam.tan_h
    var py = (Float32(1) - Float32(2) * (Float32(y) + jy) / Float32(H)) * cam.tan_h * Float32(H) / Float32(W)
    var dir = cam.fwd + cam.right * px + cam.up * py
    dir = dir * (Float32(1) / sqrt(dot(dir, dir)))
    var hh = trace_patches(d, nodes, order, cam.eye, dir)
    var c = Vec3f(Float32(0.02), Float32(0.022), Float32(0.03))   # background
    if hh.prim >= 0:
        var step = max(hh.size, Float32(1.0 / 512.0))
        var n = displaced_normal(d, hh.prim, hh.s, hh.u, step)
        if dot(n, dir) > Float32(0):
            n = -n
        var l = -dir if cam.headlight != 0 else cam.light
        var albedo = chain_color(d, crgb, coff, hh.prim, hh.s, hh.u, step) if cam.color != 0 else Vec3f(Float32(0.5), Float32(0.5), Float32(0.5))
        c = albedo * ((Float32(0.12) + max(dot(n, l), Float32(0))) * cam.exposure)
    var k = Float32(Int(cam.frame))
    var a = Vec3f(accum[unsafe_offset=3 * i], accum[unsafe_offset=3 * i + 1], accum[unsafe_offset=3 * i + 2]) if cam.frame > 0 else Vec3f(0, 0, 0)
    a = a + (c - a) * (Float32(1) / (k + Float32(1)))
    accum[unsafe_offset=3 * i] = a.x; accum[unsafe_offset=3 * i + 1] = a.y; accum[unsafe_offset=3 * i + 2] = a.z


def _demo_bvh(d: DispSurface, np: Int, mut nodes: List[BNode], mut order: List[Int32]):
    _sub_bvh(d, np, nodes, order)


def _demo_upload(ctx: DeviceContext, nb_buf: DeviceBuffer[DType.uint8], or_buf: DeviceBuffer[DType.uint8], stage: Pointer[UInt8, MutUntrackedOrigin],
                 cap: Int, nodes: List[BNode], order: List[Int32]) raises:
    var nbytes = len(nodes) * size_of[BNode]()
    if nbytes > cap or len(order) * 4 != len(or_buf):
        raise Error("patch BVH does not match its device buffers")
    for i in range(cap):
        stage[unsafe_offset=i] = UInt8(0)
    unsafe_memcpy(dest=stage, src=nodes.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]().unsafe_bitcast[UInt8](), count=nbytes)
    ctx.enqueue_copy(nb_buf, stage)
    ctx.enqueue_copy(or_buf, order.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]().unsafe_bitcast[UInt8]())
    ctx.synchronize()


def run_demo(patches: Pointer[BezierPatch, MutUntrackedOrigin], sq: Pointer[Float32, MutUntrackedOrigin], face: Pointer[Int32, MutUntrackedOrigin],
             hdr: Pointer[Int32, MutUntrackedOrigin], pyr: Pointer[UInt8, MutUntrackedOrigin], npyr: Int, chain_off: Pointer[Int32, MutUntrackedOrigin],
             chain: Pointer[UInt8, MutUntrackedOrigin], nchain: Int, fm: Pointer[Float32, MutUntrackedOrigin], greg: Pointer[UInt8, MutUntrackedOrigin],
             np: Int, nf: Int, blo: Vec3f, bhi: Vec3f, out_dir: String) raises:
    """INTERACTIVE=1: fly around the rock in the Vulkan viewer; DEMO_FRAMES=N: N frames along an orbit without a window."""
    var W = Int(_env_f("W", Float32(960))); var H = Int(_env_f("H", Float32(540)))
    var frames_env = getenv("DEMO_FRAMES")
    var headless = frames_env.byte_length() > 0
    var exag = _env_f("EXAG", Float32(1))
    var exag_off = exag if exag > Float32(0) else Float32(1)   # what 0 switches back to
    var exposure = _env_f("EXPOSURE", Float32(3))
    var half_move = getenv("HALF") != "0"
    var ctx = DeviceContext()
    var bufs = List[DeviceBuffer[DType.uint8]]()
    # The colour Ptex as bordered RGB chains (same faces as the displacement).
    var col = getenv("COLOR")
    if col.byte_length() == 0:
        col = "/mnt/storage/moana/base/ex/island/textures/isLavaRocks/Color/rockfacebg0001_geo.ptx"
    var cn = unsafe_alloc[UInt8](col.byte_length() + 1)
    for i in range(col.byte_length()):
        cn[unsafe_offset=i] = col.unsafe_ptr()[unsafe_offset=i]
    cn[unsafe_offset=col.byte_length()] = UInt8(0)
    var ncb = unsafe_alloc[Int64](1)
    var ch = external_call["disp_color_chains", Int](cn, ncb)
    var color_on = ch != 0
    var coff_h = unsafe_alloc[Int32](max(nf, 1)); var crgb_h = unsafe_alloc[UInt8](max(Int(ncb[0]), 1) if color_on else 1)
    if color_on:
        external_call["disp_color_chains_take", NoneType](ch, coff_h, crgb_h)
        print("colour chains:", Int(ncb[0]), "bytes")
    else:
        print("no colour:", col)
    var crgb = _up(ctx, bufs, crgb_h, Int(ncb[0]) if color_on else 1)
    var coff = _up(ctx, bufs, coff_h, nf)
    var dpat = _up(ctx, bufs, patches, np); var dsq = _up(ctx, bufs, sq, 3 * np); var dface = _up(ctx, bufs, face, np)
    var dhdr = _up(ctx, bufs, hdr, 3 * nf); var dpyr = _up(ctx, bufs, pyr, npyr); var doff = _up(ctx, bufs, chain_off, nf)
    var dchain = _up(ctx, bufs, chain, nchain); var dfm = _up(ctx, bufs, fm, 12 * np); var dgreg = _up(ctx, bufs, greg, np)
    var host_d = DispSurface(patches, sq, face, hdr, pyr, exag / Float32(255), Vec3f(0, 0, 0), Float32(0), chain_off, chain, fm, greg, _grid_levels(), _leaf_mode())
    var nodes = List[BNode](); var order = List[Int32]()
    _demo_bvh(host_d, np, nodes, order)
    var nb_cap = max(len(nodes), 1) * size_of[BNode]() * 2   # room for a deeper tree
    var nb_buf = ctx.enqueue_create_buffer[DType.uint8](nb_cap)
    var or_buf = ctx.enqueue_create_buffer[DType.uint8](max(len(order), 1) * 4)   # one entry per BVH primitive (sub-square)
    var nb_stage = unsafe_alloc[UInt8](nb_cap)   # the device copy reads the buffer's whole size
    _demo_upload(ctx, nb_buf, or_buf, nb_stage, nb_cap, nodes, order)
    var dnodes = nb_buf.unsafe_ptr().unsafe_bitcast[BNode]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var dorder = or_buf.unsafe_ptr().unsafe_bitcast[Int32]().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var accum_b = ctx.enqueue_create_buffer[DType.float32](3 * W * H)
    var daccum = accum_b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    ctx.synchronize()

    # The tool's default 3/4 view, the key light from the upper side.
    var c = (blo + bhi) * Float32(0.5)
    var ext = bhi - blo
    var diag = sqrt(dot(ext, ext))
    var near = _env_f("CAM_DIST", Float32(1.1))
    var eye = Vec3f(c.x + Float32(0.45) * diag * near, c.y + Float32(0.55) * diag * near, c.z + Float32(0.7) * diag * near)
    var fwd = c - eye
    fwd = fwd * (Float32(1) / sqrt(dot(fwd, fwd)))
    var tan_h = tan(Float32(40) * Float32(3.14159265) / Float32(360))
    var light = Vec3f(Float32(0.45), Float32(1), Float32(0.3))
    light = light * (Float32(1) / sqrt(dot(light, light)))
    var world_up = Vec3f(0, 1, 0)
    var viewer = ViewerHandle.unsafe_dangling()
    var title = String("Moana lava rock: tessellation-free displaced Catmull-Clark (B580)")
    var cs = unsafe_alloc[CameraState](1)
    cs[unsafe_offset=0] = CameraState(Point3f(eye.x, eye.y, eye.z), fwd, world_up, Int32(1))
    if not headless:
        var tb = unsafe_alloc[UInt8](title.byte_length() + 1)
        for i in range(title.byte_length()):
            tb[unsafe_offset=i] = title.unsafe_ptr()[unsafe_offset=i]
        tb[unsafe_offset=title.byte_length()] = UInt8(0)
        viewer = viewer_create(Int32(W), Int32(H), tb, Int32(0))
        if not _is_real(viewer):
            raise Error("cannot open the viewer window")
        viewer_set_camera_state(viewer, cs)
        print("controls: W/A/S/D move, Q/E or Space/C up/down, Shift x10, wheel speed, drag look; +/- displacement x2 / x0.5,",
              "0 limit surface, L headlight, Esc quit")
    var n_frames = Int(atof(frames_env)) if headless else 0
    var frame = 0; var headlight = 0
    var t_last = perf_counter_ns(); var frames_since = 0; var moving_ms = Float64(0); var moving_n = 0; var still_ms = Float64(0); var still_n = 0
    var it = 0
    var host = List[Float32](length=3 * W * H, fill=Float32(0))
    var cur_w = W; var cur_h = H
    while True:
        var moved = False; var rebuild = False
        if headless:
            if it >= n_frames + 32:
                break
            if it == n_frames // 2 and getenv("DEMO_SWITCH").byte_length() > 0:   # test the +/- rebuild without a window
                exag = Float32(8); rebuild = True
            if it < n_frames:   # an orbit around the rock, then 32 still frames
                var a = Float32(2) * Float32(3.14159265) * Float32(it) / Float32(max(n_frames, 1))
                var r = diag * near * Float32(0.83)
                eye = Vec3f(c.x + r * sin(a + Float32(0.57)), c.y + Float32(0.55) * diag * near, c.z + r * cos(a + Float32(0.57)))
                fwd = c - eye
                fwd = fwd * (Float32(1) / sqrt(dot(fwd, fwd)))
                moved = True
        else:
            viewer_poll_events(viewer)
            if viewer_should_close(viewer) != 0:
                break
            var key = Int(viewer_take_key(viewer))
            if key == 61 or key == 334:   # '=' / '+' and keypad +
                exag = max(exag * Float32(2), Float32(1)); rebuild = True
            elif key == 45 or key == 333:
                exag = exag * Float32(0.5) if exag > Float32(1) else Float32(0); rebuild = True
            elif key == 48:   # displacement off, and back on at the same scale
                if exag > Float32(0):
                    exag_off = exag; exag = Float32(0)
                else:
                    exag = exag_off
                rebuild = True
            elif key == 76:
                headlight = 1 - headlight; frame = 0
            if rebuild:
                print("displacement x", exag)
            viewer_get_camera_state(viewer, cs)
            if cs[unsafe_offset=0].cameraChanged != 0:
                moved = True
            eye = Vec3f(cs[unsafe_offset=0].position.x, cs[unsafe_offset=0].position.y, cs[unsafe_offset=0].position.z)
            fwd = cs[unsafe_offset=0].direction
        if rebuild:
            host_d = DispSurface(patches, sq, face, hdr, pyr, exag / Float32(255), Vec3f(0, 0, 0), Float32(0), chain_off, chain, fm, greg, _grid_levels(), _leaf_mode())
            _demo_bvh(host_d, np, nodes, order)
            _demo_upload(ctx, nb_buf, or_buf, nb_stage, nb_cap, nodes, order)   # nodes and the patch order they index
            moved = True
        if moved:
            frame = 0
        var rw = W; var rh = H
        if moved and half_move:
            rw = W // 2; rh = H // 2
        elif not moved and (cur_w != W):
            frame = 0   # back to full resolution
        cur_w = rw; cur_h = rh
        var right = cross(fwd, world_up)
        right = right * (Float32(1) / sqrt(dot(right, right)))
        var up = cross(right, fwd)
        var dd = DispSurface(dpat, dsq, dface, dhdr, dpyr, exag / Float32(255), eye, Float32(2) * tan_h / Float32(rw), doff, dchain, dfm, dgreg, _grid_levels(), _leaf_mode())
        var cam = DCam(eye, fwd, right, up, light, tan_h, exposure, Int32(rw), Int32(rh), Int32(frame), Int32(headlight), Int32(1 if color_on else 0))
        var t0 = perf_counter_ns()
        ctx.enqueue_function[demo_gpu](dd, dnodes, dorder, crgb, coff, daccum, cam, grid_dim=ceildiv(rw * rh, 64), block_dim=64)
        ctx.synchronize()
        var ms = Float64(perf_counter_ns() - t0) * 1e-6
        if moved:
            moving_ms += ms; moving_n += 1
        else:
            still_ms += ms; still_n += 1
        frame += 1
        if not headless or it == n_frames + 31:
            with accum_b.map_to_host() as hb:
                if headless:
                    for k in range(3 * rw * rh):
                        host[k] = hb[k]
                else:
                    viewer_update_framebuffer(viewer, hb.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), Int32(rw), Int32(rh))
        frames_since += 1
        var now = perf_counter_ns()
        if now - t_last > 2000000000:
            print("frame", it, ":", Float64(frames_since) / (Float64(now - t_last) * 1e-9), "fps,", rw, "x", rh, ",", frame, "spp; GPU ms/frame moving",
                  moving_ms / Float64(max(moving_n, 1)), "still", still_ms / Float64(max(still_n, 1)))
            t_last = now; frames_since = 0
        it += 1
    print("GPU ms per frame:", moving_ms / Float64(max(moving_n, 1)), "moving (", moving_n, "frames ),", still_ms / Float64(max(still_n, 1)), "still (", still_n, "frames ); exag", exag)
    if headless:   # the viewer's display transform: Reinhard, then gamma
        for k in range(3 * W * H):
            host[k] = host[k] / (host[k] + Float32(1))
        _write_ppm_rgb(out_dir + "demo_last.ppm", host, W, H)
    else:
        viewer_destroy(viewer)
    _ = len(bufs); _ = len(nodes); _ = len(order)


def main() raises:
    var out_dir = String("/tmp/claude-1000/-home-gonsolo-work-moana-3060-paper/5a168d33-8d85-45d2-a94f-7e6bdd0ebff6/scratchpad/rock/")
    var W = 960; var H = 720
    var counts = unsafe_alloc[Int64](4)
    var ptx = String(PTX)
    var cname = unsafe_alloc[UInt8](ptx.byte_length() + 1)
    for i in range(ptx.byte_length()):
        cname[unsafe_offset=i] = ptx.unsafe_ptr()[unsafe_offset=i]
    cname[unsafe_offset=ptx.byte_length()] = UInt8(0)
    var hd = external_call["disp_open", Int](cname, Int32(2), counts)
    if hd == 0:
        raise Error("cannot read " + ptx)
    var np = Int(counts[0]); var nf = Int(counts[1]); var npyr = Int(counts[2])
    var cpf = unsafe_alloc[Float32](48 * np)
    var sq = unsafe_alloc[Float32](3 * np)
    var face = unsafe_alloc[Int32](np)
    var hdr = unsafe_alloc[Int32](3 * nf)
    var pyr = unsafe_alloc[UInt8](npyr)
    var cbox = unsafe_alloc[Float32](6)
    var nchain = Int(counts[3])
    var chain_off = unsafe_alloc[Int32](nf)
    var chain = unsafe_alloc[UInt8](max(nchain, 1))
    external_call["disp_take_chains", NoneType](hd, chain_off, chain)
    var fm = unsafe_alloc[Float32](12 * np)
    var greg = unsafe_alloc[UInt8](np)
    external_call["disp_take_gregory", NoneType](hd, fm, greg)
    external_call["disp_take", NoneType](hd, cpf, sq, face, hdr, pyr, cbox)
    var patches = unsafe_alloc[BezierPatch](np)
    for p in range(np):
        var x = SIMD[DType.float32, 16](0); var y = SIMD[DType.float32, 16](0); var z = SIMD[DType.float32, 16](0)
        for k in range(16):
            x[k] = cpf[unsafe_offset=48 * p + 3 * k]; y[k] = cpf[unsafe_offset=48 * p + 3 * k + 1]; z[k] = cpf[unsafe_offset=48 * p + 3 * k + 2]
        patches[unsafe_offset=p] = BezierPatch(x, y, z)
    var texels = 0
    for f in range(nf):
        texels += (1 << Int(hdr[unsafe_offset=3 * f + 1])) * (1 << Int(hdr[unsafe_offset=3 * f + 2]))
    var blo = Vec3f(cbox[unsafe_offset=0], cbox[unsafe_offset=1], cbox[unsafe_offset=2])
    var bhi = Vec3f(cbox[unsafe_offset=3], cbox[unsafe_offset=4], cbox[unsafe_offset=5])
    print("rock:", np, "patches,", nf, "faces,", texels, "texels,", npyr, "pyramid bytes; box", blo.x, blo.y, blo.z, "..", bhi.x, bhi.y, bhi.z)
    print("bordered height chains:", nchain, "bytes =", Float32(nchain) / Float32(texels), "x the plain heights (", texels, "bytes ); height nodes (planes + residuals, 4 x 4-texel cells up)", npyr, "bytes")

    # Camera: a 3/4 view of the cage box (CAM_* overrides), and the light grazing from the side.
    var c = (blo + bhi) * Float32(0.5)
    var ext = bhi - blo
    var diag = sqrt(dot(ext, ext))
    var near = _env_f("CAM_DIST", Float32(1.1))
    var eye = Vec3f(c.x + _env_f("CAM_X", Float32(0.45)) * diag * near, c.y + _env_f("CAM_Y", Float32(0.55)) * diag * near,
                    c.z + _env_f("CAM_Z", Float32(0.7)) * diag * near)
    var look = Vec3f(c.x + _env_f("LOOK_X", Float32(0)) * diag, c.y + _env_f("LOOK_Y", Float32(0)) * diag, c.z + _env_f("LOOK_Z", Float32(0)) * diag)
    var fwd = look - eye
    fwd = fwd * (Float32(1) / sqrt(dot(fwd, fwd)))
    var right = cross(fwd, Vec3f(0, 1, 0))
    right = right * (Float32(1) / sqrt(dot(right, right)))
    var up = cross(right, fwd)
    var fov = _env_f("FOV", Float32(40)) * Float32(3.14159265) / Float32(180)
    var tan_h = tan(fov * Float32(0.5))
    var light = right * Float32(0.75) + up * Float32(0.55) - fwd * Float32(0.25)
    light = light * (Float32(1) / sqrt(dot(light, light)))
    var pixel = Float32(2) * tan_h / Float32(W) * _env_f("FOOTPRINT", Float32(1))
    var exag = _env_f("EXAG", Float32(8))
    var only = getenv("ONLY")
    var spp_n = Int(sqrt(_env_f("SPP", Float32(1))) + Float32(0.5))   # SPP=64: 8 x 8
    var col_path = getenv("COLOR")   # a colour Ptex: an RGB image as well
    var color_on = col_path.byte_length() > 0
    if color_on:
        var cc = unsafe_alloc[UInt8](col_path.byte_length() + 1)
        for i in range(col_path.byte_length()):
            cc[unsafe_offset=i] = col_path.unsafe_ptr()[unsafe_offset=i]
        cc[unsafe_offset=col_path.byte_length()] = UInt8(0)
        if external_call["disp_color_open", Int32](cc) == 0:
            raise Error("cannot read " + col_path)

    if getenv("EDGE_GAP") == "1":
        _edge_gap(DispSurface(patches, sq, face, hdr, pyr, Float32(0), eye, pixel, chain_off, chain, fm, greg, _grid_levels(), _leaf_mode()), np)
        return
    var ngreg = 0
    for p in range(np):
        ngreg += 1 if greg[unsafe_offset=p] == 1 else 0
    print("Gregory patches:", ngreg, "of", np, "(", np * (size_of[BezierPatch]() + 16) + ngreg * 48, "patch bytes incl. Fm )")

    if getenv("INTERACTIVE") == "1" or getenv("DEMO_FRAMES").byte_length() > 0:
        run_demo(patches, sq, face, hdr, pyr, npyr, chain_off, chain, nchain, fm, greg, np, nf, blo, bhi, out_dir)
        return

    var lo_b = List[Vec3f](); var hi_b = List[Vec3f]()
    for scene in range(5):   # 0: limit surface, 1: scale 1, 2: exaggerated, 3: tessellated reference of 2, 4: the cage
        var name = ["a_limit", "b_disp1", "c_disp" + String(Int(exag)), "d_tess" + String(Int(exag)), "e_cage"][scene]
        if only.byte_length() > 0 and only != name:
            continue
        var scale = Float32(0) if scene == 0 else (Float32(1) / Float32(255) if scene == 1 else exag / Float32(255))
        var d = DispSurface(patches, sq, face, hdr, pyr, scale, eye, pixel, chain_off, chain, fm, greg, _grid_levels(), _leaf_mode())
        var img = List[Float32](length=W * H, fill=Float32(0))
        var rgb = List[Float32](length=3 * W * H, fill=Float32(0))
        var hitmask = List[UInt8](length=W * H, fill=UInt8(0))
        var steps = List[Int](length=H, fill=0)
        var maxs = List[Int](length=H, fill=0)
        var mem = 0
        var t0 = perf_counter_ns()
        if scene < 3:
            var nodes = List[BNode](); var order = List[Int32]()
            if getenv("REF_HW").byte_length() == 0 and getenv("HW") != "1":
                _sub_bvh(d, np, nodes, order)
            mem = np * (size_of[BezierPatch]() + 16) + len(nodes) * size_of[BNode]() + 4 * len(order) + (npyr if scene > 0 else 0)
            print("BVH:", np * sub_count(d), "sub-squares,", len(nodes), "nodes,", (len(nodes) * size_of[BNode]() + 4 * len(order)) // 1024, "KiB")
            var tb = perf_counter_ns()
            var np_ptr = nodes.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var or_ptr = order.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var im_ptr = img.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var rg_ptr = rgb.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var hm_ptr = hitmask.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var st_ptr = steps.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var mx_ptr = maxs.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            def row(y: Int) {imm}:
                for x in range(W):
                    var acc = Float32(0); var hits = 0
                    var cr = Float32(0); var cg = Float32(0); var cb = Float32(0)
                    var col = unsafe_alloc[Float32](3)
                    for k in range(spp_n * spp_n):   # stratified sub-pixel samples
                        var jx = (Float32(k % spp_n) + Float32(0.5)) / Float32(spp_n)
                        var jy = (Float32(k // spp_n) + Float32(0.5)) / Float32(spp_n)
                        var px = (Float32(2) * (Float32(x) + jx) / Float32(W) - Float32(1)) * tan_h
                        var py = (Float32(1) - Float32(2) * (Float32(y) + jy) / Float32(H)) * tan_h * Float32(H) / Float32(W)
                        var dir = fwd + right * px + up * py
                        dir = dir * (Float32(1) / sqrt(dot(dir, dir)))
                        var h = trace_patches(d, np_ptr, or_ptr, eye, dir)
                        st_ptr[unsafe_offset=y] += h.steps
                        mx_ptr[unsafe_offset=y] = max(mx_ptr[unsafe_offset=y], h.steps)
                        if h.prim >= 0:
                            var n = displaced_normal(d, h.prim, h.s, h.u, max(h.size, Float32(1.0 / 512.0)))   # heights at the footprint
                            if dot(n, dir) > Float32(0):
                                n = -n
                            var sh = Float32(0.08) + Float32(0.75) * max(dot(n, light), Float32(0))
                            acc += sh
                            hits += 1
                            if color_on:
                                var sz = d.sq[unsafe_offset=3 * h.prim + 2]
                                external_call["disp_color", NoneType](d.face[unsafe_offset=h.prim], d.sq[unsafe_offset=3 * h.prim] + h.s * sz,
                                    d.sq[unsafe_offset=3 * h.prim + 1] + h.u * sz, max(h.size, Float32(1.0 / 1024.0)) * sz, col)
                                cr += sh * col[unsafe_offset=0]; cg += sh * col[unsafe_offset=1]; cb += sh * col[unsafe_offset=2]
                    im_ptr[unsafe_offset=y * W + x] = acc / Float32(spp_n * spp_n)
                    var inv_n = Float32(1.6) / Float32(spp_n * spp_n)   # a little exposure for the albedo
                    rg_ptr[unsafe_offset=3 * (y * W + x)] = cr * inv_n; rg_ptr[unsafe_offset=3 * (y * W + x) + 1] = cg * inv_n; rg_ptr[unsafe_offset=3 * (y * W + x) + 2] = cb * inv_n
                    col.unsafe_free()
                    if hits > 0:
                        hm_ptr[unsafe_offset=y * W + x] = UInt8(1)
            if getenv("GPU") == "1":   # the same render on the GPU (B580): device copies of everything the traversal reads
                var ctx = DeviceContext()
                var bufs = List[DeviceBuffer[DType.uint8]]()
                var nchain_b = nchain
                var dd = DispSurface(_up(ctx, bufs, patches, np), _up(ctx, bufs, sq, 3 * np), _up(ctx, bufs, face, np), _up(ctx, bufs, hdr, 3 * nf),
                                     _up(ctx, bufs, pyr, npyr), scale, eye, pixel, _up(ctx, bufs, chain_off, nf), _up(ctx, bufs, chain, nchain_b),
                                     _up(ctx, bufs, fm, 12 * np), _up(ctx, bufs, greg, np), _grid_levels(), _leaf_mode())
                var nd_d = _up(ctx, bufs, nodes.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), len(nodes))
                var or_d = _up(ctx, bufs, order.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), len(order))
                var img_b = ctx.enqueue_create_buffer[DType.float32](W * H)
                var st_b = ctx.enqueue_create_buffer[DType.int32](W * H)
                var img_d = img_b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
                var st_d = st_b.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
                var gc = GCam(eye, fwd, right, up, light, tan_h, Int32(W), Int32(H), Int32(0) if getenv("TILE") == "0" else Int32(1), Int32(atol(getenv("PROF", "0"))))
                if getenv("REF_HW").byte_length() > 0 or getenv("HW") == "1":
                    _hw_render(ctx, bufs, d, dd, np, name, gc, img, hitmask, steps, maxs)
                    _write_ppm(out_dir + name + "_hw.ppm", img, W, H)
                    with open(out_dir + name + "_hw.mask", "w") as f:
                        f.write_bytes(Span(hitmask))
                    var tot = 0; var mx = 0
                    for y in range(H):
                        tot += steps[y]; mx = max(mx, maxs[y])
                    print(name + ": HW steps/ray", Float32(tot) / Float32(W * H), "(max", mx, ")")
                    continue
                ctx.synchronize()
                var tg0 = perf_counter_ns()
                ctx.enqueue_function[render_gpu](dd, nd_d, or_d, img_d, st_d, gc, grid_dim=ceildiv(W * H, 64), block_dim=64)
                ctx.synchronize()
                var tg1 = perf_counter_ns()
                var gc5 = GCam(eye, fwd, right, up, light, tan_h, Int32(W), Int32(540), gc.tile, gc.prof)
                var best5 = 1 << 62
                for _ in range(3):
                    var ta5 = perf_counter_ns()
                    ctx.enqueue_function[render_gpu](dd, nd_d, or_d, img_d, st_d, gc5, grid_dim=ceildiv(W * 540, 64), block_dim=64)
                    ctx.synchronize()
                    best5 = min(best5, Int(perf_counter_ns()) - Int(ta5))
                print(name + ": GPU", W, "x 540 render", Float32(best5) * Float32(1e-6), "ms,", Float32(W * 540) / (Float32(best5) * Float32(1e-9)) / Float32(1e6), "Mrays/s")
                var tg2 = 0; var tg3 = 1 << 62
                for _ in range(3):   # the fastest of three launches
                    var ta = perf_counter_ns()
                    ctx.enqueue_function[render_gpu](dd, nd_d, or_d, img_d, st_d, gc, grid_dim=ceildiv(W * H, 64), block_dim=64)
                    ctx.synchronize()
                    var tb2 = perf_counter_ns()
                    if Int(tb2) - Int(ta) < tg3 - tg2:
                        tg2 = Int(ta); tg3 = Int(tb2)
                print(name + ": GPU first launch (with compile)", (tg1 - tg0) // 1000000, "ms, render", Float32(tg3 - tg2) * Float32(1e-6), "ms,",
                      Float32(W * H) / (Float32(tg3 - tg2) * Float32(1e-9)) / Float32(1e6), "Mrays/s")
                with img_b.map_to_host() as hi:
                    for k in range(W * H):
                        img[k] = hi[k]
                        if hi[k] > Float32(0):
                            hitmask[k] = UInt8(1)
                with st_b.map_to_host() as hs:
                    for k in range(W * H):
                        steps[k // W] += Int(hs[k]); maxs[k // W] = max(maxs[k // W], Int(hs[k]))
                _ = len(bufs); _ = len(nodes); _ = len(order)
            elif getenv("SERIAL") == "1":
                for y in range(H):
                    row(y)
            else:
                parallelize(row, H)
            var t1 = perf_counter_ns()
            _ = len(nodes); _ = len(order)   # the render read them through pointers
            var tot = 0; var mx = 0
            for y in range(H):
                tot += steps[y]; mx = max(mx, maxs[y])
            print(name + ": build", (tb - t0) // 1000000, "ms, render", (t1 - tb) // 1000000, "ms,", Float32(W * H) / (Float32(t1 - tb) * Float32(1e-9)) / Float32(1e6),
                  "Mrays/s, steps/ray", Float32(tot) / Float32(W * H), "(max per pixel row", mx, "), memory", mem // 1024, "KiB")
        elif scene == 4:
            var sz = unsafe_alloc[Int32](3)
            _ = external_call["disp_cage", Int32](cname, Int(0), Int(0), Int(0), sz)
            var cpos = List[Float32](length=3 * Int(sz[0]), fill=0)
            var ccnt = List[Int32](length=Int(sz[1]), fill=0)
            var cidx = List[Int32](length=Int(sz[2]), fill=0)
            _ = external_call["disp_cage", Int32](cname, Int(cpos.unsafe_ptr()), Int(ccnt.unsafe_ptr()), Int(cidx.unsafe_ptr()), sz)
            var pts = List[Vec3f](); var tri = List[Int32](); var edges = List[Int32]()   # per triangle its 2 cage edges (4 indices)
            for v in range(Int(sz[0])):
                pts.append(Vec3f(cpos[3 * v], cpos[3 * v + 1], cpos[3 * v + 2]))
            var at = 0
            for f in range(Int(sz[1])):
                var n = Int(ccnt[f])
                for k in range(1, n - 1):   # a fan: the first and last triangles carry the outer edges
                    var a = cidx[at]; var b = cidx[at + k]; var c2 = cidx[at + k + 1]
                    tri.append(a); tri.append(b); tri.append(c2)
                    edges.append(b); edges.append(c2)
                    edges.append(a if k == 1 else c2); edges.append(b if k == 1 else (cidx[at] if k == n - 2 else c2))
                at += n
            var nt = len(tri) // 3
            lo_b.clear(); hi_b.clear()
            for t in range(nt):
                var p0 = pts[Int(tri[3 * t])]; var p1 = pts[Int(tri[3 * t + 1])]; var p2 = pts[Int(tri[3 * t + 2])]
                lo_b.append(Vec3f(min(p0.x, min(p1.x, p2.x)), min(p0.y, min(p1.y, p2.y)), min(p0.z, min(p1.z, p2.z))))
                hi_b.append(Vec3f(max(p0.x, max(p1.x, p2.x)), max(p0.y, max(p1.y, p2.y)), max(p0.z, max(p1.z, p2.z))))
            var nodes = List[BNode](); var order = List[Int32]()
            for t in range(nt):
                order.append(Int32(t))
            _ = _build(nodes, order, lo_b, hi_b, 0, nt)
            var pt_ptr = pts.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var tr_ptr = tri.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var ed_ptr = edges.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var np_ptr = nodes.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var or_ptr = order.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var im_ptr = img.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var hm_ptr = hitmask.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var px_ang = Float32(2) * tan_h / Float32(W)
            def row_c(y: Int) {imm}:
                for x in range(W):
                    var px = (Float32(2) * (Float32(x) + Float32(0.5)) / Float32(W) - Float32(1)) * tan_h
                    var py = (Float32(1) - Float32(2) * (Float32(y) + Float32(0.5)) / Float32(H)) * tan_h * Float32(H) / Float32(W)
                    var dir = fwd + right * px + up * py
                    dir = dir * (Float32(1) / sqrt(dot(dir, dir)))
                    var h = trace_tris(pt_ptr, tr_ptr, np_ptr, or_ptr, eye, dir)
                    if h.prim >= 0:
                        var p0 = pt_ptr[unsafe_offset=Int(tr_ptr[unsafe_offset=3 * h.prim])]
                        var n = cross(pt_ptr[unsafe_offset=Int(tr_ptr[unsafe_offset=3 * h.prim + 1])] - p0, pt_ptr[unsafe_offset=Int(tr_ptr[unsafe_offset=3 * h.prim + 2])] - p0)
                        n = n * (Float32(1) / sqrt(dot(n, n)))
                        if dot(n, dir) > Float32(0):
                            n = -n
                        var shade = Float32(0.08) + Float32(0.75) * max(dot(n, light), Float32(0))
                        var hp = eye + dir * h.t
                        for e in range(2):   # dark cage edges, a pixel wide
                            var ea = pt_ptr[unsafe_offset=Int(ed_ptr[unsafe_offset=4 * h.prim + 2 * e])]
                            var eb = pt_ptr[unsafe_offset=Int(ed_ptr[unsafe_offset=4 * h.prim + 2 * e + 1])]
                            var ev = eb - ea
                            var l2 = dot(ev, ev)
                            if l2 > Float32(0):
                                var cr = cross(hp - ea, ev)
                                if sqrt(dot(cr, cr) / l2) < Float32(0.8) * px_ang * h.t:
                                    shade = Float32(0.12)
                        im_ptr[unsafe_offset=y * W + x] = shade
                        hm_ptr[unsafe_offset=y * W + x] = UInt8(1)
            parallelize(row_c, H)
            _ = len(nodes); _ = len(order); _ = len(pts); _ = len(tri); _ = len(edges); _ = len(cpos); _ = len(ccnt); _ = len(cidx)
            sz.unsafe_free()
            print(name + ":", nt, "triangles of the cage")
        else:
            var nref = Int(_env_f("REF_N", Float32(16)))
            var pts = List[Vec3f](); var tri = List[Int32]()
            for p in range(np):
                var f = Int(face[unsafe_offset=p])
                var res = max(1 << Int(hdr[unsafe_offset=3 * f + 1]), 1 << Int(hdr[unsafe_offset=3 * f + 2]))
                var n = min(max(Int(Float32(res) * sq[unsafe_offset=3 * p + 2] / Float32(8)), 2), nref)
                var base = len(pts)
                for j in range(n + 1):
                    for i in range(n + 1):
                        pts.append(displaced_point(d, p, Float32(i) / Float32(n), Float32(j) / Float32(n)))
                for j in range(n):
                    for i in range(n):
                        var a = Int32(base + j * (n + 1) + i)
                        tri.append(a); tri.append(a + 1); tri.append(a + Int32(n + 2))
                        tri.append(a); tri.append(a + Int32(n + 2)); tri.append(a + Int32(n + 1))
            var nt = len(tri) // 3
            lo_b.clear(); hi_b.clear()
            for t in range(nt):
                var p0 = pts[Int(tri[3 * t])]; var p1 = pts[Int(tri[3 * t + 1])]; var p2 = pts[Int(tri[3 * t + 2])]
                lo_b.append(Vec3f(min(p0.x, min(p1.x, p2.x)), min(p0.y, min(p1.y, p2.y)), min(p0.z, min(p1.z, p2.z))))
                hi_b.append(Vec3f(max(p0.x, max(p1.x, p2.x)), max(p0.y, max(p1.y, p2.y)), max(p0.z, max(p1.z, p2.z))))
            var nodes = List[BNode](); var order = List[Int32]()
            for t in range(nt):
                order.append(Int32(t))
            _ = _build(nodes, order, lo_b, hi_b, 0, nt)
            mem = len(pts) * 12 + nt * 12 + len(nodes) * size_of[BNode]() + nt * 4
            var tb = perf_counter_ns()
            var pt_ptr = pts.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var tr_ptr = tri.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var np_ptr = nodes.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var or_ptr = order.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var im_ptr = img.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            var hm_ptr = hitmask.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            def row_t(y: Int) {imm}:
                for x in range(W):
                    var px = (Float32(2) * (Float32(x) + Float32(0.5)) / Float32(W) - Float32(1)) * tan_h
                    var py = (Float32(1) - Float32(2) * (Float32(y) + Float32(0.5)) / Float32(H)) * tan_h * Float32(H) / Float32(W)
                    var dir = fwd + right * px + up * py
                    dir = dir * (Float32(1) / sqrt(dot(dir, dir)))
                    var h = trace_tris(pt_ptr, tr_ptr, np_ptr, or_ptr, eye, dir)
                    if h.prim >= 0:
                        var p0 = pt_ptr[unsafe_offset=Int(tr_ptr[unsafe_offset=3 * h.prim])]
                        var n = cross(pt_ptr[unsafe_offset=Int(tr_ptr[unsafe_offset=3 * h.prim + 1])] - p0, pt_ptr[unsafe_offset=Int(tr_ptr[unsafe_offset=3 * h.prim + 2])] - p0)
                        n = n * (Float32(1) / sqrt(dot(n, n)))
                        if dot(n, dir) > Float32(0):
                            n = -n
                        im_ptr[unsafe_offset=y * W + x] = Float32(0.08) + Float32(0.75) * max(dot(n, light), Float32(0))
                        hm_ptr[unsafe_offset=y * W + x] = UInt8(1)
            parallelize(row_t, H)
            var t1 = perf_counter_ns()
            _ = len(nodes); _ = len(order); _ = len(pts); _ = len(tri)
            print(name + ":", nt, "triangles (up to", nref, "x", nref, "quads a patch), build", (tb - t0) // 1000000, "ms, render", (t1 - tb) // 1000000,
                  "ms, memory", mem // 1024, "KiB; at one quad per texel:", texels * 2, "triangles, about", (texels * (12 + 24 + 2 * 32 * 2)) // (1024 * 1024), "MiB")
        _write_ppm(out_dir + name + ".ppm", img, W, H)
        if color_on and scene < 3:
            _write_ppm_rgb(out_dir + name + "_color.ppm", rgb, W, H)
        with open(out_dir + name + ".mask", "w") as f:
            f.write_bytes(Span(hitmask))
