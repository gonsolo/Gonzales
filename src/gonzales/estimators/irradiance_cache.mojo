"""Jensen and Buhler 2002's irradiance samples on translucent surfaces and their hierarchical dipole integral."""
from std.math import floor, max, min, sqrt
from std.memory.alloc import unsafe_alloc
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from gonzales.core.geometry import Point3f, Vec3f, cross, dot, _is_real_ptr
from gonzales.core.rng import PCG32
from gonzales.core.spectrum import SpectralSample
from gonzales.scene.shapes.primitives import PrimId, mesh_normal, mesh_pos, mesh_vi
from gonzales.scene.accel.bvh import SceneView
from gonzales.estimators.dipole import Dipole

comptime _LEAF = 8
comptime _MAX_POINTS = 4_000_000


@always_inline
def is_translucent(ref sd: SceneView, p: PrimId) -> Bool:
    """A triangle of a pbrt subsurface object, the surfaces the cache samples."""
    return p.type == Int8(0) and p.instanceIdx < Int32(0) and sd.materials[unsafe_offset=Int(p.materialIndex)].sss_walk_interior()


@fieldwise_init
struct IrradianceCache(TrivialRegisterPassable, DevicePassable):
    """Points in a binary tree per material, nodes in preorder with skip links (JB's octree, median split instead).
    `lx` holds each point's radiance through the boundary's exit lobe for the current pass, E_t / (c pi): the lobe at
    the camera hit carries the same 1 / (c pi), so A * lx stands in for A * E_t in the dipole sum."""
    var n: Int
    var pos: Pointer[Point3f, MutUntrackedOrigin]
    var nrm: Pointer[Vec3f, MutUntrackedOrigin]
    var mat: Pointer[Int32, MutUntrackedOrigin]
    var area: Pointer[Float32, MutUntrackedOrigin]
    var lx: Pointer[SpectralSample, MutUntrackedOrigin]
    var n_nodes: Int
    var center: Pointer[Point3f, MutUntrackedOrigin]
    var far2: Pointer[Float32, MutUntrackedOrigin]     # squared distance beyond which a node counts as one point
    var skip: Pointer[Int32, MutUntrackedOrigin]       # the node after this subtree
    var begin: Pointer[Int32, MutUntrackedOrigin]      # a leaf's points [begin, end); -1 for an inner node
    var end: Pointer[Int32, MutUntrackedOrigin]
    var sum: Pointer[SpectralSample, MutUntrackedOrigin]
    var mat_first: Pointer[Int32, MutUntrackedOrigin]  # a material's nodes [mat_first, mat_end)
    var mat_end: Pointer[Int32, MutUntrackedOrigin]
    var spacing: Float32
    comptime device_type: AnyType = Self

    def _to_device_type(self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "IrradianceCache"

    @staticmethod
    def build(ref sd: SceneView, prims: Pointer[PrimId, MutUntrackedOrigin], n_prims: Int, spacing_in: Float32,
              near: Float32, seed: UInt64) -> IrradianceCache:
        """Spreads points evenly over every translucent mesh (systematic sampling along its triangles' area)."""
        var n_mat = Int(sd.materialCount)
        var area_m = List[Float64](length=n_mat, fill=Float64(0))
        var st_max = Float32(0)
        for k in range(n_prims):
            var p = prims[unsafe_offset=k]
            if not is_translucent(sd, p):
                continue
            area_m[Int(p.materialIndex)] += Float64(_tri_area(sd, p))
            var m = sd.materials[unsafe_offset=Int(p.materialIndex)]
            var g = m.checker_uscale
            st_max = max(st_max, max(m.checker_tex1.r * (Float32(1) - m.checker_tex2.r * g),
                                     max(m.checker_tex1.g * (Float32(1) - m.checker_tex2.g * g),
                                         m.checker_tex1.b * (Float32(1) - m.checker_tex2.b * g))))
        var total = Float64(0)
        for m in range(n_mat):
            total += area_m[m]
        # Half the shortest reduced mean free path: the dipole's near field is about one wide.
        var s = spacing_in if spacing_in > Float32(0) else Float32(0.5) / max(st_max, Float32(1e-6))
        if total / Float64(s * s) > Float64(_MAX_POINTS):
            s = Float32(sqrt(total / Float64(_MAX_POINTS)))
        var step = List[Float64](length=n_mat, fill=Float64(0))
        var nxt = List[Float64](length=n_mat, fill=Float64(0))
        var acc = List[Float64](length=n_mat, fill=Float64(0))
        var pcg = PCG32(seed, UInt64(17))
        for m in range(n_mat):
            if area_m[m] > Float64(0):
                step[m] = area_m[m] / max(Float64(1), floor(area_m[m] / Float64(s * s) + Float64(0.5)))
                nxt[m] = Float64(pcg.next_float()) * step[m]
        var pts = List[Point3f]()
        var nrms = List[Vec3f]()
        var mats = List[Int32]()
        for k in range(n_prims):
            var p = prims[unsafe_offset=k]
            if not is_translucent(sd, p):
                continue
            var mi = Int(p.materialIndex)
            var a = Float64(_tri_area(sd, p))
            # Several points in one triangle follow an R2 sequence from a random start, not independent draws.
            var u0 = pcg.next_float(); var v0 = pcg.next_float(); var j = 0
            while nxt[mi] < acc[mi] + a:
                var fu = Float64(u0) + Float64(j) * 0.7548776662466927
                var fv = Float64(v0) + Float64(j) * 0.5698402909980532
                var su = sqrt(Float32(fu - floor(fu)))
                var (x, nn) = _tri_point(sd, p, Float32(1) - su, Float32(fv - floor(fv)) * su)
                pts.append(x); nrms.append(nn); mats.append(Int32(mi))
                nxt[mi] += step[mi]; j += 1
            acc[mi] += a
        var n = len(pts)
        var order = List[Int](capacity=n)
        var t = _Tree()
        var mat_first = unsafe_alloc[Int32](max(n_mat, 1)).unsafe_origin_cast[MutUntrackedOrigin]()
        var mat_end = unsafe_alloc[Int32](max(n_mat, 1)).unsafe_origin_cast[MutUntrackedOrigin]()
        for m in range(n_mat):
            var lo = len(order)
            for i in range(n):
                if Int(mats[i]) == m:
                    order.append(i)
            mat_first[unsafe_offset=m] = Int32(len(t.skip))
            if len(order) > lo:
                _build_node(t, order, pts, lo, len(order), near)
            mat_end[unsafe_offset=m] = Int32(len(t.skip))
        var pos = unsafe_alloc[Point3f](max(n, 1)).unsafe_origin_cast[MutUntrackedOrigin]()
        var nrm = unsafe_alloc[Vec3f](max(n, 1)).unsafe_origin_cast[MutUntrackedOrigin]()
        var mat = unsafe_alloc[Int32](max(n, 1)).unsafe_origin_cast[MutUntrackedOrigin]()
        var area = unsafe_alloc[Float32](max(n, 1)).unsafe_origin_cast[MutUntrackedOrigin]()
        for j in range(n):
            var i = order[j]
            pos[unsafe_offset=j] = pts[i]; nrm[unsafe_offset=j] = nrms[i]; mat[unsafe_offset=j] = mats[i]
            area[unsafe_offset=j] = Float32(step[Int(mats[i])])
        var nn = len(t.skip)
        var center = unsafe_alloc[Point3f](max(nn, 1)).unsafe_origin_cast[MutUntrackedOrigin]()
        var far2 = unsafe_alloc[Float32](max(nn, 1)).unsafe_origin_cast[MutUntrackedOrigin]()
        var skip = unsafe_alloc[Int32](max(nn, 1)).unsafe_origin_cast[MutUntrackedOrigin]()
        var begin = unsafe_alloc[Int32](max(nn, 1)).unsafe_origin_cast[MutUntrackedOrigin]()
        var end = unsafe_alloc[Int32](max(nn, 1)).unsafe_origin_cast[MutUntrackedOrigin]()
        for k in range(nn):
            center[unsafe_offset=k] = t.center[k]; far2[unsafe_offset=k] = t.far2[k]; skip[unsafe_offset=k] = t.skip[k]
            begin[unsafe_offset=k] = t.begin[k]; end[unsafe_offset=k] = t.end[k]
        return IrradianceCache(n, pos, nrm, mat, area,
                               unsafe_alloc[SpectralSample](max(n, 1)).unsafe_origin_cast[MutUntrackedOrigin](),
                               nn, center, far2, skip, begin, end,
                               unsafe_alloc[SpectralSample](max(nn, 1)).unsafe_origin_cast[MutUntrackedOrigin](),
                               mat_first, mat_end, s)

    def free(self):
        self.pos.unsafe_free(); self.nrm.unsafe_free(); self.mat.unsafe_free(); self.area.unsafe_free()
        self.lx.unsafe_free(); self.center.unsafe_free(); self.far2.unsafe_free(); self.skip.unsafe_free()
        self.begin.unsafe_free(); self.end.unsafe_free(); self.sum.unsafe_free()
        self.mat_first.unsafe_free(); self.mat_end.unsafe_free()

    def aggregate(self):
        """Weights the pass's lx by area and sums every subtree, children before parents."""
        for k in range(self.n_nodes - 1, -1, -1):
            var b = Int(self.begin[unsafe_offset=k])
            var s = SpectralSample(Float32(0))
            if b >= 0:
                for i in range(b, Int(self.end[unsafe_offset=k])):
                    var v = self.lx[unsafe_offset=i] * self.area[unsafe_offset=i]
                    self.lx[unsafe_offset=i] = v
                    s += v
            else:
                s = self.sum[unsafe_offset=k + 1] + self.sum[unsafe_offset=Int(self.skip[unsafe_offset=k + 1])]
            self.sum[unsafe_offset=k] = s

    def radiance(self, x: Point3f, mat_idx: Int32, dip: Dipole) -> SpectralSample:
        """sum_i R_d(|x - x_i|) A_i lx_i over the material's points, a far subtree as one point at its centroid."""
        var l = SpectralSample(Float32(0))
        var k = Int(self.mat_first[unsafe_offset=Int(mat_idx)])
        var e = Int(self.mat_end[unsafe_offset=Int(mat_idx)])
        while k < e:
            var d2 = _dist2(x, self.center[unsafe_offset=k])
            var b = Int(self.begin[unsafe_offset=k])
            if d2 > self.far2[unsafe_offset=k]:
                l += dip.rd(d2) * self.sum[unsafe_offset=k]
                k = Int(self.skip[unsafe_offset=k])
            elif b >= 0:
                for i in range(b, Int(self.end[unsafe_offset=k])):
                    l += dip.rd(_dist2(x, self.pos[unsafe_offset=i])) * self.lx[unsafe_offset=i]
                k = Int(self.skip[unsafe_offset=k])
            else:
                k += 1
        return l


struct _Tree(Movable):
    var center: List[Point3f]
    var far2: List[Float32]
    var skip: List[Int32]
    var begin: List[Int32]
    var end: List[Int32]

    def __init__(out self):
        self.center = List[Point3f](); self.far2 = List[Float32](); self.skip = List[Int32]()
        self.begin = List[Int32](); self.end = List[Int32]()


def _build_node(mut t: _Tree, mut order: List[Int], pts: List[Point3f], lo: Int, hi: Int, near: Float32):
    """Node over order[lo:hi] in preorder; split at the median of the longest axis. `near` 0 makes no node far."""
    var cx = Float32(0); var cy = Float32(0); var cz = Float32(0)
    var mn = pts[order[lo]]; var mx = pts[order[lo]]
    for j in range(lo, hi):
        var p = pts[order[j]]
        cx += p.x; cy += p.y; cz += p.z
        mn = Point3f(min(mn.x, p.x), min(mn.y, p.y), min(mn.z, p.z))
        mx = Point3f(max(mx.x, p.x), max(mx.y, p.y), max(mx.z, p.z))
    var w = Float32(1) / Float32(hi - lo)
    var c = Point3f(cx * w, cy * w, cz * w)
    var r2 = Float32(0)
    for j in range(lo, hi):
        r2 = max(r2, _dist2(c, pts[order[j]]))
    var k = len(t.skip)
    t.center.append(c); t.far2.append(near * near * r2 if near > Float32(0) else Float32(3.0e38))
    t.skip.append(Int32(0)); t.begin.append(Int32(-1)); t.end.append(Int32(-1))
    if hi - lo <= _LEAF:
        t.begin[k] = Int32(lo); t.end[k] = Int32(hi)
    else:
        var ex = mx.x - mn.x; var ey = mx.y - mn.y; var ez = mx.z - mn.z
        var axis = 0 if ex >= ey and ex >= ez else (1 if ey >= ez else 2)
        var mid = (lo + hi) // 2
        _select(order, pts, lo, hi, mid, axis)
        _build_node(t, order, pts, lo, mid, near)
        _build_node(t, order, pts, mid, hi, near)
    t.skip[k] = Int32(len(t.skip))


@always_inline
def _coord(p: Point3f, axis: Int) -> Float32:
    return p.x if axis == 0 else (p.y if axis == 1 else p.z)


def _select(mut order: List[Int], pts: List[Point3f], lo: Int, hi: Int, kth: Int, axis: Int):
    """Hoare quickselect: order[kth] gets its sorted element, smaller ones before it."""
    var l = lo
    var r = hi - 1
    while l < r:
        var pv = _coord(pts[order[(l + r) // 2]], axis)
        var i = l
        var j = r
        while i <= j:
            while _coord(pts[order[i]], axis) < pv:
                i += 1
            while _coord(pts[order[j]], axis) > pv:
                j -= 1
            if i <= j:
                var tmp = order[i]; order[i] = order[j]; order[j] = tmp
                i += 1; j -= 1
        if kth <= j:
            r = j
        elif kth >= i:
            l = i
        else:
            break


@always_inline
def _dist2(a: Point3f, b: Point3f) -> Float32:
    var dx = a.x - b.x; var dy = a.y - b.y; var dz = a.z - b.z
    return dx * dx + dy * dy + dz * dz


def _tri_area(ref sd: SceneView, p: PrimId) -> Float32:
    var m = sd.meshes[unsafe_offset=Int(p.id1)]
    var bv = Int(p.id2)
    var p0 = mesh_pos(m, mesh_vi(m, bv))
    var c = cross(mesh_pos(m, mesh_vi(m, bv + 1)) - p0, mesh_pos(m, mesh_vi(m, bv + 2)) - p0)
    return Float32(0.5) * sqrt(dot(c, c))


def _tri_point(ref sd: SceneView, p: PrimId, b1: Float32, b2: Float32) -> Tuple[Point3f, Vec3f]:
    """A point at barycentrics (b1, b2) and its geometric normal, turned to the shading normals' side."""
    var m = sd.meshes[unsafe_offset=Int(p.id1)]
    var bv = Int(p.id2)
    var i0 = mesh_vi(m, bv); var i1 = mesh_vi(m, bv + 1); var i2 = mesh_vi(m, bv + 2)
    var p0 = mesh_pos(m, i0); var p1 = mesh_pos(m, i1); var p2 = mesh_pos(m, i2)
    var b0 = Float32(1) - b1 - b2
    var x = p0 * b0 + p1 * b1 + p2 * b2
    var n = cross(p1 - p0, p2 - p0)
    n = n * (Float32(1) / max(sqrt(dot(n, n)), Float32(1e-30)))
    if _is_real_ptr(m.normals):
        var ns = mesh_normal(m, i0) * b0 + mesh_normal(m, i1) * b1 + mesh_normal(m, i2) * b2
        if dot(ns, n) < Float32(0):
            n = -n
    return (Point3f(x.x, x.y, x.z), n)
