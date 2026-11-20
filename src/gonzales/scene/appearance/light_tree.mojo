"""pbrt-v4's BVHLightSampler (Conty and Kulla 2018): a light BVH over every emitter, sampled by importance at a point.

One UInt32 blob, so the GPU upload and the scene cache move a single buffer:
  header (LT_HEADER words): n_nodes, n_emitters, off_nodes, off_trails, off_base, n_lights, 2 unused, all-bounds min xyz
    and max xyz (f32 bits)
  nodes (LT_NODE words each, pbrt's CompactLightBounds + LightBVHNode, depth first, child 0 follows its parent):
    0..2 quantized bounds (16 bits each: min x y, min z max x, max y z), 3 octahedral axis, 4 phi (f32),
    5 cos_o (15 bits) | cos_e (15 bits) << 15 | two_sided << 30, 6 child 1 or emitter index | leaf << 31
  trails (2 words per emitter): the child bits from the root, bit d at depth d; LT_ABSENT in the high word when the
    emitter has no power and so is not in the tree.
  base (n_lights + 1 words): light l owns emitters base[l] ..< base[l + 1], a mesh light one per triangle.
"""

from std.math import sqrt, acos, cos, sin, floor, ceil, max, min, abs
from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from gonzales.core.geometry import Vec3f, cross, dot, PI

comptime LT_HEADER = 16
comptime LT_NODE = 8
comptime LT_BUCKETS = 12
comptime LT_MAX_DEPTH = 63
comptime LT_ABSENT = UInt32(0x80000000)
comptime LT_ONE_MINUS_EPS = Float32(0.99999994)


@fieldwise_init
struct LightBounds(TrivialRegisterPassable):
    """pbrt's LightBounds: where the emitter is, which way it emits (cone around w, cos_o), how much (phi)."""
    var bmin: Vec3f
    var bmax: Vec3f
    var w: Vec3f
    var phi: Float32
    var cos_o: Float32
    var cos_e: Float32
    var two_sided: Bool

    @staticmethod
    def empty() -> LightBounds:
        return LightBounds(Vec3f(3.0e38), Vec3f(-3.0e38), Vec3f(0.0, 0.0, 1.0), 0.0, 1.0, 1.0, False)

    def centroid(self) -> Vec3f:
        return (self.bmin + self.bmax) * 0.5


@always_inline
def _c(v: Vec3f, i: Int) -> Float32:
    return v.x if i == 0 else (v.y if i == 1 else v.z)


@always_inline
def _safe_sqrt(x: Float32) -> Float32:
    return sqrt(max(x, Float32(0)))


@always_inline
def _safe_acos(x: Float32) -> Float32:
    return acos(min(max(x, Float32(-1)), Float32(1)))


def _angle_between(a: Vec3f, b: Vec3f) -> Float32:
    """pbrt's AngleBetween, accurate near 0 and pi."""
    if dot(a, b) < 0:
        return PI - 2 * _asin_safe((a + b).length() * 0.5)
    return 2 * _asin_safe((b - a).length() * 0.5)


@always_inline
def _asin_safe(x: Float32) -> Float32:
    var c = min(max(x, Float32(-1)), Float32(1))
    return Float32(0.5) * PI - acos(c)


def _rotate(v: Vec3f, k: Vec3f, theta: Float32) -> Vec3f:
    """v rotated by theta around the unit axis k (Rodrigues)."""
    var c = cos(theta)
    var s = sin(theta)
    return v * c + cross(k, v) * s + k * (dot(k, v) * (1 - c))


def _cone_union(wa: Vec3f, ca: Float32, wb: Vec3f, cb: Float32) -> Tuple[Vec3f, Float32]:
    """pbrt's Union(DirectionCone, DirectionCone)."""
    var ta = _safe_acos(ca)
    var tb = _safe_acos(cb)
    var td = _angle_between(wa, wb)
    if min(td + tb, PI) <= ta:
        return (wa, ca)
    if min(td + ta, PI) <= tb:
        return (wb, cb)
    var to = (ta + td + tb) * 0.5
    if to >= PI:
        return (wa, Float32(-1))
    var wr = cross(wa, wb)
    if wr.length_sq() == 0:
        return (wa, Float32(-1))
    return (_rotate(wa, wr.normalize(), to - ta), cos(to))


def light_bounds_union(a: LightBounds, b: LightBounds) -> LightBounds:
    if a.phi == 0:
        return b
    if b.phi == 0:
        return a
    var (w, cos_o) = _cone_union(a.w, a.cos_o, b.w, b.cos_o)
    return LightBounds(Vec3f(min(a.bmin.x, b.bmin.x), min(a.bmin.y, b.bmin.y), min(a.bmin.z, b.bmin.z)),
                       Vec3f(max(a.bmax.x, b.bmax.x), max(a.bmax.y, b.bmax.y), max(a.bmax.z, b.bmax.z)),
                       w, a.phi + b.phi, cos_o, min(a.cos_e, b.cos_e), a.two_sided or b.two_sided)


def _cost(b: LightBounds, ext: Vec3f, dim: Int) -> Float32:
    """pbrt's EvaluateCost: power x direction measure x box area, stretched along thin dimensions."""
    var t_o = _safe_acos(b.cos_o)
    var t_w = min(t_o + _safe_acos(b.cos_e), PI)
    var s_o = _safe_sqrt(1 - b.cos_o * b.cos_o)
    var m_omega = 2 * PI * (1 - b.cos_o) + PI / 2 * (2 * t_w * s_o - cos(t_o - 2 * t_w) - 2 * t_o * s_o + b.cos_o)
    var d = b.bmax - b.bmin
    var area = 2 * (d.x * d.y + d.y * d.z + d.z * d.x)
    var kr = max(ext.x, max(ext.y, ext.z)) / _c(ext, dim)
    return b.phi * m_omega * kr * area


@always_inline
def _quantize_cos(c: Float32) -> UInt32:
    return UInt32(floor(Float32(32767) * ((min(max(c, Float32(-1)), Float32(1)) + 1) * 0.5)))


@always_inline
def _quantize_bound(c: Float32, lo: Float32, hi: Float32) -> Float32:
    return Float32(0) if hi == lo else Float32(65535) * min(max((c - lo) / (hi - lo), Float32(0)), Float32(1))


@always_inline
def _oct_encode(f: Float32) -> UInt32:
    return UInt32(floor((min(max(f, Float32(-1)), Float32(1)) + 1) * 0.5 * 65535 + 0.5))


@always_inline
def _sign(x: Float32) -> Float32:
    return Float32(-1) if x < 0 else Float32(1)


def _oct_word(v: Vec3f) -> UInt32:
    var s = abs(v.x) + abs(v.y) + abs(v.z)
    var x = v.x / s
    var y = v.y / s
    if v.z < 0:
        var xo = x
        x = (1 - abs(y)) * _sign(xo)
        y = (1 - abs(xo)) * _sign(y)
    return _oct_encode(x) | (_oct_encode(y) << 16)


@always_inline
def _oct_decode(word: UInt32) -> Vec3f:
    var x = Float32(-1) + 2 * Float32(word & 0xFFFF) / 65535
    var y = Float32(-1) + 2 * Float32(word >> 16) / 65535
    var z = 1 - (abs(x) + abs(y))
    if z < 0:
        var xo = x
        x = (1 - abs(y)) * _sign(xo)
        y = (1 - abs(xo)) * _sign(y)
    return Vec3f(x, y, z).normalize()


def _put_node(mut words: List[UInt32], node: Int, lb: LightBounds, child: Int, leaf: Bool, amin: Vec3f, amax: Vec3f):
    var q = List[UInt32](length=6, fill=0)
    for c in range(3):
        q[c] = UInt32(floor(_quantize_bound(_c(lb.bmin, c), _c(amin, c), _c(amax, c))))
        q[3 + c] = UInt32(ceil(_quantize_bound(_c(lb.bmax, c), _c(amin, c), _c(amax, c))))
    var o = LT_HEADER + node * LT_NODE
    words[o] = q[0] | (q[1] << 16)
    words[o + 1] = q[2] | (q[3] << 16)
    words[o + 2] = q[4] | (q[5] << 16)
    words[o + 3] = _oct_word(lb.w)
    words[o + 4] = bitcast[DType.uint32](lb.phi)
    words[o + 5] = _quantize_cos(lb.cos_o) | (_quantize_cos(lb.cos_e) << 15) | (UInt32(1 << 30) if lb.two_sided else UInt32(0))
    words[o + 6] = UInt32(child) | (UInt32(1 << 31) if leaf else UInt32(0))


def _build(mut words: List[UInt32], mut trails: List[UInt64], bounds: List[LightBounds], mut order: List[Int],
           start: Int, end: Int, trail: UInt64, depth: Int, amin: Vec3f, amax: Vec3f) raises -> LightBounds:
    var node = (len(words) - LT_HEADER) // LT_NODE
    for _ in range(LT_NODE):
        words.append(0)
    if end - start == 1:
        var e = order[start]
        _put_node(words, node, bounds[e], e, True, amin, amax)
        trails[e] = trail
        return bounds[e]
    if depth >= LT_MAX_DEPTH:
        raise Error("light tree deeper than 63 levels")
    var all = LightBounds.empty()
    var cmin = Vec3f(3.0e38)
    var cmax = Vec3f(-3.0e38)
    for i in range(start, end):
        var lb = bounds[order[i]]
        all = light_bounds_union(all, lb)
        var pc = lb.centroid()
        cmin = Vec3f(min(cmin.x, pc.x), min(cmin.y, pc.y), min(cmin.z, pc.z))
        cmax = Vec3f(max(cmax.x, pc.x), max(cmax.y, pc.y), max(cmax.z, pc.z))
    var ext = all.bmax - all.bmin
    var best = Float32(3.0e38)
    var best_dim = -1
    var best_bucket = -1
    for dim in range(3):
        if _c(cmax, dim) == _c(cmin, dim) or _c(ext, dim) <= 0:
            continue
        var buckets = List[LightBounds](length=LT_BUCKETS, fill=LightBounds.empty())
        for i in range(start, end):
            var lb = bounds[order[i]]
            var b = _bucket(lb.centroid(), cmin, cmax, dim)
            buckets[b] = light_bounds_union(buckets[b], lb)
        # pbrt's loop starts at 1 as well: a split after the first bucket is never taken.
        for i in range(1, LT_BUCKETS - 1):
            var b0 = LightBounds.empty()
            var b1 = LightBounds.empty()
            for j in range(i + 1):
                b0 = light_bounds_union(b0, buckets[j])
            for j in range(i + 1, LT_BUCKETS):
                b1 = light_bounds_union(b1, buckets[j])
            var cost = _cost(b0, ext, dim) + _cost(b1, ext, dim)
            if cost > 0 and cost < best:
                best = cost
                best_dim = dim
                best_bucket = i
    var mid = (start + end) // 2
    if best_dim >= 0:
        var k = start
        for i in range(start, end):
            if _bucket(bounds[order[i]].centroid(), cmin, cmax, best_dim) <= best_bucket:
                order.swap_elements(k, i)
                k += 1
        if k != start and k != end:
            mid = k
    var b0 = _build(words, trails, bounds, order, start, mid, trail, depth + 1, amin, amax)
    var child1 = (len(words) - LT_HEADER) // LT_NODE
    var b1 = _build(words, trails, bounds, order, mid, end, trail | (UInt64(1) << UInt64(depth)), depth + 1, amin, amax)
    var lb = light_bounds_union(b0, b1)
    _put_node(words, node, lb, child1, False, amin, amax)
    return lb


@always_inline
def _bucket(pc: Vec3f, cmin: Vec3f, cmax: Vec3f, dim: Int) -> Int:
    var b = Int(Float32(LT_BUCKETS) * (_c(pc, dim) - _c(cmin, dim)) / (_c(cmax, dim) - _c(cmin, dim)))
    return min(max(b, 0), LT_BUCKETS - 1)


def build_light_tree(bounds: List[LightBounds], base: List[Int]) raises -> List[UInt32]:
    """The blob for these emitters, grouped into lights by base; emitters with no power stay out of the tree (pbrt)."""
    var n = len(bounds)
    var order = List[Int]()
    var amin = Vec3f(3.0e38)
    var amax = Vec3f(-3.0e38)
    for e in range(n):
        if bounds[e].phi > 0:
            order.append(e)
            amin = Vec3f(min(amin.x, bounds[e].bmin.x), min(amin.y, bounds[e].bmin.y), min(amin.z, bounds[e].bmin.z))
            amax = Vec3f(max(amax.x, bounds[e].bmax.x), max(amax.y, bounds[e].bmax.y), max(amax.z, bounds[e].bmax.z))
    var words = List[UInt32](length=LT_HEADER, fill=0)
    var trails = List[UInt64](length=n, fill=UInt64(LT_ABSENT) << 32)
    if len(order) > 0:
        _ = _build(words, trails, bounds, order, 0, len(order), 0, 0, amin, amax)
    var n_nodes = (len(words) - LT_HEADER) // LT_NODE
    words[0] = UInt32(n_nodes)
    words[1] = UInt32(n)
    words[2] = UInt32(LT_HEADER)
    words[3] = UInt32(len(words))
    for c in range(3):
        words[8 + c] = bitcast[DType.uint32](_c(amin, c))
        words[11 + c] = bitcast[DType.uint32](_c(amax, c))
    for e in range(n):
        words.append(UInt32(trails[e] & 0xFFFFFFFF))
        words.append(UInt32(trails[e] >> 32))
    words[4] = UInt32(len(words))
    words[5] = UInt32(len(base) - 1)
    for l in range(len(base)):
        words.append(UInt32(base[l]))
    return words^


# ---- Device side: everything below reads only the blob ----

@always_inline
def _word(t: Pointer[UInt32, MutUntrackedOrigin], i: Int) -> UInt32:
    return t[unsafe_offset=i]


@always_inline
def _f32(t: Pointer[UInt32, MutUntrackedOrigin], i: Int) -> Float32:
    return bitcast[DType.float32](t[unsafe_offset=i])


@always_inline
def _cos_sub_clamped(sin_a: Float32, cos_a: Float32, sin_b: Float32, cos_b: Float32) -> Float32:
    return Float32(1) if cos_a > cos_b else cos_a * cos_b + sin_a * sin_b


@always_inline
def _sin_sub_clamped(sin_a: Float32, cos_a: Float32, sin_b: Float32, cos_b: Float32) -> Float32:
    return Float32(0) if cos_a > cos_b else sin_a * cos_b - cos_a * sin_b


@always_inline
def _lerp16(q: UInt32, lo: Float32, hi: Float32) -> Float32:
    var s = Float32(q) / 65535
    return (1 - s) * lo + s * hi


def light_tree_importance(t: Pointer[UInt32, MutUntrackedOrigin], node: Int, p: Vec3f, n: Vec3f) -> Float32:
    """pbrt's CompactLightBounds::Importance; n = 0 (a medium) drops the receiver's cosine bound."""
    var o = Int(_word(t, 2)) + node * LT_NODE
    var w0 = _word(t, o)
    var w1 = _word(t, o + 1)
    var w2 = _word(t, o + 2)
    var bmin = Vec3f(_lerp16(w0 & 0xFFFF, _f32(t, 8), _f32(t, 11)), _lerp16(w0 >> 16, _f32(t, 9), _f32(t, 12)),
                     _lerp16(w1 & 0xFFFF, _f32(t, 10), _f32(t, 13)))
    var bmax = Vec3f(_lerp16(w1 >> 16, _f32(t, 8), _f32(t, 11)), _lerp16(w2 & 0xFFFF, _f32(t, 9), _f32(t, 12)),
                     _lerp16(w2 >> 16, _f32(t, 10), _f32(t, 13)))
    var w = _oct_decode(_word(t, o + 3))
    var phi = _f32(t, o + 4)
    var cones = _word(t, o + 5)
    var cos_o = 2 * (Float32(cones & 0x7FFF) / 32767) - 1
    var cos_e = 2 * (Float32((cones >> 15) & 0x7FFF) / 32767) - 1
    var pc = (bmin + bmax) * 0.5
    var dp = p - pc
    var dist2 = dp.length_sq()
    var d2 = max(dist2, (bmax - bmin).length() * 0.5)
    var wi = dp * (1 / sqrt(max(dist2, Float32(1e-30))))
    var cos_w = dot(w, wi)
    if (cones >> 30) & 1 != 0:
        cos_w = abs(cos_w)
    var sin_w = _safe_sqrt(1 - cos_w * cos_w)
    # The cone of directions the box subtends from p (pbrt's BoundSubtendedDirections): everything when p is inside its sphere.
    var r2 = (bmax - pc).length_sq()
    var cos_b = Float32(-1)
    if dist2 >= r2:
        cos_b = _safe_sqrt(1 - r2 / max(dist2, Float32(1e-30)))
    var sin_b = _safe_sqrt(1 - cos_b * cos_b)
    var sin_o = _safe_sqrt(1 - cos_o * cos_o)
    var cos_x = _cos_sub_clamped(sin_w, cos_w, sin_o, cos_o)
    var sin_x = _sin_sub_clamped(sin_w, cos_w, sin_o, cos_o)
    var cos_p = _cos_sub_clamped(sin_x, cos_x, sin_b, cos_b)
    if cos_p <= cos_e:
        return 0
    var importance = phi * cos_p / d2
    if dot(n, n) > 0:
        var cos_i = abs(dot(wi, n))
        var sin_i = _safe_sqrt(1 - cos_i * cos_i)
        importance *= _cos_sub_clamped(sin_i, cos_i, sin_b, cos_b)
    return max(importance, Float32(0))


@always_inline
def light_tree_present(t: Pointer[UInt32, MutUntrackedOrigin]) -> Bool:
    return _word(t, 0) != 0


def light_tree_empty() -> Pointer[UInt32, MutUntrackedOrigin]:
    """A blob with no tree: the light sampler then picks by power."""
    var t = unsafe_alloc[UInt32](LT_HEADER)
    for i in range(LT_HEADER):
        t[unsafe_offset=i] = 0
    return t


def light_tree_words(t: Pointer[UInt32, MutUntrackedOrigin]) -> Int:
    """The blob's length in words."""
    return Int(_word(t, 4)) + Int(_word(t, 5)) + 1 if _word(t, 4) != 0 else LT_HEADER


@always_inline
def light_tree_emitter(t: Pointer[UInt32, MutUntrackedOrigin], light: Int, tri: Int) -> Int:
    return Int(_word(t, Int(_word(t, 4)) + light)) + tri


def light_tree_light(t: Pointer[UInt32, MutUntrackedOrigin], e: Int) -> Tuple[Int, Int]:
    """(light, triangle) of emitter e."""
    var b = Int(_word(t, 4))
    var lo = 0
    var hi = Int(_word(t, 5)) - 1
    while lo < hi:
        var mid = (lo + hi + 1) >> 1
        if Int(_word(t, b + mid)) <= e:
            lo = mid
        else:
            hi = mid - 1
    return (lo, e - Int(_word(t, b + lo)))


def light_tree_sample(t: Pointer[UInt32, MutUntrackedOrigin], p: Vec3f, n: Vec3f, u_in: Float32) -> Tuple[Int, Float32]:
    """(emitter, pmf) picked by importance at p, or (-1, 0) when nothing can light p."""
    if not light_tree_present(t):
        return (-1, Float32(0))
    var u = min(u_in, LT_ONE_MINUS_EPS)
    var node = 0
    var pmf = Float32(1)
    var base = Int(_word(t, 2))
    while True:
        var cw = _word(t, base + node * LT_NODE + 6)
        if cw >> 31 != 0:
            if node > 0 or light_tree_importance(t, 0, p, n) > 0:
                return (Int(cw & 0x7FFFFFFF), pmf)
            return (-1, Float32(0))
        var c1 = Int(cw & 0x7FFFFFFF)
        var i0 = light_tree_importance(t, node + 1, p, n)
        var i1 = light_tree_importance(t, c1, p, n)
        var sum = i0 + i1
        if not (sum > 0):
            return (-1, Float32(0))
        var p0 = i0 / sum
        if u < p0:
            u = min(u / p0, LT_ONE_MINUS_EPS)
            pmf *= p0
            node += 1
        else:
            var p1 = i1 / sum
            u = min((u - p0) / p1, LT_ONE_MINUS_EPS)
            pmf *= p1
            node = c1


def light_tree_pmf(t: Pointer[UInt32, MutUntrackedOrigin], emitter: Int, p: Vec3f, n: Vec3f) -> Float32:
    """The probability light_tree_sample(t, p, n, .) returns emitter."""
    if not light_tree_present(t):
        return 0
    var to = Int(_word(t, 3)) + 2 * emitter
    var hi = _word(t, to + 1)
    if hi & LT_ABSENT != 0:
        return 0
    var trail = UInt64(_word(t, to)) | (UInt64(hi) << 32)
    var base = Int(_word(t, 2))
    var node = 0
    var pmf = Float32(1)
    while True:
        var cw = _word(t, base + node * LT_NODE + 6)
        if cw >> 31 != 0:
            if node == 0 and not (light_tree_importance(t, 0, p, n) > 0):
                return 0
            return pmf
        var c1 = Int(cw & 0x7FFFFFFF)
        var i0 = light_tree_importance(t, node + 1, p, n)
        var i1 = light_tree_importance(t, c1, p, n)
        var sum = i0 + i1
        if not (sum > 0):
            return 0
        if trail & 1 != 0:
            pmf *= i1 / sum
            node = c1
        else:
            pmf *= i0 / sum
            node += 1
        trail >>= 1
